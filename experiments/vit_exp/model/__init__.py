import os
import torch

import importlib


@torch.no_grad()
def load_flax_vit_weights(checkpoint_path: str, prefix: str = ''):
    """ Convert weights from .npz checkpoints for official Google Brain Flax implementation
    """
    import numpy as np
    import torch

    def _n2p(w, t=True):
        if w.ndim == 4 and w.shape[0] == w.shape[1] == w.shape[2] == 1:
            w = w.flatten()
        if t:
            if w.ndim == 4:
                w = w.transpose([3, 2, 0, 1])
            elif w.ndim == 3:
                w = w.transpose([2, 0, 1])
            elif w.ndim == 2:
                w = w.transpose([1, 0])
        return torch.from_numpy(w)

    w_p = {}

    w = np.load(checkpoint_path)

    w_p['patch_embed.proj.weight'] = _n2p(w[f'{prefix}embedding/kernel'])
    w_p['patch_embed.proj.bias'] = _n2p(w[f'{prefix}embedding/bias'])

    w_p['cls_token'] = _n2p(w[f'{prefix}cls'], t=False)

    w_p['pos_embed'] = _n2p(w[f'{prefix}Transformer/posembed_input/pos_embedding'], t=False)

    # if cls_token has pos_embed, merge it to cls_token and remove it from pos_embed
    if w_p['pos_embed'].shape[1] % 2 != 0:
        w_p['cls_token'] += w_p['pos_embed'][:, [0]]
        w_p['pos_embed'] = w_p['pos_embed'][:, 1:]

    w_p['norm.weight'] = _n2p(w[f'{prefix}Transformer/encoder_norm/scale'])
    w_p['norm.bias'] = _n2p(w[f'{prefix}Transformer/encoder_norm/bias'])

    mha_sub, b_sub, ln1_sub = (1, 3, 2)

    i = 0
    while True:
        block_prefix = f'{prefix}Transformer/encoderblock_{i}/'
        mha_prefix = block_prefix + f'MultiHeadDotProductAttention_{mha_sub}/'

        if f'{mha_prefix}query/kernel' not in w.keys():
            break

        w_p[f'blocks.{i}.norm1.weight'] = _n2p(w[f'{block_prefix}LayerNorm_0/scale'])
        w_p[f'blocks.{i}.norm1.bias'] = _n2p(w[f'{block_prefix}LayerNorm_0/bias'])

        for n_p, n in zip(('q', 'k', 'v'), ('query', 'key', 'value')):
            w_p[f'blocks.{i}.attn.{n_p}.weight'] = _n2p(w[f'{mha_prefix}{n}/kernel'], t=False).flatten(1).T
            w_p[f'blocks.{i}.attn.{n_p}.bias'] = _n2p(w[f'{mha_prefix}{n}/bias'], t=False).reshape(-1)

        w_p[f'blocks.{i}.attn.proj.weight'] = _n2p(w[f'{mha_prefix}out/kernel']).flatten(1)
        w_p[f'blocks.{i}.attn.proj.bias'] = _n2p(w[f'{mha_prefix}out/bias'])

        w_p[f'blocks.{i}.norm2.weight'] = _n2p(w[f'{block_prefix}LayerNorm_{ln1_sub}/scale'])
        w_p[f'blocks.{i}.norm2.bias'] = _n2p(w[f'{block_prefix}LayerNorm_{ln1_sub}/bias'])

        for r in range(2):
            w_p[f'blocks.{i}.mlp.fc{r + 1}.weight'] = _n2p(w[f'{block_prefix}MlpBlock_{b_sub}/Dense_{r}/kernel'])
            w_p[f'blocks.{i}.mlp.fc{r + 1}.bias'] = _n2p(w[f'{block_prefix}MlpBlock_{b_sub}/Dense_{r}/bias'])

        i += 1

    return w_p


@torch.no_grad()
def _merge_ln(state_dict):

    new_dict = {}
    for n, p in state_dict.items():
        if 'norm' not in n:
            new_dict[n] = p

    i = 0
    while True:
        if f'blocks.{i}.attn.q.weight' not in state_dict.keys():
            break

        for n in ('q', 'k', 'v'):
            new_dict[f'blocks.{i}.attn.{n}.weight'] = state_dict[f'blocks.{i}.attn.{n}.weight'] * state_dict[f'blocks.{i}.norm1.weight']
            new_dict[f'blocks.{i}.attn.{n}.bias'] = new_dict[f'blocks.{i}.attn.{n}.bias'] + state_dict[f'blocks.{i}.attn.{n}.weight'] @ state_dict[f'blocks.{i}.norm1.bias']

        new_dict[f'blocks.{i}.mlp.fc1.weight'] = state_dict[f'blocks.{i}.mlp.fc1.weight'] * state_dict[f'blocks.{i}.norm2.weight']
        new_dict[f'blocks.{i}.mlp.fc1.bias'] = new_dict[f'blocks.{i}.mlp.fc1.bias'] + state_dict[f'blocks.{i}.mlp.fc1.weight'] @ state_dict[f'blocks.{i}.norm2.bias']

        i += 1
    return new_dict


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    device: str = 'cpu',
    strict: bool = False,
    merge_ln: bool = False,
):
    if os.path.splitext(checkpoint_path)[-1].lower() in ('.npz', '.npy'):
        # numpy checkpoint
        state_dict = load_flax_vit_weights(checkpoint_path)
    else:
        state_dict = torch.load(checkpoint_path, map_location=device)['model']

    if merge_ln:
        state_dict = _merge_ln(state_dict)

    incompatible_keys = model.load_state_dict(state_dict, strict=strict)
    return incompatible_keys


def get_model(cfg, num_classes, img_size, print_msg=True):
    model_builder_module = importlib.import_module(cfg.BUILDER.MODULE)
    model_builder = getattr(model_builder_module, cfg.BUILDER.METHOD)
    model = model_builder(**cfg.BUILDER.ARGS)
    if cfg.CHECKPOINT.PATH:
        print(f"Loading checkpoint from {cfg.CHECKPOINT.PATH}")
        merge_ln = ('MS' in cfg.BUILDER.ARGS.get('norm_layer'))
        msg = load_checkpoint(model, cfg.CHECKPOINT.PATH, strict=False, merge_ln=merge_ln)
        if print_msg:
            print(msg)
        model.resample_abs_pos_embed(img_size)
    if cfg.HEAD.RESET:
        head_builder_module = importlib.import_module(cfg.HEAD.MODULE)
        head_builder = getattr(head_builder_module, cfg.HEAD.METHOD)
        model.head = head_builder(**cfg.HEAD.ARGS, out_features=num_classes)
    return model