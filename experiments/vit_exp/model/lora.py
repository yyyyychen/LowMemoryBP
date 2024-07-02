import sys
import importlib
from functools import partial
import torch
import lomem
from .vit import vit_models, Block
import math
import torch.nn.functional as F


class LoraFunction(torch.autograd.Function):

    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, x: torch.Tensor, A: torch.Tensor, B: torch.Tensor):
        xAt = x @ A.t()
        output = xAt @ B.t()

        if ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
            ctx.save_for_backward(A, B, x)

        return output

    @staticmethod
    @torch.cuda.amp.custom_bwd
    @torch.autograd.function.once_differentiable
    def backward(ctx, output_grad):
        x_grad = A_grad = B_grad = None

        if ctx.needs_input_grad[0]:
            A, B = ctx.saved_tensors[0], ctx.saved_tensors[1]
            x_grad = output_grad @ B @ A

        if ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
            A, B, x = ctx.saved_tensors
            z = output_grad.reshape(-1, output_grad.size(-1)).t() @ x.reshape(-1, x.size(-1))
            if ctx.needs_input_grad[1]:
                A_grad = B.t() @ z

            if ctx.needs_input_grad[2]:
                B_grad = z @ A.t()

        return x_grad, A_grad, B_grad


class LoraFunction_FZA(torch.autograd.Function):

    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, x: torch.Tensor, A: torch.Tensor, B: torch.Tensor):
        xAt = x @ A.t()
        output = xAt @ B.t()

        if ctx.needs_input_grad[2]:
            ctx.save_for_backward(A, B, xAt)
        return output

    @staticmethod
    @torch.cuda.amp.custom_bwd
    @torch.autograd.function.once_differentiable
    def backward(ctx, output_grad):
        x_grad = A_grad = B_grad = None

        if ctx.needs_input_grad[0]:
            A, B = ctx.saved_tensors[0], ctx.saved_tensors[1]
            x_grad = output_grad @ B @ A

        if ctx.needs_input_grad[2]:
            A, B, xAt = ctx.saved_tensors
            B_grad = output_grad.reshape(-1, output_grad.size(-1)).t() @ xAt.reshape(-1, xAt.size(-1))

        return x_grad, A_grad, B_grad


class LoRALayer():
    def __init__(
        self, 
        r: int, 
        merge_weights: bool
    ):
        self.r = r

        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights


class Linear(torch.nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        freeze_A = False,
        save_x_for_backward = False,
        merge_weights: bool = True,
        **kwargs
    ):
        torch.nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, merge_weights=merge_weights)

        # Actual trainable parameters
        if r > 0:
            save_x_for_backward_flag = save_x_for_backward or (not freeze_A)
            self.lora_func = LoraFunction.apply if save_x_for_backward_flag else LoraFunction_FZA.apply
            self.lora_A = torch.nn.Parameter(self.weight.new_zeros((r, in_features)), requires_grad=(not freeze_A))
            self.lora_B = torch.nn.Parameter(self.weight.new_zeros((out_features, r)))

            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()

    def merge_w(self):
        delta_w = self.lora_B @ self.lora_A
        return delta_w

    def reset_parameters(self):
        torch.nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            torch.nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            torch.nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        torch.nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= self.merge_w()
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += self.merge_w()
                self.merged = True       

    def forward(self, x: torch.Tensor):
        if self.r > 0 and not self.merged:
            result = F.linear(x, self.weight, bias=self.bias)
            return result + self.lora_func(x, self.lora_A, self.lora_B)
        else:
            return F.linear(x, self.weight, bias=self.bias)


class LoraAttention(torch.nn.Module):
    # modified from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self,
        dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
        qkv_lora=[True, True, True], proj_lora=False, rank=64, freeze_A=False,
        qkv_lora_save_x_for_backward = False,
        proj_lora_save_x_for_backward = False,
        **kwargs
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        linear_layer_q = partial(Linear, r=rank, freeze_A=freeze_A, save_x_for_backward=qkv_lora_save_x_for_backward) if qkv_lora[0] else torch.nn.Linear
        linear_layer_k = partial(Linear, r=rank, freeze_A=freeze_A, save_x_for_backward=qkv_lora_save_x_for_backward) if qkv_lora[1] else torch.nn.Linear
        linear_layer_v = partial(Linear, r=rank, freeze_A=freeze_A, save_x_for_backward=qkv_lora_save_x_for_backward) if qkv_lora[2] else torch.nn.Linear

        linear_layer_proj = partial(Linear, r=rank, freeze_A=freeze_A, save_x_for_backward=proj_lora_save_x_for_backward) if proj_lora else torch.nn.Linear

        self.q = linear_layer_q(dim, dim, bias=qkv_bias)
        self.k = linear_layer_k(dim, dim, bias=qkv_bias)
        self.v = linear_layer_v(dim, dim, bias=qkv_bias)

        self.attn_drop = attn_drop
        self.proj = linear_layer_proj(dim, dim)
        self.proj_drop = torch.nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        x = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.attn_drop, is_causal=False)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LoraMlp(torch.nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=torch.nn.ReLU,
        norm_layer=None,
        bias=True,
        drop=0.,
        fc_lora=[True, True],
        rank=64,
        freeze_A=False,
        fc_lora_save_x_for_backward=[False, False],
        **kwargs
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        linear_layer_1 = partial(Linear, r=rank, freeze_A=freeze_A, save_x_for_backward=fc_lora_save_x_for_backward[0]) if fc_lora[0] else torch.nn.Linear
        linear_layer_2 = partial(Linear, r=rank, freeze_A=freeze_A, save_x_for_backward=fc_lora_save_x_for_backward[1]) if fc_lora[1] else torch.nn.Linear

        self.fc1 = linear_layer_1(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = torch.nn.Dropout(drop)
        self.norm = norm_layer(hidden_features) if norm_layer is not None else torch.nn.Identity()
        self.fc2 = linear_layer_2(hidden_features, out_features, bias=bias)
        self.drop2 = torch.nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


LayerNorm = torch.nn.LayerNorm
MSLayerNorm = lomem.nn.MSLayerNorm

ReLU = torch.nn.ReLU
GELU = torch.nn.GELU
ReGELU2 = lomem.nn.ReGELU2


def mark_only_lora_as_trainable(model: torch.nn.Module, **kwargs) -> None:
    for n, p in model.named_parameters():
        if 'lora_' not in n:
            p.requires_grad = False


def vit_base_patch16_LS(img_size=224, LORA=None, vit_backbone='vit_models', norm_layer='LayerNorm', act_layer='GELU', **kwargs):
    if LORA.ATTN_MODULE:
        Attention_block = importlib.import_module(LORA.ATTN_METHOD, LORA.ATTN_MODULE)
    else:
        Attention_block = getattr(sys.modules[__name__], LORA.ATTN_METHOD)
    Attention_block = partial(Attention_block, **LORA.ARGS)

    if LORA.MLP_MODULE:
        Mlp_block = importlib.import_module(LORA.MLP_METHOD, LORA.MLP_MODULE)
    else:
        Mlp_block = getattr(sys.modules[__name__], LORA.MLP_METHOD)
    Mlp_block = partial(Mlp_block, **LORA.ARGS)

    norm_layer = getattr(sys.modules[__name__], norm_layer)
    act_layer = getattr(sys.modules[__name__], act_layer)

    vit_backbone = getattr(sys.modules[__name__], vit_backbone)

    model = vit_backbone(
        img_size = img_size, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(norm_layer, eps=1e-6), block_layers=Block, act_layer=act_layer,
        Attention_block=Attention_block, Mlp_block=Mlp_block, **kwargs)

    mark_only_lora_as_trainable(model, **LORA.ARGS)
    return model


def vit_large_patch16_LS(img_size=224, LORA=None, vit_backbone='vit_models', norm_layer='LayerNorm', act_layer='GELU', **kwargs):
    if LORA.ATTN_MODULE:
        Attention_block = importlib.import_module(LORA.ATTN_METHOD, LORA.ATTN_MODULE)
    else:
        Attention_block = getattr(sys.modules[__name__], LORA.ATTN_METHOD)
    Attention_block = partial(Attention_block, **LORA.ARGS)

    if LORA.MLP_MODULE:
        Mlp_block = importlib.import_module(LORA.MLP_METHOD, LORA.MLP_MODULE)
    else:
        Mlp_block = getattr(sys.modules[__name__], LORA.MLP_METHOD)
    Mlp_block = partial(Mlp_block, **LORA.ARGS)

    norm_layer = getattr(sys.modules[__name__], norm_layer)
    act_layer = getattr(sys.modules[__name__], act_layer)

    vit_backbone = getattr(sys.modules[__name__], vit_backbone)

    model = vit_backbone(
        img_size = img_size, patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(norm_layer, eps=1e-6), block_layers=Block, act_layer=act_layer,
        Attention_block=Attention_block, Mlp_block=Mlp_block, **kwargs)

    mark_only_lora_as_trainable(model, **LORA.ARGS)
    return model


def vit_huge_patch14_LS(img_size=224, LORA=None, vit_backbone='vit_models', norm_layer='LayerNorm', act_layer='GELU', **kwargs):
    if LORA.ATTN_MODULE:
        Attention_block = importlib.import_module(LORA.ATTN_METHOD, LORA.ATTN_MODULE)
    else:
        Attention_block = getattr(sys.modules[__name__], LORA.ATTN_METHOD)
    Attention_block = partial(Attention_block, **LORA.ARGS)

    if LORA.MLP_MODULE:
        Mlp_block = importlib.import_module(LORA.MLP_METHOD, LORA.MLP_MODULE)
    else:
        Mlp_block = getattr(sys.modules[__name__], LORA.MLP_METHOD)
    Mlp_block = partial(Mlp_block, **LORA.ARGS)

    norm_layer = getattr(sys.modules[__name__], norm_layer)
    act_layer = getattr(sys.modules[__name__], act_layer)

    vit_backbone = getattr(sys.modules[__name__], vit_backbone)

    model = vit_backbone(
        img_size = img_size, patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(norm_layer, eps=1e-6), block_layers=Block, act_layer=act_layer,
        Attention_block=Attention_block, Mlp_block=Mlp_block, **kwargs)

    mark_only_lora_as_trainable(model, **LORA.ARGS)
    return model