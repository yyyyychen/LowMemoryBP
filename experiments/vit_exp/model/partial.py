import sys
import importlib
from .vit import vit_models, Block
from functools import partial
import torch.nn.functional as F
import torch.nn as nn

import torch
import lomem

LayerNorm = torch.nn.LayerNorm
MSLayerNorm = lomem.nn.MSLayerNorm

ReLU = torch.nn.ReLU
GELU = torch.nn.GELU
ReGELU2 = lomem.nn.ReGELU2


class LinearFunction(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None):
        output = F.linear(x, weight, bias)

        if ctx.needs_input_grad[0] and not ctx.needs_input_grad[1]:
            ctx.save_for_backward(weight)
        elif ctx.needs_input_grad[1]:
            ctx.save_for_backward(weight, x)
        return output

    @staticmethod
    @torch.cuda.amp.custom_bwd
    @torch.autograd.function.once_differentiable
    def backward(ctx, output_grad):
        x_grad = weight_grad = bias_grad = None

        if ctx.needs_input_grad[0]:
            x_grad = output_grad @ ctx.saved_tensors[0]

        if ctx.needs_input_grad[1]:
            x = ctx.saved_tensors[1]
            weight_grad = output_grad.reshape(-1, output_grad.size(-1)).t() @ x.reshape(-1, x.size(-1))

        if ctx.needs_input_grad[2]:
            bias_grad = output_grad.reshape(-1, output_grad.size(-1)).sum(0)

        return x_grad, weight_grad, bias_grad


class Linear(torch.nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)

    def forward(self, input: torch.Tensor):
        return LinearFunction.apply(input, self.weight, self.bias)


class Attention(torch.nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self,
        dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
        **kwargs
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = Linear(dim, dim, bias=qkv_bias)
        self.k = Linear(dim, dim, bias=qkv_bias)
        self.v = Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = attn_drop
        self.proj = Linear(dim, dim)
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


class Mlp(torch.nn.Module):
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
        **kwargs
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = torch.nn.Dropout(drop)
        self.norm = norm_layer(hidden_features) if norm_layer is not None else torch.nn.Identity()
        self.fc2 = Linear(hidden_features, out_features, bias=bias)
        self.drop2 = torch.nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


def partial_trainable(model, key_words=None, full=False):
    if key_words is None:
        key_words = []

    key_words += ['head']

    for name, param in model.named_parameters():
        if full:
            param.requires_grad = True
        else:
            param.requires_grad = False

        for key in key_words:
            if key in name:
                param.requires_grad = True
                break


def vit_base_patch16_LS(img_size=224, key_words=None, full=False, vit_backbone='vit_models', norm_layer='LayerNorm', act_layer='GELU', **kwargs):
    """vit_base_patch16_LS for partial training"""

    norm_layer = getattr(sys.modules[__name__], norm_layer)
    act_layer = getattr(sys.modules[__name__], act_layer)

    vit_backbone = getattr(sys.modules[__name__], vit_backbone)

    model = vit_backbone(
        img_size = img_size, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(norm_layer, eps=1e-6), block_layers=Block, act_layer=act_layer, Attention_block=Attention, Mlp_block=Mlp, **kwargs)

    partial_trainable(model, key_words, full)
    return model


def vit_large_patch16_LS(img_size=224, key_words=None, full=False, vit_backbone='vit_models', norm_layer='LayerNorm', act_layer='GELU', **kwargs):
    norm_layer = getattr(sys.modules[__name__], norm_layer)
    act_layer = getattr(sys.modules[__name__], act_layer)

    vit_backbone = getattr(sys.modules[__name__], vit_backbone)

    model = vit_backbone(
        img_size=img_size, patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(norm_layer, eps=1e-6), block_layers=Block, act_layer=act_layer,
        Attention_block=Attention, Mlp_block=Mlp, **kwargs)

    partial_trainable(model, key_words, full)
    return model