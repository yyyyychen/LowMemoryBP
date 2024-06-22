import torch
from . import _C
from typing import Sequence


class LayerNormFunction(torch.autograd.Function):
    """
    Apply LayerNorm without affine transforms.
    """
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, x: torch.Tensor, normalized_shape: Sequence[int], eps: float):
        out, rstd = _C.layer_norm_fw(x, normalized_shape, eps)
        ctx.save_for_backward(out, rstd)
        return out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    @torch.autograd.function.once_differentiable
    def backward(ctx, out_grad):
        grad = _C.layer_norm_bw(out_grad, ctx.saved_tensors[0], ctx.saved_tensors[1])
        return grad, None, None


def layer_norm(input: torch.Tensor, normalized_shape: Sequence[int], eps: float) -> torch.Tensor:
    """
    Apply LayerNorm without affine transforms.
    """
    output = LayerNormFunction.apply(input, normalized_shape, eps)
    return output