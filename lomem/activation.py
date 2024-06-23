import torch
from . import _C


class ReGELU2Function(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, x: torch.Tensor):
        out, packed_flag= _C.regelu2_fw(x)
        ctx.save_for_backward(packed_flag)
        return out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    @torch.autograd.function.once_differentiable
    def backward(ctx, out_grad: torch.Tensor):
        grad = _C.regelu2_bw(out_grad, ctx.saved_tensors[0])
        return grad


def regelu2(input: torch.Tensor) -> torch.Tensor:
    """
    Apply ReGELU2.
    """
    output = ReGELU2Function.apply(input)
    return output


class ReSiLU2Function(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, x: torch.Tensor):
        out, packed_flag= _C.resilu2_fw(x)
        ctx.save_for_backward(packed_flag)
        return out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    @torch.autograd.function.once_differentiable
    def backward(ctx, out_grad: torch.Tensor):
        grad = _C.resilu2_bw(out_grad, ctx.saved_tensors[0])
        return grad


def resilu2(input: torch.Tensor) -> torch.Tensor:
    """
    Apply ReSiLU2.
    """
    output = ReSiLU2Function.apply(input)
    return output