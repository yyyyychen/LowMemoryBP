import torch
from . import _C


def apply_decorator(func):
    def wrapper(*args, **kwargs):
        result = func.apply(*args, **kwargs)
        return result
    return wrapper


@apply_decorator
class regelu2(torch.autograd.Function):
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