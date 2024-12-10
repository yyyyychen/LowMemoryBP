import torch
from torch import nn

import bitsandbytes as bnb
from bitsandbytes.autograd._functions import prod
import bitsandbytes.functional as F
from bitsandbytes.nn import Params4bit

tensor = torch.Tensor

import warnings
from warnings import warn

class MatMul4Bit(torch.autograd.Function):
    # forward is the same, but we added the fallback for pre-turing GPUs
    # backward is mostly the same, but adds one extra clause (see "elif state.CxB is not None")

    @staticmethod
    def forward(ctx, A, B, out=None, bias=None, quant_state: F.QuantState = None):
        # default of pytorch behavior if inputs are empty
        ctx.is_empty = False
        if prod(A.shape) == 0:
            ctx.is_empty = True
            ctx.A = A
            ctx.B = B
            ctx.bias = bias
            B_shape = quant_state.shape
            if A.shape[-1] == B_shape[0]:
                return torch.empty(A.shape[:-1] + B_shape[1:], dtype=A.dtype, device=A.device)
            else:
                return torch.empty(A.shape[:-1] + B_shape[:1], dtype=A.dtype, device=A.device)


        # 1. Dequantize
        # 2. MatmulnN
        output = torch.nn.functional.linear(A, F.dequantize_4bit(B, quant_state).to(A.dtype), bias)

        # 3. Save state
        ctx.state = quant_state
        ctx.dtype_A, ctx.dtype_B, ctx.dtype_bias = A.dtype, B.dtype, None if bias is None else bias.dtype

        if any(ctx.needs_input_grad[:2]):
            ctx.tensors = (A, B)
        else:
            ctx.tensors = (None, None)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.is_empty:
            bias_grad = None if ctx.bias is None else torch.zeros_like(ctx.bias)
            return torch.zeros_like(ctx.A), torch.zeros_like(ctx.B), None, bias_grad, None

        req_gradA, _, _, req_gradBias, _= ctx.needs_input_grad
        A, B = ctx.tensors

        grad_A, grad_B, grad_bias = None, None, None

        if req_gradBias:
            # compute grad_bias first before changing grad_output dtype
            grad_bias = grad_output.sum(0, dtype=ctx.dtype_bias)

        # not supported by PyTorch. TODO: create work-around
        #if req_gradB: grad_B = torch.matmul(grad_output.t(), A)
        if req_gradA: grad_A = torch.matmul(grad_output, F.dequantize_4bit(B, ctx.state).to(grad_output.dtype))

        return grad_A, grad_B, None, grad_bias, None

def matmul_4bit(A: tensor, B: tensor, quant_state: F.QuantState, out: tensor = None, bias=None):
    assert quant_state is not None
    if A.numel() == A.shape[-1] and A.requires_grad == False:
        if A.shape[-1] % quant_state.blocksize != 0:
            warn(f'Some matrices hidden dimension is not a multiple of {quant_state.blocksize} and efficient inference kernels are not supported for these (slow). Matrix input size found: {A.shape}')
            return MatMul4Bit.apply(A, B, out, bias, quant_state)
        else:
            out = F.gemv_4bit(A, B.t(), out, state=quant_state)
            if bias is not None:
                out += bias
            return out
    else:
        return MatMul4Bit.apply(A, B, out, bias, quant_state)

def Linear4bit_forward(self, x: torch.Tensor):
    # weights are cast automatically as Int8Params, but the bias has to be cast manually
    if self.bias is not None and self.bias.dtype != x.dtype:
        self.bias.data = self.bias.data.to(x.dtype)

    if getattr(self.weight, 'quant_state', None) is None:
        print('FP4 quantization state not initialized. Please call .cuda() or .to(device) on the LinearFP4 layer first.')
    if not self.compute_type_is_set:
        self.set_compute_type(x)
        self.compute_type_is_set = True

    inp_dtype = x.dtype
    if self.compute_dtype is not None:
        x = x.to(self.compute_dtype)

    bias = None if self.bias is None else self.bias.to(self.compute_dtype)
    out = matmul_4bit(x, self.weight.t(), bias=bias, quant_state=self.weight.quant_state)

    out = out.to(inp_dtype)

    return out