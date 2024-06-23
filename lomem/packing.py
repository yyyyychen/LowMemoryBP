import torch
from . import _C
from typing import Sequence
from functools import reduce


def pack_bool_to_uint8(x: torch.Tensor):
    """
    Pack eight 8-bit bool-type data to one 8-bit uint8-type data.
    """
    packed_x_1d = _C.bool_to_uint8(x)
    return packed_x_1d


def unpack_uint8_to_bool(x: torch.Tensor, shape: Sequence[int]):
    """
    Restore eight 8-bit bool-type data from one 8-bit uint8-type data.
    """
    numel = reduce(lambda x, y: x * y, shape)
    unpacked_x_1d = _C.uint8_to_bool(x)[:numel]
    unpacked_x = unpacked_x_1d.reshape(shape)
    return unpacked_x