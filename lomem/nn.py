import torch
import lomem.activation as act
import lomem.normalization as norm
from typing import Sequence


class ReGELU2(torch.nn.Module):
    """
    Apply ReGELU2.
    """
    def __init__(self):
        super().__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = act.regelu2(input)
        return output


class ReSiLU2(torch.nn.Module):
    """
    Apply ReSiLU2.
    """
    def __init__(self):
        super().__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = act.resilu2(input)
        return output


class MSLayerNorm(torch.nn.Module):
    """
    Apply MS-LayerNorm without affine transforms to the last several dimensions according to the normalized_shape argument.
    Args:
        normalized_shape (int or list or torch.Size): input shape from an expected input of size.
            If a single integer is used, it is treated as a singleton list, and this module will
            normalize over the last dimension which is expected to be of that specific size.
        eps: a value added to the denominator for numerical stability. Default: 1e-5
    """
    def __init__(self, normalized_shape: Sequence[int], eps: float = 1e-5):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.num_dims = len(normalized_shape)
        self.eps = eps

    def forward(self, input: torch.Tensor):
        output = norm.layer_norm(input, self.normalized_shape, self.eps)
        return output


class MSRMSNorm(torch.nn.Module):
    """
    Apply MS-RMSNorm without affine transforms to the last several dimensions according to the normalized_shape argument.
    Args:
        normalized_shape (int or list or torch.Size): input shape from an expected input of size.
            If a single integer is used, it is treated as a singleton list, and this module will
            normalize over the last dimension which is expected to be of that specific size.
        eps: a value added to the denominator for numerical stability. Default: 1e-5
    """
    def __init__(self, normalized_shape: Sequence[int], eps: float = 1e-5):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.num_dims = len(normalized_shape)
        self.eps = eps

    def forward(self, input: torch.Tensor):
        output = norm.rms_norm(input, self.normalized_shape, self.eps)
        return output