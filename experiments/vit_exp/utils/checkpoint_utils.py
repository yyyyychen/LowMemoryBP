import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from typing import List


def block_checkpoint(module_list: nn.ModuleList, x: torch.Tensor, block_ckpt: bool):
    if block_ckpt:
        for module in module_list:
            x = checkpoint(module, x, use_reentrant=False)
    else:
        for module in module_list:
            x = module(x)
    return x


def run_module_list(module_list: nn.ModuleList, x: torch.Tensor, start_id: int = 0, num: int = 0):
    for module in module_list[start_id:start_id + num]:
        x = module(x)
    return x


def split_checkpoint_unit(module_list: nn.ModuleList, x: torch.Tensor, num_block_per_unit: int = 0):
    if num_block_per_unit <= 0:
        x = run_module_list(module_list, x, 0, len(module_list))
    else:
        L = len(module_list)
        num_res = L % num_block_per_unit
        x = checkpoint(run_module_list, module_list, x, 0, num_res, use_reentrant=False)
        for i in range(num_res, L, num_block_per_unit):
            x = checkpoint(run_module_list, module_list, x, i, num_block_per_unit, use_reentrant=False)
    return x


def split_checkpoint_unit_by_list(module_list: nn.ModuleList, x: torch.Tensor, splits: List = None):
    if splits is None:
        x = run_module_list(module_list, x, 0, len(module_list))
    else:
        k = 0
        for num in splits:
            x = checkpoint(run_module_list, module_list, x, k, num, use_reentrant=False)
            k += num
    return x