import time
import torch
from lomem import functional
from typing import Sequence
from functools import reduce


def torch_rms_norm(input: torch.Tensor, normalized_shape: Sequence[int], eps: float):
    """
    This function is only for testing.
    """
    input_2d = input.reshape(-1, reduce(lambda x, y: x * y, normalized_shape))
    rstd = torch.rsqrt(input_2d.pow(2).mean(-1, keepdim=True) + eps)
    output = (rstd * input_2d).reshape_as(input)
    return output


def test_func(func1_name, func1, func2_name, func2, input_size, dtype, device, num_repeat=100, print_msg=False):
    x = (torch.rand(input_size, dtype=dtype, device=device) - 0.5) * 20
    x_1 = x.clone().requires_grad_()
    x_2 = x.clone().requires_grad_()

    y_1 = func1(x_1)
    y_1.norm().backward()
    y_2 = func2(x_2)
    y_2.norm().backward()

    diff_fw = (y_1 - y_2).abs().mean()
    diff_bw = (x_1.grad - x_2.grad).abs().mean()

    torch.cuda.synchronize(device)
    start_time_1 = time.time()
    for _ in range(num_repeat):
        y_1 = func1(x_1)
        y_1.norm().backward()
    torch.cuda.synchronize(device)
    end_time_1 = time.time()
    mean_time_1 = (end_time_1 - start_time_1) / num_repeat

    torch.cuda.synchronize(device)
    start_time_2 = time.time()
    for _ in range(num_repeat):
        y_2 = func2(x_2)
        y_2.norm().backward()
    torch.cuda.synchronize(device)
    end_time_2 = time.time()
    mean_time_2 = (end_time_2 - start_time_2) / num_repeat

    if print_msg:
        print(f"problem size: {input_size} | dtype: {dtype}")
        print(f"{func1_name} time: {mean_time_1 * 1000} ms")
        print(f"{func2_name} time: {mean_time_2 * 1000} ms")
        print(f"forward diff: {diff_fw}")
        print(f"backward diff: {diff_bw}")
        print()
    return diff_fw + diff_bw



if __name__ == "__main__":
    device = "cuda:0"
    num_repeat = 100
    eps = 1e-8

    error_list = []

    shape = (32, 197, 768)
    dtype = torch.float32
    print("-------------------------------------------")
    error = test_func(
        "lomem rms_norm", lambda x: functional.rms_norm(x, normalized_shape=shape[-1:], eps=eps),
        "torch rms_norm", lambda x: torch_rms_norm(x, normalized_shape=shape[-1:], eps=eps),
        shape, dtype, device, num_repeat, True)
    error_list.append(error)

    shape = (32, 197, 768)
    dtype = torch.float16
    print("-------------------------------------------")
    error = test_func(
        "lomem rms_norm", lambda x: functional.rms_norm(x, normalized_shape=shape[-1:], eps=eps),
        "torch rms_norm", lambda x: torch_rms_norm(x, normalized_shape=shape[-1:], eps=eps),
        shape, dtype, device, num_repeat, True)
    error_list.append(error)

    shape = (32, 197, 768)
    dtype = torch.bfloat16
    print("-------------------------------------------")
    error = test_func(
        "lomem rms_norm", lambda x: functional.rms_norm(x, normalized_shape=shape[-1:], eps=eps),
        "torch rms_norm", lambda x: torch_rms_norm(x, normalized_shape=shape[-1:], eps=eps),
        shape, dtype, device, num_repeat, True)
    error_list.append(error)

    shape = (64, 197, 768)
    dtype = torch.float32
    print("-------------------------------------------")
    error = test_func(
        "lomem rms_norm", lambda x: functional.rms_norm(x, normalized_shape=shape[-1:], eps=eps),
        "torch rms_norm", lambda x: torch_rms_norm(x, normalized_shape=shape[-1:], eps=eps),
        shape, dtype, device, num_repeat, True)
    error_list.append(error)

    shape = (64, 197, 768)
    dtype = torch.float16
    print("-------------------------------------------")
    error = test_func(
        "lomem rms_norm", lambda x: functional.rms_norm(x, normalized_shape=shape[-1:], eps=eps),
        "torch rms_norm", lambda x: torch_rms_norm(x, normalized_shape=shape[-1:], eps=eps),
        shape, dtype, device, num_repeat, True)
    error_list.append(error)

    shape = (64, 197, 768)
    dtype = torch.bfloat16
    print("-------------------------------------------")
    error = test_func(
        "lomem rms_norm", lambda x: functional.rms_norm(x, normalized_shape=shape[-1:], eps=eps),
        "torch rms_norm", lambda x: torch_rms_norm(x, normalized_shape=shape[-1:], eps=eps),
        shape, dtype, device, num_repeat, True)
    error_list.append(error)


    shape = (1, 197, 767)
    dtype = torch.float32
    print("-------------------------------------------")
    error = test_func(
        "lomem rms_norm", lambda x: functional.rms_norm(x, normalized_shape=shape[-1:], eps=eps),
        "torch rms_norm", lambda x: torch_rms_norm(x, normalized_shape=shape[-1:], eps=eps),
        shape, dtype, device, num_repeat, True)
    error_list.append(error)

    shape = (1, 197, 767)
    dtype = torch.float16
    print("-------------------------------------------")
    error = test_func(
        "lomem rms_norm", lambda x: functional.rms_norm(x, normalized_shape=shape[-1:], eps=eps),
        "torch rms_norm", lambda x: torch_rms_norm(x, normalized_shape=shape[-1:], eps=eps),
        shape, dtype, device, num_repeat, True)
    error_list.append(error)

    shape = (1, 197, 767)
    dtype = torch.bfloat16
    print("-------------------------------------------")
    error = test_func(
        "lomem rms_norm", lambda x: functional.rms_norm(x, normalized_shape=shape[-1:], eps=eps),
        "torch rms_norm", lambda x: torch_rms_norm(x, normalized_shape=shape[-1:], eps=eps),
        shape, dtype, device, num_repeat, True)
    error_list.append(error)


    shape = (32, 256, 5120)
    dtype = torch.float32
    print("-------------------------------------------")
    error = test_func(
        "lomem rms_norm", lambda x: functional.rms_norm(x, normalized_shape=shape[-1:], eps=eps),
        "torch rms_norm", lambda x: torch_rms_norm(x, normalized_shape=shape[-1:], eps=eps),
        shape, dtype, device, num_repeat, True)
    error_list.append(error)

    shape = (32, 256, 5120)
    dtype = torch.float16
    print("-------------------------------------------")
    error = test_func(
        "lomem rms_norm", lambda x: functional.rms_norm(x, normalized_shape=shape[-1:], eps=eps),
        "torch rms_norm", lambda x: torch_rms_norm(x, normalized_shape=shape[-1:], eps=eps),
        shape, dtype, device, num_repeat, True)
    error_list.append(error)

    shape = (32, 256, 5120)
    dtype = torch.bfloat16
    print("-------------------------------------------")
    error = test_func(
        "lomem rms_norm", lambda x: functional.rms_norm(x, normalized_shape=shape[-1:], eps=eps),
        "torch rms_norm", lambda x: torch_rms_norm(x, normalized_shape=shape[-1:], eps=eps),
        shape, dtype, device, num_repeat, True)
    error_list.append(error)

    if max(error_list) < 5e-3:
        print(f"pass! max error: {max(error_list)}")
    else:
        print(f"max error: {error}")

    x_ = torch.rand(shape, dtype=dtype, device=device)
    x_1 = x_.clone().requires_grad_()
    x_2 = x_.clone().requires_grad_()
    x_3 = x_.clone().requires_grad_()

    y_1 = functional.rms_norm(x_1, normalized_shape=shape[-1:], eps=eps)
    y_2 = torch_rms_norm(x_2, normalized_shape=shape[-1:], eps=eps)
    y_3 = torch_rms_norm(x_3.to(torch.float64), normalized_shape=shape[-1:], eps=eps)

    y_1.sum().backward()
    y_2.sum().backward()
    y_3.sum().backward()

    print("forward: lomem - torch(fp64) =", (y_1 - y_3).abs().mean())
    print("forward: torch(bf16) - torch(fp64) =", (y_2 - y_3).abs().mean())

    print("backward: lomem - torch(fp64) =", (x_1.grad - x_3.grad).abs().mean())
    print("backward: torch(bf16) - torch(fp64) =", (x_2.grad - x_3.grad).abs().mean())