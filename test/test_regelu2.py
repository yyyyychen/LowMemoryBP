import time
import torch
from lomem import activation, packing


ReGELU2_a = [-0.04922261145617846, 1.0979632065417297, -0.048740595085551286]
ReGELU2_c = [-3.1858810036855245, -0.001178821281161997, 3.190832613414926]


@activation.apply_decorator
class regelu2_test(torch.autograd.Function):
    """
    This python-based implementation is only for checking the correctness of the CUDA-based implementation.
    For practical usage, please take lomem.activation.regelu2.
    """
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, x: torch.Tensor):
        out, packed_flag= activation._C.regelu2_fw(x)
        ctx.save_for_backward(packed_flag)
        return out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    @torch.autograd.function.once_differentiable
    def backward(ctx, out_grad: torch.Tensor):
        packed_flag = ctx.saved_tensors[0]
        flag = ((packed_flag.unsqueeze(-1) >> torch.arange(0, 8, 2, dtype=torch.uint8, device=device).unsqueeze(0)) & 3).flatten()[:out_grad.numel()].reshape(out_grad.shape)
        grad = out_grad * ((flag > 0) * ReGELU2_a[0] + (flag > 1) * ReGELU2_a[1] + (flag > 2) * ReGELU2_a[2])
        return grad


@activation.apply_decorator
class regelu2_ref(torch.autograd.Function):
    """
    This python-based implementation is only for checking the correctness of the CUDA-based implementation.
    For practical usage, please take lomem.activation.regelu2.
    """
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, x: torch.Tensor):
        flag_1 = packing.pack_bool_to_uint8(
            torch.logical_or(torch.logical_and(x > ReGELU2_c[0], x < ReGELU2_c[1]), x > ReGELU2_c[2])
        )
        flag_2 = packing.pack_bool_to_uint8(x > ReGELU2_c[1])
        ctx.save_for_backward(flag_1, flag_2)
        return torch.nn.functional.gelu(x)

    @staticmethod
    @torch.cuda.amp.custom_bwd
    @torch.autograd.function.once_differentiable
    def backward(ctx, out_grad: torch.Tensor):
        shape = out_grad.shape
        flag = packing.unpack_uint8_to_bool(ctx.saved_tensors[0], shape).to(torch.int8)
        flag += packing.unpack_uint8_to_bool(ctx.saved_tensors[1], shape).to(torch.int8) * 2
        grad = out_grad * ((flag > 0) * ReGELU2_a[0] + (flag > 1) * ReGELU2_a[1] + (flag > 2) * ReGELU2_a[2])
        return grad


def test_func(func1_name, func1, func2_name, func2, input_size, dtype, device, num_repeat):
    x = (torch.rand(input_size, dtype=dtype, device=device) - 0.5) * 20
    x_1 = x.clone().requires_grad_()
    x_2 = x.clone().requires_grad_()

    y_1 = func1(x_1)
    y_1.norm().backward()
    y_2 = func2(x_2)
    y_2.norm().backward()

    diff_fw = (y_1 - y_2).norm()
    diff_bw = (x_1.grad - x_2.grad).norm()

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

    print(f"problem size: {input_size} | dtype: {dtype}")
    print(f"{func1_name} time: {mean_time_1 * 1000} ms")
    print(f"{func2_name} time: {mean_time_2 * 1000} ms")
    print(f"forward diff: {diff_fw}")
    print(f"backward diff: {diff_bw}")
    print()



if __name__ == '__main__':
    device = 'cuda:1'
    num_repeat = 100

    shape = (64, 197, 768)
    dtype = torch.float32
    print("-------------------------------------------")
    test_func("lomem regelu2", activation.regelu2, "torch gelu", torch.nn.functional.gelu, shape, dtype, device, num_repeat)
    test_func("lomem regelu2", activation.regelu2, "py-ref regelu2", regelu2_ref, shape, dtype, device, num_repeat)


    shape = (64, 197, 768)
    dtype = torch.float16
    print("-------------------------------------------")
    test_func("lomem regelu2", activation.regelu2, "torch gelu", torch.nn.functional.gelu, shape, dtype, device, num_repeat)
    test_func("lomem regelu2", activation.regelu2, "py-ref regelu2", regelu2_ref, shape, dtype, device, num_repeat)


    shape = (64, 197, 768)
    dtype = torch.bfloat16
    print("-------------------------------------------")
    test_func("lomem regelu2", activation.regelu2, "torch gelu", torch.nn.functional.gelu, shape, dtype, device, num_repeat)
    test_func("lomem regelu2", activation.regelu2, "py-ref regelu2", regelu2_ref, shape, dtype, device, num_repeat)


    shape = (13, 13, 117)
    dtype = torch.float32
    print("-------------------------------------------")
    test_func("lomem regelu2", activation.regelu2, "torch gelu", torch.nn.functional.gelu, shape, dtype, device, num_repeat)
    test_func("lomem regelu2", activation.regelu2, "py-ref regelu2", regelu2_ref, shape, dtype, device, num_repeat)


    shape = (13, 13, 117)
    dtype = torch.float16
    print("-------------------------------------------")
    test_func("lomem regelu2", activation.regelu2, "torch gelu", torch.nn.functional.gelu, shape, dtype, device, num_repeat)
    test_func("lomem regelu2", activation.regelu2, "py-ref regelu2", regelu2_ref, shape, dtype, device, num_repeat)


    shape = (13, 13, 117)
    dtype = torch.bfloat16
    print("-------------------------------------------")
    test_func("lomem regelu2", activation.regelu2, "torch gelu", torch.nn.functional.gelu, shape, dtype, device, num_repeat)
    test_func("lomem regelu2", activation.regelu2, "py-ref regelu2", regelu2_ref, shape, dtype, device, num_repeat)
