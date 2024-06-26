#include "cutils.h"


std::vector<torch::Tensor> regelu2_fw(torch::Tensor& input)
{
    at::cuda::CUDAGuard device_guard{(char)input.get_device()};

    torch::Tensor input_1d = input.flatten().contiguous();
    CHECK_DEVICE(input);
    CHECK_CONTIGUOUS(input);

    const int64_t N{input.numel()};
    torch::Tensor output_1d = torch::empty({N}, input.options());
    torch::Tensor flag = torch::empty({(N + 3) / 4}, input.options().dtype(torch::kUInt8));

    c10::ScalarType scalar_t = input.scalar_type();
    if (scalar_t == c10::ScalarType::Float) {
        regelu2_fw_1d<float>(N, input.data_ptr(), output_1d.data_ptr(), flag.data_ptr());
    } else if (scalar_t == c10::ScalarType::Half) {
        regelu2_fw_1d<half>(N, input.data_ptr(), output_1d.data_ptr(), flag.data_ptr());
    } else if (scalar_t == c10::ScalarType::BFloat16) {
        regelu2_fw_1d<nv_bfloat16>(N, input.data_ptr(), output_1d.data_ptr(), flag.data_ptr());
    } else {
        std::cout << "regelu2_fw only supports fp32, fp16 and bf16 type" << std::endl;
    }
    return {output_1d.reshape_as(input), flag};
}


torch::Tensor regelu2_bw(torch::Tensor& out_grad, torch::Tensor& packed_flag)
{
    at::cuda::CUDAGuard device_guard{(char)out_grad.get_device()};

    torch::Tensor out_grad_1d = out_grad.flatten().contiguous();
    CHECK_DEVICE(out_grad_1d);
    CHECK_CONTIGUOUS(out_grad_1d);

    CHECK_DEVICE(packed_flag);
    CHECK_CONTIGUOUS(packed_flag);

    const int64_t N{out_grad.numel()};

    torch::Tensor in_grad_1d = torch::empty({N}, out_grad.options());

    c10::ScalarType scalar_t = out_grad.scalar_type();
    if (scalar_t == c10::ScalarType::Float) {
        regelu2_bw_1d<float>(N, out_grad_1d.data_ptr(), packed_flag.data_ptr(), in_grad_1d.data_ptr());
    } else if (scalar_t == c10::ScalarType::Half) {
        regelu2_bw_1d<half>(N, out_grad_1d.data_ptr(), packed_flag.data_ptr(), in_grad_1d.data_ptr());
    } else if (scalar_t == c10::ScalarType::BFloat16) {
        regelu2_bw_1d<nv_bfloat16>(N, out_grad_1d.data_ptr(), packed_flag.data_ptr(), in_grad_1d.data_ptr());
    } else {
        std::cout << "regelu2_bw only supports fp32, fp16 and bf16 type" << std::endl;
    }
    return in_grad_1d.reshape_as(out_grad);
}