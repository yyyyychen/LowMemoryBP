#include "cutils.h"


std::vector<torch::Tensor> layer_norm_fw(torch::Tensor& input, std::vector<int64_t>& normalized_shape, const float& eps)
{
    at::cuda::CUDAGuard device_guard{(char)input.get_device()};

    int64_t normalized_size = std::accumulate(
        normalized_shape.begin(), normalized_shape.end(), int64_t(1),
        std::multiplies<int64_t>()
    );

    torch::Tensor input_2d = input.reshape({-1, normalized_size}).contiguous();

    CHECK_DEVICE(input_2d);
    CHECK_CONTIGUOUS(input_2d);

    const int64_t M{input_2d.size(0)}, N{input_2d.size(1)};

    torch::Tensor output_2d = torch::empty({M, N}, input.options());
    torch::Tensor rstd = torch::empty({M}, input.options().dtype(torch::kFloat32));

    c10::ScalarType scalar_t = input.scalar_type();
    if (scalar_t == c10::ScalarType::Float) {
        layer_norm_fw_2d<float>(M, N, eps, input.data_ptr(), output_2d.data_ptr(), rstd.data_ptr());
    } else if (scalar_t == c10::ScalarType::Half) {
        layer_norm_fw_2d<half>(M, N, eps, input.data_ptr(), output_2d.data_ptr(), rstd.data_ptr());
    } else if (scalar_t == c10::ScalarType::BFloat16) {
        layer_norm_fw_2d<nv_bfloat16>(M, N, eps, input.data_ptr(), output_2d.data_ptr(), rstd.data_ptr());
    } else {
        std::cout << "layer_norm_fw only supports fp32, fp16 and bf16 type" << std::endl;
    }
    return {output_2d.reshape_as(input), rstd};
}


torch::Tensor layer_norm_bw(torch::Tensor& out_grad, torch::Tensor& output, torch::Tensor& rstd)
{
    at::cuda::CUDAGuard device_guard{(char)out_grad.get_device()};

    const int64_t M {rstd.size(0)}, N {out_grad.numel() / M};
    torch::Tensor out_grad_2d = out_grad.reshape({M, N}).contiguous();
    torch::Tensor output_2d = output.reshape({M, N}).contiguous();

    CHECK_DEVICE(out_grad_2d);
    CHECK_CONTIGUOUS(out_grad_2d);
    CHECK_DEVICE(output_2d);
    CHECK_CONTIGUOUS(output_2d);
    CHECK_DEVICE(rstd);
    CHECK_CONTIGUOUS(rstd);

    torch::Tensor in_grad_2d = torch::empty({M, N}, out_grad.options());

    c10::ScalarType scalar_t = output.scalar_type();
    if (scalar_t == c10::ScalarType::Float) {
        layer_norm_bw_2d<float>
            (M, N, out_grad_2d.data_ptr(), output_2d.data_ptr(), rstd.data_ptr(), in_grad_2d.data_ptr());
    } else if (scalar_t == c10::ScalarType::Half) {
        layer_norm_bw_2d<half>
            (M, N, out_grad_2d.data_ptr(), output_2d.data_ptr(), rstd.data_ptr(), in_grad_2d.data_ptr());
    } else if (scalar_t == c10::ScalarType::BFloat16) {
        layer_norm_bw_2d<nv_bfloat16>
            (M, N, out_grad_2d.data_ptr(), output_2d.data_ptr(), rstd.data_ptr(), in_grad_2d.data_ptr());
    } else {
        std::cout << "layer_norm_bw only supports fp32, fp16 and bf16 type" << std::endl;
    }
    return in_grad_2d.reshape_as(out_grad);
}