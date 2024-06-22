#include "cutils.h"


std::vector<torch::Tensor> layer_norm_fw(torch::Tensor& input, std::vector<int64_t>& normalized_shape, const float& eps) {
    int64_t normalized_size = std::accumulate(
        normalized_shape.begin(), normalized_shape.end(), int64_t(1),
        std::multiplies<int64_t>()
    );

    torch::Tensor input_2d = input.reshape({-1, normalized_size}).contiguous();

    CHECK_DEVICE(input_2d);
    CHECK_CONTIGUOUS(input_2d);

    at::cuda::CUDAGuard device_guard{(char)input.get_device()};
    const int64_t M{input_2d.size(0)}, N{input_2d.size(1)};

    torch::Tensor output_2d = torch::empty({M, N}, input.options());
    torch::Tensor rstd = torch::empty({M}, input.options());

    c10::ScalarType scalar_t = input.scalar_type();
    if (scalar_t == c10::ScalarType::Float) {
        layer_norm_fw_2d<float>(M, N, eps, input.mutable_data_ptr(), output_2d.mutable_data_ptr(), rstd.mutable_data_ptr());
    } else if (scalar_t == c10::ScalarType::Half) {
        layer_norm_fw_2d<half>(M, N, eps, input.mutable_data_ptr(), output_2d.mutable_data_ptr(), rstd.mutable_data_ptr());
    } else if (scalar_t == c10::ScalarType::BFloat16) {
        layer_norm_fw_2d<nv_bfloat16>(M, N, eps, input.mutable_data_ptr(), output_2d.mutable_data_ptr(), rstd.mutable_data_ptr());
    } else {
        std::cout << "layer_norm_fw only supports fp32, fp16 and bf16 type" << std::endl;
    }
    return {output_2d.reshape_as(input), rstd};
}
