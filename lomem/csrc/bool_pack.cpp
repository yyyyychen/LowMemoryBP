#include "cutils.h"


torch::Tensor bool_to_uint8(torch::Tensor& input)
{
    torch::Tensor input_1d = input.flatten().contiguous();
    CHECK_BOOL(input);
    CHECK_DEVICE(input);
    CHECK_CONTIGUOUS(input);

    at::cuda::CUDAGuard device_guard{(char)input.get_device()};
    const int64_t N{input.numel()};
    torch::Tensor packed_flag = torch::empty({(N + 7) / 8}, input.options().dtype(torch::kUInt8));
    bool_pack_1d(N, input_1d.mutable_data_ptr(), packed_flag.mutable_data_ptr());
    return packed_flag;
}


torch::Tensor uint8_to_bool(torch::Tensor& input)
{
    torch::Tensor input_1d = input.flatten().contiguous();
    CHECK_UINT8(input);
    CHECK_DEVICE(input);
    CHECK_CONTIGUOUS(input);

    at::cuda::CUDAGuard device_guard{(char)input.get_device()};
    const int64_t N{input.numel()};
    torch::Tensor unpacked_flag = torch::empty({N * 8}, input.options().dtype(torch::kBool));
    bool_unpack_1d(N, input_1d.mutable_data_ptr(), unpacked_flag.mutable_data_ptr());
    return unpacked_flag;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("bool_to_uint8", &bool_to_uint8, "pack bool tensor into uint8 tensor");
    m.def("uint8_to_bool", &uint8_to_bool, "unpack uint8 tensor into bool tensor");
    m.def("regelu2_fw", &regelu2_fw, "the forward pass of ReGELU2");
    m.def("regelu2_bw", &regelu2_bw, "the backward pass of ReGELU2");
}