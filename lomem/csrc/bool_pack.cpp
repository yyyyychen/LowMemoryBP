#include "cutils.h"


torch::Tensor bool_to_uint8(torch::Tensor& input)
{
    at::cuda::CUDAGuard device_guard{(char)input.get_device()};

    torch::Tensor input_1d = input.flatten().contiguous();
    CHECK_BOOL(input);
    CHECK_DEVICE(input);
    CHECK_CONTIGUOUS(input);

    const int64_t N{input.numel()};
    torch::Tensor packed_flag = torch::empty({(N + 7) / 8}, input.options().dtype(torch::kUInt8));
    bool_pack_1d(N, input_1d.data_ptr(), packed_flag.data_ptr());
    return packed_flag;
}


torch::Tensor uint8_to_bool(torch::Tensor& input)
{
    at::cuda::CUDAGuard device_guard{(char)input.get_device()};

    torch::Tensor input_1d = input.flatten().contiguous();
    CHECK_UINT8(input);
    CHECK_DEVICE(input);
    CHECK_CONTIGUOUS(input);

    const int64_t N{input.numel()};
    torch::Tensor unpacked_flag = torch::empty({N * 8}, input.options().dtype(torch::kBool));
    bool_unpack_1d(N, input_1d.data_ptr(), unpacked_flag.data_ptr());
    return unpacked_flag;
}