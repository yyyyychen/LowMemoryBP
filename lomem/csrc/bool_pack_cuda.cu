#include "cutils.h"
#include "cudautils.cuh"


constexpr static int num_threads {256}; 
constexpr static int inner_repeat {8};


template <int vec_size = 1>
__global__ void
bool_pack_1d_kernel(const int64_t N, bool * input_ptr, u_int8_t * output_ptr)
{
    const int gid_blk = num_threads * inner_repeat * vec_size * blockIdx.x;
    using vec_t = Pack<bool, vec_size>;

    uint8_t packed_flag{0};
    vec_t flag_vec{};
    int64_t gid{gid_blk + threadIdx.x * vec_size};
    #pragma unroll
    for (int r = 0; r < inner_repeat; ++r, gid += num_threads * vec_size) {
        if (gid < N) {
            flag_vec = *reinterpret_cast<vec_t*>(input_ptr + gid);
            packed_flag = 0;
            #pragma unroll
            for (int k = 0; k < vec_size; ++k) {
                packed_flag |= (flag_vec.elem[k] << k);
            }
            packflagWarpReduce<vec_size>(packed_flag);
            if (!(gid & 7)) *(output_ptr + gid / 8) = packed_flag;
        }
    }
}


void bool_pack_1d(int64_t N, void * input_ptr, void * output_ptr)
{
    bool * input_ptr_ = reinterpret_cast<bool*>(input_ptr);
    u_int8_t * output_ptr_ = reinterpret_cast<u_int8_t*>(output_ptr);
    dim3 blockDim{num_threads};
    if (check_align(input_ptr_, sizeof(bool) * 8, N)) {
        constexpr int blocksize = num_threads * inner_repeat * 8;
        dim3 gridDim{(N + blocksize - 1) / blocksize};
        bool_pack_1d_kernel<8><<<gridDim, blockDim>>>(N, input_ptr_, output_ptr_);
    } else if (check_align(input_ptr_, sizeof(bool) * 4, N)) {
        constexpr int blocksize = num_threads * inner_repeat * 4;
        dim3 gridDim{(N + blocksize - 1) / blocksize};
        bool_pack_1d_kernel<4><<<gridDim, blockDim>>>(N, input_ptr_, output_ptr_);
    } else if (check_align(input_ptr_, sizeof(bool) * 2, N)) {
        constexpr int blocksize = num_threads * inner_repeat * 2;
        dim3 gridDim{(N + blocksize - 1) / blocksize};
        bool_pack_1d_kernel<2><<<gridDim, blockDim>>>(N, input_ptr_, output_ptr_);
    } else {
        constexpr int blocksize = num_threads * inner_repeat;
        dim3 gridDim{(N + blocksize - 1) / blocksize};
        bool_pack_1d_kernel<1><<<gridDim, blockDim>>>(N, input_ptr_, output_ptr_);
    }
}


__global__ void
bool_unpack_1d_kernel(const int64_t N, u_int8_t * input_ptr, bool * output_ptr)
{
    const int gid_blk = num_threads * inner_repeat * blockIdx.x;
    using vec_t = Pack<bool, 8>;

    uint8_t packed_flag{0};
    vec_t unpacked_flag{};
    int64_t gid{gid_blk + threadIdx.x};
    #pragma unroll
    for (int r = 0; r < inner_repeat; ++r, gid += num_threads) {
        if (gid < N) {
            packed_flag = *(input_ptr + gid);
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                unpacked_flag.elem[k] = (packed_flag >> k) & 1;
            }
            *reinterpret_cast<vec_t*>(output_ptr + gid * 8) = unpacked_flag;
        }
    }
}


void bool_unpack_1d(int64_t N, void * input_ptr, void * output_ptr)
{
    dim3 blockDim{num_threads};
    constexpr int blocksize = num_threads * inner_repeat;
    dim3 gridDim{(N + blocksize - 1) / blocksize};
    bool_unpack_1d_kernel<<<gridDim, blockDim>>>
        (N, reinterpret_cast<u_int8_t*>(input_ptr), reinterpret_cast<bool*>(output_ptr));
}