#include "cutils.h"
#include "cudautils.cuh"


constexpr static int num_threads {512}; 


__device__ static constexpr float ReSiLU2_a [3] {-0.04060357190528599, 1.080925428529668, -0.040321856624382146};
__device__ static constexpr float ReSiLU2_c [3] {-6.3050461001646445, -0.0008684942046214787, 6.325815242089708};


// resilu2
template <typename T>
__inline__ __device__ void resilu2_fw(float x, T& y, uint8_t& flag)
{
    y = x / (1.f + ::expf(-x));
    flag = uint8_t(x > ReSiLU2_c[0]) + uint8_t(x > ReSiLU2_c[1]) + uint8_t(x > ReSiLU2_c[2]);
}


template <typename T>
__inline__ __device__ void resilu2_bw(float out_grad, uint8_t flag, T& in_grad)
{
    in_grad = out_grad * (
        float(flag > 0) * ReSiLU2_a[0] + float(flag > 1) * ReSiLU2_a[1] + float(flag > 2) * ReSiLU2_a[2]
    );
}


template <typename T, int vec_size>
__global__ void __launch_bounds__(num_threads)
resilu2_fw_1d_kernel
(int64_t N, T * __restrict__ input_ptr, T * __restrict__ output_ptr, u_int8_t * __restrict__ flag_ptr)
{
    const int gid_blk = num_threads * vec_size * blockIdx.x;
    using vec_t = Pack<T, vec_size>;

    int64_t gid{gid_blk + threadIdx.x * vec_size};
    if (gid >= N)
        return;

    uint8_t flag;
    uint8_t packed_flag;
    vec_t input_vec;
    vec_t output_vec;

    input_vec = *reinterpret_cast<vec_t*>(input_ptr + gid);

    packed_flag = 0;
    #pragma unroll
    for (int k = 0; k < vec_size; ++k) {
        resilu2_fw(input_vec.elem[k], output_vec.elem[k], flag);
        packed_flag |= (flag <<= (2 * (k & 3)));
    }
    packflagWarpReduce<vec_size * 2>(packed_flag);
    *reinterpret_cast<vec_t*>(output_ptr + gid) = output_vec;
    if (!(gid & 3)) *(flag_ptr + gid / 4) = packed_flag;
}


template <typename T, int grad_vec_size, int flag_vec_size>
__global__ void __launch_bounds__(num_threads)
resilu2_bw_1d_kernel
(int64_t N, T * __restrict__ out_grad_ptr, uint8_t * __restrict__ packed_flag_ptr, T * __restrict__ in_grad_ptr)
{
    const int gid_blk = num_threads * grad_vec_size * blockIdx.x;
    using grad_vec_t = Pack<T, grad_vec_size>;
    using flag_vec_t = Pack<uint8_t, flag_vec_size>;

    grad_vec_t out_grad_vec;
    grad_vec_t in_grad_vec;

    flag_vec_t flag_vec;

    int64_t gid{gid_blk + threadIdx.x * grad_vec_size};
    if (gid >= N)
        return;

    out_grad_vec = *reinterpret_cast<grad_vec_t*>(out_grad_ptr + gid);
    flag_vec = *reinterpret_cast<flag_vec_t*>(packed_flag_ptr + gid / 4);

    #pragma unroll
    for (int k = 0; k < grad_vec_size; ++k) {
        int gid_k = gid + k;
        resilu2_bw(
            out_grad_vec.elem[k],
            (flag_vec.elem[k/4] >> ((gid_k & 3) * 2)) & 3,
            in_grad_vec.elem[k]
        );
    }
    *reinterpret_cast<grad_vec_t*>(in_grad_ptr + gid) = in_grad_vec;
}


template <typename T>
void resilu2_fw_1d_(int64_t N, void * input_ptr_, void * output_ptr_, void * flag_ptr_)
{
    T * input_ptr = reinterpret_cast<T*>(input_ptr_);
    T * output_ptr = reinterpret_cast<T*>(output_ptr_);
    u_int8_t * flag_ptr = reinterpret_cast<u_int8_t*>(flag_ptr_);

    dim3 blockDim(num_threads);
    if ((16 / sizeof(T) <= 4) && check_align(input_ptr, 16, N) && check_align(output_ptr, 16, N)) {
        constexpr int vec_size {16 / sizeof(T)};
        if constexpr (vec_size <= 4) {
            constexpr int blocksize = num_threads * vec_size;
            dim3 gridDim{(N + blocksize - 1) / blocksize};
            resilu2_fw_1d_kernel<T, vec_size><<<gridDim, blockDim>>>
                (N, input_ptr, output_ptr, flag_ptr);
        }
    } else if ((8 / sizeof(T) <= 4) && check_align(input_ptr, 8, N) && check_align(output_ptr, 8, N)) {
        constexpr int vec_size {8 / sizeof(T)};
        if constexpr (vec_size <= 4) {
            const int vec_size {8 / sizeof(T)};
            constexpr int blocksize = num_threads * vec_size;
            dim3 gridDim{(N + blocksize - 1) / blocksize};
            resilu2_fw_1d_kernel<T, vec_size><<<gridDim, blockDim>>>
                (N, input_ptr, output_ptr, flag_ptr);
        }
    } else if ((4 / sizeof(T) <= 4) && check_align(input_ptr, 4, N) && check_align(output_ptr, 4, N)) {
        constexpr int vec_size {4 / sizeof(T)};
        if constexpr (vec_size <= 4) {
            const int vec_size {4 / sizeof(T)};
            constexpr int blocksize = num_threads * vec_size;
            dim3 gridDim{(N + blocksize - 1) / blocksize};
            resilu2_fw_1d_kernel<T, vec_size><<<gridDim, blockDim>>>
                (N, input_ptr, output_ptr, flag_ptr);
        }
    } else {
        constexpr int vec_size {1};
        constexpr int blocksize = num_threads * vec_size;
        dim3 gridDim{(N + blocksize - 1) / blocksize};
        resilu2_fw_1d_kernel<T, vec_size><<<gridDim, blockDim>>>
            (N, input_ptr, output_ptr, flag_ptr);
    }
}


template <typename T>
void resilu2_bw_1d_(int64_t N, void * out_grad_ptr_, void * packed_flag_ptr_, void * in_grad_ptr_)
{
    T * out_grad_ptr = reinterpret_cast<T*>(out_grad_ptr_);
    u_int8_t * packed_flag_ptr = reinterpret_cast<u_int8_t*>(packed_flag_ptr_);
    T * in_grad_ptr = reinterpret_cast<T*>(in_grad_ptr_);

    dim3 blockDim(num_threads);
    if (check_align(out_grad_ptr, 16, N) && check_align(in_grad_ptr, 16, N)) {
        constexpr int grad_vec_size {16 / sizeof(T)};
        constexpr int flag_vec_size = (grad_vec_size + 4 - 1) / 4; 

        constexpr int blocksize = num_threads * grad_vec_size;
        dim3 gridDim{(N + blocksize - 1) / blocksize};
        resilu2_bw_1d_kernel<T, grad_vec_size, flag_vec_size><<<gridDim, blockDim>>>
            (N, out_grad_ptr, packed_flag_ptr, in_grad_ptr);

    } else if (check_align(out_grad_ptr, 8, N) && check_align(in_grad_ptr, 8, N)) {
        constexpr int grad_vec_size {8 / sizeof(T)};
        constexpr int flag_vec_size = (grad_vec_size + 4 - 1) / 4; 

        constexpr int blocksize = num_threads * grad_vec_size;
        dim3 gridDim{(N + blocksize - 1) / blocksize};
        resilu2_bw_1d_kernel<T, grad_vec_size, flag_vec_size><<<gridDim, blockDim>>>
            (N, out_grad_ptr, packed_flag_ptr, in_grad_ptr);

    } else if (check_align(out_grad_ptr, 4, N) && check_align(in_grad_ptr, 4, N)) {
        constexpr int grad_vec_size {4 / sizeof(T)};
        constexpr int flag_vec_size = (grad_vec_size + 4 - 1) / 4; 

        constexpr int blocksize = num_threads * grad_vec_size;
        dim3 gridDim{(N + blocksize - 1) / blocksize};
        resilu2_bw_1d_kernel<T, grad_vec_size, flag_vec_size><<<gridDim, blockDim>>>
            (N, out_grad_ptr, packed_flag_ptr, in_grad_ptr);

    } else{
        constexpr int grad_vec_size {1};
        constexpr int flag_vec_size {1}; 

        constexpr int blocksize = num_threads * grad_vec_size;
        dim3 gridDim{(N + blocksize - 1) / blocksize};
        resilu2_bw_1d_kernel<T, grad_vec_size, flag_vec_size><<<gridDim, blockDim>>>
            (N, out_grad_ptr, packed_flag_ptr, in_grad_ptr);
    }
}


template <typename T>
void resilu2_fw_1d(int64_t N, void * input_ptr, void * output_ptr, void * flag_ptr) {}

template <>
void resilu2_fw_1d<float>(int64_t N, void * input_ptr, void * output_ptr, void * flag_ptr)
{
    resilu2_fw_1d_<float>(N, input_ptr, output_ptr, flag_ptr);
}

template <>
void resilu2_fw_1d<half>(int64_t N, void * input_ptr, void * output_ptr, void * flag_ptr)
{
    resilu2_fw_1d_<half>(N, input_ptr, output_ptr, flag_ptr);
}

template <>
void resilu2_fw_1d<nv_bfloat16>(int64_t N, void * input_ptr, void * output_ptr, void * flag_ptr)
{
    resilu2_fw_1d_<nv_bfloat16>(N, input_ptr, output_ptr, flag_ptr);
}


template <typename T>
void resilu2_bw_1d(int64_t N, void * out_grad_ptr, void * packed_flag_ptr, void * in_grad_ptr) {}

template <>
void resilu2_bw_1d<float>(int64_t N, void * out_grad_ptr, void * packed_flag_ptr, void * in_grad_ptr)
{
    resilu2_bw_1d_<float>(N, out_grad_ptr, packed_flag_ptr, in_grad_ptr);
}

template <>
void resilu2_bw_1d<half>(int64_t N, void * out_grad_ptr, void * packed_flag_ptr, void * in_grad_ptr)
{
    resilu2_bw_1d_<half>(N, out_grad_ptr, packed_flag_ptr, in_grad_ptr);
}

template <>
void resilu2_bw_1d<nv_bfloat16>(int64_t N, void * out_grad_ptr, void * packed_flag_ptr, void * in_grad_ptr)
{
    resilu2_bw_1d_<nv_bfloat16>(N, out_grad_ptr, packed_flag_ptr, in_grad_ptr);
}