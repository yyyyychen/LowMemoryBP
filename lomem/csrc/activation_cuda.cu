#include "cutils.h"
#include "cudautils.cuh"


constexpr static int num_threads {128}; 
constexpr static int inner_repeat {8};


__device__ static constexpr float ReGELU2_a [3] {-0.04922261145617846, 1.0979632065417297, -0.048740595085551286};
__device__ static constexpr float ReGELU2_c [3] {-3.1858810036855245, -0.001178821281161997, 3.190832613414926};

__device__ static constexpr float ReSiLU2_a [3] {-0.04060357190528599, 1.080925428529668, -0.040321856624382146};
__device__ static constexpr float ReSiLU2_c [3] {-6.3050461001646445, -0.0008684942046214787, 6.325815242089708};

__device__ static constexpr float inv_sqrt2 {0.7071067811865475};


// regelu2
template <typename T>
__inline__ __device__ void regelu2_fw(T x, T& y, uint8_t& flag)
{
    float x_ {x};
    y = (1.f + ::erf(x_ * inv_sqrt2)) * x_ * 0.5f;
    flag = (x_ > ReGELU2_c[0]) + (x_ > ReGELU2_c[1]) + (x_ > ReGELU2_c[2]);
}


template <typename T>
__inline__ __device__ void regelu2_bw(T out_grad, uint8_t flag, T& in_grad)
{
    float out_grad_ {out_grad};
    in_grad = out_grad_ * (
        (flag > 0) * ReGELU2_a[0] + (flag > 1) * ReGELU2_a[1] + (flag > 2) * ReGELU2_a[2]
    );
}


// resilu2
template <typename T>
__inline__ __device__ void resilu2_fw(T x, T& y, uint8_t& flag)
{
    float x_ {x};
    y = x_ / (1.f + ::expf(-x_));
    flag = (x_ > ReSiLU2_c[0]) + (x_ > ReSiLU2_c[1]) + (x_ > ReSiLU2_c[2]);
}


template <typename T>
__inline__ __device__ void resilu2_bw(T out_grad, uint8_t flag, T& in_grad)
{
    float out_grad_ {out_grad};
    in_grad = out_grad_ * (
        (flag > 0) * ReSiLU2_a[0] + (flag > 1) * ReSiLU2_a[1] + (flag > 2) * ReSiLU2_a[2]
    );
}


template <typename T, int vec_size>
__global__ void
regelu2_fw_1d_kernel
(int64_t N, T * __restrict__ input_ptr, T * __restrict__ output_ptr, u_int8_t * __restrict__ flag_ptr)
{
    static_assert(vec_size <= 4, "vector size must be less than 4");

    const int gid_blk = num_threads * inner_repeat * vec_size * blockIdx.x;
    using vec_t = Pack<T, vec_size>;

    uint8_t flag;
    uint8_t packed_flag;
    vec_t input_vec [2]; // double buffer
    vec_t output_vec;
    int64_t gid{gid_blk + threadIdx.x * vec_size};
    input_vec[0] = *reinterpret_cast<vec_t*>(input_ptr + gid);
    int write_buffer = 1;
    #pragma unroll
    for (int r = 0; r < inner_repeat; ++r) {
        int gid_next = gid + num_threads * vec_size;
        int read_buffer = write_buffer ^ 1;
        if (gid < N) {
            if (gid_next < N)
                input_vec[write_buffer] = *reinterpret_cast<vec_t*>(input_ptr + gid_next);
            packed_flag = 0;
            #pragma unroll
            for (int k = 0; k < vec_size; ++k) {
                regelu2_fw(input_vec[read_buffer].elem[k], output_vec.elem[k], flag);
                packed_flag |= (flag <<= (2 * (k & 3)));
            }
            packflagWarpReduce<vec_size * 2>(packed_flag);
            *reinterpret_cast<vec_t*>(output_ptr + gid) = output_vec;
            if (!(gid & 3)) *(flag_ptr + gid / 4) = packed_flag;
        }
        gid = gid_next;
        write_buffer ^= 1;
    }
}


template <typename T, int grad_vec_size, int flag_vec_size>
__global__ void
regelu2_bw_1d_kernel
(int64_t N, T * __restrict__ out_grad_ptr, uint8_t * __restrict__ packed_flag_ptr, T * __restrict__ in_grad_ptr)
{
    const int gid_blk = num_threads * inner_repeat * grad_vec_size * blockIdx.x;
    using grad_vec_t = Pack<T, grad_vec_size>;
    using flag_vec_t = Pack<uint8_t, flag_vec_size>;

    grad_vec_t out_grad_vec [2]; // double buffer
    grad_vec_t in_grad_vec;

    flag_vec_t flag_vec [2]; // double buffer

    int64_t gid{gid_blk + threadIdx.x * grad_vec_size};
    out_grad_vec[0] = *reinterpret_cast<grad_vec_t*>(out_grad_ptr + gid);
    flag_vec[0] = *reinterpret_cast<flag_vec_t*>(packed_flag_ptr + gid / 4);
    int write_buffer = 1;
    #pragma unroll
    for (int r = 0; r < inner_repeat; ++r) {
        int gid_next = gid + num_threads * grad_vec_size;
        int read_buffer = write_buffer ^ 1;
        if (gid < N) {
            if (gid_next < N) {
                out_grad_vec[write_buffer] = *reinterpret_cast<grad_vec_t*>(out_grad_ptr + gid_next);
                flag_vec[write_buffer] = *reinterpret_cast<flag_vec_t*>(packed_flag_ptr + gid_next / 4);
            }
            #pragma unroll
            for (int k = 0; k < grad_vec_size; ++k) {
                int gid_k = gid + k;
                regelu2_bw(
                    out_grad_vec[read_buffer].elem[k],
                    (flag_vec[read_buffer].elem[k/4] >> ((gid_k & 3) * 2)) & 3,
                    in_grad_vec.elem[k]
                );
            }
            *reinterpret_cast<grad_vec_t*>(in_grad_ptr + gid) = in_grad_vec;
        }
        gid = gid_next;
        write_buffer ^= 1;
    }
}


template <typename T>
void regelu2_fw_1d_(int64_t N, void * input_ptr_, void * output_ptr_, void * flag_ptr_)
{
    T * input_ptr = reinterpret_cast<T*>(input_ptr_);
    T * output_ptr = reinterpret_cast<T*>(output_ptr_);
    u_int8_t * flag_ptr = reinterpret_cast<u_int8_t*>(flag_ptr_);

    dim3 blockDim(num_threads);
    if ((16 / sizeof(T) <= 4) && check_align(input_ptr, 16, N) && check_align(output_ptr, 16, N)) {
        constexpr int vec_size {16 / sizeof(T)};
        if constexpr (vec_size <= 4) {
            constexpr int blocksize = num_threads * inner_repeat * vec_size;
            dim3 gridDim{(N + blocksize - 1) / blocksize};
            regelu2_fw_1d_kernel<T, vec_size><<<gridDim, blockDim>>>
                (N, input_ptr, output_ptr, flag_ptr);
        }
    } else if ((8 / sizeof(T) <= 4) && check_align(input_ptr, 8, N) && check_align(output_ptr, 8, N)) {
        constexpr int vec_size {8 / sizeof(T)};
        if constexpr (vec_size <= 4) {
            const int vec_size {8 / sizeof(T)};
            constexpr int blocksize = num_threads * inner_repeat * vec_size;
            dim3 gridDim{(N + blocksize - 1) / blocksize};
            regelu2_fw_1d_kernel<T, vec_size><<<gridDim, blockDim>>>
                (N, input_ptr, output_ptr, flag_ptr);
        }
    } else if ((4 / sizeof(T) <= 4) && check_align(input_ptr, 4, N) && check_align(output_ptr, 4, N)) {
        constexpr int vec_size {4 / sizeof(T)};
        if constexpr (vec_size <= 4) {
            const int vec_size {4 / sizeof(T)};
            constexpr int blocksize = num_threads * inner_repeat * vec_size;
            dim3 gridDim{(N + blocksize - 1) / blocksize};
            regelu2_fw_1d_kernel<T, vec_size><<<gridDim, blockDim>>>
                (N, input_ptr, output_ptr, flag_ptr);
        }
    } else{
        constexpr int vec_size {1};
        constexpr int blocksize = num_threads * inner_repeat * vec_size;
        dim3 gridDim{(N + blocksize - 1) / blocksize};
        regelu2_fw_1d_kernel<T, vec_size><<<gridDim, blockDim>>>
            (N, input_ptr, output_ptr, flag_ptr);
    }
}


template <typename T>
void regelu2_bw_1d_(int64_t N, void * out_grad_ptr_, void * packed_flag_ptr_, void * in_grad_ptr_)
{
    T * out_grad_ptr = reinterpret_cast<T*>(out_grad_ptr_);
    u_int8_t * packed_flag_ptr = reinterpret_cast<u_int8_t*>(packed_flag_ptr_);
    T * in_grad_ptr = reinterpret_cast<T*>(in_grad_ptr_);

    dim3 blockDim(num_threads);
    if (check_align(out_grad_ptr, 16, N) && check_align(in_grad_ptr, 16, N)) {
        constexpr int grad_vec_size {16 / sizeof(T)};
        constexpr int flag_vec_size = (grad_vec_size + 4 - 1) / 4; 

        constexpr int blocksize = num_threads * inner_repeat * grad_vec_size;
        dim3 gridDim{(N + blocksize - 1) / blocksize};
        regelu2_bw_1d_kernel<T, grad_vec_size, flag_vec_size><<<gridDim, blockDim>>>
            (N, out_grad_ptr, packed_flag_ptr, in_grad_ptr);

    } else if (check_align(out_grad_ptr, 8, N) && check_align(in_grad_ptr, 8, N)) {
        constexpr int grad_vec_size {8 / sizeof(T)};
        constexpr int flag_vec_size = (grad_vec_size + 4 - 1) / 4; 

        constexpr int blocksize = num_threads * inner_repeat * grad_vec_size;
        dim3 gridDim{(N + blocksize - 1) / blocksize};
        regelu2_bw_1d_kernel<T, grad_vec_size, flag_vec_size><<<gridDim, blockDim>>>
            (N, out_grad_ptr, packed_flag_ptr, in_grad_ptr);

    } else if (check_align(out_grad_ptr, 4, N) && check_align(in_grad_ptr, 4, N)) {
        constexpr int grad_vec_size {4 / sizeof(T)};
        constexpr int flag_vec_size = (grad_vec_size + 4 - 1) / 4; 

        constexpr int blocksize = num_threads * inner_repeat * grad_vec_size;
        dim3 gridDim{(N + blocksize - 1) / blocksize};
        regelu2_bw_1d_kernel<T, grad_vec_size, flag_vec_size><<<gridDim, blockDim>>>
            (N, out_grad_ptr, packed_flag_ptr, in_grad_ptr);

    } else{
        constexpr int grad_vec_size {1};
        constexpr int flag_vec_size {1}; 

        constexpr int blocksize = num_threads * inner_repeat * grad_vec_size;
        dim3 gridDim{(N + blocksize - 1) / blocksize};
        regelu2_bw_1d_kernel<T, grad_vec_size, flag_vec_size><<<gridDim, blockDim>>>
            (N, out_grad_ptr, packed_flag_ptr, in_grad_ptr);
    }
}


template <typename T>
void regelu2_fw_1d(int64_t N, void * input_ptr, void * output_ptr, void * flag_ptr) {}

template <>
void regelu2_fw_1d<float>(int64_t N, void * input_ptr, void * output_ptr, void * flag_ptr)
{
    regelu2_fw_1d_<float>(N, input_ptr, output_ptr, flag_ptr);
}

template <>
void regelu2_fw_1d<half>(int64_t N, void * input_ptr, void * output_ptr, void * flag_ptr)
{
    regelu2_fw_1d_<half>(N, input_ptr, output_ptr, flag_ptr);
}

template <>
void regelu2_fw_1d<nv_bfloat16>(int64_t N, void * input_ptr, void * output_ptr, void * flag_ptr)
{
    regelu2_fw_1d_<nv_bfloat16>(N, input_ptr, output_ptr, flag_ptr);
}


template <typename T>
void regelu2_bw_1d(int64_t N, void * out_grad_ptr, void * packed_flag_ptr, void * in_grad_ptr) {}

template <>
void regelu2_bw_1d<float>(int64_t N, void * out_grad_ptr, void * packed_flag_ptr, void * in_grad_ptr)
{
    regelu2_bw_1d_<float>(N, out_grad_ptr, packed_flag_ptr, in_grad_ptr);
}

template <>
void regelu2_bw_1d<half>(int64_t N, void * out_grad_ptr, void * packed_flag_ptr, void * in_grad_ptr)
{
    regelu2_bw_1d_<half>(N, out_grad_ptr, packed_flag_ptr, in_grad_ptr);
}

template <>
void regelu2_bw_1d<nv_bfloat16>(int64_t N, void * out_grad_ptr, void * packed_flag_ptr, void * in_grad_ptr)
{
    regelu2_bw_1d_<nv_bfloat16>(N, out_grad_ptr, packed_flag_ptr, in_grad_ptr);
}


template <typename T, int vec_size>
__global__ void
resilu2_fw_1d_kernel
(int64_t N, T * __restrict__ input_ptr, T * __restrict__ output_ptr, u_int8_t * __restrict__ flag_ptr)
{
    static_assert(vec_size <= 4, "vector size must be less than 4");

    const int gid_blk = num_threads * inner_repeat * vec_size * blockIdx.x;
    using vec_t = Pack<T, vec_size>;

    uint8_t flag;
    uint8_t packed_flag;
    vec_t input_vec [2]; // double buffer
    vec_t output_vec;
    int64_t gid{gid_blk + threadIdx.x * vec_size};
    input_vec[0] = *reinterpret_cast<vec_t*>(input_ptr + gid);
    int write_buffer = 1;
    #pragma unroll
    for (int r = 0; r < inner_repeat; ++r) {
        int gid_next = gid + num_threads * vec_size;
        int read_buffer = write_buffer ^ 1;
        if (gid < N) {
            if (gid_next < N)
                input_vec[write_buffer] = *reinterpret_cast<vec_t*>(input_ptr + gid_next);
            packed_flag = 0;
            #pragma unroll
            for (int k = 0; k < vec_size; ++k) {
                resilu2_fw(input_vec[read_buffer].elem[k], output_vec.elem[k], flag);
                packed_flag |= (flag <<= (2 * (k & 3)));
            }
            packflagWarpReduce<vec_size * 2>(packed_flag);
            *reinterpret_cast<vec_t*>(output_ptr + gid) = output_vec;
            if (!(gid & 3)) *(flag_ptr + gid / 4) = packed_flag;
        }
        gid = gid_next;
        write_buffer ^= 1;
    }
}


template <typename T, int grad_vec_size, int flag_vec_size>
__global__ void
resilu2_bw_1d_kernel
(int64_t N, T * __restrict__ out_grad_ptr, uint8_t * __restrict__ packed_flag_ptr, T * __restrict__ in_grad_ptr)
{
    const int gid_blk = num_threads * inner_repeat * grad_vec_size * blockIdx.x;
    using grad_vec_t = Pack<T, grad_vec_size>;
    using flag_vec_t = Pack<uint8_t, flag_vec_size>;

    grad_vec_t out_grad_vec [2]; // double buffer
    grad_vec_t in_grad_vec;

    flag_vec_t flag_vec [2]; // double buffer

    int64_t gid{gid_blk + threadIdx.x * grad_vec_size};
    out_grad_vec[0] = *reinterpret_cast<grad_vec_t*>(out_grad_ptr + gid);
    flag_vec[0] = *reinterpret_cast<flag_vec_t*>(packed_flag_ptr + gid / 4);
    int write_buffer = 1;
    #pragma unroll
    for (int r = 0; r < inner_repeat; ++r) {
        int gid_next = gid + num_threads * grad_vec_size;
        int read_buffer = write_buffer ^ 1;
        if (gid < N) {
            if (gid_next < N) {
                out_grad_vec[write_buffer] = *reinterpret_cast<grad_vec_t*>(out_grad_ptr + gid_next);
                flag_vec[write_buffer] = *reinterpret_cast<flag_vec_t*>(packed_flag_ptr + gid_next / 4);
            }
            #pragma unroll
            for (int k = 0; k < grad_vec_size; ++k) {
                int gid_k = gid + k;
                resilu2_bw(
                    out_grad_vec[read_buffer].elem[k],
                    (flag_vec[read_buffer].elem[k/4] >> ((gid_k & 3) * 2)) & 3,
                    in_grad_vec.elem[k]
                );
            }
            *reinterpret_cast<grad_vec_t*>(in_grad_ptr + gid) = in_grad_vec;
        }
        gid = gid_next;
        write_buffer ^= 1;
    }
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
            constexpr int blocksize = num_threads * inner_repeat * vec_size;
            dim3 gridDim{(N + blocksize - 1) / blocksize};
            resilu2_fw_1d_kernel<T, vec_size><<<gridDim, blockDim>>>
                (N, input_ptr, output_ptr, flag_ptr);
        }
    } else if ((8 / sizeof(T) <= 4) && check_align(input_ptr, 8, N) && check_align(output_ptr, 8, N)) {
        constexpr int vec_size {8 / sizeof(T)};
        if constexpr (vec_size <= 4) {
            const int vec_size {8 / sizeof(T)};
            constexpr int blocksize = num_threads * inner_repeat * vec_size;
            dim3 gridDim{(N + blocksize - 1) / blocksize};
            resilu2_fw_1d_kernel<T, vec_size><<<gridDim, blockDim>>>
                (N, input_ptr, output_ptr, flag_ptr);
        }
    } else if ((4 / sizeof(T) <= 4) && check_align(input_ptr, 4, N) && check_align(output_ptr, 4, N)) {
        constexpr int vec_size {4 / sizeof(T)};
        if constexpr (vec_size <= 4) {
            const int vec_size {4 / sizeof(T)};
            constexpr int blocksize = num_threads * inner_repeat * vec_size;
            dim3 gridDim{(N + blocksize - 1) / blocksize};
            resilu2_fw_1d_kernel<T, vec_size><<<gridDim, blockDim>>>
                (N, input_ptr, output_ptr, flag_ptr);
        }
    } else {
        constexpr int vec_size {1};
        constexpr int blocksize = num_threads * inner_repeat * vec_size;
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

        constexpr int blocksize = num_threads * inner_repeat * grad_vec_size;
        dim3 gridDim{(N + blocksize - 1) / blocksize};
        resilu2_bw_1d_kernel<T, grad_vec_size, flag_vec_size><<<gridDim, blockDim>>>
            (N, out_grad_ptr, packed_flag_ptr, in_grad_ptr);

    } else if (check_align(out_grad_ptr, 8, N) && check_align(in_grad_ptr, 8, N)) {
        constexpr int grad_vec_size {8 / sizeof(T)};
        constexpr int flag_vec_size = (grad_vec_size + 4 - 1) / 4; 

        constexpr int blocksize = num_threads * inner_repeat * grad_vec_size;
        dim3 gridDim{(N + blocksize - 1) / blocksize};
        resilu2_bw_1d_kernel<T, grad_vec_size, flag_vec_size><<<gridDim, blockDim>>>
            (N, out_grad_ptr, packed_flag_ptr, in_grad_ptr);

    } else if (check_align(out_grad_ptr, 4, N) && check_align(in_grad_ptr, 4, N)) {
        constexpr int grad_vec_size {4 / sizeof(T)};
        constexpr int flag_vec_size = (grad_vec_size + 4 - 1) / 4; 

        constexpr int blocksize = num_threads * inner_repeat * grad_vec_size;
        dim3 gridDim{(N + blocksize - 1) / blocksize};
        resilu2_bw_1d_kernel<T, grad_vec_size, flag_vec_size><<<gridDim, blockDim>>>
            (N, out_grad_ptr, packed_flag_ptr, in_grad_ptr);

    } else{
        constexpr int grad_vec_size {1};
        constexpr int flag_vec_size {1}; 

        constexpr int blocksize = num_threads * inner_repeat * grad_vec_size;
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