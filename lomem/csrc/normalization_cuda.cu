#include "cutils.h"
#include "cudautils.cuh"


constexpr static int num_threads {128}; 
constexpr static int inner_repeat {8};


// template<typename T>
// __inline__ __device__ void Welford(T val, T* __restrict__ mean, T* __restrict__ m2, int* __restrict__ count) {
//     // Use Welford Online algorithem to compute mean and variance
//     // For more details you can refer to:
//     // https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
//     *count += 1;
//     T delta1 = val - *mean;
//     *mean += delta1 / (*count);
//     T delta2 = val - *mean;
//     *m2 += delta1 * delta2;
// }


// template<typename T>
// __inline__ __device__ void Welford(T b_mean, T b_m2, int b_count, T* __restrict__ mean, T* __restrict__ m2, int* __restrict__ count) {
//     if (b_count == 0) return;
//     int new_count = *count + b_count;
//     T nb_over_n = float(b_count) / float(new_count);
//     T na_over_n = float(*count) / float(new_count);
//     T delta = b_mean - *mean;
//     *mean = na_over_n * (*mean) + nb_over_n * b_mean;
//     *m2 += b_m2 + delta * delta * (*count) * nb_over_n;
//     *count = new_count;
// }


// template<typename T>
// __inline__ __device__ void WelfordWarpReduce(T* __restrict__ mean, T* __restrict__  m2, int* __restrict__ count) {
//     #pragma unroll
//     for (unsigned int offset = 16; offset > 0; offset >>= 1) {
//         T b_mean = __shfl_down_sync(0xffffffff, (float)*mean, offset, WarpSize);
//         T b_m2 = __shfl_down_sync(0xffffffff, (float)*m2, offset, WarpSize);
//         T b_count = __shfl_down_sync(0xffffffff, *count, offset, WarpSize);
//         Welford(b_mean, b_m2, b_count, mean, m2, count);
//     }
// }


// template<typename T>
// __inline__ __device__ void WelfordWarpAllReduce(T* __restrict__ mean, T* __restrict__ m2, int* __restrict__ count) {
//     //reduce to thread 0
//     WelfordWarpReduce<T>(mean, m2, count);

//     //broadcast from thread 0
//     *mean = __shfl_sync(0xffffffff, (float)*mean, 0, WarpSize);
//     *m2 = __shfl_sync(0xffffffff, (float)*m2, 0, WarpSize);
//     *count = __shfl_sync(0xffffffff, *count, 0, WarpSize);
// }


__inline__ __device__ void Welford(float val, float* __restrict__ mean, float* __restrict__ m2, int* __restrict__ count) {
    // Use Welford Online algorithem to compute mean and variance
    // For more details you can refer to:
    // https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
    *count += 1;
    float delta1 = val - *mean;
    *mean += delta1 / float(*count);
    float delta2 = val - *mean;
    *m2 += delta1 * delta2;
}


__inline__ __device__ void Welford(float b_mean, float b_m2, int b_count, float* __restrict__ mean, float* __restrict__ m2, int* __restrict__ count) {
    if (b_count == 0) return;
    int new_count = *count + b_count;
    float nb_over_n = float(b_count) / float(new_count);
    float na_over_n = float(*count) / float(new_count);
    float delta = b_mean - *mean;
    *mean = na_over_n * (*mean) + nb_over_n * b_mean;
    *m2 += b_m2 + delta * delta * (*count) * nb_over_n;
    *count = new_count;
}


__inline__ __device__ void WelfordWarpReduce(float* __restrict__ mean, float* __restrict__  m2, int* __restrict__ count) {
    #pragma unroll
    for (unsigned int offset = 16; offset > 0; offset >>= 1) {
        float b_mean = __shfl_down_sync(0xffffffff, *mean, offset, WarpSize);
        float b_m2 = __shfl_down_sync(0xffffffff, *m2, offset, WarpSize);
        float b_count = __shfl_down_sync(0xffffffff, *count, offset, WarpSize);
        Welford(b_mean, b_m2, b_count, mean, m2, count);
    }
}


__inline__ __device__ void WelfordWarpAllReduce(float* __restrict__ mean, float* __restrict__ m2, int* __restrict__ count) {
    //reduce to thread 0
    WelfordWarpReduce(mean, m2, count);

    //broadcast from thread 0
    *mean = __shfl_sync(0xffffffff, *mean, 0, WarpSize);
    *m2 = __shfl_sync(0xffffffff, *m2, 0, WarpSize);
    *count = __shfl_sync(0xffffffff, *count, 0, WarpSize);
}


__inline__ __device__ void Mean(float val, float* __restrict__ mean, int* __restrict__ count) {
    *count += 1;
    *mean += (val - *mean) / float(*count);
}


__inline__ __device__ void Mean(float b_mean, int b_count, float* __restrict__ mean, int* __restrict__ count) {
    if (b_count == 0) return;
    int new_count = *count + b_count;
    float nb_over_n = float(b_count) / float(new_count);
    float na_over_n = float(*count) / float(new_count);
    *mean = na_over_n * (*mean) + nb_over_n * b_mean;
    *count = new_count;
}


__inline__ __device__ void MeanWarpReduce(float* __restrict__  mean, int* __restrict__ count) {
    #pragma unroll
    for (unsigned int offset = 16; offset > 0; offset >>= 1) {
        float b_mean = __shfl_down_sync(0xffffffff, *mean, offset, WarpSize);
        float b_count = __shfl_down_sync(0xffffffff, *count, offset, WarpSize);
        Mean(b_mean, b_count, mean, count);
    }
}


__inline__ __device__ void MeanWarpAllReduce(float* __restrict__ mean, int* __restrict__ count) {
    //reduce to thread 0
    MeanWarpReduce(mean, count);

    //broadcast from thread 0
    *mean = __shfl_sync(0xffffffff, *mean, 0, WarpSize);
    *count = __shfl_sync(0xffffffff, *count, 0, WarpSize);
}


__inline__ __device__ void TwoMean(float val1, float val2, float* __restrict__ mean1, float* __restrict__ mean2, int* __restrict__ count) {
    *count += 1;
    *mean1 += (val1 - *mean1) / float(*count);
    *mean2 += (val2 - *mean2) / float(*count);
}


__inline__ __device__ void TwoMean(float b_mean1, float b_mean2, int b_count, float* __restrict__ mean1, float* __restrict__ mean2, int* __restrict__ count) {
    if (b_count == 0) return;
    int new_count = *count + b_count;
    *mean1 = (float(*count) * (*mean1) + float(b_count) * b_mean1) / float(new_count);
    *mean2 = (float(*count) * (*mean2) + float(b_count) * b_mean2) / float(new_count);
    *count = new_count;
}


__inline__ __device__ void TwoMeanWarpReduce(float* __restrict__ mean1, float* __restrict__ mean2, int* __restrict__ count) {
    #pragma unroll
    for (unsigned int offset = 16; offset > 0; offset >>= 1) {
        float b_mean1 = __shfl_down_sync(0xffffffff, *mean1, offset, WarpSize);
        float b_mean2 = __shfl_down_sync(0xffffffff, *mean2, offset, WarpSize);
        float b_count = __shfl_down_sync(0xffffffff, *count, offset, WarpSize);
        TwoMean(b_mean1, b_mean2, b_count, mean1, mean2, count);
    }
}


__inline__  __device__ void TwoMeanWarpAllReduce(float* __restrict__ mean1, float* __restrict__ mean2, int* __restrict__ count) {
    //reduce to thread 0
    TwoMeanWarpReduce(mean1, mean2, count);

    //broadcast from thread 0
    *mean1 = __shfl_sync(0xffffffff, *mean1, 0, WarpSize);
    *mean2 = __shfl_sync(0xffffffff, *mean2, 0, WarpSize);
    *count = __shfl_sync(0xffffffff, *count, 0, WarpSize);
}


template <typename T, int vec_size>
__global__ void
layer_norm_fw_2d_kernel
(int64_t M, int64_t N, float eps, T * __restrict__ input_ptr, T * __restrict__ output_ptr, float * __restrict__ rstd_ptr)
{
    constexpr int num_threads_n {32};
    constexpr int num_threads_m {num_threads / 32};
    constexpr int bM {num_threads_m * inner_repeat};
    const int gm_blk = bM * blockIdx.x;

    using vec_t = Pack<T, vec_size>;
    vec_t input_vec;
    vec_t output_vec;

    int gm_thr = gm_blk + threadIdx.y;
    #pragma unroll
    for (int r = 0; r < inner_repeat; ++r, gm_thr += num_threads_m) {
        if (gm_thr < M) {
            int count {0};
            float mean {0};
            float m2 {0};

            // Welford
            #pragma unroll 1
            for (int gn_thr = threadIdx.x * vec_size; gn_thr < N; gn_thr += 32 * vec_size) {
                const int gid = gm_thr * N + gn_thr;
                input_vec = *reinterpret_cast<vec_t*>(input_ptr + gid);
                #pragma unroll
                for (int k {0}; k < vec_size; ++k) {
                    Welford(input_vec.elem[k], &mean, &m2, &count);
                }
            }

            WelfordWarpAllReduce(&mean, &m2, &count);

            float rstd = rsqrt(m2 / count + eps);
            if (!threadIdx.x)
                *(rstd_ptr + gm_thr) = rstd;

            // output
            #pragma unroll 1
            for (int gn_thr = threadIdx.x * vec_size; gn_thr < N; gn_thr += 32 * vec_size) {
                const int gid = gm_thr * N + gn_thr;
                input_vec = *reinterpret_cast<vec_t*>(input_ptr + gid);
                #pragma unroll
                for (int k {0}; k < vec_size; ++k) {
                    output_vec.elem[k] = (float(input_vec.elem[k]) - mean) * rstd;
                }
                *reinterpret_cast<vec_t*>(output_ptr + gid) = output_vec;
            }
        }
    }
}


template <typename T, int vec_size>
__global__ void
layer_norm_bw_2d_kernel
(int64_t M, int64_t N, T * __restrict__ out_grad_ptr, T * __restrict__ output_ptr, float * __restrict__ rstd_ptr, T * __restrict__ in_grad_ptr)
{
    constexpr int num_threads_n {32};
    constexpr int num_threads_m {num_threads / 32};
    constexpr int bM {num_threads_m * inner_repeat};
    const int gm_blk = bM * blockIdx.x;

    using vec_t = Pack<T, vec_size>;
    vec_t out_grad_vec;
    vec_t output_vec;
    vec_t in_grad_vec;

    int gm_thr = gm_blk + threadIdx.y;
    #pragma unroll
    for (int r = 0; r < inner_repeat; ++r, gm_thr += num_threads_m) {
        if (gm_thr < M) {
            int count {0};
            float mean1 {0};
            float mean2 {0};

            #pragma unroll 1
            for (int gn_thr = threadIdx.x * vec_size; gn_thr < N; gn_thr += 32 * vec_size) {
                const int gid = gm_thr * N + gn_thr;
                out_grad_vec = *reinterpret_cast<vec_t*>(out_grad_ptr + gid);
                output_vec = *reinterpret_cast<vec_t*>(output_ptr + gid);
                #pragma unroll
                for (int k {0}; k < vec_size; ++k) {
                    TwoMean(
                        float(out_grad_vec.elem[k]),
                        float(out_grad_vec.elem[k]) * float(output_vec.elem[k]),
                        &mean1, &mean2, &count
                    );
                }
            }

            TwoMeanWarpAllReduce(&mean1, &mean2, &count);

            float rstd = *(rstd_ptr + gm_thr);
            mean1 *= rstd;
            mean2 *= rstd;

            // output
            #pragma unroll 1
            for (int gn_thr = threadIdx.x * vec_size; gn_thr < N; gn_thr += 32 * vec_size) {
                const int gid = gm_thr * N + gn_thr;
                out_grad_vec = *reinterpret_cast<vec_t*>(out_grad_ptr + gid);
                output_vec = *reinterpret_cast<vec_t*>(output_ptr + gid);
                #pragma unroll
                for (int k {0}; k < vec_size; ++k) {
                    in_grad_vec.elem[k] = float(out_grad_vec.elem[k]) * rstd - mean1 - mean2 * float(output_vec.elem[k]);
                }
                *reinterpret_cast<vec_t*>(in_grad_ptr + gid) = in_grad_vec;
            }
        }
    }
}


template <typename T>
void layer_norm_fw_2d_(int64_t M, int64_t N, float eps, void * input_ptr_, void * output_ptr_, void * rstd_ptr_)
{
    T * input_ptr = reinterpret_cast<T*>(input_ptr_);
    T * output_ptr = reinterpret_cast<T*>(output_ptr_);
    float * rstd_ptr = reinterpret_cast<float*>(rstd_ptr_);

    dim3 blockDim {32, num_threads / 32};
    constexpr int bM {num_threads / 32 * inner_repeat};
    dim3 gridDim {(M + bM - 1) / bM};

    if (check_align(input_ptr, 16, N) && check_align(output_ptr, 16, N)) {
        constexpr int vec_size {16 / sizeof(T)};
        layer_norm_fw_2d_kernel<T, vec_size><<<gridDim, blockDim>>>
            (M, N, eps, input_ptr, output_ptr, rstd_ptr);
    } else if (check_align(input_ptr, 8, N) && check_align(output_ptr, 8, N)) {
        constexpr int vec_size {8 / sizeof(T)};
        layer_norm_fw_2d_kernel<T, vec_size><<<gridDim, blockDim>>>
            (M, N, eps, input_ptr, output_ptr, rstd_ptr);
    } else if (check_align(input_ptr, 4, N) && check_align(output_ptr, 4, N)) {
        constexpr int vec_size {4 / sizeof(T)};
        layer_norm_fw_2d_kernel<T, vec_size><<<gridDim, blockDim>>>
            (M, N, eps, input_ptr, output_ptr, rstd_ptr);
    } else {
        constexpr int vec_size {1};
        layer_norm_fw_2d_kernel<T, vec_size><<<gridDim, blockDim>>>
            (M, N, eps, input_ptr, output_ptr, rstd_ptr);
    }
}


template <typename T>
void layer_norm_bw_2d_(int64_t M, int64_t N, void * out_grad_ptr_, void * output_ptr_, void * rstd_ptr_, void * in_grad_ptr_)
{
    T * out_grad_ptr = reinterpret_cast<T*>(out_grad_ptr_);
    T * output_ptr = reinterpret_cast<T*>(output_ptr_);
    float * rstd_ptr = reinterpret_cast<float*>(rstd_ptr_);
    T * in_grad_ptr = reinterpret_cast<T*>(in_grad_ptr_);

    dim3 blockDim {32, num_threads / 32};
    constexpr int bM {num_threads / 32 * inner_repeat};
    dim3 gridDim {(M + bM - 1) / bM};

    if (check_align(out_grad_ptr, 16, N) && check_align(output_ptr, 16, N) && check_align(in_grad_ptr, 16, N)) {
        constexpr int vec_size {16 / sizeof(T)};
        layer_norm_bw_2d_kernel<T, vec_size><<<gridDim, blockDim>>>
            (M, N, out_grad_ptr, output_ptr, rstd_ptr, in_grad_ptr);
    } else if (check_align(out_grad_ptr, 8, N) && check_align(output_ptr, 8, N) && check_align(in_grad_ptr, 8, N)) {
        constexpr int vec_size {8 / sizeof(T)};
        layer_norm_bw_2d_kernel<T, vec_size><<<gridDim, blockDim>>>
            (M, N, out_grad_ptr, output_ptr, rstd_ptr, in_grad_ptr);
    } else if (check_align(out_grad_ptr, 4, N) && check_align(output_ptr, 4, N) && check_align(in_grad_ptr, 4, N)) {
        constexpr int vec_size {4 / sizeof(T)};
        layer_norm_bw_2d_kernel<T, vec_size><<<gridDim, blockDim>>>
            (M, N, out_grad_ptr, output_ptr, rstd_ptr, in_grad_ptr);
    } else {
        constexpr int vec_size {1};
        layer_norm_bw_2d_kernel<T, vec_size><<<gridDim, blockDim>>>
            (M, N, out_grad_ptr, output_ptr, rstd_ptr, in_grad_ptr);
    }
}


template <typename T>
void layer_norm_fw_2d(int64_t M, int64_t N, float eps, void * input_ptr, void * output_ptr, void * rstd_ptr) {}

template <>
void layer_norm_fw_2d<float>(int64_t M, int64_t N, float eps, void * input_ptr, void * output_ptr, void * rstd_ptr)
{
    layer_norm_fw_2d_<float>(M, N, eps, input_ptr, output_ptr, rstd_ptr);
}

template <>
void layer_norm_fw_2d<half>(int64_t M, int64_t N, float eps, void * input_ptr, void * output_ptr, void * rstd_ptr)
{
    layer_norm_fw_2d_<half>(M, N, eps, input_ptr, output_ptr, rstd_ptr);
}

template <>
void layer_norm_fw_2d<nv_bfloat16>(int64_t M, int64_t N, float eps, void * input_ptr, void * output_ptr, void * rstd_ptr)
{
    layer_norm_fw_2d_<nv_bfloat16>(M, N, eps, input_ptr, output_ptr, rstd_ptr);
}


template <typename T>
void layer_norm_bw_2d(int64_t M, int64_t N, void * out_grad_ptr, void * output_ptr, void * rstd_ptr, void * in_grad_ptr) {}

template <>
void layer_norm_bw_2d<float>(int64_t M, int64_t N, void * out_grad_ptr, void * output_ptr, void * rstd_ptr, void * in_grad_ptr)
{
    layer_norm_bw_2d_<float>(M, N, out_grad_ptr, output_ptr, rstd_ptr, in_grad_ptr);
}

template <>
void layer_norm_bw_2d<half>(int64_t M, int64_t N, void * out_grad_ptr, void * output_ptr, void * rstd_ptr, void * in_grad_ptr)
{
    layer_norm_bw_2d_<half>(M, N, out_grad_ptr, output_ptr, rstd_ptr, in_grad_ptr);
}

template <>
void layer_norm_bw_2d<nv_bfloat16>(int64_t M, int64_t N, void * out_grad_ptr, void * output_ptr, void * rstd_ptr, void * in_grad_ptr)
{
    layer_norm_bw_2d_<nv_bfloat16>(M, N, out_grad_ptr, output_ptr, rstd_ptr, in_grad_ptr);
}