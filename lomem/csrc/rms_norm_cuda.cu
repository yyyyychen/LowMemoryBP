#include "cutils.h"
#include "cudautils.cuh"


constexpr static int num_threads {128}; 
constexpr static int inner_repeat {8};
static constexpr int Max_sharedColumn = 4096;


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


template <typename T, int vec_size>
__global__ void
rms_norm_fw_2d_kernel
(int64_t M, int64_t N, float eps,
T * __restrict__ input_ptr,
T * __restrict__ output_ptr,
float * __restrict__ rstd_ptr, const int sN)
{
    extern __shared__ float smem_data [];

    int smem_head_id = threadIdx.y * sN;
    const int smem_stride = sN / vec_size;

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
            float mean2 {0};

            const int gid_start = gm_thr * N;

            // Welford
            #pragma unroll 1
            for (int gn_thr = threadIdx.x * vec_size; gn_thr < sN; gn_thr += 32 * vec_size) {
                const int gid = gid_start + gn_thr;
                input_vec = *reinterpret_cast<vec_t*>(input_ptr + gid);
                #pragma unroll
                for (int k {0}; k < vec_size; ++k) {
                    float input_elem {input_vec.elem[k]};
                    Mean(input_elem * input_elem, &mean2, &count);
                    smem_data[smem_head_id + smem_stride * k + gn_thr / vec_size] = input_elem;
                }
            }

            #pragma unroll 1
            for (int gn_thr = threadIdx.x * vec_size + sN; gn_thr < N; gn_thr += 32 * vec_size) {
                const int gid = gid_start + gn_thr;
                input_vec = *reinterpret_cast<vec_t*>(input_ptr + gid);
                #pragma unroll
                for (int k {0}; k < vec_size; ++k) {
                    float input_elem {input_vec.elem[k]};
                    Mean(input_elem * input_elem, &mean2, &count);
                }
            }

            MeanWarpAllReduce(&mean2, &count);

            float rstd = rsqrt(mean2 + eps);
            if (!threadIdx.x)
                *(rstd_ptr + gm_thr) = rstd;

            // output
            #pragma unroll 1
            for (int gn_thr = threadIdx.x * vec_size; gn_thr < sN; gn_thr += 32 * vec_size) {
                const int gid = gid_start + gn_thr;
                #pragma unroll
                for (int k {0}; k < vec_size; ++k) {
                    float input_elem {smem_data[smem_head_id + smem_stride * k + gn_thr / vec_size]};
                    output_vec.elem[k] = input_elem * rstd;
                }
                *reinterpret_cast<vec_t*>(output_ptr + gid) = output_vec;
            }

            #pragma unroll 1
            for (int gn_thr = threadIdx.x * vec_size + sN; gn_thr < N; gn_thr += 32 * vec_size) {
                const int gid = gid_start + gn_thr;
                input_vec = *reinterpret_cast<vec_t*>(input_ptr + gid);
                #pragma unroll
                for (int k {0}; k < vec_size; ++k) {
                    float input_elem {input_vec.elem[k]};
                    output_vec.elem[k] = input_elem * rstd;
                }
                *reinterpret_cast<vec_t*>(output_ptr + gid) = output_vec;
            }
        }
    }
}


template <typename T, int vec_size>
__global__ void
rms_norm_bw_2d_kernel
(int64_t M, int64_t N,
T * __restrict__ out_grad_ptr, T * __restrict__ output_ptr,
float * __restrict__ rstd_ptr, T * __restrict__ in_grad_ptr, const int sN)
{
    extern __shared__ float smem_data [];

    int smem_head_id = threadIdx.y * sN;
    const int smem_stride = sN / vec_size;

    constexpr int num_threads_n {32};
    constexpr int num_threads_m {num_threads / 2 / 32};
    constexpr int bM {num_threads_m * inner_repeat};
    const int gm_blk = bM * blockIdx.x;

    float * smem_out_grad {(float *)smem_data};
    float * smem_output {(float *)smem_data + sN * num_threads_m};

    using vec_t = Pack<T, vec_size>;
    vec_t out_grad_vec;
    vec_t output_vec;
    vec_t in_grad_vec;

    int gm_thr = gm_blk + threadIdx.y;
    #pragma unroll
    for (int r = 0; r < inner_repeat; ++r, gm_thr += num_threads_m) {
        if (gm_thr < M) {
            int count {0};
            float mean2 {0};

            const int gid_start {gm_thr * N};

            #pragma unroll 1
            for (int gn_thr = threadIdx.x * vec_size; gn_thr < sN; gn_thr += 32 * vec_size) {
                const int gid = gid_start + gn_thr;
                out_grad_vec = *reinterpret_cast<vec_t*>(out_grad_ptr + gid);
                output_vec = *reinterpret_cast<vec_t*>(output_ptr + gid);
                #pragma unroll
                for (int k {0}; k < vec_size; ++k) {
                    float out_grad_elem {out_grad_vec.elem[k]};
                    float output_elem {output_vec.elem[k]};
                    int sid = smem_head_id + k * smem_stride + gn_thr / vec_size;
                    smem_out_grad[sid] = out_grad_elem;
                    smem_output[sid] = output_elem;
                    Mean(out_grad_elem * output_elem, &mean2, &count);
                }
            }

            #pragma unroll 1
            for (int gn_thr = threadIdx.x * vec_size + sN; gn_thr < N; gn_thr += 32 * vec_size) {
                const int gid = gid_start + gn_thr;
                out_grad_vec = *reinterpret_cast<vec_t*>(out_grad_ptr + gid);
                output_vec = *reinterpret_cast<vec_t*>(output_ptr + gid);
                #pragma unroll
                for (int k {0}; k < vec_size; ++k) {
                    float out_grad_elem {out_grad_vec.elem[k]};
                    float output_elem {output_vec.elem[k]};
                    Mean(out_grad_elem * output_elem, &mean2, &count);
                }
            }

            MeanWarpAllReduce(&mean2, &count);

            float rstd = *(rstd_ptr + gm_thr);
            mean2 *= rstd;

            // output
            #pragma unroll 1
            for (int gn_thr = threadIdx.x * vec_size; gn_thr < sN; gn_thr += 32 * vec_size) {
                const int gid = gid_start + gn_thr;
                #pragma unroll
                for (int k {0}; k < vec_size; ++k) {
                    int sid = smem_head_id + k * smem_stride + gn_thr / vec_size;
                    float out_grad_elem {smem_out_grad[sid]};
                    float output_elem {smem_output[sid]};
                    in_grad_vec.elem[k] = out_grad_elem * rstd - mean2 * output_elem;
                }
                *reinterpret_cast<vec_t*>(in_grad_ptr + gid) = in_grad_vec;
            }

            // output
            #pragma unroll 1
            for (int gn_thr = threadIdx.x * vec_size + sN; gn_thr < N; gn_thr += 32 * vec_size) {
                const int gid = gid_start + gn_thr;
                out_grad_vec = *reinterpret_cast<vec_t*>(out_grad_ptr + gid);
                output_vec = *reinterpret_cast<vec_t*>(output_ptr + gid);
                #pragma unroll
                for (int k {0}; k < vec_size; ++k) {
                    float out_grad_elem {out_grad_vec.elem[k]};
                    float output_elem {output_vec.elem[k]};
                    in_grad_vec.elem[k] = out_grad_elem * rstd - mean2 * output_elem;
                }
                *reinterpret_cast<vec_t*>(in_grad_ptr + gid) = in_grad_vec;
            }
        }
    }
}


template <typename T>
void rms_norm_fw_2d_(int64_t M, int64_t N, float eps, void * input_ptr_, void * output_ptr_, void * rstd_ptr_)
{
    T * input_ptr = reinterpret_cast<T*>(input_ptr_);
    T * output_ptr = reinterpret_cast<T*>(output_ptr_);
    float * rstd_ptr = reinterpret_cast<float*>(rstd_ptr_);

    dim3 blockDim {32, num_threads / 32};
    constexpr int bM {num_threads / 32 * inner_repeat};
    dim3 gridDim {(M + bM - 1) / bM};

    int sN = min(Max_sharedColumn, int(N));
    int smemsize = (num_threads / 32) * sN * sizeof(float);

    if (check_align(input_ptr, 16, N) && check_align(output_ptr, 16, N)) {
        constexpr int vec_size {16 / sizeof(T)};
        auto running_kernel = rms_norm_fw_2d_kernel<T, vec_size>;
        cudaFuncSetAttribute(running_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smemsize);
        running_kernel<<<gridDim, blockDim, smemsize>>>
            (M, N, eps, input_ptr, output_ptr, rstd_ptr, sN);
    } else if (check_align(input_ptr, 8, N) && check_align(output_ptr, 8, N)) {
        constexpr int vec_size {8 / sizeof(T)};
        auto running_kernel = rms_norm_fw_2d_kernel<T, vec_size>;
        cudaFuncSetAttribute(running_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smemsize);
        running_kernel<<<gridDim, blockDim, smemsize>>>
            (M, N, eps, input_ptr, output_ptr, rstd_ptr, sN);
    } else if (check_align(input_ptr, 4, N) && check_align(output_ptr, 4, N)) {
        constexpr int vec_size {4 / sizeof(T)};
        auto running_kernel = rms_norm_fw_2d_kernel<T, vec_size>;
        cudaFuncSetAttribute(running_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smemsize);
        running_kernel<<<gridDim, blockDim, smemsize>>>
            (M, N, eps, input_ptr, output_ptr, rstd_ptr, sN);
    } else {
        constexpr int vec_size {1};
        auto running_kernel = rms_norm_fw_2d_kernel<T, vec_size>;
        cudaFuncSetAttribute(running_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smemsize);
        running_kernel<<<gridDim, blockDim, smemsize>>>
            (M, N, eps, input_ptr, output_ptr, rstd_ptr, sN);
    }
}


template <typename T>
void rms_norm_bw_2d_(int64_t M, int64_t N, void * out_grad_ptr_, void * output_ptr_, void * rstd_ptr_, void * in_grad_ptr_)
{
    T * out_grad_ptr = reinterpret_cast<T*>(out_grad_ptr_);
    T * output_ptr = reinterpret_cast<T*>(output_ptr_);
    float * rstd_ptr = reinterpret_cast<float*>(rstd_ptr_);
    T * in_grad_ptr = reinterpret_cast<T*>(in_grad_ptr_);

    dim3 blockDim {32, num_threads / 2 / 32};
    constexpr int bM {num_threads / 2 / 32 * inner_repeat};
    dim3 gridDim {(M + bM - 1) / bM};

    int sN = min(Max_sharedColumn, int(N));
    int smemsize = 2 * (num_threads / 2 / 32) * sN * sizeof(float);

    if (check_align(out_grad_ptr, 16, N) && check_align(output_ptr, 16, N) && check_align(in_grad_ptr, 16, N)) {
        constexpr int vec_size {16 / sizeof(T)};
        auto running_kernel = rms_norm_bw_2d_kernel<T, vec_size>;
        cudaFuncSetAttribute(running_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smemsize);
        running_kernel<<<gridDim, blockDim, smemsize>>>
            (M, N, out_grad_ptr, output_ptr, rstd_ptr, in_grad_ptr, sN);
    } else if (check_align(out_grad_ptr, 8, N) && check_align(output_ptr, 8, N) && check_align(in_grad_ptr, 8, N)) {
        constexpr int vec_size {8 / sizeof(T)};
        auto running_kernel = rms_norm_bw_2d_kernel<T, vec_size>;
        cudaFuncSetAttribute(running_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smemsize);
        running_kernel<<<gridDim, blockDim, smemsize>>>
            (M, N, out_grad_ptr, output_ptr, rstd_ptr, in_grad_ptr, sN);
    } else if (check_align(out_grad_ptr, 4, N) && check_align(output_ptr, 4, N) && check_align(in_grad_ptr, 4, N)) {
        constexpr int vec_size {4 / sizeof(T)};
        auto running_kernel = rms_norm_bw_2d_kernel<T, vec_size>;
        cudaFuncSetAttribute(running_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smemsize);
        running_kernel<<<gridDim, blockDim, smemsize>>>
            (M, N, out_grad_ptr, output_ptr, rstd_ptr, in_grad_ptr, sN);
    } else {
        constexpr int vec_size {1};
        auto running_kernel = rms_norm_bw_2d_kernel<T, vec_size>;
        cudaFuncSetAttribute(running_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smemsize);
        running_kernel<<<gridDim, blockDim, smemsize>>>
            (M, N, out_grad_ptr, output_ptr, rstd_ptr, in_grad_ptr, sN);
    }
}


template <typename T>
void rms_norm_fw_2d(int64_t M, int64_t N, float eps, void * input_ptr, void * output_ptr, void * rstd_ptr) {}

template <>
void rms_norm_fw_2d<float>(int64_t M, int64_t N, float eps, void * input_ptr, void * output_ptr, void * rstd_ptr)
{
    rms_norm_fw_2d_<float>(M, N, eps, input_ptr, output_ptr, rstd_ptr);
}

template <>
void rms_norm_fw_2d<half>(int64_t M, int64_t N, float eps, void * input_ptr, void * output_ptr, void * rstd_ptr)
{
    rms_norm_fw_2d_<half>(M, N, eps, input_ptr, output_ptr, rstd_ptr);
}

template <>
void rms_norm_fw_2d<nv_bfloat16>(int64_t M, int64_t N, float eps, void * input_ptr, void * output_ptr, void * rstd_ptr)
{
    rms_norm_fw_2d_<nv_bfloat16>(M, N, eps, input_ptr, output_ptr, rstd_ptr);
}


template <typename T>
void rms_norm_bw_2d(int64_t M, int64_t N, void * out_grad_ptr, void * output_ptr, void * rstd_ptr, void * in_grad_ptr) {}

template <>
void rms_norm_bw_2d<float>(int64_t M, int64_t N, void * out_grad_ptr, void * output_ptr, void * rstd_ptr, void * in_grad_ptr)
{
    rms_norm_bw_2d_<float>(M, N, out_grad_ptr, output_ptr, rstd_ptr, in_grad_ptr);
}

template <>
void rms_norm_bw_2d<half>(int64_t M, int64_t N, void * out_grad_ptr, void * output_ptr, void * rstd_ptr, void * in_grad_ptr)
{
    rms_norm_bw_2d_<half>(M, N, out_grad_ptr, output_ptr, rstd_ptr, in_grad_ptr);
}

template <>
void rms_norm_bw_2d<nv_bfloat16>(int64_t M, int64_t N, void * out_grad_ptr, void * output_ptr, void * rstd_ptr, void * in_grad_ptr)
{
    rms_norm_bw_2d_<nv_bfloat16>(M, N, out_grad_ptr, output_ptr, rstd_ptr, in_grad_ptr);
}