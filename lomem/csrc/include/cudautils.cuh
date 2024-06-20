#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>


constexpr int WarpSize {32};


template<typename T, int pack_size>
struct GetPackType {
    using type = typename std::aligned_storage<pack_size * sizeof(T), pack_size * sizeof(T)>::type;
};


template<typename T, int pack_size>
using PackType = typename GetPackType<T, pack_size>::type;


template<typename T, int pack_size>
union Pack {
    static_assert(sizeof(PackType<T, pack_size>) == sizeof(T) * pack_size, "");
    __device__ Pack() {}
    PackType<T, pack_size> storage;
    T elem[pack_size];
};


template <typename T>
bool check_align(T* ptr, int bytes, int64_t n)
{
    uintptr_t address = reinterpret_cast<uintptr_t>(ptr);
    bool head_align = !(address % bytes);
    bool length_align = !((n * sizeof(T)) % bytes);
    return head_align && length_align;
}


template <int elem_bit>
__inline__ __device__ void packflagWarpReduce(uint8_t& flag)
{
    #pragma unroll
    for (int i = 1; i < 8 / elem_bit; i *= 2) {
        flag |= (__shfl_down_sync(0xffffffff, flag, i, WarpSize) << (i * elem_bit));
    }
}