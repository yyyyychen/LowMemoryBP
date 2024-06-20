#pragma once
#include <torch/python.h>
#include <torch/nn/functional.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <vector>
#include <iostream>


#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")


#define CHECK_BOOL(x) TORCH_CHECK(x.scalar_type() == torch::kBool, #x " must be a bool tensor")
#define CHECK_UINT8(x) TORCH_CHECK(x.scalar_type() == torch::kUInt8, #x " must be a uint8 tensor")


void bool_pack_1d(int64_t N, void * input_ptr, void * output_ptr);

torch::Tensor bool_to_uint8(torch::Tensor& input);

void bool_unpack_1d(int64_t N, void * input_ptr, void * output_ptr);

torch::Tensor uint8_to_bool(torch::Tensor& input);


template <typename T>
void regelu2_fw_1d(int64_t N, void * input_ptr, void * output_ptr, void * flag_ptr);

std::vector<torch::Tensor> regelu2_fw(torch::Tensor& input);

template <typename T>
void regelu2_bw_1d(int64_t N, void * out_grad_ptr, void * packed_flag_ptr, void * in_grad_ptr);

torch::Tensor regelu2_bw(torch::Tensor& out_grad, torch::Tensor& packed_flag);


template <typename T>
void resilu2_fw_1d(int64_t N, void * input_ptr, void * output_ptr, void * flag_ptr);

std::vector<torch::Tensor> resilu2_fw(torch::Tensor& input);

template <typename T>
void resilu2_bw_1d(int64_t N, void * out_grad_ptr, void * packed_flag_ptr, void * in_grad_ptr);

torch::Tensor resilu2_bw(torch::Tensor& out_grad, torch::Tensor& packed_flag);