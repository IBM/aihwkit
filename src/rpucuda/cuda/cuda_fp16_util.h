/**
 * (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
 *
 * Licensed under the MIT license. See LICENSE file in the project root for details.
 */

#pragma once
#include "utility_functions.h"

#ifdef RPU_DEFINE_CUDA_HALF_ARRAY
#include "cuda_fp16.h"

#ifdef RPU_BFLOAT_AS_FP16
#include "cuda_bf16.h"
#endif

#ifndef __RPU_CUDA_HALF_DEFINED
#define __RPU_CUDA_HALF_DEFINED
#define RPU_DEFINE_HALF_FUN(NAME)                                                                  \
  __device__ __forceinline__ half_t NAME(half_t value) { return h##NAME(value); }
RPU_DEFINE_HALF_FUN(rint);
RPU_DEFINE_HALF_FUN(trunc);
RPU_DEFINE_HALF_FUN(exp);
RPU_DEFINE_HALF_FUN(log2);
RPU_DEFINE_HALF_FUN(log);
RPU_DEFINE_HALF_FUN(floor);
RPU_DEFINE_HALF_FUN(ceil);
RPU_DEFINE_HALF_FUN(rsqrt);
RPU_DEFINE_HALF_FUN(sqrt);

__device__ __forceinline__ half_t fabs(half_t value) { return __habs(value); }
__device__ __forceinline__ half_t isinf(half_t value) { return __isinf(value); }
__device__ __forceinline__ half_t round(half_t value) { return hrint(value); }

#undef RPU_DEFINE_HALF_FUN
#endif
#endif
