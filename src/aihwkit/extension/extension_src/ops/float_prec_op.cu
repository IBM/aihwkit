/**
 * (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
 */

#include "cuda_util.h"
#include "float_prec_common.h"
#include "float_prec_common_gpu.h"
#include "float_prec_op.h"
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>

namespace aihwkit {
namespace detail {

template <int EL, int ML, bool saturate_to_inf = false, bool rounding = true>
__global__ void kernelPrecisionFloat(const int N, float *Y, const float *X) {
  FLOATPREC_INIT(EL, ML);

  RPU_CUDA_1D_KERNEL_LOOP(i, N) {
    float x = X[i];
    FLOATPREC_BODY(x, y, saturate_to_inf, rounding);
    Y[i] = y;
  }
}

template <int ML> __global__ void kernelPrecisionFloatExp8(const int N, float *Y, const float *X) {
  RPU_CUDA_1D_KERNEL_LOOP(i, N) {
    float x = X[i];
    uint32_t x_int = ((union {
                       float f;
                       uint32_t i;
                     }){x})
                         .i;
    uint32_t highest_not_needed_bit = 1 << (22 - ML);
    uint32_t needs_up_round = (x_int & highest_not_needed_bit);
    uint32_t valid_msk = ~((highest_not_needed_bit << 1) - 1);
    bool overflow_if = ((0x7F800000 & x_int) == 0x7F800000);
    x_int &= valid_msk;
    // set infty (with preserved sign) or round
    x_int = overflow_if ? (x_int & (uint32_t)0xFF800000) : x_int + (needs_up_round << 1);
    Y[i] = ((union {
             uint32_t i;
             float f;
           }){x_int})
               .f;
  }
}

template <int EL, int ML>
void floatPrecisionCastCUDA(const int N, float *Y, const float *X, const bool saturate_to_inf) {
  static_assert(sizeof(float) == 4);
  static_assert(EL <= 8);

  auto s = at::cuda::getCurrentCUDAStream();
  if (EL == 8 && saturate_to_inf) {
    kernelPrecisionFloatExp8<ML><<<RPU_GET_BLOCKS(N), RPU_THREADS_PER_BLOCK, 0, s>>>(N, Y, X);
  } else {
    if (saturate_to_inf) {
      kernelPrecisionFloat<EL, ML, true>
          <<<RPU_GET_BLOCKS(N), RPU_THREADS_PER_BLOCK, 0, s>>>(N, Y, X);
    } else {
      kernelPrecisionFloat<EL, ML, false>
          <<<RPU_GET_BLOCKS(N), RPU_THREADS_PER_BLOCK, 0, s>>>(N, Y, X);
    }
  }
  AT_CUDA_CHECK(cudaStreamSynchronize(s));
}
#define T(I, J) template void floatPrecisionCastCUDA<I, J>(const int, float *, const float *, bool)

#define TT(I)                                                                                      \
  T(I, 3);                                                                                         \
  T(I, 4);                                                                                         \
  T(I, 5);                                                                                         \
  T(I, 6);                                                                                         \
  T(I, 7);                                                                                         \
  T(I, 8);                                                                                         \
  T(I, 9);                                                                                         \
  T(I, 10);                                                                                        \
  T(I, 11);                                                                                        \
  T(I, 12);                                                                                        \
  T(I, 13);                                                                                        \
  T(I, 14);                                                                                        \
  T(I, 15);                                                                                        \
  T(I, 16);                                                                                        \
  T(I, 17);                                                                                        \
  T(I, 18);                                                                                        \
  T(I, 19);                                                                                        \
  T(I, 20);                                                                                        \
  T(I, 21);                                                                                        \
  T(I, 22);

TT(2);
TT(3);
TT(4);
TT(5);
TT(6);
TT(7);
TT(8);

#undef T
#undef TT
} // namespace detail
} // namespace aihwkit
