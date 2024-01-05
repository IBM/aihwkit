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

#include "float_prec_op.h"
#include "float_prec_common.h"

#define CHECK_CPU(x) TORCH_CHECK(x.device() == torch::kCPU, #x " must be a CPU tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

namespace aihwkit {
namespace detail {

template <int ML = 7> inline float precisionCastFloatExp8CPU(float v) {
  uint32_t x_int = ((union {
                     float f;
                     uint32_t i;
                   }){v})
                       .i;

  uint32_t highest_not_needed_bit = 1 << (22 - ML);
  uint32_t needs_up_round = (x_int & highest_not_needed_bit);
  uint32_t valid_msk = ~((highest_not_needed_bit << 1) - 1);
  bool overflow_if = ((0x7F800000 & x_int) == 0x7F800000);
  x_int &= valid_msk;
  // set infty (with preserved sign) or round
  x_int = overflow_if ? (x_int & (uint32_t)0xFF800000) : x_int + (needs_up_round << 1);

  return ((union {
           uint32_t i;
           float f;
         }){x_int})
      .f;
}

template <int EL, int ML, bool saturate_to_inf, bool rounding = true>
inline float precisionCastFloatCPU(float x) {
  FLOATPREC_INIT(EL, ML);
  FLOATPREC_BODY(x, y, saturate_to_inf, rounding);
  return y;
}

template <int EL, int ML>
void floatPrecisionCastCPU(const int N, float *Y, const float *X, const bool saturate_to_inf) {
  static_assert(sizeof(float) == 4);

  if (EL == 8 && saturate_to_inf) {
    for (int i = 0; i < N; i++) {
      float x = X[i];
      Y[i] = precisionCastFloatExp8CPU<ML>(x);
    }
  } else {
    if (saturate_to_inf) {
      for (int i = 0; i < N; i++) {
        float x = X[i];
        Y[i] = precisionCastFloatCPU<EL, ML, true>(x);
      }
    } else {
      for (int i = 0; i < N; i++) {
        float x = X[i];
        Y[i] = precisionCastFloatCPU<EL, ML, false>(x);
      }
    }
  }
}

#define DISPATCH_FPC2(EXPONENT, MANTISSA)                                                          \
  if (x_input.device() != torch::kCPU) {                                                           \
    detail::floatPrecisionCastCUDA<EXPONENT, MANTISSA>(                                            \
        x_input.numel(), y_output.template data_ptr<float>(), x_input.template data_ptr<float>(),  \
        saturate_to_inf);                                                                          \
  } else {                                                                                         \
    CHECK_CPU(x_input);                                                                            \
    detail::floatPrecisionCastCPU<EXPONENT, MANTISSA>(                                             \
        x_input.numel(), y_output.template data_ptr<float>(), x_input.template data_ptr<float>(),  \
        saturate_to_inf);                                                                          \
  }                                                                                                \
  break

#define DISPATCH_FPC(MANTISSA)                                                                     \
  case MANTISSA: {                                                                                 \
    switch (exponent) {                                                                            \
    case 8:                                                                                        \
      DISPATCH_FPC2(8, MANTISSA);                                                                  \
    case 7:                                                                                        \
      DISPATCH_FPC2(7, MANTISSA);                                                                  \
    case 6:                                                                                        \
      DISPATCH_FPC2(6, MANTISSA);                                                                  \
    case 5:                                                                                        \
      DISPATCH_FPC2(5, MANTISSA);                                                                  \
    case 4:                                                                                        \
      DISPATCH_FPC2(4, MANTISSA);                                                                  \
    case 3:                                                                                        \
      DISPATCH_FPC2(3, MANTISSA);                                                                  \
    case 2:                                                                                        \
      DISPATCH_FPC2(2, MANTISSA);                                                                  \
    default:                                                                                       \
      TORCH_CHECK(false, "Mantissa/exponent combination not implemented!");                        \
    };                                                                                             \
    break;                                                                                         \
  }

} // namespace detail

at::Tensor
floatPrecisionCast(at::Tensor &x_input, int exponent, int mantissa, bool saturate_to_inf) {

  CHECK_CONTIGUOUS(x_input);

  torch::Tensor y_output = torch::empty_like(x_input);

  switch (mantissa) {
    DISPATCH_FPC(8);
    DISPATCH_FPC(9);
    DISPATCH_FPC(10);
    DISPATCH_FPC(11);
    DISPATCH_FPC(7);
    DISPATCH_FPC(6);
    DISPATCH_FPC(5);
    DISPATCH_FPC(4);
    DISPATCH_FPC(3);
    DISPATCH_FPC(12);
    DISPATCH_FPC(13);
    DISPATCH_FPC(14);
    DISPATCH_FPC(15);
    DISPATCH_FPC(16);
    DISPATCH_FPC(17);
    DISPATCH_FPC(18);
    DISPATCH_FPC(19);
    DISPATCH_FPC(20);
    DISPATCH_FPC(21);
    DISPATCH_FPC(22);
  default:
    TORCH_CHECK(false, "Mantissa setting not possible.");
  }
  return y_output;
};

} // namespace aihwkit

#undef DISPATCH_FPC
#undef DISPATCH_FPC2
