/**
 * (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
 *
 * Licensed under the MIT license. See LICENSE file in the project root for details.
 */

#include <torch/extension.h>

namespace aihwkit {
namespace detail {

template <int EL, int ML>
void floatPrecisionCastCUDA(const int N, float *Y, const float *X, const bool saturate_to_inf)
#ifndef RPU_USE_CUDA
{
  TORCH_CHECK(false, "CUDA is not available.");
}
#endif
;

template <int EL, int ML>
void floatPrecisionCastCPU(const int N, float *Y, const float *X, const bool saturate_to_inf);

} // namespace detail

at::Tensor
floatPrecisionCast(at::Tensor &x_input, int exponent, int mantissa, bool saturate_to_inf);

} // namespace aihwkit
