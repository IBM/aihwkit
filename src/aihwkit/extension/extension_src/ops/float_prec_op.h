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
