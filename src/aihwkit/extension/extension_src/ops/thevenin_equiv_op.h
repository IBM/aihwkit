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

template <typename T>
void theveninEquivCUDA(
    const int B,
    const int M,
    const int N,
    const int U,
    T *vth_3d,
    T *rth_3d,
    const T *X,
    const T *Gp,
    const T *Gm,
    T tmax,
    T r_s)
#ifndef RPU_USE_CUDA
{
  TORCH_CHECK(false, "CUDA is not available.");
}
#endif
;

template <typename T>
void theveninEquivCPU(
    const int B,
    const int M,
    const int N,
    const int U,
    T *vth_3d,
    T *rth_3d,
    const T *X,
    const T *Gp,
    const T *Gm,
    T tmax,
    T r_s);

} // namespace detail

at::Tensor theveninEquiv(
    at::Tensor &x_input,
    at::Tensor &gp_values,
    at::Tensor &gm_values,
    float r_s,
    float t_max,
    int time_steps);
} // namespace aihwkit
