/**
 * (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
 *
 * Licensed under the MIT license. See LICENSE file in the project root for details.
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
