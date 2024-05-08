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

#include "thevenin_equiv_op.h"
#include "utility_functions.h"

#define CHECK_CPU(x) TORCH_CHECK(x.device() == torch::kCPU, #x " must be a CPU tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

namespace aihwkit {
namespace detail {

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
    T r_s) {

  // X is B x N  [last dim is aligned in mem]
  // G is M x N
  // output is B x M x U

  const T eps = 1e-12;
  const T seg_rows = 1; // fixed to 1
  const T rw_segs = (T)1e-6 * r_s * seg_rows;

  for (int b = 0; b < B; b++) {
    for (int i = 0; i < M; i++) {

      const int base_idx = b * (U * M) + U * i;

      for (int s = 0; s < N; s++) {
        T gm = Gm[N * i + s];
        T gp = Gp[N * i + s];
        T sum_g = gp + gm;

        T x = X[b * N + s] / tmax * (T)(U - 1);
        T x_abs = std::abs(x);

        for (int t = 0; t < U; t++) {
          int idx = base_idx + t;

          T gth_pp = ((x > (T)t) ? gp : (T)0.0) + ((x < -((T)t)) ? gm : (T)0.0);
          T gth_mm = ((x > (T)t) ? gm : (T)0.0) + ((x < -((T)t)) ? gp : (T)0.0);
          T gth_zz = (x_abs <= (T)t) ? sum_g : (T)0.0;

          T gth = gth_pp + gth_mm + gth_zz + eps;
          T rth = (T)1.0 / gth;
          T vth = ((T)0.6 * gth_pp + (T)0.2 * gth_mm + (T)0.4 * gth_zz) * rth;

          if (s == 0) {
            vth_3d[idx] = vth;
            rth_3d[idx] = rth;
          } else {
            T r_1 = rth_3d[idx] + rw_segs;
            T r_2 = rth;

            rth_3d[idx] = (r_1 * r_2) / (r_1 + r_2);
            vth_3d[idx] = (vth_3d[idx] / r_1 + vth / r_2) * rth_3d[idx];
          }
          if (s == N - 1) {
            rth_3d[idx] += 0.5 * rw_segs;
          }
        }
      }
    }
  }
}

template void theveninEquivCPU(
    const int,
    const int,
    const int,
    const int,
    float *,
    float *,
    const float *,
    const float *,
    const float *,
    float,
    float);
template void theveninEquivCPU(
    const int,
    const int,
    const int,
    const int,
    double *,
    double *,
    const double *,
    const double *,
    const double *,
    double,
    double);

} // namespace detail

#define DISPATCH_THV(TYPE)                                                                         \
  if (x_input.device() != torch::kCPU) {                                                           \
    detail::theveninEquivCUDA<TYPE>(                                                               \
        B, M, N, U, y_output.template data_ptr<TYPE>(),                                            \
        y_output.template data_ptr<TYPE>() + B * M * U, x_input.template data_ptr<TYPE>(),         \
        gp_values.template data_ptr<TYPE>(), gm_values.template data_ptr<TYPE>(), t_max, r_s);     \
  } else {                                                                                         \
    CHECK_CPU(x_input);                                                                            \
    detail::theveninEquivCPU<TYPE>(                                                                \
        B, M, N, U, y_output.template data_ptr<TYPE>(),                                            \
        y_output.template data_ptr<TYPE>() + B * M * U, x_input.template data_ptr<TYPE>(),         \
        gp_values.template data_ptr<TYPE>(), gm_values.template data_ptr<TYPE>(), t_max, r_s);     \
  }

at::Tensor theveninEquiv(
    at::Tensor &x_input,
    at::Tensor &gp_values,
    at::Tensor &gm_values,
    float r_s,
    float t_max,
    int time_steps) {

  CHECK_CONTIGUOUS(x_input);
  CHECK_CONTIGUOUS(gm_values);
  CHECK_CONTIGUOUS(gp_values);

  TORCH_CHECK(x_input.dim() == 2, " Input must be a 2D tensor.");
  TORCH_CHECK(gm_values.dim() == 2, " Gm must be a 2D tensor.");
  TORCH_CHECK(gp_values.dim() == 2, " Gp must be a 2D tensor.");
  TORCH_CHECK(gp_values.size(1) == gm_values.size(1), " Gp and Gm must be of same shape.");
  TORCH_CHECK(gp_values.size(0) == gm_values.size(0), " Gp and Gm must be of same shape.");
  TORCH_CHECK(x_input.size(1) == gm_values.size(1), " Input dim and Gp input dim should match.");

  int B = x_input.size(0);
  int N = x_input.size(1);
  int M = gp_values.size(0);
  int U = time_steps;

  // output is vth_3d, rth_3d combined
  torch::Tensor y_output = torch::zeros(at::IntArrayRef{2, B, M, U}, x_input.options());
  if (x_input.dtype() == torch::kFloat32) {
    DISPATCH_THV(float);
    //} else if (x_input.dtype() == torch::kHalf) {
    //  DISPATCH_THV(half);
  } else if (x_input.dtype() == torch::kDouble) {
    DISPATCH_THV(double);
  } else {
    TORCH_CHECK(false, "Data-type not supported");
  }

  return y_output;
};

} // namespace aihwkit

#undef DISPATCH_THV
#undef CHECK_CPU
#undef CHECK_CONTIGUOUS
