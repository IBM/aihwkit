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

template <typename T>
__global__ void kernelTheveninEquiv(
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
  const T eps = 1e-12;
  const T seg_rows = 1; // fixed to 1
  const T rw_segs = (T)1e-6 * r_s * seg_rows;
  int UM = U * M;
  int size = UM * B;

  RPU_CUDA_1D_KERNEL_LOOP(idx, size) {

    int b = idx / UM;
    int i = (idx % UM) / U;
    int t = (idx % UM) % U;

    const int base_idx = b * (UM) + U * i;

    for (int s = 0; s < N; s++) {
      T gm = Gm[N * i + s];
      T gp = Gp[N * i + s];
      T sum_g = gp + gm;

      T x = X[b * N + s] / tmax * (T)(U - 1);
      T x_abs = std::abs(x);

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
    T r_s) {
  int size = U * M * B;
  auto s = at::cuda::getCurrentCUDAStream();
  kernelTheveninEquiv<T><<<RPU_GET_BLOCKS(size), RPU_THREADS_PER_BLOCK, 0, s>>>(
      B, M, N, U, vth_3d, rth_3d, X, Gp, Gm, tmax, r_s);
  AT_CUDA_CHECK(cudaStreamSynchronize(s));
}

template void theveninEquivCUDA(
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
template void theveninEquivCUDA(
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
} // namespace aihwkit
