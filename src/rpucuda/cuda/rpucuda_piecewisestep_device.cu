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

#include "pwu_kernel_parameter.h"
#include "rpu_pulsed_meta_parameter.h"
#include "rpucuda_piecewisestep_device.h"
#include <memory>

namespace RPU {

namespace {

template <typename T>
__device__ __forceinline__ T get_interpolated_scale(
    const T &w,
    const T *piecewise_vec,
    const T &scale,
    const int &n_sections,
    const T &w_range,
    const T &wmin) {

  if (n_sections <= 0 || (w_range <= (T)0.0)) {
    return n_sections == 0 ? piecewise_vec[0] * scale : scale;
  } else {
    T w_scaled = MAX((w - wmin) / w_range * (T)n_sections, (T)0.0);
    int w_index = MIN((int)floor(w_scaled), n_sections - 1);
    T t = MIN(w_scaled - (T)w_index, (T)1.0); // convex fraction
    T t1 = (T)1.0 - t;
    return scale * (t1 * piecewise_vec[w_index] + t * piecewise_vec[w_index + 1]);
  }
}

template <typename T> struct UpdateFunctorPiecewiseStep {

  __device__ __forceinline__ void operator()(
      T &apparent_weight,
      uint32_t n,
      uint32_t negative,
      const param4_t par_4,
      const param2_t par_2,
      T &persistent_weight,
      const T *global_pars,
      const int global_params_count,
      T noise_std_dw,
      curandState &local_state) {

    UNUSED(par_2);

    // expects global_params_count to be factor of two: 16 32 64 etc
    // -- first half is piecewise up, second piecewise down
    // -- last value of halves is reserved for n_points and write_std
    const T *piecewise_vec =
        (negative > 0) ? global_pars : global_pars + (size_t)global_params_count / 2;
    const int n_points = (int)round(global_pars[global_params_count / 2 - 1]);
    const int n_sections = n_points - 1;
    const T uw_std = global_pars[global_params_count - 1]; // always last

    const T scale = (negative > 0) ? ((T)par_4.w) : (-(T)par_4.y); // [3], [1]
    const T wmax = par_4.z;                                        // [2]
    const T wmin = par_4.x;                                        // [0]
    const T w_range = wmax - wmin;

    T &w = uw_std > (T)0.0 ? persistent_weight : apparent_weight;

    // n is larger 0 in any case
    if (n == 1) {
      T interpolated_scale =
          get_interpolated_scale(w, piecewise_vec, scale, n_sections, w_range, wmin);
      if (noise_std_dw > (T)0.0) {
        T stoch_value = curand_normal(&local_state);
        stoch_value *= noise_std_dw;
        w += interpolated_scale * ((T)1.0 + stoch_value);
      } else {
        w += interpolated_scale;
      }
      w = (w > wmax) ? wmax : w;
      w = (w < wmin) ? wmin : w;
    } else {
      if (noise_std_dw > (T)0.0) {
        for (int i_updates = 0; i_updates < n; i_updates++) {
          T stoch_value = curand_normal(&local_state);
          stoch_value *= noise_std_dw;
          T interpolated_scale =
              get_interpolated_scale(w, piecewise_vec, scale, n_sections, w_range, wmin);
          w += interpolated_scale * ((T)1.0 + stoch_value);

          // better always check both bounds
          w = (w > wmax) ? wmax : w;
          w = (w < wmin) ? wmin : w;
        }
      } else {
        for (int i_updates = 0; i_updates < n; i_updates++) {
          T interpolated_scale =
              get_interpolated_scale(w, piecewise_vec, scale, n_sections, w_range, wmin);
          w += interpolated_scale;

          // better always check both bounds
          w = (w > wmax) ? wmax : w;
          w = (w < wmin) ? wmin : w;
        }
      }
    }

    // add update write noise onto apparent weight
    if (uw_std > (T)0.0) {
      T stoch_value = curand_normal(&local_state);
      apparent_weight = persistent_weight + uw_std * stoch_value;
    }
  }
};

} // namespace

#define ARGS                                                                                       \
  (this->context_, this->x_size_, this->d_size_, m_batch, nK32, use_bo64, out_trans, up,           \
   par.getName())

#define ADD_KERNELS(GC)                                                                            \
  v.push_back(                                                                                     \
      RPU::make_unique<PWUKernelParameterSingleFunctor<T, UpdateFunctorPiecewiseStep<T>, GC>>      \
          ARGS);                                                                                   \
  v.push_back(                                                                                     \
      RPU::make_unique<PWUKernelParameterBatchFunctor<T, UpdateFunctorPiecewiseStep<T>, GC>>       \
          ARGS);                                                                                   \
  v.push_back(                                                                                     \
      RPU::make_unique<PWUKernelParameterBatchSharedFunctor<T, UpdateFunctorPiecewiseStep<T>, GC>> \
          ARGS);

template <typename T>
pwukpvec_t<T> PiecewiseStepRPUDeviceCuda<T>::getUpdateKernels(
    int m_batch, int nK32, int use_bo64, bool out_trans, const PulsedUpdateMetaParameter<T> &up) {

  pwukpvec_t<T> v;
  const auto &par = getPar();

  switch (gp_count_) {
  case 32:
    ADD_KERNELS(32);
    break;
  case 64:
    ADD_KERNELS(64);
    break;
  case 128:
    ADD_KERNELS(128);
    break;
  default:
    RPU_FATAL("Number of interpolation nodes (" << gp_count_ << ") not supported for GPU.");
  }
  return v;
}

#undef ARGS
#undef ADD_KERNELS

template class PiecewiseStepRPUDeviceCuda<float>;
#ifdef RPU_USE_DOUBLE
template class PiecewiseStepRPUDeviceCuda<double>;
#endif
#ifdef RPU_USE_FP16
template class PiecewiseStepRPUDeviceCuda<half_t>;
#endif

} // namespace RPU
