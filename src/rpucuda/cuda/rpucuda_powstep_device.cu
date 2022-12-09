/**
 * (C) Copyright 2020, 2021, 2022 IBM. All Rights Reserved.
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
#include "rpucuda_powstep_device.h"
#include <memory>

namespace RPU {

template <typename T> struct UpdateFunctorPowStep {

  __device__ __forceinline__ void operator()(
      T &apparent_weight,
      uint32_t n,
      uint32_t negative,
      const float4 par_4,         // min_bound, scale_down, max_bound, scale_up,
      const float2 gamma_down_up, // gamma_down, gamma_up
      T &persistent_weight,
      const T *write_noise_std,
      const int global_params_count,
      T noise_std_dw,
      curandState &local_state) {

    UNUSED(global_params_count); // fixed

    T wmin = par_4.x; // [0]
    T wmax = par_4.z; // [2]
    T range = wmax - wmin;
    if (range == 0) {
      return;
    }
    T uw_std = *write_noise_std;
    T &w = uw_std > 0 ? persistent_weight : apparent_weight;
    // negative > 0 means sign < 0 and thus up-direction
    T scale = (negative > 0) ? (par_4.w) : (-par_4.y);                //[3] (up), [1] (down)
    T gamma = (negative > 0) ? (gamma_down_up.y) : (gamma_down_up.x); // [1] (up), [0] (down)

    // up direction: ((wmax - w) / range) ^ gamma
    // down direction:  ((w - wmin) / range) ^ gamma  == (1 - (wmax - w)/range) ^ gamma

    // n is larger 0 in any case
    if (n == 1) {
      T x = (wmax - w) / range;
      T dw = scale * ((negative > 0) ? __powf(x, gamma) : __powf((T)1.0 - x, gamma));

      if (noise_std_dw > 0) {
        T stoch_value = curand_normal(&local_state);
        stoch_value *= noise_std_dw;
        w += dw * ((T)1.0 + stoch_value);
      } else {
        w += dw;
      }
      w = (w > wmax) ? wmax : w;
      w = (w < wmin) ? wmin : w;

    } else {
      if (noise_std_dw > 0) {
        for (int i_updates = 0; i_updates < n; i_updates++) {
          T stoch_value = curand_normal(&local_state);
          stoch_value *= noise_std_dw;
          T x = (wmax - w) / range;
          T dw = scale * ((negative > 0) ? __powf(x, gamma) : __powf((T)1.0 - x, gamma));
          w += dw * ((T)1.0 + stoch_value);
          // better always check both bounds
          w = (w > wmax) ? wmax : w;
          w = (w < wmin) ? wmin : w;
        }
      } else {
        for (int i_updates = 0; i_updates < n; i_updates++) {
          T x = (wmax - w) / range;
          w += scale * ((negative > 0) ? __powf(x, gamma) : __powf((T)1.0 - x, gamma));
          // better always check both bounds
          w = (w > wmax) ? wmax : w;
          w = (w < wmin) ? wmin : w;
        }
      }
    }

    // add update write noise onto apparent weight
    if (uw_std > 0) {
      T stoch_value = curand_normal(&local_state);
      apparent_weight = persistent_weight + uw_std * stoch_value;
    }
  }
};

#define ARGS                                                                                       \
  (this->context_, this->x_size_, this->d_size_, m_batch, nK32, use_bo64, out_trans, up,           \
   par.getName())

template <typename T>
pwukpvec_t<T> PowStepRPUDeviceCuda<T>::getUpdateKernels(
    int m_batch, int nK32, int use_bo64, bool out_trans, const PulsedUpdateMetaParameter<T> &up) {

  pwukpvec_t<T> v;
  const auto &par = getPar();
  v.push_back(
      RPU::make_unique<PWUKernelParameterSingleFunctor<T, UpdateFunctorPowStep<T>, 1>> ARGS);
  v.push_back(RPU::make_unique<PWUKernelParameterBatchFunctor<T, UpdateFunctorPowStep<T>, 1>> ARGS);
  v.push_back(
      RPU::make_unique<PWUKernelParameterBatchSharedFunctor<T, UpdateFunctorPowStep<T>, 1>> ARGS);
  v.push_back(
      RPU::make_unique<
          PWUKernelParameterBatchSharedWeightOutputFunctor<T, UpdateFunctorPowStep<T>, 1>> ARGS);
  return v;
}

#undef ARGS

template class PowStepRPUDeviceCuda<float>;
#ifdef RPU_USE_DOUBLE
template class PowStepRPUDeviceCuda<double>;
#endif

} // namespace RPU
