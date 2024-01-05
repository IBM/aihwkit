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
#include "rpucuda_linearstep_device.h"
#include <memory>

namespace RPU {

template <typename T> struct UpdateFunctorLinearStepMult {

  __device__ __forceinline__ void operator()(
      T &apparent_weight,
      uint32_t n,
      uint32_t negative,
      const param4_t par_4,
      const param2_t lin_slope,
      T &persistent_weight,
      const T *write_noise_std,
      const int global_params_count,
      T noise_std_dw,
      curandState &local_state) {

    UNUSED(global_params_count); // fixed
    // negative > 0 means going up here ...
    T uw_std = *write_noise_std;
    T lin_dw = (negative > 0) ? ((T)par_4.w) : (-(T)par_4.y);        //[3], [1]
    T lin_a = (negative > 0) ? ((T)lin_slope.y) : (-(T)lin_slope.x); // [1],[0]
    T wmax = par_4.z;                                                // [2]
    T wmin = par_4.x;                                                // [0]

    T &w = uw_std > (T)0.0 ? persistent_weight : apparent_weight;

    // n is larger 0 in any case
    if (n == 1) {
      if (noise_std_dw > (T)0.0) {
        T stoch_value = curand_normal(&local_state);
        stoch_value *= noise_std_dw;
        w += (lin_a * w + lin_dw) * ((T)1.0 + stoch_value);
      } else {
        w += lin_a * w + lin_dw;
      }
      w = (w > wmax) ? wmax : w;
      w = (w < wmin) ? wmin : w;
    } else {
      if (noise_std_dw > (T)0.0) {
        for (int i_updates = 0; i_updates < n; i_updates++) {
          T stoch_value = (T)curand_normal(&local_state);
          stoch_value *= noise_std_dw;
          w += (lin_a * w + lin_dw) * ((T)1.0 + stoch_value);
          // better always check both bounds
          w = (w > wmax) ? wmax : w;
          w = (w < wmin) ? wmin : w;
        }
      } else {
        for (int i_updates = 0; i_updates < n; i_updates++) {
          w += lin_a * w + lin_dw;
          // better always check both bounds
          w = (w > wmax) ? wmax : w;
          w = (w < wmin) ? wmin : w;
        }
      }
    }

    // add update write noise onto apparent weight
    if (uw_std > (T)0.0) {
      T stoch_value = (T)curand_normal(&local_state);
      apparent_weight = w + uw_std * stoch_value;
    }
  }
};

template <typename T> struct UpdateFunctorLinearStepAdd {

  __device__ __forceinline__ void operator()(
      T &apparent_weight,
      uint32_t n,
      uint32_t negative,
      const param4_t par_4,
      const param2_t lin_slope,
      T &persistent_weight,
      const T *write_noise_std,
      const int global_params_count,
      T noise_std_dw,
      curandState &local_state) {

    UNUSED(global_params_count); // fixed

    T uw_std = *write_noise_std;
    T lin_dw = (negative > 0) ? ((T)par_4.w) : (-(T)par_4.y);        // [3] [1]
    T lin_a = (negative > 0) ? ((T)lin_slope.y) : (-(T)lin_slope.x); //[1],[0]
    T &w = uw_std > (T)0.0 ? persistent_weight : apparent_weight;
    T wmax = (T)par_4.z; // [2]
    T wmin = (T)par_4.x; // [0]

    // n is larger 0 in any case
    if (n == 1) {
      if (noise_std_dw > (T)0) {
        T stoch_value = curand_normal(&local_state);
        stoch_value *= noise_std_dw;
        w += lin_a * w + lin_dw * ((T)1.0 + stoch_value);
      } else {
        w += lin_a * w + lin_dw;
      }
      w = (w > wmax) ? wmax : w;
      w = (w < wmin) ? wmin : w;
    } else {
      if (noise_std_dw > (T)0.0) {
        for (int i_updates = 0; i_updates < n; i_updates++) {
          T stoch_value = curand_normal(&local_state);
          stoch_value *= noise_std_dw;
          w += lin_a * w + lin_dw * ((T)1.0 + stoch_value);
          w = (w > wmax) ? wmax : w;
          w = (w < wmin) ? wmin : w;
        }
      } else {
        for (int i_updates = 0; i_updates < n; i_updates++) {
          w += lin_a * w + lin_dw;
          w = (w > wmax) ? wmax : w;
          w = (w < wmin) ? wmin : w;
        }
      }
    }
    // add update write noise onto apparent weight
    if (uw_std > (T)0.0) {
      T stoch_value = curand_normal(&local_state);
      apparent_weight = w + uw_std * stoch_value;
    }
  }
};

#define ARGS                                                                                       \
  (this->context_, this->x_size_, this->d_size_, m_batch, nK32, use_bo64, out_trans, up,           \
   par.getName())

template <typename T>
pwukpvec_t<T> LinearStepRPUDeviceCuda<T>::getUpdateKernels(
    int m_batch, int nK32, int use_bo64, bool out_trans, const PulsedUpdateMetaParameter<T> &up) {

  pwukpvec_t<T> v;
  const auto &par = getPar();
  if (par.ls_mult_noise) {
    v.push_back(
        RPU::make_unique<PWUKernelParameterSingleFunctor<T, UpdateFunctorLinearStepMult<T>, 1>>
            ARGS);
    v.push_back(
        RPU::make_unique<PWUKernelParameterBatchFunctor<T, UpdateFunctorLinearStepMult<T>, 1>>
            ARGS);
    v.push_back(
        RPU::make_unique<PWUKernelParameterBatchSharedFunctor<T, UpdateFunctorLinearStepMult<T>, 1>>
            ARGS);
    v.push_back(RPU::make_unique<PWUKernelParameterBatchSharedWeightOutputFunctor<
                    T, UpdateFunctorLinearStepMult<T>, 1>> ARGS);

  } else {

    v.push_back(
        RPU::make_unique<PWUKernelParameterSingleFunctor<T, UpdateFunctorLinearStepAdd<T>, 1>>
            ARGS);
    v.push_back(
        RPU::make_unique<PWUKernelParameterBatchFunctor<T, UpdateFunctorLinearStepAdd<T>, 1>> ARGS);
    v.push_back(
        RPU::make_unique<PWUKernelParameterBatchSharedFunctor<T, UpdateFunctorLinearStepAdd<T>, 1>>
            ARGS);
    v.push_back(RPU::make_unique<PWUKernelParameterBatchSharedWeightOutputFunctor<
                    T, UpdateFunctorLinearStepAdd<T>, 1>> ARGS);
  }
  return v;
}

#undef ARGS

template class LinearStepRPUDeviceCuda<float>;
#ifdef RPU_USE_DOUBLE
template class LinearStepRPUDeviceCuda<double>;
#endif
#ifdef RPU_USE_FP16
template class LinearStepRPUDeviceCuda<half_t>;
#endif

} // namespace RPU
