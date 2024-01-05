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
#include "rpucuda_softbounds_reference_device.h"
#include <memory>

namespace RPU {

template <typename T> struct UpdateFunctorSoftBoundsReferenceMult {

  __device__ __forceinline__ void operator()(
      T &apparent_weight,
      uint32_t n,
      uint32_t negative,
      const param4_t par_4,
      const param2_t reference_2,
      T &persistent_weight,
      const T *write_noise_std,
      const int global_params_count,
      T noise_std_dw,
      curandState &local_state) {

    UNUSED(global_params_count); // fixed

    T uw_std = *write_noise_std;
    T wmax = par_4.z; // [2]
    T wmin = par_4.x; // [0]

    T &w = uw_std > (T)0.0 ? persistent_weight : apparent_weight;
    T ref = reference_2.x;

    // negative > 0 means going up here ...

    T lin_a = (T)0.0, lin_dw = (T)0.0;
    if (negative == 0) {
      lin_dw = -(T)par_4.y; // [1]
      if (wmin < (T)0.0) {
        lin_a = -lin_dw / wmin;
      }
    } else {
      lin_dw = (T)par_4.w; // [3]
      if (wmax > (T)0.0) {
        lin_a = -lin_dw / wmax;
      }
    }

    w += ref;

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
          T stoch_value = curand_normal(&local_state);
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
    w -= ref;

    if (uw_std > (T)0.0) {
      T stoch_value = curand_normal(&local_state);
      apparent_weight = w + uw_std * stoch_value;
    }
  }
};

template <typename T> struct UpdateFunctorSoftBoundsReferenceAdd {

  __device__ __forceinline__ void operator()(
      T &apparent_weight,
      uint32_t n,
      uint32_t negative,
      const param4_t par_4,
      const param2_t reference_2,
      T &persistent_weight,
      const T *write_noise_std,
      const int global_params_count,
      T noise_std_dw,
      curandState &local_state) {

    UNUSED(global_params_count); // fixed

    T uw_std = *write_noise_std;
    T &w = uw_std > (T)0.0 ? persistent_weight : apparent_weight;
    T wmax = par_4.z; // [2]
    T wmin = par_4.x; // [0]
    T ref = reference_2.x;

    T lin_a = (T)0.0, lin_dw = (T)0.0;
    if (negative == 0) {
      lin_dw = -(T)par_4.y; // [1]
      if (wmin < (T)0.0) {
        lin_a = -lin_dw / wmin;
      }
    } else {
      lin_dw = (T)par_4.w; // [3]
      if (wmax > (T)0.0) {
        lin_a = -lin_dw / wmax;
      }
    }

    w += ref;

    // n is larger 0 in any case
    if (n == 1) {
      if (noise_std_dw > (T)0.0) {
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

    w -= ref;

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
pwukpvec_t<T> SoftBoundsReferenceRPUDeviceCuda<T>::getUpdateKernels(
    int m_batch, int nK32, int use_bo64, bool out_trans, const PulsedUpdateMetaParameter<T> &up) {

  pwukpvec_t<T> v;
  const auto &par = getPar();
  if (par.mult_noise) {
    v.push_back(
        RPU::make_unique<
            PWUKernelParameterSingleFunctor<T, UpdateFunctorSoftBoundsReferenceMult<T>, 1>> ARGS);
    v.push_back(
        RPU::make_unique<
            PWUKernelParameterBatchFunctor<T, UpdateFunctorSoftBoundsReferenceMult<T>, 1>> ARGS);
    v.push_back(RPU::make_unique<PWUKernelParameterBatchSharedFunctor<
                    T, UpdateFunctorSoftBoundsReferenceMult<T>, 1>> ARGS);
    v.push_back(RPU::make_unique<PWUKernelParameterBatchSharedWeightOutputFunctor<
                    T, UpdateFunctorSoftBoundsReferenceMult<T>, 1>> ARGS);

  } else {

    v.push_back(
        RPU::make_unique<
            PWUKernelParameterSingleFunctor<T, UpdateFunctorSoftBoundsReferenceAdd<T>, 1>> ARGS);
    v.push_back(RPU::make_unique<
                PWUKernelParameterBatchFunctor<T, UpdateFunctorSoftBoundsReferenceAdd<T>, 1>> ARGS);
    v.push_back(RPU::make_unique<PWUKernelParameterBatchSharedFunctor<
                    T, UpdateFunctorSoftBoundsReferenceAdd<T>, 1>> ARGS);
    v.push_back(RPU::make_unique<PWUKernelParameterBatchSharedWeightOutputFunctor<
                    T, UpdateFunctorSoftBoundsReferenceAdd<T>, 1>> ARGS);
  }
  return v;
}

#undef ARGS

template class SoftBoundsReferenceRPUDeviceCuda<float>;
#ifdef RPU_USE_DOUBLE
template class SoftBoundsReferenceRPUDeviceCuda<double>;
#endif
#ifdef RPU_USE_FP16
template class SoftBoundsReferenceRPUDeviceCuda<half_t>;
#endif

} // namespace RPU
