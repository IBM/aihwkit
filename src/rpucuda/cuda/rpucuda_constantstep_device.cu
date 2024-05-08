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
#include "rpucuda_constantstep_device.h"

namespace RPU {

template <typename T> struct UpdateFunctorConstantStepLargeNoise {

  __device__ __forceinline__ void operator()(
      T &w,
      uint32_t n,
      uint32_t negative,
      const param4_t par_4,
      const param2_t par_2,
      T &par_1,
      const T *global_par,
      const int global_params_count,
      T noise_std_dw,
      curandState &local_state) {

    UNUSED(global_params_count);
    UNUSED(global_par);
    UNUSED(par_1);
    UNUSED(par_2);
    // negative > 0 means going up here ...
    // here we assume that noise_std_dw>0 at least
    T wmax = par_4.z;                                   // [2];
    T wmin = par_4.x;                                   //[0];
    float dw = (negative > 0) ? (par_4.w) : (-par_4.y); // [3], [1]
    float sigma = noise_std_dw;

    // n is larger 0 in any case
    if (n == 1) { // short-cut without loop
      float stoch_value = curand_normal(&local_state);
      stoch_value *= sigma;
      w += dw * ((float)1.0 + stoch_value);

      w = (w > wmax) ? wmax : w;
      w = (w < wmin) ? wmin : w;

    } else {
      for (int i = 0; i < n; i++) { // need to loop here because noise can be large and hit the
                                    // boundary and retract again because of sign reverse
        float stoch_value = curand_normal(&local_state);
        stoch_value *= sigma;
        w += dw * ((float)1.0 + stoch_value);

        w = (w > wmax) ? wmax : w;
        w = (w < wmin) ? wmin : w;
      }
    }
  }
};

#define ARGS(NAME)                                                                                 \
  (this->context_, this->x_size_, this->d_size_, m_batch, nK32, use_bo64, out_trans, up,           \
   getPar().getName() + #NAME)

template <typename T>
pwukpvec_t<T> ConstantStepRPUDeviceCuda<T>::getUpdateKernels(
    int m_batch, int nK32, int use_bo64, bool out_trans, const PulsedUpdateMetaParameter<T> &up) {

  pwukpvec_t<T> v;

  if (getPar().dw_min_std > (T)0.33) { // 3 sigma
    v.push_back(RPU::make_unique<PWUKernelParameterSingleFunctor<
                    T, UpdateFunctorConstantStepLargeNoise<T>, 1>> ARGS(FunctorLargeNoise));
    v.push_back(RPU::make_unique<PWUKernelParameterBatchFunctor<
                    T, UpdateFunctorConstantStepLargeNoise<T>, 1>> ARGS(FunctorLargeNoise));
    v.push_back(RPU::make_unique<PWUKernelParameterBatchSharedFunctor<
                    T, UpdateFunctorConstantStepLargeNoise<T>, 1>> ARGS(FunctorLargeNoise));
    v.push_back(RPU::make_unique<PWUKernelParameterBatchSharedWeightOutputFunctor<
                    T, UpdateFunctorConstantStepLargeNoise<T>, 1>> ARGS(FunctorLargeNoise));

  } else {
    // use summing approximation is save in this case
    // Update functor and kernels are in pwu_kernels.h
    v.push_back(
        RPU::make_unique<PWUKernelParameterBatchSharedFunctor<T, UpdateFunctorConstantStep<T>, 1>>
            ARGS(Functor));
    v.push_back(RPU::make_unique<PWUKernelParameterBatchSharedWeightOutputFunctor<
                    T, UpdateFunctorConstantStep<T>, 1>> ARGS(Functor));
    v.push_back(
        RPU::make_unique<PWUKernelParameterBatchFunctor<T, UpdateFunctorConstantStep<T>, 1>> ARGS(
            Functor));

    v.push_back(
        RPU::make_unique<PWUKernelParameterSingleFunctor<T, UpdateFunctorConstantStep<T>, 1>> ARGS(
            Functor));
    v.push_back(RPU::make_unique<PWUKernelParameterBatchSharedSum<T>> ARGS(Sum));
    v.push_back(RPU::make_unique<PWUKernelParameterBatchSharedSumBoundCheck<T>> ARGS(SumBC));

    v.push_back(RPU::make_unique<PWUKernelParameterBatchSum<T>> ARGS(Sum));
    v.push_back(RPU::make_unique<PWUKernelParameterBatchSumBoundCheck<T>> ARGS(SumBC));
  }

  return v;
}

#undef ARGS

template class ConstantStepRPUDeviceCuda<float>;
#ifdef RPU_USE_DOUBLE
template class ConstantStepRPUDeviceCuda<double>;
#endif
#ifdef RPU_USE_FP16
template class ConstantStepRPUDeviceCuda<half_t>;
#endif

} // namespace RPU
