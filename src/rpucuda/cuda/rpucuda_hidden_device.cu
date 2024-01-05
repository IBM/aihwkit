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
#include "rpucuda_hidden_device.h"
#include <memory>

namespace RPU {

#define UPDATE_ONCE                                                                                \
  if (hs_noise_std_dw > (T)0.0) {                                                                  \
    T stoch_value = curand_normal(&local_state);                                                   \
    stoch_value *= hs_noise_std_dw;                                                                \
    hw += hs_dw * ((T)1.0 + stoch_value);                                                          \
  } else {                                                                                         \
    hw += hs_dw;                                                                                   \
  }                                                                                                \
  if (hw > (T)1.0 || hw < (T)-1.0) {                                                               \
                                                                                                   \
    T dw = (hw > (T)1) ? ((T)par_4.w) : ((T)-par_4.y);                                             \
    hw = (T)0.0;                                                                                   \
    if (noise_std_dw > (T)0.0) {                                                                   \
      T stoch_value = curand_normal(&local_state);                                                 \
      stoch_value *= noise_std_dw;                                                                 \
      w += dw * ((T)1.0 + stoch_value);                                                            \
    } else {                                                                                       \
      w += dw;                                                                                     \
    }                                                                                              \
  }

template <typename T> struct UpdateFunctorHiddenStep {

  __device__ __forceinline__ void operator()(
      T &w,
      uint32_t n,
      uint32_t negative,
      const param4_t par_4,
      const param2_t hs_scale,
      T &hw,
      const T *global_pars,
      const T global_params_count,
      T noise_std_dw,
      curandState &local_state) {

    UNUSED(global_params_count);

    // par_4 order (min_bound, scale_down, max_bound, scale_up )
    // par_2 order (scale_down, scale_up )
    // par_1 hidden weight
    // global_pars #1:  hs_dw_min_std

    T hs_dw = (negative > 0) ? ((T)hs_scale.y) : (-(T)hs_scale.x); // [1] [0]
    T hs_noise_std_dw = global_pars[0];

    // n is larger 0 in any case
    if (n == 1) {
      UPDATE_ONCE;
    } else {
      for (int i_updates = 0; i_updates < n; i_updates++) {
        UPDATE_ONCE;
      }
    }
    // check bounds after loop is enough (only one direction)
    T wmax = (T)par_4.z; // [2]
    w = (w > wmax) ? wmax : w;
    T wmin = (T)par_4.x; // [0]
    w = (w < wmin) ? wmin : w;
  }
};

RPUCUDA_DEVICE_ADD_FUNCTOR_UPDATE_KERNELS(HiddenStep, UpdateFunctorHiddenStep<T>, 1);

template class HiddenStepRPUDeviceCuda<float>;
#ifdef RPU_USE_DOUBLE
template class HiddenStepRPUDeviceCuda<double>;
#endif
#ifdef RPU_USE_FP16
template class HiddenStepRPUDeviceCuda<half_t>;
#endif

#undef UPDATE_ONCE

} // namespace RPU
