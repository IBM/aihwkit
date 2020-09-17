/**
 * (C) Copyright 2020 IBM. All Rights Reserved.
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
#include "rpucuda_expstep_device.h"

namespace RPU {

#define UPDATE_ONCE                                                                                \
  {                                                                                                \
    T z = 2.0 * w / b_diff * a + b;                                                                \
    T y = 1.0 - A * __expf(gamma * z);                                                             \
    if (y > 0) {                                                                                   \
      if (noise_std_dw > 0) {                                                                      \
        T stoch_value = curand_normal(&local_state);                                               \
        stoch_value *= noise_std_dw;                                                               \
        w += y * (stoch_value + 1.0) * dw;                                                         \
      } else {                                                                                     \
        w += y * dw;                                                                               \
      }                                                                                            \
      w = (w > wmax) ? wmax : w;                                                                   \
      w = (w < wmin) ? wmin : w;                                                                   \
    }                                                                                              \
  }

template <typename T> struct UpdateFunctorExpStep {

  __device__ __forceinline__ void operator()(
      T &w,
      uint32_t n,
      uint32_t negative,
      const float4 par_4,
      const float2 par_2,
      T &par_1,
      const T *global_pars,
      T noise_std_dw,
      curandState &local_state)

  {
    // par_4 order (min_bound, scale_down, max_bound, scale_up )
    // global_pars see below
    T wmax = par_4.z; //[2];
    T wmin = par_4.x; //[0];
    T b_diff = (wmax - wmin);

    if (b_diff > 0) { // only do something when bounds make sense

      T A = negative ? global_pars[1] : global_pars[0];        // 1: up, 0: down
      T gamma = negative ? global_pars[3] : (-global_pars[2]); // 3: up, 2 down
      T a = global_pars[4];
      T b = global_pars[5];
      T dw = (negative > 0) ? (par_4.w) : (-par_4.y); // [3], [1]

      // n is larger 0 in any case
      if (n == 1) {
        UPDATE_ONCE;
      } else {
        for (int i_updates = 0; i_updates < n; i_updates++) {
          UPDATE_ONCE;
        }
      }
    }
  }
};
#undef UPDATE_ONCE

RPUCUDA_DEVICE_ADD_FUNCTOR_UPDATE_KERNELS(ExpStep, UpdateFunctorExpStep<T>, 6);

template class ExpStepRPUDeviceCuda<float>;
#ifdef RPU_USE_DOUBLE
template class ExpStepRPUDeviceCuda<double>;
#endif
} // namespace RPU
