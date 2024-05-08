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

#include "rpu_constantstep_device.h"
#include "utility_functions.h"
#include <iostream>
// #include <random>
#include <chrono>
#include <cmath>
#include <limits>

namespace RPU {

/********************************************************************************
 * Constant Step RPU Device
 *********************************************************************************/

template <typename T>
void ConstantStepRPUDevice<T>::populate(
    const ConstantStepRPUDeviceMetaParameter<T> &p, RealWorldRNG<T> *rng) {
  PulsedRPUDevice<T>::populate(p, rng);
}

template <typename T>
void ConstantStepRPUDevice<T>::doSparseUpdate(
    T **weights, int i, const int *x_signed_indices, int x_count, int d_sign, RNG<T> *rng) {

  T *scale_down = this->w_scale_down_[i];
  T *scale_up = this->w_scale_up_[i];
  T *w = weights[i];
  T *min_bound = this->w_min_bound_[i];
  T *max_bound = this->w_max_bound_[i];
  T dw_min_std = getPar().dw_min_std;

  if (dw_min_std > (T)0.0) {
    PULSED_UPDATE_W_LOOP(
        T dw = 0; if (sign > 0) {
          dw = ((T)1.0 + dw_min_std * rng->sampleGauss()) * scale_down[j];
          w[j] -= dw;
        } else {
          dw = ((T)1.0 + dw_min_std * rng->sampleGauss()) * scale_up[j];
          w[j] += dw;
        } w[j] = MIN(w[j], max_bound[j]);
        w[j] = MAX(w[j], min_bound[j]););
  } else {

    PULSED_UPDATE_W_LOOP(
        if (sign > 0) { w[j] -= scale_down[j]; } else { w[j] += scale_up[j]; } w[j] =
            MIN(w[j], max_bound[j]);
        w[j] = MAX(w[j], min_bound[j]););
  }
}

template <typename T>
void ConstantStepRPUDevice<T>::doDenseUpdate(T **weights, int *coincidences, RNG<T> *rng) {

  T *scale_down = this->w_scale_down_[0];
  T *scale_up = this->w_scale_up_[0];
  T *w = weights[0];
  T *min_bound = this->w_min_bound_[0];
  T *max_bound = this->w_max_bound_[0];
  T dw_min_std = getPar().dw_min_std;

  PULSED_UPDATE_W_LOOP_DENSE(
      T dw = dw_min_std > (T)0.0 ? dw_min_std * rng->sampleGauss() : (T)0.0; if (sign > 0) {
        dw = ((T)1.0 + dw) * scale_down[j];
        w[j] -= dw;
      } else {
        dw = ((T)1.0 + dw) * scale_up[j];
        w[j] += dw;
      } w[j] = MIN(w[j], max_bound[j]);
      w[j] = MAX(w[j], min_bound[j]);

  );
}

template class ConstantStepRPUDevice<float>;
#ifdef RPU_USE_DOUBLE
template class ConstantStepRPUDevice<double>;
#endif
#ifdef RPU_USE_FP16
template class ConstantStepRPUDevice<half_t>;
#endif

} // namespace RPU
