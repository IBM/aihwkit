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

#include "rpu_expstep_device.h"

namespace RPU {

template <typename T>
void ExpStepRPUDevice<T>::populate(
    const ExpStepRPUDeviceMetaParameter<T> &p, RealWorldRNG<T> *rng) {
  PulsedRPUDevice<T>::populate(p, rng);
}

template <typename T>
inline void update_once(
    T &w,
    int &sign,
    T &min_bound,
    T &max_bound,
    T &scale_down,
    T &scale_up,
    const T &es_a,
    const T &es_b,
    const T &es_A_down,
    const T &es_A_up,
    const T &es_gamma_down,
    const T &es_gamma_up,
    const T &dw_min_std,
    RNG<T> *rng) {
  T b_diff = (max_bound - min_bound);
  if (b_diff > 0.0) {
    T z = (T)2.0 * w / b_diff * es_a + es_b;

    if (sign > 0) {
      T y_down = MAX((T)1 - es_A_down * exp(es_gamma_down * (-z)), (T)0);
      w -= y_down * ((T)1.0 + dw_min_std * rng->sampleGauss()) * scale_down;

    } else {
      T y_up = MAX(1 - es_A_up * exp(es_gamma_up * z), (T)0);
      w += y_up * ((T)1.0 + dw_min_std * rng->sampleGauss()) * scale_up;
    }

    // always check both bounds
    w = MIN(w, max_bound);
    w = MAX(w, min_bound);
  }
}

template <typename T>
void ExpStepRPUDevice<T>::doSparseUpdate(
    T **weights, int i, const int *x_signed_indices, int x_count, int d_sign, RNG<T> *rng) {

  T *scale_down = this->w_scale_down_[i];
  T *scale_up = this->w_scale_up_[i];
  T *w = weights[i];
  T *min_bound = this->w_min_bound_[i];
  T *max_bound = this->w_max_bound_[i];

  const auto &par = getPar();

  PULSED_UPDATE_W_LOOP(update_once(
                           w[j], sign, min_bound[j], max_bound[j], scale_down[j], scale_up[j],
                           par.es_a, par.es_b, par.es_A_down, par.es_A_up, par.es_gamma_down,
                           par.es_gamma_up, par.dw_min_std, rng););
};

template <typename T>
void ExpStepRPUDevice<T>::doDenseUpdate(T **weights, int *coincidences, RNG<T> *rng) {

  T *scale_down = this->w_scale_down_[0];
  T *scale_up = this->w_scale_up_[0];
  T *w = weights[0];
  T *min_bound = this->w_min_bound_[0];
  T *max_bound = this->w_max_bound_[0];

  const auto &par = getPar();

  PULSED_UPDATE_W_LOOP_DENSE(update_once(
                                 w[j], sign, min_bound[j], max_bound[j], scale_down[j], scale_up[j],
                                 par.es_a, par.es_b, par.es_A_down, par.es_A_up, par.es_gamma_down,
                                 par.es_gamma_up, par.dw_min_std, rng););
}

template class ExpStepRPUDevice<float>;

#ifdef RPU_USE_DOUBLE
template class ExpStepRPUDevice<double>;
#endif

} // namespace RPU
