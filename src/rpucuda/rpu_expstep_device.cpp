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
    T &w_apparent,
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
    const T &write_noise_std,
    RNG<T> *rng) {
  T b_diff = (max_bound - min_bound);
  if (b_diff > (T)0.0) {
    T z = (T)2.0 * w / b_diff * es_a + es_b;

    if (sign > 0) {
      T y_down = MAX((T)1.0 - es_A_down * (T)expf(es_gamma_down * (-z)), (T)0.0);
      w -= y_down * ((T)1.0 + dw_min_std * rng->sampleGauss()) * scale_down;

    } else {
      T y_up = MAX((T)1.0 - es_A_up * (T)expf(es_gamma_up * z), (T)0.0);
      w += y_up * ((T)1.0 + dw_min_std * rng->sampleGauss()) * scale_up;
    }

    // always check both bounds
    w = MIN(w, max_bound);
    w = MAX(w, min_bound);

    if (write_noise_std > (T)0.0) {
      w_apparent = w + write_noise_std * rng->sampleGauss();
    }
  }
}

template <typename T>
inline void update_once_complex_noise(
    T &w,
    T &w_apparent,
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
    const T &dw_min_std_add,
    const T &dw_min_std_slope,
    const T &write_noise_std,
    RNG<T> *rng) {
  T b_diff = (max_bound - min_bound);
  if (b_diff > (T)0.0) {
    T z = (T)2.0 * w / b_diff * es_a + es_b;
    T dw;

    if (sign > 0) {
      T y_down = MAX((T)1.0 - (T)es_A_down * (T)expf(es_gamma_down * (-z)), (T)0);
      dw = -y_down * scale_down;

    } else {
      T y_up = MAX((T)1.0 - (T)es_A_up * (T)expf(es_gamma_up * z), (T)0);
      dw = y_up * scale_up;
    }

    T dw_std = dw_min_std * ((T)fabsf(dw) + dw_min_std_add + dw_min_std_slope * (T)fabsf(w));
    w += dw + dw_std * rng->sampleGauss();

    // always check both bounds
    w = MIN(w, max_bound);
    w = MAX(w, min_bound);

    if (write_noise_std > (T)0.0) {
      w_apparent = w + write_noise_std * rng->sampleGauss();
    }
  }
}

template <typename T>
void ExpStepRPUDevice<T>::doSparseUpdate(
    T **weights, int i, const int *x_signed_indices, int x_count, int d_sign, RNG<T> *rng) {

  const auto &par = getPar();
  T *scale_down = this->w_scale_down_[i];
  T *scale_up = this->w_scale_up_[i];
  T *w = par.usesPersistentWeight() ? this->w_persistent_[i] : weights[i];
  T *w_apparent = weights[i];
  T *min_bound = this->w_min_bound_[i];
  T *max_bound = this->w_max_bound_[i];

  T write_noise_std = par.getScaledWriteNoise();
  if (par.hasComplexNoise()) {
    PULSED_UPDATE_W_LOOP(update_once_complex_noise(
                             w[j], w_apparent[j], sign, min_bound[j], max_bound[j], scale_down[j],
                             scale_up[j], par.es_a, par.es_b, par.es_A_down, par.es_A_up,
                             par.es_gamma_down, par.es_gamma_up, par.dw_min_std, par.dw_min_std_add,
                             par.dw_min_std_slope, write_noise_std, rng););
  } else {

    PULSED_UPDATE_W_LOOP(update_once(
                             w[j], w_apparent[j], sign, min_bound[j], max_bound[j], scale_down[j],
                             scale_up[j], par.es_a, par.es_b, par.es_A_down, par.es_A_up,
                             par.es_gamma_down, par.es_gamma_up, par.dw_min_std, write_noise_std,
                             rng););
  }
};

template <typename T>
void ExpStepRPUDevice<T>::doDenseUpdate(T **weights, int *coincidences, RNG<T> *rng) {

  const auto &par = getPar();
  T *scale_down = this->w_scale_down_[0];
  T *scale_up = this->w_scale_up_[0];
  T *w = par.usesPersistentWeight() ? this->w_persistent_[0] : weights[0];
  T *w_apparent = weights[0];
  T *min_bound = this->w_min_bound_[0];
  T *max_bound = this->w_max_bound_[0];

  T write_noise_std = par.getScaledWriteNoise();
  if (par.hasComplexNoise()) {

    PULSED_UPDATE_W_LOOP_DENSE(
        update_once_complex_noise(
            w[j], w_apparent[j], sign, min_bound[j], max_bound[j], scale_down[j], scale_up[j],
            par.es_a, par.es_b, par.es_A_down, par.es_A_up, par.es_gamma_down, par.es_gamma_up,
            par.dw_min_std, par.dw_min_std_add, par.dw_min_std_slope, write_noise_std, rng););

  } else {
    PULSED_UPDATE_W_LOOP_DENSE(update_once(
                                   w[j], w_apparent[j], sign, min_bound[j], max_bound[j],
                                   scale_down[j], scale_up[j], par.es_a, par.es_b, par.es_A_down,
                                   par.es_A_up, par.es_gamma_down, par.es_gamma_up, par.dw_min_std,
                                   write_noise_std, rng););
  }
}

template class ExpStepRPUDevice<float>;

#ifdef RPU_USE_DOUBLE
template class ExpStepRPUDevice<double>;
#endif
#ifdef RPU_USE_FP16
template class ExpStepRPUDevice<half_t>;
#endif

} // namespace RPU
