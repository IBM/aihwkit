/**
 * (C) Copyright 2020, 2021 IBM. All Rights Reserved.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
 */

#include "rpu_selfdefine_device.h"
#include "utility_functions.h"
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>

namespace RPU {

/********************************************************************************
 * Self Defined RPU Device
 *********************************************************************************/

template <typename T>
void SelfDefineRPUDevice<T>::populate(
    const SelfDefineRPUDeviceMetaParameter<T> &p, RealWorldRNG<T> *rng) {

  PulsedRPUDevice<T>::populate(p, rng); // will clone par
}

namespace {
template <typename T>
inline void update_once(
    T &w,
    T &w_apparent,
    int &sign,
    T &scale_down,
    T &scale_up,
    T &min_bound,
    T &max_bound,
    T &interpolated_down,
    T &interpolated_up,
    const T &dw_min_std,
    const T &write_noise_std,
    RNG<T> *rng) {

  if (sign > 0) {
    w -= interpolated_down * ((T)1.0 + dw_min_std * rng->sampleGauss());
  } else {
    w += interpolated_up * ((T)1.0 + dw_min_std * rng->sampleGauss());
  }
  w = MAX(w, min_bound);
  w = MIN(w, max_bound);

  if (write_noise_std > (T)0.0) {
    w_apparent = w + write_noise_std * ((T)1.0 + dw_min_std * rng->sampleGauss());
  }
}

} // namespace

template <typename T>
void SelfDefineRPUDevice<T>::doSparseUpdate(
    T **weights, int i, const int *x_signed_indices, int x_count, int d_sign, RNG<T> *rng) {

  const auto &par = getPar();

  T *scale_down = this->w_scale_down_[i];
  T *scale_up = this->w_scale_up_[i];
  T *w = par.usesPersistentWeight() ? this->w_persistent_[i] : weights[i];
  T *w_apparent = weights[i];
  T *min_bound = this->w_min_bound_[i];
  T *max_bound = this->w_max_bound_[i];
  
  std::vector<T> sd_up_pulse = par.sd_up_pulse;
  std::vector<T> sd_down_pulse = par.sd_down_pulse;
  T sd_n_points = par.sd_n_points;
  int n_points = (int)sd_n_points;

  T interpolated_down = 0.0;
  T interpolated_up = 0.0;

  for (int n = 0; n < n_points - 1; n++) {
    T increment = abs(*max_bound - *min_bound) / (n_points - 1);
    T sd_up_weight = *max_bound - (increment * n);
    T sd_up_weight_next = *max_bound - (increment * (n + 1));
    if (*w <= sd_up_weight && *w >= sd_up_weight_next) {
      interpolated_up = sd_up_pulse[n] + ((*w - sd_up_weight) * (sd_up_pulse[n + 1] - sd_up_pulse[n]) / 
                                         (sd_up_weight_next - sd_up_weight));
      interpolated_down = sd_down_pulse[n] + ((*w - sd_up_weight) * (sd_down_pulse[n + 1] - sd_down_pulse[n]) / 
                                             (sd_up_weight_next - sd_up_weight));
      break;
    }
  }

  T write_noise_std = par.getScaledWriteNoise();
  PULSED_UPDATE_W_LOOP(update_once(
                           w[j], w_apparent[j], sign, scale_down[j], scale_up[j], 
                           min_bound[j], max_bound[j], interpolated_down, interpolated_up, par.dw_min_std, 
                           write_noise_std, rng););
}

template <typename T>
void SelfDefineRPUDevice<T>::doDenseUpdate(T **weights, int *coincidences, RNG<T> *rng) {

  const auto &par = getPar();

  T *scale_down = this->w_scale_down_[0];
  T *scale_up = this->w_scale_up_[0];
  T *w = par.usesPersistentWeight() ? this->w_persistent_[0] : weights[0];
  T *w_apparent = weights[0];
  T *min_bound = this->w_min_bound_[0];
  T *max_bound = this->w_max_bound_[0];
  T write_noise_std = par.getScaledWriteNoise();

  std::vector<T> sd_up_pulse = par.sd_up_pulse;
  std::vector<T> sd_down_pulse = par.sd_down_pulse;
  T sd_n_points = par.sd_n_points;
  int n_points = (int)sd_n_points;

  T interpolated_down = 0.0;
  T interpolated_up = 0.0;

  for (int n = 0; n < n_points - 1; n++) {
    T increment = abs(*max_bound - *min_bound) / (n_points - 1);
    T sd_up_weight = *max_bound - (increment * n);
    T sd_up_weight_next = *max_bound - (increment * (n + 1));
    if (*w <= sd_up_weight && *w >= sd_up_weight_next) {
      interpolated_up = sd_up_pulse[n] + ((*w - sd_up_weight) * (sd_up_pulse[n + 1] - sd_up_pulse[n]) / 
                                         (sd_up_weight_next - sd_up_weight));
      interpolated_down = sd_down_pulse[n] + ((*w - sd_up_weight) * (sd_down_pulse[n + 1] - sd_down_pulse[n]) / 
                                             (sd_up_weight_next - sd_up_weight));
      break;
    }
  }

  PULSED_UPDATE_W_LOOP_DENSE(update_once(
                               w[j], w_apparent[j], sign, scale_down[j], scale_up[j], 
                               min_bound[j], max_bound[j], interpolated_down, interpolated_up, par.dw_min_std, 
                               write_noise_std, rng););
}

template class SelfDefineRPUDevice<float>;
#ifdef RPU_USE_DOUBLE
template class SelfDefineRPUDevice<double>;
#endif

} // namespace RPU
