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

#include "rpu_powstep_device.h"
#include "utility_functions.h"
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>

namespace RPU {

/********************************************************************************
 * Linear Step RPU Device
 *********************************************************************************/

template <typename T>
void PowStepRPUDevice<T>::populate(
    const PowStepRPUDeviceMetaParameter<T> &p, RealWorldRNG<T> *rng) {

  PulsedRPUDevice<T>::populate(p, rng); // will clone par
  auto &par = getPar();

  T gamma = par.ps_gamma;
  T gain_std = par.ps_gamma_dtod;
  T up_down_std = par.ps_gamma_up_down_dtod;
  T up_down = par.ps_gamma_up_down;

  T up_bias = up_down > (T)0.0 ? (T)0.0 : up_down;
  T down_bias = up_down > (T)0.0 ? -up_down : (T)0.0;

  for (int i = 0; i < this->d_size_; ++i) {
    for (int j = 0; j < this->x_size_; ++j) {

      T gain = (T)1.0 + gain_std * rng->sampleGauss();
      T r = up_down_std * rng->sampleGauss();

      w_gamma_up_[i][j] = (up_bias + gain + r) * gamma;
      w_gamma_down_[i][j] = (down_bias + gain - r) * gamma;

      if (par.enforce_consistency) {
        w_gamma_up_[i][j] = (T)fabsf(w_gamma_up_[i][j]);
        w_gamma_down_[i][j] = (T)fabsf(w_gamma_down_[i][j]);
      }
    }
  }
}

template <typename T> void PowStepRPUDevice<T>::printDP(int x_count, int d_count) const {

  if (x_count < 0 || x_count > this->x_size_) {
    x_count = this->x_size_;
  }

  if (d_count < 0 || d_count > this->d_size_) {
    d_count = this->d_size_;
  }
  bool persist_if = getPar().usesPersistentWeight();

  for (int i = 0; i < d_count; ++i) {
    for (int j = 0; j < x_count; ++j) {
      std::cout.precision(5);
      std::cout << i << "," << j << ": ";
      std::cout << "[<" << this->w_max_bound_[i][j] << ",";
      std::cout << this->w_min_bound_[i][j] << ">,<";
      std::cout << this->w_scale_up_[i][j] << ",";
      std::cout << this->w_scale_down_[i][j] << ">,<";
      std::cout << w_gamma_up_[i][j] << ",";
      std::cout << w_gamma_down_[i][j] << ">]";
      std::cout.precision(10);
      std::cout << this->w_decay_scale_[i][j] << ", ";
      std::cout.precision(6);
      std::cout << this->w_diffusion_rate_[i][j] << ", ";
      std::cout << this->w_reset_bias_[i][j];
      if (persist_if) {
        std::cout << ", " << this->w_persistent_[i][j];
      }
      std::cout << "]";
    }
    std::cout << std::endl;
  }
}

namespace {
template <typename T>
inline void update_once(
    T &w,
    T &w_apparent,
    int &sign,
    T &scale_down,
    T &scale_up,
    T &gamma_down,
    T &gamma_up,
    T &min_bound,
    T &max_bound,
    const T &dw_min_std,
    const T &write_noise_std,
    RNG<T> *rng) {
  T range = max_bound - min_bound;
  if (range == (T)0.0) {
    return;
  }
  if (sign > 0) {
    w -= scale_down * (T)powf((w - min_bound) / range, gamma_down) *
         ((T)1.0 + dw_min_std * rng->sampleGauss());
  } else {
    w += scale_up * (T)powf(((max_bound - w) / range), gamma_up) *
         ((T)1.0 + dw_min_std * rng->sampleGauss());
  }
  w = MAX(w, min_bound);
  w = MIN(w, max_bound);

  if (write_noise_std > (T)0.0) {
    w_apparent = w + write_noise_std * rng->sampleGauss();
  }
}

} // namespace

template <typename T>
void PowStepRPUDevice<T>::doSparseUpdate(
    T **weights, int i, const int *x_signed_indices, int x_count, int d_sign, RNG<T> *rng) {

  const auto &par = getPar();

  T *scale_down = this->w_scale_down_[i];
  T *scale_up = this->w_scale_up_[i];
  T *gamma_down = w_gamma_down_[i];
  T *gamma_up = w_gamma_up_[i];
  T *w = par.usesPersistentWeight() ? this->w_persistent_[i] : weights[i];
  T *w_apparent = weights[i];
  T *min_bound = this->w_min_bound_[i];
  T *max_bound = this->w_max_bound_[i];

  T write_noise_std = par.getScaledWriteNoise();
  PULSED_UPDATE_W_LOOP(update_once(
                           w[j], w_apparent[j], sign, scale_down[j], scale_up[j], gamma_down[j],
                           gamma_up[j], min_bound[j], max_bound[j], par.dw_min_std, write_noise_std,
                           rng););
}

template <typename T>
void PowStepRPUDevice<T>::doDenseUpdate(T **weights, int *coincidences, RNG<T> *rng) {

  const auto &par = getPar();

  T *scale_down = this->w_scale_down_[0];
  T *scale_up = this->w_scale_up_[0];
  T *gamma_down = w_gamma_down_[0];
  T *gamma_up = w_gamma_up_[0];
  T *w = par.usesPersistentWeight() ? this->w_persistent_[0] : weights[0];
  T *w_apparent = weights[0];
  T *min_bound = this->w_min_bound_[0];
  T *max_bound = this->w_max_bound_[0];
  T write_noise_std = par.getScaledWriteNoise();

  PULSED_UPDATE_W_LOOP_DENSE(update_once(
                                 w[j], w_apparent[j], sign, scale_down[j], scale_up[j],
                                 gamma_down[j], gamma_up[j], min_bound[j], max_bound[j],
                                 par.dw_min_std, write_noise_std, rng););
}

template class PowStepRPUDevice<float>;
#ifdef RPU_USE_DOUBLE
template class PowStepRPUDevice<double>;
#endif
#ifdef RPU_USE_FP16
template class PowStepRPUDevice<half_t>;
#endif

} // namespace RPU
