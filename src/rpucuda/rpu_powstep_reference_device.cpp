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

#include "rpu_powstep_reference_device.h"
#include "utility_functions.h"
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>

namespace RPU {

/********************************************************************************
 * SoftBoundsReference RPU Device
 *********************************************************************************/

template <typename T>
void PowStepReferenceRPUDevice<T>::populate(
    const PowStepReferenceRPUDeviceMetaParameter<T> &p, RealWorldRNG<T> *rng) {

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

      // reference
      w_reference_[i][j] = par.reference_mean + par.reference_std * rng->sampleGauss();

      if (par.subtract_symmetry_point &&
          (this->w_max_bound_[i][j] > (T)0.0 && this->w_min_bound_[i][j] < (T)0.0)) {

        T gu = w_gamma_up_[i][j];
        T gd = w_gamma_down_[i][j];
        T su = this->w_scale_up_[i][j];
        T sd = this->w_scale_down_[i][j];
        T bu = this->w_max_bound_[i][j];
        T bd = this->w_min_bound_[i][j];
        T sp = 0;

        // just compute it
        T r = (bu - bd);
        int n_steps = par.n_estimation_steps;
        if (n_steps <= 0) {
          n_steps = MIN((int)roundf((T)10 * (r / MIN((T)fabsf(sd), (T)fabsf(su)))), 10000);
        }
        for (int ii = 0; ii < n_steps; ii++) {
          sp += -sd * (T)powf((sp - bd) / r, gd) + su * (T)powf((bu - sp) / r, gu);
        }
        w_reference_[i][j] += sp;

        this->w_reset_bias_[i][j] += sp;
      }
    }
  }
}

template <typename T> void PowStepReferenceRPUDevice<T>::printDP(int x_count, int d_count) const {

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
      std::cout << w_gamma_down_[i][j] << ",";
      std::cout << w_reference_[i][j] << ">]";
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
inline void update_once_reference(
    T &w,
    int &sign,
    T &scale_down,
    T &scale_up,
    T &gamma_down,
    T &gamma_up,
    T &ref,
    T &min_bound,
    T &max_bound,
    const T &dw_min_std,
    RNG<T> *rng) {
  T range = max_bound - min_bound;
  if (range == (T)0.0) {
    return;
  }
  w += ref; // first add

  if (sign > 0) {
    w -= scale_down * (T)powf((w - min_bound) / range, gamma_down) *
         ((T)1.0 + dw_min_std * rng->sampleGauss());
  } else {
    w += scale_up * (T)powf((max_bound - w) / range, gamma_up) *
         ((T)1.0 + dw_min_std * rng->sampleGauss());
  }
  w = MAX(w, min_bound);
  w = MIN(w, max_bound);

  w -= ref; // subtract again
}

} // namespace

template <typename T>
void PowStepReferenceRPUDevice<T>::doSparseUpdate(
    T **weights, int i, const int *x_signed_indices, int x_count, int d_sign, RNG<T> *rng) {

  const auto &par = getPar();

  T *scale_down = this->w_scale_down_[i];
  T *scale_up = this->w_scale_up_[i];
  T *gamma_down = w_gamma_down_[i];
  T *gamma_up = w_gamma_up_[i];
  T *ref = w_reference_[i];
  T *w = weights[i];
  T *min_bound = this->w_min_bound_[i];
  T *max_bound = this->w_max_bound_[i];

  PULSED_UPDATE_W_LOOP(update_once_reference(
                           w[j], sign, scale_down[j], scale_up[j], gamma_down[j], gamma_up[j],
                           ref[j], min_bound[j], max_bound[j], par.dw_min_std, rng););
}

template <typename T>
void PowStepReferenceRPUDevice<T>::doDenseUpdate(T **weights, int *coincidences, RNG<T> *rng) {

  const auto &par = getPar();

  T *scale_down = this->w_scale_down_[0];
  T *scale_up = this->w_scale_up_[0];
  T *gamma_down = w_gamma_down_[0];
  T *gamma_up = w_gamma_up_[0];
  T *ref = w_reference_[0];
  T *w = weights[0];
  T *min_bound = this->w_min_bound_[0];
  T *max_bound = this->w_max_bound_[0];

  PULSED_UPDATE_W_LOOP_DENSE(update_once_reference(
                                 w[j], sign, scale_down[j], scale_up[j], gamma_down[j], gamma_up[j],
                                 ref[j], min_bound[j], max_bound[j], par.dw_min_std, rng););
}

template class PowStepReferenceRPUDevice<float>;
#ifdef RPU_USE_DOUBLE
template class PowStepReferenceRPUDevice<double>;
#endif
#ifdef RPU_USE_FP16
template class PowStepReferenceRPUDevice<half_t>;
#endif

} // namespace RPU
