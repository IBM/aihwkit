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

#include "rpu_piecewisestep_device.h"
#include "utility_functions.h"
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>

namespace RPU {

/********************************************************************************
 * PiecewiseStepDevice
 *********************************************************************************/

template <typename T>
void PiecewiseStepRPUDevice<T>::populate(
    const PiecewiseStepRPUDeviceMetaParameter<T> &p, RealWorldRNG<T> *rng) {

  PulsedRPUDevice<T>::populate(p, rng); // will clone par
}

namespace {
template <typename T>
inline void update_once(
    T &w,
    T &w_apparent,
    const int &sign,
    const T &scale_down,
    const T &scale_up,
    const T &min_bound,
    const T &max_bound,
    const std::vector<T> &piecewise_up_vec,
    const std::vector<T> &piecewise_down_vec,
    const T &dw_min_std,
    const T &write_noise_std,
    RNG<T> *rng) {

  size_t n_sections = piecewise_up_vec.size() - 1;

  if (n_sections <= 0 || (max_bound <= min_bound)) {
    // short-cut for 1 value or same bounds (dummy update)
    if (sign > 0) {
      T interpolated_down =
          piecewise_down_vec.size() > 0 ? piecewise_down_vec[0] * scale_down : scale_down;
      w -= interpolated_down * ((T)1.0 + dw_min_std * rng->sampleGauss());
    } else {
      T interpolated_up = piecewise_up_vec.size() > 0 ? piecewise_up_vec[0] * scale_up : scale_up;
      w += interpolated_up * ((T)1.0 + dw_min_std * rng->sampleGauss());
    }

  } else {
    // interpolation
    T w_scaled = MAX((w - min_bound) / (max_bound - min_bound) * (T)n_sections, (T)0.0);
    size_t w_index = MIN((size_t)floorf(w_scaled), n_sections - 1);
    T t = MIN(w_scaled - (T)w_index, (T)1.0); // convex fraction
    T t1 = (T)1.0 - t;

    if (sign > 0) {
      T interpolated_down =
          scale_down * (t1 * piecewise_down_vec[w_index] + t * piecewise_down_vec[w_index + 1]);
      w -= interpolated_down * ((T)1.0 + dw_min_std * rng->sampleGauss());
    } else {
      T interpolated_up =
          scale_up * (t1 * piecewise_up_vec[w_index] + t * piecewise_up_vec[w_index + 1]);
      w += interpolated_up * ((T)1.0 + dw_min_std * rng->sampleGauss());
    }
  }

  w = MAX(w, min_bound);
  w = MIN(w, max_bound);

  if (write_noise_std > (T)0.0) {
    w_apparent = w + write_noise_std * rng->sampleGauss();
  }
}

} // namespace

template <typename T>
void PiecewiseStepRPUDevice<T>::doSparseUpdate(
    T **weights, int i, const int *x_signed_indices, int x_count, int d_sign, RNG<T> *rng) {

  const auto &par = getPar();

  T *scale_down = this->w_scale_down_[i];
  T *scale_up = this->w_scale_up_[i];
  T *w = par.usesPersistentWeight() ? this->w_persistent_[i] : weights[i];
  T *w_apparent = weights[i];
  T *min_bound = this->w_min_bound_[i];
  T *max_bound = this->w_max_bound_[i];
  T write_noise_std = par.getScaledWriteNoise();

  PULSED_UPDATE_W_LOOP(update_once(
                           w[j], w_apparent[j], sign, scale_down[j], scale_up[j], min_bound[j],
                           max_bound[j], par.piecewise_up_vec, par.piecewise_down_vec,
                           par.dw_min_std, write_noise_std, rng););
}

template <typename T>
void PiecewiseStepRPUDevice<T>::doDenseUpdate(T **weights, int *coincidences, RNG<T> *rng) {

  const auto &par = getPar();

  T *scale_down = this->w_scale_down_[0];
  T *scale_up = this->w_scale_up_[0];
  T *w = par.usesPersistentWeight() ? this->w_persistent_[0] : weights[0];
  T *w_apparent = weights[0];
  T *min_bound = this->w_min_bound_[0];
  T *max_bound = this->w_max_bound_[0];
  T write_noise_std = par.getScaledWriteNoise();

  PULSED_UPDATE_W_LOOP_DENSE(update_once(
                                 w[j], w_apparent[j], sign, scale_down[j], scale_up[j],
                                 min_bound[j], max_bound[j], par.piecewise_up_vec,
                                 par.piecewise_down_vec, par.dw_min_std, write_noise_std, rng););
}

template class PiecewiseStepRPUDevice<float>;
#ifdef RPU_USE_DOUBLE
template class PiecewiseStepRPUDevice<double>;
#endif
#ifdef RPU_USE_FP16
template class PiecewiseStepRPUDevice<half_t>;
#endif

} // namespace RPU
