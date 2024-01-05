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

#include "rpu_softbounds_reference_device.h"
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
void SoftBoundsReferenceRPUDevice<T>::populate(
    const SoftBoundsReferenceRPUDeviceMetaParameter<T> &p, RealWorldRNG<T> *rng) {

  PulsedRPUDevice<T>::populate(p, rng); // will clone par
  auto &par = getPar();

  for (int i = 0; i < this->d_size_; ++i) {

    for (int j = 0; j < this->x_size_; ++j) {

      T w_min = this->w_min_bound_[i][j];
      T scale_down = this->w_scale_down_[i][j];
      T w_max = this->w_max_bound_[i][j];
      T scale_up = this->w_scale_up_[i][j];

      if (w_max > (T)0.0 && w_min < (T)0.0) {
        // only apply an additional variation if consistent
        // no real need to compute the full slope here, but for clarity

        T current_slope_down = -scale_down / w_min;
        current_slope_down *= (T)fabsf((T)1.0 + par.slope_down_dtod * rng->sampleGauss());

        T current_slope_up = scale_up / w_max;
        current_slope_up *= (T)fabsf((T)1.0 + par.slope_up_dtod * rng->sampleGauss());

        w_max = scale_up / current_slope_up;
        w_min = -scale_down / current_slope_down;
        this->w_max_bound_[i][j] = w_max;
        this->w_min_bound_[i][j] = w_min;
      }

      // reference
      w_reference_[i][j] = par.reference_mean + par.reference_std * rng->sampleGauss();

      if (par.subtract_symmetry_point && (w_max > (T)0.0 && w_min < (T)0.0)) {
        // scale_down (1 - w / w_min) == scale_up * (1 - w / w_max)
        // scale_down -  scale_down * w / w_min == scale_up  -scale_up *  w / w_max
        // scale_up *  w / w_max -  scale_down * w / w_min == scale_up - scale_down
        // w ==  (scale_up - scale_down) / (scale_up  / w_max  -  scale_down / w_min)

        T sp = (scale_up - scale_down) / (scale_up / w_max - scale_down / w_min);
        w_reference_[i][j] += sp;

        this->w_reset_bias_[i][j] += sp;
      }
    }
  }
}

template <typename T>
void SoftBoundsReferenceRPUDevice<T>::printDP(int x_count, int d_count) const {

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
inline void update_once_mult(
    T &w,
    T &w_apparent,
    int &sign,
    T &scale_down,
    T &scale_up,
    T &ref,
    T &min_bound,
    T &max_bound,
    const T &dw_min_std,
    const T &write_noise_std,
    RNG<T> *rng) {

  w += ref;

  if (sign > 0) {
    T a = (min_bound < (T)0.0) ? w / min_bound : (T)0.0;
    w -= scale_down * ((T)1.0 - a) * ((T)1.0 + dw_min_std * rng->sampleGauss());
  } else {
    T a = (max_bound > (T)0.0) ? w / max_bound : (T)0.0;
    w += scale_up * ((T)1.0 - a) * ((T)1.0 + dw_min_std * rng->sampleGauss());
  }
  w = MAX(w, min_bound);
  w = MIN(w, max_bound);

  w -= ref;

  if (write_noise_std > (T)0.0) {
    w_apparent = w + write_noise_std * rng->sampleGauss();
  } else {
    w_apparent = w;
  }
}

template <typename T>
inline void update_once_add(
    T &w,
    T &w_apparent,
    int &sign,
    T &scale_down,
    T &scale_up,
    T &ref,
    T &min_bound,
    T &max_bound,
    const T &dw_min_std,
    const T &write_noise_std,
    RNG<T> *rng) {

  w += ref;

  if (sign > 0) {
    T a = (min_bound < (T)0.0) ? w / min_bound : (T)0.0;
    w -= scale_down * ((T)1.0 - a + dw_min_std * rng->sampleGauss());
  } else {
    T a = (max_bound > (T)0.0) ? w / max_bound : (T)0.0;
    w += scale_up * ((T)1.0 - a + dw_min_std * rng->sampleGauss());
  }
  w = MAX(w, min_bound);
  w = MIN(w, max_bound);

  w -= ref;

  if (write_noise_std > (T)0.0) {
    w_apparent = w + write_noise_std * rng->sampleGauss();
  } else {
    w_apparent = w;
  }
}
} // namespace

template <typename T>
void SoftBoundsReferenceRPUDevice<T>::doSparseUpdate(
    T **weights, int i, const int *x_signed_indices, int x_count, int d_sign, RNG<T> *rng) {

  const auto &par = getPar();

  T *scale_down = this->w_scale_down_[i];
  T *scale_up = this->w_scale_up_[i];
  T *ref = w_reference_[i];
  T *w = par.usesPersistentWeight() ? this->w_persistent_[i] : weights[i];
  T *w_apparent = weights[i];
  T *min_bound = this->w_min_bound_[i];
  T *max_bound = this->w_max_bound_[i];
  T write_noise_std = par.getScaledWriteNoise();
  if (par.mult_noise) {
    PULSED_UPDATE_W_LOOP(update_once_mult(
                             w[j], w_apparent[j], sign, scale_down[j], scale_up[j], ref[j],
                             min_bound[j], max_bound[j], par.dw_min_std, write_noise_std, rng););
  } else {
    PULSED_UPDATE_W_LOOP(update_once_add(
                             w[j], w_apparent[j], sign, scale_down[j], scale_up[j], ref[j],
                             min_bound[j], max_bound[j], par.dw_min_std, write_noise_std, rng););
  }
}

template <typename T>
void SoftBoundsReferenceRPUDevice<T>::doDenseUpdate(T **weights, int *coincidences, RNG<T> *rng) {

  const auto &par = getPar();

  T *scale_down = this->w_scale_down_[0];
  T *scale_up = this->w_scale_up_[0];
  T *ref = w_reference_[0];
  T *w = par.usesPersistentWeight() ? this->w_persistent_[0] : weights[0];
  T *w_apparent = weights[0];
  T *min_bound = this->w_min_bound_[0];
  T *max_bound = this->w_max_bound_[0];
  T write_noise_std = par.getScaledWriteNoise();

  if (par.mult_noise) {
    PULSED_UPDATE_W_LOOP_DENSE(update_once_mult(
                                   w[j], w_apparent[j], sign, scale_down[j], scale_up[j], ref[j],
                                   min_bound[j], max_bound[j], par.dw_min_std, write_noise_std,
                                   rng););
  } else {
    PULSED_UPDATE_W_LOOP_DENSE(update_once_add(
                                   w[j], w_apparent[j], sign, scale_down[j], scale_up[j], ref[j],
                                   min_bound[j], max_bound[j], par.dw_min_std, write_noise_std,
                                   rng););
  }
}

template class SoftBoundsReferenceRPUDevice<float>;
#ifdef RPU_USE_DOUBLE
template class SoftBoundsReferenceRPUDevice<double>;
#endif
#ifdef RPU_USE_FP16
template class SoftBoundsReferenceRPUDevice<half_t>;
#endif

} // namespace RPU
