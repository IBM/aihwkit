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

#include "rpu_linearstep_device.h"
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
void LinearStepRPUDevice<T>::populate(
    const LinearStepRPUDeviceMetaParameter<T> &p, RealWorldRNG<T> *rng) {

  PulsedRPUDevice<T>::populate(p, rng); // will clone par
  auto &par = getPar();

  if ((par.ls_reverse_up || par.ls_reverse_down) && !par.ls_mult_noise) {
    RPU_FATAL("Only mulitplicative noise supported with reverse up/down!");
  }

  for (int i = 0; i < this->d_size_; ++i) {

    for (int j = 0; j < this->x_size_; ++j) {

      T diff_slope_at_bound_up = par.ls_decrease_up + par.ls_decrease_up_dtod * rng->sampleGauss();
      T diff_slope_at_bound_down =
          par.ls_decrease_down + par.ls_decrease_down_dtod * rng->sampleGauss();

      if (!par.ls_allow_increasing_slope) {
        /* we force the number to be positive when requested [RRAM]*/
        diff_slope_at_bound_up = (T)fabsf(diff_slope_at_bound_up);
        diff_slope_at_bound_down = (T)fabsf(diff_slope_at_bound_down);
      }
      T w_ref_up, w_ref_down;

      if (par.ls_mean_bound_reference) {
        /* divide by mean bound, otherwise slope depends on device
           bound, which does not make sense both slopes are negative
           (sign of scale_up/scale_down here both positive and later
           corrected in update rule) */

        w_ref_up = par.ls_reverse_up ? par.w_min - par.ls_reverse_offset : par.w_max;
        w_ref_down = par.ls_reverse_down ? par.w_max + par.ls_reverse_offset : par.w_min;
      } else {
        w_ref_up = par.ls_reverse_up ? this->w_min_bound_[i][j] - par.ls_reverse_offset
                                     : this->w_max_bound_[i][j];
        w_ref_down = par.ls_reverse_down ? this->w_max_bound_[i][j] + par.ls_reverse_offset
                                         : this->w_min_bound_[i][j];
      }

      // slope should be correct because of sign of reference
      if (w_ref_up != (T)0.0) {
        w_slope_up_[i][j] = -diff_slope_at_bound_up * this->w_scale_up_[i][j] / w_ref_up;
      } else {
        w_slope_up_[i][j] = (T)0.0;
      }
      if (w_ref_down != (T)0.0) {
        w_slope_down_[i][j] = -diff_slope_at_bound_down * this->w_scale_down_[i][j] / w_ref_down;
      } else {
        w_slope_down_[i][j] = (T)0.0;
      }
    }
  }
}

template <typename T> void LinearStepRPUDevice<T>::printDP(int x_count, int d_count) const {

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
      std::cout << w_slope_up_[i][j] << ",";
      std::cout << w_slope_down_[i][j] << ">]";
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
    T &slope_down,
    T &slope_up,
    T &min_bound,
    T &max_bound,
    const T &dw_min_std,
    const T &write_noise_std,
    RNG<T> *rng) {
  if (sign > 0) {
    w -= (slope_down * w + scale_down) * ((T)1.0 + dw_min_std * rng->sampleGauss());
  } else {
    w += (slope_up * w + scale_up) * ((T)1.0 + dw_min_std * rng->sampleGauss());
  }
  w = MAX(w, min_bound);
  w = MIN(w, max_bound);

  if (write_noise_std > (T)0.0) {
    w_apparent = w + write_noise_std * rng->sampleGauss();
  }
}

template <typename T>
inline void update_once_add(
    T &w,
    T &w_apparent,
    int &sign,
    T &scale_down,
    T &scale_up,
    T &slope_down,
    T &slope_up,
    T &min_bound,
    T &max_bound,
    const T &dw_min_std,
    const T &write_noise_std,
    RNG<T> *rng) {
  if (sign > 0) {
    w -= slope_down * w + scale_down * ((T)1.0 + dw_min_std * rng->sampleGauss());
  } else {
    w += slope_up * w + scale_up * ((T)1.0 + dw_min_std * rng->sampleGauss());
  }
  w = MAX(w, min_bound);
  w = MIN(w, max_bound);

  if (write_noise_std > (T)0.0) {
    w_apparent = w + write_noise_std * rng->sampleGauss();
  }
}
} // namespace

template <typename T>
void LinearStepRPUDevice<T>::doSparseUpdate(
    T **weights, int i, const int *x_signed_indices, int x_count, int d_sign, RNG<T> *rng) {

  const auto &par = getPar();

  T *scale_down = this->w_scale_down_[i];
  T *scale_up = this->w_scale_up_[i];
  T *slope_down = w_slope_down_[i];
  T *slope_up = w_slope_up_[i];
  T *w = par.usesPersistentWeight() ? this->w_persistent_[i] : weights[i];
  T *w_apparent = weights[i];
  T *min_bound = this->w_min_bound_[i];
  T *max_bound = this->w_max_bound_[i];

  T write_noise_std = par.getScaledWriteNoise();
  if (par.ls_mult_noise) {
    PULSED_UPDATE_W_LOOP(update_once_mult(
                             w[j], w_apparent[j], sign, scale_down[j], scale_up[j], slope_down[j],
                             slope_up[j], min_bound[j], max_bound[j], par.dw_min_std,
                             write_noise_std, rng););
  } else {
    PULSED_UPDATE_W_LOOP(update_once_add(
                             w[j], w_apparent[j], sign, scale_down[j], scale_up[j], slope_down[j],
                             slope_up[j], min_bound[j], max_bound[j], par.dw_min_std,
                             write_noise_std, rng););
  }
}

template <typename T>
void LinearStepRPUDevice<T>::doDenseUpdate(T **weights, int *coincidences, RNG<T> *rng) {

  const auto &par = getPar();

  T *scale_down = this->w_scale_down_[0];
  T *scale_up = this->w_scale_up_[0];
  T *slope_down = w_slope_down_[0];
  T *slope_up = w_slope_up_[0];
  T *w = par.usesPersistentWeight() ? this->w_persistent_[0] : weights[0];
  T *w_apparent = weights[0];
  T *min_bound = this->w_min_bound_[0];
  T *max_bound = this->w_max_bound_[0];
  T write_noise_std = par.getScaledWriteNoise();

  if (par.ls_mult_noise) {
    PULSED_UPDATE_W_LOOP_DENSE(update_once_mult(
                                   w[j], w_apparent[j], sign, scale_down[j], scale_up[j],
                                   slope_down[j], slope_up[j], min_bound[j], max_bound[j],
                                   par.dw_min_std, write_noise_std, rng););
  } else {
    PULSED_UPDATE_W_LOOP_DENSE(update_once_add(
                                   w[j], w_apparent[j], sign, scale_down[j], scale_up[j],
                                   slope_down[j], slope_up[j], min_bound[j], max_bound[j],
                                   par.dw_min_std, write_noise_std, rng););
  }
}

template class LinearStepRPUDevice<float>;
#ifdef RPU_USE_DOUBLE
template class LinearStepRPUDevice<double>;
#endif
#ifdef RPU_USE_FP16
template class LinearStepRPUDevice<half_t>;
#endif

} // namespace RPU
