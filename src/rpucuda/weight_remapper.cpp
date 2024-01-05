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

#include "weight_remapper.h"
#include "math_util.h"
#include "utility_functions.h"
#include <limits>

namespace RPU {

/***********************************************************/
// ctors

template <typename T>
WeightRemapper<T>::WeightRemapper(int x_size, int d_size)
    : x_size_(x_size), d_size_(d_size), size_(d_size * x_size) {}

template <typename T>
void WeightRemapper<T>::apply(
    T *weights, T current_lr, const WeightRemapParameter &wrmpar, T *scales, T *biases) {

  switch (wrmpar.type) {
  case WeightRemapType::LayerwiseSymmetric: {

    if (!scales) {
      RPU_FATAL("Expect scales given.");
    }
    T amax = 0.0;
    PRAGMA_SIMD
    for (int i = 0; i < size_; i++) {
      T a = (T)fabsf(weights[i]);
      amax = a > amax ? a : amax;
    }
    amax /= (T)wrmpar.remapped_wmax;

    if (amax > (T)0.0) {

      PRAGMA_SIMD
      for (int i = 0; i < d_size_; i++) {
        scales[i] *= amax;
      }

      PRAGMA_SIMD
      for (int i = 0; i < size_; i++) {
        weights[i] /= amax;
      }
    }

    break;
  }
  case WeightRemapType::ChannelwiseSymmetric: {

    if (!scales) {
      RPU_FATAL("Expect scales given.");
    }

    if (max_values_.size() < (size_t)d_size_) {
      max_values_.resize(d_size_);
    }
    std::fill(max_values_.begin(), max_values_.end(), 0.0);

    // compute max per row
    PRAGMA_SIMD
    for (int i = 0; i < size_; i++) {
      int row_idx = i / x_size_;
      T a = (T)fabsf(weights[i]);
      T mv = max_values_[row_idx];

      max_values_[row_idx] = a > mv ? a : mv;
    }

    if (wrmpar.max_scale_range <= 0) {
      if (wrmpar.remapped_wmax != 1.0) {
        PRAGMA_SIMD
        for (int i = 0; i < d_size_; i++) {
          max_values_[i] /= (T)wrmpar.remapped_wmax;
        }
      }

      PRAGMA_SIMD
      for (int i = 0; i < d_size_; i++) {
        T amax = max_values_[i];
        scales[i] *= amax > (T)0.0 ? amax : (T)1.0;
      }

      PRAGMA_SIMD
      for (int i = 0; i < size_; i++) {
        int row_idx = i / x_size_;
        T amax = max_values_[row_idx];
        weights[i] /= amax > (T)0.0 ? amax : (T)1.0;
      }
    } else {
      // with bounded scale range
      if (wrmpar.remapped_wmax != 1.0) {
        RPU_FATAL("For max_scales set, expect wrmpar.remapped_wmax to be 1.");
      }

      if (old_scales_.size() < (size_t)d_size_) {
        old_scales_.resize(d_size_);
      }

      T min_scale = std::numeric_limits<T>::max();
      PRAGMA_SIMD
      for (int i = 0; i < d_size_; i++) {
        T s = scales[i];
        old_scales_[i] = s;
        min_scale = min_scale > s ? s : min_scale;
      }
      min_scale = MAX(min_scale, (T)0.0);
      if (min_scale > (T)0.0 && wrmpar.max_scale_ref) {
        min_scale = MAX(min_scale, (T)wrmpar.max_scale_ref);
      }

      PRAGMA_SIMD
      for (int i = 0; i < d_size_; i++) {
        T amax = max_values_[i];
        T new_scale = amax > (T)0.0 ? amax * old_scales_[i] : old_scales_[i];

        if (min_scale > (T)0.0) {
          new_scale = new_scale / min_scale > (T)wrmpar.max_scale_range
                          ? min_scale * (T)wrmpar.max_scale_range
                          : new_scale;
        }
        scales[i] = new_scale;
      }

      PRAGMA_SIMD
      for (int i = 0; i < size_; i++) {
        int row_idx = i / x_size_;
        weights[i] = weights[i] * old_scales_[row_idx] / scales[row_idx];
      }
    }

    break;
  }
  case WeightRemapType::ChannelwiseNorm: {

    if (!scales) {
      RPU_FATAL("Expect scales given.");
    }
    if (wrmpar.row_norm <= 0) {
      RPU_FATAL("Expect row_norm >0.");
    }

    if (norm_values_.size() < (size_t)d_size_) {
      norm_values_.resize(d_size_);
    }
    std::fill(norm_values_.begin(), norm_values_.end(), 0.0);

    // compute norm per row
    PRAGMA_SIMD
    for (int i = 0; i < size_; i++) {
      int row_idx = i / x_size_;
      T w = weights[i];
      norm_values_[row_idx] += w * w;
    }

    PRAGMA_SIMD
    for (int i = 0; i < d_size_; i++) {
      T nm = (T)sqrtf(norm_values_[i]);
      nm = nm > (T)0.0 ? nm : (T)1.0;
      nm /= (T)wrmpar.row_norm;
      nm = nm > (T)1.0 ? nm : (T)1.0;
      norm_values_[i] = nm; // ratio
      if (!wrmpar.clip_if) {
        scales[i] *= nm;
      }
    }

    PRAGMA_SIMD
    for (int i = 0; i < size_; i++) {
      int row_idx = i / x_size_;
      weights[i] /= norm_values_[row_idx];
    }

    break;
  }

  case WeightRemapType::ChannelwiseAsymmetric: {

    if (!scales || !biases) {
      RPU_FATAL("Expect scales and biases given.");
    }
    if (max_values_.size() < (size_t)d_size_) {
      max_values_.resize(d_size_);
    }
    std::fill(max_values_.begin(), max_values_.end(), std::numeric_limits<T>::lowest());
    std::fill(min_values_.begin(), min_values_.end(), std::numeric_limits<T>::max());

    PRAGMA_SIMD
    for (int i = 0; i < size_; i++) {
      int row_idx = i / x_size_; // x-size major
      T v = weights[i];

      T mxv = max_values_[row_idx];
      max_values_[row_idx] = v > mxv ? v : mxv;

      T mnv = min_values_[row_idx];
      min_values_[row_idx] = v < mnv ? v : mnv;
    }

    for (int row_idx = 0; row_idx < d_size_; row_idx++) {
      T mx = max_values_[row_idx];
      T mn = min_values_[row_idx];
      T b = biases[row_idx];
      T s = scales[row_idx];

      T half_span = (mx - mn) / (T)2.0;
      half_span = half_span > (T)0.0 ? half_span : (T)1.0;

      T new_scale = s * half_span;
      T new_b = b - new_scale - s * mn;

      PRAGMA_SIMD
      for (int j = 0; j < x_size_; j++) {
        weights[j + row_idx * x_size_] =
            weights[j + row_idx * x_size_] / half_span + (T)1.0 + mn / half_span;
      }

      scales[row_idx] = new_scale;
      biases[row_idx] = new_b;
    }

    break;
  }
  case WeightRemapType::None: {
    break;
  }
  default:
    RPU_FATAL("Remapping type not implemented for CPU.");
  } // switch
}

template <typename T>
bool WeightRemapper<T>::applySWA(
    T *swa_weights,
    T *weights,
    uint64_t iter,
    const WeightRemapParameter &wrmpar,
    T current_lr,
    T *scales,
    T *biases) {

  // stochastic weight averaging
  if (wrmpar.swa_every <= 0) {
    return false;
  }

  if ((iter <= wrmpar.swa_start) || (iter % wrmpar.swa_every != 0)) {
    return false;
  }

  uint64_t n = (iter - wrmpar.swa_start) / wrmpar.swa_every - 1;
  T ratio = (T)(n) / (T)(n + 1);
  if (scales != nullptr) {
    PRAGMA_SIMD
    for (int i = 0; i < size_; i++) {
      int row_idx = i / x_size_; // x-size major
      T w = weights[i];
      w *= scales[row_idx];
      swa_weights[i] = swa_weights[i] * ratio + w / (T)(n + 1);
    }
  } else {
    PRAGMA_SIMD
    for (int i = 0; i < size_; i++) {
      swa_weights[i] = swa_weights[i] * ratio + weights[0] / (T)(n + 1);
    }
  }

  if ((wrmpar.swa_transfer_every > 0) && ((n + 1) % wrmpar.swa_transfer_every == 0)) {
    std::cout << "SWA: do transfer [" << iter << "]\n";
    math::copy(size_, swa_weights, 1, weights, 1);
    if (scales) {
      PRAGMA_SIMD
      for (int i = 0; i < d_size_; i++) {
        scales[i] = (T)1.0;
      }
      // need to re-map the weights:
      this->apply(weights, current_lr, wrmpar, scales, biases);
    }
    return true; // modified the original weights
  }
  return false;
}

template class WeightRemapper<float>;
#ifdef RPU_USE_DOUBLE
template class WeightRemapper<double>;
#endif
#ifdef RPU_USE_FP16
template class WeightRemapper<half_t>;
#endif

} // namespace RPU
