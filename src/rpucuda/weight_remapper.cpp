/**
 * (C) Copyright 2020, 2021, 2022 IBM. All Rights Reserved.
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

  UNUSED(biases);
  UNUSED(current_lr);

  switch (wrmpar.type) {
  case WeightRemapType::LayerwiseSymmetric: {

    if (!scales) {
      RPU_FATAL("Expect scales given.");
    }
    T amax = 0.0;
    PRAGMA_SIMD
    for (int i = 0; i < size_; i++) {
      T a = fabs(weights[i]);
      amax = a > amax ? a : amax;
    }
    amax /= (T)wrmpar.remapped_wmax;

    if (amax > 0) {

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
      T a = fabs(weights[i]);
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
        scales[i] *= amax > 0 ? amax : (T)1.0;
      }

      PRAGMA_SIMD
      for (int i = 0; i < size_; i++) {
        int row_idx = i / x_size_;
        T amax = max_values_[row_idx];
        weights[i] /= amax > 0 ? amax : (T)1.0;
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
      if (min_scale > 0 && wrmpar.max_scale_ref) {
        min_scale = MAX(min_scale, (T)wrmpar.max_scale_ref);
      }

      PRAGMA_SIMD
      for (int i = 0; i < d_size_; i++) {
        T amax = max_values_[i];
        T new_scale = amax > 0 ? amax * old_scales_[i] : old_scales_[i];

        if (min_scale > 0) {
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
  case WeightRemapType::None: {
    break;
  }
  default:
    RPU_FATAL("Remapping type not implemented for CPU.");
  } // switch
}

template class WeightRemapper<float>;
#ifdef RPU_USE_DOUBLE
template class WeightRemapper<double>;
#endif

} // namespace RPU
