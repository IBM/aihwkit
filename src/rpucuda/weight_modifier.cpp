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

#include "weight_modifier.h"
#include "math_util.h"
#include "utility_functions.h"

namespace RPU {

/***********************************************************/
// ctors

template <typename T>
WeightModifier<T>::WeightModifier(int x_size, int d_size)
    : x_size_(x_size), d_size_(d_size), size_(d_size * x_size) {}

template <typename T> void WeightModifier<T>::dropConnections(T *weights, T prob) {

  PRAGMA_SIMD
  for (int i = 0; i < size_; i++) {
    if (rw_rng_.sampleUniform() < prob) {
      weights[i] = 0.0;
    }
  }
}

template <typename T>
void WeightModifier<T>::apply(
    T *new_weights, const T *weights, const WeightModifierParameter &wmpar) {

  // just copy always if not in-place [also handles WeightModifierType::Copy]
  if (new_weights != weights) {
    RPU::math::copy<T>(size_, weights, 1, new_weights, 1);
  }

  if (wmpar.copy_last_column) {
    saved_bias_.resize(d_size_);
    for (int j = 0; j < d_size_; j++) {
      saved_bias_[j] = weights[(j + 1) * x_size_ - 1];
    }
  }
  enable_during_test_ = wmpar.enable_during_test;

  T amax = wmpar.assumed_wmax; // assumed max
  if (wmpar.rel_to_actual_wmax && wmpar.type != WeightModifierType::Copy) {
    amax = 0.0;
    PRAGMA_SIMD
    for (int i = 0; i < size_; i++) {
      if (wmpar.copy_last_column && (i % x_size_) == x_size_ - 1) {
        continue;
      }
      T a = fabs(new_weights[i]);
      amax = a > amax ? a : amax;
    }
    amax = amax > 0.0 ? amax : (T)1.0;
  }

  switch (wmpar.type) {
  case WeightModifierType::Copy: {
    // do not allow in place
    if (new_weights == weights) {
      RPU_FATAL("Copy is not possible for in-place weights! ");
    }
    break; // maybe dropping below
  }

  case WeightModifierType::Discretize: {

    const T res = wmpar.res;
    if (res > 0) {
      const bool sto_round = wmpar.sto_round;
      PRAGMA_SIMD
      for (int i = 0; i < size_; i++) {
        T w = new_weights[i];
        new_weights[i] = amax * getDiscretizedValueRound(w / amax, res, sto_round, rw_rng_);
      }
    }
    break;
  }
  case WeightModifierType::MultNormal: {

    if (wmpar.std_dev > 0) {
      const T std = wmpar.std_dev * amax;
      PRAGMA_SIMD
      for (int i = 0; i < size_; i++) {
        T w = new_weights[i];
        new_weights[i] += w * std * rw_rng_.sampleGauss();
      }
    }
    break;
  }
  case WeightModifierType::AddNormal: {

    if (wmpar.std_dev > 0) {
      const T std = wmpar.std_dev * amax;
      PRAGMA_SIMD
      for (int i = 0; i < size_; i++) {
        new_weights[i] += std * rw_rng_.sampleGauss();
      }
    }

    break;
  }

  case WeightModifierType::Poly: {

    if (wmpar.std_dev > 0) {
      const T std = wmpar.std_dev;
      const T p0 = wmpar.coeff0;
      const T p1 = wmpar.coeff1;
      const T p2 = wmpar.coeff2;

      PRAGMA_SIMD
      for (int i = 0; i < size_; i++) {
        T aw = fabs(new_weights[i]) / amax;
        T sig = std * (p0 + p1 * aw + p2 * aw * aw);
        new_weights[i] += amax * sig * rw_rng_.sampleGauss();
      }
    }

    break;
  }

  case WeightModifierType::DoReFa: {

    const T res = wmpar.res;

    if (res > 0) {
      const bool sto_round = wmpar.sto_round;
      const T scale = fabs(wmpar.dorefa_clip / tanh(amax));

      PRAGMA_SIMD
      for (int i = 0; i < size_; i++) {
        T w = tanh(new_weights[i]) * scale;
        new_weights[i] = getDiscretizedValueRound(w, res, sto_round, rw_rng_);
      }
    }

    break;
  }

  case WeightModifierType::DiscretizeAddNormal: {

    const T res = wmpar.res;
    const T std = wmpar.std_dev * amax;

    if (res > 0 || std > 0) {
      const bool sto_round = wmpar.sto_round;

      PRAGMA_SIMD
      for (int i = 0; i < size_; i++) {
        T w = new_weights[i];
        w = amax * getDiscretizedValueRound(w / amax, res, sto_round, rw_rng_);
        new_weights[i] = w + std * rw_rng_.sampleGauss();
      }
    }

    break;
  }

  default:
    RPU_FATAL("Requested WeightModifierType not implemented in CPU version.");
  }

  if (wmpar.pdrop > 0.0) {
    dropConnections(new_weights, wmpar.pdrop);
  }

  if (wmpar.copy_last_column) {
    for (int j = 0; j < d_size_; j++) {
      new_weights[(j + 1) * x_size_ - 1] = saved_bias_[j];
    }
  }
}

template class WeightModifier<float>;
#ifdef RPU_USE_DOUBLE
template class WeightModifier<double>;
#endif

} // namespace RPU
