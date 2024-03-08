
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
    T *new_weights, const T *weights, const WeightModifierParameter<T> &wmpar) {

  if (wmpar.type != WeightModifierType::Copy) {
    if (wmpar.per_batch_sample) {
      RPU_FATAL("Per batch sample is not implemented in RPUCuda");
    }
  }

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

  T amax = (T)wmpar.assumed_wmax; // assumed max
  if (wmpar.rel_to_actual_wmax && wmpar.type != WeightModifierType::Copy) {
    amax = 0.0;
    PRAGMA_SIMD
    for (int i = 0; i < size_; i++) {
      if (wmpar.copy_last_column && (i % x_size_) == x_size_ - 1) {
        continue;
      }
      T a = (T)fabsf(new_weights[i]);
      amax = a > amax ? a : amax;
    }
    amax = amax > (T)0.0 ? amax : (T)1.0;
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

    const T res = (T)wmpar.res;
    if (res > (T)0.0) {
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

    if (wmpar.std_dev > (T)0.0) {
      const T std = (T)wmpar.std_dev * amax;
      PRAGMA_SIMD
      for (int i = 0; i < size_; i++) {
        T w = new_weights[i];
        new_weights[i] += w * std * rw_rng_.sampleGauss();
      }
    }
    break;
  }
  case WeightModifierType::AddNormal: {
    if (wmpar.std_dev > (T)0.0) {

      const T std = (T)wmpar.std_dev * amax;
      PRAGMA_SIMD
      for (int i = 0; i < size_; i++) {
        new_weights[i] += std * rw_rng_.sampleGauss();
      }
    }

    break;
  }

  case WeightModifierType::Poly: {

    if (wmpar.std_dev > (T)0.0 && wmpar.coeffs.size() > (size_t)0) {
      const T std = wmpar.std_dev;
      PRAGMA_SIMD
      for (int i = 0; i < size_; i++) {
        T aw = (T)fabsf(new_weights[i]) / amax;
        T paw = 1;
        T sig = wmpar.coeffs.at(0);
        for (size_t j = 1; j < wmpar.coeffs.size(); j++) {
          paw *= aw;
          sig += wmpar.coeffs.at(j) * paw;
        }
        sig *= std;
        new_weights[i] += amax * sig * rw_rng_.sampleGauss();
      }
    }
    break;
  }

  case WeightModifierType::ProgNoise: {

    if (wmpar.std_dev > (T)0.0 && wmpar.coeffs.size() > (size_t)0) {
      const T std = wmpar.std_dev / wmpar.g_max;
      PRAGMA_SIMD
      for (int i = 0; i < size_; i++) {
        T aw = (T)fabsf(new_weights[i]) / amax;
        T paw = 1;
        T sig = wmpar.coeffs.at(0);
        for (size_t j = 1; j < wmpar.coeffs.size(); j++) {
          paw *= aw;
          sig += wmpar.coeffs.at(j) * paw;
        }
        sig *= std;
        T w = new_weights[i];
        if (w < (T)0.0) {
          new_weights[i] = -(T)fabsf(w + amax * sig * rw_rng_.sampleGauss());
        } else {
          new_weights[i] = (T)fabsf(w + amax * sig * rw_rng_.sampleGauss());
        }
      }
    }
    break;
  }

  case WeightModifierType::DoReFa: {

    const T res = (T)wmpar.res;

    if (res > (T)0.0) {
      const bool sto_round = wmpar.sto_round;
      const T scale = (T)(T)fabsf(wmpar.dorefa_clip / (T)tanh((float)amax));

      PRAGMA_SIMD
      for (int i = 0; i < size_; i++) {
        T w = (T)tanh((float)new_weights[i]) * scale;
        new_weights[i] = getDiscretizedValueRound(w, res, sto_round, rw_rng_);
      }
    }

    break;
  }

  case WeightModifierType::DiscretizeAddNormal: {

    const T res = (T)wmpar.res;
    const T std = (T)wmpar.std_dev * amax;

    if (res > (T)0.0 || std > (T)0.0) {
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

  case WeightModifierType::DropConnect: {
    // will be done below
    break;
  }

  default:
    RPU_FATAL("Requested WeightModifierType not implemented in CPU version.");
  }

  if (wmpar.pdrop > (T)0.0) {
    dropConnections(new_weights, (T)wmpar.pdrop);
  }

  if (wmpar.copy_last_column) {
    for (int j = 0; j < d_size_; j++) {
      new_weights[(j + 1) * x_size_ - 1] = saved_bias_[j];
    }
  }
}

template <typename T>
void WeightModifier<T>::dumpExtra(RPU::state_t &extra, const std::string prefix) {

  RPU::state_t state;

  RPU::insert(state, "saved_bias", saved_bias_);
  RPU::insert(state, "enable_during_test", enable_during_test_);

  RPU::insertWithPrefix(extra, state, prefix);
}

template <typename T>
void WeightModifier<T>::loadExtra(
    const RPU::state_t &extra, const std::string prefix, bool strict) {

  auto state = RPU::selectWithPrefix(extra, prefix);

  RPU::load(state, "saved_bias", saved_bias_, strict);
  RPU::load(state, "enable_during_test", enable_during_test_, strict);
}

template class WeightModifier<float>;
#ifdef RPU_USE_DOUBLE
template class WeightModifier<double>;
#endif
#ifdef RPU_USE_FP16
template class WeightModifier<half_t>;
#endif

} // namespace RPU
