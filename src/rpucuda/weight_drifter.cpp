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

#include "weight_drifter.h"
#include "math_util.h"
#include "utility_functions.h"

namespace RPU {

/***********************************************************/
// ctors

template <typename T>
WeightDrifter<T>::WeightDrifter(int size) : size_(size), active_(false), current_t_(0.0) {}

template <typename T>
WeightDrifter<T>::WeightDrifter(int size, const DriftParameter<T> &par) : WeightDrifter(size) {
  par_ = par;
  par_.setSimpleDrift();
}

template <typename T>
WeightDrifter<T>::WeightDrifter(int size, const DriftParameter<T> &par, RealWorldRNG<T> *rng)
    : WeightDrifter(size) {
  populate(par, rng);
}
/***********************************************************/
template <typename T> void WeightDrifter<T>::setNu(const T *src) {
  nu_.resize(size_);
  RPU::math::copy<T>(size_, src, 1, nu_.data(), 1);
}
template <typename T> void WeightDrifter<T>::getNu(T *dst) const {
  RPU::math::copy<T>(size_, nu_.data(), 1, dst, 1);
}

/***********************************************************/
template <typename T> void WeightDrifter<T>::initialize(const T *weights) {

  t_.clear();
  t_.resize(size_);

  w0_.resize(size_);
  RPU::math::copy<T>(size_, weights, 1, w0_.data(), 1);

  previous_weights_.resize(size_);
  RPU::math::copy<T>(size_, weights, 1, previous_weights_.data(), 1);

  active_ = true;

  current_t_ = (T)0.0;
}

template <typename T>
void WeightDrifter<T>::populate(const DriftParameter<T> &par, RealWorldRNG<T> *rng) {
  par_ = par;
  if (!par_.isSimpleDrift()) {
    nu_.resize(size_);
    for (int i = 0; i < size_; i++) {
      nu_[i] = par_.nu + par_.nu_dtod * par_.nu * rng->sampleGauss();
    }
  }
}

template <typename T>
void WeightDrifter<T>::dumpExtra(RPU::state_t &extra, const std::string prefix) {

  RPU::state_t state;

  RPU::insert(state, "active", active_);
  RPU::insert(state, "current_t", current_t_);
  RPU::insert(state, "previous_weights", previous_weights_);
  RPU::insert(state, "w0", w0_);
  RPU::insert(state, "t", t_);
  RPU::insert(state, "nu", nu_);

  RPU::insertWithPrefix(extra, state, prefix);
}

template <typename T>
void WeightDrifter<T>::loadExtra(const RPU::state_t &extra, const std::string prefix, bool strict) {

  auto state = RPU::selectWithPrefix(extra, prefix);

  RPU::load(state, "previous_weights", previous_weights_, strict);
  RPU::load(state, "w0", w0_, strict);
  RPU::load(state, "t", t_, strict);
  RPU::load(state, "nu", nu_, strict);
  RPU::load(state, "current_t", current_t_, strict);
  RPU::load(state, "active", active_, strict);
}

template <typename T>
void WeightDrifter<T>::saturate(T *weights, const T *min_bounds, const T *max_bounds) {

  PRAGMA_SIMD
  for (int i = 0; i < size_; i++) {
    weights[i] = MIN(MAX(min_bounds[i], weights[i]), max_bounds[i]);
    previous_weights_[i] = MIN(MAX(min_bounds[i], previous_weights_[i]), max_bounds[i]);
  }
}

template <typename T>
void WeightDrifter<T>::apply(T *weights, T time_since_last_call, RNG<T> &rng) {

  if (previous_weights_.size() != (size_t)size_) {
    initialize(weights);
  }
  if (!par_.isSimpleDrift() && nu_.empty()) {
    RPU_FATAL("Weight drifter needs to be populated first!");
  }

  current_t_ += time_since_last_call / par_.t0;
  T reset_tol = par_.reset_tol;
  bool simple = par_.isSimpleDrift();
  T a = par_.g_offset * par_.wg_ratio + par_.w_offset;
  if ((T)fabsf(a) < reset_tol) {
    a = (T)0.0;
  }
  T w_noise_std = par_.w_read_std;
  T nu_std = par_.nu_std;
  T nu0 = par_.nu;

  PRAGMA_SIMD
  for (int i = 0; i < size_; i++) {
    T w = weights[i];
    if ((T)fabsf(previous_weights_[i] - w) > reset_tol) {
      // weight has changed and thus need a drift reset
      t_[i] = current_t_;
      w0_[i] = w;

    } else {
      // this will overwrite the current weight ! make sure that no DECAY/WNOISE is present

      T w0 = w0_[i];
      T nu = simple ? nu0 : nu_[i];
      nu = (nu_std <= (T)0.0) ? nu : nu + nu_std * nu * rng.sampleGauss();
      nu = (!par_.nu_k)
               ? nu
               : nu - par_.nu_k * (T)logf((w - par_.w_offset) / par_.wg_ratio + par_.g_offset) +
                     par_.nu_k * par_.logG0;
      T delta_t = MAX(current_t_ - t_[i], (T)1.0); // at least t0
      T nu_scale = (T)powf(delta_t, -nu);
      w = w0 * nu_scale; // overwrites w
      w = (!a) ? w : w + a * (nu_scale - (T)1.0);
    }
    w += w_noise_std > (T)0.0 ? w_noise_std * rng.sampleGauss() : (T)0.0;
    previous_weights_[i] = w;
    weights[i] = w;
  }
}

template class WeightDrifter<float>;
#ifdef RPU_USE_DOUBLE
template class WeightDrifter<double>;
#endif
#ifdef RPU_USE_FP16
template class WeightDrifter<half_t>;
#endif

} // namespace RPU
