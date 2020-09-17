/**
 * (C) Copyright 2020 IBM. All Rights Reserved.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
 */

#include "rpu_simple_device.h"
#include "math_util.h"
#include "utility_functions.h"

namespace RPU {

/******************************************************************************************/
/* SimpleRPUDevice*/

template <typename T> void SimpleRPUDevice<T>::initialize(int x_size, int d_size) {
  x_size_ = x_size;
  d_size_ = d_size;
  size_ = x_size * d_size;
  par_storage_ = nullptr;
}

// ctor
template <typename T> SimpleRPUDevice<T>::SimpleRPUDevice(int x_sz, int d_sz) {
  initialize(x_sz, d_sz);
}

template <typename T>
SimpleRPUDevice<T>::SimpleRPUDevice(
    int x_sz, int d_sz, const SimpleRPUDeviceMetaParameter<T> &par, RealWorldRNG<T> *rng)
    : SimpleRPUDevice<T>(x_sz, d_sz) {
  populate(par, rng);
}

// copy construcutor
template <typename T> SimpleRPUDevice<T>::SimpleRPUDevice(const SimpleRPUDevice<T> &other) {
  initialize(other.x_size_, other.d_size_);
  if (other.par_storage_ != nullptr) {
    par_storage_ = other.par_storage_->cloneUnique();
  }
}

// copy assignment
template <typename T>
SimpleRPUDevice<T> &SimpleRPUDevice<T>::operator=(const SimpleRPUDevice<T> &other) {

  SimpleRPUDevice<T> tmp(other);
  swap(*this, tmp);
  return *this;
};

// move constructor
template <typename T> SimpleRPUDevice<T>::SimpleRPUDevice(SimpleRPUDevice<T> &&other) {
  *this = std::move(other);
}

// move assignment
template <typename T>
SimpleRPUDevice<T> &SimpleRPUDevice<T>::operator=(SimpleRPUDevice<T> &&other) {

  initialize(other.x_size_, other.d_size_);
  par_storage_ = std::move(other.par_storage_);
  return *this;
}

/********************************************************************************/
/* compute functions  */

template <typename T> void SimpleRPUDevice<T>::decayWeights(T **weights, bool bias_no_decay) {
  decayWeights(weights, (T)1.0, bias_no_decay);
}

template <typename T>
void SimpleRPUDevice<T>::decayWeights(T **weights, T alpha, bool bias_no_decay) {
  T lifetime = getPar().lifetime;
  T decay_rate = (lifetime > 1) ? ((T)1.0 / lifetime) : (T)0.0;
  T decay_scale = (T)1.0 - alpha * decay_rate;

  if (decay_scale > 0 && decay_scale < 1.0) {
    if (!bias_no_decay) {
      RPU::math::scal<T>(this->size_, decay_scale, weights[0], 1);
    } else {
      int size = this->d_size_ * this->x_size_;
      T *w = weights[0];
      const int last_col = this->x_size_ - 1; // x-major (ie row major)
      PRAGMA_SIMD
      for (int i = 0; i < size; ++i) {
        w[i] *= (i % this->x_size_ == last_col) ? (T)1.0 : decay_scale;
      }
    }
  }
}

template <typename T> void SimpleRPUDevice<T>::diffuseWeights(T **weights, RNG<T> &rng) {

  T diffusion = getPar().diffusion;
  if (diffusion > 0.0) {
    T *w = weights[0];
    PRAGMA_SIMD
    for (int i = 0; i < this->size_; ++i) {
      w[i] += diffusion * rng.sampleGauss();
    }
  }
}

template <typename T> void SimpleRPUDevice<T>::clipWeights(T **weights, T clip) {
  // apply hard bounds

  if (clip >= 0) {
    T *w = weights[0];
    PRAGMA_SIMD
    for (int i = 0; i < this->size_; ++i) {
      w[i] = MIN(MAX(w[i], -clip), clip);
    }
  }
}

template <typename T>
void SimpleRPUDevice<T>::populate(const SimpleRPUDeviceMetaParameter<T> &p, RealWorldRNG<T> *rng) {

  par_storage_ = p.cloneUnique();
  par_storage_->initialize();
}

template class SimpleRPUDeviceMetaParameter<float>;
template class AbstractRPUDevice<float>;
template class SimpleRPUDevice<float>;
#ifdef RPU_USE_DOUBLE
template class AbstractRPUDevice<double>;
template class SimpleRPUDevice<double>;
template class SimpleRPUDeviceMetaParameter<double>;
#endif

} // namespace RPU
