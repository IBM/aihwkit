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

  if (other.wdrifter_) {
    wdrifter_ = RPU::make_unique<WeightDrifter<T>>(*other.wdrifter_);
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
  wdrifter_ = std::move(other.wdrifter_);
  return *this;
}

template <typename T>
void SimpleRPUDevice<T>::dumpExtra(RPU::state_t &extra, const std::string prefix) {
  RPU::state_t state;
  if (hasWDrifter()) {
    wdrifter_->dumpExtra(state, "wdrifter");
    RPU::insertWithPrefix(extra, state, prefix);
  }
}

template <typename T>
void SimpleRPUDevice<T>::loadExtra(
    const RPU::state_t &extra, const std::string prefix, bool strict) {
  if (hasWDrifter()) {
    auto state = RPU::selectWithPrefix(extra, prefix);
    wdrifter_->loadExtra(state, "wdrifter", strict);
  }
};

/********************************************************************************/
/* compute functions  */

template <typename T> void SimpleRPUDevice<T>::decayWeights(T **weights, bool bias_no_decay) {
  decayWeights(weights, (T)1.0, bias_no_decay);
}

template <typename T>
void SimpleRPUDevice<T>::decayWeights(T **weights, T alpha, bool bias_no_decay) {
  T lifetime = getPar().lifetime;
  T decay_rate = (lifetime > (T)1.0) ? ((T)1.0 / lifetime) : (T)0.0;
  T decay_scale = (T)1.0 - alpha * decay_rate;

  if (decay_scale > (T)0.0 && decay_scale < (T)1.0) {
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

template <typename T>
void SimpleRPUDevice<T>::driftWeights(T **weights, T time_since_last_call, RNG<T> &rng) {
  if (hasWDrifter()) {
    wdrifter_->apply(weights[0], time_since_last_call, rng);
  }
};

template <typename T> void SimpleRPUDevice<T>::diffuseWeights(T **weights, RNG<T> &rng) {

  T diffusion = getPar().diffusion;
  if (diffusion > (T)0.0) {
    T *w = weights[0];
    PRAGMA_SIMD
    for (int i = 0; i < this->size_; ++i) {
      w[i] += diffusion * rng.sampleGauss();
    }
  }
}

template <typename T> void SimpleRPUDevice<T>::clipWeights(T **weights, T clip) {
  // apply hard bounds

  if (clip >= (T)0.0) {
    T *w = weights[0];
    PRAGMA_SIMD
    for (int i = 0; i < this->size_; ++i) {
      w[i] = MIN(MAX(w[i], -clip), clip);
    }
  }
}

template <typename T>
void SimpleRPUDevice<T>::resetCols(
    T **weights, int start_col, int n_col_in, T reset_prob, RealWorldRNG<T> &rng) {

  T *w = weights[0];
  int n_col = (n_col_in >= 0) ? n_col_in : this->x_size_;

  T reset_std = getPar().reset_std;
  for (int j = 0; j < this->x_size_; ++j) {
    if ((start_col + n_col <= this->x_size_ && j >= start_col && j < start_col + n_col) ||
        (start_col + n_col > this->x_size_ &&
         ((j >= start_col) || (j < n_col - (this->x_size_ - start_col))))) {
      PRAGMA_SIMD
      for (int i = 0; i < this->d_size_; ++i) {
        if (reset_prob == (T)1.0 || rng.sampleUniform() < reset_prob) {
          int k = i * this->x_size_ + j;
          w[k] = (reset_std > (T)0.0 ? reset_std * rng.sampleGauss() : (T)0.0);
        }
      }
    }
  }
}

template <typename T>
void SimpleRPUDevice<T>::populate(const SimpleRPUDeviceMetaParameter<T> &p, RealWorldRNG<T> *rng) {

  par_storage_ = p.cloneUnique();
  par_storage_->initialize();

  if (p.drift.nu > (T)0.0) {
    wdrifter_ = RPU::make_unique<WeightDrifter<T>>(this->size_, p.drift, rng);
  } else {
    wdrifter_ = nullptr;
  }
}

template struct SimpleRPUDeviceMetaParameter<float>;
template class AbstractRPUDevice<float>;
template class SimpleRPUDevice<float>;
#ifdef RPU_USE_DOUBLE
template class AbstractRPUDevice<double>;
template class SimpleRPUDevice<double>;
template struct SimpleRPUDeviceMetaParameter<double>;
#endif
#ifdef RPU_USE_FP16
template class AbstractRPUDevice<half_t>;
template class SimpleRPUDevice<half_t>;
template struct SimpleRPUDeviceMetaParameter<half_t>;
#endif

} // namespace RPU
