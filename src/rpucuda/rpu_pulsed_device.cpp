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

#include "rpu_pulsed_device.h"
#include "math_util.h"
#include "utility_functions.h"
#include <limits>
#include <memory>

namespace RPU {

/******************************************************************************************/
/* PulsedRPUDeviceMetaParameter*/

template <typename T>
void PulsedRPUDeviceMetaParameter<T>::printToStream(std::stringstream &ss) const {
  if (this->_device_parameter_mode_manual) {
    ss << "\n\t Device parameters set manually\n";
  } else {
    ss << "Pulsed device parameter:" << std::endl;
    ss << "\t granularity (calc.):\t" << this->calcWeightGranularity() << std::endl;
    if (this->construction_seed != 0) {
      ss << "\t construction_seed:\t" << this->construction_seed << std::endl;
    }
    ss << "\t enforce_consistency:\t" << std::boolalpha << enforce_consistency << std::endl;
    ss << "\t perfect_bias:\t\t" << std::boolalpha << perfect_bias << std::endl;

    ss << "\t dw_min:\t\t" << dw_min << "\t(dtod=" << dw_min_dtod << ", ctoc=" << dw_min_std << ")"
       << std::endl;

    ss << "\t up_down:\t\t" << up_down << "\t(dtod=" << up_down_dtod << ")" << std::endl;

    ss << "\t w min:\t\t\t" << w_min << "\t(dtod=" << w_min_dtod << ")" << std::endl;
    ss << "\t w max:\t\t\t" << w_max << "\t(dtod=" << w_max_dtod << ")" << std::endl;

    ss << "\t resets to:\t\t" << reset << "\t(dtod=" << reset_dtod << ", ctoc=" << reset_std << ")"
       << std::endl;

    if (this->implementsWriteNoise() && write_noise_std > (T)0.0) {
      ss << "\t write noise std:\t" << write_noise_std << std::endl;
    }

    if (this->lifetime > 0) {
      ss << "\t lifetime [decay]:\t" << this->lifetime << "\t(dtod=" << lifetime_dtod << ")"
         << std::endl;
    }

    if (corrupt_devices_prob > 0) {
      ss << "\t corrupt_devices_prob:\t" << corrupt_devices_prob << std::endl;
      ss << "\t corrupt_devices_range:\t" << corrupt_devices_range << std::endl;
    }

    if (this->drift.nu > 0) {
      ss << "   Drift:" << std::endl;
      this->drift.printToStream(ss);
    }

    if (this->diffusion > 0) {
      ss << "   Diffusion:" << std::endl;
      ss << "\t diffusion:\t\t" << this->diffusion << "\t(dtod=" << diffusion_dtod << ")"
         << std::endl;
    }
  }
}

template struct PulsedRPUDeviceMetaParameter<float>;
template class AbstractRPUDevice<float>;

#ifdef RPU_USE_DOUBLE
template struct PulsedRPUDeviceMetaParameter<double>;
template class AbstractRPUDevice<double>;
#endif

/******************************************************************************************/
/* PulsedRPUDevice*/

template <typename T> void PulsedRPUDevice<T>::initialize() { allocateContainers(); }

template <typename T> void PulsedRPUDevice<T>::allocateContainers() {

  freeContainers();
  int d_sz = this->d_size_;
  int x_sz = this->x_size_;

  w_max_bound_ = Array_2D_Get<T>(d_sz, x_sz);
  w_min_bound_ = Array_2D_Get<T>(d_sz, x_sz);

  w_scale_up_ = Array_2D_Get<T>(d_sz, x_sz);
  w_scale_down_ = Array_2D_Get<T>(d_sz, x_sz);

  w_decay_scale_ = Array_2D_Get<T>(d_sz, x_sz);

  w_diffusion_rate_ = Array_2D_Get<T>(d_sz, x_sz);
  w_reset_bias_ = Array_2D_Get<T>(d_sz, x_sz);
  w_persistent_ = Array_2D_Get<T>(d_sz, x_sz);

  // we better set everything to zero.
  for (int j = 0; j < x_sz; ++j) {
    for (int i = 0; i < d_sz; ++i) {
      w_max_bound_[i][j] = std::numeric_limits<T>::max();
      w_min_bound_[i][j] = std::numeric_limits<T>::min();
      w_scale_up_[i][j] = (T)0.0;
      w_scale_down_[i][j] = (T)0.0;
      w_decay_scale_[i][j] = (T)1.0; // no decay
      w_diffusion_rate_[i][j] = (T)0.0;
      w_reset_bias_[i][j] = (T)0.0;
      w_persistent_[i][j] = (T)0.0;
    }
  }
  containers_allocated_ = true;
}

template <typename T> void PulsedRPUDevice<T>::freeContainers() {

  if (containers_allocated_) {

    Array_2D_Free<T>(w_max_bound_);
    Array_2D_Free<T>(w_min_bound_);
    Array_2D_Free<T>(w_scale_up_);
    Array_2D_Free<T>(w_scale_down_);
    Array_2D_Free<T>(w_decay_scale_);
    Array_2D_Free<T>(w_diffusion_rate_);
    Array_2D_Free<T>(w_reset_bias_);
    Array_2D_Free<T>(w_persistent_);

    containers_allocated_ = false;
  }
}

// ctor
template <typename T>
PulsedRPUDevice<T>::PulsedRPUDevice(int x_sz, int d_sz) : PulsedRPUDeviceBase<T>(x_sz, d_sz) {
  initialize();
}

// template <typename T>
// PulsedRPUDevice<T>::
// PulsedRPUDevice(int x_sz, int d_sz, const PulsedRPUDeviceMetaParameter<T> * par,
// 		  RealWorldRNG<T> *rng)
// {
//   initialize(x_sz,d_sz);
//   populate(par,rng);
// }

// dtor
template <typename T> PulsedRPUDevice<T>::~PulsedRPUDevice() { freeContainers(); }

// copy construcutor
template <typename T>
PulsedRPUDevice<T>::PulsedRPUDevice(const PulsedRPUDevice<T> &other)
    : PulsedRPUDeviceBase<T>(other) {

  initialize();

  for (int j = 0; j < this->x_size_; ++j) {
    for (int i = 0; i < this->d_size_; ++i) {

      w_scale_up_[i][j] = other.w_scale_up_[i][j];
      w_scale_down_[i][j] = other.w_scale_down_[i][j];

      w_max_bound_[i][j] = other.w_max_bound_[i][j];
      w_min_bound_[i][j] = other.w_min_bound_[i][j];

      w_decay_scale_[i][j] = other.w_decay_scale_[i][j];
      w_diffusion_rate_[i][j] = other.w_diffusion_rate_[i][j];

      w_reset_bias_[i][j] = other.w_reset_bias_[i][j];

      w_persistent_[i][j] = other.w_persistent_[i][j];
    }
  }
}

// copy assignment
template <typename T>
PulsedRPUDevice<T> &PulsedRPUDevice<T>::operator=(const PulsedRPUDevice<T> &other) {

  PulsedRPUDevice<T> tmp(other);
  swap(*this, tmp);
  return *this;
}

// move constructor
template <typename T> PulsedRPUDevice<T>::PulsedRPUDevice(PulsedRPUDevice<T> &&other) {

  *this = std::move(other);
}

// move assignment
template <typename T>
PulsedRPUDevice<T> &PulsedRPUDevice<T>::operator=(PulsedRPUDevice<T> &&other) {

  PulsedRPUDeviceBase<T>::operator=(std::move(other));

  containers_allocated_ = other.containers_allocated_;

  // pointers
  w_scale_up_ = other.w_scale_up_;
  w_scale_down_ = other.w_scale_down_;

  w_max_bound_ = other.w_max_bound_;
  w_min_bound_ = other.w_min_bound_;

  w_decay_scale_ = other.w_decay_scale_;
  w_diffusion_rate_ = other.w_diffusion_rate_;
  w_reset_bias_ = other.w_reset_bias_;
  w_persistent_ = other.w_persistent_;

  // set pointers to null
  other.w_scale_up_ = nullptr;
  other.w_scale_down_ = nullptr;

  other.w_max_bound_ = nullptr;
  other.w_min_bound_ = nullptr;

  other.w_decay_scale_ = nullptr;
  other.w_diffusion_rate_ = nullptr;

  other.w_reset_bias_ = nullptr;
  other.w_persistent_ = nullptr;

  return *this;
}

template <typename T> void PulsedRPUDevice<T>::getDPNames(std::vector<std::string> &names) const {

  names.clear();
  names.push_back(std::string("max_bound"));
  names.push_back(std::string("min_bound"));
  names.push_back(std::string("dwmin_up"));
  names.push_back(std::string("dwmin_down"));
  names.push_back(std::string("decay_scales"));
  names.push_back(std::string("diffusion_rates"));
  if (!getPar().legacy_params) {
    names.push_back(std::string("reset_bias"));
    names.push_back(std::string(
        "drift_nu")); // we only save the nu, not the t/w0 etc. drift will thus reset at zero
  }
  if (getPar().usesPersistentWeight()) {
    names.push_back(std::string("persistent_weights"));
  }
}

template <typename T>
void PulsedRPUDevice<T>::getDeviceParameter(std::vector<T *> &data_ptrs) const {
  // note that memory (x_sz*d_sz per ptr) assumed to be initialized from outside !!

  std::vector<std::string> names;
  getDPNames(names);

  if (data_ptrs.size() < names.size()) {
    RPU_FATAL("More data pointers expected");
  }
  int n_drift = 0;
  for (int i = 0; i < this->size_; ++i) {
    int n = 0;
    data_ptrs[n++][i] = w_max_bound_[0][i];
    data_ptrs[n++][i] = w_min_bound_[0][i];
    data_ptrs[n++][i] = w_scale_up_[0][i];
    data_ptrs[n++][i] = w_scale_down_[0][i];
    data_ptrs[n++][i] = w_decay_scale_[0][i];
    data_ptrs[n++][i] = w_diffusion_rate_[0][i];
    if (!getPar().legacy_params) {
      data_ptrs[n++][i] = w_reset_bias_[0][i];
      data_ptrs[n][i] = (T)0.0;
      n_drift = n++;
    }
    if (getPar().usesPersistentWeight()) {
      data_ptrs[n++][i] = w_persistent_[0][i];
    }
  }
  if (!getPar().legacy_params && this->hasWDrifter()) {
    this->wdrifter_->getNu(data_ptrs[n_drift]);
  }
};

template <typename T>
void PulsedRPUDevice<T>::setDeviceParameter(T **out_weights, const std::vector<T *> &data_ptrs) {

  std::vector<std::string> names;
  getDPNames(names);

  if (data_ptrs.size() < names.size()) {
    RPU_FATAL("more data pointers expected");
  }

  T dw_min = (T)0.0;
  int n_drift = 0;
  for (int i = 0; i < this->size_; ++i) {
    int n = 0;
    w_max_bound_[0][i] = data_ptrs[n++][i];
    w_min_bound_[0][i] = data_ptrs[n++][i];
    w_scale_up_[0][i] = data_ptrs[n++][i];   // assumed to be positive
    w_scale_down_[0][i] = data_ptrs[n++][i]; // assumed to be positive
    w_decay_scale_[0][i] = data_ptrs[n++][i];
    w_diffusion_rate_[0][i] = data_ptrs[n++][i];
    if (!getPar().legacy_params) {
      w_reset_bias_[0][i] = data_ptrs[n++][i];
      n_drift = n++;
    } else {
      w_reset_bias_[0][i] = (T)0.0;
    }
    if (getPar().usesPersistentWeight()) {
      w_persistent_[0][i] = data_ptrs[n++][i];
    }

    dw_min += (fabs(w_scale_up_[0][i]) + fabs(w_scale_down_[0][i])) / (T)2.0;
  }

  if (!getPar().legacy_params && this->hasWDrifter()) {
    this->wdrifter_->setNu(data_ptrs[n_drift]);
  }

  dw_min /= this->size_;
  // need dw_min for update management
  if (fabs(dw_min - getPar().dw_min) / getPar().dw_min > 2 * getPar().dw_min_dtod) {
    RPU_WARNING("DW min seems to have changed during hidden parameter set. Will update parameter "
                "with estimated value.");
    getPar().dw_min = dw_min; //!! update par. Should be possible since unique
    this->setWeightGranularity(getPar().calcWeightGranularity());
  }

  // update the weights according to the bounds
  this->onSetWeights(out_weights);
};

/********************************************************************************/
/* compute functions  */

template <typename T> void PulsedRPUDevice<T>::decayWeights(T **weights, bool bias_no_decay) {

  // maybe a bit overkill to check the bounds...
  T *w = getPar().usesPersistentWeight() ? w_persistent_[0] : weights[0];
  T *wd = w_decay_scale_[0];
  T *max_bound = w_max_bound_[0];
  T *min_bound = w_min_bound_[0];
  T *b = w_reset_bias_[0];

  if (!bias_no_decay) {
    PRAGMA_SIMD
    for (int i = 0; i < this->size_; ++i) {
      w[i] = (w[i] - b[i]) * wd[i] + b[i];
      w[i] = MIN(w[i], max_bound[i]);
      w[i] = MAX(w[i], min_bound[i]);
    }
  } else {
    const int last_col = this->x_size_ - 1; // x-major (ie row major)
    PRAGMA_SIMD
    for (int i = 0; i < this->size_; ++i) {
      T s = (i % this->x_size_ == last_col) ? (T)1.0 : wd[i];
      w[i] = (w[i] - b[i]) * s + b[i];
      w[i] = MIN(w[i], max_bound[i]);
      w[i] = MAX(w[i], min_bound[i]);
    }
  }
  applyUpdateWriteNoise(weights);
}

template <typename T>
void PulsedRPUDevice<T>::decayWeights(T **weights, T alpha, bool bias_no_decay) {

  // maybe a bit overkill to check the bounds...
  T *w = getPar().usesPersistentWeight() ? w_persistent_[0] : weights[0];
  T *wd = w_decay_scale_[0];
  T *max_bound = w_max_bound_[0];
  T *min_bound = w_min_bound_[0];
  T *b = w_reset_bias_[0];

  if (!bias_no_decay) {
    PRAGMA_SIMD
    for (int i = 0; i < this->size_; ++i) {
      T s = 1 + alpha * (wd[i] - 1);
      w[i] = (w[i] - b[i]) * s + b[i];
      w[i] = MIN(w[i], max_bound[i]);
      w[i] = MAX(w[i], min_bound[i]);
    }
  } else {
    const int last_col = this->x_size_ - 1; // x-major (ie row major)
    PRAGMA_SIMD
    for (int i = 0; i < this->size_; ++i) {
      T s = (i % this->x_size_ == last_col) ? (T)1.0 : (1 + alpha * (wd[i] - 1));
      w[i] = (w[i] - b[i]) * s + b[i];
      w[i] = MIN(w[i], max_bound[i]);
      w[i] = MAX(w[i], min_bound[i]);
    }
  }
  applyUpdateWriteNoise(weights);
}

template <typename T>
void PulsedRPUDevice<T>::driftWeights(T **weights, T time_since_last_call, RNG<T> &rng) {
  if (this->hasWDrifter()) {
    T **w = getPar().usesPersistentWeight() ? w_persistent_ : weights;
    PulsedRPUDeviceBase<T>::driftWeights(w, time_since_last_call, rng);
    this->wdrifter_->saturate(w[0], w_min_bound_[0], w_max_bound_[0]);
    applyUpdateWriteNoise(weights);
  }
}

template <typename T> void PulsedRPUDevice<T>::diffuseWeights(T **weights, RNG<T> &rng) {

  T *w = getPar().usesPersistentWeight() ? w_persistent_[0] : weights[0];
  T *diffusion_rate = &(w_diffusion_rate_[0][0]);
  T *max_bound = &(w_max_bound_[0][0]);
  T *min_bound = &(w_min_bound_[0][0]);

  PRAGMA_SIMD
  for (int i = 0; i < this->size_; ++i) {
    w[i] += diffusion_rate[i] * rng.sampleGauss();
    w[i] = MIN(w[i], max_bound[i]);
    w[i] = MAX(w[i], min_bound[i]);
  }
  applyUpdateWriteNoise(weights);
}

template <typename T> void PulsedRPUDevice<T>::clipWeights(T **weights, T clip) {
  // apply hard bounds
  T *w = getPar().usesPersistentWeight() ? w_persistent_[0] : weights[0];
  T *max_bound = &(w_max_bound_[0][0]);
  T *min_bound = &(w_min_bound_[0][0]);
  if (clip < 0.0) { // only apply bounds
    PRAGMA_SIMD
    for (int i = 0; i < this->size_; ++i) {
      w[i] = MIN(w[i], max_bound[i]);
      w[i] = MAX(w[i], min_bound[i]);
    }
  } else {
    PRAGMA_SIMD
    for (int i = 0; i < this->size_; ++i) {
      w[i] = MIN(w[i], MIN(max_bound[i], clip));
      w[i] = MAX(w[i], MAX(min_bound[i], -clip));
    }
  }
  applyUpdateWriteNoise(weights);
}

template <typename T>
void PulsedRPUDevice<T>::resetCols(
    T **weights, int start_col, int n_col, T reset_prob, RealWorldRNG<T> &rng) {

  if (getPar().usesPersistentWeight()) {
    RPU_FATAL("ResetCols is not supported with write_noise_std>0!");
  }

  T reset_std = getPar().reset_std;
  for (int j = 0; j < this->x_size_; ++j) {
    if ((start_col + n_col <= this->x_size_ && j >= start_col && j < start_col + n_col) ||
        (start_col + n_col > this->x_size_ &&
         ((j >= start_col) || (j < n_col - (this->x_size_ - start_col))))) {
      PRAGMA_SIMD
      for (int i = 0; i < this->d_size_; ++i) {
        if (reset_prob == 1 || rng.sampleUniform() < reset_prob) {
          weights[i][j] =
              w_reset_bias_[i][j] + (reset_std > 0 ? reset_std * rng.sampleGauss() : (T)0.0);
          weights[i][j] = MIN(weights[i][j], w_max_bound_[i][j]);
          weights[i][j] = MAX(weights[i][j], w_min_bound_[i][j]);
        }
      }
    }
  }
}

template <typename T>
void PulsedRPUDevice<T>::resetAtIndices(
    T **weights, std::vector<int> x_major_indices, RealWorldRNG<T> &rng) {

  if (getPar().usesPersistentWeight()) {
    RPU_FATAL("ResetIndices is not supported with write_noise_std>0!");
  }

  T reset_std = getPar().reset_std;

  for (const auto &index : x_major_indices) {
    int i = index / this->x_size_;
    int j = index % this->x_size_;

    weights[i][j] = w_reset_bias_[i][j] + (reset_std > 0 ? reset_std * rng.sampleGauss() : (T)0.0);
    weights[i][j] = MIN(weights[i][j], w_max_bound_[i][j]);
    weights[i][j] = MAX(weights[i][j], w_min_bound_[i][j]);
  }
}

template <typename T>
void PulsedRPUDevice<T>::copyInvertDeviceParameter(const PulsedRPUDeviceBase<T> *rpu_device) {
  if (!containers_allocated_) {
    RPU_FATAL("Containers empty");
  }

  if (rpu_device->getXSize() != this->x_size_ || rpu_device->getDSize() != this->d_size_) {
    RPU_FATAL("Size mismatch");
  }
  const auto *rpu = dynamic_cast<const PulsedRPUDevice<T> *>(rpu_device);
  if (rpu == nullptr) {
    RPU_FATAL("Expect RPU Pulsed device");
  }

  for (int j = 0; j < this->x_size_; ++j) {
    for (int i = 0; i < this->d_size_; ++i) {

      // scaleup/down both have same sign
      std::swap(w_scale_up_[i][j], w_scale_down_[i][j]);

      // min max have sign. mirror
      T b = w_max_bound_[i][j];
      w_max_bound_[i][j] = -w_min_bound_[i][j];
      w_min_bound_[i][j] = -b;
    }
  }
}

template <typename T> bool PulsedRPUDevice<T>::onSetWeights(T **weights) {

  // apply hard bounds to given weights
  T *w = weights[0];
  T *max_bound = &(w_max_bound_[0][0]);
  T *min_bound = &(w_min_bound_[0][0]);
  PRAGMA_SIMD
  for (int i = 0; i < this->size_; ++i) {
    w[i] = MIN(w[i], max_bound[i]);
    w[i] = MAX(w[i], min_bound[i]);
  }

  if (getPar().usesPersistentWeight()) {
    PRAGMA_SIMD
    for (int i = 0; i < this->size_; i++) {
      w_persistent_[0][i] = w[i];
    }
    applyUpdateWriteNoise(weights);
    return true; // modified device thus true
  } else {
    return false; // whether device was changed
  }
}

template <typename T> void PulsedRPUDevice<T>::applyUpdateWriteNoise(T **weights) {
  // applies new noise to ALL weight values
  T uw_std = getPar().getScaledWriteNoise();

  if (uw_std <= 0) {
    return; // nothing to be done, weights assumed to already updated
  }

  for (int i = 0; i < this->size_; i++) {
    weights[0][i] = w_persistent_[0][i] + uw_std * write_noise_rng_.sampleGauss();
  }
}

/*********************************************************************************/
/* populate */

template <typename T>
void PulsedRPUDevice<T>::populate(const PulsedRPUDeviceMetaParameter<T> &p, RealWorldRNG<T> *rng) {

  PulsedRPUDeviceBase<T>::populate(p, rng); // will clone and init parametrs

  auto &par = getPar();

  T up_down = par.up_down;
  T up_down_std = par.up_down_dtod;

  T up_bias = up_down > 0 ? (T)0.0 : up_down;
  T down_bias = up_down > 0 ? -up_down : (T)0.0;

  T gain_std = par.dw_min_dtod;

  // par.w_min = -fabs(par.w_min);
  // par.w_max = fabs(par.w_max);

  if ((par.w_min > 0) || (par.w_max < 0)) {
    RPU_FATAL("The closed interval [w_min,w_max] needs to contain 0.");
  }

  for (int j = 0; j < this->x_size_; ++j) {
    for (int i = 0; i < this->d_size_; ++i) {

      w_max_bound_[i][j] = par.w_max * (1 + par.w_max_dtod * rng->sampleGauss());
      w_min_bound_[i][j] = par.w_min * (1 + par.w_min_dtod * rng->sampleGauss());

      T gain = ((T)1.0 + gain_std * rng->sampleGauss());
      T r = up_down_std * rng->sampleGauss();

      w_scale_up_[i][j] = (up_bias + gain + r) * par.dw_min; // to reduce mults in updates
      w_scale_down_[i][j] = (down_bias + gain - r) * par.dw_min;

      // enforce consistency
      if (par.enforce_consistency) {
        w_scale_up_[i][j] = fabs(w_scale_up_[i][j]);
        w_scale_down_[i][j] = fabs(w_scale_down_[i][j]);

        if (w_min_bound_[i][j] > w_max_bound_[i][j]) {
          std::swap(w_min_bound_[i][j], w_max_bound_[i][j]);
        }
        w_max_bound_[i][j] = fabs(w_max_bound_[i][j]);
        w_min_bound_[i][j] = -fabs(w_min_bound_[i][j]);
      } else {
        // "turn off" weight if max<min
        if (w_min_bound_[i][j] > w_max_bound_[i][j]) {
          T m = w_max_bound_[i][j] + (w_min_bound_[i][j] - w_max_bound_[i][j]) / ((T)2.0);
          w_max_bound_[i][j] = m;
          w_min_bound_[i][j] = m;
        }
      }

      // corrupt devices
      if (par.corrupt_devices_prob > rng->sampleUniform()) {
        // stuck somewhere in min_max
        T mn = MAX(MIN(w_max_bound_[i][j], w_min_bound_[i][j]), -fabs(par.corrupt_devices_range));
        T mx = MIN(MAX(w_max_bound_[i][j], w_min_bound_[i][j]), fabs(par.corrupt_devices_range));

        T value = mn + (mx - mn) * rng->sampleUniform();
        w_max_bound_[i][j] = value;
        w_min_bound_[i][j] = value;
        w_scale_up_[i][j] = (T)0.0;
        w_scale_down_[i][j] = (T)0.0;
      }

      // perfect bias
      if ((par.perfect_bias) && (j == this->x_size_ - 1)) {
        w_scale_up_[i][j] = par.dw_min;
        w_scale_down_[i][j] = par.dw_min;
        w_min_bound_[i][j] = 100 * par.w_min; // essentially no bound
        w_max_bound_[i][j] = 100 * par.w_max; // essentially no bound
      }

      //--------------------
      // diffusion
      {
        T t = fabs(par.diffusion * (1 + par.diffusion_dtod * rng->sampleGauss()));
        w_diffusion_rate_[i][j] = t;
      }

      //--------------------
      // reset
      { // additive dtod
        T t = par.reset + par.reset_dtod * rng->sampleGauss();
        w_reset_bias_[i][j] = t;
      }

      //--------------------
      // decay
      {
        T t = par.lifetime * ((T)1.0 + par.lifetime_dtod * rng->sampleGauss());
        w_decay_scale_[i][j] = t > 1.0 ? (T)((T)1. - ((T)1. / t)) : (T)0.0;
      }
    }
  }
}

template <typename T> void PulsedRPUDevice<T>::printDP(int x_count, int d_count) const {
  int x_count1 = x_count;
  int d_count1 = d_count;
  if (x_count < 0 || x_count > this->x_size_) {
    x_count1 = this->x_size_;
  }

  if (d_count < 0 || d_count > this->d_size_) {
    d_count1 = this->d_size_;
  }

  bool persist_if = getPar().usesPersistentWeight();

  for (int i = 0; i < d_count1; ++i) {
    for (int j = 0; j < x_count1; ++j) {
      std::cout << "[<" << w_max_bound_[i][j] << ", ";
      std::cout << w_min_bound_[i][j] << ">, <";
      std::cout << w_scale_up_[i][j] << ", ";
      std::cout << w_scale_down_[i][j] << "> ";
      std::cout.precision(10);
      std::cout << w_decay_scale_[i][j] << ", ";
      std::cout.precision(6);
      std::cout << w_diffusion_rate_[i][j] << ", ";
      std::cout << w_reset_bias_[i][j];
      if (persist_if) {
        std::cout << ", " << w_persistent_[i][j];
      }
      std::cout << "]";
    }
    std::cout << std::endl;
  }
}

template class PulsedRPUDevice<float>;
#ifdef RPU_USE_DOUBLE
template class PulsedRPUDevice<double>;
#endif

} // namespace RPU
