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

    if (this->lifetime > 0) {
      ss << "\t lifetime [decay]:\t" << this->lifetime << "\t(dtod=" << lifetime_dtod << ")"
         << std::endl;
    }

    if (corrupt_devices_prob > 0) {
      ss << "\t corrupt_devices_prob:\t" << corrupt_devices_prob << std::endl;
      ss << "\t corrupt_devices_range:\t" << corrupt_devices_range << std::endl;
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

  sup_ = Array_2D_Get<PulsedDPStruc<T>>(d_sz, x_sz);

  w_max_bound_ = Array_2D_Get<T>(d_sz, x_sz);
  w_min_bound_ = Array_2D_Get<T>(d_sz, x_sz);
  w_scale_up_ = Array_2D_Get<T>(d_sz, x_sz);
  w_scale_down_ = Array_2D_Get<T>(d_sz, x_sz);
  w_decay_scale_ = Array_2D_Get<T>(d_sz, x_sz);
  w_diffusion_rate_ = Array_2D_Get<T>(d_sz, x_sz);
  w_reset_bias_ = Array_2D_Get<T>(d_sz, x_sz);

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

      sup_[i][j].max_bound = std::numeric_limits<T>::max();
      sup_[i][j].min_bound = std::numeric_limits<T>::min();
      sup_[i][j].scale_up = (T)0.0;
      sup_[i][j].scale_down = (T)0.0;
      sup_[i][j].diffusion_rate = (T)0.0;
      sup_[i][j].decay_scale = (T)1.0;
      sup_[i][j].reset_bias = (T)0.0;
    }
  }
  containers_allocated_ = true;
}

template <typename T> void PulsedRPUDevice<T>::freeContainers() {

  if (containers_allocated_) {

    Array_2D_Free<PulsedDPStruc<T>>(sup_);

    Array_2D_Free<T>(w_max_bound_);
    Array_2D_Free<T>(w_min_bound_);
    Array_2D_Free<T>(w_scale_up_);
    Array_2D_Free<T>(w_scale_down_);
    Array_2D_Free<T>(w_decay_scale_);
    Array_2D_Free<T>(w_diffusion_rate_);
    Array_2D_Free<T>(w_reset_bias_);

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
      sup_[i][j] = other.sup_[i][j];

      w_scale_up_[i][j] = other.w_scale_up_[i][j];
      w_scale_down_[i][j] = other.w_scale_down_[i][j];

      w_max_bound_[i][j] = other.w_max_bound_[i][j];
      w_min_bound_[i][j] = other.w_min_bound_[i][j];

      w_decay_scale_[i][j] = other.w_decay_scale_[i][j];
      w_diffusion_rate_[i][j] = other.w_diffusion_rate_[i][j];

      w_reset_bias_[i][j] = other.w_reset_bias_[i][j];
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
  sup_ = other.sup_;
  w_scale_up_ = other.w_scale_up_;
  w_scale_down_ = other.w_scale_down_;

  w_max_bound_ = other.w_max_bound_;
  w_min_bound_ = other.w_min_bound_;

  w_decay_scale_ = other.w_decay_scale_;
  w_diffusion_rate_ = other.w_diffusion_rate_;
  w_reset_bias_ = other.w_reset_bias_;

  // set pointers to null
  other.sup_ = nullptr;
  other.w_scale_up_ = nullptr;
  other.w_scale_down_ = nullptr;

  other.w_max_bound_ = nullptr;
  other.w_min_bound_ = nullptr;

  other.w_decay_scale_ = nullptr;
  other.w_diffusion_rate_ = nullptr;

  other.w_reset_bias_ = nullptr;

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

  for (int i = 0; i < this->size_; ++i) {
    data_ptrs[0][i] = w_max_bound_[0][i];
    data_ptrs[1][i] = w_min_bound_[0][i];
    data_ptrs[2][i] = w_scale_up_[0][i];
    data_ptrs[3][i] = w_scale_down_[0][i];
    data_ptrs[4][i] = w_decay_scale_[0][i];
    data_ptrs[5][i] = w_diffusion_rate_[0][i];
    if (!getPar().legacy_params) {
      data_ptrs[6][i] = w_reset_bias_[0][i];
    }
  }
};

template <typename T>
void PulsedRPUDevice<T>::setDeviceParameter(const std::vector<T *> &data_ptrs) {

  std::vector<std::string> names;
  getDPNames(names);

  if (data_ptrs.size() < names.size()) {
    RPU_FATAL("more data pointers expected");
  }

  T dw_min = (T)0.0;
  for (int i = 0; i < this->size_; ++i) {
    w_max_bound_[0][i] = data_ptrs[0][i];
    w_min_bound_[0][i] = data_ptrs[1][i];
    w_scale_up_[0][i] = data_ptrs[2][i];   // assumed to be positive
    w_scale_down_[0][i] = data_ptrs[3][i]; // assumed to be positive
    w_decay_scale_[0][i] = data_ptrs[4][i];
    w_diffusion_rate_[0][i] = data_ptrs[5][i];
    if (!getPar().legacy_params) {
      w_reset_bias_[0][i] = data_ptrs[6][i];
    } else {
      w_reset_bias_[0][i] = (T)0.0;
    }
    dw_min += (fabs(w_scale_up_[0][i]) + fabs(w_scale_down_[0][i])) / (T)2.0;

    // copy to sup
    sup_[0][i].max_bound = w_max_bound_[0][i];
    sup_[0][i].min_bound = w_min_bound_[0][i];
    sup_[0][i].scale_up = w_scale_up_[0][i];
    sup_[0][i].scale_down = w_scale_down_[0][i];

    sup_[0][i].decay_scale = w_decay_scale_[0][i];
    sup_[0][i].diffusion_rate = w_diffusion_rate_[0][i];
    sup_[0][i].reset_bias = w_reset_bias_[0][i];
  }

  dw_min /= this->size_;
  // need dw_min for update management
  this->checkDwMin(dw_min);

  getPar().dw_min = dw_min; //!! update par. Should be possible since unique
};

/********************************************************************************/
/* compute functions  */

template <typename T> void PulsedRPUDevice<T>::decayWeights(T **weights, bool bias_no_decay) {

  // maybe a bit overkill to check the bounds...
  T *wd = w_decay_scale_[0];
  T *w = weights[0];
  T *max_bound = w_max_bound_[0];
  T *min_bound = w_min_bound_[0];

  if (!bias_no_decay) {
    PRAGMA_SIMD
    for (int i = 0; i < this->size_; ++i) {
      w[i] *= wd[i];
      w[i] = MIN(w[i], max_bound[i]);
      w[i] = MAX(w[i], min_bound[i]);
    }
  } else {
    const int last_col = this->x_size_ - 1; // x-major (ie row major)
    PRAGMA_SIMD
    for (int i = 0; i < this->size_; ++i) {
      w[i] *= (i % this->x_size_ == last_col) ? (T)1.0 : wd[i];
      w[i] = MIN(w[i], max_bound[i]);
      w[i] = MAX(w[i], min_bound[i]);
    }
  }
}

template <typename T>
void PulsedRPUDevice<T>::decayWeights(T **weights, T alpha, bool bias_no_decay) {

  // maybe a bit overkill to check the bounds...

  T *wd = w_decay_scale_[0];
  T *w = weights[0];
  T *max_bound = w_max_bound_[0];
  T *min_bound = w_min_bound_[0];

  if (!bias_no_decay) {
    PRAGMA_SIMD
    for (int i = 0; i < this->size_; ++i) {
      w[i] *= 1 + alpha * (wd[i] - 1);
      w[i] = MIN(w[i], max_bound[i]);
      w[i] = MAX(w[i], min_bound[i]);
    }
  } else {
    const int last_col = this->x_size_ - 1; // x-major (ie row major)
    PRAGMA_SIMD
    for (int i = 0; i < this->size_; ++i) {
      w[i] *= (i % this->x_size_ == last_col) ? (T)1.0 : (1 + alpha * (wd[i] - 1));
      w[i] = MIN(w[i], max_bound[i]);
      w[i] = MAX(w[i], min_bound[i]);
    }
  }
}

template <typename T> void PulsedRPUDevice<T>::diffuseWeights(T **weights, RNG<T> &rng) {

  T *diffusion_rate = &(w_diffusion_rate_[0][0]);
  T *w = &(weights[0][0]);
  T *max_bound = &(w_max_bound_[0][0]);
  T *min_bound = &(w_min_bound_[0][0]);

  PRAGMA_SIMD
  for (int i = 0; i < this->size_; ++i) {
    w[i] += diffusion_rate[i] * rng.sampleGauss();
    w[i] = MIN(w[i], max_bound[i]);
    w[i] = MAX(w[i], min_bound[i]);
  }
}

template <typename T> void PulsedRPUDevice<T>::clipWeights(T **weights, T clip) {
  // apply hard bounds

  T *w = &(weights[0][0]);
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
}

template <typename T>
void PulsedRPUDevice<T>::resetCols(
    T **weights, int start_col, int n_col, T reset_prob, RealWorldRNG<T> &rng) {

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

  PulsedDPStruc<T> **sup = rpu->getDPStruc();

  for (int j = 0; j < this->x_size_; ++j) {
    for (int i = 0; i < this->d_size_; ++i) {

      // scaleup/down both have same sign
      sup_[i][j].scale_up = sup[i][j].scale_down;
      w_scale_up_[i][j] = sup[i][j].scale_down;

      sup_[i][j].scale_down = sup[i][j].scale_up;
      w_scale_down_[i][j] = sup[i][j].scale_up;

      // min max have sign. mirror
      sup_[i][j].min_bound = -sup[i][j].max_bound;
      sup_[i][j].max_bound = -sup[i][j].min_bound;

      w_max_bound_[i][j] = -sup[i][j].min_bound;
      w_min_bound_[i][j] = -sup[i][j].max_bound;
    }
  }
}

template <typename T> bool PulsedRPUDevice<T>::onSetWeights(T **weights) {
  // apply hard bounds
  this->clipWeights(weights, (T)-1.0);
  return false; // whether device was changed
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

      sup_[i][j].max_bound = par.w_max * (1 + par.w_max_dtod * rng->sampleGauss());
      sup_[i][j].min_bound = par.w_min * (1 + par.w_min_dtod * rng->sampleGauss());

      T gain = ((T)1.0 + gain_std * rng->sampleGauss());
      T r = up_down_std * rng->sampleGauss();

      sup_[i][j].scale_up = (up_bias + gain + r) * par.dw_min; // to reduce mults in updates
      sup_[i][j].scale_down = (down_bias + gain - r) * par.dw_min;

      // enforce consistency
      if (par.enforce_consistency) {
        sup_[i][j].scale_up = fabs(sup_[i][j].scale_up);
        sup_[i][j].scale_down = fabs(sup_[i][j].scale_down);

        if (sup_[i][j].min_bound > sup_[i][j].max_bound) {
          std::swap(sup_[i][j].min_bound, sup_[i][j].max_bound);
        }
        sup_[i][j].max_bound = fabs(sup_[i][j].max_bound);
        sup_[i][j].min_bound = -fabs(sup_[i][j].min_bound);
      } else {
        // "turn off" weight if max<min
        if (sup_[i][j].min_bound > sup_[i][j].max_bound) {
          T m = sup_[i][j].max_bound + (sup_[i][j].min_bound - sup_[i][j].max_bound) / ((T)2.0);
          sup_[i][j].max_bound = m;
          sup_[i][j].min_bound = m;
        }
      }

      // corrupt devices
      if (par.corrupt_devices_prob > rng->sampleUniform()) {
        // stuck somewhere in min_max
        T mn =
            MAX(MIN(sup_[i][j].max_bound, sup_[i][j].min_bound), -fabs(par.corrupt_devices_range));
        T mx =
            MIN(MAX(sup_[i][j].max_bound, sup_[i][j].min_bound), fabs(par.corrupt_devices_range));

        T value = mn + (mx - mn) * rng->sampleUniform();
        sup_[i][j].max_bound = value;
        sup_[i][j].min_bound = value;
        sup_[i][j].scale_up = (T)0.0;
        sup_[i][j].scale_down = (T)0.0;
      }

      // perfect bias
      if ((par.perfect_bias) && (j == this->x_size_ - 1)) {
        sup_[i][j].scale_up = par.dw_min;
        sup_[i][j].scale_down = par.dw_min;
        sup_[i][j].min_bound = 100 * par.w_min; // essentially no bound
        sup_[i][j].max_bound = 100 * par.w_max; // essentially no bound
      }

      //--------------------
      // diffusion
      par.diffusion = MAX(par.diffusion, (T)0.0);
      {
        T t = fabs(par.diffusion * (1 + par.diffusion_dtod * rng->sampleGauss()));
        sup_[i][j].diffusion_rate = t;
      }

      //--------------------
      // reset
      par.reset_dtod = MAX(par.reset_dtod, (T)0.0);
      par.reset_std = MAX(par.reset_std, (T)0.0);
      {
        T t = par.reset * ((T)1.0 + par.reset_dtod * rng->sampleGauss());
        sup_[i][j].reset_bias = t;
      }

      //--------------------
      // decay
      par.lifetime = MAX(par.lifetime, (T)0.0);
      {
        T t = par.lifetime * ((T)1.0 + par.lifetime_dtod * rng->sampleGauss());
        sup_[i][j].decay_scale = t > 1.0 ? (T)((T)1. - ((T)1. / t)) : (T)0.0;
      }

      //--------------------
      // copy for for linear access
      w_max_bound_[i][j] = sup_[i][j].max_bound;
      w_min_bound_[i][j] = sup_[i][j].min_bound;
      w_scale_up_[i][j] = sup_[i][j].scale_up;
      w_scale_down_[i][j] = sup_[i][j].scale_down;

      w_decay_scale_[i][j] = sup_[i][j].decay_scale;
      w_diffusion_rate_[i][j] = sup_[i][j].diffusion_rate;

      w_reset_bias_[i][j] = sup_[i][j].reset_bias;
    }
  }
}

template <typename T> void PulsedRPUDevice<T>::printDP(int x_count, int d_count) const {
  int x_count1 = x_count;
  int d_count1 = d_count;
  if (x_count < 0 || x_count > this->x_size_)
    x_count1 = this->x_size_;

  if (d_count < 0 || d_count > this->d_size_)
    d_count1 = this->d_size_;

  for (int i = 0; i < d_count1; ++i) {
    for (int j = 0; j < x_count1; ++j) {
      std::cout << "[<" << sup_[i][j].max_bound << ",";
      std::cout << sup_[i][j].min_bound << ">,<";
      std::cout << sup_[i][j].scale_up << ",";
      std::cout << sup_[i][j].scale_down << ">]";
      std::cout.precision(10);
      std::cout << sup_[i][j].decay_scale;
      std::cout.precision(6);
    }
    std::cout << std::endl;
  }
}

template class PulsedRPUDevice<float>;
#ifdef RPU_USE_DOUBLE
template class PulsedRPUDevice<double>;
#endif

} // namespace RPU
