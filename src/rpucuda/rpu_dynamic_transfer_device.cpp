/**
 * (C) Copyright 2020, 2021, 2022, 2023 IBM. All Rights Reserved.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
 */

#include "rpu_dynamic_transfer_device.h"
#include "math_util.h"
#include "utility_functions.h"
#include <algorithm>
#include <memory>
#include <sstream>

namespace RPU {

/******************************************************************************************/
/* DynamicTransferRPUDeviceMetaParameter*/

template <typename T>
void DynamicTransferRPUDeviceMetaParameter<T>::printToStream(std::stringstream &ss) const {

  ss << "\t scale_thres_with_samples:\t";
  ss << std::boolalpha << scale_thres_with_samples;
  ss << std::endl;

  ss << "\t tail_weightening:\t";
  ss << tail_weightening;
  ss << std::endl;

  ss << "\t buffer_cap:\t";
  ss << buffer_cap;
  ss << std::endl;

  ss << "\t always_write:\t";
  ss << always_write;
  ss << std::endl;

  ChoppedTransferRPUDeviceMetaParameter<T>::printToStream(ss);
};

template <typename T> void DynamicTransferRPUDeviceMetaParameter<T>::checkSupported() const {

  ChoppedTransferRPUDeviceMetaParameter<T>::checkSupported();

  if (this->in_chop_prob <= (T)0.0) {
    RPU_FATAL("In chopper prob needs to be positive.");
  }
  if (this->auto_momentum >= (T)1.0 || this->auto_momentum <= (T)0.0) {
    RPU_FATAL("Auto-momentum needs to be smaller than 1 and larger than 0.");
  }
}

template <typename T>
unsigned int
DynamicTransferRPUDeviceMetaParameter<T>::getNumInChopSamples(T current_in_chop_prob) const {

  T in_chop_freq = current_in_chop_prob;
  unsigned int n_samples = 2;
  if (in_chop_freq > 0) {
    n_samples = MAX((int)ceil((T)1.0 / in_chop_freq), 2);
  }
  return n_samples;
}

template struct DynamicTransferRPUDeviceMetaParameter<float>;
#ifdef RPU_USE_DOUBLE
template struct DynamicTransferRPUDeviceMetaParameter<double>;
#endif

/******************************************************************************************/
// dtor
template <typename T> DynamicTransferRPUDevice<T>::~DynamicTransferRPUDevice() {}

// ctor
template <typename T>
DynamicTransferRPUDevice<T>::DynamicTransferRPUDevice(int x_sz, int d_sz)
    : ChoppedTransferRPUDevice<T>(x_sz, d_sz) {}

template <typename T>
DynamicTransferRPUDevice<T>::DynamicTransferRPUDevice(
    int x_sz, int d_sz, const DynamicTransferRPUDeviceMetaParameter<T> &par, RealWorldRNG<T> *rng)
    : DynamicTransferRPUDevice<T>(x_sz, d_sz) {
  populate(par, rng);
}

// copy constructor
template <typename T>
DynamicTransferRPUDevice<T>::DynamicTransferRPUDevice(const DynamicTransferRPUDevice<T> &other)
    : ChoppedTransferRPUDevice<T>(other) {
  running_mean_ = other.running_mean_;
  running_var_ = other.running_var_;
  past_mean_ = other.past_mean_;
  in_chopper_switched_ = other.in_chopper_switched_;
  current_in_chop_prob_ = other.current_in_chop_prob_;
}

// copy assignment
template <typename T>
DynamicTransferRPUDevice<T> &
DynamicTransferRPUDevice<T>::operator=(const DynamicTransferRPUDevice<T> &other) {

  DynamicTransferRPUDevice<T> tmp(other);
  swap(*this, tmp);
  return *this;
}

// move constructor
template <typename T>
DynamicTransferRPUDevice<T>::DynamicTransferRPUDevice(
    DynamicTransferRPUDevice<T> &&other) noexcept {
  *this = std::move(other);
}

// move assignment
template <typename T>
DynamicTransferRPUDevice<T> &
DynamicTransferRPUDevice<T>::operator=(DynamicTransferRPUDevice<T> &&other) noexcept {
  ChoppedTransferRPUDevice<T>::operator=(std::move(other));
  running_mean_ = std::move(other.running_mean_);
  running_var_ = std::move(other.running_var_);
  past_mean_ = std::move(other.past_mean_);
  in_chopper_switched_ = std::move(other.in_chopper_switched_);
  current_in_chop_prob_ = other.current_in_chop_prob_;

  return *this;
}

/*********************************************************************************/
/* populate */

template <typename T>
void DynamicTransferRPUDevice<T>::populate(
    const DynamicTransferRPUDeviceMetaParameter<T> &p, RealWorldRNG<T> *rng) {

  ChoppedTransferRPUDevice<T>::populate(p, rng);
  const auto &par = getPar();

  running_mean_.resize((size_t)this->size_, 0);
  running_var_.resize((size_t)this->size_, this->rpu_device_vec_[0]->getWeightGranularity());
  past_mean_.resize((size_t)this->size_, 0);
  in_chopper_switched_.resize((size_t)par.getInSize(), false);
  current_in_chop_prob_ = par.in_chop_prob;
}

/*********************************************************************************/
/* transfer */
template <typename T>
void DynamicTransferRPUDevice<T>::readAndUpdate(
    int to_device_idx,
    int from_device_idx,
    const T lr,
    const T *vec,
    const int n_vec,
    const T reset_prob_in,
    const int i_slice_start) {

  UNUSED(reset_prob_in);

  if (lr == (T)0.0) {
    return;
  }
  // we assume that n_vec is always one (n_reads_per_transfer==1) ensured by Chopped
  if (n_vec != 1) {
    RPU_FATAL("Only single row transfer supported.");
  }

  if (to_device_idx != 1 && from_device_idx != 1) {
    RPU_FATAL("Only 2 devices supported");
  }
  const int FROM_DEVICE_IDX = 0;
  const int TO_DEVICE_IDX = 1;

  const auto &par = getPar();
  int in_size = par.getInSize();
  int out_size = par.getOutSize();
  const T weight_granularity = this->rpu_device_vec_[TO_DEVICE_IDX]->getWeightGranularity();
  const T from_weight_granularity = this->rpu_device_vec_[FROM_DEVICE_IDX]->getWeightGranularity();
  const T lr_abs = fabs(lr);

  // buffer weight is x_size major, we need to write out_size
  const bool use_cols = par.transfer_columns;
  const int w_inc = use_cols ? in_size : 1;
  const int i_w_start = use_cols ? i_slice_start : this->x_size_ * (i_slice_start);

  T *mean_w = running_mean_.data();
  T *var_w = running_var_.data();

  // only n_vec=1 supported and sequential update supported
  const T *v_in = vec;

  // first read from previous device
  this->transfer_tmp_.resize(out_size);
  T *v_out = this->transfer_tmp_.data();
  this->readVector(FROM_DEVICE_IDX, v_in, v_out, 1.0);

  // update running mean and std and do the actual write
  const unsigned int n_samples = par.getNumInChopSamples(lr_abs);
  const T current_sample = (T)((this->transfer_counter_ % (uint64_t)n_samples) + 1);
  const T sample_momentum = (T)MIN(par.tail_weightening / current_sample, (T)1.0);
  const bool previously_switched = in_chopper_switched_[i_slice_start];
  in_chopper_switched_[i_slice_start] = false;

  const bool var_momentum = par.auto_momentum;
  const bool use_thres = par.thres_scale > 0;

  T *past_mean_w = past_mean_.data();
  T *fp_w = this->transfer_buffer_vec_[FROM_DEVICE_IDX].data();
  T thres = par.thres_scale;
  const bool always_write = par.always_write;
  if (par.scale_thres_with_samples && !always_write) {
    thres /= (T)sqrt(current_sample);
  }
  T lr_scale = par.getTransferLRScale(from_weight_granularity, lr_abs);
  const int max_steps = this->transfer_pwu_->getUpPar().desired_BL;
  T buffer_cap = max_steps * par.buffer_cap;
  bool previous_in_chop = this->in_chopper_[i_slice_start];
  if (previously_switched) {
    previous_in_chop = !previous_in_chop;
  }
  const bool forget_buffer = par.forget_buffer;
  unsigned int non_zero_count = 0;
  const T momentum = par.momentum;
  int i_w = i_w_start;
  T max_past_mean_w = (T)-1.0;

  PRAGMA_SIMD
  for (int j = 0; j < out_size; j++) {

    T val = v_out[j]; // val is NOT sign corrected (see below)

    T n_steps = 0.0;
    if (previously_switched || always_write) {
      T m_w = mean_w[i_w];
      T omega = fp_w[i_w];
      T mu_past = past_mean_w[i_w];

      T mu = (always_write && !previously_switched) ? val : m_w;
      T dw = (mu - mu_past);

      dw = (previous_in_chop != this->out_chopper_[j]) ? -dw : dw;

      if (use_thres) {

        T std = MAX((T)sqrt(fabs(var_w[i_w])), weight_granularity);

        if (fabs(dw) > thres * std) {
          omega += dw * lr_scale;
        }
      } else {
        omega += dw * lr_scale;
      }

      if (fabs(omega) >= (T)1.0) {
        n_steps = MAX(MIN(truncf(omega), max_steps), -max_steps);

        if (forget_buffer) {
          omega *= momentum;
        } else {
          omega -= ((T)1.0 - momentum) * n_steps;
          if (buffer_cap > (T)0.0) {
            omega = MIN(MAX(omega, -buffer_cap), buffer_cap);
          }
        }
        non_zero_count += 1;
      }

      fp_w[i_w] = omega;

      if (previously_switched) {
        // reset mean esimation, and begin new chopper phase
        max_past_mean_w = MAX(max_past_mean_w, fabs(mu_past - m_w));
        past_mean_w[i_w] = m_w;
        mean_w[i_w] = (T)0.0;
      }
    }
    v_out[j] = -n_steps; // write sign corrected

    T w = mean_w[i_w] * (1 - sample_momentum) + val * sample_momentum;

    if (use_thres) {
      // Computes variation from current running mean, which implicitely
      // will discount the initial deviations, which also mostly are state
      // changes and should not be counted

      T d_val = (val - w);
      var_w[i_w] = var_momentum * var_w[i_w] + ((T)1.0 - var_momentum) * d_val * d_val;
    }
    mean_w[i_w] = w;

    i_w += w_inc;
  }

  if ((non_zero_count > 0) && (this->transfer_counter_ > n_samples)) {
    T write_lr = par.getWriteLR(weight_granularity);
    this->writeVector(TO_DEVICE_IDX, v_in, v_out, write_lr, 1);
  }

  bool switch_choppers = current_sample == n_samples;
  if (par.in_chop_random) {
    if (this->rw_rng_.sampleUniform() < current_in_chop_prob_) {
      switch_choppers = true;
    }
  }

  if (switch_choppers) {
    // reset chopper for each.
    this->in_chopper_[i_slice_start] = !this->in_chopper_[i_slice_start];
    in_chopper_switched_[i_slice_start] = true;
  }

  if (FROM_DEVICE_IDX == 0 && i_slice_start == in_size - 1) {
    // only advance after full matrix transfer
    this->transfer_counter_++;
  }
}

/*********************************************************************************/

template <typename T>
void DynamicTransferRPUDevice<T>::getDPNames(std::vector<std::string> &names) const {

  TransferRPUDevice<T>::getDPNames(names);
  if (!this->n_devices_) {
    return;
  }
  if (this->n_devices_ != 2) {
    RPU_FATAL("Only 2 devices supported");
  }

  std::string s1 = "buffered_FP_weight";
  names.push_back(s1);

  std::string s2 = "running_mean_weight";
  names.push_back(s2);

  std::string s3 = "running_var_weight";
  names.push_back(s3);

  std::string s4 = "past_mean_weight";
  names.push_back(s4);
}

template <typename T>
void DynamicTransferRPUDevice<T>::getDeviceParameter(T **weights, std::vector<T *> &data_ptrs) {

  std::vector<std::string> names;
  getDPNames(names);

  if (data_ptrs.size() < names.size()) {
    RPU_FATAL("Expected " << names.size() << " data pointers!");
  }
  if (!this->n_devices_) {
    return;
  }
  if (this->n_devices_ != 2) {
    RPU_FATAL("Only 2 devices supported");
  }

  TransferRPUDevice<T>::getDeviceParameter(weights, data_ptrs);

  int add_n = 4;
  size_t m = names.size() - add_n;
  for (int i = 0; i < this->size_; ++i) {
    data_ptrs[m][i] = this->transfer_buffer_vec_[0][i];
    data_ptrs[m + 1][i] = running_mean_[i];
    data_ptrs[m + 2][i] = running_var_[i];
    data_ptrs[m + 3][i] = past_mean_[i];
  }
};

template <typename T> int DynamicTransferRPUDevice<T>::getHiddenWeightsCount() const {

  if (!this->n_devices_) {
    return 0;
  }
  if (this->n_devices_ != 2) {
    RPU_FATAL("Only 2 devices supported");
  }
  int add_n = 4;

  int m = TransferRPUDevice<T>::getHiddenWeightsCount();
  return m + add_n;
}

template <typename T>
void DynamicTransferRPUDevice<T>::setHiddenWeights(const std::vector<T> &data) {
  /* hidden weights are expected in the usual row-major format (first x_size then d_size)*/

  if (!this->n_devices_) {
    return;
  }
  if (this->n_devices_ != 2) {
    RPU_FATAL("Only 2 devices supported");
  }

  TransferRPUDevice<T>::setHiddenWeights(data);

  size_t add_n = 4;
  size_t offset = (getHiddenWeightsCount() - add_n) * this->size_;

  if (data.size() < (size_t)offset + add_n * this->size_) {
    RPU_FATAL("Size mismatch for hidden weights.");
  }
  size_t size = this->size_;

  for (size_t i = 0; i < size; i++) {
    this->transfer_buffer_vec_[0][i] = data[offset + i];
    running_mean_[i] = data[offset + i + size];
    running_var_[i] = data[offset + i + 2 * size];
    past_mean_[i] = data[offset + i + 3 * size];
  }
}

template <typename T>
void DynamicTransferRPUDevice<T>::setDeviceParameter(
    T **out_weights, const std::vector<T *> &data_ptrs) {

  std::vector<std::string> names;
  getDPNames(names);

  if (data_ptrs.size() < names.size()) {
    RPU_FATAL("Expected " << names.size() << " data pointers!");
  }

  TransferRPUDevice<T>::setDeviceParameter(out_weights, data_ptrs);

  int add_n = 4;
  size_t m = names.size() - add_n;

  for (int i = 0; i < this->size_; i++) {
    this->transfer_buffer_vec_[0][i] = data_ptrs[m][i];
    running_mean_[i] = data_ptrs[m + 1][i];
    running_var_[i] = data_ptrs[m + 2][i];
    past_mean_[i] = data_ptrs[m + 3][i];
  }
};

template <typename T>
void DynamicTransferRPUDevice<T>::dumpExtra(RPU::state_t &extra, const std::string prefix) {
  ChoppedTransferRPUDevice<T>::dumpExtra(extra, prefix);

  RPU::state_t state;

  RPU::insert(state, "current_in_chop_prob", current_in_chop_prob_);
  RPU::insert(state, "in_chopper_switched", in_chopper_switched_);

  // all other vars are handled with the getDeviceParameter
  RPU::insertWithPrefix(extra, state, prefix);
}

template <typename T>
void DynamicTransferRPUDevice<T>::loadExtra(
    const RPU::state_t &extra, const std::string prefix, bool strict) {
  ChoppedTransferRPUDevice<T>::loadExtra(extra, prefix, strict);

  auto state = RPU::selectWithPrefix(extra, prefix);

  RPU::load(state, "current_in_chop_prob", current_in_chop_prob_, strict);
  RPU::load(state, "in_chopper_switched", in_chopper_switched_, strict);
}

template class DynamicTransferRPUDevice<float>;
#ifdef RPU_USE_DOUBLE
template class DynamicTransferRPUDevice<double>;
#endif

} // namespace RPU
