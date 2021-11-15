/**
 * (C) Copyright 2020, 2021 IBM. All Rights Reserved.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
 */

#include "rpu_buffered_transfer_device.h"
#include "math_util.h"
#include "utility_functions.h"
#include <algorithm>
#include <memory>
#include <sstream>

namespace RPU {

/******************************************************************************************/
/* BufferedTransferRPUDeviceMetaParameter*/

template <typename T>
void BufferedTransferRPUDeviceMetaParameter<T>::printToStream(std::stringstream &ss) const {

  TransferRPUDeviceMetaParameter<T>::printToStream(ss);

  ss << "--- Parameters special to BufferedTransfer --- " << std::endl;
  // thres
  ss << "\tthres_scale:\t\t";
  ss << thres_scale;
  ss << std::endl;

  ss << "\tstep:\t\t\t";
  ss << step;
  ss << std::endl;

  ss << "\tmomentum:\t\t";
  ss << momentum;
  ss << std::endl;
};

template <typename T>
T BufferedTransferRPUDeviceMetaParameter<T>::getTransferLR(
    int to_device_idx, int from_device_idx, T current_lr) const {

  T lr_gamma =
      TransferRPUDeviceMetaParameter<T>::getTransferLR(to_device_idx, from_device_idx, current_lr);

  if (this->gamma_vec[to_device_idx] > 0 && this->gamma_vec[from_device_idx] > 0) {
    lr_gamma *= this->gamma_vec[from_device_idx] / this->gamma_vec[to_device_idx];
  }
  return lr_gamma;
}

template struct BufferedTransferRPUDeviceMetaParameter<float>;
#ifdef RPU_USE_DOUBLE
template struct BufferedTransferRPUDeviceMetaParameter<double>;
#endif

/******************************************************************************************/
// dtor
template <typename T> BufferedTransferRPUDevice<T>::~BufferedTransferRPUDevice() {}

// ctor
template <typename T>
BufferedTransferRPUDevice<T>::BufferedTransferRPUDevice(int x_sz, int d_sz)
    : TransferRPUDevice<T>(x_sz, d_sz) {}

template <typename T>
BufferedTransferRPUDevice<T>::BufferedTransferRPUDevice(
    int x_sz, int d_sz, const BufferedTransferRPUDeviceMetaParameter<T> &par, RealWorldRNG<T> *rng)
    : BufferedTransferRPUDevice<T>(x_sz, d_sz) {
  populate(par, rng);
}

// copy constructor
template <typename T>
BufferedTransferRPUDevice<T>::BufferedTransferRPUDevice(const BufferedTransferRPUDevice<T> &other)
    : TransferRPUDevice<T>(other) {
  transfer_buffer_vec_ = other.transfer_buffer_vec_;
}

// copy assignment
template <typename T>
BufferedTransferRPUDevice<T> &
BufferedTransferRPUDevice<T>::operator=(const BufferedTransferRPUDevice<T> &other) {

  BufferedTransferRPUDevice<T> tmp(other);
  swap(*this, tmp);
  return *this;
}

// move constructor
template <typename T>
BufferedTransferRPUDevice<T>::BufferedTransferRPUDevice(BufferedTransferRPUDevice<T> &&other) {
  *this = std::move(other);
}

// move assignment
template <typename T>
BufferedTransferRPUDevice<T> &
BufferedTransferRPUDevice<T>::operator=(BufferedTransferRPUDevice<T> &&other) {
  TransferRPUDevice<T>::operator=(std::move(other));
  transfer_buffer_vec_ = std::move(transfer_buffer_vec_);

  return *this;
}

/*********************************************************************************/
/* populate */

template <typename T>
void BufferedTransferRPUDevice<T>::populate(
    const BufferedTransferRPUDeviceMetaParameter<T> &p, RealWorldRNG<T> *rng) {

  TransferRPUDevice<T>::populate(p, rng);

  transfer_buffer_vec_.resize(this->n_devices_ - 1);

  // init buffers
  for (int k = 0; k < this->n_devices_ - 1; k++) {
    transfer_buffer_vec_[k].resize(this->size_);
  }
}

/*********************************************************************************/
/* transfer */
template <typename T>
void BufferedTransferRPUDevice<T>::readAndUpdate(
    int to_device_idx,
    int from_device_idx,
    const T lr,
    const T *vec,
    const int n_vec,
    const T reset_prob_in,
    const int i_slice_start) {
  if (lr == (T)0.0) {
    return;
  }

  if (to_device_idx == from_device_idx || from_device_idx >= this->n_devices_ - 1) {
    // self update and transfer from last device not supported
    return;
  }

  this->transfer_tmp_.resize(this->d_size_);
  const auto &par = getPar();
  int in_size = par.getInSize();
  int out_size = par.getOutSize();

  T weight_granularity = this->rpu_device_vec_[to_device_idx]->getWeightGranularity();
  T buffer_granularity = par.thres_scale * weight_granularity;
  T sub_momentum = (T)1.0 - MAX(MIN(par.momentum, (T)1.0), (T)0.0);
  T step = par.step;
  T lr_abs = fabs(lr);
  T *v_out = this->transfer_tmp_.data();

  int max_steps = this->transfer_pwu_->getUpPar().desired_BL;

  // buffer weight is x_size major, we need to write out_size
  bool use_cols = par.transfer_columns;
  int w_inc = use_cols ? in_size : 1;

  // forward / update
  for (int i = 0; i < n_vec; i++) {

    const T *v_in = vec + i * in_size;

    // first read into d (will overwrite d)
    this->readVector(from_device_idx, v_in, v_out, 1.0);

    // add into to FP buffer
    T *fp_w = transfer_buffer_vec_[from_device_idx].data();
    int i_w = use_cols ? i_slice_start + i : this->x_size_ * (i_slice_start + i);

    int non_zero_count = 0;

    PRAGMA_SIMD
    for (int j = 0; j < out_size; j++) {
      T omega = fp_w[i_w];
      omega += v_out[j] * lr_abs; // multiplied with transfer LR

      T n_steps = MAX(MIN(truncf(omega / buffer_granularity), max_steps), -max_steps);
      fp_w[i_w] = omega - sub_momentum * n_steps * buffer_granularity;

      non_zero_count += ((int)n_steps) != 0;

      v_out[j] = -n_steps; // since positive update needed below
      i_w += w_inc;
    }

    if (non_zero_count > 0) {
      this->writeVector(to_device_idx, v_in, v_out, step * weight_granularity, 1);
    }
  }
}

template <typename T>
void BufferedTransferRPUDevice<T>::getDPNames(std::vector<std::string> &names) const {

  TransferRPUDevice<T>::getDPNames(names);

  for (int k = 0; k < this->n_devices_ - 1; k++) {
    std::ostringstream ss;
    ss << "buffered_FP_weight_" << k;
    names.push_back(ss.str());
  }
}

template <typename T>
void BufferedTransferRPUDevice<T>::getDeviceParameter(std::vector<T *> &data_ptrs) const {

  std::vector<std::string> names;
  getDPNames(names);

  if (data_ptrs.size() < names.size()) {
    RPU_FATAL("Expected " << names.size() << " data pointers!");
  }

  TransferRPUDevice<T>::getDeviceParameter(data_ptrs);

  int add_n = this->n_devices_ - 1;
  size_t m = names.size() - add_n;
  for (int k = 0; k < add_n; k++) {

    // "hidden weights"
    for (int i = 0; i < this->size_; ++i) {
      data_ptrs[m][i] = transfer_buffer_vec_[k][i];
    }
    m++;
  }
};

template <typename T> int BufferedTransferRPUDevice<T>::getHiddenWeightsCount() const {

  if (!this->n_devices_) {
    return 0;
  }
  int m = TransferRPUDevice<T>::getHiddenWeightsCount();
  return m + this->n_devices_ - 1;
}

template <typename T>
void BufferedTransferRPUDevice<T>::setHiddenWeights(const std::vector<T> &data) {
  /* hidden weights are expected in the usual row-major format (first x_size then d_size)*/

  if (!this->n_devices_) {
    return;
  }

  TransferRPUDevice<T>::setHiddenWeights(data);

  // lastly,  set the FP buffers
  int add_n = this->n_devices_ - 1;
  int offset = (getHiddenWeightsCount() - add_n) * this->size_;
  for (int k = 0; k < add_n; k++) {

    if (data.size() < (size_t)offset + this->size_) {
      RPU_FATAL("Size mismatch for hidden weights.");
    }

    for (int i = 0; i < this->size_; i++) {
      transfer_buffer_vec_[k][i] = data[offset + i];
    }
    offset += this->size_;
  }
}

template <typename T>
void BufferedTransferRPUDevice<T>::setDeviceParameter(
    T **out_weights, const std::vector<T *> &data_ptrs) {

  std::vector<std::string> names;
  getDPNames(names);

  if (data_ptrs.size() < names.size()) {
    RPU_FATAL("Expected " << names.size() << " data pointers!");
  }

  TransferRPUDevice<T>::setDeviceParameter(out_weights, data_ptrs);

  int add_n = this->n_devices_ - 1;
  size_t m = names.size() - add_n;

  // lastly,  set the FP buffers
  for (int k = 0; k < add_n; k++) {

    for (int i = 0; i < this->size_; i++) {
      transfer_buffer_vec_[k][i] = data_ptrs[m + k][i];
    }
  }
};

template class BufferedTransferRPUDevice<float>;
#ifdef RPU_USE_DOUBLE
template class BufferedTransferRPUDevice<double>;
#endif

} // namespace RPU
