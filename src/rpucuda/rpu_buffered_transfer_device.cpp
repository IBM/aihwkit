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

  // thres
  ss << "\t thres_scale:\t\t";
  ss << thres_scale;
  ss << std::endl;

  if (step != (T)1.0) {
    ss << "\t step:\t\t\t";
    ss << step;
    ss << std::endl;
  }
  if (momentum != (T)0.0) {
    ss << "\t momentum:\t\t";
    ss << momentum;
    ss << std::endl;
  }

  if (forget_buffer) {
    ss << "\t forget_buffer:\t\t";
    ss << std::boolalpha << forget_buffer;
    ss << std::endl;
  }
  TransferRPUDeviceMetaParameter<T>::printToStream(ss);
};

template <typename T>
T BufferedTransferRPUDeviceMetaParameter<T>::getTransferLR(
    int to_device_idx, int from_device_idx, T current_lr) const {

  T lr_gamma =
      TransferRPUDeviceMetaParameter<T>::getTransferLR(to_device_idx, from_device_idx, current_lr);

  if (this->gamma_vec[to_device_idx] > (T)0.0 && this->gamma_vec[from_device_idx] > (T)0.0) {
    lr_gamma *= this->gamma_vec[from_device_idx] / this->gamma_vec[to_device_idx];
  }
  return lr_gamma;
}

template struct BufferedTransferRPUDeviceMetaParameter<float>;
#ifdef RPU_USE_DOUBLE
template struct BufferedTransferRPUDeviceMetaParameter<double>;
#endif
#ifdef RPU_USE_FP16
template struct BufferedTransferRPUDeviceMetaParameter<half_t>;
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
BufferedTransferRPUDevice<T>::BufferedTransferRPUDevice(
    BufferedTransferRPUDevice<T> &&other) noexcept {
  *this = std::move(other);
}

// move assignment
template <typename T>
BufferedTransferRPUDevice<T> &
BufferedTransferRPUDevice<T>::operator=(BufferedTransferRPUDevice<T> &&other) noexcept {
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

  transfer_buffer_vec_.resize((size_t)this->n_devices_ - 1);

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
    const int i_slice_start,
    const int m_batch_info) {

  UNUSED(reset_prob_in);
  UNUSED(m_batch_info);

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
  T lr_abs = (T)fabsf(lr);
  T *v_out = this->transfer_tmp_.data();
  bool forget_buffer = par.forget_buffer;
  T max_steps = (T)this->transfer_pwu_->getUpPar().desired_BL;

  // buffer weight is x_size major, we need to write out_size
  bool use_cols = par.transfer_columns;
  int w_inc = use_cols ? in_size : 1;

  // forward / update
  for (size_t i = 0; i < (size_t)n_vec; i++) {

    const T *v_in = vec + (size_t)i * in_size;

    // first read from previous device
    this->readVector(from_device_idx, v_in, v_out, 1.0);

    // add into to FP buffer
    T *fp_w = transfer_buffer_vec_[from_device_idx].data();
    int i_w = use_cols ? i_slice_start + (int)i : this->x_size_ * (i_slice_start + (int)i);

    int non_zero_count = 0;
    PRAGMA_SIMD
    for (size_t j = 0; j < (size_t)out_size; j++) {
      T omega = fp_w[i_w];
      omega += v_out[j] * lr_abs;

      T n_steps = MAX(MIN((T)truncf(omega / buffer_granularity), max_steps), -max_steps);

      if (forget_buffer) {
        fp_w[i_w] = (n_steps != (T)0.0) ? omega * par.momentum : omega;
      } else {
        fp_w[i_w] =
            (n_steps != (T)0.0) ? omega - sub_momentum * n_steps * buffer_granularity : omega;
      }

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
void BufferedTransferRPUDevice<T>::getDeviceParameter(T **weights, std::vector<T *> &data_ptrs) {

  std::vector<std::string> names;
  getDPNames(names);

  if (data_ptrs.size() < names.size()) {
    RPU_FATAL("Expected " << names.size() << " data pointers!");
  }

  TransferRPUDevice<T>::getDeviceParameter(weights, data_ptrs);

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
  size_t size = this->size_;
  size_t add_n = (size_t)this->n_devices_ - 1;
  size_t offset = (getHiddenWeightsCount() - add_n) * size;

  for (size_t k = 0; k < add_n; k++) {

    if (data.size() < (size_t)offset + size) {
      RPU_FATAL("Size mismatch for hidden weights.");
    }

    for (size_t i = 0; i < size; i++) {
      transfer_buffer_vec_[k][i] = data[offset + i];
    }
    offset += size;
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
#ifdef RPU_USE_FP16
template class BufferedTransferRPUDevice<half_t>;
#endif

} // namespace RPU
