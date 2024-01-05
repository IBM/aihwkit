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

#pragma once

#include "rpu_transfer_device.h"
#include <sstream>
#include <stdio.h>

namespace RPU {

template <typename T> class BufferedTransferRPUDevice;

/* Defines the buffered transfer device.

 */

template <typename T>
struct BufferedTransferRPUDeviceMetaParameter : TransferRPUDeviceMetaParameter<T> {

  T thres_scale = (T)1.0;
  // threshold for buffer to determine whether to
  // transfer to next device. NOTE: Will be multiplied
  // with dw_min

  T step = (T)1.0; // step size to transfer to if buffer is above
                   // thres. Note, however, the step size is equal to
                   // dw_min by default (for anything step>dw_min)
                   // because of the below default update setting

  T momentum = (T)0.0; // momentum on H. after transfer the momentum fraction stays on H

  bool forget_buffer = true; // whether to forget buffer after pulse
                             // (rather than subtracting transferred
                             // fraction only)

  BufferedTransferRPUDeviceMetaParameter() : TransferRPUDeviceMetaParameter<T>() {
    initDefaults();
  };
  BufferedTransferRPUDeviceMetaParameter(
      const PulsedRPUDeviceMetaParameterBase<T> &dp, int n_devices)
      : TransferRPUDeviceMetaParameter<T>(dp, n_devices) {
    initDefaults();
  };
  BufferedTransferRPUDeviceMetaParameter(
      const PulsedRPUDeviceMetaParameterBase<T> &dp_fast,
      const PulsedRPUDeviceMetaParameterBase<T> &dp_rest,
      int n_total_devices)
      : TransferRPUDeviceMetaParameter<T>(dp_fast, dp_rest, n_total_devices) {
    initDefaults();
  };

  void initDefaults() {

    this->transfer_lr = (T)1.0;

    this->transfer_up.update_bl_management = false;
    this->transfer_up.update_management = false;
    this->transfer_up.desired_BL = 1; // to set exactly one pulse if x*d!=0
    this->transfer_up.fixed_BL = true;
    this->scale_transfer_lr = true; // transfer should scale!

    this->update_policy = VectorDeviceUpdatePolicy::SingleFixed;
    this->first_update_idx = 0; // only first is updated
    this->same_context = true;

    this->with_reset_prob = 0.0;
    this->random_selection = false;
    this->no_self_transfer = true;
  }

  std::string getName() const override {
    std::ostringstream ss;
    ss << "BufferedTransfer(" << this->vec_par.size() << ")";
    if (this->vec_par.size() > 1) {
      ss << ": " << this->vec_par[0]->getName() << " -> " << this->vec_par[1]->getName();
      ;
    }
    return ss.str();
  };

  BufferedTransferRPUDevice<T> *
  createDevice(int x_size, int d_size, RealWorldRNG<T> *rng) override {
    return new BufferedTransferRPUDevice<T>(x_size, d_size, *this, rng);
  };

  BufferedTransferRPUDeviceMetaParameter<T> *clone() const override {
    return new BufferedTransferRPUDeviceMetaParameter<T>(*this);
  };
  DeviceUpdateType implements() const override { return DeviceUpdateType::BufferedTransfer; };
  void printToStream(std::stringstream &ss) const override;

  T getTransferLR(int to_device_idx, int from_device_idx, T current_lr) const override;
};

template <typename T> class BufferedTransferRPUDevice : public TransferRPUDevice<T> {

public:
  // constructor / destructor
  BufferedTransferRPUDevice(){};
  BufferedTransferRPUDevice(int x_size, int d_size);
  BufferedTransferRPUDevice(
      int x_size,
      int d_size,
      const BufferedTransferRPUDeviceMetaParameter<T> &par,
      RealWorldRNG<T> *rng);
  ~BufferedTransferRPUDevice();

  BufferedTransferRPUDevice(const BufferedTransferRPUDevice<T> &);
  BufferedTransferRPUDevice<T> &operator=(const BufferedTransferRPUDevice<T> &);
  BufferedTransferRPUDevice(BufferedTransferRPUDevice<T> &&) noexcept;
  BufferedTransferRPUDevice<T> &operator=(BufferedTransferRPUDevice<T> &&) noexcept;

  friend void swap(BufferedTransferRPUDevice<T> &a, BufferedTransferRPUDevice<T> &b) noexcept {
    using std::swap;
    swap(static_cast<TransferRPUDevice<T> &>(a), static_cast<TransferRPUDevice<T> &>(b));

    swap(a.transfer_buffer_vec_, b.transfer_buffer_vec_);
  }

  BufferedTransferRPUDeviceMetaParameter<T> &getPar() const override {
    return static_cast<BufferedTransferRPUDeviceMetaParameter<T> &>(SimpleRPUDevice<T>::getPar());
  };

  BufferedTransferRPUDevice<T> *clone() const override {
    return new BufferedTransferRPUDevice<T>(*this);
  };

  void getDPNames(std::vector<std::string> &names) const override;
  void getDeviceParameter(T **weights, std::vector<T *> &data_ptrs) override;
  void setDeviceParameter(T **out_weights, const std::vector<T *> &data_ptrs) override;
  int getHiddenWeightsCount() const override;
  void setHiddenWeights(const std::vector<T> &data) override;

  void readAndUpdate(
      int to_device_idx,
      int from_device_idx,
      const T lr,
      const T *vec,
      const int n_vec,
      const T reset_prob,
      const int i_col,
      const int m_batch_info) override;

  std::vector<std::vector<T>> getTransferBuffers() const { return transfer_buffer_vec_; };

protected:
  void populate(const BufferedTransferRPUDeviceMetaParameter<T> &par, RealWorldRNG<T> *rng);
  std::vector<std::vector<T>> transfer_buffer_vec_;
};

} // namespace RPU
