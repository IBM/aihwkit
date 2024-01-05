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

#include "io_manager.h"
#include "pulsed_weight_updater.h"
#include "rpu_buffered_transfer_device.h"
#include "rpucuda_transfer_device.h"

namespace RPU {

template <typename T> class BufferedTransferRPUDeviceCuda : public TransferRPUDeviceCuda<T> {

public:
  explicit BufferedTransferRPUDeviceCuda(){};
  explicit BufferedTransferRPUDeviceCuda(
      CudaContextPtr c, const BufferedTransferRPUDevice<T> &other);

  ~BufferedTransferRPUDeviceCuda(){};
  BufferedTransferRPUDeviceCuda(const BufferedTransferRPUDeviceCuda<T> &other);
  BufferedTransferRPUDeviceCuda<T> &operator=(const BufferedTransferRPUDeviceCuda<T> &other);
  BufferedTransferRPUDeviceCuda(BufferedTransferRPUDeviceCuda<T> &&other);
  BufferedTransferRPUDeviceCuda<T> &operator=(BufferedTransferRPUDeviceCuda<T> &&other);

  friend void
  swap(BufferedTransferRPUDeviceCuda<T> &a, BufferedTransferRPUDeviceCuda<T> &b) noexcept {
    using std::swap;
    swap(static_cast<TransferRPUDeviceCuda<T> &>(a), static_cast<TransferRPUDeviceCuda<T> &>(b));
    swap(a.transfer_buffer_vec_, b.transfer_buffer_vec_);
  };

  void populateFrom(const AbstractRPUDevice<T> &rpu_device) override;
  BufferedTransferRPUDeviceMetaParameter<T> &getPar() const {
    return static_cast<BufferedTransferRPUDeviceMetaParameter<T> &>(
        SimpleRPUDeviceCuda<T>::getPar());
  };
  BufferedTransferRPUDeviceCuda<T> *clone() const override {
    return new BufferedTransferRPUDeviceCuda<T>(*this);
  };

  void readAndUpdate(
      int to_device_idx,
      int from_device_idx,
      int i_col_start,
      const T lr,
      const T count_lr,
      const T *x_input,
      const int n_vec,
      const PulsedUpdateMetaParameter<T> &up) override;

  std::vector<T> getHiddenWeights() const override;

protected:
  std::vector<std::unique_ptr<CudaArray<T>>> transfer_buffer_vec_;
};

} // namespace RPU
