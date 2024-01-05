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

#include "rpu_dynamic_transfer_device.h"
#include "rpucuda_chopped_transfer_device.h"

namespace RPU {

template <typename T> class DynamicTransferRPUDeviceCuda : public ChoppedTransferRPUDeviceCuda<T> {

public:
  explicit DynamicTransferRPUDeviceCuda(){};
  explicit DynamicTransferRPUDeviceCuda(CudaContextPtr c, const DynamicTransferRPUDevice<T> &other);

  ~DynamicTransferRPUDeviceCuda(){};
  DynamicTransferRPUDeviceCuda(const DynamicTransferRPUDeviceCuda<T> &other);
  DynamicTransferRPUDeviceCuda<T> &operator=(const DynamicTransferRPUDeviceCuda<T> &other);
  DynamicTransferRPUDeviceCuda(DynamicTransferRPUDeviceCuda<T> &&other);
  DynamicTransferRPUDeviceCuda<T> &operator=(DynamicTransferRPUDeviceCuda<T> &&other);

  friend void
  swap(DynamicTransferRPUDeviceCuda<T> &a, DynamicTransferRPUDeviceCuda<T> &b) noexcept {
    using std::swap;
    swap(
        static_cast<ChoppedTransferRPUDeviceCuda<T> &>(a),
        static_cast<ChoppedTransferRPUDeviceCuda<T> &>(b));

    swap(a.dev_running_mean_, b.dev_running_mean_);
    swap(a.dev_past_mean_, b.dev_past_mean_);
    swap(a.dev_feedback_, b.dev_feedback_);

    swap(a.feedback_data_, b.feedback_data_);
    swap(a.feedback_data_idx_, b.feedback_data_idx_);
    swap(a.count_lr_scale_, b.count_lr_scale_);

    swap(a.dev_transfers_since_in_chop_, b.dev_transfers_since_in_chop_);
    swap(a.dev_transfers_since_in_chop_tmp_, b.dev_transfers_since_in_chop_tmp_);
    swap(a.dev_previous_in_chopper_, b.dev_previous_in_chopper_);
    swap(a.dev_previous_in_chopper_tmp_, b.dev_previous_in_chopper_tmp_);
  };

  void populateFrom(const AbstractRPUDevice<T> &rpu_device) override;
  void dumpExtra(RPU::state_t &extra, const std::string prefix) override;
  void loadExtra(const RPU::state_t &extra, const std::string prefix, bool strict) override;

  void readAndUpdate(
      int to_device_idx,
      int from_device_idx,
      int i_slice_start,
      const T lr,
      const T count_lr,
      const T *x_input,
      const int n_vec,
      const PulsedUpdateMetaParameter<T> &up) override;

  DynamicTransferRPUDeviceMetaParameter<T> &getPar() const {
    return static_cast<DynamicTransferRPUDeviceMetaParameter<T> &>(
        SimpleRPUDeviceCuda<T>::getPar());
  };
  DynamicTransferRPUDeviceCuda<T> *clone() const override {
    return new DynamicTransferRPUDeviceCuda<T>(*this);
  };

  std::vector<T> getHiddenWeights() const override;

protected:
  T getPulseCountLearningRate(
      T lr, int current_m_batch, const PulsedUpdateMetaParameter<T> &up) override;

private:
  std::vector<T> feedback_data_;
  T count_lr_scale_ = 1.0;
  uint64_t feedback_data_idx_ = 0;

  std::unique_ptr<CudaArray<T>> dev_past_mean_ = nullptr;
  std::unique_ptr<CudaArray<T>> dev_running_mean_ = nullptr;
  std::unique_ptr<CudaArray<T>> dev_feedback_ = nullptr;
  std::unique_ptr<CudaArray<int>> dev_transfers_since_in_chop_ = nullptr;
  std::unique_ptr<CudaArray<int>> dev_transfers_since_in_chop_tmp_ = nullptr;
  std::unique_ptr<CudaArray<chop_t>> dev_previous_in_chopper_ = nullptr;
  std::unique_ptr<CudaArray<chop_t>> dev_previous_in_chopper_tmp_ = nullptr;
};

} // namespace RPU
