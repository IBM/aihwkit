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

#include "chopped_weight_output.h"
#include "io_manager.h"
#include "pulsed_weight_updater.h"
#include "rpu_chopped_transfer_device.h"
#include "rpucuda_buffered_transfer_device.h"

namespace RPU {

template <typename T> class ChoppedTransferRPUDeviceCuda : public BufferedTransferRPUDeviceCuda<T> {

public:
  explicit ChoppedTransferRPUDeviceCuda(){};
  explicit ChoppedTransferRPUDeviceCuda(CudaContextPtr c, const ChoppedTransferRPUDevice<T> &other);

  ~ChoppedTransferRPUDeviceCuda(){};
  ChoppedTransferRPUDeviceCuda(const ChoppedTransferRPUDeviceCuda<T> &other);
  ChoppedTransferRPUDeviceCuda<T> &operator=(const ChoppedTransferRPUDeviceCuda<T> &other);
  ChoppedTransferRPUDeviceCuda(ChoppedTransferRPUDeviceCuda<T> &&other);
  ChoppedTransferRPUDeviceCuda<T> &operator=(ChoppedTransferRPUDeviceCuda<T> &&other);

  friend void
  swap(ChoppedTransferRPUDeviceCuda<T> &a, ChoppedTransferRPUDeviceCuda<T> &b) noexcept {
    using std::swap;
    swap(
        static_cast<BufferedTransferRPUDeviceCuda<T> &>(a),
        static_cast<BufferedTransferRPUDeviceCuda<T> &>(b));
    swap(a.cwo_, b.cwo_);
    swap(a.m_x_, b.m_x_);
    swap(a.m_d_, b.m_d_);
  };

  void runUpdateKernel(
      pwukp_t<T> kpars,
      CudaContextPtr up_context,
      T *dev_weights,
      int m_batch,
      const BitLineMaker<T> *blm,
      const PulsedUpdateMetaParameter<T> &up,
      const T lr,
      curandState_t *dev_states,
      int one_sided = 0,
      uint32_t *x_counts_chunk = nullptr,
      uint32_t *d_counts_chunk = nullptr,
      const ChoppedWeightOutput<T> *cwo = nullptr) override;

  void populateFrom(const AbstractRPUDevice<T> &rpu_device) override;

  void transfer(
      int to_device_idx,
      int from_device_idx,
      const PulsedUpdateMetaParameter<T> &current_up,
      const T current_sgd_lr,
      const T current_count_lr) override;

  void readAndUpdate(
      int to_device_idx,
      int from_device_idx,
      int i_slice_start,
      const T lr,
      const T count_lr,
      const T *x_input,
      const int n_vec,
      const PulsedUpdateMetaParameter<T> &up) override;

  void readMatrix(int device_idx, const T *in_vec, T *out_vec, int m_batch, T alpha) override;

  void writeMatrix(
      int device_idx,
      const T *in_vec,
      const T *out_vec,
      int m_batch,
      const T lr,
      const PulsedUpdateMetaParameter<T> &up) override;

  pwukpvec_t<T> getUpdateKernels(
      int m_batch,
      int nK32,
      int use_bo64,
      bool out_trans,
      const PulsedUpdateMetaParameter<T> &up) override;

  ChoppedTransferRPUDeviceMetaParameter<T> &getPar() const {
    return static_cast<ChoppedTransferRPUDeviceMetaParameter<T> &>(
        SimpleRPUDeviceCuda<T>::getPar());
  };
  ChoppedTransferRPUDeviceCuda<T> *clone() const override {
    return new ChoppedTransferRPUDeviceCuda<T>(*this);
  };
  void dumpExtra(RPU::state_t &extra, const std::string prefix) override;
  void loadExtra(const RPU::state_t &extra, const std::string prefix, bool strict) override;

protected:
  int getTransferEvery(
      int didx, int m_batch, const PulsedUpdateMetaParameter<T> &up) const override;
  T getPulseCountLearningRate(
      T lr, int current_m_batch, const PulsedUpdateMetaParameter<T> &up) override;
  std::unique_ptr<ChoppedWeightOutput<T>> cwo_ = nullptr;
  inline T getCurrentGradStrength() const { return m_x_ * m_d_; };
  inline T getCurrentDSparsity() const { return d_sparsity_; };

private:
  T m_x_ = (T)0.0;
  T m_d_ = (T)0.0;
  T d_sparsity_ = (T)0.0;
};

} // namespace RPU
