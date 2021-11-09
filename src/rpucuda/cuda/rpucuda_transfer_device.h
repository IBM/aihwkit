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

#pragma once

#include "io_manager.h"
#include "pulsed_weight_updater.h"
#include "rpu_transfer_device.h"
#include "rpucuda_pulsed_device.h"
#include "rpucuda_vector_device.h"

namespace RPU {

template <typename T> class TransferRPUDeviceCuda : public VectorRPUDeviceCuda<T> {

public:
  explicit TransferRPUDeviceCuda(){};
  // explicit TransferRPUDeviceCuda(CudaContext * c, int x_size, int d_size);
  explicit TransferRPUDeviceCuda(CudaContext *c, const TransferRPUDevice<T> &other);

  ~TransferRPUDeviceCuda(){};
  TransferRPUDeviceCuda(const TransferRPUDeviceCuda<T> &other);
  TransferRPUDeviceCuda<T> &operator=(const TransferRPUDeviceCuda<T> &other);
  TransferRPUDeviceCuda(TransferRPUDeviceCuda<T> &&other);
  TransferRPUDeviceCuda<T> &operator=(TransferRPUDeviceCuda<T> &&other);

  friend void swap(TransferRPUDeviceCuda<T> &a, TransferRPUDeviceCuda<T> &b) noexcept {
    using std::swap;
    swap(static_cast<VectorRPUDeviceCuda<T> &>(a), static_cast<VectorRPUDeviceCuda<T> &>(b));

    swap(a.transfer_tmp_, b.transfer_tmp_);
    swap(a.transfer_vecs_, b.transfer_vecs_);
    swap(a.transfer_iom_, b.transfer_iom_);
    swap(a.transfer_pwu_, b.transfer_pwu_);
    swap(a.current_slice_indices_, b.current_slice_indices_);
    swap(a.fully_hidden_, b.fully_hidden_);
  };

  void populateFrom(const AbstractRPUDevice<T> &rpu_device) override;
  TransferRPUDeviceMetaParameter<T> &getPar() const {
    return static_cast<TransferRPUDeviceMetaParameter<T> &>(SimpleRPUDeviceCuda<T>::getPar());
  };
  TransferRPUDeviceCuda<T> *clone() const override { return new TransferRPUDeviceCuda<T>(*this); };

  void setHiddenUpdateIdx(int idx) override{};

  void runUpdateKernel(
      pwukp_t<T> kpars,
      CudaContext *up_context,
      T *dev_weights,
      int m_batch,
      const BitLineMaker<T> *blm,
      const PulsedUpdateMetaParameter<T> &up,
      const T lr,
      curandState_t *dev_states,
      int one_sided = 0,
      uint32_t *x_counts_chunk = nullptr,
      uint32_t *d_counts_chunk = nullptr) override;

  virtual void transfer(
      int to_device_idx,
      int from_device_idx,
      const PulsedUpdateMetaParameter<T> &current_up,
      const T current_lr);

  void decayWeights(T *dev_weights, bool bias_no_decay) override;
  void decayWeights(T *dev_weights, T alpha, bool bias_no_decay) override;
  void driftWeights(T *dev_weights, T time_since_epoch) override;
  void diffuseWeights(T *dev_weights) override;
  void clipWeights(T *dev_weights, T clip) override;
  void resetCols(T *dev_weights, int start_col, int n_cols, T reset_prob) override;

  // uses the getPar().transfer_up and getPar().transfer_io to make a forward with transfer_vec and
  // update
  virtual void readAndUpdate(
      int to_device_idx,
      int from_device_idx,
      int i_slice_start,
      const T lr,
      const T *x_input,
      const int n_vec,
      const PulsedUpdateMetaParameter<T> &up);

  virtual void writeMatrix(
      int device_idx,
      const T *in_vec,
      const T *out_vec,
      int m_batch,
      const T lr,
      const PulsedUpdateMetaParameter<T> &up);
  virtual void readMatrix(int device_idx, const T *in_vec, T *out_vec, int m_batch, T alpha);

  pwukpvec_t<T> getUpdateKernels(
      int m_batch,
      int nK32,
      int use_bo64,
      bool out_trans,
      const PulsedUpdateMetaParameter<T> &up) override;

  T getPulseCountLearningRate(T learning_rate) override;

protected:
  void reduceToWeights(CudaContext *c, T *dev_weights) override;

  std::unique_ptr<CudaArray<T>> transfer_tmp_ = nullptr; // no need to copy
  std::unique_ptr<CudaArray<T>> transfer_vecs_ = nullptr;
  std::unique_ptr<InputOutputManager<T>> transfer_iom_ = nullptr;
  std::unique_ptr<PulsedWeightUpdater<T>> transfer_pwu_ = nullptr;
  std::vector<int> current_slice_indices_;
  bool fully_hidden_ = false;

private:
  int getTransferEvery(int device_idx, int m_batch) const;

  void initialize(bool transfer_columns);
};

} // namespace RPU
