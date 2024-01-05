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

#include "noise_manager.h"
#include "pulsed_weight_updater.h"
#include "rpu_mixedprec_device_base.h"
#include "rpucuda_simple_device.h"

namespace RPU {

template <typename T> class MixedPrecRPUDeviceBaseCuda : public SimpleRPUDeviceCuda<T> {

public:
  explicit MixedPrecRPUDeviceBaseCuda(){};
  explicit MixedPrecRPUDeviceBaseCuda(CudaContextPtr c, int x_size, int d_size);
  explicit MixedPrecRPUDeviceBaseCuda(CudaContextPtr c, const MixedPrecRPUDeviceBase<T> &other);

  virtual ~MixedPrecRPUDeviceBaseCuda(){};
  MixedPrecRPUDeviceBaseCuda(const MixedPrecRPUDeviceBaseCuda<T> &other);
  MixedPrecRPUDeviceBaseCuda<T> &operator=(const MixedPrecRPUDeviceBaseCuda<T> &other);
  MixedPrecRPUDeviceBaseCuda(MixedPrecRPUDeviceBaseCuda<T> &&other);
  MixedPrecRPUDeviceBaseCuda<T> &operator=(MixedPrecRPUDeviceBaseCuda<T> &&other);

  friend void swap(MixedPrecRPUDeviceBaseCuda<T> &a, MixedPrecRPUDeviceBaseCuda<T> &b) noexcept {
    using std::swap;
    swap(static_cast<SimpleRPUDeviceCuda<T> &>(a), static_cast<SimpleRPUDeviceCuda<T> &>(b));
    swap(a.rpucuda_device_, b.rpucuda_device_);

    swap(a.rw_rng_, b.rw_rng_);
    swap(a.transfer_pwu_, b.transfer_pwu_);
    swap(a.noise_manager_x_, b.noise_manager_x_);
    swap(a.noise_manager_d_, b.noise_manager_d_);

    swap(a.current_zero_size_, b.current_zero_size_);
    swap(a.current_update_index_, b.current_update_index_);
    swap(a.current_row_index_, b.current_row_index_);

    swap(a.dev_zc_temp_storage_, b.dev_zc_temp_storage_);

    swap(a.dev_sparsity_x_, b.dev_sparsity_x_);
    swap(a.dev_sparsity_d_, b.dev_sparsity_d_);
    swap(a.dev_avg_sparsity_, b.dev_avg_sparsity_);

    swap(a.dev_transfer_tmp_, b.dev_transfer_tmp_);
    swap(a.dev_transfer_d_vecs_, b.dev_transfer_d_vecs_);

    swap(a.io_, b.io_);
    swap(a.up_ptr_, b.up_ptr_);
    swap(a.up_, b.up_);
    swap(a.nblocks_batch_max_, b.nblocks_batch_max_);
    swap(a.granularity_, b.granularity_);
  };
  bool hasDirectUpdate() const override { return true; };
  std::vector<T> getHiddenWeights() const override;
  void dumpExtra(RPU::state_t &extra, const std::string prefix) override;
  void loadExtra(const RPU::state_t &extra, const std::string prefix, bool strict) override;

  void decayWeights(T *dev_weights, bool bias_no_decay) override;
  void decayWeights(T *dev_weights, T alpha, bool bias_no_decay) override;
  void driftWeights(T *dev_weights, T time_since_epoch) override;
  void diffuseWeights(T *dev_weights) override;
  void clipWeights(T *dev_weights, T clip) override;
  void resetCols(T *dev_weights, int start_col, int n_cols, T reset_prob) override;
  void applyWeightUpdate(T *dev_weights, T *dw_and_current_weight_out) override {
    // for parallel: maybe one could do a sparse sync of the weights or CHI? not yet implemented
    RPU_NOT_IMPLEMENTED;
  };
  T getAvgSparsity() const;

  void populateFrom(const AbstractRPUDevice<T> &rpu_device) override;

  MixedPrecRPUDeviceBaseMetaParameter<T> &getPar() const override {
    return static_cast<MixedPrecRPUDeviceBaseMetaParameter<T> &>(SimpleRPUDeviceCuda<T>::getPar());
  };

protected:
  virtual void forwardUpdate(
      T *dev_weights,
      const T lr,
      int i_row_start,
      const T *transfer_vec,
      const int n_vec,
      const bool trans) {
    RPU_NOT_IMPLEMENTED;
  };
  virtual void transfer(T *dev_weights, const T lr);
  void doTransfer(T *dev_weights, const T lr, const int m_batch);
  void setUpPar(const PulsedUpdateMetaParameter<T> &up);
  inline void advanceUpdateCounter(int m_batch) { current_update_index_ += m_batch; };

  void computeSparsity(const T *x_values, const T *d_values, const int m_batch);

  RealWorldRNG<T> rw_rng_{0};
  std::unique_ptr<AbstractRPUDeviceCuda<T>> rpucuda_device_;
  std::unique_ptr<PulsedWeightUpdater<T>> transfer_pwu_ = nullptr;
  std::unique_ptr<NoiseManager<T>> noise_manager_x_ = nullptr;
  std::unique_ptr<NoiseManager<T>> noise_manager_d_ = nullptr;

  PulsedUpdateMetaParameter<T> up_;
  IOMetaParameter<T> io_;
  std::unique_ptr<CudaArray<T>> dev_transfer_tmp_ = nullptr;
  std::unique_ptr<CudaArray<T>> dev_transfer_d_vecs_ = nullptr;

  T granularity_ = 0.0;
  int nblocks_batch_max_;

private:
  void allocateContainers();
  void computeSparsityPartly(T *sparsity, const T *values, const int size);

  int current_zero_size_ = 0;
  int current_row_index_ = 0;
  uint64_t current_update_index_ = 0; // this is in mat-vecs!

  std::unique_ptr<CudaArray<char>> dev_zc_temp_storage_ = nullptr;
  std::unique_ptr<CudaArray<T>> dev_sparsity_d_ = nullptr;
  std::unique_ptr<CudaArray<T>> dev_sparsity_x_ = nullptr;
  std::unique_ptr<CudaArray<T>> dev_avg_sparsity_ = nullptr;

  const PulsedUpdateMetaParameter<T> *up_ptr_;
};

} // namespace RPU
