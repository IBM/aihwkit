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

#pragma once

#include "rpu_vector_device.h"
#include "rpucuda_pulsed_device.h"

namespace RPU {

template <typename T> class VectorRPUDeviceCuda : public PulsedRPUDeviceCudaBase<T> {

public:
  explicit VectorRPUDeviceCuda(){};
  // explicit VectorRPUDeviceCuda(CudaContext * c, int x_size, int d_size);
  explicit VectorRPUDeviceCuda(CudaContext *c, const VectorRPUDevice<T> &other);

  ~VectorRPUDeviceCuda(){};
  VectorRPUDeviceCuda(const VectorRPUDeviceCuda<T> &other);
  VectorRPUDeviceCuda<T> &operator=(const VectorRPUDeviceCuda<T> &other); // = default;
  VectorRPUDeviceCuda(VectorRPUDeviceCuda<T> &&other);                    // = default;
  VectorRPUDeviceCuda<T> &operator=(VectorRPUDeviceCuda<T> &&other);      // = default;

  friend void swap(VectorRPUDeviceCuda<T> &a, VectorRPUDeviceCuda<T> &b) noexcept {
    using std::swap;
    swap(
        static_cast<PulsedRPUDeviceCudaBase<T> &>(a), static_cast<PulsedRPUDeviceCudaBase<T> &>(b));
    swap(a.n_devices_, b.n_devices_);
    swap(a.dw_min_, b.dw_min_);
    swap(a.dev_weights_vec_, b.dev_weights_vec_);
    swap(a.rpucuda_device_vec_, b.rpucuda_device_vec_);
    swap(a.dev_weights_ptrs_, b.dev_weights_ptrs_);
    swap(a.context_vec_, b.context_vec_);
    swap(a.current_device_idx_, b.current_device_idx_);
    swap(a.current_update_idx_, b.current_update_idx_);
    swap(a.dev_reduce_weightening_, b.dev_reduce_weightening_);
  };

  // implement abstract functions
  std::vector<T> getHiddenWeights() const override;
  int getHiddenUpdateIdx() const override;
  void setHiddenUpdateIdx(int idx) override;

  void decayWeights(T *dev_weights, bool bias_no_decay) override;
  void decayWeights(T *dev_weights, T alpha, bool bias_no_decay) override;
  void diffuseWeights(T *dev_weights) override;
  void clipWeights(T *dev_weights, T clip) override;
  void resetCols(T *dev_weights, int start_col, int n_cols, T reset_prob) override;
  void applyWeightUpdate(T *dev_weights, T *dw_and_current_weight_out) override {
    // for parallel: would need to sync all weights separately. too costly anyway
    RPU_FATAL("Not supported for vector devices.");
  };

  void populateFrom(const AbstractRPUDevice<T> &rpu_device) override;
  T getDwMin() const override { return dw_min_; };

  VectorRPUDeviceMetaParameter<T> &getPar() const override {
    return static_cast<VectorRPUDeviceMetaParameter<T> &>(SimpleRPUDeviceCuda<T>::getPar());
  };
  VectorRPUDeviceCuda<T> *clone() const override { return new VectorRPUDeviceCuda<T>(*this); };

  void runUpdateKernel(
      pwukp_t<T> kpars,
      CudaContext *up_context,
      T *dev_weights,
      int m_batch,
      const BitLineMaker<T> *blm,
      const PulsedUpdateMetaParameter<T> &up,
      curandState_t *dev_states,
      int one_sided = 0,
      uint32_t *x_counts_chunk = nullptr,
      uint32_t *d_counts_chunk = nullptr) override;

  pwukpvec_t<T> getUpdateKernels(
      int m_batch,
      int nK32,
      int use_bo64,
      bool out_trans,
      const PulsedUpdateMetaParameter<T> &up) override;

  std::vector<T> getReduceWeightening() const;

protected:
  T dw_min_ = 0;
  int n_devices_ = 0;

  virtual void reduceToWeights(CudaContext *c, T *dev_weights);
  RealWorldRNG<T> rw_rng_{0};
  std::vector<T *> dev_weights_ptrs_;
  std::unique_ptr<CudaArray<T>> dev_weights_vec_ = nullptr;
  std::vector<std::unique_ptr<CudaContext>> context_vec_;
  std::vector<std::unique_ptr<PulsedRPUDeviceCudaBase<T>>> rpucuda_device_vec_;
  int current_device_idx_ = 0;
  unsigned long int current_update_idx_ = 0;
  std::unique_ptr<CudaArray<T>> dev_reduce_weightening_ = nullptr;

private:
  void allocateContainers();
};

} // namespace RPU
