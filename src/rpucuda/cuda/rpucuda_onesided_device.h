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

#include "forward_backward_pass.h"
#include "io_manager.h"
#include "pulsed_weight_updater.h"
#include "rpu_onesided_device.h"
#include "rpucuda_pulsed_device.h"
#include "rpucuda_vector_device.h"

namespace RPU {

template <typename T> class OneSidedRPUDeviceCuda : public VectorRPUDeviceCuda<T> {

public:
  explicit OneSidedRPUDeviceCuda(){};
  explicit OneSidedRPUDeviceCuda(CudaContextPtr c, const OneSidedRPUDevice<T> &other);

  ~OneSidedRPUDeviceCuda(){};
  OneSidedRPUDeviceCuda(const OneSidedRPUDeviceCuda<T> &other);
  OneSidedRPUDeviceCuda<T> &operator=(const OneSidedRPUDeviceCuda<T> &other);
  OneSidedRPUDeviceCuda(OneSidedRPUDeviceCuda<T> &&other);
  OneSidedRPUDeviceCuda<T> &operator=(OneSidedRPUDeviceCuda<T> &&other);

  friend void swap(OneSidedRPUDeviceCuda<T> &a, OneSidedRPUDeviceCuda<T> &b) noexcept {
    using std::swap;
    swap(static_cast<VectorRPUDeviceCuda<T> &>(a), static_cast<VectorRPUDeviceCuda<T> &>(b));
    swap(a.g_plus_, b.g_plus_);
    swap(a.g_minus_, b.g_minus_);
    swap(a.dev_reduce_weightening_inverted_, b.dev_reduce_weightening_inverted_);
    swap(a.refresh_counter_, b.refresh_counter_);
    swap(a.dev_refresh_vecs_, b.dev_refresh_vecs_);
    swap(a.refresh_pwu_, b.refresh_pwu_);
    swap(a.refresh_iom_, b.refresh_iom_);
    swap(a.refresh_fb_pass_, b.refresh_fb_pass_);
  };
  void resetCols(T *dev_weights, int start_col, int n_cols, T reset_prob) override;

  void populateFrom(const AbstractRPUDevice<T> &rpu_device) override;
  OneSidedRPUDeviceMetaParameter<T> &getPar() const override {
    return static_cast<OneSidedRPUDeviceMetaParameter<T> &>(SimpleRPUDeviceCuda<T>::getPar());
  };
  OneSidedRPUDeviceCuda<T> *clone() const override { return new OneSidedRPUDeviceCuda<T>(*this); };

  void setHiddenUpdateIdx(int idx) override{};

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

  pwukpvec_t<T> getUpdateKernels(
      int m_batch,
      int nK32,
      int use_bo64,
      bool out_trans,
      const PulsedUpdateMetaParameter<T> &up) override;

  void invert();
  inline uint64_t getRefreshCount() const { return refresh_counter_; };

protected:
  virtual int refreshWeights();
  std::unique_ptr<CudaArray<T>> dev_refresh_vecs_ = nullptr;
  std::unique_ptr<InputOutputManager<T>> refresh_iom_ = nullptr;
  std::unique_ptr<PulsedWeightUpdater<T>> refresh_pwu_ = nullptr;
  std::unique_ptr<ForwardBackwardPassIOManagedCuda<T>> refresh_fb_pass_ = nullptr;
  uint64_t refresh_counter_ = 0;

  // tmp: no need to copy
  std::unique_ptr<CudaArray<T>> dev_refresh_tmp_p_ = nullptr;
  std::unique_ptr<CudaArray<T>> dev_refresh_tmp_m_ = nullptr;
  std::unique_ptr<CudaArray<char>> dev_reset_msk_ = nullptr;
  std::unique_ptr<CudaArray<int>> dev_refresh_counters_ = nullptr;

private:
  void initialize();
  bool isInverted() const;
  int nblocks_batch_max_ = 0; // no copy
  int g_plus_ = 1;
  int g_minus_ = 0;
  std::unique_ptr<CudaArray<T>> dev_reduce_weightening_inverted_ = nullptr;
};

} // namespace RPU
