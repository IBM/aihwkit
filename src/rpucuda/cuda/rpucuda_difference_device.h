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

#include "rpu_difference_device.h"
#include "rpucuda_pulsed_device.h"
#include "rpucuda_vector_device.h"

namespace RPU {

template <typename T> class DifferenceRPUDeviceCuda : public VectorRPUDeviceCuda<T> {

public:
  explicit DifferenceRPUDeviceCuda(){};
  explicit DifferenceRPUDeviceCuda(CudaContext *c, const DifferenceRPUDevice<T> &other);

  ~DifferenceRPUDeviceCuda(){};
  DifferenceRPUDeviceCuda(const DifferenceRPUDeviceCuda<T> &other);
  DifferenceRPUDeviceCuda<T> &operator=(const DifferenceRPUDeviceCuda<T> &other);
  DifferenceRPUDeviceCuda(DifferenceRPUDeviceCuda<T> &&other);
  DifferenceRPUDeviceCuda<T> &operator=(DifferenceRPUDeviceCuda<T> &&other);

  friend void swap(DifferenceRPUDeviceCuda<T> &a, DifferenceRPUDeviceCuda<T> &b) noexcept {
    using std::swap;
    swap(static_cast<VectorRPUDeviceCuda<T> &>(a), static_cast<VectorRPUDeviceCuda<T> &>(b));
    swap(a.g_plus_, b.g_plus_);
    swap(a.g_minus_, b.g_minus_);
    swap(a.dev_reduce_weightening_inverted_, b.dev_reduce_weightening_inverted_);
  };
  void resetCols(T *dev_weights, int start_col, int n_cols, T reset_prob) override;
  void populateFrom(const AbstractRPUDevice<T> &rpu_device) override;
  DifferenceRPUDeviceMetaParameter<T> &getPar() const override {
    return static_cast<DifferenceRPUDeviceMetaParameter<T> &>(SimpleRPUDeviceCuda<T>::getPar());
  };
  DifferenceRPUDeviceCuda<T> *clone() const override {
    return new DifferenceRPUDeviceCuda<T>(*this);
  };

  void setHiddenUpdateIdx(int idx) override{};

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

  inline void invert();

protected:
private:
  inline bool isInverted() const;

  int g_plus_ = 1;
  int g_minus_ = 0;
  std::unique_ptr<CudaArray<T>> dev_reduce_weightening_inverted_ = nullptr;
};

} // namespace RPU
