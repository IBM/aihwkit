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

#include "pulsed_weight_updater.h"
#include "rpu_mixedprec_int_device.h"
#include "rpucuda_mixedprec_device_base.h"

namespace RPU {

template <typename T> class MixedPrecIntRPUDeviceCuda : public MixedPrecRPUDeviceBaseCuda<T> {

public:
  explicit MixedPrecIntRPUDeviceCuda(){};
  explicit MixedPrecIntRPUDeviceCuda(CudaContextPtr c, int x_size, int d_size);
  explicit MixedPrecIntRPUDeviceCuda(CudaContextPtr c, const MixedPrecIntRPUDevice<T> &other);

  ~MixedPrecIntRPUDeviceCuda(){};
  MixedPrecIntRPUDeviceCuda(const MixedPrecIntRPUDeviceCuda<T> &other);
  MixedPrecIntRPUDeviceCuda<T> &operator=(const MixedPrecIntRPUDeviceCuda<T> &other);
  MixedPrecIntRPUDeviceCuda(MixedPrecIntRPUDeviceCuda<T> &&other);
  MixedPrecIntRPUDeviceCuda<T> &operator=(MixedPrecIntRPUDeviceCuda<T> &&other);

  friend void swap(MixedPrecIntRPUDeviceCuda<T> &a, MixedPrecIntRPUDeviceCuda<T> &b) noexcept {
    using std::swap;
    swap(
        static_cast<MixedPrecRPUDeviceBaseCuda<T> &>(a),
        static_cast<MixedPrecRPUDeviceBaseCuda<T> &>(b));

    swap(a.dev_chi_, b.dev_chi_);
  };
  bool hasDirectUpdate() const override { return true; };
  void doDirectUpdate(
      const T *x_input,
      const T *d_input,
      T *dev_weights,
      const T lr,
      const int m_batch,
      const bool x_trans,
      const bool d_trans,
      const T beta,
      const PulsedUpdateMetaParameter<T> &up,
      T *x_buffer,
      T *d_buffer) override;

  std::vector<T> getHiddenWeights() const override;
  void populateFrom(const AbstractRPUDevice<T> &rpu_device) override;

  MixedPrecIntRPUDeviceMetaParameter<T> &getPar() const override {
    return static_cast<MixedPrecIntRPUDeviceMetaParameter<T> &>(SimpleRPUDeviceCuda<T>::getPar());
  };
  MixedPrecIntRPUDeviceCuda<T> *clone() const override {
    return new MixedPrecIntRPUDeviceCuda<T>(*this);
  };

protected:
  void forwardUpdate(
      T *dev_weights,
      const T lr,
      int i_row_start,
      const T *transfer_vec,
      const int n_vec,
      const bool trans) override;

private:
  void allocateContainers();
  void quantize(
      T *quant_values,
      const T *values,
      RPU::NoiseManager<T> *nm,
      int n_bins,
      int size,
      int m_batch,
      bool trans,
      bool stochastic_rounding,
      RPU::NoiseManagementType nm_type);

  std::unique_ptr<CudaArray<T>> dev_chi_ = nullptr;
};

} // namespace RPU
