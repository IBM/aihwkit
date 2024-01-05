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

#include "pwu_kernel_parameter_base.h"
#include "rpu_piecewisestep_device.h"
#include "rpucuda_pulsed_device.h"
#include <memory>

namespace RPU {

template <typename T> class PiecewiseStepRPUDeviceCuda : public PulsedRPUDeviceCuda<T> {

public:
  BUILD_PULSED_DEVICE_CONSTRUCTORS_CUDA(
      PiecewiseStepRPUDeviceCuda,
      PiecewiseStepRPUDevice,
      /*ctor body*/
      ,
      /*dtor body*/
      ,
      /*copy body*/
      dev_global_pars_ = nullptr;
      if (other.dev_global_pars_) {
        dev_global_pars_ = RPU::make_unique<CudaArray<T>>(*other.dev_global_pars_);
      },
      /*move assigment body*/
      dev_global_pars_ = std::move(other.dev_global_pars_);
      ,
      /*swap body*/
      swap(a.dev_global_pars_, b.dev_global_pars_);
      swap(a.gp_count_, b.gp_count_);
      ,
      /*host copy from cpu (rpu_device). Parent device params are copyied automatically*/
      const auto &par = getPar();
      int n_points = (int)par.piecewise_up_vec.size();
      gp_count_ = MAX(exp2((int)ceil(log2((float)(n_points + 1))) + 1), 32);
      if (n_points != par.piecewise_down_vec.size()) {
        RPU_FATAL("Down and up interpolation node numbers need to be the same.");
      }

      T *tmp_global_pars = new T[gp_count_]();

      for (int i = 0; i < n_points; ++i) {
        tmp_global_pars[i] = par.piecewise_up_vec[i];
        tmp_global_pars[i + gp_count_ / 2] = par.piecewise_down_vec[i];
      }
      // n_points info
      tmp_global_pars[gp_count_ / 2 - 1] = (T)n_points;
      // write std info
      tmp_global_pars[gp_count_ - 1] = getPar().getScaledWriteNoise();

      dev_global_pars_ = RPU::make_unique<CudaArray<T>>(this->context_, gp_count_, tmp_global_pars);
      this->context_->synchronize();
      delete[] tmp_global_pars;);

  pwukpvec_t<T> getUpdateKernels(
      int m_batch,
      int nK32,
      int use_bo64,
      bool out_trans,
      const PulsedUpdateMetaParameter<T> &up) override;
  T *getGlobalParamsData() override { return dev_global_pars_->getData(); };
  param_t *get2ParamsData() override { return nullptr; };
  T *get1ParamsData() override {
    return getPar().usesPersistentWeight() ? this->dev_persistent_weights_->getData() : nullptr;
  };
  T getWeightGranularityNoise() const override {
    // need to make sure that random states are enabled
    return getPar().usesPersistentWeight()
               ? PulsedRPUDeviceCuda<T>::getWeightGranularityNoise() + (T)1e-6
               : PulsedRPUDeviceCuda<T>::getWeightGranularityNoise();
  }

private:
  std::unique_ptr<CudaArray<T>> dev_global_pars_ = nullptr;
  int gp_count_ = 0;
};

} // namespace RPU
