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
#include "rpu_hidden_device.h"
#include "rpucuda_pulsed_device.h"
#include <memory>

namespace RPU {

template <typename T> class HiddenStepRPUDeviceCuda : public PulsedRPUDeviceCuda<T> {

public:
  BUILD_PULSED_DEVICE_CONSTRUCTORS_CUDA(
      HiddenStepRPUDeviceCuda,
      HiddenStepRPUDevice,
      /*ctor body*/
      dev_hs_scale_ = RPU::make_unique<CudaArray<param_t>>(this->context_, 2 * this->size_);
      dev_hidden_weights_ = RPU::make_unique<CudaArray<T>>(this->context_, this->size_);
      dev_hs_dw_min_std_ = RPU::make_unique<CudaArray<T>>(this->context_, 1);
      ,
      /*dtor body*/
      ,
      /*copy body*/
      dev_hs_dw_min_std_->assign(*other.dev_hs_dw_min_std_);
      dev_hs_scale_->assign(*other.dev_hs_scale_);
      dev_hidden_weights_->assign(*other.dev_hidden_weights_);
      ,
      /*move assigment body*/
      dev_hs_scale_ = std::move(other.dev_hs_scale_);
      dev_hidden_weights_ = std::move(other.dev_hidden_weights_);
      dev_hs_dw_min_std_ = std::move(other.dev_hs_dw_min_std_);
      ,
      /*swap body*/
      swap(a.dev_hidden_weights_, b.dev_hidden_weights_);
      swap(a.dev_hs_dw_min_std_, b.dev_hs_dw_min_std_);
      swap(a.dev_hs_scale_, b.dev_hs_scale_);
      ,
      /*host copy from cpu (rpu_device). Parent device params are copyied automatically*/
      T hs_dw_min_std = rpu_device.getPar().hs_dw_min_std;
      dev_hs_dw_min_std_->assign(&hs_dw_min_std);

      T **hs_scale_up = rpu_device.getHsScaleUp();
      T **hs_scale_down = rpu_device.getHsScaleDown();
      T **hidden_weights = rpu_device.getHiddenWeights();
      param_t *tmp_scale = new param_t[2 * this->size_];
      T *tmp_hw = new T[this->size_];

      for (int i = 0; i < this->d_size_; ++i) {
        for (int j = 0; j < this->x_size_; ++j) {

          int k = j * (this->d_size_) + i;
          int kk = j * (this->d_size_ * 2) + 2 * i;
          tmp_scale[kk] = hs_scale_down[i][j];
          tmp_scale[kk + 1] = hs_scale_up[i][j];

          tmp_hw[k] = hidden_weights[i][j]; /* actually no real need to copy the weights (they are
                                               0). do it anyway*/
        }
      } dev_hs_scale_->assign(tmp_scale);
      dev_hidden_weights_->assign(tmp_hw);
      this->context_->synchronize();

      delete[] tmp_scale;
      delete[] tmp_hw;)

  T *getGlobalParamsData() override { return dev_hs_dw_min_std_->getData(); };
  T *get1ParamsData() override { return dev_hidden_weights_->getData(); };
  param_t *get2ParamsData() override { return dev_hs_scale_->getData(); };
  void applyWeightUpdate(T *dev_weights, T *dw_and_current_weight_out) override {
    // would need to sync hidden weights separately. too costly anyway
    RPU_FATAL("Not supported for hidden step devices.");
  };

  pwukpvec_t<T> getUpdateKernels(
      int m_batch, int nK32, int use_bo64, bool out_trans, const PulsedUpdateMetaParameter<T> &up);

private:
  std::unique_ptr<CudaArray<T>> dev_hidden_weights_ = nullptr;
  std::unique_ptr<CudaArray<param_t>> dev_hs_scale_ = nullptr;
  std::unique_ptr<CudaArray<T>> dev_hs_dw_min_std_ = nullptr;
};

} // namespace RPU
