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

#include "pwu_kernel_parameter_base.h"
#include "rpu_linearstep_device.h"
#include "rpucuda_pulsed_device.h"
#include <memory>

namespace RPU {

template <typename T> class LinearStepRPUDeviceCuda : public PulsedRPUDeviceCuda<T> {

public:
  BUILD_PULSED_DEVICE_CONSTRUCTORS_CUDA(
      LinearStepRPUDeviceCuda,
      LinearStepRPUDevice,
      /*ctor body*/
      dev_slope_ =
          std::unique_ptr<CudaArray<float>>(new CudaArray<float>(this->context_, 2 * this->size_));
      ,
      /*dtor body*/
      ,
      /*copy body*/
      dev_slope_->assign(*other.dev_slope_);
      ,
      /*move assigment body*/
      dev_slope_ = std::move(other.dev_slope_);
      ,
      /*swap body*/
      swap(a.dev_slope_, b.dev_slope_);
      ,
      /*host copy from cpu (rpu_device). Parent device params are copyied automatically*/
      int d_size = this->d_size_;
      int x_size = this->x_size_;
      T **w_slope_up = rpu_device.getSlopeUp();
      T **w_slope_down = rpu_device.getSlopeDown();
      float *tmp_slope = new float[2 * this->size_];

      for (int i = 0; i < d_size; ++i) {
        for (int j = 0; j < x_size; ++j) {

          int kk = j * (d_size * 2) + 2 * i;
          tmp_slope[kk] = w_slope_down[i][j];
          tmp_slope[kk + 1] = w_slope_up[i][j];
        }
      } dev_slope_->assign(tmp_slope);
      this->context_->synchronize();
      delete[] tmp_slope;);

  pwukpvec_t<T> getUpdateKernels(
      int m_batch,
      int nK32,
      int use_bo64,
      bool out_trans,
      const PulsedUpdateMetaParameter<T> &up) override;
  float *get2ParamsData() override { return dev_slope_->getData(); };

private:
  std::unique_ptr<CudaArray<float>> dev_slope_ = nullptr;
};

} // namespace RPU
