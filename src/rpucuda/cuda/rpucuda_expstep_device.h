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
#include "rpu_expstep_device.h"
#include "rpucuda_pulsed_device.h"
#include <memory>

namespace RPU {

template <typename T> class ExpStepRPUDeviceCuda : public PulsedRPUDeviceCuda<T> {

public:
  BUILD_PULSED_DEVICE_CONSTRUCTORS_CUDA(
      ExpStepRPUDeviceCuda,
      ExpStepRPUDevice,
      /*ctor body*/
      dev_es_par_ = std::unique_ptr<CudaArray<T>>(new CudaArray<T>(this->context_, 9));
      ,
      /*dtor body*/
      ,
      /*copy body*/
      dev_es_par_->assign(*other.dev_es_par_);
      ,
      /*move assigment body*/
      dev_es_par_ = std::move(other.dev_es_par_);
      ,
      /*swap body*/
      swap(a.dev_es_par_, b.dev_es_par_);
      ,
      /*host copy from cpu (rpu_device). Parent device params are copyied automatically*/
      T es_par_arr[9];
      auto &par = getPar();
      es_par_arr[0] = par.es_A_down;
      es_par_arr[1] = par.es_A_up;
      es_par_arr[2] = par.es_gamma_down;
      es_par_arr[3] = par.es_gamma_up;
      es_par_arr[4] = par.es_a;
      es_par_arr[5] = par.es_b;
      es_par_arr[6] = par.getScaledWriteNoise();
      es_par_arr[7] = par.dw_min_std_add;
      es_par_arr[8] = par.dw_min_std_slope;
      dev_es_par_->assign(es_par_arr);
      this->context_->synchronize();)

  T *getGlobalParamsData() override { return dev_es_par_->getData(); };
  pwukpvec_t<T> getUpdateKernels(
      int m_batch,
      int nK32,
      int use_bo64,
      bool out_trans,
      const PulsedUpdateMetaParameter<T> &up) override;
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
  std::unique_ptr<CudaArray<T>> dev_es_par_ = nullptr;
};

} // namespace RPU
