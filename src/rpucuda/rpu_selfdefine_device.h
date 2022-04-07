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

#include "rng.h"
#include "rpu_pulsed_device.h"
#include "utility_functions.h"

namespace RPU {

template <typename T> class SelfDefineRPUDevice;

BUILD_PULSED_DEVICE_META_PARAMETER(
    SelfDefine,
    /*implements*/
    DeviceUpdateType::SelfDefine,
    
    /*parameter def*/    
    std::vector<T> sd_up_pulse;
    std::vector<T> sd_down_pulse;
    T sd_n_points = (T)0.00;
    ,
    /*print body*/
    ss << "\t(dtod=" << this->dw_min_dtod << ")" << std::endl;
    ss << "\t up_down:\t" << this->up_down << "\t(dtod=" << this->up_down_dtod << ")"
       << std::endl;
    ,
    /* calc weight granularity body */
    return this->dw_min;
    ,
    /*Add*/
    bool implementsWriteNoise() const override { return true; };);

template <typename T> class SelfDefineRPUDevice : public PulsedRPUDevice<T> {

  BUILD_PULSED_DEVICE_CONSTRUCTORS(
      SelfDefineRPUDevice,
      /* ctor*/
      ,
      /* dtor*/
      ,
      /* copy */
      ,
      /* move assignment */
      ,
      /* swap*/
      ,
      /* dp names*/
      ,
      /* dp2vec body*/
      ,
      /* vec2dp body*/
      ,
      /*invert copy DP */
  );

  void doSparseUpdate(
      T **weights, int i, const int *x_signed_indices, int x_count, int d_sign, RNG<T> *rng)
      override;
  void doDenseUpdate(T **weights, int *coincidences, RNG<T> *rng) override;

};
} // namespace RPU
