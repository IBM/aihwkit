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

#include "rng.h"
#include "rpu_pulsed_device.h"
#include "utility_functions.h"

namespace RPU {

template <typename T> class PiecewiseStepRPUDevice;

BUILD_PULSED_DEVICE_META_PARAMETER(
    PiecewiseStep,
    /*implements*/
    DeviceUpdateType::PiecewiseStep,

    /*parameter def*/
    std::vector<T> piecewise_up_vec{1};
    std::vector<T> piecewise_down_vec{1};
    ,
    /*print body*/
    if (piecewise_up_vec.size() > 0) {
      ss << "\t piecewise_up_vec:\t" << piecewise_up_vec.size() << " values ["
         << piecewise_up_vec.front() << " .. " << piecewise_up_vec.back() << "]" << std::endl;
    } if (piecewise_down_vec.size() > 0) {
      ss << "\t piecewise_down_vec:\t" << piecewise_down_vec.size() << " values ["
         << piecewise_down_vec.front() << " .. " << piecewise_down_vec.back() << "]" << std::endl;
    },
    /* calc weight granularity body */
    if (piecewise_up_vec.size() > 0 && piecewise_down_vec.size() > 0) {
      size_t middle = piecewise_up_vec.size() / 2;
      return this->dw_min * piecewise_up_vec[middle];
    } else { return this->dw_min; },
    /*Add*/
    bool implementsWriteNoise() const override { return true; };
    void initialize() override {
      PulsedRPUDeviceMetaParameter<T>::initialize();
      if ((piecewise_up_vec.size() == 0) || (piecewise_down_vec.size() == 0)) {
        RPU_FATAL("piecewise_up_vec and piecewise_down_vec needs to have some contents.");
      }
      if (piecewise_up_vec.size() != piecewise_down_vec.size()) {
        RPU_FATAL("piecewise_up_vec and piecewise_down_vec needs to have some size.");
      }
    };);

template <typename T> class PiecewiseStepRPUDevice : public PulsedRPUDevice<T> {

  BUILD_PULSED_DEVICE_CONSTRUCTORS(
      PiecewiseStepRPUDevice,
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
