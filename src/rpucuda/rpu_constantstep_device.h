/**
 * (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
 *
 * Licensed under the MIT license. See LICENSE file in the project root for details.
 */

#pragma once

#include "rng.h"
#include "rpu_pulsed_device.h"
#include "utility_functions.h"

namespace RPU {

template <typename T> class ConstantStepRPUDevice;

BUILD_PULSED_DEVICE_META_PARAMETER(ConstantStep,
                                   /*implements*/
                                   DeviceUpdateType::ConstantStep,
                                   /*parameter def*/
                                   ,
                                   /*print body*/
                                   ,
                                   /* calc weight granularity body */
                                   return this->dw_min;
                                   ,
                                   /*add*/
);

template <typename T> class ConstantStepRPUDevice : public PulsedRPUDevice<T> {

  BUILD_PULSED_DEVICE_CONSTRUCTORS(
      ConstantStepRPUDevice,
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
