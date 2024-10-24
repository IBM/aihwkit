/**
 * (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
 *
 * Licensed under the MIT license. See LICENSE file in the project root for details.
 */

#pragma once

#include "pwu_kernel_parameter_base.h"
#include "rpu_constantstep_device.h"
#include "rpucuda_pulsed_device.h"
#include <memory>

namespace RPU {

/* ConstantStepRPUDeviceCuda

   Implements the ConstantStep update, that is the step-size per
   pulse (dw) does not depend on the actual weight value (w).

   For smaller noise, we assume that the cycle-to-cycle
   noise times the weight update minimal step (dw) remains
   consistent with the direction of the update. E.g. if the update
   is supposed to be positive, dw + noise is assumed to be positive
   as well.

   If noise setting is too large, so that dw+noise signs changes
   become likely during multiple steps in the same direction, we use
   a slightly slower implementation, where the bounds are checked
   for each weight change.

 */
template <typename T> class ConstantStepRPUDeviceCuda : public PulsedRPUDeviceCuda<T> {

public:
  BUILD_PULSED_DEVICE_CONSTRUCTORS_CUDA(
      ConstantStepRPUDeviceCuda,
      ConstantStepRPUDevice,
      /*ctor body*/
      ,
      /*dtor body*/
      ,
      /*copy body*/
      ,
      /*move assigment body*/
      ,
      /*swap body*/
      ,
      /*host copy from cpu (rpu_device). Parent device params are copyied automatically*/
  )

  pwukpvec_t<T> getUpdateKernels(
      int m_batch,
      int nK32,
      int use_bo64,
      bool out_trans,
      const PulsedUpdateMetaParameter<T> &up) override;
};

} // namespace RPU
