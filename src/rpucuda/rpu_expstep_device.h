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

#include "rng.h"
#include "rpu_pulsed_device.h"
#include <memory>

namespace RPU {

template <typename T> class ExpStepRPUDevice;

BUILD_PULSED_DEVICE_META_PARAMETER(ExpStep,
                                   /*implements*/
                                   DeviceUpdateType::ExpStep,
                                   /*parameter def*/
                                   T es_A_up = (T)0.00081;        /* p_000081 */
                                   T es_A_down = (T)0.36833;      /* p_036833 */
                                   T es_gamma_up = (T)12.44625;   /*  p_1244625 */
                                   T es_gamma_down = (T)12.78785; /* p_1278785 */
                                   T es_a = (T)0.244;             /* p_0244 */
                                   T es_b = (T)0.2425;            /*p_02425 */
                                   ,
                                   /*print body*/
                                   ss << "\t es_A_up:\t\t" << es_A_up << std::endl;
                                   ss << "\t es_A_down:\t\t" << es_A_down << std::endl;
                                   ss << "\t es_gamma_up:\t\t" << es_gamma_up << std::endl;
                                   ss << "\t es_gamma_down:\t\t" << es_gamma_down << std::endl;
                                   ss << "\t es_a:\t\t\t" << es_a << std::endl;
                                   ss << "\t es_b:\t\t\t" << es_b << std::endl;
                                   ,
                                   /*add */

);

template <typename T> class ExpStepRPUDevice : public PulsedRPUDevice<T> {

  BUILD_PULSED_DEVICE_CONSTRUCTORS(
      ExpStepRPUDevice,
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
