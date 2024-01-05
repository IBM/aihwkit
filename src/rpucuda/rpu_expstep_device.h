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
#include <memory>

namespace RPU {

template <typename T> class ExpStepRPUDevice;

BUILD_PULSED_DEVICE_META_PARAMETER(
    ExpStep,
    /*implements*/
    DeviceUpdateType::ExpStep,
    /*parameter def*/
    T es_A_up = (T)0.00081;        /* p_000081 */
    T es_A_down = (T)0.36833;      /* p_036833 */
    T es_gamma_up = (T)12.44625;   /*  p_1244625 */
    T es_gamma_down = (T)12.78785; /* p_1278785 */
    T es_a = (T)0.244;             /* p_0244 */
    T es_b = (T)0.2425;            /*p_02425 */
    T dw_min_std_add = (T)0.0;     // additive part of dw noise
    T dw_min_std_slope = (T)0.0;   // multiplicative part of noise with abs(w)
    ,
    /*print body*/
    ss << "\t es_A_up:\t\t" << es_A_up << std::endl;
    ss << "\t es_A_down:\t\t" << es_A_down << std::endl;
    ss << "\t es_gamma_up:\t\t" << es_gamma_up << std::endl;
    ss << "\t es_gamma_down:\t\t" << es_gamma_down << std::endl;
    ss << "\t es_a:\t\t\t" << es_a << std::endl;
    ss << "\t es_b:\t\t\t" << es_b << std::endl;
    if (dw_min_std_add != (T)0.0) {
      ss << "\t dw_min_std_add:\t " << dw_min_std_add << std::endl;
    } if (dw_min_std_slope != (T)0.0) {
      ss << "\t dw_min_std_slope:\t " << dw_min_std_slope << std::endl;
    },
    /* calc weight granularity body */
    T up_down = this->up_down;
    T up_bias = up_down > (T)0.0 ? (T)0.0 : up_down;
    T down_bias = up_down > (T)0.0 ? -up_down : (T)0.0;
    T scale_up = (T)(up_bias + (T)1.0) * this->dw_min;
    T scale_down = (T)(down_bias + (T)1.0) * this->dw_min;
    T w = (T)0.0; // just take at zero
    T z = (T)2.0 * w / (this->w_max - this->w_min) * es_a + es_b;
    T dw_down = scale_down * MAX((T)1.0 - (T)es_A_down * (T)expf((es_gamma_down * (-z))), (T)0.0);
    T dw_up = scale_up * MAX((T)1.0 - (T)es_A_up * (T)expf((es_gamma_up * z)), (T)0.0);
    T weight_granularity = MAX((dw_down + dw_down) / (T)2.0, (T)0.0);
    return weight_granularity > (T)0.0 ? weight_granularity : this->dw_min;
    ,
    /*add */
    bool implementsWriteNoise() const override { return true; };
    inline bool hasComplexNoise() const {
      return (
          (this->dw_min_std > (T)0.0) && (dw_min_std_slope != (T)0.0 || dw_min_std_add != (T)0.0));
    };);

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
