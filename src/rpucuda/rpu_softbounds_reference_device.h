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

template <typename T> class SoftBoundsReferenceRPUDevice;

BUILD_PULSED_DEVICE_META_PARAMETER(
    SoftBoundsReference,
    /*implements*/
    DeviceUpdateType::SoftBoundsReference,
    /*parameter def*/
    T slope_up_dtod = (T)0.0;
    T slope_down_dtod = (T)0.0;
    T reference_mean = (T)0.0;
    T reference_std = (T)0.0;
    bool mult_noise = false;
    bool subtract_symmetry_point = false;
    ,
    /*print body*/
    ss << "\t mult_noise:\t\t\t" << std::boolalpha << mult_noise << std::endl;
    ss << "\t slope_up_dtod:\t" << slope_up_dtod << std::endl;
    ss << "\t slope_down_dtod:\t" << slope_down_dtod << std::endl;
    ss << "\t reference_mean:\t" << reference_mean << std::endl;
    ss << "\t reference_std:\t" << reference_std << std::endl;
    ss << "\t subtract_symmetry_point:\t" << std::boolalpha << subtract_symmetry_point << std::endl;
    ,
    /* calc weight granularity body */
    return this->dw_min;
    ,
    /*Add*/
    bool implementsWriteNoise() const override { return true; };);

template <typename T> class SoftBoundsReferenceRPUDevice : public PulsedRPUDevice<T> {

  BUILD_PULSED_DEVICE_CONSTRUCTORS(
      SoftBoundsReferenceRPUDevice,
      /* ctor*/
      int x_sz = this->x_size_;
      int d_sz = this->d_size_;

      w_reference_ = Array_2D_Get<T>(d_sz, x_sz);

      for (int j = 0; j < x_sz; ++j) {
        for (int i = 0; i < d_sz; ++i) {
          w_reference_[i][j] = (T)0.0;
        }
      },
      /* dtor*/
      Array_2D_Free<T>(w_reference_);
      ,
      /* copy */
      for (int j = 0; j < other.x_size_; ++j) {
        for (int i = 0; i < other.d_size_; ++i) {
          w_reference_[i][j] = other.w_reference_[i][j];
        }
      },
      /* move assignment */
      w_reference_ = other.w_reference_;
      other.w_reference_ = nullptr;
      ,
      /* swap*/
      swap(a.w_reference_, b.w_reference_);
      ,
      /* dp names*/
      names.push_back(std::string("reference"));
      ,
      /* dp2vec body*/
      int n_prev = (int)names.size();
      int size = this->x_size_ * this->d_size_;

      for (int i = 0; i < size; ++i) { data_ptrs[n_prev][i] = w_reference_[0][i]; },
      /* vec2dp body*/
      int n_prev = (int)names.size();
      int size = this->x_size_ * this->d_size_;

      for (int i = 0; i < size; ++i) { w_reference_[0][i] = data_ptrs[n_prev][i]; },
      /*invert copy DP */
      for (int j = 0; j < this->x_size_; ++j) {
        for (int i = 0; i < this->d_size_; ++i) {
          w_reference_[i][j] = -w_reference_[i][j];
        }
      }

  );

  void printDP(int x_count, int d_count) const override;
  inline T **getReference() const { return w_reference_; };

  void doSparseUpdate(
      T **weights, int i, const int *x_signed_indices, int x_count, int d_sign, RNG<T> *rng)
      override;
  void doDenseUpdate(T **weights, int *coincidences, RNG<T> *rng) override;

private:
  T **w_reference_ = nullptr;
};
} // namespace RPU
