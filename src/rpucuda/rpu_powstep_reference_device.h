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

template <typename T> class PowStepReferenceRPUDevice;

BUILD_PULSED_DEVICE_META_PARAMETER(
    PowStepReference,
    /*implements*/
    DeviceUpdateType::PowStepReference,
    /*parameter def*/
    T ps_gamma = (T)1.0;
    T ps_gamma_dtod = (T)0.1;
    T ps_gamma_up_down = (T)0.00;
    T ps_gamma_up_down_dtod = (T)0.00;
    T reference_mean = (T)0.0;
    T reference_std = (T)0.0;
    bool subtract_symmetry_point = false;
    int n_estimation_steps = -1;
    ,
    /*print body*/
    ss << "\t ps_gamma:\t\t" << ps_gamma << "\t(dtod=" << ps_gamma_dtod << ")" << std::endl;
    ss << "\t ps_gamma_up_down:\t" << ps_gamma_up_down << "\t(dtod=" << ps_gamma_up_down_dtod << ")"
       << std::endl;
    ss << "\t reference_mean:\t" << reference_mean << std::endl;
    ss << "\t reference_std:\t" << reference_std << std::endl;
    ss << "\t subtract_symmetry_point:\t" << std::boolalpha << subtract_symmetry_point;
    if (subtract_symmetry_point) {
      if (n_estimation_steps <= 0) {
        ss << "\t [auto n]";
      } else {
        ss << "\t [n = " << n_estimation_steps << " ]";
      }
    } ss
    << std::endl;
    ,
    /* calc weight granularity body */
    return this->dw_min * (T)powf((T)0.5, ps_gamma);
    ,
    /*Add*/
    bool implementsWriteNoise() const override { return false; };
    bool usesPersistentWeight() const override { return false; };);

template <typename T> class PowStepReferenceRPUDevice : public PulsedRPUDevice<T> {

  BUILD_PULSED_DEVICE_CONSTRUCTORS(
      PowStepReferenceRPUDevice,
      /* ctor*/
      int x_sz = this->x_size_;
      int d_sz = this->d_size_;

      w_gamma_down_ = Array_2D_Get<T>(d_sz, x_sz);
      w_gamma_up_ = Array_2D_Get<T>(d_sz, x_sz);
      w_reference_ = Array_2D_Get<T>(d_sz, x_sz);

      for (int j = 0; j < x_sz; ++j) {
        for (int i = 0; i < d_sz; ++i) {
          w_gamma_up_[i][j] = (T)0.0;
          w_gamma_down_[i][j] = (T)0.0;
          w_reference_[i][j] = (T)0.0;
        }
      },
      /* dtor*/
      Array_2D_Free<T>(w_gamma_down_);
      Array_2D_Free<T>(w_gamma_up_);
      Array_2D_Free<T>(w_reference_);
      ,
      /* copy */
      for (int j = 0; j < other.x_size_; ++j) {
        for (int i = 0; i < other.d_size_; ++i) {
          w_gamma_down_[i][j] = other.w_gamma_down_[i][j];
          w_gamma_up_[i][j] = other.w_gamma_up_[i][j];
          w_reference_[i][j] = other.w_reference_[i][j];
        }
      },
      /* move assignment */
      w_gamma_down_ = other.w_gamma_down_;
      w_gamma_up_ = other.w_gamma_up_;
      w_reference_ = other.w_reference_;

      other.w_gamma_down_ = nullptr;
      other.w_gamma_up_ = nullptr;
      other.w_reference_ = nullptr;
      ,
      /* swap*/
      swap(a.w_gamma_up_, b.w_gamma_up_);
      swap(a.w_gamma_down_, b.w_gamma_down_);
      swap(a.w_reference_, b.w_reference_);
      ,
      /* dp names*/
      names.push_back(std::string("gamma_up"));
      names.push_back(std::string("gamma_down"));
      names.push_back(std::string("reference"));
      ,
      /* dp2vec body*/
      int n_prev = (int)names.size();
      int size = this->x_size_ * this->d_size_;

      for (int i = 0; i < size; ++i) {
        data_ptrs[n_prev][i] = w_gamma_up_[0][i];
        data_ptrs[n_prev + 1][i] = w_gamma_down_[0][i];
        data_ptrs[n_prev + 2][i] = w_reference_[0][i];
      },
      /* vec2dp body*/
      int n_prev = (int)names.size();
      int size = this->x_size_ * this->d_size_;

      for (int i = 0; i < size; ++i) {
        w_gamma_up_[0][i] = data_ptrs[n_prev][i];
        w_gamma_down_[0][i] = data_ptrs[n_prev + 1][i];
        w_reference_[0][i] = data_ptrs[n_prev + 2][i];
      },
      /*invert copy DP */
      T **gamma_down = rpu->getGammaDown();
      T **gamma_up = rpu->getGammaUp();
      for (int j = 0; j < this->x_size_; ++j) {
        for (int i = 0; i < this->d_size_; ++i) {
          w_gamma_down_[i][j] = gamma_up[i][j];
          w_gamma_up_[i][j] = gamma_down[i][j];
          w_reference_[i][j] = -w_reference_[i][j];
        }
      }

  );

  void printDP(int x_count, int d_count) const override;

  inline T **getReference() const { return w_reference_; };
  inline T **getGammaUp() const { return w_gamma_up_; };
  inline T **getGammaDown() const { return w_gamma_down_; };

  void doSparseUpdate(
      T **weights, int i, const int *x_signed_indices, int x_count, int d_sign, RNG<T> *rng)
      override;
  void doDenseUpdate(T **weights, int *coincidences, RNG<T> *rng) override;

private:
  T **w_gamma_up_ = nullptr;
  T **w_gamma_down_ = nullptr;
  T **w_reference_ = nullptr;
};
} // namespace RPU
