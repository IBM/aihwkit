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

#include "rpu_pulsed_device.h"
#include "utility_functions.h"

namespace RPU {

template <typename T> class HiddenStepRPUDevice;

BUILD_PULSED_DEVICE_META_PARAMETER(
    HiddenStep,
    /*implements*/
    DeviceUpdateType::HiddenStep,
    /*parameter def*/
    T hs_up_down = (T)0.0;
    T hs_up_down_dtod = (T)0.01;
    T hs_dw_min_dtod = (T)0.3;
    T hs_dw_min_std = (T)0.0; /* set 0 default to allow for easier debugging (see test)*/

    T hs_dw_min = (T)1.0;        /* always 1.0 */
    T hs_hidden_states = (T)2.0; /* set small number in default for debugging*/
    ,
    /*print body*/
    ss << "\t hs_hidden_states:\t\t" << hs_hidden_states << std::endl;
    ss << "\t hs_dw_min:\t [fixed to 1/hidden_states]" << std::endl;
    ss << "\t hs_dw_min_ctoc:\t" << hs_dw_min_std << std::endl;
    ss << "\t hs_dw_min_dtod:\t" << hs_dw_min_dtod << std::endl;
    ss << "\t hs_up_down:\t\t" << hs_up_down << std::endl;
    ss << "\t hs_up_down_dtod:\t" << hs_up_down_dtod << std::endl;
    ,
    /* calc weight granularity body */
    return this->dw_min;
    ,
    /*Add*/
    void initialize() override {
      if (!this->_par_initialized) {
        PulsedRPUDeviceMetaParameter<T>::initialize();
        /* thres always at 1, so dw goes with hidden state*/
        hs_dw_min = (T)1.0 / hs_hidden_states;
        this->dw_min *= hs_hidden_states; /* scale it with hidden state*/
      }
    });

template <typename T> class HiddenStepRPUDevice : public PulsedRPUDevice<T> {

  BUILD_PULSED_DEVICE_CONSTRUCTORS(
      HiddenStepRPUDevice,
      /* ctor*/
      size_t x_sz = this->x_size_;
      size_t d_sz = this->d_size_;

      hidden_weights_ = Array_2D_Get<T>(d_sz, x_sz);
      hs_scale_up_ = Array_2D_Get<T>(d_sz, x_sz);
      hs_scale_down_ = Array_2D_Get<T>(d_sz, x_sz);

      for (size_t i = 0; i < d_sz; ++i) {
        for (size_t j = 0; j < x_sz; ++j) {
          hidden_weights_[i][j] = (T)0.0;
        }
      },
      /* dtor*/
      Array_2D_Free<T>(hs_scale_down_);
      Array_2D_Free<T>(hs_scale_up_);
      Array_2D_Free<T>(hidden_weights_);
      ,
      /* copy */
      for (size_t j = 0; j < (size_t)other.x_size_; ++j) {
        for (size_t i = 0; i < (size_t)other.d_size_; ++i) {
          hs_scale_up_[i][j] = other.hs_scale_up_[i][j];
          hs_scale_down_[i][j] = other.hs_scale_down_[i][j];
          hidden_weights_[i][j] = other.hidden_weights_[i][j];
        }
      },
      /* move assignment */
      hs_scale_down_ = other.hs_scale_down_;
      hs_scale_up_ = other.hs_scale_up_;
      hidden_weights_ = other.hidden_weights_;

      other.hidden_weights_ = nullptr;
      other.hs_scale_up_ = nullptr;
      other.hs_scale_down_ = nullptr;
      ,
      /* swap*/
      swap(a.hs_scale_down_, b.hs_scale_down_);
      swap(a.hs_scale_up_, b.hs_scale_up_);
      swap(a.hidden_weights_, b.hidden_weights_);
      ,
      /* dp names*/
      names.push_back(std::string("hs_scale_up"));
      names.push_back(std::string("hs_scale_down"));
      names.push_back(std::string("hidden_weights"));
      ,
      /* dp2vec body*/
      size_t n_prev = names.size();
      size_t size = this->x_size_ * this->d_size_;

      for (size_t i = 0; i < size; ++i) {
        data_ptrs[n_prev][i] = hs_scale_up_[0][i];
        data_ptrs[n_prev + 1][i] = hs_scale_down_[0][i];
        data_ptrs[n_prev + 2][i] = hidden_weights_[0][i];
      },
      /* vec2dp body*/
      size_t n_prev = names.size();
      size_t size = this->x_size_ * this->d_size_;

      for (size_t i = 0; i < size; ++i) {
        hs_scale_up_[0][i] = data_ptrs[n_prev][i];
        hs_scale_down_[0][i] = data_ptrs[n_prev + 1][i];
        hidden_weights_[0][i] = data_ptrs[n_prev + 2][i];
      },
      /*invert copy DP */
      T **hs_scale_down = rpu->getHsScaleDown();
      T **hs_scale_up = rpu->getHsScaleUp();

      for (size_t j = 0; j < (size_t)this->x_size_; ++j) {
        for (size_t i = 0; i < (size_t)this->d_size_; ++i) {
          hs_scale_up_[i][j] = hs_scale_down[i][j];
          hs_scale_down_[i][j] = hs_scale_up[i][j];
        }
      });

  inline T **getHsScaleUp() const { return hs_scale_up_; };
  inline T **getHsScaleDown() const { return hs_scale_down_; };
  inline T **getHiddenWeights() const { return hidden_weights_; };

  void printDP(int x_count, int d_count) const override;
  void doSparseUpdate(
      T **weights, int i, const int *x_signed_indices, int x_count, int d_sign, RNG<T> *rng)
      override;
  void doDenseUpdate(T **weights, int *coincidences, RNG<T> *rng) override;

private:
  T **hidden_weights_ = nullptr;
  T **hs_scale_up_ = nullptr;
  T **hs_scale_down_ = nullptr;
};
} // namespace RPU
