/**
 * (C) Copyright 2020, 2021, 2022 IBM. All Rights Reserved.
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
#include "rpu_JART_v1b_device.h"
#include "rpucuda_pulsed_device.h"
#include <memory>

// #define DEVICE_PARAMETER_COUNT 53
#define DEVICE_PARAMETER_COUNT 39

namespace RPU {

template <typename T> class JARTv1bRPUDeviceCuda : public PulsedRPUDeviceCuda<T> {

public:
  BUILD_PULSED_DEVICE_CONSTRUCTORS_CUDA(
      JARTv1bRPUDeviceCuda,
      JARTv1bRPUDevice,
      /*ctor body*/
      dev_ldet_A = RPU::make_unique<CudaArray<float>>(this->context_, 2 * this->size_);
      dev_device_parameters = RPU::make_unique<CudaArray<T>>(this->context_, DEVICE_PARAMETER_COUNT);
      ,
      /*dtor body*/
      ,
      /*copy body*/
      dev_ldet_A->assign(*other.dev_ldet_A);
      dev_device_parameters->assign(*other.dev_device_parameters);
      ,
      /*move assigment body*/
      dev_ldet_A = std::move(other.dev_ldet_A);
      dev_device_parameters = std::move(other.dev_device_parameters);
      ,
      /*swap body*/
      swap(a.dev_ldet_A, b.dev_ldet_A);
      swap(a.dev_device_parameters, b.dev_device_parameters);
      ,
      /*host copy from cpu (rpu_device). Parent device params are copyied automatically*/
      int d_size = this->d_size_;
      int x_size = this->x_size_;
      T **device_specific_ldet_cuda = rpu_device.getldet();
      T **device_specific_A_cuda = rpu_device.getA();
      float *tmp_ldet_A = new float[2 * this->size_];

      for (int i = 0; i < d_size; ++i) {
        for (int j = 0; j < x_size; ++j) {
          int kk = j * (d_size * 2) + 2 * i;
          // tmp_slope[kk] = w_slope_down[i][j];
          // tmp_slope[kk + 1] = w_slope_up[i][j];
          tmp_ldet_A[kk] = device_specific_ldet_cuda[i][j];
          tmp_ldet_A[kk + 1] = device_specific_A_cuda[i][j];
        }
      }
      dev_ldet_A->assign(tmp_ldet_A);

      const auto &par = getPar();
      T *tmp_global_pars = new T[DEVICE_PARAMETER_COUNT]();
      // tmp_global_pars[0] = par.read_voltage;
      // tmp_global_pars[1] = par.pulse_voltage_SET;
      // tmp_global_pars[2] = par.pulse_voltage_RESET;
      // tmp_global_pars[3] = par.pulse_length;
      // tmp_global_pars[4] = par.base_time_step;
      // tmp_global_pars[5] = par.alpha0;
      // tmp_global_pars[6] = par.alpha1;
      // tmp_global_pars[7] = par.alpha2;
      // tmp_global_pars[8] = par.alpha3;
      // tmp_global_pars[9] = par.beta0;
      // tmp_global_pars[10] = par.beta1;
      // tmp_global_pars[11] = par.c0;
      // tmp_global_pars[12] = par.c1;
      // tmp_global_pars[13] = par.c2;
      // tmp_global_pars[14] = par.c3;
      // tmp_global_pars[15] = par.d0;
      // tmp_global_pars[16] = par.d1;
      // tmp_global_pars[17] = par.d2;
      // tmp_global_pars[18] = par.d3;
      // tmp_global_pars[19] = par.f0;
      // tmp_global_pars[20] = par.f1;
      // tmp_global_pars[21] = par.f2;
      // tmp_global_pars[22] = par.f3;
      // tmp_global_pars[23] = par.g0;
      // tmp_global_pars[24] = par.g1;
      // tmp_global_pars[25] = par.h0;
      // tmp_global_pars[26] = par.h1;
      // tmp_global_pars[27] = par.h2;
      // tmp_global_pars[28] = par.h3;
      // tmp_global_pars[29] = par.j_0;
      // tmp_global_pars[30] = par.k0;
      // tmp_global_pars[31] = par.T0;
      // tmp_global_pars[32] = par.un;
      // tmp_global_pars[33] = par.Ndiscmin;
      // tmp_global_pars[34] = par.Nplug;
      // tmp_global_pars[35] = par.a;
      // tmp_global_pars[36] = par.ny0;
      // tmp_global_pars[37] = par.dWa;
      // tmp_global_pars[38] = par.Rth0;
      // tmp_global_pars[39] = par.lcell;
      // tmp_global_pars[40] = par.Rtheff_scaling;
      // tmp_global_pars[41] = par.RseriesTiOx;
      // tmp_global_pars[42] = par.R0;
      // tmp_global_pars[43] = par.Rthline;
      // tmp_global_pars[44] = par.alphaline;
      // tmp_global_pars[45] = par.current_min;
      // tmp_global_pars[46] = par.current_max;
      // tmp_global_pars[47] = par.Ndisc_min_bound;
      // tmp_global_pars[48] = par.Ndisc_max_bound;
      // tmp_global_pars[49] = par.Ndiscmax_std;
      // tmp_global_pars[50] = par.Ndiscmin_std;
      // tmp_global_pars[51] = par.ldet_std;
      // tmp_global_pars[52] = par.rdet_std;

      
      tmp_global_pars[0] = par.read_voltage;
      tmp_global_pars[1] = par.pulse_voltage_SET;
      tmp_global_pars[2] = par.pulse_voltage_RESET;
      tmp_global_pars[3] = par.pulse_length;
      tmp_global_pars[4] = par.base_time_step;
      tmp_global_pars[5] = par.alpha_SET;
      tmp_global_pars[6] = par.beta_SET;
      tmp_global_pars[7] = par.c_SET;
      tmp_global_pars[8] = par.d_SET;
      tmp_global_pars[9] = par.f_SET;
      tmp_global_pars[10] = par.g_RESET;
      tmp_global_pars[11] = par.h_RESET;
      tmp_global_pars[12] = par.g_read;
      tmp_global_pars[13] = par.h_read;
      tmp_global_pars[14] = par.j_0;
      tmp_global_pars[15] = par.k0;
      tmp_global_pars[16] = par.T0;
      tmp_global_pars[17] = par.Ndiscmin;
      tmp_global_pars[18] = par.Nplug;
      tmp_global_pars[19] = par.a_ny0;
      tmp_global_pars[20] = par.dWa;
      tmp_global_pars[21] = par.Rth_negative;
      tmp_global_pars[22] = par.Rth_positive;
      tmp_global_pars[23] = par.RseriesTiOx;
      tmp_global_pars[24] = par.R0;
      tmp_global_pars[25] = par.V_series_coefficient;
      tmp_global_pars[26] = par.V_disk_coefficient;
      tmp_global_pars[27] = par.gamma_coefficient;
      tmp_global_pars[28] = par.lcell;
      tmp_global_pars[29] = par.current_min;
      tmp_global_pars[30] = par.current_to_weight_ratio;
      tmp_global_pars[31] = par.weight_to_current_ratio;
      tmp_global_pars[32] = par.w_min;
      tmp_global_pars[33] = par.Ndisc_min_bound;
      tmp_global_pars[34] = par.Ndisc_max_bound;
      tmp_global_pars[35] = par.Ndiscmax_std;
      tmp_global_pars[36] = par.Ndiscmin_std;
      tmp_global_pars[37] = par.ldet_std;
      tmp_global_pars[38] = par.rdet_std;

      dev_device_parameters = nullptr;
      dev_device_parameters = RPU::make_unique<CudaArray<T>>(this->context_, DEVICE_PARAMETER_COUNT, tmp_global_pars);

      this->context_->synchronize();
      delete[] tmp_ldet_A;
      delete[] tmp_global_pars;
      
      // Overwrite par4
      // copy RPU to device variables
      int size = x_size * d_size;
      float *tmp = new float[4 * size];
      T *tmp_Ndisc = new T[size];

      T *w_min_bound = rpu_device.getMinBound()[0];
      T *w_max_bound = rpu_device.getMaxBound()[0];
      T *device_specific_Ndiscmin_cuda = rpu_device.getNdiscmin()[0];
      T *device_specific_Ndiscmax_cuda = rpu_device.getNdiscmax()[0];

      T *Ndiscs = rpu_device.getPersistentWeights()[0];

      for (int i = 0; i < d_size; ++i) {
        for (int j = 0; j < x_size; ++j) {

          int l = i * (x_size) + j;
          // transposed: col major required by cuBLAS .. linear arangmenet for now
          int k = j * (d_size * 4) + 4 * i;
          tmp[k] = w_min_bound[l];
          tmp[k + 1] = device_specific_Ndiscmin_cuda[l];
          tmp[k + 2] = w_max_bound[l];
          tmp[k + 3] = device_specific_Ndiscmax_cuda[l];

          int l_t = j * (d_size) + i;
          tmp_Ndisc[l_t] = Ndiscs[l];
        }
      }

      this->dev_4params_->assign(tmp);
      this->dev_persistent_weights_ = RPU::make_unique<CudaArray<T>>(this->context_, size);
      this->dev_persistent_weights_->assign(tmp_Ndisc);

      this->context_->synchronize();

      delete[] tmp;
      delete[] tmp_Ndisc;
      );

  pwukpvec_t<T> getUpdateKernels(
      int m_batch,
      int nK32,
      int use_bo64,
      bool out_trans,
      const PulsedUpdateMetaParameter<T> &up) override;
  T *getGlobalParamsData() override { return dev_device_parameters->getData(); };
  float *get2ParamsData() override { return dev_ldet_A->getData(); };
  T *get1ParamsData() override {
    return this->dev_persistent_weights_->getData();
  };
  T getWeightGranularityNoise() const override {
    return PulsedRPUDeviceCuda<T>::getWeightGranularityNoise();
  }
  // implement abstract functions
  void decayWeights(T *dev_weights, bool bias_no_decay) override;
  void decayWeights(T *dev_weights, T alpha, bool bias_no_decay) override;
  void driftWeights(T *dev_weights, T time_since_epoch) override;
  void diffuseWeights(T *dev_weights) override;
  void clipWeights(T *dev_weights, T clip) override;
  void resetCols(T *dev_weights, int start_col, int n_cols, T reset_prob) override;
  void resetAt(T *dev_weights, const char *dev_non_zero_msk) override;
  void applyWeightUpdate(T *dev_weights, T *dw_and_current_weight_out) override;

private:
  std::unique_ptr<CudaArray<float>> dev_ldet_A = nullptr;
  std::unique_ptr<CudaArray<T>> dev_device_parameters = nullptr;
};

} // namespace RPU
