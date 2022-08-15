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

#include "rng.h"
#include "rpu_pulsed_device.h"
#define _USE_MATH_DEFINES
#include <math.h>
// physical constants do not change!
#define PHYSICAL_PARAMETER_e 1.602e-19 								// elementary charge [C]
#define PHYSICAL_PARAMETER_kb 1.3807e-23								// Boltzman's constant  [VAs/K]
#define PHYSICAL_PARAMETER_zvo 2									// oxygen vacancy charge number
#define PHYSICAL_PARAMETER_eps_0 8.854e-12				      				// vacuum permittivity [As/Vm]

namespace RPU {

template <typename T> class JARTv1bRPUDevice;

BUILD_PULSED_DEVICE_META_PARAMETER(
    JARTv1b,
    /*implements*/
    DeviceUpdateType::JARTv1b,
    /*parameter def*/ 
    T alpha0 = (T) 4.81951e-5;
    // T alpha1 = (T) 2.4006e-6;
    T alpha2 = (T) 1.03685;
    T alpha3 = (T) 0.34567;
    T alpha1 = (T) alpha0*exp(-alpha2/alpha3);
    T beta0 = (T) 7.0526e-4;
    T beta1 = (T) 4.2383e-5;
    T c0 = (T) 4.004;
    T c1 = (T) 2.8646;
    T c2 = (T) 4.2125;
    T c3 = (T) 1.4134;
    T d0 = (T) 6.6103;
    T d1 = (T) 1.4524;
    T d2 = (T) 7.4235;
    T d3 = (T) 4.0585;
    T f0 = (T) 6.326e-4;
    T f1 = (T) 1.4711;
    T f2 = (T) 0.5199;
    T f3 = (T) 1.561;
    T g0 = (T) 4.84e-3;
    T g1 = (T) 0.1353;
    T h0 = (T) 5.548;
    T h1 = (T) 6.8648;
    T h2 = (T) 51.586;
    T h3 = (T) 0.36;
    T j_0 = (T) 1.054;
    T k0 = (T) 1.0526;
    // Fitting Parameters for original model
    T T0 = (T) 293;								 	// ambient temperature [K] 
    T un = (T) 4e-6; // from [1e-6:1e-5];				// electron mobility [m^2/Vs]
    T Ndiscmax = (T) 20*1e26; // from [0.001:1100];				// maximum oxygen vacancy concentration in the disc[10^26/m^3]
    T Ndiscmin = (T) 0.008*1e26; // from [0.0001:100];			// minimum oxygen vacancy concentration in the disc [10^26/m^3]
    T Nplug = (T) 20*1e26; // from [0.001:100];					// oxygen vacancy concentration in the plug [10^26/m^3]
    T a = (T) 0.25e-9; // from [0.1e-9:1e-9];					// ion hopping distance [m]
    T ny0 = (T) 2e13; // from [1e10:1e14];					// attemp frequenzy [Hz]
    T dWa = (T) 1.35; // from [0.8:1.5];					// activation energy [eV]
    T Rth0 = (T) 15.72e6; // from [1e6:20e6];					// thermal resistance of the Hafnium Oxide [K/W]
    T rdet = (T) 45e-9; // from [5e-9:100e-9];				// radius of the filament area [m]
    T lcell = (T) 3*1e-9; // from [2:5];							// length of disc and plug region [m]
    T ldet = (T) 0.4*1e-9; // from [0.1:5]; 					// length of the disc region [m]
    T Rtheff_scaling = (T) 0.27; // from [0.1:1];				// scaling factor for gradual RESET 
    T RseriesTiOx = (T) 650; // from [100:200000];			// series resistance of the TiOx layer [Ohm]
    T R0 = (T) 719.2437;									// line resistance for a current of 0 A [Ohm]
    T Rthline = (T) 90471.47;							// thermal resistance of the lines [W/K]
    T alphaline = (T) 3.92e-3;							// temperature coefficient of the lines [1/K]
    T read_voltage = (T) 0.2;
    T pulse_voltage_SET = (T) -0.342;
    T pulse_voltage_RESET = (T) 0.7065;
    T pulse_length = (T) 1e-6;
    T base_time_step = (T) 1e-8;
    T Ndisc_min_bound = (T) 0.06*1e26;
    T Ndisc_max_bound = (T) 1.9897452127440086504e26;
    T current_min =  (T) (-g0*(exp(-g1*read_voltage)-1))/(pow((1+(h0+h1*read_voltage+h2*exp(-h3*read_voltage))*pow((Ndisc_min_bound/Ndiscmin),(-j_0))),(1/k0)));
    T current_max =  (T) (-g0*(exp(-g1*read_voltage)-1))/(pow((1+(h0+h1*read_voltage+h2*exp(-h3*read_voltage))*pow((Ndisc_max_bound/Ndiscmin),(-j_0))),(1/k0)));
    T w_min = (T)-0.6;
    T w_min_dtod = (T)0.3;
    T w_max = (T)0.6;
    T w_max_dtod = (T)0.3;
    T Ninit = pow(((pow(((-g0*(exp(-g1*read_voltage)-1))/(((0-w_min)/(w_max-w_min))*(current_max-current_min)+current_min)), k0)-1)/(h0+h1*read_voltage+h2*exp(-h3*read_voltage))),(1/-j_0))*Ndiscmin; // from [0.0001:1000];				// initial oxygen vacancy concentration in the disc [10^26/m^3]
    T Ndiscmax_dtod = (T) 0;							// 
    T Ndiscmin_dtod = (T) 0;							//
    T ldet_dtod = (T) 0;							//
    T rdet_dtod = (T) 0;							//
    T Ndiscmax_std = (T) 0;							// 
    T Ndiscmin_std = (T) 0;							//
    T ldet_std = (T) 0;							//
    T rdet_std = (T) 0;							//

    T write_noise_std_real = (T)0.0;
    T write_noise_std = (T)1.0;
    ,
    /*print body*/

    ss << "\t alpha0:\t\t" << alpha0 << std::endl;
    ss << "\t alpha1:\t\t" << alpha1 << std::endl;
    ss << "\t alpha2:\t\t" << alpha2 << std::endl;
    ss << "\t alpha3:\t\t" << alpha3 << std::endl;
    ss << "\t beta0:\t\t\t" << beta0 << std::endl;
    ss << "\t beta1:\t\t\t" << beta1 << std::endl;
    ss << "\t c0:\t\t\t" << c0 << std::endl;
    ss << "\t c1:\t\t\t" << c1 << std::endl;
    ss << "\t c2:\t\t\t" << c2 << std::endl;
    ss << "\t c3:\t\t\t" << c3 << std::endl;
    ss << "\t d0:\t\t\t" << d0 << std::endl;
    ss << "\t d1:\t\t\t" << d1 << std::endl;
    ss << "\t d2:\t\t\t" << d2 << std::endl;
    ss << "\t d3:\t\t\t" << d3 << std::endl;
    ss << "\t f0:\t\t\t" << f0 << std::endl;
    ss << "\t f1:\t\t\t" << f1 << std::endl;
    ss << "\t f2:\t\t\t" << f2 << std::endl;
    ss << "\t f3:\t\t\t" << f3 << std::endl;
    ss << "\t g0:\t\t\t" << g0 << std::endl;
    ss << "\t g1:\t\t\t" << g1 << std::endl;
    ss << "\t h0:\t\t\t" << h0 << std::endl;
    ss << "\t h1:\t\t\t" << h1 << std::endl;
    ss << "\t h2:\t\t\t" << h2 << std::endl;
    ss << "\t h3:\t\t\t" << h3 << std::endl;
    ss << "\t j0:\t\t\t" << j_0 << std::endl;
    ss << "\t k0:\t\t\t" << k0 << std::endl;
    
    ss << "\t T0:\t\t\t" << T0 << std::endl;
    ss << "\t un:\t\t\t" << un << std::endl;
    ss << "\t Nplug:\t\t\t" << Nplug << std::endl;
    ss << "\t a:\t\t\t" << a << std::endl;
    ss << "\t ny0:\t\t\t" << ny0 << std::endl;
    ss << "\t dWa:\t\t\t" << dWa << std::endl;
    ss << "\t Rth0:\t\t\t" << Rth0 << std::endl;
    ss << "\t rdet:\t\t\t" << rdet << std::endl;
    ss << "\t lcell:\t\t\t" << lcell << std::endl;
    ss << "\t ldet:\t\t\t" << ldet << std::endl;
    ss << "\t Rtheff_scaling:\t" << Rtheff_scaling << std::endl;
    ss << "\t RseriesTiOx:\t\t" << RseriesTiOx << std::endl;
    ss << "\t R0:\t\t\t" << R0 << std::endl;
    ss << "\t Rthline:\t\t" << Rthline << std::endl;
    ss << "\t alphaline:\t\t" << alphaline << std::endl;
    ss << "\t read_voltage:\t\t" << read_voltage << std::endl;
    ss << "\t pulse_voltage_SET:\t" << pulse_voltage_SET << std::endl;
    ss << "\t pulse_voltage_RESET:\t" << pulse_voltage_RESET << std::endl;
    ss << "\t pulse_length:\t\t" << pulse_length << std::endl;
    ss << "\t base_time_step:\t" << base_time_step << std::endl;
    ss << "\t Ndisc_min_bound:\t" << Ndisc_min_bound << std::endl;
    ss << "\t Ndisc_max_bound:\t" << Ndisc_max_bound << std::endl;
    ss << "\t current_min:\t" << current_min << std::endl;
    ss << "\t current_max:\t" << current_max << std::endl;
    ss << "\t Ndiscmax_dtod:\t\t" << Ndiscmax_dtod << std::endl;
    ss << "\t Ndiscmin_dtod:\t\t" << Ndiscmin_dtod << std::endl;
    ss << "\t ldet_dtod:\t\t" << ldet_dtod << std::endl;
    ss << "\t rdet_dtod:\t\t" << rdet_dtod << std::endl;
    ss << "\t Ndiscmax_std:\t\t" << Ndiscmax_std << std::endl;
    ss << "\t Ndiscmin_std:\t\t" << Ndiscmin_std << std::endl;
    ss << "\t ldet_std:\t\t" << ldet_std << std::endl;
    ss << "\t rdet_std:\t\t" << rdet_std << std::endl;
    ss << "\t write_noise_std:\t\t" << write_noise_std_real << std::endl;
    ,
    /* calc weight granularity body */
    return this->dw_min;
    ,
    /*Add*/
    );

template <typename T> class JARTv1bRPUDevice : public PulsedRPUDevice<T> {

  BUILD_PULSED_DEVICE_CONSTRUCTORS(
      JARTv1bRPUDevice,
      /* ctor*/
      int x_sz = this->x_size_;
      int d_sz = this->d_size_;

      device_specific_Ndiscmax = Array_2D_Get<T>(d_sz, x_sz);
      device_specific_Ndiscmin = Array_2D_Get<T>(d_sz, x_sz);
      device_specific_ldet = Array_2D_Get<T>(d_sz, x_sz);
      device_specific_A = Array_2D_Get<T>(d_sz, x_sz);

      for (int j = 0; j < x_sz; ++j) {
        for (int i = 0; i < d_sz; ++i) {
          device_specific_Ndiscmax[i][j] = (T)0.0;
          device_specific_Ndiscmin[i][j] = (T)0.0;
          device_specific_ldet[i][j] = (T)0.0;
          device_specific_A[i][j] = (T)0.0;
        }
      },
      /* dtor*/
      Array_2D_Free<T>(device_specific_Ndiscmax);
      Array_2D_Free<T>(device_specific_Ndiscmin);
      Array_2D_Free<T>(device_specific_ldet);
      Array_2D_Free<T>(device_specific_A);
      ,
      /* copy */
      for (int j = 0; j < other.x_size_; ++j) {
        for (int i = 0; i < other.d_size_; ++i) {
          device_specific_Ndiscmax[i][j] = other.device_specific_Ndiscmax[i][j];
          device_specific_Ndiscmin[i][j] = other.device_specific_Ndiscmin[i][j];
          device_specific_ldet[i][j] = other.device_specific_ldet[i][j];
          device_specific_A[i][j] = other.device_specific_A[i][j];
        }
      },
      /* move assignment */
      device_specific_Ndiscmax = other.device_specific_Ndiscmax;
      device_specific_Ndiscmax = other.device_specific_Ndiscmax;
      device_specific_ldet = other.device_specific_ldet;
      device_specific_A = other.device_specific_A;

      other.device_specific_Ndiscmax = nullptr;
      other.device_specific_Ndiscmin = nullptr;
      other.device_specific_ldet = nullptr;
      other.device_specific_A = nullptr;
      ,
      /* swap*/
      swap(a.device_specific_Ndiscmax, b.device_specific_Ndiscmax);
      swap(a.device_specific_Ndiscmin, b.device_specific_Ndiscmin);
      swap(a.device_specific_ldet, b.device_specific_ldet);
      swap(a.device_specific_A, b.device_specific_A);
      ,
      /* dp names*/
      names.push_back(std::string("device_specific_Ndiscmax"));
      names.push_back(std::string("device_specific_Ndiscmin"));
      names.push_back(std::string("device_specific_ldet"));
      names.push_back(std::string("device_specific_A"));
      ,
      /* dp2vec body*/
      int n_prev = (int)names.size();
      int size = this->x_size_ * this->d_size_;

      for (int i = 0; i < size; ++i) {
        data_ptrs[n_prev][i] = device_specific_Ndiscmax[0][i];
        data_ptrs[n_prev + 1][i] = device_specific_Ndiscmin[0][i];
        data_ptrs[n_prev + 2][i] = device_specific_ldet[0][i];
        data_ptrs[n_prev + 3][i] = device_specific_A[0][i];
      },
      /* vec2dp body*/
      int n_prev = (int)names.size();
      int size = this->x_size_ * this->d_size_;

      for (int i = 0; i < size; ++i) {
        device_specific_Ndiscmax[0][i] = data_ptrs[n_prev][i];
        device_specific_Ndiscmin[0][i] = data_ptrs[n_prev + 1][i];
        device_specific_ldet[0][i] = data_ptrs[n_prev + 2][i];
        device_specific_A[0][i] = data_ptrs[n_prev + 3][i];
      }


      ,
      /*invert copy DP */
      for (int j = 0; j < this->x_size_; ++j) {
        for (int i = 0; i < this->d_size_; ++i) {
          std::swap(device_specific_Ndiscmax[i][j], device_specific_Ndiscmin[i][j]);
        }
      }
      // Todo: if device specific Ndisc bounds: Remap wmax&wmin to Ndisc max_bound&min_bound 

  );


  void printDP(int x_count, int d_count) const override;

  inline T **getNdiscmax() const { return device_specific_Ndiscmax; };
  inline T **getNdiscmin() const { return device_specific_Ndiscmin; };
  inline T **getldet() const { return device_specific_ldet; };
  inline T **getA() const { return device_specific_A; };

  void decayWeights(T **weights, bool bias_no_decay) override;
  void decayWeights(T **weights, T alpha, bool bias_no_decay) override;
  void driftWeights(T **weights, T time_since_last_call, RNG<T> &rng) override;
  void diffuseWeights(T **weights, RNG<T> &rng) override;
  void clipWeights(T **weights, T add_clip) override;
  bool onSetWeights(T **weights) override;
  void
  resetCols(T **weights, int start_col, int n_cols, T reset_prob, RealWorldRNG<T> &rng) override;
  void resetAtIndices(T **weights, std::vector<int> x_major_indices, RealWorldRNG<T> &rng) override;
  
  
  void doSparseUpdate(
      T **weights, int i, const int *x_signed_indices, int x_count, int d_sign, RNG<T> *rng)
      override;
  void doDenseUpdate(T **weights, int *coincidences, RNG<T> *rng) override;
  

private:
  T **device_specific_Ndiscmax = nullptr;
  T **device_specific_Ndiscmin = nullptr;
  T **device_specific_ldet = nullptr;
  T **device_specific_A = nullptr;
};

} // namespace RPU

