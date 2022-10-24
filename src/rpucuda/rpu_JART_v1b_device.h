/**
 * (C) Copyright 2022 Forschungszentrum Jülich GmbH, Zhenming Yu. All Rights reserved.
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
// #define PHYSICAL_PARAMETER_kb 1.3807e-23								// Boltzman's constant  [VAs/K]
#define PHYSICAL_PARAMETER_kb_over_e 8.61860174781523e-05
#define PHYSICAL_PARAMETER_zvo 2									// oxygen vacancy charge number
#define PHYSICAL_PARAMETER_eps_0 8.854e-12				      				// vacuum permittivity [As/Vm]

namespace RPU {

template <typename T> class JARTv1bRPUDevice;

BUILD_PULSED_DEVICE_META_PARAMETER(
    JARTv1b,
    /*implements*/
    DeviceUpdateType::JARTv1b,
    /*parameter def*/ 
    T real_write_noise_std = (T) 0.0;
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
    T Ndiscmax = (T) 20; // from [0.001:1100];				// maximum oxygen vacancy concentration in the disc[10^26/m^3]
    T Ndiscmin = (T) 0.008; // from [0.0001:100];			// minimum oxygen vacancy concentration in the disc [10^26/m^3]
    T Nplug = (T) 20; // from [0.001:100];					// oxygen vacancy concentration in the plug [10^26/m^3]
    T a = (T) 0.25e-9; // from [0.1e-9:1e-9];					// ion hopping distance [m]
    T ny0 = (T) 2e13; // from [1e10:1e14];					// attemp frequenzy [Hz]
    T dWa = (T) 1.35; // from [0.8:1.5];					// activation energy [eV]
    T Rth0 = (T) 15.72e6; // from [1e6:20e6];					// thermal resistance of the Hafnium Oxide [K/W]
    T rdisc = (T) 45e-9; // from [5e-9:100e-9];				// radius of the filament area [m]
    T lcell = (T) 3*1e-9; // from [2:5];							// length of disc and plug region [m]
    T ldisc = (T) 0.4*1e-9; // from [0.1:5]; 					// length of the disc region [m]
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
    T Ndisc_min_bound = (T) 0.06;
    T Ndisc_max_bound = (T) 1.9897452127440086504;
    T w_min = (T)-0.6;
    T w_min_dtod = (T)0.3;
    T w_max = (T)0.6;
    T w_max_dtod = (T)0.3;
    T Ndiscmax_dtod = (T) 0.0;							// 
    T Ndiscmax_dtod_upper_bound = (T) 0.0;							// 
    T Ndiscmax_dtod_lower_bound = (T) 0.0;							// 
    T Ndiscmin_dtod = (T) 0.0;							//
    T Ndiscmin_dtod_upper_bound = (T) 0.0;							// 
    T Ndiscmin_dtod_lower_bound = (T) 0.0;							// 
    T ldisc_dtod = (T) 0.0;							//
    T ldisc_dtod_upper_bound = (T) 0.0;							// 
    T ldisc_dtod_lower_bound = (T) 0.0;							// 
    T rdisc_dtod = (T) 0.0;							//
    T rdisc_dtod_upper_bound = (T) 0.0;							// 
    T rdisc_dtod_lower_bound = (T) 0.0;							// 
    T Ndiscmax_std = (T) 0.0;							// 
    T Ndiscmax_ctoc_upper_bound_old = (T) 0.0;							// 
    T Ndiscmax_ctoc_lower_bound_old = (T) 0.0;							// 
    T Ndiscmax_ctoc_upper_bound = (T) 0.0;							// 
    T Ndiscmax_ctoc_lower_bound = (T) 0.0;							// 
    T Ndiscmin_std = (T) 0.0;							//
    T Ndiscmin_ctoc_upper_bound_old = (T) 0.0;							// 
    T Ndiscmin_ctoc_lower_bound_old = (T) 0.0;							// 
    T Ndiscmin_ctoc_upper_bound = (T) 0.0;							// 
    T Ndiscmin_ctoc_lower_bound = (T) 0.0;							// 
    T ldisc_std = (T) 0.0;							//
    T ldisc_std_slope = (T) 0.0;							//
    T ldisc_ctoc_upper_bound_old = (T) 0.0;							// 
    T ldisc_ctoc_lower_bound_old = (T) 0.0;							// 
    T ldisc_ctoc_upper_bound = (T) 0.0;							// 
    T ldisc_ctoc_lower_bound = (T) 0.0;							// 
    T rdisc_std = (T) 0.0;							//
    T rdisc_std_slope = (T) 0.0;							//
    T rdisc_ctoc_upper_bound_old = (T) 0.0;							// 
    T rdisc_ctoc_lower_bound_old = (T) 0.0;							// 
    T rdisc_ctoc_upper_bound = (T) 0.0;							// 
    T rdisc_ctoc_lower_bound = (T) 0.0;							// 
    bool enable_w_max_w_min_bounds = false;					// 
    T w_max_dtod_upper_bound = (T) 0.0;							// 
    T w_max_dtod_lower_bound = (T) 0.0;							// 
    T w_min_dtod_upper_bound = (T) 0.0;							// 
    T w_min_dtod_lower_bound = (T) 0.0;							// 

    
    T current_min =  (T) (-g0*(exp(-g1*read_voltage)-1))/(pow((1+(h0+h1*read_voltage+h2*exp(-h3*read_voltage))*pow((Ndisc_min_bound/Ndiscmin),(-j_0))),(1/k0)));
    T current_max =  (T) (-g0*(exp(-g1*read_voltage)-1))/(pow((1+(h0+h1*read_voltage+h2*exp(-h3*read_voltage))*pow((Ndisc_max_bound/Ndiscmin),(-j_0))),(1/k0)));
    T Ninit = pow(((pow(((-g0*(exp(-g1*read_voltage)-1))/(((0-w_min)/(w_max-w_min))*(current_max-current_min)+current_min)), k0)-1)/(h0+h1*read_voltage+h2*exp(-h3*read_voltage))),(1/-j_0))*Ndiscmin; // from [0.0001:1000];				// initial oxygen vacancy concentration in the disc [10^26/m^3]

    T alpha_SET = (T) ((alpha1+alpha0)/(1+exp(-(pulse_voltage_SET+alpha2)/alpha3)))-alpha0;
    T beta_SET = (T) (beta1*(1-exp(-pulse_voltage_SET)))-beta0*pulse_voltage_SET;
    T c_SET = (T) c2*exp(-pulse_voltage_SET/c3)+c1*pulse_voltage_SET-c0;
    T d_SET = (T) d2*exp(-pulse_voltage_SET/d3)+d1*pulse_voltage_SET-d0;
    T f_SET = (T) f0+((f1-f0)/(1+pow((-pulse_voltage_SET/f2),f3)));

    T g_RESET = (T) -g0*(exp(-g1*pulse_voltage_RESET)-1);
    T h_RESET = (T) h0+h1*pulse_voltage_RESET+h2*exp(-h3*pulse_voltage_RESET);

    T g_read = (T) -g0*(exp(-g1*read_voltage)-1);
    T h_read = (T) h0+h1*read_voltage+h2*exp(-h3*read_voltage);

    T Original_A = (T) M_PI*pow(rdisc,2.0);

    T Rth_negative_coefficient = (T) Rth0*Original_A;
    T Rth_positive_coefficient = (T) Rth0*Rtheff_scaling*Original_A;

    T V_series_coefficient = (T) R0*alphaline*R0*Rthline;
    T V_disk_coefficient = (T) PHYSICAL_PARAMETER_zvo*PHYSICAL_PARAMETER_e*un*1e26;

    T gamma_coefficient = (T) (PHYSICAL_PARAMETER_zvo*a)/(dWa*M_PI);
    T a_ny0 = (T) a*ny0;

    T current_to_weight_ratio = (T) (w_max-w_min)/(current_max-current_min);
    T weight_to_current_ratio = (T) (current_max-current_min)/(w_max-w_min);

    T write_noise_std = (T)1.0;
    T dw_min_dtod = (T)0.0;
    T dw_min_std = (T)0.0; // ctoc of pulse
    ,
    /*print body*/
    ss << "\t write noise std:\t" << real_write_noise_std << std::endl;
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
    ss << "\t rdisc:\t\t\t" << rdisc << std::endl;
    ss << "\t lcell:\t\t\t" << lcell << std::endl;
    ss << "\t ldisc:\t\t\t" << ldisc << std::endl;
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
    ss << "\t Ndiscmax_dtod_upper_bound:\t\t" << Ndiscmax_dtod_upper_bound << std::endl;
    ss << "\t Ndiscmax_dtod_lower_bound:\t\t" << Ndiscmax_dtod_lower_bound << std::endl;
    ss << "\t Ndiscmin_dtod:\t\t" << Ndiscmin_dtod << std::endl;
    ss << "\t Ndiscmin_dtod_upper_bound:\t\t" << Ndiscmin_dtod_upper_bound << std::endl;
    ss << "\t Ndiscmin_dtod_lower_bound:\t\t" << Ndiscmin_dtod_lower_bound << std::endl;
    ss << "\t ldisc_dtod:\t\t" << ldisc_dtod << std::endl;
    ss << "\t ldisc_dtod_upper_bound:\t\t" << ldisc_dtod_upper_bound << std::endl;
    ss << "\t ldisc_dtod_lower_bound:\t\t" << ldisc_dtod_lower_bound << std::endl;
    ss << "\t rdisc_dtod:\t\t" << rdisc_dtod << std::endl;
    ss << "\t rdisc_dtod_upper_bound:\t\t" << rdisc_dtod_upper_bound << std::endl;
    ss << "\t rdisc_dtod_lower_bound:\t\t" << rdisc_dtod_lower_bound << std::endl;
    ss << "\t Ndiscmax_std:\t\t" << Ndiscmax_std << std::endl;
    ss << "\t Ndiscmax_ctoc_upper_bound_old:\t\t" << Ndiscmax_ctoc_upper_bound_old << std::endl;
    ss << "\t Ndiscmax_ctoc_lower_bound_old:\t\t" << Ndiscmax_ctoc_lower_bound_old << std::endl;
    ss << "\t Ndiscmax_ctoc_upper_bound:\t\t" << Ndiscmax_ctoc_upper_bound << std::endl;
    ss << "\t Ndiscmax_ctoc_lower_bound:\t\t" << Ndiscmax_ctoc_lower_bound << std::endl;
    ss << "\t Ndiscmin_std:\t\t" << Ndiscmin_std << std::endl;
    ss << "\t Ndiscmin_ctoc_upper_bound_old:\t\t" << Ndiscmin_ctoc_upper_bound_old << std::endl;
    ss << "\t Ndiscmin_ctoc_lower_bound_old:\t\t" << Ndiscmin_ctoc_lower_bound_old << std::endl;
    ss << "\t Ndiscmin_ctoc_upper_bound:\t\t" << Ndiscmin_ctoc_upper_bound << std::endl;
    ss << "\t Ndiscmin_ctoc_lower_bound:\t\t" << Ndiscmin_ctoc_lower_bound << std::endl;
    ss << "\t ldisc_std:\t\t" << ldisc_std << std::endl;
    ss << "\t ldisc_std_slope:\t\t" << ldisc_std_slope << std::endl;
    ss << "\t ldisc_ctoc_upper_bound_old:\t\t" << ldisc_ctoc_upper_bound_old << std::endl;
    ss << "\t ldisc_ctoc_lower_bound_old:\t\t" << ldisc_ctoc_lower_bound_old << std::endl;
    ss << "\t ldisc_ctoc_upper_bound:\t\t" << ldisc_ctoc_upper_bound << std::endl;
    ss << "\t ldisc_ctoc_lower_bound:\t\t" << ldisc_ctoc_lower_bound << std::endl;
    ss << "\t rdisc_std:\t\t" << rdisc_std << std::endl;
    ss << "\t rdisc_std_slope:\t\t" << rdisc_std_slope << std::endl;
    ss << "\t rdisc_ctoc_upper_bound_old:\t\t" << rdisc_ctoc_upper_bound_old << std::endl;
    ss << "\t rdisc_ctoc_lower_bound_old:\t\t" << rdisc_ctoc_lower_bound_old << std::endl;
    ss << "\t rdisc_ctoc_upper_bound:\t\t" << rdisc_ctoc_upper_bound << std::endl;
    ss << "\t rdisc_ctoc_lower_bound:\t\t" << rdisc_ctoc_lower_bound << std::endl;
    ss << "\t enable_w_max_w_min_bounds:\t\t" << enable_w_max_w_min_bounds << std::endl;
    ss << "\t w_max_dtod_upper_bound:\t\t" << w_max_dtod_upper_bound << std::endl;
    ss << "\t w_max_dtod_lower_bound:\t\t" << w_max_dtod_lower_bound << std::endl;
    ss << "\t w_min_dtod_upper_bound:\t\t" << w_min_dtod_upper_bound << std::endl;
    ss << "\t w_min_dtod_lower_bound:\t\t" << w_min_dtod_lower_bound << std::endl;
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

      device_specific_Ndisc_max_bound = Array_2D_Get<T>(d_sz, x_sz);
      device_specific_Ndisc_min_bound = Array_2D_Get<T>(d_sz, x_sz);
      device_specific_Ndiscmax = Array_2D_Get<T>(d_sz, x_sz);
      device_specific_Ndiscmin = Array_2D_Get<T>(d_sz, x_sz);
      device_specific_ldisc = Array_2D_Get<T>(d_sz, x_sz);
      device_specific_A = Array_2D_Get<T>(d_sz, x_sz);
      device_specific_Ndiscmax_ctoc_upper_bound = Array_2D_Get<T>(d_sz, x_sz);
      device_specific_Ndiscmin_ctoc_upper_bound = Array_2D_Get<T>(d_sz, x_sz);
      device_specific_ldisc_ctoc_upper_bound = Array_2D_Get<T>(d_sz, x_sz);
      device_specific_A_ctoc_upper_bound = Array_2D_Get<T>(d_sz, x_sz);
      device_specific_Ndiscmax_ctoc_lower_bound = Array_2D_Get<T>(d_sz, x_sz);
      device_specific_Ndiscmin_ctoc_lower_bound = Array_2D_Get<T>(d_sz, x_sz);
      device_specific_ldisc_ctoc_lower_bound = Array_2D_Get<T>(d_sz, x_sz);
      device_specific_A_ctoc_lower_bound = Array_2D_Get<T>(d_sz, x_sz);

      for (int j = 0; j < x_sz; ++j) {
        for (int i = 0; i < d_sz; ++i) {
          device_specific_Ndisc_max_bound[i][j] = (T)0.0;
          device_specific_Ndisc_min_bound[i][j] = (T)0.0;
          device_specific_Ndiscmax[i][j] = (T)0.0;
          device_specific_Ndiscmin[i][j] = (T)0.0;
          device_specific_ldisc[i][j] = (T)0.0;
          device_specific_A[i][j] = (T)0.0;
          device_specific_Ndiscmax_ctoc_upper_bound[i][j] = (T)0.0;
          device_specific_Ndiscmin_ctoc_upper_bound[i][j] = (T)0.0;
          device_specific_ldisc_ctoc_upper_bound[i][j] = (T)0.0;
          device_specific_A_ctoc_upper_bound[i][j] = (T)0.0;
          device_specific_Ndiscmax_ctoc_lower_bound[i][j] = (T)0.0;
          device_specific_Ndiscmin_ctoc_lower_bound[i][j] = (T)0.0;
          device_specific_ldisc_ctoc_lower_bound[i][j] = (T)0.0;
          device_specific_A_ctoc_lower_bound[i][j] = (T)0.0;
        }
      },
      /* dtor*/
      Array_2D_Free<T>(device_specific_Ndisc_max_bound);
      Array_2D_Free<T>(device_specific_Ndisc_min_bound);
      Array_2D_Free<T>(device_specific_Ndiscmax);
      Array_2D_Free<T>(device_specific_Ndiscmin);
      Array_2D_Free<T>(device_specific_ldisc);
      Array_2D_Free<T>(device_specific_A);
      Array_2D_Free<T>(device_specific_Ndiscmax_ctoc_upper_bound);
      Array_2D_Free<T>(device_specific_Ndiscmin_ctoc_upper_bound);
      Array_2D_Free<T>(device_specific_ldisc_ctoc_upper_bound);
      Array_2D_Free<T>(device_specific_A_ctoc_upper_bound);
      Array_2D_Free<T>(device_specific_Ndiscmax_ctoc_lower_bound);
      Array_2D_Free<T>(device_specific_Ndiscmin_ctoc_lower_bound);
      Array_2D_Free<T>(device_specific_ldisc_ctoc_lower_bound);
      Array_2D_Free<T>(device_specific_A_ctoc_lower_bound);
      ,
      /* copy */
      for (int j = 0; j < other.x_size_; ++j) {
        for (int i = 0; i < other.d_size_; ++i) {
          device_specific_Ndisc_max_bound[i][j] = other.device_specific_Ndisc_max_bound[i][j];
          device_specific_Ndisc_min_bound[i][j] = other.device_specific_Ndisc_min_bound[i][j];
          device_specific_Ndiscmax[i][j] = other.device_specific_Ndiscmax[i][j];
          device_specific_Ndiscmin[i][j] = other.device_specific_Ndiscmin[i][j];
          device_specific_ldisc[i][j] = other.device_specific_ldisc[i][j];
          device_specific_A[i][j] = other.device_specific_A[i][j];
          device_specific_Ndiscmax_ctoc_upper_bound[i][j] = other.device_specific_Ndiscmax_ctoc_upper_bound[i][j];
          device_specific_Ndiscmin_ctoc_upper_bound[i][j] = other.device_specific_Ndiscmin_ctoc_upper_bound[i][j];
          device_specific_ldisc_ctoc_upper_bound[i][j] = other.device_specific_ldisc_ctoc_upper_bound[i][j];
          device_specific_A_ctoc_upper_bound[i][j] = other.device_specific_A_ctoc_upper_bound[i][j];
          device_specific_Ndiscmax_ctoc_lower_bound[i][j] = other.device_specific_Ndiscmax_ctoc_lower_bound[i][j];
          device_specific_Ndiscmin_ctoc_lower_bound[i][j] = other.device_specific_Ndiscmin_ctoc_lower_bound[i][j];
          device_specific_ldisc_ctoc_lower_bound[i][j] = other.device_specific_ldisc_ctoc_lower_bound[i][j];
          device_specific_A_ctoc_lower_bound[i][j] = other.device_specific_A_ctoc_lower_bound[i][j];
        }
      },
      /* move assignment */
      device_specific_Ndisc_max_bound = other.device_specific_Ndisc_max_bound;
      device_specific_Ndisc_min_bound = other.device_specific_Ndisc_min_bound;
      device_specific_Ndiscmax = other.device_specific_Ndiscmax;
      device_specific_Ndiscmin = other.device_specific_Ndiscmin;
      device_specific_ldisc = other.device_specific_ldisc;
      device_specific_A = other.device_specific_A;
      device_specific_Ndiscmax_ctoc_upper_bound = other.device_specific_Ndiscmax_ctoc_upper_bound;
      device_specific_Ndiscmin_ctoc_upper_bound = other.device_specific_Ndiscmin_ctoc_upper_bound;
      device_specific_ldisc_ctoc_upper_bound = other.device_specific_ldisc_ctoc_upper_bound;
      device_specific_A_ctoc_upper_bound = other.device_specific_A_ctoc_upper_bound;
      device_specific_Ndiscmax_ctoc_lower_bound = other.device_specific_Ndiscmax_ctoc_lower_bound;
      device_specific_Ndiscmin_ctoc_lower_bound = other.device_specific_Ndiscmin_ctoc_lower_bound;
      device_specific_ldisc_ctoc_lower_bound = other.device_specific_ldisc_ctoc_lower_bound;
      device_specific_A_ctoc_lower_bound = other.device_specific_A_ctoc_lower_bound;

      other.device_specific_Ndisc_max_bound = nullptr;
      other.device_specific_Ndisc_min_bound = nullptr;
      other.device_specific_Ndiscmax = nullptr;
      other.device_specific_Ndiscmin = nullptr;
      other.device_specific_ldisc = nullptr;
      other.device_specific_A = nullptr;
      other.device_specific_Ndiscmax_ctoc_upper_bound = nullptr;
      other.device_specific_Ndiscmin_ctoc_upper_bound = nullptr;
      other.device_specific_ldisc_ctoc_upper_bound = nullptr;
      other.device_specific_A_ctoc_upper_bound = nullptr;
      other.device_specific_Ndiscmax_ctoc_lower_bound = nullptr;
      other.device_specific_Ndiscmin_ctoc_lower_bound = nullptr;
      other.device_specific_ldisc_ctoc_lower_bound = nullptr;
      other.device_specific_A_ctoc_lower_bound = nullptr;
      ,
      /* swap*/
      swap(a.device_specific_Ndisc_max_bound, b.device_specific_Ndisc_max_bound);
      swap(a.device_specific_Ndisc_min_bound, b.device_specific_Ndisc_min_bound);
      swap(a.device_specific_Ndiscmax, b.device_specific_Ndiscmax);
      swap(a.device_specific_Ndiscmin, b.device_specific_Ndiscmin);
      swap(a.device_specific_ldisc, b.device_specific_ldisc);
      swap(a.device_specific_A, b.device_specific_A);
      swap(a.device_specific_Ndiscmax_ctoc_upper_bound, b.device_specific_Ndiscmax_ctoc_upper_bound);
      swap(a.device_specific_Ndiscmin_ctoc_upper_bound, b.device_specific_Ndiscmin_ctoc_upper_bound);
      swap(a.device_specific_ldisc_ctoc_upper_bound, b.device_specific_ldisc_ctoc_upper_bound);
      swap(a.device_specific_A_ctoc_upper_bound, b.device_specific_A_ctoc_upper_bound);
      swap(a.device_specific_Ndiscmax_ctoc_lower_bound, b.device_specific_Ndiscmax_ctoc_lower_bound);
      swap(a.device_specific_Ndiscmin_ctoc_lower_bound, b.device_specific_Ndiscmin_ctoc_lower_bound);
      swap(a.device_specific_ldisc_ctoc_lower_bound, b.device_specific_ldisc_ctoc_lower_bound);
      swap(a.device_specific_A_ctoc_lower_bound, b.device_specific_A_ctoc_lower_bound);
      ,
      /* dp names*/
      names.push_back(std::string("device_specific_Ndisc_max_bound"));
      names.push_back(std::string("device_specific_Ndisc_min_bound"));
      names.push_back(std::string("device_specific_Ndiscmax"));
      names.push_back(std::string("device_specific_Ndiscmin"));
      names.push_back(std::string("device_specific_ldisc"));
      names.push_back(std::string("device_specific_A"));
      names.push_back(std::string("device_specific_Ndiscmax_ctoc_upper_bound"));
      names.push_back(std::string("device_specific_Ndiscmin_ctoc_upper_bound"));
      names.push_back(std::string("device_specific_ldisc_ctoc_upper_bound"));
      names.push_back(std::string("device_specific_A_ctoc_upper_bound"));
      names.push_back(std::string("device_specific_Ndiscmax_ctoc_lower_bound"));
      names.push_back(std::string("device_specific_Ndiscmin_ctoc_lower_bound"));
      names.push_back(std::string("device_specific_ldisc_ctoc_lower_bound"));
      names.push_back(std::string("device_specific_A_ctoc_lower_bound"));
      ,
      /* dp2vec body*/
      int n_prev = (int)names.size();
      int size = this->x_size_ * this->d_size_;

      for (int i = 0; i < size; ++i) {
        data_ptrs[n_prev][i] = device_specific_Ndisc_max_bound[0][i];
        data_ptrs[n_prev + 1][i] = device_specific_Ndisc_min_bound[0][i];
        data_ptrs[n_prev + 2][i] = device_specific_Ndiscmax[0][i];
        data_ptrs[n_prev + 3][i] = device_specific_Ndiscmin[0][i];
        data_ptrs[n_prev + 4][i] = device_specific_ldisc[0][i];
        data_ptrs[n_prev + 5][i] = device_specific_A[0][i];
        data_ptrs[n_prev + 6][i] = device_specific_Ndiscmax_ctoc_upper_bound[0][i];
        data_ptrs[n_prev + 7][i] = device_specific_Ndiscmin_ctoc_upper_bound[0][i];
        data_ptrs[n_prev + 8][i] = device_specific_ldisc_ctoc_upper_bound[0][i];
        data_ptrs[n_prev + 9][i] = device_specific_A_ctoc_upper_bound[0][i];
        data_ptrs[n_prev + 10][i] = device_specific_Ndiscmax_ctoc_lower_bound[0][i];
        data_ptrs[n_prev + 11][i] = device_specific_Ndiscmin_ctoc_lower_bound[0][i];
        data_ptrs[n_prev + 12][i] = device_specific_ldisc_ctoc_lower_bound[0][i];
        data_ptrs[n_prev + 13][i] = device_specific_A_ctoc_lower_bound[0][i];
      },
      /* vec2dp body*/
      int n_prev = (int)names.size();
      int size = this->x_size_ * this->d_size_;

      for (int i = 0; i < size; ++i) {
        device_specific_Ndisc_max_bound[0][i] = data_ptrs[n_prev][i];
        device_specific_Ndisc_min_bound[0][i] = data_ptrs[n_prev + 1][i];
        device_specific_Ndiscmax[0][i] = data_ptrs[n_prev + 2][i];
        device_specific_Ndiscmin[0][i] = data_ptrs[n_prev + 3][i];
        device_specific_ldisc[0][i] = data_ptrs[n_prev + 4][i];
        device_specific_A[0][i] = data_ptrs[n_prev + 5][i];
        device_specific_Ndiscmax_ctoc_upper_bound[0][i] = data_ptrs[n_prev + 6][i];
        device_specific_Ndiscmin_ctoc_upper_bound[0][i] = data_ptrs[n_prev + 7][i];
        device_specific_ldisc_ctoc_upper_bound[0][i] = data_ptrs[n_prev + 8][i];
        device_specific_A_ctoc_upper_bound[0][i] = data_ptrs[n_prev + 9][i];
        device_specific_Ndiscmax_ctoc_lower_bound[0][i] = data_ptrs[n_prev + 10][i];
        device_specific_Ndiscmin_ctoc_lower_bound[0][i] = data_ptrs[n_prev + 11][i];
        device_specific_ldisc_ctoc_lower_bound[0][i] = data_ptrs[n_prev + 12][i];
        device_specific_A_ctoc_lower_bound[0][i] = data_ptrs[n_prev + 13][i];
      }


      ,
      /*invert copy DP */
      for (int j = 0; j < this->x_size_; ++j) {
        for (int i = 0; i < this->d_size_; ++i) {
          std::swap(device_specific_Ndisc_max_bound[i][j], device_specific_Ndisc_min_bound[i][j]);
          std::swap(device_specific_Ndiscmax[i][j], device_specific_Ndiscmin[i][j]);
          std::swap(device_specific_Ndiscmax_ctoc_upper_bound[i][j], device_specific_Ndiscmin_ctoc_upper_bound[i][j]);
          std::swap(device_specific_Ndiscmax_ctoc_lower_bound[i][j], device_specific_Ndiscmin_ctoc_lower_bound[i][j]);
        }
      }
      // Todo: if device specific Ndisc bounds: Remap wmax&wmin to Ndisc max_bound&min_bound 

  );


  void printDP(int x_count, int d_count) const override;

  // inline T **getNdiscmaxBound() const { return device_specific_Ndisc_max_bound; };
  // inline T **getNdiscminBound() const { return device_specific_Ndisc_min_bound; };
  // inline T **getNdiscmax() const { return device_specific_Ndiscmax; };
  // inline T **getNdiscmin() const { return device_specific_Ndiscmin; };
  inline T **getldisc() const { return device_specific_ldisc; };
  inline T **getA() const { return device_specific_A; };

  T **getMaxBound() const override { return device_specific_Ndisc_max_bound; };
  T **getMinBound() const override { return device_specific_Ndisc_min_bound; };
  T **getScaleUp() const override { return device_specific_Ndiscmax; };
  T **getScaleDown() const override { return device_specific_Ndiscmin; };

  void decayWeights(T **weights, bool bias_no_decay) override{RPU_NOT_IMPLEMENTED;};
  void decayWeights(T **weights, T alpha, bool bias_no_decay) override{RPU_NOT_IMPLEMENTED;};
  void driftWeights(T **weights, T time_since_last_call, RNG<T> &rng) override{RPU_NOT_IMPLEMENTED;};
  void diffuseWeights(T **weights, RNG<T> &rng) override{RPU_NOT_IMPLEMENTED;};
  void clipWeights(T **weights, T add_clip) override;
  bool onSetWeights(T **weights) override;
  // RRAM does not have the function to reset to a 0 weight value
  void resetCols(T **weights, int start_col, int n_cols, T reset_prob, RealWorldRNG<T> &rng) override{RPU_NOT_IMPLEMENTED;};
  void resetAtIndices(T **weights, std::vector<int> x_major_indices, RealWorldRNG<T> &rng) override{RPU_NOT_IMPLEMENTED;};
  
  
  void doSparseUpdate(
      T **weights, int i, const int *x_signed_indices, int x_count, int d_sign, RNG<T> *rng)
      override;
  void doDenseUpdate(T **weights, int *coincidences, RNG<T> *rng) override;
  

private:
  T **device_specific_Ndisc_max_bound = nullptr;
  T **device_specific_Ndisc_min_bound = nullptr;
  T **device_specific_Ndiscmax = nullptr;
  T **device_specific_Ndiscmin = nullptr;
  T **device_specific_ldisc = nullptr;
  T **device_specific_A = nullptr;
  T **device_specific_Ndiscmax_ctoc_upper_bound = nullptr;
  T **device_specific_Ndiscmin_ctoc_upper_bound = nullptr;
  T **device_specific_ldisc_ctoc_upper_bound = nullptr;
  T **device_specific_A_ctoc_upper_bound = nullptr;
  T **device_specific_Ndiscmax_ctoc_lower_bound = nullptr;
  T **device_specific_Ndiscmin_ctoc_lower_bound = nullptr;
  T **device_specific_ldisc_ctoc_lower_bound = nullptr;
  T **device_specific_A_ctoc_lower_bound = nullptr;
};

} // namespace RPU

