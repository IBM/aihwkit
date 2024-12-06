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

#include "rpu_JART_v1b_device.h"
#include <stdio.h>

namespace RPU {

template <typename T>
void JARTv1bRPUDeviceMetaParameter<T>::initialize() {

  PulsedRPUDeviceMetaParameter<T>::initialize();

  // MJR: is it correct to compute this globally?
  _current_min = (T)(-g0*(exp(-g1*read_voltage)-1))/(pow((1+(h0+h1*read_voltage+h2*exp(-h3*read_voltage))*pow((Ndisc_min_bound/Ndiscmin),(-j_0))),(1/k0)));
  _current_max = (T)(-g0*(exp(-g1*read_voltage)-1))/(pow((1+(h0+h1*read_voltage+h2*exp(-h3*read_voltage))*pow((Ndisc_max_bound/Ndiscmin),(-j_0))),(1/k0)));
  // from [0.0001:1000], initial oxygen vacancy concentration in the disc [10^26/m^3]

  _Ninit =pow(((pow(((-g0*(exp(-g1*read_voltage)-1))/(((0-w_min)/(w_max-w_min))*(_current_max-_current_min)+_current_min)), k0)-1)/(h0+h1*read_voltage+h2*exp(-h3*read_voltage))),(1/-j_0))*Ndiscmin; // from [0.0001:1000];				// initial oxygen vacancy concentration in the disc [10^26/m^3]

  
  T alpha1 = alpha0*exp(-alpha2/alpha3);
  _alpha_SET = (T) ((alpha1+alpha0)/(1+exp(-(pulse_voltage_SET+alpha2)/alpha3)))-alpha0;
  _beta_SET = (T) (beta1*(1-exp(-pulse_voltage_SET)))-beta0*pulse_voltage_SET;
  _c_SET = (T) c2*exp(-pulse_voltage_SET/c3)+c1*pulse_voltage_SET-c0;
  _d_SET = (T) d2*exp(-pulse_voltage_SET/d3)+d1*pulse_voltage_SET-d0;
  _f_SET = (T) f0+((f1-f0)/(1+pow((-pulse_voltage_SET/f2),f3)));
  
  _g_RESET = (T) -g0*(exp(-g1*pulse_voltage_RESET)-1);
  _h_RESET = (T) h0+h1*pulse_voltage_RESET+h2*exp(-h3*pulse_voltage_RESET);

  _g_read = (T) -g0*(exp(-g1*read_voltage)-1);
  _h_read = (T) h0+h1*read_voltage+h2*exp(-h3*read_voltage);
  
  _Original_A = (T) M_PI*pow(rdisc,2.0);
  
  _Rth_negative_coefficient = (T) Rth0*_Original_A;
  _Rth_positive_coefficient = (T) Rth0*Rtheff_scaling*_Original_A;
  
  _V_series_coefficient = (T) R0*alphaline*R0*Rthline;
  _V_disk_coefficient = (T) PHYSICAL_PARAMETER_zvo*PHYSICAL_PARAMETER_e*un*1e26;
  
  _gamma_coefficient = (T) (PHYSICAL_PARAMETER_zvo*a)/(dWa*M_PI);
  _a_ny0 = (T) a*ny0;
}

template <typename T>
T JARTv1bRPUDeviceMetaParameter<T>::mapWeight2Ndisc(T weight) const {

  T weight_to_current_ratio = (T)(_current_max - _current_min) / (w_max - w_min);
 // Just a simple linear mapping, doesn't have to be inside the range
  T I_mem = (weight - w_min) * weight_to_current_ratio + _current_min;

  if (I_mem > (T)0.0) {
    return pow(((pow((_g_read / I_mem), k0) - 1.0) / (_h_read)), 1.0 / (-j_0)) * Ndiscmin;
  }
  return (T)0.0;
}

template <typename T>
T JARTv1bRPUDeviceMetaParameter<T>::mapNdisc2Weight(double Ndisc) const {
  T weight_to_current_ratio = (T)(_current_max - _current_min) / (w_max - w_min);
  T read_current = _g_read / (pow((1 + _h_read * pow((Ndisc / Ndiscmin), -j_0)), 1 / k0));
  return (read_current - _current_min) / weight_to_current_ratio + w_min;
}

  
  
/********************************************************************************
 * JART v1b RPU Device
y *********************************************************************************/

template <typename T>
void JARTv1bRPUDevice<T>::populate(
    const JARTv1bRPUDeviceMetaParameter<T> &p, RealWorldRNG<T> *rng) {

  PulsedRPUDevice<T>::populate(p, rng); // will clone par
  auto &par = getPar();
  if (par.Ndiscmax_ctoc_upper_bound_old < 0) {
    RPU_FATAL("Ndiscmax_ctoc_upper_bound_old needs to be 0 or positive.");
  }

  if (par.Ndiscmax_ctoc_lower_bound_old < 0) {
    RPU_FATAL("Ndiscmax_ctoc_lower_bound_old needs to be 0 or positive.");
  }

  if (par.Ndiscmin_ctoc_upper_bound_old < 0) {
    RPU_FATAL("Ndiscmin_ctoc_upper_bound_old needs to be 0 or positive.");
  }

  if (par.Ndiscmin_ctoc_lower_bound_old < 0) {
    RPU_FATAL("Ndiscmin_ctoc_lower_bound_old needs to be 0 or positive.");
  }

  if (par.ldisc_ctoc_upper_bound_old < 0) {
    RPU_FATAL("ldisc_ctoc_upper_bound_old needs to be 0 or positive.");
  }

  if (par.ldisc_ctoc_lower_bound_old < 0) {
    RPU_FATAL("ldisc_ctoc_lower_bound_old needs to be 0 or positive.");
  }

  if (par.rdisc_ctoc_upper_bound_old < 0) {
    RPU_FATAL("rdisc_ctoc_upper_bound_old needs to be 0 or positive.");
  }

  if (par.rdisc_ctoc_lower_bound_old < 0) {
    RPU_FATAL("rdisc_ctoc_lower_bound_old needs to be 0 or positive.");
  }

  if (par.Ndiscmax_dtod_upper_bound < 0) {
    RPU_FATAL("Ndiscmax_dtod_upper_bound needs to be 0 or positive.");
  }

  if (par.Ndiscmax_dtod_lower_bound < 0) {
    RPU_FATAL("Ndiscmax_dtod_lower_bound needs to be 0 or positive.");
  }

  if (par.Ndiscmin_dtod_upper_bound < 0) {
    RPU_FATAL("Ndiscmin_dtod_upper_bound needs to be 0 or positive.");
  }

  if (par.Ndiscmin_dtod_lower_bound < 0) {
    RPU_FATAL("Ndiscmin_dtod_lower_bound needs to be 0 or positive.");
  }

  if (par.ldisc_dtod_upper_bound < 0) {
    RPU_FATAL("ldisc_dtod_upper_bound needs to be 0 or positive.");
  }

  if (par.ldisc_dtod_lower_bound < 0) {
    RPU_FATAL("ldisc_dtod_lower_bound needs to be 0 or positive.");
  }

  if (par.rdisc_dtod_upper_bound < 0) {
    RPU_FATAL("rdisc_dtod_upper_bound needs to be 0 or positive.");
  }

  if (par.rdisc_dtod_lower_bound < 0) {
    RPU_FATAL("rdisc_dtod_lower_bound needs to be 0 or positive.");
  }
  
  if (par.enable_w_max_w_min_bounds) {
    if (par.w_max_dtod_upper_bound < par.w_max_dtod_lower_bound) {
      RPU_FATAL("w_max_dtod_upper_bound needs to be larger than w_max_dtod_lower_bound.");
    }
    if (par.w_min_dtod_upper_bound < par.w_min_dtod_lower_bound) {
      RPU_FATAL("w_min_dtod_upper_bound needs to be larger than w_min_dtod_lower_bound.");
    }
  }
  
  if (par.Ndiscmax_ctoc_upper_bound_old < par.Ndiscmax_ctoc_lower_bound_old) {
    RPU_FATAL("Ndiscmax_ctoc_upper_bound_old needs to be larger than Ndiscmax_ctoc_lower_bound_old.");
  }
  if (par.Ndiscmin_ctoc_upper_bound_old < par.Ndiscmin_ctoc_lower_bound_old) {
    RPU_FATAL("Ndiscmin_ctoc_upper_bound_old needs to be larger than Ndiscmin_ctoc_lower_bound_old.");
  }

  if (par.Ndiscmax_ctoc_upper_bound_old < par.Ndiscmax_ctoc_lower_bound_old) {
    RPU_FATAL("Ndiscmax_ctoc_upper_bound_old needs to be larger than Ndiscmax_ctoc_lower_bound_old.");
  }
  if (par.Ndiscmin_ctoc_upper_bound_old < par.Ndiscmin_ctoc_lower_bound_old) {
    RPU_FATAL("Ndiscmin_ctoc_upper_bound_old needs to be larger than Ndiscmin_ctoc_lower_bound_old.");
  }
  if (par.ldisc_ctoc_upper_bound_old < par.ldisc_ctoc_lower_bound_old) {
    RPU_FATAL("ldisc_ctoc_upper_bound_old needs to be larger than ldisc_ctoc_lower_bound_old.");
  }
  if (par.rdisc_ctoc_upper_bound_old < par.rdisc_ctoc_lower_bound_old) {
    RPU_FATAL("rdisc_ctoc_upper_bound_old needs to be larger than rdisc_ctoc_lower_bound_old.");
  }

  if (par.Ndiscmax_dtod_upper_bound < par.Ndiscmax_dtod_lower_bound) {
    RPU_FATAL("Ndiscmax_dtod_upper_bound needs to be larger than Ndiscmax_dtod_lower_bound.");
  }
  if (par.Ndiscmin_dtod_upper_bound < par.Ndiscmin_dtod_lower_bound) {
    RPU_FATAL("Ndiscmin_dtod_upper_bound needs to be larger than Ndiscmin_dtod_lower_bound.");
  }
  if (par.ldisc_dtod_upper_bound < par.ldisc_dtod_lower_bound) {
    RPU_FATAL("ldisc_dtod_upper_bound needs to be larger than ldisc_dtod_lower_bound.");
  }
  if (par.rdisc_dtod_upper_bound < par.rdisc_dtod_lower_bound) {
    RPU_FATAL("rdisc_dtod_upper_bound needs to be larger than rdisc_dtod_lower_bound.");
  }
  
  if (par.Ndiscmax_dtod_lower_bound < par.Ndiscmax_ctoc_lower_bound_old) {
    RPU_FATAL("For old implimentation, Ndiscmax_ctoc range cannot be smaller than Ndiscmax_dtod range.");
  }
  if (par.Ndiscmax_ctoc_upper_bound_old > 0){
    if (par.Ndiscmax_dtod_upper_bound >0){
      if (par.Ndiscmax_ctoc_upper_bound_old < par.Ndiscmax_dtod_upper_bound) {
        RPU_FATAL("For old implimentation, Ndiscmax_ctoc range cannot be smaller than Ndiscmax_dtod range.");
      }
    }else{
      RPU_FATAL("For old implimentation, Ndiscmax_ctoc range cannot be smaller than Ndiscmax_dtod range.");
    }
  }

  if (par.Ndiscmin_dtod_lower_bound < par.Ndiscmin_ctoc_lower_bound_old) {
    RPU_FATAL("For old implimentation, Ndiscmin_ctoc range cannot be smaller than Ndiscmin_dtod range.");
  }
  if (par.Ndiscmin_ctoc_upper_bound_old > 0){
    if (par.Ndiscmin_dtod_upper_bound >0){
      if (par.Ndiscmin_ctoc_upper_bound_old < par.Ndiscmin_dtod_upper_bound) {
        RPU_FATAL("For old implimentation, Ndiscmin_ctoc range cannot be smaller than Ndiscmin_dtod range.");
      }
    }else{
      RPU_FATAL("For old implimentation, Ndiscmin_ctoc range cannot be smaller than Ndiscmin_dtod range.");
    }
  }

  if (par.ldisc_dtod_lower_bound < par.ldisc_ctoc_lower_bound_old) {
    RPU_FATAL("For old implimentation, ldisc_ctoc range cannot be smaller than ldisc_dtod range.");
  }
  if (par.ldisc_ctoc_upper_bound_old > 0){
    if (par.ldisc_dtod_upper_bound >0){
      if (par.ldisc_ctoc_upper_bound_old < par.ldisc_dtod_upper_bound) {
        RPU_FATAL("For old implimentation, ldisc_ctoc range cannot be smaller than ldisc_dtod range.");
      }
    }else{
      RPU_FATAL("For old implimentation, ldisc_ctoc range cannot be smaller than ldisc_dtod range.");
    }
  }

  if (par.rdisc_dtod_lower_bound < par.rdisc_ctoc_lower_bound_old) {
    RPU_FATAL("For old implimentation, rdisc_ctoc range cannot be smaller than rdisc_dtod range.");
  }
  if (par.rdisc_ctoc_upper_bound_old > 0){
    if (par.rdisc_dtod_upper_bound >0){
      if (par.rdisc_ctoc_upper_bound_old < par.rdisc_dtod_upper_bound) {
        RPU_FATAL("For old implimentation, rdisc_ctoc range cannot be smaller than rdisc_dtod range.");
      }
    }else{
      RPU_FATAL("For old implimentation, rdisc_ctoc range cannot be smaller than rdisc_dtod range.");
    }
  }

  for (int i = 0; i < this->d_size_; ++i) {
    for (int j = 0; j < this->x_size_; ++j) {
      device_specific_Ndisc_max_bound[i][j] = par.mapWeight2Ndisc(this->w_max_bound_[i][j]);
      device_specific_Ndisc_min_bound[i][j] = par.mapWeight2Ndisc(this->w_min_bound_[i][j]);

      device_specific_Ndiscmax[i][j] = par.Ndiscmax * (1 + par.Ndiscmax_dtod * rng->sampleGauss());
      if (par.Ndiscmax_dtod_upper_bound > (T)0.0) {
        device_specific_Ndiscmax[i][j] = MIN(device_specific_Ndiscmax[i][j], par.Ndiscmax_dtod_upper_bound);
      }
      device_specific_Ndiscmax[i][j] = MAX(device_specific_Ndiscmax[i][j], par.Ndiscmax_dtod_lower_bound);

      device_specific_Ndiscmax_ctoc_upper_bound[i][j] = device_specific_Ndiscmax[i][j] * (1+par.Ndiscmax_ctoc_upper_bound);
      device_specific_Ndiscmax_ctoc_lower_bound[i][j] = device_specific_Ndiscmax[i][j] * (1+par.Ndiscmax_ctoc_lower_bound);

      device_specific_Ndiscmin[i][j] = par.Ndiscmin * (1 + par.Ndiscmin_dtod * rng->sampleGauss());
      if (par.Ndiscmin_dtod_upper_bound > (T)0.0) {
        device_specific_Ndiscmin[i][j] = MIN(device_specific_Ndiscmin[i][j], par.Ndiscmin_dtod_upper_bound);
      }
      device_specific_Ndiscmin[i][j] = MAX(device_specific_Ndiscmin[i][j], par.Ndiscmin_dtod_lower_bound);

      device_specific_Ndiscmin_ctoc_upper_bound[i][j] = device_specific_Ndiscmin[i][j] * (1+par.Ndiscmin_ctoc_upper_bound);
      device_specific_Ndiscmin_ctoc_lower_bound[i][j] = device_specific_Ndiscmin[i][j] * (1+par.Ndiscmin_ctoc_lower_bound);

      device_specific_ldisc[i][j] = par.ldisc * (1 + par.ldisc_dtod * rng->sampleGauss());
      if (par.ldisc_dtod_upper_bound > (T)0.0) {
        device_specific_ldisc[i][j] = MIN(device_specific_ldisc[i][j], par.ldisc_dtod_upper_bound);
      }
      device_specific_ldisc[i][j] = MAX(device_specific_ldisc[i][j], par.ldisc_dtod_lower_bound);

      device_specific_ldisc_ctoc_upper_bound[i][j] = device_specific_ldisc[i][j] * (1+par.ldisc_ctoc_upper_bound);
      device_specific_ldisc_ctoc_lower_bound[i][j] = device_specific_ldisc[i][j] * (1+par.ldisc_ctoc_lower_bound);

      T device_specific_rdisc = par.rdisc * (1 + par.rdisc_dtod * rng->sampleGauss());
      if (par.rdisc_dtod_upper_bound > (T)0.0) {
        device_specific_rdisc = MIN(device_specific_rdisc, par.rdisc_dtod_upper_bound);
      }
      device_specific_rdisc = MAX(device_specific_rdisc, par.rdisc_dtod_lower_bound);

      T device_specific_rdisc_ctoc_upper_bound = device_specific_rdisc * (1+par.rdisc_ctoc_upper_bound);
      T device_specific_rdisc_ctoc_lower_bound = device_specific_rdisc * (1+par.rdisc_ctoc_lower_bound);

      device_specific_A[i][j] = (T) M_PI*pow(device_specific_rdisc,2.0);

      device_specific_A_ctoc_upper_bound[i][j] = (T) M_PI*pow(device_specific_rdisc_ctoc_upper_bound,2.0);
      device_specific_A_ctoc_lower_bound[i][j] = (T) M_PI*pow(device_specific_rdisc_ctoc_lower_bound,2.0);

      this->w_persistent_[i][j] = par._Ninit;
      if (par.enable_w_max_w_min_bounds) {
        this->w_max_bound_[i][j] = par.w_max * (1 + par.w_max_dtod * rng->sampleGauss());
        this->w_max_bound_[i][j] = MIN(this->w_max_bound_[i][j], par.w_max_dtod_upper_bound);
        this->w_max_bound_[i][j] = MAX(this->w_max_bound_[i][j], par.w_max_dtod_lower_bound);

        this->w_min_bound_[i][j] = par.w_min * (1 + par.w_min_dtod * rng->sampleGauss());
        this->w_min_bound_[i][j] = MIN(this->w_min_bound_[i][j], par.w_min_dtod_upper_bound);
        this->w_min_bound_[i][j] = MAX(this->w_min_bound_[i][j], par.w_min_dtod_lower_bound);
      }
    }
  }
}

template <typename T> void JARTv1bRPUDevice<T>::printDP(int x_count, int d_count) const {

  if (x_count < 0 || x_count > this->x_size_) {
    x_count = this->x_size_;
  }

  if (d_count < 0 || d_count > this->d_size_) {
    d_count = this->d_size_;
  }

  for (int i = 0; i < d_count; ++i) {
    for (int j = 0; j < x_count; ++j) {
      std::cout.precision(5);
      std::cout << i << "," << j << ": ";
      std::cout << device_specific_Ndiscmax[i][j] << ",";
      std::cout << device_specific_Ndiscmin[i][j] << ",";
      std::cout << device_specific_ldisc[i][j] << ",";
      std::cout << device_specific_A[i][j] << ",";
      std::cout.precision(10);
      std::cout << this->w_decay_scale_[i][j] << ", ";
      std::cout.precision(6);
      std::cout << this->w_diffusion_rate_[i][j] << ", ";
      std::cout << this->w_reset_bias_[i][j];
      std::cout << ", " << this->w_persistent_[i][j];
      std::cout << "]";
    }
    std::cout << std::endl;
  }
}

template <typename T>
inline void step_SET(
    double &Ndisc,
    T &Ndiscmax,
    T &Ndiscmin,
    T &ldisc,
    T &A,
    const JARTv1bRPUDeviceMetaParameter<T>& par) {

  T I_mem = -par._alpha_SET - par._beta_SET / (pow((1.0 + pow((par._c_SET / Ndisc), par._d_SET)), par._f_SET));

    // NOTE: V_disk = I_mem*(ldisc/(V_disk_coefficient*A*Ndisc))
    // NOTE: Eion = V_disk/ldisc
  T Eion = I_mem / (par._V_disk_coefficient * A * Ndisc);

    // NOTE: T gamma = gamma_coefficient*Eion
  T gamma = par._gamma_coefficient * Eion;
  
    // NOTE: V - V_series = V_disk+V_plug+V_Schottky
  T V_other_than_series =
      par.pulse_voltage_SET - (I_mem * (par.RseriesTiOx + par.R0 + par._V_series_coefficient * I_mem * I_mem));

  T Treal = par.T0 + I_mem * V_other_than_series * par._Rth_negative_coefficient / A;

    // NOTE: dWamin = dWa_f = dWa*(sqrt(1.0-pow(gamma,2.0))-(gamma*M_PI)/2+gamma*asin(gamma)) = dWa_mean - dWa_difference
    // NOTE: dWamax = dWa_r = dWa*(sqrt(1.0-pow(gamma,2.0))+(gamma*M_PI)/2+gamma*asin(gamma)) = dWa_mean + dWa_difference
  T dWa_mean = par.dWa * (sqrt(1.0 - pow(gamma, 2.0)) + gamma * asin(gamma));
  T dWa_difference = par.dWa * ((gamma * M_PI) / 2.0);

  T denominator = PHYSICAL_PARAMETER_kb_over_e * Treal;

  T c_v0 = (par.Nplug + Ndisc) / 2.0;
  T F_limit = 1.0 - pow((Ndisc / Ndiscmax), 10.0);
  T dNdt = -(c_v0 * par._a_ny0 * F_limit *
             (exp(-(dWa_mean - dWa_difference) / denominator) -
              exp(-(dWa_mean + dWa_difference) / denominator))) /
           ldisc;

  Ndisc = Ndisc + dNdt*par.base_time_step;
}

template <typename T>
inline void step_RESET(
    double &Ndisc,
    T &Ndiscmax,
    T &Ndiscmin,
    T &ldisc,
    T &A,
    const JARTv1bRPUDeviceMetaParameter<T>& par) {
  T I_mem = -par._g_RESET/(pow((1.0+par._h_RESET*pow((Ndisc/par.Ndiscmin),-par.j_0)),1.0/par.k0));
  
  // NOTE: V - V_series = V_disk+V_plug+V_Schottky
  T V_other_than_series = par.pulse_voltage_RESET - (I_mem*(par.RseriesTiOx + par.R0 + par._V_series_coefficient*I_mem*I_mem));

  // NOTE: T gamma = gamma_coefficient*Eion
  T gamma = par._gamma_coefficient*V_other_than_series/par.lcell;

  T Treal = par.T0 + I_mem*V_other_than_series*par._Rth_positive_coefficient/A;

  // NOTE: dWamin = dWa_f = dWa*(sqrt(1.0-pow(gamma,2.0))-(gamma*M_PI)/2+gamma*asin(gamma)) = dWa_mean - dWa_difference
  // NOTE: dWamax = dWa_r = dWa*(sqrt(1.0-pow(gamma,2.0))+(gamma*M_PI)/2+gamma*asin(gamma)) = dWa_mean + dWa_difference
  T dWa_mean = par.dWa * (sqrt(1.0 - pow(gamma, 2.0)) + gamma * asin(gamma));
  T dWa_difference = par.dWa * ((gamma * M_PI) / 2.0);

  T denominator = PHYSICAL_PARAMETER_kb_over_e * Treal;

  T c_v0 = (par.Nplug + Ndisc) / 2.0;
  T F_limit = 1.0-pow((Ndiscmin/Ndisc),10.0);
  T dNdt = -(c_v0 * par._a_ny0 * F_limit *
             (exp(-(dWa_mean - dWa_difference) / denominator) -
              exp(-(dWa_mean + dWa_difference) / denominator))) /
           ldisc;

  Ndisc = Ndisc + dNdt*par.base_time_step;
}

template <typename T>
inline void apply_cycle_to_cycle_noise(
    const T &ratio,
    T &Ndiscmax,
    T &Ndiscmin,
    T &ldisc,
    T &A,
    const JARTv1bRPUDeviceMetaParameter<T>& par,
    RNG<T> *rng,
    const T &Ndiscmax_ctoc_upper_bound,
    const T &Ndiscmax_ctoc_lower_bound,
    const T &Ndiscmin_ctoc_upper_bound,
    const T &Ndiscmin_ctoc_lower_bound,
    const T &ldisc_ctoc_upper_bound,
    const T &ldisc_ctoc_lower_bound,
    const T &A_ctoc_upper_bound,
    const T &A_ctoc_lower_bound) {
  if (par.Ndiscmax_std > (T)0.0) {
    Ndiscmax = Ndiscmax * (1 + par.Ndiscmax_std * (2*rng->sampleUniform()-1));
    Ndiscmax = MIN(Ndiscmax, Ndiscmax_ctoc_upper_bound);
    Ndiscmax = MAX(Ndiscmax, Ndiscmax_ctoc_lower_bound);
  }
  if (par.Ndiscmin_std > (T)0.0) {
    Ndiscmin = Ndiscmin * (1 + par.Ndiscmin_std * (2*rng->sampleUniform()-1));
    Ndiscmin = MIN(Ndiscmin, Ndiscmin_ctoc_upper_bound);
    Ndiscmin = MAX(Ndiscmin, Ndiscmin_ctoc_lower_bound);
  }
  if ((par.ldisc_std > (T)0.0)||(par.ldisc_std_slope > (T)0.0)) {
    ldisc = ldisc * (1 + par.ldisc_std * (2*rng->sampleUniform()-1) + ratio * par.ldisc_std_slope * (2*rng->sampleUniform()-1));
    ldisc = MIN(ldisc, ldisc_ctoc_upper_bound);
    ldisc = MAX(ldisc, ldisc_ctoc_lower_bound);
  }
  if ((par.rdisc_std > (T)0.0)||(par.rdisc_std_slope > (T)0.0)) {
    T rdisc = pow(A/M_PI, 0.5) * (1 + par.rdisc_std * (2*rng->sampleUniform()-1) + ratio * par.rdisc_std_slope * (2*rng->sampleUniform()-1));
    A = M_PI*pow(rdisc,2.0);
    A = MIN(A, A_ctoc_upper_bound);
    A = MAX(A, A_ctoc_lower_bound);
  }
}


template <typename T>
inline void update_once(
    T &Ndisc,
    T &w,
    const int &sign,
    T &Ndiscmax,
    T &Ndiscmin,
    T &ldisc,
    T &A,
    const T device_specific_Ndisc_max_bound,
    const T device_specific_Ndisc_min_bound,
    const JARTv1bRPUDeviceMetaParameter<T>& par,
    RNG<T> *rng,
    const T &Ndiscmax_ctoc_upper_bound,
    const T &Ndiscmax_ctoc_lower_bound,
    const T &Ndiscmin_ctoc_upper_bound,
    const T &Ndiscmin_ctoc_lower_bound,
    const T &ldisc_ctoc_upper_bound,
    const T &ldisc_ctoc_lower_bound,
    const T &A_ctoc_upper_bound,
    const T &A_ctoc_lower_bound) {
  int n_time_steps = round(par.pulse_length / par.base_time_step);
  
  double Ndisc_double = Ndisc;
  T max_bound = MIN(device_specific_Ndisc_max_bound, Ndiscmax);
  T min_bound = MAX(device_specific_Ndisc_min_bound, Ndiscmin);

  if (sign < 0) {
    if (Ndisc_double < max_bound)
    {
      for (int i = 0; i < n_time_steps; i++) {
        step_SET(Ndisc_double, Ndiscmax, Ndiscmin, ldisc, A, par);
      }
      
      T ratio = (T)(Ndisc_double-Ndisc)/(Ndiscmax-Ndisc);
      apply_cycle_to_cycle_noise(ratio, Ndiscmax, Ndiscmin, ldisc, A, par, rng,
                                 Ndiscmax_ctoc_upper_bound, Ndiscmax_ctoc_lower_bound, Ndiscmin_ctoc_upper_bound, Ndiscmin_ctoc_lower_bound,
                                 ldisc_ctoc_upper_bound, ldisc_ctoc_lower_bound, A_ctoc_upper_bound, A_ctoc_lower_bound);
    }

  } else {
    if (Ndisc_double > min_bound)
    {
      for (int i = 0; i < n_time_steps; i++) {
        step_RESET(Ndisc_double, Ndiscmax, Ndiscmin, ldisc, A, par);
      } 
      T ratio = (T)(Ndisc-Ndisc_double)/(Ndisc-Ndiscmin);
      apply_cycle_to_cycle_noise(ratio, Ndiscmax, Ndiscmin, ldisc, A, par, rng,
                                 Ndiscmax_ctoc_upper_bound, Ndiscmax_ctoc_lower_bound, Ndiscmin_ctoc_upper_bound, Ndiscmin_ctoc_lower_bound,
                                 ldisc_ctoc_upper_bound, ldisc_ctoc_lower_bound, A_ctoc_upper_bound, A_ctoc_lower_bound);
    }
  }
  Ndisc_double = MAX(MIN(Ndisc_double, max_bound), min_bound);
  w = par.mapNdisc2Weight(Ndisc_double);
  Ndisc = Ndisc_double;
}


template <typename T>
void JARTv1bRPUDevice<T>::doSparseUpdate(
    T **weights, int i, const int *x_signed_indices, int x_count, int d_sign, RNG<T> *rng) {

  const auto &par = getPar();

  T *w = weights[i];
  T *Ndisc = this->w_persistent_[i];
  T *Ndiscmax = device_specific_Ndiscmax[i];
  T *Ndiscmin = device_specific_Ndiscmin[i];
  T *ldisc = device_specific_ldisc[i];
  T *A = device_specific_A[i];
  T *Ndisc_max_bound = device_specific_Ndisc_max_bound[i];
  T *Ndisc_min_bound = device_specific_Ndisc_min_bound[i];
  T *Ndiscmax_ctoc_upper_bound = device_specific_Ndiscmax_ctoc_upper_bound[i];
  T *Ndiscmax_ctoc_lower_bound = device_specific_Ndiscmax_ctoc_lower_bound[i];
  T *Ndiscmin_ctoc_upper_bound = device_specific_Ndiscmin_ctoc_upper_bound[i];
  T *Ndiscmin_ctoc_lower_bound = device_specific_Ndiscmin_ctoc_lower_bound[i];
  T *ldisc_ctoc_upper_bound = device_specific_ldisc_ctoc_upper_bound[i];
  T *ldisc_ctoc_lower_bound = device_specific_ldisc_ctoc_lower_bound[i];
  T *A_ctoc_upper_bound = device_specific_A_ctoc_upper_bound[i];
  T *A_ctoc_lower_bound = device_specific_A_ctoc_lower_bound[i];

  PULSED_UPDATE_W_LOOP(
      update_once(
          Ndisc[j], w[j], sign, Ndiscmax[j], Ndiscmin[j], ldisc[j], A[j],
          Ndisc_max_bound[j], Ndisc_min_bound[j], par, rng,
          Ndiscmax_ctoc_upper_bound[j], Ndiscmax_ctoc_lower_bound[j],
          Ndiscmin_ctoc_upper_bound[j], Ndiscmin_ctoc_lower_bound[j],
          ldisc_ctoc_upper_bound[j], ldisc_ctoc_lower_bound[j],
          A_ctoc_upper_bound[j], A_ctoc_lower_bound[j]););
}

template <typename T>
void JARTv1bRPUDevice<T>::doDenseUpdate(T **weights, int *coincidences, RNG<T> *rng) {

  const auto &par = getPar();

  T *w = weights[0];
  T *Ndisc = this->w_persistent_[0];
  T *Ndiscmax = device_specific_Ndiscmax[0];
  T *Ndiscmin = device_specific_Ndiscmin[0];
  T *ldisc = device_specific_ldisc[0];
  T *A = device_specific_A[0];
  T *Ndisc_max_bound = device_specific_Ndisc_max_bound[0];
  T *Ndisc_min_bound = device_specific_Ndisc_min_bound[0];
  T *Ndiscmax_ctoc_upper_bound = device_specific_Ndiscmax_ctoc_upper_bound[0];
  T *Ndiscmax_ctoc_lower_bound = device_specific_Ndiscmax_ctoc_lower_bound[0];
  T *Ndiscmin_ctoc_upper_bound = device_specific_Ndiscmin_ctoc_upper_bound[0];
  T *Ndiscmin_ctoc_lower_bound = device_specific_Ndiscmin_ctoc_lower_bound[0];
  T *ldisc_ctoc_upper_bound = device_specific_ldisc_ctoc_upper_bound[0];
  T *ldisc_ctoc_lower_bound = device_specific_ldisc_ctoc_lower_bound[0];
  T *A_ctoc_upper_bound = device_specific_A_ctoc_upper_bound[0];
  T *A_ctoc_lower_bound = device_specific_A_ctoc_lower_bound[0];

  PULSED_UPDATE_W_LOOP_DENSE(
      update_once(
          Ndisc[j], w[j], sign, Ndiscmax[j], Ndiscmin[j], ldisc[j], A[j],
          Ndisc_max_bound[j], Ndisc_min_bound[j], par, rng,
          Ndiscmax_ctoc_upper_bound[j], Ndiscmax_ctoc_lower_bound[j],
          Ndiscmin_ctoc_upper_bound[j], Ndiscmin_ctoc_lower_bound[j],
          ldisc_ctoc_upper_bound[j], ldisc_ctoc_lower_bound[j],
          A_ctoc_upper_bound[j], A_ctoc_lower_bound[j]););
}

template <typename T> void JARTv1bRPUDevice<T>::clipWeights(T **weights, T clip) {
  // apply hard bounds
  T *w = weights[0];
  T *max_bound = &(this->w_max_bound_[0][0]);
  T *min_bound = &(this->w_min_bound_[0][0]);
  if (clip < 0.0) { // only apply bounds
    PRAGMA_SIMD
    for (int i = 0; i < this->size_; ++i) {
      w[i] = MIN(w[i], max_bound[i]);
      w[i] = MAX(w[i], min_bound[i]);
    }
  } else {
    PRAGMA_SIMD
    for (int i = 0; i < this->size_; ++i) {
      w[i] = MIN(w[i], MIN(max_bound[i], clip));
      w[i] = MAX(w[i], MAX(min_bound[i], -clip));
    }
  }

  const auto &par = getPar();
  T uw_std = par.real_write_noise_std;

  if (uw_std > 0) {
    for (int i = 0; i < this->size_; i++) {
      w[i] = w[i] + uw_std * this->write_noise_rng_.sampleGauss();
    }
  }

  PRAGMA_SIMD
  for (int i = 0; i < this->size_; i++) {
    this->w_persistent_[0][i] = par.mapWeight2Ndisc(w[i]);
  }
}

template <typename T> bool JARTv1bRPUDevice<T>::onSetWeights(T **weights) {

  // apply hard bounds to given weights
  T *w = weights[0];
  T *max_bound = &(this->w_max_bound_[0][0]);
  T *min_bound = &(this->w_min_bound_[0][0]);
  PRAGMA_SIMD
  for (int i = 0; i < this->size_; ++i) {
    w[i] = MIN(w[i], max_bound[i]);
    w[i] = MAX(w[i], min_bound[i]);
  }

  const auto &par = getPar();
  T uw_std = par.real_write_noise_std;

  if (uw_std > 0) {
    for (int i = 0; i < this->size_; i++) {
      w[i] = w[i] + uw_std * this->write_noise_rng_.sampleGauss();
    }
  }

  PRAGMA_SIMD
  for (int i = 0; i < this->size_; i++) {
    this->w_persistent_[0][i] = par.mapWeight2Ndisc(w[i]);
  }
  return true;
}

template class JARTv1bRPUDevice<float>;
#ifdef RPU_USE_DOUBLE
template class JARTv1bRPUDevice<double>;
#endif

} // namespace RPU
