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

/********************************************************************************
 * JART v1b RPU Device
 *********************************************************************************/


template <typename T>
inline T invert_read_current(
    const T &I_mem,
    const T &g_read,
    const T &h_read,
    const T &j_0,
    const T &k0, 
    const T &Ndiscmin) {
  if (I_mem>0){
  return pow(((pow((g_read/I_mem), k0)-1.0)/(h_read)),1.0/(-j_0))*Ndiscmin;
  }
  else{
    return 0;
  }
}

template <typename T>
inline T map_weight_to_Ndisc(
    const T &weight,
    const T &current_min,
    const T &weight_min_bound,
    const T &weight_to_current_ratio,
    const T &g_read,
    const T &h_read,
    const T &j_0,
    const T &k0,
    const T &Original_Ndiscmin) {
  T current = (weight-weight_min_bound)*weight_to_current_ratio+current_min;
  T Ndisc = invert_read_current(current, g_read, h_read, j_0, k0, Original_Ndiscmin);
  return Ndisc;
}

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
      device_specific_Ndisc_max_bound[i][j] = map_weight_to_Ndisc(this->w_max_bound_[i][j], par.current_min, par.w_min, par.weight_to_current_ratio,
                                              par.g_read, par.h_read, par.j_0, par.k0, par.Ndiscmin);
      device_specific_Ndisc_min_bound[i][j] = map_weight_to_Ndisc(this->w_min_bound_[i][j], par.current_min, par.w_min, par.weight_to_current_ratio,
                                              par.g_read, par.h_read, par.j_0, par.k0, par.Ndiscmin);

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

      this->w_persistent_[i][j] = par.Ninit;
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
inline T calculate_current_SET(
    const double &Ndisc,
    const T &alpha_SET,
    const T &beta_SET,
    const T &c_SET,
    const T &d_SET,
    const T &f_SET) {
  return -alpha_SET-beta_SET/(pow((1+pow((c_SET/Ndisc),d_SET)),f_SET));
}

template <typename T>
inline T calculate_current_RESET_and_Read(
    const double &Ndisc,
    const T &g_RESET_or_Read,
    const T &h_RESET_or_Read,
    const T &j_0,
    const T &k0, 
    const T &Ndiscmin) {
  return g_RESET_or_Read/(pow((1+h_RESET_or_Read*pow((Ndisc/Ndiscmin),-j_0)),1/k0));
}

template <typename T>
struct Voltages_needed
{
  T other_than_V_series;
  T V_disk;
};

template <typename T>
inline Voltages_needed<T> calculate_voltages(
    const T &applied_voltage,
    const T &I_mem,
    const T &R0,
    const T &RseriesTiOx,
    const T &V_series_coefficient,
    const T &V_disk_coefficient,
    const T &lcell,
    const T &ldisc,
    const T &A,
    const double &Ndisc) {
  Voltages_needed<T> Voltages;
  // V - V_series (V_disk+V_plug+V_Schottky)
  Voltages.other_than_V_series = applied_voltage - (I_mem*(RseriesTiOx + R0 + V_series_coefficient*I_mem*I_mem));
  // V_disk
  Voltages.V_disk =  I_mem*(ldisc/(V_disk_coefficient*A*Ndisc));
  return Voltages;
}

template <typename T>
inline void step_SET(
    const T &applied_voltage_SET,
    const T &time_step,
    double &Ndisc,
    const T &alpha_SET,
    const T &beta_SET,
    const T &c_SET,
    const T &d_SET,
    const T &f_SET,
    const T &T0,
    T &Ndiscmax,
    const T &Nplug,
    const T &a_ny0,
    const T &dWa,
    const T &Rth_negative_coefficient,
    const T &RseriesTiOx,
    const T &R0,
    const T &V_series_coefficient,
    const T &V_disk_coefficient,
    const T &gamma_coefficient,
    const T &lcell,
    T &ldisc,
    T &A,
    T &max_bound) {
    T I_mem = -alpha_SET-beta_SET/(pow((1.0+pow((c_SET/Ndisc),d_SET)),f_SET));

    // NOTE: V_disk = I_mem*(ldisc/(V_disk_coefficient*A*Ndisc))
    // NOTE: Eion = V_disk/ldisc
    T Eion = I_mem/(V_disk_coefficient*A*Ndisc);

    // NOTE: T gamma = gamma_coefficient*Eion
    T gamma = gamma_coefficient*Eion;
    
    // NOTE: V - V_series = V_disk+V_plug+V_Schottky
    T V_other_than_series = applied_voltage_SET - (I_mem*(RseriesTiOx + R0 + V_series_coefficient*I_mem*I_mem));

    T Treal = T0 + I_mem*V_other_than_series*Rth_negative_coefficient/A;

    // NOTE: dWamin = dWa_f = dWa*(sqrt(1.0-pow(gamma,2.0))-(gamma*M_PI)/2+gamma*asin(gamma)) = dWa_mean - dWa_difference
    // NOTE: dWamax = dWa_r = dWa*(sqrt(1.0-pow(gamma,2.0))+(gamma*M_PI)/2+gamma*asin(gamma)) = dWa_mean + dWa_difference
    T dWa_mean = dWa*(sqrt(1.0-pow(gamma,2.0))+gamma*asin(gamma));
    T dWa_difference = dWa*((gamma*M_PI)/2.0);

    T denominator = PHYSICAL_PARAMETER_kb_over_e*Treal;

    T c_v0 = (Nplug+Ndisc)/2.0;
    T F_limit = 1.0-pow((Ndisc/Ndiscmax),10.0);
    T dNdt = -(c_v0*a_ny0*F_limit*(exp(-(dWa_mean - dWa_difference)/denominator)-exp(-(dWa_mean + dWa_difference)/denominator)))/ldisc;

    Ndisc = Ndisc + dNdt*time_step;
}

template <typename T>
inline void step_RESET(
    const T &applied_voltage_RESET,
    const T &time_step,
    double &Ndisc,
    const T &g_RESET,
    const T &h_RESET,
    const T &j_0,
    const T &k0, 
    const T &T0,
    const T &Original_Ndiscmin,
    T &Ndiscmin,
    const T &Nplug,
    const T &a_ny0,
    const T &dWa,
    const T &Rth_positive_coefficient,
    const T &RseriesTiOx,
    const T &R0,
    const T &V_series_coefficient,
    const T &V_disk_coefficient,
    const T &gamma_coefficient,
    const T &lcell,
    T &ldisc,
    T &A,
    T &min_bound) {
  T I_mem = g_RESET/(pow((1.0+h_RESET*pow((Ndisc/Original_Ndiscmin),-j_0)),1.0/k0));
  
  // NOTE: V - V_series = V_disk+V_plug+V_Schottky
  T V_other_than_series = applied_voltage_RESET - (I_mem*(RseriesTiOx + R0 + V_series_coefficient*I_mem*I_mem));

  // NOTE: T gamma = gamma_coefficient*Eion
  T gamma = gamma_coefficient*V_other_than_series/lcell;

  T Treal = T0 + I_mem*V_other_than_series*Rth_positive_coefficient/A;

  // NOTE: dWamin = dWa_f = dWa*(sqrt(1.0-pow(gamma,2.0))-(gamma*M_PI)/2+gamma*asin(gamma)) = dWa_mean - dWa_difference
  // NOTE: dWamax = dWa_r = dWa*(sqrt(1.0-pow(gamma,2.0))+(gamma*M_PI)/2+gamma*asin(gamma)) = dWa_mean + dWa_difference
  T dWa_mean = dWa*(sqrt(1.0-pow(gamma,2.0))+gamma*asin(gamma));
  T dWa_difference = dWa*((gamma*M_PI)/2.0);

  T denominator = PHYSICAL_PARAMETER_kb_over_e*Treal;

  T c_v0 = (Nplug+Ndisc)/2.0;
  T F_limit = 1.0-pow((Ndiscmin/Ndisc),10.0);
  T dNdt = -(c_v0*a_ny0*F_limit*(exp(-(dWa_mean - dWa_difference)/denominator)-exp(-(dWa_mean + dWa_difference)/denominator)))/ldisc;

  Ndisc = Ndisc + dNdt*time_step;
}

template <typename T>
inline T map_Ndisc_to_weight(
    const T &read_voltage,
    const double &Ndisc,
    const T &current_min,
    const T &weight_min_bound,
    const T &current_to_weight_ratio,
    const T &g_read,
    const T &h_read,
    const T &j_0,
    const T &k0,
    const T &Original_Ndiscmin) {
  T read_current = calculate_current_RESET_and_Read(Ndisc, g_read, h_read, j_0, k0, Original_Ndiscmin);
  T weight = (read_current-current_min)*current_to_weight_ratio+weight_min_bound;
  return weight;
}

template <typename T>
inline void apply_cycle_to_cycle_noise(
    const T &ratio,
    T &Ndiscmax,
    T &Ndiscmin,
    T &ldisc,
    T &A,
    const T &Ndiscmax_std,
    const T &Ndiscmin_std,
    const T &ldisc_std,
    const T &rdisc_std,
    const T &ldisc_std_slope,
    const T &rdisc_std_slope,
    RNG<T> *rng,
    const T &Ndiscmax_ctoc_upper_bound,
    const T &Ndiscmax_ctoc_lower_bound,
    const T &Ndiscmin_ctoc_upper_bound,
    const T &Ndiscmin_ctoc_lower_bound,
    const T &ldisc_ctoc_upper_bound,
    const T &ldisc_ctoc_lower_bound,
    const T &A_ctoc_upper_bound,
    const T &A_ctoc_lower_bound) {
  if (Ndiscmax_std > (T)0.0) {
    Ndiscmax = Ndiscmax * (1 + Ndiscmax_std * (2*rng->sampleUniform()-1));
    Ndiscmax = MIN(Ndiscmax, Ndiscmax_ctoc_upper_bound);
    Ndiscmax = MAX(Ndiscmax, Ndiscmax_ctoc_lower_bound);
  }
  if (Ndiscmin_std > (T)0.0) {
    Ndiscmin = Ndiscmin * (1 + Ndiscmin_std * (2*rng->sampleUniform()-1));
    Ndiscmin = MIN(Ndiscmin, Ndiscmin_ctoc_upper_bound);
    Ndiscmin = MAX(Ndiscmin, Ndiscmin_ctoc_lower_bound);
  }
  if ((ldisc_std > (T)0.0)||(ldisc_std_slope > (T)0.0)) {
    ldisc = ldisc * (1 + ldisc_std * (2*rng->sampleUniform()-1) + ratio * ldisc_std_slope * (2*rng->sampleUniform()-1));
    ldisc = MIN(ldisc, ldisc_ctoc_upper_bound);
    ldisc = MAX(ldisc, ldisc_ctoc_lower_bound);
  }
  if ((rdisc_std > (T)0.0)||(rdisc_std_slope > (T)0.0)) {
    T rdisc = pow(A/M_PI, 0.5) * (1 + rdisc_std * (2*rng->sampleUniform()-1) + ratio * rdisc_std_slope * (2*rng->sampleUniform()-1));
    A = M_PI*pow(rdisc,2.0);
    A = MIN(A, A_ctoc_upper_bound);
    A = MAX(A, A_ctoc_lower_bound);
  }
}

template <typename T>
inline void update_once(
    const T &read_voltage,
    const T &pulse_voltage_SET,
    const T &pulse_voltage_RESET,
    const T &pulse_length,
    const T &base_time_step,
    const T &alpha_SET,
    const T &beta_SET,
    const T &c_SET,
    const T &d_SET,
    const T &f_SET,
    const T &g_RESET,
    const T &h_RESET,
    const T &g_read,
    const T &h_read,
    const T &j_0,
    const T &k0, 
    const T &T0,
    const T &Original_Ndiscmin,
    T &Ndiscmax,
    T &Ndiscmin,
    const T &Nplug,
    const T &a_ny0,
    const T &dWa,
    const T &Rth_negative_coefficient,
    const T &Rth_positive_coefficient,
    const T &RseriesTiOx,
    const T &R0,
    const T &V_series_coefficient,
    const T &V_disk_coefficient,
    const T &gamma_coefficient,
    const T &lcell,
    T &ldisc,
    T &A,
    T &Ndisc,
    T &w,
    int &sign,
    const T &current_min,
    const T &current_to_weight_ratio,
    const T &weight_to_current_ratio,
    const T &weight_min_bound,
    const T device_specific_Ndisc_max_bound,
    const T device_specific_Ndisc_min_bound,
    const T &Ndiscmax_std,
    const T &Ndiscmax_ctoc_upper_bound,
    const T &Ndiscmax_ctoc_lower_bound,
    const T &Ndiscmin_std,
    const T &Ndiscmin_ctoc_upper_bound,
    const T &Ndiscmin_ctoc_lower_bound,
    const T &ldisc_std,
    const T &ldisc_std_slope,
    const T &ldisc_ctoc_upper_bound,
    const T &ldisc_ctoc_lower_bound,
    const T &rdisc_std,
    const T &rdisc_std_slope,
    const T &A_ctoc_upper_bound,
    const T &A_ctoc_lower_bound,
    RNG<T> *rng) {
  int pulse_counter = int (pulse_length/base_time_step);
  double Ndisc_double = Ndisc;
  T max_bound = MIN(device_specific_Ndisc_max_bound, Ndiscmax);
  T min_bound = MAX(device_specific_Ndisc_min_bound, Ndiscmin);

  if (sign < 0) {
    if (Ndisc_double < max_bound)
    {
      for (int i = 0; i < pulse_counter; i++) {
        step_SET(pulse_voltage_SET, base_time_step, Ndisc_double, alpha_SET, beta_SET, c_SET, d_SET, f_SET, T0, Ndiscmax, Nplug, a_ny0, dWa, Rth_negative_coefficient,
        RseriesTiOx, R0, V_series_coefficient, V_disk_coefficient, gamma_coefficient, lcell, ldisc, A, max_bound);
      }
      T ratio = Ndisc_double;
      ratio = (ratio-Ndisc)/(Ndiscmax-Ndisc);
      apply_cycle_to_cycle_noise(ratio, Ndiscmax, Ndiscmin, ldisc, A, Ndiscmax_std, Ndiscmin_std, ldisc_std, rdisc_std, ldisc_std_slope, rdisc_std_slope, rng,
                                 Ndiscmax_ctoc_upper_bound, Ndiscmax_ctoc_lower_bound, Ndiscmin_ctoc_upper_bound, Ndiscmin_ctoc_lower_bound,
                                 ldisc_ctoc_upper_bound, ldisc_ctoc_lower_bound, A_ctoc_upper_bound, A_ctoc_lower_bound);
      Ndisc_double = MIN(Ndisc_double, max_bound);
      w = map_Ndisc_to_weight(read_voltage, Ndisc_double, current_min, weight_min_bound, current_to_weight_ratio, g_read, h_read, j_0, k0, Original_Ndiscmin);
      Ndisc = Ndisc_double;
    }
  }else{
    if (Ndisc_double > min_bound)
    {
      for (int i = 0; i < pulse_counter; i++) {
        step_RESET(pulse_voltage_RESET, base_time_step, Ndisc_double, g_RESET, h_RESET, j_0, k0, T0, Original_Ndiscmin, Ndiscmin, Nplug, a_ny0, dWa, Rth_positive_coefficient,
        RseriesTiOx, R0, V_series_coefficient, V_disk_coefficient, gamma_coefficient, lcell, ldisc, A, min_bound);
      }
      T ratio = Ndisc_double;
      ratio = (Ndisc-ratio)/(Ndisc-Ndiscmin);
      apply_cycle_to_cycle_noise(ratio, Ndiscmax, Ndiscmin, ldisc, A, Ndiscmax_std, Ndiscmin_std, ldisc_std, rdisc_std, ldisc_std_slope, rdisc_std_slope, rng,
                                 Ndiscmax_ctoc_upper_bound, Ndiscmax_ctoc_lower_bound, Ndiscmin_ctoc_upper_bound, Ndiscmin_ctoc_lower_bound,
                                 ldisc_ctoc_upper_bound, ldisc_ctoc_lower_bound, A_ctoc_upper_bound, A_ctoc_lower_bound);
      Ndisc_double = MAX(Ndisc_double, min_bound);
      w = map_Ndisc_to_weight(read_voltage, Ndisc_double, current_min, weight_min_bound, current_to_weight_ratio, g_read, h_read, j_0, k0, Original_Ndiscmin);
      Ndisc = Ndisc_double;
    }
  } 
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

  PULSED_UPDATE_W_LOOP(update_once(par.read_voltage, par.pulse_voltage_SET, par.pulse_voltage_RESET, par.pulse_length, par.base_time_step,
                                   par.alpha_SET, par.beta_SET, par.c_SET, par.d_SET, par.f_SET, 
                                   par.g_RESET, par.h_RESET, par.g_read, par.h_read, par.j_0, par.k0,
                                   par.T0, par.Ndiscmin,
                                   Ndiscmax[j], Ndiscmin[j],
                                   par.Nplug, par.a_ny0, par.dWa,
                                   par.Rth_negative_coefficient, par.Rth_positive_coefficient, par.RseriesTiOx, par.R0,
                                   par.V_series_coefficient, par.V_disk_coefficient, par.gamma_coefficient,
                                   par.lcell,
                                   ldisc[j], A[j], Ndisc[j], w[j], sign,
                                   par.current_min, par.current_to_weight_ratio, par.weight_to_current_ratio, par.w_min,
                                   Ndisc_max_bound[j], Ndisc_min_bound[j],
                                  //  old implimentation
                                  //  par.Ndiscmax_std, par.Ndiscmax_ctoc_upper_bound, par.Ndiscmax_ctoc_lower_bound, 
                                  //  par.Ndiscmin_std, par.Ndiscmin_ctoc_upper_bound, par.Ndiscmin_ctoc_lower_bound, 
                                  //  par.ldisc_std, par.ldisc_std_slope, par.ldisc_ctoc_upper_bound, par.ldisc_ctoc_lower_bound, 
                                  //  par.rdisc_std, par.rdisc_std_slope, par.rdisc_ctoc_upper_bound, par.rdisc_ctoc_lower_bound, 
                                  par.Ndiscmax_std, Ndiscmax_ctoc_upper_bound[j], Ndiscmax_ctoc_lower_bound[j], 
                                  par.Ndiscmin_std, Ndiscmin_ctoc_upper_bound[j], Ndiscmin_ctoc_lower_bound[j], 
                                  par.ldisc_std, par.ldisc_std_slope, ldisc_ctoc_upper_bound[j], ldisc_ctoc_lower_bound[j], 
                                  par.rdisc_std, par.rdisc_std_slope, A_ctoc_upper_bound[j], A_ctoc_lower_bound[j], 
                                  rng););
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

  PULSED_UPDATE_W_LOOP_DENSE(update_once(par.read_voltage, par.pulse_voltage_SET, par.pulse_voltage_RESET, par.pulse_length, par.base_time_step,
                                         par.alpha_SET, par.beta_SET, par.c_SET, par.d_SET, par.f_SET, 
                                         par.g_RESET, par.h_RESET, par.g_read, par.h_read, par.j_0, par.k0,
                                         par.T0, par.Ndiscmin,
                                         Ndiscmax[j], Ndiscmin[j],
                                         par.Nplug, par.a_ny0, par.dWa,
                                         par.Rth_negative_coefficient, par.Rth_positive_coefficient, par.RseriesTiOx, par.R0,
                                         par.V_series_coefficient, par.V_disk_coefficient, par.gamma_coefficient,
                                         par.lcell,
                                         ldisc[j], A[j], Ndisc[j], w[j], sign,
                                         par.current_min, par.current_to_weight_ratio, par.weight_to_current_ratio, par.w_min,
                                         Ndisc_max_bound[j], Ndisc_min_bound[j],
                                        //  old implimentation
                                        //  par.Ndiscmax_std, par.Ndiscmax_ctoc_upper_bound, par.Ndiscmax_ctoc_lower_bound, 
                                        //  par.Ndiscmin_std, par.Ndiscmin_ctoc_upper_bound, par.Ndiscmin_ctoc_lower_bound, 
                                        //  par.ldisc_std, par.ldisc_std_slope, par.ldisc_ctoc_upper_bound, par.ldisc_ctoc_lower_bound, 
                                        //  par.rdisc_std, par.rdisc_std_slope, par.rdisc_ctoc_upper_bound, par.rdisc_ctoc_lower_bound, 
                                         par.Ndiscmax_std, Ndiscmax_ctoc_upper_bound[j], Ndiscmax_ctoc_lower_bound[j], 
                                         par.Ndiscmin_std, Ndiscmin_ctoc_upper_bound[j], Ndiscmin_ctoc_lower_bound[j], 
                                         par.ldisc_std, par.ldisc_std_slope, ldisc_ctoc_upper_bound[j], ldisc_ctoc_lower_bound[j], 
                                         par.rdisc_std, par.rdisc_std_slope, A_ctoc_upper_bound[j], A_ctoc_lower_bound[j], 
                                         rng););
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
    this->w_persistent_[0][i] = map_weight_to_Ndisc(w[i], par.current_min, par.w_min, par.weight_to_current_ratio,
                                              par.g_read, par.h_read, par.j_0, par.k0, par.Ndiscmin);
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
    this->w_persistent_[0][i] = map_weight_to_Ndisc(w[i], par.current_min, par.w_min, par.weight_to_current_ratio,
                                              par.g_read, par.h_read, par.j_0, par.k0, par.Ndiscmin);
  }
  return true;
}

template class JARTv1bRPUDevice<float>;
#ifdef RPU_USE_DOUBLE
template class JARTv1bRPUDevice<double>;
#endif

} // namespace RPU
