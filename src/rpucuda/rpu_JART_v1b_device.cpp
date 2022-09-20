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

#include "rpu_JART_v1b_device.h"

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
  if (par.Ndiscmax_ctoc_upper_bound < 0) {
    RPU_FATAL("Ndiscmax_ctoc_upper_bound needs to be 0 or positive.");
  }

  if (par.Ndiscmax_ctoc_lower_bound < 0) {
    RPU_FATAL("Ndiscmax_ctoc_lower_bound needs to be 0 or positive.");
  }

  if (par.Ndiscmin_ctoc_upper_bound < 0) {
    RPU_FATAL("Ndiscmin_ctoc_upper_bound needs to be 0 or positive.");
  }

  if (par.Ndiscmin_ctoc_lower_bound < 0) {
    RPU_FATAL("Ndiscmin_ctoc_lower_bound needs to be 0 or positive.");
  }

  if (par.ldet_ctoc_upper_bound < 0) {
    RPU_FATAL("ldet_ctoc_upper_bound needs to be 0 or positive.");
  }

  if (par.ldet_ctoc_lower_bound < 0) {
    RPU_FATAL("ldet_ctoc_lower_bound needs to be 0 or positive.");
  }

  if (par.rdet_ctoc_upper_bound < 0) {
    RPU_FATAL("rdet_ctoc_upper_bound needs to be 0 or positive.");
  }

  if (par.rdet_ctoc_lower_bound < 0) {
    RPU_FATAL("rdet_ctoc_lower_bound needs to be 0 or positive.");
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

  if (par.ldet_dtod_upper_bound < 0) {
    RPU_FATAL("ldet_dtod_upper_bound needs to be 0 or positive.");
  }

  if (par.ldet_dtod_lower_bound < 0) {
    RPU_FATAL("ldet_dtod_lower_bound needs to be 0 or positive.");
  }

  if (par.rdet_dtod_upper_bound < 0) {
    RPU_FATAL("rdet_dtod_upper_bound needs to be 0 or positive.");
  }

  if (par.rdet_dtod_lower_bound < 0) {
    RPU_FATAL("rdet_dtod_lower_bound needs to be 0 or positive.");
  }

  if (par.Ndiscmax_ctoc_upper_bound < par.Ndiscmax_ctoc_lower_bound) {
    RPU_FATAL("Ndiscmax_ctoc_upper_bound needs to be larger than Ndiscmax_ctoc_lower_bound.");
  }
  if (par.Ndiscmin_ctoc_upper_bound < par.Ndiscmin_ctoc_lower_bound) {
    RPU_FATAL("Ndiscmin_ctoc_upper_bound needs to be larger than Ndiscmin_ctoc_lower_bound.");
  }
  if (par.ldet_ctoc_upper_bound < par.ldet_ctoc_lower_bound) {
    RPU_FATAL("ldet_ctoc_upper_bound needs to be larger than ldet_ctoc_lower_bound.");
  }
  if (par.rdet_ctoc_upper_bound < par.rdet_ctoc_lower_bound) {
    RPU_FATAL("rdet_ctoc_upper_bound needs to be larger than rdet_ctoc_lower_bound.");
  }

  if (par.Ndiscmax_dtod_upper_bound < par.Ndiscmax_dtod_lower_bound) {
    RPU_FATAL("Ndiscmax_dtod_upper_bound needs to be larger than Ndiscmax_dtod_lower_bound.");
  }
  if (par.Ndiscmin_dtod_upper_bound < par.Ndiscmin_dtod_lower_bound) {
    RPU_FATAL("Ndiscmin_dtod_upper_bound needs to be larger than Ndiscmin_dtod_lower_bound.");
  }
  if (par.ldet_dtod_upper_bound < par.ldet_dtod_lower_bound) {
    RPU_FATAL("ldet_dtod_upper_bound needs to be larger than ldet_dtod_lower_bound.");
  }
  if (par.rdet_dtod_upper_bound < par.rdet_dtod_lower_bound) {
    RPU_FATAL("rdet_dtod_upper_bound needs to be larger than rdet_dtod_lower_bound.");
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

      device_specific_Ndiscmin[i][j] = par.Ndiscmin * (1 + par.Ndiscmin_dtod * rng->sampleGauss());
      if (par.Ndiscmin_dtod_upper_bound > (T)0.0) {
        device_specific_Ndiscmin[i][j] = MIN(device_specific_Ndiscmin[i][j], par.Ndiscmin_dtod_upper_bound);
      }
      device_specific_Ndiscmin[i][j] = MAX(device_specific_Ndiscmin[i][j], par.Ndiscmin_dtod_lower_bound);

      device_specific_ldet[i][j] = par.ldet * (1 + par.ldet_dtod * rng->sampleGauss());
      if (par.ldet_dtod_upper_bound > (T)0.0) {
        device_specific_ldet[i][j] = MIN(device_specific_ldet[i][j], par.ldet_dtod_upper_bound);
      }
      device_specific_ldet[i][j] = MAX(device_specific_ldet[i][j], par.ldet_dtod_lower_bound);

      T device_specific_rdet = par.rdet * (1 + par.rdet_dtod * rng->sampleGauss());
      if (par.rdet_dtod_upper_bound > (T)0.0) {
        device_specific_rdet = MIN(device_specific_rdet, par.rdet_dtod_upper_bound);
      }
      device_specific_rdet = MAX(device_specific_rdet, par.rdet_dtod_lower_bound);
      device_specific_A[i][j] = (T) M_PI*pow(device_specific_rdet,2.0);

      this->w_persistent_[i][j] = par.Ninit;
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
      std::cout << device_specific_ldet[i][j] << ",";
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
    const T &ldet,
    const T &A,
    const double &Ndisc) {
  Voltages_needed<T> Voltages;
  // V - V_series (V_disk+V_plug+V_Schottky)
  Voltages.other_than_V_series = applied_voltage - (I_mem*(RseriesTiOx + R0 + V_series_coefficient*I_mem*I_mem));
  // V_disk
  Voltages.V_disk =  I_mem*(ldet/(V_disk_coefficient*A*Ndisc));
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
    const T &Rth_negative,
    const T &RseriesTiOx,
    const T &R0,
    const T &V_series_coefficient,
    const T &V_disk_coefficient,
    const T &gamma_coefficient,
    const T &lcell,
    T &ldet,
    T &A,
    T &max_bound) {
    T I_mem = -alpha_SET-beta_SET/(pow((1.0+pow((c_SET/Ndisc),d_SET)),f_SET));

    // NOTE: V_disk = I_mem*(ldet/(V_disk_coefficient*A*Ndisc))
    // NOTE: Eion = V_disk/ldet
    T Eion = I_mem/(V_disk_coefficient*A*Ndisc);

    // NOTE: T gamma = gamma_coefficient*Eion
    T gamma = gamma_coefficient*Eion;
    
    // NOTE: V - V_series = V_disk+V_plug+V_Schottky
    T V_other_than_series = applied_voltage_SET - (I_mem*(RseriesTiOx + R0 + V_series_coefficient*I_mem*I_mem));

    T Treal = T0 + I_mem*V_other_than_series*Rth_negative;

    // NOTE: dWamin = dWa_f = dWa*(sqrt(1.0-pow(gamma,2.0))-(gamma*M_PI)/2+gamma*asin(gamma)) = dWa_mean - dWa_difference
    // NOTE: dWamax = dWa_r = dWa*(sqrt(1.0-pow(gamma,2.0))+(gamma*M_PI)/2+gamma*asin(gamma)) = dWa_mean + dWa_difference
    T dWa_mean = dWa*(sqrt(1.0-pow(gamma,2.0))+gamma*asin(gamma));
    T dWa_difference = dWa*((gamma*M_PI)/2.0);

    T denominator = PHYSICAL_PARAMETER_kb_over_e*Treal;

    T c_v0 = (Nplug+Ndisc)/2.0;
    T F1 = 1.0-pow((Ndisc/Ndiscmax),10.0);
    T dNdt = -(c_v0*a_ny0*F1*(exp(-(dWa_mean - dWa_difference)/denominator)-exp(-(dWa_mean + dWa_difference)/denominator)))/ldet;

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
    const T &Rth_positive,
    const T &RseriesTiOx,
    const T &R0,
    const T &V_series_coefficient,
    const T &V_disk_coefficient,
    const T &gamma_coefficient,
    const T &lcell,
    T &ldet,
    T &min_bound) {
  T I_mem = g_RESET/(pow((1.0+h_RESET*pow((Ndisc/Original_Ndiscmin),-j_0)),1.0/k0));
  
  // NOTE: V - V_series = V_disk+V_plug+V_Schottky
  T V_other_than_series = applied_voltage_RESET - (I_mem*(RseriesTiOx + R0 + V_series_coefficient*I_mem*I_mem));

  // NOTE: T gamma = gamma_coefficient*Eion
  T gamma = gamma_coefficient*V_other_than_series/lcell;

  T Treal = T0 + I_mem*V_other_than_series*Rth_positive;

  // NOTE: dWamin = dWa_f = dWa*(sqrt(1.0-pow(gamma,2.0))-(gamma*M_PI)/2+gamma*asin(gamma)) = dWa_mean - dWa_difference
  // NOTE: dWamax = dWa_r = dWa*(sqrt(1.0-pow(gamma,2.0))+(gamma*M_PI)/2+gamma*asin(gamma)) = dWa_mean + dWa_difference
  T dWa_mean = dWa*(sqrt(1.0-pow(gamma,2.0))+gamma*asin(gamma));
  T dWa_difference = dWa*((gamma*M_PI)/2.0);

  T denominator = PHYSICAL_PARAMETER_kb_over_e*Treal;

  T c_v0 = (Nplug+Ndisc)/2.0;
  T F1 = 1.0-pow((Ndiscmin/Ndisc),10.0);
  T dNdt = -(c_v0*a_ny0*F1*(exp(-(dWa_mean - dWa_difference)/denominator)-exp(-(dWa_mean + dWa_difference)/denominator)))/ldet;

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
    T &ldet,
    T &A,
    const T &Ndiscmax_std,
    const T &Ndiscmin_std,
    const T &ldet_std,
    const T &rdet_std,
    const T &ldet_std_slope,
    const T &rdet_std_slope,
    RNG<T> *rng,
    const T &Ndiscmax_ctoc_upper_bound,
    const T &Ndiscmax_ctoc_lower_bound,
    const T &Ndiscmin_ctoc_upper_bound,
    const T &Ndiscmin_ctoc_lower_bound,
    const T &ldet_ctoc_upper_bound,
    const T &ldet_ctoc_lower_bound,
    const T &rdet_ctoc_upper_bound,
    const T &rdet_ctoc_lower_bound) {
  if (Ndiscmax_std > (T)0.0) {
    Ndiscmax = Ndiscmax * (1 + Ndiscmax_std * (2*rng->sampleUniform()-1));
    if (Ndiscmax_ctoc_upper_bound > (T)0.0) {
      Ndiscmax = MIN(Ndiscmax, Ndiscmax_ctoc_upper_bound);
    }
    Ndiscmax = MAX(Ndiscmax, Ndiscmax_ctoc_lower_bound);
  }
  if (Ndiscmin_std > (T)0.0) {
    Ndiscmin = Ndiscmin * (1 + Ndiscmin_std * (2*rng->sampleUniform()-1));
    if (Ndiscmin_ctoc_upper_bound > (T)0.0) {
      Ndiscmin = MIN(Ndiscmin, Ndiscmin_ctoc_upper_bound);
    }
    Ndiscmin = MAX(Ndiscmin, Ndiscmin_ctoc_lower_bound);
  }
  if ((ldet_std > (T)0.0)||(ldet_std_slope > (T)0.0)) {
    ldet = ldet * (1 + ldet_std * (2*rng->sampleUniform()-1) + ratio * ldet_std_slope * (2*rng->sampleUniform()-1));
    if (ldet_ctoc_upper_bound > (T)0.0) {
      ldet = MIN(ldet, ldet_ctoc_upper_bound);
    }
    ldet = MAX(ldet, ldet_ctoc_lower_bound);
  }
  if ((rdet_std > (T)0.0)||(rdet_std_slope > (T)0.0)) {
    T rdet = pow(A/M_PI, 0.5) * (1 + rdet_std * (2*rng->sampleUniform()-1) + ratio * rdet_std_slope * (2*rng->sampleUniform()-1));
    if (rdet_ctoc_upper_bound > (T)0.0) {
      rdet = MIN(rdet, rdet_ctoc_upper_bound);
    }
    rdet = MAX(rdet, rdet_ctoc_lower_bound);
    A = M_PI*pow(rdet,2.0);
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
    const T &Rth_negative,
    const T &Rth_positive,
    const T &RseriesTiOx,
    const T &R0,
    const T &V_series_coefficient,
    const T &V_disk_coefficient,
    const T &gamma_coefficient,
    const T &lcell,
    T &ldet,
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
    const T &ldet_std,
    const T &ldet_std_slope,
    const T &ldet_ctoc_upper_bound,
    const T &ldet_ctoc_lower_bound,
    const T &rdet_std,
    const T &rdet_std_slope,
    const T &rdet_ctoc_upper_bound,
    const T &rdet_ctoc_lower_bound,
    RNG<T> *rng) {
  int pulse_counter = int (pulse_length/base_time_step);
  double Ndisc_double = Ndisc;
  T max_bound = MIN(device_specific_Ndisc_max_bound, Ndiscmax);
  T min_bound = MAX(device_specific_Ndisc_min_bound, Ndiscmin);

  if (sign < 0) {
    if (Ndisc_double < max_bound)
    {
      for (int i = 0; i < pulse_counter; i++) {
        step_SET(pulse_voltage_SET, base_time_step, Ndisc_double, alpha_SET, beta_SET, c_SET, d_SET, f_SET, T0, Ndiscmax, Nplug, a_ny0, dWa, Rth_negative,
        RseriesTiOx, R0, V_series_coefficient, V_disk_coefficient, gamma_coefficient, lcell, ldet, A, max_bound);
      }
      T ratio = Ndisc_double;
      ratio = (ratio-Ndisc)/(Ndiscmax-Ndisc);
      apply_cycle_to_cycle_noise(ratio, Ndiscmax, Ndiscmin, ldet, A, Ndiscmax_std, Ndiscmin_std, ldet_std, rdet_std, ldet_std_slope, rdet_std_slope, rng,
                                 Ndiscmax_ctoc_upper_bound, Ndiscmax_ctoc_lower_bound, Ndiscmin_ctoc_upper_bound, Ndiscmin_ctoc_lower_bound,
                                 ldet_ctoc_upper_bound, ldet_ctoc_lower_bound, rdet_ctoc_upper_bound, rdet_ctoc_lower_bound);
      Ndisc_double = MIN(Ndisc_double, max_bound);
      w = map_Ndisc_to_weight(read_voltage, Ndisc_double, current_min, weight_min_bound, current_to_weight_ratio, g_read, h_read, j_0, k0, Original_Ndiscmin);
      Ndisc = Ndisc_double;
    }
  }else{
    if (Ndisc_double > min_bound)
    {
      for (int i = 0; i < pulse_counter; i++) {
        step_RESET(pulse_voltage_RESET, base_time_step, Ndisc_double, g_RESET, h_RESET, j_0, k0, T0, Original_Ndiscmin, Ndiscmin, Nplug, a_ny0, dWa, Rth_positive,
        RseriesTiOx, R0, V_series_coefficient, V_disk_coefficient, gamma_coefficient, lcell, ldet, min_bound);
      }
      T ratio = Ndisc_double;
      ratio = (Ndisc-ratio)/(Ndisc-Ndiscmin);
      apply_cycle_to_cycle_noise(ratio, Ndiscmax, Ndiscmin, ldet, A, Ndiscmax_std, Ndiscmin_std, ldet_std, rdet_std, ldet_std_slope, rdet_std_slope, rng,
                                 Ndiscmax_ctoc_upper_bound, Ndiscmax_ctoc_lower_bound, Ndiscmin_ctoc_upper_bound, Ndiscmin_ctoc_lower_bound,
                                 ldet_ctoc_upper_bound, ldet_ctoc_lower_bound, rdet_ctoc_upper_bound, rdet_ctoc_lower_bound);
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
  T *ldet = device_specific_ldet[i];
  T *A = device_specific_A[i];
  T *Ndisc_max_bound = device_specific_Ndisc_max_bound[i];
  T *Ndisc_min_bound = device_specific_Ndisc_min_bound[i];

  PULSED_UPDATE_W_LOOP(update_once(par.read_voltage, par.pulse_voltage_SET, par.pulse_voltage_RESET, par.pulse_length, par.base_time_step,
                                   par.alpha_SET, par.beta_SET, par.c_SET, par.d_SET, par.f_SET, 
                                   par.g_RESET, par.h_RESET, par.g_read, par.h_read, par.j_0, par.k0,
                                   par.T0, par.Ndiscmin,
                                   Ndiscmax[j], Ndiscmin[j],
                                   par.Nplug, par.a_ny0, par.dWa,
                                   par.Rth_negative, par.Rth_positive, par.RseriesTiOx, par.R0,
                                   par.V_series_coefficient, par.V_disk_coefficient, par.gamma_coefficient,
                                   par.lcell,
                                   ldet[j], A[j], Ndisc[j], w[j], sign,
                                   par.current_min, par.current_to_weight_ratio, par.weight_to_current_ratio, par.w_min,
                                   Ndisc_max_bound[j], Ndisc_min_bound[j],
                                   par.Ndiscmax_std, par.Ndiscmax_ctoc_upper_bound, par.Ndiscmax_ctoc_lower_bound, 
                                   par.Ndiscmin_std, par.Ndiscmin_ctoc_upper_bound, par.Ndiscmin_ctoc_lower_bound, 
                                   par.ldet_std, par.ldet_std_slope, par.ldet_ctoc_upper_bound, par.ldet_ctoc_lower_bound, 
                                   par.rdet_std, par.rdet_std_slope, par.rdet_ctoc_upper_bound, par.rdet_ctoc_lower_bound, 
                                   rng););
}

template <typename T>
void JARTv1bRPUDevice<T>::doDenseUpdate(T **weights, int *coincidences, RNG<T> *rng) {

  const auto &par = getPar();

  T *w = weights[0];
  T *Ndisc = this->w_persistent_[0];
  T *Ndiscmax = device_specific_Ndiscmax[0];
  T *Ndiscmin = device_specific_Ndiscmin[0];
  T *ldet = device_specific_ldet[0];
  T *A = device_specific_A[0];
  T *Ndisc_max_bound = device_specific_Ndisc_max_bound[0];
  T *Ndisc_min_bound = device_specific_Ndisc_min_bound[0];

  PULSED_UPDATE_W_LOOP_DENSE(update_once(par.read_voltage, par.pulse_voltage_SET, par.pulse_voltage_RESET, par.pulse_length, par.base_time_step,
                                         par.alpha_SET, par.beta_SET, par.c_SET, par.d_SET, par.f_SET, 
                                         par.g_RESET, par.h_RESET, par.g_read, par.h_read, par.j_0, par.k0,
                                         par.T0, par.Ndiscmin,
                                         Ndiscmax[j], Ndiscmin[j],
                                         par.Nplug, par.a_ny0, par.dWa,
                                         par.Rth_negative, par.Rth_positive, par.RseriesTiOx, par.R0,
                                         par.V_series_coefficient, par.V_disk_coefficient, par.gamma_coefficient,
                                         par.lcell,
                                         ldet[j], A[j], Ndisc[j], w[j], sign,
                                         par.current_min, par.current_to_weight_ratio, par.weight_to_current_ratio, par.w_min,
                                         Ndisc_max_bound[j], Ndisc_min_bound[j],
                                         par.Ndiscmax_std, par.Ndiscmax_ctoc_upper_bound, par.Ndiscmax_ctoc_lower_bound, 
                                         par.Ndiscmin_std, par.Ndiscmin_ctoc_upper_bound, par.Ndiscmin_ctoc_lower_bound, 
                                         par.ldet_std, par.ldet_std_slope, par.ldet_ctoc_upper_bound, par.ldet_ctoc_lower_bound, 
                                         par.rdet_std, par.rdet_std_slope, par.rdet_ctoc_upper_bound, par.rdet_ctoc_lower_bound, 
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
