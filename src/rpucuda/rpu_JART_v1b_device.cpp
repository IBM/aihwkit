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
void JARTv1bRPUDevice<T>::populate(
    const JARTv1bRPUDeviceMetaParameter<T> &p, RealWorldRNG<T> *rng) {

  PulsedRPUDevice<T>::populate(p, rng); // will clone par
  auto &par = getPar();

  for (int i = 0; i < this->d_size_; ++i) {

    for (int j = 0; j < this->x_size_; ++j) {
      device_specific_Ndiscmax[i][j] = par.Ndiscmax + par.Ndiscmax_dtod * rng->sampleGauss();
      device_specific_Ndiscmin[i][j] = par.Ndiscmin + par.Ndiscmin_dtod * rng->sampleGauss();
      device_specific_ldet[i][j] = par.ldet + par.ldet_dtod * rng->sampleGauss();
      T device_specific_rdet = par.rdet + par.rdet_dtod * rng->sampleGauss();
      device_specific_A[i][j] = (T) M_PI*pow(device_specific_rdet,2);
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

    T V_disk = I_mem*(ldet/(V_disk_coefficient*A*Ndisc));

    // T gamma = gamma_coefficient*Eion;
    T gamma = gamma_coefficient*V_disk/ldet;
    
    // V - V_series = V_disk+V_plug+V_Schottky
    T V_other_than_series = applied_voltage_SET - (I_mem*(RseriesTiOx + R0 + V_series_coefficient*I_mem*I_mem));

    T Treal = T0 + I_mem*V_other_than_series*Rth_negative;
    // // dWamin
    // T dWa_f = dWa*(sqrt(1.0-pow(gamma,2.0))-(gamma*M_PI)/2+gamma*asin(gamma));
    // // dWamax
    // T dWa_r = dWa*(sqrt(1.0-pow(gamma,2.0))+(gamma*M_PI)/2+gamma*asin(gamma));

    T dWa_mean = dWa*(sqrt(1.0-pow(gamma,2.0))+gamma*asin(gamma));
    T dWa_difference = dWa*((gamma*M_PI)/2.0);
    // dWamin = dWa_f = dWa_mean - dWa_difference
    // dWamax = dWa_r = dWa_mean + dWa_difference

    T denominator = PHYSICAL_PARAMETER_kb_over_e*Treal;

    T c_v0 = (Nplug+Ndisc)/2.0;
    T F1 = 1.0-pow((Ndisc/Ndiscmax),10.0);
    T dNdt = -(c_v0*a_ny0*F1*(exp(-(dWa_mean - dWa_difference)/denominator)-exp(-(dWa_mean + dWa_difference)/denominator)))/ldet;

    Ndisc = Ndisc + dNdt*time_step;

    Ndisc = MIN(Ndisc, max_bound);
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
    T &A,
    T &min_bound) {
  T I_mem = g_RESET/(pow((1.0+h_RESET*pow((Ndisc/Ndiscmin),-j_0)),1.0/k0));
  
  // V - V_series = V_disk+V_plug+V_Schottky
  T V_other_than_series = applied_voltage_RESET - (I_mem*(RseriesTiOx + R0 + V_series_coefficient*I_mem*I_mem));

  // T gamma = gamma_coefficient*Eion;
  T gamma = gamma_coefficient*V_other_than_series/lcell;

  T Treal = T0 + I_mem*V_other_than_series*Rth_positive;
  // // dWamin
  // T dWa_f = dWa*(sqrt(1.0-pow(gamma,2.0))-(gamma*M_PI)/2+gamma*asin(gamma));
  // // dWamax
  // T dWa_r = dWa*(sqrt(1.0-pow(gamma,2.0))+(gamma*M_PI)/2+gamma*asin(gamma));

  T dWa_mean = dWa*(sqrt(1.0-pow(gamma,2.0))+gamma*asin(gamma));
  T dWa_difference = dWa*((gamma*M_PI)/2.0);
  // dWamin = dWa_f = dWa_mean - dWa_difference
  // dWamax = dWa_r = dWa_mean + dWa_difference

  T denominator = PHYSICAL_PARAMETER_kb_over_e*Treal;

  T c_v0 = (Nplug+Ndisc)/2.0;
  T F1 = 1.0-pow((Ndiscmin/Ndisc),10.0);
  T dNdt = -(c_v0*a_ny0*F1*(exp(-(dWa_mean - dWa_difference)/denominator)-exp(-(dWa_mean + dWa_difference)/denominator)))/ldet;

  Ndisc = Ndisc + dNdt*time_step;

  Ndisc = MAX(Ndisc, min_bound);
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
    T &Ndiscmax,
    T &Ndiscmin,
    T &ldet,
    T &A,
    const T &Ndiscmax_std,
    const T &Ndiscmin_std,
    const T &ldet_std,
    const T &rdet_std,
    RNG<T> *rng) {
  if (Ndiscmax_std > (T)0.0) {
    Ndiscmax = Ndiscmax + Ndiscmax_std * rng->sampleGauss();
  }
  if (Ndiscmin_std > (T)0.0) {
    Ndiscmin = Ndiscmin + Ndiscmin_std * rng->sampleGauss();
  }
  if (ldet_std > (T)0.0) {
    ldet = ldet + ldet_std * rng->sampleGauss();
  }
  if (rdet_std > (T)0.0) {
    T rdet = pow(A/M_PI, 0.5) + rdet_std * rng->sampleGauss();
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
    const T &Ndisc_min_bound,
    const T &Ndisc_max_bound, 
    const T &Ndiscmax_std,
    const T &Ndiscmin_std,
    const T &ldet_std,
    const T &rdet_std,
    RNG<T> *rng) {
  int pulse_counter = int (pulse_length/base_time_step);
  double Ndisc_double = Ndisc;
  T max_bound = MIN(Ndisc_max_bound, Ndiscmax);
  T min_bound = MAX(Ndisc_min_bound, Ndiscmin);

  if (sign < 0) {
    if (Ndisc_double >= max_bound)
    {
      Ndisc_double = max_bound;
    }
    else
    {
      for (int i = 0; i < pulse_counter; i++) {
        step_SET(pulse_voltage_SET, base_time_step, Ndisc_double, alpha_SET, beta_SET, c_SET, d_SET, f_SET, T0, Ndiscmax, Nplug, a_ny0, dWa, Rth_negative, RseriesTiOx, R0, V_series_coefficient, V_disk_coefficient, gamma_coefficient, lcell, ldet, A, max_bound);
      }
    }
  }else{
    if (Ndisc_double <= min_bound)
    {
      Ndisc_double = min_bound;
    }
    else
    {
      for (int i = 0; i < pulse_counter; i++) {
        step_RESET(pulse_voltage_RESET, base_time_step, Ndisc_double, g_RESET, h_RESET, j_0, k0, T0, Original_Ndiscmin, Ndiscmin, Nplug, a_ny0, dWa, Rth_positive, RseriesTiOx, R0, V_series_coefficient, V_disk_coefficient, gamma_coefficient, lcell, ldet, A, min_bound);
      }
    }
  } 

  w = map_Ndisc_to_weight(read_voltage, Ndisc_double, current_min, weight_min_bound, current_to_weight_ratio, g_read, h_read, j_0, k0, Original_Ndiscmin);
  Ndisc = Ndisc_double;
  apply_cycle_to_cycle_noise(Ndiscmax, Ndiscmin, ldet, A, Ndiscmax_std, Ndiscmin_std, ldet_std, rdet_std, rng);
}

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
void JARTv1bRPUDevice<T>::doSparseUpdate(
    T **weights, int i, const int *x_signed_indices, int x_count, int d_sign, RNG<T> *rng) {

  const auto &par = getPar();

  T *w = weights[i];
  T *Ndisc = this->w_persistent_[i];
  T *Ndiscmax = device_specific_Ndiscmax[i];
  T *Ndiscmin = device_specific_Ndiscmin[i];
  T *ldet = device_specific_ldet[i];
  T *A = device_specific_A[i];
  // T *min_bound = this->w_min_bound_[i];
  // T *max_bound = this->w_max_bound_[i];

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
                                  //  min_bound[j], max_bound[j],
                                   par.Ndisc_min_bound, par.Ndisc_max_bound,
                                   par.Ndiscmax_std, par.Ndiscmin_std, par.ldet_std, par.rdet_std,
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
  // T *min_bound = this->w_min_bound_[0];
  // T *max_bound = this->w_max_bound_[0];

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
                                         //  min_bound[j], max_bound[j],
                                         par.Ndisc_min_bound, par.Ndisc_max_bound,
                                         par.Ndiscmax_std, par.Ndiscmin_std, par.ldet_std, par.rdet_std,
                                         rng););
}



// template <typename T> void JARTv1bRPUDevice<T>::decayWeights(T **weights, bool bias_no_decay) {

//   // maybe a bit overkill to check the bounds...
//   T *w = weights[0];
//   T *wd = this->w_decay_scale_[0];
//   T *max_bound = this->w_max_bound_[0];
//   T *min_bound = this->w_min_bound_[0];
//   T *b = this->w_reset_bias_[0];

//   if (!bias_no_decay) {
//     PRAGMA_SIMD
//     for (int i = 0; i < this->size_; ++i) {
//       w[i] = (w[i] - b[i]) * wd[i] + b[i];
//       w[i] = MIN(w[i], max_bound[i]);
//       w[i] = MAX(w[i], min_bound[i]);
//     }
//   } else {
//     const int last_col = this->x_size_ - 1; // x-major (ie row major)
//     PRAGMA_SIMD
//     for (int i = 0; i < this->size_; ++i) {
//       T s = (i % this->x_size_ == last_col) ? (T)1.0 : wd[i];
//       w[i] = (w[i] - b[i]) * s + b[i];
//       w[i] = MIN(w[i], max_bound[i]);
//       w[i] = MAX(w[i], min_bound[i]);
//     }
//   }

//   const auto &par = getPar();

//   PRAGMA_SIMD
//   for (int i = 0; i < this->size_; i++) {
//     this->w_persistent_[0][i] = map_weight_to_Ndisc(w[i], par.current_min, par.w_min, par.weight_to_current_ratio,
//                                               par.g_read, par.h_read, par.j_0, par.k0, par.Ndiscmin);
//   }
// }

// template <typename T>
// void JARTv1bRPUDevice<T>::decayWeights(T **weights, T alpha, bool bias_no_decay) {

//   // maybe a bit overkill to check the bounds...
//   T *w = weights[0];
//   T *wd = this->w_decay_scale_[0];
//   T *max_bound = this->w_max_bound_[0];
//   T *min_bound = this->w_min_bound_[0];
//   T *b = this->w_reset_bias_[0];

//   if (!bias_no_decay) {
//     PRAGMA_SIMD
//     for (int i = 0; i < this->size_; ++i) {
//       T s = 1 + alpha * (wd[i] - 1);
//       w[i] = (w[i] - b[i]) * s + b[i];
//       w[i] = MIN(w[i], max_bound[i]);
//       w[i] = MAX(w[i], min_bound[i]);
//     }
//   } else {
//     const int last_col = this->x_size_ - 1; // x-major (ie row major)
//     PRAGMA_SIMD
//     for (int i = 0; i < this->size_; ++i) {
//       T s = (i % this->x_size_ == last_col) ? (T)1.0 : (1 + alpha * (wd[i] - 1));
//       w[i] = (w[i] - b[i]) * s + b[i];
//       w[i] = MIN(w[i], max_bound[i]);
//       w[i] = MAX(w[i], min_bound[i]);
//     }
//   }

//   const auto &par = getPar();

//   PRAGMA_SIMD
//   for (int i = 0; i < this->size_; i++) {
//     this->w_persistent_[0][i] = map_weight_to_Ndisc(w[i], par.current_min, par.w_min, par.weight_to_current_ratio,
//                                               par.g_read, par.h_read, par.j_0, par.k0, par.Ndiscmin);
//   }
// }

template <typename T>
void JARTv1bRPUDevice<T>::driftWeights(T **weights, T time_since_last_call, RNG<T> &rng) {
  if (this->hasWDrifter()) {
    T **w = weights;
    PulsedRPUDeviceBase<T>::driftWeights(w, time_since_last_call, rng);
    this->wdrifter_->saturate(w[0], this->w_min_bound_[0], this->w_max_bound_[0]);
    
    const auto &par = getPar();

    PRAGMA_SIMD
    for (int i = 0; i < this->size_; i++) {
    this->w_persistent_[0][i] = map_weight_to_Ndisc(w[0][i], par.current_min, par.w_min, par.weight_to_current_ratio,
                                              par.g_read, par.h_read, par.j_0, par.k0, par.Ndiscmin);
    }
  }
}

// template <typename T> void JARTv1bRPUDevice<T>::diffuseWeights(T **weights, RNG<T> &rng) {

//   T *w = weights[0];
//   T *diffusion_rate = &(this->w_diffusion_rate_[0][0]);
//   T *max_bound = &(this->w_max_bound_[0][0]);
//   T *min_bound = &(this->w_min_bound_[0][0]);

//   PRAGMA_SIMD
//   for (int i = 0; i < this->size_; ++i) {
//     w[i] += diffusion_rate[i] * rng.sampleGauss();
//     w[i] = MIN(w[i], max_bound[i]);
//     w[i] = MAX(w[i], min_bound[i]);
//   }

//   const auto &par = getPar();

//   PRAGMA_SIMD
//   for (int i = 0; i < this->size_; i++) {
//     this->w_persistent_[0][i] = map_weight_to_Ndisc(w[i], par.current_min, par.w_min, par.weight_to_current_ratio,
//                                               par.g_read, par.h_read, par.j_0, par.k0, par.Ndiscmin);
//   }
// }

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

  PRAGMA_SIMD
  for (int i = 0; i < this->size_; i++) {
    this->w_persistent_[0][i] = map_weight_to_Ndisc(w[i], par.current_min, par.w_min, par.weight_to_current_ratio,
                                              par.g_read, par.h_read, par.j_0, par.k0, par.Ndiscmin);
  }
}

// template <typename T>
// void JARTv1bRPUDevice<T>::resetCols(
//     T **weights, int start_col, int n_col, T reset_prob, RealWorldRNG<T> &rng) {

//   T reset_std = getPar().reset_std;
//   for (int j = 0; j < this->x_size_; ++j) {
//     if ((start_col + n_col <= this->x_size_ && j >= start_col && j < start_col + n_col) ||
//         (start_col + n_col > this->x_size_ &&
//          ((j >= start_col) || (j < n_col - (this->x_size_ - start_col))))) {
//       PRAGMA_SIMD
//       for (int i = 0; i < this->d_size_; ++i) {
//         if (reset_prob == 1 || rng.sampleUniform() < reset_prob) {
//           weights[i][j] =
//               this->w_reset_bias_[i][j] + (reset_std > 0 ? reset_std * rng.sampleGauss() : (T)0.0);
//           weights[i][j] = MIN(weights[i][j], this->w_max_bound_[i][j]);
//           weights[i][j] = MAX(weights[i][j], this->w_min_bound_[i][j]);
//         }
//       }
//     }
//   }

//   const auto &par = getPar();

//   PRAGMA_SIMD
//   for (int i = 0; i < this->size_; i++) {
//     this->w_persistent_[0][i] = map_weight_to_Ndisc(weights[0][i], par.current_min, par.w_min, par.weight_to_current_ratio,
//                                               par.g_read, par.h_read, par.j_0, par.k0, par.Ndiscmin);
//   }
// }

// template <typename T>
// void JARTv1bRPUDevice<T>::resetAtIndices(
//     T **weights, std::vector<int> x_major_indices, RealWorldRNG<T> &rng) {

//   T reset_std = getPar().reset_std;

//   for (const auto &index : x_major_indices) {
//     int i = index / this->x_size_;
//     int j = index % this->x_size_;

//     weights[i][j] = this->w_reset_bias_[i][j] + (reset_std > 0 ? reset_std * rng.sampleGauss() : (T)0.0);
//     weights[i][j] = MIN(weights[i][j], this->w_max_bound_[i][j]);
//     weights[i][j] = MAX(weights[i][j], this->w_min_bound_[i][j]);
//   }

//   const auto &par = getPar();

//   PRAGMA_SIMD
//   for (int i = 0; i < this->size_; i++) {
//     this->w_persistent_[0][i] = map_weight_to_Ndisc(weights[0][i], par.current_min, par.w_min, par.weight_to_current_ratio,
//                                               par.g_read, par.h_read, par.j_0, par.k0, par.Ndiscmin);
//   }
// }

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
