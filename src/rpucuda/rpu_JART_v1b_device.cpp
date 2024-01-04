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
  _current_min = (T)(-g0 * (exp(-g1 * read_voltage) - (T)1.0)) /
                    (pow(((T)1.0 + (h0 + h1 * read_voltage + h2 * exp(-h3 * read_voltage)) *
			  pow((Ndisc_min_bound / Ndiscmin), (-j_0))),
			 ((T)1.0 / k0)));
  _current_max = (T)(-g0 * (exp(-g1 * read_voltage) - (T)1.0)) /
                    (pow(((T)1.0 + (h0 + h1 * read_voltage + h2 * exp(-h3 * read_voltage)) *
			  pow((Ndisc_max_bound / Ndiscmin), (-j_0))),
                        ((T)1.0 / k0)));
  // from [0.0001:1000], initial oxygen vacancy concentration in the disc [10^26/m^3]

  _Ninit =
        pow(((pow(((-g0 * (exp(-g1 * read_voltage) - 1)) /
                   (((0 - w_min) / (w_max - w_min)) * (_current_max - _current_min) + _current_min)),
                  k0) - (T)1) /(h0 + h1 * read_voltage + h2 * exp(-h3 * read_voltage))),
            ((T)1.0 / -j_0)) *  Ndiscmin;
  
  T alpha1 = alpha0 * exp(-alpha2 / alpha3);
  _alpha_SET = ((alpha1 + alpha0) / ((T)1.0 + exp(-(pulse_voltage_SET + alpha2) / alpha3))) -
              alpha0;
  _beta_SET = beta1 * ((T)1.0 - exp(-pulse_voltage_SET)) - beta0 * pulse_voltage_SET;
  
  _c_SET = c2 * exp(-pulse_voltage_SET / c3) + c1 * pulse_voltage_SET - c0;
  _d_SET = d2 * exp(-pulse_voltage_SET / d3) + d1 * pulse_voltage_SET - d0;
  _f_SET = f0 + ((f1 - f0) / ((T)1.0 + pow((-pulse_voltage_SET / f2), f3)));
  
  _g_RESET = (T)-g0 * (exp(-g1 * pulse_voltage_RESET) - (T)1.0);
  _h_RESET = (T)h0 + h1 * pulse_voltage_RESET + h2 * exp(-h3 * pulse_voltage_RESET);

  _g_read = (T)-g0 * (exp(-g1 * read_voltage) - 1);
  _h_read = (T)h0 + h1 * read_voltage + h2 * exp(-h3 * read_voltage);
  
  _Original_A = (T)M_PI * pow(rdisc, 2.0);
  
  _Rth_negative_coefficient = (T)Rth0 * _Original_A;
  _Rth_positive_coefficient = (T)Rth0 * Rtheff_scaling * _Original_A;
  
  _V_series_coefficient = (T)R0 * alphaline * R0 * Rthline;
  _V_disk_coefficient = (T)PHYSICAL_PARAMETER_zvo * PHYSICAL_PARAMETER_e * un * 1e26;
  
  _gamma_coefficient = (T)(PHYSICAL_PARAMETER_zvo * a) / (dWa * M_PI);
  _a_ny0 = (T)a * ny0;
  
}

template <typename T>
T JARTv1bRPUDeviceMetaParameter<T>::mapWeight2Ndisc(T weight) const {

  T weight_to_current_ratio = (T)(_current_max - _current_min) / (w_max - w_min);
 // not that this uses the global bounds. Should this not be the device bounds?
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

  // re-use parameter structures
  T ** device_specific_Ndiscmax = this->w_max_bound_;
  T ** device_specific_Ndiscmin = this->w_min_bound_;

  
  for (int i = 0; i < this->d_size_; ++i) {
    for (int j = 0; j < this->x_size_; ++j) {

      device_specific_Ndiscmax[i][j] = par.mapWeight2Ndisc(this->w_max_bound_[i][j]);
      device_specific_Ndiscmin[i][j] = par.mapWeight2Ndisc(this->w_min_bound_[i][j]);

      device_specific_ldisc[i][j] = MAX(par.ldisc * ((T)1.0 + par.ldisc_dtod * rng->sampleGauss()), (T) 0.0);
      
      T device_specific_rdisc = MAX(par.rdisc * (1 + par.rdisc_dtod * rng->sampleGauss()), (T) 0.0);
      device_specific_A[i][j] = (T)M_PI * pow(device_specific_rdisc, 2.0);

      // just set to same value ?
      device_specific_Ndisc[i][j] = par._Ninit;
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
      std::cout << ", " << this->device_specific_Ndisc_[i][j];
      std::cout << "]";
    }
    std::cout << std::endl;
  }
}

template <typename T>
inline void step_SET(
    double &Ndisc,
    const T &Ndiscmax,
    const T &Ndiscmin,
    const T &ldisc,
    const T &A,
    const JARTv1bRPUDeviceMetaParameter<T>& par,
    RNG<T> *rng) {

  T I_mem = -par._alpha_SET - par._beta_SET / (pow((1.0 + pow((par._c_SET / Ndisc), par._d_SET)), par._f_SET));
  T Eion = I_mem / (par._V_disk_coefficient * A * Ndisc);

  T gamma = par._gamma_coefficient * Eion;
  T V_other_than_series =
      par._applied_voltage_SET - (I_mem * (par._RseriesTiOx + par._R0 + par._V_series_coefficient * I_mem * I_mem));

  T Treal = par.T0 + I_mem * V_other_than_series * par._Rth_negative_coefficient / A;

  T dWa_mean = par.dWa * (sqrt(1.0 - pow(gamma, 2.0)) + gamma * asin(gamma));
  T dWa_difference = par.dWa * ((gamma * M_PI) / 2.0);

  T denominator = PHYSICAL_PARAMETER_kb_over_e * Treal;

  T c_v0 = (par.Nplug + Ndisc) / 2.0;
  T F_limit = 1.0 - pow((Ndisc / Ndiscmax), 10.0);
  T dNdt = -(c_v0 * par._a_ny0 * F_limit *
             (exp(-(dWa_mean - dWa_difference) / denominator) -
              exp(-(dWa_mean + dWa_difference) / denominator))) /
           ldisc;
  if (par.dNdt_std > (T) 0.0) {
    dNdt *= (T)1.0 + par.dNdt_std * rng->sampleGauss();
  }
  Ndisc = Ndisc + dNdt * par.time_step;
  
  if (par.Ndisc_std > (T) 0.0) {
    Ndisc *= abs((T)1.0 + par.Ndisc_std * rng->sampleGauss());
  }
  Ndisc = MIN(Ndisc, Ndiscmax);
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
  T I_mem = g_RESET / (pow((1.0 + h_RESET * pow((Ndisc / Original_Ndiscmin), -j_0)), 1.0 / k0));

  // NOTE: V - V_series = V_disk+V_plug+V_Schottky
  T V_other_than_series =
      applied_voltage_RESET - (I_mem * (RseriesTiOx + R0 + V_series_coefficient * I_mem * I_mem));

  // NOTE: T gamma = gamma_coefficient*Eion
  T gamma = gamma_coefficient * V_other_than_series / lcell;

  T Treal = T0 + I_mem * V_other_than_series * Rth_positive_coefficient / A;

  // NOTE: dWamin = dWa_f = dWa*(sqrt(1.0-pow(gamma,2.0))-(gamma*M_PI)/2+gamma*asin(gamma)) =
  // dWa_mean - dWa_difference NOTE: dWamax = dWa_r =
  // dWa*(sqrt(1.0-pow(gamma,2.0))+(gamma*M_PI)/2+gamma*asin(gamma)) = dWa_mean + dWa_difference
  T dWa_mean = dWa * (sqrt(1.0 - pow(gamma, 2.0)) + gamma * asin(gamma));
  T dWa_difference = dWa * ((gamma * M_PI) / 2.0);

  T denominator = PHYSICAL_PARAMETER_kb_over_e * Treal;

  T c_v0 = (Nplug + Ndisc) / 2.0;
  T F_limit = 1.0 - pow((Ndiscmin / Ndisc), 10.0);
  T dNdt = -(c_v0 * a_ny0 * F_limit *
             (exp(-(dWa_mean - dWa_difference) / denominator) -
              exp(-(dWa_mean + dWa_difference) / denominator))) /
           ldisc;

  Ndisc = Ndisc + dNdt * time_step;
}


template <typename T>
inline void update_once(
    T &Ndisc,
    T &w,
    const int &sign,
    const T &Ndiscmax,
    const T &Ndiscmin,
    const T &ldisc,
    const T &A,
    const JARTv1bRPUDeviceMetaParameter<T>& par,
    RNG<T> *rng) {
  int n_time_steps = floor(par.pulse_length / par.base_time_step);

  double Ndisc_double = Ndisc;

  if (sign < 0) {

    for (int i = 0; i < n_time_steps; i++) {
      step_SET(Ndisc_double, Ndiscmax, Ndiscmin, ldisc, A, rng);
    }
    w = par.mapNdisc2Weight(Ndisc_double);


  } else {
    
    for (int i = 0; i < n_time_steps; i++) {
      step_RESET(
          pulse_voltage_RESET, base_time_step, Ndisc_double, g_RESET, h_RESET, j_0, k0, T0,
          Original_Ndiscmin, Ndiscmin, Nplug, a_ny0, dWa, Rth_positive_coefficient, RseriesTiOx, R0,
          V_series_coefficient, V_disk_coefficient, gamma_coefficient, lcell, ldisc, A, Ndiscmin_ctoc);
    }

    w = map_Ndisc_to_weight(
        read_voltage, Ndisc_double, current_min, weight_min_bound, current_to_weight_ratio, g_read,
        h_read, j_0, k0, Original_Ndiscmin);
  }

  Ndisc = MAX(MIN(Ndisc_double, Ndiscmax), Ndiscmin);
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
          par.read_voltage, par.pulse_voltage_SET, par.pulse_voltage_RESET, par.pulse_length,
          par.base_time_step, par.alpha_SET, par.beta_SET, par.c_SET, par.d_SET, par.f_SET,
          par.g_RESET, par.h_RESET, par.g_read, par.h_read, par.j_0, par.k0, par.T0,
          Ndiscmax[j], Ndiscmin[j], par.Nplug, par.a_ny0, par.dWa, par.Rth_negative_coefficient,
          par.Rth_positive_coefficient, par.RseriesTiOx, par.R0, par.V_series_coefficient,
          par.V_disk_coefficient, par.gamma_coefficient, par.lcell, ldisc[j], A[j], Ndisc[j], w[j],
          sign, par.current_min, par.current_to_weight_ratio, par.weight_to_current_ratio,
          par.w_min, Ndisc_max_bound[j], Ndisc_min_bound[j],
          par.Ndiscmax_std, Ndiscmax_ctoc_upper_bound[j], Ndiscmax_ctoc_lower_bound[j],
          par.Ndiscmin_std, Ndiscmin_ctoc_upper_bound[j], Ndiscmin_ctoc_lower_bound[j],
          par.ldisc_std, par.ldisc_std_slope, ldisc_ctoc_upper_bound[j], ldisc_ctoc_lower_bound[j],
          par.rdisc_std, par.rdisc_std_slope, A_ctoc_upper_bound[j], A_ctoc_lower_bound[j], rng););
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
          par.read_voltage, par.pulse_voltage_SET, par.pulse_voltage_RESET, par.pulse_length,
          par.base_time_step, par.alpha_SET, par.beta_SET, par.c_SET, par.d_SET, par.f_SET,
          par.g_RESET, par.h_RESET, par.g_read, par.h_read, par.j_0, par.k0, par.T0, par.Ndiscmin,
          Ndiscmax[j], Ndiscmin[j], par.Nplug, par.a_ny0, par.dWa, par.Rth_negative_coefficient,
          par.Rth_positive_coefficient, par.RseriesTiOx, par.R0, par.V_series_coefficient,
          par.V_disk_coefficient, par.gamma_coefficient, par.lcell, ldisc[j], A[j], Ndisc[j], w[j],
          sign, par.current_min, par.current_to_weight_ratio, par.weight_to_current_ratio,
          par.w_min, Ndisc_max_bound[j], Ndisc_min_bound[j],
          //  old implimentation
          //  par.Ndiscmax_std, par.Ndiscmax_ctoc_upper_bound, par.Ndiscmax_ctoc_lower_bound,
          //  par.Ndiscmin_std, par.Ndiscmin_ctoc_upper_bound, par.Ndiscmin_ctoc_lower_bound,
          //  par.ldisc_std, par.ldisc_std_slope, par.ldisc_ctoc_upper_bound,
          //  par.ldisc_ctoc_lower_bound, par.rdisc_std, par.rdisc_std_slope,
          //  par.rdisc_ctoc_upper_bound, par.rdisc_ctoc_lower_bound,
          par.Ndiscmax_std, Ndiscmax_ctoc_upper_bound[j], Ndiscmax_ctoc_lower_bound[j],
          par.Ndiscmin_std, Ndiscmin_ctoc_upper_bound[j], Ndiscmin_ctoc_lower_bound[j],
          par.ldisc_std, par.ldisc_std_slope, ldisc_ctoc_upper_bound[j], ldisc_ctoc_lower_bound[j],
          par.rdisc_std, par.rdisc_std_slope, A_ctoc_upper_bound[j], A_ctoc_lower_bound[j], rng););
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
    this->w_persistent_[0][i] = map_weight_to_Ndisc(
        w[i], par.current_min, par.w_min, par.weight_to_current_ratio, par.g_read, par.h_read,
        par.j_0, par.k0, par.Ndiscmin);
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
    this->w_persistent_[0][i] = map_weight_to_Ndisc(
        w[i], par.current_min, par.w_min, par.weight_to_current_ratio, par.g_read, par.h_read,
        par.j_0, par.k0, par.Ndiscmin);
  }
  return true;
}

template class JARTv1bRPUDevice<float>;
#ifdef RPU_USE_DOUBLE
template class JARTv1bRPUDevice<double>;
#endif

} // namespace RPU
