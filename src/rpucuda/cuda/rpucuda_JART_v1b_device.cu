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

#include "pwu_kernel_parameter.h"
#include "rpu_pulsed_meta_parameter.h"
#include "rpucuda_JART_v1b_device.h"
#include <memory>
#include <stdio.h>

namespace RPU {

template <typename T>
struct Voltages_holder
{
  T V_series;
  T V_disk;
  T V_plug;
  T V_Schottky;
};

template <typename T>
__device__ __forceinline__ T calculate_current_negative(
    double &Ndisc,
    const T &applied_voltage,
    const T &alpha0,
    const T &alpha1,
    const T &alpha2,
    const T &alpha3,
    const T &beta0,
    const T &beta1,
    const T &c0,
    const T &c1,
    const T &c2,
    const T &c3,
    const T &d0,
    const T &d1,
    const T &d2,
    const T &d3,
    const T &f0,
    const T &f1,
    const T &f2,
    const T &f3) {
  return -(((alpha1+alpha0)/(1+exp(-(applied_voltage+alpha2)/alpha3)))-alpha0)-((beta1*(1-exp(-applied_voltage)))
  -beta0*applied_voltage)/(pow(((1+pow(((c2*exp(-applied_voltage/c3)+c1*applied_voltage-c0)/(Ndisc/1e26)),(d2*exp(-applied_voltage/d3)+d1*applied_voltage-d0)))),(f0+((f1-f0)/(1+pow((-applied_voltage/f2),f3))))));
}

template <typename T>
__device__ __forceinline__ T calculate_current_positive(
    double &Ndisc,
    const T &applied_voltage,
    const T &g0,
    const T &g1,
    const T &h0,
    const T &h1,
    const T &h2,
    const T &h3,
    const T &j_0,
    const T &k0, 
    const T &Ndiscmin) {
  return (-g0*(exp(-g1*applied_voltage)-1))/(pow((1+(h0+h1*applied_voltage+h2*exp(-h3*applied_voltage))*pow((Ndisc/Ndiscmin),(-j_0))),(1/k0)));
}

template <typename T>
__device__ __forceinline__ T invert_positive_current(
    T &I_mem,
    const T &read_voltage,
    const T &g0,
    const T &g1,
    const T &h0,
    const T &h1,
    const T &h2,
    const T &h3,
    const T &j_0,
    const T &k0, 
    const T &Ndiscmin) {
  if (I_mem>0){
  return pow(((pow(((-g0*(exp(-g1*read_voltage)-1))/I_mem), k0)-1)/(h0+h1*read_voltage+h2*exp(-h3*read_voltage))),(1/-j_0))*Ndiscmin;
  }
  else{
    return 0;
  }
}

template <typename T>
__device__ __forceinline__ T calculate_current(
    double &Ndisc,
    const T &applied_voltage,
    const T &alpha0,
    const T &alpha1,
    const T &alpha2,
    const T &alpha3,
    const T &beta0,
    const T &beta1,
    const T &c0,
    const T &c1,
    const T &c2,
    const T &c3,
    const T &d0,
    const T &d1,
    const T &d2,
    const T &d3,
    const T &f0,
    const T &f1,
    const T &f2,
    const T &f3,
    const T &g0,
    const T &g1,
    const T &h0,
    const T &h1,
    const T &h2,
    const T &h3,
    const T &j_0,
    const T &k0, 
    const T &Ndiscmin) {
  if (applied_voltage < 0) {
    return calculate_current_negative(Ndisc, applied_voltage, alpha0, alpha1, alpha2, alpha3, beta0, beta1, c0, c1, c2, c3, d0, d1, d2, d3, f0, f1, f2, f3);
  } else {
    return calculate_current_positive(Ndisc, applied_voltage, g0, g1, h0, h1, h2, h3, j_0, k0, Ndiscmin);
  }
}

template <typename T>
__device__ __forceinline__ T calculate_T(
    const T &applied_voltage,
    T &I_mem,
    const T &T0,
    const T &Rth0,
    const T &Rtheff_scaling,
    Voltages_holder<T> &Voltages) {
  if (applied_voltage > 0) {
    return T0 + I_mem*(Voltages.V_disk+Voltages.V_plug+Voltages.V_Schottky)*Rth0*Rtheff_scaling;
  } else {
    return T0 + I_mem*(Voltages.V_disk+Voltages.V_plug+Voltages.V_Schottky)*Rth0;
  }
}

template <typename T>
__device__ __forceinline__ Voltages_holder<T> calculate_voltages(
    const T &applied_voltage,
    T &I_mem,
    const T &R0,
    const T &alphaline,
    const T &Rthline,
    const T &RseriesTiOx,
    const T &lcell,
    T &ldet,
    const int &zvo,
    const T &e,
    T &A,
    const T &Nplug,
    double &Ndisc,
    const T &un) {
  Voltages_holder<T> Voltages;
  // V_series
  Voltages.V_series = I_mem*(RseriesTiOx + (R0*(1+alphaline*R0*pow(I_mem,2)*Rthline)));
  // V_disk
  Voltages.V_disk =  I_mem*(ldet/(zvo*e*A*Ndisc*un));
  // V_plug
  Voltages.V_plug =  I_mem*((lcell-ldet)/(zvo*e*A*Nplug*un));
  // V_Schottky
  Voltages.V_Schottky =  applied_voltage-Voltages.V_series-Voltages.V_disk-Voltages.V_plug;
  return Voltages;
}

template <typename T>
__device__ __forceinline__ T calculate_F1(
    const T &applied_voltage,
    double &Ndisc,
    T &Ndiscmin,
    T &Ndiscmax) {
  if (applied_voltage > 0) {
    return 1-pow((Ndiscmin/Ndisc),10);
  } else {
    return 1-pow((Ndisc/Ndiscmax),10);
  }
}

template <typename T>
__device__ __forceinline__ T calculate_Eion(
    const T &applied_voltage,
    Voltages_holder<T> &Voltages,
    const T &lcell,
    T &ldet) {
  if (applied_voltage < 0) {
    return Voltages.V_disk/ldet;
  } else {
    return (Voltages.V_Schottky + Voltages.V_plug + Voltages.V_disk)/lcell;
  }
}

template <typename T>
__device__ __forceinline__ T calculate_dNdt(
    const T &applied_voltage,
    T &I_mem,
    double &Ndisc,
    const T &e,
    const T &kb,	
    const T &Arichardson,
    const T &mdiel,
    const T &h,
    const int &zvo,
    const T &eps_0,
    const T &T0,
    const T &eps,
    const T &epsphib,
    const T &phiBn0,
    const T &phin,
    const T &un,
    T &Ndiscmax,
    T &Ndiscmin,
    const T &Nplug,
    const T &a,
    const T &ny0,
    const T &dWa,
    const T &Rth0,
    const T &lcell,
    T &ldet,
    const T &Rtheff_scaling,
    const T &RseriesTiOx,
    const T &R0,
    const T &Rthline,
    const T &alphaline,
    T &A) {

  T c_v0 = (Nplug+Ndisc)/2;

  T F1 = calculate_F1(applied_voltage, Ndisc, Ndiscmin, Ndiscmax);

  Voltages_holder<T> Voltages = calculate_voltages(applied_voltage, I_mem, R0, alphaline, Rthline, RseriesTiOx, lcell, ldet, zvo, e, A, Nplug, Ndisc, un);

  T Eion = calculate_Eion(applied_voltage, Voltages, lcell, ldet);

  T gamma = zvo*a*Eion/(dWa*M_PI);

  T Treal = calculate_T(applied_voltage, I_mem, T0, Rth0, Rtheff_scaling, Voltages);
  
  // dWamin
  T dWa_f = dWa*(sqrt(1-pow(gamma,2))-(gamma*M_PI)/2+gamma*asin(gamma));
  // dWamax
  T dWa_r = dWa*(sqrt(1-pow(gamma,2))+(gamma*M_PI)/2+gamma*asin(gamma));
  T denominator = kb*Treal/e;
  T dNdt = -(c_v0*a*ny0*F1*(exp(-dWa_f/denominator)-exp(-dWa_r/denominator)))/ldet;
  return dNdt;
}

template <typename T>
__device__ __forceinline__ void step(
    const T &applied_voltage,
    const T &time_step,
    double &Ndisc,
    const T &alpha0,
    const T &alpha1,
    const T &alpha2,
    const T &alpha3,
    const T &beta0,
    const T &beta1,
    const T &c0,
    const T &c1,
    const T &c2,
    const T &c3,
    const T &d0,
    const T &d1,
    const T &d2,
    const T &d3,
    const T &f0,
    const T &f1,
    const T &f2,
    const T &f3,
    const T &g0,
    const T &g1,
    const T &h0,
    const T &h1,
    const T &h2,
    const T &h3,
    const T &j_0,
    const T &k0, 
    const T &e,
    const T &kb,	
    const T &Arichardson,
    const T &mdiel,
    const T &h,
    const int &zvo,
    const T &eps_0,
    const T &T0,
    const T &eps,
    const T &epsphib,
    const T &phiBn0,
    const T &phin,
    const T &un,
    const T &Original_Ndiscmin,
    T &Ndiscmax,
    T &Ndiscmin,
    const T &Nplug,
    const T &a,
    const T &ny0,
    const T &dWa,
    const T &Rth0,
    const T &lcell,
    T &ldet,
    const T &Rtheff_scaling,
    const T &RseriesTiOx,
    const T &R0,
    const T &Rthline,
    const T &alphaline,
    T &A,
    const T &Ndisc_min_bound,
    const T &Ndisc_max_bound) {


  T I_mem = calculate_current(Ndisc, applied_voltage, alpha0, alpha1, alpha2, alpha3, beta0, beta1, c0, c1, c2, c3, d0, d1, d2, d3, f0, f1, f2, f3, g0, g1, h0, h1, h2, h3, j_0, k0, Original_Ndiscmin);
  T dNdt = calculate_dNdt(applied_voltage, I_mem, Ndisc, e, kb, Arichardson, mdiel, h, zvo, eps_0, T0, eps, epsphib, phiBn0, phin, un, Ndiscmax, Ndiscmin, Nplug, a, ny0, dWa, Rth0, lcell, ldet, Rtheff_scaling, RseriesTiOx, R0, Rthline, alphaline, A);
  Ndisc = Ndisc + dNdt*time_step;

  if (Ndisc>Ndiscmax){
    Ndisc = Ndiscmax;
  }
  else if (Ndisc<Ndiscmin){
      Ndisc = Ndiscmin;
  }
}

template <typename T>
__device__ __forceinline__ T map_Ndisc_to_weight(
    const T &read_voltage,
    double &Ndisc,
    const T &current_min,
    const T &current_max,
    const T &weight_min_bound,
    const T &weight_max_bound,
    const T &g0,
    const T &g1,
    const T &h0,
    const T &h1,
    const T &h2,
    const T &h3,
    const T &j_0,
    const T &k0,
    const T &Original_Ndiscmin) {
  T read_current = calculate_current_positive(Ndisc, read_voltage, g0, g1, h0, h1, h2, h3, j_0, k0, Original_Ndiscmin);
  T weight = ((read_current-current_min)/(current_max-current_min))*(weight_max_bound-weight_min_bound)+weight_min_bound;
  return weight;
}

template <typename T>
__device__ __forceinline__ void apply_cycle_to_cycle_noise(
    T &Ndiscmax,
    T &Ndiscmin,
    T &ldet,
    T &A,
    const T &Ndiscmax_std,
    const T &Ndiscmin_std,
    const T &ldet_std,
    const T &rdet_std,
    curandState &local_state) {
  if (Ndiscmax_std > (T)0.0) {
    T stoch_value = curand_normal(&local_state);
    Ndiscmax = Ndiscmax + Ndiscmax_std * stoch_value;
  }
  if (Ndiscmin_std > (T)0.0) {
    T stoch_value = curand_normal(&local_state);
    Ndiscmin = Ndiscmin + Ndiscmin_std * stoch_value;
  }
  if (ldet_std > (T)0.0) {
    T stoch_value = curand_normal(&local_state);
    ldet = ldet + ldet_std * stoch_value;
  }
  if (rdet_std > (T)0.0) {
    T stoch_value = curand_normal(&local_state);
    T rdet = pow(A/M_PI, 1/2) + rdet_std * stoch_value;
    A = M_PI*pow(rdet,2);
  }
}

template <typename T>
__device__ __forceinline__ void update_once(
    const T &read_voltage,
    const T &pulse_voltage_SET,
    const T &pulse_voltage_RESET,
    const T &pulse_length,
    const T &base_time_step,
    const T &alpha0,
    const T &alpha1,
    const T &alpha2,
    const T &alpha3,
    const T &beta0,
    const T &beta1,
    const T &c0,
    const T &c1,
    const T &c2,
    const T &c3,
    const T &d0,
    const T &d1,
    const T &d2,
    const T &d3,
    const T &f0,
    const T &f1,
    const T &f2,
    const T &f3,
    const T &g0,
    const T &g1,
    const T &h0,
    const T &h1,
    const T &h2,
    const T &h3,
    const T &j_0,
    const T &k0, 
    const T &e,
    const T &kb,	
    const T &Arichardson,
    const T &mdiel,
    const T &h,
    const int &zvo,
    const T &eps_0,
    const T &T0,
    const T &eps,
    const T &epsphib,
    const T &phiBn0,
    const T &phin,
    const T &un,
    const T &Original_Ndiscmin,
    T &Ndiscmax,
    T &Ndiscmin,
    const T &Nplug,
    const T &a,
    const T &ny0,
    const T &dWa,
    const T &Rth0,
    const T &lcell,
    T &ldet,
    const T &Rtheff_scaling,
    const T &RseriesTiOx,
    const T &R0,
    const T &Rthline,
    const T &alphaline,
    T &A,
    T &Ndisc,
    T &w,
    uint32_t &negative,
    const T &current_min,
    const T &current_max,
    const T &weight_min_bound,
    const T &weight_max_bound,
    const T &Ndisc_min_bound,
    const T &Ndisc_max_bound, 
    const T &Ndiscmax_std,
    const T &Ndiscmin_std,
    const T &ldet_std,
    const T &rdet_std,
    curandState &local_state) {
  int pulse_counter = int (pulse_length/base_time_step);
  double Ndisc_double = Ndisc;
  // printf("w before update %.20f\n", w);
  // printf("Ndisc before update %.20e\n", Ndisc);
  // printf("Ndisc_double before update %.20e\n", Ndisc_double);

  if (negative > 0) {
    for (int i = 0; i < pulse_counter; i++) {
      step(pulse_voltage_SET, base_time_step, Ndisc_double, alpha0, alpha1, alpha2, alpha3, beta0, beta1, c0, c1, c2, c3, d0, d1, d2, d3, f0, f1, f2, f3, g0, g1, h0, h1, h2, h3, j_0, k0, e, kb, Arichardson, mdiel, h, zvo, eps_0, T0, eps, epsphib, phiBn0, phin, un, Original_Ndiscmin, Ndiscmax, Ndiscmin, Nplug, a, ny0, dWa, Rth0, lcell, ldet, Rtheff_scaling, RseriesTiOx, R0, Rthline, alphaline, A, Ndisc_min_bound, Ndisc_max_bound);
    }
    if (Ndisc_double>Ndisc_max_bound){
      Ndisc_double = Ndisc_max_bound;
    }
  }else{
    for (int i = 0; i < pulse_counter; i++) {
      step(pulse_voltage_RESET, base_time_step, Ndisc_double, alpha0, alpha1, alpha2, alpha3, beta0, beta1, c0, c1, c2, c3, d0, d1, d2, d3, f0, f1, f2, f3, g0, g1, h0, h1, h2, h3, j_0, k0, e, kb, Arichardson, mdiel, h, zvo, eps_0, T0, eps, epsphib, phiBn0, phin, un, Original_Ndiscmin, Ndiscmax, Ndiscmin, Nplug, a, ny0, dWa, Rth0, lcell, ldet, Rtheff_scaling, RseriesTiOx, R0, Rthline, alphaline, A, Ndisc_min_bound, Ndisc_max_bound);
    }
    if (Ndisc_double<Ndisc_min_bound){
      Ndisc_double = Ndisc_min_bound;
    }
  } 

  w = map_Ndisc_to_weight(read_voltage, Ndisc_double, current_min, current_max, weight_min_bound, weight_max_bound, g0, g1, h0, h1, h2, h3, j_0, k0, Original_Ndiscmin);
  Ndisc = Ndisc_double;
  apply_cycle_to_cycle_noise(Ndiscmax, Ndiscmin, ldet, A, Ndiscmax_std, Ndiscmin_std, ldet_std, rdet_std, local_state);
  // printf("w after update %.20f\n", w);
  // printf("Ndisc after update %.20e\n", Ndisc);
  // printf("Ndisc_double after update %.20e\n", Ndisc_double);
}

template <typename T> struct UpdateFunctorJARTv1b {

  __device__ __forceinline__ void operator()(
      T &apparent_weight,
      uint32_t n,
      uint32_t negative,
      float4 &par_4,
      float2 &par_2,
      // const float4 par_4,
      // const float2 par_2,
      T &persistent_weight,
      const T *global_pars,
      const int global_params_count,
      T noise_std_dw,
      curandState &local_state) {

    UNUSED(global_params_count); // fixed

    const T read_voltage        = global_pars[0];
    const T pulse_voltage_SET   = global_pars[1];
    const T pulse_voltage_RESET = global_pars[2];
    const T pulse_length        = global_pars[3];
    const T base_time_step      = global_pars[4];
    const T alpha0              = global_pars[5];
    const T alpha1              = global_pars[6];
    const T alpha2              = global_pars[7];
    const T alpha3              = global_pars[8];
    const T beta0               = global_pars[9];
    const T beta1               = global_pars[10];
    const T c0                  = global_pars[11];
    const T c1                  = global_pars[12];
    const T c2                  = global_pars[13];
    const T c3                  = global_pars[14];
    const T d0                  = global_pars[15];
    const T d1                  = global_pars[16];
    const T d2                  = global_pars[17];
    const T d3                  = global_pars[18];
    const T f0                  = global_pars[19];
    const T f1                  = global_pars[20];
    const T f2                  = global_pars[21];
    const T f3                  = global_pars[22];
    const T g0                  = global_pars[23];
    const T g1                  = global_pars[24];
    const T h0                  = global_pars[25];
    const T h1                  = global_pars[26];
    const T h2                  = global_pars[27];
    const T h3                  = global_pars[28];
    const T j_0                 = global_pars[29];
    const T k0                  = global_pars[30];
    const T e                   = global_pars[31];
    const T kb                  = global_pars[32];
    const T Arichardson         = global_pars[33];
    const T mdiel               = global_pars[34];
    const T h                   = global_pars[35];
    const int zvo               = global_pars[36];
    const T eps_0               = global_pars[37];
    const T T0                  = global_pars[38];
    const T eps                 = global_pars[39];
    const T epsphib             = global_pars[40];
    const T phiBn0              = global_pars[41];
    const T phin                = global_pars[42];
    const T un                  = global_pars[43];
    const T Ndiscmin            = global_pars[44];
    const T Nplug               = global_pars[45];
    const T a                   = global_pars[46];
    const T ny0                 = global_pars[47];
    const T dWa                 = global_pars[48];
    const T Rth0                = global_pars[49];
    const T lcell               = global_pars[50];
    const T Rtheff_scaling      = global_pars[51];
    const T RseriesTiOx         = global_pars[52];
    const T R0                  = global_pars[53];
    const T Rthline             = global_pars[54];
    const T alphaline           = global_pars[55];
    const T current_min         = global_pars[56];
    const T current_max         = global_pars[57];
    const T Ndisc_min_bound     = global_pars[58];
    const T Ndisc_max_bound     = global_pars[59];
    const T Ndiscmax_std        = global_pars[60];
    const T Ndiscmin_std        = global_pars[61];
    const T ldet_std            = global_pars[62];
    const T rdet_std            = global_pars[63];
    
    const T &weight_min_bound = par_4.x;                          // [0]
    T &device_specific_Ndiscmin_cuda = par_4.y; // [1]
    const T &weight_max_bound = par_4.z;                          // [2]
    T &device_specific_Ndiscmax_cuda = par_4.w; // [3]

    T &device_specific_ldet_cuda = par_2.x; // [0]
    T &device_specific_A_cuda = par_2.y; // [1]

    T &w = apparent_weight;
    T &Ndisc = persistent_weight;
    printf("w before update %.20f\n", apparent_weight);
    printf("Ndisc before update %.20e\n", persistent_weight);

    // n is larger 0 in any case
    if (n == 1) {
      update_once(read_voltage, pulse_voltage_SET, pulse_voltage_RESET, pulse_length, base_time_step,
                                  alpha0, alpha1, alpha2, alpha3, beta0, beta1,
                                  c0, c1, c2, c3, d0, d1, d2, d3,
                                  f0, f1, f2, f3, g0, g1, h0, h1, h2, h3, j_0, k0,
                                  e, kb, Arichardson, mdiel, h, zvo, eps_0,
                                  T0, eps, epsphib, phiBn0, phin, un, Ndiscmin,
                                  device_specific_Ndiscmax_cuda, device_specific_Ndiscmin_cuda,
                                  Nplug, a, ny0, dWa, Rth0, lcell,
                                  device_specific_ldet_cuda,
                                  Rtheff_scaling, RseriesTiOx, R0, Rthline, alphaline,
                                  device_specific_A_cuda, Ndisc, w, negative,
                                  current_min, current_max,
                                  weight_min_bound, weight_max_bound,
                                  Ndisc_min_bound, Ndisc_max_bound,
                                  Ndiscmax_std, Ndiscmin_std, ldet_std, rdet_std,
                                  local_state);
    } else {
      for (int i_updates = 0; i_updates < n; i_updates++) {
        update_once(read_voltage, pulse_voltage_SET, pulse_voltage_RESET, pulse_length, base_time_step,
                                   alpha0, alpha1, alpha2, alpha3, beta0, beta1,
                                   c0, c1, c2, c3, d0, d1, d2, d3,
                                   f0, f1, f2, f3, g0, g1, h0, h1, h2, h3, j_0, k0,
                                   e, kb, Arichardson, mdiel, h, zvo, eps_0,
                                   T0, eps, epsphib, phiBn0, phin, un, Ndiscmin,
                                   device_specific_Ndiscmax_cuda, device_specific_Ndiscmin_cuda,
                                   Nplug, a, ny0, dWa, Rth0, lcell,
                                   device_specific_ldet_cuda,
                                   Rtheff_scaling, RseriesTiOx, R0, Rthline, alphaline,
                                   device_specific_A_cuda, Ndisc, w, negative,
                                   current_min, current_max,
                                   weight_min_bound, weight_max_bound,
                                   Ndisc_min_bound, Ndisc_max_bound,
                                   Ndiscmax_std, Ndiscmin_std, ldet_std, rdet_std,
                                   local_state);
      }
    }
    printf("w after update %.20f\n", apparent_weight);
    printf("Ndisc after update %.20e\n", persistent_weight);
  }
};

#define ARGS                                                                                       \
  (this->context_, this->x_size_, this->d_size_, m_batch, nK32, use_bo64, out_trans, up,           \
   par.getName())

template <typename T>
pwukpvec_t<T> JARTv1bRPUDeviceCuda<T>::getUpdateKernels(
    int m_batch, int nK32, int use_bo64, bool out_trans, const PulsedUpdateMetaParameter<T> &up) {

  pwukpvec_t<T> v;
  const auto &par = getPar();
  v.push_back(
      RPU::make_unique<PWUKernelParameterSingleFunctor<T, UpdateFunctorJARTv1b<T>, DEVICE_PARAMETER_COUNT>>
          ARGS);
  v.push_back(
      RPU::make_unique<PWUKernelParameterBatchFunctor<T, UpdateFunctorJARTv1b<T>, DEVICE_PARAMETER_COUNT>>
          ARGS);
  v.push_back(
      RPU::make_unique<PWUKernelParameterBatchSharedFunctor<T, UpdateFunctorJARTv1b<T>, DEVICE_PARAMETER_COUNT>>
          ARGS);

  return v;
}

#undef ARGS

template <typename T>
__global__ void kernelMapWeightToNdisc(
  T *weights, 
  T *Ndiscs, 
  int size, 
  T read_voltage,
  T current_min, 
  T current_max, 
  T weight_min_bound,
  T weight_max_bound,
  T g0,
  T g1,
  T h0,
  T h1,
  T h2,
  T h3,
  T j_0,
  T k0,
  T Ndiscmin) {

  RPU_CUDA_1D_KERNEL_LOOP(idx, size) {
    T current = ((weights[idx]-weight_min_bound)/(weight_max_bound-weight_min_bound))*(current_max-current_min)+current_min;
    Ndiscs[idx] = pow(((pow(((-g0*(exp(-g1*read_voltage)-1))/current), k0)-1)/(h0+h1*read_voltage+h2*exp(-h3*read_voltage))),(1/-j_0))*Ndiscmin;
    }
}

template <typename T>
void map_weight_to_Ndisc(
  const CudaContext *context,
  T *w,
  T *Ndiscs,
  const int size,
  const T read_voltage,
  const T current_min, 
  const T current_max,
  const T weight_min_bound,
  const T weight_max_bound,
  const T g0,
  const T g1,
  const T h0,
  const T h1,
  const T h2,
  const T h3,
  const T j_0,
  const T k0,
  const T Ndiscmin) {

  int nthreads = context->getNThreads();
  int nblocks = context->getNBlocks(size, nthreads);
  kernelMapWeightToNdisc<T><<<nblocks, nthreads, 0, context->getStream()>>>(w, Ndiscs, size, read_voltage, current_min, current_max, weight_min_bound, weight_max_bound, g0, g1, h0, h1, h2, h3, j_0, k0, Ndiscmin);
}
template void map_weight_to_Ndisc<float>(const CudaContext *, float *, float *, const int, const float, const float, const float, const float, const float, const float, const float, const float, const float, const float, const float, const float, const float, const float);
#ifdef RPU_USE_DOUBLE
template void map_weight_to_Ndisc<double>(const CudaContext *, double *, double *, const int, const double, const double, const double, const double, const double, const double, const double, const double, const double, const double, const double, const double, const double, const double);
#endif

template <typename T>
void JARTv1bRPUDeviceCuda<T>::applyWeightUpdate(T *weights, T *dw_and_current_weight_out) {

  // if (getPar().usesPersistentWeight()) {
  //   RPU_FATAL("ApplyWeightUpdate is not supported with write_noise_std>0!");
  // }
  RPU::math::elemaddcopysat<T>(
      this->context_, weights, dw_and_current_weight_out, this->size_,
      this->dev_4params_->getDataConst());
  
  const auto &par = getPar();
  T *Ndisc = get1ParamsData();

  map_weight_to_Ndisc<T>(
      this->context_, weights, Ndisc, this->size_,
      par.read_voltage, par.current_min, par.current_max, par.w_min, par.w_max,
      par.g0, par.g1, par.h0, par.h1, par.h2, par.h3, par.j_0, par.k0, par.Ndiscmin);

}

template <typename T>
void JARTv1bRPUDeviceCuda<T>::decayWeights(T *weights, T alpha, bool bias_no_decay) {

  RPU::math::elemscalealpha<T>(
      this->context_, weights, bias_no_decay ? MAX(this->size_ - this->d_size_, 0) : this->size_,
      this->dev_decay_scale_->getData(), this->dev_4params_->getData(), alpha,
      this->dev_reset_bias_ != nullptr ? this->dev_reset_bias_->getData() : nullptr);
  
  const auto &par = getPar();
  T *Ndisc = get1ParamsData();

  map_weight_to_Ndisc<T>(
      this->context_, weights, Ndisc, this->size_,
      par.read_voltage, par.current_min, par.current_max, par.w_min, par.w_max,
      par.g0, par.g1, par.h0, par.h1, par.h2, par.h3, par.j_0, par.k0, par.Ndiscmin);

}

template <typename T> void JARTv1bRPUDeviceCuda<T>::decayWeights(T *weights, bool bias_no_decay) {

  const auto &par = getPar();

  RPU::math::elemscale<T>(
      this->context_, weights, bias_no_decay ? MAX(this->size_ - this->d_size_, 0) : this->size_,
      this->dev_decay_scale_->getData(), this->dev_4params_->getData(),
      this->dev_reset_bias_ != nullptr ? this->dev_reset_bias_->getData() : nullptr);
  
  T *Ndisc = get1ParamsData();

  map_weight_to_Ndisc<T>(
      this->context_, weights, Ndisc, this->size_,
      par.read_voltage, par.current_min, par.current_max, par.w_min, par.w_max,
      par.g0, par.g1, par.h0, par.h1, par.h2, par.h3, par.j_0, par.k0, par.Ndiscmin);

}

template <typename T>
void JARTv1bRPUDeviceCuda<T>::driftWeights(T *weights, T time_since_last_call) {

  PulsedRPUDeviceCudaBase<T>::driftWeights(weights, time_since_last_call);
  this->wdrifter_cuda_->saturate(weights, this->dev_4params_->getData());
  
  const auto &par = getPar();
  T *Ndisc = get1ParamsData();

  map_weight_to_Ndisc<T>(
      this->context_, weights, Ndisc, this->size_,
      par.read_voltage, par.current_min, par.current_max, par.w_min, par.w_max,
      par.g0, par.g1, par.h0, par.h1, par.h2, par.h3, par.j_0, par.k0, par.Ndiscmin);

}

template <typename T> void JARTv1bRPUDeviceCuda<T>::diffuseWeights(T *weights) {

  if (this->dev_diffusion_rate_ == nullptr) {
    return; // no diffusion
  }

  if (this->dev_diffusion_nrnd_ == nullptr) {
    this->initDiffusionRnd();
    this->rnd_context_->randNormal(
        this->dev_diffusion_nrnd_->getData(), this->dev_diffusion_nrnd_->getSize());
  }
  this->rnd_context_->synchronize();

  RPU::math::elemasb02<T>(
      this->context_, weights, this->size_, this->dev_diffusion_nrnd_->getData(),
      this->dev_diffusion_rate_->getData(), this->dev_4params_->getData());

  this->rnd_context_->recordWaitEvent(this->context_->getStream());
  this->rnd_context_->randNormal(
      this->dev_diffusion_nrnd_->getData(), this->dev_diffusion_nrnd_->getSize());

  // Note: write noise will use the same rand to save memory. If
  // diffusion + writenoise is often needed one might want to add an
  // extra variable for the random numbers
  
  const auto &par = getPar();
  T *Ndisc = get1ParamsData();

  map_weight_to_Ndisc<T>(
      this->context_, weights, Ndisc, this->size_,
      par.read_voltage, par.current_min, par.current_max, par.w_min, par.w_max,
      par.g0, par.g1, par.h0, par.h1, par.h2, par.h3, par.j_0, par.k0, par.Ndiscmin);

}

template <typename T> void JARTv1bRPUDeviceCuda<T>::clipWeights(T *weights, T clip) {

  RPU::math::elemsat<T>(this->context_, weights, this->size_, this->dev_4params_->getData());
  if (clip >= 0) {
    RPU::math::aclip<T>(this->context_, weights, this->size_, clip);
  }
  
  const auto &par = getPar();
  T *Ndisc = get1ParamsData();

  map_weight_to_Ndisc<T>(
      this->context_, weights, Ndisc, this->size_,
      par.read_voltage, par.current_min, par.current_max, par.w_min, par.w_max,
      par.g0, par.g1, par.h0, par.h1, par.h2, par.h3, par.j_0, par.k0, par.Ndiscmin);

}

template <typename T>
void JARTv1bRPUDeviceCuda<T>::resetAt(T *dev_weights, const char *dev_non_zero_msk) {

  const auto &par = getPar();

  // if (par.usesPersistentWeight()) {
  //   RPU_FATAL("ResetAt is not supported with write_noise_std>0!");
  // }

  RPU::math::elemresetsatmsk<T>(
      this->context_, dev_weights, this->size_, dev_non_zero_msk,
      this->dev_reset_bias_ == nullptr ? nullptr : this->dev_reset_bias_->getDataConst(), par.reset_std,
      this->dev_4params_->getData());
  
  T *Ndisc = get1ParamsData();

  map_weight_to_Ndisc<T>(
      this->context_, dev_weights, Ndisc, this->size_,
      par.read_voltage, par.current_min, par.current_max, par.w_min, par.w_max,
      par.g0, par.g1, par.h0, par.h1, par.h2, par.h3, par.j_0, par.k0, par.Ndiscmin);

}

template <typename T>
void JARTv1bRPUDeviceCuda<T>::resetCols(T *weights, int start_col, int n_cols, T reset_prob) {
  // col-major in CUDA.

  if (this->dev_reset_bias_ == nullptr) {
    return; // no reset
  }

  // if (getPar().usesPersistentWeight()) {
  //   RPU_FATAL("ResetCols is not supported with write_noise_std>0!");
  // }

  if (this->dev_reset_nrnd_ == nullptr) {
    PulsedRPUDeviceCuda<T>::initResetRnd();
  }
  int n = n_cols * this->d_size_;
  int offset = start_col * this->d_size_;
  this->rnd_context_->randNormal(
      this->dev_reset_nrnd_->getData(), n_cols * this->d_size_, 0.0, getPar().reset_std);
  if (reset_prob < 1) {
    this->rnd_context_->randUniform(this->dev_reset_flag_->getData(), n_cols * this->d_size_);
  }
  this->context_->recordWaitEvent(this->rnd_context_->getStream());

  if (n >= this->size_) {
    // reset whole matrix
    RPU::math::elemresetsat<T>(
        this->context_, weights, this->size_, this->dev_reset_bias_->getDataConst(),
        this->dev_reset_nrnd_->getDataConst(), this->dev_reset_flag_->getDataConst(), reset_prob,
        this->dev_4params_->getData());

  } else if (offset + n <= this->size_) {
    // one pass enough
    RPU::math::elemresetsat<T>(
        this->context_, weights + offset, n, this->dev_reset_bias_->getDataConst() + offset,
        this->dev_reset_nrnd_->getDataConst(), this->dev_reset_flag_->getDataConst(), reset_prob,
        this->dev_4params_->getData() + 4 * offset);
  } else {
    // two passes
    int m = this->size_ - offset;

    RPU::math::elemresetsat<T>(
        this->context_, weights + offset, m, this->dev_reset_bias_->getDataConst() + offset,
        this->dev_reset_nrnd_->getDataConst(), this->dev_reset_flag_->getDataConst(), reset_prob,
        this->dev_4params_->getData() + 4 * offset);

    RPU::math::elemresetsat<T>(
        this->context_, weights, n - m, this->dev_reset_bias_->getDataConst(),
        this->dev_reset_nrnd_->getDataConst() + m, this->dev_reset_flag_->getDataConst() + m, reset_prob,
        this->dev_4params_->getData());
  }
  
  const auto &par = getPar();
  T *Ndisc = get1ParamsData();

  map_weight_to_Ndisc<T>(
      this->context_, weights, Ndisc, this->size_,
      par.read_voltage, par.current_min, par.current_max, par.w_min, par.w_max,
      par.g0, par.g1, par.h0, par.h1, par.h2, par.h3, par.j_0, par.k0, par.Ndiscmin);

}

template class JARTv1bRPUDeviceCuda<float>;
#ifdef RPU_USE_DOUBLE
template class JARTv1bRPUDeviceCuda<double>;
#endif

} // namespace RPU
