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
__device__ __forceinline__ T map_Ndisc_to_weight(
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
  T read_current = g_read/(pow((1+h_read*pow((Ndisc/Original_Ndiscmin),-j_0)),1/k0));
  T weight = (read_current-current_min)*current_to_weight_ratio+weight_min_bound;
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

template <typename T> struct UpdateFunctorJARTv1b {

  __device__ __forceinline__ void operator()(
      T &apparent_weight,
      uint32_t n,
      uint32_t negative,
      float4 &par_4,
      float2 &par_2,
      T &persistent_weight,
      const T *global_pars,
      const int global_params_count,
      T noise_std_dw,
      curandState &local_state) {

    UNUSED(global_params_count); // fixed

    const T read_voltage            = global_pars[0];
    const T pulse_voltage_SET       = global_pars[1];
    const T pulse_voltage_RESET     = global_pars[2];
    const T pulse_length            = global_pars[3];
    const T base_time_step          = global_pars[4];
    const T alpha_SET               = global_pars[5];
    const T beta_SET                = global_pars[6];
    const T c_SET                   = global_pars[7];
    const T d_SET                   = global_pars[8];
    const T f_SET                   = global_pars[9];
    const T g_RESET                 = global_pars[10];
    const T h_RESET                 = global_pars[11];
    const T g_read                  = global_pars[12];
    const T h_read                  = global_pars[13];
    const T j_0                     = global_pars[14];
    const T k0                      = global_pars[15];
    const T T0                      = global_pars[16];
    const T Ndiscmin                = global_pars[17];
    const T Nplug                   = global_pars[18];
    const T a_ny0                   = global_pars[19];
    const T dWa                     = global_pars[20];
    const T Rth_negative            = global_pars[21];
    const T Rth_positive            = global_pars[22];
    const T RseriesTiOx             = global_pars[23];
    const T R0                      = global_pars[24];
    const T V_series_coefficient    = global_pars[25];
    const T V_disk_coefficient      = global_pars[26];
    const T gamma_coefficient       = global_pars[27];
    const T lcell                   = global_pars[28];
    const T current_min             = global_pars[29];
    const T current_to_weight_ratio = global_pars[30];
    const T weight_to_current_ratio = global_pars[31];
    const T w_min                   = global_pars[32];
    const T Ndisc_min_bound         = global_pars[33];
    const T Ndisc_max_bound         = global_pars[34];
    const T Ndiscmax_std            = global_pars[35];
    const T Ndiscmin_std            = global_pars[36];
    const T ldet_std                = global_pars[37];
    const T rdet_std                = global_pars[38];
    
    // const T &weight_min_bound = par_4.x;                          // [0]
    T &device_specific_Ndiscmin_cuda = par_4.y; // [1]
    // const T &weight_max_bound = par_4.z;                          // [2]
    T &device_specific_Ndiscmax_cuda = par_4.w; // [3]

    T &device_specific_ldet_cuda = par_2.x; // [0]
    T &device_specific_A_cuda = par_2.y; // [1]

    T &w = apparent_weight;
    T &Ndisc = persistent_weight;

    uint32_t pulse_counter = uint32_t (pulse_length/base_time_step);
    // n is larger 0 in any case
    pulse_counter = pulse_counter *n;
    double Ndisc_double = Ndisc;
    T max_bound = fmin(Ndisc_max_bound, device_specific_Ndiscmax_cuda);
    T min_bound = fmax(Ndisc_min_bound, device_specific_Ndiscmin_cuda);

if (negative > 0) {
  if (Ndisc_double >= max_bound)
  {
    Ndisc_double = max_bound;
  }
  else
  {
    for (int i_updates = 0; i_updates < pulse_counter; i_updates++) {
      T I_mem = -alpha_SET-beta_SET/(pow((1+pow((c_SET/Ndisc),d_SET)),f_SET));

      T V_disk = I_mem*(device_specific_ldet_cuda/(V_disk_coefficient*device_specific_A_cuda*Ndisc_double));

      // T gamma = gamma_coefficient*Eion;
      T gamma = gamma_coefficient*V_disk/device_specific_ldet_cuda;
      
      // V - V_series = V_disk+V_plug+V_Schottky
      T V_other_than_series = pulse_voltage_SET - (I_mem*(RseriesTiOx + R0 + V_series_coefficient*I_mem*I_mem));

      T Treal = T0 + I_mem*V_other_than_series*Rth_negative;
      // // dWamin
      // T dWa_f = dWa*(sqrt(1-pow(gamma,2.0))-(gamma*M_PI)/2+gamma*asin(gamma));
      // // dWamax
      // T dWa_r = dWa*(sqrt(1-pow(gamma,2.0))+(gamma*M_PI)/2+gamma*asin(gamma));

      T dWa_mean = dWa*(sqrt(1-pow(gamma,2.0))+gamma*asin(gamma));
      T dWa_difference = dWa*((gamma*M_PI)/2.0);
      // dWamin = dWa_f = dWa_mean - dWa_difference
      // dWamax = dWa_r = dWa_mean + dWa_difference

      T denominator = PHYSICAL_PARAMETER_kb_over_e*Treal;

      T c_v0 = (Nplug+Ndisc_double)/2.0;
      T F1 = 1-pow((Ndisc_double/device_specific_Ndiscmax_cuda),10.0);
      T dNdt = -(c_v0*a_ny0*F1*(exp(-(dWa_mean - dWa_difference)/denominator)-exp(-(dWa_mean + dWa_difference)/denominator)))/device_specific_ldet_cuda;

      Ndisc_double = Ndisc_double + dNdt*base_time_step;
    }
    if (Ndisc_double >= max_bound)
    {
      Ndisc_double = max_bound;
    }
  }
  
  
}else{
  if (Ndisc_double <= min_bound)
  {
    Ndisc_double = min_bound;
  }
  else
  {
    for (int i_updates = 0; i_updates < pulse_counter; i_updates++) {
      T I_mem = g_RESET/(pow((1+h_RESET*pow((Ndisc/Ndiscmin),-j_0)),1/k0));
      
      // V - V_series = V_disk+V_plug+V_Schottky
      T V_other_than_series = pulse_voltage_RESET - (I_mem*(RseriesTiOx + R0 + V_series_coefficient*I_mem*I_mem));

      // T gamma = gamma_coefficient*Eion;
      T gamma = gamma_coefficient*V_other_than_series/lcell;

      T Treal = T0 + I_mem*V_other_than_series*Rth_positive;
      // // dWamin
      // T dWa_f = dWa*(sqrt(1-pow(gamma,2.0))-(gamma*M_PI)/2+gamma*asin(gamma));
      // // dWamax
      // T dWa_r = dWa*(sqrt(1-pow(gamma,2.0))+(gamma*M_PI)/2+gamma*asin(gamma));

      T dWa_mean = dWa*(sqrt(1-pow(gamma,2.0))+gamma*asin(gamma));
      T dWa_difference = dWa*((gamma*M_PI)/2.0);
      // dWamin = dWa_f = dWa_mean - dWa_difference
      // dWamax = dWa_r = dWa_mean + dWa_difference

      T denominator = PHYSICAL_PARAMETER_kb_over_e*Treal;

      T c_v0 = (Nplug+Ndisc_double)/2.0;
      T F1 = 1-pow((device_specific_Ndiscmin_cuda/Ndisc_double),10.0);
      T dNdt = -(c_v0*a_ny0*F1*(exp(-(dWa_mean - dWa_difference)/denominator)-exp(-(dWa_mean + dWa_difference)/denominator)))/device_specific_ldet_cuda;

      Ndisc_double = Ndisc_double + dNdt*base_time_step;
    }
    if (Ndisc_double <= min_bound)
    {
      Ndisc_double = min_bound;
    }
  }
}

    w = map_Ndisc_to_weight(read_voltage, Ndisc_double, current_min, w_min, current_to_weight_ratio, g_read, h_read, j_0, k0, Ndiscmin);
    Ndisc = Ndisc_double;
    printf("w after update %.20f\n", apparent_weight);
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
