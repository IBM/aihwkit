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

#include "pwu_kernel_parameter.h"
#include "rpu_pulsed_meta_parameter.h"
#include "rpucuda_JART_v1b_device.h"
#include <memory>
#include <stdio.h>

namespace RPU {

template <typename T>
__device__ __forceinline__ T map_Ndisc_to_weight(
    const double &Ndisc,
    const T &current_min,
    const T &weight_min_bound,
    const T &current_to_weight_ratio,
    const T &g_read,
    const T &h_read,
    const T &j_0,
    const T &k0,
    const T &Original_Ndiscmin) {
  T read_current =
      g_read / (__powf((1 + h_read * __powf((Ndisc / Original_Ndiscmin), -j_0)), 1 / k0));
  T weight = (read_current - current_min) * current_to_weight_ratio + weight_min_bound;
  return weight;
}

template <typename T>
__device__ __forceinline__ void apply_cycle_to_cycle_noise(
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
    curandState &local_state,
    const T &Ndiscmax_ctoc_upper_bound,
    const T &Ndiscmax_ctoc_lower_bound,
    const T &Ndiscmin_ctoc_upper_bound,
    const T &Ndiscmin_ctoc_lower_bound,
    const T &ldisc_ctoc_upper_bound,
    const T &ldisc_ctoc_lower_bound,
    const T &rdisc_ctoc_upper_bound,
    const T &rdisc_ctoc_lower_bound) {
  if (Ndiscmax_std > (T)0.0) {
    T stoch_value = 2 * curand_uniform(&local_state) - 1;
    Ndiscmax = Ndiscmax * (1 + Ndiscmax_std * stoch_value);
    if (Ndiscmax_ctoc_upper_bound > (T)0.0) {
      Ndiscmax = MIN(Ndiscmax, Ndiscmax_ctoc_upper_bound);
    }
    Ndiscmax = MAX(Ndiscmax, Ndiscmax_ctoc_lower_bound);
  }
  if (Ndiscmin_std > (T)0.0) {
    T stoch_value = 2 * curand_uniform(&local_state) - 1;
    Ndiscmin = Ndiscmin * (1 + Ndiscmin_std * stoch_value);
    if (Ndiscmin_ctoc_upper_bound > (T)0.0) {
      Ndiscmin = MIN(Ndiscmin, Ndiscmin_ctoc_upper_bound);
    }
    Ndiscmin = MAX(Ndiscmin, Ndiscmin_ctoc_lower_bound);
  }
  if ((ldisc_std > (T)0.0) || (ldisc_std_slope > (T)0.0)) {
    T stoch_value_1 = 2 * curand_uniform(&local_state) - 1;
    T stoch_value_2 = 2 * curand_uniform(&local_state) - 1;
    ldisc = ldisc * (1 + ldisc_std * stoch_value_1 + ratio * ldisc_std_slope * stoch_value_2);
    if (ldisc_ctoc_upper_bound > (T)0.0) {
      ldisc = MIN(ldisc, ldisc_ctoc_upper_bound);
    }
    ldisc = MAX(ldisc, ldisc_ctoc_lower_bound);
  }
  if ((rdisc_std > (T)0.0) || (rdisc_std_slope > (T)0.0)) {
    T stoch_value_1 = 2 * curand_uniform(&local_state) - 1;
    T stoch_value_2 = 2 * curand_uniform(&local_state) - 1;
    T rdisc =
        sqrtf(A / M_PI) * (1 + rdisc_std * stoch_value_1 + ratio * rdisc_std_slope * stoch_value_2);
    if (rdisc_ctoc_upper_bound > (T)0.0) {
      rdisc = MIN(rdisc, rdisc_ctoc_upper_bound);
    }
    rdisc = MAX(rdisc, rdisc_ctoc_lower_bound);
    A = M_PI * rdisc * rdisc;
  }
}

template <typename T> struct UpdateFunctorJARTv1b {

  __device__ __forceinline__ void operator()(
      T &apparent_weight,
      uint32_t n,
      uint32_t negative,
      const param4_t par_4,
      const param2_t par_2,
      T &persistent_weight,
      const T *global_pars,
      const int global_params_count,
      T noise_std_dw,
      curandState &local_state) {

    UNUSED(global_params_count); // fixed

    const T pulse_voltage_SET = global_pars[0];
    const T pulse_voltage_RESET = global_pars[1];
    const T pulse_length = global_pars[2];
    const T base_time_step = global_pars[3];
    const T alpha_SET = global_pars[4];
    const T beta_SET = global_pars[5];
    const T c_SET = global_pars[6];
    const T d_SET = global_pars[7];
    const T f_SET = global_pars[8];
    const T g_RESET = global_pars[9];
    const T h_RESET = global_pars[10];
    const T g_read = global_pars[11];
    const T h_read = global_pars[12];
    const T j_0 = global_pars[13];
    const T k0 = global_pars[14];
    const T T0 = global_pars[15];
    const T Original_Ndiscmin = global_pars[16];
    const T Nplug = global_pars[17];
    const T a_ny0 = global_pars[18];
    const T dWa = global_pars[19];
    const T Rth_negative_coefficient = global_pars[20];
    const T Rth_positive_coefficient = global_pars[21];
    const T RseriesTiOx = global_pars[22];
    const T R0 = global_pars[23];
    const T V_series_coefficient = global_pars[24];
    const T V_disk_coefficient = global_pars[25];
    const T gamma_coefficient = global_pars[26];
    const T lcell = global_pars[27];
    const T current_min = global_pars[28];
    const T current_to_weight_ratio = global_pars[29];
    const T weight_to_current_ratio = global_pars[30];
    const T w_min = global_pars[31];
    const T Ndisc_max_bound = global_pars[32];
    const T Ndisc_min_bound = global_pars[33];
    const T Ndiscmax_std = global_pars[34];
    const T Ndiscmax_ctoc_upper_bound = global_pars[35];
    const T Ndiscmax_ctoc_lower_bound = global_pars[36];
    const T Ndiscmin_std = global_pars[37];
    const T Ndiscmin_ctoc_upper_bound = global_pars[38];
    const T Ndiscmin_ctoc_lower_bound = global_pars[39];
    const T ldisc_std = global_pars[40];
    const T ldisc_std_slope = global_pars[41];
    const T ldisc_ctoc_upper_bound = global_pars[42];
    const T ldisc_ctoc_lower_bound = global_pars[43];
    const T rdisc_std = global_pars[44];
    const T rdisc_std_slope = global_pars[45];
    const T rdisc_ctoc_upper_bound = global_pars[46];
    const T rdisc_ctoc_lower_bound = global_pars[47];
    T half_pi = M_PI / 2.0;
    T R = RseriesTiOx + R0;

    T device_specific_Ndisc_min_bound_cuda = par_4.x; // [0]
    T device_specific_Ndisc_max_bound_cuda = par_4.z; // [2]
    T device_specific_Ndiscmin_cuda = par_4.y;        // [1]
    T device_specific_Ndiscmax_cuda = par_4.w;        // [3]
    T device_specific_ldisc_cuda = par_2.x;           // [0]
    T device_specific_A_cuda = par_2.y;               // [1]

    T &w = apparent_weight;
    T &Ndisc = persistent_weight;

    uint32_t pulse_count = floor(pulse_length / base_time_step);
    pulse_count *= n;
    double Ndisc_double = Ndisc;

    if (negative > 0) {
      for (int i_updates = 0; i_updates < pulse_count; i_updates++) {
        T ratio = Ndisc_double;
        ratio = (ratio - Ndisc) / (device_specific_Ndiscmax_cuda - Ndisc);

        apply_cycle_to_cycle_noise(
            ratio, device_specific_Ndiscmax_cuda, device_specific_Ndiscmin_cuda,
            device_specific_ldisc_cuda, device_specific_A_cuda, Ndiscmax_std, Ndiscmin_std,
            ldisc_std, rdisc_std, ldisc_std_slope, rdisc_std_slope, local_state,
            Ndiscmax_ctoc_upper_bound, Ndiscmax_ctoc_lower_bound, Ndiscmin_ctoc_upper_bound,
            Ndiscmin_ctoc_lower_bound, ldisc_ctoc_upper_bound, ldisc_ctoc_lower_bound,
            rdisc_ctoc_upper_bound, rdisc_ctoc_lower_bound);

        T max_bound = MIN(Ndisc_max_bound, device_specific_Ndiscmax_cuda);
        T min_bound = MAX(Ndisc_min_bound, device_specific_Ndiscmin_cuda);

        Ndisc_double = MAX(MIN(Ndisc_double, max_bound), min_bound);

        T gamma_V_disk_coefficient_A =
            gamma_coefficient / (V_disk_coefficient * device_specific_A_cuda);
        T Rth_negative_coefficient_A = Rth_negative_coefficient / device_specific_A_cuda;
        T a_ny0_l = a_ny0 / device_specific_ldisc_cuda;
        T I_mem =
            -alpha_SET - beta_SET / (__powf((1.0 + __powf((c_SET / Ndisc_double), d_SET)), f_SET));
        T gamma = gamma_V_disk_coefficient_A * I_mem / Ndisc_double;
        T V_other_than_series =
            pulse_voltage_SET - (I_mem * (R + V_series_coefficient * I_mem * I_mem));
        T Treal = T0 + I_mem * V_other_than_series * Rth_negative_coefficient_A;
        T dWa_mean = dWa * (sqrtf(1.0 - gamma * gamma) + gamma * asinf(gamma));
        T dWa_difference = dWa * (gamma * half_pi);
        T denominator = PHYSICAL_PARAMETER_kb_over_e * Treal;
        T c_v0 = (Nplug + Ndisc_double) / 2.0;
        T F_limit = 1.0 - __powf((Ndisc_double / device_specific_Ndiscmax_cuda), 10.0);
        T dNdt = -c_v0 * a_ny0_l * F_limit *
                 (__expf(-(dWa_mean - dWa_difference) / denominator) -
                  __expf(-(dWa_mean + dWa_difference) / denominator));

        Ndisc_double = Ndisc_double + dNdt * base_time_step;
        Ndisc_double = MAX(MIN(Ndisc_double, max_bound), min_bound);
      }

      w = map_Ndisc_to_weight(
          Ndisc_double, current_min, w_min, current_to_weight_ratio, g_read, h_read, j_0, k0,
          Original_Ndiscmin);
      Ndisc = (T)Ndisc_double;

    } else {
      for (int i_updates = 0; i_updates < pulse_count; i_updates++) {
        T ratio = Ndisc_double;
        ratio = (Ndisc - ratio) / (Ndisc - device_specific_Ndiscmin_cuda);

        apply_cycle_to_cycle_noise(
            ratio, device_specific_Ndiscmax_cuda, device_specific_Ndiscmin_cuda,
            device_specific_ldisc_cuda, device_specific_A_cuda, Ndiscmax_std, Ndiscmin_std,
            ldisc_std, rdisc_std, ldisc_std_slope, rdisc_std_slope, local_state,
            Ndiscmax_ctoc_upper_bound, Ndiscmax_ctoc_lower_bound, Ndiscmin_ctoc_upper_bound,
            Ndiscmin_ctoc_lower_bound, ldisc_ctoc_upper_bound, ldisc_ctoc_lower_bound,
            rdisc_ctoc_upper_bound, rdisc_ctoc_lower_bound);

        T max_bound = MIN(Ndisc_max_bound, device_specific_Ndiscmax_cuda);
        T min_bound = MAX(Ndisc_min_bound, device_specific_Ndiscmin_cuda);
        Ndisc_double = MAX(MIN(Ndisc_double, max_bound), min_bound);

        T gamma_V_disk_coefficient_l = gamma_coefficient / lcell;
        T Rth_positive_coefficient_A = Rth_positive_coefficient / device_specific_A_cuda;
        T I_mem =
            g_RESET /
            (__powf((1 + h_RESET * __powf((Ndisc_double / Original_Ndiscmin), -j_0)), 1.0 / k0));
        T V_other_than_series =
            pulse_voltage_RESET - (I_mem * (R + V_series_coefficient * I_mem * I_mem));
        T gamma = gamma_V_disk_coefficient_l * V_other_than_series;
        T Treal = T0 + I_mem * V_other_than_series * Rth_positive_coefficient_A;
        T dWa_mean = dWa * (sqrtf(1.0 - gamma * gamma) + gamma * asinf(gamma));
        T dWa_difference = dWa * (gamma * half_pi);
        T denominator = PHYSICAL_PARAMETER_kb_over_e * Treal;
        T c_v0 = (Nplug + Ndisc_double) / 2.0;
        T F_limit = 1.0 - __powf((device_specific_Ndiscmin_cuda / Ndisc_double), 10.0);
        T dNdt = -(c_v0 * a_ny0 * F_limit *
                   (__expf(-(dWa_mean - dWa_difference) / denominator) -
                    __expf(-(dWa_mean + dWa_difference) / denominator))) /
                 device_specific_ldisc_cuda;
        Ndisc_double = Ndisc_double + dNdt * base_time_step;
        Ndisc_double = MAX(MIN(Ndisc_double, max_bound), min_bound);
      }

      w = map_Ndisc_to_weight(
          Ndisc_double, current_min, w_min, current_to_weight_ratio, g_read, h_read, j_0, k0,
          Original_Ndiscmin);
      Ndisc = (T)Ndisc_double;
    }
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
  v.push_back(RPU::make_unique<PWUKernelParameterSingleFunctor<
                  T, UpdateFunctorJARTv1b<T>, DEVICE_PARAMETER_COUNT>> ARGS);
  v.push_back(
      RPU::make_unique<
          PWUKernelParameterBatchFunctor<T, UpdateFunctorJARTv1b<T>, DEVICE_PARAMETER_COUNT>> ARGS);
  v.push_back(RPU::make_unique<PWUKernelParameterBatchSharedFunctor<
                  T, UpdateFunctorJARTv1b<T>, DEVICE_PARAMETER_COUNT>> ARGS);

  return v;
}

#undef ARGS

template <typename T>
__global__ void kernelMapWeightToNdisc(
    T *weights,
    T *Ndiscs,
    int size,
    T current_min,
    T weight_min_bound,
    T weight_to_current_ratio,
    T g_read,
    T h_read,
    T j_0,
    T k0,
    T Ndiscmin) {

  RPU_CUDA_1D_KERNEL_LOOP(idx, size) {
    T current = (weights[idx] - weight_min_bound) * weight_to_current_ratio + current_min;
    Ndiscs[idx] =
        __powf(((__powf((g_read / current), k0) - 1.0) / (h_read)), 1.0 / (-j_0)) * Ndiscmin;
  }
}

template <typename T>
void map_weight_to_Ndisc(
    const CudaContext *context,
    T *w,
    T *Ndiscs,
    const int size,
    const T current_min,
    const T weight_min_bound,
    const T weight_to_current_ratio,
    const T g_read,
    const T h_read,
    const T j_0,
    const T k0,
    const T Ndiscmin) {

  int nthreads = context->getNThreads();
  int nblocks = context->getNBlocks(size, nthreads);
  kernelMapWeightToNdisc<T><<<nblocks, nthreads, 0, context->getStream()>>>(
      w, Ndiscs, size, current_min, weight_min_bound, weight_to_current_ratio, g_read, h_read, j_0,
      k0, Ndiscmin);
}
template void map_weight_to_Ndisc<float>(
    const CudaContext *,
    float *,
    float *,
    const int,
    const float,
    const float,
    const float,
    const float,
    const float,
    const float,
    const float,
    const float);
#ifdef RPU_USE_DOUBLE
template void map_weight_to_Ndisc<double>(
    const CudaContext *,
    double *,
    double *,
    const int,
    const double,
    const double,
    const double,
    const double,
    const double,
    const double,
    const double,
    const double);
#endif

template <typename T>
void JARTv1bRPUDeviceCuda<T>::applyWeightUpdate(T *weights, T *dw_and_current_weight_out) {

  if (getPar().real_write_noise_std > 0) {
    RPU_FATAL("ApplyWeightUpdate is not supported with write_noise_std>0!");
  }
  RPU::math::elemaddcopysat<T>(
      this->context_, weights, dw_and_current_weight_out, this->size_, this->get4ParamsData());

  const auto &par = getPar();
  T *Ndisc = get1ParamsData();

  map_weight_to_Ndisc<T>(
      this->context_, weights, Ndisc, this->size_, par.current_min, par.w_min,
      par.weight_to_current_ratio, par.g_read, par.h_read, par.j_0, par.k0, par.Ndiscmin);
}

template <typename T> void JARTv1bRPUDeviceCuda<T>::clipWeights(T *weights, T clip) {

  RPU::math::elemsat<T>(this->context_, weights, this->size_, this->get4ParamsData());
  if (clip >= 0) {
    RPU::math::aclip<T>(this->context_, weights, this->size_, clip);
  }

  const auto &par = getPar();

  if (par.real_write_noise_std > 0) {
    // re-uses the diffusion rnd
    if (this->dev_diffusion_nrnd_ == nullptr) {
      this->initDiffusionRnd();
      this->rnd_context_->randNormal(
          this->dev_diffusion_nrnd_->getData(), this->dev_diffusion_nrnd_->getSize());
    }
    this->rnd_context_->synchronize();

    RPU::math::elemweightedsum<T>(
        this->context_, weights, this->size_, weights, (T)1.0, this->dev_diffusion_nrnd_->getData(),
        par.real_write_noise_std);

    this->rnd_context_->recordWaitEvent(this->context_->getStream());
    this->rnd_context_->randNormal(
        this->dev_diffusion_nrnd_->getData(), this->dev_diffusion_nrnd_->getSize());
  }
  T *Ndisc = get1ParamsData();

  map_weight_to_Ndisc<T>(
      this->context_, weights, Ndisc, this->size_, par.current_min, par.w_min,
      par.weight_to_current_ratio, par.g_read, par.h_read, par.j_0, par.k0, par.Ndiscmin);
}

template class JARTv1bRPUDeviceCuda<float>;
#ifdef RPU_USE_DOUBLE
template class JARTv1bRPUDeviceCuda<double>;
#endif

} // namespace RPU
