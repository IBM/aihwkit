/**
 * (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
 *
 * Licensed under the MIT license. See LICENSE file in the project root for details.
 */

#include "pwu_kernel_parameter.h"
#include "rpu_pulsed_meta_parameter.h"
#include "rpucuda_powstep_device.h"
#include <memory>

namespace RPU {

template <typename T> struct UpdateFunctorPowStep {

  __device__ __forceinline__ void operator()(
      T &apparent_weight,
      uint32_t n,
      uint32_t negative,
      const param4_t par_4,         // min_bound, scale_down, max_bound, scale_up,
      const param2_t gamma_down_up, // gamma_down, gamma_up
      T &persistent_weight,
      const T *write_noise_std,
      const int global_params_count,
      T noise_std_dw,
      curandState &local_state) {

    UNUSED(global_params_count); // fixed

    T wmin = par_4.x; // [0]
    T wmax = par_4.z; // [2]
    T range = wmax - wmin;
    if (range == (T)0.0) {
      return;
    }
    T uw_std = *write_noise_std;
    T &w = uw_std > (T)0.0 ? persistent_weight : apparent_weight;
    // negative > 0 means sign < 0 and thus up-direction
    T scale = (negative > 0) ? ((T)par_4.w) : (-(T)par_4.y);                //[3] (up), [1] (down)
    T gamma = (negative > 0) ? ((T)gamma_down_up.y) : ((T)gamma_down_up.x); // [1] (up), [0] (down)

    // up direction: ((wmax - w) / range) ^ gamma
    // down direction:  ((w - wmin) / range) ^ gamma  == (1 - (wmax - w)/range) ^ gamma

    // n is larger 0 in any case
    if (n == 1) {
      T x = (wmax - w) / range;
      T dw = scale * ((negative > 0) ? (T)__powf(x, gamma) : (T)__powf((T)1.0 - x, gamma));

      if (noise_std_dw > (T)0.0) {
        T stoch_value = curand_normal(&local_state);
        stoch_value *= noise_std_dw;
        w += dw * ((T)1.0 + stoch_value);
      } else {
        w += dw;
      }
      w = (w > wmax) ? wmax : w;
      w = (w < wmin) ? wmin : w;

    } else {
      if (noise_std_dw > (T)0.0) {
        for (int i_updates = 0; i_updates < n; i_updates++) {
          T stoch_value = curand_normal(&local_state);
          stoch_value *= noise_std_dw;
          T x = (wmax - w) / range;
          T dw = scale * ((negative > 0) ? (T)__powf(x, gamma) : (T)__powf((T)1.0 - x, gamma));
          w += dw * ((T)1.0 + stoch_value);
          // better always check both bounds
          w = (w > wmax) ? wmax : w;
          w = (w < wmin) ? wmin : w;
        }
      } else {
        for (int i_updates = 0; i_updates < n; i_updates++) {
          T x = (wmax - w) / range;
          w += scale * ((negative > 0) ? (T)__powf(x, gamma) : (T)__powf((T)1.0 - x, gamma));
          // better always check both bounds
          w = (w > wmax) ? wmax : w;
          w = (w < wmin) ? wmin : w;
        }
      }
    }

    // add update write noise onto apparent weight
    if (uw_std > (T)0.0) {
      T stoch_value = curand_normal(&local_state);
      apparent_weight = persistent_weight + uw_std * stoch_value;
    }
  }
};

#define ARGS                                                                                       \
  (this->context_, this->x_size_, this->d_size_, m_batch, nK32, use_bo64, out_trans, up,           \
   par.getName())

template <typename T>
pwukpvec_t<T> PowStepRPUDeviceCuda<T>::getUpdateKernels(
    int m_batch, int nK32, int use_bo64, bool out_trans, const PulsedUpdateMetaParameter<T> &up) {

  pwukpvec_t<T> v;
  const auto &par = getPar();
  v.push_back(
      RPU::make_unique<PWUKernelParameterSingleFunctor<T, UpdateFunctorPowStep<T>, 1>> ARGS);
  v.push_back(RPU::make_unique<PWUKernelParameterBatchFunctor<T, UpdateFunctorPowStep<T>, 1>> ARGS);
  v.push_back(
      RPU::make_unique<PWUKernelParameterBatchSharedFunctor<T, UpdateFunctorPowStep<T>, 1>> ARGS);
  v.push_back(
      RPU::make_unique<
          PWUKernelParameterBatchSharedWeightOutputFunctor<T, UpdateFunctorPowStep<T>, 1>> ARGS);
  return v;
}

#undef ARGS

/*********************************************************************************/
/* infinite granularity update */

namespace {
constexpr int IG_THREADS_PER_BLOCK = 256;

inline int ig_num_blocks(int total_size) {
  return (total_size + IG_THREADS_PER_BLOCK - 1) / IG_THREADS_PER_BLOCK;
}

template <typename T>
__device__ __forceinline__ void igStoreWeight(
    T *weights,
    T *persistent_weights,
    const T *write_noise_std,
    curandState_t *random_states,
    int idx,
    T w) {
  if (persistent_weights == nullptr) {
    weights[idx] = w;
    return;
  }

  persistent_weights[idx] = w;
  T uw_std = write_noise_std == nullptr ? (T)0.0 : *write_noise_std;
  if (uw_std > (T)0.0 && random_states != nullptr) {
    curandState local_state = random_states[idx];
    weights[idx] = w + uw_std * (T)curand_normal(&local_state);
    random_states[idx] = local_state;
  } else {
    weights[idx] = w;
  }
}

template <typename T>
__global__ void kernelIGPowStep(
    T *weights,
    const T *grad_matrix,
    const param_t *params4,
    const param_t *params_gamma,
    int total_size,
    T *persistent_weights,
    const T *write_noise_std,
    curandState_t *random_states) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_size) {
    return;
  }

  T G = grad_matrix[idx];
  if (G == (T)0.0) {
    return;
  }

  param4_t p4 = reinterpret_cast<const param4_t *>(params4)[idx];
  param2_t pg = reinterpret_cast<const param2_t *>(params_gamma)[idx];

  T wmin = p4.x;
  T wmax = p4.z;
  T range = wmax - wmin;
  if (range == (T)0.0) {
    return;
  }

  T abs_G = (G > (T)0.0) ? G : -G;
  T w = persistent_weights == nullptr ? weights[idx] : persistent_weights[idx];
  T x = (wmax - w) / range;
  T dw = G < (T)0.0 ? -(T)p4.y * abs_G * __powf((T)1.0 - x, (T)pg.x)
                    : (T)p4.w * abs_G * __powf(x, (T)pg.y);

  w += dw;
  w = (w > wmax) ? wmax : w;
  w = (w < wmin) ? wmin : w;
  igStoreWeight(weights, persistent_weights, write_noise_std, random_states, idx, w);
}
} // namespace

template <typename T>
void PowStepRPUDeviceCuda<T>::doInfiniteGranularityUpdate(
    T *dev_weights, const T *grad_matrix, curandState_t *dev_states) {
  int total_size = this->x_size_ * this->d_size_;
  kernelIGPowStep<T>
      <<<ig_num_blocks(total_size), IG_THREADS_PER_BLOCK, 0, this->context_->getStream()>>>(
          dev_weights, grad_matrix, this->get4ParamsData(), this->get2ParamsData(), total_size,
          this->get1ParamsData(), this->getGlobalParamsData(), dev_states);
}

template class PowStepRPUDeviceCuda<float>;
#ifdef RPU_USE_DOUBLE
template class PowStepRPUDeviceCuda<double>;
#endif
#ifdef RPU_USE_FP16
template class PowStepRPUDeviceCuda<half_t>;
#endif

} // namespace RPU
