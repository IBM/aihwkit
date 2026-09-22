/**
 * (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
 *
 * Licensed under the MIT license. See LICENSE file in the project root for details.
 */

#include "pwu_kernel_parameter.h"
#include "rpu_pulsed_meta_parameter.h"
#include "rpucuda_powstep_reference_device.h"
#include <memory>

namespace RPU {

template <typename T> struct UpdateFunctorPowStepReference {

  __device__ __forceinline__ void operator()(
      T &weight,
      uint32_t n,
      uint32_t negative,
      const param4_t par_4,         // min_bound, scale_down, max_bound, scale_up,
      const param2_t gamma_down_up, // gamma_down, gamma_up
      T &ref,
      const T *global_par,
      const int global_params_count,
      T noise_std_dw,
      curandState &local_state) {

    UNUSED(global_params_count); // fixed
    UNUSED(global_par);          // fixed

    T wmin = par_4.x; // [0]
    T wmax = par_4.z; // [2]
    T range = wmax - wmin;
    if (range == (T)0.0) {
      return;
    }
    T &w = weight;

    // first add reference:
    w += ref;

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

    // subtract reference again
    w -= ref;
  }
};

#define ARGS                                                                                       \
  (this->context_, this->x_size_, this->d_size_, m_batch, nK32, use_bo64, out_trans, up,           \
   par.getName())

template <typename T>
pwukpvec_t<T> PowStepReferenceRPUDeviceCuda<T>::getUpdateKernels(
    int m_batch, int nK32, int use_bo64, bool out_trans, const PulsedUpdateMetaParameter<T> &up) {

  pwukpvec_t<T> v;
  const auto &par = getPar();
  v.push_back(
      RPU::make_unique<PWUKernelParameterSingleFunctor<T, UpdateFunctorPowStepReference<T>, 1>>
          ARGS);
  v.push_back(
      RPU::make_unique<PWUKernelParameterBatchFunctor<T, UpdateFunctorPowStepReference<T>, 1>>
          ARGS);
  v.push_back(
      RPU::make_unique<PWUKernelParameterBatchSharedFunctor<T, UpdateFunctorPowStepReference<T>, 1>>
          ARGS);
  v.push_back(
      RPU::make_unique<
          PWUKernelParameterBatchSharedWeightOutputFunctor<T, UpdateFunctorPowStepReference<T>, 1>>
          ARGS);

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
__global__ void kernelIGPowStepReference(
    T *weights,
    const T *grad_matrix,
    const param_t *params4,
    const param_t *params_gamma,
    const T *reference,
    int total_size) {
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

  T ref = reference[idx];
  T abs_G = (G > (T)0.0) ? G : -G;
  T w = weights[idx] + ref;
  T x = (wmax - w) / range;
  T dw = G < (T)0.0 ? -(T)p4.y * abs_G * __powf((T)1.0 - x, (T)pg.x)
                    : (T)p4.w * abs_G * __powf(x, (T)pg.y);

  w += dw;
  w = (w > wmax) ? wmax : w;
  w = (w < wmin) ? wmin : w;
  weights[idx] = w - ref;
}
} // namespace

template <typename T>
void PowStepReferenceRPUDeviceCuda<T>::doInfiniteGranularityUpdate(
    T *dev_weights, const T *grad_matrix, curandState_t *dev_states) {
  UNUSED(dev_states);
  int total_size = this->x_size_ * this->d_size_;
  kernelIGPowStepReference<T>
      <<<ig_num_blocks(total_size), IG_THREADS_PER_BLOCK, 0, this->context_->getStream()>>>(
          dev_weights, grad_matrix, this->get4ParamsData(), this->get2ParamsData(),
          this->get1ParamsData(), total_size);
}

template class PowStepReferenceRPUDeviceCuda<float>;
#ifdef RPU_USE_DOUBLE
template class PowStepReferenceRPUDeviceCuda<double>;
#endif
#ifdef RPU_USE_FP16
template class PowStepReferenceRPUDeviceCuda<half_t>;
#endif

} // namespace RPU
