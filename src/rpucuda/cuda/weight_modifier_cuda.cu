/**
 * (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
 */

#include "cuda_fp16_util.h"
#include "cuda_math_util.h"
#include "weight_modifier_cuda.h"

namespace RPU {

#define RPU_WM_KERNEL_LOOP(STOCH_IF, BODY)                                                         \
  int tid = blockDim.x * blockIdx.x + threadIdx.x;                                                 \
  int total_threads = blockDim.x * gridDim.x;                                                      \
  int size = size_in;                                                                              \
  int size_without_bias = copy_last_column ? (size - d_size) : size;                               \
  const bool stoch_if = STOCH_IF;                                                                  \
                                                                                                   \
  curandState local_state;                                                                         \
  if (stoch_if && tid < size) {                                                                    \
    local_state = random_states[tid];                                                              \
  }                                                                                                \
                                                                                                   \
  for (int i_stride = 0; i_stride < size; i_stride += total_threads) {                             \
    int i = i_stride + tid;                                                                        \
    if (i < size_without_bias) {                                                                   \
      {                                                                                            \
        BODY;                                                                                      \
      }                                                                                            \
    } else if ((i < size) && (new_weights != weights)) {                                           \
      new_weights[i] = weights[i];                                                                 \
    }                                                                                              \
  }                                                                                                \
                                                                                                   \
  if (stoch_if && tid < size) {                                                                    \
    random_states[tid] = local_state;                                                              \
  }

template <typename T>
__global__ void kernelModifyWeightsDiscretize(
    int size_in,
    int d_size,
    const bool copy_last_column,
    T *new_weights,
    const T *weights,
    const T res_in, // need to larger than zero!!
    const bool sto_round,
    const T assumed_wmax,
    const T *wmax,
    curandState_t *random_states) {
  const T res = res_in;
  T amax = (wmax) ? (*wmax) : assumed_wmax;
  amax = amax > (T)0.0 ? amax : (T)1.0;

  RPU_WM_KERNEL_LOOP(
      sto_round,

      T value = weights[i] / amax;
      value /= res;

      if (stoch_if) {
        T stoch_value = curand_uniform(&local_state);
        value += stoch_value - (T)0.5;
      }

      new_weights[i] = amax * res * round(value););
}

template <typename T>
__global__ void kernelModifyWeightsDoReFa(
    int size_in,
    int d_size,
    const bool copy_last_column,
    T *new_weights,
    const T *weights,
    const T res_in, // need to larger than zero!!
    const bool sto_round,
    const T dorefa_clip,
    T assumed_wmax,
    T *wmax,
    curandState_t *random_states) {
  T amax = (wmax) ? (*wmax) : assumed_wmax;
  amax = amax > (T)0.0 ? amax : (T)1.0;

  const T res = res_in;
  const T scale = fabs(dorefa_clip / (T)tanhf(amax));

  RPU_WM_KERNEL_LOOP(
      sto_round,

      T value = weights[i];
      value = (T)tanhf(value) * scale;

      value /= res;

      if (stoch_if) {
        T stoch_value = curand_uniform(&local_state);
        value += stoch_value - (T)0.5;
      }

      new_weights[i] = res * round(value););
}

template <typename T>
__global__ void kernelModifyWeightsDiscretizeAddNormal(
    int size_in,
    int d_size,
    const bool copy_last_column,
    T *new_weights,
    const T *weights,
    const T res_in, // need to larger than zero!!
    const bool sto_round_in,
    const T stddev_in,
    const T assumed_wmax,
    const T *wmax,
    curandState_t *random_states) {
  const T res = res_in;
  const T stddev = stddev_in;
  const bool sto_round = sto_round_in;
  T amax = (wmax) ? (*wmax) : assumed_wmax;
  amax = amax > (T)0.0 ? amax : (T)1.0;

  RPU_WM_KERNEL_LOOP(
      true,

      T value = weights[i] / amax;

      value /= res;

      if (sto_round) { value += (T)curand_uniform(&local_state) - (T)0.5; }

      value = res * round(value);
      T stoch_value = curand_normal(&local_state);
      new_weights[i] = amax * (value + stddev * stoch_value););
}

template <typename T>
__global__ void kernelModifyWeightsAddNormal(
    int size_in,
    int d_size,
    const bool copy_last_column,
    T *new_weights,
    const T *weights,
    const T stddev_in,
    const T assumed_wmax,
    const T *wmax,
    curandState_t *random_states) {
  T amax = (wmax) ? (*wmax) : assumed_wmax;
  amax = amax > (T)0.0 ? amax : (T)1.0;

  const T stddev = amax * stddev_in;

  RPU_WM_KERNEL_LOOP(true,

                     T stoch_value = curand_normal(&local_state);
                     new_weights[i] = weights[i] + stddev * stoch_value;);
}

template <typename T>
__global__ void kernelModifyWeightsMultNormal(
    int size_in,
    int d_size,
    const bool copy_last_column,
    T *new_weights,
    const T *weights,
    const T stddev_in,
    const T assumed_wmax,
    const T *wmax,
    curandState_t *random_states) {
  T amax = (wmax) ? (*wmax) : assumed_wmax;
  amax = amax > (T)0.0 ? amax : (T)1.0;

  const T stddev = stddev_in * amax;

  RPU_WM_KERNEL_LOOP(true,

                     T w = weights[i];
                     T stoch_value = curand_normal(&local_state);

                     new_weights[i] = w * ((T)1.0 + stddev * stoch_value););
}

template <typename T, bool preserve_sign>
__global__ void kernelModifyWeightsProgNoiseN(
    int size_in,
    int d_size,
    const bool copy_last_column,
    T *new_weights,
    const T *weights,
    const T stddev_in, // additional scale:
    int n_coeffs,
    T *coeffs,
    const T assumed_wmax,
    const T *wmax,
    curandState_t *random_states) {
  T amax = (wmax) ? (*wmax) : assumed_wmax;
  amax = amax > (T)0.0 ? amax : (T)1.0;

  const T stddev = stddev_in;

  RPU_WM_KERNEL_LOOP(
      true,

      T w = weights[i];
      T stoch_value = curand_normal(&local_state); T aw = fabs(w) / amax;

      T paw = 1; T sig = coeffs[0];

      for (int j = 1; j < n_coeffs; j++) {
        paw *= aw;
        sig += coeffs[j] * paw;
      }

      sig *= stddev;
      T out_w = w + amax * sig * stoch_value;

      if (preserve_sign) { new_weights[i] = (w < (T)0.0) ? -fabs(out_w) : fabs(out_w); } else {
        new_weights[i] = out_w;
      });
}

template <typename T, bool preserve_sign>
__global__ void kernelModifyWeightsProgNoise(
    int size_in,
    int d_size,
    const bool copy_last_column,
    T *new_weights,
    const T *weights,
    const T stddev_in, // additional scale:
    const T p0_in,
    const T p1_in,
    const T p2_in,
    const T assumed_wmax,
    const T *wmax,
    curandState_t *random_states) {

  T amax = (wmax) ? (*wmax) : assumed_wmax;
  amax = amax > (T)0.0 ? amax : (T)1.0;

  const T stddev = stddev_in;
  const T p0 = p0_in;
  const T p1 = p1_in;
  const T p2 = p2_in;

  RPU_WM_KERNEL_LOOP(
      true,

      T w = weights[i];
      T stoch_value = curand_normal(&local_state);

      T aw = fabs(w) / amax;

      T sig = (p0 + aw * p1 + aw * aw * p2) * stddev; T out_w = w + amax * sig * stoch_value;

      if (preserve_sign) { new_weights[i] = (w < (T)0.0) ? -fabs(out_w) : fabs(out_w); } else {
        new_weights[i] = out_w;
      });
}

#define PCM_T_READ (double)250.0e-9
#define PCM_P0 0.26348f
#define PCM_P1 1.9650f
#define PCM_P2 -1.1731f

/*This is the new ZEUS baseline definition of PCM noise. NOTE: this
  does ONLY include the drift variation NOT the mean drift */
template <typename T>
__global__ void kernelModifyWeightsPCMNoise(
    int size_in,
    int d_size,
    const bool copy_last_column, // in case of bias
    T *new_weights,
    const T *weights,
    const T t_minus_t0_in,  // relative to t0, only for drift variation, not mean
    const T noise_scale_in, // additional scale, scaling both prog noise and 1/f (read) noise
    const T gmax_in,
    const T t0_in,
    const T zero_thres_in, // in % of P0!!
    const T prob_at_reset_in,
    const T prob_at_gmax_in,
    const T prob_at_random_in,
    const T log_sqrt_factor_in, // should be sqrt(log((((double) t)+t_read) / (2.0*t_read)))
    const T assumed_wmax,
    const T *wmax,
    curandState_t *random_states) {
  T amax = (wmax) ? (*wmax) : assumed_wmax;
  amax = amax > (T)0.0 ? amax : (T)1.0;
  const T t0 = t0_in;
  const T t = t_minus_t0_in + t0;

  const T noise_scale = noise_scale_in;

  const T gmax = gmax_in;
  const T log_sqrt_factor = log_sqrt_factor_in;
  const T prob_at_gmax = prob_at_gmax_in;
  const T prob_at_reset = prob_at_reset_in;
  const T prob_at_random = prob_at_random_in;
  const T zero_thres = zero_thres_in * noise_scale * (T)PCM_P0;

  RPU_WM_KERNEL_LOOP(
      true, T w = weights[i]; T g = MAX(MIN(w, amax), -amax) * gmax / amax;

      g = fabs(g) > zero_thres ? g : (T)0.0; // to avoid programming small numbers

      bool w_negative = g < (T)0.0;

      if (prob_at_reset > (T)0.0) {
        T stoch_value = curand_uniform(&local_state);
        g = (stoch_value < prob_at_reset) ? (T)0.0 : g;
      }

      if (prob_at_gmax > (T)0.0) {
        T stoch_value1 = curand_uniform(&local_state);
        T stoch_value2 = curand_uniform(&local_state);

        if (stoch_value1 < prob_at_gmax) {
          g = (stoch_value2 > (T)0.5) ? (g - gmax) : (g + gmax);
        }
      }

      if (prob_at_random > (T)0.0) {
        T stoch_value1 = curand_uniform(&local_state);
        T stoch_value2 = curand_uniform(&local_state);

        if (stoch_value1 < prob_at_random) {
          stoch_value2 *= gmax;
          g = (g > (T)0.0) ? stoch_value2 : -stoch_value2;
        }
      }

      g = MIN(MAX(g, -gmax), gmax); // clip the target weights to gmax

      T w_final = (T)0.0;
      // exactly zero if close to zero anyway (in this case one will
      // not programm it and leave it at RESET)
      if (fabs(g) > (T)0.0) {
        T stoch_value = curand_normal(&local_state);
        T ag_rel = fabs(g / gmax);
        T sig_prog = (T)PCM_P0 + ag_rel * (T)PCM_P1 + ag_rel * ag_rel * (T)PCM_P2;

        T g_prog = g + noise_scale * sig_prog * stoch_value;
        T ag_prog_rel = fabs(g_prog / gmax);

        // drift
        T g_drift = g_prog;
        if (t > t0) {
          T ag_rel_for_drift = MAX(ag_prog_rel, (T)1e-6);
          stoch_value = curand_normal(&local_state);

          // we do not consider the mean drift here [assuming it can
          // be corrected], only the variation mu_drift =
          // np.minimum(np.maximum(-0.0155*np.log(Grel) + 0.0244,
          // 0.049), 0.1)

          T sig_drift =
              MIN(MAX((T)-0.0125 * (T)logf(ag_rel_for_drift) - (T)0.0059, (T)0.008), (T)0.045);
          T nu_drift = fabs(stoch_value * sig_drift);

          g_drift = g_prog * (T)powf(t / t0, -nu_drift);
        }

        // 1/f noise
        T g_final = g_drift;

        if (t > (T)0.0) {
          stoch_value = curand_normal(&local_state);

          T Qs = MIN((T)0.0088 / MAX((T)powf(ag_prog_rel, (T)0.65), (T)1e-3), (T)0.2);
          T sig_noise = Qs * log_sqrt_factor;

          g_final = g_drift + noise_scale * fabs(g_drift) * sig_noise * stoch_value;
        }

        w_final = g_final / gmax * amax;

        // clip. cannot change signs. Note that boundary noise will be reflected
        w_final = w_negative ? -fabs(w_final) : fabs(w_final);
      } new_weights[i] = w_final;

  );
}

template <typename T>
__global__ void kernelModifyWeightsDropConnections(
    int size_in,
    int d_size,
    const bool copy_last_column,
    T *new_weights,
    const T *weights,
    const T prob_in,
    curandState_t *random_states) {

  const T prob = prob_in;

  RPU_WM_KERNEL_LOOP(
      true,

      T stoch_value = curand_uniform(&local_state);
      if (stoch_value < prob) { new_weights[i] = (T)0.0; });
}

// ctor
template <typename T>
WeightModifierCuda<T>::WeightModifierCuda(CudaContextPtr context, int x_size, int d_size)
    : context_(context), x_size_(x_size), d_size_(d_size), size_(x_size * d_size),
      enable_during_test_(false) {}

template <typename T>
void WeightModifierCuda<T>::apply(
    T *new_weights, const T *weights, const WeightModifierParameter<T> &wmpar) {

  int nthreads = context_->getNThreads();
  auto s = context_->getStream();
  int nblocks = context_->getNStrideBlocks(size_, nthreads);

  bool done = false;
  enable_during_test_ = wmpar.enable_during_test;

  T *amax = nullptr;
  if (wmpar.rel_to_actual_wmax && wmpar.type != WeightModifierType::Copy) {
    if (!amaximizer_) {
      amaximizer_ = RPU::make_unique<Maximizer<T>>(
          context_, wmpar.copy_last_column ? (size_ - d_size_) : size_, true);
    }
    amaximizer_->compute(weights, 1, false);
    amax = amaximizer_->getMaxValues();
  }

  if (wmpar.type != WeightModifierType::Copy) {
    if (wmpar.per_batch_sample) {
      RPU_FATAL("Per batch sample is not implemented in RPUCuda");
    }
  }

  // note: all methods need to work in
  switch (wmpar.type) {
  case WeightModifierType::Copy: {

    if (new_weights == weights) {
      RPU_FATAL("cannot use WeightModifierType::Copy with in-place weights.");
    }
    // copies below
    break; // maybe dropping below though
  }

  case WeightModifierType::Discretize: {

    if (wmpar.res > (T)0.0) {

      kernelModifyWeightsDiscretize<T><<<nblocks, nthreads, 0, s>>>(
          size_, d_size_, wmpar.copy_last_column, new_weights, weights, wmpar.res, wmpar.sto_round,
          wmpar.assumed_wmax, amax,
          wmpar.sto_round ? context_->getRandomStates(nblocks * nthreads) : nullptr);
      done = true;
    }
    break;
  }
  case WeightModifierType::DoReFa: {
    if (wmpar.res > (T)0.0) {

      kernelModifyWeightsDoReFa<T><<<nblocks, nthreads, 0, s>>>(
          size_, d_size_, wmpar.copy_last_column, new_weights, weights, wmpar.res, wmpar.sto_round,
          wmpar.dorefa_clip, wmpar.assumed_wmax, amax,
          wmpar.sto_round ? context_->getRandomStates(nblocks * nthreads) : nullptr);
      done = true;
    }
    break;
  }

  case WeightModifierType::MultNormal: {
    if (wmpar.std_dev > (T)0.0) {

      kernelModifyWeightsMultNormal<T><<<nblocks, nthreads, 0, s>>>(
          size_, d_size_, wmpar.copy_last_column, new_weights, weights, wmpar.std_dev,
          wmpar.assumed_wmax, amax, context_->getRandomStates(nblocks * nthreads));
      done = true;
    }
    break;
  }

  case WeightModifierType::AddNormal: {
    if (wmpar.std_dev > (T)0.0) {

      kernelModifyWeightsAddNormal<T><<<nblocks, nthreads, 0, s>>>(
          size_, d_size_, wmpar.copy_last_column, new_weights, weights, wmpar.std_dev,
          wmpar.assumed_wmax, amax, context_->getRandomStates(nblocks * nthreads));
      done = true;
    }
    break;
  }

  case WeightModifierType::Poly: {
    int n_coeffs = wmpar.coeffs.size();
    if (wmpar.std_dev > (T)0.0 && n_coeffs > 0) {

      if (n_coeffs <= 3) {
        kernelModifyWeightsProgNoise<T, false><<<nblocks, nthreads, 0, s>>>(
            size_, d_size_, wmpar.copy_last_column, new_weights, weights, wmpar.std_dev,
            wmpar.coeffs.at(0), (n_coeffs > 1) ? wmpar.coeffs.at(1) : (T)0.0,
            (n_coeffs > 2) ? wmpar.coeffs.at(2) : (T)0.0, wmpar.assumed_wmax, amax,
            context_->getRandomStates(nblocks * nthreads));

      } else {
        // n-poly

        if (wmpar.coeffs.size() != coeffs_.size() || dev_coeffs_ == nullptr) {
          dev_coeffs_ = RPU::make_unique<CudaArray<T>>(context_, n_coeffs, wmpar.coeffs.data());
          coeffs_ = wmpar.coeffs;
          context_->synchronize();
        } else if (coeffs_ != wmpar.coeffs) {
          dev_coeffs_->assign(wmpar.coeffs.data());
          coeffs_ = wmpar.coeffs;
        }

        kernelModifyWeightsProgNoiseN<T, false><<<nblocks, nthreads, 0, s>>>(
            size_, d_size_, wmpar.copy_last_column, new_weights, weights, wmpar.std_dev, n_coeffs,
            dev_coeffs_->getData(), wmpar.assumed_wmax, amax,
            context_->getRandomStates(nblocks * nthreads));
      }
      done = true;
    }
    break;
  }

  case WeightModifierType::ProgNoise: {
    int n_coeffs = wmpar.coeffs.size();
    if (wmpar.std_dev > (T)0.0 && n_coeffs > 0) {

      T std = wmpar.std_dev / wmpar.g_max;

      if (n_coeffs <= 3) {
        kernelModifyWeightsProgNoise<T, true><<<nblocks, nthreads, 0, s>>>(
            size_, d_size_, wmpar.copy_last_column, new_weights, weights, std, wmpar.coeffs.at(0),
            (n_coeffs > 1) ? wmpar.coeffs.at(1) : (T)0.0,
            (n_coeffs > 2) ? wmpar.coeffs.at(2) : (T)0.0, wmpar.assumed_wmax, amax,
            context_->getRandomStates(nblocks * nthreads));

      } else {
        // n-poly

        if (wmpar.coeffs.size() != coeffs_.size() || dev_coeffs_ == nullptr) {
          dev_coeffs_ = RPU::make_unique<CudaArray<T>>(context_, n_coeffs, wmpar.coeffs.data());
          coeffs_ = wmpar.coeffs;
          context_->synchronize();
        } else if (coeffs_ != wmpar.coeffs) {
          dev_coeffs_->assign(wmpar.coeffs.data());
          coeffs_ = wmpar.coeffs;
        }

        kernelModifyWeightsProgNoiseN<T, true><<<nblocks, nthreads, 0, s>>>(
            size_, d_size_, wmpar.copy_last_column, new_weights, weights, std, n_coeffs,
            dev_coeffs_->getData(), wmpar.assumed_wmax, amax,
            context_->getRandomStates(nblocks * nthreads));
      }
      done = true;
    }
    break;
  }

  case WeightModifierType::PCMNoise: {

    T t = wmpar.pcm_t_inference + wmpar.pcm_t0;
    T log_sqrt_factor = sqrt(log((((double)t) + PCM_T_READ) / (2.0 * PCM_T_READ)));

    kernelModifyWeightsPCMNoise<T><<<nblocks, nthreads, 0, s>>>(
        size_, d_size_, wmpar.copy_last_column, new_weights, weights, wmpar.pcm_t_inference,
        wmpar.std_dev, wmpar.g_max, wmpar.pcm_t0, wmpar.pcm_zero_thres, wmpar.pcm_prob_at_reset,
        wmpar.pcm_prob_at_gmax, wmpar.pcm_prob_at_random, log_sqrt_factor, wmpar.assumed_wmax, amax,
        context_->getRandomStates(nblocks * nthreads));
    done = true;

    break;
  }
  case WeightModifierType::DiscretizeAddNormal: {
    if (wmpar.res > (T)0.0 || wmpar.std_dev > (T)0.0) {

      kernelModifyWeightsDiscretizeAddNormal<T><<<nblocks, nthreads, 0, s>>>(
          size_, d_size_, wmpar.copy_last_column, new_weights, weights, wmpar.res, wmpar.sto_round,
          wmpar.std_dev, wmpar.assumed_wmax, amax, context_->getRandomStates(nblocks * nthreads));
      done = true;
    }
    break;
  }

  case WeightModifierType::DropConnect: {
    // to not get into the default branch. Drop connect is handled below
    break;
  }

  default:
    RPU_FATAL("Requested WeightModifierType not implemented.");
  }

  // need to copy in case some parameters were set to 0
  if (!done && new_weights != weights) {
    RPU::math::copy<T>(context_, size_, weights, 1, new_weights, 1);
  }

  if (wmpar.pdrop > (T)0.0) {

    if (new_weights == weights) {
      RPU_FATAL("cannot use pdrop>0 with in-place weights.");
    }

    kernelModifyWeightsDropConnections<T><<<nblocks, nthreads, 0, s>>>(
        size_, d_size_, wmpar.copy_last_column, new_weights, new_weights, wmpar.pdrop,
        context_->getRandomStates(nblocks * nthreads));
  }
}

template <typename T>
void WeightModifierCuda<T>::dumpExtra(RPU::state_t &extra, const std::string prefix) {
  RPU::state_t state;

  // don't handle maximizers (no states)
  RPU::insert(state, "enable_during_test", enable_during_test_);
  RPU::insert(state, "coeffs", coeffs_);
  RPU::insert(state, "dev_coeffs", dev_coeffs_);

  RPU::insertWithPrefix(extra, state, prefix);
}

template <typename T>
void WeightModifierCuda<T>::loadExtra(
    const RPU::state_t &extra, const std::string prefix, bool strict) {

  using V = std::vector<T>;
  auto state = RPU::selectWithPrefix(extra, prefix);

  RPU::load(state, "enable_during_test", enable_during_test_, strict);
  RPU::load(state, "coeffs", coeffs_, strict);
  RPU::load(this->context_, state, "dev_coeffs", dev_coeffs_, strict);
}

template class WeightModifierCuda<float>;
#ifdef RPU_USE_DOUBLE
template class WeightModifierCuda<double>;
#endif
#ifdef RPU_USE_FP16
template class WeightModifierCuda<half_t>;
#endif

#undef RPU_WM_KERNEL_LOOP
} // namespace RPU
