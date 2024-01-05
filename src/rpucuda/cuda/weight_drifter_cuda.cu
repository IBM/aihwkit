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
#include "weight_drifter_cuda.h"

namespace RPU {

template <typename T>
__global__ void kernelDriftWeights(
    int size_in,
    T *weights,
    T *previous_weights,
    T *w0_values,
    T *nu_values,
    T *t_values,
    const T current_t_in,
    const T reset_tol_in,
    const T simple_nu_in,
    const T nu_std_in,
    const T nu_k_in,
    const T a_in,
    const T logG0,
    const T wg_ratio,
    const T w_offset,
    const T g_offset,
    const T w_read_std_in,

    curandState_t *random_states) {

  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  int total_threads = blockDim.x * gridDim.x;
  int size = size_in;
  const T reset_tol = reset_tol_in;
  const T current_t = current_t_in;

  const T nu_std = nu_std_in;
  const T nu_k = nu_k_in;
  const T simple_nu = simple_nu_in;
  const bool is_simple_drift = nu_values == nullptr;
  const T w_read_std = w_read_std_in;
  const T a = a_in;
  const bool stoch_if = w_read_std > (T)0.0 || nu_std > (T)0.0;

  curandState local_state;
  if (stoch_if && tid < size) {
    local_state = random_states[tid];
  }

  for (int i_stride = 0; i_stride < size; i_stride += total_threads) {
    int i = i_stride + tid;

    if (i < size) {

      T w = weights[i];
      T w_prev = previous_weights[i];
      if (fabs(w_prev - w) > reset_tol) {
        // weight has changed and thus need a drift reset
        t_values[i] = current_t;
        w0_values[i] = w;
      } else {
        // this will overwrite the current weight ! make sure that no DECAY/WNOISE is present

        T w0 = w0_values[i];
        T nu = is_simple_drift ? simple_nu : nu_values[i];
        T t = t_values[i];

        nu = (nu_std <= (T)0.0) ? nu : nu + nu_std * nu * (T)curand_normal(&local_state);
        nu = (nu_k == (T)0.0)
                 ? nu
                 : nu - nu_k * (T)__log10f((w - w_offset) / wg_ratio + g_offset) + nu_k * logG0;
        T delta_t = MAX(current_t - t, (T)1.0); // at least t0
        T nu_scale = (T)powf(delta_t, -nu);
        w = w0 * nu_scale; // overwrites w
        w = (a == (T)0.0) ? w : w + a * (nu_scale - (T)1.0);
      }
      w += (w_read_std > (T)0.0) ? w_read_std * (T)curand_normal(&local_state) : (T)0.0;

      previous_weights[i] = w;
      weights[i] = w;
    }

  } // stride loop

  if (stoch_if && tid < size) {
    random_states[tid] = local_state;
  }
}

// ctor
template <typename T>
WeightDrifterCuda<T>::WeightDrifterCuda(CudaContextPtr context, int size)
    : context_(context), size_(size), max_size_(size), active_(false), current_t_(0.0) {}

template <typename T>
WeightDrifterCuda<T>::WeightDrifterCuda(
    CudaContextPtr context, const WeightDrifter<T> &wd, int x_size, int d_size)
    : WeightDrifterCuda(context, x_size * d_size) {
  populateFrom(wd, x_size, d_size);
}

// copy ctor
template <typename T> WeightDrifterCuda<T>::WeightDrifterCuda(const WeightDrifterCuda<T> &other) {

  context_ = other.context_;
  size_ = other.size_;
  active_ = other.active_;
  current_t_ = other.current_t_;
  max_size_ = other.max_size_;
  par_ = other.par_;

  if (active_) {
    dev_previous_weights_ = RPU::make_unique<CudaArray<T>>(*other.dev_previous_weights_);
    dev_w0_ = RPU::make_unique<CudaArray<T>>(*other.dev_w0_);
    dev_t_ = RPU::make_unique<CudaArray<T>>(*other.dev_t_);
  }
  if (other.dev_nu_) {
    dev_nu_ = RPU::make_unique<CudaArray<T>>(*other.dev_nu_);
  }

  context_->synchronize();
}

template <typename T>
void WeightDrifterCuda<T>::dumpExtra(RPU::state_t &extra, const std::string prefix) {
  RPU::state_t state;

  RPU::insert(state, "active", active_);
  RPU::insert(state, "current_t", current_t_);
  RPU::insert(state, "dev_previous_weights", dev_previous_weights_);
  RPU::insert(state, "dev_w0", dev_w0_);
  RPU::insert(state, "dev_t", dev_t_);
  RPU::insert(state, "dev_nu", dev_nu_);

  RPU::insertWithPrefix(extra, state, prefix);
}

template <typename T>
void WeightDrifterCuda<T>::loadExtra(
    const RPU::state_t &extra, const std::string prefix, bool strict) {

  auto state = RPU::selectWithPrefix(extra, prefix);

  RPU::load(context_, state, "dev_previous_weights", dev_previous_weights_, strict);
  RPU::load(context_, state, "dev_w0", dev_w0_, strict);
  RPU::load(context_, state, "dev_t", dev_t_, strict);
  RPU::load(context_, state, "dev_nu", dev_nu_, strict);
  RPU::load(state, "current_t", current_t_, strict);
  RPU::load(state, "active", active_, strict);
}

template <typename T>
void WeightDrifterCuda<T>::populateFrom(const WeightDrifter<T> &wd, int x_size, int d_size) {
  // only copies the parameter from nu. Other parameters are set when set to active.

  if (x_size * d_size != size_ || wd.getSize() != size_) {
    RPU_FATAL("Size mismatch!");
  }

  active_ = false;
  par_ = wd.getPar();
  dev_nu_ = nullptr;
  if (wd.getNu() != nullptr) {
    dev_nu_ = RPU::make_unique<CudaArray<T>>(context_, size_);
    context_->synchronize();
    dev_nu_->assignTranspose(wd.getNu(), d_size, x_size);
  }
  context_->synchronize();
}

template <typename T> void WeightDrifterCuda<T>::saturate(T *weights, param_t *dev_4params) {
  if (!active_) {
    RPU_FATAL("Apply should be called first!");
  }
  RPU::math::elemsat(context_, weights, size_, dev_4params);
  RPU::math::elemsat(context_, dev_previous_weights_->getData(), size_, dev_4params);
}

template <typename T> void WeightDrifterCuda<T>::initialize(const T *weights) {

  // only inits the volatile parameters, ie not nu_dtod

  current_t_ = (T)0.0;
  active_ = true;
  max_size_ = size_;

  dev_t_ = RPU::make_unique<CudaArray<T>>(context_, size_);
  dev_w0_ = RPU::make_unique<CudaArray<T>>(context_, size_);
  dev_previous_weights_ = RPU::make_unique<CudaArray<T>>(context_, size_);
  context_->synchronize();

  dev_t_->setConst((T)0.0);
  RPU::math::copy<T>(context_, size_, weights, 1, dev_w0_->getData(), 1);
  RPU::math::copy<T>(context_, size_, weights, 1, dev_previous_weights_->getData(), 1);

  int max_size = context_->getSMCount() *
                 (context_->maxThreadsPerBlock() / context_->getNThreads()) *
                 context_->getNThreads();
  max_size_ = MIN(max_size, size_);

  context_->synchronize();
}

template <typename T> void WeightDrifterCuda<T>::apply(T *weights, T time_since_last_call) {

  if (!active_) {
    initialize(weights);
  }

  if (!par_.isSimpleDrift() && dev_nu_ == nullptr) {
    RPU_FATAL("Weight drifter needs to be populated first!");
  }

  int nthreads = context_->getNThreads();
  int nblocks = context_->getNBlocks(MIN(max_size_, size_), nthreads);
  auto s = context_->getStream();

  current_t_ += time_since_last_call / par_.t0;
  T a = par_.g_offset * par_.wg_ratio + par_.w_offset;
  if ((T)fabsf(a) < par_.reset_tol) {
    a = (T)0.0;
  }

  kernelDriftWeights<T><<<nblocks, nthreads, 0, s>>>(
      size_, weights, dev_previous_weights_->getData(), dev_w0_->getData(),
      par_.isSimpleDrift() ? nullptr : dev_nu_->getData(), dev_t_->getData(), current_t_,
      par_.reset_tol, par_.nu, par_.nu_std, par_.nu_k, a, par_.logG0, par_.wg_ratio, par_.w_offset,
      par_.g_offset, par_.w_read_std,
      par_.usesRandom() ? context_->getRandomStates(nblocks * nthreads) : nullptr);
}

template class WeightDrifterCuda<float>;
#ifdef RPU_USE_DOUBLE
template class WeightDrifterCuda<double>;
#endif
#ifdef RPU_USE_FP16
template class WeightDrifterCuda<half_t>;
#endif

} // namespace RPU
