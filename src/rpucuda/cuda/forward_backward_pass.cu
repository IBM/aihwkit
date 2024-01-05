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

#include "io_manager.h"

#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <random>

#include "cuda_fp16_util.h"
#include "cuda_math_util.h"
#include "cuda_util.h"
#include "forward_backward_pass.h"
#include "io_iterator.h"

namespace RPU {
namespace {
/*********************************************************************************/
/* output add wnoise single batch*/
template <typename T>
__global__ void kernelOutputWeightNoise(
    T *output,
    const T *input,
    const int size,
    const T *x_norm_value,
    const T w_noise_std,
    curandState *random_states) {
  T noise_std = (*x_norm_value) * w_noise_std;
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  curandState local_state;
  int total_threads = blockDim.x * gridDim.x;
  int total_states = MIN(size, total_threads);

  if (tid < total_states) {
    local_state = random_states[tid];
  }
  for (int idx = tid; idx < size; idx += total_threads) {
    T value = input[idx];
    value += noise_std * (T)curand_normal(&local_state);

    output[idx] = value;
  }
  if (tid < total_states) {
    random_states[tid] = local_state;
  }
}

/* output add wnoise batch*/
template <typename T>
__global__ void kernelOutputWeightNoiseBatch(
    T *output,
    const T *input,
    const int size,
    const int m_batch,
    const bool trans, // true if m_batch first dimensions
    const T *noise_var_values,
    curandState *random_states) {
  const int total_size = size * m_batch;
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  T stoch_value;
  curandState local_state;
  int total_threads = blockDim.x * gridDim.x;
  int total_states = MIN(total_size, total_threads);

  if (tid < total_states) {
    local_state = random_states[tid];
  }

  for (int idx = tid; idx < total_size; idx += total_threads) {
    T value = input[idx];
    int bidx = trans ? (idx % m_batch) : (idx / size);
    T noise_var = noise_var_values[bidx];
    stoch_value = curand_normal(&local_state);
    value += stoch_value * sqrt(noise_var);
    output[idx] = value;
  }
  if (tid < total_states) {
    random_states[tid] = local_state;
  }
}

template <typename T>
__global__ void kernelElemSqrtAddNoiseBatch(
    T *output,
    const T *input,
    const T *noise_var_values,
    const int total_size,
    curandState *random_states) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  T stoch_value;
  curandState local_state;
  int total_threads = blockDim.x * gridDim.x;
  int total_states = MIN(total_size, total_threads);
  if (tid < total_states) {
    local_state = random_states[tid];
  }
  for (int idx = tid; idx < total_size; idx += total_threads) {
    T value = input[idx];
    T noise_var = noise_var_values[idx];
    stoch_value = curand_normal(&local_state);
    value += stoch_value * sqrt(noise_var);
    output[idx] = value;
  }
  if (tid < total_states) {
    random_states[tid] = local_state;
  }
}

// compute x_j*(1-(1-j/n)^2)
template <typename T>
__global__ void kernelInputPositionCoding(
    T *output,      // x' out
    const T *input, // x in
    const int size,
    const int m_batch,
    const bool trans) {
  const int total_size = size * m_batch;

  RPU_CUDA_1D_KERNEL_LOOP(idx, total_size) {
    T value = input[idx];
    int xidx = trans ? (idx / m_batch) : (idx % size);
    T z = (T)1.0 - (T)xidx / (T)size;
    value *= (T)1.0 - z * z;
    output[idx] = value;
  }
}

// computes c_i and y_i -= c_i*GEMM_i
// c_i = a_i*(a_i*(0.05*a_i - 0.2) + 0.5);
template <typename T>
__global__ void kernelElemMulCAdd(
    T *output, const T *input, const T *a_values, const T *pc_gemm_values, const int total_size) {

  const T K2 = 0.05;
  const T K1 = 0.2;
  const T K0 = 0.5;

  RPU_CUDA_1D_KERNEL_LOOP(idx, total_size) {
    T value = input[idx];

    T a_i = a_values[idx];
    T gemm_i = pc_gemm_values[idx];
    T c_i = a_i * (a_i * (K2 * a_i - K1) + K0);
    value -= c_i * gemm_i;
    output[idx] = value;
  }
};
} // namespace

/*******************************************************************************/

// copy construcutor
template <typename T>
ForwardBackwardPassIOManagedCuda<T>::ForwardBackwardPassIOManagedCuda(
    const ForwardBackwardPassIOManagedCuda<T> &other) {
  x_size_ = other.x_size_;
  d_size_ = other.d_size_;
  context_ = other.context_;
  fb_pars_ = other.fb_pars_;
}

// copy assignment
template <typename T>
ForwardBackwardPassIOManagedCuda<T> &
ForwardBackwardPassIOManagedCuda<T>::operator=(const ForwardBackwardPassIOManagedCuda<T> &other) {

  ForwardBackwardPassIOManagedCuda<T> tmp(other);
  swap(*this, tmp);
  context_->synchronize();
  return *this;
}

// move constructor
template <typename T>
ForwardBackwardPassIOManagedCuda<T>::ForwardBackwardPassIOManagedCuda(
    ForwardBackwardPassIOManagedCuda<T> &&other) {
  *this = std::move(other);
}

// move assignment
template <typename T>
ForwardBackwardPassIOManagedCuda<T> &
ForwardBackwardPassIOManagedCuda<T>::operator=(ForwardBackwardPassIOManagedCuda<T> &&other) {

  x_size_ = other.x_size_;
  d_size_ = other.d_size_;
  context_ = other.context_;
  other.context_ = nullptr;
  fb_pars_ = std::move(other.fb_pars_);
  return *this;
}

template <typename T>
void ForwardBackwardPassIOManagedCuda<T>::populateFrom(const FBParameter<T> &fb_pars_host) {

  auto populate = [this](MVParameterCuda<T> &mv_pars, const MVParameter<T> &mv_pars_host) -> void {
    if (mv_pars_host.out_noise_values.size() > (size_t)0) {
      mv_pars.out_noise_values = CudaArray<T>(this->context_, mv_pars_host.out_noise_values);
    }

    if (mv_pars_host.v_offset.size() > (size_t)0) {
      mv_pars.v_offset = CudaArray<T>(this->context_, mv_pars_host.v_offset);
    }
    if (mv_pars_host.out_nonlinearity.size() > (size_t)0) {
      mv_pars.out_nonlinearity = CudaArray<T>(this->context_, mv_pars_host.out_nonlinearity);
    }
    mv_pars.out_nonlinearity_factor = mv_pars_host.out_nonlinearity_factor;

    if (mv_pars_host.w_asymmetry.size() > (size_t)0) {
      mv_pars.w_asymmetry = CudaArray<T>(this->context_, mv_pars_host.w_asymmetry.size());
      mv_pars.w_asymmetry.assignTranspose(
          mv_pars_host.w_asymmetry.data(), this->d_size_, this->x_size_);
    }
    this->context_->synchronize();
  };
  populate(fb_pars_.fwd, fb_pars_host.fwd);
  populate(fb_pars_.bwd, fb_pars_host.bwd);
}

template <typename T>
void ForwardBackwardPassIOManagedCuda<T>::dumpExtra(RPU::state_t &extra, const std::string prefix) {
  RPU::state_t state;

  RPU::insert(state, "fwd.v_offset", fb_pars_.fwd.v_offset);
  RPU::insert(state, "fwd.w_asymmetry", fb_pars_.fwd.w_asymmetry);
  RPU::insert(state, "fwd.out_nonlinearity", fb_pars_.fwd.out_nonlinearity);
  RPU::insert(state, "fwd.out_nonlinearity_factor", fb_pars_.fwd.out_nonlinearity_factor);
  RPU::insert(state, "fwd.out_noise_values", fb_pars_.fwd.out_noise_values);

  RPU::insert(state, "bwd.v_offset", fb_pars_.bwd.v_offset);
  RPU::insert(state, "bwd.w_asymmetry", fb_pars_.bwd.w_asymmetry);
  RPU::insert(state, "bwd.out_nonlinearity", fb_pars_.bwd.out_nonlinearity);
  RPU::insert(state, "bwd.out_nonlinearity_factor", fb_pars_.bwd.out_nonlinearity_factor);
  RPU::insert(state, "bwd.out_noise_values", fb_pars_.bwd.out_noise_values);

  RPU::insertWithPrefix(extra, state, prefix);
}

template <typename T>
void ForwardBackwardPassIOManagedCuda<T>::loadExtra(
    const RPU::state_t &extra, const std::string prefix, bool strict) {

  using V = std::vector<T>;
  auto state = RPU::selectWithPrefix(extra, prefix);
  V tmp;

  // forward
  RPU::load(this->context_, state, "fwd.v_offset", fb_pars_.fwd.v_offset, strict);
  RPU::load(this->context_, state, "fwd.w_asymmetry", fb_pars_.fwd.w_asymmetry, strict);
  RPU::load(this->context_, state, "fwd.out_nonlinearity", fb_pars_.fwd.out_nonlinearity, strict);
  RPU::load(state, "fwd.out_nonlinearity_factor", fb_pars_.fwd.out_nonlinearity_factor, strict);
  RPU::load(this->context_, state, "fwd.out_noise_values", fb_pars_.fwd.out_noise_values, strict);

  // backward
  RPU::load(this->context_, state, "bwd.v_offset", fb_pars_.bwd.v_offset, strict);
  RPU::load(this->context_, state, "bwd.w_asymmetry", fb_pars_.bwd.w_asymmetry, strict);
  RPU::load(this->context_, state, "bwd.out_nonlinearity", fb_pars_.bwd.out_nonlinearity, strict);
  RPU::load(state, "bwd.out_nonlinearity_factor", fb_pars_.bwd.out_nonlinearity_factor, strict);
  RPU::load(this->context_, state, "bwd.out_noise_values", fb_pars_.bwd.out_noise_values, strict);
}

template <typename T>
void ForwardBackwardPassIOManagedCuda<T>::applyOutputWeightNoise(
    InputOutputManager<T> &iom, const bool out_trans, const bool tranposed) {

  auto io = iom.getIO();
  if (io.w_noise <= (T)0.0) {
    return;
  }

  auto m_batch = iom.getMBatch();
  auto in_trans = iom.getInTrans();
  auto context = iom.getContext();
  auto in_size = iom.getInSize();
  auto out_size = iom.getOutSize();
  auto nblocks_om = iom.getOutBlocks();
  auto nblocks_om_batch = iom.getOutBlocksBatch(m_batch);
  auto nthreads = iom.getNThreads();

  T *wnoise_buffer = context->template getSharedBuffer<T>(RPU_BUFFER_TEMP_1, m_batch);

  if (!dev_ones_ || dev_ones_->getSize() < in_size) {
    dev_ones_ = RPU::make_unique<CudaArray<T>>(context, in_size);
    dev_ones_->setConst((T)1.0);
  }

  // In buffer is in_size_ x m_batch or m_batch x in_size_ (if in_trans)
  const T *in_temp = iom.getInBuffer();
  // out buffer is out_size_ x m_batch or m_batch x out_size_ (if out_trans)
  T *out_temp = iom.getOutBuffer();

  if (m_batch == 1 && sizeof(T) >= 4) {

    // directly calculate vector norm, (over-)use same buffer
    RPU::math::nrm2(context, in_size, in_temp, 1, wnoise_buffer);

    kernelOutputWeightNoise<<<nblocks_om, nthreads, 0, context->getStream()>>>(
        out_temp,
        out_temp, // in-place
        out_size,
        wnoise_buffer, // this is actually only the norm value, see above
        io.w_noise, context_->getRandomStates(MIN(nblocks_om * nthreads, out_size)));

  } else {
    T *in_buffer = context->template getSharedBuffer<T>(RPU_BUFFER_TEMP_2, m_batch * in_size);

    RPU::math::elempow2(context, in_buffer, in_size * m_batch, in_temp);

    // we just use gemm for the reduction. That way we can also scale by the noise level
    RPU::math::gemm<T>(
        context, false, in_trans, 1,
        m_batch,                 // M
        in_size,                 // K
        io.w_noise * io.w_noise, // weight noise variance scale
        dev_ones_->getDataConst(), 1, in_buffer, (in_trans) ? m_batch : in_size, (T)0.0,
        wnoise_buffer, 1);

    context->template releaseSharedBuffer<T>(RPU_BUFFER_TEMP_2);

    // now we have the variance of the wnoise for each batch in the wnoise_buffer

    // generate and add the weight noise to the applied output
    kernelOutputWeightNoiseBatch<<<nblocks_om_batch, nthreads, 0, context->getStream()>>>(
        out_temp,
        out_temp, // in-place
        out_size, m_batch, out_trans, wnoise_buffer,
        context->getRandomStates(MIN(nblocks_om_batch * nthreads, out_size * m_batch)));
  }
  context->template releaseSharedBuffer<T>(RPU_BUFFER_TEMP_1);
}

template <typename T>
void ForwardBackwardPassIOManagedCuda<T>::applyOutputPCMReadNoise(
    const T *dev_weights, InputOutputManager<T> &iom, const bool out_trans, const bool transposed) {

  auto io = iom.getIO();
  if (io.w_noise <= (T)0.0) {
    return;
  }

  auto m_batch = iom.getMBatch();
  auto in_trans = iom.getInTrans();
  auto context = iom.getContext();
  auto in_size = iom.getInSize();
  auto out_size = iom.getOutSize();
  auto nblocks_om = iom.getOutBlocks();
  auto nblocks_om_batch = iom.getOutBlocksBatch(m_batch);
  auto nthreads = iom.getNThreads();

  // compute the overall noise contributions is sigma_i propto sqrt(sum_j
  // |Wij|*xj^2) eventually we might want to use cutlass for
  // this, but let's wait once we have more time since it seems
  // more complicated than thought

  // In buffer is in_size_ x m_batch or m_batch x in_size_ (if in_trans)
  const T *in_temp = iom.getInBuffer();
  // Out buffer is out_size_ x m_batch or m_batch x out_size_ (if out_trans)
  T *out_temp = iom.getOutBuffer();

  // unfortunately needs a lot of extra buffer memory (CUTLASS will help)
  T *batch_buffer = context->template getSharedBuffer<T>(RPU_BUFFER_TEMP_1, m_batch * out_size);
  T *weight_buffer = context->template getSharedBuffer<T>(RPU_BUFFER_TEMP_3, in_size * out_size);
  T *in_buffer = context->template getSharedBuffer<T>(RPU_BUFFER_TEMP_2, in_size * m_batch);

  // compute thr abs weights current
  RPU::math::elemabs(context, weight_buffer, dev_weights, out_size * in_size);

  // in_temp can be overwritten since it use assumed to have been used already
  RPU::math::elempow2(context, in_buffer, in_size * m_batch, in_temp);

  // we just use gemm also scale by the noise level
  // assumed_wmax is set to 1. if otherwise, need to be set from outside
  T noise_level = io.w_noise;
  this->gemm(
      context, weight_buffer, in_buffer, in_size, in_trans, batch_buffer, out_size, out_trans,
      m_batch, noise_level * noise_level, (T)0.0, transposed);

  // need to do the sqrt still and add to
  // generate and add the weight noise to the applied output
  kernelElemSqrtAddNoiseBatch<<<nblocks_om_batch, nthreads, 0, context->getStream()>>>(
      out_temp,
      out_temp, // in-place
      batch_buffer, out_size * m_batch,
      context_->getRandomStates(MIN(nblocks_om_batch * nthreads, out_size * m_batch)));

  context->template releaseSharedBuffer<T>(RPU_BUFFER_TEMP_2);
  context->template releaseSharedBuffer<T>(RPU_BUFFER_TEMP_3);
  context->template releaseSharedBuffer<T>(RPU_BUFFER_TEMP_1);
}

/* output dtod noise*/
template <typename T>
__global__ void kernelOutputNoiseOtoOBatch(
    T *output,
    const int out_size,
    const bool trans, // true if m_batch first dimensions
    const int m_batch,
    const T *out_noise_values,
    curandState *random_states) {
  const int tid = blockDim.x * blockIdx.x + threadIdx.x;
  T stoch_value;
  curandState local_state;
  const int total_threads = blockDim.x * gridDim.x;
  const int total_size = m_batch * out_size;
  int total_states = MIN(total_size, total_threads);
  if (tid < total_states) {
    local_state = random_states[tid];
  }
  for (int idx = tid; idx < total_size; idx += total_threads) {
    T value = output[idx];
    const int out_idx = trans ? (idx / m_batch) : (idx % out_size);
    const T noise_std = out_noise_values[out_idx];
    stoch_value = curand_normal(&local_state);
    value += stoch_value * noise_std;
    output[idx] = value;
  }
  if (tid < total_states) {
    random_states[tid] = local_state;
  }
}

template <typename T>
void ForwardBackwardPassIOManagedCuda<T>::applyOutputNoiseOtoO(
    InputOutputManager<T> &iom,
    const MVParameterCuda<T> &mv_pars,
    const bool out_trans,
    const bool tranposed) {

  auto io = iom.getIO();
  auto m_batch = iom.getMBatch();
  auto context = iom.getContext();
  auto out_size = iom.getOutSize();
  auto nblocks_om_batch = iom.getOutBlocksBatch(m_batch);
  auto nthreads = iom.getNThreads();

  // out buffer is out_size_ x m_batch or m_batch x out_size_ (if out_trans)
  T *out_temp = iom.getOutBuffer();

  kernelOutputNoiseOtoOBatch<<<nblocks_om_batch, nthreads, 0, context->getStream()>>>(
      out_temp, out_size, out_trans, m_batch, mv_pars.out_noise_values.getDataConst(),
      context_->getRandomStates(MIN(nblocks_om_batch * nthreads, out_size * m_batch)));
}

/* output non-linearity*/
template <typename T>
__global__ void kernelOutputNonLinearityBatch(
    T *output,
    const int size,
    const bool trans, // true if m_batch first dimensions
    const int m_batch,
    const T nonlinearity_factor,
    const T *nonlinearity) {
  const int total_size = size * m_batch;
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  int total_threads = blockDim.x * gridDim.x;
  T nlf = nonlinearity_factor;

  for (int idx = tid; idx < total_size; idx += total_threads) {
    T value = output[idx];
    int out_idx = trans ? (idx / m_batch) : (idx % size);
    value = nlf * value / ((T)1.0 + fabs(nonlinearity[out_idx] * value));
    output[idx] = value;
  }
}

template <typename T>
void ForwardBackwardPassIOManagedCuda<T>::applyOutputNonLinearity(
    InputOutputManager<T> &iom,
    const MVParameterCuda<T> &mv_pars,
    const bool out_trans,
    const bool tranposed) {

  auto io = iom.getIO();
  if (!io.hasNLCalibration()) {
    return;
  }

  auto m_batch = iom.getMBatch();
  auto context = iom.getContext();
  auto out_size = iom.getOutSize();
  auto nblocks_om_batch = iom.getOutBlocksBatch(m_batch);
  auto nthreads = iom.getNThreads();

  // out buffer is out_size_ x m_batch or m_batch x out_size_ (if out_trans)
  T *out_temp = iom.getOutBuffer();

  kernelOutputNonLinearityBatch<<<nblocks_om_batch, nthreads, 0, context->getStream()>>>(
      out_temp, out_size, out_trans, m_batch, mv_pars.out_nonlinearity_factor,
      mv_pars.out_nonlinearity.getDataConst());
}

template <typename T>
void ForwardBackwardPassIOManagedCuda<T>::applyIrDrop(
    const T *dev_weights, InputOutputManager<T> &iom, const bool out_trans, const bool transposed) {
  auto io = iom.getIO();
  if (io.ir_drop <= (T)0.0) {
    return;
  }

  auto m_batch = iom.getMBatch();
  auto in_trans = iom.getInTrans();
  auto context = iom.getContext();
  auto in_size = iom.getInSize();
  auto out_size = iom.getOutSize();
  auto nblocks_om_batch = iom.getOutBlocksBatch(m_batch);
  auto nthreads = iom.getNThreads();

  // computes for each output (row):
  //
  // a_i = sum_j(|w_ij|*|x_j|)*n/Gw*gmax
  // c_i = a_i*(a_i*(0.05*a_i - 0.2) + 0.5);
  // x'_j = x_j * (1 - (1-j/n)^2)
  // y_i = y_i_ideal - c_i*sum_j(w_ij * x'_j)

  // In buffer is in_size_ x m_batch or m_batch x in_size_ (if in_trans)
  const T *in_temp = iom.getInBuffer();
  // Out buffer is out_size_ x m_batch or m_batch x out_size_ (if out_trans)
  T *out_temp = iom.getOutBuffer();

  // unfortunately needs a lot of extra buffer memory (CUTLASS will help)
  T *batch_buffer =
      context->template getSharedBuffer<T>(RPU_BUFFER_TEMP_1, m_batch * max(in_size, out_size));
  T *weight_buffer = context->template getSharedBuffer<T>(RPU_BUFFER_TEMP_3, in_size * out_size);
  T *a_buffer = context->template getSharedBuffer<T>(RPU_BUFFER_TEMP_2, m_batch * out_size);
  T *b_buffer = context->template getSharedBuffer<T>(RPU_BUFFER_TEMP_4, m_batch * out_size);

  // compute thr abs weights current
  RPU::math::elemabs(context_, weight_buffer, dev_weights, in_size * out_size);

  RPU::math::elemabs(context_, batch_buffer, in_temp, in_size * m_batch);

  // compute the a_i
  this->gemm(
      context, weight_buffer, batch_buffer, in_size, in_trans, a_buffer, out_size, out_trans,
      m_batch, (T)in_size / io.ir_drop_Gw_div_gmax, (T)0.0, transposed);

  // compute x_j*(1-(1-j/n)^2)
  kernelInputPositionCoding<<<nblocks_om_batch, nthreads, 0, context_->getStream()>>>(
      batch_buffer, in_temp, in_size, m_batch, in_trans);

  // compute the position filtered GEMM
  this->gemm(
      context, dev_weights, batch_buffer, in_size, in_trans, b_buffer, out_size, out_trans, m_batch,
      io.ir_drop, (T)0.0, transposed);

  // computes c_i and y_i -= c_i*GEMM_i
  kernelElemMulCAdd<<<nblocks_om_batch, nthreads, 0, context->getStream()>>>(
      out_temp,
      out_temp,          // in-place
      a_buffer,          // this is the a_i
      b_buffer,          // this is GEMM_i
      out_size * m_batch // all have same shape
  );

  context->template releaseSharedBuffer<T>(RPU_BUFFER_TEMP_1);
  context->template releaseSharedBuffer<T>(RPU_BUFFER_TEMP_2);
  context->template releaseSharedBuffer<T>(RPU_BUFFER_TEMP_3);
  context->template releaseSharedBuffer<T>(RPU_BUFFER_TEMP_4);
}

/* output v offset*/
template <typename T>
__global__ void kernelOutputVoffsetBatch(
    T *output,
    const int size,
    const bool trans, // true if m_batch first dimensions
    const int m_batch,
    const T r_series,
    const T *v_offset,
    const T *y_ref_values,
    const T rs_max_total) {
  const int total_size = size * m_batch;
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  int total_threads = blockDim.x * gridDim.x;
  T rs = r_series;

  for (int idx = tid; idx < total_size; idx += total_threads) {
    T y = output[idx];
    int out_idx = trans ? (idx / m_batch) : (idx % size);
    int b_idx = trans ? (idx % m_batch) : (idx / size);
    T v_offs = v_offset[out_idx];
    T y_ref = (y_ref_values == nullptr) ? (T)0.0 : y_ref_values[b_idx];
    T y_pos = y + y_ref;
    if (rs > (T)0.0) {
      y_pos /= ((T)1.0 + MIN(rs * fabs(y_pos), rs_max_total));
      y_ref /= ((T)1.0 + MIN(rs * fabs(y_ref), rs_max_total));
    }
    output[idx] = ((T)1.0 - v_offs) * y_pos - ((T)1.0 + v_offs) * y_ref;
  }
}

template <typename T>
void ForwardBackwardPassIOManagedCuda<T>::applyVoltageOffsets(
    const T *dev_weights,
    InputOutputManager<T> &iom,
    const MVParameterCuda<T> &mv_pars,
    const bool out_trans,
    const bool transposed) {
  auto io = iom.getIO();

  if (!io.hasVoltageOffsets()) {
    return;
  }

  auto m_batch = iom.getMBatch();
  auto in_trans = iom.getInTrans();
  auto context = iom.getContext();
  auto in_size = iom.getInSize();
  auto out_size = iom.getOutSize();
  auto nblocks_om_batch = iom.getOutBlocksBatch(m_batch);
  auto nthreads = iom.getNThreads();

  // out buffer is out_size_ x m_batch or m_batch x out_size_ (if out_trans)
  const T *in_temp = iom.getInBuffer();
  T *out_temp = iom.getOutBuffer();

  T *y_ref = nullptr;
  if (io.v_offset_w_min != (T)0.0) {
    y_ref = context->template getSharedBuffer<T>(RPU_BUFFER_TEMP_2, m_batch);

    // reduce x
    if (!dev_ones_ || dev_ones_->getSize() < in_size) {
      dev_ones_ = RPU::make_unique<CudaArray<T>>(context, in_size);
      dev_ones_->setConst((T)1.0);
    }

    // we just use gemm for the reduction: in_temp should be all
    // positive (negative) in case of two pass
    RPU::math::gemv<T>(
        context, !in_trans, in_trans ? m_batch : in_size, in_trans ? in_size : m_batch,
        (T)-io.v_offset_w_min, in_temp, in_trans ? m_batch : in_size, dev_ones_->getDataConst(), 1,
        (T)0.0, y_ref, 1);
  }

  kernelOutputVoffsetBatch<<<nblocks_om_batch, nthreads, 0, context->getStream()>>>(
      out_temp, out_size, out_trans, m_batch, MAX(io.r_series, (T)0.0),
      mv_pars.v_offset.getDataConst(), y_ref, io.r_series_max_total);

  if (y_ref != nullptr) {
    context->template releaseSharedBuffer<T>(RPU_BUFFER_TEMP_2);
  }
}

template <typename T>
void ForwardBackwardPassIOManagedCuda<T>::computeAnalogMVSinglePass(
    T *dev_weights,
    InputOutputManager<T> &iom,
    const MVParameterCuda<T> &mv_pars,
    const bool out_trans,
    const bool transposed) {

  // input is prepared. Now do FP MV
  this->gemm(
      iom.getContext(), dev_weights, iom.getInBuffer(), iom.getInSize(), iom.getInTrans(),
      iom.getOutBuffer(), iom.getOutSize(), out_trans, iom.getMBatch(), (T)1.0, (T)0.0, transposed);

  // Add the non-idealities on the output
  auto io = iom.getIO();

  // IR drop
  applyIrDrop(dev_weights, iom, out_trans, transposed);

  // voltage offsets
  if (io.hasVoltageOffsets()) {
    applyVoltageOffsets(dev_weights, iom, mv_pars, out_trans, transposed);
  }

  // output noise
  switch (io.w_noise_type) {
  case OutputWeightNoiseType::AdditiveConstant: {
    // WARNING: overwrites inBuffer
    applyOutputWeightNoise(iom, out_trans, transposed);
    break;
  }
  case OutputWeightNoiseType::PCMRead: {
    // WARNING: overwrites inBuffer
    applyOutputPCMReadNoise(dev_weights, iom, out_trans, transposed);

    break;
  }
  case OutputWeightNoiseType::None: {
    break;
  }
  default:
    RPU_FATAL("Output noise type  not implemented.");
  }
};

template <typename T>
template <typename OutputIteratorT>
bool ForwardBackwardPassIOManagedCuda<T>::computeAnalogMV(
    OutputIteratorT out_values,
    const bool out_trans,
    T *dev_weights,
    InputOutputManager<T> &iom,
    const MVParameterCuda<T> &mv_pars,
    const bool transposed) {

  auto mv_type = iom.getIO().mv_type;
  if (iom.getIO().isPerfect()) {
    mv_type = AnalogMVType::Ideal; // just to be safe
  }

  switch (mv_type) {
  case AnalogMVType::Ideal: {
    this->gemm(
        iom.getContext(), dev_weights, iom.getInBuffer(), iom.getInSize(), iom.getInTrans(),
        iom.getOutBuffer(), iom.getOutSize(), out_trans, iom.getMBatch(), iom.getOutScale(), (T)0.0,
        transposed);
    return finalizeOutput(out_values, iom, mv_pars, out_trans, transposed);
  }
  case AnalogMVType::OnePass: {
    // this is the standard one pass MV
    computeAnalogMVSinglePass(dev_weights, iom, mv_pars, out_trans, transposed);
    return finalizeOutput(out_values, iom, mv_pars, out_trans, transposed);
  }

  case AnalogMVType::PosNegSeparate: {

    int m_batch = iom.getMBatch();
    int in_size = iom.getInSize();
    int out_size = iom.getOutSize();
    int total_in_size = m_batch * in_size;
    int total_out_size = m_batch * out_size;
    auto io = iom.getIO();

    T *pos_neg_buffer_in = context_->template getSharedBuffer<T>(
        RPU_BUFFER_POS_NEG_IN, MAX(total_in_size, total_out_size));
    T *pos_neg_buffer_out = context_->template getSharedBuffer<T>(
        RPU_BUFFER_POS_NEG_OUT, MAX(total_in_size, total_out_size));

    T *neg_weights = dev_weights;
    const T *in_temp = iom.getInBuffer();
    T *out_temp = iom.getOutBuffer();
    iom.setInBuffer(pos_neg_buffer_in);

    // neg part
    RPU::math::elemmin(this->context_, pos_neg_buffer_in, total_in_size, (T)0.0, in_temp);

    int w_size = this->d_size_ * this->x_size_;
    if (io.w_read_asymmetry_dtod > (T)0.0 && (in_size * out_size == w_size)) {
      neg_weights = context_->template getSharedBuffer<T>(RPU_BUFFER_WEIGHT, w_size);
      RPU::math::elemmul(
          context_, neg_weights, w_size, dev_weights, mv_pars.w_asymmetry.getDataConst());
    }
    bool bound_success = false;

    iom.setOutBuffer(pos_neg_buffer_out);
    computeAnalogMVSinglePass(neg_weights, iom, mv_pars, out_trans, transposed);

    // pos part
    RPU::math::elemmax(this->context_, pos_neg_buffer_in, total_in_size, (T)0.0, in_temp);

    // second pass for positive
    iom.setOutBuffer(out_temp);
    computeAnalogMVSinglePass(dev_weights, iom, mv_pars, out_trans, transposed);

    // add output buffers and then IOM
    RPU::math::elemweightedsum(
        context_, out_temp, total_out_size, pos_neg_buffer_out, (T)1.0, out_temp, (T)1.0);
    bound_success = finalizeOutput(out_values, iom, mv_pars, out_trans, transposed);

    context_->template releaseSharedBuffer<T>(RPU_BUFFER_POS_NEG_IN);
    context_->template releaseSharedBuffer<T>(RPU_BUFFER_POS_NEG_OUT);

    if (io.w_read_asymmetry_dtod > (T)0.0) {
      context_->template releaseSharedBuffer<T>(RPU_BUFFER_WEIGHT);
    }

    return bound_success;
  }
  case AnalogMVType::PosNegSeparateDigitalSum: {

    int m_batch = iom.getMBatch();
    int in_size = iom.getInSize();
    int out_size = iom.getOutSize();
    int total_in_size = m_batch * in_size;
    int total_out_size = m_batch * out_size;
    auto io = iom.getIO();
    T *pos_neg_buffer_in = context_->template getSharedBuffer<T>(
        RPU_BUFFER_POS_NEG_IN, MAX(total_in_size, total_out_size));
    T *pos_neg_buffer_out =
        context_->template getSharedBuffer<T>(RPU_BUFFER_POS_NEG_OUT, total_out_size);

    T *neg_weights = dev_weights;
    const T *in_temp = iom.getInBuffer();
    T *out_temp = iom.getOutBuffer();

    iom.setInBuffer(pos_neg_buffer_in);
    // neg part
    RPU::math::elemmin(this->context_, pos_neg_buffer_in, total_in_size, (T)0.0, in_temp);

    int w_size = this->d_size_ * this->x_size_;
    if (io.w_read_asymmetry_dtod > (T)0.0 && (in_size * out_size == w_size)) {
      neg_weights = context_->template getSharedBuffer<T>(RPU_BUFFER_WEIGHT, w_size);
      RPU::math::elemmul(
          context_, neg_weights, w_size, dev_weights, mv_pars.w_asymmetry.getDataConst());
    }
    bool bound_success = false;

    computeAnalogMVSinglePass(neg_weights, iom, mv_pars, out_trans, transposed);

    bound_success = finalizeOutput(pos_neg_buffer_out, iom, mv_pars, out_trans, transposed);

    // second pass for positive
    RPU::math::elemmax(context_, pos_neg_buffer_in, total_in_size, (T)0.0, in_temp);

    computeAnalogMVSinglePass(dev_weights, iom, mv_pars, out_trans, transposed);

    bound_success =
        finalizeOutput(pos_neg_buffer_in, iom, mv_pars, out_trans, transposed) && bound_success;

    //   // add output buffers after IOM
    RPU::math::addWithIterator(
        context_, out_values, pos_neg_buffer_out, pos_neg_buffer_in, total_out_size);

    context_->template releaseSharedBuffer<T>(RPU_BUFFER_POS_NEG_IN);
    context_->template releaseSharedBuffer<T>(RPU_BUFFER_POS_NEG_OUT);

    if (io.w_read_asymmetry_dtod > (T)0.0) {
      context_->template releaseSharedBuffer<T>(RPU_BUFFER_WEIGHT);
    }

    return bound_success;
  }

  default:
    RPU_FATAL("AnalogMVType Unknown.");
  }
};

#define OARG(NUM_T)                                                                                \
  , const bool, NUM_T *, InputOutputManager<NUM_T> &, const MVParameterCuda<NUM_T> &, const bool
template class ForwardBackwardPassIOManagedCuda<float>;
RPU_GEN_OITER_TEMPLATES(
    float, bool, ForwardBackwardPassIOManagedCuda<float>::computeAnalogMV, OARG(float));

#ifdef RPU_USE_DOUBLE
template class ForwardBackwardPassIOManagedCuda<double>;
RPU_GEN_OITER_TEMPLATES(
    double, bool, ForwardBackwardPassIOManagedCuda<double>::computeAnalogMV, OARG(double));
#endif
#ifdef RPU_USE_FP16
template class ForwardBackwardPassIOManagedCuda<half_t>;
RPU_GEN_OITER_TEMPLATES(
    half_t, bool, ForwardBackwardPassIOManagedCuda<half_t>::computeAnalogMV, OARG(half_t));
#endif

#undef OARG

} // namespace RPU
