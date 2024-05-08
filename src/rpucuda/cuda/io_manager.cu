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

#define STOCH_DEFINITIONS(STOCHIF, SIZE)                                                           \
  int total_threads = blockDim.x * gridDim.x;                                                      \
  int total_states = MIN(total_threads, SIZE);                                                     \
  int tid = blockDim.x * blockIdx.x + threadIdx.x;                                                 \
  T bu = bound_upper;                                                                              \
  T bl = bound_lower;                                                                              \
  T stoch_value;                                                                                   \
  curandState local_state;                                                                         \
  if (STOCHIF && tid < total_states) {                                                             \
    local_state = random_states[tid];                                                              \
  }

#define STOCH_FINALIZE(STOCHIF)                                                                    \
  if (STOCHIF && tid < total_states) {                                                             \
    random_states[tid] = local_state;                                                              \
  }

#define DISCRETIZE_VALUE_STOCH                                                                     \
  if (res > (T)0.0) {                                                                              \
    value /= res;                                                                                  \
    if (sr) {                                                                                      \
      stoch_value = curand_uniform(&local_state);                                                  \
      value += stoch_value - (T)0.5;                                                               \
    }                                                                                              \
    value = res * RPU_ROUNDFUN(value);                                                             \
  }

#define ADD_NOISE                                                                                  \
  if (std > (T)0.0) {                                                                              \
    value += std * (T)curand_normal(&local_state);                                                 \
  }

#define APPLY_ASYMMETRY                                                                            \
  if (asymmetry != (T)0.0) {                                                                       \
    value = value < (T)0.0 ? value * ((T)1.0 - asymmetry) : value;                                 \
  }

#define DISCRETIZE_VALUE                                                                           \
  if (res > (T)0.0) {                                                                              \
    value /= res;                                                                                  \
    value = res * RPU_ROUNDFUN(value);                                                             \
  }

#define BOUND_CHECK                                                                                \
  {                                                                                                \
    value = (value > bu) ? bu : value;                                                             \
    value = (value < bl) ? bl : value;                                                             \
  }

#define STRIDE_LOOP(SIZE, OUTVALUE, BODY)                                                          \
  T value;                                                                                         \
  for (int i_stride = 0; i_stride < SIZE; i_stride += total_threads) {                             \
    int idx = i_stride + tid;                                                                      \
                                                                                                   \
    if (idx < SIZE) {                                                                              \
      value = input[idx];                                                                          \
      BODY;                                                                                        \
      output[idx] = OUTVALUE;                                                                      \
    }                                                                                              \
  }

// if LOCAL_NM_SCALE is zero no need to scale up, since value is zero
#define APPLY_INPUT_NOISE_MANAGMENT(LOCAL_NM_SCALE)                                                \
  { value = LOCAL_NM_SCALE > (T)0.0 ? value / LOCAL_NM_SCALE : value; }

#define APPLY_OUTPUT_NOISE_MANAGMENT(LOCAL_NM_SCALE)                                               \
  { value = (LOCAL_NM_SCALE > (T)0.0) ? value * LOCAL_NM_SCALE : (T)0.0; }

namespace RPU {

/*********************************************************************************/
/*Input Management: shortcut for no SR (no noise needed)*/
template <typename T, typename InputIteratorT, bool noise_management>
__global__ void kernelInputManagement_noSR(
    T *output,
    InputIteratorT input,
    const int size,
    const T *nm_scale_value,
    const T bound_lower,
    const T bound_upper,
    const T resolution,
    const T asymmetry) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  int total_threads = blockDim.x * gridDim.x;

  T bu = bound_upper;
  T bl = bound_lower;
  T res = resolution;

  T local_nm_scale = (T)0.0;

  if (noise_management) {
    local_nm_scale = *nm_scale_value;
  }

  STRIDE_LOOP(
      size, value,

      if (noise_management) { APPLY_INPUT_NOISE_MANAGMENT(local_nm_scale); }

      DISCRETIZE_VALUE;

      BOUND_CHECK;

      APPLY_ASYMMETRY;

  );
}

/*********************************************************************************/
/*Input Management Single Batch */
template <typename T, typename InputIteratorT, bool noise_management>
__global__ void kernelInputManagement(
    T *output,
    InputIteratorT input,
    const int size_in,
    const T *nm_scale_value,
    const T bound_lower,
    const T bound_upper,
    const T resolution,
    const bool sto_round,
    const T inp_noise_std,
    const T asymmetry,
    curandState *random_states) {
  int size = size_in;
  T res = resolution;
  bool sr = sto_round && (res > (T)0.0);
  T std = inp_noise_std;
  bool stoch_if = sr || std > (T)0.0;

  STOCH_DEFINITIONS(stoch_if, size);

  T local_nm_scale = (T)0.0;
  if (noise_management) {
    local_nm_scale = *nm_scale_value;
  }

  STRIDE_LOOP(
      size, value,

      if (noise_management) { APPLY_INPUT_NOISE_MANAGMENT(local_nm_scale); }

      DISCRETIZE_VALUE_STOCH;

      ADD_NOISE;

      BOUND_CHECK;

      APPLY_ASYMMETRY;

  );

  STOCH_FINALIZE(stoch_if);
}

/*********************************************************************************/
/*Input Management Batch*/
template <typename T, typename InputIteratorT, bool noise_management>
__global__ void kernelInputManagementBatch(
    T *output,
    InputIteratorT input,
    const int size_in,
    const int m_batch_in,
    const bool trans_in, // true if m_batch first dimensions
    const T *nm_scale_values,
    const T bound_lower,
    const T bound_upper,
    const T resolution,
    const bool sto_round,
    const T inp_noise_std,
    const T asymmetry,
    curandState *random_states) {
  int size = size_in;

  bool trans = trans_in;
  int m_batch = m_batch_in;
  int total_size = size * m_batch;

  T res = resolution;
  bool sr = sto_round && (res > (T)0.0);
  T std = inp_noise_std;
  bool stoch_if = sr || std > (T)0.0;

  STOCH_DEFINITIONS(stoch_if, total_size);

  STRIDE_LOOP(
      total_size, value,

      if (noise_management) {
        int bidx = (trans) ? (idx % m_batch) : (idx / size);
        T local_nm_scale = nm_scale_values[bidx];

        APPLY_INPUT_NOISE_MANAGMENT(local_nm_scale);
      }

      DISCRETIZE_VALUE_STOCH;

      ADD_NOISE;

      BOUND_CHECK;

      APPLY_ASYMMETRY;

  );

  STOCH_FINALIZE(stoch_if);
}

/*********************************************************************************/
/*Input Management with bound check. Single Batch */
template <typename T, typename InputIteratorT, bool noise_management>
__global__ void kernelInputBoundManagement(
    T *output,
    InputIteratorT input,
    const int size_in,
    T *nm_scale_value,
    T *scale_value_out,
    const T bm_scale,
    const T bound_lower,
    const T bound_upper,
    const T resolution,
    const bool sto_round,
    const T inp_noise_std,
    const T asymmetry,
    curandState *random_states,
    int *bound_exceeded) {
  int size = size_in;
  T res = resolution;
  bool sr = sto_round && (res > (T)0.0);
  T bms = bm_scale;
  T std = inp_noise_std;
  bool stoch_if = sr || std > (T)0.0;

  STOCH_DEFINITIONS(stoch_if, size);

  T local_scale = 1.0;
  if (noise_management) {
    local_scale = *nm_scale_value;
  }

  local_scale *= bms;

  STRIDE_LOOP(size, value,

              APPLY_INPUT_NOISE_MANAGMENT(local_scale);

              DISCRETIZE_VALUE_STOCH;

              ADD_NOISE;

              BOUND_CHECK;

              APPLY_ASYMMETRY;

  );

  STOCH_FINALIZE(stoch_if);

  if (tid == total_threads - 1) {
    *scale_value_out = local_scale;
    *bound_exceeded = 0;
  }
}

/*********************************************************************************/
/*Helper for setting up the new bound management round*/
template <typename T>
__global__ void kernelUpdateScaleValuesAndInitialize(
    T *scale_values,
    int *exceeded_values,
    int *any_exceeded,
    const int m_batch_in,
    const T *nm_scale_values,
    const T reduction_due_to_bm_in) {

  // always called when round_number>1

  const int m_batch = m_batch_in;
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  int total_threads = blockDim.x * gridDim.x;
  const T bm_scale = reduction_due_to_bm_in;
  bool reread_nm_scales = nm_scale_values != nullptr;

  for (int i_stride = 0; i_stride < m_batch; i_stride += total_threads) {
    int idx = i_stride + tid;

    if (idx < m_batch) {

      if (exceeded_values[idx] > 0) {
        if (reread_nm_scales) {
          T nmsc = nm_scale_values[idx];
          scale_values[idx] = nmsc > (T)0.0 ? nmsc * bm_scale : bm_scale;
        } else {
          scale_values[idx] *= bm_scale;
        }
      }
      exceeded_values[idx] = 0;
    }
  }
  if (tid == 0) {
    *any_exceeded = 0; // initialize
  }
}

/*********************************************************************************/
/* Input management with bound management*/
template <typename T, typename InputIteratorT>
__global__ void kernelInputBoundManagementBatch(
    T *output,
    InputIteratorT input,
    const int size_in,
    const int m_batch_in,
    const bool trans_in,   // true if m_batch first dimensions
    const T *scale_values, // already updated
    const T bound_lower,
    const T bound_upper,
    const T resolution,
    const bool sto_round,
    const T inp_noise_std,
    const T asymmetry,
    curandState *random_states) {

  const int size = size_in;
  const int m_batch = m_batch_in;
  const bool trans = trans_in;
  const int total_size = size * m_batch;
  T std = inp_noise_std;
  T res = resolution;
  bool sr = sto_round && (res > (T)0.0);
  bool stoch_if = sr || std > (T)0.0;

  STOCH_DEFINITIONS(stoch_if, total_size);

  STRIDE_LOOP(total_size, value,

              int sidx = trans ? (idx % m_batch) : (idx / size);
              T svalue = scale_values[sidx];

              APPLY_INPUT_NOISE_MANAGMENT(svalue);

              DISCRETIZE_VALUE_STOCH;

              ADD_NOISE;

              BOUND_CHECK;

              APPLY_ASYMMETRY;);

  STOCH_FINALIZE(stoch_if);
}

/*********************************************************************************/
/*Output management: single batch */
template <typename T, typename OutputIteratorT, bool noise_management>
__global__ void kernelOutputManagement(
    OutputIteratorT output,
    const T *input,
    const int size,
    const T std,
    const T *nm_scale_value,
    const T bound_lower,
    const T bound_upper,
    const T res,
    const bool sto_round,
    const T out_scale,
    const T asymmetry,
    curandState *random_states) {
  bool sr = sto_round;
  bool stoch_if = sr || std > (T)0.0;
  STOCH_DEFINITIONS(stoch_if, size);

  T local_nm_scale = (T)0.0;
  if (noise_management) {
    local_nm_scale = *nm_scale_value;
  }

  RPU_CUDA_1D_KERNEL_LOOP(idx, size) {
    T value = input[idx];

    APPLY_ASYMMETRY;

    ADD_NOISE;

    DISCRETIZE_VALUE_STOCH;

    BOUND_CHECK;

    if (noise_management) {
      APPLY_OUTPUT_NOISE_MANAGMENT(local_nm_scale);
    }

    output[idx] = value * out_scale;
  }

  STOCH_FINALIZE(stoch_if);
}

/*********************************************************************************/
/* output without bound management*/
template <typename T, typename OutputIteratorT, bool noise_management>
__global__ void kernelOutputManagementBatch(
    OutputIteratorT output,
    const T *input,
    const int size,
    const int m_batch,
    const bool trans, // true if m_batch first dimensions
    const T *nm_scale_values,
    const T std, // out_noise
    const T bound_lower,
    const T bound_upper,
    const T res,
    const T sto_round,
    const T out_scale,
    const T asymmetry,
    curandState *random_states) {
  const int total_size = size * m_batch;
  const bool sr = sto_round;
  const bool stoch_if = sr || std > (T)0.0;

  STOCH_DEFINITIONS(stoch_if, total_size);

  T local_nm_scale = (T)0.0;

  RPU_CUDA_1D_KERNEL_LOOP(idx, total_size) {

    T value = input[idx];

    if (noise_management) {
      int bidx = trans ? (idx % m_batch) : (idx / size);
      local_nm_scale = nm_scale_values[bidx];
    }

    ADD_NOISE;

    APPLY_ASYMMETRY;

    DISCRETIZE_VALUE_STOCH;

    BOUND_CHECK;

    if (noise_management) {
      APPLY_OUTPUT_NOISE_MANAGMENT(local_nm_scale);
    }

    output[idx] = value * out_scale;
  }
  STOCH_FINALIZE(stoch_if);
}

/*********************************************************************************/
/* output with bound management*/
template <typename T, typename OutputIteratorT>
__global__ void kernelOutputBoundManagement(
    OutputIteratorT output,
    const T *input,
    const int in_size,
    const T out_noise,
    const T *dev_scale_value,
    const T bound_lower,
    const T bound_upper,
    const T resolution,
    const bool sto_round,
    const T out_scale,
    const bool bm_test_negative_bound,
    const T asymmetry,
    curandState *random_states,
    int *bound_exceeded) {
  const int size = in_size;
  T std = out_noise;
  T res = resolution;
  T local_scale = *dev_scale_value; // set by input
  int exceeded = 0;
  const bool sr = sto_round;
  const T osc = out_scale;
  bool stoch_if = sr || std > (T)0.0;
  const bool test_neg = bm_test_negative_bound;

  STOCH_DEFINITIONS(stoch_if, size);

  STRIDE_LOOP(
      size, value * osc,

      ADD_NOISE;

      APPLY_ASYMMETRY;

      DISCRETIZE_VALUE_STOCH;

      if (value > bu) {
        value = bu;
        exceeded++;
      }

      if (value < bl) {
        value = bl;
        exceeded += test_neg ? 0 : 1;
      }

      APPLY_OUTPUT_NOISE_MANAGMENT(local_scale);

  );
  STOCH_FINALIZE(stoch_if);

  if (exceeded > 0) {
    atomicAdd(bound_exceeded, exceeded);
  }
}

/*********************************************************************************/
/*Output Bound Management: Batch */

template <typename T, typename OutputIteratorT>
__global__ void kernelOutputBoundManagementBatch(
    OutputIteratorT output,
    const T *input,
    const int size_in,
    const int m_batch_in,
    const bool trans_in, // true if m_batch first dimensions
    const T *scale_values,
    int *exceeded_values_out, // ASSUMES TO BE INIT TO ZERO (InputBoundM)
    int *any_exceeded_out,    // should be zero
    const T out_noise,
    const T bound_lower,
    const T bound_upper,
    const T resolution,
    const bool sto_round,
    const T out_scale,
    const bool bm_test_negative_bound,
    const T asymmetry,
    curandState *random_states) {

  const int m_batch = m_batch_in;
  const int size = size_in;
  const bool trans = trans_in;
  const T std = out_noise;
  const int total_size = size * m_batch;
  const T res = resolution;
  const T osc = out_scale;
  const bool sr = sto_round;
  const bool stoch_if = sr || std > (T)0.0;
  const bool test_neg = bm_test_negative_bound;

  STOCH_DEFINITIONS(stoch_if, total_size);

  int exceeded = 0;

  STRIDE_LOOP(
      total_size, value * osc,

      int sidx = trans ? (idx % m_batch) : (idx / size);
      T local_scale = scale_values[sidx];

      ADD_NOISE;

      APPLY_ASYMMETRY;

      DISCRETIZE_VALUE_STOCH;

      if (value > bu) {
        value = bu;
        exceeded_values_out[sidx] = 1;
        exceeded++;
      } if (value < bl) {
        value = bl;
        if (test_neg) {
          exceeded_values_out[sidx] = 1;
          exceeded++;
        }
      }

      APPLY_OUTPUT_NOISE_MANAGMENT(local_scale););

  STOCH_FINALIZE(stoch_if);

  if (exceeded > 0) {
    atomicAdd(any_exceeded_out, 1);
  }
}

/********************************************************************/
/* INPUTOUTPUTMANAGER */
/********************************************************************/
#define RPU_IO_USE_SINGLE_BATCH_VERSION 1
#define RPU_IO_THREADS_PER_BLOCK 512

#define LAUNCH_NM_KERNEL(KNAME, TEMP, BLOCKS, ARGS)                                                \
  if (io_->noise_management != NoiseManagementType::None) {                                        \
    KNAME<T, TEMP, true><<<BLOCKS, nthreads_, 0, s>>> ARGS;                                        \
  } else {                                                                                         \
    KNAME<T, TEMP, false><<<BLOCKS, nthreads_, 0, s>>> ARGS;                                       \
  }

template <typename T>
InputOutputManager<T>::InputOutputManager(CudaContextPtr c, int in_size, int out_size)
    : context_(c), in_size_(in_size), out_size_(out_size) {

  dev_any_exceeded_ = RPU::make_unique<CudaArray<int>>(context_, 1);
  output_maximizer_ = RPU::make_unique<Maximizer<T>>(context_, out_size_, false); // MAX
  noise_manager_ = RPU::make_unique<NoiseManager<T>>(context_, in_size_);

  CUDA_CALL(cudaMallocHost((void **)&h_exceeded_, sizeof(int)));

  // block & threads
  nthreads_ = RPU_IO_THREADS_PER_BLOCK;
  nblocks_batch_max_ = context_->getSMCount() * (context_->maxThreadsPerBlock() / nthreads_);

  bound_management_factor_ = 2.0;
  buffer_m_batch_ = 0;

  this->initializeBatchBuffer(1);
}

template <typename T> InputOutputManager<T>::~InputOutputManager() { cudaFreeHost(h_exceeded_); }

template <typename T> void InputOutputManager<T>::initializeBatchBuffer(int m_batch) {

  if (buffer_m_batch_ < m_batch) {
    buffer_m_batch_ = m_batch;

    dev_scale_values_ = RPU::make_unique<CudaArray<T>>(context_, m_batch);
    dev_bound_exceeded_ = RPU::make_unique<CudaArray<int>>(context_, m_batch);
  }
}

/* Inits the IO management for processing. Computed max values,
   adjusts m_batch and intializes buffers and returns the output
   buffer. It should be used as:

   iom->initWithInput(d_input,...);

   T * out_buffer = iom->getOutBuffer();
   T * in_buffer = iom->getInBuffer();

   bool success = false;
   while (!success) {
     int m_batch_current = iom->applyToInput(d_input);

     my_compute_routine(out_buffer,in_buffer,m_batch_current);

     success = iom->applyToOutput(d_output,...);
   }

*/
template <typename T>
template <typename InputIteratorT>
void InputOutputManager<T>::initWithInput(
    InputIteratorT dev_input,
    const IOMetaParameter<T> &io,
    const int in_size,
    const int m_batch,
    const bool input_trans,
    const T add_out_scale,
    const bool is_test) {

  // save for apply
  io_ = &io;
  temp_trans_ = input_trans;
  temp_m_batch_ = m_batch;
  temp_out_scale_ = add_out_scale * io.out_scale;
  temp_is_test_ = is_test;
  temp_in_size_ = in_size;

  // in_size can be changed momentarily but only when noise management etc. is turned off.
  if (in_size != in_size_ && (io.noise_management != NoiseManagementType::None ||
                              io.bound_management != BoundManagementType::None)) {
    RPU_FATAL("Changing in_size not supported in the current configuration!");
  }

  if (buffer_m_batch_ < m_batch) {
    // re-initialize batch buffer
    this->initializeBatchBuffer(m_batch);
  }
  temp_input_applied_ =
      context_->template getSharedBuffer<T>(RPU_BUFFER_IN, m_batch * temp_in_size_);
  temp_output_applied_ = context_->template getSharedBuffer<T>(RPU_BUFFER_OUT, m_batch * out_size_);

  // noise management
  this->noise_manager_->compute(dev_input, io.noise_management, io, m_batch, input_trans, is_test);

  if (io.bound_management != BoundManagementType::None) {
    // add some logic for the bound management later
    bound_management_round_ = 0;
    reduction_due_to_bound_management_ = (T)1.0 / bound_management_factor_;
  }
}

template <typename T> void InputOutputManager<T>::releaseBuffer() {
  context_->template releaseSharedBuffer<T>(RPU_BUFFER_IN);
  context_->template releaseSharedBuffer<T>(RPU_BUFFER_OUT);
  temp_input_applied_ = nullptr;
  temp_output_applied_ = nullptr;
}

template <typename T>
template <typename InputIteratorT>
void InputOutputManager<T>::applyToInputWithBoundManagement(InputIteratorT dev_input) {

  cudaStream_t s = context_->getStream();
  int m_batch = temp_m_batch_;

  if (temp_in_size_ != in_size_) {
    RPU_FATAL("flexible in_size not suported");
  }
  int in_size = in_size_;

  // this will be called within a loop until all are not exceeded
  reduction_due_to_bound_management_ *= bound_management_factor_;
  bound_management_round_++;

  if (bound_management_round_ == 2 &&
      io_->bound_management == BoundManagementType::IterativeWorstCase) {

    if (io_->noise_management != NoiseManagementType::AbsMaxNPSum) {
      // recompute.
      this->noise_manager_->compute(
          dev_input, NoiseManagementType::AbsMaxNPSum, *io_, m_batch, temp_trans_, temp_is_test_);
    }
    reduction_due_to_bound_management_ = 1.0;
  }

  if (bound_management_round_ > 20) {
    std::cout << "Bound management already at " << reduction_due_to_bound_management_ << "\n";
  }

  T *nm_scale_values = noise_manager_->getScaleValues();

  if (m_batch == RPU_IO_USE_SINGLE_BATCH_VERSION) {
    int nblocks_im = getInBlocks();
    LAUNCH_NM_KERNEL(
        kernelInputBoundManagement, InputIteratorT, nblocks_im,
        (temp_input_applied_, dev_input, in_size, nm_scale_values, dev_scale_values_->getData(),
         reduction_due_to_bound_management_, -io_->inp_bound, io_->inp_bound, io_->inp_res,
         io_->inp_sto_round, io_->inp_noise, io_->inp_asymmetry,
         context_->getRandomStates(MIN(nblocks_im * nthreads_, in_size)),
         dev_any_exceeded_->getData()));

  } else {
    // here we simply recompute every batch
    bool nm = io_->noise_management != NoiseManagementType::None;

    if (bound_management_round_ == 1) {
      dev_bound_exceeded_->setConst(0);
      dev_any_exceeded_->setConst(0);
      if (nm) {
        RPU::math::copy(context_, m_batch, nm_scale_values, 1, dev_scale_values_->getData(), 1);
      } else {
        dev_scale_values_->setConst((T)1.0);
      }
    } else {
      // first update the scale values according to the exceeded
      // information from outputBoundManagement
      int nblocks = min(context_->getNBlocks(m_batch, nthreads_), nblocks_batch_max_);

      kernelUpdateScaleValuesAndInitialize<T><<<nblocks, nthreads_, 0, s>>>(
          dev_scale_values_->getData(), dev_bound_exceeded_->getData(),
          dev_any_exceeded_->getData(), m_batch, nm_scale_values,
          reduction_due_to_bound_management_);
    }

    // run
    int nblocks_im_batch = getInBlocksBatch(m_batch);
    kernelInputBoundManagementBatch<<<nblocks_im_batch, nthreads_, 0, s>>>(
        temp_input_applied_, dev_input, in_size, m_batch, temp_trans_, dev_scale_values_->getData(),
        -io_->inp_bound, io_->inp_bound, io_->inp_res, io_->inp_sto_round, io_->inp_noise,
        io_->inp_asymmetry,
        context_->getRandomStates(MIN(nblocks_im_batch * nthreads_, m_batch * in_size)));
  }
}

template <typename T>
template <typename InputIteratorT>
void InputOutputManager<T>::applyToInput(InputIteratorT dev_input) {
  if (io_->isPerfect()) {
    // short-cut (still need to copy though to apply the iterator)
    int m_batch = temp_m_batch_;
    RPU::math::copyWithIterator(context_, getInBuffer(), dev_input, m_batch * temp_in_size_);
    return;
  }
  if (io_->bound_management != BoundManagementType::None) {
    applyToInputWithBoundManagement(dev_input);
  } else {
    cudaStream_t s = context_->getStream();
    int m_batch = temp_m_batch_;
    T *nm_scale_values = noise_manager_->getScaleValues();
    int in_size = temp_in_size_; // here the in size is flexible

    // no bound management
    if (m_batch == RPU_IO_USE_SINGLE_BATCH_VERSION) {
      int nblocks_im = getInBlocks();

      if (io_->inp_sto_round || io_->inp_noise > (T)0.0) {
        LAUNCH_NM_KERNEL(
            kernelInputManagement, InputIteratorT, nblocks_im,
            (temp_input_applied_, dev_input, in_size, nm_scale_values, -io_->inp_bound,
             io_->inp_bound, io_->inp_res, io_->inp_sto_round, io_->inp_noise, io_->inp_asymmetry,
             context_->getRandomStates(MIN(nblocks_im * nthreads_, in_size))));
      } else {
        LAUNCH_NM_KERNEL(
            kernelInputManagement_noSR, InputIteratorT, nblocks_im,
            (temp_input_applied_, dev_input, in_size, nm_scale_values, -io_->inp_bound,
             io_->inp_bound, io_->inp_res, io_->inp_asymmetry));
      }
    } else {
      int nblocks_im_batch = getInBlocksBatch(m_batch);
      LAUNCH_NM_KERNEL(
          kernelInputManagementBatch, InputIteratorT, nblocks_im_batch,
          (temp_input_applied_, dev_input, in_size, m_batch, temp_trans_, nm_scale_values,
           -io_->inp_bound, io_->inp_bound, io_->inp_res, io_->inp_sto_round, io_->inp_noise,
           io_->inp_asymmetry,
           context_->getRandomStates(MIN(nblocks_im_batch * nthreads_, in_size * m_batch))));
    }
  }
}

template <typename T>
template <typename OutputIteratorT>
bool InputOutputManager<T>::applyToOutputWithBoundManagement(
    OutputIteratorT dev_output, const bool out_trans, const bool with_out_noise) {

  cudaStream_t s = context_->getStream();
  int m_batch = temp_m_batch_;
  int nblocks_om_batch = getOutBlocksBatch(m_batch);

  // actual bound management
  if (m_batch == RPU_IO_USE_SINGLE_BATCH_VERSION) {
    int nblocks_om = getOutBlocks();
    kernelOutputBoundManagement<<<nblocks_om, nthreads_, 0, s>>>(
        dev_output, temp_output_applied_, out_size_, with_out_noise ? io_->out_noise : (T)0.0,
        dev_scale_values_->getData(), -io_->out_bound, io_->out_bound, io_->out_res,
        io_->out_sto_round, temp_out_scale_, io_->bm_test_negative_bound, io_->out_asymmetry,
        context_->getRandomStates(MIN(nblocks_om * nthreads_, out_size_)),
        dev_any_exceeded_->getData());
  } else {

    kernelOutputBoundManagementBatch<<<nblocks_om_batch, nthreads_, 0, s>>>(
        dev_output, temp_output_applied_, out_size_, m_batch, out_trans,
        dev_scale_values_->getData(),
        dev_bound_exceeded_->getData(), // out
        dev_any_exceeded_->getData(),   // out
        with_out_noise ? io_->out_noise : (T)0.0, -io_->out_bound, io_->out_bound, io_->out_res,
        io_->out_sto_round, temp_out_scale_, io_->bm_test_negative_bound, io_->out_asymmetry,
        context_->getRandomStates(MIN(nblocks_om_batch * nthreads_, out_size_ * m_batch)));
  }

  dev_any_exceeded_->copyTo(h_exceeded_);
  return (
      ((*h_exceeded_) == 0) || (reduction_due_to_bound_management_ > (T)io_->max_bm_factor) ||
      (io_->inp_res > (T)0.0 &&
       reduction_due_to_bound_management_ > io_->max_bm_res / io_->inp_res));
}

template <typename T>
template <typename OutputIteratorT>
bool InputOutputManager<T>::applyToOutput(
    OutputIteratorT dev_output, const bool out_trans, const bool with_out_noise) {

  if (io_->isPerfect()) {
    // short-cut (still need to copy though to apply the iterator)
    int m_batch = temp_m_batch_;
    const T *tmp = getOutBuffer();
    RPU::math::copyWithIterator(context_, dev_output, tmp, m_batch * out_size_);
    return true;
  }

  // do the bound/ noise management (and ADC/DAC + out noise)
  if (io_->bound_management != BoundManagementType::None) {

    // bound management
    return applyToOutputWithBoundManagement(dev_output, out_trans, with_out_noise);

  } else {

    cudaStream_t s = context_->getStream();
    int m_batch = temp_m_batch_;
    int nblocks_om_batch =
        MIN(nblocks_batch_max_, this->context_->getNBlocks(out_size_ * m_batch, nthreads_));

    T *nm_scale_values = noise_manager_->getScaleValues();

    // no bound management
    if (m_batch == RPU_IO_USE_SINGLE_BATCH_VERSION) {
      int nblocks_om = MIN(context_->getNBlocks(out_size_, nthreads_), nblocks_batch_max_);
      LAUNCH_NM_KERNEL(
          kernelOutputManagement, OutputIteratorT, nblocks_om,
          (dev_output, temp_output_applied_, out_size_, with_out_noise ? io_->out_noise : (T)0.0,
           nm_scale_values, -io_->out_bound, io_->out_bound, io_->out_res, io_->out_sto_round,
           temp_out_scale_, io_->out_asymmetry,
           context_->getRandomStates(MIN(nblocks_om * nthreads_, out_size_))));

    } else {
      // batched
      LAUNCH_NM_KERNEL(
          kernelOutputManagementBatch, OutputIteratorT, nblocks_om_batch,
          (dev_output, temp_output_applied_, out_size_, m_batch, out_trans, nm_scale_values,
           with_out_noise ? io_->out_noise : (T)0.0, -io_->out_bound, io_->out_bound, io_->out_res,
           io_->out_sto_round, temp_out_scale_, io_->out_asymmetry,
           context_->getRandomStates(MIN(nblocks_om_batch * nthreads_, out_size_ * m_batch))));
    }

    return true; // no bound managment
  }
}

template <typename T>
void InputOutputManager<T>::dumpExtra(RPU::state_t &extra, const std::string prefix) {
  context_->synchronize();
  RPU::state_t state;

  noise_manager_->dumpExtra(state, "noise_manager");
  // output_maximizer_->dumpExtra(state, "output_maximizer");

  // will not handle buffers in to store data between applyToInput and applyToOutput
  RPU::insertWithPrefix(extra, state, prefix);
}

template <typename T>
void InputOutputManager<T>::loadExtra(
    const RPU::state_t &extra, const std::string prefix, bool strict) {
  context_->synchronize();

  auto state = RPU::selectWithPrefix(extra, prefix);

  noise_manager_->loadExtra(state, "noise_manager", strict);
  // output_maximizer_->loadExtra(state, "output_maximizer", strict);
}

// init necessary templates..
#define OARG(NUM_T) , const bool, const bool
#define IARG(NUM_T)                                                                                \
  , const IOMetaParameter<NUM_T> &, const int, const int, const bool, const NUM_T, const bool

template class InputOutputManager<float>;
RPU_GEN_IITER_TEMPLATES(float, void, InputOutputManager<float>::applyToInput, );
RPU_GEN_IITER_TEMPLATES(float, void, InputOutputManager<float>::initWithInput, IARG(float));
RPU_GEN_OITER_TEMPLATES(float, bool, InputOutputManager<float>::applyToOutput, OARG(float));

#ifdef RPU_USE_DOUBLE
template class InputOutputManager<double>;
RPU_GEN_IITER_TEMPLATES(double, void, InputOutputManager<double>::applyToInput, );
RPU_GEN_IITER_TEMPLATES(double, void, InputOutputManager<double>::initWithInput, IARG(double));
RPU_GEN_OITER_TEMPLATES(double, bool, InputOutputManager<double>::applyToOutput, OARG(double));
#endif
#ifdef RPU_USE_FP16
template class InputOutputManager<half_t>;
RPU_GEN_IITER_TEMPLATES(half_t, void, InputOutputManager<half_t>::applyToInput, );
RPU_GEN_IITER_TEMPLATES(half_t, void, InputOutputManager<half_t>::initWithInput, IARG(half_t));
RPU_GEN_OITER_TEMPLATES(half_t, bool, InputOutputManager<half_t>::applyToOutput, OARG(half_t));
#endif

#undef OARG
#undef IARG

#undef RPU_IO_USE_SINGLE_BATCH_VERSION
#undef RPU_IO_THREADS_PER_BLOCK

#undef APPLY_OUTPUT_NOISE_MANAGMENT
#undef APPLY_INPUT_NOISE_MANAGMENT
#undef NOISE_MANAGEMENT_ZERO_VALUE

#undef STOCH_DEFINITIONS
#undef DISCRETIZE_VALUE
#undef DISCRETIZE_VALUE_STOCH
#undef STOCH_FINALIZE
#undef STRIDE_LOOP
#undef BOUND_CHECK
#undef ADD_NOISE

#undef LAUNCH_NM_KERNEL
} // namespace RPU
