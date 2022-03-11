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

#include "io_manager.h"

#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <random>

#include "cuda_math_util.h"
#include "cuda_util.h"
#include "forward_backward_pass.h"
#include "io_iterator.h"
#include <cub/cub.cuh>

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
  if (res > 0) {                                                                                   \
    value /= res;                                                                                  \
    if (sr) {                                                                                      \
      stoch_value = curand_uniform(&local_state);                                                  \
      value += stoch_value - 0.5;                                                                  \
    }                                                                                              \
    value = res * RPU_ROUNDFUN(value);                                                             \
  }

#define ADD_NOISE                                                                                  \
  if (std > 0) {                                                                                   \
    value += std * curand_normal(&local_state);                                                    \
  }

#define DISCRETIZE_VALUE                                                                           \
  if (res > 0) {                                                                                   \
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
  { value = LOCAL_NM_SCALE > 0.0 ? value / LOCAL_NM_SCALE : value; }

#define APPLY_OUTPUT_NOISE_MANAGMENT(LOCAL_NM_SCALE)                                               \
  { value = (LOCAL_NM_SCALE > 0.0) ? value * LOCAL_NM_SCALE : 0.0; }

namespace RPU {

/*********************************************************************************/
/*Input Management: shortcut for no SR (no noise needed)*/
template <typename T, typename InputIteratorT, bool noise_management>
__global__ void kernelInputManagement_noSR(
    T *output,
    InputIteratorT input,
    const int size,
    const float *nm_scale_value,
    const T bound_lower,
    const T bound_upper,
    const T resolution) {
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
      BOUND_CHECK;);
}

/*********************************************************************************/
/*Input Management Single Batch */
template <typename T, typename InputIteratorT, bool noise_management>
__global__ void kernelInputManagement(
    T *output,
    InputIteratorT input,
    const int size_in,
    const float *nm_scale_value,
    const T bound_lower,
    const T bound_upper,
    const T resolution,
    const bool sto_round,
    const T inp_noise_std,
    curandState *random_states) {
  int size = size_in;
  T res = resolution;
  bool sr = sto_round && (res > 0);
  T std = inp_noise_std;
  bool stoch_if = sr || std > 0;

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

      BOUND_CHECK;);

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
    const float *nm_scale_values,
    const T bound_lower,
    const T bound_upper,
    const T resolution,
    const bool sto_round,
    const T inp_noise_std,
    curandState *random_states) {
  int size = size_in;

  bool trans = trans_in;
  int m_batch = m_batch_in;
  int total_size = size * m_batch;

  T res = resolution;
  bool sr = sto_round && (res > 0);
  T std = inp_noise_std;
  bool stoch_if = sr || std > 0;

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

      BOUND_CHECK;);

  STOCH_FINALIZE(stoch_if);
}

/*********************************************************************************/
/*Input Management with bound check. Single Batch */
template <typename T, typename InputIteratorT, bool noise_management>
__global__ void kernelInputBoundManagement(
    T *output,
    InputIteratorT input,
    const int size_in,
    float *nm_scale_value,
    float *scale_value_out,
    const T bm_scale,
    const T bound_lower,
    const T bound_upper,
    const T resolution,
    const bool sto_round,
    const T inp_noise_std,
    curandState *random_states,
    int *bound_exceeded) {
  int size = size_in;
  T res = resolution;
  bool sr = sto_round && (res > 0);
  T bms = bm_scale;
  T std = inp_noise_std;
  bool stoch_if = sr || std > 0;

  STOCH_DEFINITIONS(stoch_if, size);

  T local_scale = 1.0;
  if (noise_management) {
    local_scale = *nm_scale_value;
  }

  local_scale *= bms;

  STRIDE_LOOP(size, value,

              APPLY_INPUT_NOISE_MANAGMENT(local_scale);

              DISCRETIZE_VALUE_STOCH; ADD_NOISE; BOUND_CHECK;);

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
    float *scale_values,
    int *exceeded_values,
    int *any_exceeded,
    const int m_batch_in,
    const float *nm_scale_values,
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
          float nmsc = nm_scale_values[idx];
          scale_values[idx] = nmsc > 0 ? nmsc * bm_scale : bm_scale;
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
    const bool trans_in,       // true if m_batch first dimensions
    const float *scale_values, // already updated
    const T bound_lower,
    const T bound_upper,
    const T resolution,
    const bool sto_round,
    const T inp_noise_std,
    curandState *random_states) {

  const int size = size_in;
  const int m_batch = m_batch_in;
  const bool trans = trans_in;
  const int total_size = size * m_batch;
  T std = inp_noise_std;
  T res = resolution;
  bool sr = sto_round && (res > 0);
  bool stoch_if = sr || std > 0;

  STOCH_DEFINITIONS(stoch_if, total_size);

  STRIDE_LOOP(total_size, value,

              int sidx = trans ? (idx % m_batch) : (idx / size);
              T svalue = scale_values[sidx];

              APPLY_INPUT_NOISE_MANAGMENT(svalue);

              DISCRETIZE_VALUE_STOCH;

              ADD_NOISE;

              BOUND_CHECK;);

  STOCH_FINALIZE(stoch_if);
}
/*********************************************************************************/
/* Input management with bound management for selected batch with copying*/
template <typename T, typename InputIteratorT, bool noise_management>
__global__ void kernelInputBoundManagementBatchSelected(
    T *output,
    InputIteratorT input,
    const int size_in,
    const int m_batch_in,
    const int m_batch_selected_in,
    const int *selected_bidx,
    const bool trans_in, // true if m_batch first dimensions
    const float *nm_scale_values,
    const T reduction_due_to_bm_in,
    const T bound_lower,
    const T bound_upper,
    const T resolution,
    const bool sto_round,
    const T inp_noise_std,
    curandState *random_states) {

  const int size = size_in;
  const int m_batch = m_batch_in;
  const bool trans = trans_in;
  const T bm_scale = reduction_due_to_bm_in;
  const int m_batch_selected = m_batch_selected_in;

  const int total_size = size * m_batch_selected;
  T res = resolution;
  bool sr = sto_round && (res > 0);
  T std = inp_noise_std;
  const bool stoch_if = sr || std > 0;

  STOCH_DEFINITIONS(stoch_if, total_size);

  // STRIDE LOOP
  T value;
  T local_scale;
  for (int i_stride = 0; i_stride < total_size; i_stride += total_threads) {
    int idx_sel = i_stride + tid;

    if (idx_sel < total_size) {
      int bidx_sel = trans ? (idx_sel % m_batch_selected) : (idx_sel / size);
      int xidx = trans ? (idx_sel / m_batch_selected) : (idx_sel % size);

      int bidx = selected_bidx[bidx_sel];
      int idx = trans ? (bidx + m_batch * xidx) : (xidx + size * bidx);

      if (noise_management) {
        local_scale = nm_scale_values[bidx];
      }
      value = input[idx];

      if (noise_management) {
        APPLY_INPUT_NOISE_MANAGMENT(local_scale);
      }

      value /= bm_scale; // always apply bms, it will not be zero

      DISCRETIZE_VALUE_STOCH;

      ADD_NOISE;

      BOUND_CHECK;

      output[idx_sel] = value;
    }
  }

  STOCH_FINALIZE(stoch_if);
}
/*********************************************************************************/
/* output add wnoise single batch*/
template <typename T>
__global__ void kernelOutputWeightNoise(
    T *output,
    const T *input,
    const int size_in,
    const T *x_norm_value,
    const T w_noise_std,
    curandState *random_states) {
  T noise_std = (*x_norm_value) * w_noise_std;
  const int size = size_in;
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  T stoch_value;
  curandState local_state;
  int total_threads = blockDim.x * gridDim.x;
  int total_states = MIN(size, total_threads);

  if (tid < total_states) {
    local_state = random_states[tid];
  }

  STRIDE_LOOP(size, value, stoch_value = curand_normal(&local_state);
              value += noise_std * stoch_value);

  STOCH_FINALIZE(true);
}

/* output add wnoise batch*/
template <typename T>
__global__ void kernelOutputWeightNoiseBatch(
    T *output,
    const T *input,
    const int size_in,
    const int m_batch_in,
    const bool trans_in, // true if m_batch first dimensions
    const T *noise_var_values,
    curandState *random_states) {
  const bool trans = trans_in;
  const int size = size_in;
  const int m_batch = m_batch_in;
  const int total_size = size * m_batch;

  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  T stoch_value;
  curandState local_state;
  int total_threads = blockDim.x * gridDim.x;
  int total_states = MIN(total_size, total_threads);
  if (tid < total_states) {
    local_state = random_states[tid];
  }

  STRIDE_LOOP(total_size, value,
              int bidx = trans ? (idx % m_batch) : (idx / size);
              T noise_var =
                  noise_var_values[bidx]; // "if (noise_var>0)"  is actually slightly slower...
              stoch_value = curand_normal(&local_state); value += stoch_value * sqrt(noise_var););
  STOCH_FINALIZE(true);
}

/* output add wnoise batch*/
template <typename T>
__global__ void kernelElemSqrtAddNoiseBatch(
    T *output,
    const T *input,
    const T *noise_var_values,
    const int total_size_in,
    curandState *random_states) {
  const int total_size = total_size_in;

  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  T stoch_value;
  curandState local_state;
  int total_threads = blockDim.x * gridDim.x;
  int total_states = MIN(total_size, total_threads);
  if (tid < total_states) {
    local_state = random_states[tid];
  }

  STRIDE_LOOP(total_size, value, T noise_var = noise_var_values[idx];
              stoch_value = curand_normal(&local_state); value += stoch_value * sqrt(noise_var););
  STOCH_FINALIZE(true);
}

// compute x_j*(1-(1-j/n)^2)
template <typename T>
__global__ void kernelInputPositionCoding(
    T *output,      // x' out
    const T *input, // x in
    const int size_in,
    const int m_batch_in,
    const bool trans_in) {
  const bool trans = trans_in;
  const int size = size_in;
  const int m_batch = m_batch_in;
  const int total_size = size * m_batch;

  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  int total_threads = blockDim.x * gridDim.x;

  STRIDE_LOOP(total_size, value,

              int xidx = trans ? (idx / m_batch) : (idx % size);
              T z = (T)1.0 - (T)xidx / (T)size; value *= (T)1.0 - z * z;);
}

// computes c_i and y_i -= c_i*GEMM_i
// c_i = a_i*(a_i*(0.05*a_i - 0.2) + 0.5);
template <typename T>
__global__ void kernelElemMulCAdd(
    T *output,
    const T *input,
    const T *a_values,
    const T *pc_gemm_values,
    const int total_size_in) {
  const int total_size = total_size_in;

  int total_threads = blockDim.x * gridDim.x;
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  const T K2 = 0.05;
  const T K1 = 0.2;
  const T K0 = 0.5;

  STRIDE_LOOP(total_size, value, T a_i = a_values[idx]; T gemm_i = pc_gemm_values[idx];
              T c_i = a_i * (a_i * (K2 * a_i - K1) + K0); value -= c_i * gemm_i;);
}

/*********************************************************************************/
/*Output management: single batch */
template <typename T, typename OutputIteratorT, bool noise_management>
__global__ void kernelOutputManagement(
    OutputIteratorT output,
    const T *input,
    const int size_in,
    const T out_noise,
    const float *nm_scale_value,
    const T bound_lower,
    const T bound_upper,
    const T resolution,
    const bool sto_round,
    const T out_scale,
    curandState *random_states) {
  int size = size_in;
  T res = resolution;
  T noise_std = out_noise;
  T osc = out_scale;
  bool sr = sto_round;
  bool stoch_if = sr || noise_std > 0;
  STOCH_DEFINITIONS(stoch_if, size);

  T local_nm_scale = 0.0;
  if (noise_management) {
    local_nm_scale = *nm_scale_value;
  }

  STRIDE_LOOP(
      size, value * osc,

      if (noise_std > 0) {
        stoch_value = curand_normal(&local_state);
        value += noise_std * stoch_value;
      }

      DISCRETIZE_VALUE_STOCH;

      BOUND_CHECK;

      if (noise_management) { APPLY_OUTPUT_NOISE_MANAGMENT(local_nm_scale); });

  STOCH_FINALIZE(stoch_if);
}

/*********************************************************************************/
/* output without bound management*/
template <typename T, typename OutputIteratorT, bool noise_management>
__global__ void kernelOutputManagementBatch(
    OutputIteratorT output,
    const T *input,
    const int size_in,
    const int m_batch_in,
    const bool trans_in, // true if m_batch first dimensions
    const float *nm_scale_values,
    const T out_noise,
    const T bound_lower,
    const T bound_upper,
    const T resolution,
    const T sto_round,
    const T out_scale,
    curandState *random_states) {
  const bool trans = trans_in;
  const int size = size_in;
  const int m_batch = m_batch_in;
  const int total_size = size * m_batch;
  const T res = resolution;
  const T noise_std = out_noise;
  const T osc = out_scale;
  const bool sr = sto_round;
  const bool stoch_if = sr || noise_std > 0;

  STOCH_DEFINITIONS(stoch_if, total_size);

  T local_nm_scale = (T)0.0;

  STRIDE_LOOP(
      total_size, value * osc,

      if (noise_management) {
        int bidx = trans ? (idx % m_batch) : (idx / size);
        local_nm_scale = nm_scale_values[bidx];
      }

      if (noise_std > 0) {
        stoch_value = curand_normal(&local_state);
        value += noise_std * stoch_value;
      }

      DISCRETIZE_VALUE_STOCH;

      BOUND_CHECK;

      if (noise_management) { APPLY_OUTPUT_NOISE_MANAGMENT(local_nm_scale); });
  STOCH_FINALIZE(stoch_if);
}

/*********************************************************************************/
/* output shift management*/
template <typename T, typename OutputIteratorT, bool noise_management>
__global__ void kernelOutputShiftManagementBatch(
    OutputIteratorT output,
    const T *input,
    const int size_in,
    const int m_batch_in,
    const bool trans_in, // true if m_batch first dimensions
    const float *nm_scale_values,
    const float *shift_values,
    const T bound_lower,
    const T bound_upper,
    const T resolution,
    const bool sto_round,
    curandState *random_states) {
  const bool trans = trans_in;
  const int size = size_in;
  const int m_batch = m_batch_in;
  const int total_size = size * m_batch;
  const T res = resolution;
  const bool sr = sto_round;

  T local_nm_scale = 0.0;

  STOCH_DEFINITIONS(sr, total_size);

  STRIDE_LOOP(
      total_size, value,

      int bidx = trans ? (idx % m_batch) : (idx / size);
      T local_shift = shift_values[bidx];

      if (noise_management) { local_nm_scale = nm_scale_values[bidx]; }

      value += bu - local_shift;

      DISCRETIZE_VALUE_STOCH;

      BOUND_CHECK;

      if (noise_management) { APPLY_OUTPUT_NOISE_MANAGMENT(local_nm_scale); });
  STOCH_FINALIZE(sr);
}

/*********************************************************************************/
/* output with bound management*/
template <typename T, typename OutputIteratorT>
__global__ void kernelOutputBoundManagement(
    OutputIteratorT output,
    const T *input,
    const int in_size,
    const T out_noise,
    const float *dev_scale_value,
    const T bound_lower,
    const T bound_upper,
    const T resolution,
    const bool sto_round,
    const T out_scale,
    const bool bm_test_negative_bound,
    curandState *random_states,
    int *bound_exceeded) {
  const int size = in_size;
  T noise_std = out_noise;
  T res = resolution;
  T local_scale = *dev_scale_value; // set by input
  int exceeded = 0;
  const bool sr = sto_round;
  const T osc = out_scale;
  bool stoch_if = sr || noise_std > 0.0;
  const bool test_neg = bm_test_negative_bound;

  STOCH_DEFINITIONS(stoch_if, size);

  STRIDE_LOOP(
      size, value * osc,

      if (noise_std > 0) {
        stoch_value = curand_normal(&local_state);
        value += noise_std * stoch_value;
      }

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
    const float *scale_values,
    int *exceeded_values_out, // ASSUMES TO BE INIT TO ZERO (InputBoundM)
    int *any_exceeded_out,    // should be zero
    const T out_noise,
    const T bound_lower,
    const T bound_upper,
    const T resolution,
    const bool sto_round,
    const T out_scale,
    const bool bm_test_negative_bound,
    curandState *random_states) {

  const int m_batch = m_batch_in;
  const int size = size_in;
  const bool trans = trans_in;
  const T noise_std = out_noise;
  const int total_size = size * m_batch;
  const T res = resolution;
  const T osc = out_scale;
  const bool sr = sto_round;
  const bool stoch_if = sr || noise_std > 0.0;
  const bool test_neg = bm_test_negative_bound;

  STOCH_DEFINITIONS(stoch_if, total_size);

  int exceeded = 0;

  STRIDE_LOOP(
      total_size, value * osc,

      int sidx = trans ? (idx % m_batch) : (idx / size);
      T local_scale = scale_values[sidx];

      if (noise_std > 0) {
        stoch_value = curand_normal(&local_state);
        value += noise_std * stoch_value;
      }

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

/*********************************************************************************/
/*Output Bound Management: Selected Batch */

template <typename T, typename OutputIteratorT, bool noise_management>
__global__ void kernelOutputBoundManagementBatchSelected(
    OutputIteratorT output,
    const T *input,
    const int size_in,
    const int m_batch_in,
    const int m_batch_selected_in,
    const int *selected_bidx,
    const bool trans_in, // true if m_batch first dimensions
    const float *nm_scale_values,
    const T reduction_due_to_bm_in,
    int *exceeded_values_out, // ASSUMES TO BE INIT TO ZERO (InputBoundM)
    int *any_exceeded_out,    // should be zero
    const T out_noise,
    const T bound_lower,
    const T bound_upper,
    const T resolution,
    const bool sto_round,
    const T out_scale,
    const bool bm_test_negative_bound,
    curandState *random_states) {

  const int m_batch = m_batch_in;
  const int size = size_in;
  const bool trans = trans_in;
  const T noise_std = out_noise;
  const T res = resolution;
  const T osc = out_scale;
  const bool sr = sto_round;
  const bool stoch_if = noise_std > 0.0 || sr;
  const T bm_scale = reduction_due_to_bm_in;
  const int m_batch_selected = m_batch_selected_in;
  const int total_size = size * m_batch_selected;
  const bool test_neg = bm_test_negative_bound;

  int exceeded = 0;

  STOCH_DEFINITIONS(stoch_if, total_size);

  // STRIDE LOOP
  T value;
  for (int i_stride = 0; i_stride < total_size; i_stride += total_threads) {
    int idx_sel = i_stride + tid;

    if (idx_sel < total_size) {

      value = input[idx_sel];

      int bidx_sel = trans ? (idx_sel % m_batch_selected) : (idx_sel / size);
      int bidx = selected_bidx[bidx_sel];
      int xidx = trans ? (idx_sel / m_batch_selected) : (idx_sel % size);
      int idx = trans ? (bidx + m_batch * xidx) : (xidx + size * bidx);

      T local_scale = bm_scale;
      if (noise_management) {
        local_scale *= nm_scale_values[bidx];
      }

      if (noise_std > 0) {
        stoch_value = curand_normal(&local_state);
        value += noise_std * stoch_value;
      }

      DISCRETIZE_VALUE_STOCH;

      if (value > bu) {
        value = bu;
        exceeded_values_out[bidx] = 1;
        exceeded++;
      }
      if (value < bl) {
        value = bl;
        if (test_neg) {
          exceeded_values_out[bidx] = 1;
          exceeded++;
        }
      }

      APPLY_OUTPUT_NOISE_MANAGMENT(local_scale);

      output[idx] = value * osc;
    }
  }

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
#define RPU_IO_BM_SELECTING 0 // CAUTION: SELECTING SEEMS TO BE BUGGY.... need check. Turned off

#define LAUNCH_NM_KERNEL(KNAME, TEMP, BLOCKS, ARGS)                                                \
  if (io_->noise_management != NoiseManagementType::None) {                                        \
    KNAME<T, TEMP, true><<<BLOCKS, nthreads_, 0, s>>> ARGS;                                        \
  } else {                                                                                         \
    KNAME<T, TEMP, false><<<BLOCKS, nthreads_, 0, s>>> ARGS;                                       \
  }

template <typename T>
InputOutputManager<T>::InputOutputManager(CudaContext *c, int in_size, int out_size)
    : context_(c), in_size_(in_size), out_size_(out_size) {
  noise_manager_ = RPU::make_unique<NoiseManager<T>>(context_, in_size_);
  output_maximizer_ = RPU::make_unique<Maximizer<T>>(context_, out_size_, false); // MAX

  dev_any_exceeded_ = RPU::make_unique<CudaArray<int>>(context_, 1);
  dev_selected_m_batch_ = RPU::make_unique<CudaArray<int>>(context_, 1);

  CUDA_CALL(cudaMallocHost((void **)&h_exceeded_, sizeof(int)));

  // block & threads
  nthreads_ = RPU_IO_THREADS_PER_BLOCK;

  nblocks_batch_max_ = context_->getSMCount() * (context_->maxThreadsPerBlock() / nthreads_);
  nblocks_om_ = MIN(context_->getNBlocks(out_size_, nthreads_), nblocks_batch_max_);
  nblocks_im_ = MIN(context_->getNBlocks(in_size_, nthreads_), nblocks_batch_max_);

  bound_management_factor_ = 2.0;
  buffer_m_batch_ = 0;

  this->initializeBatchBuffer(1);
}

template <typename T> InputOutputManager<T>::~InputOutputManager() { cudaFreeHost(h_exceeded_); }

template <typename T>
void InputOutputManager<T>::setSharedBuffer(
    int m_batch,
    std::shared_ptr<CudaArray<T>> in_buffer,
    std::shared_ptr<CudaArray<T>> out_buffer) {

  if (in_buffer) {
    dev_input_applied_ = in_buffer;
    if (dev_input_applied_->getSize() < m_batch * in_size_) {
      RPU_FATAL("In buffer size too small.");
    }
  }

  if (out_buffer) {
    dev_output_applied_ = out_buffer;
    if (dev_output_applied_->getSize() < m_batch * out_size_) {
      RPU_FATAL("Out buffer size too small.");
    }
  }
}

template <typename T> void InputOutputManager<T>::initializeBatchBuffer(int m_batch) {

  if (buffer_m_batch_ < m_batch) {
    buffer_m_batch_ = m_batch;

    if (!dev_input_applied_ || dev_input_applied_->getSize() < m_batch * in_size_) {
      dev_input_applied_ = std::make_shared<CudaArray<T>>(context_, m_batch * in_size_);
    }

    if (!dev_output_applied_ || dev_output_applied_->getSize() < m_batch * out_size_) {
      dev_output_applied_ = std::make_shared<CudaArray<T>>(context_, m_batch * out_size_);
    }

    dev_scale_values_ = RPU::make_unique<CudaArray<float>>(context_, m_batch);
    dev_bound_exceeded_ = RPU::make_unique<CudaArray<int>>(context_, m_batch);

    dev_selected_bidx_ = RPU::make_unique<CudaArray<int>>(context_, m_batch);

    nblocks_om_batch_ =
        MIN(nblocks_batch_max_, this->context_->getNBlocks(out_size_ * m_batch, nthreads_));
    nblocks_im_batch_ =
        MIN(nblocks_batch_max_, this->context_->getNBlocks(in_size_ * m_batch, nthreads_));

    size_t byte_size = 0;
    RPU::cub::CountingInputIterator<int> itr(0);
    RPU::cub::DeviceSelect::Flagged(
        (void *)nullptr, byte_size, itr, dev_bound_exceeded_->getData(),
        dev_selected_bidx_->getData(), dev_selected_m_batch_->getData(), m_batch,
        context_->getStream());

    dev_flagged_temp_storage_ =
        RPU::make_unique<CudaArray<char>>(context_, (byte_size + 31) / 32 * 32);
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
    const int m_batch,
    const bool input_trans,
    const T add_out_scale,
    const bool is_test) {

  // save for apply
  io_ = &io;
  temp_trans_ = input_trans;
  temp_m_batch_ = m_batch;
  temp_out_scale_ = add_out_scale * io_->out_scale;
  temp_is_test_ = is_test;

  if (buffer_m_batch_ < m_batch) {
    // re-initialize batch buffer
    this->initializeBatchBuffer(m_batch);
  }

  // noise management
  this->noise_manager_->compute(dev_input, io.noise_management, io, m_batch, input_trans, is_test);

  // bound management
  currently_selecting_bidx_ = false;
  bm_with_selecting_ = RPU_IO_BM_SELECTING && (m_batch + out_size_ > 10000); // better heuristic?

  if (io_->noise_management == NoiseManagementType::None) {
    bm_with_selecting_ = false; // not supported.
  }

  if (io_->bound_management != BoundManagementType::None) {
    // add some logic for the bound management later
    bound_management_round_ = 0;
    reduction_due_to_bound_management_ = 1.0 / bound_management_factor_;
  }
}

template <typename T> T *InputOutputManager<T>::getInBuffer() {
  return dev_input_applied_->getData();
}

template <typename T> T *InputOutputManager<T>::getOutBuffer() {
  return dev_output_applied_->getData();
}

template <typename T>
template <typename InputIteratorT>
int InputOutputManager<T>::applyToInputWithBoundManagement(InputIteratorT dev_input) {

  cudaStream_t s = context_->getStream();
  int m_batch = temp_m_batch_;
  int m_batch_out = m_batch;

  // this will be called within a loop until all are not exceeded
  reduction_due_to_bound_management_ *= bound_management_factor_;
  bound_management_round_++;

  bool reset_round = false;
  if (bound_management_round_ == 2 &&
      io_->bound_management == BoundManagementType::IterativeWorstCase) {

    if (io_->noise_management != NoiseManagementType::AbsMaxNPSum) {
      // recompute.
      this->noise_manager_->compute(
          dev_input, NoiseManagementType::AbsMaxNPSum, *io_, m_batch, temp_trans_, temp_is_test_);
    }
    // reset
    reduction_due_to_bound_management_ = 1.0;

    reset_round =
        !bm_with_selecting_; // not selecting needs reset round (selecting needs NM though)
  }

  if (bound_management_round_ > 20.0) {
    std::cout << "Bound management already at " << reduction_due_to_bound_management_ << "\n";
  }

  float *nm_scale_values = noise_manager_->getScaleValues();

  if (m_batch == RPU_IO_USE_SINGLE_BATCH_VERSION) {

    LAUNCH_NM_KERNEL(
        kernelInputBoundManagement, InputIteratorT, nblocks_im_,
        (dev_input_applied_->getData(), dev_input, in_size_, nm_scale_values,
         dev_scale_values_->getData(), reduction_due_to_bound_management_, -io_->inp_bound,
         io_->inp_bound, io_->inp_res, io_->inp_sto_round, io_->inp_noise,
         context_->getRandomStates(MIN(nblocks_im_ * nthreads_, in_size_)),
         dev_any_exceeded_->getData()));

  } else if (!bm_with_selecting_ || bound_management_round_ == 1 || reset_round) {
    // here we simply recompute every batch
    bool nm = io_->noise_management != NoiseManagementType::None;

    if (bound_management_round_ == 1) {
      dev_bound_exceeded_->setConst(0);
      dev_any_exceeded_->setConst(0);
      if (nm) {
        RPU::math::copy(context_, m_batch, nm_scale_values, 1, dev_scale_values_->getData(), 1);
      } else {
        dev_scale_values_->setConst((float)1.0);
      }
    } else {
      // first update the scale values according to the exceeded
      // information from outputBoundManagement
      int nblocks = min(context_->getNBlocks(m_batch, nthreads_), nblocks_batch_max_);

      kernelUpdateScaleValuesAndInitialize<T><<<nblocks, nthreads_, 0, s>>>(
          dev_scale_values_->getData(), dev_bound_exceeded_->getData(),
          dev_any_exceeded_->getData(), m_batch, (reset_round) ? nm_scale_values : nullptr,
          reduction_due_to_bound_management_);
    }

    // run
    kernelInputBoundManagementBatch<<<nblocks_im_batch_, nthreads_, 0, s>>>(
        dev_input_applied_->getData(), dev_input, in_size_, m_batch, temp_trans_,
        dev_scale_values_->getData(), -io_->inp_bound, io_->inp_bound, io_->inp_res,
        io_->inp_sto_round, io_->inp_noise,
        context_->getRandomStates(MIN(nblocks_im_batch_ * nthreads_, m_batch * in_size_)));

  } else {

    // with selecting batches
    // the logic here:
    // 1) first test the whole batch: IBM -> GEMM -> OBM (above)
    // 2) If some batches exceeded the output bound dev_bound_exceeded will be > 0 for those batch
    // indices 3) We copy exceeded batches into d_out and recompute on those only 4) then save the
    // recomputed at the correct index position in the output
    // std::cout << "starting IBM with "  << reduction_due_to_bound_management_ << std::endl;

    if (nm_scale_values == nullptr) {
      RPU_FATAL("BM selecting needs nm_scale_values to be defined.");
    }

    currently_selecting_bidx_ = true;
    // dev_bound_exceeded_->printValues();

    // it is bound_management_round_>1 here
    RPU::cub::CountingInputIterator<int> itr(0);
    size_t byte_size = dev_flagged_temp_storage_->getSize();
    RPU::cub::DeviceSelect::Flagged(
        (void *)dev_flagged_temp_storage_->getData(), byte_size, itr,
        dev_bound_exceeded_->getData(), dev_selected_bidx_->getData(),
        dev_selected_m_batch_->getData(), m_batch, s);

    // need to set exceed to zero for the following OBM
    dev_bound_exceeded_->setConst(0);
    dev_any_exceeded_->setConst(0);
    dev_selected_m_batch_->copyTo(&m_batch_selected_); // needed on CPU for GEMM outside and
                                                       // below...
    m_batch_out = m_batch_selected_;

    // run + copy only those batch to d_out which have exceeded (and scale them accordingly)
    int nblocks =
        min(context_->getNBlocks(m_batch_selected_ * in_size_, nthreads_), nblocks_batch_max_);

    LAUNCH_NM_KERNEL(
        kernelInputBoundManagementBatchSelected, InputIteratorT, nblocks,
        (dev_input_applied_->getData(), dev_input, in_size_, m_batch, m_batch_selected_,
         dev_selected_bidx_->getDataConst(), temp_trans_, nm_scale_values,
         reduction_due_to_bound_management_, -io_->inp_bound, io_->inp_bound, io_->inp_res,
         io_->inp_sto_round, io_->inp_noise,
         context_->getRandomStates(MIN(nblocks * nthreads_, in_size_ * m_batch))));
  }

  return m_batch_out;
}

template <typename T>
template <typename InputIteratorT>
int InputOutputManager<T>::applyToInput(InputIteratorT dev_input) {
  if (io_->is_perfect) {
    // short-cut (still need to copy though to apply the iterator)
    int m_batch = temp_m_batch_;
    RPU::math::copyWithIterator(context_, getInBuffer(), dev_input, m_batch * in_size_);
    return m_batch;
  }

  if (io_->bound_management != BoundManagementType::None) {

    return applyToInputWithBoundManagement(dev_input);

  } else {

    cudaStream_t s = context_->getStream();
    int m_batch = temp_m_batch_;
    float *nm_scale_values = noise_manager_->getScaleValues();

    // no bound management
    if (m_batch == RPU_IO_USE_SINGLE_BATCH_VERSION) {

      if (io_->inp_sto_round || io_->inp_noise > 0) {
        LAUNCH_NM_KERNEL(
            kernelInputManagement, InputIteratorT, nblocks_im_,
            (dev_input_applied_->getData(), dev_input, in_size_, nm_scale_values, -io_->inp_bound,
             io_->inp_bound, io_->inp_res, io_->inp_sto_round, io_->inp_noise,
             context_->getRandomStates(MIN(nblocks_im_ * nthreads_, in_size_))));
      } else {
        LAUNCH_NM_KERNEL(
            kernelInputManagement_noSR, InputIteratorT, nblocks_im_,
            (dev_input_applied_->getData(), dev_input, in_size_, nm_scale_values, -io_->inp_bound,
             io_->inp_bound, io_->inp_res));
      }
    } else {

      LAUNCH_NM_KERNEL(
          kernelInputManagementBatch, InputIteratorT, nblocks_im_batch_,
          (dev_input_applied_->getData(), dev_input, in_size_, m_batch, temp_trans_,
           nm_scale_values, -io_->inp_bound, io_->inp_bound, io_->inp_res, io_->inp_sto_round,
           io_->inp_noise,
           context_->getRandomStates(MIN(nblocks_im_batch_ * nthreads_, in_size_ * m_batch))));
    }
    return m_batch;
  }
}

template <typename T>
template <typename OutputIteratorT>
bool InputOutputManager<T>::applyToOutputWithBoundManagement(
    OutputIteratorT dev_output, const bool out_trans) {

  cudaStream_t s = context_->getStream();
  int m_batch = temp_m_batch_;

  // actual bound management

  if (m_batch == RPU_IO_USE_SINGLE_BATCH_VERSION) {
    kernelOutputBoundManagement<<<nblocks_om_, nthreads_, 0, s>>>(
        dev_output, dev_output_applied_->getData(), out_size_, io_->out_noise,
        dev_scale_values_->getData(), -io_->out_bound, io_->out_bound, io_->out_res,
        io_->out_sto_round, temp_out_scale_, io_->bm_test_negative_bound,
        context_->getRandomStates(MIN(nblocks_om_ * nthreads_, out_size_)),
        dev_any_exceeded_->getData());
  } else {

    if (!currently_selecting_bidx_) {

      kernelOutputBoundManagementBatch<<<nblocks_om_batch_, nthreads_, 0, s>>>(
          dev_output, dev_output_applied_->getData(), out_size_, m_batch, out_trans,
          dev_scale_values_->getData(),
          dev_bound_exceeded_->getData(), // out
          dev_any_exceeded_->getData(),   // out
          io_->out_noise, -io_->out_bound, io_->out_bound, io_->out_res, io_->out_sto_round,
          temp_out_scale_, io_->bm_test_negative_bound,
          context_->getRandomStates(MIN(nblocks_om_batch_ * nthreads_, out_size_ * m_batch)));

    } else {
      // only add those batches to the output that exceeded in the previous round
      int nblocks =
          min(context_->getNBlocks(m_batch_selected_ * out_size_, nthreads_), nblocks_batch_max_);

      float *nm_scale_values = noise_manager_->getScaleValues();

      LAUNCH_NM_KERNEL(
          kernelOutputBoundManagementBatchSelected, OutputIteratorT, nblocks,
          (dev_output, dev_output_applied_->getData(), out_size_, m_batch, m_batch_selected_,
           dev_selected_bidx_->getDataConst(), out_trans, nm_scale_values,
           reduction_due_to_bound_management_, dev_bound_exceeded_->getData(),
           dev_any_exceeded_->getData(), io_->out_noise, -io_->out_bound, io_->out_bound,
           io_->out_res, io_->out_sto_round, temp_out_scale_, io_->bm_test_negative_bound,
           context_->getRandomStates(MIN(nblocks * nthreads_, out_size_ * m_batch))));
    }
  }

  dev_any_exceeded_->copyTo(h_exceeded_);
  return (
      ((*h_exceeded_) == 0) || (reduction_due_to_bound_management_ > io_->max_bm_factor) ||
      (io_->inp_res > 0 && reduction_due_to_bound_management_ > io_->max_bm_res / io_->inp_res));
}

template <typename T> void InputOutputManager<T>::applyOutputWeightNoise(const bool out_trans) {
  if (io_->w_noise > 0) {

    int m_batch = temp_m_batch_;
    int in_trans = temp_trans_;

    if (!dev_wnoise_buffer_ || dev_wnoise_buffer_->getSize() < m_batch) {
      // on the fly init when wnoise is requested to save a little memory
      dev_wnoise_buffer_ = RPU::make_unique<CudaArray<T>>(context_, m_batch);
      dev_wnoise_ones_ = RPU::make_unique<CudaArray<T>>(context_, in_size_);
      dev_wnoise_ones_->setConst((T)1.0);
    }

    T *in_temp = getInBuffer(); // this is in_size_ x m_batch or m_batch x in_size_ (if in_trans)
    T *out_temp =
        getOutBuffer(); // this is out_size_ x m_batch or m_batch x out_size_ (if out_trans)

    if (m_batch == 1) {

      // directly calculate vector norm, (over-)use same buffer
      RPU::math::nrm2(context_, in_size_, in_temp, 1, dev_wnoise_buffer_->getData());

      kernelOutputWeightNoise<<<nblocks_om_, nthreads_, 0, context_->getStream()>>>(
          out_temp,
          out_temp, // in-place
          out_size_,
          dev_wnoise_buffer_->getDataConst(), // this is actually only the norm value, see above
          io_->w_noise, context_->getRandomStates(MIN(nblocks_om_ * nthreads_, out_size_)));

    } else {

      // in_temp can be overwritten since it use assumed to have been used already
      RPU::math::elempow2(context_, in_temp, in_size_ * m_batch); // in-place

      // we just use gemm for the reduction. That way we can also scale by the noise level
      RPU::math::gemm<T>(
          context_, false, in_trans, 1,
          m_batch,                     // M
          in_size_,                    // K
          io_->w_noise * io_->w_noise, // weight noise variance scale
          dev_wnoise_ones_->getDataConst(), 1, in_temp, (in_trans) ? m_batch : in_size_, (T)0.0,
          dev_wnoise_buffer_->getData(), 1);

      // now we have the variance of the wnoise for each batch in the dev_wnoise_buffer_

      // generate and add the weight noise to the applied output
      kernelOutputWeightNoiseBatch<<<nblocks_om_batch_, nthreads_, 0, context_->getStream()>>>(
          out_temp,
          out_temp, // in-place
          out_size_, m_batch, out_trans, dev_wnoise_buffer_->getDataConst(),
          context_->getRandomStates(MIN(nblocks_om_batch_ * nthreads_, out_size_ * m_batch)));
    }
  }
}

template <typename T>
void InputOutputManager<T>::applyOutputPCMReadNoise(const T *dev_weights, const bool out_trans) {
  if (io_->w_noise > 0) {
    // compute the overall noise contributions is sigma_i propto sqrt(sum_j
    // |Wij|*xj^2) eventually we might want to use cutlass for
    // this, but let's wait once we have more time since it seems
    // more complicated than thought

    int m_batch = temp_m_batch_;
    int in_trans = temp_trans_;

    T *in_temp = getInBuffer(); // this is in_size_ x m_batch or m_batch x in_size_ (if in_trans)
    T *out_temp =
        getOutBuffer(); // this is out_size_ x m_batch or m_batch x out_size_ (if out_trans)

    // unfortunately needs a lot of extra buffer memory (CUTLASS will help)
    RPU_GET_CUDA_BUFFER(context_, T, dev_extra_batch_buffer_, m_batch * out_size_);
    RPU_GET_CUDA_BUFFER(context_, T, dev_extra_weight_buffer_, in_size_ * out_size_);

    // compute thr abs weights current
    RPU::math::elemabs(
        context_, dev_extra_weight_buffer_->getData(), dev_weights, out_size_ * in_size_);

    // COMPUTE SQR IN-PLACE thus OVERWRITES IN BUFFER!!
    // in_temp can be overwritten since it use assumed to have been used already
    RPU::math::elempow2(context_, in_temp, in_size_ * m_batch); // in-place

    // we just use gemm also scale by the noise level
    // assumed_wmax is set to 1. if otherwise, need to be set from outside
    T noise_level = io_->w_noise;
    detail::forwardMatrix(
        context_, dev_extra_weight_buffer_->getData(), in_temp, in_size_, in_trans,
        dev_extra_batch_buffer_->getData(), out_size_, out_trans, m_batch,
        noise_level * noise_level);

    // MIGHT WANT TO SUPPORT SCALING IT TO ACTUAL WMAX!?.

    // need to do the sqrt still and add to

    // generate and add the weight noise to the applied output
    kernelElemSqrtAddNoiseBatch<<<nblocks_om_batch_, nthreads_, 0, context_->getStream()>>>(
        out_temp,
        out_temp, // in-place
        dev_extra_batch_buffer_->getData(), out_size_ * m_batch,
        context_->getRandomStates(MIN(nblocks_om_batch_ * nthreads_, out_size_ * m_batch)));
  }
}

template <typename T>
void InputOutputManager<T>::applyIrDrop(const T *dev_weights, const bool out_trans) {
  if (io_->ir_drop > 0.0) {
    // computes for each output (row):
    //
    // a_i = sum_j(|w_ij|*|x_j|)*n/Gw*gmax
    // c_i = a_i*(a_i*(0.05*a_i - 0.2) + 0.5);
    // x'_j = x_j * (1 - (1-j/n)^2)
    // y_i = y_i_ideal - c_i*sum_j(w_ij * x'_j)

    int m_batch = temp_m_batch_;
    int in_trans = temp_trans_;

    T *in_temp = getInBuffer(); // this is in_size_ x m_batch or m_batch x in_size_ (if in_trans)
    T *out_temp =
        getOutBuffer(); // this is out_size_ x m_batch or m_batch x out_size_ (if out_trans)

    // unfortunately needs a lot of extra buffer memory (CUTLASS will help)
    RPU_GET_CUDA_BUFFER(context_, T, dev_extra_batch_buffer_, m_batch * MAX(in_size_, out_size_));
    RPU_GET_CUDA_BUFFER(context_, T, dev_extra_weight_buffer_, MAX(in_size_, m_batch) * out_size_);
    RPU_GET_CUDA_BUFFER(context_, T, dev_a_buffer_, m_batch * out_size_);

    // compute thr abs weights current
    RPU::math::elemabs(
        context_, dev_extra_weight_buffer_->getData(), dev_weights, out_size_ * in_size_);

    RPU::math::elemabs(context_, dev_extra_batch_buffer_->getData(), in_temp, in_size_ * m_batch);

    // compute the a_i
    detail::forwardMatrix(
        context_, dev_extra_weight_buffer_->getData(), dev_extra_batch_buffer_->getData(), in_size_,
        in_trans, dev_a_buffer_->getData(), out_size_, out_trans, m_batch,
        in_size_ / io_->ir_drop_Gw_div_gmax);

    // compute x_j*(1-(1-j/n)^2)
    kernelInputPositionCoding<<<nblocks_om_batch_, nthreads_, 0, context_->getStream()>>>(
        dev_extra_batch_buffer_->getData(), in_temp, in_size_, m_batch, in_trans);

    // compute the position filtered GEMM
    detail::forwardMatrix(
        context_, dev_weights, dev_extra_batch_buffer_->getData(), in_size_, in_trans,
        dev_extra_weight_buffer_->getData(), // re-use  buffer
        out_size_, out_trans, m_batch, io_->ir_drop);

    // computes c_i and y_i -= c_i*GEMM_i
    kernelElemMulCAdd<<<nblocks_om_batch_, nthreads_, 0, context_->getStream()>>>(
        out_temp,
        out_temp,                            // in-place
        dev_a_buffer_->getData(),            // this is the a_i
        dev_extra_weight_buffer_->getData(), // this is GEMM_i
        out_size_ * m_batch                  // all have same shape
    );
  }
}

template <typename T>
void InputOutputManager<T>::applyOutputNonIdealities(const T *dev_weights, const bool out_trans) {

  applyIrDrop(dev_weights, out_trans);

  switch (io_->w_noise_type) {
  case OutputWeightNoiseType::AdditiveConstant: {
    // overwrites inBuffer
    applyOutputWeightNoise(out_trans);
    break;
  }
  case OutputWeightNoiseType::PCMRead: {
    // overwrites inBuffer
    applyOutputPCMReadNoise(dev_weights, out_trans);

    break;
  }
  case OutputWeightNoiseType::None: {
    break;
  }
  default:
    RPU_FATAL("Output noise type  not implemented.");
  }
}

template <typename T>
template <typename OutputIteratorT>
bool InputOutputManager<T>::applyToOutput(
    OutputIteratorT dev_output, const T *dev_weights, const bool out_trans) {

  if (io_->is_perfect) {
    // short-cut (still need to copy though to apply the iterator)
    int m_batch = temp_m_batch_;
    const T *tmp = getOutBuffer();
    RPU::math::copyWithIterator(context_, dev_output, tmp, m_batch * out_size_);
    return true;
  }

  // first apply the output transforms
  applyOutputNonIdealities(dev_weights, out_trans);

  // do the bound/ noise management (and ADC/DAC + out noise)
  if (io_->bound_management != BoundManagementType::None &&
      io_->bound_management != BoundManagementType::Shift) {

    // bound management
    return applyToOutputWithBoundManagement(dev_output, out_trans);

  } else {

    cudaStream_t s = context_->getStream();
    int m_batch = temp_m_batch_;

    float *nm_scale_values = noise_manager_->getScaleValues();

    if (io_->bound_management == BoundManagementType::Shift) {
      // 1) just add noise (no NM scaling yet)
      // 2) compute max
      // 3) shift everything so that max is at out_bound
      // 4) discritize and scale (by the NM input max)

      // just use batched version here
      // apply noise (without the bounds)
      if (io_->out_noise > 0) {
        kernelOutputManagementBatch<T, T *, false><<<nblocks_om_batch_, nthreads_, 0, s>>>(
            dev_output_applied_->getData(), dev_output_applied_->getDataConst(), out_size_, m_batch,
            out_trans, nullptr, io_->out_noise, -std::numeric_limits<T>::max(),
            std::numeric_limits<T>::max(), -1., false,
            (T)1.0, // no scale
            context_->getRandomStates(MIN(nblocks_om_batch_ * nthreads_, out_size_ * m_batch)));
      }

      // compute max // this needs to be a maximizer not absmaximizer !!
      output_maximizer_->compute(dev_output_applied_->getDataConst(), m_batch, out_trans);

      // shift and discretize
      LAUNCH_NM_KERNEL(
          kernelOutputShiftManagementBatch, OutputIteratorT, nblocks_om_batch_,
          (dev_output, dev_output_applied_->getDataConst(), out_size_, m_batch, out_trans,
           nm_scale_values, output_maximizer_->getMaxValues(), -io_->out_bound, io_->out_bound,
           io_->out_res, io_->out_sto_round,
           context_->getRandomStates(MIN(nblocks_om_batch_ * nthreads_, out_size_ * m_batch))));

      return true; // shift

    } else {

      // no bound management
      if (m_batch == RPU_IO_USE_SINGLE_BATCH_VERSION) {
        LAUNCH_NM_KERNEL(
            kernelOutputManagement, OutputIteratorT, nblocks_om_,
            (dev_output, dev_output_applied_->getData(), out_size_, io_->out_noise, nm_scale_values,
             -io_->out_bound, io_->out_bound, io_->out_res, io_->out_sto_round, temp_out_scale_,
             context_->getRandomStates(MIN(nblocks_om_ * nthreads_, out_size_))));

      } else {
        // batched
        LAUNCH_NM_KERNEL(
            kernelOutputManagementBatch, OutputIteratorT, nblocks_om_batch_,
            (dev_output, dev_output_applied_->getData(), out_size_, m_batch, out_trans,
             nm_scale_values, io_->out_noise, -io_->out_bound, io_->out_bound, io_->out_res,
             io_->out_sto_round, temp_out_scale_,
             context_->getRandomStates(MIN(nblocks_om_batch_ * nthreads_, out_size_ * m_batch))));
      }

      return true; // no bound managment
    }
  }
}

// init necessary templates..
#define OARG(NUM_T) , const NUM_T *, const bool
#define IARG(NUM_T) , const IOMetaParameter<NUM_T> &, const int, const bool, const NUM_T, const bool

template class InputOutputManager<float>;
RPU_GEN_IITER_TEMPLATES(float, int, InputOutputManager<float>::applyToInput, );
RPU_GEN_IITER_TEMPLATES(float, void, InputOutputManager<float>::initWithInput, IARG(float));
RPU_GEN_OITER_TEMPLATES(float, bool, InputOutputManager<float>::applyToOutput, OARG(float));

#ifdef RPU_USE_DOUBLE
template class InputOutputManager<double>;
RPU_GEN_IITER_TEMPLATES(double, int, InputOutputManager<double>::applyToInput, );
RPU_GEN_IITER_TEMPLATES(double, void, InputOutputManager<double>::initWithInput, IARG(double));
RPU_GEN_OITER_TEMPLATES(double, bool, InputOutputManager<double>::applyToOutput, OARG(double));
#endif

#undef OARG
#undef IARG

#undef RPU_IO_USE_SINGLE_BATCH_VERSION
#undef RPU_IO_THREADS_PER_BLOCK
#undef RPU_IO_BM_SELECTING

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
