/**
 * (C) Copyright 2020 IBM. All Rights Reserved.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
 */

#pragma once

#include "cuda_math_util.h"
#include <cub/cub.cuh>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <random>

namespace RPU {

/*********************************************************************************/
/* kernel update W */
/*********************************************************************************/

__device__ __forceinline__ bool testBit(uint32_t mask, int bit) {
  return (mask & (((uint32_t)1) << bit)) > 0;
}

// some helper templates
template <bool trans>
__device__ __forceinline__ int
getIdxToLoad(int batch_index, int count_index, int sz, int m_batch, int count_offset);

/*getNfromCount the "loading" functions from the two bitlines. 2
 versions for the two formats: 32 bit (first bit sign bit) or 64
 bit, where higher value BYTE word conatins all the signs.

 This formats are chosen by the type count_t which is either uint32 or uint64

 one_sided is either -1,0,1. If -1,1 only negative or positive
 values are loaded other bits discarded. Note that in the case of -1
 the negative sign is also reversed to positive so that positve
 steps are taken on the device. In case of 0 both negative and
 positive steps are kept and performed on the weights.
*/
template <int one_sided, typename count_t>
__device__ __forceinline__ void getNfromCount(
    uint32_t &n,
    uint32_t &negative,
    bool &mixed,
    count_t *x_ptr,
    count_t *d_ptr,
    int nK32,
    int shared_x_offset,
    int shared_d_offset);

template <>
__device__ __forceinline__ int
getIdxToLoad<true>(int batch_index, int count_index, int sz, int m_batch, int count_offset) {
  // batch first order
  int batch_idx_aligned = batch_index + count_index * m_batch;
  return (batch_idx_aligned / sz) * count_offset + (batch_idx_aligned % sz);
}

template <>
__device__ __forceinline__ int
getIdxToLoad<false>(int batch_index, int count_index, int sz, int m_batch, int count_offset) {
  return count_index + batch_index * count_offset;
}

#define COMMA ,
#define DEFINE_GETNFROMCOUNT32(ONE_SIDED, OS_ADD)                                                  \
  template <>                                                                                      \
  __device__ __forceinline__ void getNfromCount<ONE_SIDED COMMA uint32_t>(                         \
      uint32_t & n, uint32_t & negative, bool &mixed, uint32_t *x_ptr, uint32_t *d_ptr, int nK32,  \
      int shared_x_offset, int shared_d_offset) {                                                  \
    uint32_t x = *x_ptr;                                                                           \
    uint32_t d = *d_ptr;                                                                           \
                                                                                                   \
    /* never mixed pos/neg within one read in this format*/                                        \
    mixed = false;                                                                                 \
    negative = ((x & 1) ^ (d & 1));                                                                \
                                                                                                   \
    OS_ADD                                                                                         \
                                                                                                   \
    uint32_t x_and_d = x & d;                                                                      \
    n = __popc(x_and_d);                                                                           \
    n -= ((x_and_d)&1);                                                                            \
                                                                                                   \
    if (nK32 > 1) {                                                                                \
      int i_d = 0;                                                                                 \
      int i_x = 0;                                                                                 \
      PRAGMA(unroll)                                                                               \
      for (int i = 0; i < (nK32 - 1); i++) {                                                       \
        i_x += shared_x_offset;                                                                    \
        i_d += shared_d_offset;                                                                    \
        x = *(x_ptr + i_x);                                                                        \
        d = *(d_ptr + i_d);                                                                        \
        n += __popc(x & d);                                                                        \
      }                                                                                            \
    }                                                                                              \
  }

DEFINE_GETNFROMCOUNT32(0, );
DEFINE_GETNFROMCOUNT32(
    -1,
    if (negative) {
      n = 0;
      return;
    } else { negative = 1; });
DEFINE_GETNFROMCOUNT32(
    1, if (!negative) {
      n = 0;
      return;
    });
#undef DEFINE_GETNFROMCOUNT32

#define DEFINE_GETNFROMCOUNTFP(FPTYPE, ONE_SIDED, OS_ADD)                                          \
  template <>                                                                                      \
  __device__ __forceinline__ void getNfromCount<ONE_SIDED COMMA FPTYPE>(                           \
      uint32_t & n, uint32_t & negative, bool &mixed, FPTYPE *x_ptr, FPTYPE *d_ptr, int nK32,      \
      int shared_x_offset, int shared_d_offset) {                                                  \
    FPTYPE x = *x_ptr;                                                                             \
    FPTYPE d = *d_ptr;                                                                             \
                                                                                                   \
    /* never mixed pos/neg within one read in this format*/                                        \
    mixed = false;                                                                                 \
    negative = ((x < 0) != (d < 0)) ? 1 : 0;                                                       \
    if ((x == 0) || (d == 0)) {                                                                    \
      n = 0;                                                                                       \
      return;                                                                                      \
    };                                                                                             \
                                                                                                   \
    OS_ADD                                                                                         \
                                                                                                   \
    n = abs(RPU_ROUNDFUN(d * x));                                                                  \
  }

DEFINE_GETNFROMCOUNTFP(float, 0, );
DEFINE_GETNFROMCOUNTFP(
    float,
    -1,
    if (negative) {
      n = 0;
      return;
    } else { negative = 1; });
DEFINE_GETNFROMCOUNTFP(
    float, 1, if (!negative) {
      n = 0;
      return;
    });

#ifdef RPU_USE_DOUBLE
DEFINE_GETNFROMCOUNTFP(double, 0, );
DEFINE_GETNFROMCOUNTFP(
    double,
    -1,
    if (negative) {
      n = 0;
      return;
    } else { negative = 1; });
DEFINE_GETNFROMCOUNTFP(
    double, 1, if (!negative) {
      n = 0;
      return;
    });
#endif
#undef DEFINE_GETNFROMCOUNTFP

#define DEFINE_GETNFROMCOUNT64(ONE_SIDED, OS_ADD)                                                  \
  template <>                                                                                      \
  __device__ __forceinline__ void getNfromCount<ONE_SIDED COMMA uint64_t>(                         \
      uint32_t & n, uint32_t & negative, bool &mixed, uint64_t *x_ptr, uint64_t *d_ptr, int nK32,  \
      int shared_x_offset, int shared_d_offset) {                                                  \
    /* -- nK32 is ignored (assumed 1). larger K will be in put into the batch order*/              \
    /* -- this is the bit-wise negative version */                                                 \
    /* -- 64 bit : upper bits have negative info*/                                                 \
                                                                                                   \
    uint64_t x = *x_ptr;                                                                           \
    uint64_t d = *d_ptr;                                                                           \
                                                                                                   \
    uint32_t neg = ((uint32_t)(x >> 32)) ^ ((uint32_t)(d >> 32));                                  \
    /* probably overkill, anyway do it explicitly*/                                                \
    uint32_t data = ((uint32_t)(x & (uint64_t)0x00000000ffffffff)) &                               \
                    ((uint32_t)(d & (uint64_t)0x00000000ffffffff));                                \
    /*uint32_t data = ((uint32_t) x) & ((uint32_t)d );*/                                           \
                                                                                                   \
    OS_ADD                                                                                         \
                                                                                                   \
    bool all_negative = (~(neg | ~data)) == 0;                                                     \
    bool all_positive = (~(~neg | ~data)) == 0;                                                    \
                                                                                                   \
    if (all_negative || all_positive) {                                                            \
      /* in case of all zero data both are true. Do not set negative in this case*/                \
      negative = (all_positive) ? 0 : 1;                                                           \
      n = __popc(data);                                                                            \
      mixed = false;                                                                               \
    } else {                                                                                       \
      /* some mixed pos and negative. Just return the data directly. The delta will be computed    \
       * later*/                                                                                   \
      mixed = true;                                                                                \
      negative = neg;                                                                              \
      n = data;                                                                                    \
    }                                                                                              \
  }

DEFINE_GETNFROMCOUNT64(0, );
DEFINE_GETNFROMCOUNT64(-1, data &= ~neg; neg = ~0;);
DEFINE_GETNFROMCOUNT64(1, data &= neg; neg = ~0;);
#undef DEFINE_GETNFROMCOUNT64
#undef COMMA

#define RPU_FUNCTOR_INIT_VARS                                                                      \
  T w = 0;                                                                                         \
  float4 par_4;                                                                                    \
  float2 par_2;                                                                                    \
  T par_1 = 0;                                                                                     \
  bool use_par_1 = params_1 != nullptr;                                                            \
  UpdateFunctor up_fun;                                                                            \
  static_assert(global_params_count <= 32, "global params count exceeds warpsize");                \
  __shared__ T global_par[global_params_count];                                                    \
  if (global_params != nullptr) {                                                                  \
    if (threadIdx.x < global_params_count) {                                                       \
      global_par[threadIdx.x] = global_params[threadIdx.x];                                        \
    }                                                                                              \
    __syncthreads();                                                                               \
  }

//T * global_par = global_params;					\

#define RPU_FUNCTOR_LOAD_PARAMS                                                                    \
  {                                                                                                \
    w = weights[idx];                                                                              \
    if (params != nullptr) {                                                                       \
      par_4 = reinterpret_cast<float4 *>(params)[idx];                                             \
    }                                                                                              \
    if (params_2 != nullptr) {                                                                     \
      par_2 = reinterpret_cast<float2 *>(params_2)[idx];                                           \
    }                                                                                              \
    if (use_par_1) {                                                                               \
      par_1 = params_1[idx];                                                                       \
    }                                                                                              \
  }

/*********************************************************************************/
/*********************************************************************************/
/* UPDATE Functor  */
template <typename T> struct UpdateFunctorConstantStep {

  __device__ __forceinline__ void operator()(
      T &w,
      uint32_t n,
      uint32_t negative,
      const float4 par_4,
      const float2 par_2,
      T &par_1,
      const T *global_par,
      T noise_std_dw,
      curandState &local_state) {

    // note that only w and par_1 will be written back when used. Thus it can be a "hidden_weights"
    // type note that we here assume that stoch_value is < 1, or if larger, then it did not hit the
    // bound.

    float dw = (negative > 0) ? (par_4.w) : (-par_4.y);
    T wmax = par_4.z;
    T wmin = par_4.x;
    float sigma = noise_std_dw;
    // n is larger 0 in any case
    if (n == 1) {
      if (sigma > 0) {
        float stoch_value = curand_normal(&local_state);
        stoch_value *= sigma;
        w += dw * ((float)1.0 + stoch_value);
      } else {
        w += dw;
      }
    } else {
      if (sigma > 0) {
        float stoch_value = curand_normal(&local_state);
        stoch_value *= sigma;
        w += dw * n *
             ((float)1.0 + rsqrtf((float)n) * stoch_value); // rsqrt(x) = 1/sqrt(x) is faster
      } else {
        w += dw * n;
      }
    }

    // better always check both bounds
    w = (w > wmax) ? wmax : w;
    w = (w < wmin) ? wmin : w;
  }
};

/*********************************************************************************/
/*********************************************************************************/
/** Single Batch versions **/

template <
    typename T,
    int one_sided,
    typename count_t,
    typename UpdateFunctor,
    int global_params_count = 1,
    typename std::enable_if<(global_params_count > 0), int>::type = 0>
__global__ void kernelUpdateWFunctor(
    T *weights,
    int size,
    count_t *x_counts,
    int x_size,
    count_t *d_counts,
    int d_size,
    float *params,
    float *params_2,
    T *params_1,
    T *global_params,
    int nK32in,
    const T dw_min_std,
    curandState *random_states) {
  // call with <<< d_size*x_size/NUM_THREADS_PER_BLOCK,NUM_THREADS_PER_BLOCK>>>

  // W in col major, that is first d_ then x_
  // params expected in the following order:  (min_bound, scale_down, max_bound, scale_up )
  // params2 is expected: (slope_down, slope_up)
  const uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;

  const int xsz = x_size;
  const int dsz = d_size;
  const int sz = xsz * dsz;
  const int total_threads = blockDim.x * gridDim.x;
  const int nK32 = nK32in;

  RPU_FUNCTOR_INIT_VARS;

  const T noise_std_dw = dw_min_std;
  curandState local_state;
  uint32_t negative = 0;
  uint32_t n;
  bool mixed = false;

  if (dw_min_std > 0 && tid < size) {
    local_state = random_states[tid];
  }

  for (int i_stride = 0; i_stride < sz; i_stride += total_threads) {

    int idx = tid + i_stride;

    if (idx < sz) { // stride over all elements of W

      int dIdx = idx % dsz; // first d
      int xIdx = idx / dsz;

      RPU_FUNCTOR_LOAD_PARAMS;

      getNfromCount<one_sided, count_t>(
          n, negative, mixed, &x_counts[xIdx], &d_counts[dIdx], nK32, xsz, dsz);

      if (n > 0) {
        up_fun(w, n, negative, par_4, par_2, par_1, global_par, noise_std_dw, local_state);
        weights[idx] = w;
        if (use_par_1) {
          params_1[idx] = par_1;
        }
      }
    }
  }
  if (dw_min_std > 0 && tid < sz) {
    random_states[tid] = local_state;
  }
}

/*********************************************************************************/
/*********************************************************************************/
// non shared batch versions

#define RPU_UPDATE_WITH_SUM_N_INNER(BOUND_CHECK_BODY)                                              \
  if (mixed) {                                                                                     \
    BOUND_CHECK_BODY {                                                                             \
      PRAGMA(unroll)                                                                               \
      for (int i_bit = 0; i_bit < 32; i_bit++) {                                                   \
        uint32_t bit_n = testBit(n, i_bit) ? 1 : 0;                                                \
        if (bit_n != 0) {                                                                          \
          uint32_t bit_neg = testBit(negative, i_bit) ? 1 : 0;                                     \
          if (bit_neg == last_negative) {                                                          \
            sum_n += 1;                                                                            \
          } else {                                                                                 \
            if (sum_n > 0) {                                                                       \
              up_fun(                                                                              \
                  w, sum_n, last_negative, par_4, par_2, par_1, nullptr, noise_std_dw,             \
                  local_state);                                                                    \
            }                                                                                      \
            sum_n = 1;                                                                             \
            last_negative = bit_neg;                                                               \
          }                                                                                        \
        }                                                                                          \
      }                                                                                            \
    }                                                                                              \
  } else {                                                                                         \
    if ((n == 0) || (last_negative == negative)) {                                                 \
      sum_n += n;                                                                                  \
    } else {                                                                                       \
      if (sum_n > 0) {                                                                             \
        up_fun(w, sum_n, last_negative, par_4, par_2, par_1, nullptr, noise_std_dw, local_state);  \
      }                                                                                            \
      sum_n = n;                                                                                   \
      last_negative = negative;                                                                    \
    }                                                                                              \
  }

#define RPU_UPDATE_WITH_SUM_N_INNER_BOUND_CHECK                                                    \
  RPU_UPDATE_WITH_SUM_N_INNER(                                                                     \
      if (sum_n > 0) {                                                                             \
        up_fun(w, sum_n, last_negative, par_4, par_2, par_1, nullptr, noise_std_dw, local_state);  \
      } sum_n = 0;                                                                                 \
      last_negative = 0;                                                                           \
                                                                                                   \
      int pos_n = __popc((~negative) & n); int neg_n = __popc((negative)&n);                       \
      float dw_pos = (float)pos_n; float dw_neg = (float)neg_n;                                    \
                                                                                                   \
      if (noise_std_dw > 0) {                                                                      \
        if (pos_n > 0) {                                                                           \
          float stoch_value = curand_normal(&local_state);                                         \
          dw_pos += sqrtf(dw_pos) * noise_std_dw;                                                  \
        }                                                                                          \
        if (neg_n > 0) {                                                                           \
          float stoch_value = curand_normal(&local_state);                                         \
          dw_neg += sqrtf(dw_neg) * noise_std_dw;                                                  \
        }                                                                                          \
      } dw_pos *= par_4.y;                                                                         \
      dw_neg *= par_4.w;                                                                           \
                                                                                                   \
      if ((w - dw_pos >= par_4.x) && (w + dw_neg <= par_4.z)) { w -= dw_pos - dw_neg; } else)

template <typename T, int one_sided, typename count_t, bool x_trans, bool d_trans>
__global__ void kernelUpdateWBatchSum(
    T *weights,
    int size_in,
    count_t *x_counts,
    int x_size,
    count_t *d_counts,
    int d_size,
    float *params,
    int nK32_in,
    int m_batch_in,
    const T dw_min_std,
    curandState *random_states,
    kagg_t *Kn = nullptr) {
  int m_batch = m_batch_in;
  if (Kn != nullptr) {
    m_batch = ((*Kn) + 31) / 32; // overwrite m_batch in case of BO64 UBLM
  }

  volatile unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
  const int total_threads = blockDim.x * gridDim.x;
  curandState local_state;
  const T noise_std_dw = dw_min_std;
  if (noise_std_dw > 0)
    local_state = random_states[tid];

  const int xsz = x_size;
  const int dsz = d_size;
  const int sz = size_in;
  const int nK32 = nK32_in;
  count_t *xptr, *dptr;

  T w;
  float4 par_4;
  float2 par_2;
  T par_1; // dummy
  UpdateFunctorConstantStep<T> up_fun;

  const int x_count_offset = xsz * nK32;
  const int d_count_offset = dsz * nK32;

  for (int i_stride = 0; i_stride < sz; i_stride += total_threads) {

    int idx = tid + i_stride;

    if (idx < sz) { // stride over all elements of W

      int dIdx = idx % dsz; // first d
      int xIdx = idx / dsz;

      par_4 = reinterpret_cast<float4 *>(params)[idx];
      w = weights[idx];

      uint32_t sum_n = 0;
      uint32_t last_negative = 0;

      for (int i_batch = 0; i_batch < m_batch; i_batch++) {

        xptr = &x_counts[getIdxToLoad<x_trans>(i_batch, xIdx, xsz, m_batch, x_count_offset)];
        dptr = &d_counts[getIdxToLoad<d_trans>(i_batch, dIdx, dsz, m_batch, d_count_offset)];

        bool mixed;
        uint32_t negative;
        uint32_t n;

        getNfromCount<one_sided, count_t>(n, negative, mixed, xptr, dptr, nK32, xsz, dsz);

        RPU_UPDATE_WITH_SUM_N_INNER();

      } // batch
      // last update
      if (sum_n > 0) {
        up_fun(w, sum_n, last_negative, par_4, par_2, par_1, nullptr, noise_std_dw, local_state);
      }

      weights[idx] = w;
    }
  }
  if (noise_std_dw > 0)
    random_states[tid] = local_state;
}

template <typename T, int one_sided, typename count_t, bool x_trans, bool d_trans>
__global__ void kernelUpdateWBatchSumBoundCheck(
    T *weights,
    int size_in,
    count_t *x_counts,
    int x_size,
    count_t *d_counts,
    int d_size,
    float *params,
    int nK32_in,
    int m_batch_in,
    const T dw_min_std,
    curandState *random_states,
    kagg_t *Kn = nullptr) {
  int m_batch = m_batch_in;
  if (Kn != nullptr) {
    m_batch = ((*Kn) + 31) / 32; // overwrite m_batch in case of BO64 UBLM
  }

  volatile unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
  const int total_threads = blockDim.x * gridDim.x;
  curandState local_state;
  const T noise_std_dw = dw_min_std;
  if (noise_std_dw > 0)
    local_state = random_states[tid];

  const int xsz = x_size;
  const int dsz = d_size;
  const int sz = size_in;
  const int nK32 = nK32_in;
  count_t *xptr, *dptr;

  T w;
  float4 par_4;
  float2 par_2; // dummy
  T par_1;      // dummy
  UpdateFunctorConstantStep<T> up_fun;

  const int x_count_offset = xsz * nK32;
  const int d_count_offset = dsz * nK32;

  for (int i_stride = 0; i_stride < sz; i_stride += total_threads) {

    int idx = tid + i_stride;

    if (idx < sz) { // stride over all elements of W

      int dIdx = idx % dsz; // first d
      int xIdx = idx / dsz;

      par_4 = reinterpret_cast<float4 *>(params)[idx];
      w = weights[idx];

      uint32_t sum_n = 0;
      uint32_t last_negative = 0;

      for (int i_batch = 0; i_batch < m_batch; i_batch++) {

        xptr = &x_counts[getIdxToLoad<x_trans>(i_batch, xIdx, xsz, m_batch, x_count_offset)];
        dptr = &d_counts[getIdxToLoad<d_trans>(i_batch, dIdx, dsz, m_batch, d_count_offset)];

        bool mixed;
        uint32_t n;
        uint32_t negative;

        getNfromCount<one_sided, count_t>(n, negative, mixed, xptr, dptr, nK32, xsz, dsz);

        RPU_UPDATE_WITH_SUM_N_INNER_BOUND_CHECK;

      } // batch
      // last update
      if (sum_n > 0) {
        up_fun(w, sum_n, last_negative, par_4, par_2, par_1, nullptr, noise_std_dw, local_state);
      }

      weights[idx] = w;
    }
  }
  if (noise_std_dw > 0)
    random_states[tid] = local_state;
}

template <
    typename T,
    int one_sided,
    typename count_t,
    bool x_trans,
    bool d_trans,
    typename UpdateFunctor,
    int global_params_count = 1,
    typename std::enable_if<(global_params_count > 0), int>::type = 0>
__global__ void kernelUpdateWBatchFunctor(
    T *weights,
    int size_in,
    count_t *x_counts,
    int x_size,
    count_t *d_counts,
    int d_size,
    float *params,
    float *params_2,
    T *params_1,
    T *global_params,
    int nK32_in,
    int m_batch_in,
    const T dw_min_std,
    curandState *random_states,
    kagg_t *Kn = nullptr) {

  int m_batch = m_batch_in;
  if (Kn != nullptr) {
    m_batch = ((*Kn) + 31) / 32; // overwrite m_batch in case of BO64 UBLM
  }

  volatile unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
  const int total_threads = blockDim.x * gridDim.x;
  curandState local_state;
  const T noise_std_dw = dw_min_std;
  if (noise_std_dw > 0)
    local_state = random_states[tid];

  const int xsz = x_size;
  const int dsz = d_size;
  const int sz = size_in;
  const int nK32 = nK32_in;
  count_t *xptr, *dptr;

  RPU_FUNCTOR_INIT_VARS;

  const int x_count_offset = xsz * nK32;
  const int d_count_offset = dsz * nK32;

  for (int i_stride = 0; i_stride < sz; i_stride += total_threads) {

    int idx = tid + i_stride;

    if (idx < sz) { // stride over all elements of W

      int dIdx = idx % dsz; // first d
      int xIdx = idx / dsz;

      // first load the parameters (same for each batch)
      RPU_FUNCTOR_LOAD_PARAMS;

      for (int i_batch = 0; i_batch < m_batch; i_batch++) {

        xptr = &x_counts[getIdxToLoad<x_trans>(i_batch, xIdx, xsz, m_batch, x_count_offset)];
        dptr = &d_counts[getIdxToLoad<d_trans>(i_batch, dIdx, dsz, m_batch, d_count_offset)];

        bool mixed;
        uint32_t negative;
        uint32_t n;

        getNfromCount<one_sided, count_t>(n, negative, mixed, xptr, dptr, nK32, xsz, dsz);

        if (mixed) {
          // not n/negative are bit strings
          PRAGMA(unroll)
          for (int i_bit = 0; i_bit < 32; i_bit++) {
            uint32_t bit_n = testBit(n, i_bit) ? 1 : 0;
            if (bit_n != 0) {
              uint32_t bit_neg = testBit(negative, i_bit) ? 1 : 0;
              up_fun(w, 1, bit_neg, par_4, par_2, par_1, global_par, noise_std_dw, local_state);
            }
          }

        } else {
          // now n is count
          if (n > 0) {
            up_fun(w, n, negative, par_4, par_2, par_1, global_par, noise_std_dw, local_state);
          }
        }
      }
      weights[idx] = w;
      if (use_par_1) {
        params_1[idx] = par_1;
      }
    }
  }
  if (noise_std_dw > 0)
    random_states[tid] = local_state;
}

/*********************************************************************************/
/*********************************************************************************/
// shared memory versions
#define D_BLOCK_SIZE 32
#define D_BLOCK_SIZE_BITS 5

#define RPU_UWBS_DEF_AND_STRIDE_LOOP(NOISEIF)                                                      \
  const int batch_stride = batch_stride_in;                                                        \
  const int xsz = x_size;                                                                          \
  const int dsz = d_size;                                                                          \
  const int nK32 = nK32_in;                                                                        \
  const int x_block_size = (blockDim.x >> D_BLOCK_SIZE_BITS);                                      \
  const int load_d_offset = (nK32 << D_BLOCK_SIZE_BITS);                                           \
                                                                                                   \
  const int x_memoffset = load_d_offset * batch_stride;                                            \
  const int x_count_offset = xsz * nK32;                                                           \
  const int d_count_offset = dsz * nK32;                                                           \
                                                                                                   \
  const int d_sub_idx = (threadIdx.x & (D_BLOCK_SIZE - 1));                                        \
  const int x_sub_idx = (threadIdx.x >> D_BLOCK_SIZE_BITS);                                        \
  const int x_sub_idx_load = (threadIdx.x % x_block_size);                                         \
  const int d_load_batch_idx = (threadIdx.x >> D_BLOCK_SIZE_BITS);                                 \
  const int x_load_batch_idx = (threadIdx.x / x_block_size);                                       \
                                                                                                   \
  curandState local_state;                                                                         \
  const T noise_std_dw = dw_min_std;                                                               \
  bool noiseif = NOISEIF;                                                                          \
  const int tid = blockDim.x * blockIdx.x + threadIdx.x;                                           \
  if (noiseif)                                                                                     \
    local_state = random_states[tid];                                                              \
                                                                                                   \
  const int load_x_offset = nK32 * x_block_size;                                                   \
  const int n_block_threads = blockDim.x * gridDim.x;                                              \
                                                                                                   \
  const int num_d_blocks = (dsz + D_BLOCK_SIZE - 1) >> D_BLOCK_SIZE_BITS;                          \
  const int num_x_blocks = (xsz + x_block_size - 1) / x_block_size;                                \
                                                                                                   \
  const int total_tid = num_d_blocks * num_x_blocks * blockDim.x;                                  \
                                                                                                   \
  for (int i_tid_stride = 0; i_tid_stride < total_tid; i_tid_stride += n_block_threads) {          \
                                                                                                   \
    int d_index = 0;                                                                               \
    int x_index = 0;                                                                               \
    int x_block_start = 0;                                                                         \
                                                                                                   \
    {                                                                                              \
      const int bid = blockIdx.x + i_tid_stride / blockDim.x;                                      \
      const int d_block_idx = bid % num_d_blocks;                                                  \
      const int x_block_idx = bid / num_d_blocks;                                                  \
                                                                                                   \
      d_index = (d_block_idx << D_BLOCK_SIZE_BITS) + d_sub_idx;                                    \
      x_block_start = x_block_size * x_block_idx;                                                  \
      x_index = x_block_start + x_sub_idx;                                                         \
                                                                                                   \
      if (x_block_start >= xsz) {                                                                  \
        if ((i_tid_stride > 0) && noiseif)                                                         \
          random_states[tid] = local_state;                                                        \
        return;                                                                                    \
      }                                                                                            \
    }                                                                                              \
    int idx = x_index * dsz + d_index;                                                             \
    bool within_range = ((x_index < xsz) && (d_index < dsz));

#define RPU_UWBS_READ_INTO_SHARED                                                                  \
  {                                                                                                \
                                                                                                   \
    __syncthreads();                                                                               \
                                                                                                   \
    int n_load_batch = blockDim.x / D_BLOCK_SIZE;                                                  \
    for (int i_load_stride = 0; i_load_stride < batch_stride; i_load_stride += n_load_batch) {     \
                                                                                                   \
      if (d_index < dsz) {                                                                         \
                                                                                                   \
        const int i_batch = d_load_batch_idx + i_load_stride;                                      \
                                                                                                   \
        if ((i_batch + i_stride < m_batch) && (i_batch < batch_stride)) {                          \
                                                                                                   \
          int d_index_load_batch =                                                                 \
              getIdxToLoad<d_trans>(i_batch + i_stride, d_index, dsz, m_batch, d_count_offset);    \
                                                                                                   \
          const int d_shared_load_index = load_d_offset * i_batch + d_sub_idx;                     \
                                                                                                   \
          int offset = 0;                                                                          \
          int org_offset = 0;                                                                      \
          PRAGMA(unroll)                                                                           \
          for (int j = 0; j < nK32; j++) {                                                         \
            shared_d_and_x_counts[d_shared_load_index + offset] =                                  \
                d_counts[d_index_load_batch + org_offset];                                         \
            offset += D_BLOCK_SIZE;                                                                \
            org_offset += dsz;                                                                     \
          }                                                                                        \
        }                                                                                          \
      }                                                                                            \
    }                                                                                              \
                                                                                                   \
    n_load_batch = blockDim.x / x_block_size;                                                      \
                                                                                                   \
    const int x_index_load = x_block_start + (x_sub_idx_load);                                     \
    for (int i_load_stride = 0; i_load_stride < batch_stride; i_load_stride += n_load_batch) {     \
                                                                                                   \
      if (x_index_load < xsz) {                                                                    \
        const int i_batch = x_load_batch_idx + i_load_stride;                                      \
                                                                                                   \
        if ((i_batch + i_stride < m_batch) && (i_batch < batch_stride)) {                          \
                                                                                                   \
          int x_index_load_batch = getIdxToLoad<x_trans>(                                          \
              i_batch + i_stride, x_index_load, xsz, m_batch, x_count_offset);                     \
          const int x_shared_load_index = x_memoffset + load_x_offset * i_batch + x_sub_idx_load;  \
          int offset = 0;                                                                          \
          int org_offset = 0;                                                                      \
          PRAGMA(unroll)                                                                           \
          for (int j = 0; j < nK32; j++) {                                                         \
            shared_d_and_x_counts[x_shared_load_index + offset] =                                  \
                x_counts[x_index_load_batch + org_offset];                                         \
            offset += x_block_size;                                                                \
            org_offset += xsz;                                                                     \
          }                                                                                        \
        }                                                                                          \
      }                                                                                            \
    }                                                                                              \
  }                                                                                                \
                                                                                                   \
  __syncthreads();

#define RPU_UWBS_LOAD_COUNTS_N(COUNT_T)                                                            \
                                                                                                   \
  int d_shared_index = d_sub_idx + d_shared_offset;                                                \
  int x_shared_index = x_sub_idx + x_shared_offset;                                                \
  d_shared_offset += load_d_offset;                                                                \
  x_shared_offset += load_x_offset;                                                                \
                                                                                                   \
  uint32_t n = 0;                                                                                  \
  bool mixed = false;                                                                              \
  getNfromCount<one_sided, COUNT_T>(                                                               \
      n, negative, mixed, shared_d_and_x_counts + x_shared_index,                                  \
      shared_d_and_x_counts + d_shared_index, nK32, x_block_size, D_BLOCK_SIZE);

#define RPU_UWBS_CLOSE_STRIDE_LOOP                                                                 \
  }                                                                                                \
                                                                                                   \
  if (noiseif)                                                                                     \
    random_states[tid] = local_state;

/*********************************************************************************/
// fast specialized version for standard pulsed update
template <typename T, int one_sided, typename count_t, bool x_trans, bool d_trans>
__global__ void kernelUpdateWBatchSharedSum(
    T *weights,
    int size_in,
    count_t *x_counts,
    int x_size,
    count_t *d_counts,
    int d_size,
    float *params,
    int nK32_in,
    int m_batch_in,
    int batch_stride_in,
    const T dw_min_std,
    curandState *random_states,
    kagg_t *Kn = nullptr) {

  // -- params expected in the following order:  (min_bound, scale_down, max_bound, scale_up )
  // -- threadIdx.x will be distributed in 32 d-values rest x values.   (threadIdx.y==1)
  // -- blockDim.x like wise cover W. (blockIdx.y==1)
  // -- batch is strided. (need batch_stride * sizeof(int) * nK32 * (32 + blockDim.x/32) memory )
  // -- shared memory is used to save x and d values
  // -- nthreads needs to be divisible by 32 (D_BLOCK_SIZE)!!

  extern __shared__ __align__(sizeof(uint64_t)) uint32_t shared_d_and_x_counts_32[];
  count_t *shared_d_and_x_counts = reinterpret_cast<count_t *>(shared_d_and_x_counts_32);
  int m_batch = m_batch_in;
  if (Kn != nullptr) {
    m_batch = ((*Kn) + 31) / 32; // overwrite m_batch in case of BO64 UBLM
  }

  T w = 0;
  float4 par_4;
  float2 par_2; // dummy
  T par_1;      // dummy
  UpdateFunctorConstantStep<T> up_fun;

  RPU_UWBS_DEF_AND_STRIDE_LOOP(noise_std_dw > 0);

  if (within_range) {
    par_4 = reinterpret_cast<float4 *>(params)[idx];
    w = weights[idx];
  }

  for (int i_stride = 0; i_stride < m_batch; i_stride += batch_stride) { // for batch loading

    RPU_UWBS_READ_INTO_SHARED;

    if (within_range) {

      // loop over all batches in the batch_stride
      int d_shared_offset = 0;
      int x_shared_offset = x_memoffset;
      uint32_t negative = 0;
      uint32_t sum_n = 0;
      uint32_t last_negative = 0;

      for (int i_batch = 0; (i_batch < batch_stride) && (i_batch + i_stride < m_batch); i_batch++) {

        RPU_UWBS_LOAD_COUNTS_N(count_t);

        RPU_UPDATE_WITH_SUM_N_INNER();

      } // batch
      // last update
      if (sum_n > 0) {
        up_fun(w, sum_n, last_negative, par_4, par_2, par_1, nullptr, noise_std_dw, local_state);
      }

    } // within range
  }   // batch strides
  if (within_range) {
    weights[idx] = w;
  }

  RPU_UWBS_CLOSE_STRIDE_LOOP;
}

template <typename T, int one_sided, typename count_t, bool x_trans, bool d_trans>
__global__ void kernelUpdateWBatchSharedSumBoundCheck(
    T *weights,
    int size_in,
    count_t *x_counts,
    int x_size,
    count_t *d_counts,
    int d_size,
    float *params,
    int nK32_in,
    int m_batch_in,
    int batch_stride_in,
    const T dw_min_std,
    curandState *random_states,
    kagg_t *Kn = nullptr) {

  // -- same as kernelUpdateWBatchShared but with additional bound check

  extern __shared__ __align__(sizeof(uint64_t)) uint32_t shared_d_and_x_counts_32[];
  count_t *shared_d_and_x_counts = reinterpret_cast<count_t *>(shared_d_and_x_counts_32);
  int m_batch = m_batch_in;
  if (Kn != nullptr) {
    m_batch = ((*Kn) + 31) / 32; // overwrite m_batch in case of BO64 UBLM
  }

  T w = 0;
  float4 par_4;
  float2 par_2;
  T par_1; // dummy
  UpdateFunctorConstantStep<T> up_fun;

  RPU_UWBS_DEF_AND_STRIDE_LOOP(noise_std_dw > 0);

  if (within_range) {
    par_4 = reinterpret_cast<float4 *>(params)[idx];
    w = weights[idx];
  }

  for (int i_stride = 0; i_stride < m_batch; i_stride += batch_stride) { // for batch loading

    RPU_UWBS_READ_INTO_SHARED;

    if (within_range) {

      // loop over all batches in the batch_stride
      int d_shared_offset = 0;
      int x_shared_offset = x_memoffset;
      uint32_t negative = 0;
      uint32_t sum_n = 0;
      uint32_t last_negative = 0;

      for (int i_batch = 0; (i_batch < batch_stride) && (i_batch + i_stride < m_batch); i_batch++) {

        RPU_UWBS_LOAD_COUNTS_N(count_t);

        RPU_UPDATE_WITH_SUM_N_INNER_BOUND_CHECK;

      } // batch
      // last update
      if (sum_n > 0) {
        up_fun(w, sum_n, last_negative, par_4, par_2, par_1, nullptr, noise_std_dw, local_state);
      }

    } // within range
  }   // batch strides

  if (within_range) {
    weights[idx] = w;
  }

  RPU_UWBS_CLOSE_STRIDE_LOOP;
}

/*********************************************************************************/
// general shared version for functor update

template <
    typename T,
    int one_sided,
    typename count_t,
    bool x_trans,
    bool d_trans,
    typename UpdateFunctor,
    int global_params_count = 1,
    typename std::enable_if<(global_params_count > 0), int>::type = 0>
__global__ void kernelUpdateWBatchSharedFunctor(
    T *weights,
    int size_in,
    count_t *x_counts,
    int x_size,
    count_t *d_counts,
    int d_size,
    float *params,
    float *params_2,
    T *params_1,
    T *global_params,
    int nK32_in,
    int m_batch_in,
    int batch_stride_in,
    const T dw_min_std,
    curandState *random_states,
    kagg_t *Kn = nullptr) {
  // -- same as kernelUpdateWBatchShared but with simple loop.
  // -- Functor needs to be specified
  // -- params expected in the following order:  (min_bound, dw_down, max_bound, dw_up )
  // -- threadIdx.x will be distributed in 32 d-values rest x values.   (threadIdx.y==1)
  // -- blockDim.x like wise cover W. (blockIdx.y==1)
  // -- batch is strided. (need batch_stride * sizeof(int) * nK32 * (32 + blockDim.x/32) memory )
  // -- shared memory is used to save x and d values
  // -- nthreads needs to be divisible by 32 (D_BLOCK_SIZE)!!

  extern __shared__ __align__(sizeof(uint64_t)) uint32_t shared_d_and_x_counts_32[];
  count_t *shared_d_and_x_counts = reinterpret_cast<count_t *>(shared_d_and_x_counts_32);
  int m_batch = m_batch_in;
  if (Kn != nullptr) {
    m_batch = ((*Kn) + 31) / 32; // overwrite m_batch in case of BO64 UBLM
  }

  RPU_FUNCTOR_INIT_VARS;

  RPU_UWBS_DEF_AND_STRIDE_LOOP(noise_std_dw > 0);

  if (within_range) {
    RPU_FUNCTOR_LOAD_PARAMS;
  }

  for (int i_stride = 0; i_stride < m_batch; i_stride += batch_stride) { // for batch loading

    RPU_UWBS_READ_INTO_SHARED;

    if (within_range) {

      // loop over all batches in the batch_stride
      int d_shared_offset = 0;
      int x_shared_offset = x_memoffset;
      uint32_t negative = 0;

      for (int i_batch = 0; (i_batch < batch_stride) && (i_batch + i_stride < m_batch); i_batch++) {

        RPU_UWBS_LOAD_COUNTS_N(count_t);

        if (mixed) {

          PRAGMA(unroll)
          for (int i_bit = 0; i_bit < 32; i_bit++) {
            uint32_t bit_n = testBit(n, i_bit) ? 1 : 0;
            if (bit_n != 0) {
              uint32_t bit_neg = testBit(negative, i_bit) ? 1 : 0;
              up_fun(w, 1, bit_neg, par_4, par_2, par_1, global_par, noise_std_dw, local_state);
            }
          }
        } else { // not mixed
          // now n is already the number n

          if (n > 0) {
            up_fun(w, n, negative, par_4, par_2, par_1, global_par, noise_std_dw, local_state);
          }
        }
      } // batch

    } // within range
  }   // batch strides

  if (within_range) {
    weights[idx] = w;
    if (use_par_1) {
      params_1[idx] = par_1;
    }
  }

  RPU_UWBS_CLOSE_STRIDE_LOOP;
}

} // namespace RPU

#undef RPU_FUNCTOR_LOAD_PARAMS
#undef RPU_UPDATE_WITH_SUM_N_INNER
#undef RPU_UPDATE_WITH_SUM_N_INNER_BOUND_CHECK
#undef RPU_UWBS_DEF_AND_STRIDE_LOOP
#undef RPU_UWBS_READ_INTO_SHARED
#undef RPU_UWBS_LOAD_COUNTS_N
#undef RPU_UWBS_CLOSE_STRIDE_LOOP
#undef RPU_FUNCTOR_INIT_VARS
#undef D_BLOCK_SIZE
#undef D_BLOCK_SIZE_BITS
