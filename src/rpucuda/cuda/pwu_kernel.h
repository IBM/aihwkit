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

#pragma once

#include "cuda_fp16_util.h"
#include "cuda_math_util.h"

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

 The non-trans count ordering is (k: nK32, x: x_size, b: m_batch)
   x  0 1 2 0 1 2 0 1 2 0 1 2
   k  0 0 0 1 1 1 0 0 0 1 1 1
   b  0 0 0 0 0 0 1 1 1 1 1 1

 The trans version is (second k is always + x_size ):
   x  0 0 1 0 0 1 1 2 2 1 2 2
   k  0 0 0 1 1 1 0 0 0 1 1 1
   b  0 1 0 0 1 0 1 0 1 1 0 1


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
    int shared_d_offset,
    bool enforce_mixed = false); // enforces mixed if possible in the format (only U64)

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
      int shared_x_offset, int shared_d_offset, bool enforce_mixed) {                              \
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
      int shared_x_offset, int shared_d_offset, bool enforce_mixed) {                              \
    FPTYPE x = *x_ptr;                                                                             \
    FPTYPE d = *d_ptr;                                                                             \
                                                                                                   \
    /* never mixed pos/neg within one read in this format*/                                        \
    mixed = false;                                                                                 \
    negative = ((x < (FPTYPE)0.0) != (d < (FPTYPE)0.0)) ? 1 : 0;                                   \
    if ((x == (FPTYPE)0.0) || (d == (FPTYPE)0.0)) {                                                \
      n = 0;                                                                                       \
      return;                                                                                      \
    };                                                                                             \
                                                                                                   \
    OS_ADD                                                                                         \
                                                                                                   \
    n = (uint32_t)fabs(RPU_ROUNDFUN(d * x));                                                       \
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
#ifdef RPU_USE_FP16
DEFINE_GETNFROMCOUNTFP(half_t, 0, );
DEFINE_GETNFROMCOUNTFP(
    half_t,
    -1,
    if (negative) {
      n = 0;
      return;
    } else { negative = 1; });
DEFINE_GETNFROMCOUNTFP(
    half_t, 1, if (!negative) {
      n = 0;
      return;
    });
#endif

#undef DEFINE_GETNFROMCOUNTFP

#define DEFINE_GETNFROMCOUNT64(ONE_SIDED, OS_ADD)                                                  \
  template <>                                                                                      \
  __device__ __forceinline__ void getNfromCount<ONE_SIDED COMMA uint64_t>(                         \
      uint32_t & n, uint32_t & negative, bool &mixed, uint64_t *x_ptr, uint64_t *d_ptr, int nK32,  \
      int shared_x_offset, int shared_d_offset, bool enforce_mixed) {                              \
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
    if (!enforce_mixed && (all_negative || all_positive)) {                                        \
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
  param4_t par_4;                                                                                  \
  param2_t par_2;                                                                                  \
  T par_1 = 0;                                                                                     \
  bool use_par_1 = params_1 != nullptr;                                                            \
  UpdateFunctor up_fun;                                                                            \
  __shared__ T global_par[global_params_count];                                                    \
  if (global_params != nullptr) {                                                                  \
    for (int gidx = threadIdx.x; gidx < global_params_count; gidx += blockDim.x) {                 \
      global_par[gidx] = global_params[gidx];                                                      \
    }                                                                                              \
    __syncthreads();                                                                               \
  }

#define RPU_FUNCTOR_LOAD_PARAMS                                                                    \
  {                                                                                                \
    w = weights[idx];                                                                              \
    if (params != nullptr) {                                                                       \
      par_4 = reinterpret_cast<param4_t *>(params)[idx];                                           \
    }                                                                                              \
    if (params_2 != nullptr) {                                                                     \
      par_2 = reinterpret_cast<param2_t *>(params_2)[idx];                                         \
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
      const param4_t par_4,
      const param2_t par_2,
      T &par_1,
      const T *global_par,
      const int global_params_count,
      T noise_std_dw,
      curandState &local_state) {

    // note that only w and par_1 will be written back when used. Thus it can be a "hidden_weights"
    // type note that we here assume that stoch_value is < 1, or if larger, then it did not hit the
    // bound.
    UNUSED(global_params_count);
    UNUSED(global_par);
    UNUSED(par_1);
    UNUSED(par_2);

    T dw = (negative > 0) ? ((T)par_4.w) : (-(T)par_4.y);
    T wmax = (T)par_4.z;
    T wmin = (T)par_4.x;
    T sigma = noise_std_dw;
    // n is larger 0 in any case
    if (n == 1) {
      if (sigma > (T)0.0) {
        T stoch_value = (T)curand_normal(&local_state);
        stoch_value *= sigma;
        w += dw * ((T)1.0 + stoch_value);
      } else {
        w += dw;
      }
    } else {
      if (sigma > (T)0.0) {
        T stoch_value = (T)curand_normal(&local_state);
        stoch_value *= sigma;
        w += dw * (T)n * ((T)1.0 + rsqrt((T)n) * stoch_value); // rsqrt(x) = 1/sqrt(x) is faster
      } else {
        w += dw * (T)n;
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
    count_t *x_counts,
    int x_size,
    count_t *d_counts,
    int d_size,
    param_t *params,
    param_t *params_2,
    T *params_1,
    T *global_params,
    int nK32in,
    T dw_min_std,
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

  if (dw_min_std > (T)0.0 && tid < sz) {
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
        up_fun(
            w, n, negative, par_4, par_2, par_1, global_par, global_params_count, noise_std_dw,
            local_state);
        weights[idx] = w;
        if (use_par_1) {
          params_1[idx] = par_1;
        }
      }
    }
  }
  if (dw_min_std > (T)0.0 && tid < sz) {
    random_states[tid] = local_state;
  }
}

/*********************************************************************************/
/*********************************************************************************/
// non shared batch versions

#define RPU_UPDATE_WITH_SUM_N_INNER(BOUND_CHECK_BODY)                                              \
  if (mixed) {                                                                                     \
    BOUND_CHECK_BODY {                                                                             \
      for (int i_bit = 0; i_bit < 32; i_bit++) {                                                   \
        uint32_t bit_n = testBit(n, i_bit) ? 1 : 0;                                                \
        if (bit_n != 0) {                                                                          \
          uint32_t bit_neg = testBit(negative, i_bit) ? 1 : 0;                                     \
          if (bit_neg == last_negative) {                                                          \
            sum_n += 1;                                                                            \
          } else {                                                                                 \
            if (sum_n > 0) {                                                                       \
              up_fun(                                                                              \
                  w, sum_n, last_negative, par_4, par_2, par_1, nullptr, 0, noise_std_dw,          \
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
        up_fun(                                                                                    \
            w, sum_n, last_negative, par_4, par_2, par_1, nullptr, 0, noise_std_dw, local_state);  \
      }                                                                                            \
      sum_n = n;                                                                                   \
      last_negative = negative;                                                                    \
    }                                                                                              \
  }

#define RPU_UPDATE_WITH_SUM_N_INNER_BOUND_CHECK                                                    \
  RPU_UPDATE_WITH_SUM_N_INNER(                                                                     \
      if (sum_n > 0) {                                                                             \
        up_fun(                                                                                    \
            w, sum_n, last_negative, par_4, par_2, par_1, nullptr, 0, noise_std_dw, local_state);  \
      } sum_n = 0;                                                                                 \
      last_negative = 0;                                                                           \
                                                                                                   \
      int pos_n = __popc((~negative) & n); int neg_n = __popc((negative)&n); T dw_pos = (T)pos_n;  \
      T dw_neg = (T)neg_n;                                                                         \
                                                                                                   \
      if (noise_std_dw > (T)0.0) {                                                                 \
        if (pos_n > 0) {                                                                           \
          T stoch_value = (T)curand_normal(&local_state);                                          \
          dw_pos += sqrt(dw_pos) * noise_std_dw;                                                   \
        }                                                                                          \
        if (neg_n > 0) {                                                                           \
          T stoch_value = (T)curand_normal(&local_state);                                          \
          dw_neg += sqrt(dw_neg) * noise_std_dw;                                                   \
        }                                                                                          \
      } dw_pos *= (T)par_4.y;                                                                      \
      dw_neg *= (T)par_4.w;                                                                        \
                                                                                                   \
      if ((w - dw_pos >= (T)par_4.x) && (w + dw_neg <= (T)par_4.z)) {                              \
        w -= dw_pos - dw_neg;                                                                      \
      } else)

template <typename T, int one_sided, typename count_t, bool x_trans, bool d_trans>
__global__ void kernelUpdateWBatchSum(
    T *weights,
    count_t *x_counts,
    int x_size,
    count_t *d_counts,
    int d_size,
    param_t *params,
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
  if (noise_std_dw > (T)0.0)
    local_state = random_states[tid];

  const int xsz = x_size;
  const int dsz = d_size;
  const int sz = xsz * dsz;
  const int nK32 = nK32_in;
  count_t *xptr, *dptr;

  T w;
  param4_t par_4;
  param2_t par_2;
  T par_1; // dummy
  UpdateFunctorConstantStep<T> up_fun;

  const int x_count_offset = xsz * nK32;
  const int d_count_offset = dsz * nK32;

  for (int i_stride = 0; i_stride < sz; i_stride += total_threads) {

    int idx = tid + i_stride;

    if (idx < sz) { // stride over all elements of W

      int dIdx = idx % dsz; // first d
      int xIdx = idx / dsz;

      par_4 = reinterpret_cast<param4_t *>(params)[idx];
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

        RPU_UPDATE_WITH_SUM_N_INNER({});

      } // batch
      // last update
      if (sum_n > 0) {
        up_fun(w, sum_n, last_negative, par_4, par_2, par_1, nullptr, 0, noise_std_dw, local_state);
      }

      weights[idx] = w;
    }
  }
  if (noise_std_dw > (T)0.0)
    random_states[tid] = local_state;
}

template <typename T, int one_sided, typename count_t, bool x_trans, bool d_trans>
__global__ void kernelUpdateWBatchSumBoundCheck(
    T *weights,
    count_t *x_counts,
    int x_size,
    count_t *d_counts,
    int d_size,
    param_t *params,
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
  if (noise_std_dw > (T)0.0)
    local_state = random_states[tid];

  const int xsz = x_size;
  const int dsz = d_size;
  const int sz = xsz * dsz;
  const int nK32 = nK32_in;
  count_t *xptr, *dptr;

  T w;
  param4_t par_4;
  param2_t par_2; // dummy
  T par_1;        // dummy
  UpdateFunctorConstantStep<T> up_fun;

  const int x_count_offset = xsz * nK32;
  const int d_count_offset = dsz * nK32;

  for (int i_stride = 0; i_stride < sz; i_stride += total_threads) {

    int idx = tid + i_stride;

    if (idx < sz) { // stride over all elements of W

      int dIdx = idx % dsz; // first d
      int xIdx = idx / dsz;

      par_4 = reinterpret_cast<param4_t *>(params)[idx];
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
        up_fun(w, sum_n, last_negative, par_4, par_2, par_1, nullptr, 0, noise_std_dw, local_state);
      }

      weights[idx] = w;
    }
  }
  if (noise_std_dw > (T)0.0)
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
    count_t *x_counts,
    int x_size,
    count_t *d_counts,
    int d_size,
    param_t *params,
    param_t *params_2,
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
  if (noise_std_dw > (T)0.0)
    local_state = random_states[tid];

  const int xsz = x_size;
  const int dsz = d_size;
  const int sz = xsz * dsz;
  const int nK32 = nK32_in;
  count_t *xptr, *dptr;

  RPU_FUNCTOR_INIT_VARS;

  const int x_count_offset = xsz * nK32;
  const int d_count_offset = dsz * nK32;

  for (int idx = tid; idx < sz; idx += total_threads) {

    int dIdx = idx % dsz; // first d
    int xIdx = idx / dsz;

    // first load the parameters (same for each batch)
    RPU_FUNCTOR_LOAD_PARAMS;

    for (int i_batch = 0; i_batch < m_batch; i_batch++) {

      xptr = &x_counts[getIdxToLoad<x_trans>(i_batch, xIdx, xsz, m_batch, x_count_offset)];
      dptr = &d_counts[getIdxToLoad<d_trans>(i_batch, dIdx, dsz, m_batch, d_count_offset)];

      bool mixed = false;
      uint32_t negative = 0;
      uint32_t n = 0;

      getNfromCount<one_sided, count_t>(n, negative, mixed, xptr, dptr, nK32, xsz, dsz);
      if (mixed) {
        // now n/negative are bit strings
        for (int i_bit = 0; i_bit < 32; i_bit++) {
          uint32_t bit_n = testBit(n, i_bit) ? 1 : 0;
          if (bit_n != 0) {
            uint32_t bit_neg = testBit(negative, i_bit) ? 1 : 0;
            up_fun(
                w, 1, bit_neg, par_4, par_2, par_1, global_par, global_params_count, noise_std_dw,
                local_state);
          }
        }
      } else {
        // now n is count
        if (n > 0) {
          up_fun(
              w, n, negative, par_4, par_2, par_1, global_par, global_params_count, noise_std_dw,
              local_state);
        }
      }
    }
    weights[idx] = w;
    if (use_par_1) {
      params_1[idx] = par_1;
    }
  }
  if (noise_std_dw > (T)0.0)
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
  if (noiseif) {                                                                                   \
    local_state = random_states[tid];                                                              \
  }                                                                                                \
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
          for (int j = 0; j < nK32; j++) {                                                         \
            shared_d_and_x_counts[x_shared_load_index + offset] =                                  \
                x_counts[x_index_load_batch + org_offset];                                         \
            offset += x_block_size;                                                                \
            org_offset += xsz;                                                                     \
          }                                                                                        \
        }                                                                                          \
      }                                                                                            \
    }                                                                                              \
  }

#define RPU_UWBS_LOAD_COUNTS_N(COUNT_T, ENFORCE_MIXED)                                             \
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
      shared_d_and_x_counts + d_shared_index, nK32, x_block_size, D_BLOCK_SIZE, ENFORCE_MIXED);

#define RPU_UWBS_CLOSE_STRIDE_LOOP                                                                 \
  }                                                                                                \
                                                                                                   \
  if (noiseif) {                                                                                   \
    random_states[tid] = local_state;                                                              \
  }

/*********************************************************************************/
// fast specialized version for standard pulsed update
template <typename T, int one_sided, typename count_t, bool x_trans, bool d_trans>
__global__ void kernelUpdateWBatchSharedSum(
    T *weights,
    count_t *x_counts,
    int x_size,
    count_t *d_counts,
    int d_size,
    param_t *params,
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
  param4_t par_4;
  param2_t par_2; // dummy
  T par_1;        // dummy
  UpdateFunctorConstantStep<T> up_fun;

  RPU_UWBS_DEF_AND_STRIDE_LOOP(noise_std_dw > (T)0.0);

  if (within_range) {
    par_4 = reinterpret_cast<param4_t *>(params)[idx];
    w = weights[idx];
  }

  for (int i_stride = 0; i_stride < m_batch; i_stride += batch_stride) { // for batch loading

    __syncthreads();
    RPU_UWBS_READ_INTO_SHARED;
    __syncthreads();

    if (within_range) {

      // loop over all batches in the batch_stride
      int d_shared_offset = 0;
      int x_shared_offset = x_memoffset;
      uint32_t negative = 0;
      uint32_t sum_n = 0;
      uint32_t last_negative = 0;

      for (int i_batch = 0; (i_batch < batch_stride) && (i_batch + i_stride < m_batch); i_batch++) {

        RPU_UWBS_LOAD_COUNTS_N(count_t, false);

        RPU_UPDATE_WITH_SUM_N_INNER({});

      } // batch
      // last update
      if (sum_n > 0) {
        up_fun(w, sum_n, last_negative, par_4, par_2, par_1, nullptr, 0, noise_std_dw, local_state);
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
    count_t *x_counts,
    int x_size,
    count_t *d_counts,
    int d_size,
    param_t *params,
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
  param4_t par_4;
  param2_t par_2;
  T par_1; // dummy
  UpdateFunctorConstantStep<T> up_fun;

  RPU_UWBS_DEF_AND_STRIDE_LOOP(noise_std_dw > (T)0.0);

  if (within_range) {
    par_4 = reinterpret_cast<param4_t *>(params)[idx];
    w = weights[idx];
  }

  for (int i_stride = 0; i_stride < m_batch; i_stride += batch_stride) { // for batch loading

    __syncthreads();
    RPU_UWBS_READ_INTO_SHARED;
    __syncthreads();

    if (within_range) {

      // loop over all batches in the batch_stride
      int d_shared_offset = 0;
      int x_shared_offset = x_memoffset;
      uint32_t negative = 0;
      uint32_t sum_n = 0;
      uint32_t last_negative = 0;

      for (int i_batch = 0; (i_batch < batch_stride) && (i_batch + i_stride < m_batch); i_batch++) {

        RPU_UWBS_LOAD_COUNTS_N(count_t, false);

        RPU_UPDATE_WITH_SUM_N_INNER_BOUND_CHECK;

      } // batch
      // last update
      if (sum_n > 0) {
        up_fun(w, sum_n, last_negative, par_4, par_2, par_1, nullptr, 0, noise_std_dw, local_state);
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
    count_t *x_counts,
    int x_size,
    count_t *d_counts,
    int d_size,
    param_t *params,
    param_t *params_2,
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

  RPU_UWBS_DEF_AND_STRIDE_LOOP(noise_std_dw > (T)0.0);

  if (within_range) {
    RPU_FUNCTOR_LOAD_PARAMS;
  }

  for (int i_stride = 0; i_stride < m_batch; i_stride += batch_stride) { // for batch loading

    __syncthreads();
    RPU_UWBS_READ_INTO_SHARED;
    __syncthreads();

    if (within_range) {

      // loop over all batches in the batch_stride
      int d_shared_offset = 0;
      int x_shared_offset = x_memoffset;
      uint32_t negative = 0;

      for (int i_batch = 0; (i_batch < batch_stride) && (i_batch + i_stride < m_batch); i_batch++) {

        RPU_UWBS_LOAD_COUNTS_N(count_t, false);

        if (mixed) {

          for (int i_bit = 0; i_bit < 32; i_bit++) {
            uint32_t bit_n = testBit(n, i_bit) ? 1 : 0;
            if (bit_n != 0) {
              uint32_t bit_neg = testBit(negative, i_bit) ? 1 : 0;
              up_fun(
                  w, 1, bit_neg, par_4, par_2, par_1, global_par, global_params_count, noise_std_dw,
                  local_state);
            }
          }
        } else { // not mixed
          // now n is already the number n

          if (n > 0) {
            up_fun(
                w, n, negative, par_4, par_2, par_1, global_par, global_params_count, noise_std_dw,
                local_state);
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

/*********************************************************************************/
// general shared version for functor update with intermediate weight output

// helper function: Test whether weight is to be stored
__device__ __forceinline__ bool isWeightOutputVal(
    const int x_index,
    const int d_index,
    const int i_weight_output,
    const int val_start,
    const bool wo_column,
    const int x_size,
    const int d_size) {
  if (wo_column) {
    // loops through x, all d
    int val_weight_output = (i_weight_output + val_start) % x_size;
    return x_index == val_weight_output;
  } else {
    // loops through d, all x
    int val_weight_output = (i_weight_output + val_start) % d_size;
    return d_index == val_weight_output;
  }
}
// helper function: whether after the current batch a weight output is needed
template <typename count_t>
__device__ __forceinline__ uint32_t isWeightOutputBatch(
    const int current_batch,
    const int wo_every,
    const int wo_batch_start,
    const int n_weight_outputs,
    const int K = 0,
    const uint32_t *weight_output_signals_shared = nullptr,
    const int i_load_batch = 0,
    const int org_m_batch = 0) {
  if (n_weight_outputs <= 0) {
    return 0;
  }
  return (wo_every > 0) ? ((current_batch + wo_batch_start + 1) % wo_every == 0) : 0;
}

template <>
__device__ __forceinline__ uint32_t isWeightOutputBatch<uint64_t>(
    const int current_batch,
    const int wo_every,
    const int wo_batch_start,
    const int n_weight_outputs,
    const int K,
    const uint32_t *weight_output_signals_shared,
    const int i_load_batch,
    const int org_m_batch) {
  if (n_weight_outputs <= 0) {
    return 0;
  }
  if (weight_output_signals_shared == nullptr) {
    // no ublm, but batch still in BO64 truncated format

    int i_org_batch = (current_batch * 32) / K; // org batch that might already have started earlier
    int bit_string_n = ((i_org_batch + 1) * K - 1); // up to end bit of current batch
    int iB = bit_string_n / 32;
    int end_pos = bit_string_n % 32;
    int iB_next = iB;
    uint32_t weight_output_signal = 0;

    while (i_org_batch < org_m_batch && (iB_next == iB)) {

      if (isWeightOutputBatch<uint32_t>(i_org_batch, wo_every, wo_batch_start, n_weight_outputs)) {
        weight_output_signal |= ((uint32_t)0x00000001 << end_pos);
      }

      i_org_batch += 1;
      bit_string_n += K;
      iB_next = bit_string_n / 32;
      end_pos = bit_string_n % 32;
    }
    return weight_output_signal;
  } else {
    return weight_output_signals_shared[i_load_batch];
  }
}

// helper function: gets the output index of the weight storage
__device__ __forceinline__ int getWeightOutputIdx(
    const int x_index,
    const int d_index,
    const int i_weight_output,
    const bool wo_column,
    const int x_size,
    const int d_size,
    const int n_weight_outputs,
    const int val_start,
    const bool flexible_in_size) {
  if (flexible_in_size) {
    // just col-major
    if (wo_column) {
      return d_index + d_size * i_weight_output;
    } else {
      return i_weight_output + n_weight_outputs * x_index;
    }
  } else {
    // size weight output resembles a weight matrix, we need d_size
    // major (col major) NOTE: the row/col is identical to the original
    // weight matrix. If wrapped around, then a new x_size*d_size chunk
    // is used

    if (wo_column) {
      int val_wo = (val_start + i_weight_output) % x_size;
      return d_index + d_size * (val_wo + i_weight_output / x_size * x_size);
    } else {
      int val_wo = (val_start + i_weight_output) % d_size;
      return val_wo + d_size * x_index + i_weight_output / d_size * d_size * x_size;
    }
  }
}

// helper to update the chopper quantities and counters
template <typename T>
__device__ __forceinline__ void updateChopper(
    bool &current_chop_neg,
    int i_weight_output,
    chop_t &x_chop,
    chop_t &d_chop,
    float *x_switching_probs,
    float *d_switching_probs,
    const int x_index,
    const int d_index,
    const int x_size,
    const int d_size,
    const int val_start,
    const bool wo_column,
    const T in_chop_prob,
    const T out_chop_prob,
    const uint64_t nwo_counter_start) {
  if (wo_column) {
    // in is x, out is d
    // in chopper:
    int val_weight_output = (i_weight_output + val_start) % x_size;
    if (x_index == val_weight_output && in_chop_prob > (T)0.0) {
      if (x_switching_probs == nullptr) {
        int i_every = (int)ceilf((T)1.0 / in_chop_prob);
        x_chop =
            (((nwo_counter_start + i_weight_output) / x_size) % i_every == 0) ? -x_chop : x_chop;
      } else {
        x_chop = (T)x_switching_probs[i_weight_output * x_size + x_index] < in_chop_prob ? -x_chop
                                                                                         : x_chop;
      }
    }
    // out chopper
    if (val_weight_output == x_size - 1 && out_chop_prob > (T)0.0) {
      d_chop = (T)d_switching_probs[i_weight_output * d_size + d_index] < out_chop_prob ? -d_chop
                                                                                        : d_chop;
    }
  } else {
    // in is d, out is x
    // in chopper:
    int val_weight_output = (i_weight_output + val_start) % d_size;
    if (d_index == val_weight_output && in_chop_prob > (T)0.0) {
      if (d_switching_probs == nullptr) {
        int i_every = (int)ceil((T)1.0 / in_chop_prob);
        d_chop =
            (((nwo_counter_start + i_weight_output) / d_size) % i_every == 0) ? -d_chop : d_chop;
      } else {
        d_chop = (T)d_switching_probs[i_weight_output * d_size + d_index] < in_chop_prob ? -d_chop
                                                                                         : d_chop;
      }
    }
    // out chopper
    if (val_weight_output == d_size - 1 && out_chop_prob > (T)0.0) {
      x_chop = (T)x_switching_probs[i_weight_output * x_size + x_index] < out_chop_prob ? -x_chop
                                                                                        : x_chop;
    }
  }
  current_chop_neg = x_chop != d_chop;
}

#define RPU_SAVE_WEIGHT_OUTPUT                                                                     \
  if (isWeightOutputVal(x_index, d_index, i_weight_output, wo_val_start, wo_column, xsz, dsz)) {   \
    int wo_idx = getWeightOutputIdx(                                                               \
        x_index, d_index, i_weight_output, wo_column, xsz, dsz, n_wo, wo_val_start,                \
        wo_flexible_in_size);                                                                      \
    weight_output[wo_idx] = w;                                                                     \
    weight_output_out_chopper[wo_idx] = wo_column ? d_chop : x_chop;                               \
    if (0 == (wo_column ? d_index : x_index)) {                                                    \
      weight_output_in_chopper[i_weight_output] = wo_column ? x_chop : d_chop;                     \
    }                                                                                              \
    /* //printf("X %d, D %d, B %d:  WO\n", x_index, d_index, current_batch);	*/                    \
  }                                                                                                \
  updateChopper(                                                                                   \
      current_chop_neg, i_weight_output, x_chop, d_chop, x_switching_probs, d_switching_probs,     \
      x_index, d_index, xsz, dsz, wo_val_start, wo_column, in_chop_prob, out_chop_prob,            \
      nwo_counter - n_weight_outputs);

template <
    typename T,
    int one_sided,
    typename count_t,
    bool x_trans,
    bool d_trans,
    typename UpdateFunctor,
    int global_params_count = 1,
    typename std::enable_if<(global_params_count > 0 && one_sided == 0), int>::type = 0>
__global__ void kernelUpdateWBatchSharedWeightOutputFunctor(
    T *weights,
    count_t *x_counts,
    int x_size,
    count_t *d_counts,
    int d_size,
    param_t *params,
    param_t *params_2,
    T *params_1,
    T *global_params,
    int BL_in,
    int m_batch_in,
    int batch_stride_in,
    const T dw_min_std,
    T *weight_output,
    chop_t *weight_output_in_chopper,  // this is of size n_wo
    chop_t *weight_output_out_chopper, // these need to be  n_wo*out_size
    int n_weight_outputs,
    uint64_t nwo_counter, // this is the END index
    const int wo_every_in,
    const bool wo_column_in, // whether col or row, column size is d_size, row size os x_size
    const int wo_val_start_in,
    const int wo_batch_start_in,
    const bool wo_flexible_in_size_in,
    const T in_chop_prob_in,
    const T out_chop_prob_in,
    const chop_t *x_chopper_in,
    const chop_t *d_chopper_in,
    chop_t *x_chopper_out, // CANNOT be in-place with x_chopper_in!
    chop_t *d_chopper_out, // CANNOT be in-place with d_chopper_in!
    float *x_switching_probs,
    float *d_switching_probs,
    curandState *random_states,
    uint32_t *weight_output_signals = nullptr, // only needed for Bo64
    int org_m_batch = 0,                       // only needed for Bo64
    kagg_t *Kn = nullptr                       // only needed for Bo64
) {
  // -- same as kernelUpdateWBatchShared but with simple loop.
  // -- Functor needs to be specified
  // -- params expected in the following order:  (min_bound, dw_down, max_bound, dw_up )
  // -- threadIdx.x will be distributed in 32 d-values rest x values.   (threadIdx.y==1)
  // -- blockDim.x like wise cover W. (blockIdx.y==1)
  // -- batch is strided. (need UBLMBO64 + batch_stride * sizeof(uint32_t) * nK32 * (32 +
  // blockDim.x/32) memory )
  // -- In case of ublm b064 only: UBLMBO64 = (batch_stride + 1)/2 * sizeof(uint64_t) to store the
  // weight_output_signals
  // -- shared memory is used to save x and d values
  // -- nthreads needs to be divisible by 32 (D_BLOCK_SIZE)!!

  // -- weight output will read transient weight rows/cols for intermedate batches

  extern __shared__ __align__(sizeof(uint64_t)) uint32_t external_mem[];
  count_t *shared_d_and_x_counts = reinterpret_cast<count_t *>(external_mem);
  uint32_t *weight_output_signals_shared = nullptr;
  const int n_wo = n_weight_outputs;
  if (std::is_same<count_t, uint64_t>::value) {
    if (n_wo > 0 && weight_output_signals != nullptr) {
      shared_d_and_x_counts = shared_d_and_x_counts + (batch_stride_in + 1) / 2;
      weight_output_signals_shared = external_mem;
    }
  }
  const int BL = BL_in;
  const int nK32_in = BL / 32 + 1;

  int m_batch = m_batch_in;
  if (Kn != nullptr) {
    m_batch = ((*Kn) + 31) / 32; // overwrite m_batch in case of BO64 UBLM
  }

  RPU_FUNCTOR_INIT_VARS;

  RPU_UWBS_DEF_AND_STRIDE_LOOP(noise_std_dw > (T)0.0);

  // in this loop the x_index / d_index change

  int i_weight_output = 0;
  chop_t x_chop = 1;
  chop_t d_chop = 1;
  bool current_chop_neg = false;
  const bool wo_column = wo_column_in;
  const int wo_val_start = wo_val_start_in;
  const int wo_batch_start = wo_batch_start_in;
  const bool wo_flexible_in_size = wo_flexible_in_size_in;
  const int wo_every = wo_every_in;
  const T in_chop_prob = in_chop_prob_in;
  const T out_chop_prob = out_chop_prob_in;

  if (within_range) {
    RPU_FUNCTOR_LOAD_PARAMS;
    if (x_chopper_in) {
      x_chop = x_chopper_in[x_index];
    }
    if (d_chopper_in) {
      d_chop = d_chopper_in[d_index];
    }
    current_chop_neg = x_chop != d_chop;
  }

  for (int i_stride = 0; i_stride < m_batch; i_stride += batch_stride) { // for batch loading

    __syncthreads();
    RPU_UWBS_READ_INTO_SHARED;

    // this is loaded again for each x/d stride. How to best avoid? Just load at the beginning for
    // all batches?
    if (std::is_same<count_t, uint64_t>::value) {
      if (n_wo > 0 && weight_output_signals_shared != nullptr) {
        for (int i_load_batch = threadIdx.x;
             (i_load_batch < batch_stride) && (i_load_batch + i_stride < m_batch);
             i_load_batch += blockDim.x) {
          weight_output_signals_shared[i_load_batch] =
              weight_output_signals[i_load_batch + i_stride];
        }
      }
    }
    __syncthreads();

    if (within_range) {

      // loop over all batches in the batch_stride
      int d_shared_offset = 0;
      int x_shared_offset = x_memoffset;
      uint32_t negative = 0;

      for (int i_load_batch = 0;
           (i_load_batch < batch_stride) && (i_load_batch + i_stride < m_batch); i_load_batch++) {
        // NOTE: this "batch" is actually nB in case of u64

        RPU_UWBS_LOAD_COUNTS_N(count_t, true);

        int current_batch = i_load_batch + i_stride;
        uint32_t weight_output_signal = isWeightOutputBatch<count_t>(
            current_batch, wo_every, wo_batch_start, n_wo, BL, weight_output_signals_shared,
            i_load_batch, org_m_batch);

        if (mixed) {
          // this is enforced for uint64 and never happens
          // for other formats. This is thus always uint64
          // UNROLLING seems quite a bit slower !?
          for (int i_bit = 0; i_bit < 32; i_bit++) {
            uint32_t bit_n = testBit(n, i_bit) ? 1 : 0;
            if (bit_n != 0) {
              uint32_t bit_neg = (testBit(negative, i_bit) != current_chop_neg) ? 1 : 0;
              up_fun(
                  w, 1, bit_neg, par_4, par_2, par_1, global_par, global_params_count, noise_std_dw,
                  local_state);
            }
            if (testBit(weight_output_signal, i_bit)) {
              // do the weight_output

              RPU_SAVE_WEIGHT_OUTPUT;

              i_weight_output++;
            }
          }
        } else { // not mixed. This can only happen for u_int32 and
          // now n is already the number n

          if (n > 0) {
            negative = ((negative > 0) != current_chop_neg) ? 1 : 0;
            up_fun(
                w, n, negative, par_4, par_2, par_1, global_par, global_params_count, noise_std_dw,
                local_state);
          }
          // if (current_chop_neg) {
          //   printf("X %d, D %d, B %d:  [-]  %d\n", x_index, d_index, current_batch, negative? -
          //   (int) n: int (n));
          // } else {
          //   printf("X %d, D %d, B %d:  [+]  %d\n", x_index, d_index, current_batch, negative? -
          //   (int) n: int (n));
          // }

          // TODO: add some proportional noise when either x or d are on?
          if (weight_output_signal) {

            RPU_SAVE_WEIGHT_OUTPUT;

            i_weight_output++;
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
    if (d_index == 0 && x_chopper_out) {
      x_chopper_out[x_index] = x_chop;
    }
    if (x_index == 0 && d_chopper_out) {
      d_chopper_out[d_index] = d_chop;
    }
  }
  RPU_UWBS_CLOSE_STRIDE_LOOP;
}

/* Pulse counter*/
template <typename T, int one_sided, typename count_t, bool x_trans, bool d_trans>
__global__ void kernelPulseCounter(
    uint64_t *pos_pulses,
    uint64_t *neg_pulses,
    count_t *x_counts,
    int x_size,
    count_t *d_counts,
    int d_size,
    int nK32_in,
    int m_batch_in,
    kagg_t *Kn = nullptr) {
  int m_batch = m_batch_in;
  if (Kn != nullptr) {
    m_batch = ((*Kn) + 31) / 32; // overwrite m_batch in case of BO64 UBLM
  }
  volatile unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
  const int total_threads = blockDim.x * gridDim.x;

  const int xsz = x_size;
  const int dsz = d_size;
  const int sz = xsz * dsz;
  const int nK32 = nK32_in;
  count_t *xptr, *dptr;

  uint64_t n_pos = 0;
  uint64_t n_neg = 0;

  const int x_count_offset = xsz * nK32;
  const int d_count_offset = dsz * nK32;

  for (int i_stride = 0; i_stride < sz; i_stride += total_threads) {

    int idx = tid + i_stride;

    if (idx < sz) { // stride over all elements of W

      int dIdx = idx % dsz; // first d
      int xIdx = idx / dsz;

      n_pos = pos_pulses[idx];
      n_neg = neg_pulses[idx];

      uint32_t sum_n = 0;
      uint32_t last_negative = 0;

      for (int i_batch = 0; i_batch < m_batch; i_batch++) {

        xptr = &x_counts[getIdxToLoad<x_trans>(i_batch, xIdx, xsz, m_batch, x_count_offset)];
        dptr = &d_counts[getIdxToLoad<d_trans>(i_batch, dIdx, dsz, m_batch, d_count_offset)];

        bool mixed;
        uint32_t negative;
        uint32_t n;

        getNfromCount<one_sided, count_t>(n, negative, mixed, xptr, dptr, nK32, xsz, dsz);

        if (mixed) {
          for (int i_bit = 0; i_bit < 32; i_bit++) {
            uint32_t bit_n = testBit(n, i_bit) ? 1 : 0;
            if (bit_n != 0) {
              uint32_t bit_neg = testBit(negative, i_bit) ? 1 : 0;
              if (bit_neg == last_negative) {
                sum_n += 1;
              } else {
                if (sum_n > 0) {
                  if (last_negative) {
                    n_neg += sum_n;
                  } else {
                    n_pos += sum_n;
                  }
                }
                sum_n = 1;
                last_negative = bit_neg;
              }
            }
          }
        } else {
          if ((n == 0) || (last_negative == negative)) {
            sum_n += n;
          } else {
            if (sum_n > 0) {
              if (last_negative) {
                n_neg += sum_n;
              } else {
                n_pos += sum_n;
              }
            }
            sum_n = n;
            last_negative = negative;
          }
        }

      } // batch
      // last update
      if (sum_n > 0) {
        if (last_negative) {
          n_neg += sum_n;
        } else {
          n_pos += sum_n;
        }
      }
      pos_pulses[idx] = n_pos;
      neg_pulses[idx] = n_neg;
    }
  }
}

} // namespace RPU
#undef RPU_SAVE_WEIGHT_OUTPUT
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
