/**
 * (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
 *
 * Licensed under the MIT license. See LICENSE file in the project root for details.
 */

#pragma once

#include "pwu_kernel.h"

namespace RPU {

__device__ __forceinline__ bool shouldDecayHS(uint8_t prev, uint8_t curr) {
  return (prev == 1 && curr == 3) || (prev == 3 && curr == 1) || (prev == 2 && curr == 4) ||
         (prev == 4 && curr == 2) || (prev == 1 && curr == 2) || (prev == 2 && curr == 1) ||
         (prev == 3 && curr == 4) || (prev == 4 && curr == 3);
}

__device__ __forceinline__ uint8_t classifyHS(bool x_bit, bool d_bit, bool x_neg, bool d_neg) {
  if (x_bit && d_bit) {
    return 0;
  }

  bool same_sign = (x_neg == d_neg);

  if (x_bit && !d_bit) {
    return same_sign ? 1 : 3;
  }

  if (!x_bit && d_bit) {
    return same_sign ? 2 : 4;
  }

  return 255;
}

#ifndef D_BLOCK_SIZE
#define RPU_HS_DEFINED_D_BLOCK_SIZE
#define D_BLOCK_SIZE 32
#endif

#ifndef D_BLOCK_SIZE_BITS
#define RPU_HS_DEFINED_D_BLOCK_SIZE_BITS
#define D_BLOCK_SIZE_BITS 5
#endif

#ifndef RPU_FUNCTOR_INIT_VARS
#define RPU_HS_DEFINED_RPU_FUNCTOR_INIT_VARS
#define RPU_FUNCTOR_INIT_VARS                                                                      \
  T w = 0;                                                                                         \
  param4_t par_4;                                                                                  \
  param2_t par_2;                                                                                  \
  T par_1 = 0;                                                                                     \
  bool use_par_1 = params_1 != nullptr;                                                            \
  UpdateFunctor up_fun;                                                                            \
  __shared__ T global_par[global_params_count];                                                    \
  if (global_params != nullptr) {                                                                  \
    for (int gidx = threadIdx.x; gidx < global_params_count; gidx += blockDim.x) {                \
      global_par[gidx] = global_params[gidx];                                                      \
    }                                                                                              \
    __syncthreads();                                                                               \
  }
#endif

#ifndef RPU_FUNCTOR_LOAD_PARAMS
#define RPU_HS_DEFINED_RPU_FUNCTOR_LOAD_PARAMS
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
#endif

#ifndef RPU_UWBS_DEF_AND_STRIDE_LOOP
#define RPU_HS_DEFINED_RPU_UWBS_DEF_AND_STRIDE_LOOP
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
  for (int i_tid_stride = 0; i_tid_stride < total_tid; i_tid_stride += n_block_threads) {         \
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
        if ((i_tid_stride > 0) && noiseif) {                                                       \
          random_states[tid] = local_state;                                                        \
        }                                                                                          \
        return;                                                                                    \
      }                                                                                            \
    }                                                                                              \
    int idx = x_index * dsz + d_index;                                                             \
    bool within_range = ((x_index < xsz) && (d_index < dsz));
#endif

#ifndef RPU_UWBS_READ_INTO_SHARED
#define RPU_HS_DEFINED_RPU_UWBS_READ_INTO_SHARED
#define RPU_UWBS_READ_INTO_SHARED                                                                  \
  {                                                                                                \
    int n_load_batch = blockDim.x / D_BLOCK_SIZE;                                                  \
    for (int i_load_stride = 0; i_load_stride < batch_stride; i_load_stride += n_load_batch) {    \
      if (d_index < dsz) {                                                                         \
        const int i_batch = d_load_batch_idx + i_load_stride;                                      \
        if ((i_batch + i_stride < m_batch) && (i_batch < batch_stride)) {                          \
          int d_index_load_batch =                                                                 \
              getIdxToLoad<d_trans>(i_batch + i_stride, d_index, dsz, m_batch, d_count_offset);    \
          const int d_shared_load_index = load_d_offset * i_batch + d_sub_idx;                     \
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
    const int x_index_load = x_block_start + x_sub_idx_load;                                       \
    for (int i_load_stride = 0; i_load_stride < batch_stride; i_load_stride += n_load_batch) {    \
      if (x_index_load < xsz) {                                                                    \
        const int i_batch = x_load_batch_idx + i_load_stride;                                      \
        if ((i_batch + i_stride < m_batch) && (i_batch < batch_stride)) {                          \
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
#endif

#ifndef RPU_UWBS_CLOSE_STRIDE_LOOP
#define RPU_HS_DEFINED_RPU_UWBS_CLOSE_STRIDE_LOOP
#define RPU_UWBS_CLOSE_STRIDE_LOOP                                                                 \
  }                                                                                                \
  if (noiseif) {                                                                                   \
    random_states[tid] = local_state;                                                              \
  }
#endif

template <
    typename T,
    int one_sided,
    typename count_t,
    bool x_trans,
    bool d_trans,
    typename UpdateFunctor,
    int global_params_count = 2,
    typename std::enable_if<(global_params_count > 1), int>::type = 0>
__global__ void kernelUpdateWBatchSharedFunctorHS(
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
    uint8_t *dev_hs_states,
    curandState *random_states,
    unsigned long long *dev_hs_counts = nullptr) {

  extern __shared__ __align__(sizeof(uint64_t)) uint32_t shared_d_and_x_counts_32[];
  count_t *shared_d_and_x_counts = reinterpret_cast<count_t *>(shared_d_and_x_counts_32);

  static_assert(
      std::is_same<count_t, uint32_t>::value,
      "kernelUpdateWBatchSharedFunctorHS requires uint32_t bitline counts.");

  const int m_batch = m_batch_in;

  RPU_FUNCTOR_INIT_VARS;
  RPU_UWBS_DEF_AND_STRIDE_LOOP(noise_std_dw > (T)0.0);

  uint8_t hs_state = 1;
  T hs_decay = (T)1.0;
  if (within_range) {
    RPU_FUNCTOR_LOAD_PARAMS;
    hs_state = dev_hs_states[d_index * xsz + x_index];
    if (global_params != nullptr) {
      hs_decay = global_par[1];
    }
  }

  for (int i_stride = 0; i_stride < m_batch; i_stride += batch_stride) {

    __syncthreads();
    RPU_UWBS_READ_INTO_SHARED;
    __syncthreads();

    if (within_range) {
      int d_shared_offset = 0;
      int x_shared_offset = x_memoffset;

      for (int i_batch = 0; (i_batch < batch_stride) && (i_batch + i_stride < m_batch); i_batch++) {

        int d_shared_index = d_sub_idx + d_shared_offset;
        int x_shared_index = x_sub_idx + x_shared_offset;
        d_shared_offset += load_d_offset;
        x_shared_offset += load_x_offset;

        const uint32_t x_sign_word = (uint32_t)shared_d_and_x_counts[x_shared_index];
        const uint32_t d_sign_word = (uint32_t)shared_d_and_x_counts[d_shared_index];
        const bool x_neg = (x_sign_word & 1u) > 0u;
        const bool d_neg = (d_sign_word & 1u) > 0u;

        uint32_t negative = (x_neg != d_neg) ? 1u : 0u;
        bool allow_update = true;
        if (one_sided == -1) {
          if (negative > 0u) {
            allow_update = false;
          } else {
            negative = 1u;
          }
        } else if (one_sided == 1) {
          if (negative == 0u) {
            allow_update = false;
          }
        }

        for (int i_k32 = 0; i_k32 < nK32; i_k32++) {
          const uint32_t x_word =
              (uint32_t)shared_d_and_x_counts[x_shared_index + i_k32 * x_block_size];
          const uint32_t d_word =
              (uint32_t)shared_d_and_x_counts[d_shared_index + i_k32 * D_BLOCK_SIZE];

          const int bit_start = (i_k32 == 0) ? 1 : 0;
          for (int b = bit_start; b < 32; b++) {
            const bool x_bit = ((x_word >> b) & 1u) > 0u;
            const bool d_bit = ((d_word >> b) & 1u) > 0u;

            if (!x_bit && !d_bit) {
              continue;
            }

            uint8_t new_hs = classifyHS(x_bit, d_bit, x_neg, d_neg);
            if (new_hs == 255) {
              continue;
            }

            if (shouldDecayHS(hs_state, new_hs)) {
              w *= hs_decay;
            }

            // Optional transition counting (only when HS tracking is enabled, i.e.
            // dev_hs_counts != nullptr). Mirrors the CPU 4x4 layout: 16 counters
            // per d-row, index = (prev-1)*4 + (curr-1) for states HS1..HS4.
            if (dev_hs_counts != nullptr && hs_state >= 1 && hs_state <= 4 && new_hs >= 1 &&
                new_hs <= 4) {
              atomicAdd(
                  &dev_hs_counts[d_index * 16 + (hs_state - 1) * 4 + (new_hs - 1)],
                  (unsigned long long)1);
            }

            if (x_bit && d_bit && allow_update) {
              up_fun(
                  w, 1, negative, par_4, par_2, par_1, global_par, global_params_count, noise_std_dw,
                  local_state);
            }

            hs_state = new_hs;
          }
        }
      }
    }
  }

  if (within_range) {
    weights[idx] = w;
    dev_hs_states[d_index * xsz + x_index] = hs_state;
    if (use_par_1) {
      params_1[idx] = par_1;
    }
  }

  RPU_UWBS_CLOSE_STRIDE_LOOP;
}

}

#ifdef RPU_HS_DEFINED_RPU_UWBS_CLOSE_STRIDE_LOOP
#undef RPU_UWBS_CLOSE_STRIDE_LOOP
#undef RPU_HS_DEFINED_RPU_UWBS_CLOSE_STRIDE_LOOP
#endif

#ifdef RPU_HS_DEFINED_RPU_UWBS_READ_INTO_SHARED
#undef RPU_UWBS_READ_INTO_SHARED
#undef RPU_HS_DEFINED_RPU_UWBS_READ_INTO_SHARED
#endif

#ifdef RPU_HS_DEFINED_RPU_UWBS_DEF_AND_STRIDE_LOOP
#undef RPU_UWBS_DEF_AND_STRIDE_LOOP
#undef RPU_HS_DEFINED_RPU_UWBS_DEF_AND_STRIDE_LOOP
#endif

#ifdef RPU_HS_DEFINED_RPU_FUNCTOR_LOAD_PARAMS
#undef RPU_FUNCTOR_LOAD_PARAMS
#undef RPU_HS_DEFINED_RPU_FUNCTOR_LOAD_PARAMS
#endif

#ifdef RPU_HS_DEFINED_RPU_FUNCTOR_INIT_VARS
#undef RPU_FUNCTOR_INIT_VARS
#undef RPU_HS_DEFINED_RPU_FUNCTOR_INIT_VARS
#endif

#ifdef RPU_HS_DEFINED_D_BLOCK_SIZE_BITS
#undef D_BLOCK_SIZE_BITS
#undef RPU_HS_DEFINED_D_BLOCK_SIZE_BITS
#endif

#ifdef RPU_HS_DEFINED_D_BLOCK_SIZE
#undef D_BLOCK_SIZE
#undef RPU_HS_DEFINED_D_BLOCK_SIZE
#endif
