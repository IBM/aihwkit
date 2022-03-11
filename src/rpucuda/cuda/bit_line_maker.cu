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

#include "bit_line_maker.h"

#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <random>

#include "cuda_math_util.h"
#include "cuda_util.h"
#include "io_iterator.h"
#include <cub/cub.cuh>

namespace RPU {

#define LASTK32MASK 0xFFFFFFFF >> ((nK32 << 5) - Kplus1)

#define RPU_BLM_DEFINE_NK32                                                                        \
  const int nK32 = (Kplus1 + 31) >> 5;                                                             \
  const int nK32m1 = nK32 - 1;

#define COMMA ,

#define RPU_BLM_SWITCH_TRANS_TEMPLATE(X_TRANS, D_TRANS, OUT_TRANS, KERNEL, ARGS, ADDTEMP)          \
  if (OUT_TRANS) {                                                                                 \
    if (X_TRANS & D_TRANS) {                                                                       \
      KERNEL<T, true, true, true ADDTEMP><<<nblocks, nthreads_, 0, s>>> ARGS;                      \
    } else if (!X_TRANS & !D_TRANS) {                                                              \
      KERNEL<T, false, false, true ADDTEMP><<<nblocks, nthreads_, 0, s>>> ARGS;                    \
    } else if (!X_TRANS & D_TRANS) {                                                               \
      KERNEL<T, false, true, true ADDTEMP><<<nblocks, nthreads_, 0, s>>> ARGS;                     \
    } else {                                                                                       \
      KERNEL<T, true, false, true ADDTEMP><<<nblocks, nthreads_, 0, s>>> ARGS;                     \
    }                                                                                              \
  } else {                                                                                         \
    if (X_TRANS & D_TRANS) {                                                                       \
      KERNEL<T, true, true, false ADDTEMP><<<nblocks, nthreads_, 0, s>>> ARGS;                     \
    } else if (!X_TRANS & !D_TRANS) {                                                              \
      KERNEL<T, false, false, false ADDTEMP><<<nblocks, nthreads_, 0, s>>> ARGS;                   \
    } else if (!X_TRANS & D_TRANS) {                                                               \
      KERNEL<T, false, true, false ADDTEMP><<<nblocks, nthreads_, 0, s>>> ARGS;                    \
    } else {                                                                                       \
      KERNEL<T, true, false, false ADDTEMP><<<nblocks, nthreads_, 0, s>>> ARGS;                    \
    }                                                                                              \
  }

#define RPU_BLM_SWITCH_TRANS_TEMPLATE_UM(X_TRANS, D_TRANS, OUT_TRANS, UM, UBLM, KERNEL, ARGS)      \
  if (UM && UBLM) {                                                                                \
    RPU_BLM_SWITCH_TRANS_TEMPLATE(                                                                 \
        X_TRANS, D_TRANS, OUT_TRANS, KERNEL, ARGS, COMMA true COMMA true);                         \
  } else if (UM && !UBLM) {                                                                        \
    RPU_BLM_SWITCH_TRANS_TEMPLATE(                                                                 \
        X_TRANS, D_TRANS, OUT_TRANS, KERNEL, ARGS, COMMA true COMMA false);                        \
  } else if (!UM && !UBLM) {                                                                       \
    RPU_BLM_SWITCH_TRANS_TEMPLATE(                                                                 \
        X_TRANS, D_TRANS, OUT_TRANS, KERNEL, ARGS, COMMA false COMMA false);                       \
  } else {                                                                                         \
    RPU_BLM_SWITCH_TRANS_TEMPLATE(                                                                 \
        X_TRANS, D_TRANS, OUT_TRANS, KERNEL, ARGS, COMMA true COMMA false);                        \
  }

#define RPU_BLM_DEBUG_DEFINE_K_NO_STOROUND                                                         \
  int Kplus1 = K + 1;                                                                              \
  int nK32 = (Kplus1 + 31) / 32;                                                                   \
  T resolution = 0.01;

#define RPU_BLM_DEBUG_DEFINE_K                                                                     \
  RPU_BLM_DEBUG_DEFINE_K_NO_STOROUND;                                                              \
  bool sto_round = false;

#define RPU_BLM_DEBUG_DEFINE_K_BATCH                                                               \
  RPU_BLM_DEBUG_DEFINE_K;                                                                          \
  int m_batch = 1;

#define RPU_BLM_DEBUG_INIT(NSTATES)                                                                \
                                                                                                   \
  CudaContext c{-1, false};                                                                        \
  CudaArray<uint32_t> dev_counts(&c, size *nK32);                                                  \
  CudaArray<T> dev_indata(&c, size, indata);                                                       \
                                                                                                   \
  CudaArray<curandState> dev_states(&c, NSTATES);                                                  \
  curandSetup(dev_states, size, fake_seed);                                                        \
  CUDA_CALL(cudaDeviceSynchronize());                                                              \
                                                                                                   \
  cudaEvent_t start, stop;                                                                         \
  CUDA_CALL(cudaEventCreate(&start));                                                              \
  CUDA_CALL(cudaEventCreate(&stop));                                                               \
  CUDA_CALL(cudaEventRecord(start, c.getStream()));

#define RPU_BLM_DEBUG_FINISH                                                                       \
  CUDA_CALL(cudaEventRecord(stop, c.getStream()));                                                 \
  cudaEventSynchronize(stop);                                                                      \
  CUDA_CALL(cudaPeekAtLastError());                                                                \
  CUDA_CALL(cudaDeviceSynchronize());                                                              \
                                                                                                   \
  float milliseconds = 0;                                                                          \
  cudaEventElapsedTime(&milliseconds, start, stop);                                                \
                                                                                                   \
  *timing = milliseconds;                                                                          \
  dev_counts.copyTo(counts);                                                                       \
  CUDA_CALL(cudaDeviceSynchronize());

#define RPU_BLM_DEBUG_BATCH_INIT(NSTATES, COUNTSTYPE)                                              \
  CudaContext c{-1, false};                                                                        \
  CudaArray<COUNTSTYPE> dev_counts(&c, size *m_batch *nK32);                                       \
  CudaArray<COUNTSTYPE> dev_counts2(&c, size *m_batch *nK32);                                      \
  dev_counts.setConst(0);                                                                          \
  dev_counts2.setConst(0);                                                                         \
  T *tmp = new T[size * m_batch];                                                                  \
  for (int i = 0; i < m_batch; i++) {                                                              \
    for (int j = 0; j < size; j++) {                                                               \
      tmp[i * size + j] = indata[j];                                                               \
    }                                                                                              \
  }                                                                                                \
  CudaArray<T> dev_indata(&c, size *m_batch, tmp);                                                 \
  CudaArray<T> dev_indata2(&c, size *m_batch, tmp);                                                \
                                                                                                   \
  CudaArray<curandState> dev_states(&c, NSTATES);                                                  \
  curandSetup(dev_states, size, fake_seed);                                                        \
                                                                                                   \
  CUDA_CALL(cudaDeviceSynchronize());                                                              \
                                                                                                   \
  cudaEvent_t start, stop;                                                                         \
  CUDA_CALL(cudaEventCreate(&start));                                                              \
  CUDA_CALL(cudaEventCreate(&stop));                                                               \
  CUDA_CALL(cudaEventRecord(start, c.getStream()));

#define RPU_BLM_DEBUG_BATCH_FINISH(COUNTT)                                                         \
  CUDA_CALL(cudaEventRecord(stop, c.getStream()));                                                 \
  cudaEventSynchronize(stop);                                                                      \
  CUDA_CALL(cudaPeekAtLastError());                                                                \
  CUDA_CALL(cudaDeviceSynchronize());                                                              \
                                                                                                   \
  float milliseconds = 0;                                                                          \
  cudaEventElapsedTime(&milliseconds, start, stop);                                                \
                                                                                                   \
  *timing = milliseconds;                                                                          \
                                                                                                   \
  COUNTT *tmp32 = new COUNTT[size * m_batch * nK32];                                               \
  dev_counts.copyTo(tmp32);                                                                        \
  CUDA_CALL(cudaDeviceSynchronize());                                                              \
                                                                                                   \
  int batch_idx = m_batch - 1;                                                                     \
  for (int j = 0; j < size * nK32; j++) {                                                          \
    counts[j] = tmp32[(batch_idx)*size * nK32 + j];                                                \
  }                                                                                                \
                                                                                                   \
  CUDA_CALL(cudaDeviceSynchronize());                                                              \
                                                                                                   \
  delete[] tmp;                                                                                    \
  delete[] tmp32;

template <bool trans, bool out_trans, typename count_t>
__device__ __forceinline__ int
getCountsIdx(int idx, int sz, int m_batch, int count_offset, int K = 0, kagg_t Kc = 0, int nB = 0);

template <bool trans> __device__ __forceinline__ int getBatchIdx(int idx, int sz, int m_batch);

template <typename T, bool um>
__device__ __forceinline__ T getScale(const T *scale_values, int batch_idx);

template <bool ublm>
__device__ __forceinline__ int getK(const int *K_values, int batch_idx, int Kplus1);

template <typename T, bool ublm>
__device__ __forceinline__ T getScaleProb(const T scaleprob, const int K, const T lr_div_dwmin);

template <bool ublm, typename count_t>
__device__ __forceinline__ int getnB(const kagg_t *Kn, int m_batch, int Kplus1);

template <bool ublm, typename count_t>
__device__ __forceinline__ int getKc(const kagg_t *Kc_values, int batch_idx, int Kplus1);

template <typename count_t>
__device__ __forceinline__ void getCountsSimpleLoop(
    float value,
    bool negative,
    count_t *c,
    int nK32m1,
    int K,
    curandState &local_state,
    int nK32,
    int sz,
    kagg_t Kc);

#define DISCRETIZE_VALUE_STOCH_DEFINITIONS                                                         \
  T res = resolution;                                                                              \
  bool sr = sto_round & (res > 0);                                                                 \
  T stoch_value;

#define DISCRETIZE_VALUE(RES)                                                                      \
  if (RES > 0) {                                                                                   \
    value /= RES;                                                                                  \
    value = RES * RPU_ROUNDFUN(value);                                                             \
  }

#define DISCRETIZE_VALUE_STOCH(STATEVAR)                                                           \
  if (sr)                                                                                          \
    stoch_value = curand_uniform(&STATEVAR);                                                       \
                                                                                                   \
  if (res > 0) {                                                                                   \
    value /= res;                                                                                  \
    if (sr)                                                                                        \
      value += stoch_value - 0.5;                                                                  \
    value = res * RPU_ROUNDFUN(value);                                                             \
  }

namespace test_helper {
// helper function for debugging

int getCounts(uint32_t *counts, int i, int K, int size, bool negtest) {
  int icounts = 0;
  int nK32 = (K + 1 + 31) / 32;
  uint32_t one = 1;
  uint32_t negative = counts[i] & one;
  for (int j = 0; j < nK32; j++) {
    uint32_t c = counts[i + j * size];
    for (int l = 0; l < 32; l++) {
      if ((c & (one << l)) != 0) {
        icounts++;
      };
    }
  }

  if (negtest && (negative == 1))
    return -icounts + 1;
  else
    return icounts;
}

template <typename T>
void checkCounts(
    const T *x_input,
    int x_size,
    const T *d_input,
    int d_size,
    int BL,
    T A,
    T B,
    CudaArray<uint32_t> *dev_x_counts,
    CudaArray<uint32_t> *dev_d_counts) {
  T *host_x_input = new T[x_size];
  T *host_d_input = new T[d_size];

  CUDA_CALL(cudaMemcpy(host_d_input, d_input, d_size * sizeof(T), cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpy(host_x_input, x_input, x_size * sizeof(T), cudaMemcpyDeviceToHost));

  uint32_t *x_counts = new uint32_t[dev_x_counts->getSize()];
  uint32_t *d_counts = new uint32_t[dev_d_counts->getSize()];

  dev_d_counts->copyTo(d_counts);
  dev_x_counts->copyTo(x_counts);
  int dzero = 0;
  int xzero = 0;

  for (int i = 0; i < d_size; i++) {
    int d = getCounts(d_counts, i, BL, d_size, true);
    if (i < 100)
      std::cout << "D[" << i << "]: " << d << " (" << (T)d / BL << ")"
                << " in: " << host_d_input[i] * A << std::endl;
    if (fabs(d) == 0)
      dzero++;
  }

  for (int i = 0; i < x_size; i++) {
    int x = getCounts(x_counts, i, BL, x_size, true);

    if (fabs(x) == 0)
      xzero++;

    if (i < 100)
      std::cout << "X[" << i << "]: " << x << " (" << (T)x / BL << ")"
                << " in: " << host_x_input[i] * B << std::endl;
  }

  delete[] x_counts;
  delete[] d_counts;
  delete[] host_x_input;
  delete[] host_d_input;
}
} // namespace test_helper

/******************************************************************************
 * COUNT KERNELS *
 ******************************************************************************/

// *********************************************************************************
// kernelUpdateGetCounts_Linear
template <typename T, int ITEMS_PER_THREAD, typename InputIterator>
__global__ void kernelUpdateGetCounts_Linear(
    InputIterator source_value,
    int size_in,
    T scaleprob,
    uint32_t *counts,
    int Kplus1,
    curandState *random_states,
    T resolution,
    bool sto_round) {
  // call <<
  // size*Kplus1/ITEMS_PER_THREAD/RPU_THREADS_PER_BLOCK_UPDATE,RPU_THREADS_PER_BLOCK_UPDATE>>

  // 2) assume that Kplus1<= 32 !!
  // 3) assume that ITEM_PER_THREAD is power of 2
  // 4) NEEDS Kplus1/ITEM_PERT_PER_THREAD*size random states !
  // 5) assume ITEMS_PER_THREAD <= Kplus1 and mod(Kplus1,ITEMS_PER_THREAD)==0 !!

  const int Kp1 = Kplus1;
  const int size = size_in;
  const int tid = blockDim.x * blockIdx.x + threadIdx.x;
  const int nKthreads = (Kp1 / ITEMS_PER_THREAD);
  const int sourceId = tid / nKthreads;
  const int kidsub = (tid * ITEMS_PER_THREAD) % Kp1;
  const int kidthread = tid % nKthreads;
  const uint32_t one = 1;

  if (tid >= (size * nKthreads))
    return;

  DISCRETIZE_VALUE_STOCH_DEFINITIONS;

  curandState local_state;
  T value;
  bool negative;

  if (sourceId < size) {

    value = source_value[sourceId]; // not memory optimized but good  for K=32 (broadcasted)
    local_state = random_states[tid];

    // input management
    negative = value < 0;
    value = (negative) ? -value : value;

    value *= scaleprob;

    if (kidsub == 0) {      // only once per K
      counts[sourceId] = 0; // need to set zero (all Kthreads need to be inside a warp!)

      DISCRETIZE_VALUE_STOCH(local_state);
    }
    // need to broadcast value to within K threads
    value = __shfl_up_sync(0xFFFFFFFF, value, kidthread);

    uint32_t bitwise = 0;
#pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
      stoch_value = curand_uniform(&local_state);
      if (stoch_value < value) {
        bitwise |= (1 << (kidsub + i));
      }
    }
    random_states[tid] = local_state;

    if (kidsub == 0) {
      bitwise = (negative) ? (bitwise | one) : (bitwise & ~one);
    }
    if (bitwise > 0)
      atomicOr(&counts[sourceId], bitwise);
  }
}

namespace test_helper {

template <typename T, int ITEMS_PER_THREAD>
int debugKernelUpdateGetCounts_Linear(
    T *indata, int size, T scaleprob, uint32_t *counts, int K, T *timing, bool fake_seed) {
  // counts should be: size*nk32;
  RPU_BLM_DEBUG_DEFINE_K;

  if ((nK32 > 1) || ((Kplus1 % ITEMS_PER_THREAD) != 0) || (Kplus1 < ITEMS_PER_THREAD) ||
      (32 % Kplus1 != 0)) {
    std::cerr << "Kplus1: ITEM_PER_THREAD combination not supported. ";
    return 1;
  }
  int n = size * (Kplus1 / ITEMS_PER_THREAD);
  dim3 nthreads = dim3(RPU_THREADS_PER_BLOCK_UPDATE);
  int nblocks = (n + RPU_THREADS_PER_BLOCK_UPDATE - 1) / RPU_THREADS_PER_BLOCK_UPDATE;

  RPU_BLM_DEBUG_INIT(n);

  kernelUpdateGetCounts_Linear<T, ITEMS_PER_THREAD, const T *>
      <<<nblocks, nthreads, 0, c.getStream()>>>(
          dev_indata.getData(), size, scaleprob, dev_counts.getData(), Kplus1, dev_states.getData(),
          resolution, sto_round);

  RPU_BLM_DEBUG_FINISH;
  return 0;
}

template int
debugKernelUpdateGetCounts_Linear<float, 1>(float *, int, float, uint32_t *, int, float *, bool);
template int
debugKernelUpdateGetCounts_Linear<float, 2>(float *, int, float, uint32_t *, int, float *, bool);
template int
debugKernelUpdateGetCounts_Linear<float, 4>(float *, int, float, uint32_t *, int, float *, bool);
template int
debugKernelUpdateGetCounts_Linear<float, 8>(float *, int, float, uint32_t *, int, float *, bool);
template int
debugKernelUpdateGetCounts_Linear<float, 16>(float *, int, float, uint32_t *, int, float *, bool);
template int
debugKernelUpdateGetCounts_Linear<float, 32>(float *, int, float, uint32_t *, int, float *, bool);

#ifdef RPU_USE_DOUBLE
template int debugKernelUpdateGetCounts_Linear<double, 1>(
    double *, int, double, uint32_t *, int, double *, bool);
template int debugKernelUpdateGetCounts_Linear<double, 2>(
    double *, int, double, uint32_t *, int, double *, bool);
template int debugKernelUpdateGetCounts_Linear<double, 4>(
    double *, int, double, uint32_t *, int, double *, bool);
template int debugKernelUpdateGetCounts_Linear<double, 8>(
    double *, int, double, uint32_t *, int, double *, bool);
template int debugKernelUpdateGetCounts_Linear<double, 16>(
    double *, int, double, uint32_t *, int, double *, bool);
template int debugKernelUpdateGetCounts_Linear<double, 32>(
    double *, int, double, uint32_t *, int, double *, bool);
#endif
} // namespace test_helper

// *********************************************************************************
// kernelUpdateGetCountsBatch_Loop2

#define GET_COUNTS_INNER_LOOP(SCALEPROB)                                                           \
                                                                                                   \
  negative = value < 0;                                                                            \
  value = (negative) ? -value : value;                                                             \
                                                                                                   \
  value *= SCALEPROB;                                                                              \
                                                                                                   \
  if (laneId == 0) {                                                                               \
    DISCRETIZE_VALUE_STOCH(local_state);                                                           \
  }                                                                                                \
  value = __shfl_sync(0xFFFFFFFF, value, 0);                                                       \
                                                                                                   \
  int isize = 0;                                                                                   \
                                                                                                   \
  PRAGMA(unroll)                                                                                   \
  for (int i = 0; i < nK32; i++) {                                                                 \
                                                                                                   \
    stoch_value = curand_uniform(&local_state);                                                    \
                                                                                                   \
    ballot = __ballot_sync(0xFFFFFFFF, stoch_value < value);                                       \
                                                                                                   \
    if (laneId == 0) {                                                                             \
      if (i == 0) {                                                                                \
        ballot = (negative) ? (ballot | one) : (ballot & ~one);                                    \
      }                                                                                            \
                                                                                                   \
      if (i == nK32m1) {                                                                           \
        ballot = ballot & lastK32mask;                                                             \
      }                                                                                            \
                                                                                                   \
      *(c + isize) = ballot;                                                                       \
      isize += sz;                                                                                 \
    }                                                                                              \
  }

#define GET_COUNTS_LOOP(PROB, SIZE, COUNTS, SCALEPROB)                                             \
  sz = SIZE;                                                                                       \
  if (sourceId < sz) {                                                                             \
    value = PROB[sourceId];                                                                        \
                                                                                                   \
    c = &COUNTS[sourceId];                                                                         \
                                                                                                   \
    GET_COUNTS_INNER_LOOP(SCALEPROB);                                                              \
  }

#define GET_COUNTS_LOOP_BATCH(PROB, SIZE, COUNTS, SCALEPROB, TRANS, OUTTRANS)                      \
  {                                                                                                \
    sz = SIZE;                                                                                     \
    int counts_offset = nK32 * sz;                                                                 \
    int n = m_batch * sz;                                                                          \
    int n32 = n << 5;                                                                              \
    for (int i_stride = 0; i_stride < n32; i_stride += total_threads) {                            \
      int sourceId = (tid + i_stride) >> 5;                                                        \
      if (sourceId < n) {                                                                          \
        value = PROB[sourceId];                                                                    \
        c = &COUNTS[getCountsIdx<TRANS, OUTTRANS, uint32_t>(                                       \
            sourceId, sz, m_batch, counts_offset)];                                                \
        GET_COUNTS_INNER_LOOP(SCALEPROB);                                                          \
      }                                                                                            \
    }                                                                                              \
  }

template <
    typename T,
    bool x_trans,
    bool d_trans,
    bool out_trans,
    typename XInputIteratorT,
    typename DInputIteratorT>
__global__ void kernelUpdateGetCountsBatch_Loop2(
    XInputIteratorT x_prob,
    int x_size_in,
    T x_scaleprob,
    uint32_t *x_counts,
    DInputIteratorT d_prob,
    int d_size_in,
    T d_scaleprob,
    uint32_t *d_counts,
    int Kplus1,
    int m_batch_in,
    curandState *random_states,
    T resolution,
    bool sto_round) {
  // call << (size (of states)/numwarpsperblock,1),warpSize*numwarpsperblock >>

  // -- let each warp compute 32 K values
  // -- no limit for K , however BLocked design might be better for larger K

  const int tid = blockDim.x * blockIdx.x + threadIdx.x; // can be larger that x_size or d_size
  const int total_threads = blockDim.x * gridDim.x;
  const int x_size = x_size_in;
  const int d_size = d_size_in;
  const int m_batch = m_batch_in;

  const int max_size = (((d_size > x_size) ? d_size : x_size) * m_batch) << 5;

  if (tid >= max_size)
    return;

  curandState local_state;
  local_state = random_states[tid];

  RPU_BLM_DEFINE_NK32;
  const uint32_t lastK32mask = LASTK32MASK;
  const uint32_t one = 1;

  const int laneId = threadIdx.x & 0x1f;

  DISCRETIZE_VALUE_STOCH_DEFINITIONS;

  uint32_t ballot = 0;

  T value;
  uint32_t *c;
  bool negative;
  int sz;

  // NOTE: need to re-order in update from SIZE*nK32 format, when it is trans!

  // x input
  GET_COUNTS_LOOP_BATCH(x_prob, x_size, x_counts, x_scaleprob, x_trans, out_trans);

  // d input
  GET_COUNTS_LOOP_BATCH(d_prob, d_size, d_counts, d_scaleprob, d_trans, out_trans);

  // save new random states
  random_states[tid] = local_state;
}

namespace test_helper {
template <typename T>
int debugKernelUpdateGetCountsBatch_Loop2(
    T *indata, int size, T scaleprob, uint32_t *counts, int K, T *timing, bool fake_seed) {
  // counts should be: size*nk32 allocated !
  RPU_BLM_DEBUG_DEFINE_K_BATCH;

  int nthreads = RPU_THREADS_PER_BLOCK_UPDATE;
  int numwarpsperblock = RPU_THREADS_PER_BLOCK_UPDATE / 32;

  int n_items = 12;
  int m =
      MIN((size * m_batch + n_items - 1) / n_items, numwarpsperblock * 8); // stripped per thread
  int nblocks = (m + numwarpsperblock - 1) / numwarpsperblock;
  int n = m * 32;
  RPU_BLM_DEBUG_BATCH_INIT(n, uint32_t);

  kernelUpdateGetCountsBatch_Loop2<T, false, false, false><<<nblocks, nthreads, 0, c.getStream()>>>(
      dev_indata.getData(), size, scaleprob, dev_counts.getData(), dev_indata2.getData(), size,
      scaleprob, dev_counts2.getData(), Kplus1, m_batch, dev_states.getData(), resolution,
      sto_round);

  RPU_BLM_DEBUG_BATCH_FINISH(uint32_t);
  return 0;
}
template int debugKernelUpdateGetCountsBatch_Loop2<float>(
    float *, int, float, unsigned int *, int, float *, bool);
#ifdef RPU_USE_DOUBLE
template int debugKernelUpdateGetCountsBatch_Loop2<double>(
    double *, int, double, unsigned int *, int, double *, bool);
#endif

} // namespace test_helper

// *********************************************************************************
// kernelUpdateGetCountsBatch_SimpleLoop2

template <>
__device__ __forceinline__ int getCountsIdx<false, false, uint32_t>(
    int idx, int sz, int m_batch, int counts_offset, int K, kagg_t Kc, int nB) {

  return idx / sz * counts_offset + (idx % sz);
}

template <>
__device__ __forceinline__ int getCountsIdx<false, true, uint32_t>(
    int idx, int sz, int m_batch, int counts_offset, int K, kagg_t Kc, int nB) {
  // batchidx = idx/sz;
  // x_idx = idx % sz
  int transposed_idx = idx / sz + m_batch * (idx % sz);
  return (transposed_idx / sz) * counts_offset + (transposed_idx % sz);
}

template <>
__device__ __forceinline__ int getCountsIdx<true, false, uint32_t>(
    int idx, int sz, int m_batch, int counts_offset, int K, kagg_t Kc, int nB) {
  // batchidx * count_offset  + i_x
  return (idx % m_batch) * counts_offset + (idx / m_batch);
}

template <>
__device__ __forceinline__ int getCountsIdx<true, true, uint32_t>(
    int idx, int sz, int m_batch, int counts_offset, int K, kagg_t Kc, int nB) {
  // idx already transposed. leave it
  return idx / sz * counts_offset + (idx % sz);
}

template <>
__device__ __forceinline__ int getCountsIdx<true, true, uint64_t>(
    int idx, int sz, int m_batch, int counts_offset, int K, kagg_t Kc, int nB) {
  int iB = Kc >> 5;           // start word [new batch idx]
  int xd_idx = idx / m_batch; // NOTE: nK32==1 REQUIRED (and not checked!!)
  return xd_idx * nB + iB;
}
template <>
__device__ __forceinline__ int getCountsIdx<false, true, uint64_t>(
    int idx, int sz, int m_batch, int counts_offset, int K, kagg_t Kc, int nB) {
  int iB = Kc >> 5;      // start word [new batch idx]
  int xd_idx = idx % sz; // NOTE: nK32==1 REQUIRED (and not checked!!)
  return xd_idx * nB + iB;
}

template <>
__device__ __forceinline__ int getCountsIdx<true, false, uint64_t>(
    int idx, int sz, int m_batch, int counts_offset, int K, kagg_t Kc, int nB) {
  printf("ERROR. Not implemented\n");
  return 0;
}
template <>
__device__ __forceinline__ int getCountsIdx<false, false, uint64_t>(
    int idx, int sz, int m_batch, int counts_offset, int K, kagg_t Kc, int nB) {
  printf("ERROR. Not implemented\n");
  return 0;
}

template <> __device__ __forceinline__ int getBatchIdx<true>(int idx, int sz, int m_batch) {
  return idx % m_batch;
}

template <> __device__ __forceinline__ int getBatchIdx<false>(int idx, int sz, int m_batch) {
  return idx / sz;
}

template <>
__device__ __forceinline__ float getScale<float, true>(const float *scale_values, int batch_idx) {
  // UM + inverse (for B (that is x))
  return scale_values[batch_idx];
}

template <>
__device__ __forceinline__ float getScale<float, false>(const float *scale_values, int batch_idx) {
  return 1.0;
}

#ifdef RPU_USE_DOUBLE
template <>
__device__ __forceinline__ double
getScale<double, true>(const double *scale_values, int batch_idx) {
  // UM + inverse (for B (that is x))
  return scale_values[batch_idx];
}

template <>
__device__ __forceinline__ double
getScale<double, false>(const double *scale_values, int batch_idx) {
  return 1.0;
}
#endif

template <> // UBLM
__device__ __forceinline__ int getK<true>(const int *K_values, int batch_idx, int Kplus1) {
  return K_values[batch_idx];
}

template <> // UBLM
__device__ __forceinline__ int getK<false>(const int *K_values, int batch_idx, int Kplus1) {
  return Kplus1 - 1;
}

template <> // UBLM
__device__ __forceinline__ int
getKc<true, uint64_t>(const kagg_t *Kc_values, int batch_idx, int Kplus1) {
  return Kc_values[batch_idx];
}

template <> // UBLM
__device__ __forceinline__ int
getKc<false, uint64_t>(const kagg_t *Kc_values, int batch_idx, int Kplus1) {
  return batch_idx * (Kplus1 - 1);
}

template <> // UBLM
__device__ __forceinline__ int
getKc<true, uint32_t>(const kagg_t *Kc_values, int batch_idx, int Kplus1) {
  return 0; // dummy
}

template <> // UBLM
__device__ __forceinline__ int
getKc<false, uint32_t>(const kagg_t *Kc_values, int batch_idx, int Kplus1) {
  return 0; // dummy
}

template <> // UBLM
__device__ __forceinline__ int getnB<true, uint64_t>(const kagg_t *Kn, int m_batch, int Kplus1) {
  return ((*Kn) + 31) >> 5;
}

template <> // UBLM
__device__ __forceinline__ int getnB<false, uint64_t>(const kagg_t *Kn, int m_batch, int Kplus1) {
  return (m_batch * (Kplus1 - 1) + 31) >> 5;
}

template <> // UBLM
__device__ __forceinline__ int getnB<false, uint32_t>(const kagg_t *Kn, int m_batch, int Kplus1) {
  return 1; // dummy
}
template <> // UBLM
__device__ __forceinline__ int getnB<true, uint32_t>(const kagg_t *Kn, int m_batch, int Kplus1) {
  return 1; // dummy
}

template <>
__device__ __forceinline__ float
getScaleProb<float, true>(const float scaleprob, const int K, const float lr_div_dwmin) {
  return sqrt(lr_div_dwmin / K);
};

// UBLM
template <>
__device__ __forceinline__ float
getScaleProb<float, false>(const float scaleprob, const int K, const float lr_div_dwmin) {
  return scaleprob;
};

#ifdef RPU_USE_DOUBLE
template <>
__device__ __forceinline__ double
getScaleProb<double, true>(const double scaleprob, const int K, const double lr_div_dwmin) {
  return sqrt(lr_div_dwmin / K);
};

template <>
__device__ __forceinline__ double
getScaleProb<double, false>(const double scaleprob, const int K, const double lr_div_dwmin) {
  return scaleprob;
};

#endif

template <>
__device__ __forceinline__ void getCountsSimpleLoop<uint32_t>(
    float value,
    bool negative,
    uint32_t *c,
    int nK32m1,
    int K,
    curandState &local_state,
    int nK32,
    int sz,
    kagg_t Kc) {

  uint32_t ballot = (negative) ? 1 : 0;
  int nK32m1_local = MIN(K >> 5, nK32m1);
  int nn = (nK32m1_local > 0) ? 31 : K;
  PRAGMA(unroll)
  for (int j = 1; j <= nn; j++) {
    float stoch_value = curand_uniform(&local_state);
    ballot |= (stoch_value < value) ? (((uint32_t)1) << j) : (uint32_t)0;
  }
  *c = ballot;
  if (nK32 > 1) {
    ballot = 0;
    int offset = 0;
    PRAGMA(unroll)
    for (int i = 1; i < nK32; i++) {
      offset += sz;
      if (i > nK32m1_local) {
        *(c + offset) = 0;
      } else {
        ballot = 0;
        nn = (i == nK32m1_local) ? (K & 0x1f) : 31;
        PRAGMA(unroll)
        for (int j = 0; j <= nn; j++) {
          float stoch_value = curand_uniform(&local_state);
          ballot |= (stoch_value < value) ? (((uint32_t)1) << j) : (uint32_t)0;
        }
        *(c + offset) = ballot;
      }
    }
  }
}

template <> // count_t
__device__ __forceinline__ void getCountsSimpleLoop<uint64_t>(
    float value,
    bool negative,
    uint64_t *c,
    int nK32m1,
    int K,
    curandState &local_state,
    int nK32,
    int sz,
    kagg_t Kc) {
  static_assert(sizeof(uint64_t) == sizeof(unsigned long long int), "uint64 issue");

  // nK32m1 NEEDS TO BE 0 (otherwise not supported)
  // needs OUTTRANS!!
  int bit_pos_start = Kc & 0x1f;

  uint32_t ballot = 0;
  uint32_t neg_word = (negative) ? (0xffffffff >> (32 - K)) : 0;

  PRAGMA(unroll)
  for (int j = 0; j < K; j++) { // start from zero (no negative bit)
    float stoch_value = curand_uniform(&local_state);
    ballot |= (stoch_value < value) ? (((uint32_t)1) << j) : (uint32_t)0;
  }

  uint64_t ballot64 = (uint64_t)(ballot << bit_pos_start); // may overflow upper bits
  ballot64 |= (((uint64_t)(neg_word << bit_pos_start)) << 32);
  atomicOr((unsigned long long int *)c, (unsigned long long int)ballot64); // save

  if (bit_pos_start + K > 32) {
    // in case of overflow
    ballot64 = (ballot >> (32 - bit_pos_start));
    ballot64 |= (((uint64_t)(neg_word >> (32 - bit_pos_start))) << 32);

    atomicOr((unsigned long long int *)(c + 1), (unsigned long long int)ballot64); // save
  }
}

#define GET_COUNTS_SIMPLE_LOOP_BATCH(                                                              \
    PROB, SIZE, COUNTS, SCALEPROB, TRANS, OUTTRANS, SPROPOP, TIDSTART, TIDEND, TIDN)               \
  {                                                                                                \
    if ((tid >= TIDSTART) && (tid < TIDEND)) {                                                     \
      int sz = SIZE;                                                                               \
      int counts_offset = nK32 * sz;                                                               \
      int n = m_batch * sz;                                                                        \
                                                                                                   \
      for (int i_stride = 0; i_stride < n; i_stride += TIDN) {                                     \
                                                                                                   \
        int idx = (tid - TIDSTART + i_stride);                                                     \
        if (idx < n) {                                                                             \
          T value = PROB[idx];                                                                     \
          int batch_idx = getBatchIdx<TRANS>(idx, sz, m_batch);                                    \
          int K = getK<update_bl_management>(K_values, batch_idx, Kplus1);                         \
          if ((K == 0) || (value == 0)) {                                                          \
            continue;                                                                              \
          }                                                                                        \
          kagg_t Kc = getKc<update_bl_management, count_t>(Kc_values, batch_idx, Kplus1);          \
          T scaleprob = getScaleProb<T, update_bl_management>(SCALEPROB, K, lr_div_dwmin);         \
          T scale = getScale<T, update_management>(scale_values, batch_idx);                       \
          count_t *c = &COUNTS[getCountsIdx<TRANS, OUTTRANS, count_t>(                             \
              idx, sz, m_batch, counts_offset, K, Kc, nB)];                                        \
          T sprob = scaleprob SPROPOP scale;                                                       \
          bool negative = value < 0;                                                               \
          value = (negative) ? -value : value;                                                     \
          value *= sprob;                                                                          \
          DISCRETIZE_VALUE_STOCH(local_state);                                                     \
                                                                                                   \
          getCountsSimpleLoop<count_t>(value, negative, c, nK32m1, K, local_state, nK32, sz, Kc);  \
        }                                                                                          \
      }                                                                                            \
    }                                                                                              \
  }

template <
    typename T,
    bool x_trans,
    bool d_trans,
    bool out_trans,
    bool update_management,
    bool update_bl_management,
    typename count_t,
    typename XInputIteratorT,
    typename DInputIteratorT>
__global__ void kernelUpdateGetCountsBatch_SimpleLoop2(
    XInputIteratorT x_prob,
    int x_size_in,
    T x_scaleprob_in,
    count_t *x_counts,
    DInputIteratorT d_prob,
    int d_size_in,
    T d_scaleprob_in,
    count_t *d_counts,
    int Kplus1_in,
    int m_batch_in,
    curandState *random_states,
    T resolution,
    bool sto_round,
    const T *scale_values = nullptr,
    const int *K_values = nullptr,
    const T lr_div_dwmin_in = 1.0,
    const kagg_t *Kc_values = nullptr,
    const kagg_t *Kn = nullptr)

{
  // -- each thread computes all the  K values
  // -- no limit for number of threads. However, occupy all stream processors once should reduce
  // overhead
  // -- for UM: scale values should be sqrt(amax_x/amax_d) for D
  // -- for UM: scale values has to be strictly POSITIVE (NON-zero)!!
  // -- ASSUMES: NGRID>1 !! (nblocks>1)
  // -- RANDOMSTATES need to have 1 for each tid.
  // -- CAUTION: counts should be set to zero!!!
  //
  // In the case of uint64_t:
  // -- using atomics to save the counts. CAUTION: counts should be set to zero!!!
  // -- ONLY for out_trans=true
  // -- only K <= 31 supported (nK32==1)
  // -- CAUTION: some bit boundardy issue? Very seldom 64 version seems one bit off.. ignore. Has no
  // relevance with noise
  //             could be just a rounding issue

  const int tid = blockDim.x * blockIdx.x + threadIdx.x; // can be larger that x_size or d_size
  const int x_size = x_size_in;
  const int d_size = d_size_in;
  const int m_batch = m_batch_in;
  const int Kplus1 = Kplus1_in;
  const int nB = getnB<update_bl_management, count_t>(Kn, m_batch, Kplus1);

  const T x_scaleprob = x_scaleprob_in;
  const T d_scaleprob = d_scaleprob_in;

  int nx_blocks = ceil(gridDim.x * ((T)x_size / (T)(x_size + d_size)));
  int nd_blocks = gridDim.x - nx_blocks;
  if ((nd_blocks <= 0) && (d_size > 0)) {
    nx_blocks = gridDim.x - 1; // ASSUMES gridDim.x>1 !~
    nd_blocks = 1;
  }
  const int tid_nx = nx_blocks * blockDim.x;
  const int tid_nd = nd_blocks * blockDim.x;

  if ((tid < tid_nx) && (tid > x_size * m_batch))
    return;
  if ((tid >= tid_nx) && (tid - tid_nx > d_size * m_batch))
    return;

  const T lr_div_dwmin = lr_div_dwmin_in;

  curandState local_state = random_states[tid];
  RPU_BLM_DEFINE_NK32;

  DISCRETIZE_VALUE_STOCH_DEFINITIONS;

  // x input
  GET_COUNTS_SIMPLE_LOOP_BATCH(
      x_prob, x_size, x_counts, x_scaleprob, x_trans, out_trans, /, 0, tid_nx, tid_nx);

  // d input
  GET_COUNTS_SIMPLE_LOOP_BATCH(
      d_prob, d_size, d_counts, d_scaleprob, d_trans, out_trans, *, tid_nx, tid_nx + tid_nd,
      tid_nd);

  // save new random states
  random_states[tid] = local_state;
}

namespace test_helper {
template <typename T>
int debugKernelUpdateGetCountsBatch_SimpleLoop2(
    T *indata, int size, T scaleprob, uint32_t *counts, int K, T *timing, bool fake_seed) {
  // counts should be: size*nk32 allocated !
  RPU_BLM_DEBUG_DEFINE_K_BATCH;

  int nthreads = RPU_THREADS_PER_BLOCK_UPDATE;

  int m = MIN(size * m_batch, nthreads * 12);
  int nblocks = MAX((m + nthreads - 1) / nthreads, 2);
  std::cout << "nblocks, nthreads: " << nblocks << ", " << nthreads << std::endl;
  int n = m;
  RPU_BLM_DEBUG_BATCH_INIT(n, uint32_t);

  kernelUpdateGetCountsBatch_SimpleLoop2<T, false, false, false, false, false>
      <<<nblocks, nthreads, 0, c.getStream()>>>(
          dev_indata.getData(), size, scaleprob, dev_counts.getData(), dev_indata2.getData(), size,
          scaleprob, dev_counts2.getData(), Kplus1, m_batch, dev_states.getData(), resolution,
          sto_round);

  RPU_BLM_DEBUG_BATCH_FINISH(uint32_t);
  return 0;
}
template int debugKernelUpdateGetCountsBatch_SimpleLoop2<float>(
    float *, int, float, unsigned int *, int, float *, bool);
#ifdef RPU_USE_DOUBLE
template int debugKernelUpdateGetCountsBatch_SimpleLoop2<double>(
    double *, int, double, unsigned int *, int, double *, bool);
#endif
} // namespace test_helper

// *********************************************************************************
// kernelUpdateGetCounts_Loop2
template <typename T, typename XInputIteratorT, typename DInputIteratorT>
__global__ void kernelUpdateGetCounts_Loop2(
    XInputIteratorT x_prob,
    int x_size_in,
    T x_scaleprob,
    uint32_t *x_counts,
    DInputIteratorT d_prob,
    int d_size_in,
    T d_scaleprob,
    uint32_t *d_counts,
    int Kplus1,
    curandState *random_states,
    T resolution,
    bool sto_round) {
  // call << (size/numwarpsperblock,1),warpSize*numwarpsperblock >>

  // -- let each warp compute 32 K values
  // -- no limit for K , however BLocked design might be better for larger K

  volatile int tid = blockDim.x * blockIdx.x + threadIdx.x;
  const int x_size = x_size_in;
  const int d_size = d_size_in;

  const int max_size = ((x_size > d_size) ? x_size : d_size) << 5;

  if (tid >= max_size)
    return;

  curandState local_state = random_states[tid];

  RPU_BLM_DEFINE_NK32;
  const uint32_t one = 1;
  const uint32_t lastK32mask = LASTK32MASK;

  const int laneId = threadIdx.x & 0x1f;
  // const uint32_t sourceId =  blockIdx.x*warps_per_block + warpId;
  const int sourceId = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);

  DISCRETIZE_VALUE_STOCH_DEFINITIONS;

  uint32_t ballot = 0;

  T value;
  uint32_t *c;
  bool negative;
  int sz;

  // x input
  GET_COUNTS_LOOP(x_prob, x_size, x_counts, x_scaleprob);

  // d input
  GET_COUNTS_LOOP(d_prob, d_size, d_counts, d_scaleprob);

  // save new random states
  random_states[tid] = local_state;
}

namespace test_helper {
template <typename T>
int debugKernelUpdateGetCounts_Loop2(
    T *indata, int size, T scaleprob, uint32_t *counts, int K, T *timing, bool fake_seed) {
  // counts should be: size*nk32 allocated !
  RPU_BLM_DEBUG_DEFINE_K;

  dim3 nthreads = dim3(RPU_THREADS_PER_BLOCK_UPDATE);
  int numwarpsperblock = RPU_THREADS_PER_BLOCK_UPDATE / 32;
  dim3 nblocks = dim3((size + numwarpsperblock - 1) / numwarpsperblock);

  int n = size * 32;

  RPU_BLM_DEBUG_INIT(n);

  kernelUpdateGetCounts_Loop2<<<nblocks, nthreads, 0, c.getStream()>>>(
      dev_indata.getData(), size, scaleprob, dev_counts.getData(), dev_indata.getData(), 0,
      scaleprob, dev_counts.getData(), Kplus1, dev_states.getData(), resolution, sto_round);

  RPU_BLM_DEBUG_FINISH;
  return 0;
}
template int
debugKernelUpdateGetCounts_Loop2<float>(float *, int, float, unsigned int *, int, float *, bool);
#ifdef RPU_USE_DOUBLE
template int debugKernelUpdateGetCounts_Loop2<double>(
    double *, int, double, unsigned int *, int, double *, bool);
#endif

} // namespace test_helper

// *********************************************************************************
// kernelUpdateGetImplicitPulses
#define GET_COUNTS_SIMPLE_LOOP_BATCH_IMPLICIT(                                                     \
    PROB, SIZE, COUNTS, SCALEPROB, RES, TRANS, OUTTRANS, SPROPOP, TIMESK, TIDSTART, TIDEND, TIDN)  \
  {                                                                                                \
    if ((tid >= TIDSTART) && (tid < TIDEND)) {                                                     \
      int sz = SIZE;                                                                               \
      int n = m_batch * sz;                                                                        \
                                                                                                   \
      for (int i_stride = 0; i_stride < n; i_stride += TIDN) {                                     \
                                                                                                   \
        int idx = (tid - TIDSTART + i_stride);                                                     \
        if (idx < n) {                                                                             \
          T value = PROB[idx];                                                                     \
          int batch_idx = getBatchIdx<TRANS>(idx, sz, m_batch);                                    \
          int K = getK<update_bl_management>(K_values, batch_idx, Kplus1);                         \
          T scaleprob = getScaleProb<T, update_bl_management>(SCALEPROB, K, lr_div_dwmin);         \
          T scale = getScale<T, update_management>(scale_values, batch_idx);                       \
          T *c = &COUNTS[getCountsIdx<TRANS, OUTTRANS, uint32_t>(idx, sz, m_batch, sz, K, 0, 1)];  \
          T sprob = scaleprob SPROPOP scale;                                                       \
          value *= sprob;                                                                          \
          value = MIN(MAX(value, -1.0), 1.0);                                                      \
          DISCRETIZE_VALUE(RES);                                                                   \
          *c = value TIMESK;                                                                       \
        }                                                                                          \
      }                                                                                            \
    }                                                                                              \
  }

template <
    typename T,
    bool x_trans,
    bool d_trans,
    bool out_trans,
    bool update_management,
    bool update_bl_management,
    typename XInputIteratorT,
    typename DInputIteratorT>
__global__ void kernelUpdateGetCountsBatchImplicit(
    XInputIteratorT x_prob,
    int x_size_in,
    T x_scaleprob_in,
    T x_res_in,
    T *x_counts,
    DInputIteratorT d_prob,
    int d_size_in,
    T d_scaleprob_in,
    T d_res_in,
    T *d_counts,
    int Kplus1_in,
    int m_batch_in,
    const T *scale_values = nullptr,
    const int *K_values = nullptr,
    const T lr_div_dwmin_in = 1.0)

{
  // same as SimpleLoop2 but implicit pulses (that is no pulses, just discretization)
  const int tid = blockDim.x * blockIdx.x + threadIdx.x; // can be larger that x_size or d_size
  const int x_size = x_size_in;
  const int d_size = d_size_in;
  const int m_batch = m_batch_in;
  const int Kplus1 = Kplus1_in;

  const T x_res = x_res_in;
  const T d_res = d_res_in;
  const T x_scaleprob = x_scaleprob_in;
  const T d_scaleprob = d_scaleprob_in;

  int nx_blocks = ceil(gridDim.x * ((T)x_size / (T)(x_size + d_size)));
  int nd_blocks = gridDim.x - nx_blocks;
  if ((nd_blocks <= 0) && (d_size > 0)) {
    nx_blocks = gridDim.x - 1; // ASSUMES gridDim.x>1 !~
    nd_blocks = 1;
  }
  const int tid_nx = nx_blocks * blockDim.x;
  const int tid_nd = nd_blocks * blockDim.x;

  if ((tid < tid_nx) && (tid > x_size * m_batch))
    return;
  if ((tid >= tid_nx) && (tid - tid_nx > d_size * m_batch))
    return;

  const T lr_div_dwmin = lr_div_dwmin_in;

  // x input // gets the K mult
  GET_COUNTS_SIMPLE_LOOP_BATCH_IMPLICIT(
      x_prob, x_size, x_counts, x_scaleprob, x_res, x_trans, out_trans, /, *K, 0, tid_nx, tid_nx);

  // d input
  GET_COUNTS_SIMPLE_LOOP_BATCH_IMPLICIT(
      d_prob, d_size, d_counts, d_scaleprob, d_res, d_trans, out_trans, *, , tid_nx,
      tid_nx + tid_nd, tid_nd);
}

namespace test_helper {
template <typename T>
int debugKernelUpdateGetCountsBatchImplicit(
    T *indata, int size, T scaleprob, T *counts, int K, T *timing, bool fake_seed) {
  // counts should be: size allocated !
  RPU_BLM_DEBUG_DEFINE_K_NO_STOROUND;
  int m_batch = 1;

  int nthreads = RPU_THREADS_PER_BLOCK_UPDATE;

  int m = MIN(size * m_batch, nthreads * 12);
  int nblocks = MAX((m + nthreads - 1) / nthreads, 2);
  std::cout << "nblocks, nthreads: " << nblocks << ", " << nthreads << std::endl;
  int n = m;
  RPU_BLM_DEBUG_BATCH_INIT(n, T);

  kernelUpdateGetCountsBatchImplicit<T, false, false, false, false, false>
      <<<nblocks, nthreads, 0, c.getStream()>>>(
          dev_indata.getData(), size, scaleprob, resolution, dev_counts.getData(),
          dev_indata2.getData(), size, scaleprob, resolution, dev_counts2.getData(), Kplus1,
          m_batch);

  RPU_BLM_DEBUG_BATCH_FINISH(T);
  return 0;
}
template int
debugKernelUpdateGetCountsBatchImplicit<float>(float *, int, float, float *, int, float *, bool);
#ifdef RPU_USE_DOUBLE
template int debugKernelUpdateGetCountsBatchImplicit<double>(
    double *, int, double, double *, int, double *, bool);
#endif
} // namespace test_helper

/****************************************************************************************************************/
/* BITLINEMAKER */
/******************************************************************************************************************/

#define RPU_BLM_ITEMS_PER_THREAD 4
#define RPU_BLM_BL_TO_SELECT_SIMPLE_LOOP 0

template <typename T>
BitLineMaker<T>::BitLineMaker(CudaContext *c, int x_size, int d_size)
    : context_{c}, x_size_{x_size}, d_size_{d_size}, umh_{nullptr}, buffer_m_batch_{0} {

  max_block_count_ =
      context_->getSMCount() * (context_->maxThreadsPerSM() / RPU_THREADS_PER_BLOCK_UPDATE);

  nthreads_ = RPU_THREADS_PER_BLOCK_UPDATE;
}

template <typename T>
BLMOutputFormat BitLineMaker<T>::getFormat(int use_bo64, bool implicit_pulses) {

  if (implicit_pulses && use_bo64 == 0) {
    return BLMOutputFormat::FP;
  } else if (use_bo64 == 1 && !implicit_pulses) {
    return BLMOutputFormat::BO64;
  } else if (use_bo64 == 2 && !implicit_pulses) {
    return BLMOutputFormat::UI32BO64;
  } else if (use_bo64 == 0 && !implicit_pulses) {
    return BLMOutputFormat::UI32;
  } else {
    RPU_FATAL("Not able to determine BLM output format");
  }
}

template <typename T> T *BitLineMaker<T>::getXData() const {
  return format_ == BLMOutputFormat::FP ? dev_x_->getData() : nullptr;
};

template <typename T> T *BitLineMaker<T>::getDData() const {
  return format_ == BLMOutputFormat::FP ? dev_d_->getData() : nullptr;
};

template <typename T> uint32_t *BitLineMaker<T>::getXCountsData() const {
  return (format_ == BLMOutputFormat::UI32 || format_ == BLMOutputFormat::UI32BO64)
             ? dev_x_counts_->getData()
             : nullptr;
};

template <typename T> uint32_t *BitLineMaker<T>::getDCountsData() const {
  return (format_ == BLMOutputFormat::UI32 || format_ == BLMOutputFormat::UI32BO64)
             ? dev_d_counts_->getData()
             : nullptr;
};

template <typename T> uint64_t *BitLineMaker<T>::getXCountsBo64Data() const {
  return (format_ == BLMOutputFormat::BO64 || format_ == BLMOutputFormat::UI32BO64)
             ? dev_x_counts_bo64_->getData()
             : nullptr;
};

template <typename T> uint64_t *BitLineMaker<T>::getDCountsBo64Data() const {
  return (format_ == BLMOutputFormat::BO64 || format_ == BLMOutputFormat::UI32BO64)
             ? dev_d_counts_bo64_->getData()
             : nullptr;
};

template <typename T> kagg_t *BitLineMaker<T>::getKnData(bool ublm) const {
  return umh_->getKnData(ublm);
};

template <typename T> int BitLineMaker<T>::getBo64Batch(int m_batch) const {
  return umh_->getBo64Batch(m_batch, current_BL_);
};

template <typename T> void BitLineMaker<T>::copyXCountsToHost(uint32_t *dest) const {
  if (!(format_ == BLMOutputFormat::UI32 || format_ == BLMOutputFormat::UI32BO64)) {
    RPU_FATAL("Wrong format!");
  }
  dev_x_counts_->copyTo(dest);
};

template <typename T> void BitLineMaker<T>::copyDCountsToHost(uint32_t *dest) const {
  if (!(format_ == BLMOutputFormat::UI32 || format_ == BLMOutputFormat::UI32BO64)) {
    RPU_FATAL("Wrong format!");
  }
  dev_d_counts_->copyTo(dest);
};

template <typename T> void BitLineMaker<T>::copyXCountsBo64ToHost(uint64_t *dest) const {
  if (!(format_ == BLMOutputFormat::BO64 || format_ == BLMOutputFormat::UI32BO64)) {
    RPU_FATAL("Wrong format!");
  }
  dev_x_counts_bo64_->copyTo(dest);
};

template <typename T> void BitLineMaker<T>::copyDCountsBo64ToHost(uint64_t *dest) const {
  if (!(format_ == BLMOutputFormat::BO64 || format_ == BLMOutputFormat::UI32BO64)) {
    RPU_FATAL("Wrong format!");
  }
  dev_d_counts_bo64_->copyTo(dest);
};

template <typename T>
void BitLineMaker<T>::initializeBLBuffers(int m_batch, int BL, int use_bo64, bool implicit_pulses) {

  buffer_m_batch_ = m_batch;
  buffer_BL_ = BL;
  format_ = getFormat(use_bo64, implicit_pulses);

  if (format_ == BLMOutputFormat::FP) {
    dev_d_ = RPU::make_unique<CudaArray<T>>(context_, d_size_ * m_batch);
    dev_x_ = RPU::make_unique<CudaArray<T>>(context_, x_size_ * m_batch);

  } else {
    int nK32 = BL / 32 + 1; // equivalent to ((BL+1) + 31)/32
    if (format_ == BLMOutputFormat::UI32 || format_ == BLMOutputFormat::UI32BO64) {
      dev_d_counts_ = RPU::make_unique<CudaArray<uint32_t>>(context_, d_size_ * (nK32)*m_batch);
      dev_x_counts_ = RPU::make_unique<CudaArray<uint32_t>>(context_, x_size_ * (nK32)*m_batch);
    }

    if (format_ == BLMOutputFormat::BO64 || format_ == BLMOutputFormat::UI32BO64) {
      if (nK32 > 1) {
        RPU_FATAL("BL>31 is not supported for BO64");
      }
      dev_d_counts_bo64_ = RPU::make_unique<CudaArray<uint64_t>>(context_, d_size_ * m_batch);
      dev_x_counts_bo64_ = RPU::make_unique<CudaArray<uint64_t>>(context_, x_size_ * m_batch);
    }
  }

  context_->synchronize();

  DEBUG_OUT("BLM init BL buffers with batch " << m_batch << " and BL " << BL << ".");
}

template <typename T> void BitLineMaker<T>::getCountsDebug(uint32_t *x_counts, uint32_t *d_counts) {

  if (!(format_ == BLMOutputFormat::UI32 || format_ == BLMOutputFormat::UI32BO64)) {
    RPU_FATAL("Wrong format output requested!");
  }

  dev_x_counts_->copyTo(x_counts);
  dev_d_counts_->copyTo(d_counts);
}

template <typename T> void BitLineMaker<T>::getFPCounts(T *x_counts, T *d_counts) {

  if (format_ != BLMOutputFormat::FP) {
    RPU_FATAL("Wrong format output requested!");
  }

  dev_x_->copyTo(x_counts);
  dev_d_->copyTo(d_counts);
}

#define RPU_BLM_START_KERNEL_LINEAR(ITEM_PER_THREAD)                                               \
  int n = (Kplus1 / ITEM_PER_THREAD);                                                              \
                                                                                                   \
  int nblocks = context_->getNBlocks(x_size_ * n, nthreads_);                                      \
  kernelUpdateGetCounts_Linear<T, ITEM_PER_THREAD><<<nblocks, nthreads_, 0, s>>>(                  \
      x_in, x_size_, B, dev_x_counts_->getData(), Kplus1,                                          \
      context_->getRandomStates(nthreads_ * nblocks), res, sr);                                    \
                                                                                                   \
  nblocks = context_->getNBlocks(d_size_ * n, nthreads_);                                          \
  kernelUpdateGetCounts_Linear<T, ITEM_PER_THREAD><<<nblocks, nthreads_, 0, s>>>(                  \
      d_in, d_size_, A, dev_d_counts_->getData(), Kplus1,                                          \
      context_->getRandomStates(nthreads_ * nblocks), res, sr);

template <typename T>
template <typename XInputIteratorT, typename DInputIteratorT>
void BitLineMaker<T>::makeCounts(
    XInputIteratorT x_in,
    DInputIteratorT d_in,
    const PulsedUpdateMetaParameter<T> &up,
    const T dw_min,
    const T lr,
    const int m_batch,
    const bool x_trans,
    const bool d_trans,
    const bool out_trans,
    const int use_bo64,
    const bool implicit_pulses) {
  // use_bo64==1 : direct bo64
  // use_bo64==2 : translate into bo64

  // just a double-check. In principle input implicit_pulses could
  // be omitted. However, it is a safe-guard to make sure the kpars
  // settings are correct.
  if (up.needsImplicitPulses() != implicit_pulses) {
    RPU_FATAL("mixed up implicit pulses settings");
  }

  // update management
  T A = 0;
  T B = 0;
  up.calculateBlAB(current_BL_, A, B, lr, dw_min);
  current_lr_ = lr; // save for rpu device if needed

  bool update_management = up.update_management;
  bool update_bl_management = up.update_bl_management;
  bool sr = up.sto_round;
  T res = up.res;

  cudaStream_t s = this->context_->getStream();

  if (format_ != getFormat(use_bo64, implicit_pulses) || (buffer_BL_ / 32 < current_BL_ / 32) ||
      (buffer_m_batch_ < m_batch)) {
    initializeBLBuffers(m_batch, current_BL_, use_bo64, implicit_pulses);
  }

  if ((use_bo64 > 0) && !out_trans) {
    RPU_FATAL("out_trans=false not supported for BO64");
  }

  bool um_if = update_management || update_bl_management;
  T *scale_values = nullptr;
  int *K_values = nullptr;

  if (um_if || use_bo64 > 0) {
    if (umh_ == nullptr) {
      umh_ = RPU::make_unique<UpdateManagementHelper<T>>(context_, x_size_, d_size_);
    }
    if (um_if) {
      umh_->computeKandScaleValues(
          x_in, d_in, dw_min, lr, update_management, update_bl_management, m_batch, x_trans,
          d_trans, current_BL_);

      scale_values = umh_->getScaleValueData();
      K_values = umh_->getKValueData();
    }
  }

  // ------- generate the requested bit lines

  switch (up.pulse_type) {

  case PulseType::DeterministicImplicit: {
    // we do not actually generate bitlines here but discretize and
    // scale x and d according to the update_management

    // we first scale x and d explicitely (to simulate the selection
    // of the bit line from memory) and then multiply. Multplication
    // is done in the update kernel directly.

    // Note that we here also multiply with BL already (after the
    // discritezation) so that counts are simply generated by
    //
    // x_val = blm->getXData(); d_val = blm->getDData();
    //
    // n = floor(x_val*d_val + 0.4999);
    //
    // sign is seperately taken into account so that floor does the
    // right thing.

    if (format_ != BLMOutputFormat::FP) {
      RPU_FATAL("Expects to be in float mode!");
    }

    int m = (d_size_ + x_size_) * m_batch;
    int nblocks = context_->getNBlocks(m, nthreads_);
    nblocks = MAX(MIN(max_block_count_, nblocks), 2);

    RPU_BLM_SWITCH_TRANS_TEMPLATE_UM(
        x_trans, d_trans, out_trans, update_management, update_bl_management,
        kernelUpdateGetCountsBatchImplicit,
        (x_in, x_size_, B, up.x_res_implicit, dev_x_->getData(), d_in, d_size_, A,
         up.d_res_implicit, dev_d_->getData(), current_BL_ + 1, m_batch, scale_values, K_values,
         lr / dw_min));

  } break;

  case PulseType::StochasticCompressed: {
    // here we generate stochastic bitlines. These are either in 64
    // bit format (32 bits for sign and 32 bits for data) or
    // standard 32 bit format. In the latter case the first bit is
    // the sign bit, which iis the same for the whole word. Longer
    // bitlines can be accomodated by adding words (nK32>1; only in
    // 32-bit case).

    if (format_ == BLMOutputFormat::FP) {
      RPU_FATAL("Expects to be NOT in floating point mode!");
    }

    int Kplus1 = current_BL_ + 1;
    bool possible_linear = (Kplus1 <= 32) && ((32 % Kplus1) == 0);
    // always do simple_loop: fastest in any case
    bool simple_loop = (Kplus1 > RPU_BLM_BL_TO_SELECT_SIMPLE_LOOP) || um_if;
    simple_loop |= use_bo64 == 1; // direct bo64 only supported in simple loop

    if ((m_batch == 1) && (possible_linear && (!simple_loop))) {

      if (!possible_linear) {

        // one block  is a little bit faster than TWOBLOCKS
        int nblocks = context_->getNBlocks(MAX(d_size_, x_size_) * 32, nthreads_);

        kernelUpdateGetCounts_Loop2<<<nblocks, nthreads_, 0, s>>>(
            x_in, x_size_, B, dev_x_counts_->getData(), d_in, d_size_, A, dev_d_counts_->getData(),
            Kplus1, context_->getRandomStates(nthreads_ * nblocks), res, sr);

      } else { // fast path for smaller K values (needs to be K<=32!)

        if ((Kplus1 % RPU_BLM_ITEMS_PER_THREAD) != 0) {
          // just set to 2 (smallest possible)

          RPU_BLM_START_KERNEL_LINEAR(2);

        } else {

          RPU_BLM_START_KERNEL_LINEAR(RPU_BLM_ITEMS_PER_THREAD);
        }
      }

    } else {
      // batch or single batch with simple loop

      if (simple_loop) {

        int m = (d_size_ + x_size_) * m_batch;
        int nblocks = context_->getNBlocks(m, nthreads_);
        nblocks = MAX(MIN(max_block_count_, nblocks), 2);

        if (use_bo64 == 1) {

          // need to set buffers to zero
          dev_x_counts_bo64_->setConst(0);
          dev_d_counts_bo64_->setConst(0);

          kagg_t *Kc_values = nullptr;
          if (update_bl_management) {
            umh_->computeKc(m_batch);
            umh_->computeKn(m_batch);
            Kc_values = umh_->getKcValueData();
          }

          RPU_BLM_SWITCH_TRANS_TEMPLATE_UM(
              x_trans, d_trans, out_trans, update_management, update_bl_management,
              kernelUpdateGetCountsBatch_SimpleLoop2,
              (x_in, x_size_, B, dev_x_counts_bo64_->getData(), d_in, d_size_, A,
               dev_d_counts_bo64_->getData(), current_BL_ + 1, m_batch,
               context_->getRandomStates(nthreads_ * nblocks), res, sr, scale_values, K_values,
               lr / dw_min, Kc_values, umh_->getKnData(update_bl_management)));

        } else {

          // need to set buffers to zero for zero short-cut
          dev_x_counts_->setConst(0);
          dev_d_counts_->setConst(0);

          RPU_BLM_SWITCH_TRANS_TEMPLATE_UM(
              x_trans, d_trans, out_trans, update_management, update_bl_management,
              kernelUpdateGetCountsBatch_SimpleLoop2,
              (x_in, x_size_, B, dev_x_counts_->getData(), d_in, d_size_, A,
               dev_d_counts_->getData(), current_BL_ + 1, m_batch,
               context_->getRandomStates(nthreads_ * nblocks), res, sr, scale_values, K_values,
               lr / dw_min));
        }
      } else {
        int m = MAX(d_size_, x_size_) * m_batch * 32;
        int nblocks = context_->getNBlocks(m, nthreads_);
        nblocks = MIN(max_block_count_, nblocks);

        RPU_BLM_SWITCH_TRANS_TEMPLATE(
            x_trans, d_trans, out_trans, kernelUpdateGetCountsBatch_Loop2,
            (x_in, x_size_, B, dev_x_counts_->getData(), d_in, d_size_, A, dev_d_counts_->getData(),
             current_BL_ + 1, m_batch, context_->getRandomStates(nthreads_ * nblocks), res, sr), );
      }
    }
    // translate to BO64 if necessary
    if (use_bo64 > 1) {
      umh_->translateTransToBatchOrder64(
          dev_x_counts_bo64_->getData(), dev_d_counts_bo64_->getData(), dev_x_counts_->getData(),
          dev_d_counts_->getData(), m_batch, current_BL_, update_bl_management);
    }

    DEBUG_CALL(context_->synchronizeDevice(); CudaArray<T> dev_x(context_, x_size_);
               CudaArray<T> dev_d(context_, d_size_);
               RPU::math::copyWithIterator(context_, dev_x.getData(), x_in, x_size_);
               RPU::math::copyWithIterator(context_, dev_d.getData(), d_in, d_size_);
               context_->synchronizeDevice(); test_helper::checkCounts(
                   dev_x.getData(), x_size_, dev_d.getData(), d_size_, current_BL_, A, B,
                   &*dev_x_counts_, &*dev_d_counts_);
               context_->synchronizeDevice(););
  } break;

  default:
    RPU_FATAL("PulseType not supported by BitLineMaker");
  }
};

template class BitLineMaker<float>;
#ifdef RPU_USE_DOUBLE
template class BitLineMaker<double>;
#endif

#define RPU_BLM_ITER_TEMPLATE(NUM_T, XITERT, DITERT)                                               \
  template void BitLineMaker<NUM_T>::makeCounts(                                                   \
      XITERT, DITERT, const PulsedUpdateMetaParameter<NUM_T> &, const NUM_T, const NUM_T,          \
      const int, const bool, const bool, const bool, const int, const bool);

#define TRANSFLOAT(TRANS) TRANS, float

RPU_BLM_ITER_TEMPLATE(float, const float *, const float *);
RPU_BLM_ITER_TEMPLATE(float, float *, float *);
RPU_BLM_ITER_TEMPLATE(float, IndexReaderInputIterator<float>, const float *);
RPU_BLM_ITER_TEMPLATE(float, IndexReaderTransInputIterator<float>, const float *);

RPU_BLM_ITER_TEMPLATE(
    float, IndexReaderTransInputIterator<float>, PermuterTransInputIterator<float>);
RPU_BLM_ITER_TEMPLATE(
    float, IndexReaderSliceInputIterator<TRANSFLOAT(true)>, SliceInputIterator<TRANSFLOAT(true)>);
RPU_BLM_ITER_TEMPLATE(
    float, IndexReaderSliceInputIterator<TRANSFLOAT(false)>, SliceInputIterator<TRANSFLOAT(false)>);

RPU_BLM_ITER_TEMPLATE(float, const float *, PermuterTransInputIterator<float>);
RPU_BLM_ITER_TEMPLATE(float, const float *, SliceInputIterator<TRANSFLOAT(true)>);
RPU_BLM_ITER_TEMPLATE(float, const float *, SliceInputIterator<TRANSFLOAT(false)>);
RPU_BLM_ITER_TEMPLATE(float, IndexReaderSliceInputIterator<TRANSFLOAT(true)>, const float *);
RPU_BLM_ITER_TEMPLATE(float, IndexReaderSliceInputIterator<TRANSFLOAT(false)>, const float *);

#undef TRANSFLOAT

#ifdef RPU_USE_DOUBLE
#define TRANSDOUBLE(TRANS) TRANS, double

RPU_BLM_ITER_TEMPLATE(double, const double *, const double *);
RPU_BLM_ITER_TEMPLATE(double, double *, double *);
RPU_BLM_ITER_TEMPLATE(double, IndexReaderInputIterator<double>, const double *);
RPU_BLM_ITER_TEMPLATE(double, IndexReaderTransInputIterator<double>, const double *);
RPU_BLM_ITER_TEMPLATE(
    double, IndexReaderTransInputIterator<double>, PermuterTransInputIterator<double>);
RPU_BLM_ITER_TEMPLATE(
    double,
    IndexReaderSliceInputIterator<TRANSDOUBLE(true)>,
    SliceInputIterator<TRANSDOUBLE(true)>);
RPU_BLM_ITER_TEMPLATE(
    double,
    IndexReaderSliceInputIterator<TRANSDOUBLE(false)>,
    SliceInputIterator<TRANSDOUBLE(false)>);

RPU_BLM_ITER_TEMPLATE(double, const double *, PermuterTransInputIterator<double>);
RPU_BLM_ITER_TEMPLATE(double, const double *, SliceInputIterator<TRANSDOUBLE(true)>);
RPU_BLM_ITER_TEMPLATE(double, const double *, SliceInputIterator<TRANSDOUBLE(false)>);
RPU_BLM_ITER_TEMPLATE(double, IndexReaderSliceInputIterator<TRANSDOUBLE(true)>, const double *);
RPU_BLM_ITER_TEMPLATE(double, IndexReaderSliceInputIterator<TRANSDOUBLE(false)>, const double *);

#undef TRANSDOUBLE
#endif

#undef RPU_BLM_ITER_TEMPLATE

#undef RPU_BLM_SWITCH_TRANS_TEMPLATE
#undef RPU_BLM_SWITCH_TRANS_TEMPLATE_UM
#undef RPU_BLM_ITEMS_PER_THREAD
#undef RPU_BLM_START_KERNEL_LINEAR
#undef RPU_BLM_BL_TO_SELECT_SIMPLE_LOOP
#undef LASTK32MASK
#undef RPU_BLM_DEFINE_NK32
#undef RPU_BLM_DEFINE_NK32_BATCH
#undef COMMA
#undef RPU_BLM_DEBUG_INIT
#undef RPU_BLM_DEBUG_FINISH
#undef RPU_BLM_DEBUG_BATCH_INIT
#undef RPU_BLM_DEBUG_BATCH_FINISH
#undef DISCRETIZE_VALUE_STOCH_DEFINITIONS
#undef DISCRETIZE_VALUE_STOCH
#undef DISCRETIZE_VALUE
#undef GET_COUNTS_INNER_LOOP
#undef GET_COUNTS_LOOP
#undef GET_COUNTS_LOOP_BATCH
#undef GET_COUNTS_SIMPLE_LOOP_BATCH
#undef RPU_BLM_BLOCKS_PER_SM
#undef RPU_BLM_DEBUG_DEFINE_K_BATCH
#undef RPU_BLM_DEBUG_DEFINE_K_NO_STOROUND
#undef RPU_BLM_DEBUG_DEFINE_K
} // namespace RPU
