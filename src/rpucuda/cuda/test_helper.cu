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

#include "cuda_math_util.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <random>

#include "pwu_kernel.h"
#include "test_helper.h"
#include "update_management_helper.h"

#define TOL 1e-5

namespace RPU {

namespace test_helper {

#define DEBUG_KERNEL_UPDATE_BATCH_INIT(CONTEXT, STATE_SIZE, M_BATCH)                               \
  int size = x_size * d_size;                                                                      \
  param_t *tmp = new param_t[size * 4];                                                            \
  for (int i = 0; i < size; i++) {                                                                 \
    int k = i * 4;                                                                                 \
    tmp[k] = -bound;                                                                               \
    tmp[k + 1] = dw_min;                                                                           \
    tmp[k + 2] = bound;                                                                            \
    tmp[k + 3] = dw_min;                                                                           \
  }                                                                                                \
  CudaArray<param_t> dev_4params(CONTEXT, size * 4, tmp);                                          \
  delete[] tmp;                                                                                    \
  CudaArray<T> dev_weights(CONTEXT, size, weights);                                                \
  CudaArray<uint32_t> dev_x_counts(CONTEXT, x_size *nK32 *M_BATCH, x_counts);                      \
  CudaArray<uint32_t> dev_d_counts(CONTEXT, d_size *nK32 *M_BATCH, d_counts);                      \
                                                                                                   \
  CudaArray<curandState> dev_states(CONTEXT, STATE_SIZE);                                          \
  curandSetup(dev_states);                                                                         \
  CUDA_CALL(cudaDeviceSynchronize());                                                              \
                                                                                                   \
  cudaEvent_t start, stop;                                                                         \
  CUDA_CALL(cudaEventCreate(&start));                                                              \
  CUDA_CALL(cudaEventCreate(&stop));                                                               \
  CUDA_CALL(cudaEventRecord(start, CONTEXT->getStream()));

#define DEBUG_KERNEL_UPDATE_CLEANUP(CONTEXT)                                                       \
  CUDA_CALL(cudaEventRecord(stop, CONTEXT->getStream()));                                          \
  CUDA_CALL(cudaDeviceSynchronize());                                                              \
  cudaEventSynchronize(stop);                                                                      \
  CUDA_CALL(cudaPeekAtLastError());                                                                \
  float milliseconds = 0;                                                                          \
  cudaEventElapsedTime(&milliseconds, start, stop);                                                \
  *timings = milliseconds;                                                                         \
  CUDA_CALL(cudaDeviceSynchronize());                                                              \
  dev_weights.copyTo(weights);                                                                     \
  CUDA_CALL(cudaDeviceSynchronize());                                                              \
  CUDA_CALL(cudaEventDestroy(start));                                                              \
  CUDA_CALL(cudaEventDestroy(stop));                                                               \
  return 0;

template <typename T>
int debugKernelUpdateW(
    T *weights,
    uint32_t *x_counts,
    int x_size,
    uint32_t *d_counts,
    int d_size,
    int nK32,
    T dw_min,
    T dw_min_std,
    T bound,
    T *timings) {
  auto c_container = CudaContext(-1, false);
  CudaContextPtr c = &c_container;

  int nthreads = RPU_THREADS_PER_BLOCK_UPDATE;
  int nblocks = (d_size * x_size + RPU_THREADS_PER_BLOCK_UPDATE - 1) / RPU_THREADS_PER_BLOCK_UPDATE;

  DEBUG_KERNEL_UPDATE_BATCH_INIT(c, size, 1);

  kernelUpdateWFunctor<T, 0, uint32_t, UpdateFunctorConstantStep<T>>
      <<<nblocks, nthreads, 0, c->getStream()>>>(
          dev_weights.getData(), dev_x_counts.getData(), x_size, dev_d_counts.getData(), d_size,
          dev_4params.getData(), (param_t *)nullptr, (T *)nullptr, (T *)nullptr, nK32, dw_min_std,
          dev_states.getData());

  DEBUG_KERNEL_UPDATE_CLEANUP(c);
}
#ifdef RPU_USE_DOUBLE
template int debugKernelUpdateW<double>(
    double *, unsigned int *, int, unsigned int *, int, int, double, double, double, double *);
#endif
#ifdef RPU_USE_FP16
template int debugKernelUpdateW<half_t>(
    half_t *, unsigned int *, int, unsigned int *, int, int, half_t, half_t, half_t, half_t *);
#endif
template int debugKernelUpdateW<float>(
    float *, unsigned int *, int, unsigned int *, int, int, float, float, float, float *);

template <typename T>
int debugKernelUpdateWBatch(
    T *weights,
    uint32_t *x_counts,
    int x_size,
    uint32_t *d_counts,
    int d_size,
    int nK32,
    int m_batch,
    bool trans,
    T dw_min,
    T dw_min_std,
    T bound,
    int kernel_type,
    T *timings) {
  auto c_container = CudaContext(-1, false);
  CudaContextPtr c = &c_container;
  int nthreads = RPU_THREADS_PER_BLOCK_UPDATE;
  // number of blocks to fill the GPU with memory usage
  int nblocks = MIN(10, (x_size * d_size + nthreads - 1) / nthreads); // maybe more blocks?

  CudaArray<T> dev_decay_scales(c, x_size * d_size);
  dev_decay_scales.setConst(1);

  CudaArray<T> dev_diff_rates(c, x_size * d_size);
  dev_diff_rates.setConst(0);

  DEBUG_KERNEL_UPDATE_BATCH_INIT(c, nthreads * nblocks, m_batch);

  switch (kernel_type) {

  case 0:

    if (trans) {
      std::cout << "kernelUpdateWBatchSum<uint32_t,true,true>" << std::endl;
      kernelUpdateWBatchSum<T, 0, uint32_t, true, true><<<nblocks, nthreads, 0, c->getStream()>>>(
          dev_weights.getData(), dev_x_counts.getData(), x_size, dev_d_counts.getData(), d_size,
          dev_4params.getData(), nK32, m_batch, dw_min_std, dev_states.getData());
    } else {
      std::cout << "kernelUpdateWBatchSum<uint32_t,false,false>" << std::endl;
      kernelUpdateWBatchSum<T, 0, uint32_t, false, false><<<nblocks, nthreads, 0, c->getStream()>>>(
          dev_weights.getData(), dev_x_counts.getData(), x_size, dev_d_counts.getData(), d_size,
          dev_4params.getData(), nK32, m_batch, dw_min_std, dev_states.getData());
    }
    break;

  case 1:

    if (trans) {
      std::cout << "kernelUpdateWBatchSumBoundCheck<uint32_t,true,true>" << std::endl;
      kernelUpdateWBatchSumBoundCheck<T, 0, uint32_t, true, true>
          <<<nblocks, nthreads, 0, c->getStream()>>>(
              dev_weights.getData(), dev_x_counts.getData(), x_size, dev_d_counts.getData(), d_size,
              dev_4params.getData(), nK32, m_batch, dw_min_std, dev_states.getData());
    } else {
      std::cout << "kernelUpdateWBatchSumBoundCheck<uint32_t,false,false>" << std::endl;
      kernelUpdateWBatchSumBoundCheck<T, 0, uint32_t, false, false>
          <<<nblocks, nthreads, 0, c->getStream()>>>(
              dev_weights.getData(), dev_x_counts.getData(), x_size, dev_d_counts.getData(), d_size,
              dev_4params.getData(), nK32, m_batch, dw_min_std, dev_states.getData());
    }
    break;

  case 2:

    if (trans) {
      std::cout << "kernelUpdateWBatchFunctor<uint32_t,true,true,UpdateFunctorConstantStep>"
                << std::endl;
      kernelUpdateWBatchFunctor<T, 0, uint32_t, true, true, UpdateFunctorConstantStep<T>>
          <<<nblocks, nthreads, 0, c->getStream()>>>(
              dev_weights.getData(), dev_x_counts.getData(), x_size, dev_d_counts.getData(), d_size,
              dev_4params.getData(), (param_t *)nullptr, (T *)nullptr, (T *)nullptr, nK32, m_batch,
              dw_min_std, dev_states.getData());
    } else {
      std::cout << "kernelUpdateWBatchFunctor<uint32_t,false,false,UpdateFunctorConstantStep>"
                << std::endl;
      kernelUpdateWBatchFunctor<T, 0, uint32_t, false, false, UpdateFunctorConstantStep<T>>
          <<<nblocks, nthreads, 0, c->getStream()>>>(
              dev_weights.getData(), dev_x_counts.getData(), x_size, dev_d_counts.getData(), d_size,
              dev_4params.getData(), (param_t *)nullptr, (T *)nullptr, (T *)nullptr, nK32, m_batch,
              dw_min_std, dev_states.getData());
    }
    break;

  default:
    RPU_FATAL("kernel type known");
  }

  DEBUG_KERNEL_UPDATE_CLEANUP(c);
}

#ifdef RPU_USE_DOUBLE
template int debugKernelUpdateWBatch<double>(
    double *,
    unsigned int *,
    int,
    unsigned int *,
    int,
    int,
    int,
    bool,
    double,
    double,
    double,
    int,
    double *);
#endif
#ifdef RPU_USE_FP16
template int debugKernelUpdateWBatch<half_t>(
    half_t *,
    unsigned int *,
    int,
    unsigned int *,
    int,
    int,
    int,
    bool,
    half_t,
    half_t,
    half_t,
    int,
    half_t *);
#endif
template int debugKernelUpdateWBatch<float>(
    float *,
    unsigned int *,
    int,
    unsigned int *,
    int,
    int,
    int,
    bool,
    float,
    float,
    float,
    int,
    float *);

template <typename T>
int debugKernelUpdateWBatchShared(
    T *weights,
    uint32_t *x_counts,
    int x_size,
    uint32_t *d_counts,
    int d_size,
    int K,
    int m_batch,
    bool trans,
    T dw_min,
    T dw_min_std,
    T bound,
    int kernel_type,
    T *timings) {
  auto c_container = CudaContext(-1, false);
  CudaContextPtr c = &c_container;

  int Kplus1 = K + 1;
  int nK32 = (Kplus1 + 31) / 32;
  if ((kernel_type == 1) && nK32 > 1) {
    std::cout << "Batch order 64 only implemented for nK32==1 .\n";
    return 1;
  }
  int nthreads = 512;
  int blocks_per_sm = 2;
  // many registers so try 2 blocks per sm if half memory
  int nblocks =
      MIN((blocks_per_sm * c->getSMCount()), ((x_size * d_size + nthreads - 1) / nthreads));
  int item_size;
  if (kernel_type == 1) {
    item_size = sizeof(uint64_t);
  } else {
    item_size = sizeof(uint32_t);
  }

  int shared_mem = (nthreads / 32 + 32) * nK32 * item_size;
  int batch_load_stride_max =
      (c->getSharedMemPerBlock() / blocks_per_sm + shared_mem - 1) / shared_mem;
  int batch_load_stride = MIN(batch_load_stride_max, m_batch);
  shared_mem *= batch_load_stride;

  CudaArray<T> dev_decay_scales(c, x_size * d_size);
  dev_decay_scales.setConst(1);

  CudaArray<T> dev_diff_rates(c, x_size * d_size);
  dev_diff_rates.setConst(0);

  std::cout << "Shared mem : " << shared_mem << std::endl;
  std::cout << "Batch stride : " << batch_load_stride << std::endl;

  UpdateManagementHelper<T> umh(c, x_size, d_size);
  CudaArray<uint64_t> dev_x_counts_bo64(c, x_size * m_batch);
  CudaArray<uint64_t> dev_d_counts_bo64(c, d_size * m_batch);

  if (kernel_type == 1) { // do once to init the buffers
    if (nK32 > 1) {
      RPU_FATAL("K setting not supported");
    }
    CudaArray<uint32_t> dev_x_counts(c, x_size * nK32 * m_batch, x_counts);
    CudaArray<uint32_t> dev_d_counts(c, d_size * nK32 * m_batch, d_counts);
    CUDA_TIMING_INIT;
    CUDA_TIMING_START(c);
    umh.translateTransToBatchOrder64(
        dev_x_counts_bo64.getData(), dev_d_counts_bo64.getData(), dev_x_counts.getData(),
        dev_d_counts.getData(), m_batch, K, false);
    CUDA_TIMING_STOP(c, "translate batch without buffer");
    CUDA_TIMING_DESTROY;
  }

  DEBUG_KERNEL_UPDATE_BATCH_INIT(c, nthreads * nblocks, m_batch);

  switch (kernel_type) {
  case 0: {
    if (trans) {
      std::cout << "kernelUpdateWBatchSharedFunctor<uint32_t,true,true,UpdateFunctorConstantStep>"
                << std::endl;
      kernelUpdateWBatchSharedFunctor<T, 0, uint32_t, true, true, UpdateFunctorConstantStep<T>>
          <<<nblocks, nthreads, shared_mem, c->getStream()>>>(
              dev_weights.getData(), dev_x_counts.getData(), x_size, dev_d_counts.getData(), d_size,
              dev_4params.getData(), nullptr, nullptr, nullptr, nK32, m_batch, batch_load_stride,
              dw_min_std, dev_states.getData());

    } else {
      std::cout << "kernelUpdateWBatchSharedFunctor<uint32_t,false,false,UpdateFunctorConstantStep>"
                << std::endl;
      kernelUpdateWBatchSharedFunctor<T, 0, uint32_t, false, false, UpdateFunctorConstantStep<T>>
          <<<nblocks, nthreads, shared_mem, c->getStream()>>>(
              dev_weights.getData(), dev_x_counts.getData(), x_size, dev_d_counts.getData(), d_size,
              dev_4params.getData(), nullptr, nullptr, nullptr, nK32, m_batch, batch_load_stride,
              dw_min_std, dev_states.getData());
    }
    break;
  }
  case 1: {
    if (trans) {
      std::cout << "kernelUpdateWBatchSharedFunctor<uint64_t,true,true,UpdateFunctorConstantStep>"
                << std::endl;
      umh.translateTransToBatchOrder64(
          dev_x_counts_bo64.getData(), dev_d_counts_bo64.getData(), dev_x_counts.getData(),
          dev_d_counts.getData(), m_batch, K, false);

      kernelUpdateWBatchSharedFunctor<T, 0, uint64_t, true, true, UpdateFunctorConstantStep<T>>
          <<<nblocks, nthreads, shared_mem, c->getStream()>>>(
              dev_weights.getData(), dev_x_counts_bo64.getData(), x_size,
              dev_d_counts_bo64.getData(), d_size, dev_4params.getData(), nullptr, nullptr, nullptr,
              nK32, umh.getBo64Batch(m_batch, K), batch_load_stride, dw_min_std,
              dev_states.getData(), umh.getKnData(false, m_batch));
    } else
      std::cout << "Batch order 64 only implemented for trans yet.\n";

    break;
  }
  case 2: {
    if (trans) {
      std::cout << "kernelUpdateWBatchSharedSum<uint32_t,true,true>" << std::endl;
      kernelUpdateWBatchSharedSum<T, 0, uint32_t, true, true>
          <<<nblocks, nthreads, shared_mem, c->getStream()>>>(
              dev_weights.getData(), dev_x_counts.getData(), x_size, dev_d_counts.getData(), d_size,
              dev_4params.getData(), nK32, m_batch, batch_load_stride, dw_min_std,
              dev_states.getData());
    } else {
      std::cout << "kernelUpdateWBatchSharedSum<uint32_t,false, false>" << std::endl;
      kernelUpdateWBatchSharedSum<T, 0, uint32_t, false, false>
          <<<nblocks, nthreads, shared_mem, c->getStream()>>>(
              dev_weights.getData(), dev_x_counts.getData(), x_size, dev_d_counts.getData(), d_size,
              dev_4params.getData(), nK32, m_batch, batch_load_stride, dw_min_std,
              dev_states.getData());
    }
    break;
  }
  case 3: {
    if (trans) {
      std::cout << "kernelUpdateWBatchSharedSumBoundCheck<uint32_t,true,true>" << std::endl;
      kernelUpdateWBatchSharedSumBoundCheck<T, 0, uint32_t, true, true>
          <<<nblocks, nthreads, shared_mem, c->getStream()>>>(
              dev_weights.getData(), dev_x_counts.getData(), x_size, dev_d_counts.getData(), d_size,
              dev_4params.getData(), nK32, m_batch, batch_load_stride, dw_min_std,
              dev_states.getData());
    } else {
      std::cout << "kernelUpdateWBatchSharedSumBoundCheck<uint32_t,false,false>" << std::endl;
      kernelUpdateWBatchSharedSumBoundCheck<T, 0, uint32_t, false, false>
          <<<nblocks, nthreads, shared_mem, c->getStream()>>>(
              dev_weights.getData(), dev_x_counts.getData(), x_size, dev_d_counts.getData(), d_size,
              dev_4params.getData(), nK32, m_batch, batch_load_stride, dw_min_std,
              dev_states.getData());
    }
    break;
  }
  }

  DEBUG_KERNEL_UPDATE_CLEANUP(c);
}

#ifdef RPU_USE_DOUBLE
template int debugKernelUpdateWBatchShared<double>(
    double *,
    unsigned int *,
    int,
    unsigned int *,
    int,
    int,
    int,
    bool,
    double,
    double,
    double,
    int,
    double *);
#endif
#ifdef RPU_USE_FP16
template int debugKernelUpdateWBatchShared<half_t>(
    half_t *,
    unsigned int *,
    int,
    unsigned int *,
    int,
    int,
    int,
    bool,
    half_t,
    half_t,
    half_t,
    int,
    half_t *);
#endif
template int debugKernelUpdateWBatchShared<float>(
    float *,
    unsigned int *,
    int,
    unsigned int *,
    int,
    int,
    int,
    bool,
    float,
    float,
    float,
    int,
    float *);

} // namespace test_helper
} // namespace RPU

#undef DEBUG_KERNEL_UPDATE_BATCH_INIT
#undef DEBUG_KERNEL_UPDATE_CLEANUP
