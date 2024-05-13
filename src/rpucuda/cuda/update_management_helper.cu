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

#include "bit_line_maker.h"
#include "update_management_helper.h"

#include <cmath>
#include <iostream>
#include <memory>

#include "cuda_fp16_util.h"
#include "cuda_math_util.h"
#include "cuda_util.h"
#include "io_iterator.h"
#include "rpu_cub.h"

namespace RPU {

// /*********************************************************************************/
/*--------- K<32 special path for large batch sizes.  -------------------------------*/
template <bool ublm>
__device__ __forceinline__ kagg_t getKn(const int m_batch, const int BL, const kagg_t *nK);

template <bool ublm>
__device__ __forceinline__ kagg_t getBlockAggregate(
    const int m_batch,
    const int BL,
    int tid_stride,
    int thread_block_size,
    kagg_t *Kc_block_aggregate);
template <bool ublm>
__device__ __forceinline__ int getK(int batch_idx, const int BL, int *K_values);

template <bool ublm>
__device__ __forceinline__ kagg_t
getCurrentKc(int batch_idx, const int BL, kagg_t *Kc_block, kagg_t Kn_aggregate);

template <>
__device__ __forceinline__ kagg_t getKn<false>(const int m_batch, const int BL, const kagg_t *Kn) {

  return m_batch * BL;
}

template <>
__device__ __forceinline__ kagg_t getKn<true>(const int m_batch, const int BL, const kagg_t *Kn) {
  return *Kn;
}

template <>
__device__ __forceinline__ kagg_t getBlockAggregate<false>(
    const int m_batch,
    const int BL,
    int tid_stride,
    int thread_block_size,
    kagg_t *Kc_block_aggregate) {
  if ((thread_block_size + tid_stride) >= m_batch)
    return (m_batch % thread_block_size) * BL;
  else
    return thread_block_size * BL;
}

template <>
__device__ __forceinline__ kagg_t getBlockAggregate<true>(
    const int m_batch,
    const int BL,
    int tid_stride,
    int thread_block_size,
    kagg_t *Kc_block_aggregate) {

  int bid = tid_stride / thread_block_size;
  return Kc_block_aggregate[bid];
}

template <> __device__ __forceinline__ int getK<false>(int batch_idx, const int BL, int *K_values) {
  return BL;
}

template <> __device__ __forceinline__ int getK<true>(int batch_idx, const int BL, int *K_values) {
  return K_values[batch_idx];
}

template <>
__device__ __forceinline__ kagg_t
getCurrentKc<false>(int batch_idx, const int BL, kagg_t *Kc_block, kagg_t Kn_aggregate) {
  return batch_idx * BL - Kn_aggregate;
}

template <>
__device__ __forceinline__ kagg_t
getCurrentKc<true>(int batch_idx, const int BL, kagg_t *Kc_block, kagg_t Kn_aggregate) {
  return Kc_block[batch_idx];
}

template <bool update_bl_management = false, int thread_block_size = 512>
__global__ void kernelTranslateTransFormatToBatchOrder64Format(
    const uint32_t *x_counts,
    uint64_t *x_counts_BO64_format,
    int x_size_in,
    const uint32_t *d_counts,
    uint64_t *d_counts_BO64_format,
    int d_size_in,
    const int m_batch_in,
    const int BL_in,
    kagg_t *Kn_in = nullptr,
    int *K_values_in = nullptr,
    kagg_t *Kc_block_in = nullptr,
    kagg_t *Kc_block_aggregate_in = nullptr) {

  // -- each block takes one x/d value.
  // -- expects OUTTRANS format !!

  __shared__ uint32_t c_shared[thread_block_size];
  __shared__ uint32_t neg_shared[thread_block_size];

  const int m_batch = m_batch_in;
  const int BL = BL_in;
  kagg_t Kn = getKn<update_bl_management>(m_batch, BL, Kn_in);

  const int x_size = x_size_in;
  const int d_size = d_size_in;
  const int add_size = x_size + d_size;

  int nB = ((Kn + 31) >> 5); // compressed K on batch

  // loop xd indeces
  for (int bid_stride = 0; bid_stride < add_size; bid_stride += gridDim.x) {

    int bid = blockIdx.x + bid_stride;

    // select x or d
    const uint32_t *counts;
    uint64_t *out_counts;
    int xd_index;
    if (bid < x_size) {
      counts = x_counts;
      out_counts = x_counts_BO64_format;
      xd_index = bid;
    } else if (bid < add_size) {
      counts = d_counts;
      out_counts = d_counts_BO64_format;
      xd_index = bid - x_size;
    } else {
      return;
    }

    const int start_idx = xd_index * m_batch; // expects trans order !!
    const int out_start_idx = xd_index * nB;  // reduced batch size

    int total_nB = 0;
    uint32_t last_neg = 0;
    uint32_t last_c = 0;
    int current_nB = 0;
    kagg_t Kc_aggregate = 0;
    int K_left_over = 0;
    // loop over batch
    for (int tid_stride = 0; tid_stride < m_batch; tid_stride += blockDim.x) {

      if (threadIdx.x > 0) {
        c_shared[threadIdx.x] = 0;
        neg_shared[threadIdx.x] = 0;
      }

      if (threadIdx.x == current_nB) { // to avoid a sync, see below
        c_shared[0] = last_c;
        neg_shared[0] = last_neg;
      }

      const int batch_idx = threadIdx.x + tid_stride;

      kagg_t Kc_block_aggregate = getBlockAggregate<update_bl_management>(
          m_batch, BL, tid_stride, thread_block_size, Kc_block_aggregate_in);

      kagg_t current_Kc = Kc_block_aggregate;
      int K = 0;
      if (batch_idx < m_batch) {
        K = getK<update_bl_management>(batch_idx, BL, K_values_in);
        current_Kc = getCurrentKc<update_bl_management>(batch_idx, BL, Kc_block_in, Kc_aggregate);
      }

      Kc_block_aggregate += K_left_over;
      current_Kc += K_left_over;

      __syncthreads(); // need to sync for shared

      if (batch_idx < m_batch) {

        uint32_t c = counts[start_idx + batch_idx];
        uint32_t negative = 0;
        if ((c & ((uint32_t)1)) > 0) {
          negative = 0xffffffff >> (32 - K);
        }
        c >>= 1; // get rid of negative bit

        // set bit in shared
        int i_word_start = current_Kc >> 5;
        int i_word_end = (current_Kc + K) >> 5;
        int i_bit_start = current_Kc & 0x1f;

        atomicOr(&c_shared[i_word_start], c << i_bit_start);
        atomicOr(&neg_shared[i_word_start], negative << i_bit_start);

        if (i_word_start != i_word_end) { // most 31 bits per batch, so only 1 overlap possible
          atomicOr(
              &c_shared[i_word_end],
              c >> (32 - i_bit_start)); // (32 - i_bit_start) first bits were already set above
          atomicOr(&neg_shared[i_word_end], negative >> (32 - i_bit_start));
        }
      }

      __syncthreads();

      Kc_aggregate += Kc_block_aggregate;
      kagg_t current_nB =
          Kc_block_aggregate >> 5; // there might be some left over bits. put into next round
      bool last_loop = tid_stride + blockDim.x >= m_batch;

      K_left_over = Kc_aggregate & 0x1f;
      bool left_overs = K_left_over > 0;

      if ((threadIdx.x < current_nB) || ((threadIdx.x == current_nB) && last_loop && left_overs)) {
        uint64_t c64 =
            (((uint64_t)neg_shared[threadIdx.x]) << 32) | ((uint64_t)c_shared[threadIdx.x]);
        out_counts[out_start_idx + total_nB + threadIdx.x] = c64;
      } else if ((threadIdx.x == current_nB) && left_overs) { // save left overs
        last_neg = neg_shared[current_nB];
        last_c = c_shared[current_nB];
      }
      total_nB += current_nB;
    }
  }
}

namespace test_helper {
template <typename T, bool ublm>
int debugKernelTranslateTransFormatToBatchOrder64Format(
    T *indata, int size, int m_batch, T scaleprob, int K) {

  // counts should be: size*nk32 allocated !
  if (K > 31)
    return 1;

  DebugPulsedUpdateMetaParameter<T> up;
  up.res = 0.01;
  up.sto_round = false;
  up.update_bl_management = ublm;
  up.update_management = ublm;
  up.scaleprob = scaleprob;
  up.desired_BL = K;

  // std::cout << "m_batch: " << m_batch << " size: " << size << std::endl;
  const int nthreads = RPU_THREADS_PER_BLOCK_UPDATE;

  CUDA_TIMING_INIT;
  auto c_container = CudaContext(-1, false);
  CudaContextPtr c = &c_container;

  T *tmp = new T[size * m_batch];
  for (int i = 0; i < m_batch; i++) {
    for (int j = 0; j < size; j++) {
      tmp[i * size + j] = indata[j];
    }
  }
  CudaArray<T> dev_indata(c, size * m_batch, tmp);
  c->synchronize();
  delete[] tmp;

  T dwmin = 0.001;
  T lr = 0.01;
  BitLineMaker<T> blm(c, size, size);
  blm.makeCounts(
      dev_indata.getData(), dev_indata.getData(), up, dwmin, lr, m_batch, false, false, true, 2,
      false); // compute B64 to init buffer for below

  UpdateManagementHelper<T> *umh = blm.getUmh();
  c->synchronize();

  int nBmax = m_batch; // at most m_batch, likely smaller
  CudaArray<uint64_t> dev_counts_out(c, size * nBmax);
  CudaArray<uint64_t> dev_counts_out2(c, size * nBmax);

  int nblocks = size + size;
  // std::cout << "nblocks, nthreads: " << nblocks << ", " << nthreads << std::endl;

  CUDA_TIMING_START(c);

  kagg_t *nK = nullptr;

  int *K_values = nullptr;
  kagg_t *Kc_block = nullptr;
  kagg_t *Kc_block_aggregate = nullptr;

  if (ublm) {
    // redo computation for timing
    umh->computeKc(m_batch); // Will also compute Kn...
    nK = umh->getKnData(true, m_batch);
    K_values = umh->getKValueData();
    umh->computeKcBlock(m_batch);
    Kc_block = umh->getKcBlockData();
    Kc_block_aggregate = umh->getKcBlockAggregateData();
  }
  CUDA_TIMING_STOP(c, "get Kn/Kcblock ");
  CUDA_TIMING_START(c);
  kernelTranslateTransFormatToBatchOrder64Format<ublm, nthreads>
      <<<nblocks, nthreads, 0, c->getStream()>>>(
          blm.getXCountsData(), dev_counts_out.getData(), size, blm.getDCountsData(),
          dev_counts_out2.getData(), size, m_batch, K, nK, K_values, Kc_block, Kc_block_aggregate);

  CUDA_TIMING_STOP(c, "Counts translated");

  kagg_t Kn = umh->getKnValue(ublm, m_batch, K);

  kagg_t nB = (Kn + 31) / 32;

  // check translated:
  int *Kvalues = new int[m_batch];
  if (ublm)
    umh->getKValues().copyTo(Kvalues);

  uint32_t *orig_counts = new uint32_t[m_batch * size];
  uint64_t *counts_out = new uint64_t[m_batch * size];
  uint64_t *counts_out_ref = new uint64_t[m_batch * size];
  dev_counts_out2.copyTo(counts_out);
  blm.copyDCountsToHost(orig_counts);

  for (int j = 0; j < m_batch * size; j++) {
    counts_out_ref[j] = 0;
  }

  c->synchronize();

  int return_int = 0;

  kagg_t Kc = 0;
  for (int i_batch = 0; i_batch < m_batch; i_batch++) {
    if (ublm)
      Kc += Kvalues[i_batch];
    else
      Kc += K;
  }
  int nBref = (Kc + 31) >> 5;
  uint32_t one = 1;
  // translate reference
  for (int idx = 0; idx < size; idx++) {
    Kc = 0;
    for (int i_batch = 0; i_batch < m_batch; i_batch++) {

      uint32_t c = orig_counts[i_batch + m_batch * idx];
      uint32_t neg = c & one;
      c >>= 1; // get rid of sign bit;
      int k = K;
      if (ublm)
        k = Kvalues[i_batch];

      for (int i = 0; i < k; i++) { // k is smaller than 32 because nK32==1
        kagg_t current_cK = Kc + i;
        kagg_t iB = (current_cK) >> 5;
        int ibit = (current_cK)&0x1f;
        if ((c & (one << i)) > 0) {
          counts_out_ref[iB + idx * nBref] |= ((uint64_t)1) << ibit;
        }
        if (neg > 0) {
          counts_out_ref[iB + idx * nBref] |= ((uint64_t)1) << (ibit + 32);
        }
      }

      Kc += k;
    }
  }
  // std::cout << "nB should be " << nBref << " and is " << nB << ".\n";

  if (nB != nBref) {

    return_int = 1;
  }
  for (int j = 0; j < nBref * size; j++) {

    if (counts_out_ref[j] != counts_out[j]) {
      std::cerr << j << ":" << counts_out[j] << " should be " << counts_out_ref[j] << std::endl;
      return_int = 1;
    }

    if ((j > 100) && return_int)
      break;
  }

  delete[] counts_out;
  delete[] orig_counts;
  delete[] counts_out_ref;
  delete[] Kvalues;

  CUDA_TIMING_DESTROY;

  return return_int;
}
template int
debugKernelTranslateTransFormatToBatchOrder64Format<float, true>(float *, int, int, float, int);
template int
debugKernelTranslateTransFormatToBatchOrder64Format<float, false>(float *, int, int, float, int);
#ifdef RPU_USE_DOUBLE
template int
debugKernelTranslateTransFormatToBatchOrder64Format<double, true>(double *, int, int, double, int);
template int
debugKernelTranslateTransFormatToBatchOrder64Format<double, false>(double *, int, int, double, int);
#endif
#ifdef RPU_USE_FP16
template int
debugKernelTranslateTransFormatToBatchOrder64Format<half_t, true>(half_t *, int, int, half_t, int);
template int
debugKernelTranslateTransFormatToBatchOrder64Format<half_t, false>(half_t *, int, int, half_t, int);
#endif
} // namespace test_helper

template <typename T>
__global__ void kernelUMGetScaleAndKValues(
    T *scale_values,
    int *K_values,
    T *x_amax_values,
    T *d_amax_values,
    const int m_batch,
    const bool ublm_in,
    const T weight_granularity_in,
    const T lr_in,
    const int Kmax_in,
    const T um_reg_scale,
    const T um_grad_scale) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  bool ublm = ublm_in;
  T weight_granularity = weight_granularity_in;
  T lr = fabs(lr_in);
  int Kmax = Kmax_in;

  if (tid < m_batch) {
    T x_val = x_amax_values[tid];
    T d_val = d_amax_values[tid] * um_grad_scale;
    T k_val = lr * x_val * d_val / weight_granularity;

    if (k_val > (T)0.0) {
      if (k_val > (T)Kmax) {
        d_val *= (T)Kmax / k_val;
      }
      scale_values[tid] = sqrt(x_val / d_val);

      if (ublm) {
        int K = ceil(k_val);
        K_values[tid] = (K <= Kmax) ? K : Kmax;
      }
    } else {
      // dummy: lr, x, or d is all zero
      scale_values[tid] = 1;
      if (ublm) {
        K_values[tid] = 1;
      }
    }
    // note:  K values are not set in case of ~ublm
  }
}

template <int thread_block_size>
__global__ void kernelGetKBlockAggregate(
    int *K_values, int m_batch_in, kagg_t *Kc_block, kagg_t *Kc_block_aggregate) {

  const int m_batch = m_batch_in;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__
      typename RPU_CUB_NS_QUALIFIER BlockScan<kagg_t, thread_block_size>::TempStorage temp_storage;

  int K = 0;
  if (tid < m_batch) {
    K = K_values[tid];
  }
  kagg_t Kc = 0;
  kagg_t block_aggregate = 0;
  RPU_CUB_NS_QUALIFIER BlockScan<kagg_t, thread_block_size>(temp_storage)
      .ExclusiveSum(K, Kc, block_aggregate);

  if (tid < m_batch) {
    Kc_block[tid] = Kc;
  }

  if (threadIdx.x == 0) {
    Kc_block_aggregate[blockIdx.x] = block_aggregate;
  }
}

/*********************************************************************************************************************/
/* UPDATEMANAGERHELPER */
/*********************************************************************************************************************/
#define RPU_UMH_B64_NTHREADS 512

template <typename T>
UpdateManagementHelper<T>::UpdateManagementHelper(CudaContextPtr c, int x_size, int d_size)
    : context_{c}, x_size_{x_size}, d_size_{d_size}, buffer_m_batch_{0} {
  nthreads_ = RPU_THREADS_PER_BLOCK_UPDATE;
  x_maximizer_ = RPU::make_unique<Maximizer<T>>(c, x_size_, true);
  d_maximizer_ = RPU::make_unique<Maximizer<T>>(c, d_size_, true);
  dev_sumabsmax_value_ = RPU::make_unique<CudaArray<T>>(context_, 2);
}

template <typename T> void UpdateManagementHelper<T>::initializeBuffers(int m_batch) {

  buffer_m_batch_ = m_batch;
  dev_K_values_ = RPU::make_unique<CudaArray<int>>(context_, m_batch);
  dev_Kc_values_ = RPU::make_unique<CudaArray<kagg_t>>(context_, m_batch + 1);
  dev_Kc_values_->setConst(0);
  dev_scale_values_ = RPU::make_unique<CudaArray<T>>(context_, m_batch);

  // for translate
  const int nthreads = RPU_UMH_B64_NTHREADS;
  int nblocks = context_->getNBlocks(m_batch, nthreads);

  dev_Kc_block_ = RPU::make_unique<CudaArray<kagg_t>>(context_, m_batch);
  dev_Kc_block_aggregate_ = RPU::make_unique<CudaArray<kagg_t>>(context_, nblocks);

  // Determine temporary device storage requirements
  void *temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  auto s = context_->getStream();

  CUDA_CALL(RPU_CUB_NS_QUALIFIER DeviceScan::InclusiveSum(
      temp_storage, temp_storage_bytes, dev_K_values_->getData(), dev_Kc_values_->getData() + 1,
      m_batch, s));
  context_->synchronize();
  dev_Kc_temp_storage_ = RPU::make_unique<CudaArray<char>>(context_, (int)temp_storage_bytes);

  // average max sum
  CUDA_CALL(RPU_CUB_NS_QUALIFIER DeviceReduce::Sum(
      nullptr, temp_storage_bytes, x_maximizer_->getMaxValues(), dev_sumabsmax_value_->getData(),
      m_batch, s));
  dev_sumabsmax_temp_storage_ = RPU::make_unique<CudaArray<char>>(context_, temp_storage_bytes);
  context_->synchronize();
}

template <typename T> void UpdateManagementHelper<T>::computeKcBlock(int m_batch) {

  // CAUTION: needs K_values to be already computed !!
  const int nthreads = RPU_UMH_B64_NTHREADS;
  int nblocks = context_->getNBlocks(m_batch, nthreads);

  kernelGetKBlockAggregate<nthreads><<<nblocks, nthreads, 0, context_->getStream()>>>(
      dev_K_values_->getData(), m_batch, dev_Kc_block_->getData(),
      dev_Kc_block_aggregate_->getData());
}

template <typename T> void UpdateManagementHelper<T>::computeKc(int m_batch) {

  // CAUTION: needs K_values to be already computed !!
  size_t temp_storage_bytes = dev_Kc_temp_storage_->getSize();
  CUDA_CALL(RPU_CUB_NS_QUALIFIER DeviceScan::InclusiveSum(
      (void *)dev_Kc_temp_storage_->getData(), temp_storage_bytes, dev_K_values_->getData(),
      dev_Kc_values_->getData() + 1, m_batch, context_->getStream()));
}

template <typename T>
kagg_t UpdateManagementHelper<T>::getKnValue(bool ublm, int m_batch, int K) const {
  if (!ublm) {
    return m_batch * K;
  }
  kagg_t Kn = 0;
  CudaArray<kagg_t> tmp(context_, 1);
  kagg_t *kndata = getKnData(ublm, m_batch);
  if (kndata != nullptr) {
    tmp.assignFromDevice(kndata);
    tmp.copyTo(&Kn);
  }
  return Kn;
}

template <typename T>
void UpdateManagementHelper<T>::getAverageAbsMax(T &m_x, T &m_d, int m_batch) const {
  // CAUTION needs computeKandScaleValues to be called !

  if (m_batch == 1) {
    x_maximizer_->copyMaxValuesToHost(&m_x);
    d_maximizer_->copyMaxValuesToHost(&m_d);
    return;
  }

  // first compute the average of the max over batch
  size_t ssz = dev_sumabsmax_temp_storage_->getSize();
  CUDA_CALL(RPU_CUB_NS_QUALIFIER DeviceReduce::Sum(
      (void *)dev_sumabsmax_temp_storage_->getData(), ssz, x_maximizer_->getMaxValues(),
      dev_sumabsmax_value_->getData(), m_batch, context_->getStream()));
  CUDA_CALL(RPU_CUB_NS_QUALIFIER DeviceReduce::Sum(
      (void *)dev_sumabsmax_temp_storage_->getData(), ssz, d_maximizer_->getMaxValues(),
      dev_sumabsmax_value_->getData() + 1, m_batch, context_->getStream()));
  T result[2];
  dev_sumabsmax_value_->copyTo(result);
  m_x = result[0] / (T)m_batch;
  m_d = result[1] / (T)m_batch;
}

template <typename T>
void UpdateManagementHelper<T>::getAverageLogAbsMax(T &m_x, T &m_d, int m_batch) const {
  // CAUTION needs computeKandScaleValues to be called !

  if (m_batch == 1) {
    x_maximizer_->copyMaxValuesToHost(&m_x);
    d_maximizer_->copyMaxValuesToHost(&m_d);
    return;
  }

  // first compute the average of the max over batch
  size_t ssz = dev_sumabsmax_temp_storage_->getSize();
  LogInputIterator<T> x_input_iter(x_maximizer_->getMaxValues());
  LogInputIterator<T> d_input_iter(d_maximizer_->getMaxValues());

  CUDA_CALL(RPU_CUB_NS_QUALIFIER DeviceReduce::Sum(
      (void *)dev_sumabsmax_temp_storage_->getData(), ssz, x_input_iter,
      dev_sumabsmax_value_->getData(), m_batch, context_->getStream()));
  CUDA_CALL(RPU_CUB_NS_QUALIFIER DeviceReduce::Sum(
      (void *)dev_sumabsmax_temp_storage_->getData(), ssz, d_input_iter,
      dev_sumabsmax_value_->getData() + 1, m_batch, context_->getStream()));
  T result[2];
  dev_sumabsmax_value_->copyTo(result);
  m_x = expf(result[0] / (T)m_batch);
  m_d = expf(result[1] / (T)m_batch);
}

template <typename T> void UpdateManagementHelper<T>::getAbsMax(T &m_x, T &m_d, int m_batch) const {
  // CAUTION needs computeKandScaleValues to be called !

  if (m_batch == 1) {
    x_maximizer_->copyMaxValuesToHost(&m_x);
    d_maximizer_->copyMaxValuesToHost(&m_d);
    return;
  }

  // first compute the average of the max over batch
  size_t ssz = dev_sumabsmax_temp_storage_->getSize();
  CUDA_CALL(RPU_CUB_NS_QUALIFIER DeviceReduce::Max(
      (void *)dev_sumabsmax_temp_storage_->getData(), ssz, x_maximizer_->getMaxValues(),
      dev_sumabsmax_value_->getData(), m_batch, context_->getStream()));
  CUDA_CALL(RPU_CUB_NS_QUALIFIER DeviceReduce::Max(
      (void *)dev_sumabsmax_temp_storage_->getData(), ssz, d_maximizer_->getMaxValues(),
      dev_sumabsmax_value_->getData() + 1, m_batch, context_->getStream()));
  T result[2];
  dev_sumabsmax_value_->copyTo(&result[0]);
  m_x = result[0];
  m_d = result[1];
}

template <typename T>
void UpdateManagementHelper<T>::translateTransToBatchOrder64(
    uint64_t *x_counts_bo64,
    uint64_t *d_counts_bo64,
    const uint32_t *x_counts,
    const uint32_t *d_counts,
    const int m_batch,
    const int BL,
    const bool update_bl_management) {
  // needs K values to be precomputed for ublm !!

  if (BL > 31) {
    RPU_FATAL("ERROR: BO64 format only supported for BL<32");
  }
  if (buffer_m_batch_ < m_batch) {
    this->initializeBuffers(m_batch);
  }

  const int nthreads = RPU_UMH_B64_NTHREADS; // how many ? test...
  int nblocks = d_size_ + x_size_;

  if (update_bl_management) {
    this->computeKcBlock(m_batch);
    this->computeKc(m_batch);

    kernelTranslateTransFormatToBatchOrder64Format<true, nthreads>
        <<<nblocks, nthreads, 0, context_->getStream()>>>(
            x_counts, x_counts_bo64, x_size_, d_counts, d_counts_bo64, d_size_, m_batch, BL,
            this->getKnData(true, m_batch), this->getKValueData(), this->getKcBlockData(),
            this->getKcBlockAggregateData());

    // context_->synchronize();
  } else {
    // no update bl management
    kernelTranslateTransFormatToBatchOrder64Format<false, nthreads>
        <<<nblocks, nthreads, 0, context_->getStream()>>>(
            x_counts, x_counts_bo64, x_size_, d_counts, d_counts_bo64, d_size_, m_batch, BL);
  }
}

template <typename T>
template <typename XInputIteratorT, typename DInputIteratorT>
void UpdateManagementHelper<T>::computeKandScaleValues(
    XInputIteratorT x_in,
    DInputIteratorT d_in,
    const T weight_granularity,
    const T lr,
    const bool update_management,
    const bool update_bl_management,
    const int m_batch,
    const bool x_trans,
    const bool d_trans,
    const int Kmax,
    const T um_reg_scale,
    const T um_grad_scale) {

  if ((!update_management) && (!update_bl_management)) {
    return;
  } else {

    // get max values
    x_maximizer_->compute(x_in, m_batch, x_trans);
    d_maximizer_->compute(d_in, m_batch, d_trans);

    // initilize if necessary
    if (buffer_m_batch_ < m_batch) {
      this->initializeBuffers(m_batch);
    }

    // compute
    int nblocks = context_->getNBlocks(m_batch, nthreads_);
    kernelUMGetScaleAndKValues<<<nblocks, nthreads_, 0, context_->getStream()>>>(
        dev_scale_values_->getData(), dev_K_values_->getData(), x_maximizer_->getMaxValues(),
        d_maximizer_->getMaxValues(), m_batch, update_bl_management, weight_granularity, lr, Kmax,
        um_reg_scale, um_grad_scale);
  }
}

#define RPU_UMH_ITER_TEMPLATE(NUM_T, XITERT, DITERT)                                               \
  template void UpdateManagementHelper<NUM_T>::computeKandScaleValues(                             \
      XITERT, DITERT, const NUM_T, const NUM_T, const bool, const bool, const int, const bool,     \
      const bool, const int, const NUM_T, const NUM_T);

#define TRANSFLOAT(TRANS) TRANS, float

template class UpdateManagementHelper<float>;

RPU_UMH_ITER_TEMPLATE(float, const float *, const float *);
RPU_UMH_ITER_TEMPLATE(float, float *, float *);
RPU_UMH_ITER_TEMPLATE(float, IndexReaderInputIterator<float>, const float *);
RPU_UMH_ITER_TEMPLATE(float, IndexReaderTransInputIterator<float>, const float *);
RPU_UMH_ITER_TEMPLATE(
    float, IndexReaderTransInputIterator<float>, PermuterTransInputIterator<float>);
RPU_UMH_ITER_TEMPLATE(
    float, IndexReaderSliceInputIterator<TRANSFLOAT(true)>, SliceInputIterator<TRANSFLOAT(true)>);
RPU_UMH_ITER_TEMPLATE(
    float, IndexReaderSliceInputIterator<TRANSFLOAT(false)>, SliceInputIterator<TRANSFLOAT(false)>);

RPU_UMH_ITER_TEMPLATE(float, const float *, PermuterTransInputIterator<float>);
RPU_UMH_ITER_TEMPLATE(float, const float *, SliceInputIterator<TRANSFLOAT(true)>);
RPU_UMH_ITER_TEMPLATE(float, const float *, SliceInputIterator<TRANSFLOAT(false)>);
RPU_UMH_ITER_TEMPLATE(float, IndexReaderSliceInputIterator<TRANSFLOAT(true)>, const float *);
RPU_UMH_ITER_TEMPLATE(float, IndexReaderSliceInputIterator<TRANSFLOAT(false)>, const float *);
RPU_UMH_ITER_TEMPLATE(float, EyeInputIterator<float>, const float *);
RPU_UMH_ITER_TEMPLATE(float, const float *, EyeInputIterator<float>);

#undef TRANSFLOAT

#ifdef RPU_USE_DOUBLE
#define TRANSDOUBLE(TRANS) TRANS, double

template class UpdateManagementHelper<double>;

RPU_UMH_ITER_TEMPLATE(double, const double *, const double *);
RPU_UMH_ITER_TEMPLATE(double, double *, double *);
RPU_UMH_ITER_TEMPLATE(
    double, IndexReaderTransInputIterator<double>, PermuterTransInputIterator<double>);
RPU_UMH_ITER_TEMPLATE(double, IndexReaderInputIterator<double>, const double *);
RPU_UMH_ITER_TEMPLATE(double, IndexReaderTransInputIterator<double>, const double *);
RPU_UMH_ITER_TEMPLATE(
    double,
    IndexReaderSliceInputIterator<TRANSDOUBLE(true)>,
    SliceInputIterator<TRANSDOUBLE(true)>);
RPU_UMH_ITER_TEMPLATE(
    double,
    IndexReaderSliceInputIterator<TRANSDOUBLE(false)>,
    SliceInputIterator<TRANSDOUBLE(false)>);

RPU_UMH_ITER_TEMPLATE(double, const double *, PermuterTransInputIterator<double>);
RPU_UMH_ITER_TEMPLATE(double, const double *, SliceInputIterator<TRANSDOUBLE(true)>);
RPU_UMH_ITER_TEMPLATE(double, const double *, SliceInputIterator<TRANSDOUBLE(false)>);
RPU_UMH_ITER_TEMPLATE(double, IndexReaderSliceInputIterator<TRANSDOUBLE(true)>, const double *);
RPU_UMH_ITER_TEMPLATE(double, IndexReaderSliceInputIterator<TRANSDOUBLE(false)>, const double *);
RPU_UMH_ITER_TEMPLATE(double, EyeInputIterator<double>, const double *);
RPU_UMH_ITER_TEMPLATE(double, const double *, EyeInputIterator<double>);

#undef TRANSDOUBLE
#endif

#ifdef RPU_USE_FP16
#define TRANSHALF(TRANS) TRANS, half_t

template class UpdateManagementHelper<half_t>;

RPU_UMH_ITER_TEMPLATE(half_t, const half_t *, const half_t *);
RPU_UMH_ITER_TEMPLATE(half_t, half_t *, half_t *);
RPU_UMH_ITER_TEMPLATE(
    half_t, IndexReaderTransInputIterator<half_t>, PermuterTransInputIterator<half_t>);
RPU_UMH_ITER_TEMPLATE(half_t, IndexReaderInputIterator<half_t>, const half_t *);
RPU_UMH_ITER_TEMPLATE(half_t, IndexReaderTransInputIterator<half_t>, const half_t *);
RPU_UMH_ITER_TEMPLATE(
    half_t, IndexReaderSliceInputIterator<TRANSHALF(true)>, SliceInputIterator<TRANSHALF(true)>);
RPU_UMH_ITER_TEMPLATE(
    half_t, IndexReaderSliceInputIterator<TRANSHALF(false)>, SliceInputIterator<TRANSHALF(false)>);

RPU_UMH_ITER_TEMPLATE(half_t, const half_t *, PermuterTransInputIterator<half_t>);
RPU_UMH_ITER_TEMPLATE(half_t, const half_t *, SliceInputIterator<TRANSHALF(true)>);
RPU_UMH_ITER_TEMPLATE(half_t, const half_t *, SliceInputIterator<TRANSHALF(false)>);
RPU_UMH_ITER_TEMPLATE(half_t, IndexReaderSliceInputIterator<TRANSHALF(true)>, const half_t *);
RPU_UMH_ITER_TEMPLATE(half_t, IndexReaderSliceInputIterator<TRANSHALF(false)>, const half_t *);
RPU_UMH_ITER_TEMPLATE(half_t, EyeInputIterator<half_t>, const half_t *);
RPU_UMH_ITER_TEMPLATE(half_t, const half_t *, EyeInputIterator<half_t>);

#undef TRANSHALF
#endif

#undef RPU_UMH_ITER_TEMPLATE
} // namespace RPU
