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

#include "maximizer.h"

#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <random>

#include "cuda_math_util.h"
#include "cuda_util.h"
#include <cub/cub.cuh>

#include "io_iterator.h"

namespace RPU {

template <typename InputIteratorT, bool abs_if = true>
__global__ void kernelMaximizeBatchTrans(
    InputIteratorT input,
    const int total_size_in,
    const int m_batch_in,
    float *max_values,
    float *max_values0) {

  // -- only use this version if m_batch < blockDim.x !!!
  // -- probably: strided version would be faster...

  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  extern __shared__ int block_max_values[]; // assumes that shared is of size nthreads*sizeof(int)

  const int size = total_size_in;
  const int m_batch = m_batch_in;

  if (abs_if) {
    block_max_values[threadIdx.x] = 0;
  } else {
    block_max_values[threadIdx.x] = INT_MIN;
  }
  __syncthreads();

  if (tid < m_batch) {
    if (abs_if) {
      max_values0[tid] = 0; // for next round
    } else {
      max_values0[tid] = -FLT_MAX; // for next round
    }
  }

  if (tid < size) {

    float value = input[tid]; // typecast to float. (because float to int)

    int midx = tid % m_batch;

    if (abs_if) {
      value = (value >= 0) ? value : -value;
    }

    atomicMax(&(block_max_values[midx]), __float_as_int(value));
  }
  __syncthreads();

  int bidx = threadIdx.x;
  if (bidx < m_batch) {
    atomicMax((int *)&(max_values[bidx]), block_max_values[bidx]);
  }
}

template <typename InputIteratorT, bool abs_if = true>
__global__ void kernelMaximizeBatchTrans_LargeBatch(
    InputIteratorT input,
    const int total_size_in,
    const int m_batch_in,
    float *max_values,
    float *max_values0) {

  // -- use this version if m_batch >= blockDim.x
  // -- just uses atomic on global memory

  int tid = blockDim.x * blockIdx.x + threadIdx.x;

  const int size = total_size_in;
  const int m_batch = m_batch_in;

  if (tid < m_batch) {
    if (abs_if) {
      max_values0[tid] = 0; // for next round
    } else {
      max_values0[tid] = -FLT_MAX; // for next round
    }
  }

  if (tid < size) {

    float value = input[tid];

    int midx = tid % m_batch;

    if (abs_if) {
      value = (value >= 0) ? value : -value;
    }

    atomicMax((int *)&max_values[midx], __float_as_int(value));
  }
}

template <typename T> struct IndexReader {
  __host__ __device__ IndexReader(T *data_in) { data = data_in; }
  __host__ __device__ __forceinline__ T operator()(const int &idx) const {
    return (idx > 0) ? data[idx - 1] : 0;
  }

  __host__ __device__ __forceinline__ void setData(T *data_in) { data = data_in; }

  T *data;
};

template <typename T> struct BatchTransposer {
  __host__ __device__ BatchTransposer(T *data_in, int size_in, int m_batch_in) {
    m_batch = m_batch_in;
    size = size_in;
    data = data_in;
  }
  __host__ __device__ __forceinline__ T operator()(const int &idx) const {
    return data[(idx / size) + (idx % size) * m_batch];
  }

  __host__ __device__ __forceinline__ void setSizeAndBatch(int size_in, int m_batch_in) {
    m_batch = m_batch_in;
    size = size_in;
  }

  __host__ __device__ __forceinline__ void setData(T *data_in) { data = data_in; }

  T *data;
  int size;
  int m_batch;
};

namespace test_helper {

template <typename T>
void debugMaxBatched(const T *indata, int size, int m_batch, bool trans, float *max_values) {

  int *offsets = new int[m_batch + 1];

  for (int i = 0; i <= m_batch; i++) {
    offsets[i] = i * size;
  }

  CudaContext c(-1, false);
  CudaArray<T> dev_in(&c, size * m_batch, indata);
  CudaArray<float> dev_max_values(&c, m_batch);
  dev_max_values.setConst(0);
  CudaArray<float> dev_max_values0(&c, m_batch);

  CudaArray<int> dev_offsets(&c, m_batch + 1, offsets);

  CUDA_CALL(cudaPeekAtLastError());
  CUDA_CALL(cudaDeviceSynchronize());

  // test transform input iterator
  int *tmp = new int[size * m_batch];
  for (int i = 0; i < size * m_batch; i++) {
    tmp[i] = i + 1;
  }
  CudaArray<int> dev_in_index(&c, size * m_batch, tmp);
  CUDA_CALL(cudaDeviceSynchronize());

  IndexReader<T> idx_reader(dev_in.getData());
  cub::TransformInputIterator<T, IndexReader<T>, int *> in_itr(dev_in_index.getData(), idx_reader);

  cub::CountingInputIterator<int> index(0);
  BatchTransposer<T> batch_transposer(dev_in.getData(), size, m_batch);
  cub::TransformInputIterator<T, BatchTransposer<T>, cub::CountingInputIterator<int>> in_trans_itr(
      index, batch_transposer);

  IndexReader<int> idx_reader_host(tmp);
  cub::TransformInputIterator<int, IndexReader<int>, int *> test_host(tmp, idx_reader_host);
  std::cout << test_host[0] << std::endl;

  CustomMaxAbs max_abs;
  // Determine temporary device storage requirements
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceSegmentedReduce::Reduce(
      d_temp_storage, temp_storage_bytes, in_itr, dev_max_values.getData(), m_batch,
      dev_offsets.getData(), dev_offsets.getData() + 1, max_abs, 0, c.getStream());
  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  CUDA_CALL(cudaDeviceSynchronize());

  int nthreads = c.getNThreads();
  int nblocks = c.getNBlocks(size * m_batch, nthreads);
  cudaStream_t s = c.getStream();

  CUDA_TIMING_INIT;
  CUDA_TIMING_START(c);

  if (trans) {

    // this works, too, but has some performance hit, because of non-aligned memory reads
    // cub::DeviceSegmentedReduce::Reduce(d_temp_storage, temp_storage_bytes,
    // 				   in_trans_itr, dev_max_values.getData(),
    // 				   m_batch, dev_offsets.getData(),
    // 				   dev_offsets.getData()+1, max_abs,0,c.getStream());

    if (m_batch > nthreads) {
      kernelMaximizeBatchTrans_LargeBatch<<<nblocks, nthreads, 0, s>>>(
          in_itr, size * m_batch, m_batch, dev_max_values.getData(), dev_max_values0.getData());

    } else {
      kernelMaximizeBatchTrans<<<nblocks, nthreads, nthreads * sizeof(int), s>>>(
          in_itr, size * m_batch, m_batch, dev_max_values.getData(), dev_max_values0.getData());
    }

  } else {
    // only trans==false
    // Fast Segmented reduction (much faster than loop from outside)
    cub::DeviceSegmentedReduce::Reduce(
        d_temp_storage, temp_storage_bytes, in_itr, dev_max_values.getData(), m_batch,
        dev_offsets.getData(), dev_offsets.getData() + 1, max_abs, 0, c.getStream());
  }

  CUDA_TIMING_STOP(c, "Max Batch");

  CUDA_CALL(cudaPeekAtLastError());
  CUDA_CALL(cudaDeviceSynchronize());
  dev_max_values.copyTo(max_values);
  CUDA_CALL(cudaDeviceSynchronize());
  cudaFree(d_temp_storage);
  delete[] offsets;
  delete[] tmp;
}
#ifdef RPU_USE_DOUBLE
template void debugMaxBatched<double>(double const *, int, int, bool, float *);
#endif
template void debugMaxBatched<float>(float const *, int, int, bool, float *);
} // namespace test_helper

/****************************************************************************************************************/
/* MAXIMIZER */
/******************************************************************************************************************/
#define LAUNCH_MAX_KERNEL(KNAME, SHARED_MEM, ARGS)                                                 \
  if (abs_if_) {                                                                                   \
    KNAME<InputIteratorT, true><<<nblocks, nthreads, SHARED_MEM, s>>> ARGS;                        \
  } else {                                                                                         \
    KNAME<InputIteratorT, false><<<nblocks, nthreads, SHARED_MEM, s>>> ARGS;                       \
  }

template <typename T>
Maximizer<T>::Maximizer(CudaContext *c, int size, bool abs_if)
    : size_{size}, context_{c}, buffer_m_batch_{0}, abs_if_{abs_if} {
  // initialize for m_batch=1
  dev_max_values_ = RPU::make_unique<CudaArray<float>>(context_, 1);
  size_t temp_storage_bytes = 0;
  if (abs_if_) {
    cub::DeviceReduce::Reduce(
        nullptr, temp_storage_bytes, dev_max_values_->getData(), dev_max_values_->getData(), size_,
        max_abs_op_, 0, context_->getStream());
  } else {
    cub::DeviceReduce::Max(
        nullptr, temp_storage_bytes, dev_max_values_->getData(), dev_max_values_->getData(), size_,
        context_->getStream());
  }

  dev_v_temp_storage_ = RPU::make_unique<CudaArray<char>>(context_, temp_storage_bytes);
}

template <typename T> void Maximizer<T>::initializeBatchBuffer(int m_batch) {

  if ((m_batch > 1) && (buffer_m_batch_ != m_batch)) {
    buffer_m_batch_ = m_batch;

    dev_max_values_ = RPU::make_unique<CudaArray<float>>(context_, m_batch);
    dev_max_values0_ = RPU::make_unique<CudaArray<float>>(context_, m_batch);
    dev_max_values0_->setConst(abs_if_ ? 0 : std::numeric_limits<T>::min());

    int *offsets = new int[m_batch + 1];

    // not trans
    for (int i = 0; i <= m_batch; i++) {
      offsets[i] = i * size_;
    }

    dev_offsets_ = RPU::make_unique<CudaArray<int>>(context_, m_batch + 1, offsets);

    size_t temp_storage_bytes = 0;
    if (abs_if_) {
      cub::DeviceSegmentedReduce::Reduce(
          nullptr, temp_storage_bytes, dev_max_values_->getData(), dev_max_values_->getData(),
          m_batch, dev_offsets_->getData(), dev_offsets_->getData() + 1, max_abs_op_, 0,
          context_->getStream());
    } else {
      cub::DeviceSegmentedReduce::Max(
          nullptr, temp_storage_bytes, dev_max_values_->getData(), dev_max_values_->getData(),
          m_batch, dev_offsets_->getData(), dev_offsets_->getData() + 1, context_->getStream());
    }
    dev_m_temp_storage_ = RPU::make_unique<CudaArray<char>>(context_, temp_storage_bytes);

    context_->synchronize();
    delete[] offsets;
    // dev_offsets_->printValues();
  }
}

template <typename T> void Maximizer<T>::setZeroBelow(T thres) {
  RPU::math::elemsetbelowzero(
      context_, dev_max_values_->getData(), dev_max_values_->getSize(), (float)thres);
}

template <typename T> void Maximizer<T>::saturateAbove(T thres) {
  RPU::math::elemmin<float>(
      context_, dev_max_values_->getData(), dev_max_values_->getSize(), (float)thres);
}

template <typename T>
template <typename InputIteratorT>
void Maximizer<T>::compute(InputIteratorT dev_input, int m_batch, bool trans) {

  // does not check for positive m_batch!
  cudaStream_t s = context_->getStream();

  if (m_batch == 1) {
    size_t ssz = dev_v_temp_storage_->getSize();
    if (abs_if_) {
      cub::DeviceReduce::Reduce(
          (void *)dev_v_temp_storage_->getData(), ssz, dev_input, dev_max_values_->getData(), size_,
          max_abs_op_, (T)0, s);
    } else {
      cub::DeviceReduce::Max(
          (void *)dev_v_temp_storage_->getData(), ssz, dev_input, dev_max_values_->getData(), size_,
          s);
    }

  } else {

    if (trans) {

      if (buffer_m_batch_ < m_batch) {
        this->initializeBatchBuffer(m_batch);
      }

      std::swap(dev_max_values_, dev_max_values0_);
      int nthreads = context_->getNThreads();
      int n = size_ * m_batch;
      int nblocks = context_->getNBlocks(n, nthreads);
      if (m_batch <= nthreads) {
        int shared_mem = nthreads * sizeof(int);

        LAUNCH_MAX_KERNEL(
            kernelMaximizeBatchTrans, shared_mem,
            (dev_input, n, m_batch, dev_max_values_->getData(), dev_max_values0_->getData()));

      } else {
        // simple atomic global memory version
        LAUNCH_MAX_KERNEL(
            kernelMaximizeBatchTrans_LargeBatch, 0,
            (dev_input, n, m_batch, dev_max_values_->getData(), dev_max_values0_->getData()));
      }

    } else {
      if (buffer_m_batch_ != m_batch) { // !! need to reinitilize offsets when batch changes !
        this->initializeBatchBuffer(m_batch);
      }

      // Fast Segmented reduction (much faster than loop from outside)
      size_t ssz = dev_m_temp_storage_->getSize();
      if (abs_if_) {
        cub::DeviceSegmentedReduce::Reduce(
            (void *)dev_m_temp_storage_->getData(), ssz, dev_input, dev_max_values_->getData(),
            m_batch, dev_offsets_->getData(), dev_offsets_->getData() + 1, max_abs_op_, (T)0.0, s);
      } else {
        cub::DeviceSegmentedReduce::Max(
            (void *)dev_m_temp_storage_->getData(), ssz, dev_input, dev_max_values_->getData(),
            m_batch, dev_offsets_->getData(), dev_offsets_->getData() + 1, s);
      }
    }
  }
}

#define ARGS1 , int, bool

template class Maximizer<float>;
RPU_GEN_IITER_TEMPLATES(float, void, Maximizer<float>::compute, ARGS1);
template void Maximizer<float>::compute(NegateInputIterator<float> ARGS1);

#ifdef RPU_USE_DOUBLE
template class Maximizer<double>;
RPU_GEN_IITER_TEMPLATES(double, void, Maximizer<double>::compute, ARGS1);
template void Maximizer<double>::compute(NegateInputIterator<double> ARGS1);
#endif

#undef RPU_MX_TEMPLATE
#undef LAUNCH_MAX_KERNEL
#undef ARGS1

} // namespace RPU
