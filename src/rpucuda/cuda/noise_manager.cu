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

#include "noise_manager.h"

#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <random>

#include "cuda_math_util.h"
#include "cuda_util.h"
#include "rpu_cub.h"

#include "io_iterator.h"

namespace RPU {

template <typename T> struct NonZeroFunctor {
  __device__ __forceinline__ T operator()(const T &a) const {
    return a == (T)0.0 ? (T)0.0 : (T)1.0;
  }
};

template <typename T>
__global__ void kernelAbsMaxNPSum(
    T *scale_values,
    const int m_batch,
    const T *amax_values,
    const T *psum_values,
    const T *nsum_values,
    const T out_bound,
    const T assumed_wmax,
    const T bm_max // io.max_bm_res/io.inp_res

) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;

  if (tid < m_batch) {
    // w_max*MAX(psum,nsum)/out_bound < scale
    T w_max = assumed_wmax;
    T amax = amax_values[tid];
    T psum = psum_values[tid];
    T nsum = -nsum_values[tid];
    T sum = MAX(psum, nsum);
    scale_values[tid] = MAX(amax, MIN(sum * w_max / out_bound, amax * bm_max));
    ;
  }
}

template <typename InputIteratorT, typename T>
__global__ void kernelNPSumBatchTrans(
    InputIteratorT input,
    const int total_size_in,
    const int m_batch_in,
    T *psum_values,
    T *nsum_values,
    T *psum_values0,
    T *nsum_values0) {

  // -- only use this version if m_batch < blockDim.x !!!
  // -- probably: strided version would be faster...

  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  // assumes that shared is of size 2*nthreads*sizeof(T) !!!!!!!!!!
  extern __shared__ __align__(sizeof(double)) unsigned char rpu_smem_nm[];
  T *block_sum_values = reinterpret_cast<T *>(rpu_smem_nm);

  T *block_psum_values = &block_sum_values[0];
  T *block_nsum_values = &block_sum_values[blockDim.x];

  const int size = total_size_in;
  const int m_batch = m_batch_in;

  block_psum_values[threadIdx.x] = (T)0.0;
  block_nsum_values[threadIdx.x] = (T)0.0;

  __syncthreads();

  if (tid < m_batch) {
    psum_values0[tid] = (T)0.0;
    nsum_values0[tid] = (T)0.0;
  }

  if (tid < size) {

    T value = input[tid];
    int midx = tid % m_batch;

    if (value >= (T)0.0) {
      atomicAdd(&(block_psum_values[midx]), value);
    } else {
      atomicAdd(&(block_nsum_values[midx]), value);
    }
  }
  __syncthreads();

  int bidx = threadIdx.x;
  if (bidx < m_batch) {
    atomicAdd(&(psum_values[bidx]), block_psum_values[bidx]);
    atomicAdd(&(nsum_values[bidx]), block_nsum_values[bidx]);
  }
}

template <typename InputIteratorT, typename T>
__global__ void kernelNPSumBatchTrans_LargeBatch(
    InputIteratorT input,
    const int total_size_in,
    const int m_batch_in,
    T *psum_values,
    T *nsum_values,
    T *psum_values0,
    T *nsum_values0) {

  // -- use this version if m_batch >= blockDim.x
  // -- just uses atomic on global memory

  int tid = blockDim.x * blockIdx.x + threadIdx.x;

  const int size = total_size_in;
  const int m_batch = m_batch_in;

  if (tid < m_batch) {
    psum_values0[tid] = (T)0.0;
    nsum_values0[tid] = (T)0.0;
  }

  if (tid < size) {

    T value = input[tid];
    int midx = tid % m_batch;

    if (value >= (T)0.0) {
      atomicAdd(&psum_values[midx], value);
    } else {
      atomicAdd(&nsum_values[midx], value);
    }
  }
}

template <typename T>
__global__ void kernelAverageAbsMaxSetScales(
    T *scales, T *ravg, const T *sum, const int m_batch_in, T decay_rate_in) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  const int m_batch = m_batch_in;
  T decay_rate = decay_rate_in;
  T max_avg = (*sum) / (T)m_batch;
  T run_avg = *ravg;

  if (tid < m_batch) {
    scales[tid] = run_avg * ((T)1.0 - decay_rate) + decay_rate * max_avg;
  }
  if (tid == m_batch) {
    *ravg = run_avg * ((T)1.0 - decay_rate) + decay_rate * max_avg;
  }
}

template <typename T>
__global__ void kernelAverageAbsMaxSingleMomentum(
    T *ravg, const T *sum, const int m_batch, const T *nz_value, T decay_rate) {
  // just single block!
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid == 0) {
    T sum_value = *sum;
    T nz = m_batch > 1 ? *nz_value : (sum_value != (T)0.0 ? (T)1.0 : (T)0.0);
    if (nz > (T)0.5) { // at least one non-zero
      T max_avg = (sum_value) / nz;
      *ravg = *ravg * ((T)1.0 - decay_rate) + decay_rate * max_avg;
    }
  }
}

template <typename T>
__global__ void kernelAbsMaxSingleMomentum(T *ravg, const T *amax, T decay_rate) {
  // just single block!
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid == 0) {
    T amax_value = *amax;
    *ravg = *ravg * ((T)1.0 - decay_rate) + decay_rate * amax_value;
  }
}

/****************************************************************************************************************/
/* NoiseManager */
/******************************************************************************************************************/
#define LAUNCH_NPSUM_KERNEL(KNAME, SHARED_MEM, ARGS)                                               \
  KNAME<InputIteratorT, T><<<nblocks, nthreads, SHARED_MEM, s>>> ARGS;

template <typename T>
NoiseManager<T>::NoiseManager(CudaContextPtr c, int size)
    : size_(size), context_(c), buffer_m_batch_(0), last_m_batch_(0), const_set_if_(false) {
  // initialize for m_batch=1
  dev_scale_values_ = RPU::make_unique<CudaArray<T>>(context_, 1);
  dev_psum_values_ = RPU::make_unique<CudaArray<T>>(context_, 1);
  dev_nsum_values_ = RPU::make_unique<CudaArray<T>>(context_, 1);
  dev_ravg_scale_value_ = RPU::make_unique<CudaArray<T>>(context_, 1);
  dev_ravg_scale_value_->setConst(1.0);
  dev_nzeros_value_ = RPU::make_unique<CudaArray<T>>(context_, 1);

  amaximizer_ = RPU::make_unique<Maximizer<T>>(context_, size, true);
  maximizer_ = RPU::make_unique<Maximizer<T>>(context_, size, false);

  size_t temp_storage_bytes = 0;
  RPU_CUB_NS_QUALIFIER DeviceReduce::Reduce(
      nullptr, temp_storage_bytes, dev_psum_values_->getData(), dev_psum_values_->getData(), size_,
      nsum_op_, (T)0.0, context_->getStream());

  dev_v_temp_storage_ = RPU::make_unique<CudaArray<char>>(context_, temp_storage_bytes);
  context_->synchronize();
}

template <typename T> void NoiseManager<T>::initializeBatchBuffer(int m_batch) {
  // this inits all the buffers needed for PMSum only !!

  if ((m_batch > 1) && (buffer_m_batch_ < m_batch)) {
    buffer_m_batch_ = m_batch;

    dev_psum_values_ = RPU::make_unique<CudaArray<T>>(context_, m_batch);
    dev_psum_values0_ = RPU::make_unique<CudaArray<T>>(context_, m_batch);
    dev_psum_values0_->setConst((T)0.0);

    dev_nsum_values_ = RPU::make_unique<CudaArray<T>>(context_, m_batch);
    dev_nsum_values0_ = RPU::make_unique<CudaArray<T>>(context_, m_batch);
    dev_nsum_values0_->setConst((T)0.0);

    int *offsets = new int[m_batch + 1];

    // not trans
    for (int i = 0; i <= m_batch; i++) {
      offsets[i] = i * size_;
    }

    dev_offsets_ = RPU::make_unique<CudaArray<int>>(context_, m_batch + 1, offsets);

    size_t temp_storage_bytes = 0;
    RPU_CUB_NS_QUALIFIER DeviceSegmentedReduce::Reduce(
        nullptr, temp_storage_bytes, dev_psum_values_->getData(), dev_psum_values_->getData(),
        m_batch, dev_offsets_->getData(), dev_offsets_->getData() + 1, psum_op_, 0,
        context_->getStream());
    dev_m_temp_storage_ = RPU::make_unique<CudaArray<char>>(context_, temp_storage_bytes);

    const_set_if_ = false;
    context_->synchronize();
    delete[] offsets;
  }
}

template <typename T>
void NoiseManager<T>::dumpExtra(RPU::state_t &extra, const std::string prefix) {

  using V = std::vector<T>;
  RPU::state_t state;
  context_->synchronize();

  RPU::insert(state, "ravg_initialized", ravg_initialized_);
  RPU::insert(state, "const_set_if", const_set_if_);

  RPU::insert(state, "dev_ravg_scale_value", dev_ravg_scale_value_);
  RPU::insert(state, "dev_scale_values", dev_scale_values_);
  RPU::insert(state, "dev_nzeros_value", dev_nzeros_value_);

  // amaximizer_->dumpExtra(state, "amaximizer");
  // maximizer_->dumpExtra(state, "maximizer");

  // will not handle buffers in to store data between applyToInput and applyToOutput

  RPU::insertWithPrefix(extra, state, prefix);
}

template <typename T>
void NoiseManager<T>::loadExtra(const RPU::state_t &extra, const std::string prefix, bool strict) {

  context_->synchronize();
  auto state = RPU::selectWithPrefix(extra, prefix);

  // amaximizer_->loadExtra(state, "amaximizer", strict);
  // maximizer_->loadExtra(state, "maximizer", strict);

  RPU::load(state, "ravg_initialized", ravg_initialized_, strict);
  RPU::load(state, "const_set_if", const_set_if_, strict);

  RPU::load(context_, state, "dev_ravg_scale_value", dev_ravg_scale_value_, strict);
  RPU::load(context_, state, "dev_scale_values", dev_scale_values_, strict);
  RPU::load(context_, state, "dev_nzeros_value", dev_nzeros_value_, strict);
}

template <typename T>
template <typename InputIteratorT>
void NoiseManager<T>::computeNPSum(InputIteratorT dev_input, int m_batch, bool trans) {
  cudaStream_t s = context_->getStream();

  if (m_batch == 1) {
    size_t ssz = dev_v_temp_storage_->getSize();
    RPU_CUB_NS_QUALIFIER DeviceReduce::Reduce(
        (void *)dev_v_temp_storage_->getData(), ssz, dev_input, dev_psum_values_->getData(), size_,
        psum_op_, (T)0, s);
    RPU_CUB_NS_QUALIFIER DeviceReduce::Reduce(
        (void *)dev_v_temp_storage_->getData(), ssz, dev_input, dev_nsum_values_->getData(), size_,
        nsum_op_, (T)0, s);

  } else {

    if (buffer_m_batch_ < m_batch) {
      this->initializeBatchBuffer(m_batch);
    }

    if (trans) {

      std::swap(dev_psum_values_, dev_psum_values0_);
      std::swap(dev_nsum_values_, dev_nsum_values0_);

      int nthreads = context_->getNThreads();
      int n = size_ * m_batch;
      int nblocks = context_->getNBlocks(n, nthreads);
      if (m_batch <= nthreads) {
        int shared_mem = 2 * nthreads * sizeof(T);

        LAUNCH_NPSUM_KERNEL(
            kernelNPSumBatchTrans, shared_mem,
            (dev_input, n, m_batch, dev_psum_values_->getData(), dev_nsum_values_->getData(),
             dev_psum_values0_->getData(), dev_nsum_values0_->getData()));

      } else {
        // simple atomic global memory version
        LAUNCH_NPSUM_KERNEL(
            kernelNPSumBatchTrans_LargeBatch, 0,
            (dev_input, n, m_batch, dev_psum_values_->getData(), dev_nsum_values_->getData(),
             dev_psum_values0_->getData(), dev_nsum_values0_->getData()));
      }

    } else {

      // Fast Segmented reduction
      size_t ssz = dev_m_temp_storage_->getSize();
      RPU_CUB_NS_QUALIFIER DeviceSegmentedReduce::Reduce(
          (void *)dev_m_temp_storage_->getData(), ssz, dev_input, dev_psum_values_->getData(),
          m_batch, dev_offsets_->getData(), dev_offsets_->getData() + 1, psum_op_, (T)0.0, s);
      RPU_CUB_NS_QUALIFIER DeviceSegmentedReduce::Reduce(
          (void *)dev_m_temp_storage_->getData(), ssz, dev_input, dev_nsum_values_->getData(),
          m_batch, dev_offsets_->getData(), dev_offsets_->getData() + 1, nsum_op_, (T)0.0, s);
    }
  }
}

template <typename T> void NoiseManager<T>::setAverageAbsMax(T value) {
  dev_ravg_scale_value_->setConst(value);
  dev_scale_values_->setConst(value);
  ravg_initialized_ = true;
  context_->synchronize();
}

template <typename T> T NoiseManager<T>::getAverageAbsMax() const {
  T tmp;
  dev_ravg_scale_value_->copyTo(&tmp);
  return tmp;
};

template <typename T>
template <typename InputIteratorT>
void NoiseManager<T>::compute(
    InputIteratorT dev_input,
    const NoiseManagementType &nm_type,
    const IOMetaParameter<T> &io,
    int m_batch,
    bool trans,
    bool is_test) {

  // does not check for positive m_batch!
  nm_type_ = nm_type;

  switch (nm_type_) {

  case NoiseManagementType::None: {
    return;
  }
  case NoiseManagementType::Constant: {
    if (m_batch > dev_scale_values_->getSize()) {
      dev_scale_values_ = RPU::make_unique<CudaArray<T>>(context_, m_batch);
      const_set_if_ = false;
    }
    if (!const_set_if_) {
      dev_scale_values_->setConst(io.nm_thres > (T)0.0 ? (T)io.nm_thres : (T)1.0);
      const_set_if_ = true;
    }
    return;
  }

  case NoiseManagementType::Max: {
    this->maximizer_->compute(dev_input, m_batch, trans);
    if (io.nm_thres > (T)0.0) {
      this->maximizer_->saturateAbove(io.nm_thres);
    }

    return;
  }

  case NoiseManagementType::AbsMax: {
    this->amaximizer_->compute(dev_input, m_batch, trans);
    if (io.nm_thres > (T)0.0) {
      this->amaximizer_->saturateAbove(io.nm_thres);
    }

    return;
  }
  case NoiseManagementType::AbsMaxNPSum: {
    if (m_batch > dev_scale_values_->getSize()) {
      dev_scale_values_ = RPU::make_unique<CudaArray<T>>(context_, m_batch);
    }

    // get amax and npsum
    this->amaximizer_->compute(dev_input, m_batch, trans);
    if (io.nm_thres > (T)0.0) {
      this->amaximizer_->saturateAbove(io.nm_thres);
    }

    this->computeNPSum(dev_input, m_batch, trans);

    // combine
    int nthreads = context_->getNThreads();
    int nblocks = context_->getNBlocks(m_batch, nthreads);
    cudaStream_t s = context_->getStream();

    kernelAbsMaxNPSum<T><<<nblocks, nthreads, 0, s>>>(
        dev_scale_values_->getData(), m_batch, this->amaximizer_->getMaxValues(),
        dev_psum_values_->getDataConst(), dev_nsum_values_->getDataConst(),
        isinf((float)io.out_bound) ? (T)1.0 : io.out_bound, io.nm_assumed_wmax,
        io.inp_res > (T)0.0 ? io.max_bm_res / io.inp_res : (T)1.0);
    return;
  }

  case NoiseManagementType::AverageAbsMax:
  case NoiseManagementType::AverageAbsMaxSingleValue: {
    // CAUTION: the running average will not be saved for checkpointing... so there might be a
    // glitch when continueing training from checkpoint...
    // ALSO: average max is computed across
    // mbatch whereas for CPU it is based running average of single mat-vecs

    if ((nm_type_ == NoiseManagementType::AverageAbsMax) &&
        (m_batch > dev_scale_values_->getSize())) {

      dev_scale_values_ = RPU::make_unique<CudaArray<T>>(context_, m_batch);

      cudaStream_t s = context_->getStream();
      int nthreads = context_->getNThreads();

      // set scales to ravg [first time, could be set from outside]
      kernelAverageAbsMaxSetScales<T>
          <<<context_->getNBlocks(m_batch + 1, nthreads), nthreads, 0, s>>>(
              dev_scale_values_->getData(), dev_ravg_scale_value_->getData(),
              dev_ravg_scale_value_->getDataConst(), m_batch, (T)0.0);
    }

    if (!is_test) {

      this->amaximizer_->compute(dev_input, m_batch, trans);
      context_->synchronize();
      cudaStream_t s = context_->getStream();
      int nthreads = context_->getNThreads();
      int nblocks = context_->getNBlocks(m_batch, nthreads);

      if (m_batch > 1) {
        // first compute the average of the max over batch
        if (!dev_a_temp_storage_ || m_batch > last_m_batch_) {
          dev_avgmax_value_ = RPU::make_unique<CudaArray<T>>(context_, 1);

          size_t temp_storage_bytes = 0;
          RPU_CUB_NS_QUALIFIER DeviceReduce::Sum(
              nullptr, temp_storage_bytes, amaximizer_->getMaxValues(),
              dev_avgmax_value_->getData(), m_batch, s);
          dev_a_temp_storage_ = RPU::make_unique<CudaArray<char>>(context_, temp_storage_bytes);
          last_m_batch_ = m_batch;
          context_->synchronize();
        }

        size_t ssz = dev_a_temp_storage_->getSize();
        RPU_CUB_NS_QUALIFIER DeviceReduce::Sum(
            (void *)dev_a_temp_storage_->getData(), ssz, amaximizer_->getMaxValues(),
            dev_avgmax_value_->getData(), m_batch, s);
      }

      if (nm_type_ == NoiseManagementType::AverageAbsMax) {
        // now update the running scale and set the current scales constant for all m_batch
        kernelAverageAbsMaxSetScales<T>
            <<<context_->getNBlocks(m_batch + 1, nthreads), nthreads, 0, s>>>(
                dev_scale_values_->getData(), dev_ravg_scale_value_->getData(),
                m_batch > 1 ? dev_avgmax_value_->getData() : amaximizer_->getMaxValues(),
                dev_scale_values_->getSize(),
                ravg_initialized_ ? MIN(io.nm_decay, (T)1.0) : (T)1.0);

      } else {
        // count non-zero

        if (m_batch > 1) {
          NonZeroFunctor<T> nonzero_functor;
          RPU_CUB_NS_QUALIFIER TransformInputIterator<T, NonZeroFunctor<T>, T *> nz_input(
              amaximizer_->getMaxValues(), nonzero_functor);
          // temp storage already requested above
          size_t ssz = dev_a_temp_storage_->getSize();
          RPU_CUB_NS_QUALIFIER DeviceReduce::Sum(
              (void *)dev_a_temp_storage_->getData(), ssz, nz_input, dev_nzeros_value_->getData(),
              m_batch, s);
        }
        // just update the running avg value as only single output requested
        kernelAverageAbsMaxSingleMomentum<T><<<1, 1, 0, s>>>(
            dev_ravg_scale_value_->getData(),
            m_batch > 1 ? dev_avgmax_value_->getData() : amaximizer_->getMaxValues(), m_batch,
            m_batch > 1 ? dev_nzeros_value_->getData() : nullptr,
            ravg_initialized_ ? (T)MIN(io.nm_decay, (T)1.0)
                              : (T)1.0); // Note that meaning of decay is per batch here
      }
      ravg_initialized_ = true;
    }
    return;
  }
  case NoiseManagementType::AbsMaxSingleValue: {

    // this is overall max over m_batch
    if (!is_test) {

      this->amaximizer_->compute(dev_input, m_batch, trans);
      context_->synchronize();
      cudaStream_t s = context_->getStream();
      int nthreads = context_->getNThreads();
      int nblocks = context_->getNBlocks(m_batch, nthreads);

      if (m_batch > 1) {
        // another max pass
        if (!dev_a_temp_storage_ || m_batch > last_m_batch_) {
          dev_avgmax_value_ = RPU::make_unique<CudaArray<T>>(context_, 1);

          size_t temp_storage_bytes = 0;
          RPU_CUB_NS_QUALIFIER DeviceReduce::Sum(
              nullptr, temp_storage_bytes, amaximizer_->getMaxValues(),
              dev_avgmax_value_->getData(), m_batch, s);
          dev_a_temp_storage_ = RPU::make_unique<CudaArray<char>>(context_, temp_storage_bytes);
          last_m_batch_ = m_batch;
          context_->synchronize();
        }

        size_t ssz = dev_a_temp_storage_->getSize();
        RPU_CUB_NS_QUALIFIER DeviceReduce::Max(
            (void *)dev_a_temp_storage_->getData(), ssz, amaximizer_->getMaxValues(),
            dev_avgmax_value_->getData(), m_batch, s);
      }

      // just update the running avg value as only single output requested
      kernelAbsMaxSingleMomentum<T><<<1, 1, 0, s>>>(
          dev_ravg_scale_value_->getData(),
          m_batch > 1 ? dev_avgmax_value_->getData() : amaximizer_->getMaxValues(),
          ravg_initialized_ ? MIN(io.nm_decay, (T)1.0)
                            : (T)1.0); // Note that meaning of decay is per batch here

      ravg_initialized_ = true;
    }
    return;
  }

  default:
    RPU_FATAL("Noise management type not implemented.");
  }
}

template <typename T> T *NoiseManager<T>::getScaleValues() const {
  switch (nm_type_) {
  case NoiseManagementType::None:
    return nullptr;
  case NoiseManagementType::AbsMaxNPSum:
  case NoiseManagementType::Constant:
  case NoiseManagementType::AverageAbsMax:
    return dev_scale_values_->getData();
  case NoiseManagementType::AbsMax:
    return amaximizer_->getMaxValues();
  case NoiseManagementType::Max:
    return maximizer_->getMaxValues();
  case NoiseManagementType::AverageAbsMaxSingleValue:
  case NoiseManagementType::AbsMaxSingleValue:
    if (!ravg_initialized_) {
      RPU_FATAL("Running average not yet initializated. Cannot use getScaleValues() yet.");
    }
    return dev_ravg_scale_value_->getData();
  default:
    RPU_FATAL("Noise management type not implemented.");
  }
};

#define ARGS1(NUM_T) , const NoiseManagementType &, const IOMetaParameter<NUM_T> &, int, bool, bool
#define ARGS2 , int, bool

template class NoiseManager<float>;
RPU_GEN_IITER_TEMPLATES(float, void, NoiseManager<float>::compute, ARGS1(float));
RPU_GEN_IITER_TEMPLATES(float, void, NoiseManager<float>::computeNPSum, ARGS2);

#ifdef RPU_USE_DOUBLE
template class NoiseManager<double>;
RPU_GEN_IITER_TEMPLATES(double, void, NoiseManager<double>::compute, ARGS1(double));
RPU_GEN_IITER_TEMPLATES(double, void, NoiseManager<double>::computeNPSum, ARGS2);
#endif

#ifdef RPU_USE_FP16
template class NoiseManager<half_t>;
RPU_GEN_IITER_TEMPLATES(half_t, void, NoiseManager<half_t>::compute, ARGS1(half_t));
RPU_GEN_IITER_TEMPLATES(half_t, void, NoiseManager<half_t>::computeNPSum, ARGS2);
#endif

#undef ARGS1
#undef ARGS2
#undef LAUNCH_NPSUM_KERNEL

} // namespace RPU
