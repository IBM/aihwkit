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

#include "cuda_fp16_util.h"
#include "cuda_math_util.h"
#include "io_iterator.h"
#include "rpu_cub.h"
#include "weight_clipper_cuda.h"

namespace RPU {

template <typename T> struct StdFunctor {
  StdFunctor(T size, T *sum) : size_(size), sum_(sum){};

  __device__ __forceinline__ T operator()(const T &a) const {
    T m = *sum_ / size_;
    return T((a - m) * (a - m));
  }
  T size_;
  T *sum_;
};

template <typename T> __global__ void kernelAClipC(T *values, int size, T *a, T c, T max_clip) {

  T abs_a = fabs(*a / c);
  if (max_clip > (T)0.0) {
    abs_a = MIN(abs_a, max_clip);
  }

  RPU_CUDA_1D_KERNEL_LOOP(tid, size) { values[tid] = MIN(MAX(values[tid], -abs_a), abs_a); }
}

template <typename T>
__global__ void kernelAClipSqrt(T *values, int size, T *a, T sigma, T max_clip) {

  T abs_a = sqrt(fabs(*a) / (T)(size - 1)) * sigma;
  if (max_clip > (T)0.0) {
    abs_a = MIN(abs_a, max_clip);
  }

  RPU_CUDA_1D_KERNEL_LOOP(tid, size) { values[tid] = MIN(MAX(values[tid], -abs_a), abs_a); }
}

// ctor
template <typename T>
WeightClipperCuda<T>::WeightClipperCuda(CudaContextPtr context, int x_size, int d_size)
    : context_(context), x_size_(x_size), d_size_(d_size), size_(x_size * d_size) {

  T *tmp = nullptr;
  StdFunctor<T> std_functor((T)x_size_, tmp);
  RPU_CUB_NS_QUALIFIER TransformInputIterator<T, StdFunctor<T>, T *> std_input(tmp, std_functor);

  RPU_CUB_NS_QUALIFIER DeviceReduce::Sum(
      nullptr, temp_storage_bytes_, std_input, tmp, size_, context_->getStream());
  dev_temp_storage_ = RPU::make_unique<CudaArray<char>>(context, temp_storage_bytes_);
}

template <typename T>
void WeightClipperCuda<T>::apply(T *weights, const WeightClipParameter &wclpar) {

  int nthreads = context_->getNThreads();
  int nblocks = context_->getNBlocks(size_, nthreads);
  auto s = context_->getStream();

  switch (wclpar.type) {
  case WeightClipType::None: {
    break;
  }
  case WeightClipType::AverageChannelMax: {

    if (!row_amaximizer_) {
      row_amaximizer_ = RPU::make_unique<Maximizer<T>>(context_, x_size_, true);
      dev_sum_value_ = RPU::make_unique<CudaArray<T>>(context_, 1);
    }
    row_amaximizer_->compute(weights, d_size_, true);

    RPU_CUB_NS_QUALIFIER DeviceReduce::Sum(
        dev_temp_storage_->getData(), temp_storage_bytes_, row_amaximizer_->getMaxValues(),
        dev_sum_value_->getData(), d_size_, s);

    kernelAClipC<T><<<nblocks, nthreads, 0, s>>>(
        weights, size_, dev_sum_value_->getData(), (T)d_size_, wclpar.fixed_value);
    break;
  }

  case WeightClipType::LayerGaussian: {

    if (!dev_sum_value_) {
      dev_sum_value_ = RPU::make_unique<CudaArray<T>>(context_, 1);
    }
    if (!dev_std_value_) {
      dev_std_value_ = RPU::make_unique<CudaArray<T>>(context_, 1);
    }

    StdFunctor<T> std_functor((T)size_, dev_sum_value_->getData());
    RPU_CUB_NS_QUALIFIER TransformInputIterator<T, StdFunctor<T>, T *> std_input(
        weights, std_functor);

    // mean (sum)
    RPU_CUB_NS_QUALIFIER DeviceReduce::Sum(
        dev_temp_storage_->getData(), temp_storage_bytes_, weights, dev_sum_value_->getData(),
        size_, s);

    // std
    RPU_CUB_NS_QUALIFIER DeviceReduce::Sum(
        dev_temp_storage_->getData(), temp_storage_bytes_, std_input, dev_std_value_->getData(),
        size_, s);

    kernelAClipSqrt<T><<<nblocks, nthreads, 0, s>>>(
        weights, size_, dev_std_value_->getData(), wclpar.sigma, wclpar.fixed_value);

    break;
  }

  case WeightClipType::FixedValue: {

    if (wclpar.fixed_value > 0) {
      RPU::math::aclip(context_, weights, size_, (T)wclpar.fixed_value);
    }
    break;
  }

  default:
    RPU_FATAL("Clipping type not implemented.");
  } // switch
}

template class WeightClipperCuda<float>;
#ifdef RPU_USE_DOUBLE
template class WeightClipperCuda<double>;
#endif
#ifdef RPU_USE_FP16
template class WeightClipperCuda<half_t>;
#endif

#undef RPU_WM_KERNEL_LOOP
} // namespace RPU
