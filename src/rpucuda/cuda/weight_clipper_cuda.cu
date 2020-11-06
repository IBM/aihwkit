#include "cuda_math_util.h"
#include "io_iterator.h"
#include "weight_clipper_cuda.h"
#include <cub/cub.cuh>

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

template <typename T> __global__ void kernelAClipC(T *values, int size, T *a, T c) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  T abs_a = fabs(*a / c);
  if (tid < size) {
    values[tid] = MIN(MAX(values[tid], -abs_a), abs_a);
  }
}

template <typename T> __global__ void kernelAClipSqrt(T *values, int size, T *a, T sigma) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  T abs_a = sqrtf(fabs(*a) / (size - 1)) * sigma;
  if (tid < size) {
    values[tid] = MIN(MAX(values[tid], -abs_a), abs_a);
  }
}

// ctor
template <typename T>
WeightClipperCuda<T>::WeightClipperCuda(CudaContext *context, int x_size, int d_size)
    : context_(context), x_size_(x_size), d_size_(d_size), size_(x_size * d_size) {

  T *tmp = nullptr;
  StdFunctor<T> std_functor((T)x_size_, tmp);
  cub::TransformInputIterator<T, StdFunctor<T>, T *> std_input(tmp, std_functor);

  cub::DeviceReduce::Sum(
      nullptr, temp_storage_bytes_, std_input, tmp, size_, context_->getStream());
  dev_temp_storage_ = RPU::make_unique<CudaArray<char>>(context, temp_storage_bytes_);
}

template <typename T>
void WeightClipperCuda<T>::apply(T *weights, const WeightClipParameter &wclpar) {
  // does a weight remap to the scales.

  int nthreads = context_->getNThreads();
  int nblocks = context_->getNBlocks(size_, nthreads);
  auto s = context_->getStream();

  // this is to remap the weights
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

    cub::DeviceReduce::Sum(
        dev_temp_storage_->getData(), temp_storage_bytes_, row_amaximizer_->getMaxValues(),
        dev_sum_value_->getData(), d_size_, s);

    kernelAClipC<T>
        <<<nblocks, nthreads, 0, s>>>(weights, size_, dev_sum_value_->getData(), (T)d_size_);
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
    cub::TransformInputIterator<T, StdFunctor<T>, T *> std_input(weights, std_functor);

    // mean (sum)
    cub::DeviceReduce::Sum(
        dev_temp_storage_->getData(), temp_storage_bytes_, weights, dev_sum_value_->getData(),
        size_, s);

    // std
    cub::DeviceReduce::Sum(
        dev_temp_storage_->getData(), temp_storage_bytes_, std_input, dev_std_value_->getData(),
        size_, s);

    kernelAClipSqrt<T>
        <<<nblocks, nthreads, 0, s>>>(weights, size_, dev_std_value_->getData(), wclpar.sigma);

    break;
  }

  case WeightClipType::FixedValue: {
    if (wclpar.fixed_value >= 0) {
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

#undef RPU_WM_KERNEL_LOOP
} // namespace RPU
