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

#include "cuda_math_util.h"
#include "io_iterator.h"
#include "weight_remapper_cuda.h"

namespace RPU {

template <typename T>
__global__ void kernelRemapScale(
    T *weights_out,
    T *scales, // d_size
    const T *weights_in,
    const int x_size_in,    //
    const int d_size_in,    // major
    const float *max_value, // scalar!
    const T
        remapped_wmax // this is the value the max(w) should be mapped to in the array (usually 1.0)
) {
  int x_size = x_size_in;
  int d_size = d_size_in;
  int size = x_size * d_size;

  T mvalue = (T)fabs(*max_value) / remapped_wmax;
  mvalue = mvalue > 0 ? mvalue : (T)1.0;

  RPU_CUDA_1D_KERNEL_LOOP(idx, size) {
    T w = weights_in[idx];
    weights_out[idx] = w / mvalue;

    if (idx < d_size) {
      scales[idx] *= mvalue;
    }
  }
}

/*Scale row-wise [d_size major, ie trans]*/
template <typename T>
__global__ void kernelRemapScaleRowwise(
    T *weights_out,
    T *scales, // d_size
    const T *weights_in,
    const int x_size_in,     //
    const int d_size_in,     // major
    const float *max_values, // d_size
    const T remapped_wmax_in // this is the value the max(w) should be mapped to in the array
                             // (usually 1.0)
) {
  int x_size = x_size_in;
  int d_size = d_size_in;
  int size = x_size * d_size;
  const T remapped_wmax = remapped_wmax_in;

  RPU_CUDA_1D_KERNEL_LOOP(idx, size) {
    T w = weights_in[idx];
    int didx = idx % d_size;

    T mvalue = fabs(max_values[didx]) / remapped_wmax;
    mvalue = mvalue > 0 ? mvalue : (T)1.0;

    weights_out[idx] = w / mvalue;

    if (idx < d_size) {
      scales[idx] *= mvalue;
    }
  }
}

/*Scale row-wise [d_size major, ie trans] bounded scales in range in respect to min_scale_value*/
template <typename T>
__global__ void kernelRemapScaleRowwiseRange(
    T *weights_out,
    T *scales_out,   // !! Cannot be in-place, otherwise some threads read it !!
    const T *scales, // d_size
    const T *weights_in,
    const int x_size_in,     //
    const int d_size_in,     // major
    const float *max_values, // d_size
    const float *neg_min_scale_value,
    const T max_scale_range_in,
    const T max_scale_ref) {
  int x_size = x_size_in;
  int d_size = d_size_in;
  int size = x_size * d_size;
  T max_scale_range = fabs(max_scale_range_in);
  const T min_scale = max_scale_range > 0 ? MAX(fabs(*neg_min_scale_value), max_scale_ref) : (T)0.0;

  RPU_CUDA_1D_KERNEL_LOOP(idx, size) {

    T w = weights_in[idx];
    int didx = idx % d_size;

    T mvalue = fabs(max_values[didx]);
    mvalue = mvalue > 0 ? mvalue : (T)1.0;
    T old_scale = fabs(scales[didx]);
    T new_scale = old_scale * mvalue; // this essential is the rowwise max value

    if (min_scale > 0) {

      T range = new_scale / min_scale;
      // clipping
      new_scale = range > max_scale_range ? max_scale_range * min_scale : new_scale;
    }

    weights_out[idx] = w * old_scale / new_scale;

    if (idx < d_size) {
      scales_out[idx] = new_scale;
    }
  }
}

// ctor
template <typename T>
WeightRemapperCuda<T>::WeightRemapperCuda(CudaContextPtr context, int x_size, int d_size)
    : context_(context), x_size_(x_size), d_size_(d_size), size_(x_size * d_size),
      max_size_(x_size * d_size) {}

template <typename T> int WeightRemapperCuda<T>::getNBlocks(int nthreads) {
  // nblocks here because max_size changes after init.
  return context_->getNBlocks(MIN(max_size_, size_), nthreads);
}

template <typename T>
void WeightRemapperCuda<T>::apply(
    T *weights, T current_lr, const WeightRemapParameter &wrmpar, T *scales, T *biases) {
  // does a weight remap to the scales.

  int nthreads = context_->getNThreads();
  auto s = context_->getStream();

  // this is to remap the weights
  switch (wrmpar.type) {
  case WeightRemapType::LayerwiseSymmetric: {

    if (!scales) {
      RPU_FATAL("Expect scales given.");
    }

    if (!amaximizer_) {
      amaximizer_ = RPU::make_unique<Maximizer<T>>(context_, size_, true);
    }
    amaximizer_->compute(weights, 1, false); // over whole matrix
    kernelRemapScale<T><<<getNBlocks(nthreads), nthreads, 0, s>>>(
        weights, scales, weights, x_size_, d_size_, amaximizer_->getMaxValues(),
        wrmpar.remapped_wmax);

    amaximizer_->compute(weights, 1, false); // over whole matrix

    break;
  }
  case WeightRemapType::ChannelwiseSymmetric: {

    if (!scales) {
      RPU_FATAL("Expect scales given.");
    }
    if (!row_amaximizer_) {
      row_amaximizer_ = RPU::make_unique<Maximizer<T>>(context_, x_size_, true);
    }
    row_amaximizer_->compute(weights, d_size_, true);

    if (wrmpar.max_scale_range <= 0) {

      kernelRemapScaleRowwise<T><<<getNBlocks(nthreads), nthreads, 0, s>>>(
          weights, scales, weights, x_size_, d_size_, row_amaximizer_->getMaxValues(),
          wrmpar.remapped_wmax);

    } else {
      if (wrmpar.remapped_wmax != 1.0) {
        RPU_FATAL("For max_scales set, expect wrmpar.remapped_wmax to be 1.");
      }

      // scales are forced to be within min(scales),..min(scale)*max_scale_range. If scales would
      // fall outside this range, weights can grow above 1
      if (!scale_minimizer_) {
        scale_minimizer_ = RPU::make_unique<Maximizer<T>>(context_, d_size_, false);
      }
      if (!scale_buffer_) {
        scale_buffer_ = RPU::make_unique<CudaArray<T>>(context_, d_size_);
      }

      scale_minimizer_->compute(NegateInputIterator<T>(scales), 1, false); // minus min
      RPU::math::copy(
          context_, d_size_, scales, 1, scale_buffer_->getData(), 1); // to avoid in-place

      kernelRemapScaleRowwiseRange<T><<<getNBlocks(nthreads), nthreads, 0, s>>>(
          weights, scales, scale_buffer_->getData(), weights, x_size_, d_size_,
          row_amaximizer_->getMaxValues(), scale_minimizer_->getMaxValues(), wrmpar.max_scale_range,
          wrmpar.max_scale_ref);
    }
    break;
  }

  case WeightRemapType::None: {
    break;
  }
  default:
    RPU_FATAL("Remapping type not implemented.");
  } // switch
}

template class WeightRemapperCuda<float>;
#ifdef RPU_USE_DOUBLE
template class WeightRemapperCuda<double>;
#endif

#undef RPU_WM_KERNEL_LOOP
} // namespace RPU
