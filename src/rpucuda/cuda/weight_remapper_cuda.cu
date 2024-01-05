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
#include "weight_remapper_cuda.h"

namespace RPU {

template <typename T>
__global__ void kernelRemapScale(
    T *weights_out,
    T *scales, // d_size
    const T *weights_in,
    const int x_size_in, //
    const int d_size_in, // major
    const T *max_value,  // scalar!
    const T
        remapped_wmax // this is the value the max(w) should be mapped to in the array (usually 1.0)
) {
  int x_size = x_size_in;
  int d_size = d_size_in;
  int size = x_size * d_size;

  T mvalue = fabs(*max_value) / remapped_wmax;
  mvalue = mvalue > (T)0.0 ? mvalue : (T)1.0;

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
    const T *max_values,     // d_size
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
    mvalue = mvalue > (T)0.0 ? mvalue : (T)1.0;

    weights_out[idx] = w / mvalue;

    if (idx < d_size) {
      scales[idx] *= mvalue;
    }
  }
}

/*Scale row-wise [d_size major, ie trans]*/
template <typename T>
__global__ void kernelSWA(
    T *swa_weights_out,
    const T *scales, // d_size
    const T *swa_weights_in,
    const T *weights,
    const int n_in,
    const T ratio_in,
    const int d_size_in,
    const int size_in) // major
{
  int d_size = d_size_in;
  int size = size_in;
  int n = n_in;
  T ratio = ratio_in;

  RPU_CUDA_1D_KERNEL_LOOP(idx, size) {
    T w = weights[idx];
    int didx = idx % d_size;
    T s = scales[didx];
    T swa_w = swa_weights_in[idx] * ratio;

    swa_weights_out[idx] = swa_w + w * s / (T)n;
  }
}

/*********************************************************************************/
/*row wise sumsqr in trans using atomic */
template <typename T>
__global__ void kernelRowwiseSumSqr(
    const T *weights, // assumes d_size major
    const int size_in,
    const int d_size_in,
    T *msqr_values // ! expect zero!! d_size
) {
  // -- assumes no trans, thus d_idx is major
  // -- stride for a constant d_idx per thread (mean val put into register)
  // -- total threads should best be divisible by d_size
  // -- total threads need to be larger than d_size !
  // -- e.g. nblock*nthreads >= max(d_size,size/4) to get 4 strides

  const int size = size_in;
  const int d_size = d_size_in;

  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  int total_threads = blockDim.x * gridDim.x;

  int total_threads_d_size = MIN(total_threads / d_size * d_size, size);

  T msqr = 0;

  if (tid < total_threads_d_size) {

    int d_idx = tid % d_size;

    // read/write stride loop
    for (int i_stride = 0; i_stride < size; i_stride += total_threads_d_size) {

      int idx = i_stride + tid;

      if (idx < size) {
        T value = weights[idx];
        msqr += value * value;
      }
    }

    // just do atomic. might be rather slow...
    atomicAdd(&msqr_values[d_idx], msqr);
  }
}

/*Scale row-wise but taking sqrt and extra scale [d_size major, ie trans]*/
template <typename T>
__global__ void kernelRemapScaleRowwiseSqrt(
    T *weights_out,
    T *scales, // d_size
    const T *weights_in,
    const int x_size_in,  //
    const int d_size_in,  // major
    const T *msqr_values, // d_size
    const T row_norm_in,
    const bool clip_if_in) {
  bool clip_if = clip_if_in;
  int x_size = x_size_in;
  int d_size = d_size_in;
  int size = x_size * d_size;
  T row_norm = row_norm_in;
  row_norm = row_norm > (T)0.0 ? row_norm : (T)1.0;

  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  int total_threads = blockDim.x * gridDim.x;
  for (int i_stride = 0; i_stride < size; i_stride += total_threads) {
    int idx = i_stride + tid;

    if (idx < size) {
      T w = weights_in[idx];
      int didx = idx % d_size;

      T mvalue = sqrt(fabs(msqr_values[didx]));

      T ratio = mvalue / row_norm;
      // ratio = ratio>0? ratio : (T) 1.0;
      ratio = ratio > (T)1.0 ? ratio : (T)1.0;

      weights_out[idx] = w / ratio;

      if (!clip_if && idx < d_size) {
        scales[idx] *= ratio;
      }
    }
  }
}

/*Scale row-wise by a learning rate only if exceeded*/
template <typename T>
__global__ void kernelRemapScaleExceededChannel(
    T *weights_out,
    T *scales, // d_size
    const T *weights_in,
    const int x_size_in,         //
    const int d_size_in,         // major
    const int *exceeded_channel, // d_size
    const bool clip_if_in,
    const T current_lr) {
  bool clip_if = clip_if_in;
  T amount = (T)1.0 + current_lr;
  int x_size = x_size_in;
  int d_size = d_size_in;
  int size = x_size * d_size;

  RPU_CUDA_1D_KERNEL_LOOP(idx, size) {
    T w = weights_in[idx];
    int didx = idx % d_size;

    bool exceeded = exceeded_channel[didx] > 0;

    weights_out[idx] = exceeded ? w / amount : w;

    if (!clip_if && idx < d_size) { // without clip this is not very healthy....
      scales[idx] *= amount;
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
    const int x_size_in, //
    const int d_size_in, // major
    const T *max_values, // d_size
    const T *neg_min_scale_value,
    const T max_scale_range_in,
    const T max_scale_ref) {
  int x_size = x_size_in;
  int d_size = d_size_in;
  int size = x_size * d_size;
  T max_scale_range = fabs(max_scale_range_in);
  const T min_scale =
      max_scale_range > (T)0.0 ? MAX(fabs(*neg_min_scale_value), max_scale_ref) : (T)0.0;

  RPU_CUDA_1D_KERNEL_LOOP(idx, size) {

    T w = weights_in[idx];
    int didx = idx % d_size;

    T mvalue = fabs(max_values[didx]);
    mvalue = mvalue > (T)0.0 ? mvalue : (T)1.0;
    T old_scale = fabs(scales[didx]);
    T new_scale = old_scale * mvalue; // this essential is the rowwise max value

    if (min_scale > (T)0.0) {

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

/*Scale row-wise and shift*/
template <typename T>
__global__ void kernelRemapScaleAndShiftRowwise(
    T *weights_out,
    T *scales_out, // cannot be in-place otherwise some threads might use new scale...!!!
    T *biases_out,
    const T *weights_in,
    const T *scales,        // NOTE: these need to be initialized !!! should be d_size on per output
    const T *biases,        // NOTE: these need to be initialized !!!
    const int x_size_in,    //
    const int d_size_in,    // major
    const T *max_values,    // d_size: eg. max per row
    const T *neg_max_values // d_size: eg. max of - weights
) {
  int x_size = x_size_in;
  int d_size = d_size_in;
  int size = x_size * d_size;

  RPU_CUDA_1D_KERNEL_LOOP(idx, size) {
    T w = weights_in[idx];
    int didx = idx % d_size;
    T max_value = (T)max_values[didx]; // shared mem!?? maybe not worth the effort
    T min_value = -(T)neg_max_values[didx];
    T b = biases[didx];
    T s = scales[didx];

    T half_span = (max_value - min_value) / (T)2.0;
    half_span = half_span > (T)0.0 ? half_span : (T)1.0;
    T new_scale = s * half_span;
    T new_b = b - new_scale - s * min_value;
    // T full_w = w*s + b;
    // T new_full_w = new_w*new_scale + new_b;
    /* new_full_w == full_w
       new_w*new_scale + new_b == w*s + b
       new_w == (w*s + b - new_b)/new_scale
       new_w == (w*s + new_scale + s*min_value)/new_scale
       new_w == w*s/new_scale + 1 + s*min_value/new_scale
       new_w == w/half_span + 1 + min_value/half_span
    */

    weights_out[idx] = w / half_span + (T)1.0 + min_value / half_span;

    if (idx < d_size) {
      scales_out[didx] = new_scale;
      biases_out[didx] = new_b;
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
    T *weights,
    T current_lr,
    const WeightRemapParameter &wrmpar,
    T *scales,
    T *biases,
    int *channel_exceeded) {
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

  case WeightRemapType::ChannelwiseExceeded: {

    if (!channel_exceeded) {
      RPU_FATAL(
          " Need channel exceeded information for ChannelwiseExceeded (BM type::SignalChannel) ");
    }

    kernelRemapScaleExceededChannel<T><<<getNBlocks(nthreads), nthreads, 0, s>>>(
        weights, scales, weights, x_size_, d_size_, channel_exceeded, wrmpar.clip_if, current_lr);

    break;
  }

  case WeightRemapType::ChannelwiseNorm: {

    // calculate norm over x_size dim, thus trans

    if (!msqr_values_) {
      msqr_values_ = RPU::make_unique<CudaArray<T>>(context_, d_size_);
    }

    if (wrmpar.row_norm <= 0) {
      RPU_FATAL("Expect row_norm >0.");
    }

    int nthreads1 = MIN(nthreads, (d_size_ + 31) / 32 * 32);
    const int n_stride = 8;
    int nblocks1 = this->context_->getNBlocks(MAX(size_ / n_stride, d_size_), nthreads1);

    msqr_values_->setConst((T)0.0); // since atomic...

    kernelRowwiseSumSqr<T>
        <<<nblocks1, nthreads1, 0, s>>>(weights, size_, d_size_, msqr_values_->getData());

    // T * w_host = new T[size_];
    kernelRemapScaleRowwiseSqrt<T><<<getNBlocks(nthreads), nthreads, 0, s>>>(
        weights, scales, weights, x_size_, d_size_, msqr_values_->getData(), wrmpar.row_norm,
        wrmpar.clip_if);

    break;
  }

  case WeightRemapType::ChannelwiseAsymmetric: {

    if (wrmpar.remapped_wmax != 1.0) {
      RPU_FATAL("Assymetric does not support remapped_wmax not equal to 1.");
    }

    if (!scales || !biases) {
      RPU_FATAL("Expect scales and biases given.");
    }
    if (!scale_buffer_) {
      scale_buffer_ = RPU::make_unique<CudaArray<T>>(context_, d_size_);
    }
    if (!bias_buffer_) {
      bias_buffer_ = RPU::make_unique<CudaArray<T>>(context_, d_size_);
    }
    RPU::math::copy(context_, d_size_, scales, 1, scale_buffer_->getData(), 1); // to avoid in-place
    RPU::math::copy(context_, d_size_, biases, 1, bias_buffer_->getData(), 1);  // to avoid in-place

    if (!row_maximizer_) {
      row_maximizer_ = RPU::make_unique<Maximizer<T>>(context_, x_size_, false);
    }
    if (!row_minimizer_) {
      row_minimizer_ = RPU::make_unique<Maximizer<T>>(context_, x_size_, false);
    }
    row_maximizer_->compute(weights, d_size_, true);
    row_minimizer_->compute(
        NegateInputIterator<T>(weights), d_size_, true); // compute max(-weights) actually

    kernelRemapScaleAndShiftRowwise<T><<<getNBlocks(nthreads), nthreads, 0, s>>>(
        weights, scales, biases, weights, scale_buffer_->getData(), bias_buffer_->getData(),
        x_size_, d_size_, row_maximizer_->getMaxValues(),
        row_minimizer_->getMaxValues() // max of negative weights, in fact
    );

    break;
  }
  case WeightRemapType::None: {
    break;
  }
  default:
    RPU_FATAL("Remapping type not implemented.");
  } // switch
}

template <typename T>
bool WeightRemapperCuda<T>::applySWA(
    T *swa_weights,
    T *weights,
    uint64_t iter,
    const WeightRemapParameter &wrmpar,
    T current_lr,
    T *scales,
    T *biases,
    int *channel_exceded) {
  // stochastic weight averaging

  int nthreads = context_->getNThreads();
  auto s = context_->getStream();
  // stochastic weight averaging
  if (wrmpar.swa_every <= 0) {
    return false;
  }

  if ((iter <= wrmpar.swa_start) || (iter % wrmpar.swa_every != 0)) {
    return false;
  }
  // we start at *end* of cycle (see <=)
  // iter is > 0 (see above)
  // std::cout << "SWA at iter " << iter << std::endl;
  // wrmpar.print();

  uint64_t n = (iter - wrmpar.swa_start) / wrmpar.swa_every - 1;
  T ratio = (T)(n) / (T)(n + 1);

  if (scales != nullptr) {
    kernelSWA<T><<<getNBlocks(nthreads), nthreads, 0, s>>>(
        swa_weights, scales, swa_weights, weights, n + 1, ratio, d_size_, size_);
  } else {
    RPU::math::elemweightedsum<T>(
        context_, swa_weights, size_, swa_weights, ratio, weights, (T)1.0 / (T)(n + 1));
  }

  if ((wrmpar.swa_transfer_every > 0) && ((n + 1) % wrmpar.swa_transfer_every == 0)) {
    std::cout << "SWA: do transfer [" << iter << "]\n";
    RPU::math::copy<T>(context_, size_, swa_weights, 1, weights, 1);

    if (scales != nullptr) {
      RPU::math::elemconst<T>(context_, scales, d_size_, (T)1.0);
      // need to re-map the weights:
      this->apply(weights, current_lr, wrmpar, scales, biases, channel_exceded);
    }
    return true;
  }
  return false;
}

template class WeightRemapperCuda<float>;
#ifdef RPU_USE_DOUBLE
template class WeightRemapperCuda<double>;
#endif
#ifdef RPU_USE_FP16
template class WeightRemapperCuda<half_t>;
#endif

#undef RPU_WM_KERNEL_LOOP
} // namespace RPU
