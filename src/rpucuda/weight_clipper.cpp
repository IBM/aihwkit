#include "weight_clipper.h"
#include "math_util.h"
#include "utility_functions.h"
#include <limits>

namespace RPU {

/***********************************************************/
// ctors

template <typename T>
WeightClipper<T>::WeightClipper(int x_size, int d_size)
    : x_size_(x_size), d_size_(d_size), size_(d_size * x_size) {}

template <typename T> void WeightClipper<T>::clip(T *weights, T value) {

  PRAGMA_SIMD
  for (int i = 0; i < size_; ++i) {
    weights[i] = MIN(MAX(weights[i], -value), value);
  }
}

template <typename T> void WeightClipper<T>::apply(T *weights, const WeightClipParameter &wclpar) {

  T clip_value = -1.0;

  switch (wclpar.type) {
  case WeightClipType::FixedValue: {
    clip_value = wclpar.fixed_value;
    break;
  }
  case WeightClipType::AverageChannelMax: {

    if (amax_values_.size() < (size_t)d_size_) {
      amax_values_.resize(d_size_);
    }
    std::fill(amax_values_.begin(), amax_values_.end(), 0.0);

    // compute max per row
    PRAGMA_SIMD
    for (int i = 0; i < size_; i++) {
      int row_idx = i / x_size_;
      T a = fabs(weights[i]);
      T mv = amax_values_[row_idx];

      amax_values_[row_idx] = a > mv ? a : mv;
    }

    T sum_amax = 0;
    PRAGMA_SIMD
    for (int i = 0; i < d_size_; i++) {
      sum_amax += amax_values_[i];
    }

    clip_value = fabs(sum_amax / d_size_);

    break;
  }
  case WeightClipType::LayerGaussian: {

    T mean_value = 0.0;
    T std_value = 0.0;

    // compute mean
    PRAGMA_SIMD
    for (int i = 0; i < size_; i++) {
      mean_value += weights[i];
    }
    mean_value /= size_;

    // compute std
    T s = size_ - 1;
    PRAGMA_SIMD
    for (int i = 0; i < size_; i++) {
      T m = weights[i] - mean_value;
      std_value += m * m / s;
    }
    std_value = sqrt(std_value);

    clip_value = std_value * wclpar.sigma;
    break;
  }

  case WeightClipType::None: {
    clip_value = -1.0;
    break;
  }
  default:
    RPU_FATAL("Clipping type not implemented for CPU.");
  } // switch

  // do the clippping
  clip(weights, clip_value);
}

template class WeightClipper<float>;
#ifdef RPU_USE_DOUBLE
template class WeightClipper<double>;
#endif

} // namespace RPU
