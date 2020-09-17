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

#include "rpu_forward_backward_pass.h"
#include "utility_functions.h"

namespace RPU {

/*FP forward / backward pass */
template <typename T>
void ForwardBackwardPass<T>::forwardVector(
    T **weights,
    const T *x_input,
    const int x_inc,
    T *d_output,
    const int d_inc,
    const T alpha,
    const bool is_test) {

  RPU::math::gemv<T>(
      CblasRowMajor, CblasNoTrans, this->d_size_, this->x_size_, alpha, weights[0], this->x_size_,
      x_input, x_inc, (T)0.0, d_output, d_inc);
}

template <typename T>
void ForwardBackwardPass<T>::backwardVector(
    T **weights, const T *d_input, const int d_inc, T *x_output, const int x_inc, const T alpha) {

  RPU::math::gemv<T>(
      CblasRowMajor, CblasTrans, this->d_size_, this->x_size_, alpha, weights[0], this->x_size_,
      d_input, d_inc, (T)0.0, x_output, x_inc);
}

template class ForwardBackwardPass<float>;
#ifdef RPU_USE_DOUBLE
template class ForwardBackwardPass<double>;
#endif

/**********************************************/
/* noise management                          */

template <typename T>
inline T computeNoiseManagement(
    const T *input,
    const int size,
    const int inc,
    const NoiseManagementType nm_type,
    const IOMetaParameter<T> &io) {
  if (nm_type == NoiseManagementType::None) {
    return 1.0;
  }

  switch (nm_type) {
  case NoiseManagementType::AbsMax: {
    int max_index = RPU::math::iamax<T>(size, input, inc);
    T amax_input_value = fabs(input[max_index * inc]);

    return io.nm_thres > 0 ? MIN(amax_input_value, io.nm_thres) : amax_input_value;
  }
  case NoiseManagementType::Constant: {
    return io.nm_thres > 0 ? (T)io.nm_thres : (T)1.0;
  }
  case NoiseManagementType::Max: {
    T max_input_value = RPU::math::max<T>(size, input, inc);
    return io.nm_thres > 0 ? MIN(max_input_value, io.nm_thres) : max_input_value;
  }
  default:
    RPU_FATAL("Noise Management type not implemented!");
  }
}

/*********************************************************************/
/* Noisy forward / backward pass with ADC/DAC and IO management*/

template <typename T> void ForwardBackwardPassIOManaged<T>::allocateContainers() {

  if (!containers_allocated_) {
    freeContainers();

    tmp_x_values_ = new T[this->x_size_]();
    tmp_d_values_ = new T[this->d_size_]();
    containers_allocated_ = true;
  }
}

template <typename T> void ForwardBackwardPassIOManaged<T>::freeContainers() {

  if (containers_allocated_) {

    delete[] tmp_d_values_;
    delete[] tmp_x_values_;

    tmp_d_values_ = nullptr;
    tmp_x_values_ = nullptr;

    containers_allocated_ = false;
  }
}

// ctor
template <typename T>
ForwardBackwardPassIOManaged<T>::ForwardBackwardPassIOManaged(
    int x_size, int d_size, std::shared_ptr<RNG<T>> rng)
    : ForwardBackwardPass<T>(x_size, d_size), rng_(rng) {
  allocateContainers();
}

// dtor
template <typename T> ForwardBackwardPassIOManaged<T>::~ForwardBackwardPassIOManaged() {
  freeContainers();
}

// copy construcutor
template <typename T>
ForwardBackwardPassIOManaged<T>::ForwardBackwardPassIOManaged(
    const ForwardBackwardPassIOManaged<T> &other)
    : ForwardBackwardPass<T>(other) {
  f_io_ = other.f_io_;
  b_io_ = other.b_io_;
  rng_ = other.rng_;

  if (other.containers_allocated_) {
    allocateContainers();
    // tmp not copied
  }
}

// copy assignment
template <typename T>
ForwardBackwardPassIOManaged<T> &
ForwardBackwardPassIOManaged<T>::operator=(const ForwardBackwardPassIOManaged<T> &other) {

  ForwardBackwardPassIOManaged<T> tmp(other);
  swap(*this, tmp);
  return *this;
}

// move constructor
template <typename T>
ForwardBackwardPassIOManaged<T>::ForwardBackwardPassIOManaged(
    ForwardBackwardPassIOManaged<T> &&other) {

  *this = std::move(other);
}

// move assignment
template <typename T>
ForwardBackwardPassIOManaged<T> &
ForwardBackwardPassIOManaged<T>::operator=(ForwardBackwardPassIOManaged<T> &&other) {

  ForwardBackwardPass<T>::operator=(std::move(other));

  f_io_ = other.f_io_;
  b_io_ = other.b_io_;

  // pointers
  tmp_d_values_ = other.tmp_d_values_;
  tmp_x_values_ = other.tmp_x_values_;

  // set pointers to null
  other.tmp_d_values_ = nullptr;
  other.tmp_x_values_ = nullptr;

  rng_ = std::move(other.rng_);

  containers_allocated_ = other.containers_allocated_;

  return *this;
}

template <typename T>
void ForwardBackwardPassIOManaged<T>::setIOPar(
    const IOMetaParameter<T> &f_io, const IOMetaParameter<T> &b_io) {
  f_io_ = f_io;
  b_io_ = b_io;

  // check the parameters
  f_io_.initializeForForward();
  b_io_.initializeForBackward();
}

template <typename T> void ForwardBackwardPassIOManaged<T>::ensureImplemented() {
  if (b_io_.w_noise_type != OutputWeightNoiseType::AdditiveConstant &&
      b_io_.w_noise_type != OutputWeightNoiseType::None) {
    RPU_FATAL("CPU version of NonIdealityType not yet implemented ");
  }

  if (f_io_.w_noise_type != OutputWeightNoiseType::AdditiveConstant &&
      f_io_.w_noise_type != OutputWeightNoiseType::None) {
    RPU_FATAL("CPU version of NonIdealityType not yet implemented ");
  }

  checked_implemented_ = true;
}

template <typename T>
void ForwardBackwardPassIOManaged<T>::forwardVector(
    T **weights,
    const T *x_input,
    const int x_inc,
    T *d_output,
    const int d_inc,
    const T alpha,
    const bool is_test) {
  if (f_io_.is_perfect) {
    // short-cut for FP
    ForwardBackwardPass<T>::forwardVector(
        weights, x_input, x_inc, d_output, d_inc, f_io_.out_scale * alpha, is_test);
    return;
  }

  T *x_value = tmp_x_values_;
  T nm_scale_value =
      computeNoiseManagement(x_input, this->x_size_, x_inc, f_io_.noise_management, f_io_);
  bool nm = f_io_.noise_management != NoiseManagementType::None;
  bool bm = f_io_.bound_management != BoundManagementType::None;

  if (nm) {

    if (nm_scale_value <= 0.0 && f_io_.inp_noise <= 0.0) {
      // short cut. output will be zero anyway
      int i_d = 0;
      PRAGMA_SIMD
      for (int i = 0; i < this->d_size_; ++i) {
        d_output[i_d] = (T)0.0;
        i_d += d_inc;
      }
      return;
    }
  }

  T out_scale = (T)1.0;
  out_scale = f_io_.out_scale * alpha;

  bool bound_test_passed = false;
  T reduction_due_to_bound_management = 0.5;
  T scale = 1.;
  bool scaling = false;

  T inp_noise = f_io_.inp_noise;
  int bm_round = 0;
  while (bound_test_passed == false) {

    bound_test_passed = true;
    reduction_due_to_bound_management *= 2.0;

    bm_round++;

    if (reduction_due_to_bound_management > 500.0) {
      std::cout << "Bound management already at " << reduction_due_to_bound_management << "\n";
    }

    scaling = false;
    scale = (T)1.0;

    if (nm && nm_scale_value > 0.) {
      scale /= nm_scale_value;
      scaling = true;
    }

    if (bm) {
      scale /= reduction_due_to_bound_management;
      scaling = true;
    }

    int j_x = 0;
    if (scaling) {
      PRAGMA_SIMD
      for (int j = 0; j < this->x_size_; ++j) {

        T value = x_input[j_x];
        j_x += x_inc;

        value *= scale;

        value = getDiscretizedValue(value, f_io_.inp_res, f_io_.inp_sto_round, *rng_);

        if (inp_noise > 0) {
          value += inp_noise * rng_->sampleGauss();
        }

        value = (value > f_io_.inp_bound) ? f_io_.inp_bound : value;
        value = (value < -f_io_.inp_bound) ? -f_io_.inp_bound : value;

        x_value[j] = value;
      }
    } else {
      PRAGMA_SIMD
      for (int j = 0; j < this->x_size_; ++j) {

        T value = x_input[j_x];
        j_x += x_inc;

        value = getDiscretizedValue(value, f_io_.inp_res, f_io_.inp_sto_round, *rng_);

        if (inp_noise > 0) {
          value += inp_noise * rng_->sampleGauss();
        }

        value = (value > f_io_.inp_bound) ? f_io_.inp_bound : value;
        value = (value < -f_io_.inp_bound) ? -f_io_.inp_bound : value;

        x_value[j] = value;
      }
    }

    if (f_io_.out_noise > 0) {
      int i_d = 0;
      PRAGMA_SIMD
      for (int i = 0; i < this->d_size_; ++i) {
        d_output[i_d] = rng_->sampleGauss();
        i_d += d_inc;
      }
    }

    RPU::math::gemv<T>(
        CblasRowMajor, CblasNoTrans, this->d_size_, this->x_size_, 1.0, weights[0], this->x_size_,
        x_value, 1, f_io_.out_noise, d_output, d_inc);

    if (f_io_.w_noise > 0 && f_io_.w_noise_type == OutputWeightNoiseType::AdditiveConstant) {
      T x_norm = RPU::math::nrm2<T>(this->x_size_, x_value, 1);
      T w_std = f_io_.w_noise * x_norm;
      int i_d = 0;
      PRAGMA_SIMD
      for (int i = 0; i < this->d_size_; ++i) {
        d_output[i_d] += w_std * rng_->sampleGauss();
        i_d += d_inc;
      }
    }

    int i_d = 0;
    PRAGMA_SIMD
    for (int i = 0; i < this->d_size_; ++i) {

      T value = d_output[i_d];

      value = getDiscretizedValue(value, f_io_.out_res, f_io_.out_sto_round, *rng_);

      if (value > f_io_.out_bound) {
        value = f_io_.out_bound;
        bound_test_passed = false;
      } else if (value < -f_io_.out_bound) {
        value = -f_io_.out_bound;
        bound_test_passed = !f_io_.bm_test_negative_bound;
      }

      d_output[i_d] = value;
      i_d += d_inc;
    }

    if (bm) {
      bound_test_passed =
          bound_test_passed || ((reduction_due_to_bound_management > f_io_.max_bm_factor) ||
                                ((f_io_.inp_res > 0) && (reduction_due_to_bound_management >
                                                         f_io_.max_bm_res / f_io_.inp_res)));
    } else {
      bound_test_passed = true;
    }
  }

  if (scaling || out_scale != 1.0) {
    RPU::math::scal<T>(this->d_size_, out_scale / scale, d_output, d_inc);
  }
};

#define CHECK_INPUT_BOUNDS                                                                         \
  value = getDiscretizedValue(value, b_io_.inp_res, b_io_.inp_sto_round, *rng_);                   \
  value = (value > b_io_.inp_bound) ? b_io_.inp_bound : value;                                     \
  value = (value < -b_io_.inp_bound) ? -b_io_.inp_bound : value;

template <typename T>
void ForwardBackwardPassIOManaged<T>::backwardVector(
    T **weights, const T *d_input, const int d_inc, T *x_output, const int x_inc, const T alpha) {

  if (b_io_.is_perfect) {
    // short-cut for FP
    ForwardBackwardPass<T>::backwardVector(
        weights, d_input, d_inc, x_output, x_inc, b_io_.out_scale * alpha);
    return;
  }

  if (!checked_implemented_) {
    ensureImplemented();
  }

  // io managed version
  T *d_value = tmp_d_values_;
  T nm_scale_value =
      computeNoiseManagement(d_input, this->d_size_, d_inc, b_io_.noise_management, b_io_);
  bool nm = b_io_.noise_management != NoiseManagementType::None;
  bool scaling = nm && nm_scale_value > 0.0;
  T out_scale = b_io_.out_scale * alpha;

  if (nm && nm_scale_value <= 0.0) {
    // max is zero. output is just zero. short-cut
    int j_x = 0;
    PRAGMA_SIMD
    for (int j = 0; j < this->x_size_; j++) {
      x_output[j_x] = (T)0.0;
      j_x += x_inc;
    }
    return;
  }

  int i_d = 0;
  if (scaling) {
    PRAGMA_SIMD
    for (int i = 0; i < this->d_size_; ++i) {
      T value = d_input[i_d];
      i_d += d_inc;
      value /= nm_scale_value;

      CHECK_INPUT_BOUNDS;

      d_value[i] = value;
    }
  } else {
    PRAGMA_SIMD
    for (int i = 0; i < this->d_size_; ++i) {
      T value = d_input[i_d];
      i_d += d_inc;

      CHECK_INPUT_BOUNDS;

      d_value[i] = value;
    }
  }

  if (b_io_.out_noise > 0) {
    int j_x = 0;
    PRAGMA_SIMD
    for (int j = 0; j < this->x_size_; j++) {
      x_output[j_x] = rng_->sampleGauss();
      j_x += x_inc;
    }
  }

  RPU::math::gemv<T>(
      CblasRowMajor, CblasTrans, this->d_size_, this->x_size_, 1.0, weights[0], this->x_size_,
      d_value, 1, b_io_.out_noise, x_output, x_inc);

  if (b_io_.w_noise > 0 && b_io_.w_noise_type == OutputWeightNoiseType::AdditiveConstant) {
    T d_norm = RPU::math::nrm2<T>(this->d_size_, d_value, 1);
    T w_std = b_io_.w_noise * d_norm;
    int j_x = 0;
    PRAGMA_SIMD
    for (int j = 0; j < this->x_size_; j++) {
      x_output[j_x] += w_std * rng_->sampleGauss();
      j_x += x_inc;
    }
  }

  int j_x = 0;
  PRAGMA_SIMD
  for (int j = 0; j < this->x_size_; j++) {

    T value = x_output[j_x];

    value = getDiscretizedValue(value, b_io_.out_res, b_io_.out_sto_round, *rng_);

    value = (value > b_io_.out_bound) ? b_io_.out_bound : value;
    value = (value < -b_io_.out_bound) ? -b_io_.out_bound : value;

    x_output[j_x] = value;
    j_x += x_inc;
  }

  if (scaling || out_scale != 1.0) {
    RPU::math::scal<T>(this->x_size_, out_scale * nm_scale_value, x_output, x_inc);
  }
};
#undef CHECK_INPUT_BOUNDS

template class ForwardBackwardPassIOManaged<float>;
#ifdef RPU_USE_DOUBLE
template class ForwardBackwardPassIOManaged<double>;
#endif

} // namespace RPU
