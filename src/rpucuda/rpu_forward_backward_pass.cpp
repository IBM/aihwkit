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
    T &auxilary_variable,
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
  case NoiseManagementType::AverageAbsMax:
  case NoiseManagementType::AverageAbsMaxSingleValue: {
    int max_index = RPU::math::iamax<T>(size, input, inc);
    T amax_input_value = fabs(input[max_index * inc]);
    if (auxilary_variable < (T)0.0) {
      auxilary_variable = amax_input_value;
    } else {
      auxilary_variable =
          auxilary_variable * ((T)1.0 - io.nm_decay) + io.nm_decay * amax_input_value;
    }
    return auxilary_variable;
  }

  case NoiseManagementType::AbsMaxNPSum: {
    int max_index = RPU::math::iamax<T>(size, input, inc);
    T amax_input_value = fabs(input[max_index * inc]);

    T psum = 0;
    T nsum = 0;
    int j_x = 0;
    PRAGMA_SIMD
    for (int j = 0; j < size; j++) {
      T x = input[j_x];
      psum += x > 0 ? x : (T)0.0;
      nsum += x < 0 ? x : (T)0.0;
      j_x += inc;
    }
    T sum = MAX(psum, -nsum);
    amax_input_value = io.nm_thres > 0 ? MIN(amax_input_value, io.nm_thres) : amax_input_value;

    T npsum_scale = sum * io.nm_assumed_wmax / io.out_bound;
    if (io.inp_res > 0) {
      npsum_scale = MIN(amax_input_value / io.inp_res * io.max_bm_res, npsum_scale);
    }
    return MAX(amax_input_value, npsum_scale);
  }
  default:
    RPU_FATAL("Noise Management type not implemented!");
  }
}

/*********************************************************************/
/* Noisy forward / backward pass with ADC/DAC and IO management*/

// ctor
template <typename T>
ForwardBackwardPassIOManaged<T>::ForwardBackwardPassIOManaged(
    int x_size, int d_size, std::shared_ptr<RNG<T>> rng)
    : ForwardBackwardPass<T>(x_size, d_size) {
  rng_ = rng;
  aux_nm_value_ = (T)-1.0;
}

template <typename T>
void ForwardBackwardPassIOManaged<T>::setIOPar(
    const IOMetaParameter<T> &f_io, const IOMetaParameter<T> &b_io) {
  f_io_ = f_io;
  b_io_ = b_io;

  // check the parameters
  f_io_.initializeForForward();
  b_io_.initializeForBackward();
  checked_implemented_ = false; // need to check in forward because CUDA also shares this with CPU
}

template <typename T> void ForwardBackwardPassIOManaged<T>::ensureImplemented() {}

template <typename T>
void ForwardBackwardPassIOManaged<T>::applyOutputWeightNoise(
    T **weights,
    T *out_values,
    const int out_size,
    const int out_inc,
    const T *in_values,
    const int in_size,
    IOMetaParameter<T> &io,
    bool transposed) {

  if (io.w_noise_type == OutputWeightNoiseType::None) {
    return;
  }

  switch (io.w_noise_type) {
  case OutputWeightNoiseType::AdditiveConstant:
    if (io.w_noise > 0) {
      T x_norm = RPU::math::nrm2<T>(in_size, in_values, 1);
      T w_std = io.w_noise * x_norm;
      int i_out = 0;
      PRAGMA_SIMD
      for (int i = 0; i < out_size; ++i) {
        out_values[i_out] += w_std * rng_->sampleGauss();
        i_out += out_inc;
      }
    }
    break;
  case OutputWeightNoiseType::PCMRead:
    if (io.w_noise > 0) {
      T w_std = io.w_noise;
      tmp_in_values_.resize(in_size);

      PRAGMA_SIMD
      for (int j = 0; j < in_size; ++j) {
        tmp_in_values_[j] = in_values[j] * in_values[j];
      }
      // likely realtively slow. Since |W|*x.^2 without GEMV...
      int i_out = 0;
      for (int i = 0; i < out_size; ++i) {
        T accum = 0.0;
        if (transposed) {
          PRAGMA_SIMD
          for (int j = 0; j < in_size; ++j) {
            accum += fabs(weights[j][i]) * tmp_in_values_[j];
          }
        } else {
          PRAGMA_SIMD
          for (int j = 0; j < in_size; ++j) {
            accum += fabs(weights[i][j]) * tmp_in_values_[j];
          }
        }
        out_values[i_out] += w_std * sqrtf(accum) * rng_->sampleGauss();
        i_out += out_inc;
      }
    }
    break;
  default:
    RPU_FATAL("Output noise type not implemented")
  }
}

template <typename T>
void ForwardBackwardPassIOManaged<T>::applyIrDrop(
    T **weights,
    T *out_values,
    int out_size,
    const int out_inc,
    const T *in_values, // inc = 1 expected
    const int in_size,
    IOMetaParameter<T> &io,
    bool transposed) {

  if (io.ir_drop <= 0.0) {
    return;
  }
  tmp_in_values_.resize(in_size);
  tmp_c_values_.resize(out_size);
  tmp_out_values_.resize(out_size);

  T a_scale = in_size / io.ir_drop_Gw_div_gmax;
  // a_i = sum_j(|w_ij|*|x_j|)*n/Gw*gmax
  for (int i = 0; i < out_size; ++i) {
    T accum = 0.0;
    if (transposed) {
      PRAGMA_SIMD
      for (int j = 0; j < in_size; ++j) {
        accum += fabs(weights[j][i]) * fabs(in_values[j]);
      }
    } else {
      PRAGMA_SIMD
      for (int j = 0; j < in_size; ++j) {
        accum += fabs(weights[i][j]) * fabs(in_values[j]);
      }
    }
    T a = a_scale * accum;
    // c_i = a_i*(a_i*(0.05*a_i - 0.2) + 0.5);
    tmp_c_values_[i] = a * (a * ((T)0.05 * a - (T)0.2) + (T)0.5);
  }

  // compute x_j*(1-(1-j/n)^2)
  PRAGMA_SIMD
  for (int j = 0; j < in_size; ++j) {
    T p = ((T)1 - (T)j / in_size);
    tmp_in_values_[j] = in_values[j] * (1 - p * p);
  }

  // y_i = y_i_ideal - ir_drop*c_i*sum_j(w_ij * x'_j)
  RPU::math::gemv<T>(
      CblasRowMajor, transposed ? CblasTrans : CblasNoTrans, this->d_size_, this->x_size_,
      io.ir_drop, weights[0], this->x_size_, tmp_in_values_.data(), 1, (T)0.0,
      tmp_out_values_.data(), 1);

  int i_out = 0;
  PRAGMA_SIMD
  for (int i = 0; i < out_size; ++i) {
    out_values[i_out] -= tmp_c_values_[i] * tmp_out_values_[i];
    i_out += out_inc;
  }
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
  if (!checked_implemented_) {
    ensureImplemented();
    checked_implemented_ = true;
  }
  tmp_x_values_.resize(this->x_size_);

  T *x_value = tmp_x_values_.data();
  T nm_scale_value = computeNoiseManagement(
      x_input, this->x_size_, x_inc, f_io_.noise_management, aux_nm_value_, f_io_);
  bool nm = f_io_.noise_management != NoiseManagementType::None;
  bool sm = f_io_.bound_management == BoundManagementType::Shift;
  bool bm = f_io_.bound_management != BoundManagementType::None && !sm;

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
  if (!sm) { // NOT scaled for ShiftManagement (ONLY during forward...)
    out_scale = f_io_.out_scale * alpha;
  }

  bool bound_test_passed = false;
  T reduction_due_to_bound_management = 0.5;
  T scale = 1.;
  bool scaling = false;

  T inp_noise = f_io_.inp_noise;
  int bm_round = 0;
  while (bound_test_passed == false) {

    bound_test_passed = true;
    reduction_due_to_bound_management *= 2.0;

    if (bm_round == 1 && f_io_.bound_management == BoundManagementType::IterativeWorstCase &&
        f_io_.noise_management != NoiseManagementType::AbsMaxNPSum) {
      nm_scale_value = computeNoiseManagement(
          x_input, this->x_size_, x_inc, NoiseManagementType::AbsMaxNPSum, aux_nm_value_, f_io_);
      reduction_due_to_bound_management = 1.0; // reset to 1.0
    }

    bm_round++;

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

    if (f_io_.w_noise_type != OutputWeightNoiseType::None) {
      applyOutputWeightNoise(
          weights, d_output, this->d_size_, d_inc, x_value, this->x_size_, f_io_, false);
    }

    if (f_io_.ir_drop != (T)0) {
      applyIrDrop(weights, d_output, this->d_size_, d_inc, x_value, this->x_size_, f_io_, false);
    }

    if (sm) {
      // this is for softmax specialization. We use the range from
      // the max and clip below, ie shift the max to the pos range
      // bound (out_bound (NEED TO BE SET TO FINITE VALUE FOR THIS))

      int max_index = RPU::math::iamax<T>(this->d_size_, d_output, d_inc);
      T max_output_value = fabs(d_output[max_index * d_inc]);
      T shift_value = f_io_.out_bound - max_output_value;
      int i_d1 = 0;
      PRAGMA_SIMD
      for (int i = 0; i < this->d_size_; ++i) {
        d_output[i_d1] += shift_value; // will be clipped below
        i_d1 += d_inc;
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

    if (bm && !sm) {
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

  // io managed version
  tmp_d_values_.resize(this->d_size_);

  T *d_value = tmp_d_values_.data();
  T nm_scale_value = computeNoiseManagement(
      d_input, this->d_size_, d_inc, b_io_.noise_management, aux_nm_value_, b_io_);
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

  if (b_io_.w_noise_type != OutputWeightNoiseType::None) {
    applyOutputWeightNoise(
        weights, x_output, this->x_size_, x_inc, d_value, this->d_size_, b_io_, true);
  }

  if (b_io_.ir_drop != (T)0) {
    applyIrDrop(weights, x_output, this->x_size_, x_inc, d_value, this->d_size_, b_io_, true);
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
