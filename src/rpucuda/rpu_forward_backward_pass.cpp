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

  UNUSED(is_test);
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

template <typename T>
void ForwardBackwardPass<T>::gemv(
    T **weights,
    const T *in_values,
    const int in_size,
    const int in_inc,
    T *out_values,
    const int out_size,
    const int out_inc,
    const T alpha,
    const T beta,
    const bool transposed) {
  // y = alpha * W * x + beta * y

  if (transposed) { // backward
    RPU::math::gemv<T>(
        CblasRowMajor, CblasTrans, in_size, out_size, alpha, weights[0], out_size, in_values,
        in_inc, beta, out_values, out_inc);
  } else {
    RPU::math::gemv<T>(
        CblasRowMajor, CblasNoTrans, out_size, in_size, alpha, weights[0], in_size, in_values,
        in_inc, beta, out_values, out_inc);
  }
}

template <typename T>
void ForwardBackwardPass<T>::dumpExtra(RPU::state_t &extra, const std::string prefix) {

  RPU::state_t state;

  RPU::insert(state, "fwd.v_offset", fb_pars_.fwd.v_offset);
  RPU::insert(state, "fwd.w_asymmetry", fb_pars_.fwd.w_asymmetry);
  RPU::insert(state, "fwd.out_nonlinearity", fb_pars_.fwd.out_nonlinearity);
  RPU::insert(state, "fwd.out_nonlinearity_factor", fb_pars_.fwd.out_nonlinearity_factor);
  RPU::insert(state, "fwd.out_noise_values", fb_pars_.fwd.out_noise_values);

  RPU::insert(state, "bwd.v_offset", fb_pars_.bwd.v_offset);
  RPU::insert(state, "bwd.w_asymmetry", fb_pars_.bwd.w_asymmetry);
  RPU::insert(state, "bwd.out_nonlinearity", fb_pars_.bwd.out_nonlinearity);
  RPU::insert(state, "bwd.out_nonlinearity_factor", fb_pars_.bwd.out_nonlinearity_factor);
  RPU::insert(state, "bwd.out_noise_values", fb_pars_.bwd.out_noise_values);

  RPU::insertWithPrefix(extra, state, prefix);
}

template <typename T>
void ForwardBackwardPass<T>::loadExtra(
    const RPU::state_t &extra, const std::string prefix, bool strict) {

  auto state = RPU::selectWithPrefix(extra, prefix);
  RPU::load(state, "fwd.v_offset", fb_pars_.fwd.v_offset, strict);
  RPU::load(state, "fwd.w_asymmetry", fb_pars_.fwd.w_asymmetry, strict);
  RPU::load(state, "fwd.out_nonlinearity", fb_pars_.fwd.out_nonlinearity, strict);
  RPU::load(state, "fwd.out_nonlinearity_factor", fb_pars_.fwd.out_nonlinearity_factor, strict);
  RPU::load(state, "fwd.out_noise_values", fb_pars_.fwd.out_noise_values, strict);

  RPU::load(state, "bwd.v_offset", fb_pars_.bwd.v_offset, strict);
  RPU::load(state, "bwd.w_asymmetry", fb_pars_.bwd.w_asymmetry, strict);
  RPU::load(state, "bwd.out_nonlinearity", fb_pars_.bwd.out_nonlinearity, strict);
  RPU::load(state, "bwd.out_nonlinearity_factor", fb_pars_.bwd.out_nonlinearity_factor, strict);
  RPU::load(state, "bwd.out_noise_values", fb_pars_.bwd.out_noise_values, strict);
}

template class ForwardBackwardPass<float>;
#ifdef RPU_USE_DOUBLE
template class ForwardBackwardPass<double>;
#endif
#ifdef RPU_USE_FP16
template class ForwardBackwardPass<half_t>;
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
    T amax_input_value = (T)fabsf(input[max_index * inc]);

    return io.nm_thres > (T)0.0 ? MIN(amax_input_value, io.nm_thres) : amax_input_value;
  }
  case NoiseManagementType::Constant: {
    return io.nm_thres > (T)0.0 ? (T)io.nm_thres : (T)1.0;
  }
  case NoiseManagementType::Max: {
    T max_input_value = RPU::math::max<T>(size, input, inc);
    return io.nm_thres > (T)0.0 ? MIN(max_input_value, io.nm_thres) : max_input_value;
  }
  case NoiseManagementType::AverageAbsMax:
  case NoiseManagementType::AverageAbsMaxSingleValue: {
    int max_index = RPU::math::iamax<T>(size, input, inc);
    T amax_input_value = (T)fabsf(input[max_index * inc]);
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
    T amax_input_value = (T)fabsf(input[max_index * inc]);

    T bound = isinf((float)io.out_bound) ? (T)1.0 : io.out_bound;
    T psum = 0;
    T nsum = 0;
    int j_x = 0;
    PRAGMA_SIMD
    for (int j = 0; j < size; j++) {
      T x = input[j_x];
      psum += x > (T)0.0 ? x : (T)0.0;
      nsum += x < (T)0.0 ? x : (T)0.0;
      j_x += inc;
    }
    T sum = MAX(psum, -nsum);
    amax_input_value = io.nm_thres > (T)0.0 ? MIN(amax_input_value, io.nm_thres) : amax_input_value;

    T npsum_scale = sum * io.nm_assumed_wmax / bound;
    if (io.inp_res > (T)0.0) {
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

// dtor
template <typename T> ForwardBackwardPassIOManaged<T>::~ForwardBackwardPassIOManaged() {
  if (neg_weights_ != nullptr) {
    Array_2D_Free<T>(neg_weights_);
  }
}

// copy construcutor
template <typename T>
ForwardBackwardPassIOManaged<T>::ForwardBackwardPassIOManaged(
    const ForwardBackwardPassIOManaged<T> &other)
    : ForwardBackwardPass<T>(other) {

  f_io_ = other.f_io_;
  b_io_ = other.b_io_;
  checked_implemented_ = other.checked_implemented_;
  rng_ = other.rng_;
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
    ForwardBackwardPassIOManaged<T> &&other) noexcept {
  *this = std::move(other);
}

// move assignment
template <typename T>
ForwardBackwardPassIOManaged<T> &
ForwardBackwardPassIOManaged<T>::operator=(ForwardBackwardPassIOManaged<T> &&other) noexcept {

  ForwardBackwardPass<T>::operator=(std::move(other));

  f_io_ = other.f_io_;
  b_io_ = other.b_io_;
  checked_implemented_ = other.checked_implemented_;
  rng_ = std::move(other.rng_);

  return *this;
}

template <typename T>
void ForwardBackwardPassIOManaged<T>::populateFBParameter(
    const IOMetaParameter<T> &f_io, const IOMetaParameter<T> &b_io) {
  f_io_ = f_io;
  b_io_ = b_io;

  // check the parameters
  f_io_.initializeForForward(this->x_size_, this->d_size_);
  b_io_.initializeForBackward(this->x_size_, this->d_size_);
  checked_implemented_ = false;

  // v offset forward
  auto populate = [this](
                      IOMetaParameter<T> &io, MVParameter<T> &mv_pars, size_t in_size,
                      size_t out_size) -> void {
    mv_pars.v_offset = io.v_offset_vec;
    if ((mv_pars.v_offset.size() != out_size) && io.hasVoltageOffsets()) {
      mv_pars.v_offset.resize(out_size);
      for (size_t i = 0; i < out_size; i++) {
        mv_pars.v_offset[i] = MAX(io.v_offset_std * rng_->sampleGauss(), (T)0.0);
      }
    }

    mv_pars.out_nonlinearity = io.out_nonlinearity_vec;
    mv_pars.out_nonlinearity_factor = (T)1.0;
    T out_bound = io.out_bound > (T)0.0 && io.out_bound < std::numeric_limits<T>::infinity()
                      ? io.out_bound
                      : (T)1.0;

    if (io.hasNLCalibration()) {
      if (mv_pars.out_nonlinearity.size() != out_size) {
        mv_pars.out_nonlinearity.resize(out_size);

        for (size_t i = 0; i < out_size; i++) {
          mv_pars.out_nonlinearity[i] = (T)fabsf(
              io.out_nonlinearity / out_bound *
              ((T)1.0 + MAX(io.out_nonlinearity_std, (T)0.0) * rng_->sampleGauss()));
        }
      }
    }
    if (io.slope_calibration > (T)0) {
      T sum_nonlinearity = (T)0.0;
      if (io.hasOutNonlinearity()) {
        for (size_t i = 0; i < out_size; i++) {
          sum_nonlinearity += (T)fabsf(mv_pars.out_nonlinearity[i]);
        }
      }
      // T r_correction  = (1 + io.v_offset_w_min*io.v_offset_w_min) * io.r_series;
      T r_correction = io.r_series;
      T f = (T)1.0 +
            (sum_nonlinearity / (T)out_size + r_correction) * io.slope_calibration * out_bound;
      mv_pars.out_nonlinearity_factor = f * f;
    }

    if (io.w_read_asymmetry_dtod) {
      size_t size = out_size * in_size;
      mv_pars.w_asymmetry.resize(size);

      for (size_t i = 0; i < size; i++) {
        mv_pars.w_asymmetry[i] = ((T)1.0 + io.w_read_asymmetry_dtod * rng_->sampleGauss());
      }
    }

    if (io.out_noise_std > (T)0.0) {
      mv_pars.out_noise_values.resize(out_size);

      for (size_t i = 0; i < out_size; i++) {
        mv_pars.out_noise_values[i] =
            (T)fabsf(io.out_noise * ((T)1.0 + io.out_noise_std * rng_->sampleGauss()));
      }
    }
  };

  populate(f_io_, this->fb_pars_.fwd, this->x_size_, this->d_size_);
  populate(b_io_, this->fb_pars_.bwd, this->d_size_, this->x_size_);
}

template <typename T> void ForwardBackwardPassIOManaged<T>::ensureImplemented() {}

/*********************************************************************/
/* Non-idealities */

template <typename T>
void ForwardBackwardPassIOManaged<T>::applyOutputWeightNoise(
    T **weights,
    T *out_values,
    const int out_size,
    const int out_inc,
    const T *in_values,
    const int in_size,
    const IOMetaParameter<T> &io,
    const bool transposed) {

  if (io.w_noise_type == OutputWeightNoiseType::None) {
    return;
  }

  switch (io.w_noise_type) {
  case OutputWeightNoiseType::AdditiveConstant:
    if (io.w_noise > (T)0.0) {
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
    if (io.w_noise > (T)0.0) {
      T w_std = io.w_noise;
      tmp_in_values_.resize(in_size);

      PRAGMA_SIMD
      for (int j = 0; j < in_size; ++j) {
        tmp_in_values_[j] = in_values[j] * in_values[j];
      }
      // likely relatively slow. Since |W|*x.^2 without GEMV...
      int i_out = 0;
      for (int i = 0; i < out_size; ++i) {
        T accum = 0.0;
        if (transposed) {
          PRAGMA_SIMD
          for (int j = 0; j < in_size; ++j) {
            accum += (T)fabsf(weights[j][i]) * tmp_in_values_[j];
          }
        } else {
          PRAGMA_SIMD
          for (int j = 0; j < in_size; ++j) {
            accum += (T)fabsf(weights[i][j]) * tmp_in_values_[j];
          }
        }
        out_values[i_out] += (T)w_std * (T)sqrtf(accum) * rng_->sampleGauss();
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
    const int out_size,
    const int out_inc,
    const T *in_values, // inc = 1 expected
    const T *current_values,
    const int in_size,
    const IOMetaParameter<T> &io,
    const bool transposed) {

  if (io.ir_drop <= (T)0.0) {
    return;
  }
  tmp_in_values_.resize(in_size);
  tmp_c_values_.resize(out_size);
  tmp_out_values_.resize(out_size);

  T a_scale = (T)in_size / io.ir_drop_Gw_div_gmax;
  // a_i = sum_j(|w_ij|*|x_j|)*n/Gw*gmax
  for (int i = 0; i < out_size; ++i) {
    T a = a_scale * current_values[i];
    // c_i = a_i*(a_i*(0.05*a_i - 0.2) + 0.5);
    tmp_c_values_[i] = a * (a * ((T)0.05 * a - (T)0.2) + (T)0.5);
  }

  // compute x_j*(1-(1-j/n)^2)
  PRAGMA_SIMD
  for (int j = 0; j < in_size; ++j) {
    T p = ((T)1 - (T)j / (T)in_size);
    tmp_in_values_[j] = in_values[j] * ((T)1.0 - p * p);
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
void ForwardBackwardPassIOManaged<T>::applyVoltageOffsets(
    T *out_values,
    const int out_size,
    const int out_inc,
    const T *in_values,
    const int in_size,
    const MVParameter<T> &mv_pars,
    const IOMetaParameter<T> &io) {

  T rs = MAX(io.r_series, (T)0.0);
  T rs_max_total = io.r_series_max_total;

  if (io.v_offset_w_min == (T)0.0) {

    int i_out = 0;
    if (rs <= (T)0.0) {
      // (1 - Vo_i) * y_i
      PRAGMA_SIMD
      for (int i = 0; i < out_size; ++i) {
        T y = out_values[i_out];
        T v_offs = mv_pars.v_offset[i];
        out_values[i_out] = y * ((T)1.0 - v_offs);
        i_out += out_inc;
      }
    } else {
      // (1 - Vo_i) * y_i / (1 + r_i * |y_i|)
      PRAGMA_SIMD
      for (int i = 0; i < out_size; ++i) {
        T y = out_values[i_out];
        T v_offs = mv_pars.v_offset[i];
        T nom = ((T)1.0 + MIN(rs * (T)fabsf(y), rs_max_total));
        out_values[i_out] = ((T)1.0 - v_offs) * y / nom;
        i_out += out_inc;
      }
    }
  } else {
    T x_accum = (T)0.0;

    PRAGMA_SIMD
    for (int j = 0; j < in_size; ++j) {
      x_accum += in_values[j];
    }

    int i_out = 0;
    if (rs <= (T)0.0) {
      // (1 - Vo_i) * y_i - 2 * Vo_i * g_ref \sum_j x_j
      T y_ref2 = (T)(-io.v_offset_w_min) * x_accum * (T)2.0;
      PRAGMA_SIMD
      for (int i = 0; i < out_size; ++i) {
        T y = out_values[i_out];
        T v_offs = mv_pars.v_offset[i];
        out_values[i_out] = y * ((T)1.0 - v_offs) - v_offs * y_ref2;
        i_out += out_inc;
      }
    } else {
      // (1 - Vo_i) * (y_i + y_ref) / (1 + rs*(y_i + y_ref)) - (1 +
      // Vo_i) * y_ref \sum_j x_j / (1 + rs * y_ref))
      T y_ref = (-io.v_offset_w_min) * x_accum;
      PRAGMA_SIMD
      for (int i = 0; i < out_size; ++i) {
        T y = out_values[i_out];
        T v_offs = mv_pars.v_offset[i];
        T y_pos = y + y_ref;
        T p_nom = ((T)1.0 + MIN(rs * (T)fabsf(y_pos), rs_max_total));
        T n_nom = ((T)1.0 + MIN(rs * (T)fabsf(y_ref), rs_max_total));
        out_values[i_out] = ((T)1.0 - v_offs) * y_pos / p_nom - ((T)1.0 + v_offs) * y_ref / n_nom;
        i_out += out_inc;
      }
    }
  }
}

template <typename T>
const T *ForwardBackwardPassIOManaged<T>::computeTotalCurrent(
    T **weights, const int out_size, const T *in_values, const int in_size, bool transposed) {

  current_buffer_values_.resize(out_size);
  for (int i = 0; i < out_size; ++i) {
    T accum = 0.0;
    if (transposed) {
      PRAGMA_SIMD
      for (int j = 0; j < in_size; ++j) {
        accum += (T)fabsf(weights[j][i]) * (T)fabsf(in_values[j]);
      }
    } else {
      PRAGMA_SIMD
      for (int j = 0; j < in_size; ++j) {
        accum += (T)fabsf(weights[i][j]) * (T)fabsf(in_values[j]);
      }
    }
    current_buffer_values_[i] = accum;
  }

  return current_buffer_values_.data();
}

template <typename T>
void ForwardBackwardPassIOManaged<T>::applyNonIdealities(
    T **weights,
    T *out_values,
    const int out_size,
    const int out_inc,
    const T *in_values, // inc = 1 expected
    const int in_size,
    const MVParameter<T> &mv_pars,
    const IOMetaParameter<T> &io,
    const bool transposed) {

  // IR drop
  if (io.ir_drop > (T)0) {
    auto current = computeTotalCurrent(weights, out_size, in_values, in_size, transposed);
    applyIrDrop(
        weights, out_values, out_size, out_inc, in_values, current, in_size, io, transposed);
  }
  // voltage offsets
  if (io.hasVoltageOffsets()) {
    applyVoltageOffsets(out_values, out_size, out_inc, in_values, in_size, mv_pars, io);
  }

  // weight dependent noise
  if (io.w_noise_type != OutputWeightNoiseType::None) {
    applyOutputWeightNoise(
        weights, out_values, out_size, out_inc, in_values, in_size, io, transposed);
  }
}

/*********************************************************************/
/* prepare input */

#define ARGS                                                                                       \
  T *out_values, const T *in_values, const int in_size, const int in_inc, const T scale,           \
      const IOMetaParameter<T> &io, std::shared_ptr<RNG<T>> &rng

#define ARGS_CALL out_values, in_values, in_size, in_inc, scale, io, rng

template <typename T, bool scaling, bool with_noise, bool sto_round_if, bool with_asymmetry>
inline void prepareInputImplStage4(ARGS) {

  T bound = io.inp_bound > (T)0.0 ? io.inp_bound : std::numeric_limits<T>::infinity();
  T noise = io.inp_noise;
  T asymmetry_scale = ((T)1.0 - io.inp_asymmetry);
  T res = io.inp_res;

  int j_idx = 0;
  PRAGMA_SIMD
  for (int j = 0; j < in_size; ++j) {

    T value = in_values[j_idx];
    j_idx += in_inc;
    if (scaling) {
      value *= scale;
    }
    value = getDiscretizedValueSR<sto_round_if>(value, res, *rng);

    value = (value > bound) ? bound : value;
    value = (value < -bound) ? -bound : value;

    // inp noise after the bound + DAC ?!
    if (noise > (T)0.0) {
      value += noise * rng->sampleGauss();
    }

    if (with_asymmetry) {
      value = value < (T)0 ? value * asymmetry_scale : value;
    }

    out_values[j] = value;
  }
}

template <typename T, bool scaling, bool with_noise, bool sto_round_if>
inline void prepareInputImplStage3(ARGS) {

  if (io.inp_asymmetry != (T)0.0) {
    prepareInputImplStage4<T, scaling, with_noise, sto_round_if, true>(ARGS_CALL);
  } else {
    prepareInputImplStage4<T, scaling, with_noise, sto_round_if, false>(ARGS_CALL);
  }
}

template <typename T, bool scaling, bool with_noise> inline void prepareInputImplStage2(ARGS) {

  if (io.inp_sto_round) {
    prepareInputImplStage3<T, scaling, with_noise, true>(ARGS_CALL);
  } else {
    prepareInputImplStage3<T, scaling, with_noise, false>(ARGS_CALL);
  }
}

template <typename T, bool scaling> inline void prepareInputImplStage1(ARGS) {
  if (io.inp_noise > (T)0.0) {
    prepareInputImplStage2<T, scaling, true>(ARGS_CALL);
  } else {
    prepareInputImplStage2<T, scaling, false>(ARGS_CALL);
  }
}
#undef ARGS
#undef ARGS_CALL

template <typename T>
T *ForwardBackwardPassIOManaged<T>::prepareInput(
    const T *in_values,
    const int in_size,
    const int in_inc,
    const T scale,
    const bool scaling,
    const IOMetaParameter<T> &io) {

  in_buffer_values_.resize(in_size);

  if (scaling) {
    prepareInputImplStage1<T, true>(
        in_buffer_values_.data(), in_values, in_size, in_inc, scale, io, rng_);
  } else {
    prepareInputImplStage1<T, false>(
        in_buffer_values_.data(), in_values, in_size, in_inc, scale, io, rng_);
  }

  return in_buffer_values_.data();
}

/*********************************************************************/
/* finalize output */

#define ARGS                                                                                       \
  T *out_values, const int out_size, const int out_inc, const MVParameter<T> &mv_pars,             \
      const IOMetaParameter<T> &io, std::shared_ptr<RNG<T>> &rng

#define ARGS_CALL out_values, out_size, out_inc, mv_pars, io, rng

template <
    typename T,
    bool with_noise,
    bool with_asymmetry,
    bool with_bm,
    bool sto_round_if,
    bool with_nonlinearity>
inline bool finalizeOutputImplStage5(ARGS) {
  int idx = 0;
  bool bound_test_passed = true;
  const T bound = io.out_bound > (T)0.0 ? io.out_bound : std::numeric_limits<T>::infinity();
  const T asymmetry_scale = ((T)1.0 - io.out_asymmetry);
  const T res = io.out_res;
  const T nlf = mv_pars.out_nonlinearity_factor;

  PRAGMA_SIMD
  for (int i = 0; i < out_size; ++i) {

    T value = out_values[idx];

    if (with_nonlinearity) {
      value = nlf * value / ((T)1.0 + (T)fabsf(mv_pars.out_nonlinearity[i] * value));
    }

    if (with_asymmetry) {
      // after NL (because experimental feature anyway and easier for CUDA)
      value = value < (T)0 ? value * asymmetry_scale : value;
    }

    if (with_noise) {
      const T noise_std = io.out_noise_std > (T)0.0 ? mv_pars.out_noise_values[i] : io.out_noise;
      value += noise_std * rng->sampleGauss();
    }

    value = getDiscretizedValueSR<sto_round_if>(value, res, *rng);

    if (with_bm) {
      if (value > bound) {
        value = bound;
        bound_test_passed = false;
      } else if (value < -bound) {
        value = -bound;
        bound_test_passed = !io.bm_test_negative_bound;
      }
    } else {
      value = value > bound ? bound : value;
      value = value < -bound ? -bound : value;
    }

    out_values[idx] = value;
    idx += out_inc;
  }
  return bound_test_passed;
}

template <typename T, bool with_noise, bool with_asymmetry, bool with_bm, bool sto_round_if>
inline bool finalizeOutputImplStage4(ARGS) {
  if (io.hasNLCalibration()) {
    return finalizeOutputImplStage5<T, with_noise, with_asymmetry, with_bm, sto_round_if, true>(
        ARGS_CALL);
  } else {
    return finalizeOutputImplStage5<T, with_noise, with_asymmetry, with_bm, sto_round_if, false>(
        ARGS_CALL);
  }
}

template <typename T, bool with_noise, bool with_asymmetry, bool with_bm>
inline bool finalizeOutputImplStage3(ARGS) {
  if (io.out_sto_round) {
    return finalizeOutputImplStage4<T, with_noise, with_asymmetry, with_bm, true>(ARGS_CALL);
  } else {
    return finalizeOutputImplStage4<T, with_noise, with_asymmetry, with_bm, false>(ARGS_CALL);
  }
}

template <typename T, bool with_noise, bool with_asymmetry>
inline bool finalizeOutputImplStage2(ARGS) {
  if (io.bound_management != BoundManagementType::None) {
    return finalizeOutputImplStage3<T, with_noise, with_asymmetry, true>(ARGS_CALL);
  } else {
    return finalizeOutputImplStage3<T, with_noise, with_asymmetry, false>(ARGS_CALL);
  }
}

template <typename T, bool with_asymmetry> inline bool finalizeOutputImplStage1(ARGS) {
  if (io.out_noise > (T)0.0 || io.out_noise_std > (T)0.0) {
    return finalizeOutputImplStage2<T, true, with_asymmetry>(ARGS_CALL);
  } else {
    return finalizeOutputImplStage2<T, true, with_asymmetry>(ARGS_CALL);
  }
}

#undef ARGS
#undef ARGS_CALL

template <typename T>
bool ForwardBackwardPassIOManaged<T>::finalizeOutput(
    T *out_values,
    const int out_size,
    const int out_inc,
    const MVParameter<T> &mv_pars,
    const IOMetaParameter<T> &io) {

  if (io.out_asymmetry > (T)0.0) {
    return finalizeOutputImplStage1<T, true>(out_values, out_size, out_inc, mv_pars, io, rng_);
  } else {
    return finalizeOutputImplStage1<T, false>(out_values, out_size, out_inc, mv_pars, io, rng_);
  }
}

/********************************************************************************/
/*  analog MAC */

template <typename T>
inline void ForwardBackwardPassIOManaged<T>::computeAnalogMVSinglePass(
    T **weights,
    const T *in_values,
    const int in_size,
    const int in_inc,
    T *out_values,
    const int out_size,
    const int out_inc,
    const T alpha,
    const T beta,
    const MVParameter<T> &mv_pars,
    const IOMetaParameter<T> &io,
    const bool transposed) {
  ForwardBackwardPass<T>::gemv(
      weights, in_values, in_size, in_inc, out_values, out_size, out_inc, alpha, beta, transposed);

  applyNonIdealities(
      weights, out_values, out_size, out_inc, in_values, in_size, mv_pars, io, transposed);
}

template <typename T>
inline bool ForwardBackwardPassIOManaged<T>::computeAnalogMV(
    T **weights,
    const T *org_in_values,
    const int in_size,
    const int in_inc,
    T *out_values,
    const int out_size,
    const int out_inc,
    const T scale,
    const bool scaling,
    const MVParameter<T> &mv_pars,
    const IOMetaParameter<T> &io,
    const bool transposed,
    const bool is_test) {

  // not used. We don't distinguish between evaluation and
  // training. Noise will be always present
  UNUSED(is_test);

  // scale, apply bound, discretize and scale and input noise
  T *in_values = prepareInput(org_in_values, in_size, in_inc, scale, scaling, io);
  switch (io.mv_type) {
  case AnalogMVType::OnePass: {
    // this is the standard one pass MV
    computeAnalogMVSinglePass(
        weights, in_values, in_size, 1, out_values, out_size, out_inc, 1.0, 0.0, mv_pars, io,
        transposed);
    return finalizeOutput(out_values, out_size, out_inc, mv_pars, io);
  }

  case AnalogMVType::PosNegSeparateDigitalSum:
  case AnalogMVType::PosNegSeparate: {
    pos_neg_buffer_values_.resize(in_size);
    out_buffer_values_.resize(out_size);

    // note: input noise is applied already above... ignore
    // first pass negative
    PRAGMA_SIMD
    for (int i = 0; i < in_size; ++i) {
      pos_neg_buffer_values_[i] = in_values[i] < (T)0 ? in_values[i] : (T)0.0;
    }

    // this will be extremely ineffecient...
    T **neg_weights = weights;
    if (io.w_read_asymmetry_dtod > (T)0.0) {
      if (neg_weights_ == nullptr) {
        neg_weights_ = Array_2D_Get<T>(this->d_size_, this->x_size_);
      }

      neg_weights = neg_weights_;
      for (int i = 0; i < this->d_size_ * this->x_size_; ++i) {
        neg_weights[0][i] = weights[0][i] * mv_pars.w_asymmetry[i];
      }
    }
    bool bound_success = false;

    computeAnalogMVSinglePass(
        neg_weights, pos_neg_buffer_values_.data(), in_size, 1, out_buffer_values_.data(), out_size,
        1, 1.0, 0.0, mv_pars, io, transposed);

    if (io.mv_type == AnalogMVType::PosNegSeparateDigitalSum) {
      bound_success = finalizeOutput(out_buffer_values_.data(), out_size, 1, mv_pars, io);
    }

    // second pass for positive, added to negative
    PRAGMA_SIMD
    for (int i = 0; i < in_size; ++i) {
      pos_neg_buffer_values_[i] = in_values[i] > (T)0 ? in_values[i] : (T)0.0;
    }

    computeAnalogMVSinglePass(
        weights, pos_neg_buffer_values_.data(), in_size, 1, out_values, out_size, out_inc, 1.0, 0.0,
        mv_pars, io, transposed);

    if (io.mv_type == AnalogMVType::PosNegSeparateDigitalSum) {
      bound_success = finalizeOutput(out_values, out_size, out_inc, mv_pars, io) && bound_success;
    }

    int i_out = 0;
    PRAGMA_SIMD
    for (int j = 0; j < out_size; ++j) {
      out_values[i_out] += out_buffer_values_[j];
      i_out += out_inc;
    }

    if (io.mv_type == AnalogMVType::PosNegSeparate) {
      bound_success = finalizeOutput(out_values, out_size, out_inc, mv_pars, io);
    }

    return bound_success;
  }
  default:
    RPU_FATAL("AnalogMVType not implemented.");
  }
}

/********************************************************************************/
/*  public entries with noise / bound management */

template <typename T>
void ForwardBackwardPassIOManaged<T>::forwardVector(
    T **weights,
    const T *x_input,
    const int x_inc,
    T *d_output,
    const int d_inc,
    const T alpha,
    const bool is_test) {
  if (f_io_.isPerfect()) {
    // short-cut for FP
    ForwardBackwardPass<T>::forwardVector(
        weights, x_input, x_inc, d_output, d_inc, f_io_.out_scale * alpha, is_test);
    return;
  }
  if (!checked_implemented_) {
    ensureImplemented();
    checked_implemented_ = true;
  }
  T nm_scale_value = computeNoiseManagement(
      x_input, this->x_size_, x_inc, f_io_.noise_management, aux_nm_value_, f_io_);
  bool nm = f_io_.noise_management != NoiseManagementType::None;
  bool bm = f_io_.bound_management != BoundManagementType::None;

  if (nm && (nm_scale_value <= (T)0.0) && (f_io_.inp_noise <= (T)0.0)) {
    // short cut. output will be zero anyway
    int i_d = 0;
    PRAGMA_SIMD
    for (int i = 0; i < this->d_size_; ++i) {
      d_output[i_d] = (T)0.0;
      i_d += d_inc;
    }
    return;
  }

  T out_scale = (T)1.0;
  out_scale = f_io_.out_scale * alpha;

  bool bound_test_passed = false;
  T reduction_due_to_bound_management = 0.5;
  T scale = 1.;
  bool scaling = false;

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

    if (nm && nm_scale_value > (T)0.0) {
      scale /= nm_scale_value;
      scaling = true;
    }

    if (bm) {
      scale /= reduction_due_to_bound_management;
      scaling = true;
    }

    bound_test_passed = computeAnalogMV(
        weights, x_input, this->x_size_, x_inc, d_output, this->d_size_, d_inc, scale, scaling,
        this->fb_pars_.fwd, f_io_, false, is_test);

    if (bm) {
      bound_test_passed =
          bound_test_passed || (((int)reduction_due_to_bound_management > f_io_.max_bm_factor) ||
                                ((f_io_.inp_res > (T)0.0) && (reduction_due_to_bound_management >
                                                              f_io_.max_bm_res / f_io_.inp_res)));
    } else {
      bound_test_passed = true;
    }
  }

  if (scaling || out_scale != (T)1.0) {
    RPU::math::scal<T>(this->d_size_, out_scale / scale, d_output, d_inc);
  }
};

template <typename T>
void ForwardBackwardPassIOManaged<T>::backwardVector(
    T **weights, const T *d_input, const int d_inc, T *x_output, const int x_inc, const T alpha) {

  if (b_io_.isPerfect()) {
    // short-cut for FP
    ForwardBackwardPass<T>::backwardVector(
        weights, d_input, d_inc, x_output, x_inc, b_io_.out_scale * alpha);
    return;
  }

  // io managed version
  b_io_.bound_management = BoundManagementType::None; // not supported
  T nm_scale_value = computeNoiseManagement(
      d_input, this->d_size_, d_inc, b_io_.noise_management, aux_nm_value_, b_io_);
  bool nm = b_io_.noise_management != NoiseManagementType::None;
  T out_scale = b_io_.out_scale * alpha;
  bool scaling = nm && nm_scale_value > (T)0.0;

  if (nm && (nm_scale_value <= (T)0.0)) {
    // max is zero. output is just zero. short-cut
    int j_x = 0;
    PRAGMA_SIMD
    for (int j = 0; j < this->x_size_; j++) {
      x_output[j_x] = (T)0.0;
      j_x += x_inc;
    }
    return;
  }

  computeAnalogMV(
      weights, d_input, this->d_size_, d_inc, x_output, this->x_size_, x_inc,
      (T)1.0 / nm_scale_value, scaling, this->fb_pars_.bwd, b_io_, true, false);

  if (scaling || out_scale != (T)1.0) {
    RPU::math::scal<T>(this->x_size_, out_scale * nm_scale_value, x_output, x_inc);
  }
};
#undef CHECK_INPUT_BOUNDS

template <typename T>
void ForwardBackwardPassIOManaged<T>::dumpExtra(RPU::state_t &extra, const std::string prefix) {

  ForwardBackwardPass<T>::dumpExtra(extra, prefix);

  RPU::state_t state;
  RPU::insert(state, "aux_nm_value", aux_nm_value_);
  RPU::insertWithPrefix(extra, state, prefix);
}

template <typename T>
void ForwardBackwardPassIOManaged<T>::loadExtra(
    const RPU::state_t &extra, const std::string prefix, bool strict) {

  ForwardBackwardPass<T>::loadExtra(extra, prefix, strict);

  auto state = RPU::selectWithPrefix(extra, prefix);
  RPU::load(state, "aux_nm_value", aux_nm_value_, strict);
}

template class ForwardBackwardPassIOManaged<float>;
#ifdef RPU_USE_DOUBLE
template class ForwardBackwardPassIOManaged<double>;
#endif
#ifdef RPU_USE_FP16
template class ForwardBackwardPassIOManaged<half_t>;
#endif

} // namespace RPU
