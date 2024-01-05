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

#include "rpu.h"
#include "rng.h"

#include "math_util.h"
#include "utility_functions.h"

#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <random>

#ifdef _MSC_VER
#include <intrin.h>
#endif

#ifdef RPU_USE_MKL
#include "mkl.h"
#else
#ifdef RPU_USE_OPENBLAS
extern "C" {
#include "cblas.h"
}
#endif
#endif

namespace RPU {

template <typename T> void RPUAbstract<T>::printToStream(std::stringstream &ss) const {
  ss << "RPUAbstract<" << this->getDataTypeName() << ">(" << d_size_ << "," << x_size_ << ")"
     << std::endl;
}

/********************************************************************************/
template <typename T>
void RPUAbstract<T>::forwardMatrix(
    const T *X_input, T *D_output, int m_batch, bool x_trans, bool d_trans, bool is_test) {

  int x_offset = x_trans ? 1 : x_size_;
  int d_offset = d_trans ? 1 : d_size_;
  int x_inc = x_trans ? m_batch : 1;
  int d_inc = d_trans ? m_batch : 1;

  const T *x_input = X_input;
  T *d_output = D_output;

  for (size_t i = 0; i < (size_t)m_batch; i++) {
    this->forwardVector(x_input + i * x_offset, d_output + i * d_offset, x_inc, d_inc, is_test);
  }
}

template <typename T>
void RPUAbstract<T>::backwardMatrix(
    const T *D_input, T *X_output, int m_batch, bool d_trans, bool x_trans) {
  int x_offset = x_trans ? 1 : x_size_;
  int d_offset = d_trans ? 1 : d_size_;
  int x_inc = x_trans ? m_batch : 1;
  int d_inc = d_trans ? m_batch : 1;

  const T *d_input = D_input;
  T *x_output = X_output;

  for (size_t i = 0; i < (size_t)m_batch; i++) {
    this->backwardVector(d_input + i * d_offset, x_output + i * x_offset, d_inc, x_inc);
  }
}

template <typename T>
void RPUAbstract<T>::updateMatrix(
    const T *X_input, const T *D_input, int m_batch, bool x_trans, bool d_trans) {
  size_t x_offset = x_trans ? 1 : x_size_;
  size_t d_offset = d_trans ? 1 : d_size_;
  int x_inc = x_trans ? m_batch : 1;
  int d_inc = d_trans ? m_batch : 1;

  const T *x_input = X_input;
  const T *d_input = D_input;

  for (size_t i = 0; i < (size_t)m_batch; i++) {
    this->updateVector(x_input + i * x_offset, d_input + i * d_offset, x_inc, d_inc);
  }
}

template <typename T>
void RPUAbstract<T>::forwardMatrixBias(
    const T *X_input_without_bias,
    T *D_output,
    int m_batch,
    bool x_trans,
    bool d_trans,
    bool is_test) {
  size_t x_offset = x_trans ? 1 : x_size_ - 1;
  size_t d_offset = d_trans ? 1 : d_size_;
  int x_inc = x_trans ? m_batch : 1;
  int d_inc = d_trans ? m_batch : 1;

  const T *x_input_without_bias = X_input_without_bias;
  T *d_output = D_output;

  for (size_t i = 0; i < (size_t)m_batch; i++) {
    this->forwardVectorBias(
        x_input_without_bias + i * x_offset, d_output + i * d_offset, x_inc, d_inc, is_test);
  }
}

template <typename T>
void RPUAbstract<T>::backwardMatrixBias(
    const T *D_input, T *X_output_without_bias, int m_batch, bool d_trans, bool x_trans) {
  size_t x_offset = x_trans ? 1 : x_size_ - 1;
  size_t d_offset = d_trans ? 1 : d_size_;
  int x_inc = x_trans ? m_batch : 1;
  int d_inc = d_trans ? m_batch : 1;

  const T *d_input = D_input;
  T *x_output_without_bias = X_output_without_bias;

  for (size_t i = 0; i < (size_t)m_batch; i++) {
    this->backwardVectorBias(
        d_input + i * d_offset, x_output_without_bias + i * x_offset, d_inc, x_inc);
  }
}

template <typename T>
void RPUAbstract<T>::updateMatrixBias(
    const T *X_input_without_bias, const T *D_input, int m_batch, bool x_trans, bool d_trans) {
  size_t x_offset = x_trans ? 1 : x_size_ - 1;
  size_t d_offset = d_trans ? 1 : d_size_;
  int x_inc = x_trans ? m_batch : 1;
  int d_inc = d_trans ? m_batch : 1;

  const T *x_input_without_bias = X_input_without_bias;
  const T *d_input = D_input;

  for (size_t i = 0; i < (size_t)m_batch; i++) {
    this->updateVectorBias(
        x_input_without_bias + i * x_offset, d_input + i * d_offset, x_inc, d_inc);
  }
}

template <> std::string RPUAbstract<float>::getDataTypeName() const { return "float"; }

#ifdef RPU_USE_DOUBLE
template <> std::string RPUAbstract<double>::getDataTypeName() const { return "double"; }
#endif

#ifdef RPU_USE_FP16
template <> std::string RPUAbstract<half_t>::getDataTypeName() const {
#ifdef RPU_BFLOAT_AS_FP16
  return "bfloat16";
#else
  return "float16";
#endif
}
#endif

/********************************************************************************/
template class RPUAbstract<float>;
#ifdef RPU_USE_DOUBLE
template class RPUAbstract<double>;
#endif
#ifdef RPU_USE_FP16
template class RPUAbstract<half_t>;
#endif

/********************************************************************************
 * SimpleMetaParameter
 *********************************************************************************/

// SimpleMetaParameter
template <typename T> void SimpleMetaParameter<T>::printToStream(std::stringstream &ss) const {
  if (use_delayed_update) {
    ss << "Using DELAYED update." << std::endl;
  }
  if (lifetime > (T)0.0) {
    ss << "\t lifetime [decay]:\t" << lifetime << std::endl;
  }
  if (drift.nu > (T)0.0) {
    drift.printToStream(ss);
  }
  if (diffusion > (T)0.0) {
    ss << "\t diffusion:\t\t" << diffusion << std::endl;
    flicker.printToStream(ss);
  }
}

template <typename T> void FlickerParameter<T>::printToStream(std::stringstream &ss) const {
  ss << "\t n_flicker:\t\t" << n << std::endl;
  ss << "\t flicker_r:\t\t" << r << std::endl;
  ss << "\t flicker_q:\t\t" << q << std::endl;
  ss << "\t flicker_h:\t\t" << h << std::endl;
  ss << "\t flicker_wreset:\t" << std::boolalpha << wreset << std::endl;
  ss << "\t flicker_wreset_tol:\t" << wreset_tol << std::endl;
}

template struct SimpleMetaParameter<float>;
template struct FlickerParameter<float>;
template struct DriftParameter<float>;
#ifdef RPU_USE_DOUBLE
template struct SimpleMetaParameter<double>;
template struct FlickerParameter<double>;
template struct DriftParameter<double>;
#endif
#ifdef RPU_USE_FP16
template struct SimpleMetaParameter<half_t>;
template struct FlickerParameter<half_t>;
template struct DriftParameter<half_t>;
#endif

/********************************************************************************
 * RPUSimple<T>
 *********************************************************************************/

template <typename T> void RPUSimple<T>::initialize(int x_sz, int d_sz) {

  this->x_size_ = x_sz;
  this->d_size_ = d_sz;

  use_delayed_update_ = false;

  weights_ = Array_2D_Get<T>(d_sz, x_sz);
  fb_weights_ = nullptr;
  delta_weights_extern_.resize(d_sz, nullptr); // this is a pointer array of row pointers

  for (int i = 0; i < d_sz; ++i) {
    for (int j = 0; j < x_sz; ++j) {
      weights_[i][j] = (T)((i + 1) * 0.01 + (0.01 * j + 1));
    }
  }
  weights_buffer_ = Array_2D_Get<T>(d_sz, x_sz);

  matrix_indices_ = nullptr;
  matrix_indices_set_ = false;

  temp_x_vector_bias_.resize(x_sz);

  rng_ = std::make_shared<RNG<T>>(0);
  rw_rng_ = std::make_shared<RealWorldRNG<T>>(0);

  last_update_m_batch_ = 1;

  fwd_alpha_ = (T)1.0;
  bwd_alpha_ = (T)1.0;
}

/*********************************************************************************/
template <typename T> RPUSimple<T>::RPUSimple(int x_sz, int d_sz) : RPUAbstract<T>(x_sz, d_sz) {

  this->initialize(x_sz, d_sz);

  DEBUG_CALL(this->disp(););
  DEBUG_OUT("RPUSimple constructed.");
}

/*********************************************************************************/
template <typename T> RPUSimple<T>::~RPUSimple() {

  rng_ = nullptr;
  rw_rng_ = nullptr;

  if (!shared_weights_if_) {
    Array_2D_Free<T>(weights_);
  } else {
    delete[] weights_;
    weights_ = nullptr;
  }

  Array_2D_Free<T>(weights_buffer_);

  if (fb_weights_ != nullptr) {
    Array_2D_Free<T>(fb_weights_);
  }

  matrix_indices_ = nullptr; // memory externally governed

  DEBUG_OUT("RPUSimple DESTRUCTED");
}

/*********************************************************************************/
// copy constructor
template <typename T> RPUSimple<T>::RPUSimple(const RPUSimple<T> &other) : RPUAbstract<T>(other) {

  this->initialize(other.x_size_, other.d_size_);

  this->setWeights(*other.weights_);

  this->copyWeightsToBuffer();

  // note: cannot copy external shared weight pointer... user needs to
  // ensure that again setShared is called after copying
  shared_weights_if_ = false;

  if (other.wdrifter_) {
    // call copy constructor
    wdrifter_ = RPU::make_unique<WeightDrifter<T>>(*other.wdrifter_);
  }

  // no copy needed
  wclipper_ = nullptr;
  wremapper_ = nullptr;

  // cannot copy external weight pointer... user needs to call it again
  delta_weights_extern_[0] = nullptr;

  if (other.fb_weights_) {
    fb_weights_ = Array_2D_Get<T>(this->d_size_, this->x_size_);
    RPU::math::copy<T>(this->x_size_ * this->d_size_, other.fb_weights_[0], 1, fb_weights_[0], 1);
    fb_weight_modifier_ =
        RPU::make_unique<WeightModifier<T>>(this->x_size_, this->d_size_); // no copy
  }

  use_delayed_update_ = other.use_delayed_update_;

  par_ = other.par_;

  flicker_probs_ = other.flicker_probs_;
  flicker_states_ = other.flicker_states_;

  matrix_indices_ = other.matrix_indices_;
  matrix_indices_set_ = other.matrix_indices_set_;

  // note: RNG / temp_values are not copied.
  last_update_m_batch_ = other.last_update_m_batch_;

  fwd_alpha_ = other.fwd_alpha_;
  bwd_alpha_ = other.bwd_alpha_;

  DEBUG_CALL(this->disp(););
  DEBUG_OUT("RPUSimple copy constructed.");
}

/*********************************************************************************/
// copy assignment
template <typename T> RPUSimple<T> &RPUSimple<T>::operator=(const RPUSimple<T> &other) {
  RPUSimple<T> tmp(other);
  swap(*this, tmp);
  return *this;
}
// move constructor
template <typename T> RPUSimple<T>::RPUSimple(RPUSimple<T> &&other) noexcept {
  *this = std::move(other);
}

// move assignment
template <typename T> RPUSimple<T> &RPUSimple<T>::operator=(RPUSimple<T> &&other) noexcept {

  RPUAbstract<T>::operator=(std::move(other));

  use_delayed_update_ = other.use_delayed_update_;

  rng_ = std::move(other.rng_);
  rw_rng_ = std::move(other.rw_rng_);

  par_ = other.par_;

  weights_ = other.weights_;
  other.weights_ = nullptr;

  shared_weights_if_ = other.shared_weights_if_;

  weights_buffer_ = other.weights_buffer_;
  other.weights_buffer_ = nullptr;

  fb_weights_ = other.fb_weights_;
  other.fb_weights_ = nullptr;

  delta_weights_extern_ = std::move(other.delta_weights_extern_);

  fb_weight_modifier_ = std::move(other.fb_weight_modifier_);

  temp_x_vector_bias_ = other.temp_x_vector_bias_;
  temp_x_matrix_bias_ = other.temp_x_matrix_bias_;
  temp_tensor_ = other.temp_tensor_;
  flicker_states_ = other.flicker_states_;
  flicker_probs_ = other.flicker_probs_;

  matrix_indices_ = other.matrix_indices_;
  other.matrix_indices_ = nullptr;

  matrix_indices_set_ = other.matrix_indices_set_;
  other.matrix_indices_set_ = false;

  wdrifter_ = std::move(other.wdrifter_);
  wremapper_ = std::move(other.wremapper_);

  last_update_m_batch_ = other.last_update_m_batch_;

  fwd_alpha_ = other.fwd_alpha_;
  bwd_alpha_ = other.bwd_alpha_;

  return *this;
}

/*********************************************************************************/
/* dump / load state */

template <typename T> void RPUSimple<T>::dumpExtra(RPU::state_t &extra, const std::string prefix) {
  // This dumps all the temporary buffers needed for setting the
  // state.  It will not store the meta-parameters and the sizes it
  // will also not store the buffers that are already handled by
  // getDeviceParameters
  using V = std::vector<T>;

  V tmp(this->x_size_ * this->d_size_);
  RPU::state_t state;

  if (this->fb_weights_) {
    RPU::math::copy<T>(tmp.size(), this->fb_weights_[0], 1, tmp.data(), 1);
    RPU::insert(state, "fb_weights", tmp);
    fb_weight_modifier_->dumpExtra(state, "fb_weight_modifier");
  }

  RPU::math::copy<T>(tmp.size(), this->weights_buffer_[0], 1, tmp.data(), 1);
  if (use_delayed_update_) {
    RPU::insert(state, "weights_buffer", tmp);
  }

  if (wdrifter_) {
    wdrifter_->dumpExtra(state, "wdrifter");
  }
  if (wclipper_) {
    wclipper_->dumpExtra(state, "wclipper");
  }
  if (wremapper_) {
    wremapper_->dumpExtra(state, "wremapper");
  }

  // ignore the temporary buffers
  // matrix_indices_ not handled... set from outside
  // RNG (seeds) not copied

  RPU::insert(state, "learning_rate", this->learning_rate_);
  RPU::insert(state, "flicker_probs", flicker_probs_);
  RPU::insert(state, "flicker_states", flicker_states_);
  RPU::insert(state, "use_delayed_update", use_delayed_update_);
  RPU::insert(state, "last_update_m_batch", last_update_m_batch_);
  RPU::insert(state, "fwd_alpha", fwd_alpha_);
  RPU::insert(state, "bwd_alpha", bwd_alpha_);

  RPU::insertWithPrefix(extra, state, prefix);
}

template <typename T>
void RPUSimple<T>::loadExtra(const RPU::state_t &extra, const std::string prefix, bool strict) {

  using V = std::vector<T>;
  auto state = RPU::selectWithPrefix(extra, prefix);

  if (state.count("fb_weights")) {
    V tmp;
    RPU::load(state, "fb_weights", tmp, strict);
    if (tmp.size()) {
      if (fb_weights_ == nullptr) {
        fb_weights_ = Array_2D_Get<T>(this->d_size_, this->x_size_);
        fb_weight_modifier_ = RPU::make_unique<WeightModifier<T>>(this->x_size_, this->d_size_);
      }
      RPU::math::copy<T>(tmp.size(), tmp.data(), 1, fb_weights_[0], 1);
    }
  }

  if (state.count("wdrifter")) {
    if (!wdrifter_) {
      wdrifter_ = RPU::make_unique<WeightDrifter<T>>(this->x_size_ * this->d_size_, getPar().drift);
    }
    wdrifter_->loadExtra(state, "wdrifter", strict);
  }

  if (state.count("wclipper")) {
    if (!wclipper_) {
      wclipper_ = RPU::make_unique<WeightClipper<T>>(this->x_size_, this->d_size_);
    }
    wclipper_->loadExtra(state, "wclipper", strict);
  }

  if (state.count("wremapper")) {
    if (!wremapper_) {
      wremapper_ = RPU::make_unique<WeightRemapper<T>>(this->x_size_, this->d_size_);
    }
    wremapper_->loadExtra(state, "wremapper", strict);
  }

  RPU::load(state, "flicker_probs", flicker_probs_, strict);
  RPU::load(state, "learning_rate", this->learning_rate_, strict);
  RPU::load(state, "use_delayed_update", use_delayed_update_, strict);
  RPU::load(state, "last_update_m_batch", last_update_m_batch_, strict);
  RPU::load(state, "fwd_alpha", fwd_alpha_, strict);
  RPU::load(state, "bwd_alpha", bwd_alpha_, strict);
  RPU::load(state, "flicker_states", flicker_states_, strict);

  if (use_delayed_update_) {
    V tmp;
    RPU::load(state, "weights_buffer", tmp, strict);
    RPU::math::copy<T>(tmp.size(), tmp.data(), 1, weights_buffer_[0], 1);
  }
}

/*********************************************************************************/
/* General forward/backward/update */

template <typename T>
void RPUSimple<T>::forward(
    const T *X_input,
    T *D_output,
    bool bias,
    int m_batch,
    bool x_trans,
    bool d_trans,
    bool is_test) {

  if ((m_batch == 1) && (!x_trans) && (!d_trans)) {
    if (bias) {
      this->forwardVectorBias(X_input, D_output, 1, 1, is_test);
    } else {
      this->forwardVector(X_input, D_output, 1, 1, is_test);
    }
  } else {
    if (bias) {
      this->forwardMatrixBias(X_input, D_output, m_batch, x_trans, d_trans, is_test);
    } else {
      this->forwardMatrix(X_input, D_output, m_batch, x_trans, d_trans, is_test);
    }
  }
}

template <typename T>
void RPUSimple<T>::backward(
    const T *D_input, T *X_output, bool bias, int m_batch, bool d_trans, bool x_trans) {
  if ((m_batch == 1) && (!x_trans) && (!d_trans)) {
    if (bias) {
      this->backwardVectorBias(D_input, X_output);
    } else {
      this->backwardVector(D_input, X_output);
    }
  } else {
    if (bias) {
      this->backwardMatrixBias(D_input, X_output, m_batch, d_trans, x_trans);
    } else {
      this->backwardMatrix(D_input, X_output, m_batch, d_trans, x_trans);
    }
  }
}

template <typename T>
void RPUSimple<T>::update(
    const T *X_input, const T *D_input, bool bias, int m_batch, bool x_trans, bool d_trans) {
  last_update_m_batch_ = m_batch; // this is mini-batchsize * reuse_factor !

  // update weights
  if ((m_batch == 1) && (!x_trans) && (!d_trans)) {
    // short-cut for vectors
    if (bias)
      this->updateVectorBias(X_input, D_input);
    else
      this->updateVector(X_input, D_input);
  } else {
    if (bias)
      this->updateMatrixBias(X_input, D_input, m_batch, x_trans, d_trans);
    else
      this->updateMatrix(X_input, D_input, m_batch, x_trans, d_trans);
  }
}

/*********************************************************************************/
/* Matrix forward/backward/update */

template <typename T>
void RPUSimple<T>::forwardMatrix(
    const T *X_input, T *D_output, int m_batch, bool x_trans, bool d_trans, bool is_test) {

  if (d_trans) {
    RPU::math::gemm<T>(
        CblasRowMajor, CblasNoTrans,
        x_trans ? CblasNoTrans : CblasTrans, // inverse meaning...
        this->d_size_,                       // M
        m_batch,                             // N
        this->x_size_,                       // K
        this->fwd_alpha_, getFBWeights(is_test)[0], this->x_size_, X_input,
        x_trans ? m_batch : this->x_size_, (float)0.0, D_output, m_batch);
  } else {
    RPU::math::gemm<T>(
        CblasRowMajor, x_trans ? CblasTrans : CblasNoTrans, CblasTrans,
        m_batch,       // M
        this->d_size_, // N
        this->x_size_, // K
        this->fwd_alpha_, X_input, x_trans ? m_batch : this->x_size_, getFBWeights(is_test)[0],
        this->x_size_, (float)0.0, D_output, this->d_size_);
  }
}

template <typename T>
void RPUSimple<T>::backwardMatrix(
    const T *D_input, T *X_output, int m_batch, bool d_trans, bool x_trans) {

  if (x_trans) {
    RPU::math::gemm<T>(
        CblasRowMajor, CblasTrans, d_trans ? CblasNoTrans : CblasTrans, this->x_size_, m_batch,
        this->d_size_, this->bwd_alpha_, getFBWeights(false)[0], this->x_size_, D_input,
        d_trans ? m_batch : this->d_size_, (T)0.0, X_output, m_batch);
  } else {
    RPU::math::gemm<T>(
        CblasRowMajor, d_trans ? CblasTrans : CblasNoTrans, CblasNoTrans,
        m_batch,       // M
        this->x_size_, // N
        this->d_size_, // K
        this->bwd_alpha_, D_input, d_trans ? m_batch : this->d_size_, getFBWeights(false)[0],
        this->x_size_, (T)0.0, X_output, this->x_size_);
  }
}

template <typename T>
void RPUSimple<T>::updateMatrix(
    const T *X_input, const T *D_input, int m_batch, bool x_trans, bool d_trans) {
  RPU::math::gemm<T>(
      CblasRowMajor, d_trans ? CblasNoTrans : CblasTrans, x_trans ? CblasTrans : CblasNoTrans,
      this->d_size_, // M
      this->x_size_, // N
      m_batch,       // K
      -this->getAlphaLearningRate(), D_input, d_trans ? m_batch : this->d_size_, X_input,
      x_trans ? m_batch : this->x_size_, this->getUpBeta(), this->getUpWeights()[0], this->x_size_);
}

template <typename T>
void RPUSimple<T>::forwardMatrixBias(
    const T *X_input_without_bias,
    T *D_output,
    int m_batch,
    bool x_trans,
    bool d_trans,
    bool is_test) {
  // TODO: use a better way to do this with GEMM LDA etc.
  T *bias_buffer = this->copyToMatrixBiasBuffer(X_input_without_bias, m_batch, x_trans);
  this->forwardMatrix(bias_buffer, D_output, m_batch, x_trans, d_trans, is_test);
  this->releaseMatrixBiasBuffer();
}

template <typename T>
void RPUSimple<T>::backwardMatrixBias(
    const T *D_input, T *X_output_without_bias, int m_batch, bool d_trans, bool x_trans) {
  // TODO: use a better way to do this with GEMM LDA etc.
  T *bias_buffer = this->getMatrixBiasBuffer(m_batch);
  this->backwardMatrix(D_input, bias_buffer, m_batch, d_trans, x_trans);
  this->copyFromMatrixBiasBuffer(X_output_without_bias, m_batch, x_trans, bias_buffer);
  this->releaseMatrixBiasBuffer();
}

template <typename T>
void RPUSimple<T>::updateMatrixBias(
    const T *X_input_without_bias, const T *D_input, int m_batch, bool x_trans, bool d_trans) {
  // TODO: use a better way to do this with GEMM LDA etc.
  T *bias_buffer = this->copyToMatrixBiasBuffer(X_input_without_bias, m_batch, x_trans);
  this->updateMatrix(bias_buffer, D_input, m_batch, x_trans, d_trans);
  this->releaseMatrixBiasBuffer();
}

template <typename T> T *RPUSimple<T>::getMatrixBiasBuffer(int m_batch) {
  temp_x_matrix_bias_.resize((size_t)m_batch * this->x_size_);
  return temp_x_matrix_bias_.data();
}

template <typename T>
T *RPUSimple<T>::copyToMatrixBiasBuffer(const T *X_input_without_bias, int m_batch, bool x_trans) {
  T *bias_buffer = getMatrixBiasBuffer(m_batch);
  RPU::math::makeBias<T>(bias_buffer, X_input_without_bias, this->x_size_, m_batch, x_trans);
  return bias_buffer;
}

template <typename T>
void RPUSimple<T>::copyFromMatrixBiasBuffer(
    T *X_input_without_bias, int m_batch, bool x_trans, T *bias_buffer) {
  RPU::math::copyWithoutBias<T>(X_input_without_bias, bias_buffer, this->x_size_, m_batch, x_trans);
}

/*********************************************************************************/
/* Vector forward/backward/update */

template <typename T>
void RPUSimple<T>::forwardVector(
    const T *x_input, T *d_output, int x_inc, int d_inc, bool is_test) {
  RPU::math::gemv<T>(
      CblasRowMajor, CblasNoTrans, this->d_size_, this->x_size_, this->fwd_alpha_,
      getFBWeights(is_test)[0], this->x_size_, x_input, x_inc, (T)0.0, d_output, d_inc);
}

template <typename T>
void RPUSimple<T>::backwardVector(const T *d_input, T *x_output, int d_inc, int x_inc) {
  RPU::math::gemv<T>(
      CblasRowMajor, CblasTrans, this->d_size_, this->x_size_, this->bwd_alpha_,
      getFBWeights(false)[0], this->x_size_, d_input, d_inc, (T)0.0, x_output, x_inc);
}

template <typename T>
void RPUSimple<T>::updateVector(const T *x_input, const T *d_input, int x_inc, int d_inc) {
  DEBUG_OUT("RPU::updateVector. LR " << -this->getAlphaLearningRate());

  if (!this->getDeltaWeights()) {
    RPU::math::ger<T>(
        CblasRowMajor, this->d_size_, this->x_size_, -this->getAlphaLearningRate(), d_input, d_inc,
        x_input, x_inc, this->getUpWeights()[0], this->x_size_);
  } else {
    if (x_inc == 1 && d_inc == 1) {
      RPUSimple<T>::updateMatrix(x_input, d_input, 1, false, false);
    } else {
      RPU_FATAL("updateVector for delta weights and xd_inc>1 is not implemented.");
    }
  }
}

template <typename T>
void RPUSimple<T>::copyFromVectorBiasBuffer(T *x_output_without_bias, int x_inc) {
  RPU::math::copy<T>(
      this->x_size_ - 1, temp_x_vector_bias_.data(), 1, x_output_without_bias, x_inc);
}

template <typename T>
T *RPUSimple<T>::copyToVectorBiasBuffer(const T *x_input_without_bias, int x_inc) {
  RPU::math::copy<T>(this->x_size_ - 1, x_input_without_bias, x_inc, temp_x_vector_bias_.data(), 1);
  temp_x_vector_bias_[this->x_size_ - 1] = 1.;
  return temp_x_vector_bias_.data();
}

template <typename T>
void RPUSimple<T>::forwardVectorBias(
    const T *x_input_without_bias, T *d_output, int x_inc, int d_inc, bool is_test) {
  T *bias_buffer = this->copyToVectorBiasBuffer(x_input_without_bias, x_inc);
  this->forwardVector(bias_buffer, d_output, 1, d_inc, is_test);
}

template <typename T>
void RPUSimple<T>::backwardVectorBias(
    const T *d_input, T *x_output_without_bias, int d_inc, int x_inc) {
  this->backwardVector(d_input, this->getVectorBiasBuffer(), d_inc, 1);
  this->copyFromVectorBiasBuffer(x_output_without_bias, x_inc);
}

template <typename T>
void RPUSimple<T>::updateVectorBias(
    const T *x_input_without_bias, const T *d_input, int x_inc, int d_inc) {
  T *bias_buffer = this->copyToVectorBiasBuffer(x_input_without_bias, x_inc);
  this->updateVector(bias_buffer, d_input, 1, d_inc);
}

/*********************************************************************************/
/* Tensor forward/backward/update */

template <typename T>
void RPUSimple<T>::getTensorBuffer(T **x_tensor_ptr, T **d_tensor_ptr, int m_batch, int dim3) {

  int n = (this->x_size_ + this->d_size_) * dim3 * m_batch;
  temp_tensor_.resize(n);

  // permute 132
  *x_tensor_ptr = temp_tensor_.data();
  *d_tensor_ptr = temp_tensor_.data() + this->x_size_ * dim3 * m_batch;
}

template <typename T>
void RPUSimple<T>::permute132(
    T *out_tensor, const T *in_tensor, int dim1, int dim2, int dim3, bool bias2) {
  // adds bias to original dimension dim2
  math::permute132<T>(out_tensor, in_tensor, dim1, dim2, dim3, bias2);
}

template <typename T>
void RPUSimple<T>::forwardTensor(
    const T *X_input, T *D_output, bool bias, int m_batch, int dim3, bool trans, bool is_test) {
  if ((dim3 == 1) || (!trans))
    this->forward(X_input, D_output, bias, dim3 * m_batch, trans, trans, is_test);
  else {
    int x_size = this->getXSize();
    int d_size = this->getDSize();

    T *x_tensor = nullptr;
    T *d_tensor = nullptr;

    this->getTensorBuffer(&x_tensor, &d_tensor, m_batch, dim3);

    this->permute132(x_tensor, X_input, m_batch, x_size, dim3, bias);

    this->forwardMatrix(x_tensor, d_tensor, m_batch * dim3, true, true, is_test);

    this->permute132(D_output, d_tensor, m_batch, dim3, d_size, false);
  }
}

template <typename T>
void RPUSimple<T>::backwardTensor(
    const T *D_input, T *X_output, bool bias, int m_batch, int dim3, bool trans) {
  if ((dim3 == 1) || (!trans))
    this->backward(D_input, X_output, bias, dim3 * m_batch, trans, trans);
  else {
    int x_size = this->getXSize();
    int d_size = this->getDSize();

    T *x_tensor = nullptr;
    T *d_tensor = nullptr;

    this->getTensorBuffer(&x_tensor, &d_tensor, m_batch, dim3);

    this->permute132(d_tensor, D_input, m_batch, d_size, dim3, false);

    this->backward(d_tensor, x_tensor, bias, m_batch * dim3, true, true);

    int b = (bias) ? 1 : 0;
    this->permute132(X_output, x_tensor, m_batch, dim3, x_size - b, false);
  }
}

template <typename T>
void RPUSimple<T>::updateTensor(
    const T *X_input, const T *D_input, bool bias, int m_batch, int dim3, bool trans) {
  if ((dim3 == 1) || (!trans))
    this->update(X_input, D_input, bias, m_batch * dim3, trans, trans);
  else {
    int x_size = this->getXSize();
    int d_size = this->getDSize();

    T *x_tensor = nullptr;
    T *d_tensor = nullptr;

    this->getTensorBuffer(&x_tensor, &d_tensor, m_batch, dim3);

    this->permute132(x_tensor, X_input, m_batch, x_size, dim3, bias);
    this->permute132(d_tensor, D_input, m_batch, d_size, dim3, false);

    this->update(x_tensor, d_tensor, false, m_batch * dim3, true, true);
  }
}

/*********************************************************************************/
/* Indexed forward/backward/update */

template <typename T>
void RPUSimple<T>::copyIndexedInput(
    T *out_tensor,
    const T *src_tensor,
    const int total_input_size,
    const int *indices_shifted,
    const int size,
    const int m_batch,
    const int dim3,
    const bool trans,
    const int m_batch_slice,
    const int *batch_indices) {

  // logic: this function should be overloaded in RPUCudaSimple for
  // the Indexed to work.

  // for RPUCuda_pulsed: we overload the forwardIndexed etc and call
  // a forwardMatrixIterator<InputIteratorT,float *> which is
  // basically the original forwardMatrix but calling the
  // InputIteratorT type in BLM. Thus BLM will handle the conversion
  // from indexed. The old Forward_Matrix will be just calling a
  // standard forwardMatrixIterator<const float *,float *>

  // all other RPUs just use the RPU/RPUCudaSimple::copyIndexed
  // and forwardIndex without the need to implement anything new.

  int m = m_batch;
  int input_matrix_size = total_input_size / dim3;
  int M = m_batch * size;
  int sz_all = size * m_batch * dim3;

  bool batch_slice = m_batch_slice > 0;

  if (trans) {
    // here we additioanlly permute 132
    if (batch_slice) {
      sz_all = size * m_batch_slice * dim3;
      M = m_batch_slice * dim3;

      for (int idx = 0; idx < sz_all; idx++) {

        int i_dim3 = (idx % M) / m_batch_slice;
        int i_batch_slice = idx % m_batch_slice;
        int i_xd = idx / M;
        int batch_idx = batch_indices[i_batch_slice + m_batch * i_dim3];
        int ind_idx = batch_idx + m_batch * i_xd;

        int j_shifted = indices_shifted[ind_idx];
        out_tensor[idx] = (j_shifted <= 1)
                              ? (T)j_shifted
                              : src_tensor[(j_shifted - 2) + i_dim3 * input_matrix_size];
      }
    } else {

      int L = m_batch * dim3;

      for (int idx = 0; idx < sz_all; idx++) {

        int i = (idx % L) / m * M + (idx % m) + idx / L * m;
        int j_shifted = indices_shifted[i % M];
        out_tensor[idx] = (j_shifted <= 1)
                              ? (T)j_shifted
                              : src_tensor[(j_shifted - 2) + i / M * input_matrix_size];
      }
    }
  } else { // no trans
    if (batch_slice) {

      sz_all = size * m_batch_slice * dim3;
      M = m_batch_slice * size;

      for (int idx = 0; idx < sz_all; idx++) {

        int i_dim3 = idx / M;
        int i_batch_slice = (idx % M) / size;
        int i_xd = idx % size;
        int batch_idx = batch_indices[i_batch_slice + m_batch * i_dim3];
        int ind_idx = batch_idx * size + i_xd;

        int j_shifted = indices_shifted[ind_idx];
        out_tensor[idx] = (j_shifted <= 1)
                              ? (T)j_shifted
                              : src_tensor[(j_shifted - 2) + i_dim3 * input_matrix_size];
      }
    } else {

      for (int idx = 0; idx < sz_all; idx++) {

        int j_shifted = indices_shifted[idx % M];
        out_tensor[idx] = (j_shifted <= 1)
                              ? (T)j_shifted
                              : src_tensor[(j_shifted - 2) + idx / M * input_matrix_size];
      }
    }
  }
}

template <typename T>
void RPUSimple<T>::copyIndexedOutput(
    T *out_tensor,
    const T *src_tensor,
    const int total_output_size,
    const int *indices_shifted,
    const int size,
    const int m_batch,
    const int dim3,
    const bool trans,
    const int m_batch_slice,
    const int *batch_indices) {
  bool batch_slice = m_batch_slice > 0;

  // output iterator
  int m = m_batch;
  int output_matrix_size = total_output_size / dim3;
  int M = m_batch * size;
  int sz_all = size * m_batch * dim3;

  if (trans) {
    // here we additionally permute 132

    if (batch_slice) {
      sz_all = size * m_batch_slice * dim3;
      M = m_batch_slice * dim3;

      for (int idx = 0; idx < sz_all; idx++) {

        int i_dim3 = (idx % M) / m_batch_slice;
        int i_batch_slice = idx % m_batch_slice;
        int i_xd = idx / M;
        int batch_idx = batch_indices[i_batch_slice + m_batch * i_dim3];
        int ind_idx = batch_idx + m_batch * i_xd;

        int j_shifted = indices_shifted[ind_idx];
        if (j_shifted > 1)
          out_tensor[(j_shifted - 2) + i_dim3 * output_matrix_size] += src_tensor[idx];
      }

    } else {

      int L = m_batch * dim3;

      for (int idx = 0; idx < sz_all; idx++) {
        int i = (idx % L) / m * M + (idx % m) + idx / L * m;
        int j_shifted = indices_shifted[i % M];
        if (j_shifted > 1)
          out_tensor[(j_shifted - 2) + i / M * output_matrix_size] += src_tensor[idx];
      }
    }
  } else { // no trans
    if (batch_slice) {
      sz_all = size * m_batch_slice * dim3;
      M = m_batch_slice * size;

      for (int idx = 0; idx < sz_all; idx++) {

        int i_dim3 = idx / M;
        int i_batch_slice = (idx % M) / size;
        int i_xd = idx % size;
        int batch_idx = batch_indices[i_batch_slice + m_batch * i_dim3];
        int ind_idx = batch_idx * size + i_xd;

        int j_shifted = indices_shifted[ind_idx];
        if (j_shifted > 1)
          out_tensor[(j_shifted - 2) + i_dim3 * output_matrix_size] += src_tensor[idx];
      }

    } else {
      for (int idx = 0; idx < sz_all; idx++) {
        int j_shifted = indices_shifted[idx % M];
        if (j_shifted > 1)
          out_tensor[(j_shifted - 2) + idx / M * output_matrix_size] += src_tensor[idx];
      }
    }
  }
}

template <typename T>
void RPUSimple<T>::copySliceInput(
    T *out_tensor,
    const T *src_tensor,
    const int size,
    const int m_batch,
    const int dim3,
    const bool trans,
    const int m_batch_slice,
    const int *batch_indices) {
  int input_matrix_size = m_batch * size;
  int sz_all = size * m_batch_slice * dim3;

  if (trans) {

    int M = m_batch_slice * dim3;

    for (int idx = 0; idx < sz_all; idx++) {

      int i_dim3 = (idx % M) / m_batch_slice;
      int i_batch_slice = idx % m_batch_slice;
      int i_xd = idx / M;
      int batch_idx = batch_indices[i_batch_slice + m_batch * i_dim3];
      int ind_idx = batch_idx + m_batch * i_xd;

      out_tensor[idx] = src_tensor[ind_idx + i_dim3 * input_matrix_size];
    }
  } else { // no trans

    int M = m_batch_slice * size;

    for (int idx = 0; idx < sz_all; idx++) {

      int i_dim3 = idx / M;
      int i_batch_slice = (idx % M) / size;
      int i_xd = idx % size;
      int batch_idx = batch_indices[i_batch_slice + m_batch * i_dim3];
      int ind_idx = batch_idx * size + i_xd;

      out_tensor[idx] = src_tensor[ind_idx + i_dim3 * input_matrix_size];
    }
  }
}

template <typename T>
void RPUSimple<T>::copySliceOutput(
    T *out_tensor,
    const T *src_tensor,
    const int size,
    const int m_batch,
    const int dim3,
    const bool trans,
    const int m_batch_slice,
    const int *batch_indices) {
  int output_matrix_size = m_batch * size;
  int sz_all = size * m_batch_slice * dim3;

  if (trans) {
    // here we additionally permute 132

    int M = m_batch_slice * dim3;

    for (int idx = 0; idx < sz_all; idx++) {

      int i_dim3 = (idx % M) / m_batch_slice;
      int i_batch_slice = idx % m_batch_slice;
      int i_xd = idx / M;
      int batch_idx = batch_indices[i_batch_slice + m_batch * i_dim3];
      int ind_idx = batch_idx + m_batch * i_xd;

      out_tensor[ind_idx + i_dim3 * output_matrix_size] = src_tensor[idx];
    }
  } else { // no trans
    int M = m_batch_slice * size;

    for (int idx = 0; idx < sz_all; idx++) {

      int i_dim3 = idx / M;
      int i_batch_slice = (idx % M) / size;
      int i_xd = idx % size;
      int batch_idx = batch_indices[i_batch_slice + m_batch * i_dim3];
      int ind_idx = batch_idx * size + i_xd;

      out_tensor[ind_idx + i_dim3 * output_matrix_size] = src_tensor[idx];
    }
  }
}

template <typename T> void RPUSimple<T>::setZero(T *v, const int size) {
  for (int i = 0; i < size; i++) {
    v[i] = (T)0.0;
  }
}

template <typename T>
void RPUSimple<T>::forwardIndexed(
    const T *X_input,
    T *D_output,
    int total_input_size,
    int m_batch,
    int dim3,
    bool trans,
    bool is_test) {
  // EXPECTS forward index to be set properly !!
  // total_input_size is size of X_input

  T *x_tensor = nullptr;
  T *d_tensor = nullptr;
  this->getTensorBuffer(&x_tensor, &d_tensor, m_batch, dim3);

  this->copyIndexedInput(
      x_tensor, X_input, total_input_size, this->getMatrixIndices(), this->getXSize(), m_batch,
      dim3, trans);
  if ((dim3 > 1) && trans) {
    this->forwardMatrix(x_tensor, d_tensor, m_batch * dim3, trans, trans, is_test);
    this->permute132(D_output, d_tensor, m_batch, dim3, this->getDSize(), false);
  } else {
    this->forwardMatrix(x_tensor, D_output, m_batch * dim3, trans, trans, is_test);
  }
}

template <typename T>
void RPUSimple<T>::forwardIndexedSlice(
    const T *X_input,
    T *D_output,
    int total_input_size,
    int m_batch,
    int dim3,
    bool trans,
    int m_batch_slice,
    const int *batch_indices,
    bool is_test) {
  T *x_tensor = nullptr;
  T *d_tensor = nullptr;

  this->getTensorBuffer(&x_tensor, &d_tensor, m_batch_slice, dim3);

  this->copyIndexedInput(
      x_tensor, X_input, total_input_size, this->getMatrixIndices(), this->getXSize(), m_batch,
      dim3, trans, m_batch_slice, batch_indices);

  this->forwardMatrix(x_tensor, d_tensor, m_batch_slice * dim3, trans, trans, is_test);

  this->copySliceOutput(
      D_output, d_tensor, this->getDSize(), m_batch, dim3, trans, m_batch_slice, batch_indices);
}

template <typename T>
void RPUSimple<T>::backwardIndexed(
    const T *D_input, T *X_output, int total_output_size, int m_batch, int dim3, bool trans) {
  // -- EXPECTS backward index to be set properly !!
  // -- total_output_size is size of X_output
  // -- bias is handled within the indeces

  T *x_tensor = nullptr;
  T *d_tensor = nullptr;

  this->getTensorBuffer(&x_tensor, &d_tensor, m_batch, dim3);

  if ((dim3 == 1) || (!trans)) {
    this->backwardMatrix(D_input, x_tensor, m_batch * dim3, trans, trans);
  } else {
    this->permute132(d_tensor, D_input, m_batch, this->getDSize(), dim3, false);
    this->backwardMatrix(d_tensor, x_tensor, m_batch * dim3, trans, trans);
  }

  // need to set to zero
  this->setZero(X_output, total_output_size);

  this->copyIndexedOutput(
      X_output, x_tensor, total_output_size, this->getMatrixIndices(), this->getXSize(), m_batch,
      dim3, trans);
}

template <typename T>
void RPUSimple<T>::backwardIndexedSlice(
    const T *D_input,
    T *X_output,
    int total_output_size,
    int m_batch,
    int dim3,
    bool trans,
    int m_batch_slice,
    const int *batch_indices) {
  T *x_tensor = nullptr;
  T *d_tensor = nullptr;

  this->getTensorBuffer(&x_tensor, &d_tensor, m_batch_slice, dim3);

  this->copySliceInput(
      d_tensor, D_input, this->getDSize(), m_batch, dim3, trans, m_batch_slice, batch_indices);

  this->backwardMatrix(d_tensor, x_tensor, m_batch_slice * dim3, trans, trans);

  this->copyIndexedOutput(
      X_output, x_tensor, total_output_size, this->getMatrixIndices(), this->getXSize(), m_batch,
      dim3, trans, m_batch_slice, batch_indices);
}

template <typename T>
void RPUSimple<T>::updateIndexed(
    const T *X_input, const T *D_input, int total_x_input_size, int m_batch, int dim3, bool trans) {
  T *x_tensor = nullptr;
  T *d_tensor = nullptr;

  this->getTensorBuffer(&x_tensor, &d_tensor, m_batch, dim3);

  this->copyIndexedInput(
      x_tensor, X_input, total_x_input_size, this->getMatrixIndices(), this->getXSize(), m_batch,
      dim3, trans);
  if (trans) {
    this->permute132(d_tensor, D_input, m_batch, this->getDSize(), dim3, false);
    this->update(x_tensor, d_tensor, false, m_batch * dim3, trans, trans);
  } else {
    this->update(x_tensor, D_input, false, m_batch * dim3, trans, trans);
  }
}

template <typename T>
void RPUSimple<T>::updateIndexedSlice(
    const T *X_input,
    const T *D_input,
    int total_x_input_size,
    int m_batch,
    int dim3,
    bool trans,
    int m_batch_slice,
    const int *batch_indices) {

  T *x_tensor = nullptr;
  T *d_tensor = nullptr;

  this->getTensorBuffer(&x_tensor, &d_tensor, m_batch_slice, dim3);

  this->copyIndexedInput(
      x_tensor, X_input, total_x_input_size, this->getMatrixIndices(), this->getXSize(), m_batch,
      dim3, trans, m_batch_slice, batch_indices);

  this->copySliceInput(
      d_tensor, D_input, this->getDSize(), m_batch, dim3, trans, m_batch_slice, batch_indices);

  this->update(x_tensor, d_tensor, false, m_batch_slice * dim3, trans, trans);
}

/*********************************************************************************/
/* delayed update using weight buffer*/

template <typename T> void RPUSimple<T>::copyWeightsFromBuffer() {
  RPU::math::copy<T>(
      this->x_size_ * this->d_size_, this->getWeightsBuffer()[0], 1, this->getWeightsPtr()[0], 1);
}

template <typename T> void RPUSimple<T>::copyWeightsToBuffer() {
  RPU::math::copy<T>(
      this->x_size_ * this->d_size_, this->getWeightsPtr()[0], 1, this->getWeightsBuffer()[0], 1);
}

template <typename T> void RPUSimple<T>::applyDelayedWeights() {
  if (!getPar().use_delayed_update) {
    RPU_FATAL("Applying delayed weights can only be used when use_delayed_weights is turned on.");
  }
  if (this->getDeltaWeights()) {
    RPU_FATAL("Applying delayed weights while using external weight update is not possible.");
  }

  if (!use_delayed_update_ && getPar().use_delayed_update) {
    // first time, copy weights to buffer
    use_delayed_update_ = true;
    this->copyWeightsToBuffer();
  } else {
    this->copyWeightsFromBuffer();
  }
}

/*********************************************************************************/
/* Set/Get weights related*/

template <typename T> void RPUSimple<T>::setWeightsUniformRandom(T min_value, T max_value) {
  T **w = this->getWeightsPtr();
  for (int j = 0; j < this->x_size_; ++j) {
    for (int i = 0; i < this->d_size_; ++i) {
      w[i][j] = rng_->sampleUniform(min_value, max_value);
    }
  }
}

template <typename T> void RPUSimple<T>::setWeights(const T *weightsptr) {
  T *w = this->getWeightsPtr()[0];
  if (weightsptr != w) {
    int size = this->d_size_ * this->x_size_;
    memcpy(w, weightsptr, size * sizeof(T));
  }
}

template <typename T> void RPUSimple<T>::setWeightsWithAlpha(const T *weightsptr, T assumed_wmax) {

  if (assumed_wmax <= (T)0.0) {
    this->setWeights(weightsptr);
  } else {
    DEBUG_OUT("WARNING: scaling weights with ALPHA scale!");
    int sz = this->x_size_ * this->d_size_;
    T *w = new T[sz];
    memcpy(w, weightsptr, sz * sizeof(T));

    int imax = RPU::math::iamax<T>(sz, w, 1);
    T alpha = (T)fabsf(w[imax] / assumed_wmax);
    RPU::math::scal<T>(sz, (T)1.0 / alpha, w, 1);

    this->setAlphaScale(alpha);
    this->setWeights(w);

    delete[] w;
  }
}

namespace detail {
template <typename T> void combine_wb(T *wb, const T *w, const T *b, int xsz, int dsz) {
  int k = 0;
  for (int i = 0; i < dsz; i++) {
    for (int j = 0; j < xsz - 1; j++) {
      wb[j + xsz * i] = w[k++];
    }
  }
  for (int i = 0; i < dsz; i++) {
    wb[xsz * (i + 1) - 1] = b[i];
  }
}
} // namespace detail

template <typename T>
void RPUSimple<T>::setWeightsAndBiasWithAlpha(
    const T *weightsptr, const T *biasptr, T assumed_wmax, bool real_if, int n_loops) {
  // expects weights in row major order (x_size-1 then d_size).
  // The last d_size_ column is given in the biasptr

  int sz = this->x_size_ * this->d_size_;
  T *w = new T[sz];
  detail::combine_wb(w, weightsptr, biasptr, this->x_size_, this->d_size_);

  if (assumed_wmax > (T)0.0) {
    DEBUG_OUT("WARNING: scaling weights with ALPHA scale!");
    int imax = RPU::math::iamax<T>(sz, w, 1);
    T alpha = (T)fabsf(w[imax] / assumed_wmax);
    RPU::math::scal<T>(sz, (T)1.0 / alpha, w, 1);
    this->setAlphaScale(alpha);
  }

  if (real_if) {
    this->setWeightsReal(w, n_loops);
  } else {
    this->setWeights(w);
  }

  delete[] w;
}

template <typename T>
void RPUSimple<T>::setWeightsAndBias(
    const T *weightsptr, const T *biasptr, bool real_if, int n_loops) {
  this->setWeightsAndBiasWithAlpha(weightsptr, biasptr, -1.0, real_if, n_loops);
}

template <typename T> void RPUSimple<T>::setSharedWeights(T *weightsptr) {
  if (!shared_weights_if_) {
    this->getWeights(weightsptr); // copy existing weights to given workspace.
    delete[] *weights_;           // delete allocated memory array but not the pointer
  }
  *weights_ = weightsptr;
  shared_weights_if_ = true;
  if ((this->d_size_ > 0) &&
      (weights_[this->d_size_ - 1] != *weights_ + (size_t)this->x_size_ * this->d_size_)) {
    // only set if changed
    for (int i = 0; i < this->d_size_; ++i) {
      weights_[i] = *weights_ + (size_t)this->x_size_ * i;
    }
  }
}

template <typename T> T RPUSimple<T>::getAlphaLearningRate() const {
  return (getDeltaWeights()) ? -(T)1.0 / this->bwd_alpha_
                             : this->getLearningRate() / this->bwd_alpha_;
}

template <typename T> void RPUSimple<T>::setFwdAlpha(const T fwd_alpha, bool with_noise) {
  if (with_noise) {
    fwd_alpha_ = fwd_alpha * ((T)1.0 + this->getPar().alpha_std * rw_rng_->sampleGauss());
    DEBUG_OUT("Forward alpha set to: " << fwd_alpha_);
    alpha_warning();
  } else {
    fwd_alpha_ = fwd_alpha;
  }
}

template <typename T> void RPUSimple<T>::setBwdAlpha(const T bwd_alpha, bool with_noise) {
  if (with_noise) {
    bwd_alpha_ = bwd_alpha * ((T)1.0 + this->getPar().alpha_std * rw_rng_->sampleGauss());
    DEBUG_OUT("Backward alpha set to: " << bwd_alpha_);
    alpha_warning();
  } else {
    bwd_alpha_ = bwd_alpha;
  }
}

template <typename T> void RPUSimple<T>::setAlphaScale(const T alpha) {
  setFwdAlpha(alpha);
  setBwdAlpha(alpha);
}

template <typename T>
void RPUSimple<T>::getAndResetWeightUpdate(T *prev_weight_and_dw_out, T scale) {

  T *w = this->getWeightsPtr()[0];
  int size = this->d_size_ * this->x_size_;
  PRAGMA_SIMD
  for (int i = 0; i < size; ++i) {
    T w_old = prev_weight_and_dw_out[i];
    prev_weight_and_dw_out[i] = scale * (w[i] - w_old);
    w[i] = w_old;
  }
}

template <typename T> void RPUSimple<T>::applyWeightUpdate(T *dw_and_current_weight_out) {
  T *w = this->getWeightsPtr()[0];
  int size = this->d_size_ * this->x_size_;
  PRAGMA_SIMD
  for (int i = 0; i < size; ++i) {
    T dw = dw_and_current_weight_out[i];
    w[i] += dw;
    dw_and_current_weight_out[i] = w[i];
  }
}

template <typename T> void RPUSimple<T>::getWeights(T *weightsptr) const {
  T *w = this->getWeightsPtr()[0];
  memcpy(weightsptr, w, sizeof(T) * this->x_size_ * this->d_size_);
}

template <typename T> void RPUSimple<T>::setDeltaWeights(T *dw_extern) {
  // need to wrap around a 2D array. WILL BE SLOWER THAN IT SHOULD BE
  // TODO: get rid of the 2D array legacy for CPU... not needed anyway
  if (dw_extern) {

    if (dw_extern == delta_weights_extern_[0]) {
      // make no op if already set
      return;
    }

    ENFORCE_NO_DELAYED_UPDATE;

    delta_weights_extern_[0] = dw_extern;
    for (size_t i = 0; i < (size_t)this->d_size_; i++) {
      delta_weights_extern_[i] = delta_weights_extern_[0] + (size_t)this->x_size_ * i;
    }
  } else {
    // caution: only first element is set to nullptr. others
    // are still pointing somewhere... should not matter
    // though as first is tested only
    delta_weights_extern_[0] = nullptr;
  }
}

template <typename T> T **RPUSimple<T>::getFBWeights(bool is_test) const {
  bool use_fb =
      fb_weights_ && (!is_test || (fb_weight_modifier_ && fb_weight_modifier_->enableDuringTest()));
  return use_fb ? fb_weights_ : weights_;
}

template <typename T> T **RPUSimple<T>::getUpWeights() {
  // This is called from the Update routines to check which weight
  // is used for calculation. If dw is defined, then it will use the
  // DW mode, meaning that it will write into delta_weights the DW
  // and keep the weights. For HW RPU models that might include
  // first using W and then writing the difference to dW
  //
  // In case of delayed update is also returns the buffered weights
  // instead of the "actual" weight. Combination of external weights
  // and buffers are not possible

  if (use_delayed_update_) {
    return getWeightsBuffer();
  } else {

    T **dw = delta_weights_extern_.data();
    return (dw[0] != nullptr) ? dw : this->getWeightsPtr();
  }
}

template <typename T> T RPUSimple<T>::getUpBeta() const {
  return (this->getDeltaWeights() == nullptr) ? (T)1.0 : (T)0.0;
}

/********************************************************************************/
/* Utilities */

template <typename T> void RPUSimple<T>::setRandomSeed(unsigned int seed) {
  DEBUG_OUT("Simple: Set seed.");
  rng_->setSeed((randomint_t)seed);
  rw_rng_->setSeed((randomint_t)seed);
}

template <typename T> void RPUSimple<T>::printWeights(int x_count, int d_count) {

  if (x_count < 0 || x_count > this->x_size_)
    x_count = this->x_size_;

  if (d_count < 0 || d_count > this->d_size_)
    d_count = this->d_size_;

  for (int i = 0; i < d_count; ++i) {
    for (int j = 0; j < x_count; ++j) {
      std::cout << this->getWeightsPtr()[i][j] << ",";
    }
    std::cout << std::endl;
  }
}

template <typename T> void RPUSimple<T>::printToStream(std::stringstream &ss) const {
  ss << "RPUSimple<" << this->getDataTypeName() << ">(" << this->d_size_ << "," << this->x_size_
     << ")" << std::endl;
}

template <typename T> void RPUSimple<T>::printParametersToStream(std::stringstream &ss) const {
  getPar().printToStream(ss);
}

template <typename T> void RPUSimple<T>::setLearningRate(T lr) {
  if (lr != this->getLearningRate()) {
    RPUAbstract<T>::setLearningRate(lr);
  }
}

/*********************************************************************************/
/* other weight operators*/

template <typename T> void RPUSimple<T>::decayWeights(T alpha, bool bias_no_decay) {

  T lifetime = getPar().lifetime;
  T decay_rate = (lifetime > (T)1.0) ? ((T)1.0 / lifetime) : (T)0.0;
  T decay_scale = (T)1.0 - alpha * decay_rate;

  if (decay_scale > (T)0.0 && decay_scale < (T)1.0) {
    if (!bias_no_decay) {
      RPU::math::scal<T>(this->x_size_ * this->d_size_, decay_scale, this->getWeightsPtr()[0], 1);
    } else {
      int size = this->d_size_ * this->x_size_;
      T *w = this->getWeightsPtr()[0];
      const int last_col = this->x_size_ - 1; // x-major (ie row major)
      PRAGMA_SIMD
      for (int i = 0; i < size; ++i) {
        w[i] *= (i % this->x_size_ == last_col) ? (T)1.0 : decay_scale;
      }
    }
  }
}

template <typename T> void RPUSimple<T>::decayWeights(bool bias_no_decay) {
  RPUSimple<T>::decayWeights(1., bias_no_decay);
}

template <typename T> void RPUSimple<T>::driftWeights(T time_since_last_call) {
  if (!wdrifter_) {
    wdrifter_ = RPU::make_unique<WeightDrifter<T>>(
        this->x_size_ * this->d_size_, getPar().drift); // simpleDrift
  }
  wdrifter_->apply(this->getWeightsPtr()[0], time_since_last_call, *rng_);
}

template <typename T> void RPUSimple<T>::clipWeights(T clip) {

  if (clip >= (T)0.0) {
    int size = this->d_size_ * this->x_size_;
    T *w = this->getWeightsPtr()[0];
    PRAGMA_SIMD
    for (int i = 0; i < size; ++i) {
      w[i] = MIN(MAX(w[i], -clip), clip);
    }
  }
}

template <typename T> void RPUSimple<T>::clipWeights(const WeightClipParameter &wclpar) {

  if (wclipper_ == nullptr) {
    wclipper_ = RPU::make_unique<WeightClipper<T>>(this->x_size_, this->d_size_);
  }
  wclipper_->apply(getWeightsPtr()[0], wclpar);
}

template <typename T> void RPUSimple<T>::diffuseWeights() {

  T diffusion = getPar().diffusion;
  if (diffusion > (T)0.0) {
    int size = this->d_size_ * this->x_size_;
    T *w = this->getWeightsPtr()[0];
    PRAGMA_SIMD
    for (int i = 0; i < size; ++i) {
      w[i] += diffusion * rng_->sampleGauss();
    }
  }
}

/*********************************************************************************/
/*********************************************************************************/
template <typename T> uint64_t *RPUSimple<T>::initFlickerStates() {

  ENFORCE_NO_DELAYED_UPDATE;

  const auto &flicker = getPar().flicker;

  if (flicker.n > 64) {
    RPU_FATAL("only <=64 flicker supported");
  }

  int size = this->d_size_ * this->x_size_;

  flicker_states_.resize(size);
  for (int j = 0; j < size; j++) {
    for (int i = 0; i < flicker.n; i++) {
      flicker_states_[j] ^=
          (rng_->sampleUniform() < flicker.q) ? ((uint64_t)1 << (uint64_t)i) : (uint64_t)0;
    }
  }
  flicker_probs_.resize(flicker.n);
  flicker_probs_[0] = flicker.q; // flicker.q is 0->1
  for (int i = 1; i < flicker.n; i++) {
    flicker_probs_[i] = flicker_probs_[i - 1] / flicker.r;
  }

  if (flicker.wreset) {
    // delayed anyway not supported for pink. use weight buffer
    this->copyWeightsToBuffer();
  }

  return flicker_states_.data();
}

template <typename T> void RPUSimple<T>::diffuseWeightsPink() {

  ENFORCE_NO_DELAYED_UPDATE;

  T diffusion = getPar().diffusion;
  if (diffusion <= (T)0.0) {
    return;
  }

  const auto &flicker = getPar().flicker;
  int size = this->d_size_ * this->x_size_;

  if (flicker_states_.size() == (size_t)0) {
    initFlickerStates();
  }

  T amp = (T)sqrtf((T)logf(flicker.r) * diffusion / (flicker.q * ((T)1.0 - flicker.q)));

  T *w = this->getWeightsPtr()[0];
  T *wb = this->getWeightsBuffer()[0];
  T sc = ((T)1.0 - flicker.q) / flicker.q;
  T H = -flicker.h / flicker.q;
  const uint64_t zero = 0;
  const uint64_t one = 1;

  PRAGMA_SIMD
  for (int j = 0; j < size; j++) {
    uint64_t fls = flicker_states_[j];
#ifdef _MSC_VER
    T last_noise_value = ((T)__popcnt64(fls));
#else
    T last_noise_value = ((T)__builtin_popcountll(fls));
#endif
    if (flicker.wreset && (T)fabsf(w[j] - wb[j]) > flicker.wreset_tol) {

      if (flicker.h > (T)0.0) {
        // only reset 1's with prob 1-exp(-h/r^i)
        for (int i = 0; i < flicker.n; i++) {
          uint64_t cbit = one << (uint64_t)i;
          fls ^= ((fls & cbit) > 0)
                     ? ((rng_->sampleUniform() >= (T)expf(H * flicker_probs_[i])) ? cbit : zero)
                     : zero; // use >= to get 1-exp
        }
      } else {
        // reset all with eq
        fls = 0;
        for (int i = 0; i < flicker.n; i++) {
          fls ^= (rng_->sampleUniform() < flicker.q) ? (one << (uint64_t)i) : zero;
        }
      }
    } else {
      for (int i = 0; i < flicker.n; i++) {
        T prob = flicker_probs_[i]; // based on flicker.q
        uint64_t cbit = one << (uint64_t)i;
        T stoch_value = rng_->sampleUniform();
        fls ^= ((fls & cbit) > 0) ? ((stoch_value < prob * sc) ? cbit : zero)
                                  :                      // 1->0  therefore change prop to q1
                   ((stoch_value < prob) ? cbit : zero); // 0->1	q0
      }
    }
#ifdef _MSC_VER
    T noise_value = ((T)__popcnt64(fls));
#else
    T noise_value = ((T)__builtin_popcountll(fls));
#endif
    w[j] += amp * (noise_value - last_noise_value);
    flicker_states_[j] = fls;
    if (flicker.wreset) {
      wb[j] = w[j];
    }
  }
}

template <typename T>
bool RPUSimple<T>::swaWeights(
    const WeightRemapParameter &wrmpar, T *swa_weights, uint64_t iter, T *scales, T *biases) {
  if (wremapper_ == nullptr) {
    wremapper_ = RPU::make_unique<WeightRemapper<T>>(this->x_size_, this->d_size_);
  }
  return wremapper_->applySWA(
      swa_weights, getWeightsPtr()[0], iter, wrmpar, this->getAlphaLearningRate(), scales, biases);
}

template <typename T>
void RPUSimple<T>::remapWeights(const WeightRemapParameter &wrmpar, T *scales, T *biases) {

  if (wremapper_ == nullptr) {
    wremapper_ = RPU::make_unique<WeightRemapper<T>>(this->x_size_, this->d_size_);
  }
  wremapper_->apply(getWeightsPtr()[0], this->getAlphaLearningRate(), wrmpar, scales, biases);
}

template <typename T> void RPUSimple<T>::modifyFBWeights(const WeightModifierParameter<T> &wmpar) {

  if (fb_weights_ == nullptr) {
    fb_weights_ = Array_2D_Get<T>(this->d_size_, this->x_size_);
    fb_weight_modifier_ = RPU::make_unique<WeightModifier<T>>(this->x_size_, this->d_size_);
  }

  // modify FB weights
  fb_weight_modifier_->apply(fb_weights_[0], this->getWeightsPtr()[0], wmpar);
}

/*********************************************************************************/

template class RPUSimple<float>;
#ifdef RPU_USE_DOUBLE
template class RPUSimple<double>;
#endif
#ifdef RPU_USE_FP16
template class RPUSimple<half_t>;
#endif

} // namespace RPU
