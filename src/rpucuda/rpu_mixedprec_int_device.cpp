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

#include "rpu_mixedprec_int_device.h"
#include "math_util.h"
#include "utility_functions.h"
#include <algorithm>
#include <memory>
#include <sstream>

namespace RPU {

/******************************************************************************************/
/* MixedPrecIntRPUDeviceMetaParameter*/

template <typename T>
void MixedPrecIntRPUDeviceMetaParameter<T>::printToStream(std::stringstream &ss) const {

  ss << "\t\bUpdate with accumulation on Chi in integers and momentum quantization: " << std::endl;
  ss << "\t momentum_chi: \t";
  ss << momentum_chi << std::endl;
  ss << "\t momentum_nm: \t";
  ss << momentum_nm << std::endl;

  ss << "\t n_x_bins: \t";
  ss << n_x_bins << std::endl;
  ss << "\t n_d_bins: \t";
  ss << n_d_bins << std::endl;
  if (stoc_round_x || stoc_round_d) {
    ss << "\t stoc_round (x / d): \t";
    ss << std::boolalpha << stoc_round_x << " / " << stoc_round_d << std::endl;
  }
  MixedPrecRPUDeviceBaseMetaParameter<T>::printToStream(ss);
}

template <typename T> void MixedPrecIntRPUDeviceMetaParameter<T>::initialize() {
  if (!this->_par_initialized) {
    MixedPrecRPUDeviceBaseMetaParameter<T>::initialize();

    if (n_x_bins <= 2 || n_d_bins <= 2) {
      RPU_FATAL("Expect n_bins to be at least 3.");
    }

    if (momentum_chi > (T)1.0 || momentum_nm > (T)1.0 || momentum_nm < (T)0.0 ||
        momentum_chi < (T)0.0) {
      RPU_FATAL("momentum should be in 0..1");
    }
  }
}

template struct MixedPrecIntRPUDeviceMetaParameter<float>;
#ifdef RPU_USE_DOUBLE
template struct MixedPrecIntRPUDeviceMetaParameter<double>;
#endif
#ifdef RPU_USE_FP16
template struct MixedPrecIntRPUDeviceMetaParameter<half_t>;
#endif

/******************************************************************************************/

template <typename T> void MixedPrecIntRPUDevice<T>::initialize(int x_sz, int d_sz) {

  freeContainers();
  chi_ = Array_2D_Get<int32_t>(d_sz, x_sz);

  PRAGMA_SIMD
  for (int i = 0; i < x_sz * d_sz; i++) {
    chi_[0][i] = 0;
  }

  md_ = -1.0;
  mx_ = -1.0f;
}

template <typename T> void MixedPrecIntRPUDevice<T>::freeContainers() {
  if (chi_ != nullptr) {
    Array_2D_Free<int32_t>(chi_);
    chi_ = nullptr;
  }
}

// dtor
template <typename T> MixedPrecIntRPUDevice<T>::~MixedPrecIntRPUDevice() { freeContainers(); }

// ctor
template <typename T>
MixedPrecIntRPUDevice<T>::MixedPrecIntRPUDevice(int x_sz, int d_sz)
    : MixedPrecRPUDeviceBase<T>(x_sz, d_sz) {}

template <typename T>
MixedPrecIntRPUDevice<T>::MixedPrecIntRPUDevice(
    int x_sz, int d_sz, const MixedPrecIntRPUDeviceMetaParameter<T> &par, RealWorldRNG<T> *rng)
    : MixedPrecIntRPUDevice<T>(x_sz, d_sz) {
  populate(par, rng);
}

// copy construcutor
template <typename T>
MixedPrecIntRPUDevice<T>::MixedPrecIntRPUDevice(const MixedPrecIntRPUDevice<T> &other)
    : MixedPrecRPUDeviceBase<T>(other) {

  initialize(other.x_size_, other.d_size_);

  if (other.chi_ != nullptr) {
    for (int i = 0; i < other.d_size_ * other.x_size_; ++i) {
      chi_[0][i] = other.chi_[0][i];
    }
  }
  mx_ = other.mx_;
  md_ = other.md_;
  qx_ = other.qx_;
  qx_index_ = other.qx_index_;
}

// copy assignment
template <typename T>
MixedPrecIntRPUDevice<T> &
MixedPrecIntRPUDevice<T>::operator=(const MixedPrecIntRPUDevice<T> &other) {

  MixedPrecIntRPUDevice<T> tmp(other);
  swap(*this, tmp);
  return *this;
}

// move constructor
template <typename T>
MixedPrecIntRPUDevice<T>::MixedPrecIntRPUDevice(MixedPrecIntRPUDevice<T> &&other) noexcept {
  *this = std::move(other);
}

// move assignment
template <typename T>
MixedPrecIntRPUDevice<T> &
MixedPrecIntRPUDevice<T>::operator=(MixedPrecIntRPUDevice<T> &&other) noexcept {
  MixedPrecRPUDeviceBase<T>::operator=(std::move(other));

  chi_ = std::move(other.chi_);
  mx_ = other.mx_;
  md_ = other.md_;
  qx_ = other.qx_;
  qx_index_ = other.qx_index_;

  return *this;
}

/*********************************************************************************/
/* populate */

template <typename T>
void MixedPrecIntRPUDevice<T>::populate(
    const MixedPrecIntRPUDeviceMetaParameter<T> &p, RealWorldRNG<T> *rng) {

  MixedPrecRPUDeviceBase<T>::populate(p, rng);
  initialize(this->x_size_, this->d_size_);
}

template <typename T>
void MixedPrecIntRPUDevice<T>::dumpExtra(RPU::state_t &extra, const std::string prefix) {
  MixedPrecRPUDeviceBase<T>::dumpExtra(extra, prefix);

  RPU::state_t state;

  RPU::insert(state, "md", md_);
  RPU::insert(state, "mx", mx_);

  RPU::insertWithPrefix(extra, state, prefix);
}

template <typename T>
void MixedPrecIntRPUDevice<T>::loadExtra(
    const RPU::state_t &extra, const std::string prefix, bool strict) {
  MixedPrecRPUDeviceBase<T>::loadExtra(extra, prefix, strict);

  auto state = RPU::selectWithPrefix(extra, prefix);

  RPU::load(state, "md", md_, strict);
  RPU::load(state, "mx", mx_, strict);
}

/*********************************************************************************/
/* transfer */

template <typename T>
void MixedPrecIntRPUDevice<T>::forwardUpdate(
    T **weights,
    const T lr,
    int j_row_start,
    const T *transfer_d_vec,
    const int n_vec,
    const bool trans) {

  if (!lr) { // not used actually
    return;
  }

  if (trans) { // not needed really
    RPU_NOT_IMPLEMENTED;
  }

  const auto &par = getPar();
  if (this->transfer_tmp_.size() < (size_t)this->x_size_) {
    this->transfer_tmp_.resize(this->x_size_);
  }
  T d_width = md_ / (T)(par.n_d_bins / 2);
  T x_width = mx_ / (T)(par.n_x_bins / 2); // needs to be integer div
  T momentum = par.momentum_chi;
  T thres = MAX((T)roundf(this->granularity_ / (T)fabsf(lr) / d_width / x_width), (T)1.0);

  // forward / update
  for (size_t j = 0; j < (size_t)n_vec; j++) {
    int32_t *chi_row = chi_[j_row_start + j];

    PRAGMA_SIMD
    for (size_t i = 0; i < (size_t)this->x_size_; i++) {
      T value = (T)chi_row[i];
      T dw = (T)truncf(value / thres);
      this->transfer_tmp_[i] = dw;
      chi_row[i] = (int32_t)value - (int32_t)roundf(((T)1.0 - momentum) * thres * dw);
    }

    this->transfer_pwu_->updateVectorWithDevice(
        weights, this->transfer_tmp_.data(), 1, transfer_d_vec + (size_t)this->d_size_ * j, 1,
        this->granularity_, n_vec, &*this->rpu_device_);
  }
}

/*********************************************************************************/
/* update */

template <typename T>
void MixedPrecIntRPUDevice<T>::doDirectVectorUpdate(
    T **weights,
    const T *x_input,
    const int x_inc,
    const T *d_input,
    const int d_inc,
    const T learning_rate,
    const int m_batch_info,
    const PulsedUpdateMetaParameter<T> &up) {

  this->setUpPar(up);
  const auto &par = getPar();

  int max_index = RPU::math::iamax<T>(this->x_size_, x_input, x_inc);
  T x_amax = (T)fabsf(x_input[max_index * x_inc]);

  max_index = RPU::math::iamax<T>(this->d_size_, d_input, d_inc);
  T d_amax = (T)fabsf(d_input[max_index * d_inc]);

  T momentum = md_ < (T)0.0 ? (T)0.0 : par.momentum_nm;
  if (d_amax > (T)0.0) {
    md_ = momentum * md_ + ((T)1.0 - momentum) * d_amax;
  }
  if (x_amax > (T)0.0) {
    mx_ = momentum * mx_ + ((T)1.0 - momentum) * x_amax;
  }
  // this is floored
  int16_t half_x_bins = (int16_t)(par.n_x_bins / 2);
  int16_t half_d_bins = (int16_t)(par.n_d_bins / 2);

  T x_width = mx_ / ((T)par.n_x_bins / (T)2.0);
  T d_width = md_ / ((T)par.n_d_bins / (T)2.0);

  int i_stop = this->x_size_ * x_inc;
  int i = -1;
  int kx = 0;

  qx_.resize(this->x_size_);
  qx_index_.resize(this->x_size_);
  const bool stochastic_rounding_x = par.stoc_round_x;
  T stoch_value = 0.0;
  // sparse outer product
  PRAGMA_SIMD
  for (int i_x = 0; i_x < i_stop; i_x += x_inc) {
    T x = x_input[i_x];
    i++;

    if (stochastic_rounding_x) {
      stoch_value = this->rng_.sampleUniform() - (T)0.5;
    }

    // quantize
    int16_t qx = (int16_t)roundf(x / x_width + stoch_value);

    if (qx == (int16_t)0) {
      continue;
    }
    qx_index_[kx] = (int16_t)i;
    qx_[kx++] = MIN(MAX(qx, -half_x_bins), half_x_bins);
  }

  int kd = 0;
  int j = -1;
  int j_stop = this->d_size_ * d_inc;
  const bool stochastic_rounding_d = par.stoc_round_d;
  stoch_value = 0.0;
  for (int j_d = 0; j_d < j_stop; j_d += d_inc) {
    T d = d_input[j_d];
    j++;

    if (stochastic_rounding_d) {
      stoch_value = this->rng_.sampleUniform() - (T)0.5;
    }

    // quantize
    int16_t qd = (int16_t)roundf(d / d_width + stoch_value);
    if (qd == (int16_t)0) {
      continue;
    }
    qd = MIN(MAX(qd, -half_d_bins), half_d_bins);
    int32_t *chi_row = chi_[j];
    kd++;
    PRAGMA_SIMD
    for (int ii = 0; ii < kx; ii++) {
      int16_t idx = qx_index_[ii];
      int16_t qx = qx_[ii];
      chi_row[idx] += qd * qx;
    }
  }

  this->doTransfer(weights, learning_rate, m_batch_info);
  this->computeSparsity(kx, kd);
  this->advanceUpdateCounter();
}

template <typename T> bool MixedPrecIntRPUDevice<T>::onSetWeights(T **weights) {

  // reset chi
  initialize(this->x_size_, this->d_size_);
  MixedPrecRPUDeviceBase<T>::onSetWeights(weights);

  return true; // modified device thus true
}

template <typename T> void MixedPrecIntRPUDevice<T>::getChi(T *data) const {
  for (int i = 0; i < this->size_; ++i) {
    data[i] = (T)chi_[0][i];
  }
}

template <typename T> void MixedPrecIntRPUDevice<T>::setChi(const T *data) {
  for (int i = 0; i < this->size_; ++i) {
    chi_[0][i] = (int16_t)roundf(data[i]);
  }
}

template class MixedPrecIntRPUDevice<float>;
#ifdef RPU_USE_DOUBLE
template class MixedPrecIntRPUDevice<double>;
#endif
#ifdef RPU_USE_FP16
template class MixedPrecIntRPUDevice<half_t>;
#endif

} // namespace RPU
