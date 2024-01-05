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

#include "rpu_chopped_transfer_device.h"
#include "math_util.h"
#include "utility_functions.h"
#include <algorithm>
#include <memory>
#include <sstream>

namespace RPU {

/******************************************************************************************/
/* ChoppedTransferRPUDeviceMetaParameter*/

template <typename T>
void ChoppedTransferRPUDeviceMetaParameter<T>::printToStream(std::stringstream &ss) const {

  ss << "\t in_chop_prob:\t\t";
  ss << in_chop_prob;
  if (in_chop_random) {
    ss << "\t [random]";
  } else {
    ss << "\t [regular]";
  }

  ss << std::endl;

  ss << "\t out_chop_prob:\t\t" << out_chop_prob << std::endl;

  ss << "\t auto_scale:\t\t";
  ss << std::boolalpha << auto_scale;
  ss << std::endl;

  if (correct_gradient_magnitudes) {
    ss << "\t [correct gradient magnitudes]";
  }
  ss << std::endl;

  ss << "\t auto_momentum:\t\t" << auto_momentum << std::endl;

  ss << "\t auto_momentum:\t\t" << auto_momentum << std::endl;

  if (no_buffer) {
    ss << "\t buffer not used.";
    ss << std::endl;
  } else {
    if (buffer_granularity > (T)0.0) {
      ss << "\t buffer_granularity:\t";
      ss << buffer_granularity;
      ss << std::endl;
    }
    if (auto_granularity > (T)0.0) {
      ss << "\t auto_granularity:\t" << auto_granularity;
      ss << std::endl;
    }
  }

  BufferedTransferRPUDeviceMetaParameter<T>::printToStream(ss);
};

template <typename T> void ChoppedTransferRPUDeviceMetaParameter<T>::checkSupported() const {

  // NOTE: will also use one-hot transfer always (and ignore the
  // transfer_vecs). This is the default currently anyway

  if (!this->singleDeviceUpdate()) {
    RPU_FATAL("Multiple device update not supported for Chopped Transfer Device");
  }

  if (!this->same_context) {
    RPU_FATAL("Only same context supported");
  }

  if (!this->fullyHidden()) {
    RPU_FATAL("Expects a fully hidden fast device.");
  }

  if ((this->n_reads_per_transfer != 1) || (this->random_selection != false) ||
      (this->with_reset_prob > (T)0.0)) {

    RPU_FATAL("In / out chopper not implemented the given parameters. \nRequired: "
              "n_devices==2, n_reads_per_transfer==1, random_selection=false).\n");
  }
}

template <typename T>
void ChoppedTransferRPUDeviceMetaParameter<T>::updateAutoScale(T &m, T new_val, int m_batch) const {
  if (m <= (T)0.0) {
    m = new_val;
    return;
  }
  if (new_val <= (T)0.0) {
    return;
  }
  T tau = ((T)1.0 - auto_momentum) / (T)m_batch;

  m = ((T)1.0 - tau) * m + tau * new_val;
}

template <typename T>
T ChoppedTransferRPUDeviceMetaParameter<T>::getWriteLR(T weight_granularity) const {
  return this->step * weight_granularity;
}

template <typename T>
T ChoppedTransferRPUDeviceMetaParameter<T>::getTransferLRScale(
    T from_weight_granularity, T to_weight_granularity, T lr, T count_lr, int current_m_batch)
    const {
  T bg = getBufferGranularity(from_weight_granularity, current_m_batch);
  T lr_scale;
  if (correct_gradient_magnitudes) {
    // needs to divide by count_lr
    bg *= to_weight_granularity / from_weight_granularity;
    lr_scale = (T)fabsf(lr) / count_lr / bg;
  } else {
    lr_scale = (T)fabsf(lr) / bg;
  }
  // RPU_INFO("LR-C scale: " << lr_scale);
  return lr_scale;
}

template <typename T>
T ChoppedTransferRPUDeviceMetaParameter<T>::getAutoTransferEvery(
    T n_states, const PulsedUpdateMetaParameter<T> &up) const {
  UNUSED(up);
  return (T)n_states / (T)this->getInSize() * (T)fabsf(this->transfer_every);
}

template <typename T>
T ChoppedTransferRPUDeviceMetaParameter<T>::getBufferGranularity(
    T weight_granularity, int current_m_batch) const {
  UNUSED(current_m_batch);
  T bg = buffer_granularity > (T)0.0 ? buffer_granularity : (T)1.0;
  if (auto_granularity > (T)0.0) {
    T period = (T)this->getInSize() * (T)fabsf(this->transfer_every);
    bg *= weight_granularity * auto_granularity / period;
  } else {
    bg *= weight_granularity;
  }
  return bg;
}

template <typename T>
T ChoppedTransferRPUDeviceMetaParameter<T>::getPulseCountAutoLR(
    T m_x,
    T m_d,
    T d_sparsity,
    T weight_granularity,
    T transfer_every,
    const PulsedUpdateMetaParameter<T> &up) const {
  UNUSED(d_sparsity);
  UNUSED(transfer_every);

  T count_lr;

  T g = (T)up.desired_BL * weight_granularity;
  count_lr = this->fast_lr * g;

  if ((m_x > (T)0.0) && (m_d > (T)0.0)) {
    count_lr /= m_x * m_d;
  }

  return count_lr;
}

template struct ChoppedTransferRPUDeviceMetaParameter<float>;
#ifdef RPU_USE_DOUBLE
template struct ChoppedTransferRPUDeviceMetaParameter<double>;
#endif
#ifdef RPU_USE_FP16
template struct ChoppedTransferRPUDeviceMetaParameter<half_t>;
#endif

/******************************************************************************************/
// dtor
template <typename T> ChoppedTransferRPUDevice<T>::~ChoppedTransferRPUDevice() {}

// ctor
template <typename T>
ChoppedTransferRPUDevice<T>::ChoppedTransferRPUDevice(int x_sz, int d_sz)
    : BufferedTransferRPUDevice<T>(x_sz, d_sz) {}

template <typename T>
ChoppedTransferRPUDevice<T>::ChoppedTransferRPUDevice(
    int x_sz, int d_sz, const ChoppedTransferRPUDeviceMetaParameter<T> &par, RealWorldRNG<T> *rng)
    : ChoppedTransferRPUDevice<T>(x_sz, d_sz) {
  populate(par, rng);
}

// copy constructor
template <typename T>
ChoppedTransferRPUDevice<T>::ChoppedTransferRPUDevice(const ChoppedTransferRPUDevice<T> &other)
    : BufferedTransferRPUDevice<T>(other) {
  out_chopper_ = other.out_chopper_;
  in_chopper_ = other.in_chopper_;
  m_x_ = other.m_x_;
  m_d_ = other.m_d_;
  d_sparsity_ = other.d_sparsity_;
  transfer_counter_ = other.transfer_counter_;
}

// copy assignment
template <typename T>
ChoppedTransferRPUDevice<T> &
ChoppedTransferRPUDevice<T>::operator=(const ChoppedTransferRPUDevice<T> &other) {

  ChoppedTransferRPUDevice<T> tmp(other);
  swap(*this, tmp);
  return *this;
}

// move constructor
template <typename T>
ChoppedTransferRPUDevice<T>::ChoppedTransferRPUDevice(
    ChoppedTransferRPUDevice<T> &&other) noexcept {
  *this = std::move(other);
}

// move assignment
template <typename T>
ChoppedTransferRPUDevice<T> &
ChoppedTransferRPUDevice<T>::operator=(ChoppedTransferRPUDevice<T> &&other) noexcept {
  BufferedTransferRPUDevice<T>::operator=(std::move(other));
  in_chopper_ = std::move(other.in_chopper_);
  out_chopper_ = std::move(other.out_chopper_);
  m_x_ = other.m_x_;
  m_d_ = other.m_d_;
  d_sparsity_ = other.d_sparsity_;
  transfer_counter_ = other.transfer_counter_;
  return *this;
}

/*********************************************************************************/
/* populate */

template <typename T>
void ChoppedTransferRPUDevice<T>::populate(
    const ChoppedTransferRPUDeviceMetaParameter<T> &p, RealWorldRNG<T> *rng) {

  BufferedTransferRPUDevice<T>::populate(p, rng);

  const auto &par = getPar();
  in_chopper_.resize(par.getInSize(), false);
  out_chopper_.resize(par.getOutSize(), false);
  m_x_ = (T)0.0;
  m_d_ = (T)0.0;
  d_sparsity_ = (T)0.0;
  transfer_counter_ = 0;
}

template <typename T>
int ChoppedTransferRPUDevice<T>::getTransferEvery(
    int didx, int m_batch, const PulsedUpdateMetaParameter<T> &up) const {
  const auto &par = getPar();
  if (par.usesAutoTransferEvery()) {
    T t = par.getAutoTransferEvery(this->rpu_device_vec_[didx]->getNumStates(), up);
    return MAX(1, (int)roundf(t));
  }
  return BufferedTransferRPUDevice<T>::getTransferEvery(didx, m_batch, up);
}

template <typename T>
void ChoppedTransferRPUDevice<T>::dumpExtra(RPU::state_t &extra, const std::string prefix) {

  BufferedTransferRPUDevice<T>::dumpExtra(extra, prefix);

  RPU::state_t state;

  RPU::insert(state, "in_chopper", in_chopper_);
  RPU::insert(state, "out_chopper", out_chopper_);
  RPU::insert(state, "tmp_count_lr", tmp_count_lr_);
  RPU::insert(state, "m_x", m_x_);
  RPU::insert(state, "m_d", m_d_);
  RPU::insert(state, "d_sparsity", d_sparsity_);
  RPU::insert(state, "transfer_counter", transfer_counter_);

  RPU::insertWithPrefix(extra, state, prefix);
}

template <typename T>
void ChoppedTransferRPUDevice<T>::loadExtra(
    const RPU::state_t &extra, const std::string prefix, bool strict) {

  BufferedTransferRPUDevice<T>::loadExtra(extra, prefix, strict);
  auto state = RPU::selectWithPrefix(extra, prefix);

  RPU::load(state, "in_chopper", in_chopper_, strict);
  RPU::load(state, "out_chopper", out_chopper_, strict);

  RPU::load(state, "tmp_count_lr", tmp_count_lr_, strict);
  RPU::load(state, "m_x", m_x_, strict);
  RPU::load(state, "m_d", m_d_, strict);
  RPU::load(state, "d_sparsity", d_sparsity_, strict);
  RPU::load(state, "transfer_counter", transfer_counter_, strict);
}

/*********************************************************************************/
/* init update cycle */
template <typename T>
void ChoppedTransferRPUDevice<T>::initUpdateCycle(
    T **weights,
    const PulsedUpdateMetaParameter<T> &up,
    T current_lr,
    int m_batch_info,
    const T *x_input,
    const int x_inc,
    const T *d_input,
    const int d_inc) {
  // called once per vector update
  const auto &par = getPar();

  if (par.auto_scale) {
    // a little inefficient since the max might have been already computed in the updater..
    T x_abs_max = Find_Absolute_Max<T>(x_input, this->x_size_, x_inc);
    T d_abs_max = Find_Absolute_Max<T>(d_input, this->d_size_, d_inc);
    par.updateAutoScale(m_x_, x_abs_max, m_batch_info);
    par.updateAutoScale(m_d_, d_abs_max, m_batch_info);
  }
  if (up.d_sparsity) {
    par.updateAutoScale(d_sparsity_, up._d_sparsity, m_batch_info);
  }

  BufferedTransferRPUDevice<T>::initUpdateCycle(
      weights, up, current_lr, m_batch_info, x_input, x_inc, d_input, d_inc);
}

template <typename T>
T ChoppedTransferRPUDevice<T>::getPulseCountLearningRate(
    T lr, int current_m_batch, const PulsedUpdateMetaParameter<T> &up) {

  // called once per vector update
  const auto &par = getPar();
  T count_lr;
  if (par.auto_scale && par.fast_lr > (T)0.0) {
    T transfer_every = (T)this->getTransferEvery(0, current_m_batch, up);
    count_lr = par.getPulseCountAutoLR(
        m_x_, m_d_, d_sparsity_, this->rpu_device_vec_[0]->getWeightGranularity(), transfer_every,
        up);

  } else {
    count_lr = BufferedTransferRPUDevice<T>::getPulseCountLearningRate(lr, current_m_batch, up);
  }
  setCurrentCountLR(count_lr);
  return count_lr;
}

/*********************************************************************************/
/* transfer */
template <typename T>
void ChoppedTransferRPUDevice<T>::readAndUpdate(
    int to_device_idx,
    int from_device_idx,
    const T lr,
    const T *vec,
    const int n_vec,
    const T reset_prob_in,
    const int i_slice_start,
    const int m_batch_info) {

  UNUSED(reset_prob_in);

  if (lr == (T)0.0) {
    return;
  }

  if (to_device_idx != 1 && from_device_idx != 1) {
    RPU_FATAL("Only 2 devices supported");
  }
  if (n_vec != 1) {
    RPU_FATAL("Only 1 read per transfer supported");
  }

  const int FROM_DEVICE_IDX = 0;
  const int TO_DEVICE_IDX = 1;

  const auto &par = getPar();
  int in_size = par.getInSize();
  int out_size = par.getOutSize();
  this->transfer_tmp_.resize(out_size);
  T *v_out = this->transfer_tmp_.data();
  const T from_weight_granularity = this->rpu_device_vec_[FROM_DEVICE_IDX]->getWeightGranularity();
  const T to_weight_granularity = this->rpu_device_vec_[TO_DEVICE_IDX]->getWeightGranularity();
  const T sub_momentum = (T)1.0 - MAX(MIN(par.momentum, (T)1.0), (T)0.0);
  const T lr_scale = par.getTransferLRScale(
      from_weight_granularity, to_weight_granularity, lr, getCurrentCountLR(), m_batch_info);
  const bool forget_buffer = par.forget_buffer;
  const T max_steps = (T)this->transfer_pwu_->getUpPar().desired_BL;

  // buffer weight is x_size major, we need to write out_size
  const bool use_cols = par.transfer_columns;
  const int w_inc = use_cols ? in_size : 1;
  const bool no_buffer = par.no_buffer;
  T *fp_w = this->transfer_buffer_vec_[FROM_DEVICE_IDX].data();

  // forward / update
  const T *v_in = vec;

  // first read from previous device
  this->readVector(FROM_DEVICE_IDX, v_in, v_out, (T)1.0);

  // add into to FP buffer
  int i_w = use_cols ? i_slice_start : this->x_size_ * i_slice_start;

  int non_zero_count = 0;
  bool in_chop = in_chopper_[(size_t)(i_slice_start)];

  PRAGMA_SIMD
  for (int j = 0; j < out_size; j++) {

    T omega = no_buffer ? (T)0.0 : fp_w[i_w];
    T val = v_out[j] * lr_scale;
    T val_signed = (in_chop != out_chopper_[j]) ? -val : val;

    omega += val_signed;
    T n_steps = 0.0;

    if ((T)fabsf(omega) >= (T)1.0) {
      n_steps = MAX(MIN((T)truncf(omega), max_steps), -max_steps);
      if (forget_buffer) {
        omega *= par.momentum;
      } else {
        omega -= sub_momentum * n_steps;
      }
      non_zero_count += 1;
    }
    fp_w[i_w] = omega;
    v_out[j] = -n_steps; // since positive update needed below
    i_w += w_inc;
  }

  if (non_zero_count > 0) {
    this->writeVector(TO_DEVICE_IDX, v_in, v_out, par.getWriteLR(to_weight_granularity), 1);
  }

  if (par.in_chop_prob > (T)0.0) {
    // randomize in_choppers
    if (par.in_chop_random) {
      if (this->rw_rng_.sampleUniform() < par.in_chop_prob) {
        in_chopper_[(size_t)(i_slice_start)] = !in_chopper_[(size_t)(i_slice_start)];
      }
    } else {
      const uint64_t n_samples = MAX((int)ceilf((T)1.0 / par.in_chop_prob), 2);
      if ((this->transfer_counter_ % n_samples) + 1 == n_samples) {
        in_chopper_[(size_t)(i_slice_start)] = !in_chopper_[(size_t)(i_slice_start)];
      }
    }
  }

  if (i_slice_start == in_size - 1 && par.out_chop_prob > (T)0.0) {
    // there should always be a i_slice_start==1 even if warping with
    // left over, as then this is called twice.
    for (int j = 0; j < out_size; j++) {
      if (this->rw_rng_.sampleUniform() < par.out_chop_prob) {
        out_chopper_[j] = !out_chopper_[j];
      }
    }
  }

  if (i_slice_start == in_size - 1) {
    // only advance after full matrix transfer
    transfer_counter_++;
  }
}

template <typename T>
void ChoppedTransferRPUDevice<T>::doSparseUpdate(
    T **weights, int i, const int *x_signed_indices, int x_count, int d_sign, RNG<T> *rng) {

  const auto &par = getPar();

  x_signed_indices_tmp_.resize(x_count);
  const auto &d_chopper = (par.transfer_columns) ? out_chopper_ : in_chopper_;
  const auto &x_chopper = (par.transfer_columns) ? in_chopper_ : out_chopper_;

  PRAGMA_SIMD
  for (int j = 0; j < x_count; ++j) {
    int idx_signed = x_signed_indices[j];
    int idx = idx_signed < 0 ? -idx_signed - 1 : idx_signed - 1;
    x_signed_indices_tmp_[j] = x_chopper[idx] ? -x_signed_indices[j] : x_signed_indices[j];
  }

  TransferRPUDevice<T>::doSparseUpdate(
      weights, i, x_signed_indices_tmp_.data(), x_count, d_chopper[i] ? -d_sign : d_sign, rng);
}

template <typename T>
void ChoppedTransferRPUDevice<T>::doDenseUpdate(T **weights, int *coincidences, RNG<T> *rng) {

  const auto &par = getPar();

  const auto &d_chopper = (par.transfer_columns) ? out_chopper_ : in_chopper_;
  const auto &x_chopper = (par.transfer_columns) ? in_chopper_ : out_chopper_;

  int idx = 0;
  for (int i = 0; i < this->d_size_; ++i) {
    bool dc = d_chopper[i];
    PRAGMA_SIMD
    for (int j = 0; j < this->x_size_; ++j) {
      bool c = x_chopper[j] != dc;
      coincidences[idx] = c ? -coincidences[idx] : coincidences[idx];
      idx++;
    }
  }

  TransferRPUDevice<T>::doDenseUpdate(weights, coincidences, rng);
}

template class ChoppedTransferRPUDevice<float>;
#ifdef RPU_USE_DOUBLE
template class ChoppedTransferRPUDevice<double>;
#endif
#ifdef RPU_USE_FP16
template class ChoppedTransferRPUDevice<half_t>;
#endif

} // namespace RPU
