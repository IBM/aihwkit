/**
 * (C) Copyright 2020, 2021, 2022, 2023 IBM. All Rights Reserved.
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

  ss << "\t out_chop_prob:\t\t";
  ss << out_chop_prob;
  ss << std::endl;

  ss << "\t auto_scale:\t\t";
  ss << std::boolalpha << auto_scale;
  if (auto_scale && experimental_adjust_auto_scale_with_transfer_every) {
    ss << " [adjusted with transfer every]";
  }
  ss << std::endl;

  ss << "\t auto_momentum:\t\t";
  ss << auto_momentum;
  ss << std::endl;

  if (no_buffer) {
    ss << "\t buffer not used.";
    ss << std::endl;
  } else {
    if (buffer_granularity > 0) {
      ss << "\t buffer_granularity:\t";
      ss << buffer_granularity;
      if (auto_granularity > (T)0.0) {
        ss << "\t [auto adjustment " << auto_granularity << "]";
      }
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
  T tau = ((T)1.0 - auto_momentum) / m_batch;

  m = ((T)1.0 - tau) * m + tau * new_val;
}

template <typename T>
T ChoppedTransferRPUDeviceMetaParameter<T>::getWriteLR(T weight_granularity) const {
  return this->step * weight_granularity;
}

template <typename T>
T ChoppedTransferRPUDeviceMetaParameter<T>::getTransferLRScale(T weight_granularity, T lr) const {
  T bg = getBufferGranularity(weight_granularity);
  return fabs(lr) / bg;
}

template <typename T>
T ChoppedTransferRPUDeviceMetaParameter<T>::getAutoTransferEvery(
    T n_states, const PulsedUpdateMetaParameter<T> &up) const {
  UNUSED(up);
  return n_states / this->getInSize() * fabs(this->transfer_every);
}

template <typename T>
T ChoppedTransferRPUDeviceMetaParameter<T>::getBufferGranularity(T weight_granularity) const {
  if (auto_granularity > (T)0.0) {
    T g = this->getInSize() * fabs(this->transfer_every) / weight_granularity;
    return buffer_granularity * auto_granularity / g;
  } else {
    return buffer_granularity;
  }
}

template <typename T>
T ChoppedTransferRPUDeviceMetaParameter<T>::getPulseCountAutoLR(
    T m_x, T m_d, T weight_granularity, T transfer_every, const PulsedUpdateMetaParameter<T> &up)
    const {

  T g = up.desired_BL * weight_granularity;
  T count_lr = this->fast_lr * g;

  if (experimental_adjust_auto_scale_with_transfer_every) {
    T n_transfers = transfer_every * this->getInSize();
    count_lr /= n_transfers;
    count_lr *= 256; // typical half tile size to get the order similar
  }

  if ((m_x > (T)0.0) && (m_d > (T)0.0)) {
    count_lr /= m_x * m_d;
  }
  // avoid severe clipping
  count_lr = MIN(count_lr, 100 * g);
  return count_lr;
}

template struct ChoppedTransferRPUDeviceMetaParameter<float>;
#ifdef RPU_USE_DOUBLE
template struct ChoppedTransferRPUDeviceMetaParameter<double>;
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
  in_chopper_.resize(par.getInSize(), true);
  out_chopper_.resize(par.getOutSize(), false);
  m_x_ = (T)0.0;
  m_d_ = (T)0.0;
  transfer_counter_ = 0;

  // T buffer_granularity =
  // par.getBufferGranularity(this->rpu_device_vec_[0]->getWeightGranularity()); std::cout << "Tile
  // [" << this->getDSize() << " x " << this->getXSize()
  //           << "]: " << buffer_granularity << std::endl;
}

template <typename T>
int ChoppedTransferRPUDevice<T>::getTransferEvery(
    int didx, int m_batch, const PulsedUpdateMetaParameter<T> &up) const {
  const auto &par = getPar();
  if (par.usesAutoTransferEvery()) {
    T t = par.getAutoTransferEvery(this->rpu_device_vec_[didx]->getNumStates(), up);
    return MAX(1, (int)round(t));
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

  BufferedTransferRPUDevice<T>::initUpdateCycle(
      weights, up, current_lr, m_batch_info, x_input, x_inc, d_input, d_inc);
}

template <typename T>
T ChoppedTransferRPUDevice<T>::getPulseCountLearningRate(
    T lr, int current_m_batch, const PulsedUpdateMetaParameter<T> &up) {

  // called once per vector update
  // sets the current_count_lr for convinience
  const auto &par = getPar();
  if (par.auto_scale && par.fast_lr > 0) {
    T transfer_every = (T)this->getTransferEvery(0, current_m_batch, up);
    T count_lr = par.getPulseCountAutoLR(
        m_x_, m_d_, this->rpu_device_vec_[0]->getWeightGranularity(), transfer_every, up);

    tmp_count_lr_ = count_lr;
  } else {
    tmp_count_lr_ =
        BufferedTransferRPUDevice<T>::getPulseCountLearningRate(lr, current_m_batch, up);

    // scale so that it is constant for tile size / dw_min / bl - change
    // tmp_count_lr_ /= par.getInSize() * fabs(par.transfer_every);
  }

  return tmp_count_lr_;
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
    const int i_slice_start) {

  UNUSED(reset_prob_in);

  if (lr == (T)0.0) {
    return;
  }

  if (to_device_idx != 1 && from_device_idx != 1) {
    RPU_FATAL("Only 2 devices supported");
  }
  const int FROM_DEVICE_IDX = 0;
  const int TO_DEVICE_IDX = 1;

  const auto &par = getPar();
  int in_size = par.getInSize();
  int out_size = par.getOutSize();
  this->transfer_tmp_.resize(out_size);
  T *v_out = this->transfer_tmp_.data();
  const T from_weight_granularity = this->rpu_device_vec_[FROM_DEVICE_IDX]->getWeightGranularity();
  const T sub_momentum = (T)1.0 - MAX(MIN(par.momentum, (T)1.0), (T)0.0);
  const T lr_scale = par.getTransferLRScale(from_weight_granularity, lr);
  const bool forget_buffer = par.forget_buffer;
  const T max_steps = (T)this->transfer_pwu_->getUpPar().desired_BL;

  // buffer weight is x_size major, we need to write out_size
  const bool use_cols = par.transfer_columns;
  const int w_inc = use_cols ? in_size : 1;
  const bool no_buffer = par.no_buffer;
  T *fp_w = this->transfer_buffer_vec_[FROM_DEVICE_IDX].data();

  // forward / update
  for (int i = 0; i < n_vec; i++) {

    const T *v_in = vec + (size_t)(i * in_size);

    // first read from previous device
    this->readVector(FROM_DEVICE_IDX, v_in, v_out, (T)1.0);

    // add into to FP buffer
    int i_w = use_cols ? i_slice_start + i : this->x_size_ * (i_slice_start + i);

    int non_zero_count = 0;
    bool in_chop = in_chopper_[(size_t)(i_slice_start + i)];
    PRAGMA_SIMD
    for (int j = 0; j < out_size; j++) {

      T omega = no_buffer ? (T)0.0 : fp_w[i_w];
      T val = v_out[j] * lr_scale;
      T val_signed = (in_chop != out_chopper_[j]) ? -val : val;

      omega += val_signed;
      T n_steps = 0.0;

      if (fabs(omega) >= (T)1.0) {
        n_steps = MAX(MIN(truncf(omega), max_steps), -max_steps);
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
      const T weight_granularity = this->rpu_device_vec_[TO_DEVICE_IDX]->getWeightGranularity();
      this->writeVector(TO_DEVICE_IDX, v_in, v_out, par.getWriteLR(weight_granularity), 1);
    }

    if (FROM_DEVICE_IDX == 0 && par.in_chop_prob > (T)0.0) {
      // randomize in_choppers
      if (par.in_chop_random) {
        if (this->rw_rng_.sampleUniform() < par.in_chop_prob) {
          in_chopper_[(size_t)(i_slice_start + i)] = !in_chopper_[(size_t)(i_slice_start + i)];
        }
      } else {
        const uint64_t n_samples = MAX((int)ceil((T)1.0 / par.in_chop_prob), 2);
        if ((this->transfer_counter_ % n_samples) + 1 == n_samples) {
          in_chopper_[(size_t)(i_slice_start + i)] = !in_chopper_[(size_t)(i_slice_start + i)];
        }
      }
    }
  }

  if (FROM_DEVICE_IDX == 0 && i_slice_start == in_size - 1 && par.out_chop_prob > (T)0.0) {
    // there should always be a i_slice_start==1 even if warping with
    // left over, as then this is called twice.
    for (int j = 0; j < out_size; j++) {
      if (this->rw_rng_.sampleUniform() < par.out_chop_prob) {
        out_chopper_[j] = !out_chopper_[j];
      }
    }
  }

  if (FROM_DEVICE_IDX == 0 && i_slice_start == in_size - 1) {
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

} // namespace RPU
