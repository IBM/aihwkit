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

#include "rpu_transfer_device.h"
#include "math_util.h"
#include "utility_functions.h"
#include <algorithm>
#include <memory>
#include <sstream>

namespace RPU {

/**************************************************************************************/
/* TransferRPUDeviceMetaParameter*/
template <typename T>
TransferRPUDeviceMetaParameter<T>::TransferRPUDeviceMetaParameter(
    const PulsedRPUDeviceMetaParameterBase<T> &dp_fast,
    const PulsedRPUDeviceMetaParameterBase<T> &dp_rest,
    int n_total_devices) {
  this->vec_par.clear();
  if (n_total_devices < 2) {
    RPU_FATAL("More or equal than 2 devices expected.");
  }
  this->appendVecPar(dp_fast.clone());
  for (int i = 1; i < n_total_devices; i++) {
    this->appendVecPar(dp_rest.clone());
  }
};

template <typename T>
void TransferRPUDeviceMetaParameter<T>::printToStream(std::stringstream &ss) const {
  ss << this->getName() << std::endl;
  // gamma
  ss << "\tgamma:\t\t\t";
  if (this->_par_initialized)
    for (size_t k = 0; k < this->gamma_vec.size(); k++) {
      ss << this->gamma_vec[k] << " ";
    }
  else {
    ss << gamma;
  }
  ss << std::endl;

  // every
  ss << "\ttransfer_every [init]: \t";
  if (this->_par_initialized)
    for (size_t k = 0; k < transfer_every_vec.size(); k++) {
      ss << transfer_every_vec[k] << " ";
    }
  else {
    ss << transfer_every;
  }
  if (units_in_mbatch) {
    ss << " [in mbatches]";
  }
  ss << std::endl;

  // lr
  ss << "\ttransfer_lr: \t\t";
  if (this->_par_initialized)
    for (size_t k = 0; k < transfer_lr_vec.size(); k++) {
      ss << transfer_lr_vec[k] << " ";
    }
  else {
    ss << transfer_lr;
  }
  if (scale_transfer_lr) {
    ss << " [scaled with current LR]";
  }
  ss << std::endl;

  ss << "\tn_cols_per_transfer: \t" << n_cols_per_transfer;
  if (with_reset_prob) {
    ss << "\t[with reset p=" << with_reset_prob << "]";
  }
  if (random_column) {
    ss << "\t[random column]";
  }
  ss << std::endl;

  ss << "   Transfer IO: \n";
  transfer_io.printToStream(ss);
  ss << "   Transfer Update Parameter: \n";
  transfer_up.printToStream(ss);

  for (size_t k = 0; k < this->vec_par.size(); k++) {
    ss << "   Device Parameter " << k << ": " << this->vec_par[k]->getName() << std::endl;
    ss << "   ";
    this->vec_par[k]->printToStream(ss);
  }
};

template <typename T>
void TransferRPUDeviceMetaParameter<T>::initializeWithSize(int x_size, int d_size) {
  // check for _par_initialized ? Maybe just force?

  VectorRPUDeviceMetaParameter<T>::initialize();

  size_t n_devices = this->vec_par.size();

  if (n_devices < 2) {
    // makes no sense
    RPU_FATAL("Need at least 2 devices");
  }

  this->update_policy = VectorDeviceUpdatePolicy::SingleFixed;
  this->first_update_idx = 0; // only first is updated
  this->same_context = true;

  // Only the first device might be difference from the rest,
  // because we use only 2 pulsed weight updater
  auto impl = this->vec_par[1]->implements();
  for (size_t i = 2; i < n_devices; i++) {
    if (impl != this->vec_par[i]->implements()) {
      RPU_FATAL("Only the first device can be a difference RPU device. ");
    }
  }

  // weightening of devices to get final weights
  if (this->gamma_vec.size() > 0) {
    if (this->gamma_vec.size() != n_devices) {
      RPU_FATAL("If gamma_vec is set manually expect the same size as number of devices.");
    }
    T g = 0;
    for (size_t i = 0; i < n_devices - 1; i++) {
      g += this->gamma_vec[i];
    }
    if (this->gamma_vec[n_devices - 1] == 0) {
      RPU_FATAL("Expect that last device has some constribution to the network weights. [otherwise "
                "why transfer?]");
    }
    gamma = g;
  }
  if (this->gamma_vec.size() == 0) {
    this->gamma_vec.resize(n_devices);
    for (size_t i = 0; i < n_devices; i++) {
      this->gamma_vec[n_devices - i - 1] = pow(gamma, (T)i);
    }
  }

  if (transfer_lr_vec.size() == 0) {
    transfer_lr_vec.resize(n_devices);
    std::fill(transfer_lr_vec.begin(), transfer_lr_vec.end(), transfer_lr);
  }
  if (transfer_lr_vec.size() != n_devices) {
    RPU_FATAL("Expect transfer_lr_vec of size n_devices.");
  }

  if (n_cols_per_transfer > x_size) {
    // should not be needed anyway
    n_cols_per_transfer = x_size;
    RPU_WARNING("too many transfers in one shot. Use x_size instead.");
  }

  if (!transfer_every) {
    transfer_every = (T)x_size / n_cols_per_transfer;
  }

  if (transfer_every_vec.size() == 0) {
    T n = transfer_every;
    for (size_t i = 0; i < n_devices; i++) {
      transfer_every_vec.push_back(n);
      n *= (T)x_size / n_cols_per_transfer;
    }
    if (no_self_transfer) {
      transfer_every_vec[n_devices - 1] = 0;
    }
  }

  if (transfer_every_vec.size() != n_devices) {
    RPU_FATAL("Expect transfer_every_vec to be of length n_devices");
  }

  // IO
  transfer_io.initializeForForward();

  // we turn BM off.
  if (transfer_io.bound_management != BoundManagementType::None) {
    RPU_WARNING("Transfer bound management turned off.");
    transfer_io.bound_management = BoundManagementType::None;
  }

  // up
  transfer_up.initialize();
}

template <typename T>
T TransferRPUDeviceMetaParameter<T>::getTransferLR(
    int to_device_idx, int from_device_idx, T current_lr) const {

  T lr = transfer_lr_vec[from_device_idx];
  if (scale_transfer_lr) {
    lr *= current_lr;
  }
  return lr;
}

template struct TransferRPUDeviceMetaParameter<float>;
#ifdef RPU_USE_DOUBLE
template struct TransferRPUDeviceMetaParameter<double>;
#endif

/******************************************************************************************/
// dtor
template <typename T> TransferRPUDevice<T>::~TransferRPUDevice() {}

// ctor
template <typename T>
TransferRPUDevice<T>::TransferRPUDevice(int x_sz, int d_sz) : VectorRPUDevice<T>(x_sz, d_sz) {}

template <typename T>
TransferRPUDevice<T>::TransferRPUDevice(
    int x_sz, int d_sz, const TransferRPUDeviceMetaParameter<T> &par, RealWorldRNG<T> *rng)
    : TransferRPUDevice<T>(x_sz, d_sz) {
  populate(par, rng);
}

// copy construcutor
template <typename T>
TransferRPUDevice<T>::TransferRPUDevice(const TransferRPUDevice<T> &other)
    : VectorRPUDevice<T>(other) {

  transfer_fb_pass_ = make_unique<ForwardBackwardPassIOManaged<T>>(*other.transfer_fb_pass_);
  transfer_pwu_ = make_unique<PulsedRPUWeightUpdater<T>>(*other.transfer_pwu_);

  current_col_indices_ = other.current_col_indices_;
  transfer_vecs_ = other.transfer_vecs_;
  transfer_every_ = other.transfer_every_;
  fully_hidden_ = other.fully_hidden_;
}

// copy assignment
template <typename T>
TransferRPUDevice<T> &TransferRPUDevice<T>::operator=(const TransferRPUDevice<T> &other) {

  TransferRPUDevice<T> tmp(other);
  swap(*this, tmp);
  return *this;
}

// move constructor
template <typename T> TransferRPUDevice<T>::TransferRPUDevice(TransferRPUDevice<T> &&other) {
  *this = std::move(other);
}

// move assignment
template <typename T>
TransferRPUDevice<T> &TransferRPUDevice<T>::operator=(TransferRPUDevice<T> &&other) {
  VectorRPUDevice<T>::operator=(std::move(other));

  current_col_indices_ = other.current_col_indices_;
  other.current_col_indices_.clear();

  transfer_vecs_ = other.transfer_vecs_;
  other.transfer_vecs_.clear();

  transfer_every_ = other.transfer_every_;
  other.transfer_every_.clear();

  transfer_fb_pass_ = std::move(other.transfer_fb_pass_);
  transfer_pwu_ = std::move(other.transfer_pwu_);

  fully_hidden_ = other.fully_hidden_;

  return *this;
}

template <typename T> void TransferRPUDevice<T>::setTransferVecs(const T *transfer_vecs) {
  transfer_vecs_.resize(this->x_size_ * this->x_size_); //!!  square matrix
  std::fill(transfer_vecs_.begin(), transfer_vecs_.end(), (T)0.0);

  if (transfer_vecs == nullptr) {
    // initialize transfer vectors with unit vectors. This might be overridden
    for (size_t i = 0; i < transfer_vecs_.size(); i += this->x_size_ + 1) {
      transfer_vecs_[i] = 1.0;
    }
  } else {
    for (size_t i = 0; i < transfer_vecs_.size(); i++) {
      transfer_vecs_[i] = transfer_vecs[i];
    }
  }
}

/*********************************************************************************/
/* populate */

template <typename T>
void TransferRPUDevice<T>::populate(
    const TransferRPUDeviceMetaParameter<T> &p, RealWorldRNG<T> *rng) {

  VectorRPUDevice<T>::populate(p, rng);
  auto &par = getPar();
  par.initializeWithSize(this->x_size_, this->d_size_);
  auto shared_rng = std::make_shared<RNG<T>>(0); // we just take a new one here (seeds...)
  transfer_fb_pass_ =
      RPU::make_unique<ForwardBackwardPassIOManaged<T>>(this->x_size_, this->d_size_, shared_rng);
  transfer_fb_pass_->setIOPar(par.transfer_io, par.transfer_io);
  // NOTE: the OUT_SCALE might be different for the transfer!! How to account for that?!?

  transfer_pwu_ =
      RPU::make_unique<PulsedRPUWeightUpdater<T>>(this->x_size_, this->d_size_, shared_rng);
  transfer_pwu_->setUpPar(par.transfer_up);

  this->reduce_weightening_.resize(this->n_devices_);
  for (int k = 0; k < this->n_devices_; k++) {
    this->reduce_weightening_[k] = par.gamma_vec[k];
  }

  setTransferVecs();

  transfer_every_ = par.transfer_every_vec; // already checked for length

  current_col_indices_.resize(this->n_devices_);
  std::fill(current_col_indices_.begin(), current_col_indices_.end(), 0);

  this->current_device_idx_ = 0; // only zero is updated, ignore current idx

  fully_hidden_ = getPar().fullyHidden(); // save
}

/*********************************************************************************/
/* transfer */
template <typename T> T **TransferRPUDevice<T>::getDeviceWeights(int device_idx) const {

  if (fully_hidden_ && device_idx == this->n_devices_ - 1) {
    return last_weight_;
  } else {
    return this->weights_vec_[device_idx];
  }
}

template <typename T>
int TransferRPUDevice<T>::getTransferEvery(int from_device_idx, int m_batch) const {

  if (getPar().units_in_mbatch) {
    return MAX(RPU_ROUNDFUN(transfer_every_[from_device_idx] * m_batch), 0);
  } else {
    return MAX(RPU_ROUNDFUN(transfer_every_[from_device_idx]), 0);
  }
}

template <typename T>
void TransferRPUDevice<T>::forwardUpdate(
    int to_device_idx,
    int from_device_idx,
    const T lr,
    const T *x_input,
    const int n_vec,
    const bool trans,
    const T reset_prob,
    const int i_col) {

  if (!lr) {
    return;
  }

  if (to_device_idx == from_device_idx) {
    // self update not supported per default
    return;
  }

  if (transfer_tmp_.size() < (size_t)this->d_size_) {
    transfer_tmp_.resize(this->d_size_);
  }

  // forward / update
  T **W;
  for (int i = 0; i < n_vec; i++) {
    const T *x = x_input + i * this->x_size_;

    transfer_fb_pass_->forwardVector(
        this->weights_vec_[from_device_idx], x, 1, &transfer_tmp_[0], 1, (T)1.0, false);

    // potentially reset here (because of possible same device to-from):
    // NOTE that with_reset_prob is COL-wise prob (elem device prob is 1)
    if (this->rw_rng_.sampleUniform() < reset_prob) {
      W = getDeviceWeights(from_device_idx);
      this->rpu_device_vec_[from_device_idx]->resetCols(W, i_col, n_vec, 1, this->rw_rng_);
    }

    // update according to device
    W = getDeviceWeights(to_device_idx);

    transfer_pwu_->updateVectorWithDevice(
        W, x, 1, &transfer_tmp_[0], 1,
        -fabs(lr), // need to be negative...
        1, &*this->rpu_device_vec_[to_device_idx]);
  }
}

template <typename T>
void TransferRPUDevice<T>::transfer(int to_device_idx, int from_device_idx, T current_lr) {
  int i_col = current_col_indices_[from_device_idx];
  const auto &par = getPar();
  if (par.random_column) {
    i_col = MAX(MIN(floor(this->rw_rng_.sampleUniform() * this->x_size_), this->x_size_ - 1), 0);
  }

  // transfer_vecs_ is always x_size-major (that is trans==false)
  T *tvec = &transfer_vecs_[0] + i_col * this->x_size_;
  int n_rest = this->x_size_ - i_col; // actually always x_size

  T lr = par.getTransferLR(to_device_idx, from_device_idx, current_lr);

  int n_transfer = MIN(par.n_cols_per_transfer, this->x_size_);

  if (n_rest < n_transfer) {
    // rest

    forwardUpdate(
        to_device_idx, from_device_idx, lr, tvec, n_rest, false, par.with_reset_prob, i_col);
    // from beginning
    forwardUpdate(
        to_device_idx, from_device_idx, lr, &transfer_vecs_[0], n_transfer - n_rest, false,
        par.with_reset_prob, 0);

  } else {
    forwardUpdate(
        to_device_idx, from_device_idx, lr, tvec, n_transfer, false, par.with_reset_prob, i_col);
  }

  current_col_indices_[from_device_idx] = (i_col + n_transfer) % this->x_size_;
}

/*********************************************************************************/
/* update */
/********************************************************************************/

template <typename T>
void TransferRPUDevice<T>::doSparseUpdate(
    T **weights, int i, const int *x_signed_indices, int x_count, int d_sign, RNG<T> *rng) {

  this->rpu_device_vec_[0]->doSparseUpdate(
      this->weights_vec_[0], i, x_signed_indices, x_count, d_sign, rng);

  // we do reduce to weights in the finishUpdateCycle, because transfer is done there, too.
}

template <typename T>
void TransferRPUDevice<T>::finishUpdateCycle(
    T **weights, const PulsedUpdateMetaParameter<T> &up, T current_lr, int m_batch_info) {

  VectorRPUDevice<T>::finishUpdateCycle(
      weights, up, current_lr, m_batch_info); // first increment to avoid zero first transfer

  // directly onto last weight if fully hidden. No reduce needed
  last_weight_ = fully_hidden_ ? weights : nullptr;

  // we transfer the device here to cope with the sparse update below.
  for (int j = 0; j < this->n_devices_; j++) {

    int every = getTransferEvery(j, m_batch_info);
    if (every > 0 && this->current_update_idx_ % every == 0) {
      // last is self-update (does nothing per default, but could implement refresh in child)
      transfer(MIN(j + 1, this->n_devices_ - 1), j, current_lr);
    }
  }
  this->reduceToWeights(weights);
}

template <typename T> bool TransferRPUDevice<T>::onSetWeights(T **weights) {
  /* all weights are set to the last weight scaled by its significance*/

  // all weights are set to *identical* values...
  // return VectorRPUDevice<T>::onSetWeights(weights);

  if (!this->n_devices_) {
    RPU_FATAL("First populate device then set the weights");
  }

  int last_idx = this->n_devices_ - 1;

  for (int j = 0; j < this->n_devices_; j++) {
    PRAGMA_SIMD
    for (int i = 0; i < this->size_; i++) {
      this->weights_vec_[j][0][i] = 0.0;
    }
  }
  if (fully_hidden_) {
    // no need to copy, as we directly use the weights
    this->rpu_device_vec_[last_idx]->onSetWeights(weights);

  } else {

    // only last gets all the weights, scaled by its significance
    T gamma_last = this->reduce_weightening_[last_idx];
    if (!gamma_last) {
      RPU_FATAL("last gamma should not be zero.");
    }

    PRAGMA_SIMD
    for (int i = 0; i < this->size_; i++) {
      this->weights_vec_[last_idx][0][i] = weights[0][i] / gamma_last;
    }

    for (int k = 0; k < this->n_devices_; k++) {
      this->rpu_device_vec_[k]->onSetWeights(this->weights_vec_[k]);
    }

    this->reduceToWeights(weights);
  }

  return true; // modified device thus true
}

template <typename T> void TransferRPUDevice<T>::reduceToWeights(T **weights) const {
  // note that we do not reduce if fully hidden since the last
  // weight is then equivalent to the actual given weight.
  if (!fully_hidden_) {
    VectorRPUDevice<T>::reduceToWeights(weights);
  }
}

#define COMMA ,
#define LOOP_WITH_HIDDEN(FUN, ADD_ARGS)                                                            \
  if (!fully_hidden_) {                                                                            \
    VectorRPUDevice<T>::FUN(weights ADD_ARGS);                                                     \
  } else {                                                                                         \
    for (size_t k = 0; k < this->rpu_device_vec_.size() - 1; k++) {                                \
      this->rpu_device_vec_[k]->FUN(this->weights_vec_[k] ADD_ARGS);                               \
    }                                                                                              \
    this->rpu_device_vec_.back()->FUN(weights ADD_ARGS);                                           \
  }

template <typename T> void TransferRPUDevice<T>::decayWeights(T **weights, bool bias_no_decay) {
  LOOP_WITH_HIDDEN(decayWeights, COMMA bias_no_decay);
}

template <typename T>
void TransferRPUDevice<T>::decayWeights(T **weights, T alpha, bool bias_no_decay) {
  LOOP_WITH_HIDDEN(decayWeights, COMMA alpha COMMA bias_no_decay);
}

template <typename T> void TransferRPUDevice<T>::diffuseWeights(T **weights, RNG<T> &rng) {
  LOOP_WITH_HIDDEN(diffuseWeights, COMMA rng);
}

template <typename T> void TransferRPUDevice<T>::clipWeights(T **weights, T clip) {
  LOOP_WITH_HIDDEN(clipWeights, COMMA clip);
}

#undef COMMA
#undef LOOP_WITH_HIDDEN

template <typename T>
void TransferRPUDevice<T>::setDeviceParameter(const std::vector<T *> &data_ptrs) {

  VectorRPUDevice<T>::setDeviceParameter(data_ptrs);

  // take dwmin only from the first
  this->dw_min_ = this->rpu_device_vec_[0]->getDwMin();
}

template class TransferRPUDevice<float>;
#ifdef RPU_USE_DOUBLE
template class TransferRPUDevice<double>;
#endif

} // namespace RPU
