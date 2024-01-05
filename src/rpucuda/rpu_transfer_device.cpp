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
  this->appendVecPar(dp_fast);
  for (int i = 1; i < n_total_devices; i++) {
    this->appendVecPar(dp_rest);
  }
};

template <typename T>
void TransferRPUDeviceMetaParameter<T>::printToStream(std::stringstream &ss) const {

  // gamma
  ss << "\t gamma:\t\t\t";
  if (this->_par_initialized)
    for (size_t k = 0; k < this->gamma_vec.size(); k++) {
      ss << this->gamma_vec[k] << "  ";
    }
  else {
    ss << gamma;
  }
  ss << std::endl;

  // every
  ss << "\t transfer_every [init]:\t";
  if (this->_par_initialized)
    for (size_t k = 0; k < transfer_every_vec.size() - 1; k++) {
      if (transfer_every_vec[k] < (T)0.0) {
        ss << "auto  ";
      } else {
        ss << transfer_every_vec[k] << "  ";
      }
    }
  else {
    ss << transfer_every;
  }
  if (units_in_mbatch) {
    ss << " [in mbatches]";
  }
  ss << std::endl;

  // lr
  if (fast_lr > (T)0.0) {
    ss << "\t fast_lr:\t\t";
    ss << fast_lr;
    ss << std::endl;
  }

  ss << "\t transfer_lr: \t\t";
  if (this->_par_initialized)
    for (size_t k = 0; k < transfer_lr_vec.size(); k++) {
      ss << transfer_lr_vec[k] << "  ";
    }
  else {
    ss << transfer_lr;
  }
  if (scale_transfer_lr) {
    ss << "\t [scaled with current LR]";
  }
  ss << std::endl;

  ss << "\t n_reads_per_transfer: \t" << n_reads_per_transfer;
  if (transfer_columns) {
    ss << "\t [reading columns]";
  } else {
    ss << "\t [reading rows]";
  }

  if (with_reset_prob > (T)0.0) {
    ss << "\t [with reset p=" << with_reset_prob << "]";
  }
  if (random_selection) {
    ss << "\t [random selection]";
  }
  ss << std::endl;

  ss << "\t\bTransfer IO: \n";
  transfer_io.printToStream(ss);
  ss << "\t\bTransfer Update Parameter: \n";
  transfer_up.printToStream(ss);

  for (size_t k = 0; k < this->vec_par.size(); k++) {
    ss << "\t\bDevice Parameter " << k << ": " << this->vec_par[k]->getName() << std::endl;
    ss << "\t\b";
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

  if (transfer_columns) {
    _in_size = x_size;
    _out_size = d_size;
  } else {
    _in_size = d_size;
    _out_size = x_size;
  }

  this->update_policy = VectorDeviceUpdatePolicy::SingleFixed;
  this->first_update_idx = 0; // only first is updated
  this->same_context = true;

  // Only the first device might be different from the rest,
  // because we use only 2 pulsed weight updater
  auto impl = this->vec_par[1]->implements();
  for (size_t i = 2; i < n_devices; i++) {
    if (impl != this->vec_par[i]->implements()) {
      RPU_FATAL("Only the first device can be a different RPU device. ");
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
    if (this->gamma_vec[n_devices - 1] == (T)0.0) {
      RPU_FATAL("Expect that last device has some constribution to the network weights. [otherwise "
                "why transfer?]");
    }
    gamma = g;
  }
  if (this->gamma_vec.size() == 0) {
    this->gamma_vec.resize(n_devices);
    for (size_t i = 0; i < n_devices; i++) {
      this->gamma_vec[n_devices - i - 1] = (T)powf(gamma, (T)i);
    }
  }

  if (transfer_lr_vec.size() == 0) {
    transfer_lr_vec.resize(n_devices);
    std::fill(transfer_lr_vec.begin(), transfer_lr_vec.end(), transfer_lr);
  }
  if (transfer_lr_vec.size() != n_devices) {
    RPU_FATAL("Expect transfer_lr_vec of size n_devices.");
  }

  if (n_reads_per_transfer > _in_size) {
    // should not be needed anyway
    n_reads_per_transfer = _in_size;
    RPU_WARNING("too many transfers in one shot. Using full transfer instead.");
  }

  if (transfer_every_vec.size() == 0) {
    T n = transfer_every;
    for (size_t i = 0; i < n_devices; i++) {
      transfer_every_vec.push_back(n);
      n *= (T)_out_size / (T)n_reads_per_transfer;
    }
    if (no_self_transfer) {
      transfer_every_vec[n_devices - 1] = 0;
    }
  }

  if (transfer_every_vec.size() != n_devices) {
    RPU_FATAL("Expect transfer_every_vec to be of length n_devices");
  }

  if (with_reset_prob > (T)0.0 && !transfer_columns) {
    RPU_FATAL("Reset prob is only implemented for column-transfer so far.");
  }

  // IO
  if (transfer_columns) {
    transfer_io.initializeForForward(x_size, d_size);
  } else {
    transfer_io.initializeForBackward(x_size, d_size);
  }

  // we turn BM off.
  if (transfer_io.bound_management != BoundManagementType::None) {
    RPU_WARNING("Transfer bound management turned off.");
    transfer_io.bound_management = BoundManagementType::None;
  }
  if (transfer_io.noise_management != NoiseManagementType::None) {
    RPU_WARNING("Transfer noise management turned off.");
    transfer_io.noise_management = NoiseManagementType::None;
  }

  // up
  transfer_up.initialize();
}

template <typename T>
T TransferRPUDeviceMetaParameter<T>::getTransferLR(
    int to_device_idx, int from_device_idx, T current_lr) const {

  UNUSED(to_device_idx);

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
#ifdef RPU_USE_FP16
template struct TransferRPUDeviceMetaParameter<half_t>;
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

  transfer_fb_pass_ = RPU::make_unique<ForwardBackwardPassIOManaged<T>>(*other.transfer_fb_pass_);
  transfer_pwu_ = RPU::make_unique<PulsedRPUWeightUpdater<T>>(*other.transfer_pwu_);

  current_slice_indices_ = other.current_slice_indices_;
  transfer_vecs_ = other.transfer_vecs_;
  transfer_every_ = other.transfer_every_;
  fully_hidden_ = other.fully_hidden_;
  last_weight_ = other.last_weight_;
}

// copy assignment
template <typename T>
TransferRPUDevice<T> &TransferRPUDevice<T>::operator=(const TransferRPUDevice<T> &other) {

  TransferRPUDevice<T> tmp(other);
  swap(*this, tmp);
  return *this;
}

// move constructor
template <typename T>
TransferRPUDevice<T>::TransferRPUDevice(TransferRPUDevice<T> &&other) noexcept {
  *this = std::move(other);
}

// move assignment
template <typename T>
TransferRPUDevice<T> &TransferRPUDevice<T>::operator=(TransferRPUDevice<T> &&other) noexcept {
  VectorRPUDevice<T>::operator=(std::move(other));

  current_slice_indices_ = std::move(other.current_slice_indices_);
  transfer_vecs_ = std::move(other.transfer_vecs_);
  transfer_every_ = std::move(other.transfer_every_);
  transfer_fb_pass_ = std::move(other.transfer_fb_pass_);
  transfer_pwu_ = std::move(other.transfer_pwu_);
  last_weight_ = std::move(other.last_weight_);
  fully_hidden_ = other.fully_hidden_;

  return *this;
}

template <typename T> void TransferRPUDevice<T>::setTransferVecs(const T *transfer_vecs) {
  size_t in_size = getPar().getInSize();

  transfer_vecs_.resize(in_size * in_size); //!!  square matrix
  std::fill(transfer_vecs_.begin(), transfer_vecs_.end(), (T)0.0);

  if (transfer_vecs == nullptr) {
    // initialize transfer vectors with unit vectors. This might be overridden
    for (size_t i = 0; i < transfer_vecs_.size(); i += in_size + 1) {
      transfer_vecs_[i] = 1.0;
    }
  } else {
    // Caution: No size check!
    for (size_t i = 0; i < transfer_vecs_.size(); i++) {
      transfer_vecs_[i] = transfer_vecs[i];
    }
  }
}

template <typename T> int TransferRPUDevice<T>::resetCounters(bool force) {

  current_slice_indices_.resize(this->n_devices_);
  std::fill(current_slice_indices_.begin(), current_slice_indices_.end(), (int)0);
  return VectorRPUDevice<T>::resetCounters(force);
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

  transfer_fb_pass_->populateFBParameter(par.transfer_io, par.transfer_io);
  transfer_pwu_ =
      RPU::make_unique<PulsedRPUWeightUpdater<T>>(this->x_size_, this->d_size_, shared_rng);
  transfer_pwu_->setUpPar(par.transfer_up);

  this->reduce_weightening_.resize(this->n_devices_);
  for (int k = 0; k < this->n_devices_; k++) {
    this->reduce_weightening_[k] = par.gamma_vec[k];
  }
  resetCounters(); // "state" pars
  setTransferVecs();
  transfer_every_ = par.transfer_every_vec; // already checked for length
  fully_hidden_ = getPar().fullyHidden();   // save
}

/*********************************************************************************/
/* getPulseCountLearningRate */
/* Here we compute the LR for the A matrix (the SGD update). Because
   of the device properties it is beneficial to use a constant LR
   here, but scale the buffer with the scheduled SGD learning rate
   later*/
template <typename T>
T TransferRPUDevice<T>::getPulseCountLearningRate(
    T learning_rate, int current_m_batch, const PulsedUpdateMetaParameter<T> &up) {

  UNUSED(current_m_batch);
  UNUSED(up);

  const auto &par = getPar();

  if (par.fast_lr > (T)0.0) {
    return par.fast_lr;
  } else {
    return learning_rate;
  }
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
int TransferRPUDevice<T>::getTransferEvery(
    int from_device_idx, int m_batch, const PulsedUpdateMetaParameter<T> &up) const {
  UNUSED(up);
  if (getPar().units_in_mbatch) {
    return MAX((int)ceilf((float)transfer_every_[from_device_idx] * (float)m_batch), 0);
  } else {
    return MAX((int)roundf(transfer_every_[from_device_idx]), 0);
  }
}

template <typename T>
void TransferRPUDevice<T>::readVector(int device_idx, const T *in_vec, T *out_vec, T alpha) {
  T **W = getDeviceWeights(device_idx);
  if (getPar().transfer_columns) {
    transfer_fb_pass_->forwardVector(W, in_vec, 1, out_vec, 1, alpha, false);
  } else {
    transfer_fb_pass_->backwardVector(W, in_vec, 1, out_vec, 1, alpha);
  }
}

template <typename T>
void TransferRPUDevice<T>::writeVector(
    int device_idx, const T *in_vec, const T *out_vec, const T lr, const int m_batch_info) {

  T **W = getDeviceWeights(device_idx);
  if (getPar().transfer_columns) {
    // in_vec is x_input
    transfer_pwu_->updateVectorWithDevice(
        W, in_vec, 1, out_vec, 1, lr, m_batch_info, &*this->rpu_device_vec_[device_idx]);
  } else {
    // in_vec is d_input
    transfer_pwu_->updateVectorWithDevice(
        W, out_vec, 1, in_vec, 1, lr, m_batch_info, &*this->rpu_device_vec_[device_idx]);
  }
}

template <typename T>
void TransferRPUDevice<T>::readAndUpdate(
    int to_device_idx,
    int from_device_idx,
    const T lr,
    const T *vec, // these are the selected transfer vecs
    const int n_vec,
    const T reset_prob,
    const int i_slice,
    const int m_batch_info) {

  if (lr == (T)0.0) {
    return;
  }

  if (to_device_idx == from_device_idx) {
    // self update not supported per default
    return;
  }
  const auto &par = getPar();

  int in_size = par.getInSize();
  int out_size = par.getOutSize();

  transfer_tmp_.resize(out_size);

  // forward or backward / update
  for (size_t i = 0; i < (size_t)n_vec; i++) {
    const T *v = vec + i * in_size;

    readVector(from_device_idx, v, transfer_tmp_.data(), -1.0); // scale -1 for pos update

    if (this->rw_rng_.sampleUniform() < reset_prob && par.transfer_columns) {
      // potentially reset here (because of possible same device to-from):
      // NOTE that with_reset_prob is COL-wise prob (elem device prob is 1)
      T **W_from = getDeviceWeights(from_device_idx);
      this->rpu_device_vec_[from_device_idx]->resetCols(W_from, i_slice, n_vec, 1, this->rw_rng_);
    }

    // update according to device
    writeVector(to_device_idx, v, transfer_tmp_.data(), lr, n_vec);
  }
}

template <typename T>
void TransferRPUDevice<T>::transfer(
    int to_device_idx, int from_device_idx, T current_lr, int m_batch_info) {

  int i_slice = current_slice_indices_[from_device_idx];
  const auto &par = getPar();

  int in_size = par.getInSize();

  if (par.random_selection) {
    i_slice = MAX(MIN((int)floorf((float)this->rw_rng_.sampleUniform() * in_size), in_size - 1), 0);
  }

  // transfer_vecs_ is always in_size-major (that is trans==false)
  T *tvec = &transfer_vecs_[0] + (size_t)(i_slice * in_size);
  int n_rest = in_size - i_slice;

  T lr = par.getTransferLR(to_device_idx, from_device_idx, current_lr);

  int n_transfer = MIN(par.n_reads_per_transfer, in_size);

  if (n_rest < n_transfer) {
    // rest

    readAndUpdate(
        to_device_idx, from_device_idx, lr, tvec, n_rest, par.with_reset_prob, i_slice,
        m_batch_info);
    // from beginning
    readAndUpdate(
        to_device_idx, from_device_idx, lr, &transfer_vecs_[0], n_transfer - n_rest,
        par.with_reset_prob, 0, m_batch_info);

  } else {
    readAndUpdate(
        to_device_idx, from_device_idx, lr, tvec, n_transfer, par.with_reset_prob, i_slice,
        m_batch_info);
  }

  current_slice_indices_[from_device_idx] = (i_slice + n_transfer) % in_size;
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
void TransferRPUDevice<T>::doDenseUpdate(T **weights, int *coincidences, RNG<T> *rng) {

  this->rpu_device_vec_[0]->doDenseUpdate(this->weights_vec_[0], coincidences, rng);
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
    int every = getTransferEvery(j, m_batch_info, up);
    if (every > 0 && this->current_update_idx_ % every == 0) {
      // last is self-update (does nothing per default, but could implement refresh in child)
      transfer(MIN(j + 1, this->n_devices_ - 1), j, current_lr, m_batch_info);
    }
  }
  this->reduceToWeights(weights);
}

template <typename T> bool TransferRPUDevice<T>::onSetWeights(T **weights) {
  /* all weights are set to the last weight scaled by its significance*/

  if (!this->n_devices_) {
    RPU_FATAL("First populate device then set the weights");
  }

  // need to reset device index etc for consistency with CUDA
  resetCounters();

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

template <typename T>
void TransferRPUDevice<T>::driftWeights(T **weights, T time_since_last_call, RNG<T> &rng) {
  LOOP_WITH_HIDDEN(driftWeights, COMMA time_since_last_call COMMA rng);
}

template <typename T>
void TransferRPUDevice<T>::resetCols(
    T **weights, int start_col, int n_cols, T reset_prob, RealWorldRNG<T> &rng) {
  // only applied to the updated (first) device !
  this->rpu_device_vec_[0]->resetCols(this->weights_vec_[0], start_col, n_cols, reset_prob, rng);
  reduceToWeights(weights);
}

template <typename T>
void TransferRPUDevice<T>::getDeviceParameter(T **weights, std::vector<T *> &data_ptrs) {
  if (fully_hidden_) {
    // weight might have changed because of hidden weight change
    RPU::math::copy<T>(this->size_, weights[0], 1, this->weights_vec_[this->n_devices_ - 1][0], 1);
  }
  VectorRPUDevice<T>::getDeviceParameter(weights, data_ptrs);
}

template <typename T>
void TransferRPUDevice<T>::setDeviceParameter(T **out_weights, const std::vector<T *> &data_ptrs) {
  VectorRPUDevice<T>::setDeviceParameter(out_weights, data_ptrs);

  if (fully_hidden_) {
    // weight might have changed because of hidden weight change
    RPU::math::copy<T>(
        this->size_, this->weights_vec_[this->n_devices_ - 1][0], 1, out_weights[0], 1);
  }
}

template <typename T>
void TransferRPUDevice<T>::dumpExtra(RPU::state_t &extra, const std::string prefix) {
  VectorRPUDevice<T>::dumpExtra(extra, prefix);

  RPU::state_t state;

  RPU::insert(state, "current_slice_indices", current_slice_indices_);

  // RPU::insert(state, "transfer_vecs", transfer_vecs_);
  RPU::insert(state, "transfer_every", transfer_every_);

  transfer_fb_pass_->dumpExtra(state, "transfer_fb_pass");
  transfer_pwu_->dumpExtra(state, "transfer_pwu");

  RPU::insertWithPrefix(extra, state, prefix);
}

template <typename T>
void TransferRPUDevice<T>::loadExtra(
    const RPU::state_t &extra, const std::string prefix, bool strict) {

  VectorRPUDevice<T>::loadExtra(extra, prefix, strict);

  auto state = RPU::selectWithPrefix(extra, prefix);

  RPU::load(state, "current_slice_indices", current_slice_indices_, strict);
  if (state.count("transfer_vecs")) {
    RPU::load(state, "transfer_vecs", transfer_vecs_, strict);
  }
  if (state.count("transfer_every")) {
    RPU::load(state, "transfer_every", transfer_every_, strict);
  }
  transfer_fb_pass_->loadExtra(state, "transfer_fb_pass", strict);
  transfer_pwu_->loadExtra(state, "transfer_pwu", strict);
}

#undef COMMA
#undef LOOP_WITH_HIDDEN

template class TransferRPUDevice<float>;
#ifdef RPU_USE_DOUBLE
template class TransferRPUDevice<double>;
#endif
#ifdef RPU_USE_FP16
template class TransferRPUDevice<half_t>;
#endif

} // namespace RPU
