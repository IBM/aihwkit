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

#include "rpu_mixedprec_device_base.h"
#include "math_util.h"
#include "utility_functions.h"
#include <algorithm>
#include <memory>
#include <sstream>

namespace RPU {

/******************************************************************************************/
/* MixedPrecRPUDeviceBaseMetaParameter*/

template <typename T>
void MixedPrecRPUDeviceBaseMetaParameter<T>::printToStream(std::stringstream &ss) const {

  if (granularity > (T)0.0) {
    ss << "\t granularity: \t\t";
    ss << granularity << std::endl;
  }

  if (n_rows_per_transfer != 0) {
    ss << "\t transfer_every: \t";
    ss << transfer_every;
    ss << " [in mbatches]";
    ss << std::endl;
  }

  if (n_rows_per_transfer == 0) {
    ss << "\t no transfer!";
  } else if (n_rows_per_transfer < 0) {
    ss << "\t full transfer.";
  } else {
    ss << "\t n_rows_per_transfer: \t" << n_rows_per_transfer;
  }

  if (n_rows_per_transfer > 0 && random_row) {
    ss << "\t [random row]";
  }
  ss << std::endl;

  if (compute_sparsity) {
    ss << "\t compute_sparsity: \t" << compute_sparsity << std::endl;
  }

  if (device_par) {
    ss << "\t\bDevice Parameter: " << device_par->getName() << std::endl;
    ss << "   ";
    device_par->printToStream(ss);
  }
};

template <typename T> void MixedPrecRPUDeviceBaseMetaParameter<T>::initialize() {
  if (!this->_par_initialized) {
    SimpleRPUDeviceMetaParameter<T>::initialize();

    if (!device_par) {
      RPU_FATAL("Expect device_par to be defined.");
    }
    device_par->initialize();
  }
};

// copy construcutor
template <typename T>
MixedPrecRPUDeviceBaseMetaParameter<T>::MixedPrecRPUDeviceBaseMetaParameter(
    const MixedPrecRPUDeviceBaseMetaParameter<T> &other)
    : SimpleRPUDeviceMetaParameter<T>(other) {
  // deep copy
  device_par = std::unique_ptr<AbstractRPUDeviceMetaParameter<T>>(other.device_par->clone());

  transfer_every = other.transfer_every;
  n_rows_per_transfer = other.n_rows_per_transfer;
  random_row = other.random_row;
  granularity = other.granularity;
  compute_sparsity = other.compute_sparsity;
}

// copy assignment
template <typename T>
MixedPrecRPUDeviceBaseMetaParameter<T> &MixedPrecRPUDeviceBaseMetaParameter<T>::operator=(
    const MixedPrecRPUDeviceBaseMetaParameter<T> &other) {

  MixedPrecRPUDeviceBaseMetaParameter<T> tmp(other);
  swap(*this, tmp);
  return *this;
}

// move constructor
template <typename T>
MixedPrecRPUDeviceBaseMetaParameter<T>::MixedPrecRPUDeviceBaseMetaParameter(
    MixedPrecRPUDeviceBaseMetaParameter<T> &&other) noexcept {

  *this = std::move(other);
}

// move assignment
template <typename T>
MixedPrecRPUDeviceBaseMetaParameter<T> &MixedPrecRPUDeviceBaseMetaParameter<T>::operator=(
    MixedPrecRPUDeviceBaseMetaParameter<T> &&other) noexcept {

  SimpleRPUDeviceMetaParameter<T>::operator=(std::move(other));

  device_par = std::move(other.device_par);

  transfer_every = other.transfer_every;
  n_rows_per_transfer = other.n_rows_per_transfer;
  random_row = other.random_row;
  granularity = other.granularity;
  compute_sparsity = other.compute_sparsity;

  return *this;
}

template <typename T>
bool MixedPrecRPUDeviceBaseMetaParameter<T>::setDevicePar(
    const AbstractRPUDeviceMetaParameter<T> &par) {
  auto *dp = dynamic_cast<const PulsedRPUDeviceMetaParameterBase<T> *>(&par);
  if (dp == nullptr) {
    return false;
  } else {
    device_par = dp->cloneUnique();
    return true;
  }
}

template struct MixedPrecRPUDeviceBaseMetaParameter<float>;
#ifdef RPU_USE_DOUBLE
template struct MixedPrecRPUDeviceBaseMetaParameter<double>;
#endif
#ifdef RPU_USE_FP16
template struct MixedPrecRPUDeviceBaseMetaParameter<half_t>;
#endif

/******************************************************************************************/

#define CHECK_RPU_DEVICE_INIT                                                                      \
  if (!this->rpu_device_) {                                                                        \
    RPU_FATAL("First populate device then set the weights");                                       \
  }

// ctor
template <typename T>
MixedPrecRPUDeviceBase<T>::MixedPrecRPUDeviceBase(int x_sz, int d_sz)
    : SimpleRPUDevice<T>(x_sz, d_sz) {}

// copy construcutor
template <typename T>
MixedPrecRPUDeviceBase<T>::MixedPrecRPUDeviceBase(const MixedPrecRPUDeviceBase<T> &other)
    : SimpleRPUDevice<T>(other) {

  transfer_pwu_ = RPU::make_unique<PulsedRPUWeightUpdater<T>>(*other.transfer_pwu_);
  rpu_device_ = other.rpu_device_->cloneUnique();

  current_row_index_ = other.current_row_index_;
  current_update_index_ = other.current_update_index_;
  transfer_tmp_ = other.transfer_tmp_;
  transfer_d_vecs_ = other.transfer_d_vecs_;
  avg_sparsity_ = other.avg_sparsity_;
  granularity_ = other.granularity_;
}

// copy assignment
template <typename T>
MixedPrecRPUDeviceBase<T> &
MixedPrecRPUDeviceBase<T>::operator=(const MixedPrecRPUDeviceBase<T> &other) {

  MixedPrecRPUDeviceBase<T> tmp(other);
  swap(*this, tmp);
  return *this;
}

// move constructor
template <typename T>
MixedPrecRPUDeviceBase<T>::MixedPrecRPUDeviceBase(MixedPrecRPUDeviceBase<T> &&other) noexcept {
  *this = std::move(other);
}

// move assignment
template <typename T>
MixedPrecRPUDeviceBase<T> &
MixedPrecRPUDeviceBase<T>::operator=(MixedPrecRPUDeviceBase<T> &&other) noexcept {
  SimpleRPUDevice<T>::operator=(std::move(other));

  transfer_pwu_ = std::move(other.transfer_pwu_);
  rpu_device_ = std::move(other.rpu_device_);

  current_row_index_ = other.current_row_index_;
  current_update_index_ = other.current_update_index_;
  transfer_tmp_ = std::move(other.transfer_tmp_);
  transfer_d_vecs_ = std::move(other.transfer_d_vecs_);
  avg_sparsity_ = other.avg_sparsity_;
  granularity_ = other.granularity_;

  return *this;
}

/*********************************************************************************/
/* populate */

template <typename T>
void MixedPrecRPUDeviceBase<T>::populate(
    const MixedPrecRPUDeviceBaseMetaParameter<T> &p, RealWorldRNG<T> *rng) {

  SimpleRPUDevice<T>::populate(p, rng);
  auto &par = getPar();

  current_update_index_ = 0;
  current_row_index_ = 0;
  avg_sparsity_ = 0;
  up_ptr_ = nullptr;

  if (par.device_par == nullptr) {
    RPU_FATAL("Expect device parameter in device_par!");
  }

  rpu_device_ = par.device_par->createDeviceUnique(this->x_size_, this->d_size_, rng);

  auto shared_rng = std::make_shared<RNG<T>>(0); // we just take a new one here (seeds...)
  transfer_pwu_ =
      RPU::make_unique<PulsedRPUWeightUpdater<T>>(this->x_size_, this->d_size_, shared_rng);

  granularity_ = 0.0;
  if (dynamic_cast<PulsedRPUDeviceBase<T> *>(&*rpu_device_) != nullptr) {
    granularity_ = dynamic_cast<PulsedRPUDeviceBase<T> *>(&*rpu_device_)->getWeightGranularity();
  }
  if (par.granularity > (T)0.0) {
    // overwrites
    granularity_ = par.granularity;
  }
  if (granularity_ <= (T)0.0) {
    RPU_FATAL("Cannot establish granularity from device. Need explicit setting >=0.");
  }
}

template <typename T>
void MixedPrecRPUDeviceBase<T>::dumpExtra(RPU::state_t &extra, const std::string prefix) {
  SimpleRPUDevice<T>::dumpExtra(extra, prefix);

  RPU::state_t state;

  rpu_device_->dumpExtra(state, "rpu_device");
  transfer_pwu_->dumpExtra(state, "transfer_pwu");

  RPU::insert(state, "granularity", granularity_);
  RPU::insert(state, "transfer_tmp", transfer_tmp_);
  RPU::insert(state, "current_row_index", current_row_index_);
  RPU::insert(state, "current_update_index", current_update_index_);
  RPU::insert(state, "avg_sparsity", avg_sparsity_);

  // transfer_d_vecs not handled (generated on the fly)

  RPU::insertWithPrefix(extra, state, prefix);
}

template <typename T>
void MixedPrecRPUDeviceBase<T>::loadExtra(
    const RPU::state_t &extra, const std::string prefix, bool strict) {
  SimpleRPUDevice<T>::loadExtra(extra, prefix, strict);

  auto state = RPU::selectWithPrefix(extra, prefix);

  rpu_device_->loadExtra(state, "rpu_device", strict);
  transfer_pwu_->loadExtra(state, "transfer_pwu", strict);

  RPU::load(state, "granularity", granularity_, strict);
  RPU::load(state, "transfer_tmp", transfer_tmp_, strict);
  RPU::load(state, "current_row_index", current_row_index_, strict);
  RPU::load(state, "current_update_index", current_update_index_, strict);
  RPU::load(state, "avg_sparsity", avg_sparsity_, strict);
}

/*********************************************************************************/
/* transfer */

template <typename T>
void MixedPrecRPUDeviceBase<T>::doTransfer(T **weights, const T lr, const int m_batch_info) {
  const auto &par = getPar();
  int every = par.transfer_every * m_batch_info;
  if (every > 0 && current_update_index_ > 0 && (current_update_index_ % every == 0)) {
    transfer(weights, lr);
  }
}

template <typename T>
void MixedPrecRPUDeviceBase<T>::setUpPar(const PulsedUpdateMetaParameter<T> &up) {
  if (&up != up_ptr_) {
    up_ptr_ = &up;
    PulsedUpdateMetaParameter<T> up_modified(up);
    // should be both true to get the desired step.
    up_modified.update_management = true;
    up_modified.update_bl_management = true;

    transfer_pwu_->setUpPar(up_modified);
  }
}

template <typename T> void MixedPrecRPUDeviceBase<T>::computeSparsity(const int kx, const int kd) {
  const auto &par = getPar();
  if (par.compute_sparsity) {
    avg_sparsity_ = ((T)current_update_index_ * avg_sparsity_ +
                     (T)((this->d_size_ - kd) * (this->x_size_ - kx)) / (T)this->size_) /
                    (T)(current_update_index_ + 1);
  }
}

template <typename T> void MixedPrecRPUDeviceBase<T>::transfer(T **weights, const T lr) {
  // updating the matrix with rows of using one-hot transfer vectors

  const auto &par = getPar();
  if (par.n_rows_per_transfer == 0 || (T)fabsf(lr) == (T)0) {
    return;
  }
  int n_transfers = par.n_rows_per_transfer;
  if (n_transfers < 0) {
    n_transfers = this->d_size_;
  }
  n_transfers = MIN(n_transfers, this->d_size_);
  int i_row = current_row_index_;
  if (par.random_row && (n_transfers < this->d_size_)) {
    i_row = MAX(
        MIN((int)floorf(this->rw_rng_.sampleUniform() * (T)this->d_size_), this->d_size_ - 1), 0);
  }

  int d2_size = this->d_size_ * this->d_size_;
  // create transfer vectors on the fly
  if (transfer_d_vecs_.size() < (size_t)d2_size) {
    transfer_d_vecs_.resize(d2_size, (T)0.0);
    for (int i = 0; i < d2_size; i += this->d_size_ + 1) {
      transfer_d_vecs_[i] = 1.0;
    }
  }

  T *tvec = transfer_d_vecs_.data() + (size_t)i_row * this->d_size_;
  int n_rest = this->d_size_ - i_row;

  if (n_rest < n_transfers) {
    // rest
    forwardUpdate(weights, lr, i_row, tvec, n_rest, false);
    // from beginning
    forwardUpdate(weights, lr, 0, transfer_d_vecs_.data(), n_transfers - n_rest, false);

  } else {
    forwardUpdate(weights, lr, i_row, tvec, n_transfers, false);
  }
  current_row_index_ = (i_row + n_transfers) % this->d_size_;
}

template <typename T> bool MixedPrecRPUDeviceBase<T>::onSetWeights(T **weights) {

  CHECK_RPU_DEVICE_INIT;

  // weight setting should reset the internal parameters
  current_update_index_ = 0;
  current_row_index_ = 0;
  avg_sparsity_ = 0;

  this->rpu_device_->onSetWeights(weights);

  return true; // modified device thus true
}

template <typename T>
void MixedPrecRPUDeviceBase<T>::getDPNames(std::vector<std::string> &names) const {

  CHECK_RPU_DEVICE_INIT;

  names.clear();
  rpu_device_->getDPNames(names);
  std::ostringstream ss;
  ss << "hidden_weights_chi";
  names.push_back(ss.str());
}

template <typename T>
void MixedPrecRPUDeviceBase<T>::getDeviceParameter(T **weights, std::vector<T *> &data_ptrs) {

  CHECK_RPU_DEVICE_INIT;

  std::vector<std::string> names;
  getDPNames(names);

  if (data_ptrs.size() < names.size()) {
    RPU_FATAL("Expected " << names.size() << " data pointers!");
  }

  std::vector<T *> v(data_ptrs.begin(), data_ptrs.end() - 1);
  rpu_device_->getDeviceParameter(weights, v);

  // "hidden weights Chi"
  size_t m = names.size() - 1;
  getChi(data_ptrs[m]);
}

template <typename T> int MixedPrecRPUDeviceBase<T>::getHiddenWeightsCount() const {

  CHECK_RPU_DEVICE_INIT;
  return rpu_device_->getHiddenWeightsCount() + 1;
}

template <typename T> void MixedPrecRPUDeviceBase<T>::setHiddenWeights(const std::vector<T> &data) {
  /* hidden weights are expected in the usual row-major format (first x_size then d_size)*/

  CHECK_RPU_DEVICE_INIT;

  size_t m = rpu_device_->getHiddenWeightsCount();
  if (data.size() < (size_t)((m + 1) * this->size_)) {
    RPU_FATAL("Size mismatch for hidden weights.");
  }

  std::vector<T> v(data.begin(), data.end() - this->size_);
  rpu_device_->setHiddenWeights(v);

  size_t offset = m * this->size_;
  setChi(data.data() + offset);
}

template <typename T>
void MixedPrecRPUDeviceBase<T>::setDeviceParameter(
    T **out_weights, const std::vector<T *> &data_ptrs) {

  CHECK_RPU_DEVICE_INIT;
  rpu_device_->setDeviceParameter(out_weights, data_ptrs);
};

template <typename T> void MixedPrecRPUDeviceBase<T>::printDP(int x_count, int d_count) const {

  CHECK_RPU_DEVICE_INIT;

  size_t x_count1 = MAX(MIN(x_count, this->x_size_), 0);
  size_t d_count1 = MAX(MIN(d_count, this->d_size_), 0);

  rpu_device_->printDP((int)x_count1, (int)d_count1);

  T *chi = new T[this->size_];
  getChi(chi);
  std::cout << "  Hidden weight [Chi] " << std::endl;
  for (size_t i = 0; i < d_count1; ++i) {
    for (size_t j = 0; j < x_count1; ++j) {
      size_t k = j + i * this->x_size_;
      std::cout << chi[k] << ", ";
    }
    std::cout << ";" << std::endl;
    ;
  }
  std::cout << std::endl;
}

template <typename T>
void MixedPrecRPUDeviceBase<T>::decayWeights(T **weights, bool bias_no_decay) {
  CHECK_RPU_DEVICE_INIT;
  this->rpu_device_->decayWeights(weights, bias_no_decay);
}

template <typename T>
void MixedPrecRPUDeviceBase<T>::decayWeights(T **weights, T alpha, bool bias_no_decay) {
  CHECK_RPU_DEVICE_INIT;
  this->rpu_device_->decayWeights(weights, alpha, bias_no_decay);
}

template <typename T> void MixedPrecRPUDeviceBase<T>::diffuseWeights(T **weights, RNG<T> &rng) {
  CHECK_RPU_DEVICE_INIT;
  this->rpu_device_->diffuseWeights(weights, rng);
}

template <typename T> void MixedPrecRPUDeviceBase<T>::clipWeights(T **weights, T clip) {
  CHECK_RPU_DEVICE_INIT;
  this->rpu_device_->clipWeights(weights, clip);
}

template <typename T>
void MixedPrecRPUDeviceBase<T>::driftWeights(T **weights, T time_since_last_call, RNG<T> &rng) {
  CHECK_RPU_DEVICE_INIT;
  this->rpu_device_->driftWeights(weights, time_since_last_call, rng);
}

template <typename T>
void MixedPrecRPUDeviceBase<T>::resetCols(
    T **weights, int start_col, int n_cols, T reset_prob, RealWorldRNG<T> &rng) {
  CHECK_RPU_DEVICE_INIT;
  this->rpu_device_->resetCols(weights, start_col, n_cols, reset_prob, rng);
}

#undef CHECK_RPU_DEVICE_INIT

template class MixedPrecRPUDeviceBase<float>;
#ifdef RPU_USE_DOUBLE
template class MixedPrecRPUDeviceBase<double>;
#endif
#ifdef RPU_USE_FP16
template class MixedPrecRPUDeviceBase<half_t>;
#endif

} // namespace RPU
