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

#include "rpu_vector_device.h"
#include "math_util.h"
#include "utility_functions.h"
#include <memory>
#include <sstream>

namespace RPU {

/******************************************************************************************/
/* Parameter classs */
template <typename T>
VectorRPUDeviceMetaParameter<T>::VectorRPUDeviceMetaParameter(
    const PulsedRPUDeviceMetaParameterBase<T> &dp, int n_devices) {
  vec_par.clear();
  for (int i = 0; i < n_devices; i++) {
    appendVecPar(dp);
  }
}

// copy construcutor
template <typename T>
VectorRPUDeviceMetaParameter<T>::VectorRPUDeviceMetaParameter(
    const VectorRPUDeviceMetaParameter<T> &other)
    : PulsedRPUDeviceMetaParameterBase<T>(other) {
  // deep copy
  for (size_t i = 0; i < other.vec_par.size(); i++) {
    appendVecPar(*other.vec_par[i]);
  }
  same_context = other.same_context;
  update_policy = other.update_policy;
  gamma_vec = other.gamma_vec;
  first_update_idx = other.first_update_idx;
}

// copy assignment
template <typename T>
VectorRPUDeviceMetaParameter<T> &
VectorRPUDeviceMetaParameter<T>::operator=(const VectorRPUDeviceMetaParameter<T> &other) {

  VectorRPUDeviceMetaParameter<T> tmp(other);
  swap(*this, tmp);
  return *this;
}

// move constructor
template <typename T>
VectorRPUDeviceMetaParameter<T>::VectorRPUDeviceMetaParameter(
    VectorRPUDeviceMetaParameter<T> &&other) noexcept {
  *this = std::move(other);
}

// move assignment
template <typename T>
VectorRPUDeviceMetaParameter<T> &
VectorRPUDeviceMetaParameter<T>::operator=(VectorRPUDeviceMetaParameter<T> &&other) noexcept {

  SimpleRPUDeviceMetaParameter<T>::operator=(std::move(other));
  same_context = other.same_context;
  update_policy = other.update_policy;
  first_update_idx = other.first_update_idx;
  gamma_vec = std::move(other.gamma_vec);
  vec_par = std::move(other.vec_par);

  return *this;
}

template <typename T>
bool VectorRPUDeviceMetaParameter<T>::appendVecPar(const AbstractRPUDeviceMetaParameter<T> &par) {
  auto *dp = dynamic_cast<PulsedRPUDeviceMetaParameterBase<T> *>(par.clone());
  if (dp == nullptr) {
    return false;
  } else {
    vec_par.push_back(std::unique_ptr<PulsedRPUDeviceMetaParameterBase<T>>(dp));
    return true;
  }
}

template struct VectorRPUDeviceMetaParameter<float>;
#ifdef RPU_USE_DOUBLE
template struct VectorRPUDeviceMetaParameter<double>;
#endif
#ifdef RPU_USE_FP16
template struct VectorRPUDeviceMetaParameter<half_t>;
#endif

/******************************************************************************************/
/* VectorRPUDevice*/

template <typename T> void VectorRPUDevice<T>::allocateContainers(int n_devices) {
  freeContainers();
  n_devices_ = n_devices;
  weights_vec_ = Array_3D_Get<T>(n_devices_, this->d_size_, this->x_size_);

  // set zero
  for (int k = 0; k < n_devices_; k++) {
    for (int i = 0; i < this->size_; i++) {
      weights_vec_[k][0][i] = (T)0.0;
    }
  }
}

template <typename T> void VectorRPUDevice<T>::freeContainers() {
  if (weights_vec_ != nullptr) {
    Array_3D_Free<T>(weights_vec_, n_devices_);
  }
  n_devices_ = 0;
  rpu_device_vec_.clear();
  reduce_weightening_.clear();
}

template <typename T>
VectorRPUDevice<T>::VectorRPUDevice(int x_sz, int d_sz) : PulsedRPUDeviceBase<T>(x_sz, d_sz) {}

template <typename T>
VectorRPUDevice<T>::VectorRPUDevice(
    int x_sz, int d_sz, const VectorRPUDeviceMetaParameter<T> &p, RealWorldRNG<T> *rng)
    : VectorRPUDevice(x_sz, d_sz) {
  populate(p, rng);
}

// dtor
template <typename T> VectorRPUDevice<T>::~VectorRPUDevice() { freeContainers(); }

// copy construcutor
template <typename T>
VectorRPUDevice<T>::VectorRPUDevice(const VectorRPUDevice<T> &other)
    : PulsedRPUDeviceBase<T>(other) {

  reduce_weightening_ = other.reduce_weightening_;
  current_device_idx_ = other.current_device_idx_;
  current_update_idx_ = other.current_update_idx_;

  allocateContainers(other.n_devices_);

  if (other.weights_vec_ != nullptr) {
    RPU::math::copy<T>(
        this->size_ * n_devices_, other.weights_vec_[0][0], 1, weights_vec_[0][0], 1);
  }

  rpu_device_vec_.clear();
  for (unsigned int k = 0; k < other.rpu_device_vec_.size(); k++) {
    rpu_device_vec_.push_back(
        std::unique_ptr<PulsedRPUDeviceBase<T>>(other.rpu_device_vec_[k]->clone()));
  }
}

// copy assignment
template <typename T>
VectorRPUDevice<T> &VectorRPUDevice<T>::operator=(const VectorRPUDevice<T> &other) {

  VectorRPUDevice<T> tmp(other);
  swap(*this, tmp);
  return *this;
}

// move constructor
template <typename T> VectorRPUDevice<T>::VectorRPUDevice(VectorRPUDevice<T> &&other) noexcept {

  *this = std::move(other);
}

// move assignment
template <typename T>
VectorRPUDevice<T> &VectorRPUDevice<T>::operator=(VectorRPUDevice<T> &&other) noexcept {

  PulsedRPUDeviceBase<T>::operator=(std::move(other));

  n_devices_ = other.n_devices_;
  current_device_idx_ = other.current_device_idx_;
  current_update_idx_ = other.current_update_idx_;

  weights_vec_ = other.weights_vec_;
  rpu_device_vec_ = std::move(other.rpu_device_vec_);
  reduce_weightening_ = other.reduce_weightening_;

  other.reduce_weightening_.clear();
  other.weights_vec_ = nullptr;

  return *this;
}

template <typename T> void VectorRPUDevice<T>::getDPNames(std::vector<std::string> &names) const {

  names.clear();
  for (unsigned int k = 0; k < rpu_device_vec_.size(); k++) {
    std::vector<std::string> n;
    rpu_device_vec_[k]->getDPNames(n);
    for (size_t i = 0; i < n.size(); i++) {
      std::ostringstream ss;
      ss << n[i] << "_" << k;
      names.push_back(ss.str());
    }
    std::ostringstream ss;
    ss << "hidden_weights_" << k;
    names.push_back(ss.str());
  }
}

template <typename T> int VectorRPUDevice<T>::getHiddenWeightsCount() const {
  if (!n_devices_) {
    return 0;
  }
  int m = n_devices_;
  for (int k = 0; k < n_devices_; k++) {
    m += rpu_device_vec_[k]->getHiddenWeightsCount();
  }
  return m;
}

template <typename T> void VectorRPUDevice<T>::setHiddenWeights(const std::vector<T> &data) {
  /* hidden weights are expected in the usual row-major format (first x_size then d_size)*/

  if (!n_devices_) {
    return;
  }
  size_t offset = 0;
  size_t size = this->size_;
  for (size_t k = 0; k < (size_t)n_devices_; k++) {
    size_t m = rpu_device_vec_[k]->getHiddenWeightsCount();
    if (data.size() <
        (size_t)offset +
            (size_t)(m + 1) * size) { // m+1 because we have a hidden in vector. this is the first
      RPU_FATAL("Size mismatch for hidden weights.");
    }
    // first this device's hidden weights
    for (size_t i = 0; i < size; i++) {
      weights_vec_[k][0][i] = data[offset + i];
    }
    offset += this->size_;
    std::vector<T> tmp_data(m * size);
    for (size_t i = 0; i < m * size; i++) {
      tmp_data[i] = data[offset + i];
    }
    offset += this->size_ * m;
    rpu_device_vec_[k]->setHiddenWeights(tmp_data);
  }
}

template <typename T> void VectorRPUDevice<T>::setHiddenUpdateIdx(int idx) {
  // we only change the current idx
  current_device_idx_ = idx;
}

template <typename T> int VectorRPUDevice<T>::getHiddenUpdateIdx() const {
  // we only change the current idx
  return current_device_idx_;
}

template <typename T>
void VectorRPUDevice<T>::getDeviceParameter(T **weights, std::vector<T *> &data_ptrs) {
  UNUSED(weights);

  std::vector<std::string> names;
  getDPNames(names);

  if (data_ptrs.size() < names.size()) {
    RPU_FATAL("Expected " << names.size() << " data pointers!");
  }

  size_t m = 0;
  for (size_t k = 0; k < rpu_device_vec_.size(); k++) {
    std::vector<std::string> n;
    rpu_device_vec_[k]->getDPNames(n);

    std::vector<T *> v;
    for (size_t i = 0; i < n.size(); i++) {
      v.push_back(data_ptrs[m + i]);
    }
    rpu_device_vec_[k]->getDeviceParameter(weights_vec_[k], v);
    m += n.size();

    // "hidden weights"
    for (int i = 0; i < this->size_; ++i) {
      data_ptrs[m][i] = weights_vec_[k][0][i];
    }
    m++;
  }
};

template <typename T>
void VectorRPUDevice<T>::setDeviceParameter(T **out_weights, const std::vector<T *> &data_ptrs) {

  std::vector<std::string> names;
  getDPNames(names);

  if (data_ptrs.size() < names.size()) {
    RPU_FATAL("Expected " << names.size() << " data pointers!");
  }

  T weight_granularity = (T)0.0;
  size_t m = 0;
  for (size_t k = 0; k < rpu_device_vec_.size(); k++) {
    std::vector<std::string> n;
    rpu_device_vec_[k]->getDPNames(n);

    std::vector<T *> v;
    for (size_t i = 0; i < n.size(); i++) {
      v.push_back(data_ptrs[m + i]);
    }
    m += n.size();

    // Following logic: The lowest hierachy weights will overwrite the
    // higher compound weights.
    for (int i = 0; i < this->size_; ++i) {
      weights_vec_[k][0][i] = data_ptrs[m][i];
    }

    rpu_device_vec_[k]->setDeviceParameter(weights_vec_[k], v);

    T weight_granularity_device = rpu_device_vec_[k]->getWeightGranularity();
    weight_granularity += weight_granularity_device;

    m++;
  }
  this->reduceToWeights(out_weights); // need to update the weights
  weight_granularity /= rpu_device_vec_.size();
  this->setWeightGranularity(weight_granularity);
};

template <typename T>
void VectorRPUDevice<T>::dumpExtra(RPU::state_t &extra, const std::string prefix) {

  PulsedRPUDeviceBase<T>::dumpExtra(extra, prefix);

  RPU::state_t state;

  for (size_t k = 0; k < rpu_device_vec_.size(); k++) {
    rpu_device_vec_[k]->dumpExtra(state, std::to_string(k));
  }
  RPU::insert(state, "reduce_weightening", reduce_weightening_);
  RPU::insert(state, "current_device_idx", current_device_idx_);
  RPU::insert(state, "current_update_idx", current_update_idx_);

  RPU::insertWithPrefix(extra, state, prefix);
}

template <typename T>
void VectorRPUDevice<T>::loadExtra(
    const RPU::state_t &extra, const std::string prefix, bool strict) {

  PulsedRPUDeviceBase<T>::loadExtra(extra, prefix, strict);

  auto state = RPU::selectWithPrefix(extra, prefix);

  for (size_t k = 0; k < rpu_device_vec_.size(); k++) {
    rpu_device_vec_[k]->loadExtra(state, std::to_string(k), strict);
  }
  RPU::load(state, "reduce_weightening", reduce_weightening_, strict);
  RPU::load(state, "current_device_idx", current_device_idx_, strict);
  RPU::load(state, "current_update_idx", current_update_idx_, strict);
}

template <typename T> void VectorRPUDevice<T>::printDP(int x_count, int d_count) const {

  int x_count1 = x_count;
  int d_count1 = d_count;
  if (x_count < 0 || x_count > this->x_size_)
    x_count1 = this->x_size_;

  if (d_count < 0 || d_count > this->d_size_)
    d_count1 = this->d_size_;

  for (unsigned int k = 0; k < rpu_device_vec_.size(); k++) {
    std::cout << "Vector device idx " << k << std::endl;
    rpu_device_vec_[k]->printDP(x_count1, d_count1);

    std::cout << "  Hidden weight idx " << k << std::endl;
    for (int i = 0; i < d_count1; ++i) {
      for (int j = 0; j < x_count1; ++j) {
        std::cout << weights_vec_[k][i][j] << ", ";
      }
    }
    std::cout << std::endl;
  }
}

template <typename T> int VectorRPUDevice<T>::resetCounters(bool force) {

  current_update_idx_ = 0;
  if (force || (getPar().update_policy != VectorDeviceUpdatePolicy::SingleFixed)) {
    current_device_idx_ = getPar().first_update_idx;
    return -1;
  } else {
    return current_device_idx_;
  }
}

/*********************************************************************************/
/* populate */

template <typename T>
void VectorRPUDevice<T>::populate(const VectorRPUDeviceMetaParameter<T> &p, RealWorldRNG<T> *rng) {

  PulsedRPUDeviceBase<T>::populate(p, rng);

  auto &par = getPar();
  allocateContainers((int)par.vec_par.size()); // will set n_devices
  resetCounters(true);

  // construct the sub devices and set the weight granularity as the
  // average of all
  rpu_device_vec_.clear();
  reduce_weightening_.clear();
  T weight_granularity = (T)0.0;
  for (int k = 0; k < n_devices_; k++) {
    rpu_device_vec_.push_back(std::unique_ptr<PulsedRPUDeviceBase<T>>(
        par.vec_par[k]->createDevice(this->x_size_, this->d_size_, rng)));
    weight_granularity += rpu_device_vec_.back()->getWeightGranularity();

    reduce_weightening_.push_back((T)1.0 / (T)n_devices_); // average per default
  }
  weight_granularity = weight_granularity / (T)n_devices_;
  this->setWeightGranularity(weight_granularity);

  // default weightening can be overwritten by given gamma_vec
  if (par.gamma_vec.size()) {
    if (par.gamma_vec.size() != (size_t)n_devices_) {
      RPU_FATAL("Gamma vector should have the same length as number of devices (or empty)!");
    }
    reduce_weightening_ = par.gamma_vec;
  }
}

/*********************************************************************************/
/* update */
template <typename T>
void VectorRPUDevice<T>::initUpdateCycle(
    T **weights,
    const PulsedUpdateMetaParameter<T> &up,
    T current_lr,
    int m_batch_info,
    const T *x_input,
    const int x_inc,
    const T *d_input,
    const int d_inc) {
  const auto &par = getPar();

  switch (par.update_policy) {
  case VectorDeviceUpdatePolicy::SingleRandom: {
    this->current_device_idx_ = (int)floorf(rw_rng_.sampleUniform() * (T)this->n_devices_);
    break;
  }
  case VectorDeviceUpdatePolicy::SingleSequential: {
    this->current_device_idx_ += 1;
    this->current_device_idx_ = this->current_device_idx_ % this->n_devices_;
    break;
  }
  default: {
  }
  }

  if (par.singleDeviceUpdate()) {
    rpu_device_vec_[current_device_idx_]->initUpdateCycle(
        weights, up, current_lr, m_batch_info, x_input, x_inc, d_input, d_inc);
  } else {
    for (size_t k = 0; k < rpu_device_vec_.size(); k++) {
      rpu_device_vec_[k]->initUpdateCycle(
          weights, up, current_lr, m_batch_info, x_input, x_inc, d_input, d_inc);
    }
  }
}

template <typename T>
void VectorRPUDevice<T>::doSparseUpdate(
    T **weights, int i, const int *x_signed_indices, int x_count, int d_sign, RNG<T> *rng) {

  if (getPar().singleDeviceUpdate()) {
    rpu_device_vec_[current_device_idx_]->doSparseUpdate(
        weights_vec_[current_device_idx_], i, x_signed_indices, x_count, d_sign, rng);
  } else {

    for (size_t k = 0; k < rpu_device_vec_.size(); k++) {
      rpu_device_vec_[k]->doSparseUpdate(
          weights_vec_[k], i, x_signed_indices, x_count, d_sign, rng);
    }
  }

  size_t m = rpu_device_vec_.size();
  for (int jj = 0; jj < x_count; jj++) {
    int j = x_signed_indices[jj];
    j = (j < 0) ? -j - 1 : j - 1;

    T w = (T)0.0;
    for (size_t k = 0; k < m; k++) {
      w += reduce_weightening_[k] * weights_vec_[k][i][j];
    }
    weights[i][j] = w;
  }
}

template <typename T>
void VectorRPUDevice<T>::doDenseUpdate(T **weights, int *coincidences, RNG<T> *rng) {
  if (getPar().singleDeviceUpdate()) {
    rpu_device_vec_[current_device_idx_]->doDenseUpdate(
        weights_vec_[current_device_idx_], coincidences, rng);
  } else {

    for (size_t k = 0; k < rpu_device_vec_.size(); k++) {
      rpu_device_vec_[k]->doDenseUpdate(weights_vec_[k], coincidences, rng);
    }
  }
  this->reduceToWeights(weights);
}

template <typename T>
void VectorRPUDevice<T>::finishUpdateCycle(
    T **weights, const PulsedUpdateMetaParameter<T> &up, T current_lr, int m_batch_info) {

  if (getPar().singleDeviceUpdate()) {
    rpu_device_vec_[current_device_idx_]->finishUpdateCycle(weights, up, current_lr, m_batch_info);
  } else {
    for (size_t k = 0; k < rpu_device_vec_.size(); k++) {
      rpu_device_vec_[k]->finishUpdateCycle(weights, up, current_lr, m_batch_info);
    }
  }

  // count updates
  current_update_idx_++;
}

/********************************************************************************/
/* compute functions  */
template <typename T> void VectorRPUDevice<T>::reduceToWeights(T **weights) const {
  // here: average weights from all devices
  // use gemv
  RPU::math::gemv(
      CblasColMajor, CblasNoTrans, this->size_, n_devices_, (T)1.0, weights_vec_[0][0], this->size_,
      reduce_weightening_.data(), 1, (T)0.0, weights[0], 1);
}

template <typename T> void VectorRPUDevice<T>::decayWeights(T **weights, bool bias_no_decay) {
  decayWeights(weights, (T)1.0, bias_no_decay);
}

template <typename T>
void VectorRPUDevice<T>::decayWeights(T **weights, T alpha, bool bias_no_decay) {

#pragma omp parallel for
  for (int k = 0; k < (int)rpu_device_vec_.size(); k++) {
    rpu_device_vec_[k]->decayWeights(weights_vec_[k], alpha, bias_no_decay);
  }
  reduceToWeights(weights);
}

template <typename T>
void VectorRPUDevice<T>::driftWeights(T **weights, T time_since_last_call, RNG<T> &rng) {

#pragma omp parallel for
  for (int k = 0; k < (int)rpu_device_vec_.size(); k++) {
    rpu_device_vec_[k]->driftWeights(weights_vec_[k], time_since_last_call, rng);
  }
  reduceToWeights(weights);
}

template <typename T> void VectorRPUDevice<T>::diffuseWeights(T **weights, RNG<T> &rng) {
#pragma omp parallel for
  for (int k = 0; k < (int)rpu_device_vec_.size(); k++) {
    rpu_device_vec_[k]->diffuseWeights(weights_vec_[k], rng);
  }
  reduceToWeights(weights);
}

template <typename T> void VectorRPUDevice<T>::clipWeights(T **weights, T clip) {
#pragma omp parallel for
  for (int k = 0; k < (int)rpu_device_vec_.size(); k++) {
    rpu_device_vec_[k]->clipWeights(weights_vec_[k], clip);
  }
  reduceToWeights(weights);
}

template <typename T>
void VectorRPUDevice<T>::resetCols(
    T **weights, int start_col, int n_cols, T reset_prob, RealWorldRNG<T> &rng) {
#pragma omp parallel for
  for (int k = 0; k < (int)rpu_device_vec_.size(); k++) {
    rpu_device_vec_[k]->resetCols(weights_vec_[k], start_col, n_cols, reset_prob, rng);
  }
  reduceToWeights(weights);
}

template <typename T> bool VectorRPUDevice<T>::onSetWeights(T **weights) {
  // note: we use this to update the internal weights for each device.
  // all weights are set to *identical* values...

  T *w = weights[0];
  int fixed_index = resetCounters();

  // e.g. apply hard bounds
  for (int k = 0; k < (int)rpu_device_vec_.size(); k++) {
    // only in case of SingleFixed policy set the weight chosen.
    if ((fixed_index >= 0 && fixed_index == k) || (fixed_index < 0)) {
      for (int i = 0; i < this->size_; i++) {
        weights_vec_[k][0][i] = w[i];
      }
    }

    // reset all counter etc. always (hidden parameter might have changed)
    rpu_device_vec_[k]->onSetWeights(weights_vec_[k]);
  }
  reduceToWeights(weights);

  return true; // modified device thus true
}

template class VectorRPUDevice<float>;
#ifdef RPU_USE_DOUBLE
template class VectorRPUDevice<double>;
#endif
#ifdef RPU_USE_FP16
template class VectorRPUDevice<half_t>;
#endif

} // namespace RPU
