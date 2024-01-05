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

#include "rpu_onesided_device.h"
#include "math_util.h"
#include "utility_functions.h"
#include <memory>
#include <sstream>

namespace RPU {

template <typename T>
void OneSidedRPUDeviceMetaParameter<T>::printToStream(std::stringstream &ss) const {
  // every
  ss << "\t\bOneSided parameter: \n";
  ss << "\t refresh_every: \t" << refresh_every << " [MACC]" << std::endl;
  if (refresh_every > 0) {
    ss << "\t\bRefresh forward IO parameter:" << std::endl;
    refresh_io.printToStream(ss);
    ss << "\t\bRefresh update parameter:" << std::endl;
    refresh_up.printToStream(ss);
  }

  if (this->vec_par.size() > 0) {
    ss << "\t\bOneSided device parameter (" << this->vec_par[0]->getName() << "):" << std::endl;
    this->vec_par[0]->printToStream(ss);
  }
};

template <typename T> void OneSidedRPUDeviceMetaParameter<T>::initialize() {
  // different parameter settings are not allowed because we
  // like to be able to invert. For this we mirror copy the exact
  // DP . This does not work when the specifications of the RPU
  // arrays are different.

  VectorRPUDeviceMetaParameter<T>::initialize();

  if (!this->vec_par.size()) {
    RPU_FATAL("Expect non-empty vec par");
  }

  this->vec_par.resize(1);
  this->appendVecPar(*this->vec_par[0]);

  this->update_policy = VectorDeviceUpdatePolicy::All;
  this->first_update_idx = 0;
  this->gamma_vec.clear(); // fixed
};

template <typename T>
bool OneSidedRPUDeviceMetaParameter<T>::appendVecPar(const AbstractRPUDeviceMetaParameter<T> &par) {
  auto *dp = dynamic_cast<PulsedRPUDeviceMetaParameter<T> *>(par.clone());
  if (dp == nullptr) {
    return false;
  }
  if (this->vec_par.size() > 1) {
    return false;
  }
  if (this->vec_par.size() == 1 && (typeid(*this->vec_par[0]) != typeid(*dp))) {
    return false;
  }

  this->vec_par.push_back(std::unique_ptr<PulsedRPUDeviceMetaParameter<T>>(dp));
  return true;
};

template struct OneSidedRPUDeviceMetaParameter<float>;
#ifdef RPU_USE_DOUBLE
template struct OneSidedRPUDeviceMetaParameter<double>;
#endif
#ifdef RPU_USE_FP16
template struct OneSidedRPUDeviceMetaParameter<half_t>;
#endif

/************************************************************************************/
/* OneSidedRPUDevice*/

// ctor
template <typename T>
OneSidedRPUDevice<T>::OneSidedRPUDevice(int x_sz, int d_sz) : VectorRPUDevice<T>(x_sz, d_sz) {
  a_indices_.resize(x_sz);
  b_indices_.resize(x_sz);
}

template <typename T>
OneSidedRPUDevice<T>::OneSidedRPUDevice(
    int x_sz, int d_sz, const OneSidedRPUDeviceMetaParameter<T> &par, RealWorldRNG<T> *rng)
    : OneSidedRPUDevice<T>(x_sz, d_sz) {
  populate(par, rng);
}

// copy construcutor
template <typename T>
OneSidedRPUDevice<T>::OneSidedRPUDevice(const OneSidedRPUDevice<T> &other)
    : VectorRPUDevice<T>(other) {
  g_plus_ = other.g_plus_;
  g_minus_ = other.g_minus_;
  a_indices_ = other.a_indices_;
  b_indices_ = other.b_indices_;

  refresh_fb_pass_ = RPU::make_unique<ForwardBackwardPassIOManaged<T>>(*other.refresh_fb_pass_);
  refresh_pwu_ = RPU::make_unique<PulsedRPUWeightUpdater<T>>(*other.refresh_pwu_);
  refresh_counter_ = other.refresh_counter_;
  refresh_vecs_ = other.refresh_vecs_;
}

// copy assignment
template <typename T>
OneSidedRPUDevice<T> &OneSidedRPUDevice<T>::operator=(const OneSidedRPUDevice<T> &other) {

  OneSidedRPUDevice<T> tmp(other);
  swap(*this, tmp);
  return *this;
}

// move constructor
template <typename T> OneSidedRPUDevice<T>::OneSidedRPUDevice(OneSidedRPUDevice<T> &&other) {
  *this = std::move(other);
}

// move assignment
template <typename T>
OneSidedRPUDevice<T> &OneSidedRPUDevice<T>::operator=(OneSidedRPUDevice<T> &&other) {
  VectorRPUDevice<T>::operator=(std::move(other));
  g_plus_ = other.g_plus_;
  g_minus_ = other.g_minus_;
  a_indices_ = std::move(other.a_indices_);
  b_indices_ = std::move(other.b_indices_);

  refresh_fb_pass_ = std::move(other.refresh_fb_pass_);
  refresh_pwu_ = std::move(other.refresh_pwu_);
  refresh_counter_ = other.refresh_counter_;
  refresh_vecs_ = std::move(other.refresh_vecs_);

  return *this;
}

template <typename T> void OneSidedRPUDevice<T>::setRefreshVecs() {
  refresh_vecs_.resize(this->x_size_ * this->x_size_); //!!  square matrix
  std::fill(refresh_vecs_.begin(), refresh_vecs_.end(), (T)0.0);

  // initialize refresh vectors with unit vectors. This might be overridden
  for (size_t i = 0; i < refresh_vecs_.size(); i += this->x_size_ + 1) {
    refresh_vecs_[i] = 1.0;
  }
}

template <typename T> int OneSidedRPUDevice<T>::resetCounters(bool force) {
  refresh_counter_ = 0;
  return VectorRPUDevice<T>::resetCounters(force);
}

/*********************************************************************************/
/* populate */

template <typename T>
void OneSidedRPUDevice<T>::populate(
    const OneSidedRPUDeviceMetaParameter<T> &p, RealWorldRNG<T> *rng) {

  VectorRPUDevice<T>::populate(p, rng);

  // copy DP with switch of up<->down [to correct for up_down bias meaning]
  if (this->rpu_device_vec_.size() != 2) {
    RPU_FATAL("Expect exactly 2 devices.");
  }

  const auto &par = getPar();

  if (par.copy_inverted) {
    this->rpu_device_vec_[1]->copyInvertDeviceParameter(&*this->rpu_device_vec_[0]);
  }
  g_plus_ = 1;
  g_minus_ = 0;

  this->reduce_weightening_.resize(2);
  this->reduce_weightening_[g_plus_] = 1;
  this->reduce_weightening_[g_minus_] = -1;

  // init refresh

  this->setRefreshVecs();
  auto shared_rng = std::make_shared<RNG<T>>(0); // we just take a new one here (seeds...)
  refresh_fb_pass_ =
      RPU::make_unique<ForwardBackwardPassIOManaged<T>>(this->x_size_, this->d_size_, shared_rng);
  refresh_fb_pass_->populateFBParameter(par.refresh_io, par.refresh_io);

  refresh_pwu_ =
      RPU::make_unique<PulsedRPUWeightUpdater<T>>(this->x_size_, this->d_size_, shared_rng);
  refresh_pwu_->setUpPar(par.refresh_up);
}

/*********************************************************************************/
/* update */

template <typename T> bool OneSidedRPUDevice<T>::isInverted() const { return g_plus_ == 0; }

template <typename T> inline void OneSidedRPUDevice<T>::invert() {
  std::swap(g_plus_, g_minus_);
  this->reduce_weightening_[g_plus_] = 1;
  this->reduce_weightening_[g_minus_] = -1;
}

template <typename T>
void OneSidedRPUDevice<T>::finishUpdateCycle(
    T **weights, const PulsedUpdateMetaParameter<T> &up, T current_lr, int m_batch_info) {

  VectorRPUDevice<T>::finishUpdateCycle(weights, up, current_lr, m_batch_info);

  const auto &par = getPar();
  if (par.refresh_every > 0) {
    int refresh_every = par.refresh_every;
    if (par.units_in_mbatch) {
      refresh_every *= m_batch_info;
    }
    int refresh_count = 0;
    if (this->current_update_idx_ % refresh_every == 0) {
      refresh_count += refreshWeights();
    }
    if (refresh_count > 0) {
      this->reduceToWeights(weights);
    }
    refresh_counter_ += refresh_count;
  }
}

/*NOTE: assumes that initUpdateCycle is called already to init the containers!*/
template <typename T>
void OneSidedRPUDevice<T>::doSparseUpdate(
    T **weights, int d_index, const int *x_signed_indices, int x_count, int d_sign, RNG<T> *rng) {

  int a_count = 0;
  int b_count = 0;

  for (int jj = 0; jj < x_count; jj++) {
    int j_signed = x_signed_indices[jj];
    int sign = (j_signed < 0) ? -d_sign : d_sign;

    if (sign > 0) { // a per default g-
      a_indices_[a_count++] =
          (j_signed > 0)
              ? j_signed
              : -j_signed; // always one sided update (to positive side, see also -1 below)

    } else { // b per default g+
      b_indices_[b_count++] = (j_signed > 0) ? j_signed : -j_signed;
    }
  }

  if (a_count > 0) {
    this->rpu_device_vec_[g_minus_]->doSparseUpdate(
        this->weights_vec_[g_minus_], d_index, a_indices_.data(), a_count, -1, rng);
  }
  if (b_count > 0) {
    this->rpu_device_vec_[g_plus_]->doSparseUpdate(
        this->weights_vec_[g_plus_], d_index, b_indices_.data(), b_count, -1, rng);
  }
  // update the changed weight indices // note that this is very
  // repetitive since the same indices might be present all the
  // time. However, should be cached.
  for (int jj = 0; jj < x_count; jj++) {
    int j_signed = x_signed_indices[jj];
    int j = (j_signed < 0) ? -j_signed - 1 : j_signed - 1;
    weights[d_index][j] =
        this->weights_vec_[g_plus_][d_index][j] - this->weights_vec_[g_minus_][d_index][j];
  }
}

template <typename T>
void OneSidedRPUDevice<T>::doDenseUpdate(T **weights, int *coincidences, RNG<T> *rng) {

  coincidences_p_.resize(this->size_);
  coincidences_m_.resize(this->size_);

  PRAGMA_SIMD
  for (int i = 0; i < this->size_; i++) {
    int c = coincidences[i];

    coincidences_p_[i] = c < 0 ? c : 0;
    coincidences_m_[i] = c > 0 ? -c : 0;
  }

  this->rpu_device_vec_[g_plus_]->doDenseUpdate(
      this->weights_vec_[g_plus_], coincidences_p_.data(), rng);
  this->rpu_device_vec_[g_minus_]->doDenseUpdate(
      this->weights_vec_[g_minus_], coincidences_m_.data(), rng);

  // TODO: this might be better called in finish update cycle and only once per mini-batch?
  this->reduceToWeights(weights);
}

/********************************************************************************/
/* refresh weights  */

template <typename T>
inline bool OneSidedRPUDevice<T>::refreshCriterion(
    T &wp, T &wm, T &w_max, T &w_min, T &upper_thres, T &lower_thres) const {
  return (wp > wm) ? (wp / w_max > upper_thres && wm / w_min > lower_thres)
                   : (wp / w_max > lower_thres && wm / w_min > upper_thres);
}

template <typename T> int OneSidedRPUDevice<T>::refreshWeights() {

  const auto &par = getPar();

  if (refresh_p_tmp_.size() < (size_t)this->d_size_) {
    refresh_p_tmp_.resize(this->d_size_);
    refresh_m_tmp_.resize(this->d_size_);
    refresh_p_vec_.resize(this->d_size_);
    refresh_m_vec_.resize(this->d_size_);
  }

  T w_max = (T)fabsf(
      static_cast<PulsedRPUDeviceMetaParameter<T> &>(this->rpu_device_vec_[g_plus_]->getPar())
          .w_max);
  T w_min = (T)fabsf(
      static_cast<PulsedRPUDeviceMetaParameter<T> &>(this->rpu_device_vec_[g_minus_]->getPar())
          .w_max); // also max because of the one-sided-ness
  T upper_thres = par.refresh_upper_thres;
  T lower_thres = par.refresh_lower_thres;
  T **weights_p = this->weights_vec_[g_plus_];
  T **weights_m = this->weights_vec_[g_minus_];

  std::vector<int> refresh_indices;

  int refresh_counter = 0;

  for (int j_col = 0; j_col < this->x_size_; j_col++) {

    T *x = &refresh_vecs_[j_col * this->x_size_];
    // read out with forward pass
    refresh_fb_pass_->forwardVector(weights_p, x, 1, refresh_p_tmp_.data(), 1, (T)1.0, false);
    refresh_fb_pass_->forwardVector(weights_m, x, 1, refresh_m_tmp_.data(), 1, (T)1.0, false);

    int refresh_p_counter = 0;
    int refresh_m_counter = 0;

    refresh_indices.resize(0);
    for (int i = 0; i < this->d_size_; i++) {
      T wp = refresh_p_tmp_[i];
      T wm = refresh_m_tmp_[i];
      refresh_p_vec_[i] = 0.0;
      refresh_m_vec_[i] = 0.0;
      if (refreshCriterion(wp, wm, w_max, w_min, upper_thres, lower_thres)) {
        if (wp > wm) {
          refresh_p_vec_[i] = wp - wm;
          refresh_p_counter++;
        } else {
          refresh_m_vec_[i] = wm - wp;
          refresh_m_counter++;
        }
        refresh_indices.push_back(i * this->x_size_ + j_col);
      }
    }
    // reset (note: it is made sure during init that we have a PulsedRPUDevice)
    static_cast<PulsedRPUDevice<T> *>(&*this->rpu_device_vec_[g_plus_])
        ->resetAtIndices(weights_p, refresh_indices, this->rw_rng_);
    static_cast<PulsedRPUDevice<T> *>(&*this->rpu_device_vec_[g_minus_])
        ->resetAtIndices(weights_m, refresh_indices, this->rw_rng_);

    // do the refresh write
    // writing might be quite big. Probably need closed loop? Or can let training do the rest

    // CAUTION: this refresh does also increase the update
    // counter... Note that we use 1 as m_batch info since not
    // every time the update is called.

    if (refresh_p_counter > 0) {
      refresh_pwu_->updateVectorWithDevice(
          weights_p, x, 1, refresh_p_vec_.data(), 1,
          -1.0, // LR
          1, &*this->rpu_device_vec_[g_plus_]);
    }
    if (refresh_m_counter > 0) {
      refresh_pwu_->updateVectorWithDevice(
          weights_m, x, 1, refresh_m_vec_.data(), 1,
          -1.0, // LR
          1, &*this->rpu_device_vec_[g_minus_]);
    }

    refresh_counter += refresh_p_counter + refresh_m_counter;
  }

  return refresh_counter;
}

/********************************************************************************/
/* compute functions  */

template <typename T>
void OneSidedRPUDevice<T>::resetCols(
    T **weights, int start_col, int n_cols, T reset_prob, RealWorldRNG<T> &rng) {
  // CAUTION: reset_prob<1 means that it potentially resets only g_plus or g_minus !!!
  VectorRPUDevice<T>::resetCols(weights, start_col, n_cols, reset_prob, rng);
}

template <typename T> bool OneSidedRPUDevice<T>::onSetWeights(T **weights) {

  resetCounters(true);

  T *w = weights[0];

  PRAGMA_SIMD
  for (int i = 0; i < this->size_; i++) {
    this->weights_vec_[g_plus_][0][i] = w[i] > (T)0.0 ? w[i] : (T)0.0;
    this->weights_vec_[g_minus_][0][i] = w[i] < (T)0.0 ? -w[i] : (T)0.0;
  }

  this->rpu_device_vec_[g_plus_]->onSetWeights(this->weights_vec_[g_plus_]);
  this->rpu_device_vec_[g_minus_]->onSetWeights(this->weights_vec_[g_minus_]);

  this->reduceToWeights(weights);

  return true; // modified device thus true
}

template class OneSidedRPUDevice<float>;
#ifdef RPU_USE_DOUBLE
template class OneSidedRPUDevice<double>;
#endif
#ifdef RPU_USE_FP16
template class OneSidedRPUDevice<half_t>;
#endif

} // namespace RPU
