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

#include "rpu_difference_device.h"
#include "math_util.h"
#include "utility_functions.h"
#include <memory>
#include <sstream>

namespace RPU {

/************************************************************************************/
/* DifferenceRPUDevice*/

// ctor
template <typename T>
DifferenceRPUDevice<T>::DifferenceRPUDevice(int x_sz, int d_sz) : VectorRPUDevice<T>(x_sz, d_sz) {}

template <typename T>
DifferenceRPUDevice<T>::DifferenceRPUDevice(
    int x_sz, int d_sz, const DifferenceRPUDeviceMetaParameter<T> &par, RealWorldRNG<T> *rng)
    : DifferenceRPUDevice<T>(x_sz, d_sz) {
  populate(par, rng);
}

// copy construcutor
template <typename T>
DifferenceRPUDevice<T>::DifferenceRPUDevice(const DifferenceRPUDevice<T> &other)
    : VectorRPUDevice<T>(other) {
  g_plus_ = other.g_plus_;
  g_minus_ = other.g_minus_;
  a_indices_ = other.a_indices_;
  b_indices_ = other.b_indices_;
}

// copy assignment
template <typename T>
DifferenceRPUDevice<T> &DifferenceRPUDevice<T>::operator=(const DifferenceRPUDevice<T> &other) {

  DifferenceRPUDevice<T> tmp(other);
  swap(*this, tmp);
  return *this;
}

// move constructor
template <typename T> DifferenceRPUDevice<T>::DifferenceRPUDevice(DifferenceRPUDevice<T> &&other) {
  *this = std::move(other);
}

// move assignment
template <typename T>
DifferenceRPUDevice<T> &DifferenceRPUDevice<T>::operator=(DifferenceRPUDevice<T> &&other) {
  VectorRPUDevice<T>::operator=(std::move(other));
  g_plus_ = other.g_plus_;
  g_minus_ = other.g_minus_;
  a_indices_ = std::move(other.a_indices_);
  b_indices_ = std::move(other.b_indices_);

  return *this;
}

/*********************************************************************************/
/* populate */

template <typename T>
void DifferenceRPUDevice<T>::populate(
    const DifferenceRPUDeviceMetaParameter<T> &p, RealWorldRNG<T> *rng) {

  VectorRPUDevice<T>::populate(p, rng);

  // copy DP with switch of up<->down [to correct for up_down bias meaning]
  if (this->rpu_device_vec_.size() != 2) {
    RPU_FATAL("Expect exactly 2 devices.");
  }

  this->rpu_device_vec_[1]->copyInvertDeviceParameter(&*this->rpu_device_vec_[0]);

  g_plus_ = 1;
  g_minus_ = 0;

  this->reduce_weightening_.resize(2);
  this->reduce_weightening_[g_plus_] = 1;
  this->reduce_weightening_[g_minus_] = -1;
}

/*********************************************************************************/
/* update */

template <typename T> inline bool DifferenceRPUDevice<T>::isInverted() const {
  return g_plus_ == 0;
}

template <typename T> inline void DifferenceRPUDevice<T>::invert() {
  std::swap(g_plus_, g_minus_);
  this->reduce_weightening_[g_plus_] = 1;
  this->reduce_weightening_[g_minus_] = -1;
}

template <typename T>
void DifferenceRPUDevice<T>::initUpdateCycle(
    T **weights, const PulsedUpdateMetaParameter<T> &up, T current_lr, int m_batch_info) {
  if (a_indices_.size() < (size_t)up.desired_BL) {
    a_indices_.resize(up.desired_BL);
    b_indices_.resize(up.desired_BL);
  }
}

/*NOTE: assumes that initUpdateCycle is called already to init the containers!*/
template <typename T>
void DifferenceRPUDevice<T>::doSparseUpdate(
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

/********************************************************************************/
/* compute functions  */

template <typename T>
void DifferenceRPUDevice<T>::resetCols(
    T **weights, int start_col, int n_cols, T reset_prob, RealWorldRNG<T> &rng) {
  // CAUTION: reset_prob<1 means that it potentially resets only g_plus or g_minus !!!
  VectorRPUDevice<T>::resetCols(weights, start_col, n_cols, reset_prob, rng);
}

template <typename T> bool DifferenceRPUDevice<T>::onSetWeights(T **weights) {
  // note: we use this to update the internal weights for each device.
  // all weights are set to *identical* values...

  T *w = weights[0];

  PRAGMA_SIMD
  for (int i = 0; i < this->size_; i++) {
    this->weights_vec_[g_plus_][0][i] = w[i] > 0 ? w[i] : (T)0.0;
    this->weights_vec_[g_minus_][0][i] = w[i] < 0 ? -w[i] : (T)0.0;
    ;
  }

  this->rpu_device_vec_[g_plus_]->onSetWeights(this->weights_vec_[g_plus_]);
  this->rpu_device_vec_[g_minus_]->onSetWeights(this->weights_vec_[g_minus_]);

  this->reduceToWeights(weights);

  return true; // modified device thus true
}

template class DifferenceRPUDevice<float>;
#ifdef RPU_USE_DOUBLE
template class DifferenceRPUDevice<double>;
#endif

} // namespace RPU
