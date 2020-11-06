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

#include "rpu_pulsed_meta_parameter.h"
#include "rpucuda_difference_device.h"
#include <memory>

namespace RPU {

/******************************************************************************************/
/* DifferenceRPUDeviceCuda

   CUDA implementation of DifferenceRPUDevice

*/

template <typename T>
DifferenceRPUDeviceCuda<T>::DifferenceRPUDeviceCuda(
    CudaContext *c, const DifferenceRPUDevice<T> &rpu_device) {
  this->context_ = c;
  populateFrom(rpu_device); // use populate to call parent
};

// copy construcutor
template <typename T>
DifferenceRPUDeviceCuda<T>::DifferenceRPUDeviceCuda(const DifferenceRPUDeviceCuda<T> &other)
    : VectorRPUDeviceCuda<T>(other) {
  g_plus_ = other.g_plus_;
  g_minus_ = other.g_minus_;
  dev_reduce_weightening_inverted_ = nullptr;
  if (other.dev_reduce_weightening_inverted_ != nullptr) {
    dev_reduce_weightening_inverted_ = RPU::make_unique<CudaArray<T>>(this->context_, 2);
    dev_reduce_weightening_inverted_->assign(*other.dev_reduce_weightening_inverted_);
    dev_reduce_weightening_inverted_->synchronize();
  }
};

// copy assignment
template <typename T>
DifferenceRPUDeviceCuda<T> &
DifferenceRPUDeviceCuda<T>::operator=(const DifferenceRPUDeviceCuda<T> &other) {
  DifferenceRPUDeviceCuda<T> tmp(other);
  swap(*this, tmp);
  return *this;
};

// move constructor
template <typename T>
DifferenceRPUDeviceCuda<T>::DifferenceRPUDeviceCuda(DifferenceRPUDeviceCuda<T> &&other) {
  *this = std::move(other);
};

// move assignment
template <typename T>
DifferenceRPUDeviceCuda<T> &
DifferenceRPUDeviceCuda<T>::operator=(DifferenceRPUDeviceCuda<T> &&other) {
  VectorRPUDeviceCuda<T>::operator=(std::move(other));
  g_plus_ = other.g_plus_;
  g_minus_ = other.g_minus_;
  dev_reduce_weightening_inverted_ = std::move(other.dev_reduce_weightening_inverted_);
  return *this;
};

template <typename T>
void DifferenceRPUDeviceCuda<T>::populateFrom(const AbstractRPUDevice<T> &rpu_device_in) {

  const auto *rpu_device = dynamic_cast<const DifferenceRPUDevice<T> *>(&rpu_device_in);
  if (rpu_device == nullptr) {
    RPU_FATAL("Expect DifferenceRPUDevice.");
  }

  VectorRPUDeviceCuda<T>::populateFrom(rpu_device_in);

  rpu_device->getGIndices(g_plus_, g_minus_);

  // inverted
  std::vector<T> rw_inv(2);
  const T *rw = rpu_device->getReduceWeightening();
  rw_inv[0] = rw[1];
  rw_inv[1] = rw[0];

  this->dev_reduce_weightening_->assign(rw);
  dev_reduce_weightening_inverted_ = RPU::make_unique<CudaArray<T>>(this->context_, 2, &rw_inv[0]);

  this->context_->synchronize();
}

template <typename T>
void DifferenceRPUDeviceCuda<T>::resetCols(
    T *dev_weights, int start_col, int n_cols, T reset_prob) {
  VectorRPUDeviceCuda<T>::resetCols(dev_weights, start_col, n_cols, reset_prob);
}

template <typename T> inline bool DifferenceRPUDevice<T>::isInverted() const {
  return g_plus_ == 0;
}

template <typename T> inline void DifferenceRPUDevice<T>::invert() {
  std::swap(g_plus_, g_minus_);
  std::swap(this->dev_reduce_weightening_, this->dev_reduce_weightening_inverted_);
}

template <typename T>
pwukpvec_t<T> DifferenceRPUDeviceCuda<T>::getUpdateKernels(
    int m_batch, int nK32, int use_bo64, bool out_trans, const PulsedUpdateMetaParameter<T> &up) {
  if (this->rpucuda_device_vec_.size() != 2) {
    RPU_FATAL("Expect exactly two devices.");
  }
  if (getPar().singleDeviceUpdate()) {
    RPU_FATAL("Single device update not supported for Difference Device");
  }

  return VectorRPUDeviceCuda<T>::getUpdateKernels(m_batch, nK32, use_bo64, out_trans, up);
}

template <typename T>
void DifferenceRPUDeviceCuda<T>::runUpdateKernel(
    pwukp_t<T> kpars,
    CudaContext *up_context,
    T *dev_weights,
    int m_batch,
    const BitLineMaker<T> *blm,
    const PulsedUpdateMetaParameter<T> &up,
    curandState_t *dev_states,
    int one_sided,
    uint32_t *x_counts_chunk,
    uint32_t *d_counts_chunk) {
  // calling kpars->run(..,this,..) directly should cause error
  // because difference derived from abstract device..
  DEBUG_OUT("start run update kernel.");
  DEBUG_CALL(kpars->print(););

  if (one_sided != 0) {
    RPU_FATAL("Cannot use one_sided here.");
  }
  bool same_context = getPar().same_context;

  if (!same_context) {
    up_context->recordEvent();
  }

  CudaContext *cp = up_context;
  CudaContext *cm = up_context;

  if (!same_context) {
    cp = &*this->context_vec_[g_plus_];
    cm = &*this->context_vec_[g_minus_];
  }

  int n = kpars->getNStates() / 2; // each device uses different random states

  if (!same_context) {
    cp->waitEvent(up_context->getEvent());
    cm->waitEvent(up_context->getEvent());
  }

  kpars->run(
      cp->getStream(), this->dev_weights_ptrs_[g_plus_], m_batch, blm,
      static_cast<PulsedRPUDeviceCuda<T> *>(&*this->rpucuda_device_vec_[g_plus_]), // checked above
      up, dev_states,
      1, // one sided!!
      x_counts_chunk, d_counts_chunk);

  kpars->run(
      cm->getStream(), this->dev_weights_ptrs_[g_minus_], m_batch, blm,
      static_cast<PulsedRPUDeviceCuda<T> *>(&*this->rpucuda_device_vec_[g_minus_]), // checked above
      up, dev_states,
      -1, // one sided!!
      x_counts_chunk, d_counts_chunk);

  if (!same_context) {
    up_context->recordWaitEvent(cp);
    up_context->recordWaitEvent(cm);
  }

  this->reduceToWeights(up_context, dev_weights);
}

template class DifferenceRPUDeviceCuda<float>;
#ifdef RPU_USE_DOUBLE
template class DifferenceRPUDeviceCuda<double>;
#endif
} // namespace RPU
