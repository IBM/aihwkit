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

#include "rpu_pulsed.h"
#include "math_util.h"
#include "utility_functions.h"

#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>

#ifdef RPU_USE_MKL
#include "mkl.h"
#else
#ifdef RPU_USE_OPENBLAS
extern "C" {
#include "cblas.h"
}
#endif
#endif

#define CHECK_RPU_DEVICE_INIT                                                                      \
  if (rpu_device_ == nullptr) {                                                                    \
    RPU_FATAL("First populate rpu device (call populateParameter())!");                            \
  }

namespace RPU {

/********************************************************************************
 * PulsedMetaParameter<T>
 *********************************************************************************/

template <typename T> void PulsedMetaParameter<T>::initialize(int x_size, int d_size) {
  if (!_par_initialized) {
    _par_initialized = true;
    this->up.initialize();
    this->f_io.initializeForForward(x_size, d_size);
    this->b_io.initializeForBackward(x_size, d_size);
  }
}

template <typename T>
RPUPulsed<T> *PulsedMetaParameter<T>::createRPUArray(
    int x_size, int d_size, AbstractRPUDeviceMetaParameter<T> *dp) {
  auto *rpu = new RPUPulsed<T>(x_size, d_size);
  rpu->populateParameter(this, dp);
  rpu->setWeightsUniformRandom(-0.1, 0.1);
  rpu->setLearningRate(0.1);
  return rpu;
};

template <typename T> void PulsedMetaParameter<T>::print() const {
  std::stringstream ss;
  printToStream(ss);
  std::cout << ss.str();
};

template <typename T>
void PulsedMetaParameter<T>::printToStream(std::stringstream &ss, bool suppress_update) const {
  ss << "Forward:" << std::endl;
  f_io.printToStream(ss);
  ss << "Backward:" << std::endl;
  b_io.printToStream(ss);
  if (!suppress_update) {
    ss << "Update:" << std::endl;
    up.printToStream(ss);
  }
}

template struct PulsedMetaParameter<float>;
#ifdef RPU_USE_DOUBLE
template struct PulsedMetaParameter<double>;
#endif
#ifdef RPU_USE_FP16
template struct PulsedMetaParameter<half_t>;
#endif

/********************************************************************************
 * RPUPulsed<T>
 *********************************************************************************/

// ctor
template <typename T> RPUPulsed<T>::RPUPulsed(int x_sz, int d_sz) : RPUSimple<T>(x_sz, d_sz) {
  DEBUG_OUT("RPUPulsed constructed");
}

// copy construcutor
template <typename T>
RPUPulsed<T>::RPUPulsed(const RPUPulsed<T> &other) : RPUSimple<T>(other), par_(other.par_) {

  pwu_ = RPU::make_unique<PulsedRPUWeightUpdater<T>>(*other.pwu_);
  fb_pass_ = RPU::make_unique<ForwardBackwardPassIOManaged<T>>(*other.fb_pass_);

  if (other.rpu_device_ != nullptr) {
    rpu_device_ = other.rpu_device_->cloneUnique();
  }

  DEBUG_CALL(this->disp());
  DEBUG_OUT("RPUPulsed copy constructed ");
}

// copy assignment
template <typename T> RPUPulsed<T> &RPUPulsed<T>::operator=(const RPUPulsed<T> &other) {

  RPUPulsed<T> tmp(other);
  swap(*this, tmp);
  return *this;
}

// move constructor
template <typename T> RPUPulsed<T>::RPUPulsed(RPUPulsed<T> &&other) noexcept {
  *this = std::move(other);
}

// move assignment
template <typename T> RPUPulsed<T> &RPUPulsed<T>::operator=(RPUPulsed<T> &&other) noexcept {

  RPUSimple<T>::operator=(std::move(other));

  pwu_ = std::move(other.pwu_);
  fb_pass_ = std::move(other.fb_pass_);

  par_ = other.par_; // should envoke default copy constrcutor
  rpu_device_ = std::move(other.rpu_device_);
  return *this;
}

// dtor
template <typename T> RPUPulsed<T>::~RPUPulsed() {
  DEBUG_CALL(this->disp());
  DEBUG_OUT("RPUPulsed DESTRUCTED");
}

/*********************************************************************************/
template <typename T> std::unique_ptr<AbstractRPUDevice<T>> RPUPulsed<T>::cloneDevice() {

  if (rpu_device_) {
    return rpu_device_->cloneUnique();
  } else {
    RPU_FATAL("Device is not initialized!");
  }
}

/*********************************************************************************/
template <typename T> void RPUPulsed<T>::printParametersToStream(std::stringstream &ss) const {
  getMetaPar().printToStream(ss, !rpu_device_->usesUpdateParameter());
  ss << "Device:" << std::endl;
  rpu_device_->printToStream(ss);
}

template <typename T> void RPUPulsed<T>::printRPUParameter(int x_count, int d_count) const {

  rpu_device_->printDP(x_count, d_count);
}

template <typename T> void RPUPulsed<T>::printToStream(std::stringstream &ss) const {

  std::string name;
  name = rpu_device_->getPar().getName();
  ss << "RPUPulsed<" << this->getDataTypeName() << ">[" << name << "](" << this->d_size_ << ","
     << this->x_size_ << ")" << std::endl;
};

template <typename T> void RPUPulsed<T>::setLearningRate(T lr) {

  if (lr != this->getLearningRate()) {

    RPUSimple<T>::setLearningRate(lr);

    if (rpu_device_ != nullptr && rpu_device_->isPulsedDevice()) {
      // some output
      int BL = 0;
      T A = 0;
      T B = 0;
      getMetaPar().up.calculateBlAB(
          BL, A, B, lr,
          static_cast<PulsedRPUDeviceBase<T> *>(&*rpu_device_)->getWeightGranularity());
      DEBUG_OUT("\t BL = " << BL << ", A = " << A << ", B = " << B);
    }
  }
}

template <typename T> void RPUPulsed<T>::decayWeights(bool bias_no_decay) {
  CHECK_RPU_DEVICE_INIT;
  rpu_device_->decayWeights(this->getWeightsPtr(), bias_no_decay);
}

template <typename T> void RPUPulsed<T>::decayWeights(T alpha, bool bias_no_decay) {
  CHECK_RPU_DEVICE_INIT;
  rpu_device_->decayWeights(this->getWeightsPtr(), alpha, bias_no_decay);
}

template <typename T> void RPUPulsed<T>::diffuseWeights() {
  CHECK_RPU_DEVICE_INIT;
  rpu_device_->diffuseWeights(this->getWeightsPtr(), *this->rng_);
}

template <typename T> void RPUPulsed<T>::diffuseWeightsPink() {
  CHECK_RPU_DEVICE_INIT;
  ENFORCE_NO_DELAYED_UPDATE;

  RPUSimple<T>::diffuseWeightsPink();

  if (rpu_device_->implements() == DeviceUpdateType::FloatingPoint) {
    return;
  } else if (rpu_device_->implements() == DeviceUpdateType::ConstantStep) {
    rpu_device_->onSetWeights(this->getWeightsPtr());
  } else {
    RPU_FATAL("Pink noise NOT implemented for most devices");
  }
}

template <typename T> void RPUPulsed<T>::resetCols(int start_col, int n_cols, T reset_prob) {
  if (reset_prob) {
    CHECK_RPU_DEVICE_INIT;
    rpu_device_->resetCols(this->getWeightsPtr(), start_col, n_cols, reset_prob, *this->rw_rng_);
  }
}

template <typename T> void RPUPulsed<T>::driftWeights(T time_since_last_call) {
  CHECK_RPU_DEVICE_INIT;
  rpu_device_->driftWeights(this->getWeightsPtr(), time_since_last_call, *this->rng_);
}

template <typename T>
void RPUPulsed<T>::remapWeights(const WeightRemapParameter &wrmpar, T *scales, T *biases) {

  if (!pwu_->checkForFPUpdate(&*rpu_device_)) {
    RPU_FATAL("Remapping is NOT implemented for most devices");
  }

  CHECK_RPU_DEVICE_INIT;
  ENFORCE_NO_DELAYED_UPDATE;

  RPUSimple<T>::remapWeights(wrmpar, scales, biases);
}

template <typename T>
bool RPUPulsed<T>::swaWeights(
    const WeightRemapParameter &wrmpar, T *swa_weights, uint64_t iter, T *scales, T *biases) {

  CHECK_RPU_DEVICE_INIT;
  ENFORCE_NO_DELAYED_UPDATE;

  if (wrmpar.type != WeightRemapType::None && !pwu_->checkForFPUpdate(&*rpu_device_)) {
    RPU_FATAL("SWA is NOT implemented for most devices");
  }

  bool modfied = RPUSimple<T>::swaWeights(wrmpar, swa_weights, iter, scales, biases);

  if (modfied) {
    rpu_device_->onSetWeights(this->getWeightsPtr());
  }
  return modfied;
}

template <typename T> void RPUPulsed<T>::clipWeights(T clip) {

  CHECK_RPU_DEVICE_INIT;
  rpu_device_->clipWeights(this->getWeightsPtr(), clip);
}

template <typename T> void RPUPulsed<T>::clipWeights(const WeightClipParameter &wclpar) {

  CHECK_RPU_DEVICE_INIT;

  if (wclpar.type == WeightClipType::FixedValue) {
    clipWeights((T)wclpar.fixed_value); // handle outside  to support devices
  } else if (rpu_device_->implements() == DeviceUpdateType::FloatingPoint) {
    RPUSimple<T>::clipWeights(wclpar);
  } else {
    RPU_FATAL("Sophisticated clipping is NOT implemented for most training devices");
  }
}

template <typename T> void RPUPulsed<T>::setWeightsUniformRandom(T min_value, T max_value) {
  CHECK_RPU_DEVICE_INIT;
  RPUSimple<T>::setWeightsUniformRandom(min_value, max_value);
  rpu_device_->onSetWeights(this->getWeightsPtr());
}

template <typename T> void RPUPulsed<T>::setWeights(const T *weightsptr) {
  CHECK_RPU_DEVICE_INIT;
  RPUSimple<T>::setWeights(weightsptr);
  rpu_device_->onSetWeights(this->getWeightsPtr());
}

template <typename T> void RPUPulsed<T>::applyWeightUpdate(T *dw_and_current_weight_out) {
  T *w = this->getWeightsPtr()[0];
  int size = this->d_size_ * this->x_size_;
  PRAGMA_SIMD
  for (int i = 0; i < size; ++i) {
    w[i] += dw_and_current_weight_out[i];
  }
  if (rpu_device_) {
    rpu_device_->onSetWeights(this->getWeightsPtr());
  }
  memcpy(dw_and_current_weight_out, w, sizeof(T) * size); // need to make sure that this is sat
}

template <typename T> void RPUPulsed<T>::getWeightsReal(T *weightsptr) {

  CHECK_RPU_DEVICE_INIT;

  int x_sz = this->getXSize();
  T **eye = Array_2D_Get_Eye<T>(x_sz);

  T alpha = this->getFwdAlpha();
  this->setFwdAlpha(1.0, false);
  this->forwardMatrix(eye[0], weightsptr, x_sz, false, true, false);
  this->setFwdAlpha(alpha, false);

  Array_2D_Free<T>(eye);
}

template <typename T> void RPUPulsed<T>::setWeightsReal(const T *weightsptr, int n_loops) {

  CHECK_RPU_DEVICE_INIT;

  int x_sz = this->getXSize();
  int d_sz = this->getDSize();

  T *w_current = this->getWeightsPtr()[0];

  /*==== this is a slight hack to get the number of iteration appproximately right*/
  // does not matter exactly anyway
  T weight_granularity = (T)0.001;
  T w_min = (T)-1.0;
  T w_max = (T)1.0;
  auto *dpar = dynamic_cast<const PulsedRPUDeviceMetaParameter<T> *>(&rpu_device_->getPar());
  if (dpar != nullptr) {
    w_min = dpar->w_min;
    w_max = dpar->w_max;
    weight_granularity =
        dynamic_cast<PulsedRPUDeviceBase<T> *>(&*rpu_device_)
            ->getWeightGranularity(); // this should be safe since we checked the params
  }
  int BL = 0;
  T A = (T)0.0;
  T B = (T)0.0;
  getMetaPar().up.calculateBlAB(BL, A, B, this->getLearningRate(), weight_granularity);

  T mx_change = (T)BL * weight_granularity;
  T range = w_max - w_min;
  int iter = (int)roundf((T)n_loops * range / mx_change);

  /*====*/

  DEBUG_OUT("RPUPulsed: Set weights real [iter=" << iter << "]");

  T **delta = Array_2D_Get<T>(d_sz, x_sz);
  T **eye = Array_2D_Get_Eye<T>(x_sz);

  T fwd_alpha = this->getFwdAlpha();
  T bwd_alpha = this->getBwdAlpha();

  this->setFwdAlpha(1.0, false);
  this->setBwdAlpha(1.0, false);

  for (int k = 0; k < iter; ++k) {

    this->forwardMatrix(eye[0], delta[0], x_sz, false, true, false);

    // calc delta
    for (int i = 0; i < x_sz * d_sz; ++i) {
      (delta[0])[i] -= weightsptr[i];
    }
    this->updateMatrix(eye[0], delta[0], x_sz, false, true);
  }
  this->setFwdAlpha(fwd_alpha, false);
  this->setBwdAlpha(bwd_alpha, false);

  T avg_dev = 0.0;
  for (int i = 0; i < x_sz * d_sz; ++i) {
    avg_dev += (T)fabsf(weightsptr[i] - w_current[i]);
  }
  avg_dev /= x_sz * d_sz;
  DEBUG_OUT("Finished setting weights real [avg deviation=" << avg_dev << "]");

  this->copyWeightsToBuffer();

  Array_2D_Free<T>(eye);
  Array_2D_Free<T>(delta);
}

template <typename T>
void RPUPulsed<T>::getDeviceParameterNames(std::vector<std::string> &names) const {
  CHECK_RPU_DEVICE_INIT;
  rpu_device_->getDPNames(names);
}

template <typename T> void RPUPulsed<T>::getDeviceParameter(std::vector<T *> &data_ptrs) {
  // note that memory (x_sz*d_sz per ptr) assumed to be initialized from outside !!
  CHECK_RPU_DEVICE_INIT;
  rpu_device_->getDeviceParameter(this->getWeightsPtr(), data_ptrs);
};

template <typename T> void RPUPulsed<T>::setDeviceParameter(const std::vector<T *> &data_ptrs) {
  // note that memory (x_sz*d_sz per ptr) assumed to be initialized from outside !!
  CHECK_RPU_DEVICE_INIT;
  rpu_device_->setDeviceParameter(this->getWeightsPtr(), data_ptrs);
};

template <typename T> void RPUPulsed<T>::setHiddenUpdateIdx(int idx) {
  CHECK_RPU_DEVICE_INIT;
  rpu_device_->setHiddenUpdateIdx(idx);
};

template <typename T> int RPUPulsed<T>::getHiddenUpdateIdx() const {
  CHECK_RPU_DEVICE_INIT;
  return rpu_device_->getHiddenUpdateIdx();
};

template <typename T> const FBParameter<T> &RPUPulsed<T>::getFBParameter() const {
  CHECK_RPU_DEVICE_INIT;
  return fb_pass_->getFBParameter();
};

template <typename T> void RPUPulsed<T>::setFBParameter(FBParameter<T> &fb_pars) {
  CHECK_RPU_DEVICE_INIT;
  fb_pass_->setFBParameter(fb_pars);
};

/*********************************************************************************/
/* dump / load state */

template <typename T> void RPUPulsed<T>::dumpExtra(RPU::state_t &extra, const std::string prefix) {

  RPUSimple<T>::dumpExtra(extra, prefix);

  RPU::state_t state;

  pwu_->dumpExtra(state, "pwu");
  fb_pass_->dumpExtra(state, "fb_pass");
  rpu_device_->dumpExtra(state, "rpu_device");

  RPU::insertWithPrefix(extra, state, prefix);
}

template <typename T>
void RPUPulsed<T>::loadExtra(const RPU::state_t &extra, const std::string prefix, bool strict) {

  RPUSimple<T>::loadExtra(extra, prefix, strict);

  auto state = RPU::selectWithPrefix(extra, prefix);

  pwu_->loadExtra(state, "pwu", strict);
  fb_pass_->loadExtra(state, "fb_pass", strict);
  rpu_device_->loadExtra(state, "rpu_device", strict);
}

/*********************************************************************************/

template <typename T>
void RPUPulsed<T>::populateParameter(
    PulsedMetaParameter<T> *p, AbstractRPUDeviceMetaParameter<T> *dp) {
  // set parent meta pars (from device pars)
  RPUSimple<T>::populateParameter(dp);
  p->initialize(this->x_size_, this->d_size_);

  // forward
  fb_pass_ =
      RPU::make_unique<ForwardBackwardPassIOManaged<T>>(this->x_size_, this->d_size_, this->rng_);
  fb_pass_->populateFBParameter(p->f_io, p->b_io);

  // pwu
  pwu_ = RPU::make_unique<PulsedRPUWeightUpdater<T>>(this->x_size_, this->d_size_, this->rng_);
  pwu_->setUpPar(p->up);

  // note: dp could also be SimpleRPUDeviceMetaParameter
  RealWorldRNG<T> rng(dp->construction_seed);
  if (p->up.pulse_type == PulseType::None) {

    if (dynamic_cast<SimpleRPUDeviceMetaParameter<T> *>(dp) == nullptr) {
      RPU_FATAL("For PulseType::None device needs to be castable to Simple.");
    }

    SimpleRPUDeviceMetaParameter<T> dp_simple(*static_cast<SimpleRPUDeviceMetaParameter<T> *>(dp));
    rpu_device_ = dp_simple.createDeviceUnique(this->x_size_, this->d_size_, &rng);
  } else {
    // create and populate correct device
    rpu_device_ = dp->createDeviceUnique(this->x_size_, this->d_size_, &rng);
  }

  // update weights to obey bounds etc. (they might be set already)
  rpu_device_->onSetWeights(this->getWeightsPtr());

  par_ = *p; // only for local copy, cannot modify! Use getMetaPar() to access it
}

/*********************************************************************************/
/* Vector forward/backward/update */

template <typename T>
void RPUPulsed<T>::forwardVector(
    const T *x_input, T *d_output, int x_inc, int d_inc, bool is_test) {
  fb_pass_->forwardVector(
      this->getFBWeights(is_test), x_input, x_inc, d_output, d_inc, this->getFwdAlpha(), is_test);
};

template <typename T>
void RPUPulsed<T>::backwardVector(const T *d_input, T *x_output, int d_inc, int x_inc) {
  fb_pass_->backwardVector(
      this->getFBWeights(false), d_input, d_inc, x_output, x_inc, this->getBwdAlpha());
};

template <typename T>
void RPUPulsed<T>::updateVector(const T *x_input, const T *d_input, int x_inc, int d_inc) {

  if (this->getDeltaWeights()) {
    if ((x_inc != 1) || (d_inc != 1)) {
      RPU_FATAL("Update_Vector for delta weights and xd_inc>1 is not implemented.");
    }
    this->updateMatrix(x_input, d_input, 1, false, false);
  } else {

    pwu_->updateVectorWithDevice(
        this->getUpWeights(), x_input, x_inc, d_input, d_inc, this->getAlphaLearningRate(),
        this->last_update_m_batch_, // for info
        &*rpu_device_);
  }
}

// for debugging
template <typename T>
void RPUPulsed<T>::updateVectorWithCounts(
    const T *x_input,
    const T *d_input,
    int x_inc,
    int d_inc,
    uint32_t *x_counts32,
    uint32_t *d_counts32) {
  auto *rpu_device = dynamic_cast<PulsedRPUDeviceBase<T> *>(&*rpu_device_);

  if (rpu_device == nullptr) {
    RPU_FATAL("Debug function updateVectorWithCounts does not support abstract devices");
  }

  if (this->getDeltaWeights()) {
    // delta w requested. We simply do not support this for CPU
    RPU_FATAL("Delta weights are not supported for RPUPulsed with counts on CPU");
  }

  pwu_->updateVectorWithDeviceAndCounts(
      this->getUpWeights(), x_input, x_inc, d_input, d_inc, this->getAlphaLearningRate(),
      this->last_update_m_batch_, rpu_device, x_counts32, d_counts32);
}

/*********************************************************************************/
/* specialized matrix update to be able to run matrix simple */

template <typename T>
void RPUPulsed<T>::updateMatrix(
    const T *X_input, const T *D_input, int m_batch, bool x_trans, bool d_trans) {

  if (pwu_->checkForFPUpdate(&*rpu_device_)) {
    // we use the fast simple GEMM is this case. This also has the
    // correct behavior for delta weights
    RPUSimple<T>::updateMatrix(X_input, D_input, m_batch, x_trans, d_trans);
  } else {

    T *local_dw = this->getDeltaWeights();
    if (local_dw) {
      this->setDeltaWeights(nullptr); // to use the standard updateVector
      // first make copy of weights
      int size = this->d_size_ * this->x_size_;
      memcpy(local_dw, this->getUpWeights()[0], sizeof(T) * size);
    }
    // use looped version to apply update onto normal weight
    RPUAbstract<T>::updateMatrix(X_input, D_input, m_batch, x_trans, d_trans);

    if (local_dw) {
      // get and reset update
      T used_lr = this->getAlphaLearningRate();
      this->setDeltaWeights(local_dw); // changes LR potentially
      this->getAndResetWeightUpdate(local_dw, this->getAlphaLearningRate() / used_lr);
    }
  }
}

template class RPUPulsed<float>;
#ifdef RPU_USE_DOUBLE
template class RPUPulsed<double>;
#endif
#ifdef RPU_USE_FP16
template class RPUPulsed<half_t>;
#endif

#undef CHECK_RPU_DEVICE_INIT
} // namespace RPU
