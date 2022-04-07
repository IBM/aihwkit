/**
 * (C) Copyright 2020, 2021, 2022 IBM. All Rights Reserved.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
 */

#include "forward_backward_pass.h"
#include "rpucuda_pulsed.h"
#include <iostream>
//#include <random>
#include <chrono>
#include <cmath>
#include <memory>

#define CHECK_RPU_DEVICE_INIT                                                                      \
  if (rpucuda_device_ == nullptr || rpu_device_ == nullptr) {                                      \
    RPU_FATAL("First populate rpu device (call Populate_Parameter())!");                           \
  }

namespace RPU {

/********************************************************************************
 * RPUCudaPulsed<T>
 *********************************************************************************/

template <typename T> void RPUCudaPulsed<T>::initialize() {

  int d_size = this->getDSize();
  int x_size = this->getXSize();

  CudaContext *c = this->context_;
  size_ = d_size * x_size;

  // shared batch buffer for backward/forward (here m_batch=1, will resize)
  dev_batch_buffer_x_size_ = std::make_shared<CudaArray<T>>(c, x_size);
  dev_batch_buffer_d_size_ = std::make_shared<CudaArray<T>>(c, d_size);

  // forward arrays
  f_iom_ = RPU::make_unique<InputOutputManager<T>>(c, x_size, d_size);
  dev_f_x_vector_inc1_ = RPU::make_unique<CudaArray<T>>(c, x_size);
  dev_f_d_vector_inc1_ = RPU::make_unique<CudaArray<T>>(c, d_size);

  // backward arrays
  b_iom_ = RPU::make_unique<InputOutputManager<T>>(c, d_size, x_size);
  dev_b_x_vector_inc1_ = RPU::make_unique<CudaArray<T>>(c, x_size);
  dev_b_d_vector_inc1_ = RPU::make_unique<CudaArray<T>>(c, d_size);

  // update arrays
  up_pwu_ = RPU::make_unique<PulsedWeightUpdater<T>>(c, x_size, d_size);

  dev_up_x_vector_inc1_ = RPU::make_unique<CudaArray<T>>(c, x_size);
  dev_up_d_vector_inc1_ = RPU::make_unique<CudaArray<T>>(c, d_size);

  this->context_->synchronize();

  rpu_device_ = nullptr;
  rpucuda_device_ = nullptr;

  DEBUG_OUT("RPUCudaPulsed constructed");
}

template <typename T>
RPUCudaPulsed<T>::RPUCudaPulsed(CudaContext *c, int x_size, int d_size)
    : RPUCudaSimple<T>(c, x_size, d_size) {}

template <typename T>
RPUCudaPulsed<T>::RPUCudaPulsed(cudaStream_t s, int x_size, int d_size)
    : RPUCudaSimple<T>(s, x_size, d_size) {}

template <typename T> void RPUCudaPulsed<T>::initFrom(RPUPulsed<T> &rpu) {
  // this is private and only for the construction from CPU

  initialize();

  rpu_device_ = rpu.cloneDevice();
  par_ = rpu.getMetaPar();
  if (rpu_device_) {
    rpucuda_device_ = AbstractRPUDeviceCuda<T>::createFromUnique(this->context_, *rpu_device_);
  } else {
    RPU_FATAL("Expect rpu_device to be populated!");
  }
  this->context_->synchronize();

  // NOTE: weight is already copied in RPUSimple constructor
}

template <typename T>
RPUCudaPulsed<T>::RPUCudaPulsed(CudaContext *c, RPUPulsed<T> &o)
    : RPUCudaSimple<T>(c, static_cast<RPUSimple<T> &>(o)) {
  initFrom(o);
}

template <typename T>
RPUCudaPulsed<T>::RPUCudaPulsed(cudaStream_t s, RPUPulsed<T> &o)
    : RPUCudaSimple<T>(s, static_cast<RPUSimple<T> &>(o)) {
  initFrom(o);
}

template <typename T> RPUCudaPulsed<T>::~RPUCudaPulsed() { DEBUG_OUT("RPUCudaPulsed destroyed."); }

// copy constructor
template <typename T>
RPUCudaPulsed<T>::RPUCudaPulsed(const RPUCudaPulsed<T> &other) : RPUCudaSimple<T>(other) {
  // NOTE: we do not copy all the class members helper such as blm,
  // pwu etc. They get reconstructed. We only copy construct the
  // RPU_PULSED_, since all relevant parameters are set there, and
  // copy it to GPU
  if (up_pwu_ != nullptr) { // check whether it is initialized
    initialize();           // private
  }
  par_ = other.par_;

  if (other.rpu_device_) {
    rpu_device_ = other.rpu_device_->cloneUnique();
    rpucuda_device_ = AbstractRPUDeviceCuda<T>::createFromUnique(this->context_, *rpu_device_);
  } else {
    rpucuda_device_ = nullptr;
    rpu_device_ = nullptr;
  }
  par_ = other.par_;

  DEBUG_CALL(this->disp(););
  DEBUG_OUT("RPUCudaPulsed copy constructed.");
}

// copy assignment
template <typename T> RPUCudaPulsed<T> &RPUCudaPulsed<T>::operator=(const RPUCudaPulsed<T> &other) {

  RPUCudaPulsed<T> tmp(other);
  swap(*this, tmp);
  this->context_->synchronize();
  return *this;
}

// move constructor
template <typename T> RPUCudaPulsed<T>::RPUCudaPulsed(RPUCudaPulsed<T> &&other) {
  *this = std::move(other);
}

// move assignment
template <typename T> RPUCudaPulsed<T> &RPUCudaPulsed<T>::operator=(RPUCudaPulsed<T> &&other) {

  RPUCudaSimple<T>::operator=(std::move(other));

  rpu_device_ = std::move(other.rpu_device_);
  rpucuda_device_ = std::move(other.rpucuda_device_);

  par_ = other.par_;
  f_iom_ = std::move(other.f_iom_);
  b_iom_ = std::move(other.b_iom_);
  up_pwu_ = std::move(other.up_pwu_);

  dev_up_x_vector_inc1_ = std::move(other.dev_up_x_vector_inc1_);
  dev_up_d_vector_inc1_ = std::move(other.dev_up_d_vector_inc1_);
  dev_f_x_vector_inc1_ = std::move(other.dev_f_x_vector_inc1_);
  dev_f_d_vector_inc1_ = std::move(other.dev_f_d_vector_inc1_);

  dev_b_x_vector_inc1_ = std::move(other.dev_b_x_vector_inc1_);
  dev_b_d_vector_inc1_ = std::move(other.dev_b_d_vector_inc1_);

  dev_batch_buffer_d_size_ = std::move(other.dev_batch_buffer_d_size_);
  dev_batch_buffer_x_size_ = std::move(other.dev_batch_buffer_x_size_);

  size_ = other.size_;

  this->context_->synchronize();

  return *this;
}

template <typename T>
void RPUCudaPulsed<T>::populateParameter(
    PulsedMetaParameter<T> *p, PulsedRPUDeviceMetaParameter<T> *dp) {
  RPUCudaSimple<T>::populateParameter(dp);
  p->initialize();

  if (up_pwu_ == nullptr) {
    initialize();
  }

  if (p->up.pulse_type == PulseType::None) {

    if (dynamic_cast<SimpleRPUDeviceMetaParameter<T> *>(dp) == nullptr) {
      RPU_FATAL("For PulseType::None device needs to be castable to Simple.");
    }

    SimpleRPUDeviceMetaParameter<T> dp_simple(*static_cast<SimpleRPUDeviceMetaParameter<T> *>(dp));
    rpu_device_ = dp_simple.createDeviceUnique(this->x_size_, this->d_size_, nullptr);
  } else {
    // create and populate correct device
    RealWorldRNG<T> rng(dp->construction_seed);
    rpu_device_ = dp->createDeviceUnique(this->x_size_, this->d_size_, &rng);
  }
  rpucuda_device_ = AbstractRPUDeviceCuda<T>::createFromUnique(this->context_, *rpu_device_);

  this->setWeights(this->copyWeightsToHost()[0]); // set weights needs all populated

  par_ = *p; // only copy, read access with getMetaPar()
}

/*********************************************************************************/

template <typename T> void RPUCudaPulsed<T>::setLearningRate(T lr) {

  if (lr != this->getLearningRate()) {

    RPUCudaSimple<T>::setLearningRate(lr);

    if (rpucuda_device_ != nullptr && rpucuda_device_->isPulsedDevice()) {
      // some output
      int BL = 0;
      T A = 0;
      T B = 0;
      getMetaPar().up.calculateBlAB(
          BL, A, B, lr,
          static_cast<PulsedRPUDeviceCudaBase<T> &>(*rpucuda_device_).getWeightGranularity());
      DEBUG_OUT("\t BL = " << BL << ", A = " << A << ", B = " << B);
    }
  }
}

/*********************************************************************************/

template <typename T> void RPUCudaPulsed<T>::printToStream(std::stringstream &ss) const {

  CHECK_RPU_DEVICE_INIT;

  std::string name;
  name = rpucuda_device_->getPar().getName();

  std::string num = "float";
  if (sizeof(T) == 8) {
    num = "double";
  }
  ss << "RPUCudaPulsed<" << num << ">[" << name << "](" << this->d_size_ << "," << this->x_size_
     << ")" << std::endl;
};

/*********************************************************************************/
/** These functions use the rpu device and are basically a
    reimplimentation of RPU_PULSED for CUDA. Note that we cannot
    directly inherit from RPU_PULSED (triangle)!  **/

template <typename T> void RPUCudaPulsed<T>::decayWeights(T alpha, bool bias_no_decay) {
  CHECK_RPU_DEVICE_INIT;
  rpucuda_device_->decayWeights(this->dev_weights_->getData(), alpha, bias_no_decay);
}

template <typename T> void RPUCudaPulsed<T>::decayWeights(bool bias_no_decay) {
  CHECK_RPU_DEVICE_INIT;
  rpucuda_device_->decayWeights(this->dev_weights_->getData(), bias_no_decay);
}

template <typename T> void RPUCudaPulsed<T>::driftWeights(T time_since_last_call) {
  CHECK_RPU_DEVICE_INIT;
  rpucuda_device_->driftWeights(this->dev_weights_->getData(), time_since_last_call);
}

template <typename T> void RPUCudaPulsed<T>::diffuseWeights() {
  CHECK_RPU_DEVICE_INIT;
  rpucuda_device_->diffuseWeights(this->dev_weights_->getData());
}

template <typename T> void RPUCudaPulsed<T>::clipWeights(T clip) {
  CHECK_RPU_DEVICE_INIT;
  rpucuda_device_->clipWeights(this->dev_weights_->getData(), clip);
}

template <typename T> void RPUCudaPulsed<T>::clipWeights(const WeightClipParameter &wclpar) {

  if (wclpar.type == WeightClipType::FixedValue) {
    clipWeights(wclpar.fixed_value); // handle outside  to support devices
  } else if (rpu_device_->implements() == DeviceUpdateType::FloatingPoint) {
    RPUCudaSimple<T>::clipWeights(wclpar);
  } else {
    RPU_FATAL("Sophisticated clipping is NOT implemented for most training devices");
  }
}

template <typename T> void RPUCudaPulsed<T>::resetCols(int start_col, int n_cols, T reset_prob) {

  if (reset_prob) {
    CHECK_RPU_DEVICE_INIT;
    rpucuda_device_->resetCols(this->dev_weights_->getData(), start_col, n_cols, reset_prob);
  }
}

template <typename T> void RPUCudaPulsed<T>::printRPUParameter(int x_count, int d_count) const {
  // prints parameters from rpu_stoc without syncing ! However,
  // device should mirror rpu_pulsed_ any time anyway

  CHECK_RPU_DEVICE_INIT;
  rpu_device_->printDP(x_count, d_count);
}

template <typename T> void RPUCudaPulsed<T>::getWeightsReal(T *weightsptr) {

  CHECK_RPU_DEVICE_INIT;

  int x_sz = this->getXSize();
  int d_sz = this->getDSize();

  T **eye = Array_2D_Get_Eye<T>(x_sz);
  auto eye_d = CudaArray<T>(this->context_, x_sz * x_sz, eye[0]);
  auto w_buffer = CudaArray<T>(this->context_, d_sz * x_sz);
  this->context_->synchronize();

  bool is_test = true; // should not change anything

  T alpha = this->getFwdAlpha();
  this->setFwdAlpha(1.0, false);
  this->forwardMatrix(eye_d.getDataConst(), w_buffer.getData(), x_sz, false, true, is_test);
  this->setFwdAlpha(alpha, false);

  w_buffer.copyTo(weightsptr);
  this->context_->synchronize();

  Array_2D_Free<T>(eye);
}

template <typename T> void RPUCudaPulsed<T>::setWeightsReal(const T *weightsptr, int n_loops) {

  CHECK_RPU_DEVICE_INIT;

  int x_sz = this->getXSize();
  int d_sz = this->getDSize();

  /*==== slight hack to get the range right */
  T weight_granularity = 0.001;
  const auto *dpar =
      dynamic_cast<const PulsedRPUDeviceMetaParameter<T> *>(&rpucuda_device_->getPar());
  T w_min = -1;
  T w_max = 1;

  if (dpar != nullptr) {
    w_min = dpar->w_min;
    w_max = dpar->w_max;
    weight_granularity =
        static_cast<PulsedRPUDeviceCudaBase<T> &>(*rpucuda_device_).getWeightGranularity();
  }
  T A = 0;
  T B = 0;
  int BL = 0;
  getMetaPar().up.calculateBlAB(BL, A, B, this->getLearningRate(), weight_granularity);
  T mx_change = BL * weight_granularity;
  T range = fabs(w_max - w_min);
  int iter = n_loops * range / mx_change;

  /*==== */

  DEBUG_OUT("RPUCudaPulsed: Set weights real [iter=" << iter << "]");

  T **eye = Array_2D_Get_Eye<T>(x_sz);
  auto eye_d = CudaArray<T>(this->context_, x_sz * x_sz, eye[0]);
  auto w_ref_trans = CudaArray<T>(this->context_, d_sz * x_sz, weightsptr);
  auto delta = CudaArray<T>(this->context_, d_sz * x_sz, weightsptr);

  this->context_->synchronize();
  bool is_test = false; // not used
  T fwd_alpha = this->getFwdAlpha();
  T bwd_alpha = this->getBwdAlpha();
  this->setFwdAlpha(1.0, false);
  this->setBwdAlpha(1.0, false);
  for (int k = 0; k < iter; ++k) {

    this->forwardMatrix(eye_d.getDataConst(), delta.getData(), x_sz, false, true, is_test);

    RPU::math::elemaddscale<T>(
        this->context_, delta.getData(), x_sz * d_sz, w_ref_trans.getDataConst(), (T)-1.0);

    this->updateMatrix(eye_d.getDataConst(), delta.getDataConst(), x_sz, false, true);
  }
  this->setFwdAlpha(fwd_alpha, false);
  this->setBwdAlpha(bwd_alpha, false);

  this->context_->synchronize();

  T avg_dev = 0.0;
  T *w_current = this->copyWeightsToHost()[0];
  for (int i = 0; i < x_sz * d_sz; ++i) {
    avg_dev += fabs(weightsptr[i] - w_current[i]);
  }
  avg_dev /= x_sz * d_sz;
  DEBUG_OUT("Finished setting weights real [avg deviation=" << avg_dev << "]");

  Array_2D_Free<T>(eye);
}

template <typename T>
void RPUCudaPulsed<T>::getDeviceParameterNames(std::vector<std::string> &names) const {

  CHECK_RPU_DEVICE_INIT;
  rpu_device_->getDPNames(names);
}

template <typename T> void RPUCudaPulsed<T>::getDeviceParameter(std::vector<T *> &data_ptrs) const {

  CHECK_RPU_DEVICE_INIT;
  rpu_device_->setHiddenWeights(rpucuda_device_->getHiddenWeights());
  rpu_device_->getDeviceParameter(data_ptrs);
};

template <typename T> void RPUCudaPulsed<T>::setDeviceParameter(const std::vector<T *> &data_ptrs) {
  // note that memory (x_sz*d_sz per ptr) assumed to be initialized from outside !!

  CHECK_RPU_DEVICE_INIT;

  // Note: for now setting the device just keeps the old meta parameter.
  // however weight_granularity is at least estimated.
  this->copyWeightsToHost();
  rpu_device_->setDeviceParameter(this->getWeightsPtr(), data_ptrs);
  rpucuda_device_->populateFrom(*rpu_device_);

  // set device weights which might have been updated because of the hidden parameters
  RPUCudaSimple<T>::setWeights(this->getWeightsPtr()[0]);
};

template <typename T> int RPUCudaPulsed<T>::getHiddenUpdateIdx() const {
  CHECK_RPU_DEVICE_INIT;
  return rpucuda_device_->getHiddenUpdateIdx();
};

template <typename T> void RPUCudaPulsed<T>::setHiddenUpdateIdx(int idx) {
  CHECK_RPU_DEVICE_INIT;
  rpucuda_device_->setHiddenUpdateIdx(idx);
  rpu_device_->setHiddenUpdateIdx(idx);
};

template <typename T> void RPUCudaPulsed<T>::setWeights(const T *host_source) {

  CHECK_RPU_DEVICE_INIT;
  RPUSimple<T>::setWeights(host_source); // sets host

  if (rpu_device_) {
    if (rpu_device_->onSetWeights(this->getWeightsPtr())) {
      // apply bounds etc to host
      rpucuda_device_->populateFrom(*rpu_device_); // device pars have changed (due to onSetWeights)
    }
  }
  RPUCudaSimple<T>::setWeights(this->getWeightsPtr()[0]); // set device weights
}

template <typename T> void RPUCudaPulsed<T>::applyWeightUpdate(T *dw_and_current_weight_out) {

  CHECK_RPU_DEVICE_INIT;

  if (rpu_device_) {
    rpucuda_device_->applyWeightUpdate(this->dev_weights_->getData(), dw_and_current_weight_out);
  } else {
    RPUCudaSimple<T>::applyWeightUpdate(dw_and_current_weight_out);
  }
}

/*********************************************************************************/
/*********************************************************************************/
/* FORWARD */

template <typename T>
void RPUCudaPulsed<T>::forwardMatrix(
    const T *X_input, T *D_output, int m_batch, bool x_trans, bool d_trans, bool is_test) {
  this->forwardMatrixIterator(X_input, D_output, m_batch, x_trans, d_trans, is_test);
}

template <typename T>
void RPUCudaPulsed<T>::forwardIndexed(
    const T *X_input,
    T *D_output,
    int total_input_size,
    int m_batch,
    int dim3,
    bool trans,
    bool is_test) {

  const int *indices = this->getMatrixIndices();

  if (trans && (dim3 > 1)) {

    IndexReaderTransInputIterator<T> iter(
        X_input, indices, total_input_size / dim3, m_batch, this->getXSize() * m_batch,
        m_batch * dim3);

    PermuterTransOutputIterator<T> permute_iter(
        D_output, m_batch, this->getDSize() * m_batch, m_batch * dim3);

    this->forwardMatrixIterator(iter, permute_iter, m_batch * dim3, trans, trans, is_test);

  } else {

    IndexReaderInputIterator<T> iter(
        X_input, indices, total_input_size / dim3, this->getXSize() * m_batch);

    // in-place
    this->forwardMatrixIterator(iter, D_output, m_batch * dim3, trans, trans, is_test);
  }
}

template <typename T>
void RPUCudaPulsed<T>::forwardIndexedSlice(
    const T *X_input,
    T *D_output,
    int total_input_size,
    int m_batch,
    int dim3,
    bool trans,
    int m_batch_slice,
    const int *batch_indices,
    bool is_test) {

  const int *indices = this->getMatrixIndices();
  int x_size = this->getXSize();
  int d_size = this->getDSize();

  if (trans && (dim3 > 1)) {

    IndexReaderSliceInputIterator<true, T> in_iter(
        X_input, indices, total_input_size / dim3, x_size, m_batch, dim3, m_batch_slice,
        batch_indices);

    SliceOutputIterator<true, T> out_iter(
        D_output, d_size, m_batch, dim3, m_batch_slice, batch_indices);

    this->forwardMatrixIterator(in_iter, out_iter, m_batch_slice * dim3, trans, trans, is_test);

  } else {

    IndexReaderSliceInputIterator<false, T> in_iter(
        X_input, indices, total_input_size / dim3, x_size, m_batch, dim3, m_batch_slice,
        batch_indices);

    SliceOutputIterator<false, T> out_iter(
        D_output, d_size, m_batch, dim3, m_batch_slice, batch_indices);

    this->forwardMatrixIterator(in_iter, out_iter, m_batch_slice * dim3, trans, trans, is_test);
  }
}

template <typename T>
template <typename InputIteratorT, typename OutputIteratorT>
void RPUCudaPulsed<T>::forwardMatrixIterator(
    InputIteratorT X_input,
    OutputIteratorT D_output,
    int m_batch,
    bool x_trans,
    bool d_trans,
    bool is_test) {

  checkBatchBuffers(m_batch);

  RPU::detail::forwardMatrixIteratorIOManaged(
      this->context_, this->getFBWeightsCuda(is_test), X_input, this->x_size_, x_trans, D_output,
      this->d_size_, d_trans, m_batch, this->getFwdAlpha(), *f_iom_, getMetaPar().f_io, is_test,
      dev_batch_buffer_x_size_, dev_batch_buffer_d_size_);
};

template <typename T>
void RPUCudaPulsed<T>::forwardVector(
    const T *x_input, T *d_output, int x_inc, int d_inc, bool is_test) {
  T *d_output_inc1 = d_output;
  const T *x_input_inc1 = x_input;

  if (d_inc != 1) {
    d_output_inc1 = dev_f_d_vector_inc1_->getData();
  }
  if (x_inc != 1) {
    // just copy for now. Only needed for looped matrix versions anyway
    RPU::math::copy<T>(
        this->context_, this->x_size_, x_input, x_inc, dev_f_x_vector_inc1_->getData(), 1);
    x_input_inc1 = dev_f_x_vector_inc1_->getDataConst();
  }

  forwardMatrixIterator(x_input_inc1, d_output_inc1, 1, false, false, is_test);

  if (d_inc != 1) {
    RPU::math::copy<T>(this->context_, this->d_size_, d_output_inc1, 1, d_output, d_inc);
  }
}

/*********************************************************************************/
/*********************************************************************************/
/* BACKWARD */

template <typename T>
void RPUCudaPulsed<T>::backwardMatrix(
    const T *D_input, T *X_output, int m_batch, bool d_trans, bool x_trans) {
  backwardMatrixIterator(D_input, X_output, m_batch, d_trans, x_trans);
}

template <typename T>
void RPUCudaPulsed<T>::backwardIndexed(
    const T *D_input, T *X_output, int total_output_size, int m_batch, int dim3, bool trans) {

  int x_size = this->getXSize();
  const int *indices = this->getMatrixIndices();

  // need to set X_output to all zero for the atomics
  this->setZero(X_output, total_output_size);

  if ((dim3 == 1) || (!trans)) {

    IndexReaderOutputIterator<T> out_iter(
        X_output, indices, total_output_size / dim3, x_size * m_batch);

    backwardMatrixIterator(D_input, out_iter, m_batch * dim3, trans, trans);

  } else {

    IndexReaderTransOutputIterator<T> out_iter(
        X_output, indices, total_output_size / dim3, m_batch, x_size * m_batch, m_batch * dim3);

    PermuterTransInputIterator<T> permute_iter(
        D_input, m_batch, this->getDSize() * m_batch, m_batch * dim3);

    backwardMatrixIterator(permute_iter, out_iter, m_batch * dim3, trans, trans);
  }
}

template <typename T>
void RPUCudaPulsed<T>::backwardIndexedSlice(
    const T *D_input,
    T *X_output,
    int total_output_size,
    int m_batch,
    int dim3,
    bool trans,
    int m_batch_slice,
    const int *batch_indices) {

  int x_size = this->getXSize();
  int d_size = this->getDSize();
  const int *indices = this->getMatrixIndices();

  // CAUTION: need X_output to be set to zero!
  if ((dim3 == 1) || (!trans)) {

    SliceInputIterator<false, T> in_iter(
        D_input, d_size, m_batch, dim3, m_batch_slice, batch_indices);

    IndexReaderSliceOutputIterator<false, T> out_iter(
        X_output, indices, total_output_size / dim3, x_size, m_batch, dim3, m_batch_slice,
        batch_indices);

    this->backwardMatrixIterator(in_iter, out_iter, m_batch_slice * dim3, trans, trans);

  } else {
    SliceInputIterator<true, T> in_iter(
        D_input, d_size, m_batch, dim3, m_batch_slice, batch_indices);

    IndexReaderSliceOutputIterator<true, T> out_iter(
        X_output, indices, total_output_size / dim3, x_size, m_batch, dim3, m_batch_slice,
        batch_indices);

    this->backwardMatrixIterator(in_iter, out_iter, m_batch_slice * dim3, trans, trans);
  }
}

template <typename T> void RPUCudaPulsed<T>::checkBatchBuffers(const int m_batch) {
  RPU_GET_CUDA_BUFFER(this->context_, T, dev_batch_buffer_x_size_, m_batch * this->x_size_);
  RPU_GET_CUDA_BUFFER(this->context_, T, dev_batch_buffer_d_size_, m_batch * this->d_size_);
}

template <typename T>
template <typename InputIteratorT, typename OutputIteratorT>
void RPUCudaPulsed<T>::backwardMatrixIterator(
    InputIteratorT D_input, OutputIteratorT X_output, int m_batch, bool d_trans, bool x_trans) {

  checkBatchBuffers(m_batch);

  RPU::detail::backwardMatrixIteratorIOManaged(
      this->context_, this->getFBWeightsCuda(false), D_input, this->d_size_, d_trans, X_output,
      this->x_size_, x_trans, m_batch, this->getBwdAlpha(), *b_iom_, getMetaPar().b_io,
      dev_batch_buffer_d_size_, dev_batch_buffer_x_size_);
};

template <typename T>
void RPUCudaPulsed<T>::backwardVector(const T *d_input, T *x_output, int d_inc, int x_inc) {
  const T *d_input_inc1 = d_input;
  T *x_output_inc1 = x_output;

  if (x_inc != 1) {
    x_output_inc1 = dev_b_x_vector_inc1_->getData();
  }

  if (d_inc != 1) { // only needed for looped updates anyways
    RPU::math::copy<T>(
        this->context_, this->d_size_, d_input, d_inc, dev_b_d_vector_inc1_->getData(), 1);
    d_input_inc1 = dev_b_d_vector_inc1_->getDataConst();
  }
  this->backwardMatrixIterator(d_input_inc1, x_output_inc1, 1, false, false);

  if (x_inc != 1) {
    RPU::math::copy<T>(this->context_, this->x_size_, x_output_inc1, 1, x_output, x_inc);
  }
}

/*********************************************************************************/
/* UPDATE */
template <typename T> void RPUCudaPulsed<T>::finishUpdateCalculations() {
  if (getMetaPar().up.pulse_type != PulseType::None) {
    up_pwu_->waitForUpdateCalculations();
  }
}

template <typename T> void RPUCudaPulsed<T>::makeUpdateAsync() {
  if (getMetaPar().up.pulse_type != PulseType::None) {
    up_pwu_->makeUpdateAsync();
  }
}

template <typename T>
void RPUCudaPulsed<T>::updateMatrix(
    const T *X_input, const T *D_input, int m_batch, bool x_trans, bool d_trans) {
  updateMatrixIterator(X_input, D_input, m_batch, x_trans, d_trans);
}

template <typename T>
void RPUCudaPulsed<T>::updateIndexed(
    const T *X_input, const T *D_input, int total_input_size, int m_batch, int dim3, bool trans) {

  const int *indices = this->getMatrixIndices();
  int x_size = this->getXSize();

  if (trans && (dim3 > 1)) {
    IndexReaderTransInputIterator<T> in_iter(
        X_input, indices, total_input_size / dim3, m_batch, x_size * m_batch, m_batch * dim3);

    PermuterTransInputIterator<T> permute_iter(
        D_input, m_batch, this->getDSize() * m_batch, m_batch * dim3);
    updateMatrixIterator(in_iter, permute_iter, m_batch * dim3, trans, trans);

  } else {
    IndexReaderInputIterator<T> in_iter(
        X_input, indices, total_input_size / dim3, x_size * m_batch);
    updateMatrixIterator(in_iter, D_input, m_batch * dim3, trans, trans);
  }
}

template <typename T>
void RPUCudaPulsed<T>::updateIndexedSlice(
    const T *X_input,
    const T *D_input,
    int total_input_size,
    int m_batch,
    int dim3,
    bool trans,
    int m_batch_slice,
    const int *batch_indices) {

  const int *indices = this->getMatrixIndices();
  int x_size = this->getXSize();
  int d_size = this->getDSize();

  if (trans && (dim3 > 1)) {

    SliceInputIterator<true, T> d_in_iter(
        D_input, d_size, m_batch, dim3, m_batch_slice, batch_indices);

    IndexReaderSliceInputIterator<true, T> x_in_iter(
        X_input, indices, total_input_size / dim3, x_size, m_batch, dim3, m_batch_slice,
        batch_indices);

    this->updateMatrixIterator(x_in_iter, d_in_iter, m_batch_slice * dim3, trans, trans);

  } else {

    SliceInputIterator<false, T> d_in_iter(
        D_input, d_size, m_batch, dim3, m_batch_slice, batch_indices);

    IndexReaderSliceInputIterator<false, T> x_in_iter(
        X_input, indices, total_input_size / dim3, x_size, m_batch, dim3, m_batch_slice,
        batch_indices);

    this->updateMatrixIterator(x_in_iter, d_in_iter, m_batch_slice * dim3, trans, trans);
  }
}

template <typename T>
void RPUCudaPulsed<T>::updateVector(const T *x_input, const T *d_input, int x_inc, int d_inc) {

  const T *x_input_inc1 = x_input;
  const T *d_input_inc1 = d_input;

  if (x_inc !=
      1) { // could make iterators here. But never hit anyway (because matrix is used for inc>1)
    RPU::math::copy<T>(
        this->context_, this->x_size_, x_input, x_inc, dev_up_x_vector_inc1_->getData(), 1);
    x_input_inc1 = dev_up_x_vector_inc1_->getDataConst();
  }

  if (d_inc != 1) {
    RPU::math::copy<T>(
        this->context_, this->d_size_, d_input, d_inc, dev_up_d_vector_inc1_->getData(), 1);
    d_input_inc1 = dev_up_d_vector_inc1_->getDataConst();
  }

  updateMatrixIterator(x_input_inc1, d_input_inc1, 1, false, false);
}

template <typename T>
template <typename XInputIteratorT, typename DInputIteratorT>
void RPUCudaPulsed<T>::updateMatrixIterator(
    XInputIteratorT X_input, DInputIteratorT D_input, int m_batch, bool x_trans, bool d_trans) {
  this->last_update_m_batch_ = m_batch;

  const auto &up = getMetaPar().up;

  checkBatchBuffers(m_batch);
  up_pwu_->setSharedBuffer(m_batch, dev_batch_buffer_x_size_, dev_batch_buffer_d_size_);

  if (up_pwu_->checkForFPUpdate(&*rpucuda_device_, up)) {
    // we take a short-cut in case that FP update is requested:
    up_pwu_->doFPupdate(
        X_input, D_input, this->getUpWeightsCuda(), this->getAlphaLearningRate(), m_batch, x_trans,
        d_trans, this->getUpBeta());

  } else {
    T *local_dw = this->getDeltaWeights();

    if (local_dw) {
      // cannot do directly compute DW. Thus first copy weights and then apply
      this->setDeltaWeights(nullptr); // reset to normal up weights
      int sz = this->x_size_ * this->d_size_;
      RPU::math::copy<T>(this->context_, sz, this->getUpWeightsCuda(), 1, local_dw, 1);
    }
    T lr = this->getAlphaLearningRate();
    T *weights = this->getUpWeightsCuda();
    up_pwu_->update(
        X_input, D_input, weights, &*rpucuda_device_, up, lr, m_batch, x_trans, d_trans);

    if (local_dw) {
      this->setDeltaWeights(local_dw); // this might change the value of LR
      this->getAndResetWeightUpdate(local_dw, this->getAlphaLearningRate() / lr);
    }
  }
}

template class RPUCudaPulsed<float>;
#ifdef RPU_USE_DOUBLE
template class RPUCudaPulsed<double>;
#endif

#undef CHECK_RPU_DEVICE_INIT

} // namespace RPU
