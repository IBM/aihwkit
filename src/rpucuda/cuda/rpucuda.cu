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

#include "forward_backward_pass.h"
#include "rpu.h"
#include "rpucuda.h"
#include "utility_functions.h"

#include <iostream>
// #include <random>
#include <chrono>
#include <cmath>
#include <memory>

namespace RPU {

/********************************************************************************
 * RPUCudaSimple<T>
 *********************************************************************************/
template <typename T> void RPUCudaSimple<T>::initialize(CudaContextPtr c) {

  context_ = c;
  dev_weights_ = RPU::make_unique<CudaArray<T>>(c, this->x_size_ * this->d_size_);
  dev_weights_buffer_ = nullptr;
  dev_fb_weights_ = nullptr;
  dev_delta_weights_extern_ = nullptr;

  dev_x_vector_bias_ = RPU::make_unique<CudaArray<T>>(c, this->x_size_);
  dev_x_vector_bias_->setConst(1.0);

  dev_d_vector_ = RPU::make_unique<CudaArray<T>>(c, this->d_size_);
  dev_x_vector_ = RPU::make_unique<CudaArray<T>>(c, this->x_size_);

  dev_temp_tensor_ = nullptr;
  // init later only if needed
  dev_flicker_states_ = nullptr;

  context_->synchronize();
}

template <typename T>
RPUCudaSimple<T>::RPUCudaSimple(CudaContextPtr c, int x_size, int d_size)
    : RPUSimple<T>(x_size, d_size) {
  this->initialize(c);

  DEBUG_CALL(this->disp(););
  DEBUG_OUT("RPUCudaSimple constructed");
}

template <typename T>
RPUCudaSimple<T>::RPUCudaSimple(cudaStream_t s, int x_size, int d_size)
    : RPUSimple<T>(x_size, d_size) {
  internal_context_ = RPU::make_unique<CudaContext>(s);

  this->initialize(&*internal_context_);

  DEBUG_CALL(this->disp(););
  DEBUG_OUT("RPUCudaSimple constructed (on shared stream)");
}

template <typename T> void RPUCudaSimple<T>::initFrom(const RPUSimple<T> &rpu) {
  // this is private and called from constructors below
  this->setWeights(rpu.getWeightsPtr()[0]);
  // TODO:  copy flicker states etc to cuda  here !!!??
}

template <typename T>
RPUCudaSimple<T>::RPUCudaSimple(CudaContextPtr c, RPUSimple<T> &o) : RPUSimple<T>(o) {

  this->initialize(c);
  initFrom(o);

  DEBUG_OUT("RPUCudaSimple constructed from RPUSimple");
}

template <typename T>
RPUCudaSimple<T>::RPUCudaSimple(cudaStream_t s, RPUSimple<T> &o) : RPUSimple<T>(o) {

  // we are using the copy constructor of the base class RPUSimple
  internal_context_ = RPU::make_unique<CudaContext>(s);
  this->initialize(&*internal_context_);
  initFrom(o);

  DEBUG_OUT("RPUCudaSimple constructed from RPUSimple on shared stream");
}

template <typename T> RPUCudaSimple<T>::~RPUCudaSimple() {
  // no need to care about the shared_pointers
}

// copy constructor
template <typename T>
RPUCudaSimple<T>::RPUCudaSimple(const RPUCudaSimple<T> &other) : RPUSimple<T>(other) {
  other.context_->synchronizeContext();
  this->initialize(other.context_); // private

  // note: CUDA event/stream logic should be unique. It is only copied if already shared
  internal_context_ = RPU::make_unique<CudaContext>(*other.internal_context_);

  if (other.dev_weights_) {
    dev_weights_->assign(*other.dev_weights_);
  }
  // note: cannot copy external shared weight pointer... user needs to
  // ensure that again setShared is called after copying
  shared_weights_if_ = false;

  if (other.dev_weights_buffer_) {
    getWeightsBufferCuda(); // to initialize
    dev_weights_buffer_->assign(*other.dev_weights_buffer_);
  }
  if (other.dev_fb_weights_) {
    dev_fb_weights_ = RPU::make_unique<CudaArray<T>>(context_, this->x_size_ * this->d_size_);
    dev_fb_weights_->assign(*other.dev_fb_weights_);
  }

  // cannot copy external weight pointer... user needs to call it again
  dev_delta_weights_extern_ = nullptr;

  if (other.fb_wmodifier_cuda_) {
    // no copy... just new..
    fb_wmodifier_cuda_ =
        RPU::make_unique<WeightModifierCuda<T>>(context_, this->x_size_, this->d_size_);
  }

  if (other.wdrifter_cuda_) {
    wdrifter_cuda_ = RPU::make_unique<WeightDrifterCuda<T>>(*other.wdrifter_cuda_);
  }

  // no copy
  wremapper_cuda_ = nullptr;
  wclipper_cuda_ = nullptr;

  dev_x_vector_->assign(*other.dev_x_vector_);
  dev_d_vector_->assign(*other.dev_d_vector_);

  dev_x_vector_bias_->assign(*other.dev_x_vector_bias_);

  // do not care to copy the matrix/tensor/rnd buffers (will init automatically)

  if (other.dev_flicker_states_ != nullptr) {
    dev_flicker_states_ =
        RPU::make_unique<CudaArray<uint64_t>>(other.context_, other.dev_flicker_states_->getSize());
    dev_flicker_states_->assign(*other.dev_flicker_states_);
  }

  context_->synchronizeContext();

  DEBUG_CALL(this->disp(););
  DEBUG_OUT("RPUCuda_Simple copy constructed.");
}

// copy assignment
template <typename T> RPUCudaSimple<T> &RPUCudaSimple<T>::operator=(const RPUCudaSimple<T> &other) {
  RPUCudaSimple<T> tmp(other);
  swap(*this, tmp);
  context_->synchronize(); // need sync because of tmp
  return *this;
}

// move constructor
template <typename T> RPUCudaSimple<T>::RPUCudaSimple(RPUCudaSimple<T> &&other) {
  *this = std::move(other);
}

// move assignment
template <typename T> RPUCudaSimple<T> &RPUCudaSimple<T>::operator=(RPUCudaSimple<T> &&other) {

  RPUSimple<T>::operator=(std::move(other));

  context_ = other.context_;
  other.context_ = nullptr;

  internal_context_ = std::move(other.internal_context_);

  dev_weights_ = std::move(other.dev_weights_);
  dev_weights_buffer_ = std::move(other.dev_weights_buffer_);
  dev_fb_weights_ = std::move(other.dev_fb_weights_);

  dev_delta_weights_extern_ = other.dev_delta_weights_extern_;
  other.dev_delta_weights_extern_ = nullptr;

  dev_x_vector_ = std::move(other.dev_x_vector_);
  dev_d_vector_ = std::move(other.dev_d_vector_);
  dev_x_vector_bias_ = std::move(dev_x_vector_bias_);

  dev_temp_tensor_ = std::move(other.dev_temp_tensor_);

  rnd_diffusion_context_ = std::move(other.rnd_diffusion_context_);
  dev_diffusion_nrnd_ = std::move(other.dev_diffusion_nrnd_);
  dev_flicker_states_ = std::move(other.dev_flicker_states_);

  wdrifter_cuda_ = std::move(other.wdrifter_cuda_);
  wremapper_cuda_ = std::move(other.wremapper_cuda_);

  shared_weights_if_ = other.shared_weights_if_;
  return *this;
}

template <typename T>
void RPUCudaSimple<T>::dumpExtra(RPU::state_t &extra, const std::string prefix) {

  context_->synchronize();

  RPUSimple<T>::dumpExtra(extra, prefix);
  RPU::state_t state;

  if (this->use_delayed_update_) {
    RPU::insert(state, "dev_weights_buffer", dev_weights_buffer_);
  }
  RPU::insert(state, "dev_flicker_states", dev_flicker_states_);
  RPU::insert(state, "dev_fb_weights", dev_fb_weights_);

  if (dev_fb_weights_) {
    fb_wmodifier_cuda_->dumpExtra(state, "fb_wmodifier_cuda");
  }
  if (wdrifter_cuda_) {
    wdrifter_cuda_->dumpExtra(state, "wdrifter_cuda");
  }
  if (wremapper_cuda_) {
    wremapper_cuda_->dumpExtra(state, "wremapper_cuda");
  }
  if (wclipper_cuda_) {
    wclipper_cuda_->dumpExtra(state, "wclipper_cuda");
  }
  RPU::insertWithPrefix(extra, state, prefix);

  // extern not handled
  // tmp buffers not needed
}

template <typename T>
void RPUCudaSimple<T>::loadExtra(const RPU::state_t &extra, const std::string prefix, bool strict) {
  using V = std::vector<T>;
  context_->synchronize();

  RPUSimple<T>::loadExtra(extra, prefix, strict);
  auto state = RPU::selectWithPrefix(extra, prefix);

  if (this->use_delayed_update_) {
    RPU::load(context_, state, "dev_weights_buffer", dev_weights_buffer_, strict);
  }
  if (state.count("dev_fb_weights")) {
    RPU::load(context_, state, "dev_fb_weights", dev_fb_weights_, strict);
  }
  RPU::load(context_, state, "dev_flicker_states", dev_flicker_states_, strict);

  if (dev_fb_weights_) {
    if (!fb_wmodifier_cuda_) {
      fb_wmodifier_cuda_ =
          RPU::make_unique<WeightModifierCuda<T>>(context_, this->x_size_, this->d_size_);
    }
    fb_wmodifier_cuda_->loadExtra(state, "fb_wmodifier_cuda", strict);
  }

  if (state.count("wdrifter_cuda")) {
    if (!wdrifter_cuda_) {
      auto wd = WeightDrifter<T>(this->x_size_ * this->d_size_, this->getPar().drift);
      wdrifter_cuda_ =
          RPU::make_unique<WeightDrifterCuda<T>>(this->context_, wd, this->x_size_, this->d_size_);
    }
    wdrifter_cuda_->loadExtra(state, "wdrifter_cuda", strict);
  }

  if (state.count("wremapper_cuda")) {
    if (!wremapper_cuda_) {
      wremapper_cuda_ =
          RPU::make_unique<WeightRemapperCuda<T>>(context_, this->x_size_, this->d_size_);
    }
    wremapper_cuda_->loadExtra(state, "wremapper_cuda", strict);
  }

  if (state.count("wclipper_cuda")) {
    if (!wclipper_cuda_) {
      wclipper_cuda_ =
          RPU::make_unique<WeightClipperCuda<T>>(this->context_, this->x_size_, this->d_size_);
    }
    wclipper_cuda_->loadExtra(state, "wclipper_cuda", strict);
  }
}

/***************************************************************************************/
template <typename T> void RPUCudaSimple<T>::copyWeightsToHost(T *weightsptr) const {
  // copies the weights to the host and returns weight pointer, without changing the simple weights
  DEBUG_OUT("RPUCuda: Get weights.");
  T **transposed_weights = Array_2D_Get<T>(this->x_size_, this->d_size_);

  context_->synchronizeDevice();
  dev_weights_->copyTo(transposed_weights[0]);

  for (int i = 0; i < this->d_size_; ++i) {
    for (int j = 0; j < this->x_size_; ++j) {
      weightsptr[j + this->x_size_ * i] = transposed_weights[j][i];
    }
  }
  Array_2D_Free<T>(transposed_weights);
}

template <typename T> T **RPUCudaSimple<T>::copyWeightsToHost() {
  // copies the weights to the host by using the simple weights and returns weight pointer

  this->copyWeightsToHost(RPUSimple<T>::getWeightsPtr()[0]);
  return RPUSimple<T>::getWeightsPtr();
}

template <typename T> T **RPUCudaSimple<T>::getWeights() { return this->copyWeightsToHost(); }

template <typename T> void RPUCudaSimple<T>::getWeights(T *weightsptr) const {
  this->copyWeightsToHost(weightsptr);
}

template <typename T> void RPUCudaSimple<T>::setWeights(const T *host_source) {
  // expects row order weights as source and sets the device weights

  RPUSimple<T>::setWeights(host_source);
  dev_weights_->assignTranspose(
      RPUSimple<T>::getWeightsPtr()[0], this->getDSize(), this->getXSize());
}

template <typename T> void RPUCudaSimple<T>::setSharedWeights(T *device_source) {

  if (device_source != dev_weights_->getData()) {

    if (!this->shared_weights_if_) {
      context_->synchronizeDevice();
      dev_weights_->copyTo(device_source);
    }
    dev_weights_->synchronize();
    dev_weights_->setShared(device_source);
    this->shared_weights_if_ = true;
  }
}

template <typename T>
void RPUCudaSimple<T>::getAndResetWeightUpdate(T *prev_weight_and_dw_out, T scale) {
  RPU::math::elemsubcopy<T>(
      context_, dev_weights_->getData(), prev_weight_and_dw_out, dev_weights_->getSize(), scale);
}

template <typename T> void RPUCudaSimple<T>::applyWeightUpdate(T *dw_and_current_weight_out) {
  RPU::math::elemaddcopy<T>(
      context_, dev_weights_->getData(), dw_and_current_weight_out, dev_weights_->getSize());
}

template <typename T> void RPUCudaSimple<T>::setWeightsUniformRandom(T min_value, T max_value) {

  DEBUG_OUT("RPUCudaSimple weights init [" << min_value << "," << max_value << "]");
  RPUSimple<T>::setWeightsUniformRandom(min_value, max_value);
  this->setWeights(this->weights_[0]); // copies to CPU twice. Ignore
}

template <typename T> void RPUCudaSimple<T>::printToStream(std::stringstream &ss) const {
  ss << "RPUCudaSimple<" << this->getDataTypeName() << ">(" << this->d_size_ << "," << this->x_size_
     << ")" << std::endl;
};

/*********************************************************************************/
/*MATRIX/BIAS updates etc*********************************************************/

template <typename T> T *RPUCudaSimple<T>::getWeightsBufferCuda() {
  if (dev_weights_buffer_ == nullptr) {
    dev_weights_buffer_ =
        RPU::make_unique<CudaArray<T>>(this->context_, this->x_size_ * this->d_size_);
    this->context_->synchronize();
  }
  return dev_weights_buffer_->getData();
};

template <typename T> void RPUCudaSimple<T>::copyWeightsFromBuffer() {
  RPU::math::copy<T>(
      this->context_, this->x_size_ * this->d_size_, getWeightsBufferCuda(), 1,
      dev_weights_->getData(), 1);
}

template <typename T> void RPUCudaSimple<T>::copyWeightsToBuffer() {
  RPU::math::copy<T>(
      this->context_, this->x_size_ * this->d_size_, dev_weights_->getData(), 1,
      getWeightsBufferCuda(), 1);
}

template <typename T> T *RPUCudaSimple<T>::getMatrixBiasBuffer(int m_batch) {
  T *bias_buffer = context_->template getSharedBuffer<T>(RPU_BUFFER_BIAS, m_batch * this->x_size_);
  return bias_buffer;
}
template <typename T> void RPUCudaSimple<T>::releaseMatrixBiasBuffer() {
  context_->template releaseSharedBuffer<T>(RPU_BUFFER_BIAS);
}

template <typename T>
T *RPUCudaSimple<T>::copyToMatrixBiasBuffer(
    const T *X_input_without_bias, int m_batch, bool x_trans) {

  T *bias_buffer = this->getMatrixBiasBuffer(m_batch);
  RPU::math::makeBias<T>(
      this->context_, bias_buffer, X_input_without_bias, this->x_size_, m_batch, x_trans);
  return bias_buffer;
};

template <typename T>
void RPUCudaSimple<T>::copyFromMatrixBiasBuffer(
    T *X_input_without_bias, int m_batch, bool x_trans, T *bias_buffer) {
  RPU::math::copyWithoutBias<T>(
      this->context_, X_input_without_bias, bias_buffer, this->x_size_, m_batch, x_trans);
};

template <typename T>
T *RPUCudaSimple<T>::copyToVectorBiasBuffer(const T *x_input_without_bias, int x_inc) {
  RPU::math::copy<T>(
      context_, this->x_size_ - 1, x_input_without_bias, x_inc, dev_x_vector_bias_->getData(), 1);
  // last value of devx_vector_bias_ is set to 1 at the initialization  (and never changed)!!
  return dev_x_vector_bias_->getData();
};

template <typename T>
void RPUCudaSimple<T>::copyFromVectorBiasBuffer(T *x_output_without_bias, int x_inc) {
  RPU::math::copy<T>(
      context_, this->x_size_ - 1, dev_x_vector_bias_->getData(), 1, x_output_without_bias, x_inc);
};

/*********************************************************************************/
template <typename T>
void RPUCudaSimple<T>::getTensorBuffer(T **x_tensor_ptr, T **d_tensor_ptr, int m_batch, int dim3) {
  int x_size = this->getXSize();
  int d_size = this->getDSize();

  int n = (x_size + d_size) * dim3 * m_batch;
  if ((dev_temp_tensor_ == nullptr) || (dev_temp_tensor_->getSize() < n)) {
    this->context_->synchronize();
    dev_temp_tensor_ = RPU::make_unique<CudaArray<T>>(context_, n);
    this->context_->synchronize();
  }
  *x_tensor_ptr = dev_temp_tensor_->getData();
  *d_tensor_ptr = *x_tensor_ptr + (x_size)*dim3 * m_batch;
}

template <typename T>
void RPUCudaSimple<T>::permute132(
    T *out_tensor, const T *in_tensor, int dim1, int dim2, int dim3, bool bias2) {
  RPU::math::permute132<T>(context_, out_tensor, in_tensor, dim1, dim2, dim3, bias2);
}

/*********************************************************************************/
template <typename T>
void RPUCudaSimple<T>::copyIndexedInput(
    T *out_tensor,
    const T *in_tensor,
    const int total_input_size,
    const int *indices,
    const int size,
    const int m_batch,
    const int dim3,
    const bool trans,
    const int m_batch_slice,
    const int *batch_indices) {
  bool batch_slice = m_batch_slice > 0;

  if (batch_slice) {
    if (trans) {
      IndexReaderSliceInputIterator<true, T> in_iter(
          in_tensor, indices, total_input_size / dim3, size, m_batch, dim3, m_batch_slice,
          batch_indices);
      RPU::math::copyWithIterator(context_, out_tensor, in_iter, m_batch_slice * size * dim3);

    } else {
      IndexReaderSliceInputIterator<false, T> in_iter(
          in_tensor, indices, total_input_size / dim3, size, m_batch, dim3, m_batch_slice,
          batch_indices);
      RPU::math::copyWithIterator(context_, out_tensor, in_iter, m_batch_slice * size * dim3);
    }
  } else {
    if (trans) {
      IndexReaderTransInputIterator<T> in_iter(
          in_tensor, indices, total_input_size / dim3, m_batch, size * m_batch, m_batch * dim3);
      RPU::math::copyWithIterator(context_, out_tensor, in_iter, m_batch * size * dim3);
    } else {

      IndexReaderInputIterator<T> in_iter(
          in_tensor, indices, total_input_size / dim3, size * m_batch);
      RPU::math::copyWithIterator(context_, out_tensor, in_iter, m_batch * size * dim3);
    }
  }
}

template <typename T>
void RPUCudaSimple<T>::copyIndexedOutput(
    T *out_tensor,
    const T *in_tensor,
    const int total_output_size,
    const int *indices,
    const int size,
    const int m_batch,
    const int dim3,
    const bool trans,
    const int m_batch_slice,
    const int *batch_indices) {
  // CAUTION:  NO zeroying. Needs to be done from outside.
  bool batch_slice = m_batch_slice > 0;

  if (batch_slice) {
    if (trans) {
      IndexReaderSliceOutputIterator<true, T> out_iter(
          out_tensor, indices, total_output_size / dim3, size, m_batch, dim3, m_batch_slice,
          batch_indices);
      RPU::math::copyWithIterator(context_, out_iter, in_tensor, m_batch_slice * size * dim3);

    } else {
      IndexReaderSliceOutputIterator<false, T> out_iter(
          out_tensor, indices, total_output_size / dim3, size, m_batch, dim3, m_batch_slice,
          batch_indices);
      RPU::math::copyWithIterator(context_, out_iter, in_tensor, m_batch_slice * size * dim3);
    }

  } else {

    if (trans) {
      IndexReaderTransOutputIterator<T> out_iter(
          out_tensor, indices, total_output_size / dim3, m_batch, size * m_batch, m_batch * dim3);
      RPU::math::copyWithIterator(context_, out_iter, in_tensor, size * m_batch * dim3);
    } else {

      IndexReaderOutputIterator<T> out_iter(
          out_tensor, indices, total_output_size / dim3, size * m_batch);
      RPU::math::copyWithIterator(context_, out_iter, in_tensor, size * m_batch * dim3);
    }
  }
}

template <typename T>
void RPUCudaSimple<T>::copySliceInput(
    T *out_tensor,
    const T *in_tensor,
    const int size,
    const int m_batch,
    const int dim3,
    const bool trans,
    const int m_batch_slice,
    const int *batch_indices) {

  if (trans) {
    SliceInputIterator<true, T> in_iter(
        in_tensor, size, m_batch, dim3, m_batch_slice, batch_indices);
    RPU::math::copyWithIterator(context_, out_tensor, in_iter, m_batch_slice * size * dim3);

  } else {
    SliceInputIterator<false, T> in_iter(
        in_tensor, size, m_batch, dim3, m_batch_slice, batch_indices);

    RPU::math::copyWithIterator(context_, out_tensor, in_iter, m_batch_slice * size * dim3);
  }
}

template <typename T>
void RPUCudaSimple<T>::copySliceOutput(
    T *out_tensor,
    const T *in_tensor,
    const int size,
    const int m_batch,
    const int dim3,
    const bool trans,
    const int m_batch_slice,
    const int *batch_indices) {

  if (trans) {
    SliceOutputIterator<true, T> out_iter(
        out_tensor, size, m_batch, dim3, m_batch_slice, batch_indices);
    RPU::math::copyWithIterator(context_, out_iter, in_tensor, m_batch_slice * size * dim3);

  } else {
    SliceOutputIterator<false, T> out_iter(
        out_tensor, size, m_batch, dim3, m_batch_slice, batch_indices);
    RPU::math::copyWithIterator(context_, out_iter, in_tensor, m_batch_slice * size * dim3);
  }
}

/*********************************************************************************/
template <typename T>
void RPUCudaSimple<T>::forwardMatrix(
    const T *X_input, T *D_output, int m_batch, bool x_trans, bool d_trans, bool is_test) {

  RPU::detail::forwardMatrix(
      this->context_, getFBWeightsCuda(is_test), X_input, this->x_size_, x_trans, D_output,
      this->d_size_, d_trans, m_batch, this->getFwdAlpha());
};

/*********************************************************************************/
template <typename T>
void RPUCudaSimple<T>::backwardMatrix(
    const T *D_input, T *X_output, int m_batch, bool d_trans, bool x_trans) {

  RPU::detail::backwardMatrix(
      this->context_, getFBWeightsCuda(false), D_input, this->d_size_, d_trans, X_output,
      this->x_size_, x_trans, m_batch, this->getBwdAlpha());
};

/*********************************************************************************/
template <typename T> inline T *RPUCudaSimple<T>::getUpWeightsCuda() {
  // This is called from the Update routines to check which weight
  // is used for calculation. If dw is defined, then it will use the
  // DW mode, meaning that it will write into delta_weights the DW
  // and keep the weights. For HW RPU models that might include
  // first using W and then writing the difference to dW
  //
  // is also handles the delayed weights
  if (this->use_delayed_update_) {
    return getWeightsBufferCuda();
  } else {
    return (dev_delta_weights_extern_ != nullptr) ? dev_delta_weights_extern_
                                                  : dev_weights_->getData();
  }
}

/*********************************************************************************/
template <typename T>
void RPUCudaSimple<T>::updateMatrix(
    const T *X_input, const T *D_input, int m_batch, bool x_trans, bool d_trans) {
  this->last_update_m_batch_ = m_batch;

  RPU::math::gemm<T>(
      context_, d_trans, !x_trans,
      this->d_size_, // M
      this->x_size_, // N
      m_batch,       // K
      -this->getAlphaLearningRate(), D_input, d_trans ? m_batch : this->d_size_, X_input,
      x_trans ? m_batch : this->x_size_, this->getUpBeta(), this->getUpWeightsCuda(),
      this->d_size_);
}

/*********************************************************************************/
/* vector forward/backward/update */
template <typename T>
void RPUCudaSimple<T>::forwardVector(
    const T *x_input, T *d_output, int x_inc, int d_inc, bool is_test) {
  RPU::math::gemv<T>(
      context_, false, this->d_size_, this->x_size_, this->getFwdAlpha(), getFBWeightsCuda(is_test),
      this->d_size_, // because of *column* major format !
      x_input, x_inc, (T)0.0, d_output, d_inc);

  // NOTE: no synchronization is done here. Assume all inputs are on the same stream
  // if not, synchronization of contexts has to be done externaly.
};

template <typename T>
void RPUCudaSimple<T>::backwardVector(const T *d_input, T *x_output, int d_inc, int x_inc) {
  RPU::math::gemv<T>(
      context_,
      true, // transpose
      this->d_size_, this->x_size_, this->getBwdAlpha(), getFBWeightsCuda(false), this->d_size_,
      d_input, d_inc, (T)0.0, x_output, x_inc);
};

template <typename T>
void RPUCudaSimple<T>::updateVector(const T *x_input, const T *d_input, int x_inc, int d_inc) {
  if (!this->getDeltaWeights()) {
    RPU::math::ger<T>(
        context_, this->d_size_, this->x_size_, -this->getAlphaLearningRate(), d_input, d_inc,
        x_input, x_inc, getUpWeightsCuda(), this->d_size_);
  } else {
    if (x_inc == 1 && d_inc == 1) {
      RPUCudaSimple<T>::updateMatrix(x_input, d_input, 1, false, false);
    } else {
      RPU_FATAL("Update_Vector for delta weights and xd_inc>1 is not implemented.");
    }
  }
}

/*********************************************************************************/
template <typename T> void RPUCudaSimple<T>::decayWeights(T alpha, bool bias_no_decay) {
  T lifetime = this->getPar().lifetime;
  T decay_rate = (lifetime > (T)1.0) ? ((T)1.0 / lifetime) : (T)0.0;
  T decay_scale = (T)1.0 - alpha * decay_rate;

  if (decay_scale > (T)0.0 && decay_scale < (T)1.0) {
    int size = this->x_size_ * this->d_size_;
    // we have d_size major, ie col-major. Thus bias is just at the end
    RPU::math::scal<T>(
        context_, bias_no_decay ? MAX(size - this->d_size_, 0) : size, decay_scale,
        dev_weights_->getData());
  }
}

template <typename T> void RPUCudaSimple<T>::decayWeights(bool bias_no_decay) {
  RPUCudaSimple<T>::decayWeights((T)1.0, bias_no_decay);
}

/*********************************************************************************/
template <typename T> void RPUCudaSimple<T>::driftWeights(T time_since_last_call) {

  if (wdrifter_cuda_ == nullptr) {
    auto wd =
        WeightDrifter<T>(this->x_size_ * this->d_size_, this->getPar().drift); // forces simple
    wdrifter_cuda_ =
        RPU::make_unique<WeightDrifterCuda<T>>(this->context_, wd, this->x_size_, this->d_size_);
    context_->synchronize();
  }
  wdrifter_cuda_->apply(dev_weights_->getData(), time_since_last_call);
}

/*********************************************************************************/
template <typename T> void RPUCudaSimple<T>::clipWeights(T clip) {

  if (clip >= (T)0.0) {
    RPU::math::aclip<T>(context_, dev_weights_->getData(), dev_weights_->getSize(), clip);
  }
}

template <typename T> void RPUCudaSimple<T>::clipWeights(const WeightClipParameter &wclpar) {

  if (!wclipper_cuda_) {
    wclipper_cuda_ =
        RPU::make_unique<WeightClipperCuda<T>>(this->context_, this->x_size_, this->d_size_);
  }

  wclipper_cuda_->apply(dev_weights_->getData(), wclpar);
}

/*********************************************************************************/
template <typename T> void RPUCudaSimple<T>::diffuseWeights() {

  T diffusion = this->getPar().diffusion;

  if (diffusion <= (T)0.0) {
    return;
  }

  if (rnd_diffusion_context_ == nullptr) {
    // first time: init
    rnd_diffusion_context_ = RPU::make_unique<CudaContext>(context_->getGPUId());
    dev_diffusion_nrnd_ = RPU::make_unique<CudaArray<float>>(
        &*rnd_diffusion_context_, (dev_weights_->getSize() + 31) / 32 * 32);

    rnd_diffusion_context_->setRandomSeed(0);
    rnd_diffusion_context_->randNormal(
        dev_diffusion_nrnd_->getData(), dev_diffusion_nrnd_->getSize());
  }

  context_->recordWaitEvent(rnd_diffusion_context_->getStream());

  RPU::math::elemaddscale<T>(
      context_, dev_weights_->getData(), dev_weights_->getSize(), dev_diffusion_nrnd_->getData(),
      diffusion);

  rnd_diffusion_context_->recordWaitEvent(context_->getStream());
  rnd_diffusion_context_->randNormal(
      dev_diffusion_nrnd_->getData(), dev_diffusion_nrnd_->getSize());
}

/*********************************************************************************/
template <typename T> void RPUCudaSimple<T>::diffuseWeightsPink() {

  ENFORCE_NO_DELAYED_UPDATE;

  const auto &flicker = this->getPar().flicker;
  T diffusion = this->getPar().diffusion;

  if (diffusion <= (T)0.0) {
    return;
  }

  int size = dev_weights_->getSize();

  if (dev_flicker_states_ == nullptr) {
    uint64_t *init_flicker_states = RPUSimple<T>::initFlickerStates();

    dev_flicker_states_ =
        RPU::make_unique<CudaArray<uint64_t>>(context_, size, init_flicker_states);
    context_->synchronize();
  }

  const auto &par = this->getPar();

  RPU::math::elemaddpinknoise<T>(
      context_, dev_weights_->getData(), flicker.wreset ? getWeightsBufferCuda() : nullptr, size,
      diffusion, flicker.n, flicker.r, flicker.q, flicker.h, flicker.wreset_tol,
      dev_flicker_states_->getData(), context_->getRandomStates(size));
}

/***********************************************************************/

template <typename T> void RPUCudaSimple<T>::setDeltaWeights(T *dw_extern) {

  ENFORCE_NO_DELAYED_UPDATE;
  dev_delta_weights_extern_ = dw_extern;
}

template <typename T> T *RPUCudaSimple<T>::getFBWeightsCuda(bool is_test) const {
  bool use_fb = dev_fb_weights_ &&
                (!is_test || (fb_wmodifier_cuda_ && fb_wmodifier_cuda_->enableDuringTest()));
  return use_fb ? dev_fb_weights_->getData() : dev_weights_->getData();
}

template <typename T>
void RPUCudaSimple<T>::modifyFBWeights(const WeightModifierParameter<T> &wmpar) {

  ENFORCE_NO_DELAYED_UPDATE; // will get confused with the buffer

  if (dev_fb_weights_ == nullptr) {
    dev_fb_weights_ = RPU::make_unique<CudaArray<T>>(context_, this->x_size_ * this->d_size_);
    fb_wmodifier_cuda_ =
        RPU::make_unique<WeightModifierCuda<T>>(context_, this->x_size_, this->d_size_);
    context_->synchronize();
  }

  // modify FB weights
  fb_wmodifier_cuda_->apply(dev_fb_weights_->getData(), dev_weights_->getDataConst(), wmpar);
}

template <typename T>
void RPUCudaSimple<T>::remapWeights(const WeightRemapParameter &wrmpar, T *scales, T *biases) {

  ENFORCE_NO_DELAYED_UPDATE; // will get confused with the buffer

  if (wremapper_cuda_ == nullptr) {
    wremapper_cuda_ =
        RPU::make_unique<WeightRemapperCuda<T>>(context_, this->x_size_, this->d_size_);
  }

  // remap weights
  wremapper_cuda_->apply(
      dev_weights_->getData(), this->getAlphaLearningRate(), wrmpar, scales, biases);
}

template <typename T>
bool RPUCudaSimple<T>::swaWeights(
    const WeightRemapParameter &wrmpar, T *hwa_weights, uint64_t iter, T *scales, T *biases) {

  ENFORCE_NO_DELAYED_UPDATE; // will get confused with the buffer

  if (wremapper_cuda_ == nullptr) {
    wremapper_cuda_ =
        RPU::make_unique<WeightRemapperCuda<T>>(context_, this->x_size_, this->d_size_);
  }

  // swa weights
  return wremapper_cuda_->applySWA(
      hwa_weights, dev_weights_->getData(), iter, wrmpar, this->getAlphaLearningRate(), scales,
      biases);
}

/*********************************************************************************/
template <typename T> void RPUCudaSimple<T>::printWeights(int x_count, int d_count) {
  this->copyWeightsToHost(this->weights_[0]);
  RPUSimple<T>::printWeights(x_count, d_count);
}

/*********************************************************************************/
template <typename T> void RPUCudaSimple<T>::finishAllCalculations() {
  this->finishUpdateCalculations();
  this->context_->synchronizeContext();
}

template class RPUCudaSimple<float>;
#ifdef RPU_USE_DOUBLE
template class RPUCudaSimple<double>;
#endif
#ifdef RPU_USE_FP16
template class RPUCudaSimple<half_t>;
#endif

} // namespace RPU
