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

#pragma once

#include "cuda_math_util.h"
#include "cuda_util.h"
#include "io_iterator.h"
#include "rng.h"
#include "rpu.h"
#include "utility_functions.h"
#include "weight_clipper_cuda.h"
#include "weight_drifter_cuda.h"
#include "weight_modifier_cuda.h"
#include <memory>
#include <random>

namespace RPU {

template <typename T> class RPUCudaSimple : public RPUSimple<T> {

public:
  RPUCudaSimple(){};
  explicit RPUCudaSimple(CudaContext *c, int x_size, int d_size);
  explicit RPUCudaSimple(CudaContext *c, RPUSimple<T> &o);
  explicit RPUCudaSimple(cudaStream_t s, int x_size, int d_size);
  explicit RPUCudaSimple(cudaStream_t s, RPUSimple<T> &o);

  ~RPUCudaSimple();

  RPUCudaSimple(const RPUCudaSimple<T> &);
  RPUCudaSimple<T> &operator=(const RPUCudaSimple<T> &); // = default;
  RPUCudaSimple(RPUCudaSimple<T> &&);                    //= default;
  RPUCudaSimple<T> &operator=(RPUCudaSimple<T> &&);      //= default;

  friend void swap(RPUCudaSimple<T> &a, RPUCudaSimple<T> &b) noexcept {

    using std::swap;
    swap(static_cast<RPUSimple<T> &>(a), static_cast<RPUSimple<T> &>(b));

    swap(a.context_, b.context_);
    swap(a.shared_context_, b.shared_context_);

    swap(a.dev_weights_, b.dev_weights_);
    swap(a.dev_weights_buffer_, b.dev_weights_buffer_);
    swap(a.dev_fb_weights_, b.dev_fb_weights_);
    swap(a.dev_delta_weights_extern_, b.dev_delta_weights_extern_);

    swap(a.dev_x_vector_, b.dev_x_vector_);
    swap(a.dev_d_vector_, b.dev_d_vector_);

    swap(a.dev_x_vector_bias_, b.dev_x_vector_bias_);
    swap(a.dev_x_matrix_bias_, b.dev_x_matrix_bias_);

    swap(a.dev_x_matrix_bias_size_, b.dev_x_matrix_bias_size_);

    swap(a.dev_temp_tensor_, b.dev_temp_tensor_);

    swap(a.rnd_diffusion_context_, b.rnd_diffusion_context_);
    swap(a.dev_diffusion_nrnd_, b.dev_diffusion_nrnd_);

    swap(a.wdrifter_cuda_, b.wdrifter_cuda_);
    swap(a.wclipper_cuda_, b.wclipper_cuda_);
    swap(a.fb_wmodifier_cuda_, b.fb_wmodifier_cuda_);
    swap(a.shared_weights_if_, b.shared_weights_if_);
  }

  void printToStream(std::stringstream &ss) const override;

  void setDeltaWeights(T *dw) override;

protected:
  T *getDeltaWeights() const override { return dev_delta_weights_extern_; };
  T *getFBWeightsCuda(
      bool is_test) const; // for specialized forward/backward. To be used in any forward
  T *getUpWeightsCuda();   // for specialized update. To be used in any update

  void copyWeightsToHost(T *weightsptr) const;
  T **copyWeightsToHost();

  // when overriding copy methods below, _Vector_Bias can be used in derived
  T *copyToVectorBiasBuffer(const T *x_input_without_bias, int x_inc) override;
  void copyFromVectorBiasBuffer(T *x_output_without_bias, int x_inc) override;
  T *getVectorBiasBuffer() const override { return dev_x_vector_bias_->getData(); };

  // new matrix and bias stuff (for Caffe2 interface)
  void copyWeightsFromBuffer() override;
  void copyWeightsToBuffer() override;
  T *getWeightsBufferCuda();

  T *copyToMatrixBiasBuffer(const T *X_input_without_bias, int m_batch, bool x_trans) override;
  void copyFromMatrixBiasBuffer(T *X_input_without_bias, int m_batch, bool x_trans) override;
  T *getMatrixBiasBuffer(int m_batch) override;

  void forwardVector(const T *x_input, T *d_output, int x_inc, int d_inc, bool is_test) override;
  void backwardVector(const T *d_input, T *x_output, int d_inc = 1, int x_inc = 1) override;
  void updateVector(const T *x_input, const T *d_input, int x_inc = 1, int d_inc = 1) override;

  // matrix methods (avoid looped version in Abstract)
  void forwardMatrix(
      const T *X_input, T *D_output, int m_batch, bool x_trans, bool d_trans, bool is_test)
      override;
  void backwardMatrix(
      const T *D_input,
      T *X_output,
      int m_batch,
      bool d_trans = false,
      bool x_trans = false) override;
  void updateMatrix(
      const T *X_input,
      const T *D_input,
      int m_batch,
      bool x_trans = false,
      bool d_trans = false) override;

  // for tensor interface
  void getTensorBuffer(T **x_tensor, T **d_tensor, int m_batch, int dim3) override;
  void
  permute132(T *out_tensor, const T *in_tensor, int dim1, int dim2, int dim3, bool bias2) override;

  // for indexed interface
  void copyIndexedInput(
      T *out_tensor,
      const T *src_tensor,
      const int total_input_size,
      const int *indices,
      const int size,
      const int m_batch,
      const int dim3,
      const bool trans,
      const int m_batch_slice = 0,
      const int *batch_indices = nullptr) override;

  void copyIndexedOutput(
      T *out_tensor,
      const T *src_tensor,
      const int total_output_size,
      const int *indices,
      const int size,
      const int m_batch,
      const int dim3,
      const bool trans,
      const int m_batch_slice = 0,
      const int *batch_indices = nullptr) override;
  void copySliceInput(
      T *out_tensor,
      const T *src_tensor,
      const int size,
      const int m_batch,
      const int dim3,
      const bool trans,
      const int m_batch_slice,
      const int *batch_indices) override;
  void copySliceOutput(
      T *out_tensor,
      const T *src_tensor,
      const int size,
      const int m_batch,
      const int dim3,
      const bool trans,
      const int m_batch_slice,
      const int *batch_indices) override;

  void setZero(T *v, const int size) override {
    RPU::math::elemconst<T>(context_, v, size, (T)0.0);
  };

public:
  void printWeights(int x_count, int d_count) override;

  void decayWeights(bool bias_no_decay) override;
  void decayWeights(T alpha, bool bias_no_decay) override;

  void driftWeights(T time_since_last_call) override;
  void diffuseWeights() override;

  void clipWeights(T clip) override;
  void clipWeights(const WeightClipParameter &wclpar) override;

  T **getWeights() override; // host weights. implicit copy from CUDA

  void getWeights(T *weightsptr) const override;
  void setWeights(const T *weightsptr) override;
  void setWeightsUniformRandom(T min_value, T max_value) override;
  void setSharedWeights(T *weightsptr) override;

  void getAndResetWeightUpdate(T *prev_weights_and_dw_out, T scale = 1.0) override;
  void applyWeightUpdate(T *dw_and_current_weights_out) override;

  void setStream(cudaStream_t s) { context_->setStream(s); };
  cudaStream_t getStream() { return context_->getStream(); };
  int getGPUId() { return context_->getGPUId(); };

  void modifyFBWeights(const WeightModifierParameter &wmpar) override;

protected:
  CudaContext *context_ = nullptr;
  std::shared_ptr<CudaContext> shared_context_ = nullptr;
  std::unique_ptr<CudaContext> rnd_diffusion_context_ = nullptr;
  std::unique_ptr<CudaArray<float>> dev_diffusion_nrnd_ = nullptr;
  std::unique_ptr<CudaArray<T>> dev_weights_ = nullptr;
  std::unique_ptr<CudaArray<T>> dev_weights_buffer_ = nullptr;
  std::unique_ptr<CudaArray<T>> dev_fb_weights_ = nullptr;
  T *dev_delta_weights_extern_ = nullptr;
  std::unique_ptr<CudaArray<T>> dev_x_vector_ = nullptr;
  std::unique_ptr<CudaArray<T>> dev_d_vector_ = nullptr;

  std::unique_ptr<CudaArray<T>> dev_x_vector_bias_ = nullptr;
  std::unique_ptr<CudaArray<T>> dev_x_matrix_bias_ = nullptr;
  int dev_x_matrix_bias_size_ = 0;

  std::unique_ptr<CudaArray<T>> dev_temp_tensor_ = nullptr;

  std::unique_ptr<WeightClipperCuda<T>> wclipper_cuda_ = nullptr;

private:
  bool shared_weights_if_ = false;
  void
  initFrom(const RPUSimple<T> &rpu_in); // to populate from CPU->CUDA, will be called by constructor
  void initialize(CudaContext *c);
  std::unique_ptr<WeightModifierCuda<T>> fb_wmodifier_cuda_ = nullptr;
  std::unique_ptr<WeightDrifterCuda<T>> wdrifter_cuda_ = nullptr;
};

} // namespace RPU
