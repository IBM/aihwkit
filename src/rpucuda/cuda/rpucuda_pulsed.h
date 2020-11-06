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

#pragma once

#include "cuda_util.h"
#include "pulsed_weight_updater.h"
#include "rpu_pulsed.h"
#include "rpucuda.h"
#include "rpucuda_pulsed_device.h"
#include <memory>

#include "bit_line_maker.h"
#include "io_iterator.h"
#include "io_manager.h"
#include "maximizer.h"

namespace RPU {

template <typename T> class RPUCudaPulsed : public RPUCudaSimple<T> {

public:
  explicit RPUCudaPulsed(){}; // dummy
  explicit RPUCudaPulsed(CudaContext *c, int x_size, int d_size);
  explicit RPUCudaPulsed(CudaContext *c, RPUPulsed<T> &o);
  explicit RPUCudaPulsed(cudaStream_t s, int x_size, int d_size);
  explicit RPUCudaPulsed(cudaStream_t s, RPUPulsed<T> &o);

  ~RPUCudaPulsed();

  RPUCudaPulsed(const RPUCudaPulsed<T> &);
  RPUCudaPulsed<T> &operator=(const RPUCudaPulsed<T> &);
  RPUCudaPulsed(RPUCudaPulsed<T> &&);
  RPUCudaPulsed<T> &operator=(RPUCudaPulsed<T> &&);

  friend void swap(RPUCudaPulsed<T> &a, RPUCudaPulsed<T> &b) noexcept {

    using std::swap;
    swap(static_cast<RPUCudaSimple<T> &>(a), static_cast<RPUCudaSimple<T> &>(b));

    swap(a.rpu_device_, b.rpu_device_);
    swap(a.rpucuda_device_, b.rpucuda_device_);
    swap(a.par_, b.par_);

    swap(a.dev_f_x_vector_inc1_, b.dev_f_x_vector_inc1_);
    swap(a.dev_f_d_vector_inc1_, b.dev_f_d_vector_inc1_);

    swap(a.dev_b_x_vector_inc1_, b.dev_b_x_vector_inc1_);
    swap(a.dev_b_d_vector_inc1_, b.dev_b_d_vector_inc1_);

    swap(a.dev_up_x_vector_inc1_, b.dev_up_x_vector_inc1_);
    swap(a.dev_up_d_vector_inc1_, b.dev_up_d_vector_inc1_);

    swap(a.dev_batch_buffer_d_size_, b.dev_batch_buffer_d_size_);
    swap(a.dev_batch_buffer_x_size_, b.dev_batch_buffer_x_size_);

    swap(a.size_, b.size_);

    swap(a.f_iom_, b.f_iom_);
    swap(a.b_iom_, b.b_iom_);
    swap(a.up_pwu_, b.up_pwu_);
  }

  void printToStream(std::stringstream &ss) const override;
  void printParametersToStream(std::stringstream &ss) const override {
    getMetaPar().printToStream(ss);
    if (getMetaPar().up.pulse_type != PulseType::None) {
      rpu_device_->printToStream(ss);
    }
  };

protected:
  void forwardVector(const T *x_input, T *d_output, int x_inc, int d_inc, bool is_test) override;
  void backwardVector(const T *d_input, T *x_output, int d_inc, int x_inc) override;
  void updateVector(const T *x_input, const T *d_input, int x_inc, int d_inc) override;

  // matrix methods (to avoid looped version in Abstract)
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

public:
  // indexed interface optimized
  void forwardIndexed(
      const T *X_input,
      T *D_output,
      int total_input_size,
      int m_batch,
      int dim3,
      bool trans,
      bool is_test) override;
  void backwardIndexed(
      const T *D_input, T *X_output, int total_output_size, int m_batch, int dim3, bool trans)
      override;
  void updateIndexed(
      const T *X_input, const T *D_input, int total_input_size, int m_batch, int dim3, bool trans)
      override;

  void decayWeights(bool bias_no_decay) override;
  void decayWeights(T alpha, bool bias_no_decay) override;
  void diffuseWeights() override;

  void resetCols(int start_col, int n_cols, T reset_prob) override;

  void setLearningRate(T rate) override;

  void getWeightsReal(T *weightsptr) override;
  void setWeightsReal(const T *weightsptr, int n_loops = 25) override;
  void setWeights(const T *weightsptr) override;

  void applyWeightUpdate(T *dw_and_current_weights_out) override;

  void getDeviceParameterNames(std::vector<std::string> &names) const override;
  void getDeviceParameter(std::vector<T *> &data_ptrs) const override;
  void setDeviceParameter(const std::vector<T *> &data_ptrs) override;

  int getHiddenUpdateIdx() const override;
  void setHiddenUpdateIdx(int idx) override;

  void populateParameter(PulsedMetaParameter<T> *p, PulsedRPUDeviceMetaParameter<T> *dp);

  void printRPUParameter(int x_count, int d_count) const;

  // CAUTION: make sure to call Finish before weight usage !! Not done in the code
  void finishUpdateCalculations() override;
  void makeUpdateAsync() override;

  // for debugging
  void getCountsDebug(uint32_t *x_counts, uint32_t *d_counts) {
    up_pwu_->getCountsDebug(x_counts, d_counts);
  };

  virtual const PulsedMetaParameter<T> &getMetaPar() const { return par_; };

  const AbstractRPUDeviceCuda<T> &getRPUDeviceCuda() { return *rpucuda_device_; };

protected:
  std::unique_ptr<AbstractRPUDevice<T>> rpu_device_ = nullptr;
  std::unique_ptr<AbstractRPUDeviceCuda<T>> rpucuda_device_ = nullptr;

  std::unique_ptr<InputOutputManager<T>> f_iom_ = nullptr;
  std::unique_ptr<InputOutputManager<T>> b_iom_ = nullptr;
  std::unique_ptr<PulsedWeightUpdater<T>> up_pwu_ = nullptr;

private:
  PulsedMetaParameter<T> par_;

  std::unique_ptr<CudaArray<T>> dev_decay_scale_ = nullptr;
  std::unique_ptr<CudaArray<T>> dev_diffusion_rate_ = nullptr;

  // forward
  std::unique_ptr<CudaArray<T>> dev_f_x_vector_inc1_ = nullptr;
  std::unique_ptr<CudaArray<T>> dev_f_d_vector_inc1_ = nullptr;

  // backward
  std::unique_ptr<CudaArray<T>> dev_b_x_vector_inc1_ = nullptr;
  std::unique_ptr<CudaArray<T>> dev_b_d_vector_inc1_ = nullptr;

  // update
  std::unique_ptr<CudaArray<T>> dev_up_x_vector_inc1_ = nullptr;
  std::unique_ptr<CudaArray<T>> dev_up_d_vector_inc1_ = nullptr;

  int size_ = 0;

  // shared forward/update
  std::shared_ptr<CudaArray<T>> dev_batch_buffer_x_size_ = nullptr;
  std::shared_ptr<CudaArray<T>> dev_batch_buffer_d_size_ = nullptr;

  template <typename InputIteratorT, typename OutputIteratorT>
  inline void forwardMatrixIterator(
      InputIteratorT X_input,
      OutputIteratorT D_output,
      int m_batch,
      bool x_trans,
      bool d_trans,
      bool is_test);
  template <typename InputIteratorT, typename OutputIteratorT>
  inline void backwardMatrixIterator(
      InputIteratorT D_input, OutputIteratorT X_output, int m_batch, bool d_trans, bool x_trans);
  template <typename XInputIteratorT, typename DInputIteratorT>
  inline void updateMatrixIterator(
      XInputIteratorT X_input, DInputIteratorT D_input, int m_batch, bool x_trans, bool d_trans);

  void initialize();
  void checkBatchBuffers(const int m_batch);
  void initFrom(RPUPulsed<T> &rpu);
};

} // namespace RPU
