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

#pragma once

#include "cuda_math_util.h"
#include "cuda_util.h"
#include "maximizer.h"
#include "noise_manager.h"
#include "rpu_pulsed.h"

namespace RPU {

template <typename T> class InputOutputManager {

public:
  explicit InputOutputManager(CudaContextPtr c, int in_size, int out_size);
  ~InputOutputManager();

  /* Note the in_size is usually fixed to that given during
     instantiation. However, for some configuration (no noise
     managment / no bound management) it can be given as
     input. Otherwise a change will result in an error.*/

  template <typename InputIteratorT>
  void initWithInput(
      InputIteratorT dev_input,
      const IOMetaParameter<T> &io,
      const int in_size,
      const int m_batch = 1,
      const bool trans = false,
      const T add_out_scale = 1.0,
      const bool is_test = false); // additional output scaling (takes add_out_scale*io->out_scale)

  template <typename InputIteratorT> void applyToInput(InputIteratorT dev_input);

  template <typename OutputIteratorT>
  bool applyToOutput(
      OutputIteratorT dev_output, const bool trans = false, const bool with_out_noise = true);

  // only valid after init
  inline T *getInBuffer() const { return temp_input_applied_; };
  inline T *getOutBuffer() const { return temp_output_applied_; };

  // careful, no checks, mem not owned or deleted!
  inline void setInBuffer(T *in_buffer) { temp_input_applied_ = in_buffer; };
  inline void setOutBuffer(T *out_buffer) { temp_output_applied_ = out_buffer; };

  // needs to be called at end to release buffer
  void releaseBuffer();

  // for testing
  void copyTempArrayToHost(T *host_array) {
    CudaArray<T> tmp(context_, getMBatch() * getInSize());
    tmp.assignFromDevice(getInBuffer());
    tmp.copyTo(host_array);
    context_->synchronize();
  }
  void copyExceededArrayToHost(int *host_array) { dev_bound_exceeded_->copyTo(host_array); }

  bool
  applyToOutputInPlace(T *dev_output, const bool trans = false, const bool with_out_noise = true) {
    RPU::math::copy<T>(context_, getMBatch() * getOutSize(), dev_output, 1, getOutBuffer(), 1);
    return applyToOutput(dev_output, trans, with_out_noise);
  }
  void dumpExtra(RPU::state_t &extra, const std::string prefix);
  void loadExtra(const RPU::state_t &extra, const std::string prefix, bool strict);

  NoiseManager<T> *getNM() { return &*noise_manager_; };

  inline int getInSize() const { return temp_in_size_; };
  inline int getOutSize() const { return out_size_; };
  inline bool getInTrans() const { return temp_trans_; };
  inline int getMBatch() const { return temp_m_batch_; };
  inline T getOutScale() const { return temp_out_scale_; };
  inline CudaContextPtr getContext() const { return context_; };
  inline int getInBlocks() const {
    return MIN(context_->getNBlocks(temp_in_size_, nthreads_), nblocks_batch_max_);
  };
  inline int getInBlocksBatch(int m_batch) const {
    return MIN(nblocks_batch_max_, this->context_->getNBlocks(temp_in_size_ * m_batch, nthreads_));
  };
  inline int getOutBlocks() const {
    return MIN(context_->getNBlocks(out_size_, nthreads_), nblocks_batch_max_);
  };
  inline int getOutBlocksBatch(int m_batch) const {
    return MIN(nblocks_batch_max_, this->context_->getNBlocks(out_size_ * m_batch, nthreads_));
  };
  inline int getNThreads() const { return nthreads_; };
  inline const IOMetaParameter<T> &getIO() const { return *io_; };

private:
  void applyOutputWeightNoise(const bool out_trans);
  void applyOutputNonIdealities(const T *dev_weights, const bool out_trans);
  void applyOutputPCMReadNoise(const T *dev_weights, const bool out_trans);
  void applyIrDrop(const T *dev_weights, const bool out_trans);
  void initializeBatchBuffer(int m_batch);
  void setInSize(int in_size);
  void setOutSize(int out_size);

  template <typename InputIteratorT> void applyToInputWithBoundManagement(InputIteratorT dev_input);

  template <typename OutputIteratorT>
  bool applyToOutputWithBoundManagement(
      OutputIteratorT dev_output, const bool trans, const bool with_out_noise = true);

  CudaContextPtr context_ = nullptr;
  int in_size_ = 0;
  int out_size_ = 0;
  std::unique_ptr<NoiseManager<T>> noise_manager_ = nullptr;
  std::unique_ptr<Maximizer<T>> output_maximizer_ = nullptr;

  int buffer_m_batch_ = 0;
  int temp_m_batch_ = 0;
  bool temp_trans_ = false;
  T temp_out_scale_ = 1.0;
  bool temp_is_test_ = false;
  int temp_in_size_ = 0;

  T reduction_due_to_bound_management_ = 0;
  T bound_management_factor_ = 2.0;
  int bound_management_round_ = 0;
  int *h_exceeded_ = nullptr;

  int nthreads_ = 0;
  int nblocks_batch_max_ = 0;

  const IOMetaParameter<T> *io_;
  T *temp_input_applied_ = nullptr;
  T *temp_output_applied_ = nullptr;

  std::unique_ptr<CudaArray<T>> dev_scale_values_ = nullptr;
  std::unique_ptr<CudaArray<int>> dev_bound_exceeded_ = nullptr;
  std::unique_ptr<CudaArray<int>> dev_any_exceeded_ = nullptr;
};

} // namespace RPU
