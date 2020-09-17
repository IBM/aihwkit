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

#include "cuda_math_util.h"
#include "cuda_util.h"
#include "maximizer.h"
#include "noise_manager.h"
#include "rpu_pulsed.h"

namespace RPU {

template <typename T> class InputOutputManager {

public:
  explicit InputOutputManager(CudaContext *c, int in_size, int out_size);
  ~InputOutputManager();

  void setSharedBuffer(
      int m_batch,
      std::shared_ptr<CudaArray<T>> in_buffer = nullptr,
      std::shared_ptr<CudaArray<T>> out_buffer = nullptr);

  template <typename InputIteratorT>
  void initWithInput(
      InputIteratorT dev_input,
      const IOMetaParameter<T> &io,
      const int m_batch = 1,
      const bool trans = false,
      const T add_out_scale = 1.0,
      const bool is_test = false); // additional output scaling (takes add_out_scale*io->out_scale)

  template <typename InputIteratorT> int applyToInput(InputIteratorT dev_input);

  template <typename OutputIteratorT>
  bool applyToOutput(OutputIteratorT dev_output, const T *dev_weights, const bool trans = false);

  T *getInBuffer();
  T *getOutBuffer();

  // for testing
  void copyTempArrayToHost(T *host_array) { dev_input_applied_->copyTo(host_array); }
  void copyExceededArrayToHost(int *host_array) { dev_bound_exceeded_->copyTo(host_array); }

  bool applyToOutputInPlace(T *dev_output, const T *dev_weights, const bool trans = false) {
    dev_output_applied_->assignFromDevice(dev_output);
    return applyToOutput(dev_output, dev_weights, trans);
  }

  NoiseManager<T> *getNM() { return &*noise_manager_; };

private:
  void applyOutputWeightNoise(const bool out_trans);
  void applyOutputNonIdealities(const T *dev_weights, const bool out_trans);

  void initializeBatchBuffer(int m_batch);

  template <typename InputIteratorT> int applyToInputWithBoundManagement(InputIteratorT dev_input);

  template <typename OutputIteratorT>
  bool applyToOutputWithBoundManagement(OutputIteratorT dev_output, const bool trans);

  CudaContext *context_ = nullptr;
  int in_size_ = 0;
  int out_size_ = 0;
  std::unique_ptr<NoiseManager<T>> noise_manager_ = nullptr;

  int buffer_m_batch_ = 0;
  int temp_m_batch_ = 0;
  bool temp_trans_ = false;
  T temp_out_scale_ = 1.0;
  bool temp_is_test_ = false;

  bool bm_with_selecting_ = false;
  int m_batch_selected_ = 0;
  bool currently_selecting_bidx_ = false;

  T reduction_due_to_bound_management_ = 0;
  T bound_management_factor_ = 2.0;
  int bound_management_round_ = 0;
  int *h_exceeded_ = nullptr;

  int nthreads_ = 0;
  int nblocks_batch_max_ = 0;
  int nblocks_om_ = 0;
  int nblocks_im_ = 0;
  int nblocks_om_batch_ = 0;
  int nblocks_im_batch_ = 0;

  const IOMetaParameter<T> *io_;

  std::shared_ptr<CudaArray<T>> dev_input_applied_ = nullptr;
  std::shared_ptr<CudaArray<T>> dev_output_applied_ = nullptr;
  std::unique_ptr<CudaArray<float>> dev_scale_values_ = nullptr;
  std::unique_ptr<CudaArray<int>> dev_bound_exceeded_ = nullptr;
  std::unique_ptr<CudaArray<int>> dev_any_exceeded_ = nullptr;

  std::unique_ptr<CudaArray<char>> dev_flagged_temp_storage_ = nullptr;
  std::unique_ptr<CudaArray<int>> dev_selected_bidx_ = nullptr;
  std::unique_ptr<CudaArray<int>> dev_selected_m_batch_ = nullptr;

  std::unique_ptr<CudaArray<T>> dev_wnoise_buffer_ = nullptr;
  std::unique_ptr<CudaArray<T>> dev_wnoise_ones_ = nullptr;
};

} // namespace RPU
