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

#include "cuda_util.h"
#include "maximizer.h"
#include "rpu_pulsed_meta_parameter.h"

namespace RPU {

struct CustomNSum {
  template <typename T> __device__ __forceinline__ T operator()(const T &a, const T &b) const {
    T bval = b > (T)0.0 ? (T)0.0 : b;
    T aval = a > (T)0.0 ? (T)0.0 : a;
    return bval + aval;
  }
};

struct CustomPSum {
  template <typename T> __device__ __forceinline__ T operator()(const T &a, const T &b) const {
    T bval = b < (T)0.0 ? (T)0.0 : b;
    T aval = a < (T)0.0 ? (T)0.0 : a;
    return bval + aval;
  }
};

template <typename T> class NoiseManager {

public:
  explicit NoiseManager(CudaContextPtr c, int size);

  /* computes the scale values */
  template <typename InputIteratorT>
  void compute(
      InputIteratorT dev_input,
      const NoiseManagementType &nm_type,
      const IOMetaParameter<T> &io,
      int m_batch = 1,
      bool trans = false,
      bool is_test = false);

  // debugging functions
  inline void copyScaleValuesToHost(T *dest) const { dev_scale_values_->copyTo(dest); };
  void printScaleValues() const { dev_scale_values_->printValues(); };
  void printAvgAbsMax() const { dev_ravg_scale_value_->printValues(); };
  void printAbsMaxValues() const { amaximizer_->printMaxValues(); };

  /* sets the computed max values to zero below thres. Caution: This
     is in-place. does not check whether compute was called. */
  T *getScaleValues() const;

  T getAverageAbsMax() const;
  void setAverageAbsMax(T value);
  void dumpExtra(RPU::state_t &extra, const std::string prefix);
  void loadExtra(const RPU::state_t &extra, const std::string prefix, bool strict);

private:
  template <typename InputIteratorT>
  void computeNPSum(InputIteratorT dev_input, int m_batch = 1, bool trans = false);

  void initializeBatchBuffer(int m_batch);

  std::unique_ptr<CudaArray<T>> dev_scale_values_ = nullptr;
  std::unique_ptr<CudaArray<T>> dev_ravg_scale_value_ = nullptr;
  std::unique_ptr<CudaArray<T>> dev_avgmax_value_ = nullptr;
  std::unique_ptr<CudaArray<T>> dev_nzeros_value_ = nullptr;
  std::unique_ptr<Maximizer<T>> amaximizer_ = nullptr;
  std::unique_ptr<Maximizer<T>> maximizer_ = nullptr;

  NoiseManagementType nm_type_ = NoiseManagementType::None;
  int buffer_m_batch_ = 0;
  int last_m_batch_ = 0;
  int size_ = 0;
  CudaContextPtr context_ = nullptr;
  bool const_set_if_ = false;
  bool ravg_initialized_ = false;

  CustomPSum psum_op_;
  CustomNSum nsum_op_;

  std::unique_ptr<CudaArray<T>> dev_psum_values_ = nullptr;
  std::unique_ptr<CudaArray<T>> dev_psum_values0_ = nullptr;
  std::unique_ptr<CudaArray<T>> dev_nsum_values_ = nullptr;
  std::unique_ptr<CudaArray<T>> dev_nsum_values0_ = nullptr;

  std::unique_ptr<CudaArray<int>> dev_offsets_ = nullptr;

  std::unique_ptr<CudaArray<char>> dev_v_temp_storage_ = nullptr;
  std::unique_ptr<CudaArray<char>> dev_m_temp_storage_ = nullptr;
  std::unique_ptr<CudaArray<char>> dev_a_temp_storage_ = nullptr;
};

} // namespace RPU
