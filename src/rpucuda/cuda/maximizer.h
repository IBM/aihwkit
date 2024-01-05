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

namespace RPU {

struct CustomMaxAbs {
  template <typename T> __device__ __forceinline__ T operator()(const T &a, const T &b) const {
    T bval = b >= (T)0.0 ? b : -b;
    T aval = a >= (T)0.0 ? a : -a;
    return (bval > aval) ? bval : aval;
  }
};

template <typename T> class Maximizer {

public:
  explicit Maximizer(CudaContextPtr c, int size, bool abs_if = true);

  /* computes the max/amax values */
  template <typename InputIteratorT>
  void compute(InputIteratorT dev_input, int m_batch = 1, bool trans = false);

  /* sets the computed max values to zero below thres. Caution: This
     is in-place. does not check whether compute was called. */
  void setZeroBelow(T thres);
  void saturateAbove(T thres);

  inline T *getMaxValues() { return dev_max_values_->getData(); };
  inline void copyMaxValuesToHost(T *dest) { dev_max_values_->copyTo(dest); };

  void printMaxValues() { dev_max_values_->printValues(); };

private:
  void initializeBatchBuffer(int m_batch);

  int buffer_m_batch_ = 0;
  int size_ = 0;
  CudaContextPtr context_ = nullptr;

  CustomMaxAbs max_abs_op_;
  bool abs_if_ = true;

  std::unique_ptr<CudaArray<T>> dev_max_values_ = nullptr;
  std::unique_ptr<CudaArray<T>> dev_max_values0_ = nullptr;
  std::unique_ptr<CudaArray<int>> dev_offsets_ = nullptr;

  std::unique_ptr<CudaArray<char>> dev_v_temp_storage_ = nullptr;
  std::unique_ptr<CudaArray<char>> dev_m_temp_storage_ = nullptr;
};

namespace test_helper {
template <typename T>
void debugMaxBatched(const T *indata, int size, int m_batch, bool trans, T *max_values);

}

} // namespace RPU
