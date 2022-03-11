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

#include "cuda_util.h"
#include "weight_drifter.h"

namespace RPU {

template <typename T> class WeightDrifterCuda {

public:
  explicit WeightDrifterCuda(CudaContext *context, int size);
  explicit WeightDrifterCuda(CudaContext *, const WeightDrifter<T> &wd, int x_size, int d_size);
  WeightDrifterCuda(){};
  virtual ~WeightDrifterCuda() = default;

  WeightDrifterCuda(const WeightDrifterCuda<T> &); // = default;
  WeightDrifterCuda<T> &operator=(const WeightDrifterCuda<T> &) = default;
  WeightDrifterCuda(WeightDrifterCuda<T> &&) = default;
  WeightDrifterCuda<T> &operator=(WeightDrifterCuda<T> &&) = default;

  void apply(T *weights, T time_since_last_call);

  inline bool isActive() { return active_; };

  void saturate(T *weights, float *dev_4params);
  const T *getNu() const { return dev_nu_ == nullptr ? nullptr : dev_nu_->getDataConst(); };

protected:
  CudaContext *context_ = nullptr;
  int size_ = 0;
  int max_size_ = 0;
  bool active_ = false;
  T current_t_ = 0.0;

  DriftParameter<T> par_;

  std::unique_ptr<CudaArray<T>> dev_previous_weights_ = nullptr;
  std::unique_ptr<CudaArray<T>> dev_w0_ = nullptr;
  std::unique_ptr<CudaArray<T>> dev_t_ = nullptr;
  std::unique_ptr<CudaArray<T>> dev_nu_ = nullptr;

private:
  void initialize(const T *weights);
  void
  populateFrom(const WeightDrifter<T> &wd, int x_size, int d_size); // called during construction
};

} // namespace RPU
