/**
 * (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
 *
 * Licensed under the MIT license. See LICENSE file in the project root for details.
 */

#pragma once

#include "cuda_util.h"
#include "maximizer.h"
#include "weight_modifier.h"

namespace RPU {

template <typename T> class WeightModifierCuda {

public:
  explicit WeightModifierCuda(CudaContextPtr context, int x_size, int d_size);
  WeightModifierCuda(){};

  void apply(T *new_weights, const T *weights, const WeightModifierParameter<T> &wmpar);

  inline bool enableDuringTest() { return enable_during_test_; };

  void dumpExtra(RPU::state_t &extra, const std::string prefix);
  void loadExtra(const RPU::state_t &extra, const std::string prefix, bool strict);

private:
  CudaContextPtr context_ = nullptr;
  int x_size_ = 0;
  int d_size_ = 0;
  int size_ = 0;
  bool enable_during_test_ = false;
  // no need to copy
  std::unique_ptr<Maximizer<T>> amaximizer_ = nullptr;
  std::unique_ptr<Maximizer<T>> row_amaximizer_ = nullptr;
  std::unique_ptr<Maximizer<T>> row_maximizer_ = nullptr;
  std::unique_ptr<Maximizer<T>> row_minimizer_ = nullptr;
  std::vector<T> coeffs_;
  std::unique_ptr<CudaArray<T>> dev_coeffs_ = nullptr;
};

} // namespace RPU
