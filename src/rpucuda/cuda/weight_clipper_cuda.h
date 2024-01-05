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
#include "weight_clipper.h"

namespace RPU {

template <typename T> class WeightClipperCuda {

public:
  explicit WeightClipperCuda(CudaContextPtr context, int x_size, int d_size);
  WeightClipperCuda(){};

  void apply(T *weights, const WeightClipParameter &wclpar);

  void dumpExtra(RPU::state_t &extra, const std::string prefix){};
  void loadExtra(const RPU::state_t &extra, const std::string prefix, bool strict){};

private:
  CudaContextPtr context_ = nullptr;
  int x_size_ = 0;
  int d_size_ = 0;
  int size_ = 0;
  size_t temp_storage_bytes_ = 0;

  // no need to copy
  std::unique_ptr<Maximizer<T>> row_amaximizer_ = nullptr;
  std::unique_ptr<CudaArray<T>> dev_std_value_ = nullptr;
  std::unique_ptr<CudaArray<T>> dev_sum_value_ = nullptr;
  std::unique_ptr<CudaArray<char>> dev_temp_storage_ = nullptr;
};

} // namespace RPU
