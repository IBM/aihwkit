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
#include "maximizer.h"
#include "weight_remapper.h"

namespace RPU {

template <typename T> class WeightRemapperCuda {

public:
  explicit WeightRemapperCuda(CudaContextPtr context, int x_size, int d_size);
  WeightRemapperCuda(){};

  void apply(
      T *weights, T current_lr, const WeightRemapParameter &wrmpar, T *scales, T *biases = nullptr);

private:
  CudaContext *context_ = nullptr;
  int x_size_ = 0;
  int d_size_ = 0;
  int size_ = 0;
  int max_size_ = 0;

  // no need to copy
  std::unique_ptr<Maximizer<T>> amaximizer_ = nullptr;
  std::unique_ptr<Maximizer<T>> row_amaximizer_ = nullptr;
  std::unique_ptr<Maximizer<T>> scale_minimizer_ = nullptr;
  std::unique_ptr<CudaArray<T>> scale_buffer_ = nullptr;

  int getNBlocks(int nthreads);
};

} // namespace RPU
