/**
 * (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
 *
 * Licensed under the MIT license. See LICENSE file in the project root for details.
 */

#pragma once

#include "cuda_util.h"
#include "maximizer.h"
#include "weight_remapper.h"

namespace RPU {

template <typename T> class WeightRemapperCuda {

public:
  explicit WeightRemapperCuda(CudaContextPtr context, int x_size, int d_size);
  WeightRemapperCuda() {};

  void apply(
      T *weights,
      T current_lr,
      const WeightRemapParameter &wrmpar,
      T *scales,
      T *biases = nullptr,
      int *channel_exceded = nullptr);

  bool applySWA(
      T *swa_weights,
      T *weights,
      uint64_t iter,
      const WeightRemapParameter &wmpar,
      T current_lr,
      T *scales = nullptr,
      T *biases = nullptr,
      int *channel_exceded = nullptr);

  void dumpExtra(RPU::state_t &extra, const std::string prefix) {};
  void loadExtra(const RPU::state_t &extra, const std::string prefix, bool strict) {};

private:
  CudaContextPtr context_ = nullptr;
  int x_size_ = 0;
  int d_size_ = 0;
  int size_ = 0;
  int max_size_ = 0;

  // no need to copy
  std::unique_ptr<Maximizer<T>> amaximizer_ = nullptr;
  std::unique_ptr<Maximizer<T>> row_amaximizer_ = nullptr;
  std::unique_ptr<Maximizer<T>> row_maximizer_ = nullptr;
  std::unique_ptr<Maximizer<T>> row_minimizer_ = nullptr;
  std::unique_ptr<Maximizer<T>> scale_minimizer_ = nullptr;
  std::unique_ptr<CudaArray<T>> bias_buffer_ = nullptr;
  std::unique_ptr<CudaArray<T>> scale_buffer_ = nullptr;
  std::unique_ptr<CudaArray<T>> msqr_values_ = nullptr;
  std::unique_ptr<CudaArray<char>> dev_temp_storage_ = nullptr;

  int getNBlocks(int nthreads);
};

} // namespace RPU
