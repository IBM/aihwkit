#pragma once

#include "cuda_util.h"
#include "maximizer.h"
#include "weight_clipper.h"

namespace RPU {

template <typename T> class WeightClipperCuda {

public:
  explicit WeightClipperCuda(CudaContext *context, int x_size, int d_size);
  WeightClipperCuda(){};

  void apply(T *weights, const WeightClipParameter &wclpar);

private:
  CudaContext *context_ = nullptr;
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
