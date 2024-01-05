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
#include <memory>
#include <mutex>

#if defined RPU_TORCH_CUDA_BUFFERS
#include <ATen/ATen.h>
#define RPU_CUB_NS_QUALIFIER ::cub::
#endif

namespace RPU {

class CudaContext;
typedef CudaContext *CudaContextPtr;
template <typename T> class CudaArray;

template <typename T> class CudaBuffer {
public:
  CudaBuffer(){};
  CudaBuffer(const CudaBuffer<T> &);
  CudaBuffer &operator=(const CudaBuffer<T> &);
  CudaBuffer(CudaBuffer<T> &&);
  CudaBuffer<T> &operator=(CudaBuffer<T> &&);
  ~CudaBuffer() = default;

  friend void swap(CudaBuffer<T> &a, CudaBuffer<T> &b) noexcept {
    using std::swap;
    const std::lock_guard<std::recursive_mutex> locka(a.mutex_);
    const std::lock_guard<std::recursive_mutex> lockb(b.mutex_);
    swap(a.buffer_, b.buffer_);
#if defined RPU_TORCH_CUDA_BUFFERS
    swap(a.tmp_context_, b.tmp_context_);
#endif
  }

  T *get(CudaContextPtr context, int size);
  void release();

  void print(int size) const;

private:
#if defined RPU_TORCH_CUDA_BUFFERS
  at::Tensor buffer_;
  CudaContextPtr tmp_context_ = nullptr;
#else
  std::unique_ptr<CudaArray<T>> buffer_ = nullptr;
#endif
  std::recursive_mutex mutex_;
};
} // namespace RPU
