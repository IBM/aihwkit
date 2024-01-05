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

#include "cuda_buffer.h"
#include "cuda_util.h"

namespace RPU {

#if defined(RPU_TORCH_CUDA_BUFFERS)

template <typename T> void CudaBuffer<T>::print(int size) const {
  auto values = buffer_.cpu();

  int n = values.numel() > size ? size : values.numel();
  for (int i = 0; i < n; ++i) {
    std::cout << "[" << i << "]:" << values[i] << ", ";
  }
  if (n < values.numel()) {
    std::cout << "...";
  }
  std::cout << std::endl;
}

template <typename T> T *CudaBuffer<T>::get(CudaContextPtr c, int size) {
  mutex_.lock(); // need to be explicitely released to avoid multi-threading issues

  if (buffer_.numel() < size || c->getGPUId() != buffer_.device().index()) {
    // Build the buffers.
    std::vector<int64_t> dims{size};
    auto options = at::TensorOptions().device(at::kCUDA, c->getGPUId()).requires_grad(false);

#ifdef RPU_DEFINE_CUDA_HALF_ARRAY
    if (std::is_same<T, half_t>::value) {
      options = options.dtype(at::kHalf);
    }
#endif
    if (std::is_same<T, double>::value) {
      options = options.dtype(at::kDouble);
    } else {
      options = options.dtype(at::kFloat);
    }
    c->synchronize();
    buffer_ = at::empty(dims, options);
  }
  tmp_context_ = c;
#ifdef RPU_DEFINE_CUDA_HALF_ARRAY
  if (std::is_same<T, half_t>::value) {
    return reinterpret_cast<T *>(buffer_.template data_ptr<at::Half>());
  }
#endif
  return buffer_.template data_ptr<T>();
}

// copy constructor
template <typename T> CudaBuffer<T>::CudaBuffer(const CudaBuffer<T> &other) {
  if (other.buffer_.numel() > 0) {
    buffer_ = other.buffer_;
  }
}

template <typename T> void CudaBuffer<T>::release() {
  // explicitly delete to save memory (memory is cached in torch)
  if (tmp_context_ != nullptr) {
    tmp_context_->synchronize();
    buffer_ = at::empty(std::vector<int64_t>{});
    tmp_context_ = nullptr;
  }
  mutex_.unlock();
}

// copy assignment
template <typename T> CudaBuffer<T> &CudaBuffer<T>::operator=(const CudaBuffer &other) {
  CudaBuffer tmp(other);
  swap(*this, tmp);
  if (tmp.tmp_context_ != nullptr) {
    tmp_context_->synchronize();
  }
  return *this;
}

#else

template <typename T> void CudaBuffer<T>::print(int size) const {
  if (buffer_ != nullptr) {
    buffer_->printValues(size);
  }
}

template <typename T> T *CudaBuffer<T>::get(CudaContextPtr c, int size) {
  mutex_.lock(); // need to be explicitely released to avoid multi-threading issues
  if (buffer_ == nullptr || buffer_->getSize() < size || &*(buffer_->getContext()) != &*c) {
    if (buffer_ != nullptr) {
      buffer_->synchronize();
    }
    buffer_ = RPU::make_unique<CudaArray<T>>(c, size);
    c->synchronize();
  }
  return buffer_->getData();
}

// copy constructor
template <typename T> CudaBuffer<T>::CudaBuffer(const CudaBuffer<T> &other) {
  if (other.buffer_ != nullptr) {
    buffer_ = RPU::make_unique<CudaArray<T>>(*other.buffer_);
    buffer_->synchronize();
  }
}

template <typename T> void CudaBuffer<T>::release() { mutex_.unlock(); }

// copy assignment
template <typename T> CudaBuffer<T> &CudaBuffer<T>::operator=(const CudaBuffer &other) {
  CudaBuffer tmp(other);
  swap(*this, tmp);
  if (tmp.buffer_ != nullptr) {
    tmp.buffer_->synchronize();
  }
  return *this;
}

#endif

// move constructor
template <typename T> CudaBuffer<T>::CudaBuffer(CudaBuffer<T> &&other) {
  { const std::lock_guard<std::recursive_mutex> lock(other.mutex_); }
  *this = std::move(other);
}

// move assignment
template <typename T> CudaBuffer<T> &CudaBuffer<T>::operator=(CudaBuffer<T> &&other) {

  const std::lock_guard<std::recursive_mutex> lock(other.mutex_);
  buffer_ = std::move(other.buffer_);
  return *this;
}

template class CudaBuffer<float>;
#ifdef RPU_USE_DOUBLE
template class CudaBuffer<double>;
#endif
#ifdef RPU_USE_FP16
template class CudaBuffer<half_t>;
#endif

} // namespace RPU
