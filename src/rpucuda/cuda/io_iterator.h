/**
 * (C) Copyright 2020 IBM. All Rights Reserved.
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

#include "cuda_math_util.h"

namespace RPU {

struct BatchSkipper {
  explicit BatchSkipper(int skip) : skip_(skip){};

  __device__ __forceinline__ int operator()(const int &a) const { return int(a * skip_); }
  int skip_ = 1;
};

template <typename T> struct Power2 {
  __device__ __forceinline__ T operator()(const T &a) const { return T(a * a); }
};

template <typename T> class NegateInputIterator {

public:
  typedef NegateInputIterator<T> self_type;
  typedef int difference_type;
  typedef T value_type;
  typedef T *pointer;
  typedef T reference;
  typedef std::input_iterator_tag iterator_category;

  __host__ __device__ __forceinline__ NegateInputIterator(const T *data) { data_ = data; }
  __host__ __device__ __forceinline__ T operator[](int idx) const { return -data_[idx]; }

  __host__ __device__ __forceinline__ self_type operator+(int n) const {
    self_type retval(data_ + n);
    return retval;
  }
  const T *data_;
};

// Iterators
template <typename T> class IndexReaderInputIterator {

public:
  typedef IndexReaderInputIterator<T> self_type;
  typedef int difference_type;
  typedef T value_type;
  typedef T *pointer;
  typedef T reference;
  typedef std::input_iterator_tag iterator_category;

  __host__ __device__ __forceinline__ IndexReaderInputIterator(
      const T *data, const int *indices, int input_matrix_size, int M, int offset = 0) {
    data_ = data;
    M_ = M;                                 // xsize*m_batch [output_matrix_size]
    input_matrix_size_ = input_matrix_size; // dimension to repmat
    indices_ = indices;
    offset_ = offset;
  }
  __host__ __device__ __forceinline__ T operator[](int idx_in) const {
    int idx = idx_in + offset_;
    int j = indices_[idx % M_];
    return (j <= 1) ? (T)j : data_[(j - 2) + idx / M_ * input_matrix_size_];
  }

  __host__ __device__ __forceinline__ self_type operator+(int n) const {
    self_type retval(data_, indices_, input_matrix_size_, M_, offset_ + n);
    return retval;
  }

  const T *data_;
  const int *indices_;
  int M_, input_matrix_size_, offset_;
};

template <typename T> class IndexReaderTransInputIterator {

public:
  typedef IndexReaderTransInputIterator<T> self_type;
  typedef int difference_type;
  typedef T value_type;
  typedef T *pointer;
  typedef T reference;
  typedef std::input_iterator_tag iterator_category;

  __host__ __device__ __forceinline__ IndexReaderTransInputIterator(
      const T *data,
      const int *indices,
      int input_matrix_size,
      int m,
      int M,
      int L,
      int offset = 0) {
    data_ = data;
    M_ = M; // m_batch*[x_]size
    input_matrix_size_ = input_matrix_size;
    indices_ = indices;
    m_ = m; // m_batch
    L_ = L; // m_batch*dim3
    offset_ = offset;
  }

  __host__ __device__ __forceinline__ T operator[](int idx_in) const {
    // here we additioanlly permute 132
    int idx = idx_in + offset_;
    int i = (idx % L_) / m_ * M_ + (idx % m_) + idx / L_ * m_;
    int j = indices_[i % M_];
    return (j <= 1) ? (T)j : data_[(j - 2) + i / M_ * input_matrix_size_];
  }

  __host__ __device__ __forceinline__ self_type operator+(int n) const {
    self_type retval(data_, indices_, input_matrix_size_, m_, M_, L_, offset_ + n);
    return retval;
  }

  const T *data_;
  const int *indices_; // need to be T as well...
  int M_, L_, m_, input_matrix_size_, offset_;
};

template <typename T> class PermuterTransInputIterator {

public:
  typedef PermuterTransInputIterator<T> self_type;
  typedef int difference_type;
  typedef T value_type;
  typedef T *pointer;
  typedef T reference;
  typedef std::input_iterator_tag iterator_category;

  __host__ __device__ __forceinline__
  PermuterTransInputIterator(const T *data, int m, int M, int L, int offset = 0) {
    data_ = data;
    M_ = M; // m_batch*[xd_]size // is already for reverse because of output
    m_ = m; // m_batch
    L_ = L; // m_batch*dim3
    offset_ = offset;
  }

  __host__ __device__ __forceinline__ T operator[](int idx_in) const {
    int idx = idx_in + offset_;
    int i = (idx % L_) / m_ * M_ + (idx % m_) + idx / L_ * m_;
    return data_[i];
  }

  __host__ __device__ __forceinline__ self_type operator+(int n) const {
    self_type retval(data_, m_, M_, L_, offset_ + n);
    return retval;
  }

  const T *data_;
  int M_, L_, m_, offset_;
};

template <typename T> class PermuterTransOutputIterator {

private:
  // Proxy object
  struct Reference {
    T *ptr_;

    __host__ __device__ __forceinline__ Reference(T *ptr) : ptr_(ptr) {}

    __device__ __forceinline__ T operator=(T val) {
      *ptr_ = val;
      return val;
    }
  };

public:
  __host__ __device__ __forceinline__ PermuterTransOutputIterator(T *data, int m, int M, int L) {
    data_ = data;
    M_ = M; // m_batch*[xd_]size // is already for reverse because of output
    m_ = m; // m_batch
    L_ = L; // m_batch*dim3
  }

  __host__ __device__ __forceinline__ Reference operator[](int idx) const {
    int i = (idx % L_) / m_ * M_ + (idx % m_) + idx / L_ * m_;
    return Reference(data_ + i);
  }

  T *data_;
  int M_, L_, m_;
};

template <typename T> class IndexReaderOutputIterator {

private:
  // Proxy object
  struct Reference {
    T *ptr_;

    __host__ __device__ __forceinline__ Reference(T *ptr) : ptr_(ptr) {}

    __device__ __forceinline__ T operator=(T val) {
      if (ptr_ != nullptr)
        atomicAdd(ptr_, val);
      return val;
    }
  };

public:
  __host__ __device__ __forceinline__
  IndexReaderOutputIterator(T *data, const int *indices, int output_matrix_size, int M) {
    data_ = data;
    M_ = M;
    output_matrix_size_ = output_matrix_size;
    indices_ = indices;
  }

  __host__ __device__ __forceinline__ Reference operator[](int idx_in) const {

    int idx = idx_in;
    int j = indices_[idx % M_];
    if (j <= 1)
      return Reference(nullptr);
    else
      return Reference(data_ + ((j - 2) + idx / M_ * output_matrix_size_));
  }

  T *data_;
  const int *indices_;
  int M_, output_matrix_size_;
};

template <typename T> class IndexReaderTransOutputIterator {

private:
  // Proxy object
  struct Reference {
    T *ptr_;

    __host__ __device__ __forceinline__ Reference(T *ptr) : ptr_(ptr) {}

    __device__ __forceinline__ T operator=(T val) {
      if (ptr_ != nullptr)
        atomicAdd(ptr_, val);
      return val;
    }
  };

public:
  __host__ __device__ __forceinline__ IndexReaderTransOutputIterator(
      T *data, const int *indices, int output_matrix_size, int m, int M, int L) {
    data_ = data;
    M_ = M; // m_batch*[x_]size
    output_matrix_size_ = output_matrix_size;
    indices_ = indices;
    m_ = m; // m_batch
    L_ = L; // m_batch*dim3
  }

  __host__ __device__ __forceinline__ Reference operator[](int idx_in) const {

    // here we additionally permute 132
    int idx = idx_in;
    int i = (idx % L_) / m_ * M_ + (idx % m_) + idx / L_ * m_;
    int j = indices_[i % M_];
    if (j <= 1)
      return Reference(nullptr);
    else
      return Reference(data_ + ((j - 2) + i / M_ * output_matrix_size_));
  }

  T *data_;
  const int *indices_;
  int M_, L_, m_, output_matrix_size_;
};

} // namespace RPU
