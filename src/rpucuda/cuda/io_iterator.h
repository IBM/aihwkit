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

#include "cuda_math_util.h"
#include <iterator>

namespace RPU {

struct BatchSkipper {
  explicit BatchSkipper(int skip) : skip_(skip){};

  __device__ __forceinline__ int operator()(const int &a) const { return int(a * skip_); }
  int skip_ = 1;
};

template <typename T> struct Power2 {
  __device__ __forceinline__ T operator()(const T &a) const { return T(a * a); }
};

template <typename T> class LogInputIterator {

public:
  typedef LogInputIterator<T> self_type;
  typedef int difference_type;
  typedef T value_type;
  typedef T *pointer;
  typedef T reference;
  typedef std::input_iterator_tag iterator_category;

  __host__ __device__ __forceinline__ LogInputIterator(const T *data) { data_ = data; }
  __host__ __device__ __forceinline__ T operator[](int idx) const {
    T x = data_[idx];
    return (x > (T)0.0) ? (T)__logf((T)x) : (T)0.0;
  }

  __host__ __device__ __forceinline__ self_type operator+(int n) const {
    self_type retval(data_ + n);
    return retval;
  }
  const T *data_;
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

template <typename T> class IndicatorInputIterator {
public:
  typedef IndicatorInputIterator<T> self_type;
  typedef int difference_type;
  typedef T value_type;
  typedef T *pointer;
  typedef T reference;
  typedef std::input_iterator_tag iterator_category;

  __host__ __device__ __forceinline__
  IndicatorInputIterator(const T *data, const T indicator, const T scale) {
    data_ = data;
    indicator_ = indicator;
    scale_ = scale;
  }
  __host__ __device__ __forceinline__ T operator[](int idx) const {
    return static_cast<T>(data_[idx] == indicator_) * scale_;
  }

  __host__ __device__ __forceinline__ self_type operator+(int n) const {
    self_type retval(data_ + n, indicator_, scale_);
    return retval;
  }
  const T *data_;
  T indicator_;
  T scale_;
};

template <typename T, typename DataT> class DiagInputIterator {
public:
  typedef DiagInputIterator<T, DataT> self_type;
  typedef int difference_type;
  typedef T value_type;
  typedef T *pointer;
  typedef T reference;
  typedef std::input_iterator_tag iterator_category;

  __host__ __device__ __forceinline__ DiagInputIterator(const DataT *data, int dim, int offset) {
    data_ = data;
    dim_ = dim;
    offset_ = offset;
  }

  __host__ __device__ __forceinline__ T operator[](int idx) const {
    int i = idx + offset_;
    return (i % dim_ == i / dim_) ? static_cast<T>(data_[idx / dim_]) : static_cast<T>(0);
  }
  __host__ __device__ __forceinline__ self_type operator+(int shift_n) const {
    self_type retval(data_, dim_, shift_n + offset_);
    return retval;
  }

  const DataT *data_;
  int dim_;
  int offset_;
};

template <typename T> class EyeInputIterator {
public:
  typedef EyeInputIterator<T> self_type;
  typedef int difference_type;
  typedef T value_type;
  typedef T *pointer;
  typedef T reference;
  typedef std::input_iterator_tag iterator_category;

  __host__ __device__ __forceinline__ EyeInputIterator(const int n, const int offset) {
    n_ = n;
    n2_ = n * n;
    offset_ = offset;
  }
  __host__ __device__ __forceinline__ T operator[](int idx) const {
    int i = (idx + offset_) % n2_;
    return static_cast<T>(i % n_ == i / n_);
  }

  __host__ __device__ __forceinline__ self_type operator+(int shift_n) const {
    self_type retval(n_, shift_n + offset_);
    return retval;
  }
  int n_, n2_;
  int offset_;
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

// *****  batch slice iterator
template <bool trans> // trans true
__host__ __device__ __forceinline__ int setM(int xd_size, int m_batch, int dim3);

template <> // trans true
__host__ __device__ __forceinline__ int setM<true>(int xd_size, int m_batch, int dim3) {
  return m_batch * dim3;
}
template <> // trans false
__host__ __device__ __forceinline__ int setM<false>(int xd_size, int m_batch, int dim3) {
  return m_batch * xd_size;
}

template <bool trans> // trans true
__host__ __device__ __forceinline__ void getNewIdx(
    int &new_idx,
    int &i_dim3,
    int idx,
    const int *batch_indices,
    int xd_size,
    int m_batch_slice,
    int m_batch,
    int M);

template <> // trans true
__host__ __device__ __forceinline__ void getNewIdx<true>(
    int &new_idx,
    int &i_dim3,
    int idx,
    const int *batch_indices,
    int xd_size,
    int m_batch_slice,
    int m_batch,
    int M) {
  // order of IDX is m_batch_slice x dim3 x xd_size
  // order of DATA is m_batch x xd_size x dim3

  i_dim3 = (idx % M) / m_batch_slice;
  int i_batch_slice = idx % m_batch_slice;
  int i_xd = idx / M;
  int batch_idx =
      batch_indices[i_batch_slice + m_batch * i_dim3]; // note: batch_indices are m_batch*dim3 and
                                                       // need to be given with the correct offset.
  new_idx = batch_idx + m_batch * i_xd;
};

template <> // trans false
__host__ __device__ __forceinline__ void getNewIdx<false>(
    int &new_idx,
    int &i_dim3,
    int idx,
    const int *batch_indices,
    int xd_size,
    int m_batch_slice,
    int m_batch,
    int M) {
  // order if IDX is xd_index x m_batch_slice x dim3
  // order of data is xd_index x m_batch * dim3

  i_dim3 = idx / M;
  int i_batch_slice = (idx % M) / xd_size;
  int i_xd = idx % xd_size;
  int batch_idx = batch_indices[i_batch_slice + m_batch * i_dim3];
  new_idx = batch_idx * xd_size + i_xd;
};

template <bool trans, typename T> class IndexReaderSliceInputIterator {

public:
  typedef IndexReaderSliceInputIterator<trans, T> self_type;
  typedef int difference_type;
  typedef T value_type;
  typedef T *pointer;
  typedef T reference;
  typedef std::input_iterator_tag iterator_category;

  __host__ __device__ __forceinline__ IndexReaderSliceInputIterator(
      const T *data,
      const int *indices,
      const int input_matrix_size,
      const int xd_size,
      const int m_batch,
      const int dim3,
      const int m_batch_slice,
      const int *batch_indices) {

    batch_indices_ = batch_indices; // for each dim3 differently. length: m_batch_slice*dim3
    data_ = data;
    M_ = setM<trans>(xd_size, m_batch_slice, dim3);
    input_matrix_size_ = input_matrix_size;
    xd_size_ = xd_size;
    m_batch_ = m_batch;
    m_batch_slice_ = m_batch_slice;
    indices_ = indices;
    offset_ = 0;
  };

  __host__ __device__ __forceinline__ IndexReaderSliceInputIterator(
      const T *data,
      const int *indices,
      const int input_matrix_size,
      const int xd_size,
      const int m_batch,
      const int M,
      const int m_batch_slice,
      const int *batch_indices,
      const int offset) {
    batch_indices_ = batch_indices;
    data_ = data;
    M_ = M;
    input_matrix_size_ = input_matrix_size;
    xd_size_ = xd_size;
    m_batch_ = m_batch;
    m_batch_slice_ = m_batch_slice;
    indices_ = indices;
    offset_ = offset;
  }

  __host__ __device__ __forceinline__ T operator[](int idx_in) const {
    int idx = idx_in + offset_;
    int new_idx, i_dim3;
    getNewIdx<trans>(new_idx, i_dim3, idx, batch_indices_, xd_size_, m_batch_slice_, m_batch_, M_);

    int j = indices_[new_idx];
    return (j <= 1) ? (T)j : data_[(j - 2) + i_dim3 * input_matrix_size_];
  };

  __host__ __device__ __forceinline__ self_type operator+(int n) const {
    self_type retval(
        data_, indices_, input_matrix_size_, xd_size_, m_batch_, M_, m_batch_slice_, batch_indices_,
        offset_ + n);
    return retval;
  };

  const T *data_;
  const int *indices_;
  const int *batch_indices_;
  int offset_, M_, input_matrix_size_, m_batch_, m_batch_slice_, xd_size_;
};

template <bool trans, typename T> class IndexReaderSliceOutputIterator {

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
  __host__ __device__ __forceinline__ IndexReaderSliceOutputIterator(
      T *data,
      const int *indices,
      const int output_matrix_size,
      const int xd_size,
      const int m_batch,
      const int dim3,
      const int m_batch_slice,
      const int *batch_indices) {
    batch_indices_ = batch_indices; // for each dim3 differently. should be new_m
    data_ = data;
    M_ = setM<trans>(xd_size, m_batch_slice, dim3);
    output_matrix_size_ = output_matrix_size;
    xd_size_ = xd_size;
    m_batch_ = m_batch;
    m_batch_slice_ = m_batch_slice;
    indices_ = indices;
  };

  __host__ __device__ __forceinline__ Reference operator[](int idx) const {

    int new_idx, i_dim3;
    getNewIdx<trans>(new_idx, i_dim3, idx, batch_indices_, xd_size_, m_batch_slice_, m_batch_, M_);

    int j = indices_[new_idx];

    if (j <= 1)
      return Reference(nullptr);
    else
      return Reference(data_ + ((j - 2) + i_dim3 * output_matrix_size_));
  };

  T *data_;
  const int *indices_;
  const int *batch_indices_;
  int M_, output_matrix_size_, m_batch_, m_batch_slice_, xd_size_;
};

template <bool trans, typename T> class SliceInputIterator {

public:
  typedef SliceInputIterator<trans, T> self_type;
  typedef int difference_type;
  typedef T value_type;
  typedef T *pointer;
  typedef T reference;
  typedef std::input_iterator_tag iterator_category;

  __host__ __device__ __forceinline__ SliceInputIterator(
      const T *data,
      const int xd_size,
      const int m_batch,
      const int dim3,
      const int m_batch_slice,
      const int *batch_indices) {

    batch_indices_ = batch_indices; // for each dim3 differently. length: m_batch_slice*dim3
    data_ = data;
    M_ = setM<trans>(xd_size, m_batch_slice, dim3);
    input_matrix_size_ = m_batch * xd_size;
    xd_size_ = xd_size;
    m_batch_ = m_batch;
    m_batch_slice_ = m_batch_slice;
    offset_ = 0;
  };

  __host__ __device__ __forceinline__ SliceInputIterator(
      const T *data,
      const int input_matrix_size,
      const int xd_size,
      const int m_batch,
      const int M,
      const int m_batch_slice,
      const int *batch_indices,
      const int offset) {
    batch_indices_ = batch_indices;
    data_ = data;
    M_ = M;
    input_matrix_size_ = input_matrix_size;
    xd_size_ = xd_size;
    m_batch_ = m_batch;
    m_batch_slice_ = m_batch_slice;
    offset_ = offset;
  }

  __host__ __device__ __forceinline__ T operator[](int idx_in) const {
    int idx = idx_in + offset_;
    int j, i_dim3;
    getNewIdx<trans>(j, i_dim3, idx, batch_indices_, xd_size_, m_batch_slice_, m_batch_, M_);

    return data_[j + i_dim3 * input_matrix_size_];
  };

  __host__ __device__ __forceinline__ self_type operator+(int n) const {
    self_type retval(
        data_, input_matrix_size_, xd_size_, m_batch_, M_, m_batch_slice_, batch_indices_,
        offset_ + n);
    return retval;
  };

  const T *data_;
  const int *batch_indices_;
  int offset_, M_, input_matrix_size_, m_batch_, m_batch_slice_, xd_size_;
};

template <bool trans, typename T> class SliceOutputIterator {

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
  __host__ __device__ __forceinline__ SliceOutputIterator(
      T *data,
      const int xd_size,
      const int m_batch,
      const int dim3,
      const int m_batch_slice,
      const int *batch_indices) {
    batch_indices_ = batch_indices; // for each dim3 differently. should be new_m
    data_ = data;
    M_ = setM<trans>(xd_size, m_batch_slice, dim3);
    output_matrix_size_ = m_batch * xd_size;
    xd_size_ = xd_size;
    m_batch_ = m_batch;
    m_batch_slice_ = m_batch_slice;
  };

  __host__ __device__ __forceinline__ Reference operator[](int idx) const {

    int j, i_dim3;
    getNewIdx<trans>(j, i_dim3, idx, batch_indices_, xd_size_, m_batch_slice_, m_batch_, M_);

    return Reference(data_ + (j + i_dim3 * output_matrix_size_));
  };

  T *data_;
  const int *batch_indices_;
  int M_, output_matrix_size_, m_batch_, m_batch_slice_, xd_size_;
};

} // namespace RPU
