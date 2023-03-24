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
#include <iterator>

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600

#else
static __inline__ __device__ double atomicAdd(double *address, double val) {
  unsigned long long int *address_as_ull = (unsigned long long int *)address;
  unsigned long long int old = *address_as_ull, assumed;
  if (val == 0.0)
    return __longlong_as_double(old);
  do {
    assumed = old;
    old = atomicCAS(
        address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}
#endif

static_assert(sizeof(unsigned long long int) == sizeof(uint64_t), "64 bit mismatch ");

namespace RPU {
namespace math {

template <typename T>
void gemm(
    const CudaContextPtr context,
    const bool TransA,
    const bool TransB,
    const int M,
    const int N,
    const int K,
    const T alpha,
    const T *A,
    const int lda,
    const T *B,
    const int ldb,
    const T beta,
    T *C,
    const int ldc);

template <typename T>
int iamax(const CudaContextPtr context, const int N, const T *X, const int incX);

template <typename T>
void scal(const CudaContextPtr context, const int N, const T alpha, T *X, const int incX);

template <typename T>
void nrm2(const CudaContextPtr context, const int N, const T *X, const int incX, T *res);

template <typename T>
void copy(
    const CudaContextPtr context, const int N, const T *X, const int incX, T *Y, const int incY);

template <typename T>
void gemv(
    const CudaContextPtr context,
    const bool TransA,
    const int M,
    const int N,
    const T alpha,
    const T *A,
    const int lda,
    const T *X,
    const int incX,
    const T beta,
    T *Y,
    const int incY);

template <typename T>
void ger(
    const CudaContextPtr context,
    const int M,
    const int N,
    const T alpha,
    const T *X,
    const int incX,
    const T *Y,
    const int incY,
    T *A,
    const int lda);

// W += beta * A
template <typename T, typename T_A>
void elemaddscale(const CudaContextPtr context, T *W, const int size, const T_A *A, const T beta);

// W += A.*B
template <typename T>
void elemaddscale(const CudaContextPtr context, T *W, const int size, const T *A, const T *B);
// W += A
template <typename T> void elemadd(const CudaContextPtr context, T *W, const int size, const T *A);

// W = W.*W
template <typename T>
void elempow2(const CudaContextPtr context, T *W, const int size, const T *W_in = nullptr);

// V = abs(W)
template <typename T> void elemabs(const CudaContextPtr context, T *V, const T *W, const int size);

// W += sat(A.*B) // + saturate at bounds
template <typename T, typename T_A>
void elemasb02(
    const CudaContextPtr context,
    T *W,
    const int size,
    const T_A *A,
    const T *B,
    float *dev_4params); // bounds in [0,2] // 4params and 2params always float !

// sat(W *= A) (optionally with shift)
template <typename T>
void elemscale(
    const CudaContextPtr context,
    T *W,
    const int size,
    const T *A,
    float *dev_4params = nullptr,
    const T *dev_shift = nullptr);

// C = A.*B
template <typename T>
void elemmul(const CudaContextPtr context, T *C, const int size, const T *A, const T *B);

// sat(W)
template <typename T>
void elemsat(const CudaContextPtr context, T *W, const int size, float *dev_4params);

// sat(W *= 1+alpha*(A-1))
template <typename T>
void elemscalealpha(
    const CudaContextPtr context,
    T *W,
    const int size,
    const T *A,
    float *dev_4params,
    const T alpha,
    const T *dev_shift = nullptr);

// W += A, A = W
template <typename T> void elemaddcopy(const CudaContextPtr context, T *W, T *A, const int size);

// W = sat(W+A), A = W
template <typename T>
void elemaddcopysat(
    const CudaContextPtr context, T *W, T *A, const int size, const float *dev_4params);

// A = scale*(W - A_in), W = A_in
template <typename T>
void elemsubcopy(const CudaContextPtr context, T *W, T *A, const int size, const T scale = 1.0);

// MSK = P<thres
// W(MSK) = sat(A(MSK) + B(MSK))
template <typename T>
void elemresetsat(
    const CudaContextPtr context,
    T *W,
    const int size,
    const T *A,
    const float *B, // float for random
    const float *P, // float for random
    T thres,
    const float *dev_4params);

// MSK != 0
// W(MSK) = sat(reset_bias(MSK) + std*randn())
template <typename T>
void elemresetsatmsk(
    CudaContextPtr context, // non const because might use random states
    T *W,
    const int size,
    const char *msk,
    const T *reset_bias, // can be nullptr
    const T reset_std,
    const float *dev_4params);

// set all elements to a
template <typename T>
void elemconst(const CudaContextPtr context, T *X, const int size, const T alpha);
template <typename T>
void elemconst(const CudaContext *context, T *X, const int size, const T alpha);

// permute(1,2,3)
template <typename T>
void permute132(
    const CudaContextPtr context,
    T *X_out,
    const T *X_in,
    const int d1,
    const int d2, // with bias added
    const int d3,
    const bool bias);

// w = max(min(w,|a|),-|a|)
template <typename T> void aclip(const CudaContextPtr context, T *W, const int size, const T a);

// w = max(w,a) element-wise
template <typename T>
void elemmax(
    const CudaContextPtr context, T *W, const int size, const T a, const T *W_in = nullptr);

// w = min(w,a) element-wise
template <typename T>
void elemmin(
    const CudaContextPtr context, T *W, const int size, const T a, const T *W_in = nullptr);

// w = w<a?0:w elementwise
template <typename T>
void elemsetbelowzero(const CudaContextPtr context, T *W, const int size, const T a);

// w[j] = sum_i^n(m_i[j])/n
template <typename T>
void elemaverage(const CudaContextPtr context, T *W, const int size, T **Ms, const int m);

// W[j] = a*A[j] + b*B[j]
template <typename T>
void elemweightedsum(
    const CudaContextPtr context,
    T *W,
    const int size,
    const T *A,
    const T a,
    const T *B,
    const T b);

template <typename T>
void makeBias(
    const CudaContextPtr context,
    T *x_with_bias,
    const T *x_without_bias,
    const int x_size,
    const int m_batch,
    const bool trans);

template <typename T>
void copyWithoutBias(
    const CudaContextPtr context,
    T *x_without_bias,
    const T *x_with_bias,
    const int x_size,
    const int m_batch,
    const bool trans);

// copyWithIterator
template <typename OutputIteratorT, typename InputIteratorT>
void copyWithIterator(
    const CudaContextPtr context,
    OutputIteratorT out_tensor,
    InputIteratorT in_tensor,
    const int total_input_size);

// addWithIterator
template <typename OutputIteratorT, typename T>
void addWithIterator(
    const CudaContextPtr context,
    OutputIteratorT out_tensor,
    const T *in_tensor_A,
    const T *in_tensor_B,
    const int total_input_size);

// to overcome compiling issues. ONLY forks for IteratorT=T * of const T * respectively. Else it
// will cause a compilation error. To be guarded with std::is_same<>
template <typename T, typename IteratorT> T *fakeCast(IteratorT X);
template <typename T, typename IteratorT> const T *fakeCastConst(IteratorT X);
} // namespace math
} // namespace RPU
