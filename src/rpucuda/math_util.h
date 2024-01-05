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

#include "math.h"
#include "utility_functions.h"

#ifdef RPU_USE_MKL
#include "mkl.h"
#else
#ifdef RPU_USE_OPENBLAS
extern "C" {
#include "cblas.h"
}
#endif
#endif

#ifndef MIN
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#endif
#ifndef MAX
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#endif

namespace RPU {
namespace math {

template <typename T>
void gemm(
    const CBLAS_ORDER Order,
    const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB,
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

template <typename T> int iamax(const int N, const T *X, const int incX);

template <typename T> T max(const int N, const T *X, const int incX);

template <typename T> void scal(const int N, const T alpha, T *X, const int incX);

template <typename T> T nrm2(const int N, const T *X, const int incX);

template <typename T> void copy(const int N, const T *X, const int incX, T *Y, const int incY);

template <typename T>
void gemv(
    const CBLAS_ORDER Order,
    const CBLAS_TRANSPOSE TransA,
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
    const CBLAS_ORDER Order,
    const int M,
    const int N,
    const T alpha,
    const T *X,
    const int incX,
    const T *Y,
    const int incY,
    T *A,
    const int lda);

template <typename T> void permute132(T *X_out, const T *X_in, int d1, int d2, int d3, bool bias);

template <typename T>
void makeBias(
    T *x_with_bias, const T *x_without_bias, const int size, const int m_batch, const bool trans);

template <typename T>
void copyWithoutBias(
    T *x_without_bias, const T *x_with_bias, const int size, const int m_batch, const bool trans);

} // namespace math
} // namespace RPU
