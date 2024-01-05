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

#include "math_util.h"
#ifdef RPU_USE_FP16
#include "cuda_fp16.h"
#endif
#include <cstring>
#include <utility_functions.h>

namespace RPU {
namespace math {
template <>
void gemm<float>(
    const CBLAS_ORDER Order,
    const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB,
    const int M,
    const int N,
    const int K,
    const float alpha,
    const float *A,
    const int lda,
    const float *B,
    const int ldb,
    const float beta,
    float *C,
    const int ldc) {
  cblas_sgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
};

template <>
void gemm<double>(
    const CBLAS_ORDER Order,
    const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB,
    const int M,
    const int N,
    const int K,
    const double alpha,
    const double *A,
    const int lda,
    const double *B,
    const int ldb,
    const double beta,
    double *C,
    const int ldc) {
  cblas_dgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
};

#ifdef RPU_DEFINE_CUDA_HALF_ARRAY
template <>
void gemm<half_t>(
    const CBLAS_ORDER Order,
    const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB,
    const int M,
    const int N,
    const int K,
    const half_t alpha,
    const half_t *A,
    const int lda,
    const half_t *B,
    const int ldb,
    const half_t beta,
    half_t *C,
    const int ldc){
    // TODO: DOES HGEMM JUST NOT work for some reasons? MKL FP16 different from half_t ?
    // RPU_INFO("A: " << (float)A[0] << ", B[0] " << B[0] << ", C[0] " << C[0]);
    // cblas_hgemm(
    //     Order, TransA, TransB, M, N, K, alpha, (const unsigned short *) A, lda,
    //     (const unsigned short *) B, ldb, beta, (unsigned short *) C, ldc);

    // just use sgemm for now (quite slow to copy)
};
#endif

template <> int iamax<float>(const int N, const float *X, const int incX) {
  return (int)cblas_isamax(N, X, incX);
};

template <> int iamax<double>(const int N, const double *X, const int incX) {
  return (int)cblas_idamax(N, X, incX);
};

#ifdef RPU_DEFINE_CUDA_HALF_ARRAY
template <> int iamax<half_t>(const int N, const half_t *X, const int incX) {
  half_t max_value = fabsf(X[0]);
  int max_index = 0;
  int i_x = incX;
  PRAGMA_SIMD
  for (int i = 1; i < N; ++i) {
    if ((half_t)fabsf(X[i_x]) > max_value) {
      max_value = X[i_x];
      max_index = i_x;
    }
    i_x += incX;
  }
  return max_index;
};
#endif

template <typename T> T max(const int N, const T *X, const int incX) {
  T max_value = X[0];
  int i_x = incX;
  PRAGMA_SIMD
  for (int i = 1; i < N; ++i) {
    if (X[i_x] > max_value) {
      max_value = X[i_x];
    }
    i_x += incX;
  }
  return max_value;
};
template float max(const int, const float *X, const int);
template double max(const int, const double *X, const int);
#ifdef RPU_DEFINE_CUDA_HALF_ARRAY
template half_t max(const int, const half_t *X, const int);
#endif

template <>
void copy<float>(const int N, const float *X, const int incX, float *Y, const int incY) {
  cblas_scopy(N, X, incX, Y, incY);
}

template <>
void copy<double>(const int N, const double *X, const int incX, double *Y, const int incY) {
  cblas_dcopy(N, X, incX, Y, incY);
}

#ifdef RPU_DEFINE_CUDA_HALF_ARRAY
template <>
void copy<half_t>(const int N, const half_t *X, const int incX, half_t *Y, const int incY) {
  if (incY == 1 && incY == 1) {
    memcpy((void *)Y, (const void *)X, sizeof(half_t) * N);
  } else {
    int i_x = 0;
    int i_y = 0;
    PRAGMA_SIMD
    for (int i = 0; i < N; ++i) {
      Y[i_y] = X[i_x];
      i_x += incX;
      i_y += incY;
    }
  }
}
#endif

template <> void scal<float>(const int N, const float alpha, float *X, const int incX) {
  cblas_sscal(N, alpha, X, incX);
}

template <> void scal<double>(const int N, const double alpha, double *X, const int incX) {
  cblas_dscal(N, alpha, X, incX);
}

#ifdef RPU_DEFINE_CUDA_HALF_ARRAY
template <> void scal<half_t>(const int N, const half_t alpha, half_t *X, const int incX) {
  int i_x = 0;
  PRAGMA_SIMD
  for (int i = 0; i < N; ++i) {
    X[i_x] *= alpha;
    i_x += incX;
  }
}
#endif

template <> float nrm2<float>(const int N, const float *X, const int incX) {
  return cblas_snrm2(N, X, incX);
}

template <> double nrm2<double>(const int N, const double *X, const int incX) {
  return cblas_dnrm2(N, X, incX);
}

#ifdef RPU_DEFINE_CUDA_HALF_ARRAY
template <> half_t nrm2<half_t>(const int N, const half_t *X, const int incX) {
  half_t nrm_value = (half_t)0.0;
  int i_x = 0;
  PRAGMA_SIMD
  for (int i = 0; i < N; ++i) {
    half_t x = X[i_x];
    nrm_value += x * x;
    i_x += incX;
  }
  return nrm_value;
}
#endif

template <>
void gemv<float>(
    const CBLAS_ORDER Order,
    const CBLAS_TRANSPOSE TransA,
    const int M,
    const int N,
    const float alpha,
    const float *A,
    const int lda,
    const float *X,
    const int incX,
    const float beta,
    float *Y,
    const int incY) {
  cblas_sgemv(Order, TransA, M, N, alpha, A, lda, X, incX, beta, Y, incY);
}

template <>
void gemv<double>(
    const CBLAS_ORDER Order,
    const CBLAS_TRANSPOSE TransA,
    const int M,
    const int N,
    const double alpha,
    const double *A,
    const int lda,
    const double *X,
    const int incX,
    const double beta,
    double *Y,
    const int incY) {
  cblas_dgemv(Order, TransA, M, N, alpha, A, lda, X, incX, beta, Y, incY);
}

#ifdef RPU_DEFINE_CUDA_HALF_ARRAY
template <>
void gemv<half_t>(
    const CBLAS_ORDER Order,
    const CBLAS_TRANSPOSE TransA,
    const int M,
    const int K,
    const half_t alpha,
    const half_t *A,
    const int lda,
    const half_t *X,
    const int incX,
    const half_t beta,
    half_t *Y,
    const int incY) {

  if (incY != 1 || incX != 1) {
    RPU_INFO(
        "GEMV: M " << M << ", K " << K << ", incX " << incX << ", incY " << incY << ", lda "
                   << lda);
    RPU_FATAL("Larger 1 increments not possible with GEMV for half_t.");
  }
  if (CblasRowMajor == Order) {
    if (lda != K) {
      RPU_FATAL("Expected lda to be " << K << " but found " << lda << ".");
    }
    gemm<half_t>(
        CblasRowMajor, TransA, CblasNoTrans, M, 1, K, alpha, A, (TransA == CblasNoTrans) ? K : M, X,
        1, beta, Y, 1);
  } else {
    if (lda != M) {
      RPU_FATAL("Expected lda to be " << M << " but found " << lda << ".");
    }
    gemm<half_t>(
        CblasColMajor, TransA, CblasNoTrans, M, 1, K, alpha, A, (TransA == CblasNoTrans) ? M : K, X,
        K, beta, Y, M);
  }
}
#endif

template <>
void ger<float>(
    const CBLAS_ORDER Order,
    const int M,
    const int N,
    const float alpha,
    const float *X,
    const int incX,
    const float *Y,
    const int incY,
    float *A,
    const int lda) {

  cblas_sger(Order, M, N, alpha, X, incX, Y, incY, A, lda);
};

template <>
void ger<double>(
    const CBLAS_ORDER Order,
    const int M,
    const int N,
    const double alpha,
    const double *X,
    const int incX,
    const double *Y,
    const int incY,
    double *A,
    const int lda) {

  cblas_dger(Order, M, N, alpha, X, incX, Y, incY, A, lda);
};

#ifdef RPU_DEFINE_CUDA_HALF_ARRAY
template <>
void ger<half_t>(
    const CBLAS_ORDER Order,
    const int M,
    const int N,
    const half_t alpha,
    const half_t *X,
    const int incX,
    const half_t *Y,
    const int incY,
    half_t *A,
    const int lda) {
  if (incY != 1 || incX != 1 || CblasRowMajor != Order || lda != N) {
    RPU_FATAL("Larger 1 increments or col order not possible with GER for half_t.");
  }
  gemm<half_t>(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, 1, alpha, X, 1, Y, N, 1.0, A, N);
};
#endif

// permute132
template <typename T>
void permute132(
    T *X_out,
    const T *X_in,
    const int d1,
    const int d2, // with the bias
    const int d3,
    const bool bias) { // permute order to 132
  // bias is added to d2 (thus d1*d3 ones at the end)
  int output_offset = 0;

  int size12 = d1 * (d2 - bias);
  int sz = sizeof(T) * (d1);
  for (int i = 0; i < d2 - bias; ++i) {
    int input_offset = i * d1;
    for (int j = 0; j < d3; ++j) {
      std::memcpy((void *)(X_out + output_offset), (const void *)(X_in + input_offset), sz);
      output_offset += d1;
      input_offset += size12;
    }
  }
  if (bias) {
    int size13 = d1 * d3;
    for (int k = 0; k < size13; ++k) {
      X_out[output_offset + k] = (T)1.0;
    }
  }
}

template void
permute132<float>(float *, const float *, const int, const int, const int, const bool);
#ifdef RPU_USE_DOUBLE
template void
permute132<double>(double *, const double *, const int, const int, const int, const bool);
#endif
#ifdef RPU_DEFINE_CUDA_HALF_ARRAY
template void
permute132<half_t>(half_t *, const half_t *, const int, const int, const int, const bool);
#endif

// makeBias
template <typename T>
void makeBias(
    T *x_with_bias, const T *x_without_bias, const int size, const int m_batch, const bool trans) {

  if (trans) { // m_batch is first dim
    int sz = m_batch * (size - 1);
    std::memcpy((void *)x_with_bias, (const void *)x_without_bias, sizeof(T) * sz);
    for (int i = sz; i < sz + m_batch; ++i) {
      x_with_bias[i] = (T)1;
    }
  } else {
    int offset = 0;
    int offsetm1 = 0;
    int sz = sizeof(T) * (size - 1);
    for (int j = 0; j < m_batch; ++j) {
      std::memcpy((void *)(x_with_bias + offset), (const void *)(x_without_bias + offsetm1), sz);
      offset += size;
      offsetm1 += size - 1;
      x_with_bias[offset - 1] = (T)1.;
    }
  }
}

template void makeBias<float>(float *, const float *, const int, const int, const bool);
template void makeBias<int>(int *, const int *, const int, const int, const bool);
#ifdef RPU_USE_DOUBLE
template void makeBias<double>(double *, const double *, const int, const int, const bool);
#endif
#ifdef RPU_DEFINE_CUDA_HALF_ARRAY
template void makeBias<half_t>(half_t *, const half_t *, const int, const int, const bool);
#endif

// copy without Bias
template <typename T>
void copyWithoutBias(
    T *x_without_bias, const T *x_with_bias, const int size, const int m_batch, const bool trans) {
  if (trans) {
    // m_batch first
    int sz = m_batch * (size - 1);
    std::memcpy((void *)x_without_bias, (const void *)x_with_bias, sizeof(T) * sz);
  } else {
    // x_size first
    int offset = 0;
    int offsetm1 = 0;
    int sz = sizeof(T) * (size - 1);
    for (int j = 0; j < m_batch; ++j) {
      std::memcpy((void *)(x_without_bias + offsetm1), (const void *)(x_with_bias + offset), sz);
      offset += size;
      offsetm1 += size - 1;
    }
  }
}

template void copyWithoutBias<float>(float *, const float *, const int, const int, const bool);
#ifdef RPU_USE_DOUBLE
template void copyWithoutBias<double>(double *, const double *, const int, const int, const bool);
#endif
#ifdef RPU_DEFINE_CUDA_HALF_ARRAY
template void copyWithoutBias<half_t>(half_t *, const half_t *, const int, const int, const bool);
#endif
} // namespace math
} // namespace RPU
