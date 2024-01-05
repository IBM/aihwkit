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

#include "cuda_fp16_util.h"
#include "cuda_math_util.h"
#include "cuda_util.h"
#include "io_iterator.h"

#define RPU_GET_CUBLAS_HANDLE                                                                      \
  cublasHandle_t handle = context->getBlasHandle();                                                \
  CUBLAS_CALL(cublasSetStream(handle, context->getStream()))

#define RPU_SET_CUBLAS_POINTER_MODE_DEVICE                                                         \
  cublasPointerMode_t p_mode;                                                                      \
  CUBLAS_CALL(cublasGetPointerMode(handle, &p_mode));                                              \
  CUBLAS_CALL(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE))

#define RPU_SET_CUBLAS_POINTER_MODE_HOST                                                           \
  cublasPointerMode_t p_mode;                                                                      \
  CUBLAS_CALL(cublasGetPointerMode(handle, &p_mode));                                              \
  CUBLAS_CALL(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST))

#define RPU_RESTORE_CUBLAS_POINTER_MODE CUBLAS_CALL(cublasSetPointerMode(handle, p_mode))

namespace RPU {
namespace math {

template <>
void gemm<float>(
    const CudaContextPtr context,
    const bool TransA,
    const bool TransB,
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
  RPU_GET_CUBLAS_HANDLE;

  RPU_SET_CUBLAS_POINTER_MODE_HOST;
  CUBLAS_CALL(cublasSgemm(
      handle, TransA ? CUBLAS_OP_T : CUBLAS_OP_N, TransB ? CUBLAS_OP_T : CUBLAS_OP_N, M, N, K,
      &alpha, A, lda, B, ldb, &beta, C, ldc));
  RPU_RESTORE_CUBLAS_POINTER_MODE;
};

template <>
void gemm<double>(
    const CudaContextPtr context,
    const bool TransA,
    const bool TransB,
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
  RPU_GET_CUBLAS_HANDLE;
  RPU_SET_CUBLAS_POINTER_MODE_HOST;
  CUBLAS_CALL(cublasDgemm(
      handle, TransA ? CUBLAS_OP_T : CUBLAS_OP_N, TransB ? CUBLAS_OP_T : CUBLAS_OP_N, M, N, K,
      &alpha, A, lda, B, ldb, &beta, C, ldc));
  RPU_RESTORE_CUBLAS_POINTER_MODE;
};

#ifdef RPU_DEFINE_CUDA_HALF_ARRAY
template <>
void gemm<half_t>(
    const CudaContextPtr context,
    const bool TransA,
    const bool TransB,
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
    const int ldc) {
  RPU_GET_CUBLAS_HANDLE;
  RPU_SET_CUBLAS_POINTER_MODE_HOST;
  CUBLAS_CALL(cublasHgemm(
      handle, TransA ? CUBLAS_OP_T : CUBLAS_OP_N, TransB ? CUBLAS_OP_T : CUBLAS_OP_N, M, N, K,
      &alpha, A, lda, B, ldb, &beta, C, ldc));
  RPU_RESTORE_CUBLAS_POINTER_MODE;
};
#endif

template <>
void copy<float>(
    const CudaContextPtr context,
    const int N,
    const float *X,
    const int incX,
    float *Y,
    const int incY) {
  RPU_GET_CUBLAS_HANDLE;
  CUBLAS_CALL(cublasScopy(handle, N, X, incX, Y, incY));
}

template <>
void copy<double>(
    const CudaContextPtr context,
    const int N,
    const double *X,
    const int incX,
    double *Y,
    const int incY) {
  RPU_GET_CUBLAS_HANDLE;
  CUBLAS_CALL(cublasDcopy(handle, N, X, incX, Y, incY));
}

#ifdef RPU_DEFINE_CUDA_HALF_ARRAY
template <>
void copy<half_t>(
    const CudaContextPtr context,
    const int N,
    const half_t *X,
    const int incX,
    half_t *Y,
    const int incY) {
  elemcopy<half_t, half_t>(context, Y, N, incY, X, incX);
}
#endif

template <>
void scal<float>(const CudaContextPtr context, const int N, const float alpha, float *X) {
  RPU_GET_CUBLAS_HANDLE;
  RPU_SET_CUBLAS_POINTER_MODE_HOST;
  CUBLAS_CALL(cublasSscal(handle, N, &alpha, X, 1));
  RPU_RESTORE_CUBLAS_POINTER_MODE;
}

template <>
void scal<double>(const CudaContextPtr context, const int N, const double alpha, double *X) {
  RPU_GET_CUBLAS_HANDLE;
  RPU_SET_CUBLAS_POINTER_MODE_HOST;
  CUBLAS_CALL(cublasDscal(handle, N, &alpha, X, 1));
  RPU_RESTORE_CUBLAS_POINTER_MODE;
}

#ifdef RPU_DEFINE_CUDA_HALF_ARRAY
template <>
void scal<half_t>(const CudaContextPtr context, const int N, const half_t alpha, half_t *X) {
  elemscale<half_t>(context, X, N, alpha);
}
#endif

template <>
void nrm2<float>(
    const CudaContextPtr context, const int N, const float *X, const int incX, float *res) {
  RPU_GET_CUBLAS_HANDLE;
  RPU_SET_CUBLAS_POINTER_MODE_DEVICE;
  CUBLAS_CALL(cublasSnrm2(handle, N, X, incX, res));
  RPU_RESTORE_CUBLAS_POINTER_MODE;
}

template <>
void nrm2<double>(
    const CudaContextPtr context, const int N, const double *X, const int incX, double *res) {
  RPU_GET_CUBLAS_HANDLE;
  RPU_SET_CUBLAS_POINTER_MODE_DEVICE;
  CUBLAS_CALL(cublasDnrm2(handle, N, X, incX, res));
  RPU_RESTORE_CUBLAS_POINTER_MODE;
}

#ifdef RPU_DEFINE_CUDA_HALF_ARRAY
template <>
void nrm2<half_t>(
    const CudaContextPtr context, const int N, const half_t *X, const int incX, half_t *res) {
  RPU_NOT_IMPLEMENTED;
}
#endif

template <>
void gemv<float>(
    const CudaContextPtr context,
    const bool TransA,
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
  RPU_GET_CUBLAS_HANDLE;
  // col major !!
  RPU_SET_CUBLAS_POINTER_MODE_HOST;
  CUBLAS_CALL(cublasSgemv(
      handle, TransA ? CUBLAS_OP_T : CUBLAS_OP_N, M, N, &alpha, A, lda, X, incX, &beta, Y, incY));
  RPU_RESTORE_CUBLAS_POINTER_MODE;
}

template <>
void gemv<double>(
    const CudaContextPtr context,
    const bool TransA,
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
  RPU_GET_CUBLAS_HANDLE;
  RPU_SET_CUBLAS_POINTER_MODE_HOST;
  CUBLAS_CALL(cublasDgemv(
      handle, TransA ? CUBLAS_OP_T : CUBLAS_OP_N, M, N, &alpha, A, lda, X, incX, &beta, Y, incY));
  RPU_RESTORE_CUBLAS_POINTER_MODE;
}

#ifdef RPU_DEFINE_CUDA_HALF_ARRAY
template <>
void gemv<half_t>(
    const CudaContextPtr context,
    const bool TransA,
    const int M,
    const int N,
    const half_t alpha,
    const half_t *A,
    const int lda,
    const half_t *X,
    const int incX,
    const half_t beta,
    half_t *Y,
    const int incY) {
  if (incX != 1 || incY != 1) {
    RPU_FATAL("Larger 1 increments not possible with GEMV for half_t.");
  }
  gemm<half_t>(context, TransA, false, M, 1, N, alpha, A, lda, X, N, beta, Y, M);
}
#endif

template <>
void ger<float>(
    const CudaContextPtr context,
    const int M,
    const int N,
    const float alpha,
    const float *X,
    const int incX,
    const float *Y,
    const int incY,
    float *A,
    const int lda) {
  RPU_GET_CUBLAS_HANDLE;
  RPU_SET_CUBLAS_POINTER_MODE_HOST;
  CUBLAS_CALL(cublasSger(handle, M, N, &alpha, X, incX, Y, incY, A, lda));
  RPU_RESTORE_CUBLAS_POINTER_MODE;
};

template <>
void ger<double>(
    const CudaContextPtr context,
    const int M,
    const int N,
    const double alpha,
    const double *X,
    const int incX,
    const double *Y,
    const int incY,
    double *A,
    const int lda) {
  RPU_GET_CUBLAS_HANDLE;
  RPU_SET_CUBLAS_POINTER_MODE_HOST;
  CUBLAS_CALL(cublasDger(handle, M, N, &alpha, X, incX, Y, incY, A, lda));
  RPU_RESTORE_CUBLAS_POINTER_MODE;
};

#ifdef RPU_DEFINE_CUDA_HALF_ARRAY
template <>
void ger<half_t>(
    const CudaContextPtr context,
    const int M,
    const int N,
    const half_t alpha,
    const half_t *X,
    const int incX,
    const half_t *Y,
    const int incY,
    half_t *A,
    const int lda) {
  if (incX != 1 || incY != 1) {
    RPU_FATAL("Larger 1 increments not possible with GER for half_t.");
  }
  gemm<half_t>(context, true, false, M, N, 1, alpha, X, 1, Y, 1, 1.0, A, lda);
};
#endif

// W = A
template <typename T, typename T_A>
__global__ void
kernelElemCopy(T *dev_W, const int size, const int incW, const T_A *dev_A, const int incA) {
  RPU_CUDA_1D_KERNEL_LOOP(idx, size) { dev_W[incW * idx] = (T)dev_A[incA * idx]; }
}

template <typename T, typename T_A>
void elemcopy(
    const CudaContextPtr context,
    T *dev_W,
    const int size,
    const int incW,
    const T_A *dev_A,
    const int incA) {

  int nthreads = context->getNThreads();
  int nblocks = context->getNBlocks(size, nthreads);
  kernelElemCopy<T, T_A>
      <<<nblocks, nthreads, 0, context->getStream()>>>(dev_W, size, incW, dev_A, incA);
}
template void elemcopy<float, float>(
    const CudaContextPtr, float *, const int, const int, const float *, const int);
template void elemcopy<float, double>(
    const CudaContextPtr, float *, const int, const int, const double *, const int);
#ifdef RPU_USE_DOUBLE
template void elemcopy<double, double>(
    const CudaContextPtr, double *, const int, const int, const double *, const int);
template void elemcopy<double, float>(
    const CudaContextPtr, double *, const int, const int, const float *, const int);
#endif
#ifdef RPU_DEFINE_CUDA_HALF_ARRAY
template void elemcopy<half_t, half_t>(
    const CudaContextPtr, half_t *, const int, const int, const half_t *, const int);
template void elemcopy<half_t, float>(
    const CudaContextPtr, half_t *, const int, const int, const float *, const int);
template void elemcopy<half_t, double>(
    const CudaContextPtr, half_t *, const int, const int, const double *, const int);
template void elemcopy<float, half_t>(
    const CudaContextPtr, float *, const int, const int, const half_t *, const int);
template void elemcopy<double, half_t>(
    const CudaContextPtr, double *, const int, const int, const half_t *, const int);
#endif

// W *= alpha
template <typename T> __global__ void kernelElemScale(T *dev_W, const int size, const T alpha) {
  RPU_CUDA_1D_KERNEL_LOOP(idx, size) { dev_W[idx] *= alpha; }
}

template <typename T>
void elemscale(const CudaContextPtr context, T *dev_W, const int size, const T alpha) {

  int nthreads = context->getNThreads();
  int nblocks = context->getNBlocks<T>(size, nthreads, true);
  kernelElemScale<T><<<nblocks, nthreads, 0, context->getStream()>>>(dev_W, size, alpha);
}
template void elemscale<float>(const CudaContextPtr, float *, const int, const float);
#ifdef RPU_USE_DOUBLE
template void elemscale<double>(const CudaContextPtr, double *, const int, const double);
#endif
#ifdef RPU_DEFINE_CUDA_HALF_ARRAY
template <> __global__ void kernelElemScale(half_t *dev_W, const int size, const half_t alpha) {
  half2_t alpha2;
  alpha2.x = alpha;
  alpha2.y = alpha;
  RPU_CUDA_1D_KERNEL_LOOP_HALF(idx, size) { HALF2PTR(dev_W)[idx] *= alpha2; }
}
template void elemscale<half_t>(const CudaContextPtr, half_t *, const int, const half_t);
#endif

// W += A
template <typename T> __global__ void kernelElemAdd(T *dev_W, const int size, const T *dev_A) {
  RPU_CUDA_1D_KERNEL_LOOP(idx, size) { dev_W[idx] += dev_A[idx]; }
}

template <typename T>
void elemadd(const CudaContextPtr context, T *dev_W, const int size, const T *dev_A) {

  int nthreads = context->getNThreads();
  int nblocks = context->getNBlocks<T>(size, nthreads, true);
  kernelElemAdd<T><<<nblocks, nthreads, 0, context->getStream()>>>(dev_W, size, dev_A);
}
template void elemadd<float>(const CudaContextPtr, float *, const int, const float *);
#ifdef RPU_USE_DOUBLE
template void elemadd<double>(const CudaContextPtr, double *, const int, const double *);
#endif
#ifdef RPU_DEFINE_CUDA_HALF_ARRAY
template <> __global__ void kernelElemAdd(half_t *dev_W, const int size, const half_t *dev_A) {
  RPU_CUDA_1D_KERNEL_LOOP_HALF(idx, size) { HALF2PTR(dev_W)[idx] += HALF2PTRCONST(dev_A)[idx]; }
}
template void elemadd<half_t>(const CudaContextPtr, half_t *, const int, const half_t *);
#endif

// W = W.*W
template <typename T> __global__ void kernelElemPow2(T *dev_W, const int size, const T *W_in) {

  RPU_CUDA_1D_KERNEL_LOOP(idx, size) {
    T x = W_in[idx];
    dev_W[idx] = x * x;
  }
}
template <typename T>
void elempow2(const CudaContextPtr context, T *dev_W, const int size, const T *dev_W_in) {

  int nthreads = context->getNThreads();
  int nblocks = context->getNBlocks(size, nthreads);
  kernelElemPow2<T><<<nblocks, nthreads, 0, context->getStream()>>>(
      dev_W, size, dev_W_in == nullptr ? dev_W : dev_W_in);
}
template void elempow2<float>(const CudaContextPtr, float *, const int, const float *);
#ifdef RPU_USE_DOUBLE
template void elempow2<double>(const CudaContextPtr, double *, const int, const double *);
#endif
#ifdef RPU_DEFINE_CUDA_HALF_ARRAY
template void elempow2<half_t>(const CudaContextPtr, half_t *, const int, const half_t *);
#endif

// V = abs(W )
template <typename T> __global__ void kernelElemAbs(T *dev_V, const T *dev_W, const int size) {

  RPU_CUDA_1D_KERNEL_LOOP(idx, size) { dev_V[idx] = fabs(dev_W[idx]); }
}
template <typename T>
void elemabs(const CudaContextPtr context, T *dev_V, const T *dev_W, const int size) {

  int nthreads = context->getNThreads();
  int nblocks = context->getNBlocks(size, nthreads);
  kernelElemAbs<T><<<nblocks, nthreads, 0, context->getStream()>>>(dev_V, dev_W, size);
}
template void elemabs<float>(const CudaContextPtr, float *, const float *, const int);
#ifdef RPU_USE_DOUBLE
template void elemabs<double>(const CudaContextPtr, double *, const double *, const int);
#endif
#ifdef RPU_DEFINE_CUDA_HALF_ARRAY
template void elemabs<half_t>(const CudaContextPtr, half_t *, const half_t *, const int);
#endif

// W += beta*A
template <typename T, typename T_A>
__global__ void kernelElemAddScale(T *dev_W, const int size, const T_A *dev_A, const T beta) {

  T b = beta;
  RPU_CUDA_1D_KERNEL_LOOP(idx, size) {
    T w = dev_W[idx];
    T a = dev_A[idx];

    w += a * b;
    dev_W[idx] = w;
  }
}
template <typename T, typename T_A>
void elemaddscale(
    const CudaContextPtr context, T *dev_W, const int size, const T_A *dev_A, const T beta) {

  int nthreads = context->getNThreads();
  int nblocks = context->getNBlocks<T>(size, nthreads, true);
  kernelElemAddScale<T><<<nblocks, nthreads, 0, context->getStream()>>>(dev_W, size, dev_A, beta);
}
template void
elemaddscale<float, float>(const CudaContextPtr, float *, const int, const float *, const float);
#ifdef RPU_USE_DOUBLE
template void elemaddscale<double, double>(
    const CudaContextPtr, double *, const int, const double *, const double);
template void
elemaddscale<double, float>(const CudaContextPtr, double *, const int, const float *, const double);
#endif
#ifdef RPU_DEFINE_CUDA_HALF_ARRAY

template void elemaddscale<half_t, half_t>(
    const CudaContextPtr, half_t *, const int, const half_t *, const half_t);
template void
elemaddscale<half_t, float>(const CudaContextPtr, half_t *, const int, const float *, const half_t);
#endif

// W += A.*B
template <typename T>
__global__ void kernelElemAddScale(T *dev_W, const int size, const T *dev_A, const T *dev_B) {

  RPU_CUDA_1D_KERNEL_LOOP(idx, size) {
    T w = dev_W[idx];
    T a = dev_A[idx];
    T b = dev_B[idx];

    w += a * b;
    dev_W[idx] = w;
  }
}
template <typename T>
void elemaddscale(
    const CudaContextPtr context, T *dev_W, const int size, const T *dev_A, const T *dev_B) {

  int nthreads = context->getNThreads();
  int nblocks = context->getNBlocks<T>(size, nthreads, true);
  kernelElemAddScale<T><<<nblocks, nthreads, 0, context->getStream()>>>(dev_W, size, dev_A, dev_B);
}
template void
elemaddscale<float>(const CudaContextPtr, float *, const int, const float *, const float *);
#ifdef RPU_USE_DOUBLE
template void
elemaddscale<double>(const CudaContextPtr, double *, const int, const double *, const double *);
#endif
#ifdef RPU_DEFINE_CUDA_HALF_ARRAY
template <>
__global__ void
kernelElemAddScale(half_t *dev_W, const int size, const half_t *dev_A, const half_t *dev_B) {

  RPU_CUDA_1D_KERNEL_LOOP_HALF(idx, size) {
    half2_t w = HALF2PTR(dev_W)[idx];
    half2_t a = HALF2PTRCONST(dev_A)[idx];
    half2_t b = HALF2PTRCONST(dev_B)[idx];

    w += a * b;
    HALF2PTR(dev_W)[idx] = w;
  }
}
template void
elemaddscale<half_t>(const CudaContextPtr, half_t *, const int, const half_t *, const half_t *);
#endif

// W += sat(A.*B)
template <typename T, typename T_A>
__global__ void
kernelElemASB02(T *dev_W, const int size, const T_A *dev_A, const T *dev_B, param_t *dev_4params) {

  RPU_CUDA_1D_KERNEL_LOOP(idx, size) {

    T w = dev_W[idx];
    T a = dev_A[idx];
    T b = dev_B[idx];
    param4_t parij = reinterpret_cast<param4_t *>(dev_4params)[idx];

    w += a * b;
    // check bounds
    w = (w > (T)parij.z) ? (T)parij.z : w;
    w = (w < (T)parij.x) ? (T)parij.x : w;

    dev_W[idx] = w;
  }
}
template <typename T, typename T_A>
void elemasb02(
    const CudaContextPtr context,
    T *dev_W,
    const int size,
    const T_A *dev_A,
    const T *dev_B,
    param_t *dev_4params) {
  int nthreads = context->getNThreads();
  int nblocks = context->getNBlocks(size, nthreads);
  kernelElemASB02<T, T_A>
      <<<nblocks, nthreads, 0, context->getStream()>>>(dev_W, size, dev_A, dev_B, dev_4params);
}
template void elemasb02<float, float>(
    const CudaContextPtr, float *, const int, const float *, const float *, param_t *);
#ifdef RPU_USE_DOUBLE
template void elemasb02<double, double>(
    const CudaContextPtr, double *, const int, const double *, const double *, param_t *);
template void elemasb02<double, float>(
    const CudaContextPtr, double *, const int, const float *, const double *, param_t *);
#endif
#ifdef RPU_DEFINE_CUDA_HALF_ARRAY
template void elemasb02<half_t, half_t>(
    const CudaContextPtr, half_t *, const int, const half_t *, const half_t *, param_t *);
template void elemasb02<half_t, float>(
    const CudaContextPtr, half_t *, const int, const float *, const half_t *, param_t *);
#endif

// sat(W *= A) (w/shift)
template <typename T>
__global__ void kernelElemScaleSat(
    T *dev_W, const int size, const T *dev_A, param_t *dev_4params, const T *dev_shift) {

  bool with_shift = dev_shift != nullptr;
  RPU_CUDA_1D_KERNEL_LOOP(idx, size) {
    T w = dev_W[idx];
    T a = dev_A[idx];
    T s = with_shift ? dev_shift[idx] : (T)0.0;
    param4_t parij = reinterpret_cast<param4_t *>(dev_4params)[idx];

    w = (w - s) * a + s;
    // check bounds
    w = (w > (T)parij.z) ? (T)parij.z : w;
    w = (w < (T)parij.x) ? (T)parij.x : w;

    dev_W[idx] = w;
  }
}
// W *= A (w/shift)
template <typename T>
__global__ void kernelElemScale(T *dev_W, const int size, const T *dev_A, const T *dev_shift) {
  bool with_shift = dev_shift != nullptr;
  RPU_CUDA_1D_KERNEL_LOOP(idx, size) {
    T w = dev_W[idx];
    T a = dev_A[idx];
    T s = with_shift ? dev_shift[idx] : (T)0.0;
    w = (w - s) * a + s;
    dev_W[idx] = w;
  }
}

template <typename T>
void elemscale(
    const CudaContextPtr context,
    T *dev_W,
    const int size,
    const T *dev_A,
    param_t *dev_4params,
    const T *dev_shift) {

  int nthreads = context->getNThreads();
  int nblocks = context->getNBlocks(size, nthreads);
  if (dev_4params != nullptr) {
    kernelElemScaleSat<T><<<nblocks, nthreads, 0, context->getStream()>>>(
        dev_W, size, dev_A, dev_4params, dev_shift);
  } else {
    kernelElemScale<T>
        <<<nblocks, nthreads, 0, context->getStream()>>>(dev_W, size, dev_A, dev_shift);
  }
}
template void
elemscale<float>(const CudaContextPtr, float *, const int, const float *, param_t *, const float *);
#ifdef RPU_USE_DOUBLE
template void elemscale<double>(
    const CudaContextPtr, double *, const int, const double *, param_t *, const double *);
#endif
#ifdef RPU_DEFINE_CUDA_HALF_ARRAY
template void elemscale<half_t>(
    const CudaContextPtr, half_t *, const int, const half_t *, param_t *, const half_t *);
#endif

// C = A.*B
template <typename T>
__global__ void kernelElemMul(T *dev_C, const int size, const T *dev_A, const T *dev_B) {

  RPU_CUDA_1D_KERNEL_LOOP(idx, size) { dev_C[idx] = dev_A[idx] * dev_B[idx]; }
}

template <typename T>
void elemmul(
    const CudaContextPtr context, T *dev_C, const int size, const T *dev_A, const T *dev_B) {

  int nthreads = context->getNThreads();
  int nblocks = context->getNBlocks<T>(size, nthreads, true);
  kernelElemMul<T><<<nblocks, nthreads, 0, context->getStream()>>>(dev_C, size, dev_A, dev_B);
}
template void
elemmul<float>(const CudaContextPtr, float *, const int, const float *, const float *);
#ifdef RPU_USE_DOUBLE
template void
elemmul<double>(const CudaContextPtr, double *, const int, const double *, const double *);
#endif
#ifdef RPU_DEFINE_CUDA_HALF_ARRAY
template <>
__global__ void
kernelElemMul(half_t *dev_C, const int size, const half_t *dev_A, const half_t *dev_B) {
  RPU_CUDA_1D_KERNEL_LOOP_HALF(idx, size) {
    HALF2PTR(dev_C)[idx] = HALF2PTRCONST(dev_A)[idx] * HALF2PTRCONST(dev_B)[idx];
  }
}
template void
elemmul<half_t>(const CudaContextPtr, half_t *, const int, const half_t *, const half_t *);
#endif

// sat(W)
template <typename T>
__global__ void kernelElemSat(T *dev_W, const int size, param_t *dev_4params) {

  RPU_CUDA_1D_KERNEL_LOOP(idx, size) {

    T w = dev_W[idx];
    param4_t parij = reinterpret_cast<param4_t *>(dev_4params)[idx];
    // check bounds
    w = (w > (T)parij.z) ? (T)parij.z : w;
    w = (w < (T)parij.x) ? (T)parij.x : w;
    dev_W[idx] = w;
  }
}
template <typename T>
void elemsat(const CudaContextPtr context, T *dev_W, const int size, param_t *dev_4params) {

  int nthreads = context->getNThreads();
  int nblocks = context->getNBlocks(size, nthreads);
  kernelElemSat<T><<<nblocks, nthreads, 0, context->getStream()>>>(dev_W, size, dev_4params);
}
template void elemsat<float>(const CudaContextPtr, float *, const int, param_t *);
#ifdef RPU_USE_DOUBLE
template void elemsat<double>(const CudaContextPtr, double *, const int, param_t *);
#endif
#ifdef RPU_DEFINE_CUDA_HALF_ARRAY
template void elemsat<half_t>(const CudaContextPtr, half_t *, const int, param_t *);
#endif

// sat(W *= 1+(A-1)*alpha)
template <typename T>
__global__ void kernelElemScaleAlpha(
    T *dev_W,
    const int size,
    const T *dev_A,
    param_t *dev_4params,
    const T alpha,
    const T *dev_shift) {

  bool with_shift = dev_shift != nullptr;
  RPU_CUDA_1D_KERNEL_LOOP(idx, size) {

    T w = dev_W[idx];
    T a = dev_A[idx];
    T s = with_shift ? dev_shift[idx] : (T)0.0;
    param4_t parij = reinterpret_cast<param4_t *>(dev_4params)[idx];

    T scale = (T)1.0 + alpha * (a - (T)1.0);
    w = (w - s) * scale + s;

    // check bounds
    w = (w > (T)parij.z) ? (T)parij.z : w;
    w = (w < (T)parij.x) ? (T)parij.x : w;

    dev_W[idx] = w;
  }
}
template <typename T>
void elemscalealpha(
    const CudaContextPtr context,
    T *dev_W,
    const int size,
    const T *dev_A,
    param_t *dev_4params,
    const T alpha,
    const T *dev_shift) {

  int nthreads = context->getNThreads();
  int nblocks = context->getNBlocks(size, nthreads);
  kernelElemScaleAlpha<T><<<nblocks, nthreads, 0, context->getStream()>>>(
      dev_W, size, dev_A, dev_4params, alpha, dev_shift);
}
template void elemscalealpha<float>(
    const CudaContextPtr, float *, const int, const float *, param_t *, const float, const float *);
#ifdef RPU_USE_DOUBLE
template void elemscalealpha<double>(
    const CudaContextPtr,
    double *,
    const int,
    const double *,
    param_t *,
    const double,
    const double *);
#endif
#ifdef RPU_DEFINE_CUDA_HALF_ARRAY
template void elemscalealpha<half_t>(
    const CudaContextPtr,
    half_t *,
    const int,
    const half_t *,
    param_t *,
    const half_t,
    const half_t *);
#endif

// W += A, A = W
template <typename T> __global__ void kernelElemAddCopy(T *dev_W, T *dev_A, const int size) {

  RPU_CUDA_1D_KERNEL_LOOP(idx, size) {
    T w = dev_W[idx];
    w += dev_A[idx];
    dev_W[idx] = w;
    dev_A[idx] = w;
  }
}
template <typename T>
void elemaddcopy(const CudaContextPtr context, T *dev_W, T *dev_A, const int size) {

  int nthreads = context->getNThreads();
  int nblocks = context->getNBlocks<T>(size, nthreads, true);
  kernelElemAddCopy<T><<<nblocks, nthreads, 0, context->getStream()>>>(dev_W, dev_A, size);
}
template void elemaddcopy<float>(const CudaContextPtr, float *, float *, const int);
#ifdef RPU_USE_DOUBLE
template void elemaddcopy<double>(const CudaContextPtr, double *, double *, const int);
#endif
#ifdef RPU_DEFINE_CUDA_HALF_ARRAY

template <> __global__ void kernelElemAddCopy(half_t *dev_W, half_t *dev_A, const int size) {
  RPU_CUDA_1D_KERNEL_LOOP_HALF(idx, size) {
    half2_t w = HALF2PTR(dev_W)[idx];
    w += HALF2PTR(dev_A)[idx];
    HALF2PTR(dev_W)[idx] = w;
    HALF2PTR(dev_A)[idx] = w;
  }
}

template void elemaddcopy<half_t>(const CudaContextPtr, half_t *, half_t *, const int);
#endif

// W = sat(W+A), A = W
template <typename T>
__global__ void
kernelElemAddCopySat(T *dev_W, T *dev_A, const int size, const param_t *dev_4params) {

  RPU_CUDA_1D_KERNEL_LOOP(idx, size) {

    T w = dev_W[idx];
    T a = dev_A[idx];
    const param4_t parij = reinterpret_cast<const param4_t *>(dev_4params)[idx];
    w += a;
    // check bounds
    w = (w > (T)parij.z) ? (T)parij.z : w;
    w = (w < (T)parij.x) ? (T)parij.x : w;
    a = w;
    dev_W[idx] = w;
    dev_A[idx] = a;
  }
}
template <typename T>
void elemaddcopysat(
    const CudaContextPtr context, T *dev_W, T *dev_A, const int size, const param_t *dev_4params) {

  int nthreads = context->getNThreads();
  int nblocks = context->getNBlocks(size, nthreads);
  kernelElemAddCopySat<T>
      <<<nblocks, nthreads, 0, context->getStream()>>>(dev_W, dev_A, size, dev_4params);
}
template void
elemaddcopysat<float>(const CudaContextPtr, float *, float *, const int, const param_t *);
#ifdef RPU_USE_DOUBLE
template void
elemaddcopysat<double>(const CudaContextPtr, double *, double *, const int, const param_t *);
#endif
#ifdef RPU_DEFINE_CUDA_HALF_ARRAY
template void
elemaddcopysat<half_t>(const CudaContextPtr, half_t *, half_t *, const int, const param_t *);
#endif

// MSK = P<thres
// W(MSK) = sat(A(MSK) + B(MSK))
template <typename T>
__global__ void kernelElemResetSat(
    T *dev_W,
    const int size,
    const T *dev_A,
    const float *dev_B,
    const float *dev_P,
    const T thres,
    const param_t *dev_4params) {

  bool with_A = dev_A != nullptr;
  bool with_B = dev_B != nullptr;
  bool with_P = dev_P != nullptr;

  RPU_CUDA_1D_KERNEL_LOOP(idx, size) {
    T th = thres;
    T a = (with_A) ? (T)dev_A[idx] : (T)0.0;
    T p = (with_P) ? (T)dev_P[idx] : (T)0.0;
    T b = (with_B) ? (T)dev_B[idx] : (T)0.0;
    T w = dev_W[idx];
    const param4_t parij = reinterpret_cast<const param4_t *>(dev_4params)[idx];

    if (p < th) {
      w = a + b;
      // check bounds [only those that changed]
      w = (w > (T)parij.z) ? (T)parij.z : w;
      w = (w < (T)parij.x) ? (T)parij.x : w;
      dev_W[idx] = w;
    }
  }
}
template <typename T>
void elemresetsat(
    const CudaContextPtr context,
    T *dev_W,
    const int size,
    const T *dev_A,
    const float *dev_B,
    const float *dev_P,
    T thres,
    const param_t *dev_4params) {

  int nthreads = context->getNThreads();
  int nblocks = context->getNBlocks(size, nthreads);
  kernelElemResetSat<T><<<nblocks, nthreads, 0, context->getStream()>>>(
      dev_W, size, dev_A, dev_B, dev_P, thres, dev_4params);
}
template void elemresetsat<float>(
    const CudaContextPtr,
    float *,
    const int,
    const float *,
    const float *,
    const float *,
    const float,
    const param_t *);
#ifdef RPU_USE_DOUBLE
template void elemresetsat<double>(
    const CudaContextPtr,
    double *,
    const int,
    const double *,
    const float *,
    const float *,
    const double,
    const param_t *);
#endif
#ifdef RPU_DEFINE_CUDA_HALF_ARRAY
template void elemresetsat<half_t>(
    const CudaContextPtr,
    half_t *,
    const int,
    const half_t *,
    const float *,
    const float *,
    const half_t,
    const param_t *);
#endif

// MSK = P<thres
// W(MSK) = A(MSK) + B(MSK)
template <typename T>
__global__ void kernelElemReset(
    T *dev_W,
    const int size,
    const T *dev_A,
    const float *dev_B,
    const float *dev_P,
    const T thres) {

  bool with_A = dev_A != nullptr;
  bool with_B = dev_B != nullptr;
  bool with_P = dev_P != nullptr;

  RPU_CUDA_1D_KERNEL_LOOP(idx, size) {
    T th = thres;
    T a = (with_A) ? (T)dev_A[idx] : (T)0.0;
    T p = (with_P) ? (T)dev_P[idx] : (T)0.0;
    T b = (with_B) ? (T)dev_B[idx] : (T)0.0;
    T w = dev_W[idx];

    if (p < th) {
      w = a + b;
      dev_W[idx] = w;
    }
  }
}
template <typename T>
void elemreset(
    const CudaContextPtr context,
    T *dev_W,
    const int size,
    const T *dev_A,
    const float *dev_B,
    const float *dev_P,
    T thres) {

  int nthreads = context->getNThreads();
  int nblocks = context->getNBlocks(size, nthreads);
  kernelElemReset<T>
      <<<nblocks, nthreads, 0, context->getStream()>>>(dev_W, size, dev_A, dev_B, dev_P, thres);
}
template void elemreset<float>(
    const CudaContextPtr,
    float *,
    const int,
    const float *,
    const float *,
    const float *,
    const float);
#ifdef RPU_USE_DOUBLE
template void elemreset<double>(
    const CudaContextPtr,
    double *,
    const int,
    const double *,
    const float *,
    const float *,
    const double);
#endif
#ifdef RPU_DEFINE_CUDA_HALF_ARRAY
template void elemreset<half_t>(
    const CudaContextPtr,
    half_t *,
    const int,
    const half_t *,
    const float *,
    const float *,
    const half_t);
#endif

// MSK != 0
// W(MSK) = sat(reset_bias(MSK) + std*randn())
#define RESET_TOLERANCE 1e-6
template <typename T>
__global__ void kernelElemResetSatMsk(
    T *weights,
    const int size,
    const char *msk,
    const T *reset_bias,
    const T reset_std_in,
    const param_t *dev_4params,
    curandState_t *random_states) {

  volatile unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
  T reset_std = reset_std_in;
  int total_threads = blockDim.x * gridDim.x;
  curandState_t local_state;
  bool with_bias = reset_bias != nullptr;

  if (reset_std) {
    local_state = random_states[tid];
  }

  for (int idx = tid; idx < size; idx += total_threads) {

    bool reset_if = msk[idx] != 0;

    if (reset_if) {
      // assume very sparse reset thus only read if reset
      T bias = with_bias ? reset_bias[idx] : (T)0.0;
      T w;
      const param4_t parij = reinterpret_cast<const param4_t *>(dev_4params)[tid];
      if (reset_std) {
        w = bias + reset_std * (T)curand_normal(&local_state);
      } else {
        w = bias;
      }
      w = (w > (T)parij.z) ? (T)parij.z : w;
      w = (w < (T)parij.x) ? (T)parij.x : w;
      weights[idx] = w;
    }
  }
  if (reset_std) {
    random_states[tid] = local_state;
  }
}
#undef RESET_TOLERANCE
template <typename T>
void elemresetsatmsk(
    CudaContextPtr context,
    T *W,
    const int size,
    const char *msk,
    const T *reset_bias,
    const T reset_std,
    const param_t *dev_4params) {

  int nthreads = context->getNThreads();
  int nblocks_batch_max = context->getSMCount() * (context->maxThreadsPerBlock() / nthreads);
  int nblocks = MIN(context->getNBlocks(size, nthreads), nblocks_batch_max);
  kernelElemResetSatMsk<T><<<nblocks, nthreads, 0, context->getStream()>>>(
      W, size, msk, reset_bias, reset_std, dev_4params,
      context->getRandomStates(nblocks * nthreads));
}
template void elemresetsatmsk<float>(
    CudaContextPtr, float *, const int, const char *, const float *, const float, const param_t *);
#ifdef RPU_USE_DOUBLE
template void elemresetsatmsk<double>(
    CudaContextPtr,
    double *,
    const int,
    const char *,
    const double *,
    const double,
    const param_t *);
#endif
#ifdef RPU_DEFINE_CUDA_HALF_ARRAY
template void elemresetsatmsk<half_t>(
    CudaContextPtr,
    half_t *,
    const int,
    const char *,
    const half_t *,
    const half_t,
    const param_t *);
#endif

// A = W - A_in; W = A_in;
template <typename T>
__global__ void kernelElemSubCopy(T *dev_W, T *dev_A, const int size, const T scale) {

  RPU_CUDA_1D_KERNEL_LOOP(idx, size) {
    T new_w = dev_W[idx];
    T old_w = dev_A[idx];
    dev_W[idx] = old_w;
    dev_A[idx] = scale * (new_w - old_w); // dw
  }
}
template <typename T>
void elemsubcopy(const CudaContextPtr context, T *dev_W, T *dev_A, const int size, const T scale) {

  int nthreads = context->getNThreads();
  int nblocks = context->getNBlocks(size, nthreads);
  kernelElemSubCopy<T><<<nblocks, nthreads, 0, context->getStream()>>>(dev_W, dev_A, size, scale);
}
template void elemsubcopy<float>(const CudaContextPtr, float *, float *, const int, const float);
#ifdef RPU_USE_DOUBLE
template void
elemsubcopy<double>(const CudaContextPtr, double *, double *, const int, const double);
#endif
#ifdef RPU_DEFINE_CUDA_HALF_ARRAY
template void
elemsubcopy<half_t>(const CudaContextPtr, half_t *, half_t *, const int, const half_t);
#endif

// set all elements to a
template <typename T> __global__ void kernelSetConstAlpha(T *values, int size, T alpha) {
  RPU_CUDA_1D_KERNEL_LOOP(idx, size) { values[idx] = alpha; }
}

template <typename T>
void elemconst(const CudaContextPtr context, T *X, const int size, const T alpha) {
  int nthreads = context->getNThreads();
  int nblocks = context->getNBlocks(size, nthreads);
  kernelSetConstAlpha<<<nblocks, nthreads, 0, context->getStream()>>>(X, size, alpha);
}
template void elemconst<float>(const CudaContextPtr, float *, const int, const float);
#ifdef RPU_USE_DOUBLE
template void elemconst<double>(const CudaContextPtr, double *, const int, const double);
#endif
#ifdef RPU_DEFINE_CUDA_HALF_ARRAY
template void elemconst<half_t>(const CudaContextPtr, half_t *, const int, const half_t);
#endif

template void elemconst<uint32_t>(const CudaContextPtr, uint32_t *, const int, const uint32_t);
template void elemconst<uint64_t>(const CudaContextPtr, uint64_t *, const int, const uint64_t);
template void elemconst<int>(const CudaContextPtr, int *, const int, const int);
template void elemconst<char>(const CudaContextPtr, char *, const int, const char);
template void elemconst<int8_t>(const CudaContextPtr, int8_t *, const int, const int8_t);

template <typename T>
void elemconst(const CudaContext *context, T *X, const int size, const T alpha) {
  int nthreads = context->getNThreads();
  int nblocks = context->getNBlocks(size, nthreads);
  kernelSetConstAlpha<<<nblocks, nthreads, 0, context->getStream()>>>(X, size, alpha);
}
template void elemconst<float>(const CudaContext *, float *, const int, const float);
#ifdef RPU_USE_DOUBLE
template void elemconst<double>(const CudaContext *, double *, const int, const double);
#endif
#ifdef RPU_DEFINE_CUDA_HALF_ARRAY
template void elemconst<half_t>(const CudaContext *, half_t *, const int, const half_t);
#endif

template void elemconst<uint32_t>(const CudaContext *, uint32_t *, const int, const uint32_t);
template void elemconst<uint64_t>(const CudaContext *, uint64_t *, const int, const uint64_t);
template void elemconst<int>(const CudaContext *, int *, const int, const int);
template void elemconst<char>(const CudaContext *, char *, const int, const char);

// add and update pink flicker noise
// add and update pink flicker noise with weight reset (judged by comparing with buffer)
template <typename T>
__global__ void kernelAddFlickerNoiseWreset(
    T *W,
    T *Wb,
    int size_in,
    T amp_in,
    int n_in,
    T alpha_in,
    T q_in,
    T reset_H_in,
    T wreset_tol_in,
    uint64_t *flicker_states,
    curandState *rnd_states) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  T alpha = alpha_in;
  T amp = amp_in;
  int n = n_in;
  int size = size_in;
  T q0 = q_in;
  T q1 = (T)1. - q0;
  T use_wreset = Wb != nullptr; // set Wb to nullptr to avoid wreset
  T wreset_tol = wreset_tol_in;
  T reset_H = reset_H_in;

  if (tid < size) {
    uint64_t flicker_state = flicker_states[tid];
    curandState local_state = rnd_states[tid];
    T value = W[tid];
    bool wreset = false;

    if (use_wreset) {
      wreset = fabs(value - Wb[tid]) > wreset_tol;
    }

    // get last noise: we only update the difference so that the weight itself evolves  1/f
    T last_noise = (T)__popcll(flicker_state);

    // update flicker states
    T prob0 = q0;
    T prob1 = q1;

    if (wreset & reset_H <= (T)0.0) { // only in EQ reset case  (reset_H==0)
      flicker_state = 0;              // will take q0 then to flip to 1
    }

    // update flicker states (in one loop probably better for warp)
    T alpha_k = 1.;
    for (int i = 0; i < n; i++) {
      T stoch_value = curand_uniform(&local_state);
      uint64_t cbit = (uint64_t)1 << (uint64_t)i;
      flicker_state ^= ((flicker_state & cbit) > 0) ? ((stoch_value < prob1) ? cbit : (uint64_t)0)
                                                    : ((stoch_value < prob0) ? cbit : (uint64_t)0);

      alpha_k /= alpha;
      if (!wreset) {
        prob0 = q0 * alpha_k;
        prob1 = q1 * alpha_k;
      } else {
        prob0 = reset_H > (T)0.0 ? (T)0.0 : q0;   // zero remains zero in case of H
        prob1 = (T)1.0 - exp(-reset_H * alpha_k); // prob1 not used if reset_H==0
      }
    }
    rnd_states[tid] = local_state;
    flicker_states[tid] = flicker_state;

    // update the weight
    T noise = (T)__popcll(flicker_state);
    value += amp * (noise - last_noise);

    // save weight
    W[tid] = value;

    if (use_wreset) {
      Wb[tid] = value; // save copy for next round
    }
  }
}

template <typename T>
void elemaddpinknoise(
    const CudaContextPtr context,
    T *W,
    T *Wb, // set to nullptr if no reset
    const int size,
    const T rate,
    const int n_flicker,
    const T flicker_r,
    const T flicker_q,
    const T flicker_h, // reset time constant . zero for reset all
    const T flicker_wreset_tol,
    uint64_t *flicker_states,
    curandState *rnd_states) {
  int nthreads = context->getNThreads();
  int nblocks = context->getNBlocks(size, nthreads);

  // const float PI_F = 3.14159265358979f;
  // const float LN_2 = 0.69314718056f;
  T amp = sqrtf((T)logf(flicker_r) * rate / (flicker_q * ((T)1.0 - flicker_q)));

  kernelAddFlickerNoiseWreset<<<nblocks, nthreads, 0, context->getStream()>>>(
      W, Wb, size, amp, n_flicker, flicker_r, flicker_q, flicker_h, flicker_wreset_tol,
      flicker_states, rnd_states);
}

template void elemaddpinknoise<float>(
    const CudaContextPtr,
    float *,
    float *,
    const int,
    const float,
    const int,
    const float,
    const float,
    const float,
    const float,
    uint64_t *,
    curandState *);
#ifdef RPU_USE_DOUBLE
template void elemaddpinknoise<double>(
    const CudaContextPtr,
    double *,
    double *,
    const int,
    const double,
    const int,
    const double,
    const double,
    const double,
    const double,
    uint64_t *,
    curandState *);
#endif
#ifdef RPU_DEFINE_CUDA_HALF_ARRAY
template void elemaddpinknoise<half_t>(
    const CudaContextPtr,
    half_t *,
    half_t *,
    const int,
    const half_t,
    const int,
    const half_t,
    const half_t,
    const half_t,
    const half_t,
    uint64_t *,
    curandState *);
#endif

// w = max(min(w,|a|),-|a|)
template <typename T> __global__ void kernelAClip(T *values, int size, T abs_a) {
  RPU_CUDA_1D_KERNEL_LOOP(idx, size) { values[idx] = MIN(MAX(values[idx], -abs_a), abs_a); }
}

template <typename T> void aclip(const CudaContextPtr context, T *W, const int size, const T a) {

  int nthreads = context->getNThreads();
  int nblocks = context->getNBlocks(size, nthreads);
  kernelAClip<<<nblocks, nthreads, 0, context->getStream()>>>(W, size, (T)fabsf(a));
}
template void aclip<float>(const CudaContextPtr, float *, const int, const float);
#ifdef RPU_USE_DOUBLE
template void aclip<double>(const CudaContextPtr, double *, const int, const double);
#endif
#ifdef RPU_DEFINE_CUDA_HALF_ARRAY
template void aclip<half_t>(const CudaContextPtr, half_t *, const int, const half_t);
#endif

// w = max(w,a)
template <typename T> __global__ void kernelElemmax(T *values, int size, T a, const T *in_values) {
  RPU_CUDA_1D_KERNEL_LOOP(idx, size) { values[idx] = MAX(in_values[idx], a); }
}

template <typename T>
void elemmax(const CudaContextPtr context, T *W, const int size, const T a, const T *in_values) {

  int nthreads = context->getNThreads();
  int nblocks = context->getNBlocks(size, nthreads);
  kernelElemmax<<<nblocks, nthreads, 0, context->getStream()>>>(
      W, size, a, in_values != nullptr ? in_values : W);
}
template void elemmax<float>(const CudaContextPtr, float *, const int, const float, const float *);
#ifdef RPU_USE_DOUBLE
template void
elemmax<double>(const CudaContextPtr, double *, const int, const double, const double *);
#endif
#ifdef RPU_DEFINE_CUDA_HALF_ARRAY
template void
elemmax<half_t>(const CudaContextPtr, half_t *, const int, const half_t, const half_t *);
#endif

// w = min(w,a)
template <typename T> __global__ void kernelElemmin(T *values, int size, T a, const T *in_values) {
  RPU_CUDA_1D_KERNEL_LOOP(idx, size) { values[idx] = MIN(in_values[idx], a); }
}

template <typename T>
void elemmin(const CudaContextPtr context, T *W, const int size, const T a, const T *in_values) {

  int nthreads = context->getNThreads();
  int nblocks = context->getNBlocks(size, nthreads);
  kernelElemmin<<<nblocks, nthreads, 0, context->getStream()>>>(
      W, size, a, in_values != nullptr ? in_values : W);
}
template void elemmin<float>(const CudaContextPtr, float *, const int, const float, const float *);
#ifdef RPU_USE_DOUBLE
template void
elemmin<double>(const CudaContextPtr, double *, const int, const double, const double *);
#endif
#ifdef RPU_DEFINE_CUDA_HALF_ARRAY
template void
elemmin<half_t>(const CudaContextPtr, half_t *, const int, const half_t, const half_t *);
#endif

// w = w<a?0:w elementwise
template <typename T> __global__ void kernelElemSetBelowZero(T *values, int size, T a) {
  RPU_CUDA_1D_KERNEL_LOOP(idx, size) {
    T v = values[idx];
    values[idx] = v < a ? (T)0.0 : v;
  }
}

template <typename T>
void elemsetbelowzero(const CudaContextPtr context, T *W, const int size, const T a) {

  int nthreads = context->getNThreads();
  int nblocks = context->getNBlocks(size, nthreads);
  kernelElemSetBelowZero<<<nblocks, nthreads, 0, context->getStream()>>>(W, size, a);
}
template void elemsetbelowzero<float>(const CudaContextPtr, float *, const int, const float);
#ifdef RPU_USE_DOUBLE
template void elemsetbelowzero<double>(const CudaContextPtr, double *, const int, const double);
#endif
#ifdef RPU_DEFINE_CUDA_HALF_ARRAY
template void elemsetbelowzero<half_t>(const CudaContextPtr, half_t *, const int, const half_t);
#endif

// w[j] = a*A[j] + b*B[j]
// with fast paths for A-B and -B+A and A+B.
template <typename T, typename T_B>
__global__ void kernelElemAdd(T *dev_W, const int size, const T *dev_A, const T_B *dev_B) {

  RPU_CUDA_1D_KERNEL_LOOP(idx, size) { dev_W[idx] = dev_A[idx] + (T)dev_B[idx]; }
}
template <typename T, typename T_A, typename T_B>
__global__ void kernelElemSub(T *dev_W, const int size, const T_A *dev_A, const T_B *dev_B) {

  RPU_CUDA_1D_KERNEL_LOOP(idx, size) { dev_W[idx] = (T)dev_A[idx] - (T)dev_B[idx]; }
}

template <typename T, typename T_B>
__global__ void kernelElemWeightedSum(
    T *dev_W, const int size, const T *dev_A, const T a, const T_B *dev_B, const T b) {

  RPU_CUDA_1D_KERNEL_LOOP(idx, size) { dev_W[idx] = a * dev_A[idx] + b * (T)dev_B[idx]; }
}

template <typename T, typename T_B>
void elemweightedsum(
    const CudaContextPtr context,
    T *dev_W,
    const int size,
    const T *dev_A,
    const T a,
    const T_B *dev_B,
    const T b) {

  int nthreads = context->getNThreads();
  int nblocks = context->getNBlocks(size, nthreads);
  if (a == (T)1.0 && b == (T)1.0) {
    kernelElemAdd<T, T_B>
        <<<nblocks, nthreads, 0, context->getStream()>>>(dev_W, size, dev_A, dev_B);
  } else if (a == (T)1.0 && b == (T)-1.0) {
    kernelElemSub<T, T, T_B>
        <<<nblocks, nthreads, 0, context->getStream()>>>(dev_W, size, dev_A, dev_B);
  } else if (a == (T)-1.0 && b == (T)1.0) {
    kernelElemSub<T, T_B, T>
        <<<nblocks, nthreads, 0, context->getStream()>>>(dev_W, size, dev_B, dev_A);
  } else {
    kernelElemWeightedSum<T, T_B>
        <<<nblocks, nthreads, 0, context->getStream()>>>(dev_W, size, dev_A, a, dev_B, b);
  }
}
template void elemweightedsum<float, float>(
    const CudaContextPtr,
    float *,
    const int,
    const float *,
    const float,
    const float *,
    const float);
#ifdef RPU_USE_DOUBLE
template void elemweightedsum<double, double>(
    const CudaContextPtr,
    double *,
    const int,
    const double *,
    const double,
    const double *,
    const double);
template void elemweightedsum<double, float>(
    const CudaContextPtr,
    double *,
    const int,
    const double *,
    const double,
    const float *,
    const double);
#endif
#ifdef RPU_DEFINE_CUDA_HALF_ARRAY
template void elemweightedsum<half_t, half_t>(
    const CudaContextPtr,
    half_t *,
    const int,
    const half_t *,
    const half_t,
    const half_t *,
    const half_t);
template void elemweightedsum<half_t, float>(
    const CudaContextPtr,
    half_t *,
    const int,
    const half_t *,
    const half_t,
    const float *,
    const half_t);
#endif

// w[j] = sum_i^n(m_i[j])/n
template <typename T>
__global__ void kernelElemAverage(T *dev_W, const int size, T **dev_Ms, const int m) {

  RPU_CUDA_1D_KERNEL_LOOP(idx, size) {
    T w = 0;
    for (int i = 0; i < m; i++) {
      const T *w1 = dev_Ms[i];
      w += w1[idx];
    }
    dev_W[idx] = w / (T)m;
  }
}

template <typename T>
void elemaverage(const CudaContextPtr context, T *dev_W, const int size, T **dev_Ms, const int m) {

  int nthreads = context->getNThreads();
  int nblocks = context->getNBlocks(size, nthreads);
  kernelElemAverage<T><<<nblocks, nthreads, 0, context->getStream()>>>(dev_W, size, dev_Ms, m);
}
template void elemaverage<float>(const CudaContextPtr, float *, const int, float **, const int);
#ifdef RPU_USE_DOUBLE
template void elemaverage<double>(const CudaContextPtr, double *, const int, double **, const int);
#endif
#ifdef RPU_DEFINE_CUDA_HALF_ARRAY
template void elemaverage<half_t>(const CudaContextPtr, half_t *, const int, half_t **, const int);
#endif

// permute (1,3,2)
template <typename T>
__global__ void kernelPermute132Bias(
    T *dev_X_out,
    const T *dev_X_in,
    const int size,
    const int d1_in,
    const int d2, // with bias added
    const int d3,
    const bool bias) {

  volatile unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;

  int d1 = d1_in;
  int b = (bias) ? 1 : 0;
  int sz = size;
  int d13 = d1 * d3;
  int d2wob = d2 - b;
  int szwob = (b == 1) ? sz - d13 : sz;
  int d12 = d1 * d2wob;
  int idx1 = tid % d1;
  int idx2 = (tid / d1) % d2wob;
  int idx3 = tid / d12;

  if (tid < szwob) {
    T x = dev_X_in[tid];
    dev_X_out[idx2 * d13 + idx3 * d1 + idx1] = x;
  } else if (tid < sz) {
    dev_X_out[tid] = 1.0;
  }
}

template <typename T>
void permute132(
    const CudaContextPtr context,
    T *X_out,
    const T *X_in,
    const int d1,
    const int d2, // with bias added
    const int d3,
    const bool bias) {

  int nthreads = context->getNThreads();
  int nblocks = context->getNBlocks(d1 * d2 * d3, nthreads);
  kernelPermute132Bias<T>
      <<<nblocks, nthreads, 0, context->getStream()>>>(X_out, X_in, d1 * d2 * d3, d1, d2, d3, bias);
}

template void permute132<float>(
    const CudaContextPtr, float *, const float *, const int, const int, const int, const bool);
#ifdef RPU_USE_DOUBLE
template void permute132<double>(
    const CudaContextPtr, double *, const double *, const int, const int, const int, const bool);
#endif
#ifdef RPU_DEFINE_CUDA_HALF_ARRAY
template void permute132<half_t>(
    const CudaContextPtr, half_t *, const half_t *, const int, const int, const int, const bool);
#endif

// copy with bias
template <typename T>
__global__ void
kernelCopyBiasTrans(T *dest_values, const T *source_values, int size, int size_without_bias) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < size_without_bias) {
    dest_values[tid] = source_values[tid];
  } else if (tid < size) {
    dest_values[tid] = (T)1.;
  }
}

template <typename T>
__global__ void kernelCopyBias(T *dest_values, const T *source_values, int size, int ld) {

  // might be somewhat inefficient because of offset memory block access...
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  int ldmod = (tid + 1) % ld;
  int lddiv = tid / ld;
  int idx = tid - lddiv;
  if (tid < size) {
    if (ldmod == 0)
      dest_values[tid] = (T)1.0;
    else
      dest_values[tid] = source_values[idx];
  }
}

template <typename T>
void makeBias(
    const CudaContextPtr context,
    T *x_with_bias,
    const T *x_without_bias,
    const int x_size,
    const int m_batch,
    const bool trans) {

  cudaStream_t s = context->getStream();
  int size = m_batch * x_size;
  int nthreads = context->getNThreads();
  int nblocks = context->getNBlocks(size, nthreads);

  if (trans) {
    int ld = m_batch;
    // put ld ones at end
    kernelCopyBiasTrans<T>
        <<<nblocks, nthreads, 0, s>>>(x_with_bias, x_without_bias, size, size - ld);
  } else {
    // put single one each ld-1 distance: [...(x_size-1)..,1,...(x_size-1)...,1,...]
    kernelCopyBias<T><<<nblocks, nthreads, 0, s>>>(x_with_bias, x_without_bias, size, x_size);
  }
}
template void
makeBias<float>(const CudaContextPtr, float *, const float *, const int, const int, const bool);
template void
makeBias<int>(const CudaContextPtr, int *, const int *, const int, const int, const bool);
#ifdef RPU_USE_DOUBLE
template void
makeBias<double>(const CudaContextPtr, double *, const double *, const int, const int, const bool);
#endif
#ifdef RPU_DEFINE_CUDA_HALF_ARRAY
template void
makeBias<half_t>(const CudaContextPtr, half_t *, const half_t *, const int, const int, const bool);
#endif

// copy without bias (backward)
template <typename T>
void copyWithoutBias(
    const CudaContextPtr context,
    T *x_without_bias,
    const T *x_with_bias,
    const int x_size,
    const int m_batch,
    const bool trans) {

  cudaStream_t s = context->getStream();
  // context_->synchronize(); // seems to be necessary ?!? (at least during nvprof)
  if (trans) {
    // m_batch first
    int sz = m_batch * (x_size - 1);

    CUDA_CALL(cudaMemcpyAsync(
        (void *)x_without_bias, (const void *)x_with_bias, sizeof(T) * sz, cudaMemcpyDeviceToDevice,
        s));
  } else {
    // x_size first
    int w = sizeof(T) * (x_size);
    int wm1 = w - sizeof(T);
    CUDA_CALL(cudaMemcpy2DAsync(
        (void *)x_without_bias,    // dst
        wm1,                       // dpitch
        (const void *)x_with_bias, // src,
        w,                         // spitch,
        wm1,                       // width in bytes
        m_batch,                   // height
        cudaMemcpyDeviceToDevice, s));
  }
}

template void copyWithoutBias<float>(
    const CudaContextPtr, float *, const float *, const int, const int, const bool);
#ifdef RPU_USE_DOUBLE
template void copyWithoutBias<double>(
    const CudaContextPtr, double *, const double *, const int, const int, const bool);
#endif
#ifdef RPU_DEFINE_CUDA_HALF_ARRAY
template void copyWithoutBias<half_t>(
    const CudaContextPtr, half_t *, const half_t *, const int, const int, const bool);
#endif

// addWithIterator
template <typename OutputIteratorT, typename T>
__global__ void kernelAddWithIterator(
    OutputIteratorT out_tensor, const T *in_tensor_a, const T *in_tensor_b, const int sz_all) {

  RPU_CUDA_1D_KERNEL_LOOP(idx, sz_all) { out_tensor[idx] = in_tensor_a[idx] + in_tensor_b[idx]; }
}

template <typename OutputIteratorT, typename T>
void addWithIterator(
    const CudaContextPtr context,
    OutputIteratorT out_tensor,
    const T *in_tensor_a,
    const T *in_tensor_b,
    const int total_input_size) {

  int nthreads = context->getNThreads();
  int nblocks = context->getNBlocks(total_input_size, nthreads);

  kernelAddWithIterator<OutputIteratorT, T><<<nblocks, nthreads, 0, context->getStream()>>>(
      out_tensor, in_tensor_a, in_tensor_b, total_input_size);
}

#define RPU_CMU_DEFINE_AWI(OUTPUTITER, T)                                                          \
  template void addWithIterator<OUTPUTITER, T>(                                                    \
      const CudaContextPtr, OUTPUTITER, const T *, const T *, const int);

#define TRANS_FLOAT(TRANS) TRANS, float
RPU_CMU_DEFINE_AWI(float *, float);
RPU_CMU_DEFINE_AWI(PermuterTransOutputIterator<float>, float);
RPU_CMU_DEFINE_AWI(IndexReaderOutputIterator<float>, float);
RPU_CMU_DEFINE_AWI(IndexReaderTransOutputIterator<float>, float);
RPU_CMU_DEFINE_AWI(IndexReaderSliceOutputIterator<TRANS_FLOAT(true)>, float);
RPU_CMU_DEFINE_AWI(IndexReaderSliceOutputIterator<TRANS_FLOAT(false)>, float);
RPU_CMU_DEFINE_AWI(SliceOutputIterator<TRANS_FLOAT(true)>, float);
RPU_CMU_DEFINE_AWI(SliceOutputIterator<TRANS_FLOAT(false)>, float);

#undef TRANS_FLOAT

#ifdef RPU_USE_DOUBLE
#define TRANS_DOUBLE(TRANS) TRANS, double
RPU_CMU_DEFINE_AWI(double *, double);
RPU_CMU_DEFINE_AWI(PermuterTransOutputIterator<double>, double);
RPU_CMU_DEFINE_AWI(IndexReaderOutputIterator<double>, double);
RPU_CMU_DEFINE_AWI(IndexReaderTransOutputIterator<double>, double);
RPU_CMU_DEFINE_AWI(IndexReaderSliceOutputIterator<TRANS_DOUBLE(true)>, double);
RPU_CMU_DEFINE_AWI(IndexReaderSliceOutputIterator<TRANS_DOUBLE(false)>, double);
RPU_CMU_DEFINE_AWI(SliceOutputIterator<TRANS_DOUBLE(true)>, double);
RPU_CMU_DEFINE_AWI(SliceOutputIterator<TRANS_DOUBLE(false)>, double);
#undef TRANS_DOUBLE
#endif

#ifdef RPU_DEFINE_CUDA_HALF_ARRAY
#define TRANS_HALF(TRANS) TRANS, half_t
RPU_CMU_DEFINE_AWI(half_t *, half_t);
RPU_CMU_DEFINE_AWI(PermuterTransOutputIterator<half_t>, half_t);
RPU_CMU_DEFINE_AWI(IndexReaderOutputIterator<half_t>, half_t);
RPU_CMU_DEFINE_AWI(IndexReaderTransOutputIterator<half_t>, half_t);
RPU_CMU_DEFINE_AWI(IndexReaderSliceOutputIterator<TRANS_HALF(true)>, half_t);
RPU_CMU_DEFINE_AWI(IndexReaderSliceOutputIterator<TRANS_HALF(false)>, half_t);
RPU_CMU_DEFINE_AWI(SliceOutputIterator<TRANS_HALF(true)>, half_t);
RPU_CMU_DEFINE_AWI(SliceOutputIterator<TRANS_HALF(false)>, half_t);
#undef TRANS_HALF
#endif

#undef RPU_CMU_DEFINE_CWI

// copyWithIterator
template <typename OutputIteratorT, typename InputIteratorT>
__global__ void kernelCopyWithIterator(
    OutputIteratorT out_tensor, const InputIteratorT in_tensor, const int sz_all) {

  RPU_CUDA_1D_KERNEL_LOOP(idx, sz_all) { out_tensor[idx] = in_tensor[idx]; }
}

template <typename OutputIteratorT, typename InputIteratorT>
void copyWithIterator(
    const CudaContextPtr context,
    OutputIteratorT out_tensor,
    InputIteratorT in_tensor,
    const int total_input_size) {

  int nthreads = context->getNThreads();
  int nblocks = context->getNBlocks(total_input_size, nthreads);

  kernelCopyWithIterator<OutputIteratorT, InputIteratorT>
      <<<nblocks, nthreads, 0, context->getStream()>>>(out_tensor, in_tensor, total_input_size);
}

#define RPU_CMU_DEFINE_CWI(OUTPUTITER, INPUTITER)                                                  \
  template void copyWithIterator<OUTPUTITER, INPUTITER>(                                           \
      const CudaContextPtr, OUTPUTITER, INPUTITER, const int);
#define TRANS_FLOAT(TRANS) TRANS, float
#define COMMA ,

RPU_CMU_DEFINE_CWI(float *, const float *);
RPU_CMU_DEFINE_CWI(float *, float *);
RPU_CMU_DEFINE_CWI(float *, IndexReaderInputIterator<float>);
RPU_CMU_DEFINE_CWI(float *, IndexReaderTransInputIterator<float>);
RPU_CMU_DEFINE_CWI(float *, PermuterTransInputIterator<float>);
RPU_CMU_DEFINE_CWI(float *, IndexReaderSliceInputIterator<TRANS_FLOAT(true)>);
RPU_CMU_DEFINE_CWI(float *, IndexReaderSliceInputIterator<TRANS_FLOAT(false)>);
RPU_CMU_DEFINE_CWI(float *, SliceInputIterator<TRANS_FLOAT(true)>);
RPU_CMU_DEFINE_CWI(float *, SliceInputIterator<TRANS_FLOAT(false)>);
RPU_CMU_DEFINE_CWI(float *, DiagInputIterator<float COMMA chop_t>);
RPU_CMU_DEFINE_CWI(float *, EyeInputIterator<float>);
RPU_CMU_DEFINE_CWI(PermuterTransOutputIterator<float>, const float *);
RPU_CMU_DEFINE_CWI(IndexReaderOutputIterator<float>, const float *);
RPU_CMU_DEFINE_CWI(IndexReaderTransOutputIterator<float>, const float *);
RPU_CMU_DEFINE_CWI(IndexReaderSliceOutputIterator<TRANS_FLOAT(true)>, const float *);
RPU_CMU_DEFINE_CWI(IndexReaderSliceOutputIterator<TRANS_FLOAT(false)>, const float *);
RPU_CMU_DEFINE_CWI(SliceOutputIterator<TRANS_FLOAT(true)>, const float *);
RPU_CMU_DEFINE_CWI(SliceOutputIterator<TRANS_FLOAT(false)>, const float *);

#undef TRANS_FLOAT

#ifdef RPU_USE_DOUBLE
#define TRANS_DOUBLE(TRANS) TRANS, double
RPU_CMU_DEFINE_CWI(double *, const double *);
RPU_CMU_DEFINE_CWI(double *, double *);
RPU_CMU_DEFINE_CWI(double *, IndexReaderInputIterator<double>);
RPU_CMU_DEFINE_CWI(double *, IndexReaderTransInputIterator<double>);
RPU_CMU_DEFINE_CWI(double *, PermuterTransInputIterator<double>);
RPU_CMU_DEFINE_CWI(double *, IndexReaderSliceInputIterator<TRANS_DOUBLE(true)>);
RPU_CMU_DEFINE_CWI(double *, IndexReaderSliceInputIterator<TRANS_DOUBLE(false)>);
RPU_CMU_DEFINE_CWI(double *, SliceInputIterator<TRANS_DOUBLE(true)>);
RPU_CMU_DEFINE_CWI(double *, SliceInputIterator<TRANS_DOUBLE(false)>);
RPU_CMU_DEFINE_CWI(double *, DiagInputIterator<double COMMA chop_t>);
RPU_CMU_DEFINE_CWI(double *, EyeInputIterator<double>);
RPU_CMU_DEFINE_CWI(PermuterTransOutputIterator<double>, const double *);
RPU_CMU_DEFINE_CWI(IndexReaderOutputIterator<double>, const double *);
RPU_CMU_DEFINE_CWI(IndexReaderTransOutputIterator<double>, const double *);
RPU_CMU_DEFINE_CWI(IndexReaderSliceOutputIterator<TRANS_DOUBLE(true)>, const double *);
RPU_CMU_DEFINE_CWI(IndexReaderSliceOutputIterator<TRANS_DOUBLE(false)>, const double *);
RPU_CMU_DEFINE_CWI(SliceOutputIterator<TRANS_DOUBLE(true)>, const double *);
RPU_CMU_DEFINE_CWI(SliceOutputIterator<TRANS_DOUBLE(false)>, const double *);

#undef TRANS_DOUBLE
#endif

#ifdef RPU_DEFINE_CUDA_HALF_ARRAY
#define TRANS_HALF(TRANS) TRANS, half_t
RPU_CMU_DEFINE_CWI(half_t *, const half_t *);
RPU_CMU_DEFINE_CWI(half_t *, half_t *);
RPU_CMU_DEFINE_CWI(half_t *, IndexReaderInputIterator<half_t>);
RPU_CMU_DEFINE_CWI(half_t *, IndexReaderTransInputIterator<half_t>);
RPU_CMU_DEFINE_CWI(half_t *, PermuterTransInputIterator<half_t>);
RPU_CMU_DEFINE_CWI(half_t *, IndexReaderSliceInputIterator<TRANS_HALF(true)>);
RPU_CMU_DEFINE_CWI(half_t *, IndexReaderSliceInputIterator<TRANS_HALF(false)>);
RPU_CMU_DEFINE_CWI(half_t *, SliceInputIterator<TRANS_HALF(true)>);
RPU_CMU_DEFINE_CWI(half_t *, SliceInputIterator<TRANS_HALF(false)>);
RPU_CMU_DEFINE_CWI(half_t *, DiagInputIterator<half_t COMMA chop_t>);
RPU_CMU_DEFINE_CWI(half_t *, EyeInputIterator<half_t>);
RPU_CMU_DEFINE_CWI(PermuterTransOutputIterator<half_t>, const half_t *);
RPU_CMU_DEFINE_CWI(IndexReaderOutputIterator<half_t>, const half_t *);
RPU_CMU_DEFINE_CWI(IndexReaderTransOutputIterator<half_t>, const half_t *);
RPU_CMU_DEFINE_CWI(IndexReaderSliceOutputIterator<TRANS_HALF(true)>, const half_t *);
RPU_CMU_DEFINE_CWI(IndexReaderSliceOutputIterator<TRANS_HALF(false)>, const half_t *);
RPU_CMU_DEFINE_CWI(SliceOutputIterator<TRANS_HALF(true)>, const half_t *);
RPU_CMU_DEFINE_CWI(SliceOutputIterator<TRANS_HALF(false)>, const half_t *);

#undef TRANS_HALF
#endif

#undef RPU_CMU_DEFINE_CWI

// fake cast to overcome constexpr lacking
template <> const float *fakeCastConst<float, const float *>(const float *X) { return X; };
template <> float *fakeCast<float, float *>(float *X) { return X; };
#ifdef RPU_USE_DOUBLE
template <> const double *fakeCastConst<double, const double *>(const double *X) { return X; };
template <> double *fakeCast<double, double *>(double *X) { return X; };
#endif
#ifdef RPU_DEFINE_CUDA_HALF_ARRAY
template <> const half_t *fakeCastConst<half_t, const half_t *>(const half_t *X) { return X; };
template <> half_t *fakeCast<half_t, half_t *>(half_t *X) { return X; };
#endif

} // namespace math
} // namespace RPU
