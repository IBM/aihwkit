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
    const CudaContext *context,
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
    const CudaContext *context,
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

template <>
int iamax<float>(const CudaContext *context, const int N, const float *X, const int incX) {
  RPU_GET_CUBLAS_HANDLE;
  RPU_SET_CUBLAS_POINTER_MODE_HOST;
  int result = 0;
  CUBLAS_CALL(cublasIsamax(handle, N, X, incX, &result));
  RPU_RESTORE_CUBLAS_POINTER_MODE;
  return result - 1; // make 0 based index !!
};

template <>
int iamax<double>(const CudaContext *context, const int N, const double *X, const int incX) {
  RPU_GET_CUBLAS_HANDLE;
  RPU_SET_CUBLAS_POINTER_MODE_HOST;
  int result;
  CUBLAS_CALL(cublasIdamax(handle, N, X, incX, &result));
  RPU_RESTORE_CUBLAS_POINTER_MODE;
  return result - 1; // make 0 based index
};

template <>
void copy<float>(
    const CudaContext *context,
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
    const CudaContext *context,
    const int N,
    const double *X,
    const int incX,
    double *Y,
    const int incY) {
  RPU_GET_CUBLAS_HANDLE;
  CUBLAS_CALL(cublasDcopy(handle, N, X, incX, Y, incY));
}

template <>
void scal<float>(
    const CudaContext *context, const int N, const float alpha, float *X, const int incX) {
  RPU_GET_CUBLAS_HANDLE;
  RPU_SET_CUBLAS_POINTER_MODE_HOST;
  CUBLAS_CALL(cublasSscal(handle, N, &alpha, X, incX));
  RPU_RESTORE_CUBLAS_POINTER_MODE;
}

template <>
void scal<double>(
    const CudaContext *context, const int N, const double alpha, double *X, const int incX) {
  RPU_GET_CUBLAS_HANDLE;
  RPU_SET_CUBLAS_POINTER_MODE_HOST;
  CUBLAS_CALL(cublasDscal(handle, N, &alpha, X, incX));
  RPU_RESTORE_CUBLAS_POINTER_MODE;
}

template <>
void nrm2<float>(
    const CudaContext *context, const int N, const float *X, const int incX, float *res) {
  RPU_GET_CUBLAS_HANDLE;
  RPU_SET_CUBLAS_POINTER_MODE_DEVICE;
  CUBLAS_CALL(cublasSnrm2(handle, N, X, incX, res));
  RPU_RESTORE_CUBLAS_POINTER_MODE;
}

template <>
void nrm2<double>(
    const CudaContext *context, const int N, const double *X, const int incX, double *res) {
  RPU_GET_CUBLAS_HANDLE;
  RPU_SET_CUBLAS_POINTER_MODE_DEVICE;
  CUBLAS_CALL(cublasDnrm2(handle, N, X, incX, res));
  RPU_RESTORE_CUBLAS_POINTER_MODE;
}

template <>
void gemv<float>(
    const CudaContext *context,
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
    const CudaContext *context,
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

template <>
void ger<float>(
    const CudaContext *context,
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
    const CudaContext *context,
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

// W += A
template <typename T> __global__ void kernelElemAdd(T *dev_W, const int size, const T *dev_A) {

  RPU_CUDA_1D_KERNEL_LOOP(idx, size) { dev_W[idx] += dev_A[idx]; }
}
template <typename T>
void elemadd(const CudaContext *context, T *dev_W, const int size, const T *dev_A) {

  int nthreads = context->getNThreads();
  int nblocks = context->getNBlocks(size, nthreads);
  kernelElemAdd<T><<<nblocks, nthreads, 0, context->getStream()>>>(dev_W, size, dev_A);
}
template void elemadd<float>(const CudaContext *, float *, const int, const float *);
#ifdef RPU_USE_DOUBLE
template void elemadd<double>(const CudaContext *, double *, const int, const double *);
#endif

// W = W.*W
template <typename T> __global__ void kernelElemPow2(T *dev_W, const int size) {

  RPU_CUDA_1D_KERNEL_LOOP(idx, size) {
    T x = dev_W[idx];
    dev_W[idx] = x * x;
  }
}
template <typename T> void elempow2(const CudaContext *context, T *dev_W, const int size) {

  int nthreads = context->getNThreads();
  int nblocks = context->getNBlocks(size, nthreads);
  kernelElemPow2<T><<<nblocks, nthreads, 0, context->getStream()>>>(dev_W, size);
}
template void elempow2<float>(const CudaContext *, float *, const int);
#ifdef RPU_USE_DOUBLE
template void elempow2<double>(const CudaContext *, double *, const int);
#endif

// V = abs(W )
template <typename T> __global__ void kernelElemAbs(T *dev_V, const T *dev_W, const int size) {

  RPU_CUDA_1D_KERNEL_LOOP(idx, size) { dev_V[idx] = fabs(dev_W[idx]); }
}
template <typename T>
void elemabs(const CudaContext *context, T *dev_V, const T *dev_W, const int size) {

  int nthreads = context->getNThreads();
  int nblocks = context->getNBlocks(size, nthreads);
  kernelElemAbs<T><<<nblocks, nthreads, 0, context->getStream()>>>(dev_V, dev_W, size);
}
template void elemabs<float>(const CudaContext *, float *, const float *, const int);
#ifdef RPU_USE_DOUBLE
template void elemabs<double>(const CudaContext *, double *, const double *, const int);
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
    const CudaContext *context, T *dev_W, const int size, const T_A *dev_A, const T beta) {

  int nthreads = context->getNThreads();
  int nblocks = context->getNBlocks(size, nthreads);
  kernelElemAddScale<T><<<nblocks, nthreads, 0, context->getStream()>>>(dev_W, size, dev_A, beta);
}
template void
elemaddscale<float, float>(const CudaContext *, float *, const int, const float *, const float);
#ifdef RPU_USE_DOUBLE
template void elemaddscale<double, double>(
    const CudaContext *, double *, const int, const double *, const double);
template void
elemaddscale<double, float>(const CudaContext *, double *, const int, const float *, const double);
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
    const CudaContext *context, T *dev_W, const int size, const T *dev_A, const T *dev_B) {

  int nthreads = context->getNThreads();
  int nblocks = context->getNBlocks(size, nthreads);
  kernelElemAddScale<T><<<nblocks, nthreads, 0, context->getStream()>>>(dev_W, size, dev_A, dev_B);
}
template void
elemaddscale<float>(const CudaContext *, float *, const int, const float *, const float *);
#ifdef RPU_USE_DOUBLE
template void
elemaddscale<double>(const CudaContext *, double *, const int, const double *, const double *);
#endif

// W += sat(A.*B)
template <typename T, typename T_A>
__global__ void
kernelElemASB02(T *dev_W, const int size, const T_A *dev_A, const T *dev_B, float *dev_4params) {

  RPU_CUDA_1D_KERNEL_LOOP(idx, size) {
    T w = dev_W[idx];
    T a = dev_A[idx];
    T b = dev_B[idx];
    float4 parij = reinterpret_cast<float4 *>(dev_4params)[idx];

    w += a * b;
    // check bounds
    w = (w > parij.z) ? parij.z : w;
    w = (w < parij.x) ? parij.x : w;

    dev_W[idx] = w;
  }
}
template <typename T, typename T_A>
void elemasb02(
    const CudaContext *context,
    T *dev_W,
    const int size,
    const T_A *dev_A,
    const T *dev_B,
    float *dev_4params) {
  int nthreads = context->getNThreads();
  int nblocks = context->getNBlocks(size, nthreads);
  kernelElemASB02<T, T_A>
      <<<nblocks, nthreads, 0, context->getStream()>>>(dev_W, size, dev_A, dev_B, dev_4params);
}
template void elemasb02<float, float>(
    const CudaContext *, float *, const int, const float *, const float *, float *);
#ifdef RPU_USE_DOUBLE
template void elemasb02<double, double>(
    const CudaContext *, double *, const int, const double *, const double *, float *);
template void elemasb02<double, float>(
    const CudaContext *, double *, const int, const float *, const double *, float *);
#endif

// sat(W *= A)
template <typename T>
__global__ void
kernelElemScale(T *dev_W, const int size, const T *dev_A, float *dev_4params, const T *dev_shift) {

  bool with_shift = dev_shift != nullptr;
  RPU_CUDA_1D_KERNEL_LOOP(idx, size) {
    T w = dev_W[idx];
    T a = dev_A[idx];
    T s = with_shift ? dev_shift[idx] : 0.0;
    float4 parij = reinterpret_cast<float4 *>(dev_4params)[idx];

    w = (w - s) * a + s;
    // check bounds
    w = (w > parij.z) ? parij.z : w;
    w = (w < parij.x) ? parij.x : w;

    dev_W[idx] = w;
  }
}
template <typename T>
void elemscale(
    const CudaContext *context,
    T *dev_W,
    const int size,
    const T *dev_A,
    float *dev_4params,
    const T *dev_shift) {

  int nthreads = context->getNThreads();
  int nblocks = context->getNBlocks(size, nthreads);
  kernelElemScale<T>
      <<<nblocks, nthreads, 0, context->getStream()>>>(dev_W, size, dev_A, dev_4params, dev_shift);
}
template void
elemscale<float>(const CudaContext *, float *, const int, const float *, float *, const float *);
#ifdef RPU_USE_DOUBLE
template void elemscale<double>(
    const CudaContext *, double *, const int, const double *, float *, const double *);
#endif

// sat(W)
template <typename T> __global__ void kernelElemSat(T *dev_W, const int size, float *dev_4params) {

  RPU_CUDA_1D_KERNEL_LOOP(idx, size) {
    T w = dev_W[idx];
    float4 parij = reinterpret_cast<float4 *>(dev_4params)[idx];
    // check bounds
    w = (w > parij.z) ? parij.z : w;
    w = (w < parij.x) ? parij.x : w;
    dev_W[idx] = w;
  }
}
template <typename T>
void elemsat(const CudaContext *context, T *dev_W, const int size, float *dev_4params) {

  int nthreads = context->getNThreads();
  int nblocks = context->getNBlocks(size, nthreads);
  kernelElemSat<T><<<nblocks, nthreads, 0, context->getStream()>>>(dev_W, size, dev_4params);
}
template void elemsat<float>(const CudaContext *, float *, const int, float *);
#ifdef RPU_USE_DOUBLE
template void elemsat<double>(const CudaContext *, double *, const int, float *);
#endif

// sat(W *= 1+(A-1)*alpha)
template <typename T>
__global__ void kernelElemScaleAlpha(
    T *dev_W,
    const int size,
    const T *dev_A,
    float *dev_4params,
    const T alpha_in,
    const T *dev_shift) {

  volatile T alpha = alpha_in;
  bool with_shift = dev_shift != nullptr;
  RPU_CUDA_1D_KERNEL_LOOP(idx, size) {
    T w = dev_W[idx];
    T a = dev_A[idx];
    T s = with_shift ? dev_shift[idx] : 0.0;
    float4 parij = reinterpret_cast<float4 *>(dev_4params)[idx];

    T scale = 1.0 + alpha * (a - 1.0);
    w = (w - s) * scale + s;

    // check bounds
    w = (w > parij.z) ? parij.z : w;
    w = (w < parij.x) ? parij.x : w;

    dev_W[idx] = w;
  }
}
template <typename T>
void elemscalealpha(
    const CudaContext *context,
    T *dev_W,
    const int size,
    const T *dev_A,
    float *dev_4params,
    const T alpha,
    const T *dev_shift) {

  int nthreads = context->getNThreads();
  int nblocks = context->getNBlocks(size, nthreads);
  kernelElemScaleAlpha<T><<<nblocks, nthreads, 0, context->getStream()>>>(
      dev_W, size, dev_A, dev_4params, alpha, dev_shift);
}
template void elemscalealpha<float>(
    const CudaContext *, float *, const int, const float *, float *, const float, const float *);
#ifdef RPU_USE_DOUBLE
template void elemscalealpha<double>(
    const CudaContext *,
    double *,
    const int,
    const double *,
    float *,
    const double,
    const double *);
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
void elemaddcopy(const CudaContext *context, T *dev_W, T *dev_A, const int size) {

  int nthreads = context->getNThreads();
  int nblocks = context->getNBlocks(size, nthreads);
  kernelElemAddCopy<T><<<nblocks, nthreads, 0, context->getStream()>>>(dev_W, dev_A, size);
}
template void elemaddcopy<float>(const CudaContext *, float *, float *, const int);
#ifdef RPU_USE_DOUBLE
template void elemaddcopy<double>(const CudaContext *, double *, double *, const int);
#endif

// W = sat(W+A), A = W
template <typename T>
__global__ void kernelElemAddCopySat(T *dev_W, T *dev_A, const int size, const float *dev_4params) {

  RPU_CUDA_1D_KERNEL_LOOP(idx, size) {
    T w = dev_W[idx];
    T a = dev_A[idx];
    const float4 parij = reinterpret_cast<const float4 *>(dev_4params)[idx];
    w += a;
    // check bounds
    w = (w > parij.z) ? parij.z : w;
    w = (w < parij.x) ? parij.x : w;
    a = w;
    dev_W[idx] = w;
    dev_A[idx] = a;
  }
}
template <typename T>
void elemaddcopysat(
    const CudaContext *context, T *dev_W, T *dev_A, const int size, const float *dev_4params) {

  int nthreads = context->getNThreads();
  int nblocks = context->getNBlocks(size, nthreads);
  kernelElemAddCopySat<T>
      <<<nblocks, nthreads, 0, context->getStream()>>>(dev_W, dev_A, size, dev_4params);
}
template void
elemaddcopysat<float>(const CudaContext *, float *, float *, const int, const float *);
#ifdef RPU_USE_DOUBLE
template void
elemaddcopysat<double>(const CudaContext *, double *, double *, const int, const float *);
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
    const float *dev_4params) {

  RPU_CUDA_1D_KERNEL_LOOP(idx, size) {
    T th = thres;
    T p = dev_P[idx];
    T a = dev_A[idx]; // load a bit more ?
    T b = dev_B[idx];
    T w = dev_W[idx];
    const float4 parij = reinterpret_cast<const float4 *>(dev_4params)[idx];

    if (p < th) {
      w = a + b;
      // check bounds [only those that changed]
      w = (w > parij.z) ? parij.z : w;
      w = (w < parij.x) ? parij.x : w;
      dev_W[idx] = w;
    }
  }
}
template <typename T>
void elemresetsat(
    const CudaContext *context,
    T *dev_W,
    const int size,
    const T *dev_A,
    const float *dev_B,
    const float *dev_P,
    T thres,
    const float *dev_4params) {

  int nthreads = context->getNThreads();
  int nblocks = context->getNBlocks(size, nthreads);
  kernelElemResetSat<T><<<nblocks, nthreads, 0, context->getStream()>>>(
      dev_W, size, dev_A, dev_B, dev_P, thres, dev_4params);
}
template void elemresetsat<float>(
    const CudaContext *,
    float *,
    const int,
    const float *,
    const float *,
    const float *,
    const float,
    const float *);
#ifdef RPU_USE_DOUBLE
template void elemresetsat<double>(
    const CudaContext *,
    double *,
    const int,
    const double *,
    const float *,
    const float *,
    const double,
    const float *);
#endif

// MSK != 0
// W(MSK) = sat(reset_bias(MSK) + std*randn())
#define RESET_TOLERANCE 1e-6
template <typename T>
__global__ void kernelElemResetSatMsk(
    T *weights,
    const int size_in,
    const char *msk,
    const T *reset_bias,
    const T reset_std_in,
    const float *dev_4params,
    curandState_t *random_states) {

  volatile unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
  T reset_std = reset_std_in;
  int size = size_in;
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
      const float4 parij = reinterpret_cast<const float4 *>(dev_4params)[tid];
      if (reset_std) {
        w = bias + reset_std * curand_normal(&local_state);
      } else {
        w = bias;
      }
      w = (w > parij.z) ? parij.z : w;
      w = (w < parij.x) ? parij.x : w;
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
    CudaContext *context,
    T *W,
    const int size,
    const char *msk,
    const T *reset_bias,
    const T reset_std,
    const float *dev_4params) {

  int nthreads = context->getNThreads();
  int nblocks_batch_max = context->getSMCount() * (context->maxThreadsPerBlock() / nthreads);
  int nblocks = MIN(context->getNBlocks(size, nthreads), nblocks_batch_max);
  kernelElemResetSatMsk<T><<<nblocks, nthreads, 0, context->getStream()>>>(
      W, size, msk, reset_bias, reset_std, dev_4params,
      context->getRandomStates(nblocks * nthreads));
}
template void elemresetsatmsk<float>(
    CudaContext *, float *, const int, const char *, const float *, const float, const float *x);
#ifdef RPU_USE_DOUBLE
template void elemresetsatmsk<double>(
    CudaContext *, double *, const int, const char *, const double *, const double, const float *);
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
void elemsubcopy(const CudaContext *context, T *dev_W, T *dev_A, const int size, const T scale) {

  int nthreads = context->getNThreads();
  int nblocks = context->getNBlocks(size, nthreads);
  kernelElemSubCopy<T><<<nblocks, nthreads, 0, context->getStream()>>>(dev_W, dev_A, size, scale);
}
template void elemsubcopy<float>(const CudaContext *, float *, float *, const int, const float);
#ifdef RPU_USE_DOUBLE
template void elemsubcopy<double>(const CudaContext *, double *, double *, const int, const double);
#endif

// set all elements to a
template <typename T> __global__ void kernelSetConstAlpha(T *values, int size, T alpha) {

  RPU_CUDA_1D_KERNEL_LOOP(idx, size) { values[idx] = alpha; }
}

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
template void elemconst<uint32_t>(const CudaContext *, uint32_t *, const int, const uint32_t);
template void elemconst<uint64_t>(const CudaContext *, uint64_t *, const int, const uint64_t);
template void elemconst<int>(const CudaContext *, int *, const int, const int);
template void elemconst<char>(const CudaContext *, char *, const int, const char);

// w = max(min(w,|a|),-|a|)
template <typename T> __global__ void kernelAClip(T *values, int size, T abs_a) {

  RPU_CUDA_1D_KERNEL_LOOP(idx, size) { values[idx] = MIN(MAX(values[idx], -abs_a), abs_a); }
}

template <typename T> void aclip(const CudaContext *context, T *W, const int size, const T a) {

  int nthreads = context->getNThreads();
  int nblocks = context->getNBlocks(size, nthreads);
  kernelAClip<<<nblocks, nthreads, 0, context->getStream()>>>(W, size, fabs(a));
}
template void aclip<float>(const CudaContext *, float *, const int, const float);
#ifdef RPU_USE_DOUBLE
template void aclip<double>(const CudaContext *, double *, const int, const double);
#endif

// w = max(w,a)
template <typename T> __global__ void kernelElemmax(T *values, int size, T a) {

  RPU_CUDA_1D_KERNEL_LOOP(idx, size) { values[idx] = MAX(values[idx], a); }
}

template <typename T> void elemmax(const CudaContext *context, T *W, const int size, const T a) {

  int nthreads = context->getNThreads();
  int nblocks = context->getNBlocks(size, nthreads);
  kernelElemmax<<<nblocks, nthreads, 0, context->getStream()>>>(W, size, a);
}
template void elemmax<float>(const CudaContext *, float *, const int, const float);
#ifdef RPU_USE_DOUBLE
template void elemmax<double>(const CudaContext *, double *, const int, const double);
#endif

// w = min(w,a)
template <typename T> __global__ void kernelElemmin(T *values, int size, T a) {

  RPU_CUDA_1D_KERNEL_LOOP(idx, size) { values[idx] = MIN(values[idx], a); }
}

template <typename T> void elemmin(const CudaContext *context, T *W, const int size, const T a) {

  int nthreads = context->getNThreads();
  int nblocks = context->getNBlocks(size, nthreads);
  kernelElemmin<<<nblocks, nthreads, 0, context->getStream()>>>(W, size, a);
}
template void elemmin<float>(const CudaContext *, float *, const int, const float);
#ifdef RPU_USE_DOUBLE
template void elemmin<double>(const CudaContext *, double *, const int, const double);
#endif

// w = w<a?0:w elementwise
template <typename T> __global__ void kernelElemSetBelowZero(T *values, int size, T a) {

  RPU_CUDA_1D_KERNEL_LOOP(idx, size) {
    T v = values[idx];
    values[idx] = v < a ? (T)0.0 : v;
  }
}

template <typename T>
void elemsetbelowzero(const CudaContext *context, T *W, const int size, const T a) {

  int nthreads = context->getNThreads();
  int nblocks = context->getNBlocks(size, nthreads);
  kernelElemSetBelowZero<<<nblocks, nthreads, 0, context->getStream()>>>(W, size, a);
}
template void elemsetbelowzero<float>(const CudaContext *, float *, const int, const float);
#ifdef RPU_USE_DOUBLE
template void elemsetbelowzero<double>(const CudaContext *, double *, const int, const double);
#endif

// w[j] = a*A[j] + b*B[j]
// with fast paths for A-B and -B+A and A+B.
template <typename T>
__global__ void kernelElemAdd(T *dev_W, const int size, const T *dev_A, const T *dev_B) {

  RPU_CUDA_1D_KERNEL_LOOP(idx, size) { dev_W[idx] = dev_A[idx] + dev_B[idx]; }
}
template <typename T>
__global__ void kernelElemSub(T *dev_W, const int size, const T *dev_A, const T *dev_B) {

  RPU_CUDA_1D_KERNEL_LOOP(idx, size) { dev_W[idx] = dev_A[idx] - dev_B[idx]; }
}

template <typename T>
__global__ void kernelElemWeightedSum(
    T *dev_W, const int size, const T *dev_A, const T a_in, const T *dev_B, const T b_in) {

  T a = a_in;
  T b = b_in;
  RPU_CUDA_1D_KERNEL_LOOP(idx, size) { dev_W[idx] = a * dev_A[idx] + b * dev_B[idx]; }
}

template <typename T>
void elemweightedsum(
    const CudaContext *context,
    T *dev_W,
    const int size,
    const T *dev_A,
    const T a,
    const T *dev_B,
    const T b) {

  int nthreads = context->getNThreads();
  int nblocks = context->getNBlocks(size, nthreads);
  if (a == 1 && b == 1) {
    kernelElemAdd<T><<<nblocks, nthreads, 0, context->getStream()>>>(dev_W, size, dev_A, dev_B);
  } else if (a == 1 && b == -1) {
    kernelElemSub<T><<<nblocks, nthreads, 0, context->getStream()>>>(dev_W, size, dev_A, dev_B);
  } else if (a == -1 && b == 1) {
    kernelElemSub<T><<<nblocks, nthreads, 0, context->getStream()>>>(dev_W, size, dev_B, dev_A);
  } else {
    kernelElemWeightedSum<T>
        <<<nblocks, nthreads, 0, context->getStream()>>>(dev_W, size, dev_A, a, dev_B, b);
  }
}
template void elemweightedsum<float>(
    const CudaContext *,
    float *,
    const int,
    const float *,
    const float,
    const float *,
    const float);
#ifdef RPU_USE_DOUBLE
template void elemweightedsum<double>(
    const CudaContext *,
    double *,
    const int,
    const double *,
    const double,
    const double *,
    const double);
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
void elemaverage(const CudaContext *context, T *dev_W, const int size, T **dev_Ms, const int m) {

  int nthreads = context->getNThreads();
  int nblocks = context->getNBlocks(size, nthreads);
  kernelElemAverage<T><<<nblocks, nthreads, 0, context->getStream()>>>(dev_W, size, dev_Ms, m);
}
template void elemaverage<float>(const CudaContext *, float *, const int, float **, const int);
#ifdef RPU_USE_DOUBLE
template void elemaverage<double>(const CudaContext *, double *, const int, double **, const int);
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
    const CudaContext *context,
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
    const CudaContext *, float *, const float *, const int, const int, const int, const bool);
#ifdef RPU_USE_DOUBLE
template void permute132<double>(
    const CudaContext *, double *, const double *, const int, const int, const int, const bool);
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
    const CudaContext *context,
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
makeBias<float>(const CudaContext *, float *, const float *, const int, const int, const bool);
template void
makeBias<int>(const CudaContext *, int *, const int *, const int, const int, const bool);
#ifdef RPU_USE_DOUBLE
template void
makeBias<double>(const CudaContext *, double *, const double *, const int, const int, const bool);
#endif

// copy without bias (backward)
template <typename T>
void copyWithoutBias(
    const CudaContext *context,
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
    const CudaContext *, float *, const float *, const int, const int, const bool);
#ifdef RPU_USE_DOUBLE
template void copyWithoutBias<double>(
    const CudaContext *, double *, const double *, const int, const int, const bool);
#endif

// copyWithIterator
template <typename OutputIteratorT, typename InputIteratorT>
__global__ void kernelCopyWithIterator(
    OutputIteratorT out_tensor, const InputIteratorT in_tensor, const int sz_all) {

  RPU_CUDA_1D_KERNEL_LOOP(idx, sz_all) { out_tensor[idx] = in_tensor[idx]; }
}

template <typename OutputIteratorT, typename InputIteratorT>
void copyWithIterator(
    const CudaContext *context,
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
      const CudaContext *, OUTPUTITER, INPUTITER, const int);
#define TRANS_FLOAT(TRANS) TRANS, float

RPU_CMU_DEFINE_CWI(float *, const float *);
RPU_CMU_DEFINE_CWI(float *, float *);
RPU_CMU_DEFINE_CWI(float *, IndexReaderInputIterator<float>);
RPU_CMU_DEFINE_CWI(float *, IndexReaderTransInputIterator<float>);
RPU_CMU_DEFINE_CWI(float *, PermuterTransInputIterator<float>);
RPU_CMU_DEFINE_CWI(float *, IndexReaderSliceInputIterator<TRANS_FLOAT(true)>);
RPU_CMU_DEFINE_CWI(float *, IndexReaderSliceInputIterator<TRANS_FLOAT(false)>);
RPU_CMU_DEFINE_CWI(float *, SliceInputIterator<TRANS_FLOAT(true)>);
RPU_CMU_DEFINE_CWI(float *, SliceInputIterator<TRANS_FLOAT(false)>);
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
RPU_CMU_DEFINE_CWI(PermuterTransOutputIterator<double>, const double *);
RPU_CMU_DEFINE_CWI(IndexReaderOutputIterator<double>, const double *);
RPU_CMU_DEFINE_CWI(IndexReaderTransOutputIterator<double>, const double *);
RPU_CMU_DEFINE_CWI(IndexReaderSliceOutputIterator<TRANS_DOUBLE(true)>, const double *);
RPU_CMU_DEFINE_CWI(IndexReaderSliceOutputIterator<TRANS_DOUBLE(false)>, const double *);
RPU_CMU_DEFINE_CWI(SliceOutputIterator<TRANS_DOUBLE(true)>, const double *);
RPU_CMU_DEFINE_CWI(SliceOutputIterator<TRANS_DOUBLE(false)>, const double *);

#undef TRANS_DOUBLE
#endif

#undef RPU_CMU_DEFINE_CWI

// fake cast to overcome constexpr lacking
template <> const float *fakeCastConst<float, const float *>(const float *X) { return X; };
template <> float *fakeCast<float, float *>(float *X) { return X; };
#ifdef RPU_USE_DOUBLE
template <> const double *fakeCastConst<double, const double *>(const double *X) { return X; };
template <> double *fakeCast<double, double *>(double *X) { return X; };
#endif

} // namespace math
} // namespace RPU
