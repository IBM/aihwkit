/**
 * (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
 *
 * Licensed under the MIT license. See LICENSE file in the project root for details.
 */

#pragma once

// Compatibility shim that lets the CUDA-spelled RPUCuda sources compile and
// run on ROCm/HIP. It is included once from cuda_util.h before the CUDA
// runtime / cuBLAS / cuRAND headers. On a CUDA build (USE_HIP undefined) it is
// a no-op, so the CUDA path stays byte-identical.
//
// The host C string / allocation headers are pulled in before <hip/hip_runtime.h>
// so that calls to memcpy/memset/abs resolve to the host <cstring>/<cstdlib>
// declarations and not to HIP __device__ overloads.

#if defined(USE_HIP)

#include <cstdlib>
#include <cstring>

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hipblas/hipblas.h>
#include <hiprand/hiprand.h>
#include <hiprand/hiprand_kernel.h>

// ---- runtime ---------------------------------------------------------------
#define cudaError_t hipError_t
#define cudaSuccess hipSuccess
#define cudaGetErrorString hipGetErrorString
#define cudaPeekAtLastError hipPeekAtLastError
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaDeviceReset hipDeviceReset
#define cudaGetDevice hipGetDevice
#define cudaSetDevice hipSetDevice
#define cudaDeviceProp hipDeviceProp_t
#define cudaGetDeviceProperties hipGetDeviceProperties

#define cudaMalloc hipMalloc
#define cudaMallocPitch hipMallocPitch
#define cudaMallocHost hipHostMalloc
#define cudaFree hipFree
#define cudaFreeHost hipHostFree
#define cudaMemset hipMemset
#define cudaMemcpy hipMemcpy
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpy2D hipMemcpy2D
#define cudaMemcpy2DAsync hipMemcpy2DAsync
#define cudaMemset2DAsync hipMemset2DAsync
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice

#define cudaStream_t hipStream_t
#define cudaStreamCreate hipStreamCreate
#define cudaStreamCreateWithFlags hipStreamCreateWithFlags
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStreamWaitEvent hipStreamWaitEvent
#define cudaStreamNonBlocking hipStreamNonBlocking

#define cudaEvent_t hipEvent_t
#define cudaEventCreate hipEventCreate
#define cudaEventDestroy hipEventDestroy
#define cudaEventRecord hipEventRecord
#define cudaEventSynchronize hipEventSynchronize
#define cudaEventElapsedTime hipEventElapsedTime

// ---- cuBLAS -> hipBLAS -----------------------------------------------------
#define cublasHandle_t hipblasHandle_t
#define cublasStatus_t hipblasStatus_t
#define cublasPointerMode_t hipblasPointerMode_t
#define cublasCreate hipblasCreate
#define cublasDestroy hipblasDestroy
#define cublasSetStream hipblasSetStream
#define cublasGetPointerMode hipblasGetPointerMode
#define cublasSetPointerMode hipblasSetPointerMode
#define cublasSetMatrix hipblasSetMatrix
#define cublasGetMatrix hipblasGetMatrix

#define cublasSgemm hipblasSgemm
#define cublasDgemm hipblasDgemm
#define cublasHgemm hipblasHgemm
#define cublasSgemv hipblasSgemv
#define cublasDgemv hipblasDgemv
#define cublasSger hipblasSger
#define cublasDger hipblasDger
#define cublasSscal hipblasSscal
#define cublasDscal hipblasDscal
#define cublasScopy hipblasScopy
#define cublasDcopy hipblasDcopy
#define cublasSnrm2 hipblasSnrm2
#define cublasDnrm2 hipblasDnrm2

#define CUBLAS_OP_N HIPBLAS_OP_N
#define CUBLAS_OP_T HIPBLAS_OP_T
#define CUBLAS_POINTER_MODE_HOST HIPBLAS_POINTER_MODE_HOST
#define CUBLAS_POINTER_MODE_DEVICE HIPBLAS_POINTER_MODE_DEVICE

#define CUBLAS_STATUS_SUCCESS HIPBLAS_STATUS_SUCCESS
#define CUBLAS_STATUS_NOT_INITIALIZED HIPBLAS_STATUS_NOT_INITIALIZED
#define CUBLAS_STATUS_INVALID_VALUE HIPBLAS_STATUS_INVALID_VALUE
#define CUBLAS_STATUS_ARCH_MISMATCH HIPBLAS_STATUS_ARCH_MISMATCH
#define CUBLAS_STATUS_MAPPING_ERROR HIPBLAS_STATUS_MAPPING_ERROR
#define CUBLAS_STATUS_EXECUTION_FAILED HIPBLAS_STATUS_EXECUTION_FAILED
#define CUBLAS_STATUS_INTERNAL_ERROR HIPBLAS_STATUS_INTERNAL_ERROR
#define CUBLAS_STATUS_NOT_SUPPORTED HIPBLAS_STATUS_NOT_SUPPORTED
// hipBLAS has no LICENSE_ERROR status; map it to a benign unknown so the
// CUBLAS_CALL switch still compiles.
#define CUBLAS_STATUS_LICENSE_ERROR HIPBLAS_STATUS_UNKNOWN

// ---- cuRAND -> hipRAND (host generator + device per-thread state) ----------
#define curandGenerator_t hiprandGenerator_t
#define curandStatus_t hiprandStatus_t
// In CUDA, curandState, curandState_t and curandStateXORWOW are all the same
// XORWOW type. hipRAND declares hiprandState and hiprandStateXORWOW_t as
// distinct struct types, so map all three spellings to a single hipRAND type
// to keep the project's template instantiations consistent.
#define curandState hiprandState_t
#define curandState_t hiprandState_t
#define curandStateXORWOW hiprandState_t
#define curandCreateGenerator hiprandCreateGenerator
#define curandDestroyGenerator hiprandDestroyGenerator
#define curandGenerateNormal hiprandGenerateNormal
#define curandGenerateUniform hiprandGenerateUniform
#define curandSetStream hiprandSetStream
#define curandSetPseudoRandomGeneratorSeed hiprandSetPseudoRandomGeneratorSeed
#define curand_init hiprand_init
#define curand_normal hiprand_normal
#define curand_uniform hiprand_uniform
#define CURAND_RNG_PSEUDO_DEFAULT HIPRAND_RNG_PSEUDO_DEFAULT

#define CURAND_STATUS_SUCCESS HIPRAND_STATUS_SUCCESS
#define CURAND_STATUS_VERSION_MISMATCH HIPRAND_STATUS_VERSION_MISMATCH
#define CURAND_STATUS_NOT_INITIALIZED HIPRAND_STATUS_NOT_INITIALIZED
#define CURAND_STATUS_ALLOCATION_FAILED HIPRAND_STATUS_ALLOCATION_FAILED
#define CURAND_STATUS_TYPE_ERROR HIPRAND_STATUS_TYPE_ERROR
#define CURAND_STATUS_OUT_OF_RANGE HIPRAND_STATUS_OUT_OF_RANGE
#define CURAND_STATUS_LENGTH_NOT_MULTIPLE HIPRAND_STATUS_LENGTH_NOT_MULTIPLE
#define CURAND_STATUS_DOUBLE_PRECISION_REQUIRED HIPRAND_STATUS_DOUBLE_PRECISION_REQUIRED
#define CURAND_STATUS_LAUNCH_FAILURE HIPRAND_STATUS_LAUNCH_FAILURE
#define CURAND_STATUS_PREEXISTING_FAILURE HIPRAND_STATUS_PREEXISTING_FAILURE
#define CURAND_STATUS_INITIALIZATION_FAILED HIPRAND_STATUS_INITIALIZATION_FAILED
#define CURAND_STATUS_ARCH_MISMATCH HIPRAND_STATUS_ARCH_MISMATCH
#define CURAND_STATUS_INTERNAL_ERROR HIPRAND_STATUS_INTERNAL_ERROR

// ---- warp-size-independent serialized pulse-train bit format ---------------
//
// bit_line_maker.cu packs the stochastic pulse train as 32-bit words, one word
// per 32-lane logical warp (laneId = threadIdx.x & 0x1f), and the pulsed-update
// kernels read those words back with >>5 / &0x1f / __popc. That on-device bit
// layout must stay byte-identical regardless of the physical wavefront width,
// so the producer's ballot/shuffle must operate at 32-lane logical-warp
// granularity. On a 64-wide CDNA wavefront each wavefront holds two independent
// 32-lane logical warps; the native HIP __ballot/__shfl span all 64 lanes, so
// the device-side wrappers below restrict them to the caller's own 32-lane
// subgroup. On a 32-wide RDNA wavefront they reduce to a plain width-32 op.
__device__ __forceinline__ unsigned int
__rpu_logical_warp32_ballot(int pred) {
  const unsigned long long full = __ballot(pred);
  // Shift the calling lane's own 32-lane subgroup down to bits 0..31. On
  // wave32 __lane_id() is 0..31 so the shift is always 0; on wave64 lanes
  // 32..63 select the high word.
  const unsigned int shift = (__lane_id() & 0x20u);
  return static_cast<unsigned int>(full >> shift);
}
#define __ballot_sync(mask, pred) __rpu_logical_warp32_ballot(pred)
#define __shfl_sync(mask, val, src) __shfl(val, src, 32)
#define __shfl_up_sync(mask, val, delta) __shfl_up(val, delta, 32)
#define __shfl_down_sync(mask, val, delta) __shfl_down(val, delta, 32)

#endif // USE_HIP
