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

#include "cuda_buffer.h"
#include "utility_functions.h"

#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <string.h>

#include "cublas_v2.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include <cuda_runtime_api.h>
#include <curand.h>
#include <curand_kernel.h>

#define RPU_BUFFER_IN 0
#define RPU_BUFFER_OUT 1
#define RPU_BUFFER_BIAS 2
#define RPU_BUFFER_WEIGHT 3
#define RPU_BUFFER_POS_NEG_IN 4
#define RPU_BUFFER_POS_NEG_OUT 5
#define RPU_BUFFER_TEMP_BLAS 6
#define RPU_BUFFER_TEMP_0 7
#define RPU_BUFFER_TEMP_1 8
#define RPU_BUFFER_TEMP_2 9
#define RPU_BUFFER_TEMP_3 10
#define RPU_BUFFER_TEMP_4 11
#define RPU_BUFFER_DEVICE_0 12
#define RPU_BUFFER_DEVICE_1 13
#define RPU_BUFFER_CWO 14

#define RPU_MAX_BUFFER 15

#define BOLD_ON "\033[1m"
#define BOLD_OFF "\033[0m"
#define RED_ON "\033[0;31m"
#define GREEN_ON "\033[0;32m"
#define BLUE_ON "\033[0;34m"
#define COLOR_OFF "\033[0m"

#define CUBLAS_CALL(X)                                                                             \
  if (X != CUBLAS_STATUS_SUCCESS) {                                                                \
    std::ostringstream ss;                                                                         \
    ss << "CUBLAS_CALL Error ";                                                                    \
    switch (X) {                                                                                   \
    case CUBLAS_STATUS_NOT_INITIALIZED:                                                            \
      ss << "CUBLAS_STATUS_NOT_INITIALIZED";                                                       \
      break;                                                                                       \
    case CUBLAS_STATUS_INVALID_VALUE:                                                              \
      ss << "CUBLAS_STATUS_INVALID_VALUE";                                                         \
      break;                                                                                       \
    case CUBLAS_STATUS_ARCH_MISMATCH:                                                              \
      ss << "CUBLAS_STATUS_ARCH_MISMATCH";                                                         \
      break;                                                                                       \
    case CUBLAS_STATUS_MAPPING_ERROR:                                                              \
      ss << "CUBLAS_STATUS_MAPPING_ERROR";                                                         \
      break;                                                                                       \
    case CUBLAS_STATUS_EXECUTION_FAILED:                                                           \
      ss << "CUBLAS_STATUS_EXECUTION_FAILED";                                                      \
      break;                                                                                       \
    case CUBLAS_STATUS_INTERNAL_ERROR:                                                             \
      ss << "CUBLAS_STATUS_INTERNAL_ERROR";                                                        \
      break;                                                                                       \
    case CUBLAS_STATUS_NOT_SUPPORTED:                                                              \
      ss << "CUBLAS_STATUS_NOT_SUPPORTED";                                                         \
      break;                                                                                       \
    case CUBLAS_STATUS_LICENSE_ERROR:                                                              \
      ss << "CUBLAS_STATUS_LICENSE_ERROR";                                                         \
      break;                                                                                       \
    }                                                                                              \
    ss << " at " << __FILENAME__ << ":" << __LINE__ << std::endl;                                  \
    throw std::runtime_error(ss.str());                                                            \
  }

#define CUDA_CALL(x)                                                                               \
  {                                                                                                \
    cudaError_t tmpe = x;                                                                          \
    if ((tmpe) != cudaSuccess) {                                                                   \
      std::ostringstream ss;                                                                       \
      ss << "CUDA_CALL Error '" << cudaGetErrorString(tmpe) << "' at " << __FILENAME__ << ":"      \
         << __LINE__ << "\n";                                                                      \
      throw std::runtime_error(ss.str());                                                          \
    }                                                                                              \
  }

#define CURAND_CALL(X)                                                                             \
  {                                                                                                \
    curandStatus_t x = X;                                                                          \
    if ((x) != CURAND_STATUS_SUCCESS) {                                                            \
      std::ostringstream ss;                                                                       \
      ss << "CURAND_CALL Error ";                                                                  \
      switch (x) {                                                                                 \
      case CURAND_STATUS_VERSION_MISMATCH:                                                         \
        ss << "CURAND_STATUS_VERSION_MISMATCH";                                                    \
        break;                                                                                     \
      case CURAND_STATUS_NOT_INITIALIZED:                                                          \
        ss << "CURAND_STATUS_NOT_INITIALIZED";                                                     \
        break;                                                                                     \
      case CURAND_STATUS_ALLOCATION_FAILED:                                                        \
        ss << "CURAND_STATUS_ALLOCATION_FAILED";                                                   \
        break;                                                                                     \
      case CURAND_STATUS_TYPE_ERROR:                                                               \
        ss << "CURAND_STATUS_TYPE_ERROR";                                                          \
        break;                                                                                     \
      case CURAND_STATUS_OUT_OF_RANGE:                                                             \
        ss << "CURAND_STATUS_OUT_OF_RANGE";                                                        \
        break;                                                                                     \
      case CURAND_STATUS_LENGTH_NOT_MULTIPLE:                                                      \
        ss << "CURAND_STATUS_LENGTH_NOT_MULTIPLE";                                                 \
        break;                                                                                     \
      case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:                                                \
        ss << "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";                                           \
        break;                                                                                     \
      case CURAND_STATUS_LAUNCH_FAILURE:                                                           \
        ss << "CURAND_STATUS_LAUNCH_FAILURE";                                                      \
        break;                                                                                     \
      case CURAND_STATUS_PREEXISTING_FAILURE:                                                      \
        ss << "CURAND_STATUS_PREEXISTING_FAILURE";                                                 \
        break;                                                                                     \
      case CURAND_STATUS_INITIALIZATION_FAILED:                                                    \
        ss << "CURAND_STATUS_INITIALIZATION_FAILED";                                               \
        break;                                                                                     \
      case CURAND_STATUS_ARCH_MISMATCH:                                                            \
        ss << "CURAND_STATUS_ARCH_MISMATCH";                                                       \
        break;                                                                                     \
      case CURAND_STATUS_INTERNAL_ERROR:                                                           \
        ss << "CURAND_STATUS_INTERNAL_ERROR";                                                      \
        break;                                                                                     \
      }                                                                                            \
      ss << " at " << __FILENAME__ << ":" << __LINE__ << "\n";                                     \
      throw std::runtime_error(ss.str());                                                          \
    }                                                                                              \
  };

#define CUDA_TIMING_INIT                                                                           \
  cudaEvent_t timing_start, timing_stop;                                                           \
  CUDA_CALL(cudaEventCreate(&timing_start));                                                       \
  CUDA_CALL(cudaEventCreate(&timing_stop));                                                        \
  float milliseconds = 0;

#define CUDA_TIMING_START(CONTEXT) CUDA_CALL(cudaEventRecord(timing_start, (CONTEXT)->getStream()));

#define CUDA_TIMING_STOP_NO_OUTPUT(CONTEXT)                                                        \
  CUDA_CALL(cudaEventRecord(timing_stop, (CONTEXT)->getStream()));                                 \
  CUDA_CALL(cudaEventSynchronize(timing_stop));                                                    \
  cudaEventElapsedTime(&milliseconds, timing_start, timing_stop);

#define CUDA_TIMING_STOP(CONTEXT, TXT)                                                             \
  CUDA_TIMING_STOP_NO_OUTPUT(CONTEXT)                                                              \
  std::cout << "\t" << BOLD_ON << TXT << ": " << milliseconds << " msec" << BOLD_OFF << "\n";

#define CUDA_TIMING_DESTROY                                                                        \
  CUDA_CALL(cudaEventDestroy(timing_start));                                                       \
  CUDA_CALL(cudaEventDestroy(timing_stop));

#ifndef MIN
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#endif
#ifndef MAX
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#endif

#define RPU_GET_CUDA_BUFFER(CONTEXT, TYPE, BUFFER, SIZE)                                           \
  if (!BUFFER || BUFFER->getSize() < (SIZE)) {                                                     \
    BUFFER = RPU::make_unique<CudaArray<TYPE>>(CONTEXT, (SIZE));                                   \
    CONTEXT->synchronize();                                                                        \
  }

#define RPU_CUDA_1D_KERNEL_LOOP(i, n)                                                              \
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

#define RPU_CUDA_1D_KERNEL_LOOP_HALF(i, n)                                                         \
  int size2 =                                                                                      \
      ((n) + 1) / 2; /* CudaArray makes sure that last idx is dummy allocated for odd sizes*/      \
  RPU_CUDA_1D_KERNEL_LOOP(i, size2)

#define HALF2PTR(PTR) reinterpret_cast<half2_t *>(PTR)
#define HALF2PTRCONST(PTR) reinterpret_cast<const half2_t *>(PTR)

#define RPU_THREADS_PER_BLOCK 512
#define RPU_THREADS_PER_BLOCK_UPDATE 256
#define RPU_MAX_RAND_STATE_SIZE 10000
#define RPU_UPDATE_BLOCKS_PER_SM 8.1

#define RPU_GET_BLOCKS(SIZE) (SIZE + RPU_THREADS_PER_BLOCK - 1) / RPU_THREADS_PER_BLOCK

#define RPU_GEN_IITER_TEMPLATES(NUM_T, OUT_T, FUNC, ARGS)                                          \
  template OUT_T FUNC(const NUM_T *ARGS);                                                          \
  template OUT_T FUNC(NUM_T *ARGS);                                                                \
  template OUT_T FUNC(IndexReaderInputIterator<NUM_T> ARGS);                                       \
  template OUT_T FUNC(IndexReaderTransInputIterator<NUM_T> ARGS);                                  \
  template OUT_T FUNC(PermuterTransInputIterator<NUM_T> ARGS);                                     \
  template OUT_T FUNC(SliceInputIterator<true, NUM_T> ARGS);                                       \
  template OUT_T FUNC(SliceInputIterator<false, NUM_T> ARGS);                                      \
  template OUT_T FUNC(IndexReaderSliceInputIterator<true, NUM_T> ARGS);                            \
  template OUT_T FUNC(IndexReaderSliceInputIterator<false, NUM_T> ARGS);                           \
  template OUT_T FUNC(DiagInputIterator<NUM_T, chop_t> ARGS);                                      \
  template OUT_T FUNC(EyeInputIterator<NUM_T> ARGS);

#define RPU_GEN_OITER_TEMPLATES(NUM_T, OUT_T, FUNC, ARGS)                                          \
  template OUT_T FUNC(PermuterTransOutputIterator<NUM_T> ARGS);                                    \
  template OUT_T FUNC(IndexReaderTransOutputIterator<NUM_T> ARGS);                                 \
  template OUT_T FUNC(IndexReaderOutputIterator<NUM_T> ARGS);                                      \
  template OUT_T FUNC(NUM_T *ARGS);                                                                \
  template OUT_T FUNC(IndexReaderSliceOutputIterator<true, NUM_T> ARGS);                           \
  template OUT_T FUNC(IndexReaderSliceOutputIterator<false, NUM_T> ARGS);                          \
  template OUT_T FUNC(SliceOutputIterator<false, NUM_T> ARGS);                                     \
  template OUT_T FUNC(SliceOutputIterator<true, NUM_T> ARGS);

#define TRANSPOSE_D2X(IDX, X_SIZE, D_SIZE) ((IDX % D_SIZE) * X_SIZE + (IDX / D_SIZE))

#define TRANSPOSE_X2D(IDX, X_SIZE, D_SIZE) ((IDX % X_SIZE) * D_SIZE + (IDX / X_SIZE))

extern int64_t rpu_global_mem_counter;

namespace RPU {

typedef uint32_t kagg_t; // K aggregate type
typedef int8_t chop_t;   // chopper type

#ifdef RPU_PARAM_FP16
typedef half_t param_t;
typedef half2_t param2_t;
typedef struct __align__(8) { half_t x, y, z, w; }
param4_t;
#else
typedef float param_t;
typedef float2 param2_t;
typedef float4 param4_t;
#endif

class CublasEnvironment {

public:
  explicit CublasEnvironment() : CublasEnvironment(-1){};
  explicit CublasEnvironment(int gpu_id);
  ~CublasEnvironment();

  inline cublasHandle_t getHandle() { return handle_; };
  int runTest();

  void init();
  void test();

private:
  cublasHandle_t handle_;

#ifdef RPU_WITH_CUBLAS_DEVICE
public:
  cublasHandle_t *getDeviceHandle();
  int runTestDevice();

private:
  cublasHandle_t *device_handle_ = nullptr;
  bool device_handle_created_ = false;
  void createDeviceHandle();
#endif
};

template <typename T> class CudaArray;

class CudaContext;
typedef CudaContext *CudaContextPtr;

class CudaContext : public std::enable_shared_from_this<CudaContext>, public Context {

public:
  explicit CudaContext() : CudaContext(-1){};
  // NOTE: not tested on gpu_id (does a streams implicitely specifies a GPU id?)
  explicit CudaContext(int gpu_id, bool non_blocking = true);
  explicit CudaContext(cudaStream_t shared_stream, int gpu_id = -1);
  ~CudaContext();

  CudaContext(const CudaContext &);
  CudaContext &operator=(const CudaContext &);
  CudaContext(CudaContext &&);
  CudaContext &operator=(CudaContext &&);

  friend void swap(CudaContext &a, CudaContext &b) noexcept {

    using std::swap;

    swap(a.gpu_id_, b.gpu_id_);
    swap(a.stream_id_, b.stream_id_);
    swap(a.streams_, b.streams_);
    swap(a.shared_stream_id_, b.shared_stream_id_);
    swap(a.shared_streams_, b.shared_streams_);

    swap(a.env_, b.env_);
    swap(a.rng_, b.rng_);

    swap(a.rng_created_, b.rng_created_);

    swap(a.shared_, b.shared_);

    swap(a.non_blocking_, b.non_blocking_);
    swap(a.event_, b.event_);
    swap(a.prop_, b.prop_);

    swap(a.shared_random_states_, b.shared_random_states_);
    swap(a.shared_float_buffer_, b.shared_float_buffer_);
#ifdef RPU_USE_DOUBLE
    swap(a.shared_double_buffer_, b.shared_double_buffer_);
    swap(a.double_buffer_, b.double_buffer_);
#endif
#ifdef RPU_USE_FP16
    swap(a.shared_half_t_buffer_, b.shared_half_t_buffer_);
    swap(a.half_t_buffer_, b.half_t_buffer_);
#endif

    swap(a.random_states_, b.random_states_);
    swap(a.float_buffer_, b.float_buffer_);
  }

  void synchronizeDevice() const;
  void synchronizeContext()
      const; // synchronizes whole context (if multiple streams are used explicitly)
  void synchronizeStream(int idx) const; // individual stream
  void synchronizeStream() const;        // current stream
  void synchronizeWith(CudaContextPtr c) const;
  void synchronizeWith(CudaContextPtr ca, CudaContextPtr cb) const;
  void synchronize() const override { this->synchronizeStream(); }; // default current stream only

  template <typename T = float>
  int getNBlocks(
      int size, int nthreads = RPU_THREADS_PER_BLOCK, bool half_for_half_t = false) const {
#ifdef RPU_USE_FP16
    if (std::is_same<T, half_t>::value && half_for_half_t) {
      int size2 = (size + 1) / 2; // ceil
      nthreads = MIN(maxThreadsPerBlock(), nthreads);
      return (size2 + nthreads - 1) / nthreads;
    }
#endif
    nthreads = MIN(maxThreadsPerBlock(), nthreads);
    return (size + nthreads - 1) / nthreads;
  }

  int getNStrideBlocks(int size, int nthreads = RPU_THREADS_PER_BLOCK) const;
  inline int getNThreads() const { return MIN(RPU_THREADS_PER_BLOCK, maxThreadsPerBlock()); };
  inline int getNThreads(int n) const { return MIN((n + 31) / 32 * 32, getNThreads()); };
  inline cudaStream_t getStream() const {
    enforceDeviceId();
    return streams_[stream_id_];
  }
  inline cudaStream_t getStream(int idx);
  void enforceDeviceId() const;

  void setExternalStream(cudaStream_t s); // for external stream meanagement
  void releaseExternalStream();           // to be called to release the set external stream

  inline cublasHandle_t getBlasHandle() const { return env_->getHandle(); }
#ifdef RPU_WITH_CUBLAS_DEVICE
  inline cublasHandle_t *getBlasDeviceHandle() const { return env_->getDeviceHandle(); }
#endif
  inline int getGPUId() const { return gpu_id_; };

  void createRandomGenerator();
  void randNormal(float *dev_array, int size, float mean = 0, float stddev = 1);
  void randUniform(float *dev_array, int size);
  void setRandomSeed(unsigned long long rseed);

  template <typename T> T *getSharedBuffer(int id, int size);
  template <typename T> void releaseSharedBuffer(int id);
  template <typename T> void printSharedBuffer(int id, int size);

  void recordEvent();
  void waitEvent(CudaContextPtr wait_on_context);
  void waitEvent(cudaEvent_t e);
  void recordWaitEvent(CudaContextPtr wait_on_context);
  void recordWaitEvent(cudaStream_t s);
  void recordWaitEvent(cudaStream_t s, cudaEvent_t e);

  inline int getSMCount() const { return prop_->multiProcessorCount; };
  inline int getSharedMemPerBlock() const { return prop_->sharedMemPerBlock; };
  inline int getSharedMemPerSM() const {
    return this->getSharedMemPerBlock() * this->maxThreadsPerSM() / this->maxThreadsPerBlock();
  };
  inline int maxThreadsPerBlock() const { return prop_->maxThreadsPerBlock; };
  inline int maxThreadsPerSM() const { return prop_->maxThreadsPerMultiProcessor; };
  inline cudaEvent_t getEvent() const { return event_; };
  inline bool hasRandomGenerator() const { return rng_created_; };

  curandState_t *getRandomStates(int size = 0);

private:
  int gpu_id_ = 0;
  int stream_id_ = 0;
  int shared_stream_id_ = 0;
  std::recursive_mutex shared_mutex_;
  std::vector<cudaStream_t> streams_ = {};
  std::vector<cudaStream_t> shared_streams_ = {};
  CublasEnvironment *env_ = nullptr;
  curandGenerator_t rng_ = nullptr;
  bool rng_created_ = false;
  bool shared_ = false;
  bool non_blocking_ = true;
  cudaEvent_t event_ = nullptr;
  cudaDeviceProp *prop_ = nullptr;

  std::vector<std::unique_ptr<CudaArray<curandState_t>>> shared_random_states_ = {};
  std::vector<std::unique_ptr<CudaArray<curandState_t>>> random_states_ = {};
  std::vector<std::vector<CudaBuffer<float>>> shared_float_buffer_ = {};
  std::vector<std::vector<CudaBuffer<float>>> float_buffer_ = {};
#ifdef RPU_USE_DOUBLE
  std::vector<std::vector<CudaBuffer<double>>> shared_double_buffer_ = {};
  std::vector<std::vector<CudaBuffer<double>>> double_buffer_ = {};
#endif
#ifdef RPU_USE_FP16
  std::vector<std::vector<CudaBuffer<half_t>>> shared_half_t_buffer_ = {};
  std::vector<std::vector<CudaBuffer<half_t>>> half_t_buffer_ = {};
#endif
  void init();
};

template <typename T> class CudaArray {

public:
  CudaArray(){};
  explicit CudaArray(CudaContextPtr c);
  explicit CudaArray(CudaContextPtr c, int n);
  explicit CudaArray(CudaContextPtr c, int n, const T *host_array);
  CudaArray(CudaContextPtr c, const std::vector<T> &host_vector);
  ~CudaArray();

  CudaArray(const CudaArray<T> &);
  CudaArray<T> &operator=(const CudaArray<T> &);
  CudaArray(CudaArray<T> &&);
  CudaArray<T> &operator=(CudaArray<T> &&);

  friend void swap(CudaArray<T> &a, CudaArray<T> &b) noexcept {

    using std::swap;

    swap(a.values_, b.values_);
    swap(a.size_, b.size_);
    swap(a.pitch_, b.pitch_);
    swap(a.width_, b.width_);
    swap(a.height_, b.height_);
    swap(a.context_, b.context_);
    swap(a.shared_if_, b.shared_if_);
  }

  void assign(const T *host_array);
  void assignTranspose(const T *host_array, const int rows, const int cols);

  void assign(const CudaArray<T> &source);
  inline void assign(const std::shared_ptr<CudaArray<T>> source) { this->assign(*source); };

  void assignFromDevice(const T *device_array);
  void setShared(T *device_array);

  void copyTo(T *host_array) const;
  void copyTo(std::vector<T> &host_vector) const;
  std::vector<T> cpu() const;

  void setConst(T set_value);

  T *getDataSafe(CudaContextPtr c);
  inline CudaContextPtr getContext() const { return context_; };
  inline void synchronize() { context_->synchronize(); };

  inline int getSize() const { return size_; };
  inline int getWidth() const { return width_; };
  inline int getWidthBytes() const { return width_ * sizeof(T); };
  inline int getHeight() const { return height_; };
  inline size_t getPitch() const { return pitch_; };
  inline T *getData() { return values_; };
  const T *getDataConst() const { return values_; };

  int getLD() const { return (((int)this->getPitch()) / sizeof(T)); }

  void printValues(int nmax = 0) const;
  void printNZValues(int nmax = 0) const;

private:
  bool shared_if_ = false;
  T *values_ = nullptr;
  int size_ = 0;
  size_t pitch_ = 0;
  int width_ = 0;
  int height_ = 0;

  CudaContextPtr context_ = nullptr;
};

/****************************************************/
void resetCuda(int gpu_id = -1);

void curandSetup(CudaArray<curandState_t> &, unsigned long long rseed = 0, bool same_seed = false);
void curandSetup(
    CudaContextPtr c,
    std::unique_ptr<CudaArray<curandState_t>> &dev_states,
    int n,
    unsigned long long rseed = 0,
    bool same_seed = false);

/* state helper functions*/
template <typename T>
void load(CudaContextPtr context, RPU::state_t &state, std::string key, T &value, bool strict);

} // namespace RPU
