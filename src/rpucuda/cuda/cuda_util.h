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

#include "utility_functions.h"

#include <iostream>
#include <memory>
#include <queue>
#include <string.h>

#include "cuda.h"
#include "cuda_runtime.h"
#include <cuda_runtime_api.h>
#include <curand.h>
#include <curand_kernel.h>

#include "cublas_v2.h"

#define BOLD_ON "\e[1m"
#define BOLD_OFF "\e[0m"
#define RED_ON "\e[0;31m"
#define GREEN_ON "\e[0;32m"
#define BLUE_ON "\e[0;34m"
#define COLOR_OFF "\e[0m"

#define CUBLAS_CALL(x)                                                                             \
  if ((x) != CUBLAS_STATUS_SUCCESS) {                                                              \
    std::ostringstream ss;                                                                         \
    ss << "CUBLAS_CALL Error at " << __FILENAME__ << " : " << __LINE__ << "\n";                    \
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
      if (x == CURAND_STATUS_VERSION_MISMATCH)                                                     \
        ss << "CURAND_STATUS_VERSION_MISMATCH";                                                    \
      else if (x == CURAND_STATUS_NOT_INITIALIZED)                                                 \
        ss << "CURAND_STATUS_NOT_INITIALIZED";                                                     \
      else if (x == CURAND_STATUS_ALLOCATION_FAILED)                                               \
        ss << "CURAND_STATUS_ALLOCATION_FAILED";                                                   \
      else if (x == CURAND_STATUS_TYPE_ERROR)                                                      \
        ss << "CURAND_STATUS_TYPE_ERROR";                                                          \
      else if (x == CURAND_STATUS_OUT_OF_RANGE)                                                    \
        ss << "CURAND_STATUS_OUT_OF_RANGE";                                                        \
      else if (x == CURAND_STATUS_LENGTH_NOT_MULTIPLE)                                             \
        ss << "CURAND_STATUS_LENGTH_NOT_MULTIPLE";                                                 \
      else if (x == CURAND_STATUS_DOUBLE_PRECISION_REQUIRED)                                       \
        ss << "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";                                           \
      else if (x == CURAND_STATUS_LAUNCH_FAILURE)                                                  \
        ss << "CURAND_STATUS_LAUNCH_FAILURE";                                                      \
      else if (x == CURAND_STATUS_PREEXISTING_FAILURE)                                             \
        ss << "CURAND_STATUS_PREEXISTING_FAILURE";                                                 \
      else if (x == CURAND_STATUS_INITIALIZATION_FAILED)                                           \
        ss << "CURAND_STATUS_INITIALIZATION_FAILED";                                               \
      else if (x == CURAND_STATUS_ARCH_MISMATCH)                                                   \
        ss << "CURAND_STATUS_ARCH_MISMATCH";                                                       \
      else if (x == CURAND_STATUS_INTERNAL_ERROR)                                                  \
        ss << "CURAND_STATUS_INTERNAL_ERROR";                                                      \
      ss << " at " << __FILENAME__ << ":" << __LINE__ << "\n";                                     \
      throw std::runtime_error(ss.str());                                                          \
    }                                                                                              \
  };

#define CUDA_TIMING_INIT                                                                           \
  cudaEvent_t timing_start, timing_stop;                                                           \
  CUDA_CALL(cudaEventCreate(&timing_start));                                                       \
  CUDA_CALL(cudaEventCreate(&timing_stop));                                                        \
  float milliseconds = 0;

#define CUDA_TIMING_START(CONTEXT) CUDA_CALL(cudaEventRecord(timing_start, (CONTEXT).getStream()));

#define CUDA_TIMING_STOP_NO_OUTPUT(CONTEXT)                                                        \
  CUDA_CALL(cudaEventRecord(timing_stop, (CONTEXT).getStream()));                                  \
  CUDA_CALL(cudaEventSynchronize(timing_stop));                                                    \
  cudaEventElapsedTime(&milliseconds, timing_start, timing_stop);

#define CUDA_TIMING_STOP(CONTEXT, TXT)                                                             \
  CUDA_TIMING_STOP_NO_OUTPUT(CONTEXT)                                                              \
  std::cout << "\t" << GREEN_ON << TXT << ": " << milliseconds << " msec" << COLOR_OFF << "\n";

#define CUDA_TIMING_DESTROY                                                                        \
  CUDA_CALL(cudaEventDestroy(timing_start));                                                       \
  CUDA_CALL(cudaEventDestroy(timing_stop));

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

#define RPU_CUDA_1D_KERNEL_LOOP(i, n)                                                              \
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

#define RPU_GET_CUDA_BUFFER(CONTEXT, TYPE, BUFFER, SIZE)                                           \
  if (!BUFFER || BUFFER->getSize() < (SIZE)) {                                                     \
    BUFFER = make_unique<CudaArray<TYPE>>(CONTEXT, (SIZE));                                        \
  }

#define RPU_THREADS_PER_BLOCK 512
#define RPU_THREADS_PER_BLOCK_UPDATE 512
#define RPU_MAX_RAND_STATE_SIZE 10000
#define RPU_UPDATE_BLOCKS_PER_SM 1.1

#define RPU_GEN_IITER_TEMPLATES(NUM_T, OUT_T, FUNC, ARGS)                                          \
  template OUT_T FUNC(const NUM_T *ARGS);                                                          \
  template OUT_T FUNC(NUM_T *ARGS);                                                                \
  template OUT_T FUNC(IndexReaderInputIterator<NUM_T> ARGS);                                       \
  template OUT_T FUNC(IndexReaderTransInputIterator<NUM_T> ARGS);                                  \
  template OUT_T FUNC(PermuterTransInputIterator<NUM_T> ARGS);

#define RPU_GEN_OITER_TEMPLATES(NUM_T, OUT_T, FUNC, ARGS)                                          \
  template OUT_T FUNC(PermuterTransOutputIterator<NUM_T> ARGS);                                    \
  template OUT_T FUNC(IndexReaderTransOutputIterator<NUM_T> ARGS);                                 \
  template OUT_T FUNC(IndexReaderOutputIterator<NUM_T> ARGS);                                      \
  template OUT_T FUNC(NUM_T *ARGS);

#define TRANSPOSE_D2X(IDX, X_SIZE, D_SIZE) ((IDX % D_SIZE) * X_SIZE + (IDX / D_SIZE))

#define TRANSPOSE_X2D(IDX, X_SIZE, D_SIZE) ((IDX % X_SIZE) * D_SIZE + (IDX / X_SIZE))

namespace RPU {

typedef uint32_t kagg_t; // K aggregate type

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

class CudaContext {

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
    swap(a.env_, b.env_);
    swap(a.rng_, b.rng_);

    swap(a.rng_created_, b.rng_created_);

    swap(a.shared_, b.shared_);

    swap(a.non_blocking_, b.non_blocking_);
    swap(a.event_, b.event_);
    swap(a.prop_, b.prop_);

    swap(a.shared_random_states_, b.shared_random_states_);
  }

  void synchronizeDevice() const;
  void synchronizeContext()
      const; // synchronizes whole context (if multiple streams are used explicitly)
  void synchronizeStream(int idx) const; // individual stream
  void synchronizeStream() const;        // current stream
  void synchronizeWith(CudaContext *c) const;
  void synchronizeWith(CudaContext *ca, CudaContext *cb) const;
  inline void synchronize() const { this->synchronizeStream(); }; // default current stream only

  int getNBlocks(int size, int nthreads = RPU_THREADS_PER_BLOCK) const;
  int getNStrideBlocks(int size, int nthreads = RPU_THREADS_PER_BLOCK) const;
  inline int getNThreads() const { return RPU_THREADS_PER_BLOCK; };
  inline cudaStream_t getStream() const {
    enforceDeviceId();
    return streams_[stream_id_];
  }
  inline cudaStream_t getStream(int idx);
  void enforceDeviceId() const;

  void setStream(cudaStream_t s);
  inline cublasHandle_t getBlasHandle() const { return env_->getHandle(); }
#ifdef RPU_WITH_CUBLAS_DEVICE
  inline cublasHandle_t *getBlasDeviceHandle() const { return env_->getDeviceHandle(); }
#endif
  inline int getGPUId() const { return gpu_id_; };

  void createRandomGenerator();
  void randNormal(float *dev_array, int size, float mean = 0, float stddev = 1);
  void randUniform(float *dev_array, int size);
  void setRandomSeed(unsigned long long rseed);

  void recordEvent();
  void waitEvent(CudaContext *wait_on_context);
  void waitEvent(cudaEvent_t e);
  void recordWaitEvent(CudaContext *wait_on_context);
  void recordWaitEvent(cudaStream_t s);
  void recordWaitEvent(cudaStream_t s, cudaEvent_t e);

  inline int getSMCount() const { return prop_->multiProcessorCount; };
  inline int getSharedMemPerBlock() const { return prop_->sharedMemPerBlock; };
  inline int maxThreadsPerBlock() const { return prop_->maxThreadsPerBlock; };
  inline int maxThreadsPerSM() const { return prop_->maxThreadsPerMultiProcessor; };
  inline cudaEvent_t getEvent() const { return event_; };
  inline bool hasRandomGenerator() const { return rng_created_; };

  curandState_t *getRandomStates(int size = 0);

private:
  int gpu_id_ = 0;
  int stream_id_ = 0;
  std::vector<cudaStream_t> streams_ = {};
  CublasEnvironment *env_ = nullptr;
  curandGenerator_t rng_ = nullptr;
  bool rng_created_ = false;
  bool shared_ = false;
  bool non_blocking_ = true;
  cudaEvent_t event_ = nullptr;
  cudaDeviceProp *prop_ = nullptr;
  std::vector<std::unique_ptr<CudaArray<curandState_t>>> shared_random_states_ = {};
  void init();
};

template <typename T> class CudaArray {

public:
  explicit CudaArray(CudaContext *c);
  explicit CudaArray(CudaContext *c, int n);
  explicit CudaArray(CudaContext *c, int n, const T *host_array);
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

  void setConst(T set_value);

  T *getDataSafe(CudaContext *c);
  inline CudaContext *getContext() const { return context_; };
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

private:
  bool shared_if_ = false;
  T *values_ = nullptr;
  int size_ = 0;
  size_t pitch_ = 0;
  int width_ = 0;
  int height_ = 0;

  CudaContext *context_ = nullptr;
};

void resetCuda(int gpu_id = -1);

// helper for random init
void curandSetup(CudaArray<curandState_t> &, unsigned long long rseed = 0, bool same_seed = false);
void curandSetup(
    CudaContext *c,
    std::unique_ptr<CudaArray<curandState_t>> &dev_states,
    int n,
    unsigned long long rseed = 0,
    bool same_seed = false);

} // namespace RPU
