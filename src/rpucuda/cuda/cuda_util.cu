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

#include "cuda_math_util.h"
#include "cuda_util.h"
#include "utility_functions.h"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>

#define DISABLE_SHARED_MUTEX 1

#define IDX2F(i, j, ld) ((((j)-1) * (ld)) + ((i)-1))

// this should be not necesary, because device id is set individually
// per thread. However, if one would want to use 2 GPUs within one
// thread, one needs it.
#define RPU_EXPLICIT_ENFORCE_DEVICE_ID

#define SUBTRACTMEMCOUNTER(BYTES)                                                                  \
  {                                                                                                \
    std::lock_guard<std::mutex> lock(rpu_global_mem_counter_mutex);                                \
    rpu_global_mem_counter -= BYTES;                                                               \
  }
#define ADDTOMEMCOUNTER(BYTES)                                                                     \
  {                                                                                                \
    std::lock_guard<std::mutex> lock(rpu_global_mem_counter_mutex);                                \
    rpu_global_mem_counter += BYTES;                                                               \
  }

int64_t rpu_global_mem_counter = 0;
std::mutex rpu_global_mem_counter_mutex;

namespace RPU {

/*****************************************************/
/*  states */

#define RPU_LOAD_ARRAY(T)                                                                          \
  template <>                                                                                      \
  void load(                                                                                       \
      CudaContextPtr context, RPU::state_t &state, std::string key, CudaArray<T> &value,           \
      bool strict) {                                                                               \
    std::vector<double> tmp;                                                                       \
    try {                                                                                          \
      tmp = state.at(key);                                                                         \
    } catch (const std::out_of_range &oor) {                                                       \
      if (strict) {                                                                                \
        RPU_FATAL("Cannot find the cuda unique vector key `" << key << "` in state.");             \
      }                                                                                            \
      return;                                                                                      \
    }                                                                                              \
    std::vector<T> out(tmp.begin(), tmp.end());                                                    \
    if (!out.size()) {                                                                             \
      value = CudaArray<T>(context, 0);                                                            \
    } else {                                                                                       \
      value = CudaArray<T>(context, out);                                                          \
    }                                                                                              \
  }

RPU_LOAD_ARRAY(float);
#ifdef RPU_USE_DOUBLE
RPU_LOAD_ARRAY(double);
#endif
RPU_LOAD_ARRAY(int);
RPU_LOAD_ARRAY(chop_t);
RPU_LOAD_ARRAY(uint64_t);
RPU_LOAD_ARRAY(char);
RPU_LOAD_ARRAY(kagg_t);
#ifdef RPU_DEFINE_CUDA_HALF_ARRAY
RPU_LOAD_ARRAY(half_t);
#endif
#undef RPU_LOAD_ARRAY

#define RPU_LOAD_UNIQUE(T)                                                                         \
  template <>                                                                                      \
  void load(                                                                                       \
      CudaContextPtr context, RPU::state_t &state, std::string key,                                \
      std::unique_ptr<CudaArray<T>> &value, bool strict) {                                         \
    std::vector<double> tmp;                                                                       \
    try {                                                                                          \
      tmp = state.at(key);                                                                         \
    } catch (const std::out_of_range &oor) {                                                       \
      if (strict) {                                                                                \
        RPU_FATAL("Cannot find the cuda vector key `" << key << "` in state.");                    \
      }                                                                                            \
      return;                                                                                      \
    }                                                                                              \
    std::vector<T> out(tmp.begin(), tmp.end());                                                    \
    if (!out.size()) {                                                                             \
      value = nullptr;                                                                             \
    } else {                                                                                       \
      value = RPU::make_unique<CudaArray<T>>(context, out);                                        \
    }                                                                                              \
  }

RPU_LOAD_UNIQUE(float);
#ifdef RPU_USE_DOUBLE
RPU_LOAD_UNIQUE(double);
#endif
RPU_LOAD_UNIQUE(int);
RPU_LOAD_UNIQUE(chop_t);
RPU_LOAD_UNIQUE(uint64_t);
RPU_LOAD_UNIQUE(char);
RPU_LOAD_UNIQUE(kagg_t);
#ifdef RPU_DEFINE_CUDA_HALF_ARRAY
RPU_LOAD_UNIQUE(half_t);
#endif
#undef RPU_LOAD_UNIQUE

#define RPU_INSERT_CUDA(TYPE)                                                                      \
  template <> void insert(RPU::state_t &state, std::string key, const CudaArray<TYPE> &value) {    \
    std::vector<TYPE> tmp = value.cpu();                                                           \
    std::vector<double> out(tmp.begin(), tmp.end());                                               \
    state[key] = out;                                                                              \
  }

RPU_INSERT_CUDA(float);
#ifdef RPU_USE_DOUBLE
RPU_INSERT_CUDA(double);
#endif
RPU_INSERT_CUDA(int);
RPU_INSERT_CUDA(chop_t);
RPU_INSERT_CUDA(uint64_t);
RPU_INSERT_CUDA(char);
RPU_INSERT_CUDA(kagg_t);
#ifdef RPU_DEFINE_CUDA_HALF_ARRAY
RPU_INSERT_CUDA(half_t);
#endif
#undef RPU_INSERT_CUDA

#define RPU_INSERT_UNIQUE_CUDA(TYPE)                                                               \
  template <>                                                                                      \
  void insert(                                                                                     \
      RPU::state_t &state, std::string key, const std::unique_ptr<CudaArray<TYPE>> &value) {       \
    if (value != nullptr) {                                                                        \
      std::vector<TYPE> tmp = value->cpu();                                                        \
      std::vector<double> out(tmp.begin(), tmp.end());                                             \
      state[key] = out;                                                                            \
    } else {                                                                                       \
      state[key] = std::vector<double>{};                                                          \
    }                                                                                              \
  }

RPU_INSERT_UNIQUE_CUDA(float);
#ifdef RPU_USE_DOUBLE
RPU_INSERT_UNIQUE_CUDA(double);
#endif
RPU_INSERT_UNIQUE_CUDA(int);
RPU_INSERT_UNIQUE_CUDA(chop_t);
RPU_INSERT_UNIQUE_CUDA(uint64_t);
RPU_INSERT_UNIQUE_CUDA(char);
RPU_INSERT_UNIQUE_CUDA(kagg_t);
#ifdef RPU_DEFINE_CUDA_HALF_ARRAY
RPU_INSERT_UNIQUE_CUDA(half_t);
#endif
#undef RPU_INSERT_UNIQUE_CUDA

__global__ void kernelCurandSetup(unsigned long long rseed, curandState_t *state, int n) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  /* Each thread gets same seed, a different sequence
     number, no offset */
  if (id < n) {
    curand_init(rseed, id, 0, &state[id]);
  }
}

__global__ void kernelCurandSetupSameSeed(unsigned long long rseed, curandState_t *state, int n) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id < n) {
    curand_init(rseed, 0, 0, &state[id]);
  }
}

void curandSetup(CudaArray<curandState_t> &dev_states, unsigned long long rseed, bool same_seed) {
  unsigned long long seed = rseed;

  if (rseed == 0) {
    seed = (unsigned long long)std::chrono::high_resolution_clock::now().time_since_epoch().count();
  } else {
    seed = rseed;
  }
  CudaContextPtr c = dev_states.getContext();
  int m = dev_states.getSize();
  int nthreads = c->getNThreads();
  int nblocks = c->getNBlocks(m, nthreads);
  if (same_seed) {
    kernelCurandSetupSameSeed<<<nblocks, nthreads, 0, c->getStream()>>>(
        seed, dev_states.getData(), m);
  } else {
    kernelCurandSetup<<<nblocks, nthreads, 0, c->getStream()>>>(seed, dev_states.getData(), m);
  }

  c->synchronize();
}

void curandSetup(
    CudaContextPtr c,
    std::unique_ptr<CudaArray<curandState_t>> &dev_states,
    int n,
    unsigned long long rseed,
    bool same_seed) {
  int m = (n + 31) / 32 * 32;
  c->synchronizeDevice();
  dev_states = RPU::make_unique<CudaArray<curandState_t>>(c, m);
  curandSetup(*dev_states, rseed, same_seed);
}

CublasEnvironment::~CublasEnvironment() {

  DEBUG_OUT("Destroy BLAS env.");
  // DEBUG_OUT("handle : " <<this->handle_);

  // destroy device

  // destroy host
  if (handle_ != nullptr) {
    cublasDestroy(handle_);
    DEBUG_OUT("CUBLAS destroyed");
  }
#ifdef RPU_WITH_CUBLAS_DEVICE
  if (device_handle_created_) {
    DEBUG_OUT("destroy device handle");
    kernelCublasDestroy<<<1, 1>>>(device_handle_);
    CUDA_CALL(cudaDeviceSynchronize());
    CUDA_CALL(cudaFree(device_handle_));
    DEBUG_OUT("CUBLAS device destroyed");
  }
#endif
  // cudaDeviceReset();
}

CublasEnvironment::CublasEnvironment(int gpu_id) {

  DEBUG_OUT("GET BLAS env.");
  if (gpu_id >= 0) {
    CUDA_CALL(cudaSetDevice(gpu_id));
  }

  // create host
  cublasStatus_t stat = cublasCreate(&handle_);
  CUDA_CALL(cudaDeviceSynchronize());

  // DEBUG_CALL(this->test(););
  // DEBUG_OUT("handle : " <<handle_);
  if (stat != CUBLAS_STATUS_SUCCESS) {
    RPU_FATAL("CUBLAS initialization failed");
  } else
    DEBUG_OUT("CUBLAS Host initialized.");

#ifdef RPU_WITH_CUBLAS_DEVICE
  device_handle_created_ = false;
#endif
}

void CublasEnvironment::test() {
  this->runTest();
#ifdef RPU_WITH_CUBLAS_DEVICE
  if (device_handle_created_) {
    this->runTestDevice();
  }
#endif
}

static __inline__ void
modifyS(cublasHandle_t handle, float *m, int ldm, int n, int p, int q, float alpha, float beta) {
  cublasSscal(handle, n - p + 1, &alpha, &m[IDX2F(p, q, ldm)], ldm);
  cublasSscal(handle, ldm - p + 1, &beta, &m[IDX2F(p, q, ldm)], 1);
}

int CublasEnvironment::runTest() {

  // make a test run
  cublasStatus_t stat;

  int i, j;
  int M = 5;
  int N = 6;
  float *devPtrA;
  float *a = 0;
  a = (float *)malloc(M * N * sizeof(*a));
  if (!a) {
    std::cout << "CUBLAS test run failed (malloc)\n";
    return 1;
  }
  for (j = 1; j <= N; j++) {
    for (i = 1; i <= M; i++) {
      a[IDX2F(i, j, M)] = (float)((i - 1) * M + j);
    }
  }

  if (cudaMalloc((void **)&devPtrA, M * N * sizeof(*a)) != cudaSuccess) {
    std::cerr << "CUBLAS test run failed (cudaMalloc)\n";
    free(a);
    return 1;
  }

  modifyS(handle_, devPtrA, M, N, 2, 3, 16.0f, 12.0f);
  stat = cublasSetMatrix(M, N, sizeof(*a), a, M, devPtrA, M);
  if (stat != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "CUBLAS test run failed (data download)\n";
    CUDA_CALL(cudaFree(devPtrA));
    free(a);
    return 1;
  }
  stat = cublasGetMatrix(M, N, sizeof(*a), devPtrA, M, a, M);
  if (stat != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "CUBLAS test run failed (data upload)\n";
    CUDA_CALL(cudaFree(devPtrA));
    free(a);
    return 1;
  }
  CUDA_CALL(cudaFree(devPtrA));
  for (j = 1; j <= N; j++) {
    for (i = 1; i <= M; i++) {
      std::cout << a[IDX2F(i, j, M)] << ",";
    }
    std::cout << std::endl;
  }

  free(a);
  std::cout << "CUBLAS test run successful.\n";

  return 0;
}

#ifdef RPU_WITH_CUBLAS_DEVICE

__global__ void kernelCublasDestroy(cublasHandle_t *device_handle) {

  cublasStatus_t status = cublasDestroy(*device_handle);
  cudaDeviceSynchronize();
  if (status != CUBLAS_STATUS_SUCCESS) {
    printf("ERROR in destroying cublas device!\n");
  }
}

__global__ void kernelCublasCreateDevice(cublasHandle_t *device_handle) {

  cublasStatus_t status = cublasCreate(device_handle);

  cudaDeviceSynchronize();

  if (status != CUBLAS_STATUS_SUCCESS) {
    printf("ERROR in creating cublas device!\n");
    return;
  }
}

void CublasEnvironment::createDeviceHandle() {

  if (device_handle_created_)
    return;

  CUDA_CALL(cudaMalloc(&device_handle_, sizeof(cublasHandle_t)));
  CUDA_CALL(cudaDeviceSynchronize());

  kernelCublasCreateDevice<<<1, 1>>>(device_handle_);
  CUDA_CALL(cudaDeviceSynchronize());

  DEBUG_OUT("Created device handle");

  device_handle_created_ = true;
}

cublasHandle_t *CublasEnvironment::getDeviceHandle() {
  if (!device_handle_created_) {
    this->createDeviceHandle();
  }
  return device_handle_;
}

__global__ void kernelCublasTest(cublasHandle_t *device_handle, float *source, float *dest) {

  cublasStatus_t status = cublasScopy(*device_handle, 1, source, 1, dest, 1);
  cudaDeviceSynchronize();

  if ((status != CUBLAS_STATUS_SUCCESS)) {
    printf("Some problems with the CuBLAS device test.\n");
  }
}

int CublasEnvironment::runTestDevice() {

  float one = 1;
  float zero = 0;
  float *a;
  float *b;

  CUDA_CALL(cudaMalloc(&a, sizeof(float)));
  CUDA_CALL(cudaMalloc(&b, sizeof(float)));
  CUDA_CALL(cudaMemcpy(a, &one, sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(b, &zero, sizeof(float), cudaMemcpyHostToDevice));

  kernelCublasTest<<<1, 1>>>(device_handle_, a, b);
  CUDA_CALL(cudaDeviceSynchronize());

  CUDA_CALL(cudaMemcpy(&zero, b, sizeof(float), cudaMemcpyDeviceToHost));

  CUDA_CALL(cudaFree(a));
  CUDA_CALL(cudaFree(b));

  if (zero == 1) {
    std::cout << "CuBLAS device test succeded\n";
    return 0;
  } else {
    std::cerr << "ERROR in CuBLAS device test\n";
    return 1;
  }
}

#endif

//**********************************************************************//
void CudaContext::init() {
  DEBUG_OUT("Init context...");

  if (gpu_id_ >= 0) {
    CUDA_CALL(cudaSetDevice(gpu_id_));
  } else {
    CUDA_CALL(cudaGetDevice(&gpu_id_));
  }
  CUDA_CALL(cudaDeviceSynchronize());
  DEBUG_OUT("Create context on GPU " << gpu_id_);
  env_ = new CublasEnvironment(gpu_id_);
  stream_id_ = 0;
  rng_created_ = false;
  shared_ = false;
  non_blocking_ = true;

  CUDA_CALL(cudaEventCreate(&event_));

  prop_ = new cudaDeviceProp();
  CUDA_CALL(cudaGetDeviceProperties(prop_, gpu_id_));
  CUDA_CALL(cudaDeviceSynchronize());
}

CudaContext::CudaContext(int gpu_id, bool non_blocking)
    : gpu_id_(gpu_id), non_blocking_(non_blocking) {
  this->init();
  this->getStream(0);
}

CudaContext::CudaContext(cudaStream_t shared_stream, int gpu_id) : gpu_id_(gpu_id) {
  DEBUG_OUT("Create context on GPU " << gpu_id << " with shared stream (on id 0)\n");
  this->init();

  shared_ = true;
  streams_.push_back(shared_stream);
  stream_id_ = 0;

  shared_stream_id_ = 0;
  shared_streams_.push_back(shared_stream);
}

CudaContext::~CudaContext() {
  DEBUG_OUT("Destroy CudaContext...");

  shared_random_states_.clear();
  shared_float_buffer_.clear();
#ifdef RPU_USE_DOUBLE
  shared_double_buffer_.clear();
#endif
#ifdef RPU_USE_FP16
  shared_half_t_buffer_.clear();
#endif

  random_states_.clear();
  float_buffer_.clear();
#ifdef RPU_USE_DOUBLE
  double_buffer_.clear();
#endif
#ifdef RPU_USE_FP16
  half_t_buffer_.clear();
#endif

  int i_start = shared_ ? 1 : 0;
  for (int i = i_start; i < streams_.size(); i++) {
    cudaStreamSynchronize(streams_[i]);
    cudaStreamDestroy(streams_[i]);
  }

  if (event_ != nullptr) {
    cudaEventDestroy(event_);
    event_ = nullptr;
  }

  if (rng_created_) {
    curandDestroyGenerator(rng_);
    rng_created_ = false;
  }

  if (prop_ != nullptr) {
    delete prop_;
    prop_ = nullptr;
  }
  if (env_ != nullptr) {
    delete env_;
    env_ = nullptr;
  }
  DEBUG_OUT("Destroyed CudaContext.");
}

// copy constructor
CudaContext::CudaContext(const CudaContext &other) {
  // NOTE: changed to non-shared copy

  gpu_id_ = other.gpu_id_;
  this->init();

  shared_ = other.shared_;
  non_blocking_ = other.non_blocking_;

  if (other.shared_ && other.streams_.size() > 0) {
    streams_.push_back(other.streams_[0]);
  }
  shared_streams_ = other.shared_streams_;
  shared_stream_id_ = other.shared_stream_id_;

  for (int i = other.shared_ ? 1 : 0; i < other.streams_.size(); i++) {
    // rest are new streams!!
    this->getStream(i);
  }

  stream_id_ = other.stream_id_;

  if (other.rng_created_) {
    this->createRandomGenerator();
  }

  // random states and buffers won't be copied. They will be created a new

  DEBUG_OUT("CudaContext copy constructed [but only first stream shared. New streams and event!].");
}

// copy assignment
CudaContext &CudaContext::operator=(const CudaContext &other) {
  CudaContext tmp(other);
  swap(*this, tmp);
  synchronize();
  return *this;
}

// move constructor
CudaContext::CudaContext(CudaContext &&other) { *this = std::move(other); }

// move assignment
CudaContext &CudaContext::operator=(CudaContext &&other) {

  gpu_id_ = other.gpu_id_;
  stream_id_ = other.stream_id_;
  shared_stream_id_ = other.shared_stream_id_;
  shared_ = other.shared_;
  non_blocking_ = other.non_blocking_;

  prop_ = other.prop_;
  other.prop_ = nullptr;

  streams_ = std::move(other.streams_);
  shared_streams_ = std::move(other.shared_streams_);

  env_ = other.env_;
  other.env_ = nullptr;

  rng_ = other.rng_;
  other.rng_ = nullptr;

  rng_created_ = other.rng_created_;

  event_ = other.event_;
  other.event_ = nullptr;

  random_states_ = std::move(other.random_states_);
  shared_random_states_ = std::move(other.shared_random_states_);

  shared_float_buffer_ = std::move(other.shared_float_buffer_);
  float_buffer_ = std::move(other.float_buffer_);
#ifdef RPU_USE_DOUBLE
  shared_double_buffer_ = std::move(other.shared_double_buffer_);
  double_buffer_ = std::move(other.double_buffer_);
#endif
#ifdef RPU_USE_FP16
  shared_half_t_buffer_ = std::move(other.shared_half_t_buffer_);
  half_t_buffer_ = std::move(other.half_t_buffer_);
#endif

  return *this;
}

void CudaContext::synchronizeContext() const {
  enforceDeviceId();
  for (int i = 0; i < streams_.size(); i++) {
    CUDA_CALL(cudaStreamSynchronize(streams_[i]));
  }
}

void CudaContext::enforceDeviceId() const {
#ifdef RPU_EXPLICIT_ENFORCE_DEVICE_ID
  int gpu_id;
  CUDA_CALL(cudaGetDevice(&gpu_id));
  if (gpu_id != gpu_id_) {
    // std::cout << "WARNING wrong device detected: " << gpu_id << " vs. " << gpu_id_ << std::endl;
    CUDA_CALL(cudaSetDevice(gpu_id_));
  }
#endif
}

void CudaContext::synchronizeDevice() const {
  enforceDeviceId();
  CUDA_CALL(cudaDeviceSynchronize());
}

void CudaContext::synchronizeWith(CudaContextPtr c) const {

  if (this->getStream() == c->getStream()) {
    // do nothing since work on the same stream
  } else {
    this->synchronize();
    c->synchronize();
  }
}

void CudaContext::synchronizeWith(CudaContextPtr ca, CudaContextPtr cb) const {

  if (ca->getStream() != cb->getStream()) {
    ca->synchronizeWith(cb);
  }
  if (ca->getStream() != this->getStream()) {
    this->synchronize();
  }
}

void CudaContext::synchronizeStream(int idx) const {
  DEBUG_OUT("Synchronize stream idx " << idx);
  enforceDeviceId();
  if ((idx >= 0) && (idx < streams_.size())) {
    CUDA_CALL(cudaStreamSynchronize(streams_[idx]));
  }
}
void CudaContext::synchronizeStream() const {
  DEBUG_OUT("Synchronize stream id " << stream_id_);
  enforceDeviceId();
  CUDA_CALL(cudaStreamSynchronize(streams_[stream_id_]));
}

int CudaContext::getNStrideBlocks(int size, int nthreads) const {
  DEBUG_OUT("get N Stride Blocks for  size " << size);
  nthreads = MIN(maxThreadsPerBlock(), nthreads);
  int max_blocks = getSMCount() * maxThreadsPerBlock() / nthreads;
  return MIN(getNBlocks(size, nthreads), max_blocks);
}

cudaStream_t CudaContext::getStream(int idx) {

  enforceDeviceId();

  DEBUG_OUT("Try to get streams " << idx);
  if ((idx >= 0) && (idx < streams_.size())) {
    if (stream_id_ != idx) {
      stream_id_ = idx;
      CUBLAS_CALL(cublasSetStream(this->getBlasHandle(), streams_[idx]));
    }
    return streams_[idx];
  } else if (streams_.size() == idx) {

    cudaStream_t s;
    if (non_blocking_) {
      CUDA_CALL(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    } else {
      CUDA_CALL(cudaStreamCreate(&s));
    }
    streams_.push_back(s);

    stream_id_ = idx;
    CUBLAS_CALL(cublasSetStream(this->getBlasHandle(), streams_[idx]));

    DEBUG_OUT("Created stream id " << idx << " at : " << streams_[idx] << " ( s: " << s << ")");
    return streams_[idx];
  } else {
    RPU_FATAL("Requested stream size mismatch.");
  }
}

void CudaContext::setExternalStream(cudaStream_t s) {
  if (shared_) {

#ifndef DISABLE_SHARED_MUTEX
    shared_mutex_.lock();
#endif

    enforceDeviceId();
    if (s != streams_[stream_id_]) {
      if (stream_id_ != 0) {
        this->synchronizeDevice();
      } else {
        this->synchronizeStream();
      }
      CUBLAS_CALL(cublasSetStream(this->getBlasHandle(), s));
    }
    streams_[0] = s;
    stream_id_ = 0;

    auto it = std::find(shared_streams_.begin(), shared_streams_.end(), s);
    if (it != shared_streams_.end()) {
      shared_stream_id_ = it - shared_streams_.begin();
    } else {
      shared_streams_.push_back(s);
      shared_stream_id_ = shared_streams_.size() - 1;
    }
  } else {
    RPU_FATAL("setExternalStream: must be a shared context.");
  }
}

void CudaContext::releaseExternalStream() {
  if (shared_) {

#ifndef DISABLE_SHARED_MUTEX
    shared_mutex_.unlock();
#endif
  }
}

void CudaContext::createRandomGenerator() {
  if (!rng_created_) {
    enforceDeviceId();
    CURAND_CALL(curandCreateGenerator(&rng_, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CALL(curandSetStream(rng_, this->getStream()));
    rng_created_ = true;
  }
}

void CudaContext::setRandomSeed(unsigned long long rseed) {
  enforceDeviceId();

  if (!rng_created_) {
    this->createRandomGenerator();
  }

  unsigned long long seed = rseed;
  if (rseed == 0) {
    seed = (unsigned long long)std::chrono::high_resolution_clock::now().time_since_epoch().count();
  } else {
    seed = rseed;
  }
  CURAND_CALL(curandSetStream(rng_, this->getStream()));
  CURAND_CALL(curandSetPseudoRandomGeneratorSeed(rng_, seed));
  this->synchronizeStream();
}

void CudaContext::randNormal(float *dev_array, int size, float mean, float stddev) {
  if (!rng_created_) {
    setRandomSeed(0); // will create random generator on the fly
  }

  if (stddev > 0) {
    CURAND_CALL(curandSetStream(rng_, this->getStream()));
    CURAND_CALL(curandGenerateNormal(rng_, dev_array, size, mean, stddev));
  } else {
    RPU::math::elemconst(this, dev_array, size, mean);
  }
}

void CudaContext::randUniform(float *dev_array, int size) {

  if (!rng_created_) {
    setRandomSeed(0);
  }
  CURAND_CALL(curandSetStream(rng_, this->getStream()));
  CURAND_CALL(curandGenerateUniform(rng_, dev_array, size));
}

curandState_t *CudaContext::getRandomStates(int size) {

  int n = size;
  if (n <= 0) {
    n = getSMCount() * maxThreadsPerBlock();
  }

  auto *rs = &random_states_;
  auto stream_id = stream_id_;
  if (shared_ && stream_id_ == 0) {
    rs = &shared_random_states_;
    stream_id = shared_stream_id_;
  }

  if (rs->size() <= stream_id) {
    rs->resize(stream_id + 1);
  }
  if (!(*rs)[stream_id] || (n > (*rs)[stream_id]->getSize())) {
    curandSetup(this, (*rs)[stream_id], n, 0, false);
  }
  return (*rs)[stream_id]->getData();
}

template <> float *CudaContext::getSharedBuffer<float>(int id, int size) {

  auto *buffer = &float_buffer_;
  auto stream_id = stream_id_;
  if (shared_ && stream_id_ == 0) {
    buffer = &shared_float_buffer_;
    stream_id = shared_stream_id_;
    DEBUG_OUT("Get SHARED float buffer ID " << id << ", size " << size << ", stream " << stream_id);
  } else {
    DEBUG_OUT("Get float buffer ID " << id << ", size " << size << ", stream " << stream_id);
  }

  while (buffer->size() <= stream_id) {
    buffer->push_back(std::vector<CudaBuffer<float>>{RPU_MAX_BUFFER});
  }
  return (*buffer)[stream_id][id].get(this, size);
}

template <> void CudaContext::releaseSharedBuffer<float>(int id) {

  auto *buffer = &float_buffer_;
  auto stream_id = stream_id_;
  if (shared_ && stream_id_ == 0) {
    buffer = &shared_float_buffer_;
    stream_id = shared_stream_id_;
    DEBUG_OUT("Release SHARED float buffer ID " << id << ", stream " << stream_id);
  } else {
    DEBUG_OUT("Release float buffer ID " << id << ", stream " << stream_id);
  }

  (*buffer)[stream_id][id].release();
}

template <> void CudaContext::printSharedBuffer<float>(int id, int size) {

  auto *buffer = &float_buffer_;
  auto stream_id = stream_id_;
  if (shared_ && stream_id_ == 0) {
    buffer = &shared_float_buffer_;
    stream_id = shared_stream_id_;
  }
  RPU_INFO("Float buffer " << id);
  (*buffer)[stream_id][id].print(size);
}

#ifdef RPU_USE_DOUBLE
template <> double *CudaContext::getSharedBuffer<double>(int id, int size) {
  // somehow this needs to be a MAX_BUFFER vector to avoid dynamical
  // resizing. Not sure why, but dynamical allocation of the
  // CudaBuffer vector elements does not work without uniptr (which
  // then has sync problems)

  auto *buffer = &double_buffer_;
  auto stream_id = stream_id_;
  if (shared_ && stream_id_ == 0) {
    buffer = &shared_double_buffer_;
    stream_id = shared_stream_id_;
  }

  while (buffer->size() <= stream_id) {
    buffer->push_back(std::vector<CudaBuffer<double>>{RPU_MAX_BUFFER});
  }
  return (*buffer)[stream_id][id].get(this, size);
}

template <> void CudaContext::releaseSharedBuffer<double>(int id) {

  auto *buffer = &double_buffer_;
  auto stream_id = stream_id_;
  if (shared_ && stream_id_ == 0) {
    buffer = &shared_double_buffer_;
    stream_id = shared_stream_id_;
  }

  (*buffer)[stream_id][id].release();
}

template <> void CudaContext::printSharedBuffer<double>(int id, int size) {

  auto *buffer = &double_buffer_;
  auto stream_id = stream_id_;
  if (shared_ && stream_id_ == 0) {
    buffer = &shared_double_buffer_;
    stream_id = shared_stream_id_;
  }

  RPU_INFO("Double buffer " << id);
  (*buffer)[stream_id][id].print(size);
}

#endif

#ifdef RPU_USE_FP16
template <> half_t *CudaContext::getSharedBuffer<half_t>(int id, int size) {
  // somehow this needs to be a MAX_BUFFER vector to avoid dynamical
  // resizing. Not sure why, but dynamical allocation of the
  // CudaBuffer vector elements does not work without uniptr (which
  // then has sync problems)

  auto *buffer = &half_t_buffer_;
  auto stream_id = stream_id_;
  if (shared_ && stream_id_ == 0) {
    buffer = &shared_half_t_buffer_;
    stream_id = shared_stream_id_;
  }

  while (buffer->size() <= stream_id) {
    buffer->push_back(std::vector<CudaBuffer<half_t>>{RPU_MAX_BUFFER});
  }
  return (*buffer)[stream_id][id].get(this, size);
}

template <> void CudaContext::releaseSharedBuffer<half_t>(int id) {

  auto *buffer = &half_t_buffer_;
  auto stream_id = stream_id_;
  if (shared_ && stream_id_ == 0) {
    buffer = &shared_half_t_buffer_;
    stream_id = shared_stream_id_;
  }

  (*buffer)[stream_id][id].release();
}

template <> void CudaContext::printSharedBuffer<half_t>(int id, int size) {

  auto *buffer = &half_t_buffer_;
  auto stream_id = stream_id_;
  if (shared_ && stream_id_ == 0) {
    buffer = &shared_half_t_buffer_;
    stream_id = shared_stream_id_;
  }

  RPU_INFO("Bfloat buffer " << id);
  (*buffer)[stream_id][id].print(size);
}

#endif

void CudaContext::recordWaitEvent(CudaContextPtr wait_on_context) {
  this->recordWaitEvent(wait_on_context->getStream(), wait_on_context->getEvent());
}
void CudaContext::recordEvent() { CUDA_CALL(cudaEventRecord(event_, streams_[stream_id_])); }
void CudaContext::waitEvent(cudaEvent_t wait_on_event) {
  CUDA_CALL(cudaStreamWaitEvent(streams_[stream_id_], wait_on_event, 0));
}

void CudaContext::waitEvent(CudaContextPtr wait_on_context) {
  waitEvent(wait_on_context->getEvent());
}

void CudaContext::recordWaitEvent(cudaStream_t s) { this->recordWaitEvent(s, event_); }

void CudaContext::recordWaitEvent(cudaStream_t s, cudaEvent_t e) {
  if (streams_[stream_id_] != s) {
    CUDA_CALL(cudaEventRecord(e, s));
    CUDA_CALL(cudaStreamWaitEvent(streams_[stream_id_], e, 0));
  }
}

//**********************************************************************//

template <typename T>
CudaArray<T>::CudaArray(CudaContextPtr c)
    : size_(0), width_(0), height_(1), pitch_(0), context_(c) {}

template <typename T> CudaArray<T>::CudaArray(CudaContextPtr c, int n) : CudaArray(c) {
  size_ = n;
  width_ = n;
  height_ = 1; // this needs to be one! No height>1 supported yet
  if (n > 0) {
    context_->enforceDeviceId();
    int mem_size = size_ * sizeof(T);
    mem_size = (mem_size + 3) / 4 * 4; // align on 32-bit word
    CUDA_CALL(cudaMallocPitch(&values_, &pitch_, mem_size, height_));
    ADDTOMEMCOUNTER(mem_size);
  }
}

template <typename T>
CudaArray<T>::CudaArray(CudaContextPtr c, int n, const T *host_array) : CudaArray(c, n) {
  if (n > 0) {
    this->assign(host_array);
    context_->synchronize(); // better syncrhonize. Constructing is considered slow anyway
  }
}

template <typename T>
CudaArray<T>::CudaArray(CudaContextPtr c, const std::vector<T> &host_vector)
    : CudaArray(c, host_vector.size()) {
  size_t n = host_vector.size();
  if (n > 0) {
    this->assign(host_vector.data());
    context_->synchronize(); // better syncrhonize. Constructing is considered slow anyway
  }
}

template <typename T> CudaArray<T>::~CudaArray() {
  if (values_ == nullptr) {
    return;
  }

  if ((size_ > 0) && (values_ != nullptr) && (!shared_if_)) {
    // cudaDeviceSynchronize(); // too much?
    SUBTRACTMEMCOUNTER(size_ * sizeof(T));
    cudaFree(values_);
    values_ = nullptr;
    size_ = 0;
    width_ = 0;
  }

  values_ = nullptr;
}

// copy constructor
template <typename T> CudaArray<T>::CudaArray(const CudaArray<T> &other) {
  size_ = other.size_;
  width_ = other.width_;
  height_ = other.height_;
  pitch_ = other.pitch_;
  context_ = other.context_;
  values_ = nullptr;

  if (size_ > 0) {
    context_->enforceDeviceId();

    if (other.shared_if_) {
      this->setShared(other.values_);
    } else {
      CUDA_CALL(cudaMallocPitch(&values_, &pitch_, size_ * sizeof(T), height_));
      this->assign(other);
    }
    context_->synchronize(); // better synchronize. Constructing is slow anyway
  }

  DEBUG_OUT("CudaArray copy constructed.");
}

// copy assignment
template <typename T> CudaArray<T> &CudaArray<T>::operator=(const CudaArray<T> &other) {
  CudaArray<T> tmp(other);
  swap(*this, tmp);
  if (size_ > 0) {
    context_->synchronize(); // need sync because of tmp
  }
  return *this;
}

// move constructor
template <typename T> CudaArray<T>::CudaArray(CudaArray<T> &&other) { *this = std::move(other); }

// move assignment
template <typename T> CudaArray<T> &CudaArray<T>::operator=(CudaArray<T> &&other) {

  size_ = other.size_;
  other.size_ = 0;

  width_ = other.width_;
  other.width_ = 0;

  height_ = other.height_;
  other.height_ = 0;

  pitch_ = other.pitch_;
  other.pitch_ = 0;

  context_ = other.context_;
  other.context_ = nullptr;

  values_ = other.values_;
  other.values_ = nullptr;

  shared_if_ = other.shared_if_;

  return *this;
}

template <typename T> void CudaArray<T>::setConst(T set_value) {

  DEBUG_OUT(
      "Set (hsize,P,W,H): " << size_ << ", " << pitch_ << ", " << width_ * sizeof(T) << ", "
                            << height_);
  if (size_ > 0) {
    context_->enforceDeviceId();
    if (set_value != (T)0.0) {
      RPU::math::elemconst(context_, values_, size_, set_value);
    } else {
      CUDA_CALL(cudaMemset2DAsync(
          values_, pitch_, 0, this->getWidthBytes(), height_, context_->getStream()));
    }
  }
}

template <> void CudaArray<curandStateXORWOW>::setConst(curandStateXORWOW set_value) {
  RPU_FATAL("Cannot set curandstates to some values.");
}

#ifdef RPU_USE_DOUBLE
template <> void CudaArray<double *>::setConst(double *set_value) {
  RPU_FATAL("Cannot set pointer types to some values.");
}
#endif
#ifdef RPU_DEFINE_CUDA_HALF_ARRAY
template <> void CudaArray<half_t *>::setConst(half_t *set_value) {
  RPU_FATAL("Cannot set pointer types to some values.");
}
#endif

template <> void CudaArray<float *>::setConst(float *set_value) {
  RPU_FATAL("Cannot set pointer types to some values.");
}

template <typename T> void CudaArray<T>::printValues(int nmax) const {
  T *values = new T[size_];
  this->copyTo(values); // will synchronize
  int n = nmax > 0 ? MIN(nmax, size_) : size_;
  for (int i = 0; i < n; ++i) {
    std::cout << "[" << i << "]:" << values[i] << ", ";
  }
  if (n < size_) {
    std::cout << "...";
  }
  std::cout << std::endl;
  delete[] values;
}

template <typename T> void CudaArray<T>::printNZValues(int nmax) const {
  T *values = new T[size_];
  this->copyTo(values); // will synchronize
  int n = nmax > 0 ? MIN(nmax, size_) : size_;
  for (int i = 0; i < n; ++i) {
    if (values[i] != (T)0) {
      std::cout << "[" << i << "]:" << values[i] << ", ";
    }
  }
  if (n < size_) {
    std::cout << "...";
  }
  std::cout << std::endl;
  delete[] values;
}

template <> void CudaArray<curandStateXORWOW>::printValues(int nmax) const {
  RPU_FATAL("Cannot print curandstates.");
}
template <> void CudaArray<curandStateXORWOW>::printNZValues(int nmax) const {
  RPU_FATAL("Cannot print curandstates.");
}

template <> void CudaArray<int8_t>::printValues(int nmax) const {
  int8_t *values = new int8_t[size_];
  this->copyTo(values); // will synchronize
  int n = nmax > 0 ? MIN(nmax, size_) : size_;
  for (int i = 0; i < n; ++i) {
    std::cout << "[" << i << "]:" << static_cast<int>(values[i]) << ", ";
  }
  if (n < size_) {
    std::cout << "...";
  }
  std::cout << std::endl;
  delete[] values;
}

template <> void CudaArray<int8_t>::printNZValues(int nmax) const {
  int8_t *values = new int8_t[size_];
  this->copyTo(values); // will synchronize
  int n = nmax > 0 ? MIN(nmax, size_) : size_;
  for (int i = 0; i < n; ++i) {
    if (values[i] != 0) {
      std::cout << "[" << i << "]:" << static_cast<int>(values[i]) << ", ";
    }
  }
  if (n < size_) {
    std::cout << "...";
  }
  std::cout << std::endl;
  delete[] values;
}

#ifdef RPU_DEFINE_CUDA_HALF_ARRAY
template <> void CudaArray<half_t>::printValues(int nmax) const {
  half_t *values = new half_t[size_];
  this->copyTo(values); // will synchronize
  int n = nmax > 0 ? MIN(nmax, size_) : size_;
  for (int i = 0; i < n; ++i) {
    std::cout << "[" << i << "]:" << static_cast<float>(values[i]) << ", ";
  }
  if (n < size_) {
    std::cout << "...";
  }
  std::cout << std::endl;
  delete[] values;
}

template <> void CudaArray<half_t>::printNZValues(int nmax) const {
  half_t *values = new half_t[size_];
  this->copyTo(values); // will synchronize
  int n = nmax > 0 ? MIN(nmax, size_) : size_;
  for (int i = 0; i < n; ++i) {
    if (values[i] != (half_t)0.0) {
      std::cout << "[" << i << "]:" << static_cast<float>(values[i]) << ", ";
    }
  }
  if (n < size_) {
    std::cout << "...";
  }
  std::cout << std::endl;
  delete[] values;
}

template <> void CudaArray<half_t *>::printValues(int nmax) const {
  RPU_FATAL("Cannot print half_t* values.");
}
template <> void CudaArray<half_t *>::printNZValues(int nmax) const {
  RPU_FATAL("Cannot print half_t* values.");
}
#endif

template <typename T> void CudaArray<T>::assign(const T *host_array) {
  int sz = size_ * sizeof(T);
  DEBUG_OUT(
      "Assign host (hsize,P,W,H): " << sz << ", " << pitch_ << ", " << width_ * sizeof(T) << ", "
                                    << height_);
  if (size_ > 0) {
    context_->enforceDeviceId();
    context_->synchronize();
    CUDA_CALL(cudaMemcpy2DAsync(
        values_, pitch_, host_array, sz, sz, 1, cudaMemcpyHostToDevice, context_->getStream()));
  }
}

template <typename T>
void CudaArray<T>::assignTranspose(const T *host_array, const int m, const int n) {
  // col major to row major
  if (size_ <= 0) {
    return;
  }

  if (m * n != size_) {
    RPU_FATAL("Size mismatch");
  }
  T *transposed_array = new T[size_];
  for (int i = 0; i < size_; i++) {
    int i_col = (i % n);
    int i_row = (i / n);
    transposed_array[i_col * m + i_row] = host_array[i];
  }
  context_->enforceDeviceId();
  int sz = size_ * sizeof(T);
  DEBUG_OUT(
      "Assign host (hsize,P,W,H): " << sz << ", " << pitch_ << ", " << width_ * sizeof(T) << ", "
                                    << height_);
  context_->synchronize();
  CUDA_CALL(cudaMemcpy2D(
      values_, pitch_, transposed_array, sz, sz, 1, cudaMemcpyHostToDevice)); // no async
  delete[] transposed_array;
}

template <typename T> void CudaArray<T>::assign(const CudaArray<T> &source) {
  DEBUG_OUT(
      "Assign from CudaArray (S,P,W,H): " << size_ << ", " << pitch_ << ", " << width_ * sizeof(T)
                                          << ", " << height_);
  if (source.getSize() != size_) {
    RPU_FATAL("Assignment of Cuda Array failed. Size mismatch.");
  }
  if ((size_ > 0) && (source.getSize() > 0)) {
    cudaStream_t s = context_->getStream();
    context_->synchronizeWith(source.getContext());
    CUDA_CALL(cudaMemcpy2DAsync(
        values_, pitch_, source.getDataConst(), source.getPitch(), source.getWidthBytes(), 1,
        cudaMemcpyDeviceToDevice, s));
  }
}

template <typename T> void CudaArray<T>::assignFromDevice(const T *device_array) {
  DEBUG_OUT(
      "Assign device (S, P,W,H): " << size_ << ", " << pitch_ << ", " << width_ * sizeof(T) << ", "
                                   << height_);
  if ((size_ > 0)) {
    int sz = size_ * sizeof(T);
    cudaStream_t s = context_->getStream();
    context_->synchronizeDevice(); // better do device-wide. Not clear where the device array lives
    CUDA_CALL(
        cudaMemcpy2DAsync(values_, pitch_, device_array, sz, sz, 1, cudaMemcpyDeviceToDevice, s));
  }
}

template <typename T> void CudaArray<T>::setShared(T *device_array) {

  if (device_array == nullptr || size_ <= 0 || values_ == nullptr) {
    RPU_FATAL("Cannot setShared of empty or to nullptr.");
  }

  // destruct
  if (!shared_if_ && values_ != nullptr) {
    context_->synchronize();
    context_->enforceDeviceId();
    CUDA_CALL(cudaFree(values_));
    values_ = nullptr;
    shared_if_ = true;
  }
  values_ = device_array; // assign memory shared (memory is governed from outside)

  // Caution: does not CHECK THE SIZE OF THE GIVEN ARRAY!
}

template <typename T> void CudaArray<T>::copyTo(T *host_array) const {

  int sz = size_ * sizeof(T);
  DEBUG_OUT(
      "Copy to host (hsize,P,W,H): " << sz << ", " << pitch_ << ", " << width_ * sizeof(T) << ", "
                                     << height_);

  if (size_ > 0) {
    context_->enforceDeviceId();
    CUDA_CALL(cudaMemcpy2DAsync(
        host_array, sz, values_, pitch_, this->getWidthBytes(), height_, cudaMemcpyDeviceToHost,
        context_->getStream()));

    context_->synchronizeStream();
  }
}

template <typename T> void CudaArray<T>::copyTo(std::vector<T> &host_vector) const {
  host_vector.resize(size_);
  copyTo(host_vector.data());
}

template <typename T> std::vector<T> CudaArray<T>::cpu() const {
  if (!size_) {
    return std::vector<T>{};
  }
  std::vector<T> host_vector(size_);
  copyTo(host_vector.data());
  return host_vector;
}

template <typename T> T *CudaArray<T>::getDataSafe(CudaContextPtr c) {
  context_->synchronizeWith(c);
  return values_;
}

#ifdef RPU_USE_DOUBLE
template class CudaArray<double *>;
template class CudaArray<double>;
#endif
#ifdef RPU_DEFINE_CUDA_HALF_ARRAY
template class CudaArray<half_t *>;
template class CudaArray<half_t>;
#endif

template class CudaArray<float>;
template class CudaArray<float *>;

template class CudaArray<int>;
template class CudaArray<char>;
template class CudaArray<int8_t>;
template class CudaArray<uint32_t>;
template class CudaArray<uint64_t>;
template class CudaArray<curandStateXORWOW>;

// reset
void resetCuda(int gpu_id) {
  if (gpu_id >= 0) {
    CUDA_CALL(cudaSetDevice(gpu_id));
  }
  CUDA_CALL(cudaDeviceReset());
  CUDA_CALL(cudaFree(0));
  CUDA_CALL(cudaDeviceSynchronize());
}

} // namespace RPU
