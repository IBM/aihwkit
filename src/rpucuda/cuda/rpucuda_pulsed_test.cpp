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

#include "cuda.h"
#include "cuda_util.h"
#include "io_manager.h"
#include "rng.h"
#include "rpu_constantstep_device.h"
#include "rpucuda_pulsed.h"
#include "utility_functions.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <chrono>
#include <memory>
#include <numeric>
#include <random>

#define TOLERANCE 5e-4
namespace {

using namespace RPU;

template <typename T> void transpose(T *w_trans, T *w, int x_size, int d_size) {

  for (int i = 0; i < x_size; i++) {
    for (int j = 0; j < d_size; j++) {
      w_trans[j + i * d_size] = w[i + j * x_size];
    }
  }
}

template <typename T1> class RPUCudaPulsedTestFixture : public ::testing::Test {
public:
  using T = T1;
  void SetUp() {

    is_test = false;

    x_size = 100;
    d_size = 101;
    repeats = 100;
    K = 10;
    m_batch = 100; // for the batched versions
    dim3 = 4;

    T bmin = -0.7;
    T bmax = 0.7;

    PulsedMetaParameter<T> p;
    IOMetaParameter<T> p_io;
    ConstantStepRPUDeviceMetaParameter<T> dp;

    noise_value = 0.05;
    p_io.out_noise = noise_value;
    p.up.desired_BL = K;

    dp.w_max = 1.1;
    dp.w_min = -1.1;
    dp.w_min_dtod = 0.0;
    dp.w_max_dtod = 0.0;

    // dp.dw_min = 0.005;
    // dp.dw_min_std = 0.001;
    // dp.dw_min_dtod = 0.3;

    // dp.up_down = 0.0;
    // dp.up_down_dtod = 0.01;

    // peripheral circuits specs
    // p_io.inp_res = 0.01;
    // p_io.inp_sto_round = false;
    // p_io.out_res = 0.01;

    p_io.noise_management = NoiseManagementType::AbsMax;
    p_io.bound_management = BoundManagementType::Iterative;

    p.up.update_management = false;
    p.up.update_bl_management = false;

    dp.lifetime = 100;
    dp.lifetime_dtod = 10;

    dp.diffusion = 0.01;
    dp.diffusion_dtod = 0.01;
    dp.reset = 0.01;
    dp.reset_std = 0.0;
    dp.reset_dtod = 0.01;

    p.f_io = p_io;
    p.b_io = p_io;
    p.b_io.bound_management = BoundManagementType::None;

    T lr = 0.01;

    layer_pulsed = RPU::make_unique<RPUPulsed<T>>(x_size, d_size);
    layer_pulsed->populateParameter(&p, &dp);
    layer_pulsed->setLearningRate(lr);
    layer_pulsed->setWeightsUniformRandom(bmin, bmax);

    // layer_pulsed->dispParameter();
    // layer_pulsed->disp();

    // copy-construct on GPU
    context = &context_container;

    culayer_pulsed = RPU::make_unique<RPUCudaPulsed<T>>(context, *layer_pulsed);
    // culayer_pulsed->disp();

    x1.resize(x_size * m_batch);
    x2.resize(x_size * m_batch);
    x3.resize(x_size * m_batch);
    d1.resize(d_size * m_batch);
    d2.resize(d_size * m_batch);
    rx.resize(x_size * m_batch);
    rd.resize(d_size * m_batch);

    w_other = new T[d_size * x_size];
    w_other_trans = new T[d_size * x_size];

    unsigned int seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator{seed};
    std::uniform_real_distribution<float> udist(-1.2, 1.2);
    auto urnd = std::bind(udist, generator);

    // just assign some numbers from the weigt matrix
    for (int i = 0; i < x_size * m_batch; i++)
      rx[i] = (T)urnd();

    for (int j = 0; j < d_size * m_batch; j++) {
      rd[j] = (T)urnd();
    }

    std::uniform_real_distribution<float> udist2(-0.2, 0.2);
    auto urnd2 = std::bind(udist2, generator);

    for (int j = 0; j < d_size * x_size; j++) {
      w_other[j] = (T)urnd2();
    }
    transpose(w_other_trans, w_other, x_size, d_size);

    x_cuvec = RPU::make_unique<CudaArray<T>>(context, x_size);
    x_vec.resize(x_size);
    x_vec2.resize(x_size);

    d_cuvec = RPU::make_unique<CudaArray<T>>(context, d_size);
    d_vec.resize(d_size);
    d_vec2.resize(d_size);

    x_cuvec_batch = RPU::make_unique<CudaArray<T>>(context, m_batch * x_size);
    x_vec_batch.resize(m_batch * x_size);
    x_vec2_batch.resize(m_batch * x_size);

    d_cuvec_batch = RPU::make_unique<CudaArray<T>>(context, m_batch * d_size);
    d_vec_batch.resize(m_batch * d_size);
    d_vec2_batch.resize(m_batch * d_size);

    w_cuother = RPU::make_unique<CudaArray<T>>(context, x_size * d_size);
    w_cuother->assignTranspose(w_other, d_size, x_size);
  }

  void TearDown() {
    delete[] w_other;
    delete[] w_other_trans;
  }

  CudaContext context_container{-1, false};
  CudaContextPtr context;
  std::unique_ptr<RPUPulsed<T>> layer_pulsed;
  std::unique_ptr<RPUCudaPulsed<T>> culayer_pulsed;
  std::vector<T> x_vec, x_vec_batch;
  std::vector<T> x_vec2, x_vec2_batch;
  std::vector<T> d_vec, d_vec_batch;
  std::vector<T> d_vec2, d_vec2_batch;

  std::unique_ptr<CudaArray<T>> x_cuvec, x_cuvec_batch;
  std::unique_ptr<CudaArray<T>> d_cuvec, d_cuvec_batch;
  std::unique_ptr<CudaArray<T>> w_cuother;

  int x_size;
  int d_size;
  int repeats;
  int K;
  int dim3;
  int m_batch;
  bool is_test;

  std::vector<T> x1, x2, x3;
  std::vector<T> d1, d2;
  std::vector<T> rx, rd;
  T *w_other;
  T *w_other_trans;
  T noise_value;
};

template <typename T1> class RPUCudaPulsedTestFixtureNoNoise : public ::testing::Test {
public:
  using T = T1;

  void SetUp() {

    is_test = false;

    context = &context_container;

    x_size = 41;
    d_size = 50;
    repeats = 100;
    K = 10;
    m_batch = 100; // for the batched versions
    dim3 = 4;

    T bmin = -0.1;
    T bmax = 0.1;

    PulsedMetaParameter<T> p;
    IOMetaParameter<T> p_io;
    SimpleRPUDeviceMetaParameter<T> dp;

    p_io.out_noise = 0.0; // no noise in output;
    this->noise_value = p_io.out_noise;
    p.up.desired_BL = K;

    // dp.w_max = 1;
    // dp.w_min = -1;
    // dp.w_min_dtod = 0.0;
    // dp.w_max_dtod = 0.0;

    // peripheral circuits specs
    p_io.is_perfect = false;
    p_io.inp_res = -1;
    p_io.inp_sto_round = false;
    p_io.out_res = -1;
    p_io.out_bound = 0.0;

    non_linearities = false; // otherwise test will not pass for NL

    p_io.noise_management = NoiseManagementType::AbsMax;
    p_io.bound_management = BoundManagementType::Iterative;

    p.up.update_management = true;
    p.up.update_bl_management = true;

    dp.lifetime = 100;
    // dp.lifetime_dtod = 10;

    dp.diffusion = 0.01;
    // dp.diffusion_dtod = 0.01;

    p.f_io = p_io;
    p.b_io = p_io;
    p.b_io.bound_management = BoundManagementType::None;
    // p.print();
    p.f_io.out_bound = 12;

    x1.resize(x_size * m_batch);
    x2.resize(x_size * m_batch);
    x3.resize(x_size * m_batch);
    d1.resize(d_size * m_batch);
    d2.resize(d_size * m_batch);
    d3.resize(d_size * m_batch);
    rx.resize(x_size * m_batch);
    rd.resize(d_size * m_batch);

    T lr = 0.01;
    layer = RPU::make_unique<RPUSimple<T>>(x_size, d_size);
    layer->setLearningRate(lr);

    layer_pulsed = RPU::make_unique<RPUPulsed<T>>(x_size, d_size);
    layer_pulsed->populateParameter(&p, &dp);
    layer_pulsed->setLearningRate(lr);
    layer_pulsed->setWeightsUniformRandom(bmin, bmax);
    layer->setWeights(layer_pulsed->getWeights()[0]);

    // layer_pulsed->disp();

    // culayer
    culayer_pulsed = RPU::make_unique<RPUCudaPulsed<T>>(context, *layer_pulsed);
    // culayer_pulsed->dispParameter();

    unsigned int seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator{seed};
    std::uniform_real_distribution<float> udist(-1.2, 1.2);
    auto urnd = std::bind(udist, generator);

    // just assign some numbers from the weight matrix
    for (int i = 0; i < x_size * m_batch; i++)
      rx[i] = (T)urnd();

    for (int j = 0; j < d_size * m_batch; j++) {
      rd[j] = (T)urnd();
    }

    x_cuvec = RPU::make_unique<CudaArray<T>>(context, x_size);
    x_vec.resize(x_size);
    x_vec2.resize(x_size);

    d_cuvec = RPU::make_unique<CudaArray<T>>(context, d_size);
    d_vec.resize(d_size);
    d_vec2.resize(d_size);

    x_cuvec_batch = RPU::make_unique<CudaArray<T>>(context, m_batch * x_size);
    x_vec_batch.resize(m_batch * x_size);
    x_vec2_batch.resize(m_batch * x_size);

    d_cuvec_batch = RPU::make_unique<CudaArray<T>>(context, m_batch * d_size);
    d_vec_batch.resize(m_batch * d_size);
    d_vec2_batch.resize(m_batch * d_size);

    m = m_batch / dim3; // m_batch for the indexed versions

    std::vector<int> v(m);
    std::iota(v.begin(), v.end(), 0);
    int s = 0;
    batch_indices = new int[m * dim3];

    for (int i = 0; i < dim3; i++) {
      std::shuffle(v.begin(), v.end(), generator);
      for (int j = 0; j < m; j++) {
        batch_indices[s++] = v[j];
      }
    }
  }

  void TearDown() { delete batch_indices; }

  CudaContext context_container{-1, false};
  CudaContextPtr context;
  std::unique_ptr<RPUSimple<T>> layer;
  std::unique_ptr<RPUPulsed<T>> layer_pulsed;
  std::unique_ptr<RPUCudaPulsed<T>> culayer_pulsed;
  std::vector<T> x_vec, x_vec_batch;
  std::vector<T> x_vec2, x_vec2_batch;
  std::vector<T> d_vec, d_vec_batch;
  std::vector<T> d_vec2, d_vec2_batch;

  std::unique_ptr<CudaArray<T>> x_cuvec, x_cuvec_batch;
  std::unique_ptr<CudaArray<T>> d_cuvec, d_cuvec_batch;
  int x_size;
  int d_size;
  int repeats;
  int K;
  int m_batch;
  int m;
  int dim3;
  int *batch_indices;
  bool is_test;
  bool non_linearities = false;
  std::vector<T> x1, x2, x3;
  std::vector<T> d1, d2, d3;
  std::vector<T> rx, rd;

  T noise_value;
};

// types
typedef ::testing::Types<num_t> Tios;

TYPED_TEST_CASE(RPUCudaPulsedTestFixtureNoNoise, Tios);
TYPED_TEST_CASE(RPUCudaPulsedTestFixture, Tios);

TYPED_TEST(RPUCudaPulsedTestFixture, ForwardVector) {
  using T = typename TestFixture::T;

  this->x_cuvec->assign(this->rx.data());
  this->x_vec = this->rx;
  // T max_value = Find_Absolute_Max<T>(this->rx.data(), this->x_size);
  // std::cout << "Max value is " << max_value << std::endl;

  this->context->synchronizeDevice();

  // this->culayer_pulsed->printWeights(3, 3);
  // this->layer_pulsed->printWeights(3, 3);

  int nloop = this->repeats;
  double cudur = 0, dur = 0;

  T *cuavg = new T[this->d_size];
  T *avg = new T[this->d_size];

  for (int i = 0; i < this->d_size; i++) {
    cuavg[i] = 0;
    avg[i] = 0;
  }
  CUDA_TIMING_INIT;
  auto start_time = std::chrono::high_resolution_clock::now();
  auto end_time = std::chrono::high_resolution_clock::now();
  for (int loop = 0; loop < nloop; loop++) {

    // RPUCuda forward
    CUDA_TIMING_START(this->context);
    this->culayer_pulsed->forward(
        this->x_cuvec->getData(), this->d_cuvec->getData(), false, 1, false, false, this->is_test);
    CUDA_TIMING_STOP_NO_OUTPUT(this->context);
    if (loop > 0)
      cudur += milliseconds;

    // RPU forward
    start_time = std::chrono::high_resolution_clock::now();
    this->layer_pulsed->forward(
        this->x_vec.data(), this->d_vec.data(), false, 1, false, false, this->is_test);
    end_time = std::chrono::high_resolution_clock::now();
    if (loop > 0)
      dur += std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();

    this->d2 = this->d_vec;
    this->d_cuvec->copyTo(this->d1.data());
    this->context->synchronizeDevice();

    for (int i = 0; i < this->d_size; i++) {
      cuavg[i] += this->d1[i] / (num_t)nloop;
      avg[i] += this->d2[i] / (num_t)nloop;

      if (!this->noise_value) {
        EXPECT_FLOAT_EQ(this->d1[i], this->d2[i]);
      }
    }
  }

  std::cout << BOLD_ON << "\tCUDA Forward done in: " << (float)cudur / (nloop - 1) << " msec"
            << BOLD_OFF << std::endl;
  std::cout << BOLD_ON << "\tCPU Forward done in: " << (float)dur / (nloop - 1) / 1000. << " msec"
            << BOLD_OFF << std::endl;

  for (int i = 0; i < this->d_size; i++) {
    // std::cout  << "ref:" << avg[i] << ", cu out: " << cuavg[i] << std::endl;
    ASSERT_NEAR((float)cuavg[i], (float)avg[i], 1. / sqrtf(nloop));
  }

  delete[] avg;
  delete[] cuavg;
  CUDA_TIMING_DESTROY;
}

TYPED_TEST(RPUCudaPulsedTestFixtureNoNoise, ForwardMatrixBiasNoNoise) {

  this->x_cuvec_batch->assign(this->rx.data());
  this->x_vec_batch = this->rx;

  this->context->synchronizeDevice();

  // this->culayer_pulsed->printWeights(3, 3);
  // this->layer_pulsed->printWeights(3, 3);
  // this->layer->printWeights(3, 3);

  bool bias = false;

  // RPUCuda forward
  this->culayer_pulsed->forward(
      this->x_cuvec_batch->getData(), this->d_cuvec_batch->getData(), bias, this->m_batch, true,
      true, this->is_test);
  this->context->synchronizeDevice();
  this->d_cuvec_batch->copyTo(this->d1.data());

  // RPU forward
  this->layer_pulsed->forward(
      this->x_vec_batch.data(), this->d_vec_batch.data(), bias, this->m_batch, true, true,
      this->is_test);
  this->d2 = this->d_vec_batch;

  this->layer->forward(
      this->x_vec_batch.data(), this->d_vec_batch.data(), bias, this->m_batch, true, true,
      this->is_test);
  this->d3 = this->d_vec_batch;

  for (int i = 0; i < this->d_size * this->m_batch; i++) {
    // std::cout << i << ": d1 " << this->d1[i] << " vs d2 " << this->d2[i] << std::endl;
    ASSERT_NEAR((float)this->d1[i], (float)this->d2[i], TOLERANCE);

    if (!this->non_linearities) {
      ASSERT_NEAR((float)this->d2[i], (float)this->d3[i], TOLERANCE);
      ASSERT_NEAR((float)this->d1[i], (float)this->d3[i], TOLERANCE);

    } else {
      ASSERT_NE((float)this->d2[i], (float)this->d3[i]);
      ASSERT_NE((float)this->d1[i], (float)this->d3[i]);
    }
  }
}

TYPED_TEST(RPUCudaPulsedTestFixtureNoNoise, ForwardIndexedNoNoise) {
  using T = typename TestFixture::T;

  this->x_cuvec_batch->assign(this->rx.data());
  this->x_vec_batch = this->rx;

  this->context->synchronizeDevice();

  // this->culayer_pulsed->printWeights(3, 3);
  // this->layer_pulsed->printWeights(3, 3);
  // this->layer->printWeights(3, 3);

  int input_size = MIN(10, this->x_size * this->m);
  int output_size = this->x_size * this->m;

  int *index = new int[output_size];
  for (int j = 0; j < output_size; j++) {
    index[j] = (output_size - j - 1) % (input_size + 2);
  }
  T *orig_input = new T[input_size * this->dim3];
  for (int j = 0; j < input_size * this->dim3; j++) {
    orig_input[j] = this->rx[j];
  }

  T *indexed_input = new T[output_size * this->dim3];
  for (int j = 0; j < output_size * this->dim3; j++) {
    int i = j % output_size;
    int k = j / output_size;
    int ii = index[i];
    indexed_input[j] = (ii <= 1) ? (T)ii : orig_input[((int)ii - 2) + input_size * k];
    // std::cout << "ind input idx  " << j << " is " << indexed_input[j] << std::endl;
  }

  CudaArray<int> cu_index(this->context, this->x_size * this->m, index);
  CudaArray<T> cu_orig_input(this->context, input_size * this->dim3, orig_input);
  CudaArray<T> cu_indexed_input(this->context, this->x_size * this->m * this->dim3, indexed_input);
  CudaArray<T> d_cuvec_batch2(this->context, this->d_size * this->m * this->dim3);
  CudaArray<int> cu_batch_indices(this->context, this->m * this->dim3, this->batch_indices);

  // test cu Pulsed
  CUDA_TIMING_INIT;
  for (bool trans : {false, true}) {

    this->layer_pulsed->setMatrixIndices(index);
    this->layer_pulsed->forwardIndexed(
        orig_input, this->d1.data(), input_size * this->dim3, this->m, this->dim3, trans,
        this->is_test);
    this->layer_pulsed->forwardTensor(
        indexed_input, this->d2.data(), false, this->m, this->dim3, trans, this->is_test);
    // std::cout << "\nCPU Indexed (" << trans << "):";
    for (int i = 0; i < d_cuvec_batch2.getSize(); i++) {
      ASSERT_FLOAT_EQ(this->d1[i], this->d2[i]);
    }
    // std::cout << "  success!\n";

    this->context->synchronizeDevice();
    this->culayer_pulsed->setMatrixIndices(cu_index.getData());

    CUDA_TIMING_START(this->context);
    this->culayer_pulsed->forwardIndexed(
        cu_orig_input.getDataConst(), d_cuvec_batch2.getData(), input_size * this->dim3, this->m,
        this->dim3, trans, this->is_test);
    CUDA_TIMING_STOP(this->context, "Forward Indexed " << trans);
    CUDA_TIMING_START(this->context);
    this->culayer_pulsed->forwardTensor(
        cu_indexed_input.getDataConst(), this->d_cuvec_batch->getData(), false, this->m, this->dim3,
        trans, this->is_test);
    CUDA_TIMING_STOP(this->context, "Forward Tensor " << trans);
    this->context->synchronizeDevice();

    this->d_cuvec_batch->copyTo(this->d1.data());
    d_cuvec_batch2.copyTo(this->d2.data()); // indexed version

    for (int i = 0; i < d_cuvec_batch2.getSize(); i++) {
      ASSERT_FLOAT_EQ(this->d1[i], this->d2[i]);
    }
  }

  CUDA_TIMING_DESTROY;
  delete[] indexed_input;
  delete[] orig_input;
  delete[] index;
}

TYPED_TEST(RPUCudaPulsedTestFixtureNoNoise, BackwardIndexedNoNoise) {
  using T = typename TestFixture::T;

  this->d_cuvec_batch->assign(this->rd.data());
  this->d_vec_batch = this->rd;

  this->context->synchronizeDevice();

  // this->culayer_pulsed->printWeights(3, 3);
  // this->layer_pulsed->printWeights(3, 3);
  // this->layer->printWeights(3, 3);

  int input_size = this->x_size * this->m;
  int output_size = MIN(10, this->x_size * this->m);

  int *index = new int[input_size];
  for (int j = 0; j < input_size; j++) {
    index[j] = (output_size - j - 1) + 2;
    if (index[j] > output_size + 1) {
      index[j] = 1;
    }
  }

  CudaArray<int> cu_index(this->context, this->x_size * this->m, index);
  CudaArray<T> x_cuvec_batch2(this->context, this->x_size * this->m * this->dim3);
  CudaArray<T> x_cuvec_batch3(this->context, this->x_size * this->m * this->dim3);
  CudaArray<int> cu_batch_indices(this->context, this->m * this->dim3, this->batch_indices);

  // test cu Pulsed
  CUDA_TIMING_INIT;
  for (bool trans : {false, true}) {

    this->layer_pulsed->setMatrixIndices(index);
    this->layer_pulsed->backwardIndexed(
        this->d_vec_batch.data(), this->x1.data(), output_size * this->dim3, this->m, this->dim3,
        trans);
    this->layer_pulsed->backwardTensor(
        this->d_vec_batch.data(), this->x2.data(), false, this->m, this->dim3, trans);

    T *indexed_output = new T[output_size * this->dim3];
    for (int j = 0; j < output_size * this->dim3; j++) {
      indexed_output[j] = 0;
    }

    for (int j = 0; j < input_size * this->dim3; j++) {
      int i = j % input_size;
      int k = j / input_size;
      int ii = index[i];
      if (ii > 1) {
        indexed_output[((int)ii - 2) + output_size * k] += this->x2[j];
      }
    }

    // std::cout << "\nCPU Indexed (" << trans << "):";
    for (int i = 0; i < output_size * this->dim3; i++) {
      ASSERT_NEAR((float)this->x1[i], (float)indexed_output[i], 1e-6);
    }
    // std::cout << "  success!\n";

    this->context->synchronizeDevice();
    this->culayer_pulsed->setMatrixIndices(cu_index.getData());

    CUDA_TIMING_START(this->context);
    this->culayer_pulsed->backwardIndexed(
        this->d_cuvec_batch->getData(), this->x_cuvec_batch->getData(), output_size * this->dim3,
        this->m, this->dim3, trans);
    CUDA_TIMING_STOP(this->context, "Backward Indexed " << trans);
    CUDA_TIMING_START(this->context);
    this->culayer_pulsed->backwardTensor(
        this->d_cuvec_batch->getData(), x_cuvec_batch2.getData(), false, this->m, this->dim3,
        trans);
    CUDA_TIMING_STOP(this->context, "Backward Tensor " << trans);

    this->context->synchronizeDevice();

    this->x_cuvec_batch->copyTo(this->x1.data());
    x_cuvec_batch2.copyTo(this->x2.data());

    for (int j = 0; j < output_size * this->dim3; j++) {
      indexed_output[j] = 0;
    }

    for (int j = 0; j < input_size * this->dim3; j++) {
      int i = j % input_size;
      int k = j / input_size;
      int ii = index[i];
      if (ii > 1)
        indexed_output[((int)ii - 2) + output_size * k] += this->x2[j];
    }

    for (int i = 0; i < output_size * this->dim3; i++) {
      ASSERT_FLOAT_EQ((float)this->x1[i], (float)indexed_output[i]);
    }

    delete[] indexed_output;
  }

  CUDA_TIMING_DESTROY;
  delete[] index;
}

TYPED_TEST(RPUCudaPulsedTestFixtureNoNoise, BackwardMatrixBiasNoNoise) {

  this->d_cuvec_batch->assign(this->rd.data());
  this->d_vec_batch = this->rd;

  this->context->synchronizeDevice();

  // this->culayer_pulsed->printWeights(3, 3);
  // this->layer_pulsed->printWeights(3, 3);
  // this->layer->printWeights(3, 3);
  // this->culayer_pulsed->dispParameter();

  // RPUCuda backward
  this->culayer_pulsed->backward(
      this->d_cuvec_batch->getData(), this->x_cuvec_batch->getData(), true, this->m_batch, true,
      true);
  this->context->synchronizeDevice();
  this->x_cuvec_batch->copyTo(this->x1.data());
  this->context->synchronizeDevice();

  // RPU backward
  this->layer_pulsed->backward(
      this->d_vec_batch.data(), this->x_vec_batch.data(), true, this->m_batch, true, true);
  this->x2 = this->x_vec_batch;

  this->layer->backward(
      this->d_vec_batch.data(), this->x_vec_batch.data(), true, this->m_batch, true, true);
  this->x3 = this->x_vec_batch;

  for (int i = 0; i < (this->x_size - 1) * this->m_batch; i++) {
    ASSERT_NEAR((float)this->x1[i], (float)this->x2[i], TOLERANCE);

    if (!this->non_linearities) {
      ASSERT_NEAR((float)this->x1[i], (float)this->x3[i], TOLERANCE);

      ASSERT_NEAR((float)this->x2[i], (float)this->x3[i], TOLERANCE);

    } else {
      ASSERT_NE((float)this->x2[i], (float)this->x3[i]);
      ASSERT_NE((float)this->x1[i], (float)this->x3[i]);
    }
  }
}

TYPED_TEST(RPUCudaPulsedTestFixtureNoNoise, GetWeights) {
  using T = typename TestFixture::T;

  T *w1 = new T[this->x_size * this->d_size];
  T *w2 = new T[this->x_size * this->d_size];

  this->layer_pulsed->getWeights(w1);
  this->culayer_pulsed->getWeights(w2);

  for (int i = 0; i < this->d_size * this->x_size; i++) {
    ASSERT_FLOAT_EQ(w1[i], w2[i]);
  }

  delete[] w1;
  delete[] w2;
}

TYPED_TEST(RPUCudaPulsedTestFixtureNoNoise, GetWeightsReal) {
  using T = typename TestFixture::T;

  T *w1 = new T[this->x_size * this->d_size];
  T *w2 = new T[this->x_size * this->d_size];
  T *w1_ref = new T[this->x_size * this->d_size];
  T *w2_ref = new T[this->x_size * this->d_size];

  this->layer_pulsed->getWeights(w1_ref);
  this->culayer_pulsed->getWeights(w2_ref);

  this->layer_pulsed->getWeightsReal(w1);
  this->culayer_pulsed->getWeightsReal(w2);

  T avg_dev1 = 0.0, avg_dev2 = 0.0;

  for (int i = 0; i < this->d_size * this->x_size; i++) {

    avg_dev1 += fabsf(w1[i] - w1_ref[i]);
    avg_dev2 += fabsf(w2[i] - w2_ref[i]);
  }
  avg_dev1 /= this->d_size * this->x_size;
  avg_dev2 /= this->d_size * this->x_size;

  EXPECT_NEAR(avg_dev1, avg_dev2, avg_dev1);
  EXPECT_NEAR(avg_dev1, 0.001, 0.002);

  delete[] w1;
  delete[] w2;
  delete[] w1_ref;
  delete[] w2_ref;
}

TYPED_TEST(RPUCudaPulsedTestFixtureNoNoise, SetWeightsReal) {
  using T = typename TestFixture::T;

  T *w1 = new T[this->x_size * this->d_size];
  T *w2 = new T[this->x_size * this->d_size];
  T *w1_ref = new T[this->x_size * this->d_size];
  T *w2_ref = new T[this->x_size * this->d_size];

  this->layer_pulsed->getWeights(w1_ref);
  this->culayer_pulsed->getWeights(w2_ref);

  // reverse weights
  for (int i = 0; i < this->d_size * this->x_size; i++) {
    w1_ref[this->d_size * this->x_size - i - 1] = w2_ref[i];
  }
  for (int i = 0; i < this->d_size * this->x_size; i++) {
    w2_ref[i] = w1_ref[i];
  }

  this->layer_pulsed->setWeightsReal(w1_ref);
  this->culayer_pulsed->setWeightsReal(w2_ref);

  this->layer_pulsed->getWeights(w1);
  this->culayer_pulsed->getWeights(w2);

  T avg_dev1 = 0.0, avg_dev2 = 0.0;
  // T max_dev1 = 0.0, max_dev2 = 0.0;
  // int max_dev1_i = 0;
  // int max_dev2_i = 0;

  for (int i = 0; i < this->d_size * this->x_size; i++) {
    avg_dev1 += fabsf(w1[i] - w1_ref[i]);
    avg_dev2 += fabsf(w2[i] - w2_ref[i]);

    // if (max_dev1 < fabsf(w1[i] - w1_ref[i])) {
    //   max_dev1 = fabsf(w1[i] - w1_ref[i]);
    //   max_dev1_i = i;
    // }

    // if (max_dev2 < fabsf(w2[i] - w2_ref[i])) {
    //   max_dev2 = fabsf(w2[i] - w2_ref[i]);
    //   max_dev2_i = i;
    // }

    // std::cout << i << ": " << w1[i] << " vs [ref] " << w1_ref[i] << "\n";
    // std::cout << i << ": " << w2[i] << " vs [ref] " << w2_ref[i] << "\n";
  }
  avg_dev1 /= this->d_size * this->x_size;
  avg_dev2 /= this->d_size * this->x_size;
  // std::cout << "max_dev1 is " << max_dev1 << " idx is " << max_dev1_i << "\n";
  // std::cout << "max_dev2 is " << max_dev2 << " idx is " << max_dev2_i << "\n";

  EXPECT_NEAR(avg_dev1, avg_dev2, avg_dev1);
  EXPECT_NEAR(avg_dev1, 0.001, 0.002);

  delete[] w1;
  delete[] w2;
  delete[] w1_ref;
  delete[] w2_ref;
}

TYPED_TEST(RPUCudaPulsedTestFixture, ForwardMatrix) {
  using T = typename TestFixture::T;

  this->x_cuvec_batch->assign(this->rx.data());
  this->x_vec_batch = this->rx;

  this->context->synchronizeDevice();

  // this->culayer_pulsed->printWeights(3, 3);
  // this->layer_pulsed->printWeights(3, 3);

  int nloop = 2;
  double cudur = 0, dur = 0;

  T *cuavg = new T[this->d_size];
  T *avg = new T[this->d_size];

  for (int i = 0; i < this->d_size; i++) {
    cuavg[i] = 0;
    avg[i] = 0;
  }
  CUDA_TIMING_INIT;

  auto start_time = std::chrono::high_resolution_clock::now();
  auto end_time = std::chrono::high_resolution_clock::now();
  for (int loop = 0; loop < nloop; loop++) {

    // RPUCuda forward
    start_time = std::chrono::high_resolution_clock::now();
    CUDA_TIMING_START(this->context);
    this->culayer_pulsed->forward(
        this->x_cuvec_batch->getData(), this->d_cuvec_batch->getData(), false, this->m_batch, false,
        false, this->is_test);
    CUDA_TIMING_STOP_NO_OUTPUT(this->context);
    if (loop > 0)
      cudur += milliseconds;

    // RPU forward
    start_time = std::chrono::high_resolution_clock::now();
    this->layer_pulsed->forward(
        this->x_vec_batch.data(), this->d_vec_batch.data(), false, this->m_batch, false, false,
        this->is_test);
    end_time = std::chrono::high_resolution_clock::now();
    if (loop > 0)
      dur += std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();

    this->d2 = this->d_vec_batch;
    this->d_cuvec_batch->copyTo(this->d1.data());
    this->context->synchronizeDevice();

    for (int i = 0; i < this->d_size * this->m_batch; i++) {
      cuavg[i % this->m_batch] += this->d1[i] / (num_t)nloop / (num_t)this->m_batch;
      avg[i % this->m_batch] += this->d2[i] / (num_t)nloop / (num_t)this->m_batch;

      if (!this->noise_value) {
        EXPECT_FLOAT_EQ(this->d1[i], this->d2[i]);
      }
    }
  }

  std::cout << BOLD_ON << "\tCUDA Forward Matrix done in: " << (float)cudur / (nloop - 1) << " msec"
            << BOLD_OFF << std::endl;
  std::cout << BOLD_ON << "\tCPU Forward Matrix done in: " << (float)dur / 1000. / (nloop - 1)
            << " msec" << BOLD_OFF << std::endl;

  for (int i = 0; i < this->d_size; i++) {
    // std::cout  << "ref:" << avg[i] << ", cu out: " << cuavg[i] << std::endl;
    ASSERT_NEAR((float)cuavg[i], (float)avg[i], 1. / sqrtf(nloop));
  }

  delete[] avg;
  delete[] cuavg;

  CUDA_TIMING_DESTROY;
}

TYPED_TEST(RPUCudaPulsedTestFixture, BackwardVector) {
  using T = typename TestFixture::T;

  this->d_cuvec->assign(this->rd.data());
  this->d_vec = this->rd;
  // T max_value = Find_Absolute_Max<T>(this->rd.data(), this->d_size);
  // std::cout << "Max value Reference: " << max_value << std::endl;
  this->context->synchronizeDevice();

  // this->culayer_pulsed->printWeights(3, 3);
  // this->layer_pulsed->printWeights(3, 3);

  // this->culayer_pulsed->dispParameter();

  T *cuavg = new T[this->x_size];
  T *avg = new T[this->x_size];

  for (int i = 0; i < this->x_size; i++) {
    cuavg[i] = 0;
    avg[i] = 0;
  }

  int nloop = this->repeats;
  double cudur = 0, dur = 0;
  CUDA_TIMING_INIT;

  auto start_time = std::chrono::high_resolution_clock::now();
  auto end_time = std::chrono::high_resolution_clock::now();
  for (int loop = 0; loop < nloop; loop++) {

    // RPUCuda backward
    CUDA_TIMING_START(this->context);
    this->culayer_pulsed->backward(this->d_cuvec->getData(), this->x_cuvec->getData(), false, 1);
    CUDA_TIMING_STOP_NO_OUTPUT(this->context);
    if (loop > 0)
      cudur += milliseconds;

    // RPU backward
    start_time = std::chrono::high_resolution_clock::now();
    this->layer_pulsed->backward(this->d_vec.data(), this->x_vec.data(), false, 1);
    end_time = std::chrono::high_resolution_clock::now();
    if (loop > 0)
      dur += std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();

    this->x2 = this->x_vec;
    this->x_cuvec->copyTo(this->x1.data());

    this->context->synchronizeDevice();

    for (int i = 0; i < this->x_size; i++) {
      cuavg[i] += this->x1[i] / (num_t)nloop;
      avg[i] += this->x2[i] / (num_t)nloop;

      if (!this->noise_value) {
        EXPECT_FLOAT_EQ(this->x1[i], this->x2[i]);
      }
    }
  }
  std::cout << BOLD_ON << "\tCUDA Backwards done in: " << (float)cudur / (nloop - 1) << " msec"
            << BOLD_OFF << std::endl;
  std::cout << BOLD_ON << "\tCPU Backwards done in: " << (float)dur / 1000. / (nloop - 1) << " msec"
            << BOLD_OFF << std::endl;

  for (int i = 0; i < this->x_size; i++) {
    // std::cout  << "ref:" << avg[i] << ", cu out: " << cuavg[i] << std::endl;
    ASSERT_NEAR((float)cuavg[i], (float)avg[i], 1. / sqrtf(nloop));
  }

  delete[] avg;
  delete[] cuavg;
  CUDA_TIMING_DESTROY;
}

TYPED_TEST(RPUCudaPulsedTestFixture, BackwardMatrix) {
  using T = typename TestFixture::T;

  this->d_cuvec_batch->assign(this->rd.data());
  this->d_vec_batch = this->rd;
  this->context->synchronizeDevice();

  // this->culayer_pulsed->printWeights(3, 3);
  // this->layer_pulsed->printWeights(3, 3);

  T *cuavg = new T[this->x_size];
  T *avg = new T[this->x_size];

  for (int i = 0; i < this->x_size; i++) {
    cuavg[i] = 0;
    avg[i] = 0;
  }

  int nloop = this->repeats;
  double cudur = 0, dur = 0;
  CUDA_TIMING_INIT;

  auto start_time = std::chrono::high_resolution_clock::now();
  auto end_time = std::chrono::high_resolution_clock::now();
  for (int loop = 0; loop < nloop; loop++) {

    // RPUCuda backward
    CUDA_TIMING_START(this->context);
    this->culayer_pulsed->backward(
        this->d_cuvec_batch->getData(), this->x_cuvec_batch->getData(), false, this->m_batch);
    CUDA_TIMING_STOP_NO_OUTPUT(this->context);
    if (loop > 0)
      cudur += milliseconds;

    // RPU backward

    start_time = std::chrono::high_resolution_clock::now();
    this->layer_pulsed->backward(
        this->d_vec_batch.data(), this->x_vec_batch.data(), false, this->m_batch);
    end_time = std::chrono::high_resolution_clock::now();
    if (loop > 0)
      dur += std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();

    this->x2 = this->x_vec_batch;
    this->x_cuvec_batch->copyTo(this->x1.data());

    this->context->synchronizeDevice();

    for (int i = 0; i < this->x_size * this->m_batch; i++) {
      cuavg[i % this->m_batch] += this->x1[i] / (num_t)nloop / (num_t)this->m_batch;
      avg[i % this->m_batch] += this->x2[i] / (num_t)nloop / (num_t)this->m_batch;

      if (!this->noise_value) {
        // if ((fabsf(this->x2[i] - this->x1[i])) > 1e-6)
        // std::cout << "i " << i << " x_size " << this->x_size << " batch: " << this->m_batch
        //          << std::endl;
        EXPECT_FLOAT_EQ(this->x1[i], this->x2[i]);
        //!! note: could have a rounding difference!
      }
    }
  }
  std::cout << BOLD_ON << "\tCUDA Backward Mat done in: " << (float)cudur / (nloop - 1) << " msec"
            << BOLD_OFF << std::endl;
  std::cout << BOLD_ON << "\tCPU Backward Mat done in: " << (float)dur / 1000. / (nloop - 1)
            << " msec" << BOLD_OFF << std::endl;

  for (int i = 0; i < this->x_size; i++) {
    // std::cout  << "ref:" << avg[i] << ", cu out: " << cuavg[i] << std::endl;
    ASSERT_NEAR((float)cuavg[i], (float)avg[i], 1. / sqrtf(nloop));
  }

  delete[] avg;
  delete[] cuavg;
  CUDA_TIMING_DESTROY;
}

#define RPU_TEST_UPDATE(CUFUN, FUN, NLOOP)                                                         \
  this->context->synchronizeDevice();                                                              \
  bool no_noise = !this->layer_pulsed->getRPUDevice().isPulsedDevice();                            \
  int n = this->x_size * this->d_size;                                                             \
  T **refweights = Array_2D_Get<T>(this->d_size, this->x_size);                                    \
  T **w = this->layer_pulsed->getWeightsPtr();                                                     \
  for (int i = 0; i < this->d_size; i++) {                                                         \
    for (int j = 0; j < this->x_size; j++) {                                                       \
      refweights[i][j] = w[i][j];                                                                  \
    }                                                                                              \
  }                                                                                                \
                                                                                                   \
  T *cuavg = new T[n];                                                                             \
  T *avg = new T[n];                                                                               \
  T *cusig = new T[n];                                                                             \
  T *sig = new T[n];                                                                               \
                                                                                                   \
  for (int i = 0; i < n; i++) {                                                                    \
    cuavg[i] = 0;                                                                                  \
    avg[i] = 0;                                                                                    \
    cusig[i] = 0;                                                                                  \
    sig[i] = 0;                                                                                    \
  }                                                                                                \
  int nloop = NLOOP;                                                                               \
  double cudur = 0, dur = 0;                                                                       \
  CUDA_TIMING_INIT;                                                                                \
  auto start_time = std::chrono::high_resolution_clock::now();                                     \
  auto end_time = std::chrono::high_resolution_clock::now();                                       \
  for (int loop = 0; loop < nloop; loop++) {                                                       \
                                                                                                   \
    start_time = std::chrono::high_resolution_clock::now();                                        \
    CUDA_TIMING_START(this->context);                                                              \
    this->culayer_pulsed->CUFUN;                                                                   \
    CUDA_TIMING_STOP_NO_OUTPUT(this->context);                                                     \
    if (loop > 0)                                                                                  \
      cudur += milliseconds;                                                                       \
                                                                                                   \
    start_time = std::chrono::high_resolution_clock::now();                                        \
    this->layer_pulsed->FUN;                                                                       \
    end_time = std::chrono::high_resolution_clock::now();                                          \
    if (loop > 0)                                                                                  \
      dur += std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count(); \
                                                                                                   \
    T **cuweights = this->culayer_pulsed->getWeights();                                            \
    T **weights = this->layer_pulsed->getWeights();                                                \
    this->context->synchronizeDevice();                                                            \
    for (int i = 0; i < this->d_size; i++) {                                                       \
      for (int j = 0; j < this->x_size; j++) {                                                     \
        int k = j + i * this->x_size;                                                              \
        cuavg[k] += cuweights[i][j] / (num_t)nloop;                                                \
        avg[k] += weights[i][j] / (num_t)nloop;                                                    \
        if (no_noise) {                                                                            \
          ASSERT_NEAR((float)cuweights[i][j], (float)weights[i][j], TOLERANCE);                    \
        }                                                                                          \
        cusig[k] += cuweights[i][j] * cuweights[i][j] / (num_t)nloop;                              \
        sig[k] += weights[i][j] * weights[i][j] / (num_t)nloop;                                    \
                                                                                                   \
        /*weights[i][j] = refweights[i][j];*/                                                      \
      }                                                                                            \
    }                                                                                              \
    if (loop < nloop - 1) {                                                                        \
      this->culayer_pulsed->setWeights(refweights[0]);                                             \
      this->context->synchronizeDevice();                                                          \
      this->layer_pulsed->setWeights(refweights[0]);                                               \
    }                                                                                              \
  }                                                                                                \
  std::cout << BOLD_ON << "\tCUDA done in: " << (float)cudur / (nloop - 1) << " msec " << BOLD_OFF \
            << std::endl;                                                                          \
  std::cout << BOLD_ON << "\tCPU done in: " << (float)dur / 1000. / (nloop - 1) << " msec"         \
            << BOLD_OFF << std::endl;                                                              \
  for (int k = 0; k < n; k++) {                                                                    \
    T sigi = sqrt(fabsf(sig[k] - avg[k] * avg[k]));                                                \
    T cusigi = sqrt(fabsf(cusig[k] - cuavg[k] * cuavg[k]));                                        \
                                                                                                   \
    ASSERT_NEAR((float)avg[k], (float)cuavg[k], 2. / sqrtf(nloop));                                \
    ASSERT_NEAR((float)sigi, (float)cusigi, 2. / sqrtf(nloop));                                    \
  }                                                                                                \
  delete[] cuavg;                                                                                  \
  delete[] avg;                                                                                    \
  delete[] cusig;                                                                                  \
  delete[] sig;                                                                                    \
                                                                                                   \
  Array_2D_Free(refweights);                                                                       \
  CUDA_TIMING_DESTROY;

TYPED_TEST(RPUCudaPulsedTestFixture, WeightDecay) {
  using T = typename TestFixture::T;

  RPU_TEST_UPDATE(decayWeights(false), decayWeights(false), this->repeats);
}
TYPED_TEST(RPUCudaPulsedTestFixture, WeightDecayNoBias) {
  using T = typename TestFixture::T;

  RPU_TEST_UPDATE(decayWeights(true), decayWeights(true), this->repeats);
}

TYPED_TEST(RPUCudaPulsedTestFixture, WeightDecayAlpha) {
  using T = typename TestFixture::T;

  RPU_TEST_UPDATE(decayWeights(0.3, false), decayWeights(0.3, false), this->repeats);
}

TYPED_TEST(RPUCudaPulsedTestFixture, WeightDecayAlphaNoBias) {
  using T = typename TestFixture::T;

  RPU_TEST_UPDATE(decayWeights(0.3, true), decayWeights(0.3, true), this->repeats);
}

TYPED_TEST(RPUCudaPulsedTestFixture, ClipWeights) {
  using T = typename TestFixture::T;

  RPU_TEST_UPDATE(clipWeights(0.2), clipWeights(0.2), this->repeats);
}

TYPED_TEST(RPUCudaPulsedTestFixture, ResetCols) {
  using T = typename TestFixture::T;

  RPU_TEST_UPDATE(resetCols(1, 3, 1), resetCols(1, 3, 1), this->repeats);
}

TYPED_TEST(RPUCudaPulsedTestFixture, WeightDiffusion) {
  using T = typename TestFixture::T;

  RPU_TEST_UPDATE(diffuseWeights(), diffuseWeights(), this->repeats);
}

TYPED_TEST(RPUCudaPulsedTestFixture, AssignTranspose) {
  // better check with one
  this->w_cuother->assignTranspose(this->w_other, this->d_size, this->x_size);
  this->w_cuother->copyTo(this->w_other);

  for (int i = 0; i < this->d_size * this->x_size; i++) {
    ASSERT_FLOAT_EQ(this->w_other[i], this->w_other_trans[i]);
  }
}

TYPED_TEST(RPUCudaPulsedTestFixture, PreAllReduce) {
  using T = typename TestFixture::T;

  T *w1 = new T[this->x_size * this->d_size];
  T *w2 = new T[this->x_size * this->d_size];
  T *w1_ref = new T[this->x_size * this->d_size];
  T *w2_ref = new T[this->x_size * this->d_size];
  T *w_other_copy = new T[this->x_size * this->d_size];

  for (int i = 0; i < this->d_size * this->x_size; i++) {
    w_other_copy[i] = this->w_other[i];
  }

  this->culayer_pulsed->getWeights(w1_ref);
  this->layer_pulsed->getWeights(w2_ref);

  // check same weights
  for (int i = 0; i < this->d_size * this->x_size; i++) {
    ASSERT_FLOAT_EQ(w1_ref[i], w2_ref[i]);
  }
  // check transposed assignment
  this->w_cuother->copyTo(w1); // transposed
  for (int i = 0; i < this->d_size * this->x_size; i++) {
    ASSERT_FLOAT_EQ(w1[i], this->w_other_trans[i]);
  }

  this->context->synchronizeDevice();
  CUDA_TIMING_INIT;
  CUDA_TIMING_START(this->context);
  this->culayer_pulsed->getAndResetWeightUpdate(this->w_cuother->getData());
  CUDA_TIMING_STOP(this->context, "Get_And_Reset");
  this->layer_pulsed->getAndResetWeightUpdate(this->w_other);
  this->context->synchronizeDevice();

  this->culayer_pulsed->getWeights(w1);
  this->layer_pulsed->getWeights(w2);
  this->context->synchronizeDevice();

  for (int i = 0; i < this->x_size * this->d_size; i++) {
    ASSERT_FLOAT_EQ(w2[i], w_other_copy[i]);
    ASSERT_FLOAT_EQ(
        w1[i], w_other_copy[i]); // because PreReduce resets the weights to the given weights
  }

  this->w_cuother->copyTo(w1); // transposed
  transpose(this->w_other_trans, w1, this->d_size, this->x_size);

  for (int i = 0; i < this->x_size * this->d_size; i++) {
    ASSERT_FLOAT_EQ(this->w_other_trans[i], w1_ref[i] - w_other_copy[i]);
    ASSERT_FLOAT_EQ(this->w_other[i], w1_ref[i] - w_other_copy[i]);
  }

  delete[] w1;
  delete[] w2;
  delete[] w1_ref;
  delete[] w2_ref;
  delete[] w_other_copy;
  CUDA_TIMING_DESTROY;
}

TYPED_TEST(RPUCudaPulsedTestFixture, PostAllReduce) {
  using T = typename TestFixture::T;

  T *w1 = new T[this->x_size * this->d_size];
  T *w2 = new T[this->x_size * this->d_size];
  T *w1_ref = new T[this->x_size * this->d_size];
  T *w2_ref = new T[this->x_size * this->d_size];
  T *w_other_copy = new T[this->x_size * this->d_size];

  for (int i = 0; i < this->d_size * this->x_size; i++) {
    w_other_copy[i] = this->w_other[i];
  }

  this->culayer_pulsed->getWeights(w1_ref);
  this->layer_pulsed->getWeights(w2_ref);

  // check same weights
  for (int i = 0; i < this->d_size * this->x_size; i++) {
    ASSERT_FLOAT_EQ(w1_ref[i], w2_ref[i]);
  }

  // check transposed assignment
  this->w_cuother->copyTo(w1); // transposed
  for (int i = 0; i < this->d_size * this->x_size; i++) {
    ASSERT_FLOAT_EQ(w1[i], this->w_other_trans[i]);
  }

  this->context->synchronizeDevice();
  CUDA_TIMING_INIT;
  CUDA_TIMING_START(this->context);
  this->culayer_pulsed->applyWeightUpdate(this->w_cuother->getData());
  CUDA_TIMING_STOP(this->context, "Apply_Weight_Update");
  this->layer_pulsed->applyWeightUpdate(this->w_other);
  this->context->synchronizeDevice();

  this->culayer_pulsed->getWeights(w1);
  this->layer_pulsed->getWeights(w2);
  this->context->synchronizeDevice();

  for (int i = 0; i < this->x_size * this->d_size; i++) {
    ASSERT_FLOAT_EQ(w2[i], w1_ref[i] + w_other_copy[i]);
    ASSERT_FLOAT_EQ(w1[i], w1_ref[i] + w_other_copy[i]); // because PostReduce adds DW to weights
  }

  this->w_cuother->copyTo(w1); // transposed
  transpose(this->w_other_trans, w1, this->d_size, this->x_size);

  for (int i = 0; i < this->x_size * this->d_size; i++) {
    ASSERT_FLOAT_EQ(this->w_other_trans[i], w1_ref[i] + w_other_copy[i]); // copy weights to output
    ASSERT_FLOAT_EQ(this->w_other[i], w1_ref[i] + w_other_copy[i]);
  }

  delete[] w1;
  delete[] w2;
  delete[] w1_ref;
  delete[] w2_ref;
  delete[] w_other_copy;
  CUDA_TIMING_DESTROY;
}

TYPED_TEST(RPUCudaPulsedTestFixture, UpdateVector) {
  using T = typename TestFixture::T;

  this->x_cuvec->assign(this->rx.data());
  this->x_vec = this->rx;

  this->d_cuvec->assign(this->rd.data());
  this->d_vec = this->rd;

  this->context->synchronizeDevice();

  RPU_TEST_UPDATE(
      update(this->x_cuvec->getData(), this->d_cuvec->getData(), false),
      update(this->x_vec.data(), this->d_vec.data(), false), this->repeats);
}

TYPED_TEST(RPUCudaPulsedTestFixtureNoNoise, UpdateMatrixBatchIndexedNoNoise) {
  using T = typename TestFixture::T;

  this->x_cuvec_batch->assign(this->rx.data());
  this->x_vec_batch = this->rx;
  this->d_cuvec_batch->assign(this->rd.data());
  this->d_vec_batch = this->rd;

  this->context->synchronizeDevice();

  bool trans = true;

  int input_size = MIN(10, this->x_size * this->m);
  int output_size = this->x_size * this->m;

  int *index = new int[output_size];
  for (int j = 0; j < output_size; j++) {
    index[j] = (output_size - j - 1) % (input_size + 2);
  }
  T *orig_input = new T[input_size * this->dim3];
  for (int j = 0; j < input_size * this->dim3; j++) {
    orig_input[j] = this->rx[j];
  }

  T *indexed_input = new T[output_size * this->dim3];
  for (int j = 0; j < output_size * this->dim3; j++) {
    int i = j % output_size;
    int k = j / output_size;
    int ii = index[i];
    indexed_input[j] = (ii <= 1) ? (T)ii : orig_input[((int)ii - 2) + input_size * k];
    // std::cout << "ind input idx  " << j << " is " << indexed_input[j] << std::endl;
  }

  CudaArray<int> cu_index(this->context, this->x_size * this->m, index);
  CudaArray<T> cu_orig_input(this->context, input_size * this->dim3, orig_input);
  CudaArray<T> cu_indexed_input(this->context, this->x_size * this->m * this->dim3, indexed_input);
  CudaArray<int> cu_batch_indices(this->context, this->m * this->dim3, this->batch_indices);

  this->culayer_pulsed->setMatrixIndices(cu_index.getData());
  this->layer_pulsed->setMatrixIndices(index);

  this->context->synchronizeDevice();

  {
    RPU_TEST_UPDATE(
        updateIndexed(
            cu_orig_input.getData(), this->d_cuvec_batch->getData(), input_size * this->dim3,
            this->m, this->dim3, trans),
        updateTensor(indexed_input, this->d_vec_batch.data(), false, this->m, this->dim3, trans),
        2);
  }

  this->context->synchronizeDevice();

  {
    RPU_TEST_UPDATE(
        updateIndexed(
            cu_orig_input.getData(), this->d_cuvec_batch->getData(), input_size * this->dim3,
            this->m, this->dim3, trans),
        updateIndexed(
            orig_input, this->d_vec_batch.data(), input_size * this->dim3, this->m, this->dim3,
            trans),
        2);
  }

  this->context->synchronizeDevice();

  {
    RPU_TEST_UPDATE(
        updateIndexedSlice(
            cu_orig_input.getData(), this->d_cuvec_batch->getData(), input_size * this->dim3,
            this->m, this->dim3, trans, this->m, cu_batch_indices.getDataConst()),
        updateIndexedSlice(
            orig_input, this->d_vec_batch.data(), input_size * this->dim3, this->m, this->dim3,
            trans, this->m, this->batch_indices),
        2);
  }

  delete[] orig_input;
  delete[] index;
  delete[] indexed_input;
}

TYPED_TEST(RPUCudaPulsedTestFixture, UpdateMatrixBatch) {
  using T = typename TestFixture::T;

  this->x_cuvec_batch->assign(this->rx.data());
  this->x_vec_batch = this->rx;

  this->d_cuvec_batch->assign(this->rd.data());
  this->d_vec_batch = this->rd;

  RPU_TEST_UPDATE(
      update(this->x_cuvec_batch->getData(), this->d_cuvec_batch->getData(), false, this->m_batch),
      update(this->x_vec_batch.data(), this->d_vec_batch.data(), false, this->m_batch), 10);
}

TYPED_TEST(RPUCudaPulsedTestFixtureNoNoise, UpdateMatrixBatch) {
  using T = typename TestFixture::T;

  this->x_cuvec_batch->assign(this->rx.data());
  this->x_vec_batch = this->rx;

  this->d_cuvec_batch->assign(this->rd.data());
  this->d_vec_batch = this->rd;

  RPU_TEST_UPDATE(
      update(this->x_cuvec_batch->getData(), this->d_cuvec_batch->getData(), false, this->m_batch),
      update(this->x_vec_batch.data(), this->d_vec_batch.data(), false, this->m_batch), 10);
}

#undef RPU_TEST_UPDATE
} // namespace

int main(int argc, char **argv) {
  resetCuda();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
