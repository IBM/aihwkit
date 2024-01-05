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
#include "rng.h"
#include "rpu_constantstep_device.h"
#include "rpu_pulsed.h"
#include "rpu_simple_device.h"
#include "rpucuda_constantstep_device.h"
#include "rpucuda_pulsed.h"
#include "rpucuda_simple_device.h"
#include "utility_functions.h"
#include "gtest/gtest.h"
#include <chrono>
#include <iostream>
#include <memory>
#include <random>

#include "io_manager.h"

#define TOLERANCE 1e-6

namespace {

using namespace RPU;

template <typename DeviceParT> class RPUDeviceTestFixture : public ::testing::Test {
public:
  void SetUp() {

    context = &context_container;

    this->x_size = 253;
    this->d_size = 243;
    this->repeats = 100;

    this->rx = new num_t[x_size];
    this->rd = new num_t[d_size];

    this->refweights = Array_2D_Get<num_t>(d_size, x_size);
  }

  void populateLayers() {
    // only for test PulsedType::None with different devices

    PulsedMetaParameter<num_t> p;
    DeviceParT dp;
    IOMetaParameter<num_t> p_io;

    p_io.out_noise = 0.0; // no noise in output;
    dp.dw_min_std = 0.0;
    // dp.dw_min_dtod = 0.0;
    // dp.up_down_dtod = 0.0;

    p.up.pulse_type = PulseType::None;

    // peripheral circuits specs off
    p_io.inp_res = -1;
    p_io.inp_sto_round = false;
    p_io.out_res = -1;

    p_io.noise_management = NoiseManagementType::None;
    p_io.bound_management = BoundManagementType::None;

    p_io.inp_bound = 1000;
    p_io.out_bound = 1000;

    p.f_io = p_io;
    p.b_io = p_io;

    p.up.update_management = true;
    p.up.update_bl_management = false;

    num_t lr = 0.01;

    // cpu
    simple = RPU::make_unique<RPUSimple<num_t>>(x_size, d_size);
    pulsed = RPU::make_unique<RPUPulsed<num_t>>(x_size, d_size);

    pulsed->populateParameter(&p, &dp);
    pulsed->setLearningRate(lr);
    pulsed->setWeightsUniformRandom(-.5, 0.5);
    pulsed->disp();
    pulsed->getWeights(refweights[0]);

    SimpleMetaParameter<num_t> sp;
    simple->populateParameter(&sp);
    simple->setLearningRate(lr);
    simple->setWeights(refweights[0]);

    // cuda
    pulsed_cuda = RPU::make_unique<RPUCudaPulsed<num_t>>(context, *pulsed);
    pulsed_cuda->setLearningRate(lr);
    pulsed_cuda->disp();

    simple_cuda = RPU::make_unique<RPUCudaSimple<num_t>>(context, x_size, d_size);
    simple_cuda->populateParameter(&sp);
    simple_cuda->setWeights(refweights[0]);
    simple_cuda->setLearningRate(lr);

    unsigned int seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator{seed};
    std::uniform_real_distribution<float> udist(-1.2, 1.2);
    auto urnd = std::bind(udist, generator);

    // just assign some numbers from the weight matrix
    for (int i = 0; i < x_size; i++)
      rx[i] = (num_t)urnd();

    for (int j = 0; j < d_size; j++) {
      rd[j] = (num_t)urnd();
    }

    rx_cuda = RPU::make_unique<CudaArray<num_t>>(context, x_size, rx);
    rd_cuda = RPU::make_unique<CudaArray<num_t>>(context, d_size, rd);

    context->synchronizeDevice();
  };

  void runLayerTest() {

    this->populateLayers();

    CUDA_TIMING_INIT;
    // update
    double pulsed_dur = 0;
    double pulsed_cuda_dur = 0;
    double simple_dur = 0;
    double simple_cuda_dur = 0;
    auto start_time = std::chrono::high_resolution_clock::now();
    auto end_time = std::chrono::high_resolution_clock::now();
    for (int loop = 0; loop < this->repeats; loop++) {

      CUDA_TIMING_START((this->context));
      this->pulsed_cuda->update(this->rx_cuda->getData(), rd_cuda->getData(), false, 1);
      CUDA_TIMING_STOP_NO_OUTPUT((this->context));
      if (loop > 0)
        pulsed_cuda_dur += milliseconds;

      context->synchronize();

      CUDA_TIMING_START((this->context));
      this->simple_cuda->update(this->rx_cuda->getData(), rd_cuda->getData(), false, 1);
      CUDA_TIMING_STOP_NO_OUTPUT((this->context));
      if (loop > 0)
        simple_cuda_dur += milliseconds;

      context->synchronize();

      start_time = std::chrono::high_resolution_clock::now();
      this->pulsed->update(this->rx, this->rd, false, 1);
      end_time = std::chrono::high_resolution_clock::now();
      if (loop > 0)
        pulsed_dur +=
            std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();

      start_time = std::chrono::high_resolution_clock::now();
      this->simple->update(this->rx, this->rd, false, 1);
      end_time = std::chrono::high_resolution_clock::now();
      if (loop > 0)
        simple_dur +=
            std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();

      num_t **w1 = this->simple->getWeights();
      num_t **w2 = this->pulsed->getWeights();
      num_t **w3 = this->simple_cuda->getWeights();
      num_t **w4 = this->pulsed_cuda->getWeights();
      this->context->synchronizeDevice();

      for (int i = 0; i < this->d_size; i++) {
        for (int j = 0; j < this->x_size; j++) {
          ASSERT_NEAR(w1[i][j], w2[i][j], TOLERANCE);
          ASSERT_NEAR(w2[i][j], w3[i][j], 1e-5);
          ASSERT_NEAR(w3[i][j], w4[i][j], TOLERANCE);
        }
      }
      this->context->synchronizeDevice();
    }

    std::cout << BOLD_ON << "\tRPU Pulsed Cuda: done in " << pulsed_cuda_dur / (this->repeats - 1)
              << " msec" << std::endl
              << BOLD_OFF;
    // this->pulsed_cuda->printWeights(3, 1);
    std::cout << BOLD_ON << "\tRPU Pulsed: done in "
              << (float)pulsed_dur / 1000. / (this->repeats - 1) << " msec" << std::endl
              << BOLD_OFF;
    // this->pulsed->printWeights(3, 1);
    std::cout << BOLD_ON << "\tRPU Simple Cuda: done in " << simple_cuda_dur / (this->repeats - 1)
              << " msec\n"
              << BOLD_OFF;
    // this->simple_cuda->printWeights(3, 1);
    std::cout << BOLD_ON << "\tRPU Simple: done in "
              << (float)simple_dur / 1000. / (this->repeats - 1) << " msec" << std::endl
              << BOLD_OFF;
    // this->simple->printWeights(3, 1);

    CUDA_TIMING_DESTROY;
  };

  void TearDown() {
    delete[] rx;
    delete[] rd;
    Array_2D_Free(refweights);
  }

  CudaContext context_container{-1, false};
  CudaContextPtr context;
  std::unique_ptr<RPUPulsed<num_t>> pulsed;
  std::unique_ptr<RPUCudaPulsed<num_t>> pulsed_cuda;
  std::unique_ptr<RPUSimple<num_t>> simple;
  std::unique_ptr<RPUCudaSimple<num_t>> simple_cuda;

  std::unique_ptr<CudaArray<num_t>> rx_cuda;
  std::unique_ptr<CudaArray<num_t>> rd_cuda;

  int x_size;
  int d_size;
  int repeats;

  num_t *rx, *rd;
  num_t **refweights;
};

// types
typedef ::testing::Types<ConstantStepRPUDeviceMetaParameter<num_t>> MetaPar;

TYPED_TEST_CASE(RPUDeviceTestFixture, MetaPar);

TYPED_TEST(RPUDeviceTestFixture, SimpleDeviceUpdate) { this->runLayerTest(); }

} // namespace

#undef TOLERANCE

int main(int argc, char **argv) {
  resetCuda();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
