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

#include "cuda.h"
#include "cuda_util.h"
#include "io_manager.h"
#include "rng.h"
#include "rpu_pulsed.h"
#include "rpucuda_constantstep_device.h"
#include "rpucuda_expstep_device.h"
#include "rpucuda_linearstep_device.h"
#include "rpucuda_pulsed.h"
#include "rpucuda_pulsed_device.h"
#include "utility_functions.h"
#include "gtest/gtest.h"
#include <chrono>
#include <iostream>
#include <memory>
#include <random>

#define TOLERANCE 1e-5

#ifdef RPU_USE_DOUBLE
typedef double num_t;
#else
typedef float num_t;
#endif

namespace {

using namespace RPU;

template <typename DeviceParT> class RPUDeviceTestFixture : public ::testing::Test {
public:
  void SetUp() {

    context_container = RPU::make_unique<CudaContext>(-1, false);
    context = &*context_container;

    this->x_size = 53;
    this->d_size = 43;
    this->K = 10;

    this->rx.resize(x_size);
    this->rd.resize(d_size);

    this->refweights = Array_2D_Get<num_t>(d_size, x_size);
  }

  void populateLayers(int kernelidx) {
    repeats = 3;

    num_t bmin = -0.7;
    num_t bmax = 0.7;

    PulsedMetaParameter<num_t> p;
    DeviceParT dp;
    IOMetaParameter<num_t> p_io;

    p_io.out_noise = 0.0; // no noise in output;
    dp.dw_min_std = 0.0;
    // dp.dw_min_dtod = 0.0;
    // dp.asym_dtod = 0.0;

    p.up.desired_BL = K;

    dp.w_max = 1;
    dp.w_min = -1;
    dp.w_min_dtod = 0.1;
    dp.w_max_dtod = 0.1;

    dp.dw_min = 0.01;

    // peripheral circuits specs
    p_io.inp_res = -1;
    p_io.inp_sto_round = false;
    p_io.out_res = -1;

    p_io.noise_management = NoiseManagementType::AbsMax;
    p_io.bound_management = BoundManagementType::Iterative;

    p.f_io = p_io;
    p.b_io = p_io;
    p.b_io.bound_management = BoundManagementType::None;

    p.up.update_management = true;
    p.up.update_bl_management = false;

    p.up.pulse_type = PulseType::StochasticCompressed;
    p.up._debug_kernel_index = abs(kernelidx);

    dp.print();

    num_t lr = 0.05;

    layer_pulsed = RPU::make_unique<RPUPulsed<num_t>>(x_size, d_size);

    layer_pulsed->populateParameter(&p, &dp);
    layer_pulsed->setLearningRate(lr);
    layer_pulsed->setWeightsUniformRandom(bmin, bmax);
    layer_pulsed->disp();

    this->layer_pulsed->getWeights(refweights[0]);

    // culayer
    culayer_pulsed = RPU::make_unique<RPUCudaPulsed<num_t>>(context, *layer_pulsed);
    culayer_pulsed->disp();

    auto seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator{seed};
    std::uniform_real_distribution<num_t> udist(-2.0, 2.0);
    auto urnd = std::bind(udist, generator);

    // just assign some numbers from the weigt matrix
    for (int i = 0; i < x_size; i++)
      rx[i] = urnd();

    for (int j = 0; j < d_size; j++) {
      rd[j] = urnd();
    }

    x_cuvec = RPU::make_unique<CudaArray<num_t>>(this->context, this->x_size);
    x_vec.resize(this->x_size);

    d_cuvec = RPU::make_unique<CudaArray<num_t>>(this->context, this->d_size);
    d_vec.resize(this->d_size);
    this->context->synchronizeDevice();
  };

  void runLayerTest(int kernelidx) {

    this->populateLayers(kernelidx);

    this->x_cuvec->assign(this->rx.data());
    this->x_vec = this->rx;

    this->d_cuvec->assign(this->rd.data());
    this->d_vec = this->rd;

    this->context->synchronizeDevice();

    std::cout << "RPU Cuda:\n";
    this->culayer_pulsed->printWeights(3, 3);
    this->culayer_pulsed->printRPUParameter(3, 3);
    std::cout << "RPU:\n";
    this->layer_pulsed->printWeights(3, 3);
    this->layer_pulsed->printRPUParameter(3, 3);

    // update
    int nK32 = (K + 32) / 32;
    uint32_t *x_counts32 = new uint32_t[this->x_size * nK32];
    uint32_t *d_counts32 = new uint32_t[this->d_size * nK32];

    for (int loop = 0; loop < this->repeats; loop++) {

      this->culayer_pulsed->update(this->x_cuvec->getData(), this->d_cuvec->getData(), false, 1);

      this->context->synchronizeDevice();

      this->culayer_pulsed->getCountsDebug(x_counts32, d_counts32);
      this->context->synchronizeDevice();
      this->layer_pulsed->updateVectorWithCounts(
          this->x_vec.data(), this->d_vec.data(), 1, 1, x_counts32, d_counts32);

      num_t **cuweights = this->culayer_pulsed->getWeights();
      num_t **weights = this->layer_pulsed->getWeights();
      this->context->synchronizeDevice();

      for (int i = 0; i < this->d_size; i++) {
        for (int j = 0; j < this->x_size; j++) {
          ASSERT_NEAR(weights[i][j], cuweights[i][j], 1e-5);
        }
      }
      this->context->synchronizeDevice();
    }

    std::cout << "W results for RPU Cuda:\n";
    this->culayer_pulsed->printWeights(3, 3);
    std::cout << "W results for RPU:\n";
    this->layer_pulsed->printWeights(3, 3);

    num_t **cuweights = this->culayer_pulsed->getWeights();
    num_t **weights = this->layer_pulsed->getWeights();
    this->context->synchronizeDevice();

    int diff_count_rpu = 0;
    int diff_count_rpucuda = 0;
    for (int i = 0; i < this->d_size; i++) {
      for (int j = 0; j < this->x_size; j++) {
        if (fabs(weights[i][j] - refweights[i][j]) > 1e-4) {
          diff_count_rpu++;
        }
        if (fabs(cuweights[i][j] - refweights[i][j]) > 1e-4) {
          diff_count_rpucuda++;
        }
      }
    }
    // make sure that at least some updated happend
    ASSERT_TRUE(diff_count_rpu > 0);
    ASSERT_TRUE(diff_count_rpucuda > 0);

    delete[] x_counts32;
    delete[] d_counts32;
  };

  void TearDown() { Array_2D_Free(refweights); }

  std::unique_ptr<CudaContext> context_container;
  CudaContext *context;

  std::unique_ptr<RPUPulsed<num_t>> layer_pulsed;
  std::unique_ptr<RPUCudaPulsed<num_t>> culayer_pulsed;
  std::vector<num_t> x_vec, d_vec, rx, rd;
  std::unique_ptr<CudaArray<num_t>> x_cuvec;
  std::unique_ptr<CudaArray<num_t>> d_cuvec;

  int x_size;
  int d_size;
  int repeats;
  int K;

  num_t **refweights;
};

// types
typedef ::testing::Types<
    LinearStepRPUDeviceMetaParameter<num_t>,
    ExpStepRPUDeviceMetaParameter<num_t>,
    ConstantStepRPUDeviceMetaParameter<num_t>>
    MetaPar;

TYPED_TEST_CASE(RPUDeviceTestFixture, MetaPar);

TYPED_TEST(RPUDeviceTestFixture, DeviceUpdate0) { this->runLayerTest(0); }
TYPED_TEST(RPUDeviceTestFixture, DeviceUpdate1) { this->runLayerTest(1); }
TYPED_TEST(RPUDeviceTestFixture, DeviceUpdate2) { this->runLayerTest(2); }
TYPED_TEST(RPUDeviceTestFixture, DeviceUpdate3) { this->runLayerTest(3); }
TYPED_TEST(RPUDeviceTestFixture, DeviceUpdate4) { this->runLayerTest(4); }
TYPED_TEST(RPUDeviceTestFixture, DeviceUpdate5) { this->runLayerTest(5); }
TYPED_TEST(RPUDeviceTestFixture, DeviceUpdate6) { this->runLayerTest(6); }
TYPED_TEST(RPUDeviceTestFixture, DeviceUpdate7) { this->runLayerTest(7); }

TYPED_TEST(RPUDeviceTestFixture, DeviceUpdate8) { this->runLayerTest(8); }

TYPED_TEST(RPUDeviceTestFixture, DeviceUpdate9) { this->runLayerTest(9); }

} // namespace

int main(int argc, char **argv) {
  resetCuda();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
