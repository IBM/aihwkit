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
#include "rpu_pulsed.h"
#include "rpucuda_constantstep_device.h"
#include "rpucuda_expstep_device.h"
#include "rpucuda_linearstep_device.h"
#include "rpucuda_onesided_device.h"
#include "rpucuda_pulsed.h"
#include "rpucuda_pulsed_device.h"
#include "rpucuda_transfer_device.h"
#include "rpucuda_vector_device.h"
#include "utility_functions.h"
#include "gtest/gtest.h"
#include <chrono>
#include <memory>
#include <random>

#define TOLERANCE 1e-5

namespace {

using namespace RPU;

template <typename VectorDeviceParT> void setAdditionalValues(VectorDeviceParT *vp, num_t value);

template <> void setAdditionalValues(VectorRPUDeviceMetaParameter<num_t> *vp, num_t value) {}

template <> void setAdditionalValues(OneSidedRPUDeviceMetaParameter<num_t> *vp, num_t value) {}

template <> void setAdditionalValues(TransferRPUDeviceMetaParameter<num_t> *vp, num_t value) {
  vp->transfer_up.pulse_type = PulseType::None;
  vp->transfer_io.out_noise = 0.0;
  vp->transfer_io.inp_res = -1;
  vp->transfer_io.out_res = -1;
  vp->transfer_every = 1;
  vp->transfer_lr = 0.5;
  vp->n_reads_per_transfer = 1;
  vp->gamma = value;
}

template <typename VectorDeviceParT> class RPUDeviceTestFixture : public ::testing::Test {
public:
  void SetUp() {

    context = &context_container;

    this->x_size = 4;
    this->d_size = 5;
    this->K = 10;
    this->m_batch = 2;

    this->rx.resize(x_size * m_batch);
    this->rd.resize(d_size * m_batch);

    this->refweights = Array_2D_Get<num_t>(d_size, x_size);
  }

  template <typename PulsedDeviceT> void populateLayers(num_t value, PulseType pt) {
    repeats = 10;

    num_t bmin = -0.7;
    num_t bmax = 0.7;

    PulsedMetaParameter<num_t> p;
    PulsedDeviceT dp;
    IOMetaParameter<num_t> p_io;

    p_io.out_noise = 0.0; // no noise in output;
    dp.dw_min_std = 0.0;
    // dp.dw_min_dtod = 0.0;
    // dp.asym_dtod = 0.0;

    p.up.desired_BL = K;
    p.up.pulse_type = pt;

    dp.w_max = 3;
    dp.w_min = -3;
    dp.w_min_dtod = 0.1;
    dp.w_max_dtod = 0.1;

    dp.dw_min = 0.05;

    // peripheral circuits specs
    p_io.inp_res = -1;
    p_io.inp_sto_round = false;
    p_io.out_res = -1;

    p_io.noise_management = NoiseManagementType::None;
    p_io.bound_management = BoundManagementType::None;

    p.f_io = p_io;
    p.b_io = p_io;

    p.up.update_management = false;
    p.up.update_bl_management = false;

    int kernelidx = 0;

    if (kernelidx >= 0) {
      p.up._debug_kernel_index = kernelidx;
    }

    // dp.print();
    VectorDeviceParT vp(dp, 2);

    setAdditionalValues(&vp, value);

    // vp.print();
    num_t lr = 1;

    layer_pulsed = RPU::make_unique<RPUPulsed<num_t>>(x_size, d_size);

    layer_pulsed->populateParameter(&p, &vp);
    layer_pulsed->setLearningRate(lr);
    layer_pulsed->setWeightsUniformRandom(bmin, bmax);
    // layer_pulsed->disp();

    this->layer_pulsed->getWeights(refweights[0]);

    // culayer
    culayer_pulsed = RPU::make_unique<RPUCudaPulsed<num_t>>(context, *layer_pulsed);
    // culayer_pulsed->disp();

    unsigned int seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator{seed};
    std::uniform_real_distribution<float> udist(-1.2, 1.2);
    auto urnd = std::bind(udist, generator);

    // just assign some numbers from the weigt matrix
    for (int i = 0; i < x_size * m_batch; i++)
      rx[i] = (num_t)urnd();

    for (int j = 0; j < d_size * m_batch; j++) {
      rd[j] = (num_t)urnd();
    }

    x_cuvec = RPU::make_unique<CudaArray<num_t>>(this->context, this->x_size * m_batch);
    x_vec.resize(this->x_size * m_batch);

    d_cuvec = RPU::make_unique<CudaArray<num_t>>(this->context, this->d_size * m_batch);

    d_vec.resize(this->d_size * m_batch);
    this->context->synchronizeDevice();
  };

  template <typename PulsedDeviceT> void runLayerTest(num_t value, PulseType pt) {

    this->template populateLayers<PulsedDeviceT>(value, pt);

    this->x_cuvec->assign(this->rx.data());
    this->x_vec = this->rx;

    this->d_cuvec->assign(this->rd.data());
    this->d_vec = this->rd;

    this->context->synchronizeDevice();

    // update
    int nK32 = (K + 32) / 32;
    uint32_t *x_counts32 = new uint32_t[this->x_size * nK32 * this->m_batch];
    uint32_t *d_counts32 = new uint32_t[this->d_size * nK32 * this->m_batch];

    for (int loop = 0; loop < this->repeats; loop++) {

      this->culayer_pulsed->update(
          this->x_cuvec->getData(), this->d_cuvec->getData(), false, this->m_batch);
      this->context->synchronizeDevice();

      if (pt != PulseType::NoneWithDevice) {

        this->culayer_pulsed->getCountsDebug(x_counts32, d_counts32);
        this->context->synchronizeDevice();
        for (int m = 0; m < this->m_batch; m++) {
          this->layer_pulsed->updateVectorWithCounts(
              this->x_vec.data() + m * this->x_size, this->d_vec.data() + m * this->d_size, 1, 1,
              x_counts32 + m * this->x_size * nK32, d_counts32 + m * this->d_size * nK32);
        }
      } else {
        this->layer_pulsed->update(this->x_vec.data(), this->d_vec.data(), false, this->m_batch);
      }

      num_t **cuweights = this->culayer_pulsed->getWeights();
      this->context->synchronizeDevice();
      num_t **weights = this->layer_pulsed->getWeights();

      for (int i = 0; i < this->d_size; i++) {
        for (int j = 0; j < this->x_size; j++) {
          EXPECT_NEAR(weights[i][j], cuweights[i][j], TOLERANCE);
        }
      }

      num_t ***w_vec =
          static_cast<const VectorRPUDevice<num_t> &>(this->layer_pulsed->getRPUDevice())
              .getWeightVec();
      std::vector<num_t> w_vec_cuda =
          static_cast<const VectorRPUDeviceCuda<num_t> &>(this->culayer_pulsed->getRPUDeviceCuda())
              .getHiddenWeights();

      for (size_t i = 0; i < w_vec_cuda.size(); i++) {
        EXPECT_NEAR(w_vec[0][0][i], w_vec_cuda[i], TOLERANCE);
      }

      this->context->synchronizeDevice();
    }

    num_t **cuweights = this->culayer_pulsed->getWeights();
    num_t **weights = this->layer_pulsed->getWeights();
    this->context->synchronizeDevice();

    int diff_count_rpu = 0;
    int diff_count_rpucuda = 0;
    for (int i = 0; i < this->d_size; i++) {
      for (int j = 0; j < this->x_size; j++) {
        if (fabsf(weights[i][j] - refweights[i][j]) > 1e-4) {
          diff_count_rpu++;
        }
        if (fabsf(cuweights[i][j] - refweights[i][j]) > 1e-4) {
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

  CudaContext context_container{-1, false};
  CudaContextPtr context;
  std::unique_ptr<RPUPulsed<num_t>> layer_pulsed;
  std::unique_ptr<RPUCudaPulsed<num_t>> culayer_pulsed;
  std::vector<num_t> x_vec, d_vec, rx, rd;
  std::unique_ptr<CudaArray<num_t>> x_cuvec;
  std::unique_ptr<CudaArray<num_t>> d_cuvec;
  int x_size;
  int d_size;
  int m_batch;
  int repeats;
  int K;
  num_t **refweights;
};

// types
typedef ::testing::Types<
    VectorRPUDeviceMetaParameter<num_t>,
    OneSidedRPUDeviceMetaParameter<num_t>,
    TransferRPUDeviceMetaParameter<num_t>>
    MetaPar;

TYPED_TEST_CASE(RPUDeviceTestFixture, MetaPar);

TYPED_TEST(RPUDeviceTestFixture, ConstantStepDeviceUpdateGamma) {
  // -1 might tune into 64direct. This will
  // result in errors, because compared on
  // normal format basis.
  this->template runLayerTest<ConstantStepRPUDeviceMetaParameter<num_t>>(
      0.25, PulseType::StochasticCompressed);
}

TYPED_TEST(RPUDeviceTestFixture, ConstantStepDeviceUpdateFullyHidden) {
  // -1 might tune into 64direct. This will
  // result in errors, because compared on
  // normal format basis.
  this->template runLayerTest<ConstantStepRPUDeviceMetaParameter<num_t>>(
      0, PulseType::StochasticCompressed);
}

TYPED_TEST(RPUDeviceTestFixture, LinearStepDeviceUpdate) {
  // -1 might tune into 64direct. This will
  // result in errors, because compared on
  // normal format basis.
  this->template runLayerTest<LinearStepRPUDeviceMetaParameter<num_t>>(
      0.2, PulseType::StochasticCompressed);
}

} // namespace

int main(int argc, char **argv) {
  resetCuda();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
