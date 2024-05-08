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

#include "cuda_util.h"
#include "rpucuda_constantstep_device.h"
#include "rpucuda_onesided_device.h"
#include "rpucuda_pulsed.h"
#include "test_helper.h"
#include "gtest/gtest.h"
#include <chrono>
#include <memory>
#include <random>

#define TOLERANCE 1e-5

namespace {

using namespace RPU;

class RPUDeviceCudaTestFixture : public ::testing::TestWithParam<int> {
public:
  void SetUp() {

    context = &context_container;

    x_size = 5;
    d_size = 8;

    w_ref = Array_2D_Get<num_t>(d_size, x_size);

    for (int i = 0; i < x_size * d_size; i++) {
      w_ref[0][i] = MIN(MAX((num_t)0.1 * rw_rng.sampleGauss(), (num_t)-1.0), (num_t)1.0);
    }

    weights = Array_2D_Get<num_t>(d_size, x_size);
    for (int i = 0; i < d_size * x_size; i++) {
      weights[0][i] = 0;
    }

    up.pulse_type = PulseType::DeterministicImplicit;
    up.update_bl_management = true;
    up.update_management = true;
    up.desired_BL = 100;
    up.x_res_implicit = 0;
    up.d_res_implicit = 0;

    up.initialize();

    lifetime = 100;
    dp_cs.dw_min = 0.1;
    dp_cs.dw_min_dtod = 0.0;
    dp_cs.dw_min_std = 0.0;
    dp_cs.up_down_dtod = 0.0;
    dp_cs.w_max = 1;
    dp_cs.w_min = -1;
    dp_cs.w_max_dtod = 0;
    dp_cs.w_min_dtod = 0;
    dp_cs.lifetime = lifetime;

    dp = new OneSidedRPUDeviceMetaParameter<num_t>(dp_cs);

    dp->refresh_io.inp_res = -1;
    dp->refresh_io.out_res = -1;
    dp->refresh_io.out_noise = 0.0;

    dp->refresh_up = up;
    dp->refresh_up.pulse_type = PulseType::None; // perfect refresh
    dp->refresh_every = GetParam();
    // dp->print();

    rng = new RNG<num_t>(0);

    up_pwu = RPU::make_unique<PulsedWeightUpdater<num_t>>(context, x_size, d_size);

    rpu_device = this->dp->createDeviceUnique(x_size, d_size, &rw_rng);
    rpucuda_device = AbstractRPUDeviceCuda<num_t>::createFromUnique(context, *rpu_device);

    dev_weights = RPU::make_unique<CudaArray<num_t>>(context, x_size * d_size);
    dev_weights->assignTranspose(weights[0], d_size, x_size);
    context->synchronize();

    size = x_size * d_size;
  };

  void TearDown() {
    Array_2D_Free<num_t>(weights);
    Array_2D_Free<num_t>(w_ref);
    delete dp;
    delete rng;
  };

  int x_size, d_size, colidx, size;
  num_t lifetime;
  num_t **weights;
  num_t **w_ref;
  PulsedUpdateMetaParameter<num_t> up;
  OneSidedRPUDeviceMetaParameter<num_t> *dp;
  ConstantStepRPUDeviceMetaParameter<num_t> dp_cs;
  std::unique_ptr<PulsedWeightUpdater<num_t>> up_pwu;
  std::unique_ptr<CudaArray<num_t>> dev_weights;

  RNG<num_t> *rng;
  RealWorldRNG<num_t> rw_rng;
  std::unique_ptr<AbstractRPUDevice<num_t>> rpu_device;
  std::unique_ptr<AbstractRPUDeviceCuda<num_t>> rpucuda_device;
  CudaContext context_container{-1, false};
  CudaContextPtr context;
};

// define the tests
INSTANTIATE_TEST_CASE_P(RefreshEvery, RPUDeviceCudaTestFixture, ::testing::Values(0, 1, 2));

TEST_P(RPUDeviceCudaTestFixture, createDevice) {

  ASSERT_TRUE(dynamic_cast<OneSidedRPUDeviceCuda<num_t> *>(&*rpucuda_device) != nullptr);
}

TEST_P(RPUDeviceCudaTestFixture, onSetWeights) {

  if (rpu_device->onSetWeights(w_ref)) {
    rpucuda_device->populateFrom(*rpu_device); // device pars have changed (due to onSetWeights)
  }
  context->synchronize();
  int g_plus, g_minus;
  static_cast<OneSidedRPUDevice<num_t> *>(&*rpu_device)->getGIndices(g_plus, g_minus);
  ASSERT_FLOAT_EQ(g_plus, 1.0);
  ASSERT_FLOAT_EQ(g_minus, 0.0);

  std::vector<num_t> w_vec =
      static_cast<OneSidedRPUDeviceCuda<num_t> *>(&*rpucuda_device)->getHiddenWeights();
  std::vector<num_t> reduce_weightening =
      static_cast<OneSidedRPUDeviceCuda<num_t> *>(&*rpucuda_device)->getReduceWeightening();
  int size = this->x_size * this->d_size;

  for (int i = 0; i < size; i++) {
    if (w_ref[0][i] < (num_t)0.0) {
      ASSERT_FLOAT_EQ(w_vec[i], (num_t)fabsf(w_ref[0][i]));
      ASSERT_FLOAT_EQ(w_vec[i + size], 0);
    } else {
      ASSERT_FLOAT_EQ(w_vec[i + size], (num_t)fabsf(w_ref[0][i]));
      ASSERT_FLOAT_EQ(w_vec[i], 0);
    }
  }
  ASSERT_FLOAT_EQ(reduce_weightening[g_minus], (num_t)-1.0);
  ASSERT_FLOAT_EQ(reduce_weightening[g_plus], (num_t)1.0);
}

TEST_P(RPUDeviceCudaTestFixture, UpdatePos) {

  for (int i = 0; i < size; i++) {
    this->weights[0][i] = 0.0;
  }

  // just newly create from paramerers
  rpu_device = dp->createDeviceUnique(this->x_size, this->d_size, &this->rw_rng);
  rpucuda_device = AbstractRPUDeviceCuda<num_t>::createFromUnique(context, *rpu_device);

  CudaArray<num_t> dev_x(context, this->x_size);
  dev_x.setConst(1.0);
  CudaArray<num_t> dev_d(context, this->d_size);
  dev_d.setConst(-1.0);
  context->synchronize();

  if (rpu_device->onSetWeights(this->weights)) {
    rpucuda_device->populateFrom(*rpu_device); // device pars have changed (due to onSetWeights)
  }
  context->synchronize();

  up_pwu->update(
      dev_x.getDataConst(), dev_d.getDataConst(), dev_weights->getData(), &*rpucuda_device,
      this->up,
      1.0,   // lr
      1,     // batch
      false, // trans
      false);
  // should update all weight values of the hidden weight by -1
  context->synchronize();
  auto w_vec = static_cast<OneSidedRPUDeviceCuda<num_t> *>(&*rpucuda_device)->getHiddenWeights();

  // update only on fast [nothing to transfer for first row]
  int size = this->d_size * this->x_size;
  // hidden weights updated (should be about 1)
  for (int i = 0; i < size; i++) {
    ASSERT_FLOAT_EQ(w_vec[i + size], (num_t)1.0);
  }

  // negative  weights  not
  for (int i = 0; i < size; i++) {
    ASSERT_FLOAT_EQ(w_vec[i], 0.0);
  }

  dev_weights->copyTo(weights[0]);
  dev_weights->assignTranspose(weights[0], x_size, d_size);
  dev_weights->copyTo(weights[0]);

  // reduce to weight. Only if gamma is set
  for (int i = 0; i < size; i++) {
    ASSERT_FLOAT_EQ(this->weights[0][i], 1.0);
  }
}

TEST_P(RPUDeviceCudaTestFixture, UpdateNeg) {

  // just newly create from paramerers
  rpu_device = dp->createDeviceUnique(this->x_size, this->d_size, &this->rw_rng);
  rpucuda_device = AbstractRPUDeviceCuda<num_t>::createFromUnique(context, *rpu_device);

  CudaArray<num_t> dev_x(context, this->x_size);
  dev_x.setConst(1.0);
  CudaArray<num_t> dev_d(context, this->d_size);
  dev_d.setConst(1.0);
  context->synchronize();

  for (int i = 0; i < size; i++) {
    this->weights[0][i] = 0.0;
  }

  if (rpu_device->onSetWeights(this->weights)) {
    rpucuda_device->populateFrom(*rpu_device); // device pars have changed (due to onSetWeights)
  }
  context->synchronize();

  up_pwu->update(
      dev_x.getDataConst(), dev_d.getDataConst(), dev_weights->getData(), &*rpucuda_device,
      this->up,
      1.0,   // lr
      1,     // batch
      false, // trans
      false);
  // should update all weight values of the hidden weight by -1
  context->synchronize();
  auto w_vec = static_cast<OneSidedRPUDeviceCuda<num_t> *>(&*rpucuda_device)->getHiddenWeights();

  // update only on fast [nothing to transfer for first row]
  int size = this->d_size * this->x_size;
  // hidden weights updated (should be about 1)
  for (int i = 0; i < size; i++) {
    ASSERT_FLOAT_EQ(w_vec[i], (num_t)1.0);
  }

  // negative  weights  not
  for (int i = 0; i < size; i++) {
    ASSERT_FLOAT_EQ(w_vec[i + size], 0.0);
  }

  dev_weights->copyTo(weights[0]);
  dev_weights->assignTranspose(weights[0], x_size, d_size);
  dev_weights->copyTo(weights[0]);

  // reduce to weight.
  for (int i = 0; i < size; i++) {
    ASSERT_FLOAT_EQ(this->weights[0][i], -1.0);
  }
}

TEST_P(RPUDeviceCudaTestFixture, UpdateAndRefresh) {

  CudaArray<num_t> dev_x(context, this->x_size);
  CudaArray<num_t> dev_d(context, this->d_size);
  context->synchronize();

  for (int i = 0; i < size; i++) {
    this->weights[0][i] = 0.0;
  }

  if (rpu_device->onSetWeights(this->weights)) {
    rpucuda_device->populateFrom(*rpu_device); // device pars have changed (due to onSetWeights)
  }

  // rpu_device->dispInfo();
  context->synchronize();

  // pos update
  dev_x.setConst(1.0);
  dev_d.setConst(-1.0);
  up_pwu->update(
      dev_x.getDataConst(), dev_d.getDataConst(), dev_weights->getData(), &*rpucuda_device,
      this->up,
      1.0,   // lr
      1,     // batch
      false, // trans
      false);

  // neg update
  dev_d.setConst(0.9);
  up_pwu->update(
      dev_x.getDataConst(), dev_d.getDataConst(), dev_weights->getData(), &*rpucuda_device,
      this->up,
      1.0,   // lr
      1,     // batch
      false, // trans
      false);

  // all weights should be saturated
  context->synchronize();
  auto w_vec = static_cast<OneSidedRPUDeviceCuda<num_t> *>(&*rpucuda_device)->getHiddenWeights();
  int size = this->d_size * this->x_size;

  if (GetParam() == 0) {             // no refresh
    for (int i = 0; i < size; i++) { // neg
      ASSERT_NEAR(w_vec[i], 0.9, TOLERANCE);
    }
    for (int i = 0; i < size; i++) {
      ASSERT_NEAR(w_vec[i + size], 1.0, TOLERANCE); // pos
    }
  } else { // refresh 1 or 2
    for (int i = 0; i < size; i++) {
      ASSERT_NEAR(w_vec[i], 0.0, TOLERANCE); // neg
    }
    for (int i = 0; i < size; i++) {
      ASSERT_NEAR(w_vec[i + size], 0.1, TOLERANCE); // pos
    }
  }

} // namespace

TEST_P(RPUDeviceCudaTestFixture, LayerTest) {
  PulsedMetaParameter<num_t> p;

  p.up.pulse_type = PulseType::DeterministicImplicit;
  p.up.x_res_implicit = 0.01;
  p.up.d_res_implicit = 0.01;
  p.up.desired_BL = 100;

  auto layer_pulsed = RPUPulsed<num_t>(x_size, d_size);
  layer_pulsed.populateParameter(&p, dp);
  layer_pulsed.setLearningRate(0.1);
  layer_pulsed.setWeights(w_ref[0]);

  auto culayer_pulsed = RPUCudaPulsed<num_t>(context, layer_pulsed);
  // culayer_pulsed.disp();

  int m_batch = 1; // note that CUDA will refresh less often then CPU
                   // in case the refresh is smaller than m_batch
  std::vector<num_t> x(this->x_size * m_batch);
  std::vector<num_t> xm(this->x_size * m_batch);
  std::vector<num_t> d(this->d_size * m_batch);

  for (int i = 0; i < this->x_size * m_batch; i++) {
    x[i] = rw_rng.sampleGauss();
    xm[i] = -x[i];
  }
  for (int i = 0; i < this->d_size * m_batch; i++) {
    d[i] = rw_rng.sampleGauss();
  }

  CudaArray<num_t> dev_x(context, this->x_size * m_batch, x.data());
  CudaArray<num_t> dev_xm(context, this->x_size * m_batch, xm.data());
  CudaArray<num_t> dev_d(context, this->d_size * m_batch, d.data());
  context->synchronize();

  int ntimes = 10;

  for (int i = 0; i < ntimes; i++) {
    layer_pulsed.update(x.data(), d.data(), false, m_batch);
    culayer_pulsed.update(dev_x.getData(), dev_d.getData(), false, m_batch);

    layer_pulsed.update(xm.data(), d.data(), false, m_batch);
    culayer_pulsed.update(dev_xm.getData(), dev_d.getData(), false, m_batch);
  }
  this->context->synchronizeDevice();
  num_t **cuweights = culayer_pulsed.getWeights();
  num_t **weights = layer_pulsed.getWeights();

  int count =
      static_cast<const OneSidedRPUDevice<num_t> &>(layer_pulsed.getRPUDevice()).getRefreshCount();
  int cucount = static_cast<const OneSidedRPUDeviceCuda<num_t> &>(culayer_pulsed.getRPUDeviceCuda())
                    .getRefreshCount();
  ASSERT_EQ(count, cucount);
  if (GetParam()) {
    ASSERT_TRUE(count > 0);
  }

  int n_diff = 0;
  for (int i = 0; i < d_size; i++) {
    for (int j = 0; j < x_size; j++) {
      ASSERT_NEAR(weights[i][j], cuweights[i][j], TOLERANCE);
      n_diff += weights[i][j] != w_ref[i][j];
    }
  }
  ASSERT_TRUE(n_diff > 0);
}
} // namespace

int main(int argc, char **argv) {
  resetCuda();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
