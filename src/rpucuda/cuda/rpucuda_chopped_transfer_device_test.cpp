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
#include "rpu_pulsed.h"
#include "rpucuda_chopped_transfer_device.h"
#include "rpucuda_constantstep_device.h"
#include "rpucuda_pulsed.h"
#include "test_helper.h"
#include "gtest/gtest.h"
#include <chrono>
#include <memory>
#include <random>

#define TOLERANCE 1e-5

namespace {

using namespace RPU;

class RPUDeviceCudaTestFixture : public ::testing::TestWithParam<bool> {
public:
  void SetUp() {
    x_size = 3;
    d_size = 3;
    m_batch = 1;
    context = &context_container;

    w_ref = Array_2D_Get<num_t>(d_size, x_size);

    for (int i = 0; i < x_size * d_size; i++) {
      w_ref[0][i] = rw_rng.sampleGauss();
    }

    weights = Array_2D_Get<num_t>(d_size, x_size);
    for (int i = 0; i < d_size * x_size; i++) {
      weights[0][i] = 0;
    }

    up.pulse_type =
        PulseType::StochasticCompressed; // nodevice would skip the entire transfer mechanism
    up.update_bl_management = false;
    up.update_management = false;
    up.desired_BL = 10;
    up.initialize();

    dp_cs.dw_min = 0.1;
    dp_cs.dw_min_dtod = 0.0;
    dp_cs.dw_min_std = 0.0;
    dp_cs.up_down_dtod = 0.0;
    dp_cs.w_max = 100;
    dp_cs.w_min = -100;
    dp_cs.w_max_dtod = 0;
    dp_cs.w_min_dtod = 0;
    dp_cs.lifetime = 0.0;

    dp = new ChoppedTransferRPUDeviceMetaParameter<num_t>(dp_cs, 2);

    dp->gamma = 0.0;
    dp->thres_scale = (num_t)0.5 / dp_cs.dw_min;
    dp->transfer_columns = GetParam();
    dp->transfer_every = GetParam() ? x_size : d_size;
    dp->n_reads_per_transfer = 1;
    dp->units_in_mbatch = false;
    dp->forget_buffer = true;
    dp->transfer_io.inp_res = -1.0;
    dp->transfer_io.out_res = -1.0;
    dp->transfer_io.out_noise = 0.0;

    dp->transfer_up = up;
    dp->transfer_up.pulse_type = PulseType::StochasticCompressed;
    dp->transfer_up.update_bl_management = false;
    dp->transfer_up.update_management = false;
    dp->transfer_up.fixed_BL = true;
    dp->transfer_up.desired_BL = 1; // to force exactly one pulse if x*d!=0
    dp->step = 1.0;                 // step to accumulate on next device. Since
                                    // BL=1, LR=1, UM=0, and UBLM=0 it should always
                                    // be (+/-)dw_min
    dp->transfer_lr = 1.0;
    dp->scale_transfer_lr = false; // do not scale with current_lr
    dp->random_selection = false;

    rx.resize(x_size * m_batch);
    rd.resize(d_size * m_batch);

    unsigned int seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator{seed};
    std::uniform_real_distribution<float> udist(-1.2, 1.2);
    auto urnd = std::bind(udist, generator);

    // just assign some numbers from the weight matrix
    for (int i = 0; i < x_size * m_batch; i++)
      rx[i] = (num_t)urnd();

    for (int j = 0; j < d_size * m_batch; j++) {
      rd[j] = (num_t)urnd();
    }

    up_pwu = RPU::make_unique<PulsedWeightUpdater<num_t>>(context, x_size, d_size);

    rpu_device = this->dp->createDeviceUnique(x_size, d_size, &rw_rng);
    rpucuda_device = AbstractRPUDeviceCuda<num_t>::createFromUnique(context, *rpu_device);

    dev_weights = RPU::make_unique<CudaArray<num_t>>(context, x_size * d_size);
    dev_weights->assignTranspose(weights[0], d_size, x_size);
    context->synchronize();
  };

  void TearDown() {
    Array_2D_Free<num_t>(weights);
    Array_2D_Free<num_t>(w_ref);
    delete dp;
  };

  int x_size, d_size, colidx, m_batch;
  num_t lifetime;
  num_t **weights;
  num_t **w_ref;
  std::vector<num_t> rx, rd, w, w2;
  PulsedUpdateMetaParameter<num_t> up;
  ChoppedTransferRPUDeviceMetaParameter<num_t> *dp;
  ConstantStepRPUDeviceMetaParameter<num_t> dp_cs;
  std::unique_ptr<PulsedWeightUpdater<num_t>> up_pwu;
  std::unique_ptr<CudaArray<num_t>> dev_weights;

  CudaContext context_container{-1, false};
  CudaContextPtr context;
  RealWorldRNG<num_t> rw_rng;
  std::unique_ptr<AbstractRPUDevice<num_t>> rpu_device;
  std::unique_ptr<AbstractRPUDeviceCuda<num_t>> rpucuda_device;
};

// define the tests
INSTANTIATE_TEST_CASE_P(RowColumn, RPUDeviceCudaTestFixture, ::testing::Values(true, false));

TEST_P(RPUDeviceCudaTestFixture, createDevice) {

  ASSERT_TRUE(dynamic_cast<ChoppedTransferRPUDeviceCuda<num_t> *>(&*rpucuda_device) != nullptr);
}

TEST_P(RPUDeviceCudaTestFixture, Update) {

  dp->transfer_lr = 0; // no transfer here
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
  // should update all weight values of the hidden weight by -0.1 (current dw_min)
  context->synchronize();
  auto w_vec =
      static_cast<ChoppedTransferRPUDeviceCuda<num_t> *>(&*rpucuda_device)->getHiddenWeights();

  // update only on fast [nothing to transfer for first row]
  int size = this->d_size * this->x_size;
  // hidden weights updated (should be about 1)
  num_t s = 0;
  for (int i = 0; i < size; i++) {
    ASSERT_FLOAT_EQ(w_vec[i], (num_t)1.0); // although stochastic, since A,B and BL/dwmin is 1 it
                                           // should actually be exactly one
    s += w_vec[i];
  }
  // std::cout << "Average weight " << s / size << " (Expected is 0.1)" << std::endl;

  // visible  weights  not
  for (int i = 0; i < size; i++) {
    ASSERT_FLOAT_EQ(w_vec[i + size], 0.0);
  }

  dev_weights->copyTo(weights[0]);
  dev_weights->assignTranspose(weights[0], x_size, d_size);
  dev_weights->copyTo(weights[0]);

  // reduce to weight. Only if gamma is set
  for (int i = 0; i < size; i++) {
    ASSERT_FLOAT_EQ(this->weights[0][i], 0.0);
  }
}

TEST_P(RPUDeviceCudaTestFixture, UpdateAndTransfer) {

  CudaArray<num_t> dev_x(context, this->x_size);
  dev_x.setConst(1.0);
  CudaArray<num_t> dev_d(context, this->d_size);
  dev_d.setConst(-1.0);
  context->synchronize();

  if (rpu_device->onSetWeights(this->weights)) {
    rpucuda_device->populateFrom(*rpu_device); // device pars have changed (due to onSetWeights)
  }
  context->synchronize();
  int max_size = (GetParam() ? this->x_size : this->d_size);

  for (int k = 0; k < max_size; k++) {
    up_pwu->update(
        dev_x.getDataConst(), dev_d.getDataConst(), dev_weights->getData(), &*rpucuda_device,
        this->up,
        1.0,   // lr
        1,     // batch
        false, // trans
        false);
  }
  // weight values of the hidden weights should be x_size and first
  // col should be transfered once (that is set to dw_min)
  context->synchronize();
  auto w_vec =
      static_cast<ChoppedTransferRPUDeviceCuda<num_t> *>(&*rpucuda_device)->getHiddenWeights();
  dev_weights->copyTo(weights[0]);
  dev_weights->assignTranspose(weights[0], x_size, d_size);
  dev_weights->copyTo(weights[0]);

  int size = this->d_size * this->x_size;
  // for (int k = 0; k < 3; k++) {
  //   switch (k) {
  //   case 0: std::cout << " A " << k << ":" << std::endl; break;
  //   case 1: std::cout << " C " << k << ":" << std::endl; break;
  //   case 2: std::cout << " Buffer " << k << ":" << std::endl; break;
  //   }
  //   for (int i_x = 0; i_x < x_size; i_x++) {
  //     for (int i_d = 0; i_d < d_size; i_d++) {
  // 	int i = i_x + x_size * i_d + k*size;

  // 	if (k == 1) {// fully hidden, thus take weight here
  // 	  std::cout << "\t" << weights[0][i];
  // 	} else {
  // 	  std::cout << "\t" << w_vec[i];
  // 	}
  //     }
  //     std::cout << std::endl;
  //   }
  //   std::cout << std::endl;
  // }

  // update only on fast [nothing to transfer for first row]
  // hidden weights updated
  num_t s = 0;
  for (int i = 0; i < size; i++) {
    ASSERT_FLOAT_EQ(w_vec[i], max_size);
    s += w_vec[i];
  }
  // only first col of  weights should be transferred

  for (int i = 0; i < size; i++) {
    ASSERT_FLOAT_EQ(w_vec[i + size], 0.0); // should not be used
  }

  // reduce to weight.
  std::vector<num_t> rw =
      static_cast<ChoppedTransferRPUDeviceCuda<num_t> *>(&*rpucuda_device)->getReduceWeightening();

  // always fully hidden this not A in C
  for (int i = 0; i < size; i++) {
    // std::cout << "[" << i / x_size << "," << i % x_size << "]: " << this->weights[0][i] <<
    // std::endl;
    if (GetParam()) {
      ASSERT_FLOAT_EQ(this->weights[0][i], i % x_size ? (num_t)0.0 : dp_cs.dw_min * rw[1]);
    } else {
      ASSERT_FLOAT_EQ(this->weights[0][i], i >= x_size ? (num_t)0.0 : dp_cs.dw_min * rw[1]);
    }
  }
}

TEST_P(RPUDeviceCudaTestFixture, UpdateAndTransferBatch) {

  int max_size = GetParam() ? this->x_size : this->d_size;

  CudaArray<num_t> dev_x(context, this->x_size * max_size);
  dev_x.setConst(1.0);
  CudaArray<num_t> dev_d(context, this->d_size * max_size);
  dev_d.setConst(-1.0);
  context->synchronize();

  if (rpu_device->onSetWeights(this->weights)) {
    rpucuda_device->populateFrom(*rpu_device); // device pars have changed (due to onSetWeights)
  }
  context->synchronize();

  up_pwu->update(
      dev_x.getDataConst(), dev_d.getDataConst(), dev_weights->getData(), &*rpucuda_device,
      this->up,
      1.0,      // lr
      max_size, // batch
      false,    // trans
      false);
  // weight values of the hidden weights should be x_size and first
  // col should be transfered once (that is set to dw_min)
  context->synchronize();
  auto w_vec =
      static_cast<ChoppedTransferRPUDeviceCuda<num_t> *>(&*rpucuda_device)->getHiddenWeights();
  dev_weights->copyTo(weights[0]);
  dev_weights->assignTranspose(weights[0], x_size, d_size);
  dev_weights->copyTo(weights[0]);

  // update only on fast [nothing to transfer for first row]
  int size = this->d_size * this->x_size;
  // hidden weights updated
  num_t s = 0;
  for (int i = 0; i < size; i++) {
    ASSERT_FLOAT_EQ(w_vec[i], (num_t)max_size);
    s += w_vec[i];
  }
  // only first col of  weights should be transferred

  for (int i = 0; i < size; i++) {
    ASSERT_FLOAT_EQ(w_vec[i + size], 0.0); // should not be used
  }

  // reduce to weight.
  std::vector<num_t> rw =
      static_cast<TransferRPUDeviceCuda<num_t> *>(&*rpucuda_device)->getReduceWeightening();

  // always fully hidden this not A in C
  for (int i = 0; i < size; i++) {
    // std::cout << "[" << i / x_size << "," << i % x_size << "]: " << this->weights[0][i] <<
    // std::endl;
    if (GetParam()) {
      ASSERT_FLOAT_EQ(this->weights[0][i], i % x_size ? (num_t)0.0 : dp_cs.dw_min * rw[1]);
    } else {
      ASSERT_FLOAT_EQ(this->weights[0][i], i >= x_size ? (num_t)0.0 : dp_cs.dw_min * rw[1]);
    }
  }
}

TEST_P(RPUDeviceCudaTestFixture, CUDAvsCPU) {

  PulsedMetaParameter<num_t> p;
  p.up = up;
  p.up.x_res_implicit = 0.01;
  p.up.d_res_implicit = 0.01;
  p.up.pulse_type = PulseType::DeterministicImplicit;
  p.up.desired_BL = 10;

  dp->in_chop_prob = 1;
  dp->out_chop_prob = 1;

  dp->transfer_up.pulse_type = PulseType::DeterministicImplicit;
  dp->transfer_up.x_res_implicit = 0.01;
  dp->transfer_up.d_res_implicit = 0.01;
  dp->transfer_up.desired_BL = 1;

  dp->transfer_io.is_perfect = true;

  CudaArray<num_t> dev_x(context, x_size * m_batch, rx.data());
  CudaArray<num_t> dev_d(context, d_size * m_batch, rd.data());
  context->synchronize();

  auto *rpu = new RPUPulsed<num_t>(x_size, d_size);
  rpu->populateParameter(&p, dp);
  rpu->setWeights(this->weights[0]);

  rpu->setLearningRate(1.0);
  auto *rpucuda = new RPUCudaPulsed<num_t>(context->getStream(), *rpu);
  rpucuda->setLearningRate(1.0);
  context->synchronize();

  int size = this->d_size * this->x_size;
  // double check whether weights are correct
  w.resize(x_size * d_size);
  rpu->getWeights(w.data());
  w2.resize(x_size * d_size);
  rpucuda->getWeights(w2.data());

  for (int i = 0; i < size; i++) {
    ASSERT_NEAR(w[i], w2[i], 1.0e-5);
  }

  context->synchronize();
  for (int k = 0; k < this->d_size * this->x_size; k++) {
    rpu->update(rx.data(), rd.data(), false, m_batch, false, false);
    rpucuda->update(dev_x.getData(), dev_d.getData(), false, m_batch, false, false);
  }

  context->synchronize();
  w.resize(x_size * d_size);
  rpu->getWeights(w.data());
  w2.resize(x_size * d_size);
  rpucuda->getWeights(w2.data());

  std::cout << "CUDA vs. CPU:" << std::endl;
  for (int i = 0; i < size; i++) {
    std::cout << "[" << i / x_size << "," << i % x_size << "]: " << w2[i] << " \tvs. \t" << w[i]
              << std::endl;
    ASSERT_NEAR(w[i], w2[i], 1.0e-5);
  }
}

} // namespace

int main(int argc, char **argv) {
  resetCuda();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
