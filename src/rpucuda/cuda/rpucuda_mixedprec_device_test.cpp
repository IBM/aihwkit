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
#include "rpucuda_mixedprec_device.h"
#include "test_helper.h"
#include "gtest/gtest.h"
#include <chrono>
#include <memory>
#include <random>

#define TOLERANCE 1e-5

namespace {

template <typename T> void transpose(T *w_trans, T *w, int x_size, int d_size) {

  for (int i = 0; i < x_size; i++) {
    for (int j = 0; j < d_size; j++) {
      w_trans[i + j * x_size] = w[j + i * d_size];
    }
  }
}

using namespace RPU;
class MixedPrecRPUDeviceCudaTestFixtureSmall : public ::testing::Test {
public:
  void SetUp() {
    context = &context_container;
    x_size = 2;
    d_size = 3;

    ConstantStepRPUDeviceMetaParameter<num_t> dp_cs;
    dp.setDevicePar(dp_cs);
  };

  void TearDown(){};

  int x_size, d_size;
  MixedPrecRPUDeviceMetaParameter<num_t> dp;
  CudaContext context_container{-1, false};
  CudaContextPtr context;
  std::unique_ptr<AbstractRPUDevice<num_t>> rpu_device;
  std::unique_ptr<AbstractRPUDeviceCuda<num_t>> rpucuda_device;
};

class MixedPrecRPUDeviceCudaTestFixture : public ::testing::TestWithParam<int> {
public:
  void SetUp() {
    x_size = 2;
    d_size = 3;
    context = &context_container;

    w_ref = Array_2D_Get<num_t>(d_size, x_size);

    for (int i = 0; i < x_size * d_size; i++) {
      w_ref[0][i] = rw_rng.sampleGauss();
    }

    weights = Array_2D_Get<num_t>(d_size, x_size);
    for (int i = 0; i < d_size * x_size; i++) {
      weights[0][i] = 0;
    }
    x_vec = new num_t[x_size]();
    d_vec = new num_t[d_size]();
    x_vec2 = new num_t[x_size]();
    d_vec2 = new num_t[d_size]();

    for (int i = 0; i < d_size; i++) {
      d_vec[i] = rw_rng.sampleGauss();
      d_vec2[i] = rw_rng.sampleGauss();
    }
    for (int i = 0; i < x_size; i++) {
      x_vec[i] = rw_rng.sampleGauss();
      x_vec2[i] = rw_rng.sampleGauss();
    }

    up.pulse_type = PulseType::NoneWithDevice;
    up.update_bl_management = true;
    up.update_management = true;
    up.desired_BL = 31;
    up.initialize();

    lifetime = 0;
    dp_cs.dw_min = 0.1;
    dp_cs.dw_min_dtod = 0.0;
    dp_cs.dw_min_std = 0.0;
    dp_cs.up_down_dtod = 0.0;
    dp_cs.w_max = 100;
    dp_cs.w_min = -100;
    dp_cs.w_max_dtod = 0;
    dp_cs.w_min_dtod = 0;
    dp_cs.lifetime = lifetime;

    dp.setDevicePar(dp_cs);
    dp.transfer_every = x_size;
    dp.n_rows_per_transfer = 1;

    up_pwu = RPU::make_unique<PulsedWeightUpdater<num_t>>(context, x_size, d_size);

    dev_weights = RPU::make_unique<CudaArray<num_t>>(context, x_size * d_size);
    context->synchronize();
  };

  void printChiCuda() {
    context->synchronize();
    auto chi_cuda =
        static_cast<MixedPrecRPUDeviceCuda<num_t> *>(&*rpucuda_device)->getHiddenWeights();

    std::cout << "\t\tChi CUDA:\t";
    for (int i = 0; i < x_size * d_size; i++) {
      std::cout << chi_cuda[i] << ", ";
    }
    std::cout << std::endl;
  }
  void printChi() {
    std::vector<num_t> chi(x_size * d_size);
    static_cast<MixedPrecRPUDevice<num_t> *>(&*rpu_device)->getChi(chi.data());
    std::cout << "\t\tChi:\t\t";
    for (int i = 0; i < x_size * d_size; i++) {
      std::cout << chi[i] << ", ";
    }
    std::cout << std::endl;
  }

  void getWeightsAll() {
    // also sets host weights
    context->synchronizeDevice();
    std::vector<num_t> weights2(x_size * d_size);
    dev_weights->copyTo(weights2.data());
    context->synchronizeDevice();

    transpose(weights[0], weights2.data(), x_size, d_size);
  }

  void printWeightsCuda() {
    std::cout << "\t\tweights CUDA:\t";
    for (int i = 0; i < x_size * d_size; i++) {
      std::cout << weights[0][i] << ", ";
    }
    std::cout << std::endl;
  }

  void printWeights() {
    std::cout << "\t\tweights:\t";
    for (int i = 0; i < x_size * d_size; i++) {
      std::cout << w_ref[0][i] << ", ";
    }
    std::cout << std::endl;
  }

  void TearDown() {
    Array_2D_Free<num_t>(weights);
    Array_2D_Free<num_t>(w_ref);
    delete[] x_vec;
    delete[] d_vec;
    delete[] x_vec2;
    delete[] d_vec2;
  };

  CudaContext context_container{-1, false};
  CudaContextPtr context;
  int x_size, d_size, colidx;
  num_t lifetime;
  num_t **weights;
  num_t **w_ref;
  num_t *x_vec, *d_vec, *x_vec2, *d_vec2;
  PulsedUpdateMetaParameter<num_t> up;
  MixedPrecRPUDeviceMetaParameter<num_t> dp;
  ConstantStepRPUDeviceMetaParameter<num_t> dp_cs;
  std::unique_ptr<PulsedWeightUpdater<num_t>> up_pwu;
  std::unique_ptr<CudaArray<num_t>> dev_weights;
  RealWorldRNG<num_t> rw_rng;
  std::unique_ptr<AbstractRPUDevice<num_t>> rpu_device;
  std::unique_ptr<AbstractRPUDeviceCuda<num_t>> rpucuda_device;
};

// define the tests
INSTANTIATE_TEST_CASE_P(Binsize, MixedPrecRPUDeviceCudaTestFixture, ::testing::Values(0, 3, 10));

TEST_F(MixedPrecRPUDeviceCudaTestFixtureSmall, createDevice) {
  RealWorldRNG<num_t> rw_rng;
  rpu_device = dp.createDeviceUnique(x_size, d_size, &rw_rng);
  rpucuda_device = AbstractRPUDeviceCuda<num_t>::createFromUnique(context, *rpu_device);
  context->synchronize();

  ASSERT_TRUE(dynamic_cast<MixedPrecRPUDevice<num_t> *>(&*rpu_device) != nullptr);
  ASSERT_TRUE(dynamic_cast<MixedPrecRPUDeviceCuda<num_t> *>(&*rpucuda_device) != nullptr);

  // also destroy
  rpucuda_device = nullptr;
  rpu_device = nullptr;
}

TEST_P(MixedPrecRPUDeviceCudaTestFixture, CudaCPUDifference) {
  dp.n_d_bins = GetParam();
  dp.n_x_bins = GetParam();
  dp.transfer_every = 1;
  dp.n_rows_per_transfer = 1;
  dp.compute_sparsity = true;
  // dp.print();

  rpu_device = dp.createDeviceUnique(x_size, d_size, &this->rw_rng);
  rpucuda_device = AbstractRPUDeviceCuda<num_t>::createFromUnique(context, *rpu_device);

  dev_weights->assignTranspose(w_ref[0], d_size, x_size);
  CudaArray<num_t> dev_x(context, x_size, x_vec);
  CudaArray<num_t> dev_d(context, d_size, d_vec);
  CudaArray<num_t> dev_x2(context, x_size, x_vec2);
  CudaArray<num_t> dev_d2(context, d_size, d_vec2);

  CudaArray<num_t> dev_x_buffer(context, x_size);
  CudaArray<num_t> dev_d_buffer(context, d_size);

  context->synchronize();
  num_t lr = 0.5;
  int reps = 3;
  std::vector<num_t> chi(x_size * d_size);

  for (int i = 0; i < reps; i++) {
    rpu_device->doDirectVectorUpdate(w_ref, x_vec, 1, d_vec, 1, lr, 1, up);
    rpucuda_device->doDirectUpdate(
        dev_x.getData(), dev_d.getData(), dev_weights->getData(), lr, 1, false, false, 1.0f, up,
        dev_x_buffer.getData(), dev_d_buffer.getData());

    getWeightsAll();
    // printChi();
    // printChiCuda();
    // printWeights();
    // printWeightsCuda();
  }

  for (int i = 0; i < reps; i++) {
    rpu_device->doDirectVectorUpdate(w_ref, x_vec2, 1, d_vec2, 1, lr, 1, up);
    rpucuda_device->doDirectUpdate(
        dev_x2.getData(), dev_d2.getData(), dev_weights->getData(), lr, 1, false, false, 1.0f, up,
        dev_x_buffer.getData(), dev_d_buffer.getData());

    getWeightsAll();
    // printChi();
    // printChiCuda();
    // printWeights();
    // printWeightsCuda();
  }

  context->synchronize();

  auto chi_cuda =
      static_cast<MixedPrecRPUDeviceCuda<num_t> *>(&*rpucuda_device)->getHiddenWeights();
  static_cast<MixedPrecRPUDevice<num_t> *>(&*rpu_device)->getChi(chi.data());

  for (int i = 0; i < x_size * d_size; i++) {
    ASSERT_NEAR((num_t)chi[i], chi_cuda[i], TOLERANCE);
  }

  for (int i = 0; i < x_size * d_size; i++) {
    ASSERT_NEAR(weights[0][i], w_ref[0][i], TOLERANCE);
  }

  // sparsity
  num_t sp = dynamic_cast<MixedPrecRPUDevice<num_t> *>(&*rpu_device)->getAvgSparsity();
  num_t sp_cuda = dynamic_cast<MixedPrecRPUDeviceCuda<num_t> *>(&*rpucuda_device)->getAvgSparsity();
  ASSERT_FLOAT_EQ(sp, sp_cuda);
}

TEST_P(MixedPrecRPUDeviceCudaTestFixture, Update) {

  dp.n_d_bins = GetParam();
  dp.n_x_bins = GetParam();

  dynamic_cast<ConstantStepRPUDeviceMetaParameter<num_t> *>(&*dp.device_par)->dw_min = 0.2;
  up.pulse_type = PulseType::NoneWithDevice;
  dp.transfer_every = 2;
  dp.n_rows_per_transfer = d_size;
  dp.compute_sparsity = false;

  // just newly create from paramerers
  rpu_device = dp.createDeviceUnique(this->x_size, this->d_size, &this->rw_rng);
  rpucuda_device = AbstractRPUDeviceCuda<num_t>::createFromUnique(context, *rpu_device);

  CudaArray<num_t> dev_x(context, this->x_size);
  dev_x.setConst(1.0);
  CudaArray<num_t> dev_d(context, this->d_size);
  dev_d.setConst(-1.0);
  dev_weights->assignTranspose(weights[0], d_size, x_size);
  context->synchronize();

  if (rpu_device->onSetWeights(this->weights)) {
    rpucuda_device->populateFrom(*rpu_device); // device pars have changed (due to onSetWeights)
  }
  context->synchronize();

  for (int i = 0; i < dp.transfer_every; i++) {
    up_pwu->update(
        dev_x.getDataConst(), dev_d.getDataConst(), dev_weights->getData(), &*rpucuda_device,
        this->up,
        1.0,   // lr
        1,     // batch
        false, // trans
        false);
  }
  // should update all weight values of the hidden weight by -reps
  context->synchronize();
  auto w_vec = static_cast<MixedPrecRPUDeviceCuda<num_t> *>(&*rpucuda_device)->getHiddenWeights();

  // update only on fast [nothing to transfer for first row]
  int size = this->d_size * this->x_size;
  // hidden weights updated (should be about - 1), since LR is applied later
  for (int i = 0; i < size; i++) {
    ASSERT_FLOAT_EQ(w_vec[i], -dp.transfer_every);
  }

  dev_weights->copyTo(weights[0]);
  dev_weights->assignTranspose(weights[0], x_size, d_size);
  dev_weights->copyTo(weights[0]);

  // weights should be all zero still
  for (int i = 0; i < x_size * d_size; i++) {
    ASSERT_FLOAT_EQ(weights[0][i], 0);
  }
}

TEST_P(MixedPrecRPUDeviceCudaTestFixture, UpdateAndTransfer) {

  num_t dw_min = 0.2;
  dp.n_d_bins = GetParam();
  dp.n_x_bins = GetParam();

  dynamic_cast<ConstantStepRPUDeviceMetaParameter<num_t> *>(&*dp.device_par)->dw_min = dw_min;
  up.pulse_type = PulseType::StochasticCompressed;
  dp.n_rows_per_transfer = d_size;

  num_t lr = 0.05;
  dp.compute_sparsity = false;
  dp.transfer_every = roundf(dw_min / lr);

  // just newly create from paramerers
  rpu_device = dp.createDeviceUnique(this->x_size, this->d_size, &this->rw_rng);
  rpucuda_device = AbstractRPUDeviceCuda<num_t>::createFromUnique(context, *rpu_device);

  CudaArray<num_t> dev_x(context, this->x_size);
  dev_x.setConst(1.0);
  CudaArray<num_t> dev_d(context, this->d_size);
  dev_d.setConst(-1.0);
  dev_weights->setConst(0.0);
  context->synchronize();

  // Since all 1 updates, the weight update should be once every dw_min/lr times
  // thres is floord though. take one update step more
  for (int i = 0; i < dp.transfer_every + 1; i++) {
    up_pwu->update(
        dev_x.getDataConst(), dev_d.getDataConst(), dev_weights->getData(), &*rpucuda_device,
        this->up,
        lr,    // lr
        1,     // batch
        false, // trans
        false);
  }
  context->synchronize();

  auto chi_cuda =
      static_cast<MixedPrecRPUDeviceCuda<num_t> *>(&*rpucuda_device)->getHiddenWeights();

  // printChiCuda();
  // printWeightsCuda();

  dev_weights->copyTo(weights[0]);
  dev_weights->assignTranspose(weights[0], x_size, d_size);
  dev_weights->copyTo(weights[0]);

  // weights should be constant dw_min as only one transfer-update occurred
  for (int i = 0; i < x_size * d_size; i++) {
    ASSERT_FLOAT_EQ(weights[0][i], dw_min);
  }
}

} // namespace

int main(int argc, char **argv) {
  resetCuda();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
