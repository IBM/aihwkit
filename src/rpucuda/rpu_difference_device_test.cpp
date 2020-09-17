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

#include "rng.h"
#include "rpu_constantstep_device.h"
#include "rpu_difference_device.h"
#include "utility_functions.h"
#include "gtest/gtest.h"
#include <chrono>
#include <memory>
#include <random>
//#include "test_helper.h"

#define TOLERANCE 1e-5

#ifdef RPU_USE_DOUBLE
typedef double num_t;
#else
typedef float num_t;
#endif

namespace {

using namespace RPU;

class RPUDeviceTestFixture : public ::testing::TestWithParam<int> {
public:
  void SetUp() {
    x_size = 2;
    d_size = 3;

    n_pos = GetParam() + 4;
    n_neg = GetParam();
    x_indices = new int[n_pos + n_neg];
    for (int i = 0; i < n_pos; i++) {
      x_indices[i] = 1;
    }
    for (int i = 0; i < n_neg; i++) {
      x_indices[i + n_pos] = -1;
    }

    w_ref = Array_2D_Get<num_t>(d_size, x_size);

    for (int i = 0; i < x_size * d_size; i++) {
      w_ref[0][i] = rw_rng.sampleGauss();
    }

    weights = Array_2D_Get<num_t>(d_size, x_size);
    for (int i = 0; i < d_size * x_size; i++) {
      weights[0][i] = 0;
    }
    up.desired_BL = n_pos + n_neg;

    lifetime = 100;
    num_t dw_min = 1.0;

    dp_cs.dw_min = dw_min;
    dp_cs.dw_min_dtod = 0.0;
    dp_cs.dw_min_std = 0.0;
    dp_cs.up_down_dtod = 0.0;
    dp_cs.w_max = 100;
    dp_cs.w_min = -100;
    dp_cs.w_max_dtod = 0;
    dp_cs.w_min_dtod = 0;
    dp_cs.lifetime = lifetime;

    dp = new DifferenceRPUDeviceMetaParameter<num_t>(dp_cs);
    // note: meta parameters for g+ and g- need to be the same (to support inversion)
    reference_update = dw_min * n_pos - dw_min * n_neg;
    reference_update_inv = -dw_min * n_neg + dw_min * n_pos;

    rng = new RNG<num_t>(0);
  };

  void TearDown() {
    Array_2D_Free<num_t>(weights);
    Array_2D_Free<num_t>(w_ref);
    delete[] x_indices;
    delete dp;
    delete rng;
  };

  int *x_indices;
  int n_pos, n_neg, x_size, d_size;
  num_t lifetime;
  num_t **weights;
  num_t **w_ref;
  num_t reference_update_inv, reference_update;
  PulsedUpdateMetaParameter<num_t> up;
  DifferenceRPUDeviceMetaParameter<num_t> *dp;
  ConstantStepRPUDeviceMetaParameter<num_t> dp_cs;
  RNG<num_t> *rng;
  RealWorldRNG<num_t> rw_rng;
  DifferenceRPUDevice<num_t> *rpu_device;
};

// define the tests
INSTANTIATE_TEST_CASE_P(NumOfUpdates, RPUDeviceTestFixture, ::testing::Range(5, 6));

TEST_P(RPUDeviceTestFixture, Empty) {}

TEST_P(RPUDeviceTestFixture, createDevice) {

  rpu_device = this->dp->createDevice(this->x_size, this->d_size, &this->rw_rng);
  ASSERT_TRUE(rpu_device != nullptr);
  rpu_device->dispInfo();

  delete rpu_device;
}

TEST_P(RPUDeviceTestFixture, onSetWeights) {

  for (int i = 0; i < this->x_size * this->d_size; i++) {
    this->weights[0][i] = w_ref[0][i];
  }

  rpu_device = this->dp->createDevice(this->x_size, this->d_size, &this->rw_rng);
  rpu_device->onSetWeights(this->weights);
  num_t ***w_vec = rpu_device->getWeightVec();
  for (int i = 0; i < this->x_size * this->d_size; i++) {
    num_t w = this->weights[0][i];
    ASSERT_FLOAT_EQ(w, w_ref[0][i]);
    if (w >= 0) {
      ASSERT_FLOAT_EQ(w_vec[1][0][i], w_ref[0][i]);
      ASSERT_FLOAT_EQ(w_vec[0][0][i], 0);
    } else {
      ASSERT_FLOAT_EQ(w_vec[0][0][i], -w_ref[0][i]);
      ASSERT_FLOAT_EQ(w_vec[1][0][i], 0);
    }
  }
  delete rpu_device;
}

TEST_P(RPUDeviceTestFixture, doSparseUpdate) {
  rpu_device = this->dp->createDevice(this->x_size, this->d_size, &this->rw_rng);
  rpu_device->onSetWeights(this->weights);
  rpu_device->dispInfo();

  rpu_device->initUpdateCycle(this->weights, this->up, 1.0, 1);
  rpu_device->doSparseUpdate(
      this->weights, 0, this->x_indices, this->n_neg + this->n_pos, (num_t)-1, this->rng);

  ASSERT_FLOAT_EQ(this->weights[0][0], this->reference_update);
  for (int i = 1; i < this->d_size * this->x_size; i++) {
    ASSERT_FLOAT_EQ(this->weights[0][i], 0);
  }

  num_t ***w_vec = rpu_device->getWeightVec();
  ASSERT_FLOAT_EQ(w_vec[0][0][0], (*this->dp)[0]->dw_min * n_neg);
  ASSERT_FLOAT_EQ(w_vec[1][0][0], (*this->dp)[1]->dw_min * n_pos);

  delete rpu_device;
}

TEST_P(RPUDeviceTestFixture, doSparseUpdateInvert) {
  rpu_device = this->dp->createDevice(this->x_size, this->d_size, &this->rw_rng);
  rpu_device->onSetWeights(this->weights);
  rpu_device->dispInfo();

  rpu_device->initUpdateCycle(this->weights, this->up, 1.0, 1);
  rpu_device->invert();
  rpu_device->doSparseUpdate(
      this->weights, 0, this->x_indices, this->n_neg + this->n_pos, (num_t)-1, this->rng);

  ASSERT_FLOAT_EQ(this->weights[0][0], this->reference_update_inv);
  for (int i = 1; i < d_size * x_size; i++) {
    ASSERT_FLOAT_EQ(this->weights[0][i], 0);
  }

  num_t ***w_vec = rpu_device->getWeightVec();
  ASSERT_FLOAT_EQ(w_vec[0][0][0], (*this->dp)[0]->dw_min * n_pos);
  ASSERT_FLOAT_EQ(w_vec[1][0][0], (*this->dp)[1]->dw_min * n_neg);

  delete rpu_device;
}

TEST_P(RPUDeviceTestFixture, Decay) {

  for (int i = 0; i < this->x_size * this->d_size; i++) {
    this->weights[0][i] = this->w_ref[0][i];
    this->w_ref[0][i] *= ((num_t)1.0 - (num_t)1.0 / this->lifetime);
  }

  rpu_device = this->dp->createDevice(this->x_size, this->d_size, &this->rw_rng);
  rpu_device->onSetWeights(this->weights);

  rpu_device->decayWeights(this->weights, false);

  for (int i = 0; i < d_size * x_size; i++) {
    ASSERT_FLOAT_EQ(this->weights[0][i], w_ref[0][i]);
  }

  delete rpu_device;
}

TEST_P(RPUDeviceTestFixture, DecayNoBiasDecay) {

  for (int i = 0; i < this->x_size * this->d_size; i++) {
    this->weights[0][i] = w_ref[0][i];
    if (i % this->x_size != this->x_size - 1) {
      w_ref[0][i] *= ((num_t)1.0 - (num_t)1.0 / this->lifetime);
    }
  }

  rpu_device = this->dp->createDevice(this->x_size, this->d_size, &this->rw_rng);
  rpu_device->onSetWeights(this->weights);

  rpu_device->decayWeights(this->weights, true);

  for (int i = 0; i < d_size * x_size; i++) {
    ASSERT_FLOAT_EQ(this->weights[0][i], w_ref[0][i]);
  }

  delete rpu_device;
}

// test reduce to weights !!

} // namespace

int main(int argc, char **argv) {
  // resetCuda();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
