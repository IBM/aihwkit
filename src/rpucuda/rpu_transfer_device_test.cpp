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

#include "rng.h"
#include "rpu_constantstep_device.h"
#include "rpu_transfer_device.h"
#include "utility_functions.h"
#include "gtest/gtest.h"
#include <chrono>
#include <memory>
#include <random>
// #include "test_helper.h"

#define TOLERANCE 1e-5

namespace {

using namespace RPU;

class RPUDeviceTestFixture : public ::testing::TestWithParam<float> {
public:
  void SetUp() {
    x_size = 2;
    d_size = 3;

    n_pos = 3;
    n_neg = 2;
    colidx = 1;
    x_indices = new int[n_pos + n_neg];
    for (int i = 0; i < n_pos; i++) {
      x_indices[i] = colidx + 1;
    }
    for (int i = 0; i < n_neg; i++) {
      x_indices[i + n_pos] = -(colidx + 1);
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
    dp_cs.dw_min = 1;
    dp_cs.dw_min_dtod = 0.0;
    dp_cs.dw_min_std = 0.0;
    dp_cs.up_down_dtod = 0.0;
    dp_cs.w_max = 100;
    dp_cs.w_min = -100;
    dp_cs.w_max_dtod = 0;
    dp_cs.w_min_dtod = 0;
    dp_cs.lifetime = lifetime;

    dp = new TransferRPUDeviceMetaParameter<num_t>(dp_cs, 2);

    dp->gamma = GetParam(); // meaning fully hidden
    dp->transfer_lr = 1;

    dp->transfer_io.inp_res = -1;
    dp->transfer_io.out_res = -1;
    dp->transfer_io.out_noise = 0.0;

    dp->transfer_up = up;
    // dp->transfer_up.pulse_type = PulseType::None;

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
  int n_pos, n_neg, x_size, d_size, colidx;
  num_t lifetime;
  num_t **weights;
  num_t **w_ref;
  PulsedUpdateMetaParameter<num_t> up;
  TransferRPUDeviceMetaParameter<num_t> *dp;
  ConstantStepRPUDeviceMetaParameter<num_t> dp_cs;
  RNG<num_t> *rng;
  RealWorldRNG<num_t> rw_rng;
  TransferRPUDevice<num_t> *rpu_device;
};

// define the tests
INSTANTIATE_TEST_CASE_P(GammaWeightening, RPUDeviceTestFixture, ::testing::Values(0.0, 0.5));

TEST_P(RPUDeviceTestFixture, createDevice) {
  rpu_device = this->dp->createDevice(this->x_size, this->d_size, &this->rw_rng);
  ASSERT_TRUE(dynamic_cast<TransferRPUDevice<num_t> *>(&*rpu_device) != nullptr);
  // rpu_device->dispInfo();

  delete rpu_device;
}

TEST_P(RPUDeviceTestFixture, onSetWeights) {

  for (int i = 0; i < this->x_size * this->d_size; i++) {
    this->weights[0][i] = w_ref[0][i];
  }

  rpu_device = this->dp->createDevice(this->x_size, this->d_size, &this->rw_rng);

  rpu_device->onSetWeights(this->weights);
  num_t ***w_vec = rpu_device->getWeightVec();
  const num_t *reduce_weightening = rpu_device->getReduceWeightening();
  ASSERT_FLOAT_EQ(reduce_weightening[1], 1);
  ASSERT_FLOAT_EQ(reduce_weightening[0], GetParam());

  if (GetParam()) { // for fully hidden internal weights are not used...
    for (int i = 0; i < this->x_size * this->d_size; i++) {
      num_t w = this->weights[0][i];
      ASSERT_FLOAT_EQ(w, w_ref[0][i]);
      ASSERT_FLOAT_EQ(w_vec[0][0][i], 0);
      ASSERT_FLOAT_EQ(w_vec[1][0][i], w / reduce_weightening[1]);
    }
  }
  delete rpu_device;
}

TEST_P(RPUDeviceTestFixture, doSparseUpdate) {

  dp->transfer_lr = 0; // no transfer here
  rpu_device = this->dp->createDevice(this->x_size, this->d_size, &this->rw_rng);
  rpu_device->onSetWeights(this->weights); // all zero
  rpu_device->initUpdateCycle(this->weights, this->up, 1, 1);

  // last row update, net 1 (n_pos-n_neg)
  float dx = rpu_device->getWeightGranularity() * (num_t)(n_pos - n_neg);
  int rowidx = this->d_size - 1;
  rpu_device->doSparseUpdate(
      this->weights, rowidx, this->x_indices, this->n_neg + this->n_pos, (num_t)-1.0, this->rng);
  rpu_device->finishUpdateCycle(this->weights, this->up, 1, 1);

  num_t ***w_vec = rpu_device->getWeightVec();

  // update only on fast [nothing to transfer for first row]
  for (int m = 0; m < (int)this->dp->vec_par.size(); m++) {
    for (int j = 0; j < this->d_size; j++) {
      for (int i = 0; i < this->x_size; i++) {
        if (j == rowidx && i == this->colidx && m == 0) {
          ASSERT_FLOAT_EQ(w_vec[m][j][i], dx);
        } else {
          ASSERT_FLOAT_EQ(w_vec[m][j][i], 0);
        }
      }
    }
  }

  // reduce to weight. Only if gamma is set
  for (int j = 0; j < this->d_size; j++) {
    for (int i = 0; i < this->x_size; i++) {
      if (j == rowidx && i == this->colidx) {
        ASSERT_FLOAT_EQ(this->weights[j][i], dx * GetParam());
      } else {
        ASSERT_FLOAT_EQ(this->weights[j][i], 0);
      }
    }
  }
  delete rpu_device;
}

TEST_P(RPUDeviceTestFixture, doSparseUpdateWithTransfer) {
  rpu_device = this->dp->createDevice(this->x_size, this->d_size, &this->rw_rng);
  rpu_device->onSetWeights(this->weights); // all zero
  rpu_device->initUpdateCycle(this->weights, this->up, 1, 1);

  // last row update, net 1 (n_pos-n_neg)
  float dx = rpu_device->getWeightGranularity() * (num_t)(n_pos - n_neg);
  int rowidx = this->d_size - 1;
  rpu_device->doSparseUpdate(
      this->weights, rowidx, this->x_indices, this->n_neg + this->n_pos, (num_t)-1.0, this->rng);
  rpu_device->finishUpdateCycle(this->weights, this->up, 1, 1); // to signal the end of the update

  for (int i = 0; i < this->x_size; i++) {
    rpu_device->initUpdateCycle(this->weights, this->up, 1, 1);   // to signal the end of the update
    rpu_device->finishUpdateCycle(this->weights, this->up, 1, 1); // to signal the end of the update
  }
  // should have transfered all cols once.

  num_t ***w_vec = rpu_device->getWeightVec();

  // update only on fast [nothing to transfer for first row]
  for (size_t m = 0; m < this->dp->vec_par.size(); m++) {
    for (int j = 0; j < this->d_size; j++) {
      for (int i = 0; i < this->x_size; i++) {
        if (j == rowidx && i == this->colidx) {
          if (m == 1 && GetParam() == 0) {
            // if fullyHidden (GetParam==0) then actual weight is updated directly
            ASSERT_FLOAT_EQ(w_vec[m][j][i], 0);
          } else {
            // should be fully transferred, since transfer_lr = 1
            ASSERT_FLOAT_EQ(w_vec[m][j][i], dx);
          }
        } else {
          ASSERT_FLOAT_EQ(w_vec[m][j][i], 0);
        }
      }
    }
  }

  // reduce to weight. Weightening does not necessarily sums to 1
  const num_t *reduce_weightening = rpu_device->getReduceWeightening();
  num_t sc = reduce_weightening[0] + reduce_weightening[1];
  for (int j = 0; j < this->d_size; j++) {
    for (int i = 0; i < this->x_size; i++) {
      if (j == rowidx && i == this->colidx) {
        ASSERT_FLOAT_EQ(this->weights[j][i], sc * (num_t)dx);
      } else {
        ASSERT_FLOAT_EQ(this->weights[j][i], 0);
      }
    }
  }
  delete rpu_device;
}

TEST_P(RPUDeviceTestFixture, doSparseUpdateWithTransferRows) {
  this->dp->transfer_columns = false;
  rpu_device = this->dp->createDevice(this->x_size, this->d_size, &this->rw_rng);
  rpu_device->onSetWeights(this->weights); // all zero
  rpu_device->initUpdateCycle(this->weights, this->up, 1, 1);

  // last row update, net 1 (n_pos-n_neg)
  float dx = rpu_device->getWeightGranularity() * (num_t)(n_pos - n_neg);
  int rowidx = this->d_size - 1;
  rpu_device->doSparseUpdate(
      this->weights, rowidx, this->x_indices, this->n_neg + this->n_pos, (num_t)-1.0, this->rng);
  rpu_device->finishUpdateCycle(this->weights, this->up, 1, 1); // to signal the end of the update

  for (int i = 0; i < this->d_size; i++) {
    rpu_device->initUpdateCycle(this->weights, this->up, 1, 1);   // to signal the end of the update
    rpu_device->finishUpdateCycle(this->weights, this->up, 1, 1); // to signal the end of the update
  }
  // should have transfered all cols once.

  num_t ***w_vec = rpu_device->getWeightVec();

  // update only on fast [nothing to transfer for first row]
  for (size_t m = 0; m < this->dp->vec_par.size(); m++) {
    for (int j = 0; j < this->d_size; j++) {
      for (int i = 0; i < this->x_size; i++) {
        if (j == rowidx && i == this->colidx) {
          if (m == 1 && GetParam() == 0) {
            // if fullyHidden (GetParam==0) then actual weight is updated directly
            ASSERT_FLOAT_EQ(w_vec[m][j][i], 0);
          } else {
            // should be fully transferred, since transfer_lr = 1
            ASSERT_FLOAT_EQ(w_vec[m][j][i], dx);
          }
        } else {
          ASSERT_FLOAT_EQ(w_vec[m][j][i], 0);
        }
      }
    }
  }

  // reduce to weight. Weightening does not necessarily sums to 1
  const num_t *reduce_weightening = rpu_device->getReduceWeightening();
  num_t sc = reduce_weightening[0] + reduce_weightening[1];
  for (int j = 0; j < this->d_size; j++) {
    for (int i = 0; i < this->x_size; i++) {
      if (j == rowidx && i == this->colidx) {
        ASSERT_FLOAT_EQ(this->weights[j][i], sc * (num_t)dx);
      } else {
        ASSERT_FLOAT_EQ(this->weights[j][i], 0);
      }
    }
  }
  delete rpu_device;
}

TEST_P(RPUDeviceTestFixture, Decay) {

  for (int i = 0; i < this->x_size * this->d_size; i++) {
    this->weights[0][i] = w_ref[0][i];
    w_ref[0][i] *= ((num_t)1.0 - (num_t)1.0 / this->lifetime);
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
