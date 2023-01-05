/**
 * (C) Copyright 2020, 2021, 2022 IBM. All Rights Reserved.
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
#include "rpu_pulsed.h"
#include "utility_functions.h"
#include "gtest/gtest.h"
#include <chrono>
#include <memory>
#include <random>

#define TOLERANCE 1e-5

#ifdef RPU_USE_DOUBLE
typedef double T;
#else
typedef float T;
#endif

namespace {

using namespace RPU;

class RPUTestNoiseFreeFixture : public ::testing::TestWithParam<int> {
public:
  void SetUp() {

    x_size = 100;
    d_size = 101;

    IOMetaParameter<T> p_io;
    // noise free

    p.up.desired_BL = 20;

    dp.w_max = 1.1;
    dp.w_min = -1.1;
    dp.w_min_dtod = 0.0;
    dp.w_max_dtod = 0.0;

    dp.dw_min = 0.005;
    dp.dw_min_std = 0.0;
    dp.dw_min_dtod = 0.3;

    dp.up_down = 0.0;
    dp.up_down_dtod = 0.01;

    // peripheral circuits specs
    p_io.inp_res = 0.01;
    p_io.inp_sto_round = false;
    p_io.out_res = 0.01;
    p_io.out_sto_round = false;
    p_io.out_noise = 0.0;
    p_io.w_noise = 0.0;
    p_io.inp_noise = 0.0;

    // turn on all asymetries with systematic variations
    p_io.ir_drop = 1.0;
    p_io.v_offset_std = 0.03;
    p_io.w_read_asymmetry_dtod = 0.05;

    p_io.out_nonlinearity = 0.1;
    p_io.out_nonlinearity_std = 0.1;

    p_io.noise_management = NoiseManagementType::None;
    p_io.bound_management = BoundManagementType::None;

    p.up.update_management = false;
    p.up.update_bl_management = false;

    p.up.pulse_type = PulseType::DeterministicImplicit;
    p.up.x_res_implicit = 0.01;
    p.up.d_res_implicit = 0.01;

    p.f_io = p_io;
    p.b_io = p_io;

    rx.resize(x_size);
    rd.resize(d_size);

    unsigned int seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator{seed};
    std::uniform_real_distribution<T> udist(-1.2, 1.2);
    auto urnd = std::bind(udist, generator);

    // just assign some numbers from the weigt matrix
    for (int i = 0; i < x_size; i++)
      rx[i] = urnd();

    for (int j = 0; j < d_size; j++) {
      rd[j] = urnd();
    }

    std::uniform_real_distribution<T> udist2(-0.2, 0.2);
    auto urnd2 = std::bind(udist2, generator);
    w.resize(d_size * x_size);
    for (int j = 0; j < d_size * x_size; j++) {
      w[j] = urnd2();
    }
    w2 = w;
    x = rx;
    d = rd;
    x2 = rx;
    d2 = rd;
  }

  void constructRPU() {
    rpu = RPU::make_unique<RPUPulsed<T>>(x_size, d_size);
    rpu->populateParameter(&p, &dp);
    rpu->setLearningRate(0.1);
    rpu->setWeightsUniformRandom(-0.5, 0.5);
  }

  void TearDown() {}

  PulsedMetaParameter<T> p;
  ConstantStepRPUDeviceMetaParameter<T> dp;

  std::unique_ptr<RPUPulsed<T>> rpu;
  int x_size;
  int d_size;
  std::vector<T> rx, rd, w, x, d, x2, d2, w2;
};

// types
INSTANTIATE_TEST_CASE_P(MVType, RPUTestNoiseFreeFixture, ::testing::Range(0, 2));

TEST_P(RPUTestNoiseFreeFixture, ConstructAndCopy) {

  p.f_io.mv_type = (RPU::AnalogMVType)GetParam();
  p.b_io.mv_type = (RPU::AnalogMVType)GetParam();
  constructRPU();

  auto rpu2(*rpu);

  rpu->forward(rx.data(), d.data());
  rpu->backward(rd.data(), x.data());
  rpu->update(rd.data(), rx.data());
  rpu->getWeights(w.data());

  rpu2.forward(rx.data(), d2.data());
  rpu2.backward(rd.data(), x2.data());
  rpu2.update(rd.data(), rx.data());
  rpu2.getWeights(w2.data());

  for (int i = 0; i < d_size; i++) {
    ASSERT_NEAR(d[i], d2[i], TOLERANCE);
  }
  for (int i = 0; i < x_size; i++) {
    ASSERT_NEAR(x[i], x2[i], TOLERANCE);
  }
  for (int i = 0; i < x_size * d_size; i++) {
    ASSERT_NEAR(w[i], w2[i], TOLERANCE);
  }
}

TEST_P(RPUTestNoiseFreeFixture, ConstructAndMove) {

  p.f_io.mv_type = (RPU::AnalogMVType)GetParam();
  p.b_io.mv_type = (RPU::AnalogMVType)GetParam();
  constructRPU();

  rpu->forward(rx.data(), d.data());
  rpu->backward(rd.data(), x.data());

  auto rpu2 = std::move(*rpu);

  rpu2.forward(rx.data(), d2.data());
  rpu2.backward(rd.data(), x2.data());

  for (int i = 0; i < d_size; i++) {
    ASSERT_NEAR(d[i], d2[i], TOLERANCE);
  }
  for (int i = 0; i < x_size; i++) {
    ASSERT_NEAR(x[i], x2[i], TOLERANCE);
  }

  // update
  constructRPU();
  rpu->getWeights(w2.data());
  rpu->update(rd.data(), rx.data());
  rpu->getWeights(w.data());

  rpu2 = std::move(*rpu);
  rpu2.setWeights(w2.data());
  rpu2.update(rd.data(), rx.data());
  rpu2.getWeights(w2.data());

  for (int i = 0; i < x_size * d_size; i++) {
    ASSERT_NEAR(w[i], w2[i], TOLERANCE);
  }
}

} // namespace
  // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
