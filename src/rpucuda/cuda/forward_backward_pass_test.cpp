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

#define TOLERANCE 5e-3
namespace {

using namespace RPU;

template <typename T1> class FBTestFixtureNoNoise : public ::testing::Test {
public:
  using T = T1;

  void SetUp() {

    is_test = false;

    context = &context_container;

    x_size = 241;
    d_size = 250;
    K = 10;
    m_batch = 500; // for the batched versions

    IOMetaParameter<T> p_io;

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
    p_io.out_bound = 12;
    p_io.inp_bound = 1.0;

    non_linearities = true;

    p_io.noise_management = NoiseManagementType::None;
    p_io.bound_management = BoundManagementType::None;

    p.up.update_management = true;
    p.up.update_bl_management = true;
    p.up.pulse_type = PulseType::None;
    dp.lifetime = 0;
    dp.diffusion = 0.0;

    p.f_io = p_io;
    p.b_io = p_io;
    p.b_io.bound_management = BoundManagementType::None;

    x1.resize(x_size * m_batch);
    x2.resize(x_size * m_batch);
    x3.resize(x_size * m_batch);
    d1.resize(d_size * m_batch);
    d2.resize(d_size * m_batch);
    d3.resize(d_size * m_batch);
    rx.resize(x_size * m_batch);
    rd.resize(d_size * m_batch);

    unsigned int seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator{seed};
    std::uniform_real_distribution<float> udist(-1.2, 1.2);
    auto urnd = std::bind(udist, generator);

    // just assign some numbers from the weight matrix
    for (int i = 0; i < x_size * m_batch; i++)
      rx[i] = urnd();

    for (int j = 0; j < d_size * m_batch; j++) {
      rd[j] = urnd();
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
  }

  void PopulatePulsed() {

    T lr = 0.01;
    T bmin = -0.4;
    T bmax = 0.4;

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
  }

  void TestForward(bool trans, bool only_timing = false) {
    CUDA_TIMING_INIT;
    this->x_cuvec_batch->assign(this->rx.data());
    this->x_vec_batch = this->rx;

    this->context->synchronizeDevice();

    bool bias = true;
    this->culayer_pulsed->forward(
        this->x_cuvec_batch->getData(), this->d_cuvec_batch->getData(), bias, this->m_batch, trans,
        trans, this->is_test);

    // RPUCuda forward
    CUDA_TIMING_START(this->context);
    this->culayer_pulsed->forward(
        this->x_cuvec_batch->getData(), this->d_cuvec_batch->getData(), bias, this->m_batch, trans,
        trans, this->is_test);
    CUDA_TIMING_STOP(this->context, "CUDA Forward (trans: " << trans << ")");
    this->context->synchronizeDevice();

    if (only_timing) {
      return;
    }

    this->d_cuvec_batch->copyTo(this->d1.data());

    // RPU forward
    this->layer_pulsed->forward(
        this->x_vec_batch.data(), this->d_vec_batch.data(), bias, this->m_batch, trans, trans,
        this->is_test);
    this->d2 = this->d_vec_batch;

    this->layer->forward(
        this->x_vec_batch.data(), this->d_vec_batch.data(), bias, this->m_batch, trans, trans,
        this->is_test);
    this->d3 = this->d_vec_batch;

    double accum = 0.0;
    for (int i = 0; i < this->d_size * this->m_batch; i++) {
      // std::cout << i << ": d1 " << this->d1[i] << " vs d2 " << this->d2[i] << std::endl;
      ASSERT_NEAR(this->d1[i], this->d2[i], TOLERANCE);

      accum += fabsf((double)(this->d2[i] - this->d3[i]));
      accum += fabsf((double)(this->d1[i] - this->d3[i]));
    }
    ASSERT_TRUE(accum > TOLERANCE);

    CUDA_TIMING_DESTROY;
  }

  void TestBackward(bool trans) {
    CUDA_TIMING_INIT;
    this->d_cuvec_batch->assign(this->rd.data());
    this->d_vec_batch = this->rd;

    this->context->synchronizeDevice();

    // this->culayer_pulsed->printWeights(3, 3);
    // this->layer_pulsed->printWeights(3, 3);
    // this->layer->printWeights(3, 3);
    // this->culayer_pulsed->dispParameter();

    // RPUCuda backward
    this->culayer_pulsed->backward(
        this->d_cuvec_batch->getData(), this->x_cuvec_batch->getData(), true, this->m_batch, trans,
        trans);

    CUDA_TIMING_START(this->context);
    this->culayer_pulsed->backward(
        this->d_cuvec_batch->getData(), this->x_cuvec_batch->getData(), true, this->m_batch, trans,
        trans);
    CUDA_TIMING_STOP(this->context, "CUDA Backward (trans: " << trans << ")");
    this->context->synchronizeDevice();
    this->x_cuvec_batch->copyTo(this->x1.data());
    this->context->synchronizeDevice();

    // RPU backward
    this->layer_pulsed->backward(
        this->d_vec_batch.data(), this->x_vec_batch.data(), true, this->m_batch, trans, trans);
    this->x2 = this->x_vec_batch;

    this->layer->backward(
        this->d_vec_batch.data(), this->x_vec_batch.data(), true, this->m_batch, trans, trans);
    this->x3 = this->x_vec_batch;

    double accum = 0.0;
    for (int i = 0; i < (this->x_size - 1) * this->m_batch; i++) {

      ASSERT_NEAR(this->x1[i], this->x2[i], TOLERANCE);

      accum += fabs((double)(this->x2[i] - this->x3[i]));
      accum += fabs((double)(this->x1[i] - this->x3[i]));
    }
    ASSERT_TRUE(accum > TOLERANCE);

    CUDA_TIMING_DESTROY;
  }

  void TearDown() {}

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
  int K;
  int m_batch;
  bool is_test;
  bool non_linearities = false;
  std::vector<T> x1, x2, x3;
  std::vector<T> d1, d2, d3;
  std::vector<T> rx, rd;
  PulsedMetaParameter<T> p;
  SimpleRPUDeviceMetaParameter<T> dp;
  T noise_value;
};

typedef ::testing::Types<num_t, float> Tios;
TYPED_TEST_CASE(FBTestFixtureNoNoise, Tios);

TYPED_TEST(FBTestFixtureNoNoise, IRDrop) {

  this->p.f_io.mv_type = AnalogMVType::OnePass;
  this->p.b_io.mv_type = AnalogMVType::OnePass;

  this->p.f_io.ir_drop = 3;
  this->p.b_io.ir_drop = 5;

  this->PopulatePulsed();
  this->TestForward(true);
  this->TestBackward(true);
  this->TestForward(false);
  this->TestBackward(false);
}

TYPED_TEST(FBTestFixtureNoNoise, Asymmetry) {

  this->p.f_io.mv_type = AnalogMVType::OnePass;
  this->p.b_io.mv_type = AnalogMVType::OnePass;

  this->p.f_io.inp_asymmetry = 0.2;
  this->p.f_io.out_asymmetry = 0.12;
  this->p.b_io.inp_asymmetry = 0.3;
  this->p.b_io.out_asymmetry = 0.02;

  this->PopulatePulsed();
  this->TestForward(true);
  this->TestBackward(true);
  this->TestForward(false);
  this->TestBackward(false);
}

TYPED_TEST(FBTestFixtureNoNoise, NonLinearity) {

  this->p.f_io.mv_type = AnalogMVType::OnePass;
  this->p.b_io.mv_type = AnalogMVType::OnePass;

  this->p.f_io.out_nonlinearity = 0.2;
  this->p.f_io.out_nonlinearity_std = 1.0;
  this->p.b_io.out_nonlinearity = 0.3;
  this->p.b_io.out_nonlinearity_std = 1.2;

  this->PopulatePulsed();
  this->TestForward(true);
  this->TestBackward(true);
  this->TestForward(false);
  this->TestBackward(false);
}

TYPED_TEST(FBTestFixtureNoNoise, TwoPassDigitalSum) {

  this->p.f_io.mv_type = AnalogMVType::PosNegSeparateDigitalSum;
  this->p.b_io.mv_type = AnalogMVType::PosNegSeparateDigitalSum;

  this->p.f_io.inp_asymmetry = 0.2;
  this->p.f_io.out_asymmetry = 0.12;
  this->p.b_io.inp_asymmetry = 0.3;
  this->p.b_io.out_asymmetry = 0.02;

  this->p.f_io.out_nonlinearity = 0.2;
  this->p.f_io.out_nonlinearity_std = 1.0;
  this->p.b_io.out_nonlinearity = 0.3;
  this->p.b_io.out_nonlinearity_std = 1.2;

  this->p.b_io.w_read_asymmetry_dtod = 2.0;
  this->p.f_io.w_read_asymmetry_dtod = 2.0;

  this->PopulatePulsed();
  this->TestForward(true);
  this->TestBackward(true);
  this->TestForward(false);
  this->TestBackward(false);
}

TYPED_TEST(FBTestFixtureNoNoise, TwoPass) {

  this->p.f_io.mv_type = AnalogMVType::PosNegSeparate;
  this->p.b_io.mv_type = AnalogMVType::PosNegSeparate;

  this->p.f_io.inp_asymmetry = 0.2;
  this->p.f_io.out_asymmetry = 0.12;
  this->p.b_io.inp_asymmetry = 0.3;
  this->p.b_io.out_asymmetry = 0.4;

  this->p.f_io.out_nonlinearity = 0.2;
  this->p.f_io.out_nonlinearity_std = 1.0;
  this->p.b_io.out_nonlinearity = 0.3;
  this->p.b_io.out_nonlinearity_std = 1.2;

  this->p.b_io.w_read_asymmetry_dtod = 2.3;
  this->p.f_io.w_read_asymmetry_dtod = 2.0;

  this->PopulatePulsed();
  this->TestForward(true);
  this->TestBackward(true);
  this->TestForward(false);
  this->TestBackward(false);
}

TYPED_TEST(FBTestFixtureNoNoise, TwoPassDigitalSumVoffset) {

  this->p.f_io.mv_type = AnalogMVType::PosNegSeparateDigitalSum;
  this->p.b_io.mv_type = AnalogMVType::PosNegSeparateDigitalSum;

  this->p.b_io.v_offset_std = 0.05;
  this->p.f_io.v_offset_std = 0.1;

  this->p.b_io.r_series = 0.01;
  this->p.f_io.r_series = 0.02;

  this->p.b_io.slope_calibration = 0.3;
  this->p.f_io.slope_calibration = 0.5;

  this->PopulatePulsed();
  this->TestForward(true);
  this->TestBackward(true);
  this->TestForward(false);
  this->TestBackward(false);
}

TYPED_TEST(FBTestFixtureNoNoise, TwoPassDigitalSumVoffsetWmin) {

  this->p.f_io.mv_type = AnalogMVType::PosNegSeparateDigitalSum;
  this->p.b_io.mv_type = AnalogMVType::PosNegSeparateDigitalSum;

  this->p.b_io.v_offset_std = 0.05;
  this->p.f_io.v_offset_std = 0.03;

  this->p.b_io.r_series = 0.01;
  this->p.f_io.r_series = 0.02;

  this->p.f_io.v_offset_w_min = -1.0;
  this->p.b_io.v_offset_w_min = -0.5;

  this->p.b_io.slope_calibration = 0.3;
  this->p.f_io.slope_calibration = 0.5;

  this->PopulatePulsed();
  this->TestForward(true);
  this->TestBackward(true);
  this->TestForward(false);
  this->TestBackward(false);
}

TYPED_TEST(FBTestFixtureNoNoise, NoisyForward) {

  this->p.f_io.mv_type = AnalogMVType::OnePass;
  this->p.f_io.noise_management = NoiseManagementType::AbsMax;
  this->p.f_io.out_noise = 0.06;
  this->p.f_io.w_noise = 0.012;
  this->p.f_io.w_noise_type = OutputWeightNoiseType::PCMRead;
  this->p.f_io.inp_res = 256.;
  this->p.f_io.out_res = 256.;
  this->p.f_io.inp_bound = 1.;
  this->p.f_io.out_bound = 12.;

  this->PopulatePulsed();
  this->TestForward(true, true);
  this->TestForward(false, true);
}

} // namespace
int main(int argc, char **argv) {
  resetCuda();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
