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
#include "rpucuda_pulsed.h"
#include "utility_functions.h"
#include "gtest/gtest.h"
#include <chrono>
#include <memory>
#include <random>

#define TOLERANCE 1e-5

namespace {

using namespace RPU;

int discretize(
    num_t *vq,
    num_t *v,
    int size,
    num_t bound,
    num_t resolution,
    bool noise_management,
    num_t bm_scale,
    num_t post_scale) {

  int exceeds_bound = 0;
  num_t scale_value = bm_scale;
  num_t bound_upper = bound;
  num_t bound_lower = -bound;

  if (noise_management)
    scale_value *= Find_Absolute_Max(v, size);

  for (int i = 0; i < size; i++) {

    num_t x;
    x = v[i];

    x /= scale_value;

    if ((x > bound_upper) || (x < bound_lower)) {
      exceeds_bound++;
    }

    x = (x > bound_upper) ? bound_upper : x;
    x = (x < bound_lower) ? bound_lower : x;

    if (resolution > (num_t)0.0) {
      x /= resolution;
      x = (num_t)RPU_ROUNDFUNF(x);
      x *= resolution;
    }
    vq[i] = x * post_scale;
  }

  return exceeds_bound;
}

void transpose(num_t *x_trans, num_t *x, int size, int m_batch) {

  for (int i = 0; i < size; i++) {
    for (int j = 0; j < m_batch; j++) {
      x_trans[j + i * m_batch] = x[i + j * size];
    }
  }
}

class IOManagerTestFixture : public ::testing::TestWithParam<bool> {
public:
  void SetUp() {

    x_size = 530;
    d_size = 490;
    m_batch = 100;

    context = &context_container;
    iom = RPU::make_unique<InputOutputManager<num_t>>(context, x_size, d_size);

    num_t bound = 0.8;
    num_t res = 0.01;

    io.inp_bound = bound;
    io.out_bound = bound;
    io.inp_res = res;
    io.out_res = res;
    io.out_noise = 0;
    io.inp_sto_round = false;
    io.max_bm_factor = 100000;

    rx = new num_t[x_size * m_batch];
    x1 = new num_t[x_size * m_batch];
    x2 = new num_t[x_size * m_batch];
    rd = new num_t[d_size * m_batch];
    rx_trans = new num_t[x_size * m_batch];
    rd_trans = new num_t[d_size * m_batch];

    rx_res = new num_t[x_size * m_batch];
    rd_res = new num_t[d_size * m_batch];

    rxq = new num_t[x_size * m_batch];
    rdq = new num_t[d_size * m_batch];
    rxq_trans = new num_t[x_size * m_batch];
    rdq_trans = new num_t[d_size * m_batch];

    W = new num_t[x_size * d_size];

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

    std::uniform_real_distribution<float> udist2(-0.2, 1.2);
    auto urnd2 = std::bind(udist2, generator);
    for (int j = 0; j < d_size * x_size; j++) {
      W[j] = (num_t)urnd2();
    }

    transpose(rd_trans, rd, d_size, m_batch);
    transpose(rx_trans, rx, x_size, m_batch);

    curx = RPU::make_unique<CudaArray<num_t>>(context, x_size * m_batch, rx);
    curd = RPU::make_unique<CudaArray<num_t>>(context, d_size * m_batch, rd);

    curx_trans = RPU::make_unique<CudaArray<num_t>>(context, x_size * m_batch, rx_trans);
    curd_trans = RPU::make_unique<CudaArray<num_t>>(context, d_size * m_batch, rd_trans);

    dev_W = RPU::make_unique<CudaArray<num_t>>(context, d_size * x_size, W);
  }

  void TearDown() {
    delete[] rx;
    delete[] x1;
    delete[] x2;
    delete[] rd;
    delete[] rxq;
    delete[] rdq;

    delete[] rx_res;
    delete[] rd_res;

    delete[] W;
  }
  CudaContext context_container{-1, false};
  CudaContextPtr context;

  int x_size;
  int d_size;
  int m_batch;
  std::unique_ptr<InputOutputManager<num_t>> iom;
  std::unique_ptr<CudaArray<num_t>> curx, curx_trans;
  std::unique_ptr<CudaArray<num_t>> curd, curd_trans, dev_W;

  IOMetaParameter<num_t> io;
  num_t *rx, *x1, *x2, *rd, *rx_trans, *rd_trans;
  num_t *rxq, *rdq, *rxq_trans, *rdq_trans;
  num_t *rx_res, *rd_res, *W;
};

INSTANTIATE_TEST_CASE_P(NoiseManagement, IOManagerTestFixture, ::testing::Bool());

TEST_P(IOManagerTestFixture, InputManagement) {

  CUDA_TIMING_INIT;

  // make reference
  discretize(rxq, rx, x_size, io.inp_bound, io.inp_res, GetParam(), 1.0, 1.0);
  io.noise_management = GetParam() ? NoiseManagementType::AbsMax : NoiseManagementType::None;
  io.bound_management = BoundManagementType::None;
  iom->initWithInput(curx->getDataConst(), io, x_size, 1, false);

  CUDA_TIMING_START((this->context));
  iom->applyToInput(curx->getDataConst());
  if (GetParam()) {
    CUDA_TIMING_STOP((this->context), "IM [NM]");
  } else {
    CUDA_TIMING_STOP((this->context), "IM [no NM]");
  }

  iom->copyTempArrayToHost(rx_res);

  iom->releaseBuffer();
  // SR and noise is off

  for (int i = 0; i < x_size; i++) {
    ASSERT_FLOAT_EQ(rxq[i], rx_res[i]);
  }
  CUDA_TIMING_DESTROY;
}
TEST_P(IOManagerTestFixture, InputManagementBatch) {

  CUDA_TIMING_INIT;

  // make reference
  for (int i = 0; i < m_batch * x_size; i += x_size) {
    discretize(rxq + i, rx + i, x_size, io.inp_bound, io.inp_res, GetParam(), 1.0, 1.0);
  }
  io.noise_management = GetParam() ? NoiseManagementType::AbsMax : NoiseManagementType::None;
  io.bound_management = BoundManagementType::None;
  iom->initWithInput(curx->getDataConst(), io, x_size, m_batch, false);

  CUDA_TIMING_START((this->context));
  iom->applyToInput(curx->getDataConst());
  if (GetParam()) {
    CUDA_TIMING_STOP((this->context), "IM Batched [NM]");
  } else {
    CUDA_TIMING_STOP((this->context), "IM Batched [no NM]");
  }

  iom->copyTempArrayToHost(rx_res);

  for (int i = 0; i < x_size * m_batch; i++) {
    ASSERT_FLOAT_EQ(rxq[i], rx_res[i]);
  }

  // trans
  transpose(rxq_trans, rxq, x_size, m_batch);
  io.noise_management = GetParam() ? NoiseManagementType::AbsMax : NoiseManagementType::None;
  io.bound_management = BoundManagementType::None;
  iom->initWithInput(curx_trans->getDataConst(), io, x_size, m_batch, true);

  CUDA_TIMING_START((this->context));
  iom->applyToInput(curx_trans->getDataConst());
  if (GetParam()) {
    CUDA_TIMING_STOP((this->context), "IM Batched Trans [NM]");
  } else {
    CUDA_TIMING_STOP((this->context), "IM Batched Trans [no NM]");
  }

  iom->copyTempArrayToHost(rx_res);
  iom->releaseBuffer();
  for (int i = 0; i < x_size * m_batch; i++) {
    ASSERT_FLOAT_EQ(rxq_trans[i], rx_res[i]);
  }
  CUDA_TIMING_DESTROY;
}

TEST_P(IOManagerTestFixture, OutputManagement) {

  CUDA_TIMING_INIT;

  // make reference
  num_t scale_value = 1.0;
  if (GetParam())
    scale_value = Find_Absolute_Max(rx, x_size);

  discretize(rdq, rd, d_size, io.out_bound, io.out_res, false, 1.0, scale_value);

  io.noise_management = GetParam() ? NoiseManagementType::AbsMax : NoiseManagementType::None;
  io.bound_management = BoundManagementType::None;

  iom->initWithInput(curx->getDataConst(), io, x_size, 1, false);

  CUDA_TIMING_START((this->context));
  iom->applyToOutputInPlace(curd->getData(), false);
  iom->releaseBuffer();
  if (GetParam()) {
    CUDA_TIMING_STOP((this->context), "OM [NM]");
  } else {
    CUDA_TIMING_STOP((this->context), "OM [no NM]");
  }

  curd->copyTo(rd_res);

  for (int i = 0; i < d_size; i++) {
    ASSERT_FLOAT_EQ(rdq[i], rd_res[i]);
  }
  CUDA_TIMING_DESTROY;
}

TEST_P(IOManagerTestFixture, OutputManagementBatch) {

  // make reference
  num_t scale_value = 1.0;
  for (int i = 0; i < m_batch; i++) {
    if (GetParam()) {
      scale_value = Find_Absolute_Max(rx + i * x_size, x_size);
    }
    discretize(
        rdq + i * d_size, rd + i * d_size, d_size, io.out_bound, io.out_res, false, 1.0,
        scale_value);
  }
  io.noise_management = GetParam() ? NoiseManagementType::AbsMax : NoiseManagementType::None;
  io.bound_management = BoundManagementType::None;

  iom->initWithInput(curx->getDataConst(), io, x_size, m_batch, false);
  iom->applyToInput(curx->getDataConst()); // sets the scale values
  CUDA_TIMING_INIT;

  CUDA_TIMING_START((this->context));
  iom->applyToOutputInPlace(curd->getData(), false);
  iom->releaseBuffer();

  if (GetParam()) {
    CUDA_TIMING_STOP((this->context), "OM Batched [NM]");
  } else {
    CUDA_TIMING_STOP((this->context), "OM Batched [no NM]");
  }
  context->synchronize();

  curd->copyTo(rd_res);

  for (int i = 0; i < d_size * m_batch; i++) {
    ASSERT_FLOAT_EQ(rdq[i], rd_res[i]);
  }
  // trans
  transpose(rdq_trans, rdq, d_size, m_batch);
  io.noise_management = GetParam() ? NoiseManagementType::AbsMax : NoiseManagementType::None;
  io.bound_management = BoundManagementType::None;
  iom->initWithInput(curx_trans->getDataConst(), io, x_size, m_batch, true);
  iom->applyToInput(curx_trans->getDataConst()); // sets the scale values

  CUDA_TIMING_START((this->context));
  iom->applyToOutputInPlace(curd_trans->getData(), true);
  iom->releaseBuffer();
  if (GetParam()) {
    CUDA_TIMING_STOP((this->context), "OM Batched Trans [NM]");
  } else {
    CUDA_TIMING_STOP((this->context), "OM Batched Trans [no NM]");
  }
  context->synchronize();
  curd_trans->copyTo(rd_res);

  for (int i = 0; i < d_size * m_batch; i++) {
    ASSERT_FLOAT_EQ(rdq_trans[i], rd_res[i]);
  }

  CUDA_TIMING_DESTROY;
}

// define the tests
TEST_P(IOManagerTestFixture, InputBoundManagementBatch) {

  CUDA_TIMING_INIT;

  CudaArray<num_t> cubu(context, d_size * m_batch);
  cubu.setConst((num_t)2.0 * io.out_bound);
  context->synchronize();

  // make reference
  for (int i = 0; i < m_batch * x_size; i += x_size) {
    discretize(rxq + i, rx + i, x_size, io.inp_bound, io.inp_res, GetParam(), 2.0, 1.0);
  }
  io.noise_management = GetParam() ? NoiseManagementType::AbsMax : NoiseManagementType::None;
  io.bound_management = BoundManagementType::Iterative;

  iom->initWithInput(curx->getDataConst(), io, x_size, m_batch, false);
  iom->applyToInput(curx->getDataConst()); // first round

  bool success = iom->applyToOutputInPlace(cubu.getData(), false); // first round
  ASSERT_EQ(success, false);

  CUDA_TIMING_START((this->context));
  iom->applyToInput(curx->getDataConst()); // second round (2.0 scaling)

  if (GetParam()) {
    CUDA_TIMING_STOP((this->context), "IBM Batched [NM]");
  } else {
    CUDA_TIMING_STOP((this->context), "IBM Batched [no NM]");
  }

  iom->copyTempArrayToHost(rx_res);
  iom->releaseBuffer();

  for (int i = 0; i < x_size * m_batch; i++) {
    ASSERT_FLOAT_EQ(rxq[i], rx_res[i]);
  }

  // trans
  cubu.setConst((num_t)2.0 * io.out_bound);
  context->synchronize();

  transpose(rxq_trans, rxq, x_size, m_batch);

  io.noise_management = GetParam() ? NoiseManagementType::AbsMax : NoiseManagementType::None;
  io.bound_management = BoundManagementType::Iterative;
  iom->initWithInput(curx_trans->getDataConst(), io, x_size, m_batch, true);
  iom->applyToInput(curx_trans->getDataConst());             // first round
  success = iom->applyToOutputInPlace(cubu.getData(), true); // first round
  ASSERT_EQ(success, false);

  CUDA_TIMING_START((this->context));
  iom->applyToInput(curx_trans->getDataConst()); // second round
  if (GetParam()) {
    CUDA_TIMING_STOP((this->context), "IBM Batched Trans [NM]");
  } else {
    CUDA_TIMING_STOP((this->context), "IBM Batched Trans [no NM]");
  }

  iom->copyTempArrayToHost(rx_res);
  iom->releaseBuffer();

  for (int i = 0; i < x_size * m_batch; i++) {
    ASSERT_FLOAT_EQ(rxq_trans[i], rx_res[i]);
  }
  CUDA_TIMING_DESTROY;
}

// define the tests
TEST_P(IOManagerTestFixture, OutputBoundManagementBatch) {

  CUDA_TIMING_INIT;
  io.out_bound = 2.0;

  int *exceeding = new int[m_batch];
  int selected = MIN(3, m_batch - 1); // batch selected to cross threshold

  rd[selected * d_size] = 3.0;
  curd->assign(rd);
  context->synchronize();

  io.noise_management = GetParam() ? NoiseManagementType::AbsMax : NoiseManagementType::None;
  io.bound_management = BoundManagementType::Iterative;
  iom->initWithInput(curx->getDataConst(), io, x_size, m_batch, false);
  iom->applyToInput(curx->getDataConst());

  CUDA_TIMING_START((this->context));
  bool success = iom->applyToOutputInPlace(curd->getData(), false); // first round

  if (GetParam()) {
    CUDA_TIMING_STOP((this->context), "OBM Batched [NM]");
  } else {
    CUDA_TIMING_STOP((this->context), "OBM Batched [no NM]");
  }

  ASSERT_EQ(success, false);
  iom->copyExceededArrayToHost(exceeding);
  if (m_batch > 1) {
    for (int i = 0; i < m_batch; i++) {
      if (i == selected) {
        ASSERT_EQ(exceeding[i], 1);
      } else {
        ASSERT_EQ(exceeding[i], 0);
      }
    }
  }
  iom->releaseBuffer();

  // trans
  transpose(rd_trans, rd, d_size, m_batch);
  curd_trans->assign(rd_trans);

  io.noise_management = GetParam() ? NoiseManagementType::AbsMax : NoiseManagementType::None;
  io.bound_management = BoundManagementType::Iterative;

  iom->initWithInput(curx_trans->getDataConst(), io, x_size, m_batch, true);
  iom->applyToInput(curx_trans->getDataConst());

  CUDA_TIMING_START((this->context));
  success = iom->applyToOutputInPlace(curd_trans->getData(), true); // first round

  if (GetParam()) {
    CUDA_TIMING_STOP((this->context), "OBM Batched Trans [NM]");
  } else {
    CUDA_TIMING_STOP((this->context), "OBM Batched Trans [no NM]");
  }

  ASSERT_EQ(success, false);
  iom->copyExceededArrayToHost(exceeding);
  iom->releaseBuffer();

  if (m_batch > 1) {
    for (int i = 0; i < m_batch; i++) {
      if (i == selected) {
        ASSERT_EQ(exceeding[i], 1);
      } else {
        ASSERT_EQ(exceeding[i], 0);
      }
    }
  }
}

TEST_P(IOManagerTestFixture, InputOutputBoundManagementBatch) {

  CUDA_TIMING_INIT;
  io.inp_res = -1;
  io.out_res = -1;
  io.inp_bound = 1.0;
  io.out_bound = 0.2;
  io.inp_sto_round = false;

  auto D_buffer = CudaArray<num_t>(this->context, d_size * m_batch);

  // ASSERT_EQ(d_size==x_size,true); // otherwise simplification below will fail

  for (int i = 0; i < x_size * m_batch; i++) {
    rx[i] = (float)i / (x_size * m_batch) * 2.2 - 1.1;
  };
  rx[1] = 100;

  curx->assign(rx);

  context->synchronize();

  // first make reference
  bool x_trans = false;
  RPU::math::gemm<num_t>(
      context,
      false, // d_trans
      x_trans, d_size,
      m_batch, // M
      x_size,  // K
      (num_t)1.0, dev_W->getData(),
      d_size, // col major
      curx->getDataConst(), (x_trans) ? m_batch : x_size, (num_t)0.0, D_buffer.getData(), d_size);

  D_buffer.copyTo(rd_res);

  CUDA_TIMING_START((this->context));

  io.noise_management = GetParam() ? NoiseManagementType::AbsMax : NoiseManagementType::None;
  io.bound_management = BoundManagementType::Iterative;

  iom->initWithInput(curx->getDataConst(), io, x_size, m_batch, false);

  bool success = false;
  num_t *temp_x = iom->getInBuffer();
  num_t *temp_d = iom->getOutBuffer();

  while (!success) {
    iom->applyToInput(curx->getDataConst());

    x_trans = false;
    RPU::math::gemm<num_t>(
        context,
        false, // d_trans
        x_trans, d_size,
        m_batch, // M
        x_size,  // K
        (num_t)1.0, dev_W->getData(),
        d_size, // col major
        temp_x, (x_trans) ? m_batch : x_size, (num_t)0.0, temp_d, d_size);

    success = iom->applyToOutput(curd->getData(), false); // first round
  }

  if (GetParam()) {
    CUDA_TIMING_STOP((this->context), "IBM + OBM Batched [NM]");
  } else {
    CUDA_TIMING_STOP((this->context), "IBM + OBM Batched [no NM]");
  }

  iom->releaseBuffer();
  curd->copyTo(rd);
  for (int i = 0; i < d_size * m_batch; i++) {

    ASSERT_NEAR(rd_res[i], rd[i], 1e-3);
  }

  // trans
  transpose(rx_trans, rx, x_size, m_batch);
  curx->assign(rx_trans);
  context->synchronize();

  io.noise_management = GetParam() ? NoiseManagementType::AbsMax : NoiseManagementType::None;
  io.bound_management = BoundManagementType::Iterative;

  iom->initWithInput(curx->getDataConst(), io, x_size, m_batch, true);
  temp_x = iom->getInBuffer();
  temp_d = iom->getOutBuffer();

  CUDA_TIMING_START((this->context));
  success = false;
  while (!success) {
    iom->applyToInput(curx->getDataConst());

    x_trans = true;
    RPU::math::gemm<num_t>(
        context, !x_trans, true, m_batch, d_size, x_size, (num_t)1.0, temp_x,
        (x_trans) ? m_batch : x_size, dev_W->getData(), d_size, (num_t)0.0, temp_d, m_batch);

    success = iom->applyToOutput(curd->getData(), true); // first round
  }

  if (GetParam()) {
    CUDA_TIMING_STOP((this->context), "IBM + OBM Batched Trans [NM]");
  } else {
    CUDA_TIMING_STOP((this->context), "IBM + OBM Batched Trans [no NM]");
  }
  iom->releaseBuffer();

  curd->copyTo(rd);
  transpose(rd_trans, rd_res, d_size, m_batch);
  for (int i = 0; i < d_size * m_batch; i++) {
    ASSERT_NEAR(rd_trans[i], rd[i], 1e-3);
  }
}

} // namespace

int main(int argc, char **argv) {
  resetCuda();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
