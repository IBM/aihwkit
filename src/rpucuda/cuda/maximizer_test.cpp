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
#include "maximizer.h"
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

void transpose(num_t *x_trans, num_t *x, int size, int m_batch) {

  for (int i = 0; i < size; i++) {
    for (int j = 0; j < m_batch; j++) {
      x_trans[j + i * m_batch] = x[i + j * size];
    }
  }
};

class MaximizerTestFixture : public ::testing::TestWithParam<bool> {
public:
  void SetUp() {

    c = &context_container;
    x_size = 1000;
    m_batch = 1025; // for the batched versions

    x1 = new num_t[x_size * m_batch];
    rx = new num_t[x_size * m_batch];

    max_values = new num_t[this->m_batch];
    max_values2 = new num_t[this->m_batch];

    unsigned int seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator{seed};
    std::uniform_real_distribution<float> udist(-1.2, 1.2);
    auto urnd = std::bind(udist, generator);

    // just assign some numbers from the weigt matrix
    for (int i = 0; i < x_size * m_batch; i++) {
      rx[i] = (num_t)urnd();
    }
    rx[0] = -2.6;
  }

  void TearDown() {
    delete[] x1;
    delete[] rx;
    delete[] max_values;
    delete[] max_values2;
  }

  int x_size;
  int m_batch;
  CudaContext context_container{-1, false};
  CudaContextPtr c;
  num_t *max_values, *max_values2;
  num_t *x1, *rx;
};

INSTANTIATE_TEST_CASE_P(Transpose, MaximizerTestFixture, ::testing::Bool());

// BATCH
TEST_P(MaximizerTestFixture, kernelMaxBatch) {

  int m_b = 100;

  for (int i = 0; i < m_b; i++) {
    max_values[i] = Find_Absolute_Max<num_t>(this->rx + i * this->x_size, this->x_size);
  }

  if (GetParam()) {
    transpose(this->x1, this->rx, this->x_size, m_b);
    RPU::test_helper::debugMaxBatched(
        this->x1, this->x_size, m_b,
        true, // trans
        this->max_values2);
  } else {
    RPU::test_helper::debugMaxBatched(
        this->rx, this->x_size, m_b,
        false, // trans
        this->max_values2);
  }

  for (int i = 0; i < m_b; i++) {
    EXPECT_FLOAT_EQ(this->max_values[i], this->max_values2[i]);
  }
}

TEST_P(MaximizerTestFixture, kernelMaxBatch_LargeBatch) {

  int m_b = this->m_batch;

  for (int i = 0; i < m_b; i++) {
    max_values[i] = Find_Absolute_Max<num_t>(this->rx + i * this->x_size, this->x_size);
  }

  if (GetParam()) {
    transpose(this->x1, this->rx, this->x_size, m_b);
    RPU::test_helper::debugMaxBatched(
        this->x1, this->x_size, m_b,
        true, // trans
        this->max_values2);
  } else {
    RPU::test_helper::debugMaxBatched(
        this->rx, this->x_size, m_b,
        false, // trans
        this->max_values2);
  }

  for (int i = 0; i < this->m_batch; i++) {
    EXPECT_FLOAT_EQ(this->max_values[i], this->max_values2[i]);
  }
}

TEST_P(MaximizerTestFixture, MaximizerSingle) {

  CUDA_TIMING_INIT;

  if (GetParam()) {
    max_values[0] = Find_Absolute_Max<num_t>(this->rx, this->x_size);
  } else {
    max_values[0] = Find_Max<num_t>(this->rx, this->x_size);
  }
  Maximizer<num_t> mxm(c, this->x_size, GetParam());
  CudaArray<num_t> dev_x(c, this->x_size, this->rx);
  c->synchronize();
  mxm.compute(dev_x.getDataConst(), 1, false); // to init batch buffers etc
  c->synchronize();

  CUDA_TIMING_START(c);
  mxm.compute(dev_x.getDataConst(), 1, false);
  if (GetParam()) {
    CUDA_TIMING_STOP(c, "AbsMax Single");
  } else {
    CUDA_TIMING_STOP(c, "Max Single");
  }

  c->synchronize();
  mxm.copyMaxValuesToHost(this->max_values2);
  c->synchronize();

  EXPECT_FLOAT_EQ(this->max_values[0], this->max_values2[0]);
  std::cout << "max value : " << this->max_values[0] << std::endl;
  CUDA_TIMING_DESTROY;
}

TEST_P(MaximizerTestFixture, MaximizerAbsBatch) {

  CUDA_TIMING_INIT;

  for (int i = 0; i < this->m_batch; i++) {
    max_values[i] = Find_Absolute_Max<num_t>(this->rx + i * this->x_size, this->x_size);
  }
  num_t *temp;
  if (GetParam()) {
    transpose(this->x1, this->rx, this->x_size, this->m_batch);
    temp = this->x1;
  } else {
    temp = this->rx;
  }

  Maximizer<num_t> mxm(c, this->x_size, true);
  CudaArray<num_t> dev_x(c, this->x_size * this->m_batch, temp);
  c->synchronize();
  mxm.compute(dev_x.getDataConst(), this->m_batch, GetParam()); // to init batch buffers etc
  c->synchronize();

  CUDA_TIMING_START(c);
  mxm.compute(dev_x.getDataConst(), this->m_batch, GetParam());
  if (GetParam()) {
    CUDA_TIMING_STOP(c, "Max Abs Batched [trans]");
  } else {
    CUDA_TIMING_STOP(c, "Max Abs Batched");
  }

  c->synchronize();
  mxm.copyMaxValuesToHost(this->max_values2);
  c->synchronize();

  for (int i = 0; i < this->m_batch; i++) {
    EXPECT_FLOAT_EQ(this->max_values[i], this->max_values2[i]);
  }

  CUDA_TIMING_DESTROY;
}

TEST_P(MaximizerTestFixture, MaximizerBatch) {

  CUDA_TIMING_INIT;

  for (int i = 0; i < this->m_batch; i++) {
    max_values[i] = Find_Max<num_t>(this->rx + i * this->x_size, this->x_size);
  }
  num_t *temp;
  if (GetParam()) {
    transpose(this->x1, this->rx, this->x_size, this->m_batch);
    temp = this->x1;
  } else {
    temp = this->rx;
  }

  Maximizer<num_t> mxm(c, this->x_size, false);
  CudaArray<num_t> dev_x(c, this->x_size * this->m_batch, temp);
  c->synchronize();
  mxm.compute(dev_x.getDataConst(), this->m_batch, GetParam()); // to init batch buffers etc
  c->synchronize();

  CUDA_TIMING_START(c);
  mxm.compute(dev_x.getDataConst(), this->m_batch, GetParam());
  if (GetParam()) {
    CUDA_TIMING_STOP(c, "Max Batched [trans]");
  } else {
    CUDA_TIMING_STOP(c, "Max Batched");
  }

  c->synchronize();
  mxm.copyMaxValuesToHost(this->max_values2);
  c->synchronize();

  for (int i = 0; i < this->m_batch; i++) {
    EXPECT_FLOAT_EQ(this->max_values[i], this->max_values2[i]);
  }

  CUDA_TIMING_DESTROY;
}

} // namespace

int main(int argc, char **argv) {
  resetCuda();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
