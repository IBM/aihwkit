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
#include "rpu.h"
#include "rpucuda.h"
#include "gtest/gtest.h"
#include <chrono>
#include <memory>
#include <random>

#define TOLERANCE 1e-3

namespace {

using namespace RPU;

template <typename T> class RPUCudaSimpleTestFixture : public ::testing::Test {
public:
  void SetUp() {

    context = &context_container;
    is_test = true;

    x_size = 99;
    d_size = 56;
    repeats = 3;

    T bmin = (-1. / sqrtf(x_size));
    T bmax = (1. / sqrtf(x_size));

    x1.resize(x_size);
    x2.resize(x_size);
    d1.resize(d_size);
    d2.resize(d_size);
    rx.resize(x_size);
    rd.resize(d_size);

    auto p = SimpleMetaParameter<T>();

    layer_simple = std::unique_ptr<RPUSimple<T>>(p.createRPUArray(x_size, d_size));
    layer_simple->setLearningRate(1);
    layer_simple->setWeightsUniformRandom(bmin, bmax);

    culayer_simple = RPU::make_unique<RPUCudaSimple<T>>(context, *layer_simple);

    // generate random numbers
    unsigned int seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::default_random_engine generator{seed};
    std::uniform_real_distribution<float> udist(-1, 1);
    auto urnd = std::bind(udist, generator);

    for (int i = 0; i < x_size; i++)
      rx[i] = (num_t)urnd();

    for (int j = 0; j < d_size; j++)
      rd[j] = (num_t)urnd();

    x_cuvec = RPU::make_unique<CudaArray<T>>(context, x_size);
    x_vec.resize(x_size);
    x_vec2.resize(x_size);

    d_cuvec = RPU::make_unique<CudaArray<T>>(context, d_size);
    d_vec.resize(d_size);
    d_vec2.resize(d_size);
  }

  CudaContext context_container{-1, false};
  CudaContextPtr context;
  std::unique_ptr<RPUSimple<T>> layer_simple;
  std::unique_ptr<RPUCudaSimple<T>> culayer_simple;
  std::vector<T> x_vec, x_vec2, d_vec, d_vec2, x1, x2, d1, d2, rx, rd;
  std::unique_ptr<CudaArray<T>> x_cuvec;
  std::unique_ptr<CudaArray<T>> d_cuvec;
  int x_size;
  int d_size;
  int repeats;
  bool is_test;
};

class RPUCudaSimpleTestFixtureBatch : public ::testing::TestWithParam<bool> {
public:
  void SetUp() {
    context = &context_container;

    is_test = true;

    x_size = 5;
    d_size = 6;
    repeats = 1;
    m_batch = 3;

    num_t bmin = (-1. / sqrtf(x_size));
    num_t bmax = (1. / sqrtf(x_size));

    x1.resize(x_size * m_batch);
    x2.resize(x_size * m_batch);
    d1.resize(d_size * m_batch);
    d2.resize(d_size * m_batch);
    rx.resize(x_size * m_batch);
    rd.resize(d_size * m_batch);

    auto p = SimpleMetaParameter<num_t>();
    layer_simple = std::unique_ptr<RPUSimple<num_t>>(p.createRPUArray(x_size, d_size));

    layer_simple->setLearningRate(1);
    layer_simple->setWeightsUniformRandom(bmin, bmax);

    culayer_simple = RPU::make_unique<RPUCudaSimple<num_t>>(context, *layer_simple);

    // generate random numbers
    unsigned int seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::default_random_engine generator{seed};
    std::uniform_real_distribution<float> udist(-1., 1.);
    auto urnd = std::bind(udist, generator);

    for (int i = 0; i < x_size * m_batch; i++)
      rx[i] = (num_t)urnd();

    for (int j = 0; j < d_size * m_batch; j++)
      rd[j] = (num_t)urnd();

    x_cuvec = RPU::make_unique<CudaArray<num_t>>(context, x_size * m_batch);
    x_vec.resize(x_size * m_batch);
    x_vec2.resize(x_size * m_batch);

    d_cuvec = RPU::make_unique<CudaArray<num_t>>(context, d_size * m_batch);
    d_vec.resize(d_size * m_batch);
    d_vec2.resize(d_size * m_batch);
  }

  void TearDown() {}

  CudaContext context_container{-1, false};
  CudaContextPtr context;
  std::unique_ptr<RPUSimple<num_t>> layer_simple;
  std::unique_ptr<RPUCudaSimple<num_t>> culayer_simple;
  std::vector<num_t> x_vec, x_vec2, d_vec, d_vec2, x1, x2, d1, d2, rx, rd;
  std::unique_ptr<CudaArray<num_t>> x_cuvec;
  std::unique_ptr<CudaArray<num_t>> d_cuvec;
  int x_size;
  int d_size;
  int repeats;
  int m_batch;
  int is_test;
};

// types
typedef ::testing::Types<num_t> CudaTypes;

TYPED_TEST_CASE(RPUCudaSimpleTestFixture, CudaTypes);
INSTANTIATE_TEST_CASE_P(Batched, RPUCudaSimpleTestFixtureBatch, ::testing::Bool());

// define the tests

TYPED_TEST(RPUCudaSimpleTestFixture, InitSize) {
  ASSERT_EQ(this->culayer_simple->getXSize(), this->x_size);
  ASSERT_EQ(this->culayer_simple->getDSize(), this->d_size);
}

TYPED_TEST(RPUCudaSimpleTestFixture, SetGet) {

  this->culayer_simple->setWeights(this->layer_simple->getWeights()[0]);
  TypeParam **pw = this->culayer_simple->getWeights();
  TypeParam *weights = pw[0];
  TypeParam *w2 = this->layer_simple->getWeights()[0];
  for (int i = 0; i < this->d_size; ++i) {
    for (int j = 0; j < this->x_size; ++j) {
      int k = j + this->x_size * i;
      ASSERT_FLOAT_EQ(weights[k], w2[k]);
    }
  }
}

TYPED_TEST(RPUCudaSimpleTestFixture, CopyConstruct) {
  RPUCudaSimple<TypeParam> culayer_simple1(this->context, *this->layer_simple);
  TypeParam **pw = culayer_simple1.getWeights();
  TypeParam *weights = pw[0];
  TypeParam *w2 = this->layer_simple->getWeights()[0];
  for (int i = 0; i < this->d_size; ++i) {
    for (int j = 0; j < this->x_size; ++j) {
      int k = j + this->x_size * i;
      ASSERT_FLOAT_EQ(weights[k], w2[k]);
    }
  }
}

TYPED_TEST(RPUCudaSimpleTestFixture, ForwardVector) {

  this->culayer_simple->setWeights(this->layer_simple->getWeights()[0]);

  this->x_cuvec->assign(this->rx.data());
  this->x_vec = this->rx;

  for (int i = 0; i < this->repeats; ++i) {
    this->culayer_simple->forward(
        this->x_cuvec->getData(), this->d_cuvec->getData(), false, 1, false, false, this->is_test);
  }
  for (int i = 0; i < this->repeats; ++i) {
    this->layer_simple->forward(
        this->x_vec.data(), this->d_vec.data(), false, 1, false, false, this->is_test);
  }
  this->context->synchronizeDevice();

  this->d_cuvec->copyTo(this->d1.data());
  this->d2 = this->d_vec;

  for (int i = 0; i < this->d_size; ++i) {
    ASSERT_NEAR(this->d1[i], this->d2[i], TOLERANCE);
  }
}

TYPED_TEST(RPUCudaSimpleTestFixture, ForwardVectorBias) {

  this->culayer_simple->setWeights(this->layer_simple->getWeights()[0]);

  this->x_cuvec->assign(this->rx.data());
  this->x_vec = this->rx;

  for (int i = 0; i < this->repeats; ++i) {
    this->culayer_simple->forward(
        this->x_cuvec->getData(), this->d_cuvec->getData(), true, 1, false, false, this->is_test);
  }
  for (int i = 0; i < this->repeats; ++i) {
    this->layer_simple->forward(
        this->x_vec.data(), this->d_vec.data(), true, 1, false, false, this->is_test);
  }
  this->context->synchronizeDevice();

  this->d_cuvec->copyTo(this->d1.data());
  this->d2 = this->d_vec;

  for (int i = 0; i < this->d_size; ++i) {
    ASSERT_NEAR(this->d1[i], this->d2[i], TOLERANCE);
  }

  this->rx[this->x_size - 1] = 1.0;
  this->x_vec = this->rx;

  for (int i = 0; i < this->repeats; ++i) {
    this->layer_simple->forward(
        this->x_vec.data(), this->d_vec.data(), false, 1, false, false, this->is_test);
  }
  this->d1 = this->d_vec;

  for (int i = 0; i < this->d_size; ++i) {
    ASSERT_FLOAT_EQ(this->d1[i], this->d2[i]);
  }
}

TYPED_TEST(RPUCudaSimpleTestFixture, BackwardVector) {

  this->culayer_simple->setWeights(this->layer_simple->getWeights()[0]);

  this->d_cuvec->assign(this->rd.data());
  this->d_vec = this->rd;

  for (int i = 0; i < this->repeats; ++i) {
    this->culayer_simple->backward(this->d_cuvec->getData(), this->x_cuvec->getData(), false, 1);
  }
  for (int i = 0; i < this->repeats; ++i) {
    this->layer_simple->backward(this->d_vec.data(), this->x_vec.data(), false, 1);
  }
  this->context->synchronizeDevice();

  this->x_cuvec->copyTo(this->x1.data());
  this->x2 = this->x_vec;

  for (int j = 0; j < this->x_size; ++j) {
    ASSERT_NEAR(this->x1[j], this->x2[j], TOLERANCE);
  }
}

TYPED_TEST(RPUCudaSimpleTestFixture, BackwardVectorBias) {

  this->culayer_simple->setWeights(this->layer_simple->getWeights()[0]);

  this->d_cuvec->assign(this->rd.data());
  this->d_vec = this->rd;

  this->x_cuvec->assign(this->rx.data());
  this->x_vec = this->rx;
  TypeParam last = this->rx[this->x_size - 1];

  for (int i = 0; i < this->repeats; ++i) {
    this->culayer_simple->backward(this->d_cuvec->getData(), this->x_cuvec->getData(), true);
  }
  for (int i = 0; i < this->repeats; ++i) {
    this->layer_simple->backward(this->d_vec.data(), this->x_vec.data(), true);
  }
  this->context->synchronizeDevice();

  this->x_cuvec->copyTo(this->x1.data());
  this->x2 = this->x_vec;

  for (int j = 0; j < this->x_size - 1; ++j) {
    ASSERT_NEAR(this->x1[j], this->x2[j], TOLERANCE);
  }

  // should not be changed (vector longer by one)
  ASSERT_FLOAT_EQ(last, this->x1[this->x_size - 1]);
  ASSERT_FLOAT_EQ(last, this->x2[this->x_size - 1]);
}

TYPED_TEST(RPUCudaSimpleTestFixture, UpdateVector) {

  this->culayer_simple->setWeights(this->layer_simple->getWeights()[0]);

  this->d_cuvec->assign(this->rd.data());
  this->d_vec = this->rd;

  this->x_cuvec->assign(this->rx.data());
  this->x_vec = this->rx;

  for (int i = 0; i < this->repeats; ++i) {
    this->culayer_simple->update(this->x_cuvec->getData(), this->d_cuvec->getData(), false);
  }
  for (int i = 0; i < this->repeats; ++i) {
    this->layer_simple->update(this->x_vec.data(), this->d_vec.data(), false);
  }

  this->context->synchronizeDevice();

  TypeParam **pw1 = this->culayer_simple->getWeights();
  TypeParam *w1 = pw1[0];

  TypeParam **pw2 = this->layer_simple->getWeights();
  TypeParam *w2 = pw2[0];

  for (int i = 0; i < this->d_size; ++i) {
    for (int j = 0; j < this->x_size; ++j) {
      int k = j + this->x_size * i;
      ASSERT_NEAR(w1[k], w2[k], TOLERANCE);
    }
  }
}

TEST_P(RPUCudaSimpleTestFixtureBatch, UpdateMatrixBatch) {

  this->culayer_simple->setWeights(this->layer_simple->getWeights()[0]);

  this->d_cuvec->assign(this->rd.data());
  this->d_vec = this->rd;

  this->x_cuvec->assign(this->rx.data());
  this->x_vec = this->rx;

  for (int i = 0; i < this->repeats; ++i) {
    this->culayer_simple->update(
        this->x_cuvec->getData(), this->d_cuvec->getData(), false, this->m_batch, GetParam());
  }
  this->context->synchronizeDevice();

  num_t **pw1 = this->culayer_simple->getWeights();
  num_t *w1 = pw1[0];

  for (int i = 0; i < this->repeats; ++i) {
    this->layer_simple->update(
        this->x_vec.data(), this->d_vec.data(), false, this->m_batch, GetParam());
  }
  num_t **pw2 = this->layer_simple->getWeights();
  num_t *w2 = pw2[0];

  for (int i = 0; i < this->d_size; ++i) {
    for (int j = 0; j < this->x_size; ++j) {
      int k = j + this->x_size * i;
      ASSERT_NEAR(w1[k], w2[k], TOLERANCE);
    }
  }
}

TEST_P(RPUCudaSimpleTestFixtureBatch, UpdateMatrixBatchDelayed) {

  auto p = SimpleMetaParameter<num_t>();
  p.use_delayed_update = true;

  this->layer_simple =
      std::unique_ptr<RPUSimple<num_t>>(p.createRPUArray(this->x_size, this->d_size));
  this->layer_simple->setLearningRate(1);
  this->layer_simple->setWeightsUniformRandom(-0.5, 0.5);
  this->culayer_simple = RPU::make_unique<RPUCudaSimple<num_t>>(this->context, *this->layer_simple);

  ASSERT_TRUE(this->culayer_simple->isDelayedUpdate());
  ASSERT_TRUE(this->layer_simple->isDelayedUpdate());

  this->d_cuvec->assign(this->rd.data());
  this->d_vec = this->rd;

  this->x_cuvec->assign(this->rx.data());
  this->x_vec = this->rx;

  num_t **pw2 = this->layer_simple->getWeights();
  num_t *w2 = pw2[0];

  for (int i = 0; i < this->repeats; ++i) {
    this->culayer_simple->update(
        this->x_cuvec->getData(), this->d_cuvec->getData(),
        false, // bias
        this->m_batch, GetParam(), GetParam());
    this->context->synchronizeDevice();

    this->layer_simple->update(
        this->x_vec.data(), this->d_vec.data(), false, this->m_batch, GetParam(), GetParam());

    if (i < this->repeats) {
      num_t **pw1 = this->culayer_simple->getWeights();
      num_t *w1 = pw1[0];

      for (int i = 0; i < this->d_size; ++i) {
        for (int j = 0; j < this->x_size; ++j) {
          int k = j + this->x_size * i;
          ASSERT_NEAR(w1[k], w2[k], TOLERANCE);
        }
      }
    }
  }
  this->culayer_simple->applyDelayedWeights();
  this->context->synchronizeDevice();
  this->layer_simple->applyDelayedWeights();

  // after

  num_t **pw1 = this->culayer_simple->getWeights();
  num_t *w1 = pw1[0];

  pw2 = this->layer_simple->getWeights();
  w2 = pw2[0];

  for (int i = 0; i < this->d_size; ++i) {
    for (int j = 0; j < this->x_size; ++j) {
      int k = j + this->x_size * i;
      ASSERT_NEAR(w1[k], w2[k], TOLERANCE);
    }
  }
}

TEST_P(RPUCudaSimpleTestFixtureBatch, BackwardMatrixBatch) {

  this->culayer_simple->setWeights(this->layer_simple->getWeights()[0]);

  this->d_cuvec->assign(this->rd.data());
  this->d_vec = this->rd;

  for (int i = 0; i < this->repeats; ++i) {
    this->culayer_simple->backward(
        this->d_cuvec->getData(), this->x_cuvec->getData(), false, this->m_batch, GetParam(),
        false);
  }
  for (int i = 0; i < this->repeats; ++i) {
    this->layer_simple->backward(
        this->d_vec.data(), this->x_vec.data(), false, this->m_batch, GetParam(), false);
  }
  this->context->synchronizeDevice();
  this->x_cuvec->copyTo(this->x1.data());
  this->x2 = this->x_vec;

  for (int j = 0; j < this->x_size * this->m_batch; ++j) {
    ASSERT_NEAR(this->x1[j], this->x2[j], TOLERANCE);
  }
}

TEST_P(RPUCudaSimpleTestFixtureBatch, BackwardMatrixBatchTranspose) {

  this->culayer_simple->setWeights(this->layer_simple->getWeights()[0]);

  this->d_cuvec->assign(this->rd.data());
  this->d_vec = this->rd;

  for (int i = 0; i < this->repeats; ++i) {
    this->culayer_simple->backward(
        this->d_cuvec->getData(), this->x_cuvec->getData(), false, this->m_batch, GetParam(), true);
  }
  for (int i = 0; i < this->repeats; ++i) {
    this->layer_simple->backward(
        this->d_vec.data(), this->x_vec.data(), false, this->m_batch, GetParam(), true);
  }
  this->context->synchronizeDevice();
  this->x_cuvec->copyTo(this->x1.data());
  this->x2 = this->x_vec;

  for (int j = 0; j < this->x_size * this->m_batch; ++j) {
    ASSERT_NEAR(this->x1[j], this->x2[j], TOLERANCE);
  }
}

TEST_P(RPUCudaSimpleTestFixtureBatch, BackwardMatrixBatchTransposeBias) {

  this->culayer_simple->setWeights(this->layer_simple->getWeights()[0]);

  this->d_cuvec->assign(this->rd.data());
  this->d_vec = this->rd;

  this->x_cuvec->assign(this->rx.data());
  this->x_vec = this->rx;

  for (int i = 0; i < this->repeats; ++i) {
    this->culayer_simple->backward(
        this->d_cuvec->getData(), this->x_cuvec->getData(), true, this->m_batch, GetParam(), true);
  }
  for (int i = 0; i < this->repeats; ++i) {
    this->layer_simple->backward(
        this->d_vec.data(), this->x_vec.data(), true, this->m_batch, GetParam(), true);
  }
  this->context->synchronizeDevice();
  this->x_cuvec->copyTo(this->x1.data());
  this->x2 = this->x_vec;

  for (int j = 0; j < this->x_size * this->m_batch; ++j) {
    ASSERT_NEAR(this->x1[j], this->x2[j], TOLERANCE);
  }
}

TEST_P(RPUCudaSimpleTestFixtureBatch, BackwardMatrixBatchBias) {

  this->culayer_simple->setWeights(this->layer_simple->getWeights()[0]);

  this->d_cuvec->assign(this->rd.data());
  this->d_vec = this->rd;

  this->x_cuvec->assign(this->rx.data());
  this->x_vec = this->rx;

  for (int i = 0; i < this->repeats; ++i) {
    this->culayer_simple->backward(
        this->d_cuvec->getData(), this->x_cuvec->getData(), true, this->m_batch, GetParam(), false);
  }
  for (int i = 0; i < this->repeats; ++i) {
    this->layer_simple->backward(
        this->d_vec.data(), this->x_vec.data(), true, this->m_batch, GetParam(), false);
  }
  this->context->synchronizeDevice();
  this->x_cuvec->copyTo(this->x1.data());
  this->x2 = this->x_vec;

  for (int j = 0; j < this->x_size * this->m_batch; ++j) {
    ASSERT_NEAR(this->x1[j], this->x2[j], TOLERANCE);
  }
}

TEST_P(RPUCudaSimpleTestFixtureBatch, ForwardMatrixBatch) {

  this->culayer_simple->setWeights(this->layer_simple->getWeights()[0]);

  this->x_cuvec->assign(this->rx.data());
  this->x_vec = this->rx;

  for (int i = 0; i < this->repeats; ++i) {
    this->culayer_simple->forward(
        this->x_cuvec->getData(), this->d_cuvec->getData(), false, this->m_batch, GetParam(), false,
        this->is_test);
  }
  for (int i = 0; i < this->repeats; ++i) {
    this->layer_simple->forward(
        this->x_vec.data(), this->d_vec.data(), false, this->m_batch, GetParam(), false,
        this->is_test);
  }
  this->context->synchronizeDevice();
  this->d_cuvec->copyTo(this->d1.data());
  this->d2 = this->d_vec;

  for (int j = 0; j < this->d_size * this->m_batch; ++j) {
    ASSERT_NEAR(this->d1[j], this->d2[j], TOLERANCE);
  }
}

TEST_P(RPUCudaSimpleTestFixtureBatch, ForwardMatrixBatchTranspose) {

  this->culayer_simple->setWeights(this->layer_simple->getWeights()[0]);

  this->x_cuvec->assign(this->rx.data());
  this->x_vec = this->rx;

  for (int i = 0; i < this->repeats; ++i) {
    this->culayer_simple->forward(
        this->x_cuvec->getData(), this->d_cuvec->getData(), false, this->m_batch, GetParam(), true,
        this->is_test);
  }
  for (int i = 0; i < this->repeats; ++i) {
    this->layer_simple->forward(
        this->x_vec.data(), this->d_vec.data(), false, this->m_batch, GetParam(), true,
        this->is_test);
  }
  this->context->synchronizeDevice();
  this->d_cuvec->copyTo(this->d1.data());
  this->d2 = this->d_vec;

  for (int j = 0; j < this->d_size * this->m_batch; ++j) {
    ASSERT_NEAR(this->d1[j], this->d2[j], TOLERANCE);
  }
}

TEST_P(RPUCudaSimpleTestFixtureBatch, ForwardMatrixBatchTransposeBias) {

  this->culayer_simple->setWeights(this->layer_simple->getWeights()[0]);

  this->x_cuvec->assign(this->rx.data());
  this->x_vec = this->rx;

  this->d_cuvec->assign(this->rd.data()); // for comparison
  this->d_vec = this->rd;

  for (int i = 0; i < this->repeats; ++i) {
    this->culayer_simple->forward(
        this->x_cuvec->getData(), this->d_cuvec->getData(), true, this->m_batch, GetParam(), true,
        this->is_test);
  }
  for (int i = 0; i < this->repeats; ++i) {
    this->layer_simple->forward(
        this->x_vec.data(), this->d_vec.data(), true, this->m_batch, GetParam(), true,
        this->is_test);
  }
  this->context->synchronizeDevice();
  this->d_cuvec->copyTo(this->d1.data());
  this->d2 = this->d_vec;

  for (int j = 0; j < this->d_size * this->m_batch; ++j) {
    ASSERT_NEAR(this->d1[j], this->d2[j], TOLERANCE);
  }
}

TEST_P(RPUCudaSimpleTestFixtureBatch, ForwardMatrixBatchBias) {

  this->culayer_simple->setWeights(this->layer_simple->getWeights()[0]);

  this->x_cuvec->assign(this->rx.data());
  this->x_vec = this->rx;

  this->d_cuvec->assign(this->rd.data());
  this->d_vec = this->rd;

  for (int i = 0; i < this->repeats; ++i) {
    this->culayer_simple->forward(
        this->x_cuvec->getData(), this->d_cuvec->getData(), true, this->m_batch, GetParam(), false,
        this->is_test);
  }
  for (int i = 0; i < this->repeats; ++i) {
    this->layer_simple->forward(
        this->x_vec.data(), this->d_vec.data(), true, this->m_batch, GetParam(), false,
        this->is_test);
  }
  this->context->synchronizeDevice();
  this->d_cuvec->copyTo(this->d1.data());
  this->d2 = this->d_vec;

  for (int j = 0; j < this->d_size * this->m_batch; ++j) {
    ASSERT_NEAR(this->d1[j], this->d2[j], TOLERANCE);
  }

  if (GetParam()) { // x_trans
    for (int j = (this->x_size - 1) * this->m_batch; j < (this->x_size) * this->m_batch; j++) {
      this->rx[j] = 1.0;
    }
  } else {
    for (int j = 0; j < (this->x_size) * this->m_batch; j++) {
      this->x1[j] = 1;
    }
    for (int j = 0; j < (this->x_size - 1) * this->m_batch; j++) {
      int k = (j) / (this->x_size - 1);
      this->x1[j + k] = this->rx[j];
    }
    for (int j = 0; j < (this->x_size) * this->m_batch; j++) {
      this->rx[j] = this->x1[j];
    }
  }
  this->x_vec = this->rx;

  for (int i = 0; i < this->repeats; ++i) {
    this->layer_simple->forward(
        this->x_vec.data(), this->d_vec.data(), false, this->m_batch, GetParam(), false,
        this->is_test);
  }
  this->d1 = this->d_vec;
  for (int j = 0; j < this->d_size * this->m_batch; ++j) {
    ASSERT_FLOAT_EQ(this->d1[j], this->d2[j]);
  }
}

} // namespace

int main(int argc, char **argv) {
  resetCuda();
  ::testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}
