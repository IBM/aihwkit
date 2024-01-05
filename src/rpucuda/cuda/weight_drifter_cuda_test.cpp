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

#include "cuda_math_util.h"
#include "cuda_util.h"
#include "weight_drifter.h"
#include "weight_drifter_cuda.h"
#include "gtest/gtest.h"
#include <chrono>
#include <memory>
#include <random>

#define TOLERANCE 1e-5

namespace {

using namespace RPU;

void transpose(num_t *w, int dim0, int dim1) {

  num_t *w_trans = new num_t[dim0 * dim1];

  for (int i = 0; i < dim0; i++) {
    for (int j = 0; j < dim1; j++) {
      w_trans[j + i * dim1] = w[i + j * dim0];
    }
  }
  for (int k = 0; k < dim0 * dim1; k++) {
    w[k] = w_trans[k];
  }
  delete[] w_trans;
}

class WeightDrifterTestFixture : public ::testing::Test {
public:
  void SetUp() {

    x_size = 10;
    d_size = 12;
    size = x_size * d_size;
    time_since_last_call = 3.0;
    ntimes = 10;
    context = &context_container;

    context->setRandomSeed(0);
    dev_w = RPU::make_unique<CudaArray<num_t>>(context, size);
    auto dev_w_float = RPU::make_unique<CudaArray<float>>(context, size);
    dev_w2 = RPU::make_unique<CudaArray<num_t>>(context, size);

    context->randNormal(dev_w_float->getData(), size);
    RPU::math::elemcopy(context, dev_w->getData(), size, dev_w_float->getDataConst());

    dev_nu = RPU::make_unique<CudaArray<num_t>>(context, size);

    par.nu_dtod = 0.1; // ensure larger than 0 to test nu construction and simpledrift
    par.nu_std = 0.0;
    par.w_read_std = 0.0;

    wdrifter = RPU::make_unique<WeightDrifter<num_t>>(size, par, &rw_rng);
    wdrifter_cuda = RPU::make_unique<WeightDrifterCuda<num_t>>(context, *wdrifter, x_size, d_size);

    context->synchronize();
    w = new num_t[size];
    w_orig = new num_t[size];
    w2 = new num_t[size];
    nu = new num_t[size];

    dev_w->copyTo(w2);
    dev_w->copyTo(w);
    dev_w->copyTo(w_orig);
    dev_w->assignTranspose(w, d_size, x_size);
    dev_w2->assignTranspose(w, d_size, x_size);

    context->synchronize();
  }

  void TearDown() {
    delete[] w;
    delete[] w2;
    delete[] w_orig;
    delete[] nu;
  }

  CudaContext context_container{-1, false};
  CudaContextPtr context;
  RNG<num_t> rng{0};
  int size, ntimes, x_size, d_size;
  num_t time_since_last_call;
  RealWorldRNG<num_t> rw_rng{0};
  std::unique_ptr<CudaArray<num_t>> dev_w, dev_w2, dev_nu;
  std::unique_ptr<WeightDrifterCuda<num_t>> wdrifter_cuda;
  std::unique_ptr<WeightDrifter<num_t>> wdrifter;
  DriftParameter<num_t> par;
  num_t *w, *w2, *w_orig, *nu;
};

TEST_F(WeightDrifterTestFixture, Construction) {

  auto wdrifter_simple = RPU::make_unique<WeightDrifter<num_t>>(size, par);
  ASSERT_TRUE(wdrifter_simple->getNu() == nullptr);

  ASSERT_TRUE(wdrifter->getNu() != nullptr);

  dev_nu->assignFromDevice(wdrifter_cuda->getNu());
  context->synchronize();
  dev_nu->copyTo(nu);
  context->synchronize();

  for (int k = 0; k < size; k++) {
    int i = k % x_size; // x index: not transposed: first x_size
    int j = k / x_size; // d index
    ASSERT_FLOAT_EQ(nu[j + d_size * i], wdrifter->getNu()[k]);
  }
}

TEST_F(WeightDrifterTestFixture, ApplySimple) {

  CUDA_TIMING_INIT;

  par.w_read_std = 0.0;
  par.nu = 0.05;
  par.nu_std = 0.0;
  par.nu_dtod = 0.0;
  par.wg_ratio = 1.0;
  par.w_offset = 0.0;
  par.g_offset = 0.0;

  wdrifter = RPU::make_unique<WeightDrifter<num_t>>(size, par, &rw_rng);
  wdrifter_cuda = RPU::make_unique<WeightDrifterCuda<num_t>>(context, *wdrifter, x_size, d_size);

  wdrifter_cuda->apply(dev_w->getData(), time_since_last_call);
  context->synchronize();

  CUDA_TIMING_START(this->context);
  for (int i = 0; i < ntimes - 1; i++) {
    wdrifter_cuda->apply(dev_w->getData(), time_since_last_call);
  }
  CUDA_TIMING_STOP(this->context, "Apply drift repeatedly");

  dev_w->copyTo(w);
  transpose(w, d_size, x_size);

  wdrifter->apply(w2, time_since_last_call, rng);
  for (int i = 0; i < ntimes - 1; i++) {
    wdrifter->apply(w2, time_since_last_call, rng);
  }

  for (int i = 0; i < size; i++) {
    num_t w_ref =
        w_orig[i] * (num_t)(powf(((num_t)ntimes) * time_since_last_call / par.t0, -par.nu));
    ASSERT_NEAR(w_ref, w2[i], TOLERANCE);
    ASSERT_NEAR(w[i], w2[i], TOLERANCE);
  }

  CUDA_TIMING_DESTROY;
}

TEST_F(WeightDrifterTestFixture, ApplyChange) {

  dev_w->setConst(0.5);
  context->synchronize();
  for (int i = 0; i < size; i++) {
    w2[i] = 0.5;
  }

  wdrifter_cuda->apply(dev_w->getData(), time_since_last_call);
  context->synchronize();
  dev_w->copyTo(w);
  context->synchronize();
  transpose(w, d_size, x_size);

  wdrifter->apply(w2, time_since_last_call, rng);
  for (int i = 0; i < size; i++) {
    ASSERT_FLOAT_EQ(w[i], w2[i]);
  }
  // num_t w_ref = 0.5 * (powf(time_since_last_call / par.t0, -wdrifter->getNu()[0]));
  // std::cout << "w cpu :  " << w2[0] << ",  w gpu : " << w[0] << ", w ref : " << w_ref <<
  // std::endl;

  // set everything to one
  num_t c = 0.234567;
  dev_w->setConst(c);
  context->synchronize();
  for (int i = 0; i < size; i++) {
    w2[i] = c;
  }

  wdrifter_cuda->apply(dev_w->getData(), time_since_last_call);
  context->synchronize();
  dev_w->copyTo(w);
  context->synchronize();
  transpose(w, d_size, x_size);

  wdrifter->apply(w2, time_since_last_call, rng);

  // std::cout << "w cpu :  " << w2[0] << ",  w gpu : " << w[0] << ", w ref : " << c << std::endl;

  for (int i = 0; i < size; i++) {
    ASSERT_FLOAT_EQ(w[i], w2[i]);
  }

  // apply drift again
  wdrifter_cuda->apply(dev_w->getData(), time_since_last_call);
  context->synchronize();
  dev_w->copyTo(w);
  context->synchronize();
  transpose(w, d_size, x_size);

  wdrifter->apply(w2, time_since_last_call, rng);

  // w_ref = c * (powf(time_since_last_call / par.t0, -wdrifter->getNu()[0]));
  // std::cout << "w cpu :  " << w2[0] << ",  w gpu : " << w[0] << ", w ref : " << w_ref <<
  // std::endl;

  for (int i = 0; i < size; i++) {
    ASSERT_FLOAT_EQ(w[i], w2[i]);
  }
}

} // namespace

int main(int argc, char **argv) {
  resetCuda();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
