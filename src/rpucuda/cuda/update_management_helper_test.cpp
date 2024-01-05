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

#include "bit_line_maker.h"
#include "cuda.h"
#include "cuda_util.h"
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

template <typename T> class UMHTestFixture : public ::testing::Test {
public:
  void SetUp() {

    m_batch = 10;
    size = 1000;
    K = 23; // max 31
    scaleprob = 1;

    x1 = new T[size * m_batch];
    d1 = new T[size * m_batch];

    unsigned int seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::default_random_engine generator{seed};
    std::uniform_real_distribution<float> udist(-1., 1.);
    auto urnd = std::bind(udist, generator);

    for (int i = 0; i < size * m_batch; i++) {
      x1[i] = (num_t)urnd();
      d1[i] = (num_t)(5 * urnd());
    }
  };

  void TearDown() {
    delete[] x1;
    delete[] d1;
  };

  int m_batch, size, K;
  T scaleprob;
  T *x1, *d1;
};

typedef ::testing::Types<num_t> num_types;

TYPED_TEST_CASE(UMHTestFixture, num_types);

TYPED_TEST(UMHTestFixture, TranslateBatchOrder64UBLM) {
  int result =
      RPU::test_helper::debugKernelTranslateTransFormatToBatchOrder64Format<TypeParam, true>(
          this->x1, this->size, this->m_batch, this->scaleprob, this->K);
  ASSERT_EQ(result, 0);
}

TYPED_TEST(UMHTestFixture, TranslateBatchOrder64) {
  int result =
      RPU::test_helper::debugKernelTranslateTransFormatToBatchOrder64Format<TypeParam, false>(
          this->x1, this->size, this->m_batch, this->scaleprob, this->K);
  ASSERT_EQ(result, 0);
}

TYPED_TEST(UMHTestFixture, computeScaleAndK) {

  CudaContext context_container{-1, false};
  CudaContextPtr c = &context_container;
  UpdateManagementHelper<TypeParam> umh(c, this->size, this->size);
  CudaArray<TypeParam> cu_x(c, this->size * this->m_batch, this->x1);
  CudaArray<TypeParam> cu_d(c, this->size * this->m_batch, this->d1);
  c->synchronize();

  TypeParam dw_min = 0.001;
  TypeParam lr = 0.01;
  bool x_trans = false;
  bool d_trans = false;
  bool BL = 31;

  umh.computeKandScaleValues(
      cu_x.getData(), cu_d.getData(), dw_min, lr,
      true, // update_management,
      true, // update_bl_management,
      this->m_batch, x_trans, d_trans, BL, 1.0, 1.0);

  c->synchronize();
  TypeParam *scale_val = new TypeParam[this->m_batch];
  int *K_val = new int[this->m_batch];

  umh.getScaleValues(scale_val);
  umh.getKValues(K_val);

  c->synchronize();
  // reference:
  for (int i_batch = 0; i_batch < this->m_batch; i_batch++) {

    TypeParam x_abs_max_value =
        Find_Absolute_Max<TypeParam>(this->x1 + this->size * i_batch, this->size);
    TypeParam d_abs_max_value =
        Find_Absolute_Max<TypeParam>(this->d1 + this->size * i_batch, this->size);

    int bl = ceilf(lr * x_abs_max_value * d_abs_max_value / dw_min);
    if (bl > BL) {
      bl = BL;
    }

    TypeParam reg = dw_min;

    TypeParam scale = sqrtf((float)MAX(x_abs_max_value, reg) / (float)MAX(d_abs_max_value, reg));

    EXPECT_FLOAT_EQ(scale, scale_val[i_batch]); // large error ?
    ASSERT_EQ(bl, K_val[i_batch]);
  }

  delete[] scale_val;
  delete[] K_val;
}

} // namespace

int main(int argc, char **argv) {
  resetCuda();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
