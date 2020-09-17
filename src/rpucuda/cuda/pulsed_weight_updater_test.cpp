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

#include "cuda.h"
#include "cuda_util.h"
#include "pulsed_weight_updater.h"
#include "rng.h"
#include "rpucuda_pulsed.h"
#include "test_helper.h"
#include "utility_functions.h"
#include "gtest/gtest.h"
#include <chrono>
#include <memory>
#include <random>

#define TOLERANCE 1e-5

#ifdef RPU_USE_DOUBLE
typedef double num_t;
#else
typedef float num_t;
#endif

namespace {

using namespace RPU;

#define NO_REFERENCE_CHECK                                                                         \
  if ((m_batch != m_batch_test) || (dw_min_std > 0)) {                                             \
    std::cout << BOLD_ON                                                                           \
              << "\n**WARNING: No reference check possible [adjust noise and batch settings]!\n\n" \
              << BOLD_OFF;                                                                         \
    return;                                                                                        \
  }

void transposeCounts(uint32_t *c_trans, uint32_t *c, int size, int m_batch, int nK32) {

  // reorder counts
  for (int i = 0; i < m_batch; i++) {
    for (int j = 0; j < size; j++) {
      int l = i + j * m_batch;
      int s = l % size;
      int b = l / size;
      for (int k = 0; k < nK32; k++) {
        c_trans[s + b * nK32 * size + k * size] = c[j + i * nK32 * size + k * size];
      }
    }
  }
}

int getCombinedCounts(
    uint32_t *x_counts, int x_size, uint32_t *d_counts, int d_size, int K, int x_i, int d_i) {

  int nK32 = K / 32 + 1;
  int icounts = 0;
  for (int k = 0; k < nK32; k++) {
    uint32_t c = x_counts[k * x_size + x_i] & d_counts[k * d_size + d_i];
    icounts += test_helper::getCounts(&c, 0, 31, 1, false);
  }
  int negx = x_counts[x_i] & 1;
  int negd = d_counts[d_i] & 1;

  if ((negx & negd) == 1) {
    icounts--;
  }
  icounts = ((negx ^ negd) == 1) ? -icounts : icounts;

  return icounts;
}

class UpdateKernelWTestFixture : public ::testing::TestWithParam<int> {
public:
  void SetUp() {

    if (1) {
      // large W small batch
      K = 8;
      d_size = 5;
      x_size = 5;
      m_batch = 10;
    } else {
      // small W large batch
      K = 1;
      d_size = 26;
      x_size = 16;
      m_batch = 500;
    }
    m_batch_test =
        MIN(m_batch, 50); // to speed up for testing: will result in failure if batch is too large

    nK32 = K / 32 + 1;
    x_counts = new uint32_t[x_size * nK32 * m_batch];
    d_counts = new uint32_t[d_size * nK32 * m_batch];
    x_counts_trans = new uint32_t[x_size * nK32 * m_batch];
    d_counts_trans = new uint32_t[d_size * nK32 * m_batch];

    weights = new num_t[d_size * x_size];
    ref_w = new num_t[d_size * x_size];
    ref_w_batch = new num_t[d_size * x_size];

    auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::default_random_engine generator{seed};
    std::uniform_int_distribution<unsigned int> idist(0, ((uint32_t)1) << 31);
    auto irnd = std::bind(idist, generator);

    uint32_t lastmask = ~((uint32_t)0xffffffff << ((K + 1) & 0x1f));

    int d_offset = d_size * nK32;
    for (int i = 0; i < d_size * m_batch * nK32; i++) {
      d_counts[i] = irnd();
      if (lastmask > 0) {
        int i_nk32 = (i % d_offset) / d_size;
        if (i_nk32 == nK32 - 1)
          d_counts[i] &= lastmask;
      }
    }
    int x_offset = x_size * nK32;
    for (int j = 0; j < x_size * m_batch * nK32; j++) {
      x_counts[j] = irnd();
      if (lastmask > 0) {
        int i_nk32 = (j % x_offset) / x_size;
        if (i_nk32 == nK32 - 1)
          x_counts[j] &= lastmask;
      }
    }

    transposeCounts(x_counts_trans, x_counts, x_size, m_batch, nK32);
    transposeCounts(d_counts_trans, d_counts, d_size, m_batch, nK32);

    for (int i = 0; i < d_size * x_size; i++) {
      weights[i] = 0.0;
      ref_w_batch[i] = 0.0;
      ref_w[i] = 0.0;
    }

    dw_min = 0.01; // about 1/K
    bound = 0.1 * m_batch;
    dw_min_std = 0.0000; // no noise
    timing = 0.0;

    // calculate reference
    for (int i_batch = 0; i_batch < m_batch_test; i_batch++) {
      for (int i = 0; i < d_size; i++) {
        for (int j = 0; j < x_size; j++) {
          int k = i + d_size * j; // col major
          int n = getCombinedCounts(
              x_counts + i_batch * x_size * nK32, x_size, d_counts + i_batch * d_size * nK32,
              d_size, K, j, i);

          ref_w_batch[k] -= dw_min * n;
          if (n < 0)
            ref_w_batch[k] = (ref_w_batch[k] > bound) ? bound : ref_w_batch[k];
          else
            ref_w_batch[k] = (ref_w_batch[k] < -bound) ? -bound : ref_w_batch[k];

          if (i_batch == 0) {
            ref_w[k] = ref_w_batch[k];
          }
        }
      }
    }
  };

  void TearDown() {
    delete[] weights;
    delete[] ref_w;
    delete[] ref_w_batch;
    delete[] x_counts;
    delete[] d_counts;
    delete[] x_counts_trans;
    delete[] d_counts_trans;
  };

  uint32_t *x_counts, *d_counts, *x_counts_trans, *d_counts_trans;
  num_t *weights, *ref_w, *ref_w_batch;
  num_t timing, dw_min, dw_min_std, bound;
  int K, x_size, d_size, m_batch, nK32, m_batch_test;
};

// define the tests
INSTANTIATE_TEST_CASE_P(kernelCase, UpdateKernelWTestFixture, ::testing::Range(0, 4));

TEST_F(UpdateKernelWTestFixture, KernelUpdateW) {

  // calculate with kernel
  test_helper::debugKernelUpdateW(
      this->weights, this->x_counts, this->x_size, this->d_counts, this->d_size, this->nK32,
      this->dw_min, this->dw_min_std, this->bound, &this->timing);

  std::cout << BOLD_ON << "\nUpdate W:   " << timing << " msec\n" << BOLD_OFF << std::endl;

  NO_REFERENCE_CHECK;

  for (int i = 0; i < d_size * x_size; i++) {
    // std::cout << "w_ref [n="<< n[i] << "] " << ref_w[i]<< " vs. w " <<  weights[i] << std::endl;
    ASSERT_NEAR(ref_w[i], weights[i], 1e-5);
  }
}

TEST_P(UpdateKernelWTestFixture, KernelUpdateWBatch) {

  if (GetParam() > 2) {
    return;
  }
  // calculate with kernel
  test_helper::debugKernelUpdateWBatch(
      this->weights, this->x_counts, this->x_size, this->d_counts, this->d_size, this->nK32,
      this->m_batch, false, this->dw_min, this->dw_min_std, this->bound, GetParam(), &this->timing);

  std::cout << BOLD_ON << "\nUpdate W  [Batch]:   " << timing << " msec\n" << BOLD_OFF << std::endl;

  NO_REFERENCE_CHECK;

  for (int i = 0; i < d_size * x_size; i++) {
    // std::cout << "w_ref [n="<< n[i] << "] " << ref_w[i]<< " vs. w " <<  weights[i] << std::endl;
    ASSERT_NEAR(ref_w_batch[i], weights[i], 1e-5);
  }
}

TEST_P(UpdateKernelWTestFixture, KernelUpdateWBatchTrans) {
  if (GetParam() > 2) {
    return;
  }

  // calculate with kernel
  test_helper::debugKernelUpdateWBatch(
      this->weights, x_counts_trans, this->x_size, d_counts_trans, this->d_size, this->nK32,
      this->m_batch, true, this->dw_min, this->dw_min_std, this->bound, GetParam(), &this->timing);

  std::cout << BOLD_ON << "\nUpdate W Trans [Batch]:   " << timing << " msec\n"
            << BOLD_OFF << std::endl;

  NO_REFERENCE_CHECK;

  for (int i = 0; i < d_size * x_size; i++) {
    // std::cout << "w_ref [n="<< n[i] << "] " << ref_w[i]<< " vs. w " <<  weights[i] << std::endl;
    ASSERT_NEAR(ref_w_batch[i], weights[i], 1e-5);
  }
}

TEST_P(UpdateKernelWTestFixture, KernelUpdateWBatchShared) {

  if (GetParam() > 3 || GetParam() == 1) { // 1 is non trans 64. Not possible
    return;
  }

  // calculate with kernel
  test_helper::debugKernelUpdateWBatchShared(
      this->weights, this->x_counts, this->x_size, this->d_counts, this->d_size, this->K,
      this->m_batch, false, this->dw_min, this->dw_min_std, this->bound, GetParam(), &this->timing);

  std::cout << BOLD_ON << "\nUpdate W  Shared [Batch]:   " << timing << " msec\n"
            << BOLD_OFF << std::endl;

  NO_REFERENCE_CHECK;

  for (int i = 0; i < d_size * x_size; i++) {
    // std::cout << "w_ref [n="<< n[i] << "] " << ref_w[i]<< " vs. w " <<  weights[i] << std::endl;
    ASSERT_NEAR(ref_w_batch[i], weights[i], 1e-5);
  }
}

TEST_P(UpdateKernelWTestFixture, KernelUpdateWBatchSharedTrans) {

  if (GetParam() > 3) {
    return;
  }

  // calculate with kernel
  test_helper::debugKernelUpdateWBatchShared(
      this->weights, this->x_counts_trans, this->x_size, this->d_counts_trans, this->d_size,
      this->K, this->m_batch, true, this->dw_min, this->dw_min_std, this->bound, GetParam(),
      &this->timing);

  std::cout << BOLD_ON << "\nUpdate W  Shared Trans [Batch]:   " << timing << " msec\n"
            << BOLD_OFF << std::endl;

  NO_REFERENCE_CHECK;

  for (int i = 0; i < d_size * x_size; i++) {
    // std::cout << "w_ref [n="<< n[i] << "] " << ref_w[i]<< " vs. w " <<  weights[i] << std::endl;
    ASSERT_NEAR(ref_w_batch[i], weights[i], 1e-5);
  }
}

} // namespace

int main(int argc, char **argv) {
  resetCuda();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
