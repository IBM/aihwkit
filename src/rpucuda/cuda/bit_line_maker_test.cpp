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

template <typename T> void transpose(T *x_trans, T *x, int size, int m_batch) {

  for (int i = 0; i < size; i++) {
    for (int j = 0; j < m_batch; j++) {
      x_trans[j + i * m_batch] = x[i + j * size];
    }
  }
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

template <typename T> T correlation(T *x, uint32_t *counts, int sz, int K, T scaleprob) {

  int icounts = 0;
  T corr = 0;
  T mx = 0, my = 0, cx = 0, cy = 0;
  T xi;
  for (int i = 0; i < sz; i++) {
    icounts = test_helper::getCounts(counts, i, K, sz, true);
    xi = x[i] * scaleprob;
    T y = ((T)icounts) / (T)K;
    corr += xi * y;
    mx += xi;
    my += y;
    cx += xi * xi;
    cy += y * y;
    if (i < 5) {
      std::cout << xi << " vs  " << y << "[n=" << icounts << "]" << std::endl;
    }
    // EXPECT_NEAR(xi , y, 0.1);
  }
  my /= sz;
  mx /= sz;
  cy /= sz;
  cx /= sz;
  corr /= sz;
  float cc = (float)(corr - mx * my) / (sqrtf(cx - mx * mx) * sqrtf(cy - my * my));
  // std::cout << "Corr coeff:" << cc << std::endl;
  return cc;
}

template <typename T> class BitLineMakerTestFixture : public ::testing::Test {
public:
  void SetUp() {

    nsize = 2; // can be lower than the 4 below
    m_batch = 515;

    size = new int[nsize];
    size[0] = 500;
    size[1] = 5000;
    size[2] = 100;
    size[3] = 10; //// has to be 4 sizes HERE !

    nK = 3;
    K = new int[nK];
    K[0] = 7; // note that one bit has to be sign. so: K=31 is fastest
    K[1] = 31;
    K[2] = 1023; // last biggest

    scaleprob = 1; // needs to be 1 !
    timings = new T[nK * nsize];
    resolution = 0.05;

    x1 = new T[size[0] * m_batch];
    x2 = new T[size[1] * m_batch];
    x3 = new T[size[2] * m_batch];
    x4 = new T[size[3] * m_batch];

    z1 = new T[size[0] * m_batch];
    z2 = new T[size[1] * m_batch];
    z3 = new T[size[2] * m_batch];
    z4 = new T[size[3] * m_batch];

    z4 = new T[size[3] * m_batch];

    unsigned int seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::default_random_engine generator{seed};
    std::uniform_real_distribution<float> udist(-1., 1.);
    auto urnd = std::bind(udist, generator);

    uint32_t noz = 0;
    for (int i = 0; i < size[0] * m_batch; i++) {
      x1[i] = urnd();
      if (i < size[0] && fabsf(x1[i]) < (float)resolution / 2) {
        noz++;
      }
    }
    d_zeros.push_back(noz);

    noz = 0;
    for (int i = 0; i < size[1] * m_batch; i++) {
      x2[i] = urnd();
      if (i < size[1] && fabsf(x2[i]) < (float)resolution / 2) {
        noz++;
      }
    }
    d_zeros.push_back(noz);

    noz = 0;
    for (int i = 0; i < size[2] * m_batch; i++) {
      x3[i] = urnd();
      if (i < size[2] && fabsf(x3[i]) < (float)resolution / 2) {
        noz++;
      }
    }
    d_zeros.push_back(noz);

    noz = 0;
    for (int i = 0; i < size[3] * m_batch; i++) {
      x4[i] = urnd();
      if (i < size[3] && fabsf(x4[i]) < (float)resolution / 2) {
        noz++;
      }
    }
    d_zeros.push_back(noz);

    int nk32max = (K[nK - 1] + 1 + 31) / 32; // plus 1
    count1 = new uint32_t[size[0] * nk32max * m_batch];
    count2 = new uint32_t[size[1] * nk32max * m_batch];
    count3 = new uint32_t[size[2] * nk32max * m_batch];
    count4 = new uint32_t[size[3] * nk32max * m_batch];

    count21 = new uint32_t[size[0] * nk32max * m_batch];
    count22 = new uint32_t[size[1] * nk32max * m_batch];
    count23 = new uint32_t[size[2] * nk32max * m_batch];
    count24 = new uint32_t[size[3] * nk32max * m_batch];

    cptr.push_back(count1);
    cptr.push_back(count2);
    cptr.push_back(count3);
    cptr.push_back(count4);

    c2ptr.push_back(count21);
    c2ptr.push_back(count22);
    c2ptr.push_back(count23);
    c2ptr.push_back(count24);

    xptr.push_back(x1);
    xptr.push_back(x2);
    xptr.push_back(x3);
    xptr.push_back(x4);

    zptr.push_back(z1);
    zptr.push_back(z2);
    zptr.push_back(z3);
    zptr.push_back(z4);
  };

  void TearDown() {
    delete[] x1;
    delete[] x2;
    delete[] x3;
    delete[] x4;

    delete[] z1;
    delete[] z2;
    delete[] z3;
    delete[] z4;

    delete[] count1;
    delete[] count2;
    delete[] count3;
    delete[] count4;

    delete[] count21;
    delete[] count22;
    delete[] count23;
    delete[] count24;

    delete[] size;
    delete[] K;
    delete[] timings;
  };

  void testUpdateFunCorr(int (*fun)(T *, int, T, uint32_t *, uint32_t &, int, T, T *, bool)) {

    int ss = 0;
    uint32_t d_noz = 0;

    for (int isz = 0; isz < this->nsize; isz++) {
      for (int ik = 0; ik < this->nK; ik++) {

        std::cout << "[" << this->size[isz] << ", K=" << this->K[ik] << "]: \n";

        int errcode =
            fun(this->xptr[isz], this->size[isz], this->scaleprob, this->cptr[isz], d_noz,
                this->K[ik], this->resolution, &this->timings[ss], false);
        if (errcode != 0)
          continue;
        std::cout << BOLD_ON << this->timings[ss] << " msec." << BOLD_OFF << "\n\n";

        // add asserts
        float cc = correlation(
            this->xptr[isz], this->cptr[isz], this->size[isz], this->K[isz], this->scaleprob);

        EXPECT_GT(cc, 0.5);
        ASSERT_EQ(d_noz, this->d_zeros[isz]);

        ss++;
        std::cout << "\n\n";
      }
    }
  }

  void testUpdateFunEqual(
      int (*fun1)(T *, int, T, uint32_t *, uint32_t &, int, T, T *, bool),
      int (*fun2)(T *, int, T, uint32_t *, uint32_t &, int, T, T *, bool)) {

    int ss = 0;

    for (int isz = 0; isz < this->nsize; isz++) {
      for (int ik = 0; ik < this->nK; ik++) {
        uint32_t d_noz_1 = 0;
        uint32_t d_noz_2 = 0;

        std::cout << "[" << this->size[isz] << ", K=" << this->K[ik] << "]: \n";
        // same seed
        int errcode = fun1(
            this->xptr[isz], this->size[isz], this->scaleprob, this->cptr[isz], d_noz_1,
            this->K[ik], this->resolution, &this->timings[ss], true);
        int errcode2 = fun2(
            this->xptr[isz], this->size[isz], this->scaleprob, this->c2ptr[isz], d_noz_2,
            this->K[ik], this->resolution, &this->timings[ss], true);

        if (errcode != 0 || errcode2 != 0)
          continue;

        // add asserts
        ASSERT_EQ(d_noz_1, d_noz_2);

        int nK32 = this->K[ik] / 32 + 1;
        for (int ii = 0; ii < this->size[isz] * nK32; ii++) {
          if (ii < 50)
            std::cout << ii << " ";
          ASSERT_EQ(this->cptr[isz][ii], this->c2ptr[isz][ii]);
        }
      }
    }
  }

  void testUpdateFunConst(
      float prob, int (*fun)(T *, int, T, uint32_t *, uint32_t &, int, T, T *, bool)) {

    int isz = 0;
    int ik = 0;
    uint32_t d_noz = 0;

    T *x = this->xptr[isz];
    for (int i = 0; i < this->size[isz]; i++) {
      x[i] = prob;
    }

    std::cout << "[" << this->size[isz] << ", K=" << this->K[ik] << "]: \n";
    int errcode =
        fun(this->xptr[isz], this->size[isz], 1, this->cptr[isz], d_noz, this->K[ik],
            this->resolution, &this->timings[0], false);
    if (errcode != 0)
      return;
    for (int i = 0; i < this->size[isz]; i++) {
      int counts = test_helper::getCounts(this->cptr[isz], i, this->K[ik], this->size[isz], true);
      float cprob = ((float)counts) / this->K[ik];
      ASSERT_FLOAT_EQ(prob, cprob);
    }
  }

  T *x1, *x2, *x3, *x4;
  T *z1, *z2, *z3, *z4;
  uint32_t *count1, *count2, *count3, *count4;
  uint32_t *count21, *count22, *count23, *count24;
  T *timings;
  T scaleprob, resolution;
  int nK, nsize, m_batch;
  int *size, *K;
  std::vector<T *> xptr, zptr;
  std::vector<uint32_t> d_zeros;
  std::vector<uint32_t *> c2ptr;
  std::vector<uint32_t *> cptr;
};

class BitLineMakerParTestFixture : public ::testing::TestWithParam<bool> {
public:
  void SetUp() {

    x_size = 1000;
    d_size = 1001;
    m_batch = 1; // actually not necessary. Batch inside helper functions
    num_t scaleprob = 0.76;
    num_t resolution = 0.01;
    int K = 33;

    up.scaleprob = scaleprob;
    up.desired_BL = K;
    up.res = resolution;
    up.sto_round = false;
    up.d_sparsity = true;

    context = &context_container;

    blm = RPU::make_unique<BitLineMaker<num_t>>(context, x_size, d_size);

    rx = new num_t[x_size * m_batch];
    rd = new num_t[d_size * m_batch];
    rx_trans = new num_t[x_size * m_batch];
    rd_trans = new num_t[d_size * m_batch];

    nK32 = K / 32 + 1;
    x_counts = new uint32_t[x_size * m_batch * nK32];
    d_counts = new uint32_t[d_size * m_batch * nK32];
    x_counts_trans = new uint32_t[x_size * m_batch * nK32];
    d_counts_trans = new uint32_t[d_size * m_batch * nK32];

    unsigned int seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::default_random_engine generator{seed};
    std::uniform_real_distribution<float> udist(-1., 1.);
    auto urnd = std::bind(udist, generator);

    for (int i = 0; i < x_size * m_batch; i++) {
      rx[i] = (num_t)urnd();
    }
    for (int i = 0; i < d_size * m_batch; i++) {
      rd[i] = (num_t)urnd();
    }

    transpose(rd_trans, rd, d_size, m_batch);
    transpose(rx_trans, rx, x_size, m_batch);

    curx = RPU::make_unique<CudaArray<num_t>>(context, x_size * m_batch, rx);
    curd = RPU::make_unique<CudaArray<num_t>>(context, d_size * m_batch, rd);
  }

  void TearDown() {
    delete[] rx;
    delete[] rd;
    delete[] rx_trans;
    delete[] rd_trans;
    delete[] x_counts;
    delete[] d_counts;
  }
  CudaContext context_container{-1, false};
  CudaContextPtr context;
  int nK32;
  int x_size;
  int d_size;
  int m_batch;
  std::shared_ptr<BitLineMaker<num_t>> blm;
  std::shared_ptr<CudaArray<num_t>> curx;
  std::shared_ptr<CudaArray<num_t>> curd;
  DebugPulsedUpdateMetaParameter<num_t> up;
  IOMetaParameter<num_t> io;
  num_t *rx, *rd, *rx_trans, *rd_trans;
  uint32_t *x_counts, *d_counts, *x_counts_trans, *d_counts_trans;
};

INSTANTIATE_TEST_CASE_P(UM, BitLineMakerParTestFixture, ::testing::Bool());

typedef ::testing::Types<num_t> num_types;

TYPED_TEST_CASE(BitLineMakerTestFixture, num_types);

TYPED_TEST(BitLineMakerTestFixture, KernelUpdateGetCountsCorr_Linear1) {
  this->testUpdateFunCorr(RPU::test_helper::debugKernelUpdateGetCounts_Linear<TypeParam, 1>);
}

TYPED_TEST(BitLineMakerTestFixture, KernelUpdateGetCountsZero_Linear1) {
  this->testUpdateFunConst(0, RPU::test_helper::debugKernelUpdateGetCounts_Linear<TypeParam, 1>);
}

TYPED_TEST(BitLineMakerTestFixture, KernelUpdateGetCountsOne_Linear1) {
  this->testUpdateFunConst(1, RPU::test_helper::debugKernelUpdateGetCounts_Linear<TypeParam, 1>);
}

TYPED_TEST(BitLineMakerTestFixture, KernelUpdateGetCountsMinusOne_Linear1) {
  this->testUpdateFunConst(-1, RPU::test_helper::debugKernelUpdateGetCounts_Linear<TypeParam, 1>);
}

TYPED_TEST(BitLineMakerTestFixture, KernelUpdateGetCountsCorr_Linear2) {
  this->testUpdateFunCorr(RPU::test_helper::debugKernelUpdateGetCounts_Linear<TypeParam, 2>);
}

TYPED_TEST(BitLineMakerTestFixture, KernelUpdateGetCountsZero_Linear2) {
  this->testUpdateFunConst(0, RPU::test_helper::debugKernelUpdateGetCounts_Linear<TypeParam, 2>);
}

TYPED_TEST(BitLineMakerTestFixture, KernelUpdateGetCountsOne_Linear2) {
  this->testUpdateFunConst(1, RPU::test_helper::debugKernelUpdateGetCounts_Linear<TypeParam, 2>);
}

TYPED_TEST(BitLineMakerTestFixture, KernelUpdateGetCountsMinusOne_Linear2) {
  this->testUpdateFunConst(-1, RPU::test_helper::debugKernelUpdateGetCounts_Linear<TypeParam, 2>);
}

TYPED_TEST(BitLineMakerTestFixture, KernelUpdateGetCountsCorr_Linear4) {
  this->testUpdateFunCorr(RPU::test_helper::debugKernelUpdateGetCounts_Linear<TypeParam, 4>);
}

TYPED_TEST(BitLineMakerTestFixture, KernelUpdateGetCountsZero_Linear4) {
  this->testUpdateFunConst(0, RPU::test_helper::debugKernelUpdateGetCounts_Linear<TypeParam, 4>);
}

TYPED_TEST(BitLineMakerTestFixture, KernelUpdateGetCountsOne_Linear4) {
  this->testUpdateFunConst(1, RPU::test_helper::debugKernelUpdateGetCounts_Linear<TypeParam, 4>);
}

TYPED_TEST(BitLineMakerTestFixture, KernelUpdateGetCountsMinusOne_Linear4) {
  this->testUpdateFunConst(-1, RPU::test_helper::debugKernelUpdateGetCounts_Linear<TypeParam, 4>);
}

TYPED_TEST(BitLineMakerTestFixture, KernelUpdateGetCountsCorr_Loop2) {

  this->testUpdateFunCorr(RPU::test_helper::debugKernelUpdateGetCounts_Loop2);
}

TYPED_TEST(BitLineMakerTestFixture, KernelUpdateGetCountsZero_Loop2) {
  this->testUpdateFunConst(0, RPU::test_helper::debugKernelUpdateGetCounts_Loop2);
}

TYPED_TEST(BitLineMakerTestFixture, KernelUpdateGetCountsOne_Loop2) {
  this->testUpdateFunConst(1, RPU::test_helper::debugKernelUpdateGetCounts_Loop2);
}

TYPED_TEST(BitLineMakerTestFixture, KernelUpdateGetCountsMinusOne_Loop2) {
  this->testUpdateFunConst(-1, RPU::test_helper::debugKernelUpdateGetCounts_Loop2);
}

TYPED_TEST(BitLineMakerTestFixture, KernelUpdateGetCountsCorrBatch_Loop2) {

  this->testUpdateFunCorr(RPU::test_helper::debugKernelUpdateGetCountsBatch_Loop2);
}

TYPED_TEST(BitLineMakerTestFixture, KernelUpdateGetCountsZeroBatch_Loop2) {
  this->testUpdateFunConst(0, RPU::test_helper::debugKernelUpdateGetCountsBatch_Loop2);
}

TYPED_TEST(BitLineMakerTestFixture, KernelUpdateGetCountsOneBatch_Loop2) {
  this->testUpdateFunConst(1, RPU::test_helper::debugKernelUpdateGetCountsBatch_Loop2);
}

TYPED_TEST(BitLineMakerTestFixture, KernelUpdateGetCountsMinusOneBatch_Loop2) {
  this->testUpdateFunConst(-1, RPU::test_helper::debugKernelUpdateGetCountsBatch_Loop2);
}

TYPED_TEST(BitLineMakerTestFixture, KernelUpdateGetCountsCorrBatch_SimpleLoop2) {

  this->testUpdateFunCorr(RPU::test_helper::debugKernelUpdateGetCountsBatch_SimpleLoop2);
}

TYPED_TEST(BitLineMakerTestFixture, KernelUpdateGetCountsZeroBatch_SimpleLoop2) {
  this->testUpdateFunConst(0, RPU::test_helper::debugKernelUpdateGetCountsBatch_SimpleLoop2);
}

TYPED_TEST(BitLineMakerTestFixture, KernelUpdateGetCountsOneBatch_SimpleLoop2) {
  this->testUpdateFunConst(1, RPU::test_helper::debugKernelUpdateGetCountsBatch_SimpleLoop2);
}

TYPED_TEST(BitLineMakerTestFixture, KernelUpdateGetCountsMinusOneBatch_SimpleLoop2) {
  this->testUpdateFunConst(-1, RPU::test_helper::debugKernelUpdateGetCountsBatch_SimpleLoop2);
}

TEST_P(BitLineMakerParTestFixture, BO64Direct) {
  // each bit should have the same likely hood and negative bits should be exactly the same.
  bool trans = true;
  int k = 20;
  up.update_management = GetParam();
  up.update_bl_management = GetParam();
  up.desired_BL = k;
  num_t dw_min = 0.001;
  num_t lr = 0.01;

  uint64_t *x_counts_bo64 = new uint64_t[x_size * m_batch];
  uint64_t *d_counts_bo64 = new uint64_t[d_size * m_batch];

  uint64_t *x_counts_bo64_ref = new uint64_t[x_size * m_batch];
  uint64_t *d_counts_bo64_ref = new uint64_t[d_size * m_batch];

  int *nx = new int[x_size * m_batch * 64];
  int *nd = new int[d_size * m_batch * 64];
  int *nx_ref = new int[x_size * m_batch * 64];
  int *nd_ref = new int[d_size * m_batch * 64];

  for (int i = 0; i < m_batch * x_size * 64; i++) {
    nx[i] = 0;
    nx_ref[i] = 0;
  }
  for (int i = 0; i < m_batch * d_size * 64; i++) {
    nd[i] = 0;
    nd_ref[i] = 0;
  }

  int nrepeats = 1000;

  for (int i = 0; i < nrepeats; i++) {

    // translate
    blm->makeCounts(
        curx->getData(), curd->getData(), up, dw_min, lr, m_batch, trans, trans, trans, 2);

    context->synchronize();
    blm->copyXCountsBo64ToHost(x_counts_bo64_ref);
    blm->copyDCountsBo64ToHost(d_counts_bo64_ref);
    context->synchronize();
    // direct
    blm->makeCounts(
        curx->getData(), curd->getData(), up, dw_min, lr, m_batch, trans, trans, trans, 1);

    context->synchronize();
    blm->copyXCountsBo64ToHost(x_counts_bo64);
    blm->copyDCountsBo64ToHost(d_counts_bo64);
    context->synchronize();

    for (int j = 0; j < m_batch * x_size; j++) {
      // if (j < 100 && i==0) {
      //   std::cout << "ref 64: " << x_counts_bo64_ref[j] << " vs "  << x_counts_bo64[j] <<
      //   std::endl;
      //}

      for (int ibit = 0; ibit < 64; ibit++) {
        nx[j * 64 + ibit] += (x_counts_bo64[j] & (((uint64_t)1) << ibit)) > 0;
        nx_ref[j * 64 + ibit] += (x_counts_bo64_ref[j] & (((uint64_t)1) << ibit)) > 0;
      }
    }
    for (int j = 0; j < m_batch * d_size; j++) {
      for (int ibit = 0; ibit < 64; ibit++) {
        nd[j * 64 + ibit] += (d_counts_bo64[j] & (((uint64_t)1) << ibit)) > 0;
        nd_ref[j * 64 + ibit] += (d_counts_bo64_ref[j] & (((uint64_t)1) << ibit)) > 0;
      }
    }
  }
  int nfail = 0;
  for (int j = 0; j < m_batch * x_size; j++) {
    for (int ibit = 0; ibit < 64; ibit++) {
      if (ibit >= 32) {
        ASSERT_EQ(nx[j * 64 + ibit], nx_ref[j * 64 + ibit]);
      } else {
        if (fabsf(nx[j * 64 + ibit] - nx_ref[j * 64 + ibit]) > 2 * sqrt(nrepeats))
          nfail++;
      }
    }
  }
  ASSERT_NEAR(((float)nfail) / k / m_batch / x_size, 0, 0.05);

  for (int j = 0; j < m_batch * d_size; j++) {
    for (int ibit = 0; ibit < 64; ibit++) {
      if (ibit >= 32) {
        ASSERT_EQ(nd[j * 64 + ibit], nd_ref[j * 64 + ibit]);
      } else {
        EXPECT_NEAR(nd[j * 64 + ibit], nd_ref[j * 64 + ibit], 4 * sqrt(nrepeats));
      }
    }
  }

  delete[] nd;
  delete[] nx;
  delete[] nx_ref;
  delete[] nd_ref;
  delete[] x_counts_bo64;
  delete[] d_counts_bo64;
  delete[] x_counts_bo64_ref;
  delete[] d_counts_bo64_ref;
}

TEST_P(BitLineMakerParTestFixture, BitlineMakerBatch) {

  // to init the buffers
  bool trans = false;
  up.update_management = GetParam();
  up.update_bl_management = GetParam();
  num_t dw_min = 0.001;
  num_t lr = 0.01;
  blm->makeCounts(curx->getData(), curd->getData(), up, dw_min, lr, m_batch, trans, trans, trans);
  context->synchronize();

  CUDA_TIMING_INIT;
  CUDA_TIMING_START((this->context));
  blm->makeCounts(curx->getData(), curd->getData(), up, dw_min, lr, m_batch, trans, trans, trans);

  if (trans) {
    CUDA_TIMING_STOP((this->context), "Get Counts Batch [trans]");
  } else {
    CUDA_TIMING_STOP((this->context), "Get Counts Batch");
  }
  if (up.update_bl_management)
    std::cout << "Update Management/ Update BL Management is on \n";

  blm->copyXCountsToHost(x_counts);
  blm->copyDCountsToHost(d_counts);

  uint32_t *x_c_tmp = x_counts;
  uint32_t *d_c_tmp = d_counts;
  num_t *x_tmp = rx;
  num_t *d_tmp = rd;

  if (trans) {
    transposeCounts(x_counts_trans, x_counts, x_size, m_batch, nK32);
    x_c_tmp = x_counts_trans;
    x_tmp = rx_trans;

    transposeCounts(d_counts_trans, d_counts, d_size, m_batch, nK32);
    d_c_tmp = d_counts_trans;
    d_tmp = rd_trans;
  }

  for (int i = 0; i < m_batch; i++) {

    int Klocal = up.desired_BL;
    num_t x_sc = up.scaleprob;
    num_t d_sc = up.scaleprob;

    if (up.update_bl_management) {
      num_t x_abs_max_value = Find_Absolute_Max<num_t>(x_tmp + i * x_size, x_size);
      num_t d_abs_max_value = Find_Absolute_Max<num_t>(d_tmp + i * d_size, d_size);

      num_t reg = powf(dw_min, (num_t)2.0);
      x_abs_max_value = MAX(x_abs_max_value, reg);
      d_abs_max_value = MAX(d_abs_max_value, reg);

      Klocal = ceilf(lr * x_abs_max_value * d_abs_max_value / dw_min);
      Klocal = MIN(up.desired_BL, Klocal);
      num_t scaleprob = sqrtf(lr / (dw_min * (num_t)Klocal));

      num_t scale = sqrtf(x_abs_max_value / d_abs_max_value);
      x_sc = scaleprob / scale;
      d_sc = scaleprob * scale;
    }
    float cc_x = correlation(x_tmp + i * x_size, x_c_tmp + i * x_size * nK32, x_size, Klocal, x_sc);
    ASSERT_GT(cc_x, 0.5);

    float cc_d = correlation(d_tmp + i * d_size, d_c_tmp + i * d_size * nK32, d_size, Klocal, d_sc);
    ASSERT_GT(cc_d, 0.5);
  }
}

} // namespace

int main(int argc, char **argv) {
  resetCuda();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
