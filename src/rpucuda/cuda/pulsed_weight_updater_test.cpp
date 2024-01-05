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
#include "chopped_weight_output.h"
#include "cuda.h"
#include "cuda_util.h"
#include "pulsed_weight_updater.h"
#include "rng.h"
#include "rpucuda_constantstep_device.h"
#include "rpucuda_pulsed.h"
#include "test_helper.h"
#include "utility_functions.h"
#include "gtest/gtest.h"
#include <chrono>
#include <memory>
#include <random>

#define TOLERANCE 1e-4

namespace {

using namespace RPU;

#define NO_REFERENCE_CHECK                                                                         \
  if ((m_batch != m_batch_test) || (dw_min_std > (num_t)0.0)) {                                    \
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
      d_size = 12;
      x_size = 32;
      m_batch = 30;
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

    unsigned int seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
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

          ref_w_batch[k] -= dw_min * (num_t)n;
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

  void DebugUpdateKernels(
      int x_size,
      int d_size,
      int K,
      int m_batch,
      bool trans,
      num_t dw_min,
      int use_bo64,
      PulsedUpdateMetaParameter<num_t> &up,
      ChoppedWeightOutputParameter<num_t> &cwo_par,
      num_t sparsity = 3.0,
      bool flexible_in_size = false,
      bool verbose = false) {
    auto c_container = CudaContext(-1, false);
    CudaContextPtr c = &c_container;
    CUDA_TIMING_INIT;

    int Kplus1 = K + 1;
    int nK32 = (Kplus1 + 31) / 32;
    if (use_bo64 > 0 && !trans) {
      std::cout << "Batch order 64 only implemented for nK32==1 .\n";
      return;
    }

    ConstantStepRPUDeviceMetaParameter<num_t> dp;
    dp.dw_min = dw_min;
    dp.up_down_dtod = 0.0;
    dp.up_down = 0.0;
    dp.dw_min_std = 0.0;
    dp.dw_min_dtod = 0.0;
    dp.w_max = 1e12;
    dp.w_min = -1e12;
    dp.w_max_dtod = 0.0; // turn off otherwise issue with large bound (overflow possible)
    dp.w_min_dtod = 0.0;
    dp.enforce_consistency = true;

    RealWorldRNG<num_t> rng{0};
    auto rpu_device = dp.createDeviceUnique(x_size, d_size, &rng);
    auto rpucuda_device_abstract = AbstractRPUDeviceCuda<num_t>::createFromUnique(c, *rpu_device);
    auto *rpucuda_device = static_cast<PulsedRPUDeviceCuda<num_t> *>(&*rpucuda_device_abstract);
    BitLineMaker<num_t> blm(c, x_size, d_size);
    ChoppedWeightOutput<num_t> cwo(c, x_size, d_size);
    cwo.setFlexibleInSize(flexible_in_size);
    cwo.setPar(cwo_par);

    int n_x = ((x_size * m_batch) + 31) / 32 * 32;
    int n_d = ((d_size * m_batch) + 31) / 32 * 32;

    CudaArray<float> dev_x_in_float(c, n_x);
    CudaArray<float> dev_d_in_float(c, n_d);

    CudaArray<num_t> dev_x_in(c, n_x);
    CudaArray<num_t> dev_d_in(c, n_d);

    c->randNormal(dev_x_in_float.getData(), n_x, 0.0, 1.0);
    c->randNormal(dev_d_in_float.getData(), n_d, 0.0, dw_min * (num_t)K / sparsity);
    RPU::math::elemcopy(c, dev_x_in.getData(), n_x, 1, dev_x_in_float.getDataConst(), 1);
    RPU::math::elemcopy(c, dev_d_in.getData(), n_d, 1, dev_d_in_float.getDataConst(), 1);

    bool implicit_pulses = false;
    up.desired_BL = K;

    auto kernels = rpucuda_device->getUpdateKernels(m_batch, nK32, use_bo64, trans, up);

    for (size_t i_kernel = 0; i_kernel < kernels.size(); i_kernel++) {
      auto kernel = kernels[i_kernel];

      if (cwo_par.isEnabled()) {
        kernel->ensureCWO();
      }

      if (!kernel->isValid()) {
        continue;
      }

      std::cout << std::endl;
      if (verbose) {
        kernel->print();
      }

      std::vector<num_t> res_weights;
      std::vector<num_t> res_weights_outputs;
      std::vector<num_t> ref_weights;
      std::vector<num_t> ref_weights_outputs;
      std::vector<num_t> ref_weights_batch;

      CudaArray<num_t> dev_weights(c, d_size * x_size);
      dev_weights.setConst(0.0);
      cwo.setCounter(10);
      auto rand_states = c->getRandomStates(MAX(kernel->getNStates(), 1));
      c->synchronize();
      blm.makeCounts(
          dev_x_in.getData(), dev_d_in.getData(), up, dw_min, 1.0, m_batch, trans, trans, trans,
          use_bo64, implicit_pulses);
      cwo.makeWeightOutputChoppers(&blm);

      blm.makeCounts(
          dev_x_in.getData(), dev_d_in.getData(), up, dw_min, 1.0, m_batch, trans, trans, trans,
          use_bo64, implicit_pulses);

      cwo.makeWeightOutputChoppers(&blm);
      int n_wo = cwo.getNumWeightOutputs();
      if (verbose) {
        std::cout << "\nCWO: " << std::endl;
        cwo.print();
      }
      c->synchronizeDevice();
      CUDA_TIMING_START(c);
      kernel->run(
          c->getStream(), dev_weights.getData(), m_batch, &blm, &*rpucuda_device, up, rand_states,
          0, nullptr, nullptr, &cwo);
      CUDA_TIMING_STOP(c, kernel->getName());
      c->synchronizeDevice();
      blm.getAccCountsDebug(
          cwo_par.isEnabled() ? &cwo : nullptr, ref_weights, ref_weights_outputs, ref_weights_batch,
          up, dw_min, flexible_in_size, verbose);

      dev_weights.copyTo(res_weights);
      c->synchronizeDevice();

      for (size_t i = 0; i < res_weights.size(); i++) {
        if (verbose) {
          std::cout << "X " << i / d_size << ", D " << i % d_size << ": \t" << res_weights[i]
                    << "\t[res] vs. \t" << ref_weights[i] << " \t[ref]" << std::endl;
        }
        ASSERT_NEAR(res_weights[i], ref_weights[i], TOLERANCE);
      }

      if (n_wo > 0) {
        CudaArray<num_t> tmp(c, n_wo * (cwo_par.use_columns ? d_size : x_size));
        tmp.assignFromDevice(cwo.getWeightOutputData());
        tmp.copyTo(res_weights_outputs);

        for (size_t i = 0; i < res_weights_outputs.size(); i++) {
          if (ref_weights_outputs[i] == std::numeric_limits<num_t>::max()) {
            continue;
          }
          if (verbose) {
            std::cout << "WO: I " << i << ": \t" << res_weights_outputs[i] << "\t[res] vs. \t"
                      << ref_weights_outputs[i] << " \t[ref]" << std::endl;
          }
          ASSERT_NEAR(res_weights_outputs[i], ref_weights_outputs[i], TOLERANCE);
        }
      }
    }

    cwo.releaseBuffers();

    CUDA_TIMING_DESTROY;
  }

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
    ASSERT_NEAR(ref_w_batch[i], weights[i], 1e-5);
  }
}

TEST_P(UpdateKernelWTestFixture, KernelUpdateSingleBatch) {
  int x_size = 4;
  int d_size = 5;

  int use_b064 = GetParam() % 2;
  int ublm = GetParam() / 2;
  PulsedUpdateMetaParameter<num_t> up;
  ChoppedWeightOutputParameter<num_t> cwo;
  up.update_bl_management = ublm;
  up.update_management = ublm;
  int BL = 20;
  int m_batch = 1;
  num_t dw_min = 0.01;
  bool trans = false;

  // standard without chopper
  DebugUpdateKernels(x_size, d_size, BL, m_batch, trans, dw_min, use_b064, up, cwo);
}

TEST_P(UpdateKernelWTestFixture, KernelUpdateSingleBatchLargeK) {
  if (GetParam() > 1) {
    return;
  }
  int x_size = 4;
  int d_size = 5;

  int use_b064 = 0;
  int ublm = GetParam() % 2;
  PulsedUpdateMetaParameter<num_t> up;
  ChoppedWeightOutputParameter<num_t> cwo;
  up.update_bl_management = ublm;
  up.update_management = ublm;
  int BL = 62;
  int m_batch = 1;
  num_t dw_min = 0.01;
  bool trans = false;

  // standard without chopper
  DebugUpdateKernels(x_size, d_size, BL, m_batch, trans, dw_min, use_b064, up, cwo);
}

TEST_P(UpdateKernelWTestFixture, KernelUpdateLargeK) {

  int x_size = 209;
  int d_size = 25;

  int use_b064 = false;
  int ublm = GetParam() / 2;
  PulsedUpdateMetaParameter<num_t> up;
  ChoppedWeightOutputParameter<num_t> cwo;
  up.update_bl_management = ublm;
  up.update_management = ublm;
  int BL = 67;
  int m_batch = 100;
  num_t dw_min = 0.01;
  bool trans = GetParam() % 2;

  // standard without chopper
  DebugUpdateKernels(x_size, d_size, BL, m_batch, trans, dw_min, use_b064, up, cwo);
}

TEST_P(UpdateKernelWTestFixture, KernelUpdate) {

  int use_b064 = GetParam() % 2;
  int ublm = GetParam() / 2;
  PulsedUpdateMetaParameter<num_t> up;
  ChoppedWeightOutputParameter<num_t> cwo;
  up.update_bl_management = ublm;
  up.update_management = ublm;
  int BL = 5;
  int m_batch = 302;
  num_t dw_min = 0.01;
  bool trans = true;
  int d_size = 100;
  int x_size = 124;

  // standard without CWO
  DebugUpdateKernels(x_size, d_size, BL, m_batch, trans, dw_min, use_b064, up, cwo);
}

TEST_P(UpdateKernelWTestFixture, KernelUpdateWeightOutputNoChopper) {

  int use_b064 = GetParam() % 2;
  int ublm = GetParam() / 2;
  PulsedUpdateMetaParameter<num_t> up;
  ChoppedWeightOutputParameter<num_t> cwo;
  up.update_bl_management = ublm;
  up.update_management = ublm;
  int BL = 20;
  int m_batch = 200;
  num_t dw_min = 0.01;
  bool trans = true;
  int d_size = 100;
  int x_size = 124;

  cwo.every = 2;
  cwo.in_chop_prob = 0.0;
  cwo.out_chop_prob = 0.0;
  cwo.use_columns = false;
  num_t sparsity = 5;
  DebugUpdateKernels(x_size, d_size, BL, m_batch, trans, dw_min, use_b064, up, cwo, sparsity);
}

TEST_P(UpdateKernelWTestFixture, KernelUpdateWeightOutputNoChopperColumn) {

  int use_b064 = GetParam() % 2;
  int ublm = GetParam() / 2;
  PulsedUpdateMetaParameter<num_t> up;
  ChoppedWeightOutputParameter<num_t> cwo;
  up.update_bl_management = ublm;
  up.update_management = ublm;
  int BL = 20;
  int m_batch = 102;
  num_t dw_min = 0.01;
  bool trans = true;
  int d_size = 10;
  int x_size = 12;

  cwo.every = 2;
  cwo.in_chop_prob = 0.0;
  cwo.out_chop_prob = 0.0;
  cwo.use_columns = true;

  DebugUpdateKernels(x_size, d_size, BL, m_batch, trans, dw_min, use_b064, up, cwo);
}

TEST_P(UpdateKernelWTestFixture, KernelUpdateWeightOutput) {

  int use_b064 = GetParam() % 2;
  int ublm = GetParam() / 2;
  PulsedUpdateMetaParameter<num_t> up;
  ChoppedWeightOutputParameter<num_t> cwo;
  up.update_bl_management = ublm;
  up.update_management = ublm;
  int BL = 12;
  int m_batch = 5;
  num_t dw_min = 0.01;
  bool trans = true;
  int x_size = 4;
  int d_size = 4;

  cwo.every = 3;

  cwo.in_chop_prob = 1.0;
  cwo.out_chop_prob = 1.0;
  cwo.use_columns = false;

  DebugUpdateKernels(x_size, d_size, BL, m_batch, trans, dw_min, use_b064, up, cwo, 3.0);
}

TEST_P(UpdateKernelWTestFixture, KernelUpdateWeightOutputSingleBatchLargeEvery) {

  int use_b064 = GetParam() % 2;
  int ublm = 1;
  PulsedUpdateMetaParameter<num_t> up;
  ChoppedWeightOutputParameter<num_t> cwo;
  up.update_bl_management = ublm;
  up.update_management = ublm;
  int BL = 12;
  int m_batch = 1;
  num_t dw_min = 0.01;
  bool trans = true;
  int x_size = 4;
  int d_size = 4;

  cwo.every = 3;

  cwo.in_chop_prob = 1.0;
  cwo.out_chop_prob = 1.0;
  cwo.use_columns = GetParam() / 2;

  DebugUpdateKernels(x_size, d_size, BL, m_batch, trans, dw_min, use_b064, up, cwo);
}

TEST_P(UpdateKernelWTestFixture, KernelUpdateWeightOutputBatch) {

  int use_b064 = false;
  int ublm = false;
  PulsedUpdateMetaParameter<num_t> up;
  ChoppedWeightOutputParameter<num_t> cwo;
  up.update_bl_management = ublm;
  up.update_management = ublm;
  int BL = 20;
  int m_batch = 1 + GetParam();
  num_t dw_min = 0.01;
  bool trans = true;
  int x_size = 4;
  int d_size = 4;
  num_t sparsity = 3.0;

  cwo.every = 3;

  cwo.in_chop_prob = 1.0;
  cwo.out_chop_prob = 1.0;
  cwo.use_columns = false;
  bool flexible_in_size = false;

  DebugUpdateKernels(
      x_size, d_size, BL, m_batch, trans, dw_min, use_b064, up, cwo, sparsity, flexible_in_size,
      true);
}

TEST_P(UpdateKernelWTestFixture, KernelUpdateWeightOutputBatchFlexible) {

  int use_b064 = false;
  int ublm = false;
  PulsedUpdateMetaParameter<num_t> up;
  ChoppedWeightOutputParameter<num_t> cwo;
  up.update_bl_management = ublm;
  up.update_management = ublm;
  int BL = 20;
  int m_batch = 1 + GetParam();
  num_t dw_min = 0.01;
  bool trans = true;
  int x_size = 4;
  int d_size = 4;
  num_t sparsity = 3.0;
  cwo.every = 3;

  cwo.in_chop_prob = 1.0;
  cwo.out_chop_prob = 1.0;
  cwo.use_columns = false;
  int flexible_in_size = true;
  DebugUpdateKernels(
      x_size, d_size, BL, m_batch, trans, dw_min, use_b064, up, cwo, sparsity, flexible_in_size,
      true);
}

TEST_P(UpdateKernelWTestFixture, KernelUpdateWeightOutputLargeK) {
  if (GetParam() > 0) {
    return;
  }
  int use_b064 = 0;
  int ublm = true;
  PulsedUpdateMetaParameter<num_t> up;
  ChoppedWeightOutputParameter<num_t> cwo;
  up.update_bl_management = ublm;
  up.update_management = ublm;
  int BL = 67;
  int m_batch = 7;
  num_t dw_min = 0.01;
  bool trans = GetParam() % 2;

  cwo.every = 3;
  cwo.in_chop_prob = 0.8;
  cwo.out_chop_prob = 0.8;
  cwo.use_columns = GetParam() / 2;

  DebugUpdateKernels(x_size, d_size, BL, m_batch, trans, dw_min, use_b064, up, cwo, 3.0, false);
}

TEST_P(UpdateKernelWTestFixture, KernelUpdateWeightOutputUseColumns) {

  if (GetParam() > 0) {
    return;
  }

  int use_b064 = GetParam() % 2;
  int ublm = GetParam() / 2;
  PulsedUpdateMetaParameter<num_t> up;
  ChoppedWeightOutputParameter<num_t> cwo;
  up.update_bl_management = ublm;
  up.update_management = ublm;
  int BL = 31;
  int m_batch = 3;
  num_t dw_min = 0.01;
  bool trans = true;

  cwo.every = 2;
  cwo.in_chop_prob = 0.8;
  cwo.out_chop_prob = 0.8;
  cwo.use_columns = false;

  DebugUpdateKernels(
      this->x_size / 2 + 2, this->d_size, BL, m_batch, trans, dw_min, use_b064, up, cwo);
}

} // namespace

int main(int argc, char **argv) {
  resetCuda();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
