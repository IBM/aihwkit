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

#pragma once

#include "cuda_util.h"
#include "rpu_pulsed_meta_parameter.h"
#include "update_management_helper.h"

namespace RPU {

enum BLMOutputFormat {
  NotSet, // dummy init
  FP,     // floating point mode for implicit pulses
  UI32,   // standard 32-bit word of bits per x,d value [more than one word supported]. First bit is
          // sign
  BO64,   // 64 word with sign bits in upper 32-bit word and data in
          // lower. Batch major ordering. Here more than one batch
          // value can be squeezed together in one word. Useful for
          // small BL, will speed up loading across batches
  UI32BO64 // translate mode, first UI32 than compressed into BO64
};

template <typename T> class BitLineMaker {

public:
  explicit BitLineMaker(CudaContext *c, int x_size, int d_size);

  template <typename XInputIteratorT, typename DInputIteratorT>
  void makeCounts(
      XInputIteratorT x_in,
      DInputIteratorT d_in,
      const PulsedUpdateMetaParameter<T> &up,
      const T dw_min,
      const T lr,
      const int m_batch = 1,
      const bool x_trans = false,
      const bool d_trans = false,
      const bool out_trans = false,
      const int use_bo64 = 0,
      const bool implicit_pulses = false);

  BLMOutputFormat getFormat(int use_bo64, bool implicit_pulses);

  T *getXData() const;
  T *getDData() const;

  uint32_t *getXCountsData() const;
  uint32_t *getDCountsData() const;

  uint64_t *getXCountsBo64Data() const;
  uint64_t *getDCountsBo64Data() const;
  kagg_t *getKnData(bool ublm) const;
  int getBo64Batch(int m_batch) const;

  void copyXCountsToHost(uint32_t *dest) const;
  void copyDCountsToHost(uint32_t *dest) const;

  void copyXCountsBo64ToHost(uint64_t *dest) const;
  void copyDCountsBo64ToHost(uint64_t *dest) const;

  inline bool checkBuffer(int m_batch, int BL) const {
    return (BL == buffer_BL_) && (m_batch <= buffer_m_batch_);
  }
  inline int getNK32Current() const { return current_BL_ / 32 + 1; };
  void getCountsDebug(uint32_t *x_counts, uint32_t *d_counts);
  void getFPCounts(T *x_counts, T *d_counts);

  inline T getCurrentLR() const { return current_lr_; };
  // helper for debug
  UpdateManagementHelper<T> *getUmh() const { return &*umh_; };

  void initializeBLBuffers(int m_batch, int BL, int use_bo64, bool implicit_pulses);

private:
  CudaContext *context_ = nullptr;
  int x_size_ = 0;
  int d_size_ = 0;
  int nthreads_ = 0;
  int max_block_count_ = 0;
  int buffer_BL_ = 0;
  int current_BL_ = 0;
  T current_lr_ = 0;
  int buffer_m_batch_ = 0;
  BLMOutputFormat format_ = BLMOutputFormat::NotSet;

  std::unique_ptr<CudaArray<T>> dev_x_ = nullptr;
  std::unique_ptr<CudaArray<T>> dev_d_ = nullptr;

  std::unique_ptr<CudaArray<uint32_t>> dev_x_counts_ = nullptr;
  std::unique_ptr<CudaArray<uint32_t>> dev_d_counts_ = nullptr;
  std::unique_ptr<CudaArray<uint64_t>> dev_x_counts_bo64_ = nullptr;
  std::unique_ptr<CudaArray<uint64_t>> dev_d_counts_bo64_ = nullptr;

  std::shared_ptr<CudaArray<curandState_t>> dev_states_ = nullptr;

  std::unique_ptr<UpdateManagementHelper<T>> umh_ = nullptr;
};

namespace test_helper {
template <typename T>
int debugKernelUpdateGetCounts_Loop2(
    T *indata, int size, T scaleprob, uint32_t *counts, int K, T *timing, bool fake_seed);

template <typename T>
int debugKernelUpdateGetCountsBatch_Loop2(
    T *indata, int size, T scaleprob, uint32_t *counts, int K, T *timing, bool fake_seed);

template <typename T>
int debugKernelUpdateGetCountsBatch_SimpleLoop2(
    T *indata, int size, T scaleprob, uint32_t *counts, int K, T *timing, bool fake_seed);

template <typename T, int ITEMS_PER_THREAD>
int debugKernelUpdateGetCounts_Linear(
    T *indata, int size, T scaleprob, uint32_t *counts, int K, T *timing, bool fake_seed);
int getCounts(uint32_t *counts, int i, int K, int size, bool negtest);

template <typename T>
void checkCounts(
    const T *x_input,
    int x_size,
    const T *d_input,
    int d_size,
    int BL,
    T A,
    T B,
    CudaArray<uint32_t> *dev_x_counts,
    CudaArray<uint32_t> *dev_d_counts);

} // namespace test_helper

} // namespace RPU
