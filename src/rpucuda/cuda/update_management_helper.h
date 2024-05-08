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

#pragma once

#include "cuda_util.h"
#include "maximizer.h"

namespace RPU {

template <typename T> class UpdateManagementHelper {

public:
  explicit UpdateManagementHelper(CudaContextPtr c, int x_size, int d_size);

  template <typename XInputIteratorT, typename DInputIteratorT>
  void computeKandScaleValues(
      XInputIteratorT x_in,
      DInputIteratorT d_in,
      const T dw_min,
      const T lr,
      const bool update_management,
      const bool update_bl_management,
      const int m_batch,
      const bool x_trans,
      const bool d_trans,
      const int Kmax,
      const T um_reg_scale,
      const T um_grad_scale);

  void computeKcBlock(int m_batch);
  void computeKc(int m_batch);

  void translateTransToBatchOrder64(
      uint64_t *x_counts_bo64,
      uint64_t *d_counts_bo64,
      const uint32_t *x_counts,
      const uint32_t *d_counts,
      const int m_batch,
      const int BL,
      const bool update_bl_management);

  inline kagg_t *getKnData(bool ublm, int m_batch) const {
    if (ublm) {
      return dev_Kc_values_->getData() + m_batch;
    } else {
      return nullptr;
    }
  };
  inline kagg_t *getKcValueData() const { return dev_Kc_values_->getData(); };
  inline int getBo64Batch(int m_batch, int BL) const {
    return (m_batch * BL + 31) / 32;
  }; // this is for no UBLM

  inline int *getKValueData() const { return dev_K_values_->getData(); };
  inline kagg_t *getKcBlockData() const { return dev_Kc_block_->getData(); };
  inline kagg_t *getKcBlockAggregateData() const { return dev_Kc_block_aggregate_->getData(); };
  kagg_t getKnValue(bool ublm, int m_batch, int K) const;
  inline T *getScaleValueData() const { return dev_scale_values_->getData(); };

  // for debug
  inline const CudaArray<int> &getKValues() const { return *dev_K_values_; };
  inline void getScaleValues(T *dest) const { dev_scale_values_->copyTo(dest); };
  inline void getKValues(int *dest) const { dev_K_values_->copyTo(dest); };

  void getAverageAbsMax(T &m_x, T &m_d, int m_batch) const;
  void getAverageLogAbsMax(T &m_x, T &m_d, int m_batch) const;
  void getAbsMax(T &m_x, T &m_d, int m_batch) const;

private:
  void initializeBuffers(int m_batch);

  CudaContextPtr context_ = nullptr;

  int x_size_ = 0;
  int d_size_ = 0;
  int buffer_m_batch_ = 0;
  int nthreads_ = 0;

  std::unique_ptr<Maximizer<T>> x_maximizer_ = nullptr;
  std::unique_ptr<Maximizer<T>> d_maximizer_ = nullptr;

  std::unique_ptr<CudaArray<int>> dev_K_values_ = nullptr;
  std::unique_ptr<CudaArray<T>> dev_scale_values_ = nullptr;
  std::unique_ptr<CudaArray<kagg_t>> dev_Kc_values_ = nullptr;
  std::unique_ptr<CudaArray<char>> dev_Kc_temp_storage_ = nullptr;
  std::unique_ptr<CudaArray<kagg_t>> dev_Kc_block_ = nullptr;
  std::unique_ptr<CudaArray<kagg_t>> dev_Kc_block_aggregate_ = nullptr;
  std::unique_ptr<CudaArray<char>> dev_sumabsmax_temp_storage_ = nullptr;
  std::unique_ptr<CudaArray<T>> dev_sumabsmax_value_ = nullptr;
};

namespace test_helper {
template <typename T, bool ublm>
int debugKernelTranslateTransFormatToBatchOrder64Format(
    T *indata, int size, int m_batch, T scaleprob, int K);
}

} // namespace RPU
