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

#include "bit_line_maker.h"
#include "cuda_util.h"
#include "rpu_pulsed_meta_parameter.h"
#include "rpucuda_pulsed_device.h"

namespace RPU {

template <typename T> class PulsedWeightUpdater {

public:
  explicit PulsedWeightUpdater(CudaContextPtr c, int x_size, int d_size);

  template <typename XInputIteratorT, typename DInputIteratorT>
  void update(
      XInputIteratorT x_in,
      DInputIteratorT d_in,
      T *dev_weights,
      AbstractRPUDeviceCuda<T> *rpucuda_device,
      const PulsedUpdateMetaParameter<T> &up,
      const T lr,
      const int m_batch = 1,
      const bool x_trans_in = false,
      const bool d_trans_in = false);

  void getCountsDebug(uint32_t *x_counts, uint32_t *d_counts) {
    blm_->getCountsDebug(x_counts, d_counts);
  };
  bool checkForFPUpdate(
      AbstractRPUDeviceCuda<T> *rpucuda_device_in, const PulsedUpdateMetaParameter<T> &up);

  void waitForUpdateCalculations();
  void makeUpdateAsync();
  // PulsedUpdateType getUpdateType() {return update_type_;};
  template <typename XInputIteratorT, typename DInputIteratorT>
  void doFPupdate(
      XInputIteratorT x_in,
      DInputIteratorT d_in,
      T *dev_weights,
      const T lr,
      const int m_batch,
      const bool x_trans,
      const bool d_trans,
      const T beta = (T)1.0);

  template <typename XInputIteratorT, typename DInputIteratorT>
  void doDirectUpdate(
      XInputIteratorT x_in,
      DInputIteratorT d_in,
      AbstractRPUDeviceCuda<T> *rpucuda_device,
      T *dev_weights,
      const T lr,
      const PulsedUpdateMetaParameter<T> &up,
      const int m_batch,
      const bool x_trans,
      const bool d_trans,
      const T beta = (T)1.0);
  void setVerbosityLevel(int level) { verbose_ = level; };
  void dumpExtra(RPU::state_t &extra, const std::string prefix);
  void loadExtra(const RPU::state_t &extra, const std::string prefix, bool strict);

private:
  // void setUpdateType(PulsedUpdateType update_type);
  pwukpvec_t<T> getValidUpdateKernels(
      PulsedRPUDeviceCudaBase<T> *rpucuda_device,
      int m_batch,
      const PulsedUpdateMetaParameter<T> &up);

  template <typename InputIteratorT>
  const T *copyIterator2Buffer(InputIteratorT vec, T *buffer, int size);

  template <typename XInputIteratorT, typename DInputIteratorT>
  void executeUpdate(
      pwukp_t<T> kpars,
      XInputIteratorT x_in,
      DInputIteratorT d_in,
      T *dev_weights,
      PulsedRPUDeviceCudaBase<T> *rpucuda_device,
      const PulsedUpdateMetaParameter<T> &up,
      const T lr,
      const int m_batch,
      const bool x_trans_in,
      const bool d_trans_in);

  template <typename XInputIteratorT, typename DInputIteratorT>
  void tuneUpdate(
      pwukp_t<T> &opt_kernel_pars,
      pwukpvec_t<T> &v,
      XInputIteratorT x_in,
      DInputIteratorT d_in,
      T *dev_weights,
      PulsedRPUDeviceCudaBase<T> *rpucuda_device,
      const PulsedUpdateMetaParameter<T> &up,
      const T lr,
      const int m_batch,
      const bool x_trans_in,
      const bool d_trans_in);

  CudaContextPtr context_ = nullptr;
  int x_size_ = 0;
  int d_size_ = 0;
  int update_count_ = 0;
  bool is_async_update_ = false;
  int verbose_ = 0;
  DeviceUpdateType update_type_ = DeviceUpdateType::Undefined;
  pwukp_t<T> kernel_pars_;
  pwukpvec_t<T> valid_kernels_;
  std::unique_ptr<BitLineMaker<T>> blm_ = nullptr;
  std::unique_ptr<CudaContext> up_context_ = nullptr;
};

} // namespace RPU
