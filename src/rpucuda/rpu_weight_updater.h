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

#pragma once

#include "dense_bit_line_maker.h"
#include "rng.h"
#include "rpu_pulsed_device.h"
#include "rpu_pulsed_meta_parameter.h"
#include "sparse_bit_line_maker.h"
#include <memory>

namespace RPU {

template <typename T> class RPUWeightUpdater {

public:
  explicit RPUWeightUpdater(int x_size, int d_size) : x_size_(x_size), d_size_(d_size){};
  RPUWeightUpdater(){};

  friend void swap(RPUWeightUpdater<T> &a, RPUWeightUpdater<T> &b) noexcept {
    using std::swap;
    swap(a.x_size_, b.x_size_);
    swap(a.d_size_, b.d_size_);
  }

  virtual void updateVector(
      T **weights,
      const T *x_input,
      const int x_inc,
      const T *d_input,
      const int d_inc,
      const T learning_rate);

protected:
  int x_size_ = 0;
  int d_size_ = 0;
};

/* RPU stochastic version of the forward pass with noise and management ntechniques*/
template <typename T> class PulsedRPUWeightUpdater : public RPUWeightUpdater<T> {

public:
  explicit PulsedRPUWeightUpdater(int x_size, int d_size, std::shared_ptr<RNG<T>> rng);
  PulsedRPUWeightUpdater(){};
  ~PulsedRPUWeightUpdater();

  PulsedRPUWeightUpdater(const PulsedRPUWeightUpdater<T> &);
  PulsedRPUWeightUpdater<T> &operator=(const PulsedRPUWeightUpdater<T> &);
  PulsedRPUWeightUpdater(PulsedRPUWeightUpdater<T> &&);
  PulsedRPUWeightUpdater<T> &operator=(PulsedRPUWeightUpdater<T> &&);

  friend void swap(PulsedRPUWeightUpdater<T> &a, PulsedRPUWeightUpdater<T> &b) noexcept {
    using std::swap;
    swap(static_cast<RPUWeightUpdater<T> &>(a), static_cast<RPUWeightUpdater<T> &>(b));
    swap(a.containers_allocated_, b.containers_allocated_);

    swap(a.up_, b.up_);
    swap(a.sblm_, b.sblm_);
    swap(a.dblm_, b.dblm_);
    swap(a.rng_, b.rng_);
  }
  bool checkForFPUpdate(AbstractRPUDevice<T> *rpu_device_in);

  virtual void updateVectorWithDevice(
      T **weights,
      const T *x_input,
      const int x_inc,
      const T *d_input,
      const int d_inc,
      const T learning_rate,
      const int last_m_batch_info,
      AbstractRPUDevice<T> *rpu_device);

  void updateVectorWithDeviceAndCounts(
      T **weights,
      const T *x_input,
      const int x_inc,
      const T *d_input,
      const int d_inc,
      const T learning_rate,
      const int last_m_batch_info,
      PulsedRPUDeviceBase<T> *rpu_device,
      uint32_t *x_counts,
      uint32_t *d_counts);

  void setUpPar(const PulsedUpdateMetaParameter<T> &up);

private:
  void freeContainers();
  void allocateContainers();
  bool containers_allocated_ = false;
  std::shared_ptr<RNG<T>> rng_ = nullptr;
  std::unique_ptr<SparseBitLineMaker<T>> sblm_ = nullptr;
  std::unique_ptr<DenseBitLineMaker<T>> dblm_ = nullptr;

  PulsedUpdateMetaParameter<T> up_;
};

} // namespace RPU
