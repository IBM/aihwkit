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

#include "rpu_buffered_transfer_device.h"
#include <sstream>
#include <stdio.h>

namespace RPU {

template <typename T> class ChoppedTransferRPUDevice;

/* Defines the buffered transfer device.

 */

template <typename T>
struct ChoppedTransferRPUDeviceMetaParameter : BufferedTransferRPUDeviceMetaParameter<T> {

  T in_chop_prob = (T)1.0;  // in chopper is applied to the full vector of the A reads
  T in_chop_random = true;  // random or regular
  T out_chop_prob = (T)0.0; // out chopper is applied to output of the A reads

  bool auto_scale = false; // scales according the recent past gradient size

  bool correct_gradient_magnitudes = false; // scale transfer_lr with fast_lr

  T auto_momentum = (T)0.99;   // momentum for auto_scale (in batch?)
  T auto_granularity = (T)0.0; // scales by the number of mat-vecs to reach thres

  T buffer_granularity =
      (T)1.0; // does REPLACE the thres_scale (and is NOT scaled with weight_granularity)
  bool no_buffer = false; // turn off buffer (TTv1)

  ChoppedTransferRPUDeviceMetaParameter() : BufferedTransferRPUDeviceMetaParameter<T>() {
    initDefaults();
  };
  ChoppedTransferRPUDeviceMetaParameter(
      const PulsedRPUDeviceMetaParameterBase<T> &dp, int n_devices)
      : BufferedTransferRPUDeviceMetaParameter<T>(dp, n_devices) {
    initDefaults();
  };
  ChoppedTransferRPUDeviceMetaParameter(
      const PulsedRPUDeviceMetaParameterBase<T> &dp_fast,
      const PulsedRPUDeviceMetaParameterBase<T> &dp_rest,
      int n_total_devices)
      : BufferedTransferRPUDeviceMetaParameter<T>(dp_fast, dp_rest, n_total_devices) {
    initDefaults();
  };

  void initDefaults() { this->n_reads_per_transfer = 1; }

  void checkSupported() const;

  T getTransferLRScale(
      T from_weight_granularity, T to_weight_granularity, T lr, T count_lr, int m_batch) const;
  T getWriteLR(T weight_granularity) const;
  virtual T getPulseCountAutoLR(
      T m_x,
      T m_d,
      T d_sparsity,
      T weight_granularity,
      T transfer_every,
      const PulsedUpdateMetaParameter<T> &up) const;
  inline bool usesAutoTransferEvery() const { return this->transfer_every < (T)0.0; }
  T getAutoTransferEvery(T n_states, const PulsedUpdateMetaParameter<T> &up) const;
  T getBufferGranularity(T weight_granularity, int m_batch) const;
  void updateAutoScale(T &m, T new_val, int m_batch) const;

  std::string getName() const override {
    std::ostringstream ss;
    ss << "ChoppedTransfer(" << this->vec_par.size() << ")";
    if (this->vec_par.size() > 1) {
      ss << ": " << this->vec_par[0]->getName() << " -> " << this->vec_par[1]->getName();
      ;
    }
    return ss.str();
  };

  ChoppedTransferRPUDevice<T> *createDevice(int x_size, int d_size, RealWorldRNG<T> *rng) override {
    return new ChoppedTransferRPUDevice<T>(x_size, d_size, *this, rng);
  };

  ChoppedTransferRPUDeviceMetaParameter<T> *clone() const override {
    return new ChoppedTransferRPUDeviceMetaParameter<T>(*this);
  };
  DeviceUpdateType implements() const override { return DeviceUpdateType::ChoppedTransfer; };
  void printToStream(std::stringstream &ss) const override;
};

template <typename T> class ChoppedTransferRPUDevice : public BufferedTransferRPUDevice<T> {

public:
  // constructor / destructor
  ChoppedTransferRPUDevice(){};
  ChoppedTransferRPUDevice(int x_size, int d_size);
  ChoppedTransferRPUDevice(
      int x_size,
      int d_size,
      const ChoppedTransferRPUDeviceMetaParameter<T> &par,
      RealWorldRNG<T> *rng);
  ~ChoppedTransferRPUDevice();

  ChoppedTransferRPUDevice(const ChoppedTransferRPUDevice<T> &);
  ChoppedTransferRPUDevice<T> &operator=(const ChoppedTransferRPUDevice<T> &);
  ChoppedTransferRPUDevice(ChoppedTransferRPUDevice<T> &&) noexcept;
  ChoppedTransferRPUDevice<T> &operator=(ChoppedTransferRPUDevice<T> &&) noexcept;

  friend void swap(ChoppedTransferRPUDevice<T> &a, ChoppedTransferRPUDevice<T> &b) noexcept {
    using std::swap;
    swap(
        static_cast<BufferedTransferRPUDevice<T> &>(a),
        static_cast<BufferedTransferRPUDevice<T> &>(b));
    swap(a.in_chopper_, b.in_chopper_);
    swap(a.out_chopper_, b.out_chopper_);
    swap(a.m_x_, b.m_x_);
    swap(a.m_d_, b.m_d_);
    swap(a.d_sparsity_, b.d_sparsity_);
    swap(a.x_signed_indices_tmp_, b.x_signed_indices_tmp_);
    swap(a.transfer_counter_, b.transfer_counter_);
  }
  void dumpExtra(RPU::state_t &extra, const std::string prefix) override;
  void loadExtra(const RPU::state_t &extra, const std::string prefix, bool strict) override;

  ChoppedTransferRPUDeviceMetaParameter<T> &getPar() const override {
    return static_cast<ChoppedTransferRPUDeviceMetaParameter<T> &>(SimpleRPUDevice<T>::getPar());
  };

  ChoppedTransferRPUDevice<T> *clone() const override {
    return new ChoppedTransferRPUDevice<T>(*this);
  };

  void readAndUpdate(
      int to_device_idx,
      int from_device_idx,
      const T lr,
      const T *vec,
      const int n_vec,
      const T reset_prob,
      const int i_col,
      const int m_batch_info) override;

  void doSparseUpdate(
      T **weights, int i, const int *x_signed_indices, int x_count, int d_sign, RNG<T> *rng)
      override;

  void doDenseUpdate(T **weights, int *coincidences, RNG<T> *rng) override;

  void initUpdateCycle(
      T **weights,
      const PulsedUpdateMetaParameter<T> &up,
      T current_lr,
      int m_batch_info,
      const T *x_input = nullptr,
      const int x_inc = 1,
      const T *d_input = nullptr,
      const int d_inc = 1) override;
  T getPulseCountLearningRate(
      T lr, int current_m_batch, const PulsedUpdateMetaParameter<T> &up) override;

  T getMx() const { return m_x_; };
  T getMd() const { return m_d_; };

protected:
  void populate(const ChoppedTransferRPUDeviceMetaParameter<T> &par, RealWorldRNG<T> *rng);
  int getTransferEvery(
      int device_idx, int m_batch, const PulsedUpdateMetaParameter<T> &up) const override;

  std::vector<bool> in_chopper_;
  std::vector<bool> out_chopper_;
  T m_x_ = 0.0;
  T m_d_ = 0.0;
  T d_sparsity_ = 0.0;

  inline T getCurrentCountLR() const { return tmp_count_lr_; };
  inline void setCurrentCountLR(T count_lr) { tmp_count_lr_ = count_lr; };
  inline T getCurrentGradStrength() const { return m_x_ * m_d_; };

  uint64_t transfer_counter_ = 0;

private:
  std::vector<int> x_signed_indices_tmp_;
  T tmp_count_lr_ = 1.0;
};

} // namespace RPU
