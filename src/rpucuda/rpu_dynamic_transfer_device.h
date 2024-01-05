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

#include "rpu_chopped_transfer_device.h"
#include <sstream>
#include <stdio.h>

namespace RPU {

template <typename T> class DynamicTransferRPUDevice;

/* Defines the buffered transfer device.

 */

#define FEEDBACK_ESTIMATE 0
#define FEEDBACK_TARGET 1
#define FEEDBACK_MOD 2
#define FEEDBACK_N 3

template <typename T>
struct DynamicTransferRPUDeviceMetaParameter : ChoppedTransferRPUDeviceMetaParameter<T> {

  T tail_weightening = 1.0;
  T buffer_cap = 10.0; // times the max_steps (in case of forget_buffer = false)

  bool experimental_correct_accumulation = false; // for CPU only correct if chopping NOT random !
  bool experimental_fast_lr_feedback = false;
  T experimental_feedback_target = 0.1;
  int experimental_feedback_mod = 1; // times read-out period of past_momentum

  DynamicTransferRPUDeviceMetaParameter() : ChoppedTransferRPUDeviceMetaParameter<T>() {
    initDefaults();
  };
  DynamicTransferRPUDeviceMetaParameter(
      const PulsedRPUDeviceMetaParameterBase<T> &dp, int n_devices)
      : ChoppedTransferRPUDeviceMetaParameter<T>(dp, n_devices) {
    initDefaults();
  };
  DynamicTransferRPUDeviceMetaParameter(
      const PulsedRPUDeviceMetaParameterBase<T> &dp_fast,
      const PulsedRPUDeviceMetaParameterBase<T> &dp_rest,
      int n_total_devices)
      : ChoppedTransferRPUDeviceMetaParameter<T>(dp_fast, dp_rest, n_total_devices) {
    initDefaults();
  };
  inline unsigned int getNumInChopSamples() const;

  void computeCountLRFeedback(
      T &count_lr_scale,
      std::vector<T> &pm_vec,
      uint64_t &previous_update_idx,
      uint64_t current_update_idx,
      int current_m_batch) const;

  void initDefaults() { this->in_chop_random = false; };
  void checkSupported() const;

  std::string getName() const override {
    std::ostringstream ss;
    ss << "DynamicTransfer(" << this->vec_par.size() << ")";
    if (this->vec_par.size() > 1) {
      ss << ": " << this->vec_par[0]->getName() << " -> " << this->vec_par[1]->getName();
      ;
    }
    return ss.str();
  };

  DynamicTransferRPUDevice<T> *createDevice(int x_size, int d_size, RealWorldRNG<T> *rng) override {
    return new DynamicTransferRPUDevice<T>(x_size, d_size, *this, rng);
  };

  DynamicTransferRPUDeviceMetaParameter<T> *clone() const override {
    return new DynamicTransferRPUDeviceMetaParameter<T>(*this);
  };
  DeviceUpdateType implements() const override { return DeviceUpdateType::DynamicTransfer; };
  void printToStream(std::stringstream &ss) const override;
};

template <typename T> class DynamicTransferRPUDevice : public ChoppedTransferRPUDevice<T> {

public:
  // constructor / destructor
  DynamicTransferRPUDevice(){};
  DynamicTransferRPUDevice(int x_size, int d_size);
  DynamicTransferRPUDevice(
      int x_size,
      int d_size,
      const DynamicTransferRPUDeviceMetaParameter<T> &par,
      RealWorldRNG<T> *rng);
  ~DynamicTransferRPUDevice();

  DynamicTransferRPUDevice(const DynamicTransferRPUDevice<T> &);
  DynamicTransferRPUDevice<T> &operator=(const DynamicTransferRPUDevice<T> &);
  DynamicTransferRPUDevice(DynamicTransferRPUDevice<T> &&) noexcept;
  DynamicTransferRPUDevice<T> &operator=(DynamicTransferRPUDevice<T> &&) noexcept;

  friend void swap(DynamicTransferRPUDevice<T> &a, DynamicTransferRPUDevice<T> &b) noexcept {
    using std::swap;
    swap(
        static_cast<ChoppedTransferRPUDevice<T> &>(a),
        static_cast<ChoppedTransferRPUDevice<T> &>(b));
    swap(a.running_mean_, b.running_mean_);
    swap(a.past_mean_, b.past_mean_);
    swap(a.in_chopper_switched_, b.in_chopper_switched_);
    swap(a.count_lr_scale_, b.count_lr_scale_);

    swap(a.feedback_data_, b.feedback_data_);
    swap(a.feedback_data_idx_, b.feedback_data_idx_);
  }

  DynamicTransferRPUDeviceMetaParameter<T> &getPar() const override {
    return static_cast<DynamicTransferRPUDeviceMetaParameter<T> &>(SimpleRPUDevice<T>::getPar());
  };

  DynamicTransferRPUDevice<T> *clone() const override {
    return new DynamicTransferRPUDevice<T>(*this);
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

  void getDPNames(std::vector<std::string> &names) const override;
  void getDeviceParameter(T **weights, std::vector<T *> &data_ptrs) override;
  void setDeviceParameter(T **out_weights, const std::vector<T *> &data_ptrs) override;
  int getHiddenWeightsCount() const override;
  void setHiddenWeights(const std::vector<T> &data) override;
  T getPulseCountLearningRate(
      T lr, int current_m_batch, const PulsedUpdateMetaParameter<T> &up) override;

  void dumpExtra(RPU::state_t &extra, const std::string prefix) override;
  void loadExtra(const RPU::state_t &extra, const std::string prefix, bool strict) override;

  const T *getPastMean() const { return past_mean_.data(); };
  const T *getRunningMean() const { return running_mean_.data(); };

  void setCountLRScale(T count_lr_scale) { count_lr_scale_ = count_lr_scale; };
  const T getCountLRScale() const { return count_lr_scale_; };
  inline const std::vector<T> getFeedbackData() const { return feedback_data_; };
  inline uint64_t getFeedbackIdx() const { return feedback_data_idx_; };

protected:
  void populate(const DynamicTransferRPUDeviceMetaParameter<T> &par, RealWorldRNG<T> *rng);

private:
  std::vector<T> running_mean_;
  std::vector<T> past_mean_;
  std::vector<bool> in_chopper_switched_;
  T count_lr_scale_ = 1.0;

  std::vector<T> feedback_data_;
  uint64_t feedback_data_idx_ = 0;
};

} // namespace RPU
