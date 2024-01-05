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

#include "rpu_pulsed_device.h"
#include "rpu_simple_device.h"
#include <sstream>
#include <stdio.h>

namespace RPU {

template <typename T> class VectorRPUDevice;

enum class VectorDeviceUpdatePolicy { All, SingleFixed, SingleSequential, SingleRandom };

template <typename T> struct VectorRPUDeviceMetaParameter : PulsedRPUDeviceMetaParameterBase<T> {

  std::vector<std::unique_ptr<PulsedRPUDeviceMetaParameterBase<T>>> vec_par;
  bool same_context = true; // only effects cuda
  VectorDeviceUpdatePolicy update_policy = VectorDeviceUpdatePolicy::All;
  int first_update_idx = 0;
  std::vector<T> gamma_vec;

  VectorRPUDeviceMetaParameter(){};
  explicit VectorRPUDeviceMetaParameter(
      const PulsedRPUDeviceMetaParameterBase<T> &dp, int n_devices);

  VectorRPUDeviceMetaParameter(const VectorRPUDeviceMetaParameter<T> &);
  VectorRPUDeviceMetaParameter<T> &operator=(const VectorRPUDeviceMetaParameter<T> &);
  VectorRPUDeviceMetaParameter(VectorRPUDeviceMetaParameter<T> &&) noexcept;
  VectorRPUDeviceMetaParameter<T> &operator=(VectorRPUDeviceMetaParameter<T> &&) noexcept;

  friend void
  swap(VectorRPUDeviceMetaParameter<T> &a, VectorRPUDeviceMetaParameter<T> &b) noexcept {
    using std::swap;
    swap(
        static_cast<PulsedRPUDeviceMetaParameterBase<T> &>(a),
        static_cast<PulsedRPUDeviceMetaParameterBase<T> &>(b));

    swap(a.vec_par, b.vec_par);
    swap(a.same_context, b.same_context);
    swap(a.update_policy, b.update_policy);
    swap(a.gamma_vec, b.gamma_vec);
    swap(a.first_update_idx, b.first_update_idx);
  }

  inline bool singleDeviceUpdate() const { return update_policy != VectorDeviceUpdatePolicy::All; }

  std::string getName() const override {
    std::ostringstream ss;
    ss << "Vector(" << vec_par.size() << ")";
    if (vec_par.size() > 0) {
      ss << ":" << vec_par[0]->getName();
    }
    return ss.str();
  };

  /* appends a parameter vector to vec_par. Returns True if successful */
  virtual bool appendVecPar(const AbstractRPUDeviceMetaParameter<T> &par);

  VectorRPUDevice<T> *createDevice(int x_size, int d_size, RealWorldRNG<T> *rng) override {
    return new VectorRPUDevice<T>(x_size, d_size, *this, rng);
  };

  VectorRPUDeviceMetaParameter<T> *clone() const override {
    return new VectorRPUDeviceMetaParameter<T>(*this);
  };
  DeviceUpdateType implements() const override { return DeviceUpdateType::Vector; };

  /* this is for convenient access in case all pars are Pulsed. Will
   return runtime error if this is not the case though */
  PulsedRPUDeviceMetaParameter<T> *operator[](size_t idx) {
    if (idx >= vec_par.size()) {
      RPU_FATAL("Exceeding vector limit.");
    }
    auto *dp = dynamic_cast<PulsedRPUDeviceMetaParameter<T> *>(&*vec_par[idx]);
    if (!dp) {
      RPU_FATAL("Expected a Pulsed Meta Parameter Class.");
    }
    return dp;
  }

  T calcWeightGranularity() const override {
    T weight_granularity = 0.0;
    for (size_t k = 0; k < vec_par.size(); k++) {
      weight_granularity += vec_par[k]->calcWeightGranularity() / (T)vec_par.size();
    }
    return weight_granularity;
  }
  T calcNumStates() const override {
    T num_states = 0.0;
    for (size_t k = 0; k < vec_par.size(); k++) {
      num_states += vec_par[k]->calcNumStates() / (T)vec_par.size();
    }
    return num_states;
  }

  void printToStream(std::stringstream &ss) const override {
    ss << this->getName();
    ss << " [update policy ";
    switch (update_policy) {
    case VectorDeviceUpdatePolicy::All:
      ss << "All";
      break;
    case VectorDeviceUpdatePolicy::SingleFixed:
      ss << "SingleFixed";
      break;
    case VectorDeviceUpdatePolicy::SingleSequential:
      ss << "SingleSequential";
      break;
    case VectorDeviceUpdatePolicy::SingleRandom:
      ss << "SingleRandom";
      break;
    default:
      ss << "UNDEFINED";
    }
    ss << " (" << first_update_idx << ")]" << std::endl;
    ss << std::endl;
    for (size_t k = 0; k < vec_par.size(); k++) {
      ss << "\t\bDevice Parameter " << k << ": " << vec_par[k]->getName() << std::endl;
      vec_par[k]->printToStream(ss);
    }
  };
};

template <typename T> class VectorRPUDevice : public PulsedRPUDeviceBase<T> {

public:
  // constructor / destructor
  VectorRPUDevice(){};
  VectorRPUDevice(int x_size, int d_size);
  VectorRPUDevice(
      int x_size, int d_size, const VectorRPUDeviceMetaParameter<T> &p, RealWorldRNG<T> *rng);
  ~VectorRPUDevice();

  VectorRPUDevice(const VectorRPUDevice<T> &);
  VectorRPUDevice<T> &operator=(const VectorRPUDevice<T> &);
  VectorRPUDevice(VectorRPUDevice<T> &&) noexcept;
  VectorRPUDevice<T> &operator=(VectorRPUDevice<T> &&) noexcept;

  friend void swap(VectorRPUDevice<T> &a, VectorRPUDevice<T> &b) noexcept {
    using std::swap;
    swap(static_cast<PulsedRPUDeviceBase<T> &>(a), static_cast<PulsedRPUDeviceBase<T> &>(b));

    swap(a.weights_vec_, b.weights_vec_);
    swap(a.rpu_device_vec_, b.rpu_device_vec_);
    swap(a.reduce_weightening_, b.reduce_weightening_);

    swap(a.current_device_idx_, b.current_device_idx_);
    swap(a.current_update_idx_, b.current_update_idx_);
    swap(a.n_devices_, b.n_devices_);
  }

  void getDPNames(std::vector<std::string> &names) const override;
  void getDeviceParameter(T **weights, std::vector<T *> &data_ptrs) override;
  void setDeviceParameter(T **out_weights, const std::vector<T *> &data_ptrs) override;
  int getHiddenWeightsCount() const override;
  void setHiddenWeights(const std::vector<T> &data) override;
  int getHiddenUpdateIdx() const override;
  void setHiddenUpdateIdx(int idx) override;
  void dumpExtra(RPU::state_t &extra, const std::string prefix) override;
  void loadExtra(const RPU::state_t &extra, const std::string prefix, bool strict) override;

  void printDP(int x_count, int d_count) const override;
  void printToStream(std::stringstream &ss) const override { this->getPar().printToStream(ss); };
  void disp(std::stringstream &ss) const override {
    ss << this->getPar().getName() << " [" << this->x_size_ << "," << this->d_size_ << "]\n";
  };

  VectorRPUDeviceMetaParameter<T> &getPar() const override {
    return static_cast<VectorRPUDeviceMetaParameter<T> &>(SimpleRPUDevice<T>::getPar());
  };

  inline const std::vector<std::unique_ptr<PulsedRPUDeviceBase<T>>> &getRpuVec() const {
    return rpu_device_vec_;
  };
  inline T ***getWeightVec() const { return weights_vec_; };
  inline const T *getReduceWeightening() const { return reduce_weightening_.data(); };
  inline int getCurrentDeviceIdx() const { return current_device_idx_; };

  VectorRPUDevice<T> *clone() const override { return new VectorRPUDevice<T>(*this); };

  void decayWeights(T **weights, bool bias_no_decay) override;
  void decayWeights(T **weights, T alpha, bool bias_no_decay) override;
  void diffuseWeights(T **weights, RNG<T> &rng) override;
  void clipWeights(T **weights, T clip) override;
  void driftWeights(T **weights, T time_since_last_call, RNG<T> &rng) override;
  void
  resetCols(T **weights, int start_col, int n_cols, T reset_prob, RealWorldRNG<T> &rng) override;

  bool onSetWeights(T **weights) override;
  void initUpdateCycle(
      T **weights,
      const PulsedUpdateMetaParameter<T> &up,
      T current_lr,
      int m_batch_info,
      const T *x_input = nullptr,
      const int x_inc = 1,
      const T *d_input = nullptr,
      const int d_inc = 1) override;
  void finishUpdateCycle(
      T **weights, const PulsedUpdateMetaParameter<T> &up, T current_lr, int m_batch_info) override;

  void doSparseUpdate(
      T **weights, int i, const int *x_signed_indices, int x_count, int d_sign, RNG<T> *rng)
      override;
  void doDenseUpdate(T **weights, int *coincidences, RNG<T> *rng) override;

protected:
  void populate(const VectorRPUDeviceMetaParameter<T> &par, RealWorldRNG<T> *rng);
  virtual void reduceToWeights(T **weights) const;

  int n_devices_ = 0;

  std::vector<std::unique_ptr<PulsedRPUDeviceBase<T>>> rpu_device_vec_;
  T ***weights_vec_ = nullptr;
  std::vector<T> reduce_weightening_;
  int current_device_idx_ = 0;
  uint64_t current_update_idx_ = 0;
  RealWorldRNG<T> rw_rng_{0};

  virtual int resetCounters(bool force = false);

private:
  void freeContainers();
  void allocateContainers(int n_devices);
};

} // namespace RPU
