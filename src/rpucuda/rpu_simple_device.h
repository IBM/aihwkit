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

#include "math_util.h"
#include "rng.h"
#include "rpu.h"
#include "rpu_pulsed_meta_parameter.h"
#include "utility_functions.h"
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace RPU {

template <typename T> class AbstractRPUDevice;

template <typename T> class SimpleRPUDevice;

// register all available devices [better to use registry at some point]
enum class DeviceUpdateType {
  Undefined,
  FloatingPoint,
  ConstantStep,
  LinearStep,
  SoftBounds,
  HiddenStep,
  ExpStep,
  Vector,
  OneSided,
  Transfer,
  BufferedTransfer,
  MixedPrec,
  MixedPrecInt,
  PowStep,
  PiecewiseStep,
  ChoppedTransfer,
  DynamicTransfer,
  SoftBoundsReference,
  PowStepReference
};

// inherit from Simple
template <typename T> struct AbstractRPUDeviceMetaParameter : SimpleMetaParameter<T> {

  virtual ~AbstractRPUDeviceMetaParameter() {}
  bool _device_parameter_mode_manual = false;
  unsigned int construction_seed = 0;
  bool _par_initialized = false;

  virtual AbstractRPUDevice<T> *createDevice(int x_size, int d_size, RealWorldRNG<T> *rng) = 0;
  std::unique_ptr<AbstractRPUDevice<T>>
  createDeviceUnique(int x_size, int d_size, RealWorldRNG<T> *rng) {
    return std::unique_ptr<AbstractRPUDevice<T>>(
        static_cast<AbstractRPUDevice<T> *>(createDevice(x_size, d_size, rng)));
  }
  virtual std::string getName() const = 0;
  virtual AbstractRPUDeviceMetaParameter<T> *clone() const = 0;
  std::unique_ptr<AbstractRPUDeviceMetaParameter<T>> cloneUnique() const {
    return std::unique_ptr<AbstractRPUDeviceMetaParameter<T>>(clone());
  }
  virtual DeviceUpdateType implements() const = 0;
  virtual void initialize() {
    _par_initialized = true;
    _device_parameter_mode_manual = false;
    this->diffusion = MAX(this->diffusion, (T)0.0);
    this->lifetime = MAX(this->lifetime, (T)0.0);
  }

  friend void
  swap(AbstractRPUDeviceMetaParameter<T> &a, AbstractRPUDeviceMetaParameter<T> &b) noexcept {
    using std::swap;
    swap(static_cast<SimpleMetaParameter<T> &>(a), static_cast<SimpleMetaParameter<T> &>(b));
    swap(a._device_parameter_mode_manual, b._device_parameter_mode_manual);
    swap(a.construction_seed, b.construction_seed);
    swap(a._par_initialized, b._par_initialized);
  }
};

// Simple Device parameter
template <typename T> struct SimpleRPUDeviceMetaParameter : AbstractRPUDeviceMetaParameter<T> {

  T reset_std = 0.0;

  std::string getName() const override { return "SimpleRPUDevice"; };
  SimpleRPUDevice<T> *createDevice(int x_size, int d_size, RealWorldRNG<T> *rng) override {
    return new SimpleRPUDevice<T>(x_size, d_size, *this, rng);
  }
  SimpleRPUDeviceMetaParameter<T> *clone() const override {
    return new SimpleRPUDeviceMetaParameter<T>(*this);
  }
  using SimpleMetaParameter<T>::print;
  void printToStream(std::stringstream &ss) const override {
    SimpleMetaParameter<T>::printToStream(ss);
    if (reset_std > (T)0.0) {
      ss << "\t reset_std: " << reset_std << std::endl;
    }
  }
  DeviceUpdateType implements() const override { return DeviceUpdateType::FloatingPoint; };

  friend void
  swap(SimpleRPUDeviceMetaParameter<T> &a, SimpleRPUDeviceMetaParameter<T> &b) noexcept {
    using std::swap;
    swap(
        static_cast<AbstractRPUDeviceMetaParameter<T> &>(a),
        static_cast<AbstractRPUDeviceMetaParameter<T> &>(b));
  }
};

template <typename T> class AbstractRPUDevice {

public:
  // constructor / destructor
  AbstractRPUDevice(){};
  virtual ~AbstractRPUDevice() = default;

  virtual AbstractRPUDevice<T> *clone() const = 0;
  std::unique_ptr<AbstractRPUDevice<T>> cloneUnique() const {
    return std::unique_ptr<AbstractRPUDevice<T>>(clone());
  };
  virtual void getDPNames(std::vector<std::string> &names) const = 0;
  virtual void getDeviceParameter(T **weights, std::vector<T *> &data_ptrs) = 0;
  virtual void setDeviceParameter(T **out_weights, const std::vector<T *> &data_ptrs) = 0;
  virtual int getHiddenWeightsCount() const = 0;
  virtual void setHiddenWeights(const std::vector<T> &data) = 0;
  virtual int getHiddenUpdateIdx() const { return 0; };
  virtual void setHiddenUpdateIdx(int idx){};
  virtual void dumpExtra(RPU::state_t &extra, const std::string prefix) = 0;
  virtual void loadExtra(const RPU::state_t &extra, const std::string prefix, bool strict) = 0;

  virtual void printDP(int x_count, int d_count) const = 0;
  void dispMetaParameter() const {
    std::stringstream ss;
    ss << "\033[0;33m";
    printToStream(ss);
    ss << "\033[0m";
    std::cout << ss.str();
  };
  virtual void printToStream(std::stringstream &ss) const = 0;
  void dispType() const {
    std::stringstream ss;
    ss << "\033[1m";
    disp(ss);
    ss << "\033[0m";
    std::cout << ss.str();
  };
  void dispInfo() const {
    this->dispMetaParameter();
    this->dispType();
  };
  virtual void disp(std::stringstream &ss) const = 0;

  virtual int getDSize() const = 0;
  virtual int getXSize() const = 0;
  virtual AbstractRPUDeviceMetaParameter<T> &getPar() const = 0;
  virtual bool isPulsedDevice() const { return false; };

  virtual void decayWeights(T **weights, bool bias_no_decay) = 0;
  virtual void decayWeights(T **weights, T alpha, bool bias_no_decay) = 0;
  virtual void driftWeights(T **weights, T time_since_last_call, RNG<T> &rng) = 0;
  virtual void diffuseWeights(T **weights, RNG<T> &rng) = 0;
  virtual void clipWeights(T **weights, T clip) = 0;
  virtual void
  resetCols(T **weights, int start_col, int n_cols, T reset_prob, RealWorldRNG<T> &rng) = 0;
  virtual bool onSetWeights(T **weights) = 0;
  virtual DeviceUpdateType implements() const = 0;
  virtual bool hasDirectUpdate() const { return false; };
  virtual bool usesUpdateParameter() const { return !hasDirectUpdate(); };
  virtual void doDirectVectorUpdate(
      T **weights,
      const T *x_input,
      const int x_inc,
      const T *d_input,
      const int d_inc,
      const T learning_rate,
      const int m_batch_info,
      const PulsedUpdateMetaParameter<T> &up) {
    RPU_NOT_IMPLEMENTED;
  };
};

/*This re-implements the floating point weight related things to
  make it compatible with the "device" framework. It is also a FP
  fallback for the other devices.*/
template <typename T> class SimpleRPUDevice : public AbstractRPUDevice<T> {

public:
  SimpleRPUDevice(){};
  explicit SimpleRPUDevice(int x_sz, int d_sz);
  explicit SimpleRPUDevice(
      int x_sz, int d_sz, const SimpleRPUDeviceMetaParameter<T> &par, RealWorldRNG<T> *rng);
  ~SimpleRPUDevice() = default;

  SimpleRPUDevice(const SimpleRPUDevice<T> &other);
  SimpleRPUDevice<T> &operator=(const SimpleRPUDevice<T> &other); // = default;
  SimpleRPUDevice(SimpleRPUDevice<T> &&other);                    // = default;
  SimpleRPUDevice<T> &operator=(SimpleRPUDevice<T> &&other);      // = default;

  friend void swap(SimpleRPUDevice<T> &a, SimpleRPUDevice<T> &b) noexcept {
    using std::swap;
    swap(a.par_storage_, b.par_storage_);
    swap(a.size_, b.size_);
    swap(a.x_size_, b.x_size_);
    swap(a.d_size_, b.d_size_);
    swap(a.wdrifter_, b.wdrifter_);
  }

  SimpleRPUDevice<T> *clone() const override { return new SimpleRPUDevice<T>(*this); }

  void getDPNames(std::vector<std::string> &names) const override { names.clear(); };
  void getDeviceParameter(T **weights, std::vector<T *> &data_ptrs) override{};
  void setDeviceParameter(T **out_weights, const std::vector<T *> &data_ptrs) override{};
  int getHiddenWeightsCount() const override { return 0; };
  void setHiddenWeights(const std::vector<T> &data) override{};
  void printDP(int x_count, int d_count) const override{};
  void printToStream(std::stringstream &ss) const override { this->getPar().printToStream(ss); };
  void disp(std::stringstream &ss) const override {
    ss << "Device " << this->getPar().getName() << " [" << this->x_size_ << "," << this->d_size_
       << "]\n";
  };

  int getDSize() const override { return d_size_; };
  int getXSize() const override { return x_size_; };

  SimpleRPUDeviceMetaParameter<T> &getPar() const override {
    return static_cast<SimpleRPUDeviceMetaParameter<T> &>(*par_storage_);
  };

  void decayWeights(T **weights, bool bias_no_decay) override;
  void decayWeights(T **weights, T alpha, bool bias_no_decay) override;
  void driftWeights(T **weights, T time_since_last_call, RNG<T> &rng) override;
  void diffuseWeights(T **weights, RNG<T> &rng) override;
  void clipWeights(T **weights, T clip) override;
  bool onSetWeights(T **weights) override { return false; };
  void
  resetCols(T **weights, int start_col, int n_cols, T reset_prob, RealWorldRNG<T> &rng) override;

  DeviceUpdateType implements() const override { return this->getPar().implements(); };

  inline const WeightDrifter<T> *getWDrifter() const {
    if (hasWDrifter()) {
      return &*wdrifter_;
    } else {
      return nullptr;
    }
  };
  inline bool hasWDrifter() const { return wdrifter_ != nullptr; };
  void dumpExtra(RPU::state_t &extra, const std::string prefix) override;
  void loadExtra(const RPU::state_t &extra, const std::string prefix, bool strict) override;

protected:
  void populate(const SimpleRPUDeviceMetaParameter<T> &p, RealWorldRNG<T> *rng);
  int x_size_ = 0;
  int d_size_ = 0;
  int size_ = 0;
  std::unique_ptr<WeightDrifter<T>> wdrifter_ = nullptr;

private:
  void initialize(int x_sz, int d_sz);
  std::unique_ptr<AbstractRPUDeviceMetaParameter<T>> par_storage_ = nullptr;
};

} // namespace RPU
