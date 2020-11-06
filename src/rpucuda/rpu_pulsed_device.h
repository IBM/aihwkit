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

#include "math_util.h"
#include "rng.h"
#include "rpu.h"
#include "rpu_pulsed_meta_parameter.h"
#include "rpu_simple_device.h"
#include "utility_functions.h"
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace RPU {

template <typename T> class PulsedRPUDeviceBase;

template <typename T> class PulsedRPUDevice;

template <typename T> struct PulsedRPUDeviceMetaParameterBase : SimpleRPUDeviceMetaParameter<T> {

  PulsedRPUDeviceMetaParameterBase() {}

  std::string getName() const override { return "PulsedRPUDeviceParameterBase"; };
  PulsedRPUDeviceBase<T> *createDevice(int x_size, int d_size, RealWorldRNG<T> *rng) override {RPU_FATAL("Needs implementation");};
  PulsedRPUDeviceMetaParameterBase<T> *clone() const override {RPU_FATAL("Needs implementation");};
  DeviceUpdateType implements() const override { return DeviceUpdateType::Undefined; };
};

template <typename T> struct PulsedRPUDeviceMetaParameter : PulsedRPUDeviceMetaParameterBase<T> {

  bool legacy_params = false; // to not load the reset params

  T dw_min = (T)0.001;
  T dw_min_dtod = (T)0.3;
  T dw_min_std = (T)0.3; // ctoc of pulse

  T w_min = (T)-0.6;
  T w_min_dtod = (T)0.3;

  T w_max = (T)0.6;
  T w_max_dtod = (T)0.3;

  T up_down = (T)0.0;
  T up_down_dtod = (T)0.01;

  // T lifetime = 0;  from Simple
  T lifetime_dtod = (T)0.0;

  // T diffusion = 0; from Simple
  T diffusion_dtod = (T)0.0;

  bool enforce_consistency = false;
  bool perfect_bias = false;

  T corrupt_devices_prob = (T)0.0;
  T corrupt_devices_range = std::numeric_limits<T>::max();

  T reset = (T)0.0; // mean
  T reset_std = (T)0.0;
  T reset_dtod = (T)0.0;

  void printToStream(std::stringstream &ss) const override;
  using SimpleMetaParameter<T>::print;
  std::string getName() const override { return "PulsedRPUDeviceParameter"; };
  PulsedRPUDevice<T> *createDevice(int x_size, int d_size, RealWorldRNG<T> *rng) override {RPU_FATAL("Needs implementation");};
  PulsedRPUDeviceMetaParameter<T> *clone() const override {RPU_FATAL("Needs implementation");};
  DeviceUpdateType implements() const override { return DeviceUpdateType::Undefined; };
};

template <typename T> class PulsedRPUDeviceBase : public SimpleRPUDevice<T> {

public:
  // constructor / destructor
  PulsedRPUDeviceBase(){};
  explicit PulsedRPUDeviceBase(int x_sz, int d_sz) : SimpleRPUDevice<T>(x_sz, d_sz){};
  virtual ~PulsedRPUDeviceBase() = default;

  PulsedRPUDeviceBase(const PulsedRPUDeviceBase<T> &other) = default;
  PulsedRPUDeviceBase<T> &operator=(const PulsedRPUDeviceBase<T> &other) = default;
  PulsedRPUDeviceBase(PulsedRPUDeviceBase<T> &&other) = default;
  PulsedRPUDeviceBase<T> &operator=(PulsedRPUDeviceBase<T> &&other) = default;

  friend void swap(PulsedRPUDeviceBase<T> &a, PulsedRPUDeviceBase<T> &b) noexcept {
    using std::swap;
    swap(static_cast<SimpleRPUDevice<T> &>(a), static_cast<SimpleRPUDevice<T> &>(b));
  }

  virtual void copyInvertDeviceParameter(const PulsedRPUDeviceBase<T> *rpu_device) {
    RPU_FATAL("copyInvertDeviceParameter not available for this device!");
  }

  bool isPulsedDevice() const override { return true; };
  PulsedRPUDeviceBase<T> *clone() const override {RPU_FATAL("Needs implementation");};

  virtual T getDwMin() const = 0;
  void resetCols(
      T **weights, int start_col, int n_cols, T reset_prob, RealWorldRNG<T> &rng) override {RPU_FATAL("Needs implementation");};
  virtual void doSparseUpdate(
      T **weights, int i, const int *x_signed_indices, int x_count, int d_sign, RNG<T> *rng) {
    RPU_FATAL("Sparse update not available for this device!");
  };
  virtual void doDenseUpdate(T **weights, int *coincidences, RNG<T> *rng) {
    RPU_FATAL("Dense update not available for this device!");
  };
  // for Meta-devices [like vector/transfer]: called once before each update starts
  virtual void initUpdateCycle(
      T **weights, const PulsedUpdateMetaParameter<T> &up, T current_lr, int m_batch_info){};
  // called when update completed
  virtual void finishUpdateCycle(
      T **weights, const PulsedUpdateMetaParameter<T> &up, T current_lr, int m_batch_info){};

protected:
  bool checkDwMin(T dw_min) const {
    if (fabs(dw_min - this->getDwMin()) > this->getDwMin() / 2.0) {
      std::cout << "dw_min meta info might be inaccurate\n";
    }
    return true;
  };

  void populate(const SimpleRPUDeviceMetaParameter<T> &par, RealWorldRNG<T> *rng) {
    SimpleRPUDevice<T>::populate(par, rng);
  };
};

template <typename T> struct PulsedDPStruc {
  T max_bound = (T)0.0;
  T min_bound = (T)0.0;
  T scale_up = (T)0.0;
  T scale_down = (T)0.0;
  T decay_scale = (T)0.0;
  T diffusion_rate = (T)0.0;
  T reset_bias = (T)0.0;
};

template <typename T> class PulsedRPUDevice : public PulsedRPUDeviceBase<T> {

public:
  // constructor / destructor
  PulsedRPUDevice(){};
  /* populate cannot be done through constructor because parameter
     objects reside in derived. Derived populate method needs to
     make sure to call the populate of base class */

  PulsedRPUDevice(int x_size, int d_size);
  ~PulsedRPUDevice();

  PulsedRPUDevice(const PulsedRPUDevice<T> &);
  PulsedRPUDevice<T> &operator=(const PulsedRPUDevice<T> &);
  PulsedRPUDevice(PulsedRPUDevice<T> &&);
  PulsedRPUDevice<T> &operator=(PulsedRPUDevice<T> &&);

  friend void swap(PulsedRPUDevice<T> &a, PulsedRPUDevice<T> &b) noexcept {
    using std::swap;
    swap(static_cast<PulsedRPUDeviceBase<T> &>(a), static_cast<PulsedRPUDeviceBase<T> &>(b));

    swap(a.w_scale_up_, b.w_scale_up_);
    swap(a.w_scale_down_, b.w_scale_down_);
    swap(a.w_max_bound_, b.w_max_bound_);
    swap(a.w_min_bound_, b.w_min_bound_);
    swap(a.w_decay_scale_, b.w_decay_scale_);
    swap(a.w_diffusion_rate_, b.w_diffusion_rate_);
    swap(a.w_reset_bias_, b.w_reset_bias_);
    swap(a.sup_, b.sup_);

    swap(a.containers_allocated_, b.containers_allocated_);
  }

  PulsedRPUDevice<T> *clone() const override { RPU_FATAL("Needs implementation"); };

  void getDPNames(std::vector<std::string> &names) const override;
  void getDeviceParameter(std::vector<T *> &data_ptrs) const override;
  void setDeviceParameter(const std::vector<T *> &data_ptrs) override;
  void printDP(int x_count, int d_count) const override;

  inline T **getMaxBound() const { return w_max_bound_; };
  inline T **getMinBound() const { return w_min_bound_; };
  inline T **getDecayScale() const { return w_decay_scale_; };
  inline T **getDiffusionRate() const { return w_diffusion_rate_; };
  inline T **getResetBias() const { return w_reset_bias_; };
  inline T **getScaleUp() const { return w_scale_up_; };
  inline T **getScaleDown() const { return w_scale_down_; };
  inline PulsedDPStruc<T> **getDPStruc() const { return sup_; };
  PulsedRPUDeviceMetaParameter<T> &getPar() const override {
    return static_cast<PulsedRPUDeviceMetaParameter<T> &>(SimpleRPUDevice<T>::getPar());
  };

  // these are turned off for concrete devices. Only for meta [derived from Base]
  void initUpdateCycle(
      T **weights,
      const PulsedUpdateMetaParameter<T> &up,
      T current_lr,
      int m_batch_info) override final{};
  void finishUpdateCycle(
      T **weights,
      const PulsedUpdateMetaParameter<T> &up,
      T current_lr,
      int m_batch_info) override final{};

  T getDwMin() const override { return this->getPar().dw_min; };
  void decayWeights(T **weights, bool bias_no_decay) override;
  void decayWeights(T **weights, T alpha, bool bias_no_decay) override;
  void diffuseWeights(T **weights, RNG<T> &rng) override;
  void clipWeights(T **weights, T add_clip) override;
  bool onSetWeights(T **weights) override;
  void
  resetCols(T **weights, int start_col, int n_cols, T reset_prob, RealWorldRNG<T> &rng) override;
  void copyInvertDeviceParameter(const PulsedRPUDeviceBase<T> *rpu_device) override;

protected:
  void populate(const PulsedRPUDeviceMetaParameter<T> &par, RealWorldRNG<T> *rng);

  PulsedDPStruc<T> **sup_ = nullptr;

  T **w_max_bound_ = nullptr;
  T **w_min_bound_ = nullptr;
  T **w_scale_up_ = nullptr;
  T **w_scale_down_ = nullptr;
  T **w_decay_scale_ = nullptr;
  T **w_diffusion_rate_ = nullptr;
  T **w_reset_bias_ = nullptr;

private:
  void freeContainers();
  void allocateContainers();
  void initialize();
  bool containers_allocated_ = false;
};

#define BUILD_PULSED_DEVICE_CONSTRUCTORS(                                                          \
    CLASSNAME, CTOR_BODY, DTOR_BODY, COPY_BODY, MOVE_BODY, SWAP_BODY, DPNAMES_BODY, DP2V_BODY,     \
    V2DP_BODY, INVERT_COPY_BODY)                                                                   \
public:                                                                                            \
  explicit CLASSNAME(){};                                                                          \
  explicit CLASSNAME(int x_size, int d_size) : PulsedRPUDevice<T>(x_size, d_size) {                \
    initialize();                                                                                  \
  };                                                                                               \
  explicit CLASSNAME(                                                                              \
      int x_size, int d_size, const CLASSNAME##MetaParameter<T> &par, RealWorldRNG<T> *rng)        \
      : PulsedRPUDevice<T>(x_size, d_size) {                                                       \
    initialize();                                                                                  \
    populate(par, rng);                                                                            \
  };                                                                                               \
  ~CLASSNAME() {                                                                                   \
    if (initialized_) {                                                                            \
      DTOR_BODY;                                                                                   \
    }                                                                                              \
  };                                                                                               \
  CLASSNAME(const CLASSNAME<T> &other) : PulsedRPUDevice<T>(other) {                               \
    if (other.initialized_) {                                                                      \
      initialize();                                                                                \
      COPY_BODY;                                                                                   \
    }                                                                                              \
  };                                                                                               \
  CLASSNAME<T> &operator=(const CLASSNAME<T> &other) {                                             \
    CLASSNAME<T> tmp(other);                                                                       \
    swap(*this, tmp);                                                                              \
    return *this;                                                                                  \
  };                                                                                               \
  CLASSNAME(CLASSNAME<T> &&other) { *this = std::move(other); };                                   \
  CLASSNAME<T> &operator=(CLASSNAME<T> &&other) {                                                  \
    PulsedRPUDevice<T>::operator=(std::move(other));                                               \
    initialized_ = other.initialized_;                                                             \
    MOVE_BODY;                                                                                     \
    return *this;                                                                                  \
  };                                                                                               \
  friend void swap(CLASSNAME<T> &a, CLASSNAME<T> &b) noexcept {                                    \
    using std::swap;                                                                               \
    swap(static_cast<PulsedRPUDevice<T> &>(a), static_cast<PulsedRPUDevice<T> &>(b));              \
    swap(a.initialized_, b.initialized_);                                                          \
    SWAP_BODY;                                                                                     \
  };                                                                                               \
                                                                                                   \
  void copyInvertDeviceParameter(const PulsedRPUDeviceBase<T> *rpu_device) override {              \
    PulsedRPUDevice<T>::copyInvertDeviceParameter(rpu_device);                                     \
    const auto *rpu = dynamic_cast<const CLASSNAME<T> *>(rpu_device);                              \
    if (rpu == nullptr) {                                                                          \
      RPU_FATAL("Expect RPU Pulsed device");                                                       \
    };                                                                                             \
    INVERT_COPY_BODY;                                                                              \
  };                                                                                               \
                                                                                                   \
  void printToStream(std::stringstream &ss) const override { getPar().printToStream(ss); };        \
  CLASSNAME<T> *clone() const override { return new CLASSNAME<T>(*this); };                        \
                                                                                                   \
private:                                                                                           \
  void initialize() {                                                                              \
    if (!initialized_) {                                                                           \
      CTOR_BODY;                                                                                   \
      initialized_ = true;                                                                         \
    };                                                                                             \
  };                                                                                               \
  bool initialized_ = false;                                                                       \
                                                                                                   \
protected:                                                                                         \
  void populate(const CLASSNAME##MetaParameter<T> &par, RealWorldRNG<T> *rng);                     \
                                                                                                   \
public:                                                                                            \
  CLASSNAME##MetaParameter<T> &getPar() const override {                                           \
    return static_cast<CLASSNAME##MetaParameter<T> &>(SimpleRPUDevice<T>::getPar());               \
  };                                                                                               \
                                                                                                   \
  void getDPNames(std::vector<std::string> &names) const override {                                \
    PulsedRPUDevice<T>::getDPNames(names);                                                         \
    {DPNAMES_BODY};                                                                                \
  };                                                                                               \
                                                                                                   \
  void getDeviceParameter(std::vector<T *> &data_ptrs) const override {                            \
                                                                                                   \
    PulsedRPUDevice<T>::getDeviceParameter(data_ptrs);                                             \
                                                                                                   \
    std::vector<std::string> names;                                                                \
    std::vector<std::string> all_names;                                                            \
    this->getDPNames(all_names);                                                                   \
    if (all_names.size() != data_ptrs.size()) {                                                    \
      RPU_FATAL("Wrong number of arguments.");                                                     \
    }                                                                                              \
    PulsedRPUDevice<T>::getDPNames(names);                                                         \
    {DP2V_BODY};                                                                                   \
  };                                                                                               \
                                                                                                   \
  void setDeviceParameter(const std::vector<T *> &data_ptrs) override {                            \
    PulsedRPUDevice<T>::setDeviceParameter(data_ptrs);                                             \
                                                                                                   \
    std::vector<std::string> names;                                                                \
    PulsedRPUDevice<T>::getDPNames(names);                                                         \
    {V2DP_BODY};                                                                                   \
  }

#define BUILD_PULSED_DEVICE_META_PARAMETER(DEVICENAME, IMPLEMENTS, PAR_BODY, PRINT_BODY, ADD)      \
  template <typename T>                                                                            \
  struct DEVICENAME##RPUDeviceMetaParameter : public PulsedRPUDeviceMetaParameter<T> {             \
    PAR_BODY                                                                                       \
                                                                                                   \
    std::string getName() const override { return #DEVICENAME; };                                  \
    DeviceUpdateType implements() const override { return IMPLEMENTS; };                           \
                                                                                                   \
    DEVICENAME##RPUDevice<T> *                                                                     \
    createDevice(int x_size, int d_size, RealWorldRNG<T> *rng) override {                          \
      return new DEVICENAME##RPUDevice<T>(x_size, d_size, *this, rng);                             \
    };                                                                                             \
                                                                                                   \
    DEVICENAME##RPUDeviceMetaParameter<T> *clone() const override {                                \
      return new DEVICENAME##RPUDeviceMetaParameter<T>(*this);                                     \
    };                                                                                             \
                                                                                                   \
    void printToStream(std::stringstream &ss) const override {                                     \
      PulsedRPUDeviceMetaParameter<T>::printToStream(ss);                                          \
      ss << "   ";                                                                                 \
      ss << #DEVICENAME << " parameter:" << std::endl;                                             \
      PRINT_BODY                                                                                   \
    };                                                                                             \
                                                                                                   \
    ADD                                                                                            \
  }

// some macros for the Update_W function
#define PULSED_UPDATE_W_LOOP(BODY)                                                                 \
  PRAGMA_SIMD                                                                                      \
  for (int jj = 0; jj < x_count; jj++) {                                                           \
    int j_signed = x_signed_indices[jj];                                                           \
    int sign = (j_signed < 0) ? -d_sign : d_sign;                                                  \
    int j = (j_signed < 0) ? -j_signed - 1 : j_signed - 1;                                         \
    { BODY; }                                                                                      \
  }

#define PULSED_UPDATE_W_LOOP_DENSE(BODY)                                                           \
  for (int j = 0; j < this->x_size_ * this->d_size_; j++) {                                        \
    int c_signed = coincidences[j];                                                                \
    if (c_signed == 0) {                                                                           \
      continue;                                                                                    \
    }                                                                                              \
    int ac = abs(c_signed);                                                                        \
    int sign = c_signed > 0 ? 1 : -1;                                                              \
    PRAGMA_SIMD                                                                                    \
    for (int i_c = 0; i_c < ac; i_c++) {                                                           \
      BODY;                                                                                        \
    }                                                                                              \
  }

} // namespace RPU
