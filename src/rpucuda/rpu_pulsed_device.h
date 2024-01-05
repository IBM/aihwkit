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

  PulsedRPUDeviceMetaParameterBase() { this->drift.unsetSimpleDrift(); }

  std::string getName() const override { return "PulsedRPUDeviceParameterBase"; };
  PulsedRPUDeviceBase<T> *createDevice(int x_size, int d_size, RealWorldRNG<T> *rng) override {
    RPU_FATAL("Needs implementation");
  };
  PulsedRPUDeviceMetaParameterBase<T> *clone() const override {
    RPU_FATAL("Needs implementation");
  };
  DeviceUpdateType implements() const override { return DeviceUpdateType::Undefined; };

  virtual T calcWeightGranularity() const { RPU_FATAL("Needs implementation."); };
  virtual T calcNumStates() const { RPU_FATAL("Needs implementation."); };

  friend void
  swap(PulsedRPUDeviceMetaParameterBase<T> &a, PulsedRPUDeviceMetaParameterBase<T> &b) noexcept {
    using std::swap;
    swap(
        static_cast<SimpleRPUDeviceMetaParameter<T> &>(a),
        static_cast<SimpleRPUDeviceMetaParameter<T> &>(b));
  }
};

template <typename T> struct PulsedRPUDeviceMetaParameter : PulsedRPUDeviceMetaParameterBase<T> {

  bool legacy_params = false; // to not load the reset / drift params

  T dw_min = (T)0.001;
  T dw_min_dtod = (T)0.3;
  T dw_min_std = (T)0.3; // ctoc of pulse
  bool dw_min_dtod_log_normal = false;

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
  // T reset_std = (T)0.0;  from SimpleDevice
  T reset_dtod = (T)0.0;

  bool adjust_bounds_with_up_down = false;
  T adjust_bounds_with_up_down_dev = (T)0.0;

  T write_noise_std = (T)0.0;
  bool apply_write_noise_on_set = true;
  bool count_pulses = false; // whether to count the pulses. Some runtime penalty

  void printToStream(std::stringstream &ss) const override;
  using SimpleMetaParameter<T>::print;
  std::string getName() const override { return "PulsedRPUDeviceParameter"; };
  PulsedRPUDevice<T> *createDevice(int x_size, int d_size, RealWorldRNG<T> *rng) override {
    RPU_FATAL("Needs implementation");
  };
  PulsedRPUDeviceMetaParameter<T> *clone() const override { RPU_FATAL("Needs implementation"); };
  DeviceUpdateType implements() const override { return DeviceUpdateType::Undefined; };

  virtual bool implementsWriteNoise() const { return false; }; // needs to be activated in derived
  virtual bool usesPersistentWeight() const { return write_noise_std > (T)0.0; };
  inline T getScaledWriteNoise() const { return write_noise_std * this->dw_min; };

  void initialize() override {
    PulsedRPUDeviceMetaParameterBase<T>::initialize();
    if (!implementsWriteNoise() && usesPersistentWeight()) {
      RPU_FATAL("Device does not support write noise");
    }
    reset_dtod = MAX(reset_dtod, (T)0.0);
    this->reset_std = MAX(this->reset_std, (T)0.0);
    reset = MAX(reset, (T)0.0);
  };
};

template <typename T> class PulsedRPUDeviceBase : public SimpleRPUDevice<T> {

public:
  // constructor / destructor
  PulsedRPUDeviceBase(){};
  explicit PulsedRPUDeviceBase(int x_sz, int d_sz) : SimpleRPUDevice<T>(x_sz, d_sz){};
  virtual ~PulsedRPUDeviceBase() = default;

  PulsedRPUDeviceBase(const PulsedRPUDeviceBase<T> &other) = default;
  PulsedRPUDeviceBase<T> &operator=(const PulsedRPUDeviceBase<T> &other) = default;
  PulsedRPUDeviceBase(PulsedRPUDeviceBase<T> &&other) noexcept = default;
  PulsedRPUDeviceBase<T> &operator=(PulsedRPUDeviceBase<T> &&other) noexcept = default;

  friend void swap(PulsedRPUDeviceBase<T> &a, PulsedRPUDeviceBase<T> &b) noexcept {
    using std::swap;
    swap(static_cast<SimpleRPUDevice<T> &>(a), static_cast<SimpleRPUDevice<T> &>(b));
    swap(a.weight_granularity_, b.weight_granularity_);
    swap(a.num_states_, b.num_states_);
  }

  virtual void copyInvertDeviceParameter(const PulsedRPUDeviceBase<T> *rpu_device) {
    RPU_FATAL("copyInvertDeviceParameter not available for this device!");
  }

  bool isPulsedDevice() const override { return true; };
  PulsedRPUDeviceBase<T> *clone() const override { RPU_FATAL("Needs implementation"); };

  void
  resetCols(T **weights, int start_col, int n_cols, T reset_prob, RealWorldRNG<T> &rng) override {
    RPU_FATAL("Needs implementation");
  };
  virtual void doSparseUpdate(
      T **weights, int i, const int *x_signed_indices, int x_count, int d_sign, RNG<T> *rng) {
    RPU_FATAL("Sparse update not available for this device!");
  };
  virtual void doDenseUpdate(T **weights, int *coincidences, RNG<T> *rng) {
    RPU_FATAL("Dense update not available for this device!");
  };
  // for Meta-devices [like vector/transfer]: called once before each update starts
  virtual void initUpdateCycle(
      T **weights,
      const PulsedUpdateMetaParameter<T> &up,
      T current_lr,
      int m_batch_info,
      const T *x_input = nullptr,
      const int x_inc = 1,
      const T *d_input = nullptr,
      const int d_inc = 1){};
  // called when update completed
  virtual void finishUpdateCycle(
      T **weights, const PulsedUpdateMetaParameter<T> &up, T current_lr, int m_batch_info){};

  inline T getWeightGranularity() const { return weight_granularity_; };
  inline T getNumStates() const { return num_states_; };
  virtual T getPulseCountLearningRate(
      T learning_rate, int current_m_batch, const PulsedUpdateMetaParameter<T> &up) {
    UNUSED(up);
    UNUSED(current_m_batch);
    return learning_rate;
  };

  /* called from the weight updater before the call to
     initUpdateCycle. Can be used to do some additional computation on
     the input */
  virtual void
  initWithUpdateInput(const T *x_input, const int x_inc, const T *d_input, const int d_inc){};

  void dumpExtra(RPU::state_t &extra, const std::string prefix) override {
    SimpleRPUDevice<T>::dumpExtra(extra, prefix);

    RPU::state_t state;
    RPU::insert(state, "num_states", num_states_);
    RPU::insert(state, "weight_granularity", weight_granularity_);

    RPU::insertWithPrefix(extra, state, prefix);
  };

  void loadExtra(const RPU::state_t &extra, const std::string prefix, bool strict) override {
    SimpleRPUDevice<T>::loadExtra(extra, prefix, strict);

    auto state = RPU::selectWithPrefix(extra, prefix);

    RPU::load(state, "num_states", num_states_, strict);
    RPU::load(state, "weight_granularity", weight_granularity_, strict);
  };

protected:
  inline void setWeightGranularity(T weight_granularity) {
    weight_granularity_ = weight_granularity;
  };
  inline void setNumStates(T num_states) { num_states_ = num_states; };

  void populate(const PulsedRPUDeviceMetaParameterBase<T> &par, RealWorldRNG<T> *rng) {
    SimpleRPUDevice<T>::populate(par, rng);
    setWeightGranularity(par.calcWeightGranularity());
    setNumStates(par.calcNumStates());
  };

private:
  T weight_granularity_ = 0.0;
  T num_states_ = 0.0;
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
    swap(a.w_persistent_, b.w_persistent_);
    swap(a.w_reset_bias_, b.w_reset_bias_);

    swap(a.containers_allocated_, b.containers_allocated_);
  }

  PulsedRPUDevice<T> *clone() const override { RPU_FATAL("Needs implementation"); };

  void getDPNames(std::vector<std::string> &names) const override;
  void getDeviceParameter(T **weights, std::vector<T *> &data_ptrs) override;
  void setDeviceParameter(T **out_weights, const std::vector<T *> &data_ptrs) override;
  void printDP(int x_count, int d_count) const override;
  int getHiddenWeightsCount() const override;
  void setHiddenWeights(const std::vector<T> &data) override;

  inline T **getPersistentWeights() const { return w_persistent_; };
  inline T **getMaxBound() const { return w_max_bound_; };
  inline T **getMinBound() const { return w_min_bound_; };
  inline T **getDecayScale() const { return w_decay_scale_; };
  inline T **getDiffusionRate() const { return w_diffusion_rate_; };
  inline T **getResetBias() const { return w_reset_bias_; };
  inline T **getScaleUp() const { return w_scale_up_; };
  inline T **getScaleDown() const { return w_scale_down_; };
  PulsedRPUDeviceMetaParameter<T> &getPar() const override {
    return static_cast<PulsedRPUDeviceMetaParameter<T> &>(SimpleRPUDevice<T>::getPar());
  };

  /* Note: In case of persistent data these weight methods below will
     affect the perstistent state and ALL apparent weight elements
     will be re-drawn with noise (even if they were not
     e.g. clipped) */

  void decayWeights(T **weights, bool bias_no_decay) override;
  void decayWeights(T **weights, T alpha, bool bias_no_decay) override;
  void driftWeights(T **weights, T time_since_last_call, RNG<T> &rng) override;
  void diffuseWeights(T **weights, RNG<T> &rng) override;
  void clipWeights(T **weights, T add_clip) override;
  bool onSetWeights(T **weights) override;
  void
  resetCols(T **weights, int start_col, int n_cols, T reset_prob, RealWorldRNG<T> &rng) override;
  virtual void resetAtIndices(T **weights, std::vector<int> x_major_indices, RealWorldRNG<T> &rng);
  void copyInvertDeviceParameter(const PulsedRPUDeviceBase<T> *rpu_device) override;

  using PulsedRPUDeviceBase<T>::dumpExtra;
  using PulsedRPUDeviceBase<T>::loadExtra;

protected:
  void populate(const PulsedRPUDeviceMetaParameter<T> &par, RealWorldRNG<T> *rng);

  T **w_max_bound_ = nullptr;
  T **w_min_bound_ = nullptr;
  T **w_scale_up_ = nullptr;
  T **w_scale_down_ = nullptr;
  T **w_decay_scale_ = nullptr;
  T **w_diffusion_rate_ = nullptr;
  T **w_reset_bias_ = nullptr;
  T **w_persistent_ = nullptr;

  RealWorldRNG<T> write_noise_rng_{0};
  virtual void applyUpdateWriteNoise(T **weights);

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
      RPU_FATAL("Wrong device class");                                                             \
    };                                                                                             \
    INVERT_COPY_BODY;                                                                              \
  };                                                                                               \
                                                                                                   \
  void printToStream(std::stringstream &ss) const override {                                       \
    ss << "Device:" << std::endl;                                                                  \
    getPar().printToStream(ss);                                                                    \
  };                                                                                               \
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
  void getDeviceParameter(T **weights, std::vector<T *> &data_ptrs) override {                     \
                                                                                                   \
    PulsedRPUDevice<T>::getDeviceParameter(weights, data_ptrs);                                    \
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
  void setDeviceParameter(T **out_weights, const std::vector<T *> &data_ptrs) override {           \
    PulsedRPUDevice<T>::setDeviceParameter(out_weights, data_ptrs);                                \
                                                                                                   \
    std::vector<std::string> names;                                                                \
    PulsedRPUDevice<T>::getDPNames(names);                                                         \
    {V2DP_BODY};                                                                                   \
    this->onSetWeights(out_weights);                                                               \
  }

#define BUILD_PULSED_DEVICE_META_PARAMETER(                                                        \
    DEVICENAME, IMPLEMENTS, PAR_BODY, PRINT_BODY, GRANULARITY_BODY, ADD)                           \
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
    T calcWeightGranularity() const override{GRANULARITY_BODY};                                    \
    T calcNumStates() const override {                                                             \
      return (this->w_max - this->w_min) / calcWeightGranularity();                                \
    };                                                                                             \
                                                                                                   \
    void printToStream(std::stringstream &ss) const override {                                     \
      PulsedRPUDeviceMetaParameter<T>::printToStream(ss);                                          \
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
  int _total_size = this->x_size_ * this->d_size_;                                                 \
  for (int j = 0; j < _total_size; j++) {                                                          \
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
