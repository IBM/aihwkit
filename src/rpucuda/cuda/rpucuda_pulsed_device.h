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

#include "cuda_math_util.h"
#include "cuda_util.h"
#include "pwu_kernel_parameter_base.h"
#include "rpu_pulsed_device.h"
#include "rpucuda_simple_device.h"

namespace RPU {

template <typename T> struct PulsedUpdateMetaParameter;

/* Base class for all devices that support pulsed update*/
template <typename T> class PulsedRPUDeviceCudaBase : public SimpleRPUDeviceCuda<T> {
public:
  explicit PulsedRPUDeviceCudaBase() = default;
  explicit PulsedRPUDeviceCudaBase(CudaContextPtr c, int x_size, int d_size)
      : SimpleRPUDeviceCuda<T>(c, x_size, d_size){};

  ~PulsedRPUDeviceCudaBase() = default;
  PulsedRPUDeviceCudaBase(const PulsedRPUDeviceCudaBase<T> &other) = default;
  PulsedRPUDeviceCudaBase<T> &operator=(const PulsedRPUDeviceCudaBase<T> &other) = default;
  PulsedRPUDeviceCudaBase(PulsedRPUDeviceCudaBase<T> &&other) = default;
  PulsedRPUDeviceCudaBase<T> &operator=(PulsedRPUDeviceCudaBase<T> &&other) = default;

  friend void swap(PulsedRPUDeviceCudaBase<T> &a, PulsedRPUDeviceCudaBase<T> &b) noexcept {
    using std::swap;
    swap(static_cast<SimpleRPUDeviceCuda<T> &>(a), static_cast<SimpleRPUDeviceCuda<T> &>(b));
    swap(a.weight_granularity_, b.weight_granularity_);
    swap(a.num_states_, b.num_states_);
  }

  void populateFrom(const AbstractRPUDevice<T> &rpu_device_in) override {
    SimpleRPUDeviceCuda<T>::populateFrom(rpu_device_in);

    const auto &rpu_device = dynamic_cast<const PulsedRPUDeviceBase<T> &>(rpu_device_in);
    if (&rpu_device == nullptr) {
      RPU_FATAL("populateFrom expects PulsedRPUDeviceBase.");
    }
    setWeightGranularity(rpu_device.getWeightGranularity());
    setNumStates(rpu_device.getNumStates());
  };

  void dumpExtra(RPU::state_t &extra, const std::string prefix) override {
    SimpleRPUDeviceCuda<T>::dumpExtra(extra, prefix);

    RPU::state_t state;
    RPU::insert(state, "num_states", num_states_);
    RPU::insert(state, "weight_granularity", weight_granularity_);

    RPU::insertWithPrefix(extra, state, prefix);
  };

  void loadExtra(const RPU::state_t &extra, const std::string prefix, bool strict) override {
    SimpleRPUDeviceCuda<T>::loadExtra(extra, prefix, strict);

    auto state = RPU::selectWithPrefix(extra, prefix);

    RPU::load(state, "num_states", num_states_, strict);
    RPU::load(state, "weight_granularity", weight_granularity_, strict);
  };

  bool isPulsedDevice() const override { return true; };
  bool hasDirectUpdate() const override { return false; };
  void doDirectUpdate(
      const T *x_input,
      const T *d_input,
      T *dev_weights,
      const T lr,
      const int m_batch,
      const bool x_trans,
      const bool d_trans,
      const T beta,
      const RPU::PulsedUpdateMetaParameter<T> &up,
      T *x_buffer = nullptr,
      T *d_buffer = nullptr) override {
    RPU_FATAL("No direct update supported with this device.");
  }

  /* Resets columns of the weights matrix to 0 (with noise reset_std)*/
  void resetCols(T *dev_weights, int start_col, int n_cols, T reset_prob) override {
    RPU_FATAL("Needs implementation");
  };

  virtual void runUpdateKernel(
      pwukp_t<T> kpars,
      CudaContextPtr c,
      T *dev_weights,
      int m_batch,
      const BitLineMaker<T> *blm,
      const PulsedUpdateMetaParameter<T> &up,
      const T lr,
      curandState_t *dev_states,
      int one_sided = 0,
      uint32_t *x_counts_chunk = nullptr,
      uint32_t *d_counts_chunk = nullptr,
      const ChoppedWeightOutput<T> *cwo = nullptr) {
    RPU_FATAL("Needs implementation");
  };

  virtual pwukpvec_t<T> getUpdateKernels(
      int m_batch, int nK32, int use_bo64, bool out_trans, const PulsedUpdateMetaParameter<T> &up) {
    RPU_FATAL("Needs implementation");
  };

  PulsedRPUDeviceCudaBase<T> *clone() const override { RPU_FATAL("Needs implementation"); };

  inline T getWeightGranularity() const { return weight_granularity_; };
  inline T getNumStates() const { return num_states_; };
  virtual T getPulseCountLearningRate(
      T learning_rate, int current_m_batch, const PulsedUpdateMetaParameter<T> &up) {
    UNUSED(up);
    UNUSED(current_m_batch);
    return learning_rate;
  };

protected:
  inline void setWeightGranularity(T weight_granularity) {
    weight_granularity_ = weight_granularity;
  };
  inline void setNumStates(T num_states) { num_states_ = num_states; };

private:
  T weight_granularity_ = 0.0;
  T num_states_ = 0.0;
};

/* Base class for all devices that do a simple pulsed update with 6 parameters*/
template <typename T> class PulsedRPUDeviceCuda : public PulsedRPUDeviceCudaBase<T> {

public:
  explicit PulsedRPUDeviceCuda(){};
  explicit PulsedRPUDeviceCuda(CudaContextPtr c, int x_size, int d_size);
  // explicit PulsedRPUDeviceCuda(CudaContextPtr  c, const PulsedRPUDevice<T> * other);

  ~PulsedRPUDeviceCuda(){};
  PulsedRPUDeviceCuda(const PulsedRPUDeviceCuda<T> &other);
  PulsedRPUDeviceCuda<T> &operator=(const PulsedRPUDeviceCuda<T> &other);
  PulsedRPUDeviceCuda(PulsedRPUDeviceCuda<T> &&other);
  PulsedRPUDeviceCuda<T> &operator=(PulsedRPUDeviceCuda<T> &&other);

  friend void swap(PulsedRPUDeviceCuda<T> &a, PulsedRPUDeviceCuda<T> &b) noexcept {
    using std::swap;
    swap(
        static_cast<PulsedRPUDeviceCudaBase<T> &>(a), static_cast<PulsedRPUDeviceCudaBase<T> &>(b));
    swap(a.dev_diffusion_rate_, b.dev_diffusion_rate_);
    swap(a.dev_reset_bias_, b.dev_reset_bias_);
    swap(a.dev_decay_scale_, b.dev_decay_scale_);
    swap(a.dev_4params_, b.dev_4params_);
    swap(a.dev_persistent_weights_, b.dev_persistent_weights_);
    swap(a.dev_neg_pulse_counter_, b.dev_neg_pulse_counter_);
    swap(a.dev_pos_pulse_counter_, b.dev_pos_pulse_counter_);
  };

  // implement abstract functions
  void decayWeights(T *dev_weights, bool bias_no_decay) override;
  void decayWeights(T *dev_weights, T alpha, bool bias_no_decay) override;
  void driftWeights(T *dev_weights, T time_since_epoch) override;
  void diffuseWeights(T *dev_weights) override;
  void clipWeights(T *dev_weights, T clip) override;
  void resetCols(T *dev_weights, int start_col, int n_cols, T reset_prob) override;
  virtual void resetAt(T *dev_weights, const char *dev_non_zero_msk);
  void applyWeightUpdate(T *dev_weights, T *dw_and_current_weight_out) override;
  void
  populateFrom(const AbstractRPUDevice<T> &rpu_device) override; // need to be called by derived
  PulsedRPUDeviceCuda<T> *clone() const override { RPU_FATAL("Needs implementations"); };

  PulsedRPUDeviceMetaParameter<T> &getPar() const override {
    return static_cast<PulsedRPUDeviceMetaParameter<T> &>(SimpleRPUDeviceCuda<T>::getPar());
  };
  std::vector<T> getHiddenWeights() const override;

  void runUpdateKernel(
      pwukp_t<T> kpars,
      CudaContextPtr c,
      T *dev_weights,
      int m_batch,
      const BitLineMaker<T> *blm,
      const PulsedUpdateMetaParameter<T> &up,
      const T lr,
      curandState_t *dev_states,
      int one_sided = 0,
      uint32_t *x_counts_chunk = nullptr,
      uint32_t *d_counts_chunk = nullptr,
      const ChoppedWeightOutput<T> *cwo = nullptr) override;

  // for interfacing with pwu_kernel
  virtual T *getGlobalParamsData() { return nullptr; };
  virtual T *get1ParamsData() { return nullptr; };
  virtual param_t *get2ParamsData() { return nullptr; };
  virtual param_t *get4ParamsData() { return dev_4params_->getData(); }
  virtual T getWeightGranularityNoise() const { return getPar().dw_min_std; };
  virtual uint64_t *getPosPulseCountData();
  virtual uint64_t *getNegPulseCountData();
  std::vector<uint64_t> getPulseCounters() const override;

  void dumpExtra(RPU::state_t &extra, const std::string prefix) override;
  void loadExtra(const RPU::state_t &extra, const std::string prefix, bool strict) override;

protected:
  virtual void applyUpdateWriteNoise(T *dev_weights);
  std::unique_ptr<CudaArray<T>> dev_persistent_weights_ = nullptr;
  std::unique_ptr<CudaArray<uint64_t>> dev_pos_pulse_counter_ = nullptr;
  std::unique_ptr<CudaArray<uint64_t>> dev_neg_pulse_counter_ = nullptr;

private:
  void initialize();

  std::unique_ptr<CudaArray<param_t>> dev_4params_ = nullptr;
  std::unique_ptr<CudaArray<T>> dev_diffusion_rate_ = nullptr;
  std::unique_ptr<CudaArray<T>> dev_decay_scale_ = nullptr;
  std::unique_ptr<CudaArray<T>> dev_reset_bias_ = nullptr;
};

#define BUILD_PULSED_DEVICE_CONSTRUCTORS_CUDA(                                                     \
    CUDACLASS, CPUCLASS, CTOR_BODY, DTOR_BODY, COPY_BODY, MOVE_BODY, SWAP_BODY, HOST_COPY_BODY)    \
public:                                                                                            \
  explicit CUDACLASS(){};                                                                          \
  explicit CUDACLASS(CudaContextPtr c, int x_size, int d_size)                                     \
      : PulsedRPUDeviceCuda<T>(c, x_size, d_size) {                                                \
    initialize();                                                                                  \
  };                                                                                               \
                                                                                                   \
  explicit CUDACLASS(CudaContextPtr c, const CPUCLASS<T> &rpu_device)                              \
      : PulsedRPUDeviceCuda<T>(c, rpu_device.getXSize(), rpu_device.getDSize()) {                  \
    initialize();                                                                                  \
    populateFrom(rpu_device);                                                                      \
  };                                                                                               \
                                                                                                   \
  ~CUDACLASS() { DTOR_BODY; };                                                                     \
                                                                                                   \
  CUDACLASS(const CUDACLASS<T> &other) : PulsedRPUDeviceCuda<T>(other) {                           \
                                                                                                   \
    initialize();                                                                                  \
    { COPY_BODY; }                                                                                 \
    this->context_->synchronize();                                                                 \
  };                                                                                               \
                                                                                                   \
  friend void swap(CUDACLASS<T> &a, CUDACLASS<T> &b) noexcept {                                    \
    using std::swap;                                                                               \
    swap(static_cast<PulsedRPUDeviceCuda<T> &>(a), static_cast<PulsedRPUDeviceCuda<T> &>(b));      \
    { SWAP_BODY; }                                                                                 \
  };                                                                                               \
                                                                                                   \
  CUDACLASS<T> &operator=(const CUDACLASS<T> &other) {                                             \
    CUDACLASS<T> tmp(other);                                                                       \
    swap(*this, tmp);                                                                              \
    this->context_->synchronize();                                                                 \
    return *this;                                                                                  \
  };                                                                                               \
                                                                                                   \
  CUDACLASS(CUDACLASS<T> &&other) { *this = std::move(other); };                                   \
                                                                                                   \
  CUDACLASS<T> &operator=(CUDACLASS<T> &&other) {                                                  \
                                                                                                   \
    PulsedRPUDeviceCuda<T>::operator=(std::move(other));                                           \
    MOVE_BODY;                                                                                     \
    return *this;                                                                                  \
  };                                                                                               \
                                                                                                   \
  void populateFrom(const AbstractRPUDevice<T> &rpu_device_in) override {                          \
    const auto &rpu_device = dynamic_cast<const CPUCLASS<T> &>(rpu_device_in);                     \
    if (&rpu_device == nullptr) {                                                                  \
      RPU_FATAL("populateFrom expects " << #CPUCLASS << ".");                                      \
    }                                                                                              \
    PulsedRPUDeviceCuda<T>::populateFrom(rpu_device);                                              \
    { HOST_COPY_BODY; }                                                                            \
    this->context_->synchronize();                                                                 \
  }                                                                                                \
                                                                                                   \
  CUDACLASS<T> *clone() const { return new CUDACLASS<T>(*this); };                                 \
                                                                                                   \
private:                                                                                           \
  void initialize() {                                                                              \
    {CTOR_BODY};                                                                                   \
    this->context_->synchronize();                                                                 \
  };                                                                                               \
                                                                                                   \
public:                                                                                            \
  CPUCLASS##MetaParameter<T> &getPar() const override {                                            \
    return static_cast<CPUCLASS##MetaParameter<T> &>(SimpleRPUDeviceCuda<T>::getPar());            \
  }

#define RPUCUDA_DEVICE_ADD_FUNCTOR_UPDATE_KERNELS(DEVICENAME, FUNCTOR, GPCOUNT)                    \
  template <typename T>                                                                            \
  pwukpvec_t<T> DEVICENAME##RPUDeviceCuda<T>::getUpdateKernels(                                    \
      int m_batch, int nK32, int use_bo64, bool out_trans,                                         \
      const PulsedUpdateMetaParameter<T> &up) {                                                    \
                                                                                                   \
    pwukpvec_t<T> v;                                                                               \
                                                                                                   \
    v.push_back(RPU::make_unique<PWUKernelParameterSingleFunctor<T, FUNCTOR, GPCOUNT>>(            \
        this->context_, this->x_size_, this->d_size_, m_batch, nK32, use_bo64, out_trans, up,      \
        getPar().getName()));                                                                      \
                                                                                                   \
    v.push_back(RPU::make_unique<PWUKernelParameterBatchFunctor<T, FUNCTOR, GPCOUNT>>(             \
        this->context_, this->x_size_, this->d_size_, m_batch, nK32, use_bo64, out_trans, up,      \
        getPar().getName()));                                                                      \
                                                                                                   \
    v.push_back(RPU::make_unique<PWUKernelParameterBatchSharedFunctor<T, FUNCTOR, GPCOUNT>>(       \
        this->context_, this->x_size_, this->d_size_, m_batch, nK32, use_bo64, out_trans, up,      \
        getPar().getName()));                                                                      \
                                                                                                   \
    return v;                                                                                      \
  }

} // namespace RPU
