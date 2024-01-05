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

#include "rng.h"
#include "rpu.h"
#include "rpu_forward_backward_pass.h"
#include "rpu_pulsed_device.h"
#include "rpu_pulsed_meta_parameter.h"
#include "rpu_weight_updater.h"
#include <iostream>
#include <memory>
#include <random>

namespace RPU {

template <typename T> struct PulsedMetaParameter;

template <typename T> class RPUPulsed : public RPUSimple<T> {

public:
  // constructor / destructor
  RPUPulsed(){}; // for move
  RPUPulsed(int x_size, int d_size);
  ~RPUPulsed();

  RPUPulsed(const RPUPulsed<T> &);
  RPUPulsed<T> &operator=(const RPUPulsed<T> &);
  RPUPulsed(RPUPulsed<T> &&) noexcept;
  RPUPulsed<T> &operator=(RPUPulsed<T> &&) noexcept;

  friend void swap(RPUPulsed<T> &a, RPUPulsed<T> &b) noexcept {
    using std::swap;
    swap(static_cast<RPUSimple<T> &>(a), static_cast<RPUSimple<T> &>(b));

    swap(a.par_, b.par_);
    swap(a.pwu_, b.pwu_);
    swap(a.rpu_device_, b.rpu_device_);
    swap(a.fb_pass_, b.fb_pass_);
  }

  void populateParameter(PulsedMetaParameter<T> *p, AbstractRPUDeviceMetaParameter<T> *dp);

  // overloaded methods
  void decayWeights(bool bias_no_decay) override;
  void decayWeights(T alpha, bool bias_no_decay) override;
  void driftWeights(T time_since_last_call) override;
  void diffuseWeights() override;
  void diffuseWeightsPink() override;
  void clipWeights(T clip) override;
  void clipWeights(const WeightClipParameter &wclpar) override;
  void remapWeights(const WeightRemapParameter &wrmpar, T *scales, T *biases = nullptr) override;
  bool swaWeights(
      const WeightRemapParameter &wrmpar,
      T *swa_weights,
      uint64_t iter,
      T *scales = nullptr,
      T *biases = nullptr) override;
  void resetCols(int start_col, int n_cols, T reset_prob) override;

  void updateVectorWithCounts(
      const T *x_input,
      const T *d_input,
      int x_inc,
      int d_inc,
      uint32_t *x_counts32,
      uint32_t *d_counts32);

  void getWeightsReal(T *weightsptr) override;
  void setWeightsReal(const T *weightsptr, int n_loops = 25) override;
  void setWeightsUniformRandom(T min_value, T max_value) override;
  void setWeights(const T *weightsptr) override;

  void applyWeightUpdate(T *dw_and_current_weights_out) override;

  virtual const PulsedMetaParameter<T> &getMetaPar() const { return par_; };

  void getDeviceParameterNames(std::vector<std::string> &names) const override;
  void getDeviceParameter(std::vector<T *> &data_ptrs) override;
  void setDeviceParameter(const std::vector<T *> &data_ptrs) override;
  void dumpExtra(RPU::state_t &extra, const std::string prefix) override;
  void loadExtra(const RPU::state_t &extra, const std::string prefix, bool strict) override;

  int getHiddenUpdateIdx() const override;
  void setHiddenUpdateIdx(int idx) override;

  void setLearningRate(T lrate) override;
  void printToStream(std::stringstream &ss) const override;
  void printParametersToStream(std::stringstream &ss) const override;
  void printRPUParameter(int x_count, int d_count) const;

  std::unique_ptr<AbstractRPUDevice<T>> cloneDevice();
  const AbstractRPUDevice<T> &getRPUDevice() { return *rpu_device_; };

  const FBParameter<T> &getFBParameter() const;
  void setFBParameter(FBParameter<T> &fb_pars);

protected:
  void forwardVector(const T *x_input, T *d_output, int x_inc, int d_inc, bool is_test) override;
  void backwardVector(const T *d_input, T *x_output, int d_inc = 1, int x_inc = 1) override;
  void updateVector(const T *x_input, const T *d_input, int x_inc = 1, int d_inc = 1) override;

  USE_LOOPED_MATRIX_FORWARD(T);
  USE_LOOPED_MATRIX_BACKWARD(T);
  void updateMatrix(
      const T *X_input,
      const T *D_input,
      int m_batch,
      bool x_trans = false,
      bool d_trans = false) override;

  std::unique_ptr<AbstractRPUDevice<T>> rpu_device_ = nullptr;

private:
  // helpers
  std::unique_ptr<PulsedRPUWeightUpdater<T>> pwu_ = nullptr;
  std::unique_ptr<ForwardBackwardPassIOManaged<T>> fb_pass_ = nullptr;

  PulsedMetaParameter<T> par_;
  // void initialize(PulsedMetaParameter<T> *p, int x_sz, int d_sz);
};

template <typename T> struct PulsedMetaParameter {

  friend void swap(PulsedMetaParameter<T> &a, PulsedMetaParameter<T> &b) noexcept {
    using std::swap;

    swap(a.f_io, b.f_io);
    swap(a.b_io, b.b_io);
    swap(a.up, b.up);

    swap(a._par_initialized, b._par_initialized);
  }

  IOMetaParameter<T> f_io;
  IOMetaParameter<T> b_io;

  PulsedUpdateMetaParameter<T> up;

  RPUPulsed<T> *createRPUArray(int x_size, int d_size, AbstractRPUDeviceMetaParameter<T> *dp);

  bool _par_initialized = false; // for keeping track of initialize
  void initialize(int x_size, int d_size);

  void print() const;
  void printToStream(std::stringstream &ss, bool suppress_update = false) const;
};

} // namespace RPU
