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

#include "rpu_forward_backward_pass.h"
#include "rpu_pulsed_device.h"
#include "rpu_vector_device.h"
#include "rpu_weight_updater.h"
#include <sstream>
#include <stdio.h>

namespace RPU {

template <typename T> class OneSidedRPUDevice;

template <typename T> struct OneSidedRPUDeviceMetaParameter : VectorRPUDeviceMetaParameter<T> {

  int refresh_every = 0; // refresh every x updates (ie counting single vector updates)
  bool units_in_mbatch = true;
  IOMetaParameter<T> refresh_io;           // the IO for reading out during refresh
  PulsedUpdateMetaParameter<T> refresh_up; // UP parameters for refresh
  T refresh_upper_thres = 0.75;
  T refresh_lower_thres = 0.25;
  bool copy_inverted = false; // whether to use copy inverted for second device

  OneSidedRPUDeviceMetaParameter(){};
  OneSidedRPUDeviceMetaParameter(const PulsedRPUDeviceMetaParameterBase<T> &dp, int n_devices = 2)
      : VectorRPUDeviceMetaParameter<T>(dp, n_devices) {
    if (n_devices != 2) {
      RPU_FATAL("Expecting exactly 2 devices");
    }
  };

  DeviceUpdateType implements() const override { return DeviceUpdateType::OneSided; };

  std::string getName() const override {
    std::ostringstream ss;
    ss << "OneSided(" << this->vec_par.size() << ")";
    if (this->vec_par.size() > 0) {
      ss << ":" << this->vec_par[0]->getName();
    }
    return ss.str();
  };

  bool appendVecPar(const AbstractRPUDeviceMetaParameter<T> &par) override;
  void initialize() override;

  OneSidedRPUDevice<T> *createDevice(int x_size, int d_size, RealWorldRNG<T> *rng) override {
    return new OneSidedRPUDevice<T>(x_size, d_size, *this, rng);
  };

  OneSidedRPUDeviceMetaParameter<T> *clone() const override {
    return new OneSidedRPUDeviceMetaParameter<T>(*this);
  };

  void printToStream(std::stringstream &ss) const override;
};

template <typename T> class OneSidedRPUDevice : public VectorRPUDevice<T> {

public:
  // constructor / destructor
  OneSidedRPUDevice(){};
  OneSidedRPUDevice(int x_size, int d_size);
  OneSidedRPUDevice(
      int x_size, int d_size, const OneSidedRPUDeviceMetaParameter<T> &par, RealWorldRNG<T> *rng);
  ~OneSidedRPUDevice(){};

  OneSidedRPUDevice(const OneSidedRPUDevice<T> &);
  OneSidedRPUDevice<T> &operator=(const OneSidedRPUDevice<T> &);
  OneSidedRPUDevice(OneSidedRPUDevice<T> &&);
  OneSidedRPUDevice<T> &operator=(OneSidedRPUDevice<T> &&);

  friend void swap(OneSidedRPUDevice<T> &a, OneSidedRPUDevice<T> &b) noexcept {
    using std::swap;
    swap(static_cast<VectorRPUDevice<T> &>(a), static_cast<VectorRPUDevice<T> &>(b));

    swap(a.g_plus_, b.g_plus_);
    swap(a.g_minus_, b.g_minus_);
    swap(a.a_indices_, b.a_indices_);
    swap(a.b_indices_, b.b_indices_);
    swap(a.refresh_counter_, b.refresh_counter_);
    swap(a.refresh_fb_pass_, b.refresh_fb_pass_);
    swap(a.refresh_pwu_, b.refresh_pwu_);
    swap(a.refresh_vecs_, b.refresh_vecs_);
  }

  OneSidedRPUDeviceMetaParameter<T> &getPar() const override {
    return static_cast<OneSidedRPUDeviceMetaParameter<T> &>(SimpleRPUDevice<T>::getPar());
  };

  OneSidedRPUDevice<T> *clone() const override { return new OneSidedRPUDevice<T>(*this); };
  void
  resetCols(T **weights, int start_col, int n_cols, T reset_prob, RealWorldRNG<T> &rng) override;
  bool onSetWeights(T **weights) override;

  void invert();
  virtual const T *getRefreshVecs() const { return refresh_vecs_.data(); };
  void finishUpdateCycle(
      T **weights, const PulsedUpdateMetaParameter<T> &up, T current_lr, int m_batch_info) override;

  void setHiddenUpdateIdx(int idx) override{};

  void doSparseUpdate(
      T **weights, int i, const int *x_signed_indices, int x_count, int d_sign, RNG<T> *rng)
      override;
  void doDenseUpdate(T **weights, int *coincidences, RNG<T> *rng) override;
  void getGIndices(int &gplus, int &gminus) const {
    gplus = g_plus_;
    gminus = g_minus_;
  };
  virtual T **getPosWeights() { return this->getWeightVec()[g_plus_]; };
  virtual T **getNegWeights() { return this->getWeightVec()[g_minus_]; };

  inline uint64_t getRefreshCount() const { return refresh_counter_; };
  inline const ForwardBackwardPassIOManaged<T> &getRefreshFBPass() const {
    return *refresh_fb_pass_;
  };

protected:
  void populate(const OneSidedRPUDeviceMetaParameter<T> &par, RealWorldRNG<T> *rng);

  int resetCounters(bool force = false) override;
  std::unique_ptr<ForwardBackwardPassIOManaged<T>> refresh_fb_pass_ = nullptr;
  std::unique_ptr<PulsedRPUWeightUpdater<T>> refresh_pwu_ = nullptr;
  std::vector<T> refresh_vecs_;

  inline bool
  refreshCriterion(T &wp, T &wm, T &w_max, T &w_min, T &upper_thres, T &lower_thres) const;

private:
  bool isInverted() const;
  int refreshWeights();
  void setRefreshVecs();

  int g_plus_ = 1;
  int g_minus_ = 0;
  uint64_t refresh_counter_ = 0;

  std::vector<int> a_indices_;
  std::vector<int> b_indices_;

  // temporary: no need to copy
  std::vector<T> refresh_p_tmp_;
  std::vector<T> refresh_m_tmp_;
  std::vector<T> refresh_p_vec_;
  std::vector<T> refresh_m_vec_;
  std::vector<int> coincidences_p_;
  std::vector<int> coincidences_m_;
};

} // namespace RPU
