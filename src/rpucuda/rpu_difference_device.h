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

#include "rpu_pulsed_device.h"
#include "rpu_vector_device.h"
#include <sstream>
#include <stdio.h>

namespace RPU {

template <typename T> class DifferenceRPUDevice;

template <typename T> struct DifferenceRPUDeviceMetaParameter : VectorRPUDeviceMetaParameter<T> {

  DifferenceRPUDeviceMetaParameter(){};
  DifferenceRPUDeviceMetaParameter(const PulsedRPUDeviceMetaParameterBase<T> &dp, int n_devices = 2)
      : VectorRPUDeviceMetaParameter<T>(dp, n_devices) {
    if (n_devices != 2) {
      RPU_FATAL("Expecting exactly 2 devices");
    }
  };

  DeviceUpdateType implements() const override { return DeviceUpdateType::Difference; };

  std::string getName() const override {
    std::ostringstream ss;
    ss << "Difference(" << this->vec_par.size() << ")";
    if (this->vec_par.size() > 0) {
      ss << ":" << this->vec_par[0]->getName();
    }
    return ss.str();
  };

  void initialize() override {
    VectorRPUDeviceMetaParameter<T>::initialize();

    // different parameteter settings are not allowed because we
    // like to be able to invert. For this we mirror copy the exact
    // DP . This does not work when the specifications of the RPU
    // arrays are different.

    if (!this->vec_par.size()) {
      RPU_FATAL("Expect non-empty vec par");
    }

    this->vec_par.resize(1);
    this->appendVecPar(this->vec_par[0]->clone());

    this->update_policy = VectorDeviceUpdatePolicy::All;
    this->first_update_idx = 0;
    this->gamma_vec.clear(); // fixed
  }

  DifferenceRPUDevice<T> *createDevice(int x_size, int d_size, RealWorldRNG<T> *rng) override {
    return new DifferenceRPUDevice<T>(x_size, d_size, *this, rng);
  };

  DifferenceRPUDeviceMetaParameter<T> *clone() const override {
    return new DifferenceRPUDeviceMetaParameter<T>(*this);
  };
};

template <typename T> class DifferenceRPUDevice : public VectorRPUDevice<T> {

public:
  // constructor / destructor
  DifferenceRPUDevice(){};
  DifferenceRPUDevice(int x_size, int d_size);
  DifferenceRPUDevice(
      int x_size, int d_size, const DifferenceRPUDeviceMetaParameter<T> &par, RealWorldRNG<T> *rng);
  ~DifferenceRPUDevice(){};

  DifferenceRPUDevice(const DifferenceRPUDevice<T> &);
  DifferenceRPUDevice<T> &operator=(const DifferenceRPUDevice<T> &);
  DifferenceRPUDevice(DifferenceRPUDevice<T> &&);
  DifferenceRPUDevice<T> &operator=(DifferenceRPUDevice<T> &&);

  friend void swap(DifferenceRPUDevice<T> &a, DifferenceRPUDevice<T> &b) noexcept {
    using std::swap;
    swap(static_cast<VectorRPUDevice<T> &>(a), static_cast<VectorRPUDevice<T> &>(b));

    swap(a.g_plus_, b.g_plus_);
    swap(a.g_minus_, b.g_minus_);
    swap(a.a_indices_, b.a_indices_);
    swap(a.b_indices_, b.b_indices_);
  }

  DifferenceRPUDeviceMetaParameter<T> &getPar() const override {
    return static_cast<DifferenceRPUDeviceMetaParameter<T> &>(SimpleRPUDevice<T>::getPar());
  };

  DifferenceRPUDevice<T> *clone() const override { return new DifferenceRPUDevice<T>(*this); };
  void
  resetCols(T **weights, int start_col, int n_cols, T reset_prob, RealWorldRNG<T> &rng) override;
  bool onSetWeights(T **weights) override;

  void invert();

  // current_update_count not needed:
  void initUpdateCycle(
      T **weights, const PulsedUpdateMetaParameter<T> &up, T current_lr, int m_batch_info) override;
  void finishUpdateCycle(
      T **weights,
      const PulsedUpdateMetaParameter<T> &up,
      T current_lr,
      int m_batch_info) override{};

  void setHiddenUpdateIdx(int idx) override{};

  void doSparseUpdate(
      T **weights, int i, const int *x_signed_indices, int x_count, int d_sign, RNG<T> *rng)
      override;
  void getGIndices(int &gplus, int &gminus) const {
    gplus = g_plus_;
    gminus = g_minus_;
  };
  virtual T **getPosWeights() { return this->getWeightVec()[g_plus_]; };
  virtual T **getNegWeights() { return this->getWeightVec()[g_minus_]; };

protected:
  void populate(const DifferenceRPUDeviceMetaParameter<T> &par, RealWorldRNG<T> *rng);

private:
  inline bool isInverted() const;

  int g_plus_ = 1;
  int g_minus_ = 0;

  std::vector<int> a_indices_;
  std::vector<int> b_indices_;
};

} // namespace RPU
