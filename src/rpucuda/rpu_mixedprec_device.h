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
#include "rpu_mixedprec_device_base.h"
#include "rpu_pulsed_device.h"
#include "rpu_simple_device.h"
#include "rpu_weight_updater.h"
#include <sstream>
#include <stdio.h>

namespace RPU {

template <typename T> class MixedPrecRPUDevice;

/* Defines the mixed prec device.

   outer-product update is computed in digital in reduced precision (e.g. 3 bins
   for d only) and thus highly sparse. The update is stored in a Chi matrix in
   digital.

   Each mini-batch, the full Chi matrix transferred to the analog
   weights, depending on the device granularity, as suggested by
   Nandakumar et al. Front. in Neurosci. (2020).

 */

template <typename T>
struct MixedPrecRPUDeviceMetaParameter : MixedPrecRPUDeviceBaseMetaParameter<T> {

  int n_x_bins = 0;
  int n_d_bins = 0;
  bool stoc_round_d = false;
  bool stoc_round_x = false;
  T transfer_lr = 1.0;

  MixedPrecRPUDeviceMetaParameter() = default;
  ~MixedPrecRPUDeviceMetaParameter() = default;

  friend void
  swap(MixedPrecRPUDeviceMetaParameter<T> &a, MixedPrecRPUDeviceMetaParameter<T> &b) noexcept {
    using std::swap;
    swap(
        static_cast<MixedPrecRPUDeviceBaseMetaParameter<T> &>(a),
        static_cast<MixedPrecRPUDeviceBaseMetaParameter<T> &>(b));

    swap(a.n_d_bins, b.n_d_bins);
    swap(a.n_x_bins, b.n_x_bins);
    swap(a.transfer_lr, b.transfer_lr);
    swap(a.stoc_round_x, b.stoc_round_x);
    swap(a.stoc_round_d, b.stoc_round_d);
  }

  std::string getName() const override {
    std::ostringstream ss;
    if (!this->device_par) {
      ss << "MixedPrec[UNDEFINED]";
    } else {
      ss << "MixedPrec[" << this->device_par->getName() << "]";
    }
    return ss.str();
  };

  MixedPrecRPUDevice<T> *createDevice(int x_size, int d_size, RealWorldRNG<T> *rng) override {
    return new MixedPrecRPUDevice<T>(x_size, d_size, *this, rng);
  };

  MixedPrecRPUDeviceMetaParameter<T> *clone() const override {
    return new MixedPrecRPUDeviceMetaParameter<T>(*this);
  };
  DeviceUpdateType implements() const override { return DeviceUpdateType::MixedPrec; };
  void printToStream(std::stringstream &ss) const override;
  void initialize() override;
};

template <typename T> class MixedPrecRPUDevice : public MixedPrecRPUDeviceBase<T> {

public:
  // constructor / destructor
  MixedPrecRPUDevice(int x_size, int d_size);
  MixedPrecRPUDevice(
      int x_size, int d_size, const MixedPrecRPUDeviceMetaParameter<T> &par, RealWorldRNG<T> *rng);
  ~MixedPrecRPUDevice();

  MixedPrecRPUDevice(const MixedPrecRPUDevice<T> &);
  MixedPrecRPUDevice<T> &operator=(const MixedPrecRPUDevice<T> &);
  MixedPrecRPUDevice(MixedPrecRPUDevice<T> &&) noexcept;
  MixedPrecRPUDevice<T> &operator=(MixedPrecRPUDevice<T> &&) noexcept;

  friend void swap(MixedPrecRPUDevice<T> &a, MixedPrecRPUDevice<T> &b) noexcept {
    using std::swap;
    swap(static_cast<MixedPrecRPUDeviceBase<T> &>(a), static_cast<MixedPrecRPUDeviceBase<T> &>(b));

    swap(a.chi_, b.chi_);
    swap(a.qx_, b.qx_);
    swap(a.qd_, b.qd_);
    swap(a.qx_index_, b.qx_index_);
  }

  MixedPrecRPUDeviceMetaParameter<T> &getPar() const override {
    return static_cast<MixedPrecRPUDeviceMetaParameter<T> &>(SimpleRPUDevice<T>::getPar());
  };

  MixedPrecRPUDevice<T> *clone() const override { return new MixedPrecRPUDevice<T>(*this); };

  bool onSetWeights(T **weights) override;
  void getChi(T *data) const override;
  void setChi(const T *data) override;

  void forwardUpdate(
      T **weights,
      const T lr,
      int i_row_start,
      const T *transfer_vec,
      const int n_vec,
      const bool trans) override;

  void doDirectVectorUpdate(
      T **weights,
      const T *x_input,
      const int x_inc,
      const T *d_input,
      const int d_inc,
      const T learning_rate,
      const int m_batch_info,
      const PulsedUpdateMetaParameter<T> &up) override;

protected:
  void populate(const MixedPrecRPUDeviceMetaParameter<T> &par, RealWorldRNG<T> *rng);

private:
  void initialize(int x_size, int d_size);
  void freeContainers();

  // handled in base
  T **chi_ = nullptr;

  // temporary
  std::vector<T> qx_;
  std::vector<T> qd_;
  std::vector<int> qx_index_;
};

} // namespace RPU
