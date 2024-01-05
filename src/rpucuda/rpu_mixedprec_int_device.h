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

template <typename T> class MixedPrecIntRPUDevice;

/* Defines mixed prec device where the Chi matrix is in integer precision.

   Similar to the mixed prec device, outer-product update is computed
   in digital in reduced precision (e.g. 3 bins for d only) and thus
   highly sparse (internally int16 is used). The update is stored in a
   Chi matrix in digital.

   However, only integer multiplication are done for storing the Chi
   matrix using a momentum-based quantization approach.

 */

template <typename T>
struct MixedPrecIntRPUDeviceMetaParameter : MixedPrecRPUDeviceBaseMetaParameter<T> {

  int n_x_bins = 5; // should be odd, otherwise a bin is added
  int n_d_bins = 3;

  T momentum_chi = (T)0.0; // momentum for Chi
  T momentum_nm = (T)0.9;  // momentum for d and x binning

  bool stoc_round_d = false;
  bool stoc_round_x = false;

  MixedPrecIntRPUDeviceMetaParameter() = default;
  ~MixedPrecIntRPUDeviceMetaParameter() = default;

  friend void swap(
      MixedPrecIntRPUDeviceMetaParameter<T> &a, MixedPrecIntRPUDeviceMetaParameter<T> &b) noexcept {
    using std::swap;
    swap(
        static_cast<MixedPrecRPUDeviceBaseMetaParameter<T> &>(a),
        static_cast<MixedPrecRPUDeviceBaseMetaParameter<T> &>(b));

    swap(a.n_d_bins, b.n_d_bins);
    swap(a.n_x_bins, b.n_x_bins);
    swap(a.momentum_chi, b.momentum_chi);
    swap(a.momentum_nm, b.momentum_nm);
    swap(a.stoc_round_x, b.stoc_round_x);
    swap(a.stoc_round_d, b.stoc_round_d);
  }

  std::string getName() const override {
    std::ostringstream ss;
    if (!this->device_par) {
      ss << " MixedPrecInt[UNDEFINED]";
    } else {
      ss << "MixedPrecInt[" << this->device_par->getName() << "]";
    }
    return ss.str();
  };

  MixedPrecIntRPUDevice<T> *createDevice(int x_size, int d_size, RealWorldRNG<T> *rng) override {
    return new MixedPrecIntRPUDevice<T>(x_size, d_size, *this, rng);
  };

  MixedPrecIntRPUDeviceMetaParameter<T> *clone() const override {
    return new MixedPrecIntRPUDeviceMetaParameter<T>(*this);
  };
  DeviceUpdateType implements() const override { return DeviceUpdateType::MixedPrecInt; };
  void printToStream(std::stringstream &ss) const override;
  void initialize() override;
};

template <typename T> class MixedPrecIntRPUDevice : public MixedPrecRPUDeviceBase<T> {

public:
  // constructor / destructor
  MixedPrecIntRPUDevice(){};
  MixedPrecIntRPUDevice(int x_size, int d_size);
  MixedPrecIntRPUDevice(
      int x_size,
      int d_size,
      const MixedPrecIntRPUDeviceMetaParameter<T> &par,
      RealWorldRNG<T> *rng);
  ~MixedPrecIntRPUDevice();

  MixedPrecIntRPUDevice(const MixedPrecIntRPUDevice<T> &);
  MixedPrecIntRPUDevice<T> &operator=(const MixedPrecIntRPUDevice<T> &);
  MixedPrecIntRPUDevice(MixedPrecIntRPUDevice<T> &&) noexcept;
  MixedPrecIntRPUDevice<T> &operator=(MixedPrecIntRPUDevice<T> &&) noexcept;

  friend void swap(MixedPrecIntRPUDevice<T> &a, MixedPrecIntRPUDevice<T> &b) noexcept {
    using std::swap;
    swap(static_cast<MixedPrecRPUDeviceBase<T> &>(a), static_cast<MixedPrecRPUDeviceBase<T> &>(b));

    swap(a.mx_, b.mx_);
    swap(a.md_, b.md_);
    swap(a.chi_, b.chi_);
    swap(a.qx_, b.qx_);
    swap(a.qx_index_, b.qx_index_);
  }
  void dumpExtra(RPU::state_t &extra, const std::string prefix) override;
  void loadExtra(const RPU::state_t &extra, const std::string prefix, bool strict) override;

  MixedPrecIntRPUDeviceMetaParameter<T> &getPar() const override {
    return static_cast<MixedPrecIntRPUDeviceMetaParameter<T> &>(SimpleRPUDevice<T>::getPar());
  };

  MixedPrecIntRPUDevice<T> *clone() const override { return new MixedPrecIntRPUDevice<T>(*this); };

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
  void populate(const MixedPrecIntRPUDeviceMetaParameter<T> &par, RealWorldRNG<T> *rng);

private:
  void initialize(int x_size, int d_size);
  void freeContainers();

  T md_ = -1.0f;
  T mx_ = -1.0f;
  std::vector<int16_t> qx_;
  std::vector<int16_t> qx_index_;
  int32_t **chi_ = nullptr;
};

} // namespace RPU
