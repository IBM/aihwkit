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
#include "rpu_weight_updater.h"
#include <sstream>
#include <stdio.h>

namespace RPU {

template <typename T> class MixedPrecRPUDeviceBase;

/* Defines the mixed prec device base class.

   Here outer-product update is computed in digital and transfered to the analog.
 */

template <typename T> struct MixedPrecRPUDeviceBaseMetaParameter : SimpleRPUDeviceMetaParameter<T> {

  int transfer_every = 1;       // set once per mini-batch
  int n_rows_per_transfer = -1; // -1 means full array
  bool random_row = false;

  T granularity = 0.0; // will take dw_min from device if zero

  bool compute_sparsity = false;

  std::unique_ptr<AbstractRPUDeviceMetaParameter<T>> device_par = nullptr;

  MixedPrecRPUDeviceBaseMetaParameter() = default;
  MixedPrecRPUDeviceBaseMetaParameter(const MixedPrecRPUDeviceBaseMetaParameter<T> &);
  MixedPrecRPUDeviceBaseMetaParameter<T> &operator=(const MixedPrecRPUDeviceBaseMetaParameter<T> &);
  MixedPrecRPUDeviceBaseMetaParameter(MixedPrecRPUDeviceBaseMetaParameter<T> &&) noexcept;
  MixedPrecRPUDeviceBaseMetaParameter<T> &
  operator=(MixedPrecRPUDeviceBaseMetaParameter<T> &&) noexcept;
  ~MixedPrecRPUDeviceBaseMetaParameter() = default;

  friend void swap(
      MixedPrecRPUDeviceBaseMetaParameter<T> &a,
      MixedPrecRPUDeviceBaseMetaParameter<T> &b) noexcept {
    using std::swap;
    swap(
        static_cast<SimpleRPUDeviceMetaParameter<T> &>(a),
        static_cast<SimpleRPUDeviceMetaParameter<T> &>(b));

    swap(a.transfer_every, b.transfer_every);
    swap(a.n_rows_per_transfer, b.n_rows_per_transfer);
    swap(a.random_row, b.random_row);
    swap(a.granularity, b.granularity);
    swap(a.device_par, b.device_par);
    swap(a.compute_sparsity, b.compute_sparsity);
  }

  bool setDevicePar(const AbstractRPUDeviceMetaParameter<T> &par);

  std::string getName() const override {
    std::ostringstream ss;
    if (!device_par) {
      ss << "MixedPrec[UNDEFINED]";
    } else {
      ss << "MixedPrec[" << this->device_par->getName() << "]";
    }
    return ss.str();
  };

  void printToStream(std::stringstream &ss) const override;
  void initialize() override;
};

template <typename T> class MixedPrecRPUDeviceBase : public SimpleRPUDevice<T> {

public:
  // constructor / destructor
  MixedPrecRPUDeviceBase(){};
  MixedPrecRPUDeviceBase(int x_size, int d_size);
  virtual ~MixedPrecRPUDeviceBase(){};

  MixedPrecRPUDeviceBase(const MixedPrecRPUDeviceBase<T> &);
  MixedPrecRPUDeviceBase<T> &operator=(const MixedPrecRPUDeviceBase<T> &);
  MixedPrecRPUDeviceBase(MixedPrecRPUDeviceBase<T> &&) noexcept;
  MixedPrecRPUDeviceBase<T> &operator=(MixedPrecRPUDeviceBase<T> &&) noexcept;

  friend void swap(MixedPrecRPUDeviceBase<T> &a, MixedPrecRPUDeviceBase<T> &b) noexcept {
    using std::swap;
    swap(static_cast<SimpleRPUDevice<T> &>(a), static_cast<SimpleRPUDevice<T> &>(b));

    swap(a.rpu_device_, b.rpu_device_);
    swap(a.transfer_pwu_, b.transfer_pwu_);
    swap(a.granularity_, b.granularity_);
    swap(a.transfer_tmp_, b.transfer_tmp_);

    swap(a.current_row_index_, b.current_row_index_);
    swap(a.current_update_index_, b.current_update_index_);

    swap(a.transfer_d_vecs_, b.transfer_d_vecs_);
    swap(a.avg_sparsity_, b.avg_sparsity_);

    swap(a.rw_rng_, b.rw_rng_);
    swap(a.up_ptr_, b.up_ptr_);
  }

  MixedPrecRPUDeviceBaseMetaParameter<T> &getPar() const override {
    return static_cast<MixedPrecRPUDeviceBaseMetaParameter<T> &>(SimpleRPUDevice<T>::getPar());
  };

  void printDP(int x_count, int d_count) const override;
  void getDPNames(std::vector<std::string> &names) const override;
  void getDeviceParameter(T **weights, std::vector<T *> &data_ptrs) override;
  void setDeviceParameter(T **out_weights, const std::vector<T *> &data_ptrs) override;
  int getHiddenWeightsCount() const override;
  void setHiddenWeights(const std::vector<T> &data) override;

  void dumpExtra(RPU::state_t &extra, const std::string prefix) override;
  void loadExtra(const RPU::state_t &extra, const std::string prefix, bool strict) override;

  bool usesUpdateParameter() const override { return true; };
  bool onSetWeights(T **weights) override;

  void decayWeights(T **weights, bool bias_no_decay) override;
  void decayWeights(T **weights, T alpha, bool bias_no_decay) override;
  void diffuseWeights(T **weights, RNG<T> &rng) override;
  void clipWeights(T **weights, T clip) override;
  void driftWeights(T **weights, T time_since_last_call, RNG<T> &rng) override;
  void
  resetCols(T **weights, int start_col, int n_cols, T reset_prob, RealWorldRNG<T> &rng) override;

  inline T getAvgSparsity() const { return getPar().compute_sparsity ? avg_sparsity_ : (T)0; };
  inline T getGranularity() const { return granularity_; };

  virtual void setChi(const T *data) { RPU_NOT_IMPLEMENTED; };
  virtual void getChi(T *data) const { RPU_NOT_IMPLEMENTED; };

  inline const AbstractRPUDevice<T> &getRPUDevice() const { return *rpu_device_; };

  bool hasDirectUpdate() const override { return true; };

protected:
  void populate(const MixedPrecRPUDeviceBaseMetaParameter<T> &par, RealWorldRNG<T> *rng);
  void computeSparsity(const int kx, const int kd);
  virtual void transfer(T **weights, const T lr);
  virtual void forwardUpdate(
      T **weights,
      const T lr,
      int i_row_start,
      const T *transfer_vec,
      const int n_vec,
      const bool trans) {
    RPU_NOT_IMPLEMENTED;
  };

  void doTransfer(T **weights, const T lr, const int m_batch_info);
  void setUpPar(const PulsedUpdateMetaParameter<T> &up);
  inline void advanceUpdateCounter() { current_update_index_++; };

  std::unique_ptr<AbstractRPUDevice<T>> rpu_device_ = nullptr;
  std::unique_ptr<PulsedRPUWeightUpdater<T>> transfer_pwu_ = nullptr;
  T granularity_ = 0.0f;
  std::vector<T> transfer_tmp_;
  RealWorldRNG<T> rw_rng_{0};
  RNG<T> rng_{0};

private:
  int current_row_index_ = 0;
  int64_t current_update_index_ = 0;
  std::vector<T> transfer_d_vecs_;
  T avg_sparsity_ = 0.0f;
  const PulsedUpdateMetaParameter<T> *up_ptr_ = nullptr;
};

} // namespace RPU
