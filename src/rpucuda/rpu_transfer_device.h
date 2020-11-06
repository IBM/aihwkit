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

#include "rpu_forward_backward_pass.h"
#include "rpu_pulsed_device.h"
#include "rpu_simple_device.h"
#include "rpu_vector_device.h"
#include "rpu_weight_updater.h"
#include <sstream>
#include <stdio.h>

namespace RPU {

template <typename T> class TransferRPUDevice;

/* Defines the  transfer device.

   First device is used to update the weight gradients directly. The
   other devices use used to receive the transfer of the previous in
   the chain.

   gamma, gamma_vec: the weightening factor. Default is W = g^(m-1)*D[0] +
   g^(m-2)*D[1] + .. + 1.0*D[m-1] where m is the
   number of devices used altogether. It can be set by changing gamma_vec
   though

   n_cols_per_transfer: number of cols to transfer each transfer call

   transfer_every: determines the numbmer of cycles to pause between
   transfer. The number is given in mat-vec operations. Thus one has
   to take care to calculate the minibatch and n_patch for FC/CNN
   oneself. Note that this number is set during populate device.

   transfer_every is the number for the first transfer. All
   subsequent transfers are devided by the product of the number of
   cols of all previous transfers [thus transfer is slowed down
   geometrically].

   transfer_lr is the learning rate for the transfer. Per default it
   is same for all the tranfers. However, it can be set individually
   by using the vector transfer_lr_vec (with size n_devices-1)

 */

template <typename T> struct TransferRPUDeviceMetaParameter : VectorRPUDeviceMetaParameter<T> {

  T gamma = (T)0.0; // weightening factor. [gamma vec optionally inherited from vector]
  T transfer_every = (T)1.0;
  bool units_in_mbatch = false;
  int n_cols_per_transfer = 1;
  T with_reset_prob = (T)0.0;
  bool no_self_transfer = true;
  bool random_column = false;

  T transfer_lr = (T)1.0;
  std::vector<T> transfer_lr_vec;
  bool scale_transfer_lr = true;

  std::vector<T> transfer_every_vec;

  IOMetaParameter<T> transfer_io;
  PulsedUpdateMetaParameter<T> transfer_up;

  TransferRPUDeviceMetaParameter(){};
  TransferRPUDeviceMetaParameter(const PulsedRPUDeviceMetaParameterBase<T> &dp, int n_devices)
      : VectorRPUDeviceMetaParameter<T>(dp, n_devices){};

  TransferRPUDeviceMetaParameter(
      const PulsedRPUDeviceMetaParameterBase<T> &dp_fast,
      const PulsedRPUDeviceMetaParameterBase<T> &dp_rest,
      int n_total_devices);

  virtual void initializeWithSize(int x_size, int d_size);
  void initialize() override{/* do nothing */};

  inline bool fullyHidden() const { return (!gamma && this->gamma_vec.back() == 1.0); };

  std::string getName() const override {
    std::ostringstream ss;
    ss << "Transfer(" << this->vec_par.size() << ")";
    if (this->vec_par.size() > 1) {
      ss << ": " << this->vec_par[0]->getName() << " -> " << this->vec_par[1]->getName();
      ;
    }
    return ss.str();
  };

  TransferRPUDevice<T> *createDevice(int x_size, int d_size, RealWorldRNG<T> *rng) override {
    return new TransferRPUDevice<T>(x_size, d_size, *this, rng);
  };

  TransferRPUDeviceMetaParameter<T> *clone() const override {
    return new TransferRPUDeviceMetaParameter<T>(*this);
  };
  DeviceUpdateType implements() const override { return DeviceUpdateType::Transfer; };
  void printToStream(std::stringstream &ss) const override;

  virtual T getTransferLR(int to_device_idx, int from_device_idx, T current_lr) const;
};

template <typename T> class TransferRPUDevice : public VectorRPUDevice<T> {

public:
  // constructor / destructor
  TransferRPUDevice(){};
  TransferRPUDevice(int x_size, int d_size);
  TransferRPUDevice(
      int x_size, int d_size, const TransferRPUDeviceMetaParameter<T> &par, RealWorldRNG<T> *rng);
  ~TransferRPUDevice();

  TransferRPUDevice(const TransferRPUDevice<T> &);
  TransferRPUDevice<T> &operator=(const TransferRPUDevice<T> &);
  TransferRPUDevice(TransferRPUDevice<T> &&);
  TransferRPUDevice<T> &operator=(TransferRPUDevice<T> &&);

  friend void swap(TransferRPUDevice<T> &a, TransferRPUDevice<T> &b) noexcept {
    using std::swap;
    swap(static_cast<VectorRPUDevice<T> &>(a), static_cast<VectorRPUDevice<T> &>(b));

    swap(a.transfer_pwu_, b.transfer_pwu_);
    swap(a.transfer_fb_pass_, b.transfer_fb_pass_);
    swap(a.transfer_vecs_, b.transfer_vecs_);
    swap(a.transfer_every_, b.transfer_every_);
    swap(a.current_col_indices_, b.current_col_indices_);
    swap(a.fully_hidden_, b.fully_hidden_);
  }

  TransferRPUDeviceMetaParameter<T> &getPar() const override {
    return static_cast<TransferRPUDeviceMetaParameter<T> &>(SimpleRPUDevice<T>::getPar());
  };

  TransferRPUDevice<T> *clone() const override { return new TransferRPUDevice<T>(*this); };

  bool onSetWeights(T **weights) override;
  void decayWeights(T **weights, bool bias_no_decay) override;
  void decayWeights(T **weights, T alpha, bool bias_no_decay) override;
  void diffuseWeights(T **weights, RNG<T> &rng) override;
  void clipWeights(T **weights, T clip) override;

  void setDeviceParameter(const std::vector<T *> &data_ptrs) override;
  void setHiddenUpdateIdx(int idx) override{}; // ignored

  void finishUpdateCycle(
      T **weights, const PulsedUpdateMetaParameter<T> &up, T current_lr, int m_batch_info) override;

  virtual int getTransferEvery(int from_device_idx, int m_batch) const;
  virtual void setTransferVecs(const T *transfer_vecs = nullptr);
  virtual void transfer(int to_device_idx, int from_device_idx, T current_lr);
  virtual void forwardUpdate(
      int to_device_idx,
      int from_device_idx,
      const T lr,
      const T *x_input,
      const int n_vec,
      const bool trans,
      const T reset_prob,
      const int i_col);
  virtual const T *getTransferVecs() const { return &transfer_vecs_[0]; };
  T getDwMin() const override { return this->rpu_device_vec_[0]->getDwMin(); };

  void doSparseUpdate(
      T **weights, int i, const int *x_signed_indices, int x_count, int d_sign, RNG<T> *rng)
      override;

protected:
  void populate(const TransferRPUDeviceMetaParameter<T> &par, RealWorldRNG<T> *rng);
  void reduceToWeights(T **weights) const override;
  T **getDeviceWeights(int device_idx) const;

  std::unique_ptr<ForwardBackwardPassIOManaged<T>> transfer_fb_pass_ = nullptr;
  std::unique_ptr<PulsedRPUWeightUpdater<T>> transfer_pwu_ = nullptr;
  std::vector<T> transfer_vecs_;
  std::vector<T> transfer_every_;
  std::vector<int> current_col_indices_;

  // no need to swap/copy.
  T **last_weight_ = nullptr;
  std::vector<T> transfer_tmp_;
  bool fully_hidden_ = false;

private:
};

} // namespace RPU
