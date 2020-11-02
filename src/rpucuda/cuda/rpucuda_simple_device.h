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

#include "cuda_math_util.h"
#include "cuda_util.h"
#include "rpu_simple_device.h"

namespace RPU {

template <typename T> class AbstractRPUDeviceCuda {
public:
  explicit AbstractRPUDeviceCuda(){};
  virtual ~AbstractRPUDeviceCuda(){};

  virtual void decayWeights(T *dev_weights, bool bias_no_decay) = 0;
  virtual void decayWeights(T *dev_weights, T alpha, bool bias_no_decay) = 0;
  virtual void diffuseWeights(T *dev_weights) = 0;
  virtual void clipWeights(T *dev_weights, T clip) = 0;
  virtual void resetCols(T *dev_weights, int start_col, int n_cols, T reset_prob) = 0;

  virtual std::vector<T> getHiddenWeights() const = 0;
  virtual void applyWeightUpdate(T *dev_weights, T *dw_and_current_weight_out) = 0;
  virtual AbstractRPUDeviceMetaParameter<T> &getPar() const = 0;
  virtual int getHiddenUpdateIdx() const { return 0; };
  virtual void setHiddenUpdateIdx(int idx){};

  virtual void populateFrom(const AbstractRPUDevice<T> &rpu_device) = 0;
  virtual DeviceUpdateType implements() const = 0;
  virtual bool isPulsedDevice() const { return false; };
  virtual AbstractRPUDeviceCuda<T> *clone() const = 0;
  std::unique_ptr<AbstractRPUDeviceCuda<T>> cloneUnique() const {
    return std::unique_ptr<AbstractRPUDeviceCuda<T>>(clone());
  };
  static AbstractRPUDeviceCuda<T> *
  createFrom(CudaContext *c, const AbstractRPUDevice<T> &rpu_device);
  static std::unique_ptr<AbstractRPUDeviceCuda<T>>
  createFromUnique(CudaContext *c, const AbstractRPUDevice<T> &rpu_device);
};

/* Simple device. Like simple RPU, a floating point baseline. It also has a storage for the device
 * meta parameters.*/
template <typename T> class SimpleRPUDeviceCuda : public AbstractRPUDeviceCuda<T> {

public:
  explicit SimpleRPUDeviceCuda(){};
  explicit SimpleRPUDeviceCuda(CudaContext *c, int x_size, int d_size);
  explicit SimpleRPUDeviceCuda(CudaContext *c, const SimpleRPUDevice<T> &other);

  ~SimpleRPUDeviceCuda(){};
  SimpleRPUDeviceCuda(const SimpleRPUDeviceCuda<T> &other);
  SimpleRPUDeviceCuda<T> &operator=(const SimpleRPUDeviceCuda<T> &other);
  SimpleRPUDeviceCuda(SimpleRPUDeviceCuda<T> &&other);
  SimpleRPUDeviceCuda<T> &operator=(SimpleRPUDeviceCuda<T> &&other);

  friend void swap(SimpleRPUDeviceCuda<T> &a, SimpleRPUDeviceCuda<T> &b) noexcept {
    using std::swap;
    swap(a.context_, b.context_);
    swap(a.x_size_, b.x_size_);
    swap(a.d_size_, b.d_size_);
    swap(a.par_storage_, b.par_storage_);
  };

  // implement abstract functions
  std::vector<T> getHiddenWeights() const override {
    std::vector<T> tmp;
    return tmp;
  };
  void decayWeights(T *dev_weights, bool bias_no_decay) override;
  void decayWeights(T *dev_weights, T alpha, bool bias_no_decay) override;
  void diffuseWeights(T *dev_weights) override;
  void clipWeights(T *dev_weights, T clip) override;
  void resetCols(T *dev_weights, int start_col, int n_cols, T reset_prob) override {
    RPU_FATAL("Not supported by simple device.");
  };
  void applyWeightUpdate(T *dev_weights, T *dw_and_current_weight_out) override;
  void
  populateFrom(const AbstractRPUDevice<T> &rpu_device) override; // need to be called by derived
  SimpleRPUDeviceMetaParameter<T> &getPar() const override {
    return static_cast<SimpleRPUDeviceMetaParameter<T> &>(*par_storage_);
  };
  DeviceUpdateType implements() const override { return this->getPar().implements(); };
  SimpleRPUDeviceCuda<T> *clone() const override { return new SimpleRPUDeviceCuda<T>(*this); }

protected:
  int x_size_ = 0;
  int d_size_ = 0;
  int size_ = 0;
  CudaContext *context_;

  // these are helpers and not copied
  void initDiffusionRnd();
  void initRndContext();
  std::unique_ptr<CudaContext> rnd_context_;
  std::unique_ptr<CudaArray<float>> dev_diffusion_nrnd_ = nullptr;

private:
  std::unique_ptr<AbstractRPUDeviceMetaParameter<T>> par_storage_;
  void initialize(CudaContext *context, int x_size, int d_size);
};

} // namespace RPU
