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

#include "pwu_kernel_parameter.h"
#include "rpu_pulsed_meta_parameter.h"
#include "rpucuda_pulsed_device.h"
#include <memory>

#include "rpucuda_constantstep_device.h"
#include "rpucuda_difference_device.h"
#include "rpucuda_expstep_device.h"
#include "rpucuda_linearstep_device.h"
#include "rpucuda_transfer_device.h"
#include "rpucuda_vector_device.h"

namespace RPU {

/******************************************************************************************/
/* AbstractRPUDeviceCuda*/

/* static function to create the corresponing cuda device from the CPU device*/
template <typename T>
AbstractRPUDeviceCuda<T> *
AbstractRPUDeviceCuda<T>::createFrom(CudaContext *c, const AbstractRPUDevice<T> &rpu_device) {
  switch (rpu_device.getPar().implements()) {
  case DeviceUpdateType::ConstantStep:
    return new ConstantStepRPUDeviceCuda<T>(
        c, static_cast<const ConstantStepRPUDevice<T> &>(rpu_device));
  case DeviceUpdateType::LinearStep:
  case DeviceUpdateType::SoftBounds:
    return new LinearStepRPUDeviceCuda<T>(
        c, static_cast<const LinearStepRPUDevice<T> &>(rpu_device));
  case DeviceUpdateType::ExpStep:
    return new ExpStepRPUDeviceCuda<T>(c, static_cast<const ExpStepRPUDevice<T> &>(rpu_device));
  case DeviceUpdateType::Vector:
    return new VectorRPUDeviceCuda<T>(c, static_cast<const VectorRPUDevice<T> &>(rpu_device));
  case DeviceUpdateType::Difference:
    return new DifferenceRPUDeviceCuda<T>(
        c, static_cast<const DifferenceRPUDevice<T> &>(rpu_device));
  case DeviceUpdateType::Transfer:
    return new TransferRPUDeviceCuda<T>(c, static_cast<const TransferRPUDevice<T> &>(rpu_device));
  case DeviceUpdateType::FloatingPoint:
    return new SimpleRPUDeviceCuda<T>(c, static_cast<const SimpleRPUDevice<T> &>(rpu_device));
  default:
    RPU_FATAL("Pulsed device type not implemented in CUDA. Maybe not added to createFrom in "
              "rpucuda_simple_device.cu?");
  }
}

template <typename T>
std::unique_ptr<AbstractRPUDeviceCuda<T>>
AbstractRPUDeviceCuda<T>::createFromUnique(CudaContext *c, const AbstractRPUDevice<T> &rpu_device) {
  return std::unique_ptr<AbstractRPUDeviceCuda<T>>(
      AbstractRPUDeviceCuda<T>::createFrom(c, rpu_device));
}

template class AbstractRPUDeviceCuda<float>;
#ifdef RPU_USE_DOUBLE
template class AbstractRPUDeviceCuda<double>;
#endif

/******************************************************************************************/
/* SimpleRPUDeviceCuda*/

template <typename T>
void SimpleRPUDeviceCuda<T>::initialize(CudaContext *c, int x_size, int d_size) {
  context_ = c;
  x_size_ = x_size;
  d_size_ = d_size;
  size_ = d_size_ * x_size_;

  dev_diffusion_nrnd_ = nullptr;
  rnd_context_ = nullptr;

  par_storage_ = nullptr;
}

template <typename T>
SimpleRPUDeviceCuda<T>::SimpleRPUDeviceCuda(CudaContext *c, int x_size, int d_size) {
  initialize(c, x_size, d_size);
};

template <typename T>
SimpleRPUDeviceCuda<T>::SimpleRPUDeviceCuda(CudaContext *c, const SimpleRPUDevice<T> &rpu_device) {
  initialize(c, rpu_device.getXSize(), rpu_device.getDSize());
  populateFrom(rpu_device);
};

template <typename T>
SimpleRPUDeviceCuda<T>::SimpleRPUDeviceCuda(const SimpleRPUDeviceCuda<T> &other) {
  initialize(other.context_, other.x_size_, other.d_size_);
  if (other.par_storage_ != nullptr) {
    par_storage_ = other.par_storage_->cloneUnique();
  }
};

template <typename T>
SimpleRPUDeviceCuda<T> &SimpleRPUDeviceCuda<T>::operator=(const SimpleRPUDeviceCuda<T> &other) {
  SimpleRPUDeviceCuda<T> tmp(other);
  swap(*this, tmp);
  return *this;
};

template <typename T> SimpleRPUDeviceCuda<T>::SimpleRPUDeviceCuda(SimpleRPUDeviceCuda<T> &&other) {
  *this = std::move(other);
};

template <typename T>
SimpleRPUDeviceCuda<T> &SimpleRPUDeviceCuda<T>::operator=(SimpleRPUDeviceCuda<T> &&other) {

  initialize(other.context_, other.x_size_, other.d_size_);
  par_storage_ = std::move(other.par_storage_);
  return *this;
};

template <typename T>
void SimpleRPUDeviceCuda<T>::populateFrom(const AbstractRPUDevice<T> &rpu_device_in) {

  const auto &rpu_device = dynamic_cast<const SimpleRPUDevice<T> &>(rpu_device_in);
  if (&rpu_device == nullptr) {
    RPU_FATAL("populateFrom expects SimpleRPUDevice.");
  }

  initialize(context_, rpu_device.getXSize(), rpu_device.getDSize());
  par_storage_ = rpu_device_in.getPar().cloneUnique();
  context_->synchronize();
}

template <typename T>
void SimpleRPUDeviceCuda<T>::applyWeightUpdate(T *weights, T *dw_and_current_weight_out) {
  RPU::math::elemaddcopy<T>(context_, weights, dw_and_current_weight_out, x_size_ * d_size_);
}

template <typename T>
void SimpleRPUDeviceCuda<T>::decayWeights(T *weights, T alpha, bool bias_no_decay) {

  T lifetime = this->getPar().lifetime;
  T decay_rate = (lifetime > 1) ? (1.0 / lifetime) : 0.0;
  T decay_scale = 1.0 - alpha * decay_rate;

  if (decay_scale > 0.0 && decay_scale < 1.0) {
    RPU::math::scal<T>(
        context_, bias_no_decay ? MAX(size_ - d_size_, 0) : size_, decay_scale, weights, 1);
  }
}

template <typename T> void SimpleRPUDeviceCuda<T>::decayWeights(T *weights, bool bias_no_decay) {
  decayWeights(weights, 1.0, bias_no_decay);
}

template <typename T> void SimpleRPUDeviceCuda<T>::initRndContext() {

  // first time: init
  rnd_context_ = std::unique_ptr<CudaContext>(new CudaContext(context_->getGPUId()));
  rnd_context_->setRandomSeed(0);
}

template <typename T> void SimpleRPUDeviceCuda<T>::initDiffusionRnd() {

  if (rnd_context_ == nullptr) {
    initRndContext();
  }

  dev_diffusion_nrnd_ = std::unique_ptr<CudaArray<float>>(
      new CudaArray<float>(&*rnd_context_, (size_ + 31) / 32 * 32));
  rnd_context_->synchronize();
}

template <typename T> void SimpleRPUDeviceCuda<T>::diffuseWeights(T *weights) {

  T diffusion = this->getPar().diffusion;
  if (diffusion <= 0) {
    return;
  }

  if (dev_diffusion_nrnd_ == nullptr) {
    initDiffusionRnd();
    rnd_context_->randNormal(dev_diffusion_nrnd_->getData(), dev_diffusion_nrnd_->getSize());
  }

  context_->recordWaitEvent(rnd_context_->getStream());

  RPU::math::elemaddscale<T>(context_, weights, size_, dev_diffusion_nrnd_->getData(), diffusion);

  rnd_context_->recordWaitEvent(context_->getStream());
  rnd_context_->randNormal(dev_diffusion_nrnd_->getData(), dev_diffusion_nrnd_->getSize());
}

template <typename T> void SimpleRPUDeviceCuda<T>::clipWeights(T *weights, T clip) {

  if (clip >= 0) {
    RPU::math::aclip<T>(context_, weights, size_, clip);
  }
}

template class SimpleRPUDeviceCuda<float>;
#ifdef RPU_USE_DOUBLE
template class SimpleRPUDeviceCuda<double>;
#endif

} // namespace RPU
