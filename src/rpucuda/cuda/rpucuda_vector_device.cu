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

#include "rpu_pulsed_meta_parameter.h"
#include "rpucuda_vector_device.h"
#include <memory>

namespace RPU {

/******************************************************************************************/
/* VectorRPUDeviceCuda

   CUDA implementation of VectorRPUDevice

*/

#define LOOP_DEVICES_WITH_CONTEXTS_K(BODY)                                                         \
  if (!getPar().same_context) {                                                                    \
    this->context_->recordEvent();                                                                 \
  }                                                                                                \
  for (int k = 0; k < rpucuda_device_vec_.size(); k++) {                                           \
    if (!getPar().same_context) {                                                                  \
      context_vec_[k]->waitEvent(this->context_);                                                  \
    }                                                                                              \
    {BODY};                                                                                        \
    if (!getPar().same_context) {                                                                  \
      this->context_->recordWaitEvent(&*context_vec_[k]);                                          \
    }                                                                                              \
  }

template <typename T>
VectorRPUDeviceCuda<T>::VectorRPUDeviceCuda(CudaContextPtr c, const VectorRPUDevice<T> &rpu_device)
    : PulsedRPUDeviceCudaBase<T>(c, rpu_device.getXSize(), rpu_device.getDSize()) {
  populateFrom(rpu_device);
};

template <typename T> void VectorRPUDeviceCuda<T>::allocateContainers() {

  dev_weights_vec_ = nullptr;

  CudaContextPtr c = this->context_;
  context_vec_.clear();
  for (int k = 0; k < n_devices_; k++) {

    if (!getPar().same_context) {
      context_vec_.push_back(RPU::make_unique<CudaContext>(this->context_->getGPUId()));
      c = &*context_vec_[k];
    }
  }
  // put weights vector in one big matrix to do GEMV for reduce
  dev_weights_vec_ = RPU::make_unique<CudaArray<T>>(c, this->size_ * n_devices_);
  dev_reduce_weightening_ = RPU::make_unique<CudaArray<T>>(c, n_devices_);

  this->context_->synchronizeDevice();

  dev_weights_ptrs_.clear();
  for (int k = 0; k < n_devices_; k++) {
    dev_weights_ptrs_.push_back(dev_weights_vec_->getData() + k * this->size_);
  }
}

// copy
template <typename T>
VectorRPUDeviceCuda<T>::VectorRPUDeviceCuda(const VectorRPUDeviceCuda<T> &other)
    : PulsedRPUDeviceCudaBase<T>(other) {
  current_update_idx_ = other.current_update_idx_;
  current_device_idx_ = other.current_device_idx_;
  n_devices_ = other.n_devices_;

  allocateContainers();
  dev_weights_vec_->assign(*other.dev_weights_vec_);
  // dev_weight_ptrs and new context_vec are already allocated (NOT COPY)

  // need to clone all devices
  rpucuda_device_vec_.clear();
  for (int i = 0; i < n_devices_; i++) {
    rpucuda_device_vec_.push_back(
        std::unique_ptr<PulsedRPUDeviceCudaBase<T>>(other.rpucuda_device_vec_[i]->clone()));
  }

  dev_reduce_weightening_->assign(*other.dev_reduce_weightening_);

  this->context_->synchronizeDevice(); // possibly many streams
};

template <typename T>
VectorRPUDeviceCuda<T> &VectorRPUDeviceCuda<T>::operator=(const VectorRPUDeviceCuda<T> &other) {
  VectorRPUDeviceCuda<T> tmp(other);
  swap(*this, tmp);
  this->context_->synchronize();
  return *this;
};

template <typename T> VectorRPUDeviceCuda<T>::VectorRPUDeviceCuda(VectorRPUDeviceCuda<T> &&other) {
  *this = std::move(other);
};

template <typename T>
VectorRPUDeviceCuda<T> &VectorRPUDeviceCuda<T>::operator=(VectorRPUDeviceCuda<T> &&other) {

  PulsedRPUDeviceCudaBase<T>::operator=(std::move(other));

  n_devices_ = other.n_devices_;
  current_update_idx_ = other.current_update_idx_;
  current_device_idx_ = other.current_device_idx_;

  dev_reduce_weightening_ = std::move(other.dev_reduce_weightening_);
  dev_weights_vec_ = std::move(other.dev_weights_vec_);
  rpucuda_device_vec_ = std::move(other.rpucuda_device_vec_);
  other.rpucuda_device_vec_.clear();

  dev_weights_ptrs_ = other.dev_weights_ptrs_;
  other.dev_weights_ptrs_.clear();

  context_vec_ = std::move(other.context_vec_);
  other.context_vec_.clear();

  rw_rng_ = std::move(other.rw_rng_);
  return *this;
};

template <typename T>
void VectorRPUDeviceCuda<T>::populateFrom(const AbstractRPUDevice<T> &rpu_device_in) {

  const auto &rpu_device = dynamic_cast<const VectorRPUDevice<T> &>(rpu_device_in);
  if (&rpu_device == nullptr) {
    RPU_FATAL("populateFrom expects VectorRPUDevice.");
  }

  PulsedRPUDeviceCudaBase<T>::populateFrom(rpu_device_in); // will set sizes and par

  // populate vector device
  current_update_idx_ = 0;
  current_device_idx_ = rpu_device.getCurrentDeviceIdx();
  n_devices_ = 0;

  const auto &rpu_device_vec = rpu_device.getRpuVec();
  T ***weights_vec = rpu_device.getWeightVec();

  n_devices_ = getPar().vec_par.size();
  if (rpu_device_vec.size() != n_devices_) {
    RPU_FATAL("Vector dimension mismatch in rpu_device. Not populated?");
  }

  // just re-init
  allocateContainers();

  dev_reduce_weightening_->assign(rpu_device.getReduceWeightening());

  CudaArray<T> tmp_weights(this->context_, this->size_);
  this->context_->synchronize(); // necessary? probably not.

  rpucuda_device_vec_.clear();
  for (int k = 0; k < n_devices_; k++) {
    CudaContextPtr c;
    c = getPar().same_context ? this->context_ : &*context_vec_[k];
    rpucuda_device_vec_.push_back(
        std::unique_ptr<PulsedRPUDeviceCudaBase<T>>(static_cast<PulsedRPUDeviceCudaBase<T> *>(
            AbstractRPUDeviceCuda<T>::createFrom(c, *rpu_device_vec[k]))));
    // first transpose and copy to device.
    tmp_weights.assignTranspose(weights_vec[k][0], this->d_size_, this->x_size_);
    this->context_->synchronize();

    // need to manually copy the big memory. Just use BLAS
    RPU::math::copy(
        this->context_, tmp_weights.getSize(), tmp_weights.getData(), 1, dev_weights_ptrs_[k], 1);
    this->context_->synchronize(); // to be safe
  }

  this->context_->synchronizeDevice();
}

template <typename T>
void VectorRPUDeviceCuda<T>::dumpExtra(RPU::state_t &extra, const std::string prefix) {
  PulsedRPUDeviceCudaBase<T>::dumpExtra(extra, prefix);

  RPU::state_t state;

  for (size_t k = 0; k < rpucuda_device_vec_.size(); k++) {
    rpucuda_device_vec_[k]->dumpExtra(state, std::to_string(k));
  }
  RPU::insert(state, "dev_reduce_weightening", dev_reduce_weightening_);
  RPU::insert(state, "current_device_idx", current_device_idx_);
  RPU::insert(state, "current_update_idx", current_update_idx_);

  RPU::insertWithPrefix(extra, state, prefix);
}

template <typename T>
void VectorRPUDeviceCuda<T>::loadExtra(
    const RPU::state_t &extra, const std::string prefix, bool strict) {
  PulsedRPUDeviceCudaBase<T>::loadExtra(extra, prefix, strict);

  auto state = RPU::selectWithPrefix(extra, prefix);

  for (size_t k = 0; k < rpucuda_device_vec_.size(); k++) {
    rpucuda_device_vec_[k]->loadExtra(state, std::to_string(k), strict);
  }
  RPU::load(this->context_, state, "dev_reduce_weightening", dev_reduce_weightening_, strict);
  RPU::load(state, "current_device_idx", current_device_idx_, strict);
  RPU::load(state, "current_update_idx", current_update_idx_, strict);
}

template <typename T>
void VectorRPUDeviceCuda<T>::reduceToWeights(CudaContextPtr c, T *dev_weights) {

  RPU::math::gemv(
      c, false, this->size_, n_devices_, (T)1.0, dev_weights_vec_->getData(), this->size_,
      dev_reduce_weightening_->getData(), 1, (T)0.0, dev_weights, 1);
}

template <typename T>
void VectorRPUDeviceCuda<T>::decayWeights(T *dev_weights, T alpha, bool bias_no_decay) {

  LOOP_DEVICES_WITH_CONTEXTS_K(
      rpucuda_device_vec_[k]->decayWeights(dev_weights_ptrs_[k], alpha, bias_no_decay););
  reduceToWeights(this->context_, dev_weights);
}

template <typename T>
void VectorRPUDeviceCuda<T>::decayWeights(T *dev_weights, bool bias_no_decay) {

  LOOP_DEVICES_WITH_CONTEXTS_K(
      rpucuda_device_vec_[k]->decayWeights(dev_weights_ptrs_[k], bias_no_decay););
  reduceToWeights(this->context_, dev_weights);
}

template <typename T>
void VectorRPUDeviceCuda<T>::driftWeights(T *dev_weights, T time_since_last_call) {

  LOOP_DEVICES_WITH_CONTEXTS_K(
      rpucuda_device_vec_[k]->driftWeights(dev_weights_ptrs_[k], time_since_last_call););
  reduceToWeights(this->context_, dev_weights);
}

template <typename T> void VectorRPUDeviceCuda<T>::diffuseWeights(T *dev_weights) {

  LOOP_DEVICES_WITH_CONTEXTS_K(rpucuda_device_vec_[k]->diffuseWeights(dev_weights_ptrs_[k]););
  reduceToWeights(this->context_, dev_weights);
}

template <typename T> void VectorRPUDeviceCuda<T>::clipWeights(T *dev_weights, T clip) {

  LOOP_DEVICES_WITH_CONTEXTS_K(rpucuda_device_vec_[k]->clipWeights(dev_weights_ptrs_[k], clip););
  reduceToWeights(this->context_, dev_weights);
}

template <typename T>
void VectorRPUDeviceCuda<T>::resetCols(T *dev_weights, int start_col, int n_cols, T reset_prob) {

  LOOP_DEVICES_WITH_CONTEXTS_K(
      rpucuda_device_vec_[k]->resetCols(dev_weights_ptrs_[k], start_col, n_cols, reset_prob););
  reduceToWeights(this->context_, dev_weights);
}

template <typename T>
pwukpvec_t<T> VectorRPUDeviceCuda<T>::getUpdateKernels(
    int m_batch, int nK32, int use_bo64, bool out_trans, const PulsedUpdateMetaParameter<T> &up) {

  if (!rpucuda_device_vec_.size()) {
    RPU_FATAL("No RPUCuda device found.");
  }

  // not possible to have different device types for now. No easy
  // way to do (get rid of PWU(PK)?). However, probably not needed anyway
  DeviceUpdateType pt = rpucuda_device_vec_[0]->implements();
  for (int k = 0; k < rpucuda_device_vec_.size(); k++) {
    if (pt != rpucuda_device_vec_[k]->implements()) { // should be unique
      RPU_FATAL("No RPUCuda vector devices cannot be of different types (for now).");
    }
  }
  // just use vec device idx 1 for tuning the kernels
  pwukpvec_t<T> v =
      rpucuda_device_vec_[0]->getUpdateKernels(m_batch, nK32, use_bo64, out_trans, up);

  if (!getPar().singleDeviceUpdate()) {
    // need to adjust the number of requested states for execution on multiple streams
    int m = rpucuda_device_vec_.size();
    for (int i = 0; i < v.size(); i++) {
      v[i]->setNStates(v[i]->getNStates() * m);
    }
  }

  // CWO never supported, since it needs a direct read of the updated weights
  for (auto &kpars : v) {
    kpars->disableCWO();
  }

  DEBUG_OUT("getUpdateKernel " << v.size());
  return v;
}

template <typename T>
void VectorRPUDeviceCuda<T>::runUpdateKernel(
    pwukp_t<T> kpars,
    CudaContextPtr up_context,
    T *dev_weights,
    int m_batch,
    const BitLineMaker<T> *blm,
    const PulsedUpdateMetaParameter<T> &up,
    const T lr,
    curandState_t *dev_states,
    int one_sided,
    uint32_t *x_counts_chunk,
    uint32_t *d_counts_chunk,
    const ChoppedWeightOutput<T> *cwo) {
  DEBUG_OUT("start run update kernel.");
  DEBUG_CALL(kpars->print(););
  int m = rpucuda_device_vec_.size();
  const auto &par = getPar();

  if (cwo) {
    // this is because the weight is assumed to be updated
    // directly. Additonaly ops (like reduceToWeights) would not be
    // called correctly
    RPU_FATAL("CWO is not supported for vector devices.");
  }

  if (par.singleDeviceUpdate()) {
    // random states and context are shared, since only one is updated at a time.
    if (par.update_policy == VectorDeviceUpdatePolicy::SingleRandom) {
      current_device_idx_ = floorf((float)rw_rng_.sampleUniform() * m);
    } else if (par.update_policy == VectorDeviceUpdatePolicy::SingleSequential) {
      current_device_idx_ = ++current_device_idx_ % m;
    }

    this->rpucuda_device_vec_[current_device_idx_]->runUpdateKernel(
        kpars, up_context, this->dev_weights_ptrs_[current_device_idx_], m_batch, blm, up, lr,
        dev_states, one_sided, x_counts_chunk, d_counts_chunk);

  } else {
    // VectorDeviceUpdatePolicy::All

    if (!par.same_context) {
      up_context->recordEvent();
    }

    CudaContextPtr c = up_context;

    int n = kpars->getNStates() / m; // each device uses different random states

    for (int k = 0; k < m; k++) {
      if (!par.same_context) {
        context_vec_[k]->waitEvent(up_context->getEvent());
        c = &*context_vec_[k];
      }

      this->rpucuda_device_vec_[k]->runUpdateKernel(
          kpars, c, this->dev_weights_ptrs_[k], m_batch, blm, up, lr, dev_states + k * n, one_sided,
          x_counts_chunk, d_counts_chunk);

      if (!par.same_context) {
        up_context->recordWaitEvent(c);
      }
    }
  }

  reduceToWeights(up_context, dev_weights);
  current_update_idx_ += m_batch;
}

template <typename T> void VectorRPUDeviceCuda<T>::setHiddenUpdateIdx(int idx) {
  current_device_idx_ = idx;
}

template <typename T> int VectorRPUDeviceCuda<T>::getHiddenUpdateIdx() const {
  return current_device_idx_;
}

template <typename T> std::vector<T> VectorRPUDeviceCuda<T>::getHiddenWeights() const {
  std::vector<T> data;
  if (!n_devices_) {
    return data;
  }

  // get the current weights vec
  CudaArray<T> dev_w(this->context_, this->size_);
  std::vector<T> w_tmp(this->size_);
  std::vector<std::vector<T>> w_vec(this->n_devices_);

  for (int k = 0; k < n_devices_; k++) {
    w_vec[k].resize(this->size_);

    dev_w.assignFromDevice(
        this->dev_weights_vec_->getData() + k * this->size_); // this copies the "empty" hidden
    dev_w.copyTo(w_vec[k].data());

    for (int i = 0; i < this->size_; i++) {
      w_tmp[i] = w_vec[k][TRANSPOSE_X2D(i, this->x_size_, this->d_size_)];
    }
    for (int i = 0; i < this->size_; i++) {
      w_vec[k][i] = w_tmp[i];
    }
  }

  // copy all recursive hidden weights
  int offset = 0;
  for (int k = 0; k < n_devices_; k++) {
    std::vector<T> tmp_data = rpucuda_device_vec_[k]->getHiddenWeights();
    int m = tmp_data.size() / this->size_; // needs to be the same x_size and d_size.
    data.resize(offset + (m + 1) * this->size_);

    // first this weight vec
    for (int i = 0; i < this->size_; i++) {
      data[offset + i] = w_vec[k][i];
    }
    offset += this->size_;

    // then any other hidden
    for (int i = 0; i < m * this->size_; i++) {
      data[offset + i] = tmp_data[i];
    }
    offset += this->size_ * m;
  }

  return data;
}

template <typename T> std::vector<T> VectorRPUDeviceCuda<T>::getReduceWeightening() const {
  std::vector<T> vec;
  vec.resize(n_devices_);
  dev_reduce_weightening_->copyTo(&vec[0]);
  return vec;
}

template <typename T> std::vector<uint64_t> VectorRPUDeviceCuda<T>::getPulseCounters() const {
  std::vector<uint64_t> data;

  for (int k = 0; k < n_devices_; k++) {
    std::vector<uint64_t> tmp_data = rpucuda_device_vec_[k]->getPulseCounters();
    data.insert(data.end(), tmp_data.begin(), tmp_data.end());
  }
  return data;
}

#undef LOOP_DEVICES_WITH_CONTEXTS_K

template class VectorRPUDeviceCuda<float>;
#ifdef RPU_USE_DOUBLE
template class VectorRPUDeviceCuda<double>;
#endif
#ifdef RPU_USE_FP16
template class VectorRPUDeviceCuda<half_t>;
#endif

} // namespace RPU
