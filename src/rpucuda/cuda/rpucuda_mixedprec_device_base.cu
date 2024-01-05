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

#include "cuda_math_util.h"
#include "io_iterator.h"
#include "rpu_cub.h"
#include "rpu_pulsed_meta_parameter.h"
#include "rpucuda_mixedprec_device.h"
#include <memory>

namespace RPU {

/******************************************************************************************/
/* MixedPrecRPUDeviceBaseCuda

   CUDA implementation of MixedPrecRPUDeviceBase

*/

template <typename T>
MixedPrecRPUDeviceBaseCuda<T>::MixedPrecRPUDeviceBaseCuda(CudaContextPtr c, int x_size, int d_size)
    : SimpleRPUDeviceCuda<T>(c, x_size, d_size){};

template <typename T> void MixedPrecRPUDeviceBaseCuda<T>::allocateContainers() {

  this->context_->synchronizeDevice();

  noise_manager_d_ = RPU::make_unique<NoiseManager<T>>(this->context_, this->d_size_);
  noise_manager_x_ = RPU::make_unique<NoiseManager<T>>(this->context_, this->x_size_);
  transfer_pwu_ =
      RPU::make_unique<PulsedWeightUpdater<T>>(this->context_, this->x_size_, this->d_size_);

  dev_sparsity_d_ = RPU::make_unique<CudaArray<T>>(this->context_, 1);
  dev_sparsity_d_->setConst((T)0);
  dev_sparsity_x_ = RPU::make_unique<CudaArray<T>>(this->context_, 1);
  dev_sparsity_x_->setConst((T)0);
  dev_avg_sparsity_ = RPU::make_unique<CudaArray<T>>(this->context_, 1);
  dev_avg_sparsity_->setConst((T)0);

  nblocks_batch_max_ = this->context_->getSMCount() *
                       (this->context_->maxThreadsPerBlock() / this->context_->getNThreads());

  dev_zc_temp_storage_ = nullptr;
  current_zero_size_ = 0;
  current_update_index_ = 0;
  dev_transfer_tmp_ = nullptr;
  dev_transfer_d_vecs_ = nullptr;

  up_ptr_ = nullptr;

  this->context_->synchronize();
}

// copy
template <typename T>
MixedPrecRPUDeviceBaseCuda<T>::MixedPrecRPUDeviceBaseCuda(
    const MixedPrecRPUDeviceBaseCuda<T> &other)
    : SimpleRPUDeviceCuda<T>(other) {

  allocateContainers();

  rpucuda_device_ = other.rpucuda_device_->cloneUnique();

  dev_sparsity_d_->assign(*other.dev_sparsity_d_);
  dev_sparsity_x_->assign(*other.dev_sparsity_x_);
  dev_avg_sparsity_->assign(*other.dev_avg_sparsity_);

  current_row_index_ = other.current_row_index_;
  current_update_index_ = other.current_update_index_;

  io_ = other.io_;
  granularity_ = other.granularity_;

  // do not copy tmps, noise managers, up_ptrs,  rw_rng_, nor transfer_pwu
  nblocks_batch_max_ = other.nblocks_batch_max_;
  this->context_->synchronize();
};

template <typename T>
MixedPrecRPUDeviceBaseCuda<T> &
MixedPrecRPUDeviceBaseCuda<T>::operator=(const MixedPrecRPUDeviceBaseCuda<T> &other) {
  MixedPrecRPUDeviceBaseCuda<T> tmp(other);
  swap(*this, tmp);
  this->context_->synchronize();
  return *this;
};

template <typename T>
MixedPrecRPUDeviceBaseCuda<T>::MixedPrecRPUDeviceBaseCuda(MixedPrecRPUDeviceBaseCuda<T> &&other) {
  *this = std::move(other);
};

template <typename T>
MixedPrecRPUDeviceBaseCuda<T> &
MixedPrecRPUDeviceBaseCuda<T>::operator=(MixedPrecRPUDeviceBaseCuda<T> &&other) {

  SimpleRPUDeviceCuda<T>::operator=(std::move(other));

  rpucuda_device_ = std::move(other.rpucuda_device_);
  rw_rng_ = std::move(rw_rng_);

  transfer_pwu_ = std::move(other.transfer_pwu_);
  noise_manager_x_ = std::move(other.noise_manager_x_);
  noise_manager_d_ = std::move(other.noise_manager_d_);

  current_row_index_ = other.current_row_index_;
  current_update_index_ = other.current_update_index_;
  current_zero_size_ = other.current_zero_size_;

  dev_zc_temp_storage_ = std::move(other.dev_zc_temp_storage_);

  dev_sparsity_d_ = std::move(other.dev_sparsity_d_);
  dev_sparsity_x_ = std::move(other.dev_sparsity_x_);
  dev_avg_sparsity_ = std::move(other.dev_avg_sparsity_);

  dev_transfer_tmp_ = std::move(dev_transfer_tmp_);
  dev_transfer_d_vecs_ = std::move(dev_transfer_d_vecs_);

  io_ = other.io_;
  up_ptr_ = other.up_ptr_;
  up_ = other.up_;

  granularity_ = other.granularity_;
  nblocks_batch_max_ = other.nblocks_batch_max_;
  return *this;
};

template <typename T>
void MixedPrecRPUDeviceBaseCuda<T>::populateFrom(const AbstractRPUDevice<T> &rpu_device_in) {

  const auto &rpu_device = dynamic_cast<const MixedPrecRPUDeviceBase<T> &>(rpu_device_in);
  if (&rpu_device == nullptr) {
    RPU_FATAL("populateFrom expects MixedPrecRPUDeviceBase.");
  }

  SimpleRPUDeviceCuda<T>::populateFrom(rpu_device_in); // will set sizes
  allocateContainers();
  rpucuda_device_ =
      AbstractRPUDeviceCuda<T>::createFromUnique(this->context_, rpu_device.getRPUDevice());

  const auto &par = this->getPar();
  granularity_ = rpu_device.getGranularity();
  this->context_->synchronize();
}

template <typename T>
void MixedPrecRPUDeviceBaseCuda<T>::dumpExtra(RPU::state_t &extra, const std::string prefix) {
  SimpleRPUDeviceCuda<T>::dumpExtra(extra, prefix);

  RPU::state_t state;

  rpucuda_device_->dumpExtra(state, "rpucuda_device");
  transfer_pwu_->dumpExtra(state, "transfer_pwu");
  noise_manager_x_->dumpExtra(state, "noise_manager_x");
  noise_manager_d_->dumpExtra(state, "noise_manager_d");

  RPU::insert(state, "current_update_index", current_update_index_);
  RPU::insert(state, "current_row_index", current_row_index_);
  RPU::insert(state, "granularity", granularity_);

  RPU::insert(state, "dev_avg_sparsity", dev_avg_sparsity_);
  RPU::insert(state, "dev_sparsity_d", dev_sparsity_d_);
  RPU::insert(state, "dev_sparsity_x", dev_sparsity_x_);

  // dev_transfer_d_vecs not handled (generated on the fly)

  RPU::insertWithPrefix(extra, state, prefix);
}

template <typename T>
void MixedPrecRPUDeviceBaseCuda<T>::loadExtra(
    const RPU::state_t &extra, const std::string prefix, bool strict) {
  SimpleRPUDeviceCuda<T>::loadExtra(extra, prefix, strict);

  auto state = RPU::selectWithPrefix(extra, prefix);
  using V = std::vector<T>;

  rpucuda_device_->loadExtra(state, "rpucuda_device", strict);
  transfer_pwu_->loadExtra(state, "transfer_pwu", strict);
  noise_manager_x_->loadExtra(state, "noise_manager_x", strict);
  noise_manager_d_->loadExtra(state, "noise_manager_d", strict);

  RPU::load(state, "granularity", granularity_, strict);
  RPU::load(state, "current_row_index", current_row_index_, strict);
  RPU::load(state, "current_update_index", current_update_index_, strict);

  RPU::load(this->context_, state, "dev_avg_sparsity", dev_avg_sparsity_, strict);
  RPU::load(this->context_, state, "dev_sparsity_d", dev_sparsity_d_, strict);
  RPU::load(this->context_, state, "dev_sparsity_x", dev_sparsity_x_, strict);
}

template <typename T>
__global__ void kernelAddSparsity(
    T *sparsity,
    const T *x_sparsity,
    const T *d_sparsity,
    int64_t current_update_index,
    const int m_batch) {
  volatile unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;

  if (tid == 0) {
    sparsity[0] = (sparsity[0] * (T)current_update_index + x_sparsity[0] * d_sparsity[0]) /
                  (T)(current_update_index + m_batch);
  }
}

template <typename T>
void MixedPrecRPUDeviceBaseCuda<T>::computeSparsityPartly(
    T *sparsity, const T *values, const int size) {

  IndicatorInputIterator<T> input_values(values, (T)0.0, (T)1.0 / ((T)size));
  if (size > current_zero_size_) {
    // init
    current_zero_size_ = size;
    size_t temp_storage_bytes = 0;
    RPU_CUB_NS_QUALIFIER DeviceReduce::Sum(
        nullptr, temp_storage_bytes, input_values, sparsity, size, this->context_->getStream());
    dev_zc_temp_storage_ = RPU::make_unique<CudaArray<char>>(this->context_, temp_storage_bytes);
  }

  // Run sum-reduction (use T as output)
  size_t temp_storage_bytes = dev_zc_temp_storage_->getSize();
  RPU_CUB_NS_QUALIFIER DeviceReduce::Sum(
      dev_zc_temp_storage_->getData(), temp_storage_bytes, input_values, sparsity, size,
      this->context_->getStream());
}

template <typename T>
void MixedPrecRPUDeviceBaseCuda<T>::computeSparsity(
    const T *x_values, const T *d_values, const int m_batch) {

  if (!getPar().compute_sparsity) {
    return;
  }

  computeSparsityPartly(dev_sparsity_d_->getData(), d_values, this->d_size_ * m_batch);
  computeSparsityPartly(dev_sparsity_x_->getData(), x_values, this->x_size_ * m_batch);
  kernelAddSparsity<<<1, 1, 0, this->context_->getStream()>>>(
      dev_avg_sparsity_->getData(), dev_sparsity_x_->getData(), dev_sparsity_d_->getData(),
      current_update_index_, m_batch);
}

template <typename T> T MixedPrecRPUDeviceBaseCuda<T>::getAvgSparsity() const {

  if (!getPar().compute_sparsity) {
    return 0;
  } else {
    T sparsity;
    dev_avg_sparsity_->copyTo(&sparsity);
    return sparsity;
  }
}

template <typename T>
void MixedPrecRPUDeviceBaseCuda<T>::setUpPar(const PulsedUpdateMetaParameter<T> &up) {
  if (&up != up_ptr_) {
    up_ptr_ = &up;
    up_ = up;

    // should be both true to get the desired step.
    up_.update_management = true;
    up_.update_bl_management = true;
  }
}

template <typename T> void MixedPrecRPUDeviceBaseCuda<T>::transfer(T *dev_weights, const T lr) {
  // updating the matrix with rows of using one-hot transfer vectors

  const auto &par = getPar();
  if (par.n_rows_per_transfer == 0 || fabsf(lr) == 0.0f) {
    return;
  }
  int n_transfers = par.n_rows_per_transfer;
  if (n_transfers < 0) {
    n_transfers = this->d_size_;
  }
  n_transfers = MIN(n_transfers, this->d_size_);
  // std::cout << "n_transfers: "  << n_transfers << ", current_row_index:" <<  current_row_index_
  //  << std::endl;
  int i_row = current_row_index_;
  if (par.random_row) {
    i_row = MAX(
        MIN(floorf((float)this->rw_rng_.sampleUniform() * this->d_size_), this->d_size_ - 1), 0);
  }

  int d2_size = this->d_size_ * this->d_size_;
  if (!dev_transfer_d_vecs_ || dev_transfer_d_vecs_->getSize() < d2_size) {

    std::vector<T> vec;
    vec.resize(d2_size, (T)0.0);
    for (int i = 0; i < d2_size; i += this->d_size_ + 1) {
      vec[i] = 1.0;
    }
    dev_transfer_d_vecs_ = RPU::make_unique<CudaArray<T>>(this->context_, d2_size, vec.data());
    this->context_->synchronize();
  }

  T *tvec = dev_transfer_d_vecs_->getData() + i_row * this->d_size_;
  int n_rest = this->d_size_ - i_row;

  if (n_rest < n_transfers) {
    // rest
    forwardUpdate(dev_weights, lr, i_row, tvec, n_rest, false);
    // from beginning
    forwardUpdate(dev_weights, lr, 0, dev_transfer_d_vecs_->getData(), n_transfers - n_rest, false);

  } else {
    forwardUpdate(dev_weights, lr, i_row, tvec, n_transfers, false);
  }
  current_row_index_ = (i_row + n_transfers) % this->d_size_;
}

template <typename T>
void MixedPrecRPUDeviceBaseCuda<T>::doTransfer(T *dev_weights, const T lr, const int m_batch) {
  const auto &par = getPar();
  int every = par.transfer_every; // current_update_index_ is in mat-vecs, but every in m_batch
  if (every > 0 && (current_update_index_ / m_batch) > 0 &&
      ((current_update_index_ / m_batch) % every == 0)) {
    transfer(dev_weights, lr);
  }
}

template <typename T> std::vector<T> MixedPrecRPUDeviceBaseCuda<T>::getHiddenWeights() const {
  std::vector<T> data;
  if (!rpucuda_device_) {
    // not populated?
    return data;
  }
  data = rpucuda_device_->getHiddenWeights();
  return data;
}

template <typename T>
void MixedPrecRPUDeviceBaseCuda<T>::decayWeights(T *dev_weights, T alpha, bool bias_no_decay) {
  rpucuda_device_->decayWeights(dev_weights, alpha, bias_no_decay);
}

template <typename T>
void MixedPrecRPUDeviceBaseCuda<T>::decayWeights(T *dev_weights, bool bias_no_decay) {
  rpucuda_device_->decayWeights(dev_weights, bias_no_decay);
}

template <typename T>
void MixedPrecRPUDeviceBaseCuda<T>::driftWeights(T *dev_weights, T time_since_last_call) {
  rpucuda_device_->driftWeights(dev_weights, time_since_last_call);
}

template <typename T> void MixedPrecRPUDeviceBaseCuda<T>::diffuseWeights(T *dev_weights) {
  rpucuda_device_->diffuseWeights(dev_weights);
}

template <typename T> void MixedPrecRPUDeviceBaseCuda<T>::clipWeights(T *dev_weights, T clip) {
  rpucuda_device_->clipWeights(dev_weights, clip);
}

template <typename T>
void MixedPrecRPUDeviceBaseCuda<T>::resetCols(
    T *dev_weights, int start_col, int n_cols, T reset_prob) {
  rpucuda_device_->resetCols(dev_weights, start_col, n_cols, reset_prob);
}

template class MixedPrecRPUDeviceBaseCuda<float>;
#ifdef RPU_USE_DOUBLE
template class MixedPrecRPUDeviceBaseCuda<double>;
#endif
#ifdef RPU_USE_FP16
template class MixedPrecRPUDeviceBaseCuda<half_t>;
#endif

} // namespace RPU
