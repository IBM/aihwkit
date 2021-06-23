/**
 * (C) Copyright 2020, 2021 IBM. All Rights Reserved.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
 */

#include "io_iterator.h"
#include "rpu_pulsed_meta_parameter.h"
#include "rpucuda_mixedprec_device.h"
#include <cub/cub.cuh>
#include <memory>

namespace RPU {

/******************************************************************************************/
/* MixedPrecRPUDeviceCuda

   CUDA implementation of MixedPrecRPUDevice

*/
template <typename T>
MixedPrecRPUDeviceCuda<T>::MixedPrecRPUDeviceCuda(CudaContext *c, int x_size, int d_size)
    : MixedPrecRPUDeviceBaseCuda<T>(c, x_size, d_size){};

template <typename T>
MixedPrecRPUDeviceCuda<T>::MixedPrecRPUDeviceCuda(
    CudaContext *c, const MixedPrecRPUDevice<T> &rpu_device)
    : MixedPrecRPUDeviceCuda<T>(c, rpu_device.getXSize(), rpu_device.getDSize()) {
  populateFrom(rpu_device);
};

template <typename T> void MixedPrecRPUDeviceCuda<T>::allocateContainers() {
  this->context_->synchronizeDevice();
  dev_chi_ = RPU::make_unique<CudaArray<T>>(this->context_, this->size_);
  nblocks_batch_max_ = this->context_->getSMCount() *
                       (this->context_->maxThreadsPerBlock() / this->context_->getNThreads());
}

// copy
template <typename T>
MixedPrecRPUDeviceCuda<T>::MixedPrecRPUDeviceCuda(const MixedPrecRPUDeviceCuda<T> &other)
    : MixedPrecRPUDeviceBaseCuda<T>(other) {

  allocateContainers();
  dev_chi_->assign(*other.dev_chi_);
  this->context_->synchronize();
};

template <typename T>
MixedPrecRPUDeviceCuda<T> &
MixedPrecRPUDeviceCuda<T>::operator=(const MixedPrecRPUDeviceCuda<T> &other) {
  MixedPrecRPUDeviceCuda<T> tmp(other);
  swap(*this, tmp);
  return *this;
};

template <typename T>
MixedPrecRPUDeviceCuda<T>::MixedPrecRPUDeviceCuda(MixedPrecRPUDeviceCuda<T> &&other) {
  *this = std::move(other);
};

template <typename T>
MixedPrecRPUDeviceCuda<T> &MixedPrecRPUDeviceCuda<T>::operator=(MixedPrecRPUDeviceCuda<T> &&other) {

  MixedPrecRPUDeviceBaseCuda<T>::operator=(std::move(other));

  dev_chi_ = std::move(other.dev_chi_);
  nblocks_batch_max_ = nblocks_batch_max_;
  return *this;
}

template <typename T>
void MixedPrecRPUDeviceCuda<T>::populateFrom(const AbstractRPUDevice<T> &rpu_device_in) {

  const auto &rpu_device = dynamic_cast<const MixedPrecRPUDevice<T> &>(rpu_device_in);
  if (&rpu_device == nullptr) {
    RPU_FATAL("populateFrom expects MixedPrecRPUDevice.");
  }

  MixedPrecRPUDeviceBaseCuda<T>::populateFrom(rpu_device_in); // will set sizes
  allocateContainers();
  const auto &par = this->getPar();

  std::vector<T> v;
  v.resize(this->size_);
  rpu_device.getChi(v.data());
  dev_chi_->assign(v.data()); // both in x-major

  this->context_->synchronize();
}

template <typename T>
__global__ void kernelQuantizeBatch(
    T *quantized_values,
    const T *values,
    const T *nm_values,
    const int n_bins,
    const int size_in,
    const int m_batch_in,
    const bool trans_in) {

  volatile unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
  int size = size_in;
  int m_batch = m_batch_in;
  int total_size = size * m_batch;
  bool trans = trans_in;
  T half_bins = (T)(n_bins / 2); // floor
  T res = (T)1.0 / ((T)half_bins);
  T value;
  int total_threads = blockDim.x * gridDim.x;
  for (int i_stride = 0; i_stride < total_size; i_stride += total_threads) {
    int idx = i_stride + tid;

    if (idx < total_size) {
      value = values[idx];

      int sidx = trans ? (idx % m_batch) : (idx / size);
      T amax = nm_values[sidx]; // amax from noise management
      value = amax > 0.0 ? value / amax : value;
      value = RPU_ROUNDFUN(value / res);
      value = MIN(MAX(value, -half_bins), half_bins) * amax * res;

      quantized_values[idx] = value;
    }
  }
}

template <typename T>
const T *MixedPrecRPUDeviceCuda<T>::quantize(
    T *buffer_values,
    const T *values,
    RPU::NoiseManager<T> *nm,
    int n_bins,
    int size,
    int m_batch,
    bool trans) {

  if (n_bins <= 0) {
    return values;
  }

  nm->compute(values, NoiseManagementType::AbsMax, this->io_, m_batch, trans, false);
  int nthreads = this->context_->getNThreads();
  int nblocks = this->context_->getNBlocks(m_batch * size, nthreads);
  int nblocks_batch = MIN(nblocks_batch_max_, nblocks);

  cudaStream_t s = this->context_->getStream();

  kernelQuantizeBatch<<<nblocks_batch, nthreads, 0, s>>>(
      buffer_values, values, nm->getScaleValues(), n_bins, size, m_batch, trans);

  return buffer_values;
}

template <typename T>
void MixedPrecRPUDeviceCuda<T>::doDirectUpdate(
    const T *x_input,
    const T *d_input,
    T *dev_weights,
    const T lr,
    const int m_batch,
    const bool x_trans,
    const bool d_trans,
    const T beta,
    const PulsedUpdateMetaParameter<T> &up,
    T *x_buffer,
    T *d_buffer) {

  if (beta != 1.0f) {
    RPU_FATAL("beta not equal 1 is not supported.")
  }

  this->setUpPar(up);
  const auto &par = getPar();

  const T *d_val = quantize(
      d_buffer, d_input, &*this->noise_manager_d_, par.n_d_bins, this->d_size_, m_batch, d_trans);

  // % Quantize x
  const T *x_val = quantize(
      x_buffer, x_input, &*this->noise_manager_x_, par.n_x_bins, this->x_size_, m_batch, x_trans);

  // dev_chi is x-size (row) major !! (to facilitate the readout below)

  if (m_batch == 1) {
    RPU::math::ger<T>(
        this->context_, this->x_size_, this->d_size_, lr, x_val, 1, d_val, 1, dev_chi_->getData(),
        this->x_size_);
  } else {
    RPU::math::gemm<T>(
        this->context_, x_trans, !d_trans, this->x_size_, this->d_size_, m_batch, lr, x_val,
        x_trans ? m_batch : this->x_size_, d_val, d_trans ? m_batch : this->d_size_,
        1.0, // set beta to 1.0. We want to add to Chi
        dev_chi_->getData(), this->x_size_);
  }

  this->doTransfer(dev_weights, 1.0, m_batch);
  this->advanceUpdateCounter();
  this->computeSparsity(x_buffer, d_buffer, m_batch);
}

template <typename T>
__global__ void
kernelMixedPrecTransfer(T *transfer_out, T *chi, const int size, const T granularity_) {
  volatile unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;

  if (tid < size) {
    T value = chi[tid];
    T dw = truncf(value / granularity_);
    transfer_out[tid] = dw;

    chi[tid] = value - granularity_ * dw;
  }
}

template <typename T>
void MixedPrecRPUDeviceCuda<T>::forwardUpdate(
    T *dev_weights,
    const T lr,
    int i_row_start,
    const T *transfer_vec,
    const int n_vec,
    const bool trans) {

  if (!lr) {
    return;
  }
  T t_size = n_vec * this->x_size_;
  if ((this->dev_transfer_tmp_ == nullptr) || this->dev_transfer_tmp_->getSize() < t_size) {
    this->dev_transfer_tmp_ = RPU::make_unique<CudaArray<T>>(this->context_, t_size);
  }

  const auto &par = this->getPar();

  int nthreads = this->context_->getNThreads();
  int nblocks = this->context_->getNBlocks(t_size, nthreads);
  kernelMixedPrecTransfer<T><<<nblocks, nthreads, 0, this->context_->getStream()>>>(
      this->dev_transfer_tmp_->getData(), dev_chi_->getData() + i_row_start * this->x_size_, t_size,
      this->granularity_);

  // requires to turn on update_managment / bl managment as well
  this->transfer_pwu_->update(
      this->dev_transfer_tmp_->getDataConst(), // this is the transfer vector (x_size)
      transfer_vec,                            // this should be d_size, non-trans
      dev_weights, &*this->rpucuda_device_, this->up_, this->granularity_, n_vec, trans, false);
}

template <typename T> std::vector<T> MixedPrecRPUDeviceCuda<T>::getHiddenWeights() const {

  auto data = MixedPrecRPUDeviceBaseCuda<T>::getHiddenWeights();

  int offset = data.size();
  data.resize(offset + this->size_);
  dev_chi_->copyTo(data.data() + offset);

  return data;
}

template class MixedPrecRPUDeviceCuda<float>;
#ifdef RPU_USE_DOUBLE
template class MixedPrecRPUDeviceCuda<double>;
#endif
} // namespace RPU
