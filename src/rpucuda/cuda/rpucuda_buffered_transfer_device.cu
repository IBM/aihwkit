/**
 * (C) Copyright 2020, 2021, 2022 IBM. All Rights Reserved.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
 */

#include "forward_backward_pass.h"
#include "rpu_pulsed_meta_parameter.h"
#include "rpucuda_buffered_transfer_device.h"
#include <memory>

namespace RPU {

/******************************************************************************************/
/* BufferedTransferRPUDeviceCuda

   CUDA implementation of BufferedTransferRPUDevice

*/

template <typename T>
BufferedTransferRPUDeviceCuda<T>::BufferedTransferRPUDeviceCuda(
    CudaContext *c, const BufferedTransferRPUDevice<T> &rpu_device) {
  this->context_ = c;
  populateFrom(rpu_device); // use populate to call parent
};

// copy construcutor
template <typename T>
BufferedTransferRPUDeviceCuda<T>::BufferedTransferRPUDeviceCuda(
    const BufferedTransferRPUDeviceCuda<T> &other)
    : TransferRPUDeviceCuda<T>(other) {

  if (other.transfer_buffer_vec_.size()) {
    int n = other.transfer_buffer_vec_.size();
    transfer_buffer_vec_.resize(n);

    for (int i = 0; i < n; i++) {
      if (other.transfer_buffer_vec_[i]) {
        transfer_buffer_vec_[i] = RPU::make_unique<CudaArray<T>>(*(other.transfer_buffer_vec_[i]));
      }
    }
  }
  this->context_->synchronize();
};

// copy assignment
template <typename T>
BufferedTransferRPUDeviceCuda<T> &
BufferedTransferRPUDeviceCuda<T>::operator=(const BufferedTransferRPUDeviceCuda<T> &other) {
  BufferedTransferRPUDeviceCuda<T> tmp(other);
  swap(*this, tmp);
  return *this;
};

// move constructor
template <typename T>
BufferedTransferRPUDeviceCuda<T>::BufferedTransferRPUDeviceCuda(
    BufferedTransferRPUDeviceCuda<T> &&other) {
  *this = std::move(other);
};

// move assignment
template <typename T>
BufferedTransferRPUDeviceCuda<T> &
BufferedTransferRPUDeviceCuda<T>::operator=(BufferedTransferRPUDeviceCuda<T> &&other) {
  TransferRPUDeviceCuda<T>::operator=(std::move(other));

  transfer_buffer_vec_ = std::move(other.transfer_buffer_vec_);
  transfer_out_ = std::move(other.transfer_out_);

  return *this;
};

template <typename T>
void BufferedTransferRPUDeviceCuda<T>::populateFrom(const AbstractRPUDevice<T> &rpu_device_in) {

  const auto &rpu_device = dynamic_cast<const BufferedTransferRPUDevice<T> &>(rpu_device_in);
  if (&rpu_device == nullptr) {
    RPU_FATAL("populateFrom expects BufferedTransferRPUDevice.");
  }

  TransferRPUDeviceCuda<T>::populateFrom(rpu_device_in);

  std::vector<std::vector<T>> t_vec = rpu_device.getTransferBuffers();
  if (t_vec.size() != (size_t)(this->n_devices_ - 1)) {
    RPU_FATAL("Wrong number of buffers to populate from.");
  }

  transfer_buffer_vec_.resize(this->n_devices_ - 1);
  const auto &par = getPar();

  for (int k = 0; k < this->n_devices_ - 1; k++) {
    if (t_vec[k].size() != this->size_) {
      RPU_FATAL("Wrong number of elements in buffers to populate from.");
    }
    transfer_buffer_vec_[k] = RPU::make_unique<CudaArray<T>>(this->context_, this->size_);
    this->context_->synchronize();
    if (par.transfer_columns) {
      // d_major
      transfer_buffer_vec_[k]->assignTranspose(t_vec[k].data(), this->d_size_, this->x_size_);
    } else {
      // x_major in this case. Always out-size-major
      transfer_buffer_vec_[k]->assign(t_vec[k].data());
    }
  }
  this->context_->synchronize();
}

/* partially transfer using the given "readout" transfer vectors
   (with io-managed forward) and buffer the results in
   digital. Transfer only to next device if threshold is reached. */

template <typename T>
__global__ void kernelBufferedTransfer(
    T *transfer_out,
    T *W_buffer,
    T *transfer_in,
    const int size,
    const T transfer_lr,
    const T buffer_granularity,
    const T step,
    const T sub_momentum,
    const int desired_BL_in) {

  T desired_BL = (T)desired_BL_in;

  RPU_CUDA_1D_KERNEL_LOOP(idx, size) {
    T omega = W_buffer[idx];
    T x = transfer_in[idx];
    omega += x * transfer_lr;

    transfer_in[idx] = (T)0.0; // reset to zero for next round

    T n_steps = truncf(omega / buffer_granularity);
    n_steps = MIN(MAX(n_steps, -desired_BL), desired_BL);

    W_buffer[idx] = omega - sub_momentum * n_steps * buffer_granularity;
    transfer_out[idx] = -n_steps; // negative because of LR has reverse meaning
  }
}

template <typename T>
void BufferedTransferRPUDeviceCuda<T>::readAndUpdate(
    int to_device_idx,
    int from_device_idx,
    int i_slice_start,
    const T lr,
    const T *vec,
    const int n_vec,
    const PulsedUpdateMetaParameter<T> &up) {
  if (lr == (T)0.0) {
    return;
  }

  if (!transfer_buffer_vec_.size()) {
    RPU_FATAL("First populate device.");
  }
  const auto &par = this->getPar();

  int in_size = par.getInSize();
  int out_size = par.getOutSize();

  int t_size = n_vec * out_size; // transfer size
  if ((this->transfer_tmp_ == nullptr) || this->transfer_tmp_->getSize() < t_size) {
    this->transfer_tmp_ = RPU::make_unique<CudaArray<T>>(this->context_, t_size);
    this->transfer_tmp_->setConst((T)0.0);
  }
  // init second tmp as well, no need to zero this.
  if ((transfer_out_ == nullptr) || transfer_out_->getSize() < t_size) {
    transfer_out_ = RPU::make_unique<CudaArray<T>>(this->context_, t_size);
  }

  // forward/backward with transfer vectors into tmp
  this->readMatrix(from_device_idx, vec, this->transfer_tmp_->getData(), n_vec, 1.0);

  // 1) add tmp to buffer (tmp is forced to be trans==false)
  // 2) set tmp to 0
  // 3) check thres on buffer
  // 4) write step-wise transfer_out for next device
  T *B =
      transfer_buffer_vec_[from_device_idx]->getData() + i_slice_start * out_size; // out-size major

  T weight_granularity = this->rpucuda_device_vec_[to_device_idx]->getWeightGranularity();
  T thres = par.thres_scale * weight_granularity;
  T step = par.step;

  int nthreads = this->context_->getNThreads();
  int nblocks = this->context_->getNBlocks(t_size, nthreads);
  T sub_momentum = (T)1.0 - MAX(MIN(par.momentum, (T)1.0), (T)0.0);

  kernelBufferedTransfer<T><<<nblocks, nthreads, 0, this->context_->getStream()>>>(
      transfer_out_->getData(), B, this->transfer_tmp_->getData(), t_size, fabs(lr), thres, step,
      sub_momentum, up.desired_BL);

  // in reality one might want to check whether some updates are
  // actually made and don't do anything if not. However, for that
  // one would need to copy the sum of transfer_out_ to CPU which
  // would need to synchronize the cuda streams and thus would not
  // have any benefits. Thus we just start the update in any case,
  // even if transfer_out_ turns out to be all zero.

  // update according to device
  this->writeMatrix(
      to_device_idx, vec, transfer_out_->getDataConst(), n_vec, fabs(step * weight_granularity),
      up);
}

template <typename T> std::vector<T> BufferedTransferRPUDeviceCuda<T>::getHiddenWeights() const {
  std::vector<T> data;
  if (!this->n_devices_ || !transfer_buffer_vec_.size()) {
    // not populated?
    return data;
  }

  data = TransferRPUDeviceCuda<T>::getHiddenWeights();

  int offset = data.size();
  data.resize(offset + (this->n_devices_ - 1) * this->size_);
  bool transpose = this->getPar().transfer_columns;

  for (int k = 0; k < this->n_devices_ - 1; k++) {
    std::vector<T> w_vec(this->size_);

    transfer_buffer_vec_[k]->copyTo(w_vec.data());

    if (transpose) {
      for (int i = 0; i < this->size_; i++) {
        // transpose d_size maj -> x_size maj
        data[offset + i] = w_vec[TRANSPOSE_X2D(i, this->x_size_, this->d_size_)];
      }
    } else {
      // already x-major
      for (int i = 0; i < this->size_; i++) {
        data[offset + i] = w_vec[i];
      }
    }
    offset += this->size_;
  }

  return data;
}

template class BufferedTransferRPUDeviceCuda<float>;
#ifdef RPU_USE_DOUBLE
template class BufferedTransferRPUDeviceCuda<double>;
#endif
} // namespace RPU
