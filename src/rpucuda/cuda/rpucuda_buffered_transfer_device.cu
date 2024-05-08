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

#include "cuda_fp16_util.h"
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
    CudaContextPtr c, const BufferedTransferRPUDevice<T> &rpu_device) {
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
  this->context_->synchronize();
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
    const int desired_BL_in,
    const bool forget_buffer_in) {

  T desired_BL = (T)desired_BL_in;
  bool forget_buffer = forget_buffer_in;
  T momentum = -sub_momentum + (T)1.0;

  RPU_CUDA_1D_KERNEL_LOOP(idx, size) {
    T omega = W_buffer[idx];
    T x = transfer_in[idx];
    omega += x * transfer_lr;

    transfer_in[idx] = (T)0.0; // reset to zero for next round

    T n_steps = trunc(omega / buffer_granularity);
    n_steps = MIN(MAX(n_steps, -desired_BL), desired_BL);

    if (forget_buffer) {
      W_buffer[idx] = (n_steps != (T)0.0) ? omega * momentum : omega;
    } else {
      W_buffer[idx] = omega - sub_momentum * n_steps * buffer_granularity;
    }

    transfer_out[idx] = -n_steps; // negative because of LR has reverse meaning
  }
}

template <typename T>
void BufferedTransferRPUDeviceCuda<T>::readAndUpdate(
    int to_device_idx,
    int from_device_idx,
    int i_slice_start,
    const T lr,
    const T count_lr,
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

  T *transfer_tmp = this->context_->template getSharedBuffer<T>(RPU_BUFFER_DEVICE_0, t_size);
  T *transfer_out = this->context_->template getSharedBuffer<T>(RPU_BUFFER_DEVICE_1, t_size);

  // forward/backward with transfer vectors into tmp
  T alpha = 1.0;
  // if (from_device_idx == 0 && par.fast_lr > (T)0.0 && count_lr > (T)0.0) {
  //   alpha = (T)1.0 / count_lr;
  // }
  this->readMatrix(from_device_idx, vec, transfer_tmp, n_vec, alpha);

  // out-size major
  T *B = transfer_buffer_vec_[from_device_idx]->getData() + i_slice_start * out_size;
  T weight_granularity = this->rpucuda_device_vec_[to_device_idx]->getWeightGranularity();
  T thres = par.thres_scale * weight_granularity;
  T step = par.step;

  int nthreads = this->context_->getNThreads();
  int nblocks = this->context_->getNBlocks(t_size, nthreads);
  T sub_momentum = (T)1.0 - MAX(MIN(par.momentum, (T)1.0), (T)0.0);

  kernelBufferedTransfer<T><<<nblocks, nthreads, 0, this->context_->getStream()>>>(
      transfer_out, B, transfer_tmp, t_size, fabsf(lr), thres, step, sub_momentum, up.desired_BL,
      par.forget_buffer);

  // update according to device
  this->writeMatrix(to_device_idx, vec, transfer_out, n_vec, fabsf(step * weight_granularity), up);

  this->context_->template releaseSharedBuffer<T>(RPU_BUFFER_DEVICE_0);
  this->context_->template releaseSharedBuffer<T>(RPU_BUFFER_DEVICE_1);
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
#ifdef RPU_USE_FP16
template class BufferedTransferRPUDeviceCuda<half_t>;
#endif

} // namespace RPU
