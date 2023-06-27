/**
 * (C) Copyright 2020, 2021, 2022, 2023 IBM. All Rights Reserved.
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
#include "rpucuda_dynamic_transfer_device.h"
#include <limits>
#include <memory>

namespace RPU {

/******************************************************************************************/
/* DefferenceRPUDeviceCuda

   CUDA implementation of TransferRPUDevice

*/

template <typename T>
DynamicTransferRPUDeviceCuda<T>::DynamicTransferRPUDeviceCuda(
    CudaContextPtr c, const DynamicTransferRPUDevice<T> &rpu_device) {
  this->context_ = c;
  populateFrom(rpu_device); // use populate to call parent
};

// copy construcutor
template <typename T>
DynamicTransferRPUDeviceCuda<T>::DynamicTransferRPUDeviceCuda(
    const DynamicTransferRPUDeviceCuda<T> &other)
    : ChoppedTransferRPUDeviceCuda<T>(other) {
  if (other.dev_running_var_ != nullptr) {

    dev_past_mean_ = RPU::make_unique<CudaArray<T>>(*other.dev_past_mean_);
    dev_running_mean_ = RPU::make_unique<CudaArray<T>>(*other.dev_running_mean_);
    dev_running_var_ = RPU::make_unique<CudaArray<T>>(*other.dev_running_var_);
    dev_transfers_since_in_chop_ =
        RPU::make_unique<CudaArray<int>>(*other.dev_transfers_since_in_chop_);
    dev_transfers_since_in_chop_tmp_ =
        RPU::make_unique<CudaArray<int>>(*other.dev_transfers_since_in_chop_tmp_);

    dev_previous_in_chopper_ = RPU::make_unique<CudaArray<chop_t>>(*other.dev_previous_in_chopper_);
    dev_previous_in_chopper_tmp_ =
        RPU::make_unique<CudaArray<chop_t>>(*other.dev_previous_in_chopper_tmp_);

    current_in_chop_prob_ = other.current_in_chop_prob_;
    this->context_->synchronize();
  }
};

// copy assignment
template <typename T>
DynamicTransferRPUDeviceCuda<T> &
DynamicTransferRPUDeviceCuda<T>::operator=(const DynamicTransferRPUDeviceCuda<T> &other) {
  DynamicTransferRPUDeviceCuda<T> tmp(other);
  swap(*this, tmp);
  this->context_->synchronize();
  return *this;
};

// move constructor
template <typename T>
DynamicTransferRPUDeviceCuda<T>::DynamicTransferRPUDeviceCuda(
    DynamicTransferRPUDeviceCuda<T> &&other) {
  *this = std::move(other);
};

// move assignment
template <typename T>
DynamicTransferRPUDeviceCuda<T> &
DynamicTransferRPUDeviceCuda<T>::operator=(DynamicTransferRPUDeviceCuda<T> &&other) {
  ChoppedTransferRPUDeviceCuda<T>::operator=(std::move(other));

  dev_past_mean_ = std::move(other.dev_past_mean_);
  dev_running_mean_ = std::move(other.dev_running_mean_);
  dev_running_var_ = std::move(other.dev_running_var_);
  dev_transfers_since_in_chop_ = std::move(other.dev_transfers_since_in_chop_);
  dev_transfers_since_in_chop_tmp_ = std::move(other.dev_transfers_since_in_chop_tmp_);
  dev_previous_in_chopper_ = std::move(other.dev_previous_in_chopper_);
  dev_previous_in_chopper_tmp_ = std::move(other.dev_previous_in_chopper_tmp_);

  current_in_chop_prob_ = other.current_in_chop_prob_;
  return *this;
};

template <typename T>
void DynamicTransferRPUDeviceCuda<T>::populateFrom(const AbstractRPUDevice<T> &rpu_device_in) {

  const auto &rpu_device = dynamic_cast<const DynamicTransferRPUDevice<T> &>(rpu_device_in);
  if (&rpu_device == nullptr) {
    RPU_FATAL("populateFrom expects DynamicTransferRPUDevice.");
  }

  ChoppedTransferRPUDeviceCuda<T>::populateFrom(rpu_device_in);

  if (this->n_devices_ != 2) {
    RPU_FATAL("Expect at exactly two devices.");
  }
  const auto &par = getPar();
  par.checkSupported();

  int size = this->x_size_ * this->d_size_;
  dev_running_mean_ =
      RPU::make_unique<CudaArray<T>>(this->context_, size, rpu_device.getRunningMean());
  dev_running_var_ =
      RPU::make_unique<CudaArray<T>>(this->context_, size, rpu_device.getRunningVar());
  dev_past_mean_ = RPU::make_unique<CudaArray<T>>(this->context_, size, rpu_device.getPastMean());
  dev_past_mean_->setConst(0.0);

  // NOTE: not everything is copied actually. The choppers / transfer / update
  // counting will begin from scratch when copied to cuda.

  dev_transfers_since_in_chop_tmp_ =
      RPU::make_unique<CudaArray<int>>(this->context_, par.getInSize());
  dev_transfers_since_in_chop_ = RPU::make_unique<CudaArray<int>>(this->context_, par.getInSize());
  dev_transfers_since_in_chop_->setConst(0);

  dev_previous_in_chopper_tmp_ =
      RPU::make_unique<CudaArray<chop_t>>(this->context_, par.getInSize());
  dev_previous_in_chopper_ = RPU::make_unique<CudaArray<chop_t>>(this->context_, par.getInSize());
  dev_previous_in_chopper_->setConst(1);

  current_in_chop_prob_ = rpu_device.getCurrentInChopProb();

  this->context_->synchronize();
};

template <typename T>
void DynamicTransferRPUDeviceCuda<T>::dumpExtra(RPU::state_t &extra, const std::string prefix) {
  ChoppedTransferRPUDeviceCuda<T>::dumpExtra(extra, prefix);

  RPU::state_t state;

  RPU::insert(state, "current_in_chop_prob", current_in_chop_prob_);
  RPU::insert(state, "dev_transfers_since_in_chop", dev_transfers_since_in_chop_);
  RPU::insert(state, "dev_previous_in_chopper", dev_previous_in_chopper_);

  // all other vars are handled with the getDeviceParameter
  RPU::insertWithPrefix(extra, state, prefix);
}

template <typename T>
void DynamicTransferRPUDeviceCuda<T>::loadExtra(
    const RPU::state_t &extra, const std::string prefix, bool strict) {
  ChoppedTransferRPUDeviceCuda<T>::loadExtra(extra, prefix, strict);

  auto state = RPU::selectWithPrefix(extra, prefix);

  RPU::load(state, "current_in_chop_prob", current_in_chop_prob_, strict);
  RPU::load(
      this->context_, state, "dev_transfers_since_in_chop", dev_transfers_since_in_chop_, strict);
  RPU::load(this->context_, state, "dev_previous_in_chopper", dev_previous_in_chopper_, strict);
}

/*********************************************************************************/
/* partially transfer using the given "readout" transfer vectors
   (with io-managed forward) and buffer the results in
   digital. Transfer only to next device if threshold is reached. */
namespace detail {
template <typename T>
__global__ void kernelDynamicTransfer(
    T *transfer_out,
    T *buffer_weight,                      // out_size major
    T *running_mean_weight,                // out_size major
    T *running_var_weight,                 // out_size major
    T *past_mean_weight,                   // out_size major
    int *transfers_since_in_chop_out,      // in_size
    chop_t *previous_in_choppers_out,      // in_size
    const int *transfers_since_in_chop_in, // in_size
    const chop_t *previous_in_choppers_in, // in_size
    const T *transfer_in,
    const chop_t *in_chopper,  // size n_wo  NOTE: is already applied to transfer_in
    const chop_t *out_chopper, // size n_wo * out_size
    const int out_size,
    const int in_size,
    const int m_batch,
    const int start_read_idx,
    const T thres_scale_in,
    const T lr_scale_in,
    const T weight_granularity,
    const T momentum,
    const T var_momentum,
    const int max_steps_in,
    const bool scale_thres_with_samples_in,
    const bool forget_buffer,
    const T tail_weightening,
    const T buffer_cap,
    const bool always_write_in) {

  int w_size = out_size * in_size;
  int t_size = out_size * m_batch;
  T max_steps = (T)max_steps_in;
  const bool always_write = always_write_in;
  const T thres_scale = thres_scale_in;
  const bool use_thres = thres_scale > 0;
  const T lr_scale = lr_scale_in;
  const bool scale_thres_with_samples = scale_thres_with_samples_in && !always_write;

  // CAUTION: n_vec might have mulitple wraps around in_size, we need
  // to thus make sure that the same threads are working on the same
  // repeat.
  int n_repeats = (m_batch + in_size - 1) / in_size;

  RPU_CUDA_1D_KERNEL_LOOP(idx, w_size) {

    T omega, mean_w, var_w, past_mean_w;
    int buffer_idx, in_idx, out_idx_shifted, current_sample;
    chop_t previous_in_chop;
    bool used_thread_for_read = false;
    bool used_thread_for_write = false;

    for (int i_rep = 0; i_rep < n_repeats; i_rep++) {

      int inp_idx = idx + i_rep * w_size;

      if (inp_idx >= t_size) {
        break;
      }

      if (i_rep == 0) {
        // initialize and load
        used_thread_for_read = true;
        buffer_idx = (idx + start_read_idx * out_size) % w_size; // this is unique for each thread
        omega = buffer_weight[buffer_idx];
        mean_w = running_mean_weight[buffer_idx];
        past_mean_w = past_mean_weight[buffer_idx];
        if (use_thres) {
          var_w = running_var_weight[buffer_idx];
        }
        in_idx = buffer_idx / out_size;
        out_idx_shifted = idx % out_size;
        current_sample = transfers_since_in_chop_in[in_idx];
        previous_in_chop = previous_in_choppers_in[in_idx];
      }

      int i_wo = inp_idx / out_size; // in_chop always non-trans
      chop_t in_chop = in_chopper[i_wo];
      chop_t out_chop = out_chopper[inp_idx];
      T val = in_chop * transfer_in[inp_idx]; // remove the applied in_choppper
      bool in_chop_switch = previous_in_chop != in_chop;
      T n_steps = 0.0;

      if (in_chop_switch || always_write) {

        used_thread_for_write = true;

        // do the write calculation etc here the switch has already
        // happend on the current sample. So first calculate the
        // update and then update the variables

        T mu = (always_write && !in_chop_switch) ? val : mean_w;
        T mu_past = past_mean_w;

        T dw = (mu - mu_past);
        dw = (previous_in_chop != out_chop) ? -dw : dw;

        if (use_thres) {
          T thres = thres_scale;
          T std = MAX(sqrtf(fabs(var_w)), weight_granularity);

          if (scale_thres_with_samples) {
            thres /= sqrtf((T)MAX(current_sample, 1));
          }

          if (fabs(dw) > thres * std) {
            omega += dw * lr_scale;
          }

        } else {
          omega += dw * lr_scale;
        }

        if (fabs(omega) >= (T)1.0) {
          n_steps = MIN(MAX(truncf(omega), -max_steps), max_steps);

          if (forget_buffer) {
            omega *= momentum;
          } else {
            omega -= ((T)1.0 - momentum) * n_steps;
            if (buffer_cap > (T)0.0) {
              omega = MIN(MAX(omega, -buffer_cap), buffer_cap);
            }
          }
        }

        if (in_chop_switch) {
          // reset buffer and mean esimation, and begin new chopper phase
          current_sample = 0;
          previous_in_chop = in_chop;
          past_mean_w = mean_w;
          mean_w = (T)0.0;
        }

      } // writing

      // set the output
      transfer_out[inp_idx] = -n_steps; // write sign corrected

      // update the current sample
      current_sample++;
      T sample_momentum = MIN((T)tail_weightening / (T)current_sample, (T)1);

      mean_w = mean_w * ((T)1.0 - sample_momentum) + val * sample_momentum;
      if (use_thres) {
        T d_val = (val - mean_w);
        var_w = var_w * var_momentum + (d_val * d_val) * ((T)1.0 - var_momentum);
      }
    }

    if (used_thread_for_read) {
      if (use_thres) {
        running_var_weight[buffer_idx] = var_w;
      }
      running_mean_weight[buffer_idx] = mean_w;

      if (out_idx_shifted == 0) {
        transfers_since_in_chop_out[in_idx] = current_sample;
      }
    }
    if (used_thread_for_write) {
      buffer_weight[buffer_idx] = omega;
      past_mean_weight[buffer_idx] = past_mean_w;
      if (out_idx_shifted == 0) {
        previous_in_choppers_out[in_idx] = previous_in_chop;
      }
    }
  }
}
} // namespace detail

template <typename T>
void DynamicTransferRPUDeviceCuda<T>::readAndUpdate(
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
  if (!this->transfer_buffer_vec_.size()) {
    RPU_FATAL("First populate device.");
  }
  const auto &par = this->getPar();
  UNUSED(vec); // will use one-hot always

  int in_size = par.getInSize();
  int out_size = par.getOutSize();
  int t_size = n_vec * out_size; // transfer size
  T weight_granularity = this->rpucuda_device_vec_[to_device_idx]->getWeightGranularity();

  T *transfer_tmp = this->context_->template getSharedBuffer<T>(RPU_BUFFER_DEVICE_0, t_size);
  T *transfer_out = this->context_->template getSharedBuffer<T>(RPU_BUFFER_DEVICE_1, t_size);

  // forward/backward with transfer vectors into tmp
  this->readMatrix(from_device_idx, nullptr, transfer_tmp, n_vec, (T)1.0);
  T from_weight_granularity = this->rpucuda_device_vec_[from_device_idx]->getWeightGranularity();
  // out-size major
  T *fp_w = this->transfer_buffer_vec_[from_device_idx]->getData();

  int n = MIN(in_size * out_size, t_size);
  int nthreads = this->context_->getNThreads(n);
  int nblocks = this->context_->getNBlocks(n, nthreads);
  int max_steps = up.desired_BL;
  T buffer_cap = max_steps * par.buffer_cap;
  T lr_scale = par.getTransferLRScale(from_weight_granularity, lr);

  // need to copy ... otherwise large dims will fail because of block looping
  dev_transfers_since_in_chop_tmp_->assign(*dev_transfers_since_in_chop_);
  dev_previous_in_chopper_tmp_->assign(*dev_previous_in_chopper_);

  detail::kernelDynamicTransfer<T><<<nblocks, nthreads, 0, this->context_->getStream()>>>(
      transfer_out, fp_w, dev_running_mean_->getData(), dev_running_var_->getData(),
      dev_past_mean_->getData(), dev_transfers_since_in_chop_->getData(),
      dev_previous_in_chopper_->getData(), dev_transfers_since_in_chop_tmp_->getData(),
      dev_previous_in_chopper_tmp_->getData(), transfer_tmp,
      this->cwo_->getWeightOutputInChopperData(), this->cwo_->getWeightOutputOutChopperData(),
      out_size, in_size, n_vec, i_slice_start, par.thres_scale, lr_scale, weight_granularity,
      par.momentum, par.auto_momentum, max_steps, par.scale_thres_with_samples, par.forget_buffer,
      par.tail_weightening, buffer_cap, par.always_write);

  // update according to device
  int n_samples = par.getNumInChopSamples(current_in_chop_prob_);
  int n_period = n_samples * par.getInSize() * this->cwo_->getEvery();
  if (this->cwo_->getNWOCounter() > this->cwo_->getNumWeightOutputs() &&
      this->cwo_->getCounter() > n_period) {
    T write_lr = par.getWriteLR(weight_granularity);
    this->writeMatrix(to_device_idx, nullptr, transfer_out, n_vec, write_lr, up);
  }

  this->context_->template releaseSharedBuffer<T>(RPU_BUFFER_DEVICE_0);
  this->context_->template releaseSharedBuffer<T>(RPU_BUFFER_DEVICE_1);
}

template <typename T> std::vector<T> DynamicTransferRPUDeviceCuda<T>::getHiddenWeights() const {
  std::vector<T> data;
  if (!this->n_devices_ || !this->transfer_buffer_vec_.size()) {
    // not populated?
    return data;
  }

  // we skip the Buffered/Chopped on purpose here
  data = TransferRPUDeviceCuda<T>::getHiddenWeights();

  int offset = data.size();
  int add_n = 4;
  data.resize(offset + add_n * this->size_);

  auto copy_fun = [this](const CudaArray<T> *dev_vec, std::vector<T> *data_ptr, int &offset) {
    bool transpose = this->getPar().transfer_columns;
    std::vector<T> w_vec(this->size_);
    dev_vec->copyTo(w_vec.data());

    if (transpose) {
      for (int i = 0; i < this->size_; i++) {
        // transpose d_size maj -> x_size maj
        (*data_ptr)[offset + i] = w_vec[TRANSPOSE_X2D(i, this->x_size_, this->d_size_)];
      }
    } else {
      // already x-major
      for (int i = 0; i < this->size_; i++) {
        (*data_ptr)[offset + i] = w_vec[i];
      }
    }
    offset += this->size_;
  };

  copy_fun(&*this->transfer_buffer_vec_[0], &data, offset);
  copy_fun(&*dev_running_mean_, &data, offset);
  copy_fun(&*dev_running_var_, &data, offset);
  copy_fun(&*dev_past_mean_, &data, offset);

  return data;
}

template class DynamicTransferRPUDeviceCuda<float>;
#ifdef RPU_USE_DOUBLE
template class DynamicTransferRPUDeviceCuda<double>;
#endif
} // namespace RPU
