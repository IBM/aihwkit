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
#include "io_iterator.h"
#include "rpu_pulsed_meta_parameter.h"
#include "rpucuda_chopped_transfer_device.h"
#include <limits>
#include <memory>

namespace RPU {

/******************************************************************************************/
/* DefferenceRPUDeviceCuda

   CUDA implementation of TransferRPUDevice

*/

template <typename T>
ChoppedTransferRPUDeviceCuda<T>::ChoppedTransferRPUDeviceCuda(
    CudaContextPtr c, const ChoppedTransferRPUDevice<T> &rpu_device) {
  this->context_ = c;
  populateFrom(rpu_device); // use populate to call parent
};

// copy construcutor
template <typename T>
ChoppedTransferRPUDeviceCuda<T>::ChoppedTransferRPUDeviceCuda(
    const ChoppedTransferRPUDeviceCuda<T> &other)
    : BufferedTransferRPUDeviceCuda<T>(other) {

  if (other.cwo_) {
    cwo_ = RPU::make_unique<ChoppedWeightOutput<T>>(*other.cwo_);
  }
  m_x_ = other.m_x_;
  m_d_ = other.m_d_;

  this->context_->synchronizeDevice();
};

// copy assignment
template <typename T>
ChoppedTransferRPUDeviceCuda<T> &
ChoppedTransferRPUDeviceCuda<T>::operator=(const ChoppedTransferRPUDeviceCuda<T> &other) {
  ChoppedTransferRPUDeviceCuda<T> tmp(other);
  swap(*this, tmp);
  this->context_->synchronize();
  return *this;
};

// move constructor
template <typename T>
ChoppedTransferRPUDeviceCuda<T>::ChoppedTransferRPUDeviceCuda(
    ChoppedTransferRPUDeviceCuda<T> &&other) {
  *this = std::move(other);
};

// move assignment
template <typename T>
ChoppedTransferRPUDeviceCuda<T> &
ChoppedTransferRPUDeviceCuda<T>::operator=(ChoppedTransferRPUDeviceCuda<T> &&other) {
  BufferedTransferRPUDeviceCuda<T>::operator=(std::move(other));

  cwo_ = std::move(other.cwo_);
  m_x_ = other.m_x_;
  m_d_ = other.m_d_;

  return *this;
};

template <typename T>
void ChoppedTransferRPUDeviceCuda<T>::populateFrom(const AbstractRPUDevice<T> &rpu_device_in) {

  const auto &rpu_device = dynamic_cast<const ChoppedTransferRPUDevice<T> &>(rpu_device_in);
  if (&rpu_device == nullptr) {
    RPU_FATAL("populateFrom expects ChoppedTransferRPUDevice.");
  }

  BufferedTransferRPUDeviceCuda<T>::populateFrom(rpu_device_in);

  if (this->n_devices_ != 2) {
    RPU_FATAL("Expect at exactly two devices.");
  }

  this->transfer_vecs_ = nullptr; // not needed: forced one-hot

  const auto &par = getPar();
  par.checkSupported();

  m_x_ = rpu_device.getMx();
  m_d_ = rpu_device.getMd();

  // populate parameters of CWO
  cwo_ = RPU::make_unique<ChoppedWeightOutput<T>>(this->context_, this->x_size_, this->d_size_);
  ChoppedWeightOutputParameter<T> cwo_par;

  cwo_par.use_columns = par.transfer_columns;
  cwo_par.every = 1; // will be re-set later
  cwo_par.in_chop_prob = par.in_chop_prob;
  cwo_par.out_chop_prob = par.out_chop_prob;
  cwo_par.in_chop_random = par.in_chop_random;
  cwo_->setPar(cwo_par);
  cwo_->setCounter(this->current_update_idx_);
  cwo_->setFlexibleInSize(this->transfer_fb_pass_->checkFlexibleInSize(par.transfer_io));

  if (par.usesAutoTransferEvery()) {
    if (par.units_in_mbatch) {
      RPU_FATAL("Auto transfer every is not supported for units in m_batch");
    }
  }
}

template <typename T>
void ChoppedTransferRPUDeviceCuda<T>::dumpExtra(RPU::state_t &extra, const std::string prefix) {

  BufferedTransferRPUDeviceCuda<T>::dumpExtra(extra, prefix);

  RPU::state_t state;
  RPU::insert(state, "m_x", m_x_);
  RPU::insert(state, "m_d", m_d_);
  RPU::insert(state, "d_sparsity", d_sparsity_);

  cwo_->dumpExtra(state, "cwo");

  RPU::insertWithPrefix(extra, state, prefix);
}

template <typename T>
void ChoppedTransferRPUDeviceCuda<T>::loadExtra(
    const RPU::state_t &extra, const std::string prefix, bool strict) {

  BufferedTransferRPUDeviceCuda<T>::loadExtra(extra, prefix, strict);
  auto state = RPU::selectWithPrefix(extra, prefix);

  RPU::load(state, "m_x", m_x_, strict);
  RPU::load(state, "m_d", m_d_, strict);
  RPU::load(state, "d_sparsity", d_sparsity_, strict);

  cwo_->loadExtra(state, "cwo", strict);
}

template <typename T>
int ChoppedTransferRPUDeviceCuda<T>::getTransferEvery(
    int didx, int m_batch, const PulsedUpdateMetaParameter<T> &up) const {

  const auto &par = getPar();
  if (par.usesAutoTransferEvery()) {
    T t = par.getAutoTransferEvery(this->rpucuda_device_vec_[didx]->getNumStates(), up);
    return MAX(1, (int)floorf(t));
  }
  return BufferedTransferRPUDeviceCuda<T>::getTransferEvery(didx, m_batch, up);
}

template <typename T>
void ChoppedTransferRPUDeviceCuda<T>::readMatrix(
    int device_idx, const T *in_vec, T *out_vec, int m_batch, T alpha) {
  const auto &par = getPar();

  if (device_idx != 0) {
    RPU_FATAL("Only read from device index 0.");
  }

  if (in_vec != nullptr) {
    RPU_FATAL("only one-hot transfer vectors supported.");
  }
  if (m_batch != cwo_->getNumWeightOutputs()) {
    RPU_FATAL("m_batch mismatch!");
  }

  if (m_batch == 0) {
    return;
  }

  bool in_size_flexible = this->cwo_->getFlexibleInSize();

  if (in_size_flexible) {
    // in case no input dependence, we can read out in one go

    T *output_weights = cwo_->getWeightOutputData();
    chop_t *wo_chopper_data = cwo_->getWeightOutputInChopperData();
    DiagInputIterator<T, chop_t> diag_iter(wo_chopper_data, m_batch, 0);

    if (par.transfer_columns) {
      this->transfer_fb_pass_->forwardMatrixIterator(
          output_weights, diag_iter, m_batch, false, out_vec, this->d_size_, false, m_batch, alpha,
          *this->transfer_iom_, par.transfer_io, false);

    } else {
      // backward with transfer vectors.
      this->transfer_fb_pass_->backwardMatrixIterator(
          output_weights, diag_iter, m_batch, false, out_vec, this->x_size_, false, m_batch, alpha,
          *this->transfer_iom_, par.transfer_io);
    }

  } else {

    int out_size = cwo_->getOutSize();
    int in_size = cwo_->getInSize();
    int n_pass = m_batch / in_size + 1;
    int size = in_size * out_size;
    for (int i_pass = 0; i_pass < n_pass; i_pass++) {
      // we potentially need to run multiple passes in case a
      // particular col/vec is transferred twice or more, in which case
      // the weight matrix is concatenated, to not mess up the position
      // of the row/col for the analog MV.

      // NOTE: the non-read out weights might be arbitrary value. Should
      // not matter though as input is 0 for those.

      int wo_offset = i_pass * size;

      chop_t *wo_chopper_data = cwo_->getWeightOutputInChopperData() + i_pass * in_size;
      DiagInputIterator<T, chop_t> diag_iter(
          wo_chopper_data, in_size, cwo_->getValStart() * in_size);
      T *output_weights = cwo_->getWeightOutputData() + wo_offset;
      int sub_m_batch = (i_pass < n_pass - 1) ? in_size : (m_batch - in_size * (n_pass - 1));

      if (par.transfer_columns) {
        this->transfer_fb_pass_->forwardMatrixIterator(
            output_weights, diag_iter, this->x_size_, false, out_vec + wo_offset, this->d_size_,
            false, sub_m_batch, alpha, *this->transfer_iom_, par.transfer_io, false);

      } else {
        // backward with transfer vectors.
        this->transfer_fb_pass_->backwardMatrixIterator(
            output_weights, diag_iter, this->d_size_, false, out_vec + wo_offset, this->x_size_,
            false, sub_m_batch, alpha, *this->transfer_iom_, par.transfer_io);
      }
    }
  }
}

template <typename T>
void ChoppedTransferRPUDeviceCuda<T>::writeMatrix(
    int device_idx,
    const T *in_vec,
    const T *out_vec,
    int m_batch,
    const T lr,
    const PulsedUpdateMetaParameter<T> &up) {
  const auto &par = getPar();
  // note that the ptrs might point to the current weight
  T *W = this->dev_weights_ptrs_[device_idx];

  if (device_idx != 1) {
    RPU_FATAL("Only write to device index 1.");
  }
  if (in_vec != nullptr) {
    RPU_FATAL("only one hot-transfer vectors supported.");
  }
  if (m_batch == 0) {
    return;
  }

  // NOTE: here again the out_vec are the buffer weight outputs
  // (derived from the buffer after writing the weight outputs to it)
  // that could have multiple wrap around cycles but always sequential
  // in in_size. We use the EyeIterator with offset to generate the
  // corresponding one-hot input vectors

  int in_size = par.transfer_columns ? this->x_size_ : this->d_size_;

  // this iterator will wrap around: i_slice_start is val start
  EyeInputIterator<T> eye_iter(in_size, cwo_->getValStart() * in_size);

  if (par.transfer_columns) {
    this->transfer_pwu_->update(
        eye_iter, out_vec, W, &*this->rpucuda_device_vec_[device_idx], up, fabsf(lr), m_batch,
        false, false);
  } else {
    this->transfer_pwu_->update(
        out_vec, eye_iter, W, &*this->rpucuda_device_vec_[device_idx], up, fabsf(lr), m_batch,
        false, false);
  }
}

/*********************************************************************************/
/* partially transfer using the given "readout" transfer vectors
   (with io-managed forward) and buffer the results in
   digital. Transfer only to next device if threshold is reached. */

template <typename T>
__global__ void kernelChoppedTransfer(
    T *transfer_out,
    T *W_buffer,
    const T *transfer_in,
    const chop_t *in_chopper,  // size n_wo  NOTE: is already applied to transfer_in
    const chop_t *out_chopper, // size n_wo * out_size
    const int out_size,
    const int in_size,
    const int m_batch,
    const int start_read_idx,
    const T lr_scale_in,
    const T sub_momentum,
    const int max_steps_in,
    const bool forget_buffer_in,
    const bool no_buffer) {

  const T max_steps = (T)max_steps_in;
  const int w_size = out_size * in_size;
  const int t_size = out_size * m_batch;
  const T momentum = -sub_momentum + (T)1.0;
  const bool forget_buffer = forget_buffer_in;
  const T lr_scale = lr_scale_in;

  // CAUTION: n_vec might have mulitple wraps around in_size, we need
  // to thus make sure that the same threads are working on the same
  // repeat.
  int n_repeats = (m_batch + in_size - 1) / in_size;

  RPU_CUDA_1D_KERNEL_LOOP(idx, w_size) {

    int buffer_idx = (idx + start_read_idx * out_size) % w_size;
    T omega = no_buffer ? (T)0.0 : W_buffer[buffer_idx];

    for (int i_rep = 0; i_rep < n_repeats; i_rep++) {

      int inp_idx = idx + i_rep * w_size;

      if (inp_idx >= t_size) {
        break;
      }

      chop_t out_chop = out_chopper[inp_idx];
      T x = transfer_in[inp_idx];

      x = out_chop > 0 ? x : -x;
      omega += x * lr_scale;

      // writing
      T n_steps = 0;
      if (fabs(omega) >= (T)1.0) {
        n_steps = MIN(MAX(trunc(omega), -max_steps), max_steps);

        if (!no_buffer) {
          if (forget_buffer) {
            omega *= momentum;
          } else {
            omega -= sub_momentum * n_steps;
          }
        }
      }
      transfer_out[inp_idx] = -n_steps; // negative because of LR has reverse meaning
    }
    if (!no_buffer) {
      W_buffer[buffer_idx] = omega;
    }
  }
}

template <typename T>
void ChoppedTransferRPUDeviceCuda<T>::readAndUpdate(
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
  T to_weight_granularity = this->rpucuda_device_vec_[to_device_idx]->getWeightGranularity();
  T from_weight_granularity = this->rpucuda_device_vec_[from_device_idx]->getWeightGranularity();

  T *transfer_tmp = this->context_->template getSharedBuffer<T>(RPU_BUFFER_DEVICE_0, t_size);
  T *transfer_out = this->context_->template getSharedBuffer<T>(RPU_BUFFER_DEVICE_1, t_size);
  T lr_scale = par.getTransferLRScale(
      from_weight_granularity, to_weight_granularity, lr, count_lr, cwo_->getCurrentMBatch());

  // forward/backward with transfer vectors into tmp
  this->readMatrix(from_device_idx, nullptr, transfer_tmp, n_vec, (T)1.0);

  // out-size major
  T *B = this->transfer_buffer_vec_[from_device_idx]->getData();

  int n = MIN(in_size * out_size, t_size);
  int nthreads = this->context_->getNThreads(n);
  int nblocks = this->context_->getNBlocks(n, nthreads);
  T sub_momentum = (T)1.0 - MAX(MIN(par.momentum, (T)1.0), (T)0.0);

  kernelChoppedTransfer<T><<<nblocks, nthreads, 0, this->context_->getStream()>>>(
      transfer_out, B, transfer_tmp, cwo_->getWeightOutputInChopperData(),
      cwo_->getWeightOutputOutChopperData(), out_size, in_size, n_vec, i_slice_start, lr_scale,
      sub_momentum, up.desired_BL, par.forget_buffer, par.no_buffer);

  // update according to device
  T write_lr = par.getWriteLR(to_weight_granularity);
  this->writeMatrix(to_device_idx, nullptr, transfer_out, n_vec, write_lr, up);

  this->context_->template releaseSharedBuffer<T>(RPU_BUFFER_DEVICE_0);
  this->context_->template releaseSharedBuffer<T>(RPU_BUFFER_DEVICE_1);
}

template <typename T>
T ChoppedTransferRPUDeviceCuda<T>::getPulseCountLearningRate(
    T lr, int current_m_batch, const PulsedUpdateMetaParameter<T> &up) {

  const auto &par = getPar();
  T out_count_lr;

  if (par.auto_scale && par.fast_lr > (T)0.0) {

    T transfer_every = (T)this->getTransferEvery(0, current_m_batch, up);
    out_count_lr = par.getPulseCountAutoLR(
        m_x_, m_d_, d_sparsity_, this->rpucuda_device_vec_[0]->getWeightGranularity(),
        transfer_every, up);
  } else {
    out_count_lr =
        BufferedTransferRPUDeviceCuda<T>::getPulseCountLearningRate(lr, current_m_batch, up);
    // scale so that it is constant for tile size / dw_min / bl - change
    // out_count_lr /= par.getInSize() * fabs(par.transfer_every);
  }
  return out_count_lr;
}

/*********************************************************************************/
template <typename T>
void ChoppedTransferRPUDeviceCuda<T>::transfer(
    int to_device_idx,
    int from_device_idx,
    const PulsedUpdateMetaParameter<T> &current_up,
    const T current_lr,
    const T current_count_lr) {

  const auto &par = getPar();
  int in_size = par.getInSize();
  int out_size = par.getOutSize();

  int i_slice = cwo_->getValStart();
  int n_transfers = cwo_->getNumWeightOutputs(); // could be more than in_size !
  T lr = par.getTransferLR(to_device_idx, from_device_idx, current_lr);

  readAndUpdate(
      to_device_idx, from_device_idx, i_slice, lr, current_count_lr, nullptr, n_transfers,
      par.transfer_up);
  this->current_slice_indices_[from_device_idx] = (i_slice + n_transfers) % in_size;
}

/*********************************************************************************/
template <typename T>
pwukpvec_t<T> ChoppedTransferRPUDeviceCuda<T>::getUpdateKernels(
    int m_batch, int nK32, int use_bo64, bool out_trans, const PulsedUpdateMetaParameter<T> &up) {

  pwukpvec_t<T> v;

  // Note: Every will remain FIXED for the remainder (even if
  // batchsize is changed)
  auto t_every = this->getTransferEvery(0, m_batch, up);
  cwo_->setEvery(t_every);

  v = this->rpucuda_device_vec_[0]->getUpdateKernels(m_batch, nK32, use_bo64, out_trans, up);
  for (auto &kpars : v) {
    kpars->ensureCWO();
  }

  return v;
}

/*********************************************************************************/
template <typename T>
void ChoppedTransferRPUDeviceCuda<T>::runUpdateKernel(
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
  // calling kpars->run(..,this,..) directly should cause error because  derived from abstract
  // device..
  DEBUG_OUT("start run update kernel.");

  if (x_counts_chunk != nullptr || d_counts_chunk != nullptr) {
    RPU_FATAL("Chunking not allowed here.");
  }
  if (cwo) {
    RPU_FATAL("Explicit CWO as input not allowed here.");
  }

  if (!this->fully_hidden_) {
    RPU_FATAL("Expects fully hidden fast matrix.");
  }

  if (blm->getCurrentLR() == (T)0.0) {
    return;
  }

  // set full hidden
  this->dev_weights_ptrs_[this->n_devices_ - 1] = dev_weights;

  // TODO: enable asynchronous update on a separate update stream.
  //       for that one needs to make sure that wait events are
  //       inserted in the main stream and that all the context
  //       pointers are set correctly for CWO / DEVICE etc.  always
  //       same (up) context.
  CudaContextPtr c = up_context;

  // generate the choppers, advance counter, etc
  cwo_->makeWeightOutputChoppers(blm);

  this->rpucuda_device_vec_[0]->runUpdateKernel(
      kpars, c, this->dev_weights_ptrs_[0], m_batch, blm, up, lr, dev_states, one_sided, nullptr,
      nullptr, &*cwo_);

  if (up._currently_tuning) {
    return;
  }

  const auto &par = getPar();
  if (par.auto_scale) {
    T abs_m_x;
    T abs_m_d;
    blm->getAverageAbsMax(abs_m_x, abs_m_d); // this will sync...
    par.updateAutoScale(m_x_, abs_m_x, 1);
    par.updateAutoScale(m_d_, abs_m_d, 1);
  }
  if (up.d_sparsity) {
    T d_sparsity = blm->getAverageDSparsity(); // this will sync
    par.updateAutoScale(d_sparsity_, d_sparsity, 1);
  }

  // let CWO do the counting
  this->current_update_idx_ = cwo_->getCounter();

  // Note:  higher order devices not supported for CWO

  // -- do the transfer
  if (cwo_->getNumWeightOutputs() > 0) {
    transfer(1, 0, up, lr, blm->getCurrentLR());

    // always fully hidden, reduce is no-op anyway
    this->reduceToWeights(up_context, dev_weights);
  }

  cwo_->releaseBuffers();
}

template class ChoppedTransferRPUDeviceCuda<float>;
#ifdef RPU_USE_DOUBLE
template class ChoppedTransferRPUDeviceCuda<double>;
#endif
#ifdef RPU_USE_FP16
template class ChoppedTransferRPUDeviceCuda<half_t>;
#endif

} // namespace RPU
