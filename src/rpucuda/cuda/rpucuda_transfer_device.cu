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

#include "forward_backward_pass.h"
#include "rpu_pulsed_meta_parameter.h"
#include "rpucuda_transfer_device.h"
#include <limits>
#include <memory>

namespace RPU {

/******************************************************************************************/
/* DefferenceRPUDeviceCuda

   CUDA implementation of TransferRPUDevice

*/

template <typename T> void TransferRPUDeviceCuda<T>::initialize(bool transfer_columns) {
  transfer_pwu_ =
      RPU::make_unique<PulsedWeightUpdater<T>>(this->context_, this->x_size_, this->d_size_);

  if (transfer_columns) {
    transfer_iom_ =
        RPU::make_unique<InputOutputManager<T>>(this->context_, this->x_size_, this->d_size_);
  } else {
    transfer_iom_ =
        RPU::make_unique<InputOutputManager<T>>(this->context_, this->d_size_, this->x_size_);
  }
}

template <typename T>
TransferRPUDeviceCuda<T>::TransferRPUDeviceCuda(
    CudaContextPtr c, const TransferRPUDevice<T> &rpu_device) {
  this->context_ = c;
  populateFrom(rpu_device); // use populate to call parent
};

// copy construcutor
template <typename T>
TransferRPUDeviceCuda<T>::TransferRPUDeviceCuda(const TransferRPUDeviceCuda<T> &other)
    : VectorRPUDeviceCuda<T>(other) {

  if (other.transfer_vecs_ != nullptr) {
    transfer_vecs_ = RPU::make_unique<CudaArray<T>>(*other.transfer_vecs_);
  }
  if (other.transfer_fb_pass_ != nullptr) {
    transfer_fb_pass_ =
        RPU::make_unique<ForwardBackwardPassIOManagedCuda<T>>(*other.transfer_fb_pass_);
  }
  initialize(other.getPar().transfer_columns);
  current_slice_indices_ = other.current_slice_indices_;
  fully_hidden_ = other.fully_hidden_;

  this->context_->synchronizeDevice();
};

// copy assignment
template <typename T>
TransferRPUDeviceCuda<T> &
TransferRPUDeviceCuda<T>::operator=(const TransferRPUDeviceCuda<T> &other) {
  TransferRPUDeviceCuda<T> tmp(other);
  swap(*this, tmp);
  this->context_->synchronize();
  return *this;
};

// move constructor
template <typename T>
TransferRPUDeviceCuda<T>::TransferRPUDeviceCuda(TransferRPUDeviceCuda<T> &&other) {
  *this = std::move(other);
};

// move assignment
template <typename T>
TransferRPUDeviceCuda<T> &TransferRPUDeviceCuda<T>::operator=(TransferRPUDeviceCuda<T> &&other) {
  VectorRPUDeviceCuda<T>::operator=(std::move(other));

  transfer_vecs_ = std::move(other.transfer_vecs_);

  current_slice_indices_ = other.current_slice_indices_;
  other.current_slice_indices_.clear();

  fully_hidden_ = other.fully_hidden_;
  transfer_pwu_ = std::move(other.transfer_pwu_);
  transfer_iom_ = std::move(other.transfer_iom_);
  transfer_fb_pass_ = std::move(other.transfer_fb_pass_);

  return *this;
};

template <typename T>
void TransferRPUDeviceCuda<T>::populateFrom(const AbstractRPUDevice<T> &rpu_device_in) {

  const auto &rpu_device = dynamic_cast<const TransferRPUDevice<T> &>(rpu_device_in);
  if (&rpu_device == nullptr) {
    RPU_FATAL("populateFrom expects TransferRPUDevice.");
  }

  VectorRPUDeviceCuda<T>::populateFrom(rpu_device_in);

  const auto &par = getPar();

  if (!par.singleDeviceUpdate()) {
    RPU_FATAL("Multiple device update not supported for Transfer Device");
  }

  if (!par.same_context) {
    RPU_FATAL("Only same context supported");
  }

  if (this->n_devices_ < 2) {
    RPU_FATAL("Expect at least two devices.");
  }

  for (int j = 1; j < this->n_devices_ - 1; j++) {
    if (par.transfer_every_vec[0] > par.transfer_every_vec[j]) {
      RPU_FATAL("Later transfer periods need to be larger than first for CUDA.");
    }
  }
  int in_size = par.getInSize();
  transfer_vecs_ = RPU::make_unique<CudaArray<T>>(
      this->context_, in_size * in_size, rpu_device.getTransferVecs());

  initialize(par.transfer_columns); // pwu/iom

  transfer_fb_pass_ = RPU::make_unique<ForwardBackwardPassIOManagedCuda<T>>(
      this->context_, this->x_size_, this->d_size_);
  transfer_fb_pass_->populateFrom(rpu_device.getTransferFBPass().getFBParameter());

  current_slice_indices_.resize(this->n_devices_ - 1);
  std::fill(current_slice_indices_.begin(), current_slice_indices_.end(), (int)0);

  this->current_update_idx_ = 0;

  fully_hidden_ = par.fullyHidden();
}

template <typename T>
void TransferRPUDeviceCuda<T>::dumpExtra(RPU::state_t &extra, const std::string prefix) {

  VectorRPUDeviceCuda<T>::dumpExtra(extra, prefix);

  RPU::state_t state;

  RPU::insert(state, "current_slice_indices", current_slice_indices_);
  // RPU::insert(state, "transfer_vecs", transfer_vecs_);

  transfer_fb_pass_->dumpExtra(state, "transfer_fb_pass");
  transfer_pwu_->dumpExtra(state, "transfer_pwu");
  transfer_iom_->dumpExtra(state, "transfer_iom");

  RPU::insertWithPrefix(extra, state, prefix);
}

template <typename T>
void TransferRPUDeviceCuda<T>::loadExtra(
    const RPU::state_t &extra, const std::string prefix, bool strict) {

  VectorRPUDeviceCuda<T>::loadExtra(extra, prefix, strict);

  auto state = RPU::selectWithPrefix(extra, prefix);

  RPU::load(state, "current_slice_indices", current_slice_indices_, strict);
  if (state.count("transfer_vecs")) {
    RPU::load(this->context_, state, "transfer_vecs", transfer_vecs_, strict);
  }
  transfer_fb_pass_->loadExtra(state, "transfer_fb_pass", strict);
  transfer_pwu_->loadExtra(state, "transfer_pwu", strict);
  transfer_iom_->loadExtra(state, "transfer_iom", strict);
}

/*********************************************************************************/
/* getPulseCountLearningRate */
/* Here we compute the LR for the A matrix (the SGD update). Because
   of the device properties it is beneficial to use a constant LR
   here, but scale the buffer with the scheduled SGD learning rate
   later*/
template <typename T>
T TransferRPUDeviceCuda<T>::getPulseCountLearningRate(
    T learning_rate, int current_m_batch, const PulsedUpdateMetaParameter<T> &up) {

  const auto &par = getPar();

  if (par.fast_lr > (T)0.0) {
    return par.fast_lr;
  } else {
    return PulsedRPUDeviceCudaBase<T>::getPulseCountLearningRate(
        learning_rate, current_m_batch, up);
  }
}

template <typename T>
void TransferRPUDeviceCuda<T>::readMatrix(
    int device_idx, const T *in_vec, T *out_vec, int m_batch, T alpha) {
  const auto &par = getPar();

  if (par.transfer_columns) {
    // forward with transfer vectors
    transfer_fb_pass_->forwardMatrixIterator(
        this->dev_weights_ptrs_[device_idx], in_vec, this->x_size_, false, out_vec, this->d_size_,
        false, m_batch, alpha, *transfer_iom_, par.transfer_io, false);

  } else {
    // backward with transfer vectors
    transfer_fb_pass_->backwardMatrixIterator(
        this->dev_weights_ptrs_[device_idx], in_vec, this->d_size_, false, out_vec, this->x_size_,
        false, m_batch, alpha, *transfer_iom_, par.transfer_io);
  }
}

template <typename T>
void TransferRPUDeviceCuda<T>::writeMatrix(
    int device_idx,
    const T *in_vec,
    const T *out_vec,
    int m_batch,
    const T lr,
    const PulsedUpdateMetaParameter<T> &up) {
  const auto &par = getPar();
  // note that the ptrs might point to the current weight
  T *W = this->dev_weights_ptrs_[device_idx];

  if (par.transfer_columns) {
    transfer_pwu_->update(
        in_vec, out_vec, W, &*this->rpucuda_device_vec_[device_idx], up, fabsf(lr), m_batch, false,
        false);
  } else {
    transfer_pwu_->update(
        out_vec, in_vec, W, &*this->rpucuda_device_vec_[device_idx], up, fabsf(lr), m_batch, false,
        false);
  }
}

/*********************************************************************************/
/* partially transfer using the given "readout" transfer vectors
   (with io-managed forward) and the usualy device update */
template <typename T>
void TransferRPUDeviceCuda<T>::readAndUpdate(
    int to_device_idx,
    int from_device_idx,
    int i_col_start,
    const T lr,
    const T count_lr,
    const T *vec,
    const int n_vec,
    const PulsedUpdateMetaParameter<T> &up) {
  if (lr == (T)0.0) {
    return;
  }

  const auto &par = getPar();
  int out_size = par.getOutSize();
  int t_size = n_vec * out_size; // transfer size

  T *out_vec = this->context_->template getSharedBuffer<T>(RPU_BUFFER_DEVICE_0, t_size);

  // forward / backward with transfer vectors. Since we need *positive*
  // update, LR needs to be negative. However, this is not supported
  // in the PWU really. Thus we scale the output by -1 and set alpha
  // accordingly
  readMatrix(from_device_idx, vec, out_vec, n_vec, -1.0);
  // update according to device
  writeMatrix(to_device_idx, vec, out_vec, n_vec, fabsf(lr), up);

  this->context_->template releaseSharedBuffer<T>(RPU_BUFFER_DEVICE_0);
}

/*********************************************************************************/
template <typename T>
void TransferRPUDeviceCuda<T>::transfer(
    int to_device_idx,
    int from_device_idx,
    const PulsedUpdateMetaParameter<T> &current_up,
    const T current_lr,
    const T current_count_lr) {

  int i_slice = current_slice_indices_[from_device_idx];
  const auto &par = getPar();

  int in_size = par.getInSize();
  int out_size = par.getOutSize();

  if (par.random_selection) {
    i_slice = MAX(MIN(floorf((float)this->rw_rng_.sampleUniform() * in_size), in_size - 1), 0.0f);
  }

  // transfer_vecs_ is always in_size-major (that is trans==false)
  T *tvec = transfer_vecs_->getData() + i_slice * in_size;
  int n_rest = in_size - i_slice;

  T lr = par.getTransferLR(to_device_idx, from_device_idx, current_lr);

  const PulsedUpdateMetaParameter<T> *up;
  up = &par.transfer_up;

  int n_transfers = MIN(par.n_reads_per_transfer, in_size);

  if (n_rest < n_transfers) {
    // rest
    readAndUpdate(to_device_idx, from_device_idx, i_slice, lr, current_count_lr, tvec, n_rest, *up);
    // from beginning
    readAndUpdate(
        to_device_idx, from_device_idx, 0, lr, current_count_lr, transfer_vecs_->getData(),
        n_transfers - n_rest, *up);

  } else {
    readAndUpdate(
        to_device_idx, from_device_idx, i_slice, lr, current_count_lr, tvec, n_transfers, *up);
  }

  if (par.transfer_columns && this->rw_rng_.sampleUniform() < par.with_reset_prob) {
    // COL-wise prob!! device-wise reset_prob=1
    this->rpucuda_device_vec_[from_device_idx]->resetCols(
        this->dev_weights_ptrs_[from_device_idx], i_slice, n_transfers, 1);
  }

  current_slice_indices_[from_device_idx] = (i_slice + n_transfers) % in_size;
}

/*********************************************************************************/
template <typename T>
int TransferRPUDeviceCuda<T>::getTransferEvery(
    int didx, int m_batch, const PulsedUpdateMetaParameter<T> &up) const {
  UNUSED(up);
  if (getPar().units_in_mbatch) {
    return MAX((int)ceilf((float)getPar().transfer_every_vec[didx] * m_batch), 0);
  } else {
    return MAX((int)roundf(getPar().transfer_every_vec[didx]), 0);
  }
}

/*********************************************************************************/
template <typename T> inline int getNChunks(int m_batch, T every) {
  if (every <= 0) {
    return 1;
  } else {
    return MAX((int)(round((float)m_batch / every)), 1); // take next integer for period
  }
}

inline int getChunkSize(int m_batch, int nchunks) {
  return (m_batch + nchunks - 1) / nchunks; // to ensure not to have residual
}

inline uint64_t getNextTransfer(uint64_t current_update_idx, int transfer_every) {

  if (transfer_every <= 0) {
    return std::numeric_limits<uint64_t>::max();
  }
  return current_update_idx + transfer_every - (current_update_idx % transfer_every);
}

/*********************************************************************************/
template <typename T>
pwukpvec_t<T> TransferRPUDeviceCuda<T>::getUpdateKernels(
    int m_batch, int nK32, int use_bo64, bool out_trans, const PulsedUpdateMetaParameter<T> &up) {

  pwukpvec_t<T> v;

  // just get approx chunk size for tuning
  int nchunks = getNChunks(m_batch, getTransferEvery(0, m_batch, up));
  int chunk_size = getChunkSize(m_batch, nchunks);
  // use the first device as the "FAST" device that gets updates with the true gradients.
  v = this->rpucuda_device_vec_[0]->getUpdateKernels(chunk_size, nK32, use_bo64, out_trans, up);

  if (nchunks > 1) {
    for (auto &kpars : v) {
      kpars->ensureChunk();
    }
  }

  // CWO not supported
  for (auto &kpars : v) {
    kpars->disableCWO();
  }

  return v;
}

/*********************************************************************************/
template <typename T>
void TransferRPUDeviceCuda<T>::runUpdateKernel(
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
  DEBUG_CALL(kpars->print(););

  if (fully_hidden_) {
    this->dev_weights_ptrs_[this->n_devices_ - 1] = dev_weights;
  }

  // always same (up) context.
  CudaContextPtr c = up_context;

  if (x_counts_chunk != nullptr || d_counts_chunk != nullptr) {
    RPU_FATAL("Chunking not allowed here.");
  }
  if (cwo != nullptr) {
    RPU_FATAL("A given CWO not allowed here.");
  }

  // use chunking instead

  // only look at first device here as it makes no sense to transfer
  // the higer order devices more often
  int transfer_every = getTransferEvery(0, m_batch, up);
  auto next_transfer = getNextTransfer(this->current_update_idx_, transfer_every);

  if (next_transfer >= m_batch + this->current_update_idx_) {

    // just update the whole batch we do not call kpars directly to
    // also make possible to have non-pulsed devices. Note that only
    // one device is directly updated with the gradients, thus
    // tuning kpars are always unique (and valid to that rpu_device
    // only). However, the other RPU device kernels will be tuned
    // during transfer, since we use a seperate PWU object

    this->rpucuda_device_vec_[0]->runUpdateKernel(
        kpars, c, this->dev_weights_ptrs_[0], m_batch, blm, up, lr, dev_states, one_sided);

    if (up._currently_tuning) {
      return;
    }

    this->current_update_idx_ += m_batch; // first update idx

    // transfer
    if (next_transfer == this->current_update_idx_) {
      transfer(1, 0, up, lr, blm->getCurrentLR());
    }

    // other higher order devices
    for (int j = 1; j < this->n_devices_ - 1; j++) {
      // all transfer periods will be rounded up to chunk_sizes
      auto higher_order_next_transfer =
          getNextTransfer(this->current_update_idx_ - m_batch, getTransferEvery(j, m_batch, up));
      if (higher_order_next_transfer <= this->current_update_idx_) {
        transfer(j + 1, j, up, lr, 1.0);
      }
    }

  } else {
    // transfer is inbetween the mbatch, we need to chunk in some way

    // need to do it chunkwise
    int batch_start = 0;
    int nK32 = blm->getNK32Current();
    auto x_counts = blm->getXCountsData();
    auto d_counts = blm->getDCountsData();
    uint64_t final_update_idx = this->current_update_idx_ + m_batch;

    while (next_transfer <= final_update_idx) {

      int current_m_batch = (int)(next_transfer - this->current_update_idx_);

      this->rpucuda_device_vec_[0]->runUpdateKernel(
          kpars,
          c, // same context since sequence important
          this->dev_weights_ptrs_[0], current_m_batch, blm, up, lr, dev_states, one_sided,
          x_counts + batch_start * this->x_size_ * nK32, // always non-trans
          d_counts + batch_start * this->d_size_ * nK32);

      if (up._currently_tuning) {
        return;
      }

      this->current_update_idx_ += current_m_batch; // first update idx
      batch_start += current_m_batch;

      // transfer
      if (next_transfer == this->current_update_idx_) {
        transfer(1, 0, up, lr, blm->getCurrentLR());
      }

      // other higher order devices
      for (int j = 1; j < this->n_devices_ - 1; j++) {
        // all transfer periods will be rounded up to chunk_sizes
        auto higher_order_next_transfer = getNextTransfer(
            this->current_update_idx_ - current_m_batch, getTransferEvery(j, m_batch, up));
        if (higher_order_next_transfer <= this->current_update_idx_) {
          transfer(j + 1, j, up, lr, 1.0);
        }
      }
      next_transfer = getNextTransfer(this->current_update_idx_, transfer_every);
    }
  }
  // only reduce at end
  this->reduceToWeights(up_context, dev_weights);
}

/*********************************************************************************/
template <typename T>
void TransferRPUDeviceCuda<T>::reduceToWeights(CudaContextPtr context, T *dev_weights) {

  if (!fully_hidden_) {
    VectorRPUDeviceCuda<T>::reduceToWeights(context, dev_weights);
  }
}

template <typename T>
void TransferRPUDeviceCuda<T>::decayWeights(T *dev_weights, T alpha, bool bias_no_decay) {

  if (fully_hidden_) {
    this->dev_weights_ptrs_[this->n_devices_ - 1] = dev_weights;
  }
  VectorRPUDeviceCuda<T>::decayWeights(dev_weights, alpha, bias_no_decay);
}

template <typename T>
void TransferRPUDeviceCuda<T>::decayWeights(T *dev_weights, bool bias_no_decay) {

  if (fully_hidden_) {
    this->dev_weights_ptrs_[this->n_devices_ - 1] = dev_weights;
  }

  VectorRPUDeviceCuda<T>::decayWeights(dev_weights, bias_no_decay);
}

template <typename T>
void TransferRPUDeviceCuda<T>::driftWeights(T *dev_weights, T time_since_last_call) {

  if (fully_hidden_) {
    this->dev_weights_ptrs_[this->n_devices_ - 1] = dev_weights;
  }

  VectorRPUDeviceCuda<T>::driftWeights(dev_weights, time_since_last_call);
}

template <typename T> void TransferRPUDeviceCuda<T>::diffuseWeights(T *dev_weights) {

  if (fully_hidden_) {
    this->dev_weights_ptrs_[this->n_devices_ - 1] = dev_weights;
  }

  VectorRPUDeviceCuda<T>::diffuseWeights(dev_weights);
}

template <typename T> void TransferRPUDeviceCuda<T>::clipWeights(T *dev_weights, T clip) {

  if (fully_hidden_) {
    this->dev_weights_ptrs_[this->n_devices_ - 1] = dev_weights;
  }

  VectorRPUDeviceCuda<T>::clipWeights(dev_weights, clip);
}

template <typename T>
void TransferRPUDeviceCuda<T>::resetCols(T *dev_weights, int start_col, int n_cols, T reset_prob) {

  // only first will be reset
  this->rpucuda_device_vec_[0]->resetCols(
      this->dev_weights_ptrs_[0], start_col, n_cols, reset_prob);
  this->reduceToWeights(this->context_, dev_weights);
}

template class TransferRPUDeviceCuda<float>;
#ifdef RPU_USE_DOUBLE
template class TransferRPUDeviceCuda<double>;
#endif
#ifdef RPU_USE_FP16
template class TransferRPUDeviceCuda<half_t>;
#endif

} // namespace RPU
