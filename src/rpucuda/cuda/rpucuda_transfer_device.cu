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

#include "forward_backward_pass.h"
#include "rpu_pulsed_meta_parameter.h"
#include "rpucuda_transfer_device.h"
#include <memory>

namespace RPU {

/******************************************************************************************/
/* DefferenceRPUDeviceCuda

   CUDA implementation of TransferRPUDevice

*/

template <typename T> void TransferRPUDeviceCuda<T>::initialize() {
  transfer_pwu_ = RPU::make_unique<PulsedWeightUpdater<T>>(this->context_, this->x_size_, this->d_size_);
  transfer_iom_ = RPU::make_unique<InputOutputManager<T>>(this->context_, this->x_size_, this->d_size_);
}

template <typename T>
TransferRPUDeviceCuda<T>::TransferRPUDeviceCuda(
    CudaContext *c, const TransferRPUDevice<T> &rpu_device) {
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

  initialize();

  current_col_indices_ = other.current_col_indices_;
  fully_hidden_ = other.fully_hidden_;
  this->context_->synchronizeDevice();
};

// copy assignment
template <typename T>
TransferRPUDeviceCuda<T> &
TransferRPUDeviceCuda<T>::operator=(const TransferRPUDeviceCuda<T> &other) {
  TransferRPUDeviceCuda<T> tmp(other);
  swap(*this, tmp);
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
  current_col_indices_ = other.current_col_indices_;
  other.current_col_indices_.clear();
  fully_hidden_ = other.fully_hidden_;

  transfer_pwu_ = std::move(other.transfer_pwu_);
  transfer_iom_ = std::move(other.transfer_iom_);
  // ignore transfer_tmp_ or RNG

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

  transfer_vecs_ = RPU::make_unique<CudaArray<T>>(
      this->context_, this->x_size_ * this->x_size_, rpu_device.getTransferVecs());

  initialize(); // pwu/iom

  current_col_indices_.resize(this->n_devices_ - 1);
  std::fill(current_col_indices_.begin(), current_col_indices_.end(), 0);

  this->current_update_idx_ = 0;

  fully_hidden_ = par.fullyHidden();
}

/* partially transfer using the given "readout" transfer vectors
   (with io-managed forward) and the usualy device update */
template <typename T>
void TransferRPUDeviceCuda<T>::forwardUpdate(
    int to_device_idx,
    int from_device_idx,
    int i_col_start,
    const T lr,
    const T *x_input,
    const int n_vec,
    const bool trans,
    const PulsedUpdateMetaParameter<T> &up) {
  if (!lr) {
    return;
  }

  if ((transfer_tmp_ == nullptr) || transfer_tmp_->getSize() < n_vec * this->d_size_) {
    transfer_tmp_ = RPU::make_unique<CudaArray<T>>(this->context_, this->d_size_ * n_vec);
  }

  // forward with transfer vectors
  RPU::detail::forwardMatrixIteratorIOManaged(
      this->context_, this->dev_weights_ptrs_[from_device_idx], x_input, this->x_size_, trans,
      transfer_tmp_->getData(), this->d_size_, trans, n_vec,
      (T)1.0, // additional output scaling. Obey?
      *transfer_iom_, getPar().transfer_io, false);

  // update according to device
  T *W = this->dev_weights_ptrs_[to_device_idx]; /// note that the ptrs might point to the current
                                                 /// weight

  // since we need *positive* update, LR needs to be
  // negative. However, this is not supported in the PWU
  // really. Thus we scale the temp-vector by -1
  RPU::math::scal(this->context_, this->d_size_ * n_vec, (T)-1.0, transfer_tmp_->getData(), 1);

  transfer_pwu_->update(
      x_input,                       // this is the transfer vector (x_size)
      transfer_tmp_->getDataConst(), // this should be d_size
      W, &*this->rpucuda_device_vec_[to_device_idx], up, fabs(lr), n_vec, trans, trans);
}

template <typename T>
void TransferRPUDeviceCuda<T>::transfer(
    int to_device_idx,
    int from_device_idx,
    const PulsedUpdateMetaParameter<T> &current_up,
    const T current_lr) {
  int i_col = current_col_indices_[from_device_idx];
  const auto &par = getPar();
  if (par.random_column) {
    i_col = MAX(MIN(floor(this->rw_rng_.sampleUniform() * this->x_size_), this->x_size_ - 1), 0);
  }

  // transfer_vecs_ is always x_size-major (that is trans==false)
  T *tvec = transfer_vecs_->getData() + i_col * this->x_size_;
  int n_rest = this->x_size_ - i_col;

  T lr = par.getTransferLR(to_device_idx, from_device_idx, current_lr);

  const PulsedUpdateMetaParameter<T> *up;
  up = &par.transfer_up;

  int n_transfers = MIN(par.n_cols_per_transfer, this->x_size_);

  if (n_rest < n_transfers) {

    // rest
    forwardUpdate(to_device_idx, from_device_idx, i_col, lr, tvec, n_rest, false, *up);
    // from beginning
    forwardUpdate(
        to_device_idx, from_device_idx, 0, lr, transfer_vecs_->getData(), n_transfers - n_rest,
        false, *up);

  } else {
    forwardUpdate(to_device_idx, from_device_idx, i_col, lr, tvec, n_transfers, false, *up);
  }

  if (this->rw_rng_.sampleUniform() <
      par.with_reset_prob) { // COL-wise prob!! device-wise reset_prob=1
    this->rpucuda_device_vec_[from_device_idx]->resetCols(
        this->dev_weights_ptrs_[from_device_idx], i_col, n_transfers, 1);
  }

  current_col_indices_[from_device_idx] = (i_col + n_transfers) % this->x_size_;
}

template <typename T> int TransferRPUDeviceCuda<T>::getTransferEvery(int didx, int m_batch) const {

  if (getPar().units_in_mbatch) {
    return MAX(RPU_ROUNDFUN(getPar().transfer_every_vec[didx] * m_batch), 0);
  } else {
    return MAX(RPU_ROUNDFUN(getPar().transfer_every_vec[didx]), 0);
  }
}

template <typename T>
pwukpvec_t<T> TransferRPUDeviceCuda<T>::getUpdateKernels(
    int m_batch, int nK32, int use_bo64, bool out_trans, const PulsedUpdateMetaParameter<T> &up) {

  pwukpvec_t<T> v;

  // use the first device as the "FAST" device that gets updates with the true gradients.
  v = this->rpucuda_device_vec_[0]->getUpdateKernels(m_batch, nK32, use_bo64, out_trans, up);

  if (RPU_ROUNDFUN((T)m_batch / getTransferEvery(0, m_batch)) > 1) {
    for (auto &kpars : v) {
      kpars->ensureChunk();
    }
  }

  return v;
}

template <typename T>
void TransferRPUDeviceCuda<T>::runUpdateKernel(
    pwukp_t<T> kpars,
    CudaContext *up_context,
    T *dev_weights,
    int m_batch,
    const BitLineMaker<T> *blm,
    const PulsedUpdateMetaParameter<T> &up,
    curandState_t *dev_states,
    int one_sided,
    uint32_t *x_counts_chunk,
    uint32_t *d_counts_chunk) {
  // calling kpars->run(..,this,..) directly should cause error because  derived from abstract
  // device..
  DEBUG_OUT("start run update kernel.");
  DEBUG_CALL(kpars->print(););

  if (fully_hidden_) {
    this->dev_weights_ptrs_[this->n_devices_ - 1] = dev_weights;
  }

  // always same (up) context.
  CudaContext *c = up_context;

  if (x_counts_chunk != nullptr || d_counts_chunk != nullptr) {
    RPU_FATAL("Chunking not allowed here.");
  }

  int nchunks =
      RPU_ROUNDFUN((T)m_batch / getTransferEvery(0, m_batch)); // take next integer for period

  if (nchunks <= 1) {

    // just update the whole batch we do not call kpars directly to
    // also make possible to have non-pulsed devices. Note that only
    // one device is directly updated with the gradients, thus
    // tuning kpars are always unique (and valid to that rpu_device
    // only). However, the other RPU device kernels will be tuned
    // during transfer, since we use a seperate PWU object

    this->rpucuda_device_vec_[0]->runUpdateKernel(
        kpars, c, this->dev_weights_ptrs_[0], m_batch, blm, up, dev_states, one_sided);

    this->current_update_idx_ += m_batch; // first update idx

    if (!up._currently_tuning) {

      for (int j = 0; j < this->n_devices_ - 1; j++) {
        // all transfer periods will be rounded up to batches.
        int period = (getTransferEvery(j, m_batch) + m_batch - 1) / m_batch; // in m_batch
        if (this->current_update_idx_ / m_batch % period == 0) {
          transfer(j + 1, j, up, blm->getCurrentLR());
        }
      }
    }

  } else {
    // need to do it chunkwise
    int chunk_size = (m_batch + nchunks - 1) / nchunks; // to ensure not to have residual

    for (int i_chunk = 0; i_chunk < nchunks; i_chunk++) {

      int batch_start = i_chunk * chunk_size;

      // note that last chunk might be smaller.
      T current_m_batch = chunk_size - MAX(batch_start + chunk_size - m_batch, 0);

      this->rpucuda_device_vec_[0]->runUpdateKernel(
          kpars,
          c, // same context since sequence important
          this->dev_weights_ptrs_[0], current_m_batch, blm, up, dev_states, one_sided,
          blm->getXCountsData() + i_chunk * this->x_size_ * up.getNK32Default(), // always non-trans
          blm->getDCountsData() + i_chunk * this->d_size_ * up.getNK32Default());

      this->current_update_idx_ += current_m_batch; // first update idx

      if (!up._currently_tuning) {
        // transfer

        for (int j = 0; j < this->n_devices_ - 1; j++) {
          // all transfer periods will be rounded up to chunk_sizes
          int period = (getTransferEvery(j, m_batch) + chunk_size - 1) / chunk_size;
          if (this->current_update_idx_ / chunk_size % period == 0) {
            transfer(j + 1, j, up, blm->getCurrentLR());
          }
        }
      }
    }
  }
  // only reduce at end
  this->reduceToWeights(up_context, dev_weights);
}

template <typename T>
void TransferRPUDeviceCuda<T>::reduceToWeights(CudaContext *context, T *dev_weights) {

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

template class TransferRPUDeviceCuda<float>;
#ifdef RPU_USE_DOUBLE
template class TransferRPUDeviceCuda<double>;
#endif
} // namespace RPU
