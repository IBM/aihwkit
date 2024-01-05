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
#include "rpucuda_onesided_device.h"
#include <memory>

namespace RPU {

/******************************************************************************************/
/* OneSidedRPUDeviceCuda

   CUDA implementation of OneSidedRPUDevice

*/

template <typename T>
OneSidedRPUDeviceCuda<T>::OneSidedRPUDeviceCuda(
    CudaContextPtr c, const OneSidedRPUDevice<T> &rpu_device) {
  this->context_ = c;
  populateFrom(rpu_device); // use populate to call parent
};

// copy construcutor
template <typename T>
OneSidedRPUDeviceCuda<T>::OneSidedRPUDeviceCuda(const OneSidedRPUDeviceCuda<T> &other)
    : VectorRPUDeviceCuda<T>(other) {
  g_plus_ = other.g_plus_;
  g_minus_ = other.g_minus_;
  refresh_counter_ = other.refresh_counter_;

  if (other.refresh_pwu_) {
    refresh_pwu_ =
        RPU::make_unique<PulsedWeightUpdater<T>>(this->context_, this->x_size_, this->d_size_);
  }

  if (other.refresh_iom_) {
    refresh_iom_ =
        RPU::make_unique<InputOutputManager<T>>(this->context_, this->x_size_, this->d_size_);
  }

  // only fb_pass needs copy
  if (other.refresh_fb_pass_) {
    refresh_fb_pass_ =
        RPU::make_unique<ForwardBackwardPassIOManagedCuda<T>>(*other.refresh_fb_pass_);
  }

  dev_refresh_vecs_ = nullptr;
  if (other.dev_refresh_vecs_) {
    dev_refresh_vecs_ =
        RPU::make_unique<CudaArray<T>>(this->context_, this->x_size_ * this->x_size_);
    dev_refresh_vecs_->assign(*other.dev_refresh_vecs_);
  }
  dev_reduce_weightening_inverted_ = nullptr;
  if (other.dev_reduce_weightening_inverted_ != nullptr) {
    dev_reduce_weightening_inverted_ = RPU::make_unique<CudaArray<T>>(this->context_, 2);
    dev_reduce_weightening_inverted_->assign(*other.dev_reduce_weightening_inverted_);
  }
  this->context_->synchronize();
};

// copy assignment
template <typename T>
OneSidedRPUDeviceCuda<T> &
OneSidedRPUDeviceCuda<T>::operator=(const OneSidedRPUDeviceCuda<T> &other) {
  OneSidedRPUDeviceCuda<T> tmp(other);
  swap(*this, tmp);
  this->context_->synchronize();
  return *this;
};

// move constructor
template <typename T>
OneSidedRPUDeviceCuda<T>::OneSidedRPUDeviceCuda(OneSidedRPUDeviceCuda<T> &&other) {
  *this = std::move(other);
};

// move assignment
template <typename T>
OneSidedRPUDeviceCuda<T> &OneSidedRPUDeviceCuda<T>::operator=(OneSidedRPUDeviceCuda<T> &&other) {
  VectorRPUDeviceCuda<T>::operator=(std::move(other));

  refresh_pwu_ = std::move(other.refresh_pwu_);
  refresh_iom_ = std::move(other.refresh_iom_);
  refresh_fb_pass_ = std::move(other.refresh_fb_pass_);
  dev_refresh_vecs_ = std::move(other.dev_refresh_vecs_);

  refresh_counter_ = other.refresh_counter_;
  g_plus_ = other.g_plus_;
  g_minus_ = other.g_minus_;
  dev_reduce_weightening_inverted_ = std::move(other.dev_reduce_weightening_inverted_);

  return *this;
};

template <typename T>
void OneSidedRPUDeviceCuda<T>::populateFrom(const AbstractRPUDevice<T> &rpu_device_in) {

  const auto *rpu_device = dynamic_cast<const OneSidedRPUDevice<T> *>(&rpu_device_in);
  if (rpu_device == nullptr) {
    RPU_FATAL("Expect OneSidedRPUDevice.");
  }

  VectorRPUDeviceCuda<T>::populateFrom(rpu_device_in);

  refresh_pwu_ =
      RPU::make_unique<PulsedWeightUpdater<T>>(this->context_, this->x_size_, this->d_size_);
  refresh_iom_ =
      RPU::make_unique<InputOutputManager<T>>(this->context_, this->x_size_, this->d_size_);
  refresh_fb_pass_ = RPU::make_unique<ForwardBackwardPassIOManagedCuda<T>>(
      this->context_, this->x_size_, this->d_size_);
  refresh_fb_pass_->populateFrom(rpu_device->getRefreshFBPass().getFBParameter());
  this->context_->synchronize();

  rpu_device->getGIndices(g_plus_, g_minus_);
  dev_refresh_vecs_ = RPU::make_unique<CudaArray<T>>(
      this->context_, this->x_size_ * this->x_size_, rpu_device->getRefreshVecs());

  // check refresh
  const auto &par = getPar();
  if ((par.refresh_every > 0) && (!par.units_in_mbatch)) {
    RPU_FATAL("refresh_every needs to be in m_batch units for GPU (turn units_in_mbatch on).");
  }

  // inverted
  std::vector<T> rw_inv(2);
  const T *rw = rpu_device->getReduceWeightening();
  rw_inv[0] = rw[1];
  rw_inv[1] = rw[0];

  this->dev_reduce_weightening_->assign(rw);
  dev_reduce_weightening_inverted_ = RPU::make_unique<CudaArray<T>>(this->context_, 2, &rw_inv[0]);

  this->current_update_idx_ = 0;
  refresh_counter_ = rpu_device->getRefreshCount();
  this->context_->synchronize();
}

template <typename T>
void OneSidedRPUDeviceCuda<T>::resetCols(T *dev_weights, int start_col, int n_cols, T reset_prob) {
  VectorRPUDeviceCuda<T>::resetCols(dev_weights, start_col, n_cols, reset_prob);
}

template <typename T>
__global__ void kernelRefreshWeights(
    T *readout_p,
    T *readout_m,
    char *reset_msk,
    const int size_in,
    int *dev_refresh_counters,
    const T wp_max_thres_in, // upper_thres * |w_max|
    const T wp_min_thres_in, // lower_thres * |w_min|
    const T wm_max_thres_in, // lower_thres * |w_max|
    const T wm_min_thres_in  // upper_thres * |w_min|
) {
  volatile unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;

  const T wp_max_thres = wp_max_thres_in;
  const T wp_min_thres = wp_min_thres_in;
  const T wm_max_thres = wm_max_thres_in;
  const T wm_min_thres = wm_min_thres_in;
  int *p_counter = dev_refresh_counters;
  int *m_counter = p_counter + 1;

  int size = size_in;
  int total_threads = blockDim.x * gridDim.x;
  for (int idx = tid; idx < size; idx += total_threads) {

    T wp = readout_p[idx];
    T wm = readout_m[idx];
    T wp_out = 0.0;
    T wm_out = 0.0;

    bool wp_larger = wp > wm;
    bool refresh_if = wp_larger ? (wp > wp_max_thres && wm > wp_min_thres)
                                : (wp > wm_max_thres && wm > wm_min_thres);

    reset_msk[idx] = refresh_if;

    if (refresh_if) {

      if (wp_larger) {
        // positive
        atomicAdd(p_counter, 1);
        wp_out = -(wp - wm); // negate for correct direction
      } else {
        // negative
        atomicAdd(m_counter, 1);
        wm_out = -(wm - wp); // negate for correct direction
      }
    }

    readout_p[idx] = wp_out;
    readout_m[idx] = wm_out;
  }
}

template <typename T> int OneSidedRPUDeviceCuda<T>::refreshWeights() {
  // refreshes the device to avoid saturation

  const auto &par = getPar();
  auto *rpucuda_device_p =
      static_cast<PulsedRPUDeviceCuda<T> *>(&*this->rpucuda_device_vec_[g_plus_]);
  auto *rpucuda_device_m =
      static_cast<PulsedRPUDeviceCuda<T> *>(&*this->rpucuda_device_vec_[g_minus_]);
  T *weights_p = this->dev_weights_ptrs_[g_plus_];
  T *weights_m = this->dev_weights_ptrs_[g_minus_];

  if (dev_refresh_tmp_p_ == nullptr || dev_refresh_tmp_p_->getSize() < (size_t)this->size_) {
    dev_refresh_tmp_p_ = RPU::make_unique<CudaArray<T>>(this->context_, this->size_);
    dev_refresh_tmp_m_ = RPU::make_unique<CudaArray<T>>(this->context_, this->size_);
    dev_reset_msk_ = RPU::make_unique<CudaArray<char>>(this->context_, this->size_);
  }
  if (dev_refresh_counters_ == nullptr) {
    dev_refresh_counters_ = RPU::make_unique<CudaArray<int>>(this->context_, 2);
  }

  int nthreads = this->context_->getNThreads();
  if (!nblocks_batch_max_) {
    nblocks_batch_max_ =
        this->context_->getSMCount() * (this->context_->maxThreadsPerBlock() / nthreads);
  }
  int nblocks = MIN(this->context_->getNBlocks(this->size_, nthreads), nblocks_batch_max_);

  // both max because of the one-sided-ness
  T w_max = fabsf(static_cast<PulsedRPUDeviceMetaParameter<T> &>(rpucuda_device_p->getPar()).w_max);
  T w_min = fabsf(static_cast<PulsedRPUDeviceMetaParameter<T> &>(rpucuda_device_m->getPar()).w_max);
  T upper_thres = par.refresh_upper_thres;
  T lower_thres = par.refresh_lower_thres;

  // 1. we need forward "readout" of full weight matrix, once for pos, once for neg
  refresh_fb_pass_->forwardMatrixIterator(
      weights_p, dev_refresh_vecs_->getDataConst(), this->x_size_, false,
      dev_refresh_tmp_p_->getData(), this->d_size_, false, this->x_size_, (T)1.0, *refresh_iom_,
      par.refresh_io, false);

  refresh_fb_pass_->forwardMatrixIterator(
      weights_m, dev_refresh_vecs_->getDataConst(), this->x_size_, false,
      dev_refresh_tmp_m_->getData(), this->d_size_, false, this->x_size_, (T)1.0, *refresh_iom_,
      par.refresh_io, false);

  // 2. look for resets and set the write back values
  dev_refresh_counters_->setConst(0);

  kernelRefreshWeights<T><<<nblocks, nthreads, 0, this->context_->getStream()>>>(
      dev_refresh_tmp_p_->getData(), dev_refresh_tmp_m_->getData(), dev_reset_msk_->getData(),
      this->size_, dev_refresh_counters_->getData(), upper_thres * w_max, lower_thres * w_min,
      lower_thres * w_max, upper_thres * w_min);

  // 3. do the reset
  // this will add a sync. But since expected to be very sparse that might be worth it.
  int counters[2]; // P, M
  dev_refresh_counters_->copyTo(&counters[0]);
  int p_counter = counters[0];
  int m_counter = counters[1];

  if (p_counter + m_counter) {
    rpucuda_device_p->resetAt(weights_p, dev_reset_msk_->getDataConst());
    rpucuda_device_m->resetAt(weights_m, dev_reset_msk_->getDataConst());
  }

  // 4. update according to difference
  if (p_counter) {
    refresh_pwu_->update(
        dev_refresh_vecs_->getDataConst(), dev_refresh_tmp_p_->getDataConst(), weights_p,
        rpucuda_device_p, par.refresh_up, 1.0, this->x_size_, false, false);
  }

  if (m_counter) {
    refresh_pwu_->update(
        dev_refresh_vecs_->getDataConst(), dev_refresh_tmp_m_->getDataConst(), weights_m,
        rpucuda_device_m, par.refresh_up, 1.0, this->x_size_, false, false);
  }

  return p_counter + m_counter;
}

template <typename T> bool OneSidedRPUDeviceCuda<T>::isInverted() const { return g_plus_ == 0; }

template <typename T> void OneSidedRPUDeviceCuda<T>::invert() {
  std::swap(g_plus_, g_minus_);
  std::swap(this->dev_reduce_weightening_, this->dev_reduce_weightening_inverted_);
}

template <typename T>
pwukpvec_t<T> OneSidedRPUDeviceCuda<T>::getUpdateKernels(
    int m_batch, int nK32, int use_bo64, bool out_trans, const PulsedUpdateMetaParameter<T> &up) {
  if (this->rpucuda_device_vec_.size() != 2) {
    RPU_FATAL("Expect exactly two devices.");
  }
  if (getPar().singleDeviceUpdate()) {
    RPU_FATAL("Single device update not supported for OneSided Device");
  }

  return VectorRPUDeviceCuda<T>::getUpdateKernels(m_batch, nK32, use_bo64, out_trans, up);
}

template <typename T>
void OneSidedRPUDeviceCuda<T>::runUpdateKernel(
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
  // calling kpars->run(..,this,..) directly should cause error
  // because difference derived from abstract device..
  DEBUG_OUT("start run update kernel.");
  DEBUG_CALL(kpars->print(););
  if (one_sided != 0) {
    RPU_FATAL("Cannot use one_sided here.");
  }
  if (cwo) {
    RPU_FATAL("Cannot use CWO here.");
  }

  const auto &par = getPar();
  bool same_context = par.same_context;

  if (!same_context) {
    up_context->recordEvent();
  }

  CudaContextPtr cp = up_context;
  CudaContextPtr cm = up_context;

  if (!same_context) {
    cp = &*this->context_vec_[g_plus_];
    cm = &*this->context_vec_[g_minus_];
  }

  int n = kpars->getNStates() / 2; // each device uses different random states

  if (!same_context) {
    cp->waitEvent(up_context->getEvent());
    cm->waitEvent(up_context->getEvent());
  }

  kpars->run(
      cp->getStream(), this->dev_weights_ptrs_[g_plus_], m_batch, blm,
      static_cast<PulsedRPUDeviceCuda<T> *>(&*this->rpucuda_device_vec_[g_plus_]), // checked above
      up, dev_states,
      1, // one sided!!
      x_counts_chunk, d_counts_chunk);

  kpars->run(
      cm->getStream(), this->dev_weights_ptrs_[g_minus_], m_batch, blm,
      static_cast<PulsedRPUDeviceCuda<T> *>(&*this->rpucuda_device_vec_[g_minus_]), // checked above
      up, dev_states,
      -1, // one sided!!
      x_counts_chunk, d_counts_chunk);

  if (!same_context) {
    up_context->recordWaitEvent(cp);
    up_context->recordWaitEvent(cm);
  }

  if (par.refresh_every > 0 && !up._currently_tuning) {
    int refresh_every = par.refresh_every * m_batch; // always in m_batch
    int n1 = this->current_update_idx_ / refresh_every;
    int n2 = (this->current_update_idx_ + m_batch) / refresh_every;
    if (n2 > n1) {
      refresh_counter_ += refreshWeights();
    }
  }

  this->current_update_idx_ += m_batch;
  this->reduceToWeights(up_context, dev_weights);
}

template class OneSidedRPUDeviceCuda<float>;
#ifdef RPU_USE_DOUBLE
template class OneSidedRPUDeviceCuda<double>;
#endif
#ifdef RPU_USE_FP16
template class OneSidedRPUDeviceCuda<half_t>;
#endif

} // namespace RPU
