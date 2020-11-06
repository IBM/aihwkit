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

#include "pwu_kernel_parameter.h"
#include "rpu_pulsed_meta_parameter.h"
#include "rpucuda_pulsed_device.h"
#include <memory>

namespace RPU {
/******************************************************************************************/
/* PulsedRPUDeviceCuda

   Base class which maintains the basic hard bounds and dw_min
   up/down and decays etc for the pulsed updates.

   Note that it is still Abstract. Need to implement the getUpdateKernels in derived.
*/

template <typename T>
PulsedRPUDeviceCuda<T>::PulsedRPUDeviceCuda(CudaContext *c, int x_size, int d_size)
    : PulsedRPUDeviceCudaBase<T>(c, x_size, d_size){};

template <typename T> void PulsedRPUDeviceCuda<T>::initialize() {

  dev_4params_ = RPU::make_unique<CudaArray<float>>(this->context_, 4 * this->size_);
  dev_diffusion_rate_ = RPU::make_unique<CudaArray<T>>(this->context_, this->size_);
  dev_reset_bias_ = RPU::make_unique<CudaArray<T>>(this->context_, this->size_);
  dev_decay_scale_ = RPU::make_unique<CudaArray<T>>(this->context_, this->size_);

  this->context_->synchronize();
};

template <typename T>
PulsedRPUDeviceCuda<T>::PulsedRPUDeviceCuda(const PulsedRPUDeviceCuda<T> &other)
    : PulsedRPUDeviceCudaBase<T>(other) {
  initialize();

  dev_4params_->assign(*other.dev_4params_);
  dev_diffusion_rate_->assign(*other.dev_diffusion_rate_);
  dev_reset_bias_->assign(*other.dev_reset_bias_);
  dev_decay_scale_->assign(*other.dev_decay_scale_);

  this->context_->synchronize();
};

// template <typename T>
// PulsedRPUDeviceCuda<T>& PulsedRPUDeviceCuda<T>::operator=(const PulsedRPUDeviceCuda<T>& other){
//   PulsedRPUDeviceCuda<T> tmp(other);
//   swap(*this,tmp);
//   return *this;
// };

// template <typename T>
// PulsedRPUDeviceCuda<T>::PulsedRPUDeviceCuda(PulsedRPUDeviceCuda<T>&& other) {
//   *this = std::move(other);
// };

// template <typename T>
// PulsedRPUDeviceCuda<T>& PulsedRPUDeviceCuda<T>::operator=(PulsedRPUDeviceCuda<T>&& other){

//   PulsedRPUDeviceCudaBase<T>::operator=(std::move(other));

//   dev_4params_ = std::move(other.dev_4params_);
//   dev_diffusion_rate_ = std::move(other.dev_diffusion_rate_);
//   dev_reset_bias_ = std::move(other.dev_reset_bias_);
//   dev_decay_scale_ = std::move(other.dev_decay_scale_);

//   return *this;
// };

template <typename T>
void PulsedRPUDeviceCuda<T>::populateFrom(const AbstractRPUDevice<T> &rpu_device_in) {

  const auto &rpu_device = dynamic_cast<const PulsedRPUDevice<T> &>(rpu_device_in);
  if (&rpu_device == nullptr) {
    RPU_FATAL("populateFrom expects PulsedRPUDevice.");
  }

  int x_size = rpu_device.getXSize();
  int d_size = rpu_device.getDSize();
  int size = x_size * d_size;

  initialize();

  PulsedRPUDeviceCudaBase<T>::populateFrom(rpu_device_in);

  PulsedDPStruc<T> **sup = rpu_device.getDPStruc();

  // copy RPU to device variables
  float *tmp = new float[4 * size];
  T *tmp_ds = new T[size];
  T *tmp_df = new T[size];
  T *tmp_rb = new T[size];
  T *ds = rpu_device.getDecayScale()[0];
  T *df = rpu_device.getDiffusionRate()[0];
  T *rb = rpu_device.getResetBias()[0];

  for (int i = 0; i < d_size; ++i) {
    for (int j = 0; j < x_size; ++j) {

      int k = j * (d_size * 4) +
              4 * i; // transposed: col major required by cuBLAS .. linear arangmenet for now
      tmp[k] = sup[i][j].min_bound;
      tmp[k + 1] = sup[i][j].scale_down;
      tmp[k + 2] = sup[i][j].max_bound;
      tmp[k + 3] = sup[i][j].scale_up;

      int l_t = j * (d_size) + i;
      int l = i * (x_size) + j;
      tmp_ds[l_t] = ds[l];
      tmp_df[l_t] = df[l];
      tmp_rb[l_t] = rb[l];
    }
  }

  dev_4params_->assign(tmp);
  // other parameters
  dev_decay_scale_->assign(tmp_ds);
  dev_diffusion_rate_->assign(tmp_df);
  dev_reset_bias_->assign(tmp_rb);

  this->context_->synchronize();

  delete[] tmp_ds;
  delete[] tmp_df;
  delete[] tmp_rb;
  delete[] tmp;
}

template <typename T>
void PulsedRPUDeviceCuda<T>::applyWeightUpdate(T *weights, T *dw_and_current_weight_out) {
  RPU::math::elemaddcopysat<T>(
      this->context_, weights, dw_and_current_weight_out, this->size_,
      dev_4params_->getDataConst());
}

template <typename T>
void PulsedRPUDeviceCuda<T>::decayWeights(T *weights, T alpha, bool bias_no_decay) {

  RPU::math::elemscalealpha<T>(
      this->context_, weights, bias_no_decay ? MAX(this->size_ - this->d_size_, 0) : this->size_,
      dev_decay_scale_->getData(), dev_4params_->getData(), alpha);
}

template <typename T> void PulsedRPUDeviceCuda<T>::decayWeights(T *weights, bool bias_no_decay) {

  RPU::math::elemscale<T>(
      this->context_, weights, bias_no_decay ? MAX(this->size_ - this->d_size_, 0) : this->size_,
      dev_decay_scale_->getData(), dev_4params_->getData());
}

template <typename T> void PulsedRPUDeviceCuda<T>::diffuseWeights(T *weights) {

  if (this->dev_diffusion_nrnd_ == nullptr) {
    this->initDiffusionRnd();
    this->rnd_context_->randNormal(
        this->dev_diffusion_nrnd_->getData(), this->dev_diffusion_nrnd_->getSize());
  }
  this->rnd_context_->synchronize();

  RPU::math::elemasb02<T>(
      this->context_, weights, this->size_, this->dev_diffusion_nrnd_->getData(),
      dev_diffusion_rate_->getData(), dev_4params_->getData());

  this->rnd_context_->recordWaitEvent(this->context_->getStream());
  this->rnd_context_->randNormal(
      this->dev_diffusion_nrnd_->getData(), this->dev_diffusion_nrnd_->getSize());
}

template <typename T> void PulsedRPUDeviceCuda<T>::clipWeights(T *weights, T clip) {

  RPU::math::elemsat<T>(this->context_, weights, this->size_, dev_4params_->getData());
  if (clip >= 0) {
    RPU::math::aclip<T>(this->context_, weights, this->size_, clip);
  }
}

template <typename T> void PulsedRPUDeviceCuda<T>::initResetRnd() {

  if (this->rnd_context_ == nullptr) {
    this->initRndContext();
  }
  dev_reset_nrnd_ = std::unique_ptr<CudaArray<float>>(
      new CudaArray<float>(&*this->rnd_context_, (this->size_ + 31) / 32 * 32));
  dev_reset_flag_ = std::unique_ptr<CudaArray<float>>(
      new CudaArray<float>(&*this->rnd_context_, (this->size_ + 31) / 32 * 32));
  dev_reset_flag_->setConst(0);
  this->rnd_context_->synchronize();
}

template <typename T>
void PulsedRPUDeviceCuda<T>::resetCols(T *weights, int start_col, int n_cols, T reset_prob) {
  // col-major in CUDA.

  if (dev_reset_nrnd_ == nullptr) {
    initResetRnd();
  }
  int n = n_cols * this->d_size_;
  int offset = start_col * this->d_size_;
  this->rnd_context_->randNormal(
      dev_reset_nrnd_->getData(), n_cols * this->d_size_, 0.0, getPar().reset_std);
  if (reset_prob < 1) {
    this->rnd_context_->randUniform(dev_reset_flag_->getData(), n_cols * this->d_size_);
  }
  this->context_->recordWaitEvent(this->rnd_context_->getStream());

  if (n >= this->size_) {
    // reset whole matrix
    RPU::math::elemresetsat<T>(
        this->context_, weights, this->size_, dev_reset_bias_->getDataConst(),
        dev_reset_nrnd_->getDataConst(), dev_reset_flag_->getDataConst(), reset_prob,
        dev_4params_->getData());

  } else if (offset + n <= this->size_) {
    // one pass enough
    RPU::math::elemresetsat<T>(
        this->context_, weights + offset, n, dev_reset_bias_->getDataConst() + offset,
        dev_reset_nrnd_->getDataConst(), dev_reset_flag_->getDataConst(), reset_prob,
        dev_4params_->getData() + 4 * offset);
  } else {
    // two passes
    int m = this->size_ - offset;

    RPU::math::elemresetsat<T>(
        this->context_, weights + offset, m, dev_reset_bias_->getDataConst() + offset,
        dev_reset_nrnd_->getDataConst(), dev_reset_flag_->getDataConst(), reset_prob,
        dev_4params_->getData() + 4 * offset);

    RPU::math::elemresetsat<T>(
        this->context_, weights, n - m, dev_reset_bias_->getDataConst(),
        dev_reset_nrnd_->getDataConst() + m, dev_reset_flag_->getDataConst() + m, reset_prob,
        dev_4params_->getData());
  }
}

template <typename T>
void PulsedRPUDeviceCuda<T>::runUpdateKernel(
    pwukp_t<T> kpars,
    CudaContext *c,
    T *dev_weights,
    int m_batch,
    const BitLineMaker<T> *blm,
    const PulsedUpdateMetaParameter<T> &up,
    curandState_t *dev_states,
    int one_sided,
    uint32_t *x_counts_chunk,
    uint32_t *d_counts_chunk) {

  kpars->run(
      c->getStream(), dev_weights, m_batch, blm, this, up, dev_states, one_sided, x_counts_chunk,
      d_counts_chunk);
}

template class PulsedRPUDeviceCuda<float>;
#ifdef RPU_USE_DOUBLE
template class PulsedRPUDeviceCuda<double>;
#endif
} // namespace RPU
