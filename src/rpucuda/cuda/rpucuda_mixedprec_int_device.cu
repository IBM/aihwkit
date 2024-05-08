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
#include "rpucuda_mixedprec_int_device.h"
#include <memory>

namespace RPU {

/******************************************************************************************/
/* MixedPrecIntRPUDeviceCuda

   CUDA implementation of MixedPrecIntRPUDevice

*/

template <typename T>
MixedPrecIntRPUDeviceCuda<T>::MixedPrecIntRPUDeviceCuda(CudaContextPtr c, int x_size, int d_size)
    : MixedPrecRPUDeviceBaseCuda<T>(c, x_size, d_size){};

template <typename T>
MixedPrecIntRPUDeviceCuda<T>::MixedPrecIntRPUDeviceCuda(
    CudaContextPtr c, const MixedPrecIntRPUDevice<T> &rpu_device)
    : MixedPrecIntRPUDeviceCuda<T>(c, rpu_device.getXSize(), rpu_device.getDSize()) {
  populateFrom(rpu_device);
};

template <typename T> void MixedPrecIntRPUDeviceCuda<T>::allocateContainers() {

  dev_chi_ = RPU::make_unique<CudaArray<T>>(this->context_, this->size_);
  this->context_->synchronize();
}

// copy
template <typename T>
MixedPrecIntRPUDeviceCuda<T>::MixedPrecIntRPUDeviceCuda(const MixedPrecIntRPUDeviceCuda<T> &other)
    : MixedPrecRPUDeviceBaseCuda<T>(other) {

  allocateContainers();
  dev_chi_->assign(*other.dev_chi_);

  this->context_->synchronize();
};

template <typename T>
MixedPrecIntRPUDeviceCuda<T> &
MixedPrecIntRPUDeviceCuda<T>::operator=(const MixedPrecIntRPUDeviceCuda<T> &other) {
  MixedPrecIntRPUDeviceCuda<T> tmp(other);
  swap(*this, tmp);
  this->context_->synchronize();
  return *this;
};

template <typename T>
MixedPrecIntRPUDeviceCuda<T>::MixedPrecIntRPUDeviceCuda(MixedPrecIntRPUDeviceCuda<T> &&other) {
  *this = std::move(other);
};

template <typename T>
MixedPrecIntRPUDeviceCuda<T> &
MixedPrecIntRPUDeviceCuda<T>::operator=(MixedPrecIntRPUDeviceCuda<T> &&other) {

  MixedPrecRPUDeviceBaseCuda<T>::operator=(std::move(other));

  dev_chi_ = std::move(other.dev_chi_);
  return *this;
}

template <typename T>
void MixedPrecIntRPUDeviceCuda<T>::populateFrom(const AbstractRPUDevice<T> &rpu_device_in) {

  const auto &rpu_device = dynamic_cast<const MixedPrecIntRPUDevice<T> &>(rpu_device_in);
  if (&rpu_device == nullptr) {
    RPU_FATAL("populateFrom expects MixedPrecIntRPUDevice.");
  }

  MixedPrecRPUDeviceBaseCuda<T>::populateFrom(rpu_device_in); // will set sizes
  allocateContainers();

  const auto &par = this->getPar();

  this->io_.nm_decay = (T)1.0 - par.momentum_nm;
  this->io_.noise_management = NoiseManagementType::AverageAbsMaxSingleValue;

  // chi technically is integer.  bin half needs to be floored

  std::vector<T> v;
  v.resize(this->size_);
  rpu_device.getChi(v.data());
  dev_chi_->assign(v.data());

  this->context_->synchronize();
}

template <typename T>
__global__ void kernelQuantizeInt(
    T *quantized_values, const T *values, const int size, const int n_bins, const T *nm_value) {
  // Note: zero will always be included. A bin will essentially be added if n_bin is an even number

  volatile unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;

  RPU_CUDA_1D_KERNEL_LOOP(idx, size) {
    T nm = *nm_value;
    T x = values[idx];
    T half_bins = (T)n_bins / (T)2.0;
    int max =
        n_bins / 2; // integer div to make sure that rounding errors do not occur at saturation
    T width = nm / half_bins;
    T xq = RPU_ROUNDFUN(x / width);
    quantized_values[idx] = MAX(MIN(xq, (T)max), (T)-max);
  }
}

template <typename T>
__global__ void kernelQuantizeIntStocRound(
    T *quantized_values,
    const T *values,
    const int size,
    const int n_bins,
    const T *nm_value,
    curandState *random_states) {
  // Note: zero will always be included. A bin will essentially be added if n_bin is an even number

  unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
  curandState local_state = random_states[tid];

  RPU_CUDA_1D_KERNEL_LOOP(idx, size) {
    T stoch_value = curand_uniform(&local_state);
    T nm = *nm_value;
    T x = values[idx];
    T half_bins = (T)n_bins / (T)2.0;
    int max =
        n_bins / 2; // integer div to make sure that rounding errors do not occur at saturation
    T width = nm / half_bins;
    T xq = RPU_ROUNDFUN(x / width + stoch_value - (T)0.5);
    quantized_values[idx] = MAX(MIN(xq, (T)max), -(T)max);
  }
  random_states[tid] = local_state;
}

template <typename T>
void MixedPrecIntRPUDeviceCuda<T>::quantize(
    T *quant_values,
    const T *values,
    RPU::NoiseManager<T> *nm,
    int n_bins,
    int size,
    int m_batch,
    bool trans,
    bool stochastic_rounding,
    RPU::NoiseManagementType nm_type) {

  nm->compute(values, nm_type, this->io_, m_batch, trans, false);
  int nthreads = this->context_->getNThreads();
  int nblocks = this->context_->getNBlocks(m_batch * size, nthreads);
  nblocks = MIN(this->nblocks_batch_max_, nblocks);
  cudaStream_t s = this->context_->getStream();

  if (stochastic_rounding) {
    kernelQuantizeIntStocRound<<<nblocks, nthreads, 0, s>>>(
        quant_values, values, m_batch * size, n_bins, nm->getScaleValues(),
        this->context_->getRandomStates(nthreads * nblocks));
  } else {
    kernelQuantizeInt<<<nblocks, nthreads, 0, s>>>(
        quant_values, values, m_batch * size, n_bins, nm->getScaleValues());
  }
}

template <typename T>
void MixedPrecIntRPUDeviceCuda<T>::doDirectUpdate(
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

  if (beta != (T)1.0) {
    RPU_FATAL("beta not equal 1 is not supported.")
  }
  this->setUpPar(up);
  const auto &par = getPar();

  quantize(
      d_buffer, d_input, &*this->noise_manager_d_, par.n_d_bins, this->d_size_, m_batch, d_trans,
      par.stoc_round_d, NoiseManagementType::AbsMaxSingleValue);

  // % Quantize x
  quantize(
      x_buffer, x_input, &*this->noise_manager_x_, par.n_x_bins, this->x_size_, m_batch, x_trans,
      par.stoc_round_x, NoiseManagementType::AverageAbsMaxSingleValue);

  if (m_batch == 1) {
    RPU::math::ger<T>(
        this->context_, this->x_size_, this->d_size_, 1.0, x_buffer, 1, d_buffer, 1,
        dev_chi_->getData(), this->x_size_);
  } else {
    RPU::math::gemm<T>(
        this->context_, x_trans, !d_trans, this->x_size_, this->d_size_, m_batch, 1.0, x_buffer,
        x_trans ? m_batch : this->x_size_, d_buffer, d_trans ? m_batch : this->d_size_,
        1.0, // set beta to 1.0. We want to add to Chi
        dev_chi_->getData(), this->x_size_);
  }

  this->doTransfer(dev_weights, lr, m_batch);
  this->computeSparsity(x_buffer, d_buffer, m_batch);
  this->advanceUpdateCounter(m_batch);
}

template <typename T>
__global__ void kernelMixedPrecIntTransfer(
    T *transfer_out,
    T *chi,
    const int size,
    const T *x_nm,
    const T *d_nm,
    const T w_width_div_lr_chi_scale,
    const T momentum) {

  unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;

  if (tid < size) {
    T d_width = *d_nm; //  already divided by d_bins/2 below
    T x_width = *x_nm; //  already divided by x_bins/2 below
    T value = round(chi[tid]);
    T thres = MAX(round(w_width_div_lr_chi_scale / (d_width * x_width)), (T)1.0);
    T dw = trunc(value / thres); // multiplies of thres OK

    transfer_out[tid] = dw;

    chi[tid] = value - round(((T)1.0 - momentum) * thres * dw);
  }
}

template <typename T>
void MixedPrecIntRPUDeviceCuda<T>::forwardUpdate(
    T *dev_weights,
    const T lr,
    int i_row_start,
    const T *transfer_vec,
    const int n_vec,
    const bool trans) {

  if (!lr) {
    return;
  }
  int t_size = n_vec * this->x_size_;
  if ((this->dev_transfer_tmp_ == nullptr) || this->dev_transfer_tmp_->getSize() < t_size) {
    this->dev_transfer_tmp_ = RPU::make_unique<CudaArray<T>>(this->context_, t_size);
  }

  const auto &par = this->getPar();
  int chi_scale = (par.n_d_bins / 2) * (par.n_x_bins / 2);

  int nthreads = this->context_->getNThreads();
  int nblocks = this->context_->getNBlocks(t_size, nthreads);
  kernelMixedPrecIntTransfer<T><<<nblocks, nthreads, 0, this->context_->getStream()>>>(
      this->dev_transfer_tmp_->getData(), dev_chi_->getData() + i_row_start * this->x_size_, t_size,
      this->noise_manager_x_->getScaleValues(), this->noise_manager_d_->getScaleValues(),
      this->granularity_ / (T)(fabsf(lr) * chi_scale), par.momentum_chi);

  // requires to turn on update_managment / bl managment as well
  this->transfer_pwu_->update(
      this->dev_transfer_tmp_->getDataConst(), // this is the transfer vector (x_size)
      transfer_vec,                            // this should be d_size, non-trans
      dev_weights, &*this->rpucuda_device_, this->up_, this->granularity_, n_vec, trans, false);
}

template <typename T> std::vector<T> MixedPrecIntRPUDeviceCuda<T>::getHiddenWeights() const {

  auto data = MixedPrecRPUDeviceBaseCuda<T>::getHiddenWeights();

  int offset = data.size();
  data.resize(offset + this->size_);
  dev_chi_->copyTo(data.data() + offset);

  return data;
}

template class MixedPrecIntRPUDeviceCuda<float>;
#ifdef RPU_USE_DOUBLE
template class MixedPrecIntRPUDeviceCuda<double>;
#endif
#ifdef RPU_USE_FP16
template class MixedPrecIntRPUDeviceCuda<half_t>;
#endif

} // namespace RPU
