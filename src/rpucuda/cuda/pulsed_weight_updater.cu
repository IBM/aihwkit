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

#include "cuda_math_util.h"
#include "pulsed_weight_updater.h"
#include <cub/cub.cuh>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <random>

#include "io_iterator.h"
#include "pwu_kernel_parameter.h"
#include "rpucuda_pulsed_device.h"

namespace RPU {

/****************************************************************************************************************/
/* PULSEDWEIGHTUPDATER */
/******************************************************************************************************************/

template <typename T>
PulsedWeightUpdater<T>::PulsedWeightUpdater(CudaContext *c, int x_size, int d_size)
    : context_{c}, x_size_{x_size}, d_size_{d_size}

{
  blm_ = RPU::make_unique<BitLineMaker<T>>(c, x_size, d_size);

  up_context_ = nullptr;
  is_async_update_ = false;
};

template <typename T>
pwukpvec_t<T> PulsedWeightUpdater<T>::getValidUpdateKernels(
    PulsedRPUDeviceCudaBase<T> *rpucuda_device,
    int m_batch,
    const PulsedUpdateMetaParameter<T> &up) {
  pwukpvec_t<T> v;
  for (int use_bo64 : {1, 0}) { // omit 2 (ie bo64 translation)
    for (int out_trans : {true, false}) {

      pwukpvec_t<T> v2 =
          rpucuda_device->getUpdateKernels(m_batch, up.getNK32Default(), use_bo64, out_trans, up);
      for (int i = 0; i < v2.size(); i++) {
        if (v2[i]->isValid()) {
          v.push_back(v2[i]);
        }
      }
    }
    if (v.size() > 0 && (m_batch >= 1000)) {
      break; // prefer bo64 for large batch if possible
    }
  }
  return v;
}

template <typename T> void PulsedWeightUpdater<T>::makeUpdateAsync() {
  if (!is_async_update_) {
    is_async_update_ = true;
    up_context_ = RPU::make_unique<CudaContext>(context_->getGPUId());
  }
}

template <typename T> void PulsedWeightUpdater<T>::waitForUpdateCalculations() {
  if (is_async_update_) {
    // use the up_context event for it because context_ might be shared
    context_->recordWaitEvent(up_context_->getStream(), up_context_->getEvent());
  }
}

template <typename T>
template <typename XInputIteratorT, typename DInputIteratorT>
void PulsedWeightUpdater<T>::executeUpdate(
    pwukp_t<T> kpars,
    XInputIteratorT x_in,
    DInputIteratorT d_in,
    T *dev_weights,
    PulsedRPUDeviceCudaBase<T> *rpucuda_device,
    const PulsedUpdateMetaParameter<T> &up,
    const T lr,
    const int m_batch,
    const bool x_trans_in,
    const bool d_trans_in) {

  T pc_lr = rpucuda_device->getPulseCountLearningRate(lr);
  blm_->makeCounts(
      x_in, d_in, up, rpucuda_device->getWeightGranularity(), pc_lr, m_batch, x_trans_in,
      d_trans_in, kpars->getOutTrans(), kpars->getUseBo64(), kpars->getImplicitPulses());

  CudaContext *c = context_;
  if (is_async_update_) {
    up_context_->recordWaitEvent(context_->getStream(), context_->getEvent());
    c = &*up_context_;
  }
  // the original learninig rate needs to be passed
  rpucuda_device->runUpdateKernel(
      kpars, c, dev_weights, m_batch, &*blm_, up, lr, c->getRandomStates(kpars->getNStates()));
}

template <typename T>
template <typename XInputIteratorT, typename DInputIteratorT>
void PulsedWeightUpdater<T>::tuneUpdate(
    pwukp_t<T> &opt_kernel_pars,
    pwukpvec_t<T> &v,
    XInputIteratorT x_in,
    DInputIteratorT d_in,
    T *dev_weights,
    PulsedRPUDeviceCudaBase<T> *rpucuda_device,
    const PulsedUpdateMetaParameter<T> &up,
    const T lr,
    const int m_batch,
    const bool x_trans_in,
    const bool d_trans_in) {
  bool is_async_update = is_async_update_;
  is_async_update_ = false;

  CUDA_TIMING_INIT;
  int nrepeats = 3;

  CudaArray<T> dev_tmp_weights(context_, x_size_ * d_size_);

  auto *tmp_device = rpucuda_device->clone();

  PulsedUpdateMetaParameter<T> up_tuning(up);
  up_tuning._currently_tuning = true;

  dev_tmp_weights.assignFromDevice(dev_weights);
  context_->synchronizeDevice(); // maybe other streams exist.

  T min_timing = FLT_MAX;
  int min_i = 0;

  for (int k = 0; k < v.size(); k++) {

    CUDA_TIMING_START(*context_);

    for (int i = 0; i < nrepeats; i++) {
      this->executeUpdate(
          v[k], x_in, d_in, dev_tmp_weights.getData(), tmp_device, up_tuning, lr, m_batch,
          x_trans_in, d_trans_in);
    }
    CUDA_TIMING_STOP_NO_OUTPUT(*context_);

    v[k]->timing = milliseconds / nrepeats;

    if (v[k]->timing < min_timing) {
      min_timing = v[k]->timing;
      min_i = k;
    }
  }

  CUDA_TIMING_DESTROY;
  is_async_update_ = is_async_update;

  opt_kernel_pars = v[min_i];

  delete tmp_device;

  DEBUG_OUT(
      "UpdateTuner: Using " << opt_kernel_pars->getName() << " for PWU [" << opt_kernel_pars->timing
                            << "].\n");
  DEBUG_CALL(opt_kernel_pars->print());
}

template <typename T>
template <typename InputIteratorT>
const T *PulsedWeightUpdater<T>::copyIterator2Buffer(
    InputIteratorT vec, std::shared_ptr<CudaArray<T>> &buffer, int size) {
  if ((buffer == nullptr) || (buffer->getSize() < size)) {
    buffer = std::shared_ptr<CudaArray<T>>(new CudaArray<T>(context_, size));
  }
  RPU::math::copyWithIterator(context_, buffer->getData(), vec, size);

  return buffer->getDataConst();
}

template <>
template <>
const float *PulsedWeightUpdater<float>::copyIterator2Buffer(
    const float *vec, std::shared_ptr<CudaArray<float>> &buffer, int size) {
  return vec;
}

#ifdef RPU_USE_DOUBLE
template <>
template <>
const double *PulsedWeightUpdater<double>::copyIterator2Buffer(
    const double *vec, std::shared_ptr<CudaArray<double>> &buffer, int size) {
  return vec;
}
#endif

template <typename T>
void PulsedWeightUpdater<T>::setSharedBuffer(
    int m_batch, std::shared_ptr<CudaArray<T>> x_buffer, std::shared_ptr<CudaArray<T>> d_buffer) {
  if (x_buffer) {
    dev_fpx_buffer_ = x_buffer;
    if (dev_fpx_buffer_->getSize() < m_batch * x_size_) {
      RPU_FATAL("X batch buffer size too small.");
    }
  }

  if (d_buffer) {
    dev_fpd_buffer_ = d_buffer;
    if (dev_fpd_buffer_->getSize() < m_batch * d_size_) {
      RPU_FATAL("D batch buffer size too small.");
    }
  }
}

template <typename T>
template <typename XInputIteratorT, typename DInputIteratorT>
void PulsedWeightUpdater<T>::doFPupdate(
    XInputIteratorT x_in,
    DInputIteratorT d_in,
    T *dev_weights,
    const T lr,
    const int m_batch,
    const bool x_trans,
    const bool d_trans,
    const T beta) {

  const T *x_out = copyIterator2Buffer(x_in, dev_fpx_buffer_, x_size_ * m_batch);
  const T *d_out = copyIterator2Buffer(d_in, dev_fpd_buffer_, d_size_ * m_batch);

  if (m_batch == 1 && beta == 1.0) {
    RPU::math::ger<T>(context_, d_size_, x_size_, -lr, d_out, 1, x_out, 1, dev_weights, d_size_);
  } else {

    RPU::math::gemm<T>(
        context_, d_trans, !x_trans,
        d_size_, // M
        x_size_, // N
        m_batch, // K
        -lr, d_out, d_trans ? m_batch : d_size_, x_out, x_trans ? m_batch : x_size_, beta,
        dev_weights, d_size_);
  }
}

template <typename T> void PulsedWeightUpdater<T>::checkBuffers(int m_batch) {

  // make sure shared buffers are constructed
  if ((dev_fpx_buffer_ == nullptr) || (dev_fpx_buffer_->getSize() < x_size_ * m_batch)) {
    dev_fpx_buffer_ = std::make_shared<CudaArray<T>>(context_, x_size_ * m_batch);
  }
  if ((dev_fpd_buffer_ == nullptr) || (dev_fpd_buffer_->getSize() < d_size_ * m_batch)) {
    dev_fpd_buffer_ = std::make_shared<CudaArray<T>>(context_, d_size_ * m_batch);
  }
}

template <typename T>
template <typename XInputIteratorT, typename DInputIteratorT>
void PulsedWeightUpdater<T>::doDirectUpdate(
    XInputIteratorT x_in,
    DInputIteratorT d_in,
    AbstractRPUDeviceCuda<T> *rpucuda_device,
    T *dev_weights,
    const T lr,
    const PulsedUpdateMetaParameter<T> &up,
    const int m_batch,
    const bool x_trans,
    const bool d_trans,
    const T beta) {

  checkBuffers(m_batch); // make sure they are created (we need them also for float * iterator)

  const T *x_out = copyIterator2Buffer(x_in, dev_fpx_buffer_, x_size_ * m_batch);
  const T *d_out = copyIterator2Buffer(d_in, dev_fpd_buffer_, d_size_ * m_batch);

  if (!rpucuda_device->hasDirectUpdate()) {
    RPU_FATAL("Device does not support a direct update");
  }

  rpucuda_device->doDirectUpdate(
      x_out, d_out, dev_weights, lr, m_batch, x_trans, d_trans, beta, up,
      dev_fpx_buffer_->getData(), // this might overrite x_out
      dev_fpd_buffer_->getData());
}

template <typename T>
bool PulsedWeightUpdater<T>::checkForFPUpdate(
    AbstractRPUDeviceCuda<T> *rpucuda_device_in, const PulsedUpdateMetaParameter<T> &up) {

  if (rpucuda_device_in == nullptr) {
    return true;
  }
  if (rpucuda_device_in->implements() == DeviceUpdateType::FloatingPoint) {
    return true;
  }
  if (rpucuda_device_in->isPulsedDevice() && up.pulse_type == PulseType::None) {
    return true;
  }
  if (rpucuda_device_in->hasDirectUpdate()) {
    // also FP has direct, but that is handled above
    return false;
  }
  // omitting !isPulsedDevice

  return false;
}

#define FORCE_TUNING_THRES 0

template <typename T>
template <typename XInputIteratorT, typename DInputIteratorT>
void PulsedWeightUpdater<T>::update(
    XInputIteratorT x_in,
    DInputIteratorT d_in,
    T *dev_weights,
    AbstractRPUDeviceCuda<T> *rpucuda_device_in,
    const PulsedUpdateMetaParameter<T> &up,
    const T lr,
    const int m_batch,
    const bool x_trans,
    const bool d_trans) {
  // FP update if no device is given
  if (rpucuda_device_in != nullptr && rpucuda_device_in->hasDirectUpdate()) {
    doDirectUpdate(x_in, d_in, rpucuda_device_in, dev_weights, lr, up, m_batch, x_trans, d_trans);
    return;
  } else if (
      checkForFPUpdate(rpucuda_device_in, up) || (up.pulse_type == PulseType::NoneWithDevice)) {

    doFPupdate(x_in, d_in, dev_weights, lr, m_batch, x_trans, d_trans);

    if (up.pulse_type == PulseType::NoneWithDevice) {
      // apply bounds
      rpucuda_device_in->clipWeights(dev_weights, -1.0);
    }
    return;
  }

  // safe because of isPulsedDevice
  PulsedRPUDeviceCudaBase<T> *rpucuda_device =
      static_cast<PulsedRPUDeviceCudaBase<T> *>(rpucuda_device_in);
  bool force_tuning = false;

  // check need for init (or re-init)
  DeviceUpdateType update_type = rpucuda_device->implements();
  if (update_type != update_type_) //|| (!blm_->checkBuffer(m_batch,BL)))
  {
    // we do not check for change in x_size/d_size, but they are assumed to be constant as well!

    force_tuning = true;
    update_type_ = update_type;

    update_count_ = 0;

    // init kernels
    valid_kernels_ = getValidUpdateKernels(rpucuda_device, m_batch, up);
    if (valid_kernels_.size() == 0) {
      RPU_FATAL("Cannot find valid update kernels");
    }
    kernel_pars_ = valid_kernels_[0]; // this will be modified if tuned

    if (up._debug_kernel_index >= 0) {
      // set default for debugging
      // just get a valid kpars (will be overwritten if tuning is used below)
      force_tuning = false;
      int kidx = up._debug_kernel_index;
      if (up._debug_kernel_index >= valid_kernels_.size()) {
        std::cout << "DEBUG WARNING: kernel index out of range " << valid_kernels_.size()
                  << std::endl;
        kidx = 0;
      }

      kernel_pars_ = valid_kernels_[kidx];

      if (kernel_pars_->getUseBo64() == 1) {
        std::cout << "DEBUG WARNING: cannot test BO64 direct. Set to translate " << std::endl;
        kernel_pars_->forceBo64Translate();
      }
      if (kidx == 0) {
        kernel_pars_->force32();       // debug hack: might break kernel in the worst case
        kernel_pars_->forceNonTrans(); // debug hack: might break kernel in the worst case
        std::cout << "DEBUG WARNING: Kernel index 0: FORCED 32 and non-trans" << std::endl;
      }
      std::cout << "Selected kernel index " << kidx << "  out of " << valid_kernels_.size()
                << std::endl;
      kernel_pars_->print();
    }
  }

  if (update_count_ < FORCE_TUNING_THRES) { // only once again
    update_count_ += 1;
    force_tuning = force_tuning || (update_count_ == FORCE_TUNING_THRES);
  }

  // tune if requested
  if (force_tuning) {
    this->tuneUpdate(
        kernel_pars_, valid_kernels_, x_in, d_in, dev_weights, rpucuda_device, up, lr, m_batch,
        x_trans, d_trans);
  }

  // do update
  this->executeUpdate(
      kernel_pars_, x_in, d_in, dev_weights, rpucuda_device, up, lr, m_batch, x_trans, d_trans);
}

#define RPU_PWU_ITER_TEMPLATE(NUM_T, XITERT, DITERT)                                               \
  template void PulsedWeightUpdater<NUM_T>::update(                                                \
      XITERT, DITERT, NUM_T *, AbstractRPUDeviceCuda<NUM_T> *,                                     \
      const PulsedUpdateMetaParameter<NUM_T> &, const NUM_T, const int, const bool, const bool);   \
  template void PulsedWeightUpdater<NUM_T>::doFPupdate(                                            \
      XITERT, DITERT, NUM_T *, const NUM_T, const int, const bool, const bool, const NUM_T);       \
  template void PulsedWeightUpdater<NUM_T>::doDirectUpdate(                                        \
      XITERT, DITERT, AbstractRPUDeviceCuda<NUM_T> *, NUM_T *, const NUM_T,                        \
      const PulsedUpdateMetaParameter<NUM_T> &, const int, const bool, const bool, const NUM_T);   \
  template void PulsedWeightUpdater<NUM_T>::tuneUpdate(                                            \
      pwukp_t<NUM_T> &, pwukpvec_t<NUM_T> &, XITERT, DITERT, NUM_T *,                              \
      PulsedRPUDeviceCudaBase<NUM_T> *, const PulsedUpdateMetaParameter<NUM_T> &, const NUM_T,     \
      const int, const bool, const bool);                                                          \
  template void PulsedWeightUpdater<NUM_T>::executeUpdate(                                         \
      pwukp_t<NUM_T>, XITERT, DITERT, NUM_T *, PulsedRPUDeviceCudaBase<NUM_T> *,                   \
      const PulsedUpdateMetaParameter<NUM_T> &, const NUM_T, const int, const bool, const bool);

#define TRANSFLOAT(TRANS) TRANS, float
template class PulsedWeightUpdater<float>;

RPU_PWU_ITER_TEMPLATE(float, IndexReaderTransInputIterator<float>, const float *);
RPU_PWU_ITER_TEMPLATE(float, IndexReaderInputIterator<float>, const float *);
RPU_PWU_ITER_TEMPLATE(float, const float *, const float *);
RPU_PWU_ITER_TEMPLATE(
    float, IndexReaderTransInputIterator<float>, PermuterTransInputIterator<float>);
RPU_PWU_ITER_TEMPLATE(float, const float *, PermuterTransInputIterator<float>);

RPU_PWU_ITER_TEMPLATE(
    float, IndexReaderSliceInputIterator<TRANSFLOAT(true)>, SliceInputIterator<TRANSFLOAT(true)>);
RPU_PWU_ITER_TEMPLATE(
    float, IndexReaderSliceInputIterator<TRANSFLOAT(false)>, SliceInputIterator<TRANSFLOAT(false)>);

RPU_PWU_ITER_TEMPLATE(float, const float *, SliceInputIterator<TRANSFLOAT(true)>);
RPU_PWU_ITER_TEMPLATE(float, const float *, SliceInputIterator<TRANSFLOAT(false)>);
RPU_PWU_ITER_TEMPLATE(float, IndexReaderSliceInputIterator<TRANSFLOAT(true)>, const float *);
RPU_PWU_ITER_TEMPLATE(float, IndexReaderSliceInputIterator<TRANSFLOAT(false)>, const float *);

#undef TRANSFLOAT

#ifdef RPU_USE_DOUBLE
#define TRANSDOUBLE(TRANS) TRANS, double

template class PulsedWeightUpdater<double>;

RPU_PWU_ITER_TEMPLATE(double, IndexReaderTransInputIterator<double>, const double *);
RPU_PWU_ITER_TEMPLATE(double, IndexReaderInputIterator<double>, const double *);
RPU_PWU_ITER_TEMPLATE(double, const double *, const double *);
RPU_PWU_ITER_TEMPLATE(
    double, IndexReaderTransInputIterator<double>, PermuterTransInputIterator<double>);
RPU_PWU_ITER_TEMPLATE(double, const double *, PermuterTransInputIterator<double>);

RPU_PWU_ITER_TEMPLATE(
    double,
    IndexReaderSliceInputIterator<TRANSDOUBLE(true)>,
    SliceInputIterator<TRANSDOUBLE(true)>);
RPU_PWU_ITER_TEMPLATE(
    double,
    IndexReaderSliceInputIterator<TRANSDOUBLE(false)>,
    SliceInputIterator<TRANSDOUBLE(false)>);

RPU_PWU_ITER_TEMPLATE(double, const double *, SliceInputIterator<TRANSDOUBLE(true)>);
RPU_PWU_ITER_TEMPLATE(double, const double *, SliceInputIterator<TRANSDOUBLE(false)>);
RPU_PWU_ITER_TEMPLATE(double, IndexReaderSliceInputIterator<TRANSDOUBLE(true)>, const double *);
RPU_PWU_ITER_TEMPLATE(double, IndexReaderSliceInputIterator<TRANSDOUBLE(false)>, const double *);

#undef TRANSDOUBLE
#endif

#undef RPU_PWU_ITER_TEMPLATE

} // namespace RPU
