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

#include "cuda_math_util.h"
#include "io_iterator.h"
#include "pulsed_weight_updater.h"
#include "pwu_kernel_parameter.h"
#include "rpucuda_pulsed_device.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <random>

namespace RPU {

/****************************************************************************************************************/
/* PULSEDWEIGHTUPDATER */
/******************************************************************************************************************/

template <typename T>
PulsedWeightUpdater<T>::PulsedWeightUpdater(CudaContextPtr c, int x_size, int d_size)
    : context_{c}, x_size_{x_size}, d_size_{d_size}

{
  blm_ = RPU::make_unique<BitLineMaker<T>>(c, x_size, d_size);

  up_context_ = nullptr;
  is_async_update_ = false;
};

template <typename T>
void PulsedWeightUpdater<T>::dumpExtra(RPU::state_t &extra, const std::string prefix) {

  RPU::state_t state;
  context_->synchronize();

  RPU::insert(state, "is_async_update", is_async_update_);
  RPU::insert(state, "update_count", update_count_);
  RPU::insert(state, "verbose", verbose_);

  blm_->dumpExtra(state, "blm");

  RPU::insertWithPrefix(extra, state, prefix);
}

template <typename T>
void PulsedWeightUpdater<T>::loadExtra(
    const RPU::state_t &extra, const std::string prefix, bool strict) {

  context_->synchronize();
  auto state = RPU::selectWithPrefix(extra, prefix);

  RPU::load(state, "is_async_update", is_async_update_, strict);
  RPU::load(state, "update_count", update_count_, strict);
  RPU::load(state, "verbose", verbose_, strict);

  blm_->loadExtra(state, "blm", strict);
}

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

  T pc_lr = rpucuda_device->getPulseCountLearningRate(lr, m_batch, up);
  blm_->makeCounts(
      x_in, d_in, up, rpucuda_device->getWeightGranularity(), pc_lr, m_batch, x_trans_in,
      d_trans_in, kpars->getOutTrans(), kpars->getUseBo64(), kpars->getImplicitPulses());

  CudaContextPtr c = context_;
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

  T min_timing = std::numeric_limits<T>::max();
  int min_i = 0;

  for (int k = 0; k < v.size(); k++) {

    CUDA_TIMING_START(context_);

    for (int i = 0; i < nrepeats; i++) {
      this->executeUpdate(
          v[k], x_in, d_in, dev_tmp_weights.getData(), tmp_device, up_tuning, lr, m_batch,
          x_trans_in, d_trans_in);
    }
    if (verbose_ > 1) {
      CUDA_TIMING_STOP(context_, v[k]->getName());
    } else {
      CUDA_TIMING_STOP_NO_OUTPUT(context_);
    }
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

  if (verbose_ > 0) {
    std::cout << "UpdateTuner: Using " << opt_kernel_pars->getName() << " for PWU ["
              << opt_kernel_pars->timing << "]." << std::endl;
  }
  DEBUG_OUT(
      "UpdateTuner: Using " << opt_kernel_pars->getName() << " for PWU [" << opt_kernel_pars->timing
                            << "].");
  DEBUG_CALL(opt_kernel_pars->print());
}

template <typename T>
template <typename InputIteratorT>
const T *PulsedWeightUpdater<T>::copyIterator2Buffer(InputIteratorT vec, T *buffer, int size) {
  RPU::math::copyWithIterator(context_, buffer, vec, size);

  return buffer;
}

template <>
template <>
const float *
PulsedWeightUpdater<float>::copyIterator2Buffer(const float *vec, float *buffer, int size) {
  return vec;
}

#ifdef RPU_USE_DOUBLE
template <>
template <>
const double *
PulsedWeightUpdater<double>::copyIterator2Buffer(const double *vec, double *buffer, int size) {
  return vec;
}
#endif

#ifdef RPU_USE_FP16
template <>
template <>
const half_t *
PulsedWeightUpdater<half_t>::copyIterator2Buffer(const half_t *vec, half_t *buffer, int size) {
  return vec;
}
#endif

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

  T *fpx_buffer = context_->template getSharedBuffer<T>(RPU_BUFFER_IN, x_size_ * m_batch);
  T *fpd_buffer = context_->template getSharedBuffer<T>(RPU_BUFFER_OUT, d_size_ * m_batch);

  const T *x_out = copyIterator2Buffer(x_in, fpx_buffer, x_size_ * m_batch);
  const T *d_out = copyIterator2Buffer(d_in, fpd_buffer, d_size_ * m_batch);

  if (m_batch == 1 && beta == (T)1.0) {
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
  context_->template releaseSharedBuffer<T>(RPU_BUFFER_IN);
  context_->template releaseSharedBuffer<T>(RPU_BUFFER_OUT);
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

  T *fpx_buffer = context_->template getSharedBuffer<T>(RPU_BUFFER_IN, x_size_ * m_batch);
  T *fpd_buffer = context_->template getSharedBuffer<T>(RPU_BUFFER_OUT, d_size_ * m_batch);

  const T *x_out = copyIterator2Buffer(x_in, fpx_buffer, x_size_ * m_batch);
  const T *d_out = copyIterator2Buffer(d_in, fpd_buffer, d_size_ * m_batch);

  if (!rpucuda_device->hasDirectUpdate()) {
    RPU_FATAL("Device does not support a direct update");
  }

  rpucuda_device->doDirectUpdate(
      x_out, d_out, dev_weights, lr, m_batch, x_trans, d_trans, beta, up,
      fpx_buffer, // this could be in-place with x_out
      fpd_buffer);

  context_->template releaseSharedBuffer<T>(RPU_BUFFER_IN);
  context_->template releaseSharedBuffer<T>(RPU_BUFFER_OUT);
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
  if (update_type != update_type_) {
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
        DEBUG_OUT("DEBUG WARNING: kernel index out of range " << valid_kernels_.size());
        kidx = 0;
      }
      kernel_pars_ = valid_kernels_[kidx];

      if (kernel_pars_->getUseBo64() == 1) {
        DEBUG_OUT("DEBUG WARNING: cannot test BO64 direct. Set to translate ");
        kernel_pars_->forceBo64Translate();
      }
      if (kidx == 0) {
        kernel_pars_->force32();       // debug hack: might break kernel in the worst case
        kernel_pars_->forceNonTrans(); // debug hack: might break kernel in the worst case
        DEBUG_OUT("DEBUG WARNING: Kernel index 0: FORCED 32 and non-trans");
      }
      DEBUG_OUT("Selected kernel index " << kidx << "  out of " << valid_kernels_.size());
      DEBUG_CALL(kernel_pars_->print(););
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
RPU_PWU_ITER_TEMPLATE(float, EyeInputIterator<float>, const float *);
RPU_PWU_ITER_TEMPLATE(float, const float *, EyeInputIterator<float>);

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
RPU_PWU_ITER_TEMPLATE(double, EyeInputIterator<double>, const double *);
RPU_PWU_ITER_TEMPLATE(double, const double *, EyeInputIterator<double>);

#undef TRANSDOUBLE
#endif

#ifdef RPU_USE_FP16
#define TRANSHALF(TRANS) TRANS, half_t

template class PulsedWeightUpdater<half_t>;

RPU_PWU_ITER_TEMPLATE(half_t, IndexReaderTransInputIterator<half_t>, const half_t *);
RPU_PWU_ITER_TEMPLATE(half_t, IndexReaderInputIterator<half_t>, const half_t *);
RPU_PWU_ITER_TEMPLATE(half_t, const half_t *, const half_t *);
RPU_PWU_ITER_TEMPLATE(
    half_t, IndexReaderTransInputIterator<half_t>, PermuterTransInputIterator<half_t>);
RPU_PWU_ITER_TEMPLATE(half_t, const half_t *, PermuterTransInputIterator<half_t>);

RPU_PWU_ITER_TEMPLATE(
    half_t, IndexReaderSliceInputIterator<TRANSHALF(true)>, SliceInputIterator<TRANSHALF(true)>);
RPU_PWU_ITER_TEMPLATE(
    half_t, IndexReaderSliceInputIterator<TRANSHALF(false)>, SliceInputIterator<TRANSHALF(false)>);

RPU_PWU_ITER_TEMPLATE(half_t, const half_t *, SliceInputIterator<TRANSHALF(true)>);
RPU_PWU_ITER_TEMPLATE(half_t, const half_t *, SliceInputIterator<TRANSHALF(false)>);
RPU_PWU_ITER_TEMPLATE(half_t, IndexReaderSliceInputIterator<TRANSHALF(true)>, const half_t *);
RPU_PWU_ITER_TEMPLATE(half_t, IndexReaderSliceInputIterator<TRANSHALF(false)>, const half_t *);
RPU_PWU_ITER_TEMPLATE(half_t, EyeInputIterator<half_t>, const half_t *);
RPU_PWU_ITER_TEMPLATE(half_t, const half_t *, EyeInputIterator<half_t>);

#undef TRANSHALF
#endif

#undef RPU_PWU_ITER_TEMPLATE

} // namespace RPU
