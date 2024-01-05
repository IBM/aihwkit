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

#include "rpu_weight_updater.h"
#include "rpu_vector_device.h"
#include "utility_functions.h"

namespace RPU {

/*FP update */
template <typename T>
void RPUWeightUpdater<T>::updateVector(
    T **weights,
    const T *x_input,
    const int x_inc,
    const T *d_input,
    const int d_inc,
    const T learning_rate) {

  RPU::math::ger<T>(
      CblasRowMajor, this->d_size_, this->x_size_, -learning_rate, d_input, d_inc, x_input, x_inc,
      weights[0], this->x_size_);
}

template class RPUWeightUpdater<float>;
#ifdef RPU_USE_DOUBLE
template class RPUWeightUpdater<double>;
#endif
#ifdef RPU_USE_FP16
template class RPUWeightUpdater<half_t>;
#endif

/*********************************************************************/
/* Pulsed update */

template <typename T> void PulsedRPUWeightUpdater<T>::allocateContainers() {

  if (!containers_allocated_) {
    freeContainers();

    sblm_ = std::unique_ptr<SparseBitLineMaker<T>>(
        new SparseBitLineMaker<T>(this->x_size_, this->d_size_));
    dblm_ = std::unique_ptr<DenseBitLineMaker<T>>(
        new DenseBitLineMaker<T>(this->x_size_, this->d_size_));

    containers_allocated_ = true;
  }

  x_noz_ = 0;
  d_noz_ = 0;
}

template <typename T> void PulsedRPUWeightUpdater<T>::freeContainers() {

  if (containers_allocated_) {

    sblm_ = nullptr;
    dblm_ = nullptr;

    containers_allocated_ = false;
  }
}

// ctor
template <typename T>
PulsedRPUWeightUpdater<T>::PulsedRPUWeightUpdater(
    int x_size, int d_size, std::shared_ptr<RNG<T>> rng)
    : RPUWeightUpdater<T>(x_size, d_size), rng_(rng) {
  allocateContainers();
}

// dtor
template <typename T> PulsedRPUWeightUpdater<T>::~PulsedRPUWeightUpdater() { freeContainers(); }

// copy construcutor
template <typename T>
PulsedRPUWeightUpdater<T>::PulsedRPUWeightUpdater(const PulsedRPUWeightUpdater<T> &other)
    : RPUWeightUpdater<T>(other) {
  up_ = other.up_;
  rng_ = other.rng_;

  if (other.containers_allocated_) {
    allocateContainers();
    *sblm_ = *other.sblm_;
    *dblm_ = *other.dblm_;
  }
  x_noz_ = other.x_noz_;
  d_noz_ = other.d_noz_;
}

// copy assignment
template <typename T>
PulsedRPUWeightUpdater<T> &
PulsedRPUWeightUpdater<T>::operator=(const PulsedRPUWeightUpdater<T> &other) {

  PulsedRPUWeightUpdater<T> tmp(other);
  swap(*this, tmp);
  return *this;
}

// move constructor
template <typename T>
PulsedRPUWeightUpdater<T>::PulsedRPUWeightUpdater(PulsedRPUWeightUpdater<T> &&other) {

  *this = std::move(other);
}

// move assignment
template <typename T>
PulsedRPUWeightUpdater<T> &PulsedRPUWeightUpdater<T>::operator=(PulsedRPUWeightUpdater<T> &&other) {

  RPUWeightUpdater<T>::operator=(std::move(other));

  up_ = other.up_;

  // pointers
  dblm_ = std::move(other.dblm_);
  sblm_ = std::move(other.sblm_);
  rng_ = std::move(other.rng_);

  containers_allocated_ = other.containers_allocated_;
  x_noz_ = other.x_noz_;
  d_noz_ = other.d_noz_;
  return *this;
}

template <typename T>
void PulsedRPUWeightUpdater<T>::dumpExtra(RPU::state_t &extra, const std::string prefix) {

  RPUWeightUpdater<T>::dumpExtra(extra, prefix);

  RPU::state_t state;

  if (containers_allocated_) {
    dblm_->dumpExtra(state, "dblm");
    sblm_->dumpExtra(state, "sblm");
  }
  RPU::insert(state, "containers_allocated", containers_allocated_);
  RPU::insert(state, "d_noz", d_noz_);
  RPU::insert(state, "x_noz", x_noz_);

  RPU::insertWithPrefix(extra, state, prefix);
}

template <typename T>
void PulsedRPUWeightUpdater<T>::loadExtra(
    const RPU::state_t &extra, const std::string prefix, bool strict) {

  RPUWeightUpdater<T>::loadExtra(extra, prefix, strict);
  auto state = RPU::selectWithPrefix(extra, prefix);

  bool was_allocated;
  RPU::load(state, "containers_allocated", was_allocated, strict);
  if (!containers_allocated_ && was_allocated) {
    allocateContainers();
  }
  RPU::load(state, "d_noz", d_noz_, strict);
  RPU::load(state, "x_noz", x_noz_, strict);

  if (containers_allocated_) {
    dblm_->loadExtra(state, "dblm", strict);
    sblm_->loadExtra(state, "sblm", strict);
  }
}

template <typename T>
void PulsedRPUWeightUpdater<T>::setUpPar(const PulsedUpdateMetaParameter<T> &up) {
  up_ = up;
  // check the parameters
  up_.initialize();
}

template <typename T>
bool PulsedRPUWeightUpdater<T>::checkForFPUpdate(AbstractRPUDevice<T> *rpu_device_in) {

  if (rpu_device_in == nullptr) {
    return true;
  }
  if (rpu_device_in->implements() == DeviceUpdateType::FloatingPoint) {
    return true;
  }
  if (rpu_device_in->isPulsedDevice() && up_.pulse_type == PulseType::None) {
    return true;
  }
  if (rpu_device_in->hasDirectUpdate()) {
    // also FP has direct, but that is handled above
    return false;
  }
  // omitting !isPulsedDevice

  return false;
}

template <typename T>
void PulsedRPUWeightUpdater<T>::updateVectorWithDevice(
    T **weights,
    const T *x_input,
    const int x_inc,
    const T *d_input,
    const int d_inc,
    const T learning_rate,
    const int m_batch_info,
    AbstractRPUDevice<T> *rpu_device_in) {
  if (!learning_rate) {
    return; // do nothing
  }

  // handle cases with no device or FP device
  if (rpu_device_in != nullptr && rpu_device_in->hasDirectUpdate()) {
    rpu_device_in->doDirectVectorUpdate(
        weights, x_input, x_inc, d_input, d_inc, learning_rate, m_batch_info, up_);
    return;
  } else if (up_.pulse_type == PulseType::NoneWithDevice || checkForFPUpdate(rpu_device_in)) {

    RPUWeightUpdater<T>::updateVector(weights, x_input, x_inc, d_input, d_inc, learning_rate);

    if (up_.pulse_type == PulseType::NoneWithDevice) {
      rpu_device_in->clipWeights(weights, (T)-1.0);
    }
    return;
  }

  auto *rpu_device = static_cast<PulsedRPUDeviceBase<T> *>(rpu_device_in);

  // check learning rate and update management
  T weight_granularity = rpu_device->getWeightGranularity();

  // pulsed device update
  if (up_.d_sparsity) {
    up_._d_sparsity = getCurrentDSparsity();
  }
  rpu_device->initUpdateCycle(
      weights, up_, learning_rate, m_batch_info, x_input, x_inc, d_input, d_inc);
  // potentially modify the LR from the device side

  T pc_learning_rate = rpu_device->getPulseCountLearningRate(learning_rate, m_batch_info, up_);
  d_noz_ = 0;
  x_noz_ = 0;

  if (sblm_->supports(up_.pulse_type)) {
    // envoke sparse bit line maker to get the counts and indices
    int BL = sblm_->makeCounts(
        x_input, x_inc, x_noz_, d_input, d_inc, d_noz_, &*rng_,
        pc_learning_rate < (T)0.0 ? -pc_learning_rate : pc_learning_rate, weight_granularity, up_);
    // positive LR actually means that positive signs *decrease* the weight (as in SGD).
    int lr_sign = pc_learning_rate < (T)0.0 ? -1 : 1;

    if (BL > 0) {

      int *x_counts_p;
      int *x_counts_n;
      int *d_counts;
      int **x_indices_p;
      int **x_indices_n;
      int **d_indices;

      bool do_negative_separatly = sblm_->getCountsAndIndices(
          x_counts_p, x_counts_n, d_counts, x_indices_p, x_indices_n, d_indices);

      for (int k = 0; k < BL; k++) {
        if (d_counts[k] > 0) {
          for (int ii = 0; ii < d_counts[k]; ii++) {

            int i_signed = d_indices[k][ii];
            int d_sign = i_signed < 0 ? -lr_sign : lr_sign;
            int i = i_signed < 0 ? -i_signed - 1 : i_signed - 1;

            // let rpu_device decide how to update w
            if (x_counts_p[k] > 0) {
              rpu_device->doSparseUpdate(weights, i, x_indices_p[k], x_counts_p[k], d_sign, &*rng_);
            }
            if (do_negative_separatly) {
              if (x_counts_n[k] > 0) {
                rpu_device->doSparseUpdate(
                    weights, i, x_indices_n[k], x_counts_n[k], d_sign, &*rng_);
              }
            }
          }
        }
      }
    }
  } else {
    // use dense update
    int *coincidences = dblm_->makeCoincidences(
        x_input, x_inc, x_noz_, d_input, d_inc, d_noz_, &*rng_, pc_learning_rate,
        weight_granularity, up_);
    rpu_device->doDenseUpdate(weights, coincidences, &*rng_);
  }
  // always the current SGD learning rate is given here
  rpu_device->finishUpdateCycle(weights, up_, learning_rate, m_batch_info);
}

namespace test_helper {
void getSparseCountsFromCounts(
    int **&sparse_indices, int *&sparse_counts, uint32_t *&counts, int K, int size) {

  for (int k = 0; k < K; k++) { // BL
    sparse_counts[k] = 0;
  }

  for (int i = 0; i < size; i++) { // vector index
    int nK32 = (K + 1 + 31) / 32;
    uint32_t one = 1;
    uint32_t negative = counts[i] & one; // first bit is sign bit
    int s = 0;                           // overall BL
    for (int j = 0; j < nK32; j++) {     // BL in chunks of 32 bits
      uint32_t c = counts[i + j * size];
      for (int l = (j == 0 ? 1 : 0); l < 32; l++) {

        if ((c & (one << l)) != 0) {
          int iplus1 = negative > 0 ? -(i + 1) : (i + 1);
          sparse_indices[s][sparse_counts[s]++] = iplus1;
        };
        s++;
        if (s == K) {
          break;
        }
      }
    }
  }
}
} // namespace test_helper

template <typename T>
void PulsedRPUWeightUpdater<T>::updateVectorWithDeviceAndCounts(
    T **weights,
    const T *x_input,
    const int x_inc,
    const T *d_input,
    const int d_inc,
    const T learning_rate,
    const int m_batch_info,
    PulsedRPUDeviceBase<T> *rpu_device,
    uint32_t *x_counts32,
    uint32_t *d_counts32) {

  // for debugging: use cuda format of counts to update.
  // simply generate some fake bit lines (to setup the memory etc and to get the current BL)
  if (!sblm_->supports(up_.pulse_type)) {
    RPU_FATAL("Requested pulse type not supported.");
  }
  d_noz_ = 0;
  x_noz_ = 0;
  int BL = sblm_->makeCounts(
      x_input, x_inc, x_noz_, d_input, d_inc, d_noz_, &*rng_, (T)fabsf(learning_rate),
      rpu_device->getWeightGranularity(), up_);

  // translate to sparse format
  int *x_counts_p;
  int *x_counts_n;
  int *d_counts;
  int **x_indices_p;
  int **x_indices_n;
  int **d_indices;

  bool do_negative_separatly = sblm_->getCountsAndIndices(
      x_counts_p, x_counts_n, d_counts, x_indices_p, x_indices_n, d_indices);
  if (do_negative_separatly) {
    RPU_FATAL("no supported mode for debugging currently")
  }
  // could test for similarity of given counts with generated ones...

  // translate given counts
  test_helper::getSparseCountsFromCounts(x_indices_p, x_counts_p, x_counts32, BL, this->x_size_);
  test_helper::getSparseCountsFromCounts(d_indices, d_counts, d_counts32, BL, this->d_size_);

  // pulsed device update
  rpu_device->initUpdateCycle(
      weights, up_, learning_rate, m_batch_info, x_input, x_inc, d_input, d_inc);

  // sblm_->printCounts(BL);
  // for info: in BLM (cuda) makeCounts is additional info for debugging the bit line makers
  if (BL > 0) {
    int lr_sign = learning_rate < (T)0.0 ? -1 : 1;
    for (int k = 0; k < BL; k++) {
      if (d_counts[k] > 0) {
        for (int ii = 0; ii < d_counts[k]; ii++) {

          int i_signed = d_indices[k][ii];
          int d_sign = i_signed < 0 ? -lr_sign : lr_sign;
          int i = (i_signed < 0 ? -i_signed : i_signed) - 1;
          if (x_counts_p[k] > 0) {
            rpu_device->doSparseUpdate(weights, i, x_indices_p[k], x_counts_p[k], d_sign, &*rng_);
          }
        }
      }
    }
  }
  rpu_device->finishUpdateCycle(weights, up_, learning_rate, m_batch_info);
}

template class PulsedRPUWeightUpdater<float>;
#ifdef RPU_USE_DOUBLE
template class PulsedRPUWeightUpdater<double>;
#endif
#ifdef RPU_USE_FP16
template class PulsedRPUWeightUpdater<half_t>;
#endif

} // namespace RPU
