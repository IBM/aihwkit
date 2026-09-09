/**
 * (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
 *
 * Licensed under the MIT license. See LICENSE file in the project root for details.
 */

#include "rpu_weight_updater.h"
#include "rpu_vector_device.h"
#include "rpu_constantstep_device.h"
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

        // HS tracking: Process all HS states for this time slot k
        if ((up_.pulse_type == PulseType::HalfselectedStochastic ||
             up_.pulse_type == PulseType::HalfselectedStochasticStream) &&
            rpu_device->isHSTrackingEnabled()) {

          // Process all rows for HS tracking
          for (int i = 0; i < rpu_device->getDSize(); i++) {
            bool d_pulse_at_i = false;
            int d_sign_for_i = lr_sign;

            // Check if row i has d pulse
            for (int ii = 0; ii < d_counts[k]; ii++) {
              int i_signed = d_indices[k][ii];
              int d_row = i_signed < 0 ? -i_signed - 1 : i_signed - 1;
              if (d_row == i) {
                d_pulse_at_i = true;
                d_sign_for_i = i_signed < 0 ? -lr_sign : lr_sign;
                break;
              }
            }

            if (d_pulse_at_i) {
              // Row i has d pulse: check each column for HS state
              for (int j = 0; j < rpu_device->getXSize(); j++) {
                bool x_pulse_at_j = false;
                int x_sign_for_j = 1;

                // Check if column j has x pulse (positive)
                for (int jj = 0; jj < x_counts_p[k]; jj++) {
                  int j_signed = x_indices_p[k][jj];
                  int x_col = j_signed < 0 ? -j_signed - 1 : j_signed - 1;
                  if (x_col == j) {
                    x_pulse_at_j = true;
                    x_sign_for_j = j_signed < 0 ? -1 : 1;
                    break;
                  }
                }

                // Check negative x if separated
                if (!x_pulse_at_j && do_negative_separatly) {
                  for (int jj = 0; jj < x_counts_n[k]; jj++) {
                    int j_signed = x_indices_n[k][jj];
                    int x_col = j_signed < 0 ? -j_signed - 1 : j_signed - 1;
                    if (x_col == j) {
                      x_pulse_at_j = true;
                      x_sign_for_j = j_signed < 0 ? -1 : 1;
                      break;
                    }
                  }
                }

                // Update HS state for cell (i,j)
                if (i < rpu_device->getDSize() && j < rpu_device->getXSize()) {
                  HalfSelectedState prev_hs = rpu_device->getHSStates()[i][j];
                  HalfSelectedState curr_hs;

                  if (x_pulse_at_j && d_pulse_at_i) {
                    // Coincidence: skip HS tracking here, will be handled after weight update
                    continue;
                  } else if (x_pulse_at_j && !d_pulse_at_i) {
                    curr_hs = (x_sign_for_j == d_sign_for_i) ? HalfSelectedState::HS1 : HalfSelectedState::HS3;
                  } else if (!x_pulse_at_j && d_pulse_at_i) {
                    curr_hs = (x_sign_for_j == d_sign_for_i) ? HalfSelectedState::HS2 : HalfSelectedState::HS4;
                  } else {
                    continue; // No pulse, keep current state
                  }

                  bool apply_decay = rpu_device->shouldApplyHSDecay(prev_hs, curr_hs);

                  // Update transition count and state
                  rpu_device->updateHSTransitionCount(prev_hs, curr_hs, i);
                  rpu_device->getHSStates()[i][j] = curr_hs;
                }
              }
            } else {
              // Row i has no d pulse: only check x pulses for HS1/HS3
              for (int jj = 0; jj < x_counts_p[k]; jj++) {
                int j_signed = x_indices_p[k][jj];
                int x_sign = j_signed < 0 ? -1 : 1;
                int j = j_signed < 0 ? -j_signed - 1 : j_signed - 1;

                if (i < rpu_device->getDSize() && j < rpu_device->getXSize()) {
                  HalfSelectedState prev_hs = rpu_device->getHSStates()[i][j];
                  HalfSelectedState curr_hs = (x_sign == d_sign_for_i) ? HalfSelectedState::HS1 : HalfSelectedState::HS3;

                  bool apply_decay = rpu_device->shouldApplyHSDecay(prev_hs, curr_hs);

                  rpu_device->updateHSTransitionCount(prev_hs, curr_hs, i);
                  rpu_device->getHSStates()[i][j] = curr_hs;
                }
              }

              // Handle negative x pulses if separated
              if (do_negative_separatly) {
                for (int jj = 0; jj < x_counts_n[k]; jj++) {
                  int j_signed = x_indices_n[k][jj];
                  int x_sign = j_signed < 0 ? -1 : 1;
                  int j = j_signed < 0 ? -j_signed - 1 : j_signed - 1;

                  if (i < rpu_device->getDSize() && j < rpu_device->getXSize()) {
                    HalfSelectedState prev_hs = rpu_device->getHSStates()[i][j];
                    HalfSelectedState curr_hs = (x_sign == d_sign_for_i) ? HalfSelectedState::HS1 : HalfSelectedState::HS3;

                    bool apply_decay = rpu_device->shouldApplyHSDecay(prev_hs, curr_hs);

                    rpu_device->updateHSTransitionCount(prev_hs, curr_hs, i);
                    rpu_device->getHSStates()[i][j] = curr_hs;
                  }
                }
              }
            }
          }
        }

        // Process weight updates (only for coincidences)
        if (d_counts[k] > 0) {
          for (int ii = 0; ii < d_counts[k]; ii++) {

            int i_signed = d_indices[k][ii];
            int d_sign = i_signed < 0 ? -lr_sign : lr_sign;
            int i = i_signed < 0 ? -i_signed - 1 : i_signed - 1;

            // Weight update for coincidences only
            if (x_counts_p[k] > 0) {
              if ((up_.pulse_type == PulseType::HalfselectedStochastic ||
                   up_.pulse_type == PulseType::HalfselectedStochasticStream) &&
                  rpu_device->isHSTrackingEnabled()) {
                // Use virtual HS-aware update method for coincidences
                rpu_device->doSparseUpdateHS(weights, i, x_indices_p[k], x_counts_p[k], d_sign, &*rng_);
              } else {
                rpu_device->doSparseUpdate(weights, i, x_indices_p[k], x_counts_p[k], d_sign, &*rng_);
              }
            }
            if (do_negative_separatly) {
              if (x_counts_n[k] > 0) {
                if ((up_.pulse_type == PulseType::HalfselectedStochastic ||
                     up_.pulse_type == PulseType::HalfselectedStochasticStream) &&
                    rpu_device->isHSTrackingEnabled()) {
                  // Use virtual HS-aware update method for coincidences
                  rpu_device->doSparseUpdateHS(weights, i, x_indices_n[k], x_counts_n[k], d_sign, &*rng_);
                } else {
                  rpu_device->doSparseUpdate(weights, i, x_indices_n[k], x_counts_n[k], d_sign, &*rng_);
                }
              }
            }
          }

          // Post-update HS state setting for coincidence cells
          if ((up_.pulse_type == PulseType::HalfselectedStochastic ||
               up_.pulse_type == PulseType::HalfselectedStochasticStream) &&
              rpu_device->isHSTrackingEnabled()) {

            for (int ii = 0; ii < d_counts[k]; ii++) {
              int i_signed = d_indices[k][ii];
              int d_sign = i_signed < 0 ? -lr_sign : lr_sign;
              int i = i_signed < 0 ? -i_signed - 1 : i_signed - 1;

              // Check positive x pulses for coincidences
              for (int jj = 0; jj < x_counts_p[k]; jj++) {
                int j_signed = x_indices_p[k][jj];
                int j = j_signed < 0 ? -j_signed - 1 : j_signed - 1;

                if (i < rpu_device->getDSize() && j < rpu_device->getXSize()) {
                  HalfSelectedState prev_hs = rpu_device->getHSStates()[i][j];
                  // Weight update direction determines HS state
                  int update_sign = j_signed < 0 ? -d_sign : d_sign;
                  HalfSelectedState curr_hs = (update_sign > 0) ? HalfSelectedState::HS1 : HalfSelectedState::HS3;

                  bool apply_decay = rpu_device->shouldApplyHSDecay(prev_hs, curr_hs);

                  rpu_device->updateHSTransitionCount(prev_hs, curr_hs, i);
                  rpu_device->getHSStates()[i][j] = curr_hs;
                }
              }

              // Check negative x pulses for coincidences (if separated)
              if (do_negative_separatly) {
                for (int jj = 0; jj < x_counts_n[k]; jj++) {
                  int j_signed = x_indices_n[k][jj];
                  int j = j_signed < 0 ? -j_signed - 1 : j_signed - 1;

                  if (i < rpu_device->getDSize() && j < rpu_device->getXSize()) {
                    HalfSelectedState prev_hs = rpu_device->getHSStates()[i][j];
                    // Weight update direction determines HS state
                    int update_sign = j_signed < 0 ? -d_sign : d_sign;
                    HalfSelectedState curr_hs = (update_sign > 0) ? HalfSelectedState::HS1 : HalfSelectedState::HS3;

                    bool apply_decay = rpu_device->shouldApplyHSDecay(prev_hs, curr_hs);

                    rpu_device->updateHSTransitionCount(prev_hs, curr_hs, i);
                    rpu_device->getHSStates()[i][j] = curr_hs;
                  }
                }
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
