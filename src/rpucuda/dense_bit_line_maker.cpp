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

#include "dense_bit_line_maker.h"
#include "rpu_pulsed_device.h"
#include "utility_functions.h"

namespace RPU {

// init and memory
template <typename T> void DenseBitLineMaker<T>::initialize(int x_size, int d_size) {
  if (x_size != x_size_ || d_size != d_size_ || !containers_allocated_) {
    x_size_ = x_size;
    d_size_ = d_size;

    allocateContainers();
  }
}

template <typename T> void DenseBitLineMaker<T>::allocateContainers() {

  freeContainers();
  rw_rng_ = RPU::make_unique<RealWorldRNG<T>>(0);

  x_counts_ = new int[x_size_];
  d_counts_ = new int[d_size_];
  x_values_ = new T[x_size_];
  d_values_ = new T[d_size_];

  coincidences_ = new int[d_size_ * x_size_];
  containers_allocated_ = true;
}

template <typename T> void DenseBitLineMaker<T>::freeContainers() {

  if (containers_allocated_) {

    delete[] d_counts_;
    delete[] x_counts_;
    delete[] d_values_;
    delete[] x_values_;

    delete[] coincidences_;

    d_counts_ = nullptr;
    x_counts_ = nullptr;
    d_values_ = nullptr;
    x_values_ = nullptr;

    coincidences_ = nullptr;

    rw_rng_ = nullptr;

    containers_allocated_ = false;
  }
}

// ctor
template <typename T>
DenseBitLineMaker<T>::DenseBitLineMaker(int x_size, int d_size)
    : x_size_(x_size), d_size_(d_size), containers_allocated_(false) {}

// dtor
template <typename T> DenseBitLineMaker<T>::~DenseBitLineMaker() { freeContainers(); }

// copy construcutor
template <typename T> DenseBitLineMaker<T>::DenseBitLineMaker(const DenseBitLineMaker<T> &other) {

  x_size_ = other.x_size_;
  d_size_ = other.d_size_;

  if (other.containers_allocated_) {
    initialize(other.x_size_, other.d_size_);

    for (int k = 0; k < d_size_; ++k) {
      d_counts_[k] = other.d_counts_[k];
      d_values_[k] = other.d_values_[k];
    }

    for (int k = 0; k < x_size_; ++k) {
      x_counts_[k] = other.x_counts_[k];
      x_values_[k] = other.x_values_[k];
    }

    for (int k = 0; k < d_size_ * x_size_; ++k) {
      coincidences_[k] = other.coincidences_[k];
    }
  }
}

// copy assignment
template <typename T>
DenseBitLineMaker<T> &DenseBitLineMaker<T>::operator=(const DenseBitLineMaker<T> &other) {

  DenseBitLineMaker<T> tmp(other);
  swap(*this, tmp);
  return *this;
}

// move constructor
template <typename T> DenseBitLineMaker<T>::DenseBitLineMaker(DenseBitLineMaker<T> &&other) {

  *this = std::move(other);
}

// move assignment
template <typename T>
DenseBitLineMaker<T> &DenseBitLineMaker<T>::operator=(DenseBitLineMaker<T> &&other) {

  // pointers
  d_counts_ = other.d_counts_;
  x_counts_ = other.x_counts_;
  d_values_ = other.d_values_;
  x_values_ = other.x_values_;

  coincidences_ = other.coincidences_;

  // set pointers to null
  other.d_counts_ = nullptr;
  other.x_counts_ = nullptr;
  other.d_values_ = nullptr;
  other.x_values_ = nullptr;

  other.coincidences_ = nullptr;

  // other values
  x_size_ = other.x_size_;
  d_size_ = other.d_size_;

  rw_rng_ = std::move(other.rw_rng_);

  containers_allocated_ = other.containers_allocated_;

  return *this;
}

/**************************************************************************************/
/* Take the mean of the probability as counts. No variation in the count number.   */

template <typename T>
inline void DenseBitLineMaker<T>::generateCountsMean(
    int *counts,
    const T *v,
    const int v_inc,
    const int v_size,
    int &v_noz,
    const T p,
    RNG<T> *rng,
    const int BL,
    const T res,
    const bool sto_round,
    const T lr) {
  int j_v = 0;
  for (int j = 0; j < v_size; j++) {

    T v_value = lr < (T)0.0 ? -v[j_v] : v[j_v];
    j_v += v_inc;

    T pp = getDiscretizedValue((T)fabsf((T)v_value) * p, res, sto_round, *rng);

    if (pp == (T)0.0) {
      counts[j] = 0;
      v_noz++;
      continue;
    }

    int ntimes = MAX(MIN((int)RPU_ROUNDFUNF((T)BL * pp), BL), 0);
    counts[j] = (v_value >= (T)0.0) ? ntimes : -ntimes;
  }
}

/** generates coincidences by just multiplying the respective count
 prob. no random fluctuations.  Coincidences might be pos of
 negative, depending on the direction of update**/
template <typename T>
void DenseBitLineMaker<T>::generateCoincidences(
    int *coincidences,
    const int *x_counts,
    const int x_size,
    const int *d_counts,
    const int d_size,
    const int BL) {
  T bl = (T)BL;
  int idx = 0;
  for (int i = 0; i < d_size; ++i) {
    T dc = (T)d_counts[i];
    if (dc != (T)0.0) {
      dc /= bl;
      PRAGMA_SIMD
      for (int j = 0; j < x_size; ++j) {
        coincidences[idx++] = (int)RPU_ROUNDFUNF(dc * (T)x_counts[j]);
      }
    } else {
      // need to set to zero
      PRAGMA_SIMD
      for (int j = 0; j < x_size; ++j) {
        coincidences[idx++] = 0;
      }
    }
  }
}

/**************************************************************************************/
/* just discretize. Deterministic, could be generated according to Murat's bitlines   */

template <typename T>
inline void DenseBitLineMaker<T>::generateDetImplicit(
    T *pcounts,
    const T *v,
    const int v_inc,
    const int v_size,
    int &v_noz,
    const T p,
    RNG<T> *rng,
    const int BL,
    const T res,
    const bool sto_round,
    const T lr) {
  int j_v = 0;
  for (int j = 0; j < v_size; j++) {

    T v_value = lr < (T)0.0 ? -v[j_v] : v[j_v];
    j_v += v_inc;

    T pp = getDiscretizedValue<T>((T)fabsf(v_value) * p, res, sto_round, *rng);

    if (pp == (T)0.0) {
      v_noz++;
    }

    pp = MAX(MIN(pp, (T)1.0), (T)0.0);
    pcounts[j] = (v_value >= (T)0.0) ? pp : -pp;
  }
}

template <typename T>
void DenseBitLineMaker<T>::generateCoincidencesDetI(
    int *coincidences,
    const T *x_values,
    const int x_size,
    const T *d_values,
    const int d_size,
    const int BL) {
  int idx = 0;
  for (int i = 0; i < d_size; ++i) {
    T dc = d_values[i];
    if (dc != (T)0.0) {
      dc *= BL;
      PRAGMA_SIMD
      for (int j = 0; j < x_size; ++j) {
        coincidences[idx++] = (int)RPU_ROUNDFUNF(dc * x_values[j]);
      }
    } else {
      // need to set to zero
      PRAGMA_SIMD
      for (int j = 0; j < x_size; ++j) {
        coincidences[idx++] = 0;
      }
    }
  }
}

// makeCounts
template <typename T>
int *DenseBitLineMaker<T>::makeCoincidences(
    const T *x_in,
    const int x_inc,
    int &x_noz,
    const T *d_in,
    const int d_inc,
    int &d_noz,
    RNG<T> *rng,
    const T lr,
    const T dw_min,
    const PulsedUpdateMetaParameter<T> &up) {

  T A = 0;
  T B = 0;
  int BL = 0;
  // negative lr allowed in the below, thus (T)fabsf(lr)
  if (up.update_bl_management || up.update_management) {

    T x_abs_max = Find_Absolute_Max<T>(x_in, x_size_, x_inc);
    T d_abs_max = Find_Absolute_Max<T>(d_in, d_size_, d_inc);

    up.performUpdateManagement(
        BL, A, B, up.desired_BL, x_abs_max, d_abs_max, (T)fabsf((float)lr), dw_min);
  } else {

    up.calculateBlAB(BL, A, B, (T)fabsf((float)lr), dw_min);
  }

  if (!containers_allocated_) {
    initialize(x_size_, d_size_);
  }

  switch (up.pulse_type) {

  case PulseType::MeanCount:
    // x counts
    generateCountsMean(
        x_counts_, x_in, x_inc, x_size_, x_noz, B, rng, BL, up.res, up.sto_round, lr);
    // d counts
    generateCountsMean(
        d_counts_, d_in, d_inc, d_size_, d_noz, A, rng, BL, up.res, up.sto_round, lr);

    generateCoincidences(coincidences_, x_counts_, x_size_, d_counts_, d_size_, BL);
    break;

  case PulseType::DeterministicImplicit: {
    // note that GPU version does not support stoc round / bl_thres / sparsity for this type.

    // x counts
    generateDetImplicit(
        x_values_, x_in, x_inc, x_size_, x_noz, B, rng, BL, up.x_res_implicit, up.sto_round, lr);

    // d counts
    generateDetImplicit(
        d_values_, d_in, d_inc, d_size_, d_noz, A, rng, BL, up.d_res_implicit, up.sto_round, lr);

    generateCoincidencesDetI(coincidences_, x_values_, x_size_, d_values_, d_size_, BL);
  } break;
  default:
    RPU_FATAL("PulseType not supported");
  }

  return coincidences_;
}

template <typename T> bool DenseBitLineMaker<T>::supports(RPU::PulseType pulse_type) const {
  return PulseType::MeanCount == pulse_type || PulseType::DeterministicImplicit == pulse_type;
}

template <typename T> void DenseBitLineMaker<T>::printCounts(int max_n) const {

  if (!containers_allocated_) {
    RPU_FATAL("Containter not yet allocated");
  }

  std::cout << "\n\nX_counts:\n";
  for (int k = 0; k < MIN(x_size_, max_n); k++) {
    std::cout << x_counts_[k] << ", ";
  }
  std::cout << "\n\nD_counts:\n";
  for (int k = 0; k < MIN(d_size_, max_n); k++) {
    std::cout << d_counts_[k] << ", ";
  }
  std::cout << std::endl;
}

template class DenseBitLineMaker<float>;
#ifdef RPU_USE_DOUBLE
template class DenseBitLineMaker<double>;
#endif
#ifdef RPU_USE_FP16
template class DenseBitLineMaker<half_t>;
#endif

} // namespace RPU
