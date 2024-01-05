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

#pragma once

#include "rng.h"
#include "rpu_pulsed_meta_parameter.h"
#include <memory>

namespace RPU {

template <typename T> class DenseBitLineMaker {

public:
  explicit DenseBitLineMaker(int x_size, int d_size);
  DenseBitLineMaker(){};
  virtual ~DenseBitLineMaker();
  DenseBitLineMaker(const DenseBitLineMaker<T> &);
  DenseBitLineMaker<T> &operator=(const DenseBitLineMaker<T> &);
  DenseBitLineMaker(DenseBitLineMaker<T> &&);
  DenseBitLineMaker<T> &operator=(DenseBitLineMaker<T> &&);

  friend void swap(DenseBitLineMaker<T> &a, DenseBitLineMaker<T> &b) noexcept {
    using std::swap;

    swap(a.x_size_, b.x_size_);
    swap(a.d_size_, b.d_size_);
    swap(a.containers_allocated_, b.containers_allocated_);
    swap(a.d_counts_, b.d_counts_);
    swap(a.x_counts_, b.x_counts_);
    swap(a.coincidences_, b.coincidences_);
    swap(a.rw_rng_, b.rw_rng_);
  }

  /* returns current BL*/
  virtual int *makeCoincidences(
      const T *x_in,
      const int x_inc,
      int &x_noz,
      const T *d_in,
      const int d_inc,
      int &d_noz,
      RNG<T> *rng,
      const T lr,
      const T dw_min,
      const PulsedUpdateMetaParameter<T> &up);

  void printCounts(int max_n) const;
  bool supports(RPU::PulseType pulse_type) const;

  /* Ignore the buffer / counts, as they will be generated anew each sample.*/
  void dumpExtra(RPU::state_t &extra, const std::string prefix){};
  void loadExtra(const RPU::state_t &extra, const std::string prefix, bool strict){};

private:
  void freeContainers();
  void allocateContainers();
  void initialize(int x_size, int d_size);
  inline void generateCoincidences(
      int *coincidences,
      const int *x_counts,
      const int x_size,
      const int *d_counts,
      const int d_size,
      const int BL);

  inline void generateCoincidencesDetI(
      int *coincidences,
      const T *x_values,
      const int x_size,
      const T *d_values,
      const int d_size,
      const int BL);

  inline void generateCountsMean(
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
      const T lr);

  inline void generateDetImplicit(
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
      const T lr);

  int x_size_ = 0;
  int d_size_ = 0;

  bool containers_allocated_ = false;
  std::unique_ptr<RealWorldRNG<T>> rw_rng_ = nullptr;

  int *coincidences_ = nullptr;
  int *d_counts_ = nullptr;
  int *x_counts_ = nullptr;

  T *d_values_ = nullptr;
  T *x_values_ = nullptr;
};

} // namespace RPU
