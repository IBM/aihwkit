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

template <typename T> class SparseBitLineMaker {

public:
  explicit SparseBitLineMaker(int x_size, int d_size);
  SparseBitLineMaker(){};
  virtual ~SparseBitLineMaker();
  SparseBitLineMaker(const SparseBitLineMaker<T> &);
  SparseBitLineMaker<T> &operator=(const SparseBitLineMaker<T> &);
  SparseBitLineMaker(SparseBitLineMaker<T> &&);
  SparseBitLineMaker<T> &operator=(SparseBitLineMaker<T> &&);

  friend void swap(SparseBitLineMaker<T> &a, SparseBitLineMaker<T> &b) noexcept {
    using std::swap;

    swap(a.x_size_, b.x_size_);
    swap(a.d_size_, b.d_size_);
    swap(a.max_BL_, b.max_BL_);
    swap(a.n_indices_used_, b.n_indices_used_);

    swap(a.d_indices_, b.d_indices_);
    swap(a.x_indices_p_, b.x_indices_p_);
    swap(a.x_indices_n_, b.x_indices_n_);
    swap(a.d_counts_, b.d_counts_);
    swap(a.x_counts_p_, b.x_counts_p_);
    swap(a.x_counts_n_, b.x_counts_n_);
  }

  /* returns current BL*/
  virtual int makeCounts(
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

  /* returns whether x_n indices/counts are used*/
  bool getCountsAndIndices(
      int *&x_counts_p,
      int *&x_counts_n,
      int *&d_counts,
      int **&x_indices_p,
      int **&x_indices_n,
      int **&d_indices);

  void printCounts(int BL) const;
  bool supports(RPU::PulseType pulse_type) const;

  /* Ignore the buffer / indices, as they will be generated anew each sample.*/
  void dumpExtra(RPU::state_t &extra, const std::string prefix){};
  void loadExtra(const RPU::state_t &extra, const std::string prefix, bool strict){};

private:
  void freeContainers();
  void allocateContainers(int max_BL);
  void initialize(int x_size, int d_size, int max_BL);

  int x_size_ = 0;
  int d_size_ = 0;
  int max_BL_ = 0; // tracks the size of the containers
  bool n_indices_used_ = false;

  int **d_indices_ = nullptr;
  int **x_indices_p_ = nullptr;
  int **x_indices_n_ = nullptr;
  int *d_counts_ = nullptr;
  int *x_counts_p_ = nullptr;
  int *x_counts_n_ = nullptr;
};

} // namespace RPU
