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

#include "rng.h"
#include "utility_functions.h"
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <math.h>
#include <memory>
#include <random>
#include <time.h>

namespace RPU {

/********************************************************/
/* RealWorldRNG                                         */
template <typename T> RealWorldRNG<T>::RealWorldRNG(unsigned int seed) { setSeed(seed); }

template <typename T> void RealWorldRNG<T>::setSeed(unsigned int seed) {
  if (!seed) {
    seed_ = (unsigned int)std::chrono::high_resolution_clock::now().time_since_epoch().count();
  } else {
    seed_ = seed;
  }
  gen_ = std::unique_ptr<std::default_random_engine>(
      new std::default_random_engine((unsigned int)seed_));
}

/********************************************************/
/* RNG                                                  */

template <typename T> RNG<T>::RNG(unsigned int rseed) : gauss_list_size_(FIXED_LIST_SIZE) {
  this->setSeed(rseed);
}

template <typename T> RNG<T>::~RNG() {
  if (gauss_numbers_list_ != nullptr) {
    delete[] gauss_numbers_list_;
    gauss_numbers_list_ = nullptr;
  }
}

/*********************************************************************************/
// copy constructor
template <typename T> RNG<T>::RNG(const RNG<T> &other) {
  // do never copy random numbers. Just re-generate
  gauss_list_size_ = other.gauss_list_size_;
  this->setSeed(other.seed_);
}

// copy assignment
template <typename T> RNG<T> &RNG<T>::operator=(const RNG<T> &other) {
  RNG<T> tmp(other);
  swap(*this, tmp);
  return *this;
}
// move constructor
template <typename T> RNG<T>::RNG(RNG<T> &&other) noexcept { *this = std::move(other); }

// move assignment
template <typename T> RNG<T> &RNG<T>::operator=(RNG<T> &&other) noexcept {

  gauss_numbers_list_ = other.gauss_numbers_list_;
  other.gauss_numbers_list_ = nullptr;
  gauss_list_size_ = other.gauss_list_size_;
  seed_ = other.seed_;
  return *this;
}

template <typename T> void RNG<T>::randomizeSeed() {
  unsigned int seed =
      (unsigned int)std::chrono::high_resolution_clock::now().time_since_epoch().count();
  srand(seed);
  fast_srand((randomint_t)(rand() % RPU_MAX_RAND_RANGE));
  generateNewList();
  seed_ = 0;
}

template <typename T> void RNG<T>::setSeed(unsigned int seed) {
  seed_ = seed;
  if (seed == 0) {
    randomizeSeed();
  } else {
    srand((randomint_t)seed);
    fast_srand((randomint_t)seed);
    generateNewList();
  }
}

template <typename T> void RNG<T>::generateNewList() { generateNewList(gauss_list_size_); }
template <typename T> void RNG<T>::generateNewList(int list_size) {
#ifdef RPU_USE_FASTMOD
  if (list_size != FIXED_LIST_SIZE) {
    RPU_FATAL("Fast mode needs constant list size (" << FIXED_LIST_SIZE << ") got: " << list_size);
  }
#endif
  unsigned long long seed = seed_;
  if (seed == 0) {
    seed = (unsigned long long)std::chrono::high_resolution_clock::now().time_since_epoch().count();
  }
  std::default_random_engine generator{(unsigned int)seed};
  std::normal_distribution<float> random_dist{};
  auto nrnd = std::bind(random_dist, generator);

  float *numbers = new float[list_size];

  for (int i = 0; i < list_size; ++i) {
    numbers[i] = nrnd();
  }

  if (gauss_numbers_list_ != nullptr) {
    delete[] gauss_numbers_list_;
  }

  gauss_list_size_ = list_size;
  gauss_numbers_list_ = numbers;
}

template class RNG<float>;
template class RealWorldRNG<float>;
#ifdef RPU_USE_DOUBLE
template class RNG<double>;
template class RealWorldRNG<double>;
#endif
#ifdef RPU_USE_FP16
template class RNG<half_t>;
template class RealWorldRNG<half_t>;
#endif

} // namespace RPU
