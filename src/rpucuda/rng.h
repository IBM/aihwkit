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

#include "math_util.h"
#include <chrono>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <memory>
#include <random>
#include <time.h>

#ifdef RPU_USE_FASTRAND
#define RPU_MAX_RAND_RANGE 0x7FFF
#define RANDFUN fastrand
#include <stdint.h>
typedef int_fast16_t randomint_t;
#else
#define RANDFUN rand
#define RPU_MAX_RAND_RANGE RAND_MAX
typedef int randomint_t;
#endif

// NEED TO BE 0x7FFF for FASTRAND!!!
#define FIXED_LIST_SIZE 32768
#define FIXED_LIST_SIZE_MSK 0x7FFF
namespace RPU {

static randomint_t g_seed;

// Used to seed the generator.
inline void fast_srand(randomint_t seed) { g_seed = seed; }
// fastrand routine returns one integer, similar output value range as C lib.
inline randomint_t fastrand() {
  g_seed = (214013 * g_seed + 2531011);
  return (g_seed >> 16) & 0x7FFF;
}

/* this is used for construction (populate device) */
template <typename T> class RealWorldRNG {
public:
  explicit RealWorldRNG(unsigned int seed);
  RealWorldRNG() : RealWorldRNG(0){};

  void setSeed(unsigned int seed);

  FORCE_INLINE T sampleGauss() { return ndist_(*gen_); };

  FORCE_INLINE T sampleUniform() { return udist_(*gen_); };

private:
  unsigned int seed_ = 0;
  std::unique_ptr<std::default_random_engine> gen_;
  std::uniform_real_distribution<float> udist_{0.0f, 1.0f};
  std::normal_distribution<float> ndist_{0.0f, 1.0f};
};

/* Faster approximative RNG for CPU. This is used for everything
 else on the CPU. Note that GPU ALWAYS uses "real world
 random numbers !
*/
template <typename T> class RNG {

public:
  explicit RNG(unsigned int seed);
  RNG() : RNG<T>(0){};
  ~RNG();

  RNG(const RNG<T> &);
  RNG<T> &operator=(const RNG<T> &);
  RNG(RNG<T> &&) noexcept;
  RNG<T> &operator=(RNG<T> &&) noexcept;

  friend void swap(RNG<T> &a, RNG<T> &b) noexcept {
    using std::swap;
    swap(a.gauss_list_size_, b.gauss_list_size_);
    swap(a.seed_, b.seed_);
    swap(a.gauss_numbers_list_, b.gauss_numbers_list_);
  }

  void generateNewList();
  void generateNewList(int list_size);

  void randomizeSeed();
  void setSeed(unsigned int seed);

  FORCE_INLINE randomint_t sample() { return RANDFUN(); }

  FORCE_INLINE T sampleUniform() { return (float)RANDFUN() / (float)RPU_MAX_RAND_RANGE; }

  FORCE_INLINE T sampleUniform(float min_max) {
    return (((float)RANDFUN() / (float)RPU_MAX_RAND_RANGE) - 0.5f) * 2.0f * min_max;
  }

  FORCE_INLINE T sampleUniform(float min_value, float max_value) {
    return (((float)RANDFUN() / (float)RPU_MAX_RAND_RANGE) * (max_value - min_value)) + min_value;
  }

  FORCE_INLINE T sampleGauss() {
#ifdef RPU_USE_FASTRAND
#ifdef RPU_USE_FASTMOD
    return gauss_numbers_list_[RANDFUN()]; // assume 0x7FFF for list size
#else
    return gauss_numbers_list_[RANDFUN() % gauss_list_size_];
#endif
#else
#ifdef RPU_USE_FASTMOD
    return gauss_numbers_list_[RANDFUN() & FIXED_LIST_SIZE_MSK];
#else
    return gauss_numbers_list_[RANDFUN() % gauss_list_size_];
#endif
#endif
  }

private:
  int gauss_list_size_;
  unsigned int seed_;
  float *gauss_numbers_list_ = nullptr;
};

} // namespace RPU
