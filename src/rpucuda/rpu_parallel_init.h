/**
 * (C) Copyright 2026 IBM. All Rights Reserved.
 *
 * Licensed under the MIT license. See LICENSE file in the project root for details.
 */

#pragma once

#include "rng.h"
#include <cstdint>
#include <cstdlib>
#include <vector>

namespace RPU {

inline bool useParallelDeviceInit(int size) {
#ifdef _OPENMP
  const char *value = std::getenv("AIHWKIT_PARALLEL_DEVICE_INIT");
  return size >= 65536 && value != nullptr && value[0] == '1' && value[1] == '\0';
#else
  (void)size;
  return false;
#endif
}

template <typename T>
std::vector<unsigned int> makeDeviceInitRowSeeds(RealWorldRNG<T> *rng, int rows) {
  std::vector<unsigned int> seeds(rows);
  for (int i = 0; i < rows; ++i) {
    // Two 16-bit draws avoid repeated row streams on large devices.
    uint32_t high = (uint32_t)((float)rng->sampleUniform() * 65536.0f);
    uint32_t low = (uint32_t)((float)rng->sampleUniform() * 65536.0f);
    uint32_t seed = (high << 16) | low;
    seeds[i] = seed ? seed : (uint32_t)(i + 1);
  }
  return seeds;
}

} // namespace RPU
