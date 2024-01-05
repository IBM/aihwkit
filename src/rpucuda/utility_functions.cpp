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

#include "utility_functions.h"
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string.h>
#include <unordered_map>
#include <vector>

namespace RPU {

#ifdef RPU_DEFINE_CUDA_HALF_ARRAY
std::ostream &operator<<(std::ostream &out, const half_t &value) {
  out << (float)value;
  return out;
};
#endif

template <typename T_VEC>
void load(RPU::state_t &state, std::string key, T_VEC &value, bool strict) {
  std::vector<double> tmp;
  try {
    tmp = state.at(key);
  } catch (const std::out_of_range &oor) {
    if (strict) {
      RPU_FATAL("Cannot find the vector key `" << key << "` in state.");
    }
    return; // do nothing
  }
  T_VEC out(tmp.begin(), tmp.end());
  value = out;
}

#define RPU_LOAD_VECTOR(T) template void load(RPU::state_t &, std::string, std::vector<T> &, bool);

RPU_LOAD_VECTOR(float);
RPU_LOAD_VECTOR(double);
RPU_LOAD_VECTOR(int);
RPU_LOAD_VECTOR(uint64_t);
RPU_LOAD_VECTOR(int64_t);
RPU_LOAD_VECTOR(bool);
#ifdef RPU_DEFINE_CUDA_HALF_ARRAY
RPU_LOAD_VECTOR(half_t);
#endif
#undef RPU_LOAD_VECTOR

#define RPU_LOAD_SINGLE(T)                                                                         \
  template <> void load(RPU::state_t &state, std::string key, T &value, bool strict) {             \
    std::vector<double> tmp;                                                                       \
    try {                                                                                          \
      tmp = state.at(key);                                                                         \
    } catch (const std::out_of_range &oor) {                                                       \
      if (strict) {                                                                                \
        RPU_FATAL("Cannot find the single key `" << key << "` in state.");                         \
      }                                                                                            \
      return;                                                                                      \
    }                                                                                              \
    value = (T)tmp[0];                                                                             \
  }

RPU_LOAD_SINGLE(float);
RPU_LOAD_SINGLE(double);
RPU_LOAD_SINGLE(int);
RPU_LOAD_SINGLE(uint64_t);
RPU_LOAD_SINGLE(int64_t);
RPU_LOAD_SINGLE(bool);
#ifdef RPU_DEFINE_CUDA_HALF_ARRAY
RPU_LOAD_SINGLE(half_t);
#endif
#undef RPU_LOAD_SINGLE

template <typename T_VEC> void insert(RPU::state_t &state, std::string key, const T_VEC &value) {
  std::vector<double> tmp(value.begin(), value.end());
  state[key] = tmp;
}

#define RPU_INSERT_VECTOR(T)                                                                       \
  template void insert(RPU::state_t &, std::string, const std::vector<T> &value);

RPU_INSERT_VECTOR(float);
RPU_INSERT_VECTOR(double);
RPU_INSERT_VECTOR(int);
RPU_INSERT_VECTOR(uint64_t);
RPU_INSERT_VECTOR(int64_t);
RPU_INSERT_VECTOR(bool);
#ifdef RPU_DEFINE_CUDA_HALF_ARRAY
RPU_INSERT_VECTOR(half_t);
#endif
#undef RPU_INSERT_VECTOR

#define RPU_INSERT_SINGLE(T)                                                                       \
  template <> void insert(RPU::state_t &state, std::string key, const T &value) {                  \
    std::vector<double> tmp{(double)value};                                                        \
    state[key] = tmp;                                                                              \
  }

RPU_INSERT_SINGLE(float);
RPU_INSERT_SINGLE(double);
RPU_INSERT_SINGLE(int);
RPU_INSERT_SINGLE(uint64_t);
RPU_INSERT_SINGLE(int64_t);
RPU_INSERT_SINGLE(bool);
#ifdef RPU_DEFINE_CUDA_HALF_ARRAY
RPU_INSERT_SINGLE(half_t);
#endif
#undef RPU_INSERT_SINGLE

/* inserts the state into the extra and adding the prefix to the keys*/
void insertWithPrefix(RPU::state_t &extra, const RPU::state_t &state, std::string prefix) {
  std::string full_prefix = prefix + ".";
  for (const auto &i : state) {
    if (extra.count(i.first)) {
      RPU_FATAL("Key " << i.first << " already exists in state map.");
    }
    extra[full_prefix + i.first] = i.second;
  }
}

/* extracts all variables starting with prefix */
RPU::state_t selectWithPrefix(const RPU::state_t &extra, std::string prefix) {
  RPU::state_t state;
  std::string full_prefix = prefix + ".";

  for (const auto &i : extra) {
    if (i.first.find(full_prefix) == 0) {
      std::string name = i.first.substr(full_prefix.length());
      state[name] = i.second;
    }
  }
  return state;
}
} // namespace RPU
