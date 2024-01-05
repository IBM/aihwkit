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
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string.h>
#include <unordered_map>
#include <vector>
#ifdef RPU_USE_WITH_TORCH
#include <ATen/ATen.h>
#endif

// NOTE: fp16 only available for CUDA
#ifdef RPU_USE_CUDA

#ifdef RPU_USE_FP16
#define RPU_DEFINE_CUDA_HALF_ARRAY
#ifdef RPU_BFLOAT_AS_FP16
#include "cuda_bf16.h"
typedef __nv_bfloat16 half_t;
typedef __nv_bfloat162 half2_t;
#else
#include "cuda_fp16.h"
typedef half half_t;
typedef half2 half2_t;
#endif
#else
#ifdef RPU_PARAM_FP16
#define RPU_DEFINE_CUDA_HALF_ARRAY
#include "cuda_fp16.h"
typedef half half_t;
typedef half2 half2_t;
#endif
#endif
#endif

#define UNUSED(X) (void)X

#ifdef _MSC_VER
#define PRAGMA(DIRECTIVE) __pragma(DIRECTIVE)
#define PRAGMA_SIMD
#else
#define PRAGMA(DIRECTIVE) _Pragma(#DIRECTIVE)
#define PRAGMA_SIMD PRAGMA(omp simd)
#endif

#ifdef __GNUC__
#define FORCE_INLINE __attribute__((always_inline)) inline
#else
#define FORCE_INLINE inline
#endif

#ifndef __FILENAME__
#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#endif

#define RPU_FATAL(MSG)                                                                             \
  {                                                                                                \
    std::ostringstream ss;                                                                         \
    ss << "Error in " << __FILENAME__ << ":" << __LINE__ << "  ";                                  \
    ss << MSG;                                                                                     \
    throw std::runtime_error(ss.str());                                                            \
  }

#define RPU_INFO(MSG)                                                                              \
  {                                                                                                \
    std::cout << "Info in " << __FILENAME__ << ":" << __LINE__ << "  ";                            \
    std::cout << MSG << std::endl;                                                                 \
  }

#define RPU_NOT_IMPLEMENTED RPU_FATAL("Feature not yet implemented!");

#define RPU_WARNING(MSG)                                                                           \
  {                                                                                                \
    std::ostringstream ss;                                                                         \
    ss << "WARNING in " << __FILENAME__ << ":" << __LINE__ << "  ";                                \
    ss << MSG;                                                                                     \
  }

// #define RPU_DEBUG

#define ENFORCE_NO_DELAYED_UPDATE                                                                  \
  if (this->isDelayedUpdate()) {                                                                   \
    RPU_FATAL("Not supported during delayed update count.");                                       \
  }

#ifndef DEBUG_OUT
#ifdef RPU_DEBUG
#define DEBUG_OUT(x) std::cout << __FILENAME__ << ":" << __LINE__ << " : " << x << std::endl;
#define DEBUG_CALL(x)                                                                              \
  { x; }
#else
#define DEBUG_OUT(x)
#define DEBUG_CALL(x)
#endif
#endif

// Caution: round() might be the best, but slower on GPU. Also GPU and
// CPU have sometimes different rounding behavior for some reasons. In
// case of RINT CPU/GPU results are consistent. Note that using rint()
// means that for very low resolutions, e.g. 1.5, 2, 2.5, 3, 3.5 it
// rounds to 2 2 2 3 4 where round() does 2 2 3 3 4. Probably not very
// critical as it only is in effect for exactly half-way numbers (0.5)
// Further the rounding mode (FENV) is set in rpu abstract. Not sure
// whether this is the best place [each thread needs it?]. However,
// default should be anyway to-neareast (for GPU this even cannot be
// changed apparently) so we should be fine using RINT here.
#define RPU_ROUNDFUN rint
#define RPU_ROUNDFUNF rintf

// for testing
#ifdef RPU_USE_DOUBLE
using num_t = double;
#else
#ifdef RPU_USE_FP16
using num_t = half_t;
#else
using num_t = float;
#endif
#endif

#ifdef RPU_USE_FP16
#ifdef RPU_BFLOAT_AS_FP16
#define RPU_DATA_TYPE_NAME(T)                                                                      \
  (std::is_same<T, half_t>::value ? "bfloat16"                                                     \
                                  : (std::is_same<T, double>::value ? "double" : "float"))
#else
#define RPU_DATA_TYPE_NAME(T)                                                                      \
  (std::is_same<T, half_t>::value ? "float16"                                                      \
                                  : (std::is_same<T, double>::value ? "double" : "float"))
#endif
#else
#define RPU_DATA_TYPE_NAME(T) (std::is_same<T, double>::value ? "double" : "float")
#endif

namespace RPU {

using state_t = std::unordered_map<std::string, std::vector<double>>;

#ifdef RPU_DEFINE_CUDA_HALF_ARRAY
std::ostream &operator<<(std::ostream &out, const half_t &value);
#endif

/* state helper functions */
template <typename T_VEC>
void load(RPU::state_t &state, std::string key, T_VEC &value, bool strict);
template <typename T_VEC> void insert(RPU::state_t &state, std::string key, const T_VEC &value);

void insertWithPrefix(RPU::state_t &extra, const RPU::state_t &state, std::string prefix);
RPU::state_t selectWithPrefix(const RPU::state_t &extra, std::string prefix);

/* Context class */
class Context {
public:
  virtual ~Context() = default;
  virtual void synchronize() const {};
};

typedef Context *ContextPtr;

template <typename T, typename... Args> std::unique_ptr<T> make_unique(Args &&...args) {
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

template <typename T, typename RNGClass>
inline T getDiscretizedValue(T value, T res, bool sto_round, RNGClass &rng) {

  return (res <= (T)0.0)
             ? value
             : (sto_round ? (T)RPU_ROUNDFUNF(value / res + (rng.sampleUniform() - (T)0.5)) * res
                          : (T)RPU_ROUNDFUNF(value / res) * res);
}

template <typename T, typename RNGClass>
inline T getDiscretizedValueRound(T value, T res, bool sto_round, RNGClass &rng) {

  return (res <= (T)0.0)
             ? value
             : (sto_round ? (T)roundf(value / res + (rng.sampleUniform() - (T)0.5)) * res
                          : (T)roundf(value / res) * res);
}

template <bool sto_round, typename T, typename RNGClass>
inline T getDiscretizedValueSR(T value, T res, RNGClass &rng) {
  if (sto_round) {
    return getDiscretizedValue(value, res, true, rng);
  }
  return (res <= (T)0.0) ? value : ((T)RPU_ROUNDFUNF(value / res) * res);
}

template <typename T> inline T **Array_2D_Get(size_t r, size_t c) {
  T **arr = new T *[r];
  arr[0] = new T[(size_t)r * c];
  for (size_t i = 0; i < r; ++i) {
    arr[i] = *arr + (size_t)c * i;
  }
  return arr;
}

template <typename T> inline T **Array_2D_Get_Eye(size_t n) {

  T **eye = Array_2D_Get<T>(n, n);

  for (size_t j = 0; j < n; ++j) {
    for (size_t i = 0; i < n; ++i) {
      if (i == j) {
        eye[i][j] = (T)1.0;
      } else {
        eye[i][j] = (T)0.0;
      }
    }
  }
  return eye;
}

template <typename T> void Array_2D_Free(T **arr) {
  if (arr != nullptr) {
    delete[] (*arr);
    *arr = nullptr;
    delete[] arr;
    arr = nullptr;
  }
}

template <typename T> inline T ***Array_3D_Get(size_t n, size_t r, size_t c) {
  T ***arr = new T **[n];
  for (size_t j = 0; j < n; j++) {
    arr[j] = new T *[r];
  }
  arr[0][0] = new T[r * c * n];
  for (size_t j = 0; j < n; j++) {
    for (size_t i = 0; i < r; ++i) {
      arr[j][i] = arr[0][0] + c * i + j * r * c;
    }
  }
  return arr;
}

template <typename T> void Array_3D_Free(T ***arr, size_t n) {
  if (arr != nullptr) {
    delete[] arr[0][0];
    arr[0][0] = nullptr;

    for (size_t j = 0; j < n; j++) {
      delete[] arr[j];
      arr[j] = nullptr;
    }

    delete[] arr;
    arr = nullptr;
  }
}

template <typename T> T Find_Absolute_Max(const T *data, int data_length, int inc = 1) {
  T max_input_value = 0;
  for (int i = 0; i < data_length * inc; i += inc) {
    T abs_value = (data[i] >= (T)0.0) ? data[i] : -data[i];
    if (abs_value > max_input_value) {
      max_input_value = abs_value;
    }
  }
  return max_input_value;
}

template <typename T> T Find_Max(const T *data, int data_length) {
  T max_input_value = data[data_length - 1];
  for (int i = 0; i < data_length - 1; ++i) {
    if (data[i] > max_input_value) {
      max_input_value = data[i];
    }
  }
  return max_input_value;
}

template <typename T> T Find_Min(const T *data, int data_length) {
  T min_input_value = data[data_length - 1];
  for (int i = 0; i < data_length; ++i) {
    if (data[i] < min_input_value) {
      min_input_value = data[i];
    }
  }
  return min_input_value;
}
} // namespace RPU
