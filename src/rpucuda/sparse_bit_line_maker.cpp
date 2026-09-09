/**
 * (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
 *
 * Licensed under the MIT license. See LICENSE file in the project root for details.
 */

#include "sparse_bit_line_maker.h"
#include "rpu_pulsed_device.h"
#include "utility_functions.h"

namespace RPU {

// init and memory
template <typename T> void SparseBitLineMaker<T>::initialize(int x_size, int d_size, int max_BL) {
  x_size_ = x_size;
  d_size_ = d_size;

  allocateContainers(max_BL);
}

template <typename T> void SparseBitLineMaker<T>::allocateContainers(int max_BL) {

  freeContainers();

  max_BL_ = max_BL;
  x_indices_p_ = Array_2D_Get<int>(max_BL_, x_size_);
  x_indices_n_ = Array_2D_Get<int>(max_BL_, x_size_);
  d_indices_ = Array_2D_Get<int>(max_BL_, d_size_);

  x_counts_p_ = new int[max_BL_]();
  x_counts_n_ = new int[max_BL_]();
  d_counts_ = new int[max_BL_]();
}

template <typename T> void SparseBitLineMaker<T>::freeContainers() {

  if (max_BL_ > 0) {
    Array_2D_Free<int>(x_indices_p_);
    Array_2D_Free<int>(x_indices_n_);
    Array_2D_Free<int>(d_indices_);

    delete[] d_counts_;
    delete[] x_counts_p_;
    delete[] x_counts_n_;

    d_counts_ = nullptr;
    x_counts_p_ = nullptr;
    x_counts_n_ = nullptr;

    max_BL_ = 0;
  }
}

// ctor
template <typename T>
SparseBitLineMaker<T>::SparseBitLineMaker(int x_size, int d_size)
    : x_size_(x_size), d_size_(d_size), max_BL_(0) {}

// dtor
template <typename T> SparseBitLineMaker<T>::~SparseBitLineMaker() { freeContainers(); }

// copy construcutor
template <typename T>
SparseBitLineMaker<T>::SparseBitLineMaker(const SparseBitLineMaker<T> &other) {

  x_size_ = other.x_size_;
  d_size_ = other.d_size_;
  max_BL_ = other.max_BL_;
  n_indices_used_ = other.n_indices_used_;

  if (other.max_BL_ > 0) {
    initialize(other.x_size_, other.d_size_, other.max_BL_);

    for (int j = 0; j < x_size_; ++j) {
      for (int k = 0; k < max_BL_; ++k) {
        x_indices_p_[k][j] = other.x_indices_p_[k][j];
        x_indices_n_[k][j] = other.x_indices_n_[k][j];
      }
    }

    for (int i = 0; i < d_size_; ++i) {
      for (int k = 0; k < max_BL_; ++k) {
        d_indices_[k][i] = other.d_indices_[k][i];
      }
    }

    for (int k = 0; k < max_BL_; ++k) {
      d_counts_[k] = other.d_counts_[k];
      x_counts_p_[k] = other.x_counts_p_[k];
      x_counts_n_[k] = other.x_counts_n_[k];
    }
  }
}

// copy assignment
template <typename T>
SparseBitLineMaker<T> &SparseBitLineMaker<T>::operator=(const SparseBitLineMaker<T> &other) {

  SparseBitLineMaker<T> tmp(other);
  swap(*this, tmp);
  return *this;
}

// move constructor
template <typename T> SparseBitLineMaker<T>::SparseBitLineMaker(SparseBitLineMaker<T> &&other) {

  *this = std::move(other);
}

// move assignment
template <typename T>
SparseBitLineMaker<T> &SparseBitLineMaker<T>::operator=(SparseBitLineMaker<T> &&other) {

  // pointers
  d_indices_ = other.d_indices_;
  x_indices_p_ = other.x_indices_p_;
  x_indices_n_ = other.x_indices_n_;

  d_counts_ = other.d_counts_;
  x_counts_p_ = other.x_counts_p_;
  x_counts_n_ = other.x_counts_n_;

  // set pointers to null
  other.d_indices_ = nullptr;
  other.x_indices_p_ = nullptr;
  other.x_indices_n_ = nullptr;

  other.d_counts_ = nullptr;
  other.x_counts_p_ = nullptr;
  other.x_counts_n_ = nullptr;

  // other values
  x_size_ = other.x_size_;
  d_size_ = other.d_size_;
  max_BL_ = other.max_BL_;
  n_indices_used_ = other.n_indices_used_;

  return *this;
}

/*********************************************************************************/
/* bit line generator functions [maybe at some point these can be
   more abstract and put into a registry of some sorts]*/

/* Stochastic with seperate bit trains for positive and negative*/
template <typename T>
inline void generateCountsPN(
    int *counts_p,
    int *counts_n,
    int **indices_p,
    int **indices_n,
    const T *v,
    int v_inc,
    int v_size,
    T P,
    RNG<T> *rng,
    int BL,
    T res,
    bool sto_round,
    int &noz) {

  PRAGMA_SIMD
  for (int k = 0; k < BL; k++) {
    counts_p[k] = 0;
    counts_n[k] = 0;
  }

  int j_v = 0;
  for (int j = 0; j < v_size; j++) {

    T v_value = v[j_v];
    j_v += v_inc;

    T PP = getDiscretizedValue<T>((T)fabsf(v_value) * P, res, sto_round, *rng);

    if (PP == (T)0.0) {
      noz++;
      continue;
    }

    int jplus1_signed = (v_value > (T)0.0) ? j + 1 : -(j + 1);
    if (v_value > (T)0.0) {
      PRAGMA_SIMD
      for (int k = 0; k < BL; k++) {
        if (PP > rng->sampleUniform()) {
          indices_p[k][counts_p[k]++] = jplus1_signed;
        }
      }
    } else {
      for (int k = 0; k < BL; k++) {
        if (PP > rng->sampleUniform()) {
          indices_n[k][counts_n[k]++] = jplus1_signed; // always negative here
        }
      }
    }
  }
}
/* Stochastic Compressed with same bit trains for positive and negative*/
template <typename T>
FORCE_INLINE void generateCounts(
    int *counts,
    int **indices,
    const T *v,
    int v_inc,
    int v_size,
    T P,
    RNG<T> *rng,
    int BL,
    T res,
    bool sto_round,
    int &noz) {

  PRAGMA_SIMD
  for (int k = 0; k < BL; k++) {
    counts[k] = 0;
  }

  int j_v = 0;
  for (int j = 0; j < v_size; j++) {

    T v_value = v[j_v];
    j_v += v_inc;

    T PP = getDiscretizedValue<T>((T)fabsf(v_value) * P, res, sto_round, *rng);
    if (PP == (T)0.0) {
      noz++;
      continue;
    }

    int jplus1_signed = (v_value > (T)0.0) ? j + 1 : -(j + 1);
    PRAGMA_SIMD
    for (int k = 0; k < BL; k++) {
      if (PP > rng->sampleUniform()) {
        indices[k][counts[k]++] = jplus1_signed;
      }
    }
  }
}

/* Other types of bitlines using a functor. BitlineFunctorT needs to
   be of struct type and implement the function getBit, see below */
template <typename T, typename BitlineFunctorT>
inline void generateCountsFunctor(
    int *counts,
    int **indices,
    const T *v,
    int v_inc,
    int v_size,
    T P,
    RNG<T> *rng,
    int BL,
    T res,
    bool sto_round,
    int &noz,
    BitlineFunctorT blfun) {

  PRAGMA_SIMD
  for (int k = 0; k < BL; k++) {
    counts[k] = 0;
  }

  int j_v = 0;
  for (int j = 0; j < v_size; j++) {

    T v_value = v[j_v];
    j_v += v_inc;

    // this is the scaled [0..1] input value
    T PP = getDiscretizedValue<T>((T)fabsf(v_value) * P, res, sto_round, *rng);

    if (PP == (T)0.0) {
      noz++;
      continue;
    }

    int jplus1_signed = (v_value > 0) ? j + 1 : -(j + 1);
    PRAGMA_SIMD
    for (int k = 0; k < BL; k++) {
      if (blfun.getBit(PP, k, BL, rng)) {
        indices[k][counts[k]++] = jplus1_signed;
      }
    }
  }
}

/* Stochastic Stream with continuous pulse trains */
template <typename T>
FORCE_INLINE void generateStreamCounts(
    int *counts,
    int **indices,
    const T *v,
    int v_inc,
    int v_size,
    T P,
    RNG<T> *rng,
    int BL,
    T res,
    bool sto_round,
    int &noz,
    int start_pos = 0) {

  PRAGMA_SIMD
  for (int k = 0; k < BL; k++) {
    counts[k] = 0;
  }

  int j_v = 0;
  for (int j = 0; j < v_size; j++) {

    T v_value = v[j_v];
    j_v += v_inc;

    T PP = getDiscretizedValue<T>((T)fabsf(v_value) * P, res, sto_round, *rng);
    if (PP == (T)0.0) {
      noz++;
      continue;
    }

    // Calculate stream length based on probability
    // Clamp PP to [0, 1] range to prevent overflow
    T PP_clamped = MIN(PP, (T)1.0);

    // Stochastic rounding: handle fractional part probabilistically
    T stream_length_float = BL * PP_clamped;
    int stream_length_base = (int)stream_length_float;
    T fractional_part = stream_length_float - stream_length_base;

    // Use fractional part as probability to round up
    int stream_length = stream_length_base;
    if (fractional_part > 0 && rng->sampleUniform() < fractional_part) {
      stream_length++;
    }

    // Safety check: ensure stream_length doesn't exceed BL
    stream_length = MIN(stream_length, BL);

    // if (debug_mode && j == 0) { // Only print for first element to avoid spam
    //   std::cout << "[DEBUG] generateStreamCounts: j=" << j << ", v_value=" << v_value
    //             << ", PP=" << PP << ", PP_clamped=" << PP_clamped
    //             << ", stream_length=" << stream_length
    //             << ", start_pos=" << start_pos << std::endl;
    // }
    if (stream_length <= 0) {
      noz++;
      continue;
    }

    int jplus1_signed = (v_value > (T)0.0) ? j + 1 : -(j + 1);

    // Generate continuous pulse stream with wrapping
    for (int k = 0; k < stream_length; k++) {
      int pos = (start_pos + k) % BL;
      indices[pos][counts[pos]++] = jplus1_signed;
    }
  }
}

// makeCounts
template <typename T>
int SparseBitLineMaker<T>::makeCounts(
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

  // Set debug mode for PulsedUpdateMetaParameter

  if (up.update_bl_management || up.update_management) {

    T x_abs_max = Find_Absolute_Max<T>(x_in, x_size_, x_inc);
    T d_abs_max = Find_Absolute_Max<T>(d_in, d_size_, d_inc);

    // if (debug_mode_) {
    //   std::cout << "[DEBUG] makeCounts: Using updateManagement path" << std::endl;
    //   std::cout << "[DEBUG] makeCounts: x_abs_max=" << x_abs_max << ", d_abs_max=" << d_abs_max << std::endl;
    //   std::cout << "[DEBUG] makeCounts: lr=" << lr << ", dw_min=" << dw_min << std::endl;
    // }

    up.performUpdateManagement(BL, A, B, up.desired_BL, x_abs_max, d_abs_max, lr, dw_min);
  } else {
    // if (debug_mode_) {
    //   std::cout << "[DEBUG] makeCounts: Using calculateBlAB path" << std::endl;
    //   std::cout << "[DEBUG] makeCounts: lr=" << lr << ", dw_min=" << dw_min << std::endl;
    // }

    up.calculateBlAB(BL, A, B, lr, dw_min);
  }

  // if (debug_mode_) {
  //   std::cout << "[DEBUG] makeCounts: Final values - A=" << A << ", B=" << B << ", BL=" << BL << std::endl;
  // }

  if (BL == 0) {
    return 0;
  }
  if (MAX(BL, up.desired_BL) > max_BL_) {
    initialize(x_size_, d_size_, MAX(BL, up.desired_BL));
  }

  switch (up.pulse_type) {

  case PulseType::Stochastic:
    // x counts
    generateCountsPN<T>(
        x_counts_p_, x_counts_n_, x_indices_p_, x_indices_n_, x_in, x_inc, x_size_, B, rng, BL,
        up.res, up.sto_round, x_noz);
    // d counts // can be compressed
    generateCounts<T>(
        d_counts_, d_indices_, d_in, d_inc, d_size_, A, rng, BL, up.res, up.sto_round, d_noz);

    n_indices_used_ = true;
    break;

  case PulseType::StochasticCompressed:

    // x counts
    generateCounts<T>(
        x_counts_p_, x_indices_p_, x_in, x_inc, x_size_, B, rng, BL, up.res, up.sto_round, x_noz);

    // d counts
    generateCounts<T>(
        d_counts_, d_indices_, d_in, d_inc, d_size_, A, rng, BL, up.res, up.sto_round, d_noz);

    n_indices_used_ = false;
    break;

  case PulseType::StochasticStream: {

    // if (debug_mode_) {
    //   std::cout << "[DEBUG] StochasticStream: A=" << A << ", B=" << B << ", BL=" << BL << std::endl;
    // }

    // x counts - start from position 0
    generateStreamCounts<T>(
        x_counts_p_, x_indices_p_, x_in, x_inc, x_size_, B, rng, BL, up.res, up.sto_round, x_noz, 0);

    // d counts - start from random position
    int random_start = (int)(rng->sampleUniform() * BL);
    // if (debug_mode_) {
    //   std::cout << "[DEBUG] StochasticStream: d random_start=" << random_start << std::endl;
    // }
    generateStreamCounts<T>(
        d_counts_, d_indices_, d_in, d_inc, d_size_, A, rng, BL, up.res, up.sto_round, d_noz, random_start);

    n_indices_used_ = false;
    break;
  }

  case PulseType::HalfselectedStochastic:
    // Similar to Stochastic but with HS tracking capability
    // x counts
    generateCountsPN<T>(
        x_counts_p_, x_counts_n_, x_indices_p_, x_indices_n_, x_in, x_inc, x_size_, B, rng, BL,
        up.res, up.sto_round, x_noz);
    // d counts
    generateCounts<T>(
        d_counts_, d_indices_, d_in, d_inc, d_size_, A, rng, BL, up.res, up.sto_round, d_noz);

    n_indices_used_ = true;
    break;

  case PulseType::HalfselectedStochasticStream: {
    // Similar to StochasticStream but with HS tracking capability
    // x counts - start from position 0
    generateStreamCounts<T>(
        x_counts_p_, x_indices_p_, x_in, x_inc, x_size_, B, rng, BL, up.res, up.sto_round, x_noz, 0);

    // d counts - start from random position
    int random_start = (int)(rng->sampleUniform() * BL);
    generateStreamCounts<T>(
        d_counts_, d_indices_, d_in, d_inc, d_size_, A, rng, BL, up.res, up.sto_round, d_noz, random_start);

    n_indices_used_ = false;
    break;
  }

  default:
    RPU_FATAL("PulseType not supported");
  }

  return BL; // this is the actual
}

template <typename T> bool SparseBitLineMaker<T>::supports(RPU::PulseType pulse_type) const {
  return PulseType::StochasticCompressed == pulse_type || PulseType::Stochastic == pulse_type ||
         PulseType::StochasticStream == pulse_type || PulseType::HalfselectedStochastic == pulse_type ||
         PulseType::HalfselectedStochasticStream == pulse_type;
}

template <typename T> void SparseBitLineMaker<T>::printCounts(int BL) const {

  if (max_BL_ < BL) {
    RPU_FATAL("Containter not yet allocated or BL too big");
  }

  // std::cout << "\n\nX_counts_p:\n";
  // for (int k = 0; k < BL; k++) {
  //   std::cout << "\nk=" << k << std::endl;
  //   for (int i = 0; i < x_counts_p_[k]; i++) {
  //     std::cout << x_indices_p_[k][i] << ", ";
  //   }
  // }
  // std::cout << "\n\nD_counts_p:\n";
  // for (int k = 0; k < BL; k++) {
  //   std::cout << "\nk=" << k << std::endl;
  //   for (int i = 0; i < d_counts_[k]; i++) {
  //     std::cout << d_indices_[k][i] << ", ";
  //   }
  // }
  // if (n_indices_used_) {
  //   std::cout << "\n\nX_counts_n:\n";
  //   for (int k = 0; k < BL; k++) {
  //     std::cout << "\nk=" << k << std::endl;
  //     for (int i = 0; i < x_counts_n_[k]; i++) {
  //       std::cout << x_indices_n_[k][i] << ", ";
  //     }
  //   }
  // }
}

// public access to results. Need to call makeCounts before.
template <typename T>
bool SparseBitLineMaker<T>::getCountsAndIndices(
    int *&x_counts_p,
    int *&x_counts_n,
    int *&d_counts,
    int **&x_indices_p,
    int **&x_indices_n,
    int **&d_indices) {
  if (!max_BL_) {
    RPU_FATAL("Containers not allocated!");
  }
  x_counts_p = x_counts_p_;
  x_counts_n = x_counts_n_;
  d_counts = d_counts_;

  x_indices_p = x_indices_p_;
  x_indices_n = x_indices_n_;
  d_indices = d_indices_;

  return n_indices_used_;
}

template class SparseBitLineMaker<float>;
#ifdef RPU_USE_DOUBLE
template class SparseBitLineMaker<double>;
#endif
#ifdef RPU_USE_FP16
template class SparseBitLineMaker<half_t>;
#endif

} // namespace RPU
