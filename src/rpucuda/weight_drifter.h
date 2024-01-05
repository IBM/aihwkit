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
#include <memory>

namespace RPU {

template <typename T> struct DriftParameter {

  T nu = (T)0.0;      // zero ! Set to non-zero to cause the drifter to initialize
  T nu_dtod = (T)0.0; // in percent of nu

  // C-to-C of nu. However, this probably makes not too much
  // physical sense, because each time it will compute the actual
  // weight independent of the last nu drift value by just looking
  // at a stored w0 (at t_init), and computing
  // w0*(delat_t/t0)^(-nu_actual) Maybe more realistic to use w
  // "read noise" for modelling variations (see below, w_read_std)
  T nu_std = (T)0.0;

  T wg_ratio = (T)1; // (wmax-wmin)/(gmax-gmin)
  T g_offset = (T)0; // gmin
  T w_offset = (T)0; // w(gmin), i.e. to what value gmin is mapped to in w-space

  T nu_k = (T)0.0;  // vary of nu with W  nu(R) = nu0 - k log(G/G0)
  T logG0 = (T)0.0; // default is actually 0.5muS, see https://ieeexplore.ieee.org/document/8753712

  T t0 = (T)1.0; // time between write and first read. [ms]

  T reset_tol = (T)1e-7; // should be very small (smaller than
                         // minimal weight update), as weight will
                         // get overwritten anyway if difference
                         // smaller than this. Just to detect any
                         // weight update. CAUTION: will not work
                         // with DECAY!

  T w_read_std = (T)0.0; // additional weight read noise

  bool _is_simple_drift = false;

  inline bool isSimpleDrift() const { return _is_simple_drift; }
  inline void setSimpleDrift() {
    nu_dtod = (T)0.0;
    _is_simple_drift = true;
  }
  inline void unsetSimpleDrift() { _is_simple_drift = false; }

  void print() const {
    std::stringstream ss;
    printToStream(ss);
    std::cout << ss.str();
  };

  void printToStream(std::stringstream &ss) const {
    ss << "\t nu:\t\t\t" << nu << "\t(dtod=";
    if (_is_simple_drift) {
      ss << "NA";
    } else {
      ss << nu_dtod;
    }
    ss << ", ctoc=" << nu_std << ")" << std::endl;
    ss << "\t t0 [ms]:\t\t" << t0 << std::endl;

    if (w_read_std > (T)0.0) {
      ss << "\t w_read_std:\t\t" << w_read_std << std::endl;
    }

    if (wg_ratio != (T)1 || g_offset != (T)0 || w_offset != (T)0) {
      ss << "\t wg ratio / offsets:\t" << wg_ratio << " / " << w_offset << " / " << g_offset
         << std::endl;
    }
    if (nu_k != (T)0.0) {
      ss << "\t nu_k / log G0:\t" << nu_k << " / " << logG0 << std::endl;
    }

    if (fabsf(reset_tol) > (float)1e-6) {
      ss << "\t reset_tol:\t\t" << reset_tol << std::endl;
    }
  }

  inline bool usesRandom() { return (w_read_std > (T)0.0 || nu_std > (T)0.0); };
};

/***********************************************************************************/

template <typename T> class WeightDrifter {

public:
  explicit WeightDrifter(int size);
  explicit WeightDrifter(int size, const DriftParameter<T> &par); // forces SimpleDrift
  explicit WeightDrifter(int size, const DriftParameter<T> &par, RealWorldRNG<T> *rng);
  WeightDrifter(){};
  virtual ~WeightDrifter() = default;

  WeightDrifter(const WeightDrifter<T> &) = default;
  WeightDrifter<T> &operator=(const WeightDrifter<T> &) = default;
  WeightDrifter(WeightDrifter<T> &&) = default;
  WeightDrifter<T> &operator=(WeightDrifter<T> &&) = default;

  void populate(const DriftParameter<T> &par, RealWorldRNG<T> *rng);

  /*Applies the weight drift to all unchanged weight elements
    (judged by reset_tol) and resets those that have
    changed. time_since_last_call is the time between the calls,
    typipcally the time to process a mini-batch for the
    network. Units are milliseconds */
  void apply(T *weights, T time_since_last_call, RNG<T> &rng);

  void saturate(T *weights, const T *min_bounds, const T *max_bounds);

  inline bool isActive() const { return active_; };
  inline const T *getNu() const { return nu_.size() != (size_t)size_ ? nullptr : nu_.data(); };

  void getNu(T *dst) const;
  void setNu(const T *src);

  inline const DriftParameter<T> &getPar() const { return par_; };
  inline int getSize() const { return size_; };

  void dumpExtra(RPU::state_t &extra, const std::string prefix);
  void loadExtra(const RPU::state_t &extra, const std::string prefix, bool strict);

protected:
  int size_ = 0;
  bool active_ = false;
  T current_t_ = 0.0;

  DriftParameter<T> par_;

  std::vector<T> previous_weights_;
  std::vector<T> w0_;
  std::vector<T> t_;
  std::vector<T> nu_;

private:
  void initialize(const T *weights);
};

} // namespace RPU
