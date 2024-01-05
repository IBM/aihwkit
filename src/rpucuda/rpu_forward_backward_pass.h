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

template <typename T> class MVParameter {

public:
  std::vector<T> out_noise_values;
  std::vector<T> v_offset;
  std::vector<T> w_asymmetry;
  std::vector<T> out_nonlinearity;
  T out_nonlinearity_factor = 0.0;
};

template <typename T> class FBParameter {

public:
  MVParameter<T> fwd;
  MVParameter<T> bwd;
};

template <typename T> class ForwardBackwardPass {

public:
  explicit ForwardBackwardPass(int x_size, int d_size) : x_size_(x_size), d_size_(d_size){};
  ForwardBackwardPass(){};
  virtual ~ForwardBackwardPass(){};

  ForwardBackwardPass(const ForwardBackwardPass<T> &) = default;
  ForwardBackwardPass<T> &operator=(const ForwardBackwardPass<T> &) = default;
  ForwardBackwardPass(ForwardBackwardPass<T> &&) = default;
  ForwardBackwardPass<T> &operator=(ForwardBackwardPass<T> &&) = default;

  friend void swap(ForwardBackwardPass<T> &a, ForwardBackwardPass<T> &b) noexcept {
    using std::swap;
    swap(a.x_size_, b.x_size_);
    swap(a.d_size_, b.d_size_);
    swap(a.fb_pars_, b.fb_pars_);
  }

  const FBParameter<T> &getFBParameter() const { return fb_pars_; };
  void setFBParameter(FBParameter<T> fb_pars) { fb_pars_ = fb_pars; };

  virtual void forwardVector(
      T **weights,
      const T *x_input,
      const int x_inc,
      T *d_output,
      const int d_inc,
      const T alpha,
      const bool is_test);

  virtual void backwardVector(
      T **weights, const T *d_input, const int d_inc, T *x_output, const int x_inc, const T alpha);

  inline void gemv(
      T **weights,
      const T *in_values,
      const int in_size,
      const int in_inc,
      T *out_vales,
      const int out_size,
      const int out_inc,
      const T alpha,
      const T beta,
      const bool transpose);

  virtual void dumpExtra(RPU::state_t &extra, const std::string prefix);
  virtual void loadExtra(const RPU::state_t &extra, const std::string prefix, bool strict);

protected:
  int x_size_ = 0;
  int d_size_ = 0;

  // parameter storage
  FBParameter<T> fb_pars_;
};

/* RPU stochastic version of the forward pass with noise and management ntechniques*/
template <typename T> class ForwardBackwardPassIOManaged : public ForwardBackwardPass<T> {

public:
  explicit ForwardBackwardPassIOManaged(int x_size, int d_size, std::shared_ptr<RNG<T>> rng);
  ForwardBackwardPassIOManaged(){};

  ~ForwardBackwardPassIOManaged();
  ForwardBackwardPassIOManaged(const ForwardBackwardPassIOManaged<T> &);
  ForwardBackwardPassIOManaged<T> &operator=(const ForwardBackwardPassIOManaged<T> &);
  ForwardBackwardPassIOManaged(ForwardBackwardPassIOManaged<T> &&) noexcept;
  ForwardBackwardPassIOManaged<T> &operator=(ForwardBackwardPassIOManaged<T> &&) noexcept;

  friend void
  swap(ForwardBackwardPassIOManaged<T> &a, ForwardBackwardPassIOManaged<T> &b) noexcept {
    using std::swap;
    swap(static_cast<ForwardBackwardPass<T> &>(a), static_cast<ForwardBackwardPass<T> &>(b));
    swap(a.f_io_, b.f_io_);
    swap(a.b_io_, b.b_io_);
    swap(a.checked_implemented_, b.checked_implemented_);
    swap(a.rng_, b.rng_);

    // others are tmps so far
  }

  void dumpExtra(RPU::state_t &extra, const std::string prefix) override;
  void loadExtra(const RPU::state_t &extra, const std::string prefix, bool strict) override;

  void forwardVector(
      T **weights,
      const T *x_input,
      const int x_inc,
      T *d_output,
      const int d_inc,
      const T alpha,
      const bool is_test) override;

  void backwardVector(
      T **weights, const T *d_input, const int d_inc, T *x_output, const int x_inc, const T alpha)
      override;

  void populateFBParameter(const IOMetaParameter<T> &f_io_, const IOMetaParameter<T> &b_io_);

  inline bool computeAnalogMV(
      T **weights,
      const T *org_in_values,
      const int in_size,
      const int in_inc,
      T *out_values,
      const int out_size,
      const int out_inc,
      const T scale,
      const bool scaling,
      const MVParameter<T> &mv_pars,
      const IOMetaParameter<T> &io,
      const bool transposed,
      const bool is_test);

protected:
  inline void applyOutputWeightNoise(
      T **weights,
      T *out_values,
      const int out_size,
      const int out_inc,
      const T *in_values,
      const int in_size,
      const IOMetaParameter<T> &io,
      const bool transposed);

  inline void applyIrDrop(
      T **weights,
      T *out_values,
      const int out_size,
      const int out_inc,
      const T *in_values,
      const T *current_values,
      const int in_size,
      const IOMetaParameter<T> &io,
      const bool transposed);

  inline void applyVoltageOffsets(
      T *out_values,
      const int out_size,
      const int out_inc,
      const T *in_values,
      const int in_size,
      const MVParameter<T> &mv_pars,
      const IOMetaParameter<T> &io);

  inline void applyNonIdealities(
      T **weights,
      T *out_values,
      int out_size,
      const int out_inc,
      const T *in_values,
      const int in_size,
      const MVParameter<T> &mv_pars,
      const IOMetaParameter<T> &io,
      const bool transposed);

  inline const T *computeTotalCurrent(
      T **weights,
      const int out_size,
      const T *in_values,
      const int in_size,
      const bool transposed);

  inline T *prepareInput(
      const T *in_values,
      const int in_size,
      const int in_inc,
      const T scale,
      const bool scaling,
      const IOMetaParameter<T> &io);

  inline bool finalizeOutput(
      T *out_values,
      const int out_size,
      const int out_inc,
      const MVParameter<T> &mv_pars,
      const IOMetaParameter<T> &io);

  inline void computeAnalogMVSinglePass(
      T **weights,
      const T *in_values,
      const int in_size,
      const int in_inc,
      T *out_values,
      const int out_size,
      const int out_inc,
      const T alpha,
      const T beta,
      const MVParameter<T> &mv_pars,
      const IOMetaParameter<T> &io,
      const bool transposed);

private:
  inline void ensureImplemented();

  // tmp for non-ideal computations
  std::vector<T> tmp_in_values_;
  std::vector<T> tmp_out_values_;
  std::vector<T> tmp_c_values_;

  // buffers are not to be used in the non-ideal computations
  std::vector<T> current_buffer_values_;
  std::vector<T> in_buffer_values_;
  std::vector<T> out_buffer_values_;
  std::vector<T> pos_neg_buffer_values_;

  T **neg_weights_ = nullptr;

  T aux_nm_value_ = -1.0;
  IOMetaParameter<T> f_io_;
  IOMetaParameter<T> b_io_;
  bool checked_implemented_ = false;
  std::shared_ptr<RNG<T>> rng_ = nullptr;
};

} // namespace RPU
