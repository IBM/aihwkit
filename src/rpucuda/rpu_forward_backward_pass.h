/**
 * (C) Copyright 2020 IBM. All Rights Reserved.
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

template <typename T> class ForwardBackwardPass {

public:
  explicit ForwardBackwardPass(int x_size, int d_size) : x_size_(x_size), d_size_(d_size){};
  ForwardBackwardPass(){};

  friend void swap(ForwardBackwardPass<T> &a, ForwardBackwardPass<T> &b) noexcept {
    using std::swap;
    swap(a.x_size_, b.x_size_);
    swap(a.d_size_, b.d_size_);
  }

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

protected:
  int x_size_ = 0;
  int d_size_ = 0;
};

/* RPU stochastic version of the forward pass with noise and management ntechniques*/
template <typename T> class ForwardBackwardPassIOManaged : public ForwardBackwardPass<T> {

public:
  explicit ForwardBackwardPassIOManaged(int x_size, int d_size, std::shared_ptr<RNG<T>> rng);
  ForwardBackwardPassIOManaged(){};
  ~ForwardBackwardPassIOManaged();

  ForwardBackwardPassIOManaged(const ForwardBackwardPassIOManaged<T> &);
  ForwardBackwardPassIOManaged<T> &operator=(const ForwardBackwardPassIOManaged<T> &);
  ForwardBackwardPassIOManaged(ForwardBackwardPassIOManaged<T> &&);
  ForwardBackwardPassIOManaged<T> &operator=(ForwardBackwardPassIOManaged<T> &&);

  friend void
  swap(ForwardBackwardPassIOManaged<T> &a, ForwardBackwardPassIOManaged<T> &b) noexcept {
    using std::swap;
    swap(static_cast<ForwardBackwardPass<T> &>(a), static_cast<ForwardBackwardPass<T> &>(b));
    swap(a.containers_allocated_, b.containers_allocated_);
    swap(a.f_io_, b.f_io_);
    swap(a.b_io_, b.b_io_);
    swap(a.tmp_x_values_, b.tmp_x_values_);
    swap(a.tmp_d_values_, b.tmp_d_values_);
  }

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

  void setIOPar(const IOMetaParameter<T> &f_io_, const IOMetaParameter<T> &b_io_);

private:
  void ensureImplemented();

  void freeContainers();
  void allocateContainers();

  T *tmp_x_values_ = nullptr;
  T *tmp_d_values_ = nullptr;

  bool checked_implemented_ = false;
  IOMetaParameter<T> f_io_;
  IOMetaParameter<T> b_io_;
  bool containers_allocated_ = false;
  std::shared_ptr<RNG<T>> rng_ = nullptr;
};

} // namespace RPU
