/**
 * (C) Copyright 2020, 2021 IBM. All Rights Reserved.
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
  virtual ~ForwardBackwardPass(){};

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

protected:
  void applyOutputWeightNoise(
      T **weights,
      T *out_values,
      const int out_size,
      const int out_inc,
      const T *in_values,
      const int in_size,
      IOMetaParameter<T> &io,
      bool transposed);

  void applyIrDrop(
      T **weights,
      T *out_values,
      int out_size,
      const int out_inc,
      const T *in_values,
      const int in_size,
      IOMetaParameter<T> &io,
      bool transposed);

private:
  void ensureImplemented();

  std::vector<T> tmp_d_values_;
  std::vector<T> tmp_x_values_;

  std::vector<T> tmp_in_values_;
  std::vector<T> tmp_out_values_;
  std::vector<T> tmp_c_values_;

  T aux_nm_value_ = -1.0;
  IOMetaParameter<T> f_io_;
  IOMetaParameter<T> b_io_;
  bool checked_implemented_ = false;
  std::shared_ptr<RNG<T>> rng_ = nullptr;
};

} // namespace RPU
