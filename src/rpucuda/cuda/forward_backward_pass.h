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

#include "cuda_util.h"
#include "io_manager.h"

namespace RPU {

namespace detail {

template <typename T>
void forwardMatrix(
    const CudaContext *context,
    const T *dev_weights,
    const T *X_input,
    const int x_size,
    const bool x_trans,
    T *D_output,
    const int d_size,
    const bool d_trans,
    const int m_batch,
    const T alpha) {
  if (m_batch == 1) {

    RPU::math::gemv<T>(
        context, false, d_size, x_size, alpha, dev_weights, d_size, X_input,
        1, // x_inc
        (T)0.0, D_output,
        1 // d_inc
    );
  } else {

    if (d_trans) {
      RPU::math::gemm<T>(
          context, // is col based !!
          !x_trans, true, m_batch, d_size, x_size, alpha, X_input, (x_trans) ? m_batch : x_size,
          dev_weights, d_size, (T)0.0, D_output, m_batch);
    } else {
      RPU::math::gemm<T>(
          context, false, x_trans, d_size,
          m_batch, // M
          x_size,  // K
          alpha, dev_weights,
          d_size, // col major
          X_input, (x_trans) ? m_batch : x_size, (T)0.0, D_output, d_size);
    }
  }
};

template <typename T>
void backwardMatrix(
    const CudaContext *context,
    const T *dev_weights,
    const T *D_input,
    const int d_size,
    const bool d_trans,
    T *X_output,
    const int x_size,
    const bool x_trans,
    const int m_batch,
    const T alpha) {

  if (m_batch == 1) {
    // backward
    RPU::math::gemv<T>(
        context, true, d_size, x_size, alpha, dev_weights, d_size, D_input, 1, (T)0.0, X_output, 1);
  } else {
    if (x_trans) {
      RPU::math::gemm<T>(
          context, !d_trans, false, m_batch, x_size, d_size, alpha, D_input,
          (d_trans) ? m_batch : d_size, dev_weights, d_size, (T)0.0, X_output, m_batch);
    } else {
      RPU::math::gemm<T>(
          context, true, d_trans,
          x_size,  // N
          m_batch, // M
          d_size,  // K
          alpha, dev_weights, d_size, D_input, (d_trans) ? m_batch : d_size, (T)0.0, X_output,
          x_size);
    }
  }
};

/* Implementation of forward pass with IO manager*/
template <typename T, typename InputIteratorT, typename OutputIteratorT>
void forwardMatrixIteratorIOManaged(
    CudaContext *context,
    T *dev_weights,
    InputIteratorT X_input,
    int x_size,
    bool x_trans,
    OutputIteratorT D_output,
    int d_size,
    bool d_trans,
    int m_batch,
    T alpha,
    InputOutputManager<T> &f_iom,
    const IOMetaParameter<T> &f_io_pars,
    const bool is_test,
    std::shared_ptr<CudaArray<T>> in_buffer = nullptr,
    std::shared_ptr<CudaArray<T>> out_buffer = nullptr) {

  if (f_io_pars.is_perfect) {
    if (std::is_same<InputIteratorT, const T *>::value &&
        std::is_same<OutputIteratorT, T *>::value) {
      // perfect short-cut outside. This avoids the buffer copy if possible (only FC)

      forwardMatrix(
          context, dev_weights, RPU::math::fakeCastConst<T, InputIteratorT>(X_input), x_size,
          x_trans, RPU::math::fakeCast<T, OutputIteratorT>(D_output), d_size, d_trans, m_batch,
          (T)alpha * f_io_pars.out_scale);
      return;
    }
  }

  // init IO
  f_iom.setSharedBuffer(m_batch, in_buffer, out_buffer);

  f_iom.initWithInput(X_input, f_io_pars, m_batch, x_trans, alpha, is_test);

  T *X_temp = f_iom.getInBuffer();
  T *D_temp = f_iom.getOutBuffer();

  bool bound_test_passed = false;
  while (bound_test_passed == false) {

    // input management
    int current_m_batch = f_iom.applyToInput(X_input);

    forwardMatrix(
        context, dev_weights, X_temp, x_size, x_trans, D_temp, d_size, d_trans, current_m_batch,
        f_io_pars.is_perfect ? alpha * f_io_pars.out_scale : (T)1.0);
    // output management
    bound_test_passed = f_iom.applyToOutput(D_output, dev_weights, d_trans);
  }
}

/* Implementation of backward pass with IO manager
   Note that bound management is not supported */
template <typename T, typename InputIteratorT, typename OutputIteratorT>
void backwardMatrixIteratorIOManaged(
    CudaContext *context,
    T *dev_weights,
    InputIteratorT D_input,
    int d_size,
    bool d_trans,
    OutputIteratorT X_output,
    int x_size,
    bool x_trans,
    int m_batch,
    T alpha,
    InputOutputManager<T> &b_iom,
    const IOMetaParameter<T> &b_io_pars,
    std::shared_ptr<CudaArray<T>> in_buffer = nullptr,
    std::shared_ptr<CudaArray<T>> out_buffer = nullptr) {

  if (b_io_pars.is_perfect) {
    if (std::is_same<InputIteratorT, const T *>::value &&
        std::is_same<OutputIteratorT, T *>::value) {
      // perfect short-cut outside. This avoids the buffer copy if possible (only FC)

      backwardMatrix(
          context, dev_weights, RPU::math::fakeCastConst<T, InputIteratorT>(D_input), d_size,
          d_trans, RPU::math::fakeCast<T, OutputIteratorT>(X_output), x_size, x_trans, m_batch,
          alpha * b_io_pars.out_scale);

      return;
    }
  } else {
    // input management
    if (b_io_pars.bound_management != BoundManagementType::None) {
      RPU_FATAL("Bound management is not supported for backward pass.");
    }
  }

  b_iom.setSharedBuffer(m_batch, in_buffer, out_buffer);
  b_iom.initWithInput(D_input, b_io_pars, m_batch, d_trans, alpha);
  b_iom.applyToInput(D_input);

  T *D_temp = b_iom.getInBuffer();
  T *X_temp = b_iom.getOutBuffer();

  backwardMatrix(
      context, dev_weights, D_temp, d_size, d_trans, X_temp, x_size, x_trans, m_batch,
      // alpha is taken care of in io_manager (because it is done in digital)
      b_io_pars.is_perfect ? alpha * b_io_pars.out_scale : (T)1.0);

  // output management
  b_iom.applyToOutput(X_output, dev_weights, x_trans);
}

} // namespace detail

} // namespace RPU
