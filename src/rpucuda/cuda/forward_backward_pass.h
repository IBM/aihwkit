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

#include "cuda_util.h"
#include "io_manager.h"

namespace RPU {

namespace detail {

template <typename T>
void forwardMatrix(
    const CudaContextPtr context,
    const T *dev_weights,
    const T *X_input,
    const int x_size,
    const bool x_trans,
    T *D_output,
    const int d_size,
    const bool d_trans,
    const int m_batch,
    const T alpha,
    const T beta = 0.0) {
  if (m_batch == 1) {
    RPU::math::gemv<T>(
        context, false, d_size, x_size, alpha, dev_weights, d_size, X_input,
        1, // x_inc
        beta, D_output,
        1 // d_inc
    );
  } else {

    if (d_trans) {
      RPU::math::gemm<T>(
          context, // is col based !!
          !x_trans, true, m_batch, d_size, x_size, alpha, X_input, (x_trans) ? m_batch : x_size,
          dev_weights, d_size, beta, D_output, m_batch);
    } else {
      RPU::math::gemm<T>(
          context, false, x_trans, d_size,
          m_batch, // M
          x_size,  // K
          alpha, dev_weights,
          d_size, // col major
          X_input, (x_trans) ? m_batch : x_size, beta, D_output, d_size);
    }
  }
};

template <typename T>
void backwardMatrix(
    const CudaContextPtr context,
    const T *dev_weights,
    const T *D_input,
    const int d_size,
    const bool d_trans,
    T *X_output,
    const int x_size,
    const bool x_trans,
    const int m_batch,
    const T alpha,
    const T beta = 0.0) {

  if (m_batch == 1) {
    // backward
    RPU::math::gemv<T>(
        context, true, d_size, x_size, alpha, dev_weights, d_size, D_input, 1, beta, X_output, 1);
  } else {
    if (x_trans) {
      RPU::math::gemm<T>(
          context, !d_trans, false, m_batch, x_size, d_size, alpha, D_input,
          (d_trans) ? m_batch : d_size, dev_weights, d_size, beta, X_output, m_batch);
    } else {
      RPU::math::gemm<T>(
          context, true, d_trans,
          x_size,  // N
          m_batch, // M
          d_size,  // K
          alpha, dev_weights, d_size, D_input, (d_trans) ? m_batch : d_size, beta, X_output,
          x_size);
    }
  }
};

} // namespace detail

template <typename T> class MVParameterCuda {
public:
  MVParameterCuda(){};
  CudaArray<T> out_noise_values;
  CudaArray<T> v_offset;
  CudaArray<T> w_asymmetry;
  CudaArray<T> out_nonlinearity;
  T out_nonlinearity_factor = 0.0;

  friend void swap(MVParameterCuda<T> &a, MVParameterCuda<T> &b) noexcept {
    using std::swap;
    swap(a.out_noise_values, b.out_noise_values);
    swap(a.v_offset, b.v_offset);
    swap(a.w_asymmetry, b.w_asymmetry);
    swap(a.out_nonlinearity_factor, b.out_nonlinearity_factor);
    swap(a.out_nonlinearity, b.out_nonlinearity);
  }
};

template <typename T> class FBParameterCuda {
public:
  FBParameterCuda(){};
  MVParameterCuda<T> fwd;
  MVParameterCuda<T> bwd;

  friend void swap(FBParameterCuda<T> &a, FBParameterCuda<T> &b) noexcept {
    using std::swap;
    swap(a.fwd, b.fwd);
    swap(a.bwd, b.bwd);
  }
};

template <typename T> class ForwardBackwardPassIOManagedCuda {

public:
  explicit ForwardBackwardPassIOManagedCuda(CudaContextPtr context, int x_size, int d_size)
      : x_size_(x_size), d_size_(d_size), context_(context){};
  ForwardBackwardPassIOManagedCuda(){};

  ~ForwardBackwardPassIOManagedCuda() = default;
  ForwardBackwardPassIOManagedCuda(const ForwardBackwardPassIOManagedCuda<T> &);
  ForwardBackwardPassIOManagedCuda<T> &operator=(const ForwardBackwardPassIOManagedCuda<T> &);
  ForwardBackwardPassIOManagedCuda(ForwardBackwardPassIOManagedCuda<T> &&);
  ForwardBackwardPassIOManagedCuda<T> &operator=(ForwardBackwardPassIOManagedCuda<T> &&);

  friend void
  swap(ForwardBackwardPassIOManagedCuda<T> &a, ForwardBackwardPassIOManagedCuda<T> &b) noexcept {
    using std::swap;
    swap(a.x_size_, b.x_size_);
    swap(a.d_size_, b.d_size_);
    swap(a.context_, b.context_);
    swap(a.fb_pars_, b.fb_pars_);
  }
  void dumpExtra(RPU::state_t &extra, const std::string prefix);
  void loadExtra(const RPU::state_t &extra, const std::string prefix, bool strict);

  void populateFrom(const FBParameter<T> &fb_pars_host);
  bool checkFlexibleInSize(const IOMetaParameter<T> &io) {
    // TODO: need to check whether that is all correct...
    if (io.w_noise > (T)0.0 || io.ir_drop > (T)0.0 || io.w_read_asymmetry_dtod > (T)0.0) {
      return false;
    } else {
      return true;
    };
  }

  /* Implementation of forward pass with IO manager*/
  template <typename InputIteratorT, typename OutputIteratorT>
  void forwardMatrixIterator(
      T *dev_weights,
      InputIteratorT X_input,
      int in_size,
      bool x_trans,
      OutputIteratorT D_output,
      int out_size,
      bool d_trans,
      int m_batch,
      T alpha,
      InputOutputManager<T> &f_iom,
      const IOMetaParameter<T> &f_io,
      const bool is_test) {

    if (in_size != this->x_size_ && !checkFlexibleInSize(f_io) || out_size != this->d_size_) {
      RPU_FATAL("x/d size mismatch!");
    }

    if (f_io.isPerfect()) {
      if (std::is_same<InputIteratorT, const T *>::value &&
          std::is_same<OutputIteratorT, T *>::value) {
        // perfect short-cut outside. This avoids the buffer copy if possible (only FC)
        RPU::detail::forwardMatrix(
            f_iom.getContext(), dev_weights, RPU::math::fakeCastConst<T, InputIteratorT>(X_input),
            in_size, x_trans, RPU::math::fakeCast<T, OutputIteratorT>(D_output), out_size, d_trans,
            m_batch, (T)alpha * f_io.out_scale);
        return;
      }
    }

    // init IO
    f_iom.initWithInput(X_input, f_io, in_size, m_batch, x_trans, alpha, is_test);

    bool bound_test_passed = false;
    while (bound_test_passed == false) {
      // input management
      f_iom.applyToInput(X_input);
      bound_test_passed =
          computeAnalogMV(D_output, d_trans, dev_weights, f_iom, fb_pars_.fwd, false);
    }
    f_iom.releaseBuffer();
  }

  template <typename InputIteratorT, typename OutputIteratorT>
  void backwardMatrixIterator(
      T *dev_weights,
      InputIteratorT D_input,
      int in_size,
      bool d_trans,
      OutputIteratorT X_output,
      int out_size,
      bool x_trans,
      int m_batch,
      T alpha,
      InputOutputManager<T> &b_iom,
      const IOMetaParameter<T> &b_io) {

    if (in_size != this->d_size_ && !checkFlexibleInSize(b_io) || out_size != this->x_size_) {
      RPU_FATAL("x/d size mismatch!");
    }

    if (b_io.isPerfect()) {
      if (std::is_same<InputIteratorT, const T *>::value &&
          std::is_same<OutputIteratorT, T *>::value) {
        // perfect short-cut outside. This avoids the buffer copy if possible (only FC)

        RPU::detail::backwardMatrix(
            b_iom.getContext(), dev_weights, RPU::math::fakeCastConst<T, InputIteratorT>(D_input),
            in_size, d_trans, RPU::math::fakeCast<T, OutputIteratorT>(X_output), out_size, x_trans,
            m_batch, alpha * b_io.out_scale);
        return;
      }
    } else {
      // input management
      if (b_io.bound_management != BoundManagementType::None) {
        RPU_FATAL("Bound management is not supported for backward pass.");
      }
    }

    b_iom.initWithInput(D_input, b_io, in_size, m_batch, d_trans, alpha);
    b_iom.applyToInput(D_input);
    computeAnalogMV(X_output, x_trans, dev_weights, b_iom, fb_pars_.bwd, true);
    b_iom.releaseBuffer();
  }

protected:
  // WARNING: uses and overwrites inBuffer
  void computeAnalogMVSinglePass(
      T *dev_weights,
      InputOutputManager<T> &iom,
      const MVParameterCuda<T> &mv_pars,
      const bool out_trans,
      const bool transposed);

  template <typename OutputIteratorT>
  bool computeAnalogMV(
      OutputIteratorT out_values,
      const bool out_trans,
      T *dev_weights,
      InputOutputManager<T> &iom,
      const MVParameterCuda<T> &mv_pars,
      const bool transposed);

  inline void gemm(
      const CudaContextPtr context,
      const T *dev_weights,
      const T *in_values,
      const int in_size,
      const bool in_trans,
      T *out_values,
      const int out_size,
      const bool out_trans,
      const int m_batch,
      const T alpha,
      const T beta,
      const bool transposed) {
    if (transposed) {
      // backward
      RPU::detail::backwardMatrix(
          context, dev_weights, in_values, in_size, in_trans, out_values, out_size, out_trans,
          m_batch, alpha, beta);
    } else {
      RPU::detail::forwardMatrix(
          context, dev_weights, in_values, in_size, in_trans, out_values, out_size, out_trans,
          m_batch, alpha, beta);
    }
  };

  int x_size_ = 0;
  int d_size_ = 0;
  CudaContextPtr context_ = nullptr;

private:
  inline void
  applyOutputWeightNoise(InputOutputManager<T> &iom, const bool out_trans, const bool tranposed);

  inline void applyOutputNoiseOtoO(
      InputOutputManager<T> &iom,
      const MVParameterCuda<T> &mv_pars,
      const bool out_trans,
      const bool tranposed);

  inline void applyOutputPCMReadNoise(
      const T *dev_weights, InputOutputManager<T> &iom, const bool out_trans, const bool tranposed);

  inline void applyIrDrop(
      const T *dev_weights,
      InputOutputManager<T> &iom,
      const bool out_trans,
      const bool transposed);

  template <typename OutputIteratorT>
  inline bool finalizeOutput(
      OutputIteratorT out_values,
      InputOutputManager<T> &iom,
      const MVParameterCuda<T> &mv_pars,
      const bool out_trans,
      const bool transposed) {
    auto io = iom.getIO();
    if (io.hasNLCalibration()) {
      applyOutputNonLinearity(iom, mv_pars, out_trans, transposed);
    }
    if (io.out_noise_std > (T)0.0) {
      applyOutputNoiseOtoO(iom, mv_pars, out_trans, transposed);
      return iom.applyToOutput(out_values, out_trans, false);
    }
    return iom.applyToOutput(out_values, out_trans);
  }

  void applyOutputNonLinearity(
      InputOutputManager<T> &iom,
      const MVParameterCuda<T> &mv_pars,
      const bool out_trans,
      const bool tranposed);

  void applyVoltageOffsets(
      const T *dev_weights,
      InputOutputManager<T> &iom,
      const MVParameterCuda<T> &mv_pars,
      const bool out_trans,
      const bool transposed);

  std::unique_ptr<CudaArray<T>> dev_ones_ = nullptr;

  FBParameterCuda<T> fb_pars_;
};

} // namespace RPU
