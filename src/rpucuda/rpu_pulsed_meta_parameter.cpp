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

#include "rpu_pulsed_meta_parameter.h"
#include "math_util.h"
#include "rng.h"
#include "utility_functions.h"

#include <iostream>
// #include <random>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>

namespace RPU {

/********************************************************************************
 * IOMetaParameter<T>
 *********************************************************************************/
namespace detail {
template <typename T> T checkRes(T res) {
  T r = res < (T)0.0 ? (T)0.0 : res;
  return r > (T)1.0 ? (T)1. / r : r;
}

template <typename T> void checkAndSetRes(T &res, T &res_in, T range) {
  if (res_in != (T)0.0) {
    RPU_FATAL("Cannot re-set resolutions after parameters were intialized!");
  }
  res_in = checkRes(res);
  res = res_in * range;
}
} // namespace detail

template <typename T> void IOMetaParameter<T>::initializeForForward(int x_size, int d_size) {

  if (!_par_initialized) {
    // NOTE: will only init parameters once!!
    _par_initialized = true;

    if (mv_type == AnalogMVType::Ideal) {
      is_perfect = true;
    }

    if (is_perfect) {
      mv_type = AnalogMVType::Ideal;
      return;
    }

    // forward pass
    detail::checkAndSetRes(this->out_res, this->_out_res, (T)2.0 * this->out_bound);
    detail::checkAndSetRes(this->inp_res, this->_inp_res, (T)2.0 * this->inp_bound);

    if (this->noise_management != NoiseManagementType::None) {
      if (this->inp_bound != (T)1.0) {
        RPU_FATAL("Forward noise managment expects bound==1");
      }
    } else {
      this->nm_thres = (T)0.0;
    }
    if (this->out_bound <= (T)0.0) {
      this->out_bound = std::numeric_limits<T>::infinity();
    }

    if (this->inp_bound <= (T)0.0) {
      this->inp_bound = std::numeric_limits<T>::infinity();
    }
    if (v_offset_vec.size() > 0 && v_offset_vec.size() != (size_t)d_size) {
      RPU_FATAL("Size mismatch in user-defined v_offsets for forward.");
    }
    if (v_offset_vec.size() > 0 && v_offset_vec.size() != (size_t)d_size) {
      RPU_FATAL("Size mismatch in user-defined v_offsets for forward.");
    }
  }
  UNUSED(x_size);
}

template <typename T> void IOMetaParameter<T>::initializeForBackward(int x_size, int d_size) {

  if (!_par_initialized) {
    // NOTE: will only init parameters once !
    _par_initialized = true;

    if (mv_type == AnalogMVType::Ideal) {
      is_perfect = true;
    }

    if (is_perfect) {
      mv_type = AnalogMVType::Ideal;
      return;
    }

    // backward pass
    detail::checkAndSetRes(this->out_res, this->_out_res, (T)2.0 * this->out_bound);
    detail::checkAndSetRes(this->inp_res, this->_inp_res, (T)2.0 * this->inp_bound);

    if (this->noise_management != NoiseManagementType::None) {
      if (this->inp_bound != (T)1.0) {
        RPU_FATAL("Backward noise managment expects input bound==1");
      }
    } else {
      this->nm_thres = (T)0.0;
    }

    if (this->out_bound <= (T)0.0) {
      this->out_bound = std::numeric_limits<T>::infinity();
    }

    if (this->inp_bound <= (T)0.0) {
      this->inp_bound = std::numeric_limits<T>::infinity();
    }

    if (this->bound_management != BoundManagementType::None) {
      this->bound_management = BoundManagementType::None;
      // keep silent.
    }

    if (v_offset_vec.size() > 0 && v_offset_vec.size() != (size_t)x_size) {
      RPU_FATAL("Size mismatch in user-defined v_offsets for backward.");
    }
  }
  UNUSED(d_size);
}

/********************************************************************************
 * PulsedUpdateMetaParameter<T>
 *********************************************************************************/

template <typename T> void PulsedUpdateMetaParameter<T>::initialize() {

  if (!_par_initialized) {
    _par_initialized = true;
    // update
    // always turn on UM for UBL management
    update_management = update_management || update_bl_management;

    detail::checkRes(res);
    detail::checkRes(x_res_implicit);
    detail::checkRes(d_res_implicit);
  }

  if (_currently_tuning) {
    RPU_FATAL("Currently tuning cannot be set to True by user!");
  }
}

template <typename T>
void PulsedUpdateMetaParameter<T>::calculateBlAB(
    int &BL, T &A, T &B, T lr, T weight_granularity) const {
  if (lr < (T)0.0) {
    RPU_FATAL("lr should be positive !");
  } else if (lr == (T)0.0) {
    A = (T)0.0;
    B = (T)0.0;
    BL = 0;
    return;
  }

  if (fixed_BL || update_bl_management) {
    BL = desired_BL; // actually max for UBLM
    A = (T)sqrtf(lr / (weight_granularity * (T)BL));
    B = A;
  } else {
    if ((weight_granularity * (T)desired_BL) < lr) {
      A = (T)1.0;
      B = (T)1.0;
      BL = MAX((int)ceilf(lr / weight_granularity), 1);
    } else {
      BL = desired_BL;
      A = (T)sqrtf(lr / (weight_granularity * (T)BL));
      B = A;
    }
  }
}

template <typename T>
void PulsedUpdateMetaParameter<T>::performUpdateManagement(
    int &BL,
    T &A,
    T &B,
    const int max_BL,
    const T x_abs_max,
    const T d_abs_max,
    const T lr,
    const T weight_granularity) const {

  this->calculateBlAB(BL, A, B, lr, weight_granularity);
  if (lr > (T)0.0) {

    if (d_abs_max == (T)0 || x_abs_max == (T)0) {
      A = 1;
      B = 1;
      BL = 0;
      return;
    }

    if (update_bl_management || update_management) {

      T x_val = x_abs_max;
      T d_val = um_grad_scale * d_abs_max;
      T k_val = lr * x_val * d_val / weight_granularity;

      if (this->update_bl_management) {
        BL = (int)ceilf(k_val);
        if (BL > max_BL) {
          BL = max_BL; // the set BL is the *max BL* in case of update_bl_management  !
        }
        A = (T)sqrtf(lr / (weight_granularity * (T)BL));
        B = A;
      }

      if (this->update_management) {

        if (k_val > (T)max_BL) {
          // avoid clipping of x
          d_val *= (T)max_BL / k_val;
        }

        A *= (T)sqrtf(x_val / d_val);
        B *= (T)sqrtf(d_val / x_val);

        // that is:
        //     prob(x) = B * x = x * sqrt(d_amax / x_amax) * sqrt(lr / dw_min / BL)
        //     prob at x_amax == 1 ->  d_max * x_max * lr / dw_min / BL = 1
        //     --> lr = dw_min * BL / (d_max * x_max)
        // um_grad_scale will bias it towards the gradient d (clipping
        // more if smaller than 1)
      }
    }
  }
}

template struct PulsedUpdateMetaParameter<float>;
template struct IOMetaParameter<float>;
#ifdef RPU_USE_DOUBLE
template struct IOMetaParameter<double>;
template struct PulsedUpdateMetaParameter<double>;
#endif
#ifdef RPU_USE_FP16
template struct IOMetaParameter<half_t>;
template struct PulsedUpdateMetaParameter<half_t>;
#endif

/********************************************************************************
 * DebugPulsedUpdateMetaParameter<T>
 *********************************************************************************/
template <typename T>
void DebugPulsedUpdateMetaParameter<T>::calculateBlAB(
    int &BL, T &A, T &B, T lr, T weight_granularity) const {
  BL = this->desired_BL;
  A = scaleprob;
  B = scaleprob;
}

template struct DebugPulsedUpdateMetaParameter<float>;
#ifdef RPU_USE_DOUBLE
template struct DebugPulsedUpdateMetaParameter<double>;
#endif
#ifdef RPU_USE_FP16
template struct DebugPulsedUpdateMetaParameter<half_t>;
#endif

} // namespace RPU
