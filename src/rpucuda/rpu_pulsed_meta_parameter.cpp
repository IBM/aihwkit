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

#include "rpu_pulsed_meta_parameter.h"
#include "math_util.h"
#include "rng.h"
#include "utility_functions.h"

#include <iostream>
//#include <random>
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
  if (res_in != 0) {
    RPU_FATAL("Cannot re-set resolutions after parameters were intialized!");
  }
  res_in = checkRes(res);
  res = res_in * range;
}
} // namespace detail

template <typename T> void IOMetaParameter<T>::initializeForForward() {

  if (!_par_initialized) {
    // NOTE: will only init parameters once!!
    _par_initialized = true;

    if (is_perfect) {
      return;
    }

    // forward pass
    detail::checkAndSetRes(this->out_res, this->_out_res, (T)2.0 * this->out_bound);
    detail::checkAndSetRes(this->inp_res, this->_inp_res, (T)2.0 * this->inp_bound);

    if (this->noise_management != NoiseManagementType::None) {
      if (this->inp_bound != 1.0) {
        RPU_FATAL("Forward noise managment expects bound==1");
      }
    } else {
      this->nm_thres = (T)0.0;
    }

    if ((this->out_bound <= 0.0) || (this->inp_bound <= 0.0)) {
      RPU_FATAL("Forward bounds need to be >0");
    }

    if (isinf(this->out_bound)) {
      RPU_FATAL("Forward out bound needs to be finite");
    }
  }
}

template <typename T> void IOMetaParameter<T>::initializeForBackward() {

  if (!_par_initialized) {
    // NOTE: will only init parameters once !
    _par_initialized = true;
    if (is_perfect) {
      return;
    }
    // backward pass
    detail::checkAndSetRes(this->out_res, this->_out_res, (T)2.0 * this->out_bound);
    detail::checkAndSetRes(this->inp_res, this->_inp_res, (T)2.0 * this->inp_bound);

    if (isinf(this->out_bound)) {
      RPU_FATAL("Backward out bound needs to be finite");
    }

    if (this->noise_management != NoiseManagementType::None) {
      if (this->inp_bound != 1) {
        RPU_FATAL("Backward noise managment expects input bound==1");
      }
    } else {
      this->nm_thres = (T)0.0;
    }

    if ((this->out_bound <= 0.0) || (this->inp_bound <= 0.0)) {
      RPU_FATAL("Backward bounds need to be >0");
    }

    if (this->bound_management != BoundManagementType::None) {
      this->bound_management = BoundManagementType::None;
      // keep silent.
    }
  }
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
  }

  if (_currently_tuning) {
    RPU_FATAL("Currently tuning cannot be set to True by user!");
  }
}

template <typename T>
void PulsedUpdateMetaParameter<T>::calculateBlAB(int &BL, T &A, T &B, T lr, T dw_min) const {
  if (lr < 0.0) {
    RPU_FATAL("lr should be positive !");
  } else if (lr == 0.0) {
    A = (T)0.0;
    B = (T)0.0;
    BL = 0;
  }

  if (fixed_BL || update_bl_management) {
    BL = desired_BL; // actually max for UBLM
    A = sqrt(lr / (dw_min * BL));
    B = A;
  } else {
    if ((dw_min * desired_BL) < lr) {
      A = (T)1.0;
      B = (T)1.0;
      BL = (int)ceil(lr / dw_min);
    } else {
      BL = desired_BL;
      A = sqrt(lr / (dw_min * BL));
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
    const T dw_min) const {

  this->calculateBlAB(BL, A, B, lr, dw_min);
  if (lr > 0.0) {
    if (this->update_bl_management || this->update_management) {

      T reg = dw_min * dw_min;
      T x_abs_max_value = (x_abs_max < reg) ? reg : x_abs_max;
      T d_abs_max_value = (d_abs_max < reg) ? reg : d_abs_max;

      if (this->update_bl_management) {

        BL = (int)ceil(lr * x_abs_max_value * d_abs_max_value / dw_min);
        if (BL > max_BL) {
          BL = max_BL; // the set BL is the *max BL* in case of update_bl_management  !
        }
        A = sqrt(lr / (dw_min * BL));
        B = A;
      }
      if (this->update_management) {

        A *= sqrt(x_abs_max_value / d_abs_max_value);
        B *= sqrt(d_abs_max_value / x_abs_max_value);
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

/********************************************************************************
 * DebugPulsedUpdateMetaParameter<T>
 *********************************************************************************/
template <typename T>
void DebugPulsedUpdateMetaParameter<T>::calculateBlAB(int &BL, T &A, T &B, T lr, T dw_min) const {
  BL = this->desired_BL;
  A = scaleprob;
  B = scaleprob;
}

template struct DebugPulsedUpdateMetaParameter<float>;

#ifdef RPU_USE_DOUBLE
template struct DebugPulsedUpdateMetaParameter<double>;
#endif

} // namespace RPU
