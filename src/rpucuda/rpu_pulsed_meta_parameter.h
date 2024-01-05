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

#include "math_util.h"
#include <iostream>
#include <limits>
#include <memory>
#include <random>

namespace RPU {

enum class NoiseManagementType {
  None,
  AbsMax,
  AbsMaxNPSum,
  Max,
  Constant,
  AverageAbsMax,
  AverageAbsMaxSingleValue,
  AbsMaxSingleValue
};

enum class BoundManagementType { None, Iterative, IterativeWorstCase };

enum class OutputWeightNoiseType { None, AdditiveConstant, PCMRead };

enum class PulseType {
  None,
  StochasticCompressed,
  Stochastic,
  NoneWithDevice,
  MeanCount,
  DeterministicImplicit
};

enum class AnalogMVType {
  Ideal,
  OnePass,
  PosNegSeparate,
  PosNegSeparateDigitalSum,
  SplitMode,
  BitWise
};

template <typename T> struct IOMetaParameter {
  bool _par_initialized = false;

  bool is_perfect = false; // short-cut to use pure floating point (only out_scale will be applied)
  AnalogMVType mv_type = AnalogMVType::OnePass;
  OutputWeightNoiseType w_noise_type = OutputWeightNoiseType::None;
  T ir_drop = (T)0.0;
  T ir_drop_Gw_div_gmax = (T)5.7143e5; // physical ratio of wire conductance to physical gmax

  T v_offset_std = (T)0.0;     // systematic voltage offset variation at output
  std::vector<T> v_offset_vec; // systematic voltage offset variation at output
  T v_offset_w_min = (T)0.0;   // reference value for offset. typically -1 (for w_min = -1)
  T r_series = (T)0.0;
  T r_series_max_total = std::numeric_limits<T>::max();

  T inp_asymmetry = (T)0.0; // negative input number are scaled with (1 - asymmetry)
  T out_asymmetry = (T)0.0;
  T w_read_asymmetry_dtod = (T)0.0; // neg input weight-scale (e.g. pcm polarity) NOTE: only
                                    // observed for AnalogMVType::PosNegSeparate

  T inp_bound = (T)1.0;
  T inp_res = (T)1.0 / ((T)powf((T)2.0, (T)7.0) - (T)2.0);
  T _inp_res = (T)0.0; // this is the unscaled version saved for output..
  bool inp_sto_round = false;
  T inp_noise = (T)0.0;
  T out_noise = (T)0.06;
  T out_noise_std = (T)0.0; // systematic variation in percent of out_noise
  T w_noise = (T)0.0;
  T out_bound = (T)12.0;
  T out_res = (T)1.0 / ((T)powf((T)2.0, (T)9.0) - (T)2.0);
  T _out_res = (T)0;
  bool out_sto_round = false;
  T out_scale = (T)1.0;

  // this is: y/(1 + nl_i*|y|) with
  // nl_i = out_nonlinearity / out_bound * (1 + out_nonlinearity_std * rng_->sampleGauss());
  // thus an out_nonlinearity of 1 would reduce the output by half at out_bound
  T out_nonlinearity = (T)0.0;
  T out_nonlinearity_std = (T)0.0; // output to output line systematic variation
  std::vector<T> out_nonlinearity_vec;
  // y/out_bound value where the slope of the NL is 1 on average. This calibration is done in ADC
  T slope_calibration = (T)0.5;

  NoiseManagementType noise_management = NoiseManagementType::AbsMax;
  T nm_thres = (T)0.0;
  T nm_assumed_wmax = (T)0.6;
  T nm_decay = (T)1e-3; // minibatches for AverageAbsMax

  BoundManagementType bound_management = BoundManagementType::None;
  bool bm_test_negative_bound = true;
  int max_bm_factor = 1000; // absolute max of BM
  T max_bm_res =
      (T)0.25; // bounds BM to less than max_bm_res times the input number of states (1/inp_res)

  void initializeForForward(int x_size, int d_size);
  void initializeForBackward(int x_size, int d_size);
  void print() const {
    std::stringstream ss;
    printToStream(ss);
    std::cout << ss.str();
  };

  inline bool hasVoltageOffsets() const {
    return v_offset_std > (T)0 || r_series > (T)0 || v_offset_vec.size() > (size_t)0;
  };
  inline bool isPerfect() const { return mv_type == AnalogMVType::Ideal || is_perfect; }
  inline bool hasOutNonlinearity() const {
    return (out_nonlinearity > (T)0 || out_nonlinearity_vec.size() > (size_t)0) && (!isPerfect());
  };
  inline bool hasNLCalibration() const { return hasOutNonlinearity() || r_series > (T)0.0; }
  void printToStream(std::stringstream &ss) const {

    if (!isPerfect()) {

      switch (mv_type) {
      case AnalogMVType::Ideal:
        ss << "\t mv_type:\t\tIdeal\n";
        break;
      case AnalogMVType::OnePass:
        ss << "\t mv_type:\t\tOnePass\n";
        break;
      case AnalogMVType::PosNegSeparate:
        ss << "\t mv_type:\t\tPosNegSeparate\n";
        break;
      case AnalogMVType::PosNegSeparateDigitalSum:
        ss << "\t mv_type:\t\tPosNegSeparateDigitalSum\n";
        break;
      default:
        RPU_FATAL("MV type not implemented.");
      }

      ss << "\t inp/out_bound:\t\t" << inp_bound << " / " << out_bound << std::endl;
      if (_par_initialized) {
        ss << "\t DAC/ADC:\t\t" << 1.0f / MAX((float)_inp_res, 0.0f) << " / "
           << 1.0f / MAX((float)_out_res, 0.0f) << std::endl;
      } else {
        ss << "\t DAC/ADC:\t\t" << 1.0f / MAX((float)inp_res, 0.0f) << " / "
           << 1.0f / MAX((float)out_res, 0.0f) << std::endl;
      }
      if (inp_sto_round || out_sto_round) {
        ss << "\t sto_round:\t\t" << std::boolalpha << inp_sto_round;
        ss << " / " << std::boolalpha << out_sto_round << std::endl;
      }
      ss << "\t out_noise:\t\t" << out_noise << std::endl;
      if (inp_noise > (T)0.0) {
        ss << "\t inp_noise:\t\t" << inp_noise << std::endl;
      }
      if (w_noise > (T)0.0 && w_noise_type != OutputWeightNoiseType::None) {
        ss << "\t w_noise:\t\t" << w_noise << std::endl;
        if (w_noise_type == OutputWeightNoiseType::AdditiveConstant) {
          ss << "\t w_noise_type:\t\t" << (int)OutputWeightNoiseType::AdditiveConstant
             << " (AdditiveConstant) " << std::endl;
        }
        if (w_noise_type == OutputWeightNoiseType::PCMRead) {
          ss << "\t w_noise_type:\t\t" << (int)OutputWeightNoiseType::PCMRead << " (PCMRead) "
             << std::endl;
        }
      }
      if (ir_drop > (T)0.0) {
        ss << "\t ir_drop [scale]:\t" << ir_drop << "  (ir_drop_Gw_div_gmax is "
           << ir_drop_Gw_div_gmax << ")" << std::endl;
      }
      if (hasVoltageOffsets()) {
        ss << "\t r_series:\t\t" << r_series << std::endl;
        ss << "\t r_series_max_total:\t" << r_series_max_total << std::endl;
        if (v_offset_std > (T)0.0) {
          ss << "\t v_offset_std:\t\t" << v_offset_std << std::endl;
        } else if (v_offset_vec.size() > (size_t)0) {
          // compute here ?
          ss << "\t v_offset_std:\t\tuser-defined" << std::endl;
        }
        ss << "\t v_offset_w_min:\t" << v_offset_w_min << std::endl;
      }

      if (hasOutNonlinearity()) {
        if (out_nonlinearity > (T)0.0 || out_nonlinearity_std > (T)0.0) {
          ss << "\t out_nonlinearity:\t" << out_nonlinearity << std::endl;
          ss << "\t out_nonlinearity_std:\t" << out_nonlinearity_std << std::endl;
        } else {
          ss << "\t out_nonlinearity:\tuser-defined" << std::endl;
        }
      }
      if (hasNLCalibration()) {
        ss << "\t slope_calibration:\t" << slope_calibration << std::endl;
      }

      if (inp_asymmetry != (T)0.0) {
        ss << "\t inp_asymmetry:\t\t" << inp_asymmetry << std::endl;
      }
      if (out_asymmetry != (T)0.0) {
        ss << "\t out_asymmetry:\t\t" << out_asymmetry << std::endl;
      }

      if (w_read_asymmetry_dtod > (T)0.0 && mv_type == AnalogMVType::PosNegSeparate) {
        ss << "\t w_read_asymmetry_dtod:\t" << w_read_asymmetry_dtod << std::endl;
      }
    }
    if (out_scale != (T)1.0) {
      ss << "\t out_scale:\t\t" << out_scale << std::endl;
    }
    if (!isPerfect()) {
      if (noise_management == NoiseManagementType::AbsMax && nm_thres > (T)0.0) {
        ss << "\t noise_management [nm_thres]:\t" << nm_thres << std::endl;
      } else if (noise_management == NoiseManagementType::AbsMaxNPSum) {
        ss << "\t noise_management [NPSum;wmax]:\t" << nm_assumed_wmax << std::endl;
      } else if (noise_management == NoiseManagementType::Constant && nm_thres > (T)0.0) {
        ss << "\t noise_management: \t" << nm_thres << " (Constant scale)" << std::endl;
      } else {
        ss << "\t noise_management:\t" << (int)noise_management << std::endl;
      }
      ss << "\t bound_management ";
      if (bm_test_negative_bound) {
        ss << "[+/-]:";
      } else {
        ss << "[+]:\t";
      }
      switch (bound_management) {
      case BoundManagementType::None:
        ss << "None";
        break;
      case BoundManagementType::Iterative:
        ss << "Iterative";
        break;
      case BoundManagementType::IterativeWorstCase:
        ss << "Iterative [2nd round with NM::AbsMaxNPSum]";
        break;
      default:
        ss << "UNKNOWN.";
        break; // should never happen
      };
    } else {
      ss << "\t using ideal floating point.";
    }

    ss << std::endl;
  }
};

template <typename T> struct PulsedUpdateMetaParameter {

  bool fixed_BL = true;
  int desired_BL = 31;

  bool update_management = true;
  bool update_bl_management = true;
  T um_grad_scale = 1.0; // bias gradient for UM (ie 0.5 means more clipping of gradient)
  T um_reg_scale = 1.0;  // scale for regularizer of UM / UBLM (scale=1 means reg = dw_min**2)
  bool sto_round = false;
  bool d_sparsity = false; // whether compute d sparsity
  T _d_sparsity = 0.0;     // hidden container

  T res = (T)0; // this is taken to be in the range 0..1 as positive and negative phases are done
                // separately

  T x_res_implicit = (T)0; // in case of implicit pulsing. Assumes range 0..1
  T d_res_implicit = (T)0;

  bool _par_initialized = false;
  bool _currently_tuning = false;
  int _debug_kernel_index = -1; // for PWU debugging.

  PulseType pulse_type = PulseType::StochasticCompressed;

  inline bool needsImplicitPulses() const {
    return pulse_type == PulseType::DeterministicImplicit || pulse_type == PulseType::None;
  };

  void initialize();
  inline int getNK32Default() const { return desired_BL / 32 + 1; };

  virtual void calculateBlAB(int &BL, T &A, T &B, T lr, T weight_granularity) const;
  virtual void performUpdateManagement(
      int &BL,
      T &A,
      T &B,
      const int max_BL,
      const T x_abs_max,
      const T d_abs_max,
      const T lr,
      const T weight_granularity) const;
  void print() const {
    std::stringstream ss;
    printToStream(ss);
    std::cout << ss.str();
  };
  void printToStream(std::stringstream &ss) const {
    if (pulse_type == PulseType::None) {
      ss << "\t using ideal floating point." << std::endl;
    } else if (pulse_type == PulseType::NoneWithDevice) {
      ss << "\t using ideal floating point (with device)." << std::endl;
    } else {
      if (needsImplicitPulses()) {
        ss << "\t using implicit pulsing scheme." << std::endl;
        if (x_res_implicit > (T)0.0) {
          ss << "\t nx (x_res):\t\t" << 1.f / (float)x_res_implicit << std::endl;
        }
        if (d_res_implicit > (T)0.0) {
          ss << "\t nd (d_res):\t\t" << 1.f / (float)d_res_implicit << std::endl;
        }
      }
      ss << "\t desired_BL:\t\t" << desired_BL << std::endl;
      ss << "\t fixed_BL:\t\t" << std::boolalpha << fixed_BL << std::endl;
      ss << "\t update_management:\t" << std::boolalpha << update_management << std::endl;
      if (um_grad_scale != (T)1.0) {
        ss << "\t um_grad_scale:\t" << um_grad_scale << std::endl;
      }
      ss << "\t update_management:\t" << std::boolalpha << update_management << std::endl;
      ss << "\t update_bl_management:\t" << std::boolalpha << update_bl_management << std::endl;
      ss << "\t up_DAC_stoc_round:\t" << sto_round << std::endl;
      ss << "\t up_DAC:\t\t" << 1.0f / MAX((float)res, 0.0f) << std::endl;
      ss << "\t pulse_type:\t\t" << (int)pulse_type << std::endl;
    }
  }
};

// just for setting the A/B to fixed value for debug
template <typename T> struct DebugPulsedUpdateMetaParameter : PulsedUpdateMetaParameter<T> {
  T scaleprob = 1;
  void calculateBlAB(int &BL, T &A, T &B, T lr, T dw_min) const override;
};

}; // namespace RPU
