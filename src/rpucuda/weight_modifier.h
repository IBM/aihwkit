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

enum class WeightModifierType {
  Copy, // does nothing, just copy (e.g. for delayed weight update), however, could also drop
  Discretize,
  MultNormal,
  AddNormal,
  DiscretizeAddNormal,
  DoReFa,
  Poly,
  PCMNoise,
  DropConnect,
  ProgNoise,
};

template <typename T> struct WeightModifierParameter {
  T std_dev = 0.0;
  bool per_batch_sample = false;
  T res = 0.1;
  bool sto_round = false;
  T dorefa_clip = 0.6;
  T pdrop = 0.0;
  bool enable_during_test = false;
  bool copy_last_column = false;
  bool rel_to_actual_wmax = true;
  T assumed_wmax = 1.0;
  T g_max = 25.0;

  T pcm_zero_thres = 0.0; // for PCMNoise in units of of baseline noise
  T pcm_t_inference = 0.0;
  T pcm_prob_at_reset = 0.0;
  T pcm_prob_at_gmax = 0.0;
  T pcm_prob_at_random = 0.0;

  T pcm_t0 = 20.0;

  WeightModifierType type = WeightModifierType::Copy;
  std::vector<T> coeffs = {0.26348 / 25.0, 0.0768, -0.001877 * 25.0};

  inline std::string getTypeName() const {
    switch (type) {
    case WeightModifierType::Copy:
      return "Copy";
    case WeightModifierType::MultNormal:
      return "MultNormal";
    case WeightModifierType::Discretize:
      return "Discretize";
    case WeightModifierType::AddNormal:
      return "AddNormal";
    case WeightModifierType::DoReFa:
      return "DoReFa";
    case WeightModifierType::DiscretizeAddNormal:
      return "DiscretizeAddNormal";
    case WeightModifierType::Poly:
      return "Poly";
    case WeightModifierType::ProgNoise:
      return "ProgNoise";
    case WeightModifierType::PCMNoise:
      return "PCMNoise";
    case WeightModifierType::DropConnect:
      return "DropConnect";
    default:
      return "Unknown";
    }
  };

  void print() const {
    std::stringstream ss;
    printToStream(ss);
    std::cout << ss.str();
  };

  void printToStream(std::stringstream &ss) const {
    ss << "\t weight modifier type:\t" << getTypeName() << std::endl;
    if (type != WeightModifierType::Copy) {
      if (type == WeightModifierType::Poly || type == WeightModifierType::MultNormal ||
          type == WeightModifierType::AddNormal || type == WeightModifierType::ProgNoise ||
          type == WeightModifierType::DiscretizeAddNormal || type == WeightModifierType::PCMNoise) {
        ss << "\t std_dev:\t\t" << std_dev << std::endl;
      }
      ss << "\t rel_to_actual_wmax:\t" << rel_to_actual_wmax << std::endl;
      ss << "\t assumed_wmax:\t\t" << assumed_wmax << std::endl;
    }
    if (copy_last_column) {
      ss << "\t copy_last_column:\t" << copy_last_column << std::endl;
    }
    if (type == WeightModifierType::PCMNoise) {
      if (pcm_t_inference > 0) {
        ss << "\t pcm_t_inference:\t\t" << pcm_t_inference << std::endl;
      }
      if (pcm_zero_thres > 0) {
        ss << "\t pcm_zero_thres:\t\t" << pcm_zero_thres << std::endl;
      }
      if (pcm_prob_at_reset > 0) {
        ss << "\t pcm_prob_at_reset:\t\t" << pcm_prob_at_reset << std::endl;
      }
      if (pcm_prob_at_gmax > 0) {
        ss << "\t pcm_prob_at_gmax:\t\t" << pcm_prob_at_reset << std::endl;
      }
    }
    if (pdrop > 0.0) {
      ss << "\t pdrop:\t\t\t" << pdrop << std::endl;
    }
    if (type == WeightModifierType::DoReFa) {
      ss << "\t dorefa clip:\t\t" << dorefa_clip << std::endl;
    }
    if (type == WeightModifierType::ProgNoise || type == WeightModifierType::PCMNoise) {
      ss << "\t g_max:\t\t" << g_max << std::endl;
    }

    if (type == WeightModifierType::Poly) {
      for (int i = 0; i < (int)coeffs.size(); i++) {
        ss << "\t coeff [" << i << "]:\t" << coeffs[i] << std::endl;
      }
    }
    if (type == WeightModifierType::Discretize || type == WeightModifierType::DiscretizeAddNormal ||
        type == WeightModifierType::DoReFa) {
      ss << "\t res:\t\t\t" << res << std::endl;
    }
    if (enable_during_test) {
      ss << "\t enabled during test." << std::endl;
    }

    ss << std::endl;
  }

  inline bool usesRandom() {
    return (
        pdrop > 0 || (type == WeightModifierType::Discretize && sto_round) ||
        type == WeightModifierType::MultNormal || type == WeightModifierType::Poly ||
        type == WeightModifierType::PCMNoise || type == WeightModifierType::AddNormal ||
        type == WeightModifierType::ProgNoise || type == WeightModifierType::DiscretizeAddNormal ||
        (type == WeightModifierType::DoReFa && sto_round));
  };
};

template <typename T> class WeightModifier {

public:
  explicit WeightModifier(int x_size, int d_size);
  WeightModifier(){};

  /* buffers the weight changes and redraws the drop connection*/
  void apply(T *new_weights, const T *weights, const WeightModifierParameter<T> &wmpar);

  inline bool enableDuringTest() { return enable_during_test_; };

  void dumpExtra(RPU::state_t &extra, const std::string prefix);
  void loadExtra(const RPU::state_t &extra, const std::string prefix, bool strict);

private:
  void dropConnections(T *weights, T prob);

  int x_size_ = 0;
  int d_size_ = 0;
  int size_ = 0;
  std::vector<T> saved_bias_;
  bool enable_during_test_ = false;
  RealWorldRNG<T> rw_rng_{0};
};

} // namespace RPU
