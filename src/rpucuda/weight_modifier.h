/**
 * (C) Copyright 2020, 2021, 2022 IBM. All Rights Reserved.
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
  Poly
};

// no template. Just double
struct WeightModifierParameter {

  double std_dev = 0.0;
  double res = 0.1;
  bool sto_round = false;
  double dorefa_clip = 0.6;
  double pdrop = 0.0;
  bool enable_during_test = false;
  bool copy_last_column = false;
  bool rel_to_actual_wmax = true;
  double assumed_wmax = 1.0;

  // expects to gmax normalized norm_prog_coeff.
  // [0.26348 / 25.0, 0.0768, -0.001877 * 25.0]
  double coeff0 = 0.26348 / 25.0;
  double coeff1 = 0.0768;
  double coeff2 = -0.001877 * 25.0;

  WeightModifierType type = WeightModifierType::Copy;

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
          type == WeightModifierType::AddNormal ||
          type == WeightModifierType::DiscretizeAddNormal) {
        ss << "\t std_dev:\t\t" << std_dev << std::endl;
      }
      ss << "\t rel_to_actual_wmax:\t" << rel_to_actual_wmax << std::endl;
      ss << "\t assumed_wmax:\t\t" << assumed_wmax << std::endl;
    }
    if (copy_last_column) {
      ss << "\t copy_last_column:\t" << copy_last_column << std::endl;
    }
    if (pdrop > 0.0) {
      ss << "\t pdrop:\t\t\t" << pdrop << std::endl;
    }
    if (type == WeightModifierType::DoReFa) {
      ss << "\t dorefa clip:\t\t" << dorefa_clip << std::endl;
    }
    if (type == WeightModifierType::Poly) {
      ss << "\t coeff0,1,2:\t\t" << coeff0 << ", " << coeff1 << ", " << coeff2 << std::endl;
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
        type == WeightModifierType::AddNormal || type == WeightModifierType::DiscretizeAddNormal ||
        (type == WeightModifierType::DoReFa && sto_round));
  };
};

template <typename T> class WeightModifier {

public:
  explicit WeightModifier(int x_size, int d_size);
  WeightModifier(){};

  /* buffers the weight changes and redraws the drop connection*/
  void apply(T *new_weights, const T *weights, const WeightModifierParameter &wmpar);

  inline bool enableDuringTest() { return enable_during_test_; };

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
