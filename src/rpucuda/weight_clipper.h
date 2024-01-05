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

enum class WeightClipType {
  None,
  FixedValue,
  LayerGaussian,
  AverageChannelMax,
};

// no template. Just double
struct WeightClipParameter {

  WeightClipType type = WeightClipType::None;
  double fixed_value = -1.0; // will always be applied if larger >0
  double sigma = 2.5;

  void print() const {
    std::stringstream ss;
    printToStream(ss);
    std::cout << ss.str();
  };

  void printToStream(std::stringstream &ss) const {

    ss << "\t weight clipper type:\t" << getTypeName() << std::endl;
    if (type == WeightClipType::None) {
      return;
    }
    ss << "\t fixed value:\t\t" << fixed_value << std::endl;
    if (type == WeightClipType::LayerGaussian) {
      ss << "\t sigma:\t\t" << sigma << std::endl;
    }

    ss << std::endl;
  };

  inline std::string getTypeName() const {
    switch (type) {
    case WeightClipType::None:
      return "None";
    case WeightClipType::FixedValue:
      return "FixedValue";
    case WeightClipType::AverageChannelMax:
      return "AverageChannelMax";
    case WeightClipType::LayerGaussian:
      return "LayerGaussian";
    default:
      return "Unknown";
    }
  };
};

template <typename T> class WeightClipper {

public:
  explicit WeightClipper(int x_size, int d_size);
  WeightClipper(){};

  /* in-place clipping of weights */
  void apply(T *weights, const WeightClipParameter &wclpar);

  void dumpExtra(RPU::state_t &extra, const std::string prefix){};
  void loadExtra(const RPU::state_t &extra, const std::string prefix, bool strict){};

private:
  void clip(T *weights, T clip_value);
  std::vector<T> amax_values_;

  int x_size_ = 0;
  int d_size_ = 0;
  int size_ = 0;
};

} // namespace RPU
