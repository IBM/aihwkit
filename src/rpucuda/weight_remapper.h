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

enum class WeightRemapType { None, LayerwiseSymmetric, ChannelwiseSymmetric };

// no template. Just double
struct WeightRemapParameter {

  WeightRemapType type = WeightRemapType::None;
  double max_scale_range = 0.0; // for Symmetric: whether to bound the diversity of scales
  double max_scale_ref = 0.0;   // reference is minimum scale if larger than this
  double remapped_wmax = 1.0;   // For Channelwise/LayerwiseSymmetric: where the max is mapped to
  void print() const {
    std::stringstream ss;
    printToStream(ss);
    std::cout << ss.str();
  };

  void printToStream(std::stringstream &ss) const {

    ss << "\t weight remapping type:\t" << getTypeName() << std::endl;
    if (type == WeightRemapType::None) {
      return;
    }
    if (max_scale_range > 0 && type == WeightRemapType::ChannelwiseSymmetric) {
      ss << "\t max scale range:\t" << max_scale_range << std::endl;
    }
    if (remapped_wmax != 1.0) {
      ss << "\t remapped_wmax:\t\t" << remapped_wmax << std::endl;
    }

    ss << std::endl;
  };

  inline std::string getTypeName() const {
    switch (type) {
    case WeightRemapType::None:
      return "None";
    case WeightRemapType::LayerwiseSymmetric:
      return "LayerwiseSymmetric";
    case WeightRemapType::ChannelwiseSymmetric:
      return "ChannelwiseSymmetric";
    default:
      return "Unknown";
    }
  };
};

template <typename T> class WeightRemapper {

public:
  explicit WeightRemapper(int x_size, int d_size);
  WeightRemapper(){};

  /* in-place remap of weights */
  void apply(
      T *weights, T current_lr, const WeightRemapParameter &wmpar, T *scales, T *biases = nullptr);

private:
  std::vector<T> max_values_;
  std::vector<T> min_values_;
  std::vector<T> old_scales_;

  int x_size_ = 0;
  int d_size_ = 0;
  int size_ = 0;
};

} // namespace RPU
