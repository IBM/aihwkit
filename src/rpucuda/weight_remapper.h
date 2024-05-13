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

enum class WeightRemapType {
  None,
  LayerwiseSymmetric,
  ChannelwiseSymmetric,
  ChannelwiseNorm,
  ChannelwiseExceeded,
  LayerwiseAsymmetric,
  ChannelwiseAsymmetric
};

// no template. Just double
struct WeightRemapParameter {

  WeightRemapType type = WeightRemapType::None;
  double max_scale_range = 0.0; // for Symmetric: whether to bound the diversity of scales
  double max_scale_ref = 0.0;   // reference is minimum scale if larger than this
  double row_norm = 1.0;        // max norm of row for ChannelwiseNorm
  double remapped_wmax = 1.0;   // For Channelwise/LayerwiseSymmetric: where the max is mapped to
  bool clip_if = true;          // only for norm
  int swa_every = 0;            // stochastic weight averaging
  int swa_transfer_every =
      0; // stochastic weight averaging -> copy to weight [counting in the number of swa applied]
  uint64_t swa_start = 0; // start iter to do SWA
  void print() const {
    std::stringstream ss;
    printToStream(ss);
    std::cout << ss.str();
  };

  void printToStream(std::stringstream &ss) const {

    if (swa_every > 0) {
      ss << "\t swa_every:\t\t" << swa_every << std::endl;
      ss << "\t swa_start:\t\t" << swa_start << std::endl;
    }
    if (swa_transfer_every > 0) {
      ss << "\t swa_transfer_every:\t" << swa_transfer_every << std::endl;
    }

    ss << "\t weight remapping type:\t" << getTypeName() << std::endl;
    if (type == WeightRemapType::None) {
      return;
    }
    if (type == WeightRemapType::ChannelwiseNorm) {
      ss << "\t row norm:\t\t" << row_norm << std::endl;
    }
    if (type == WeightRemapType::ChannelwiseNorm || type == WeightRemapType::ChannelwiseExceeded) {
      ss << "\t clip if:\t\t" << clip_if << std::endl;
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
    case WeightRemapType::ChannelwiseNorm:
      return "ChannelwiseNorm";
    case WeightRemapType::ChannelwiseSymmetric:
      return "ChannelwiseSymmetric";
    case WeightRemapType::ChannelwiseExceeded:
      return "ChannelwiseExceeded";
    case WeightRemapType::LayerwiseAsymmetric:
      return "LayerwiseAsymmetric";
    case WeightRemapType::ChannelwiseAsymmetric:
      return "ChannelwiseAsymmetric";
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

  /* intermittantly saving of the running average weight. Returns true if weight was modified */
  bool applySWA(
      T *swa_weights,
      T *weights,
      uint64_t iter,
      const WeightRemapParameter &wmpar,
      T current_lr,
      T *scales = nullptr,
      T *biases = nullptr);

  void dumpExtra(RPU::state_t &extra, const std::string prefix){};
  void loadExtra(const RPU::state_t &extra, const std::string prefix, bool strict){};

private:
  std::vector<T> max_values_;
  std::vector<T> min_values_;
  std::vector<T> old_scales_;
  std::vector<T> norm_values_;

  int x_size_ = 0;
  int d_size_ = 0;
  int size_ = 0;
};

} // namespace RPU
