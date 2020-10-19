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
  double fixed_value = 1.0;
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
    if (type == WeightClipType::FixedValue) {
      ss << "\t fixed value:\t\t" << fixed_value << std::endl;
    }
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

private:
  void clip(T *weights, T clip_value);
  std::vector<T> amax_values_;

  int x_size_ = 0;
  int d_size_ = 0;
  int size_ = 0;
};

} // namespace RPU
