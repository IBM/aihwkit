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

#pragma once

#include "cuda_util.h"
#include "maximizer.h"
#include "rpu_pulsed_meta_parameter.h"

namespace RPU {

template <typename T> class NoiseManager {

public:
  explicit NoiseManager(CudaContext *c, int size);

  /* computes the scale values */
  template <typename InputIteratorT>
  void compute(
      InputIteratorT dev_input,
      const NoiseManagementType &nm_type,
      const IOMetaParameter<T> &io,
      int m_batch = 1,
      bool trans = false,
      bool is_test = false);

  /* sets the computed max values to zero below thres. Caution: This
     is in-place. does not check whether compute was called. */

  inline void copyScaleValuesToHost(float *dest) const { dev_scale_values_->copyTo(dest); };

  void printScaleValues() const { dev_scale_values_->printValues(); };
  float *getScaleValues() const;

private:
  void initializeBatchBuffer(int m_batch);

  std::unique_ptr<CudaArray<float>> dev_scale_values_ = nullptr; // need float here
  std::unique_ptr<Maximizer<T>> amaximizer_ = nullptr;
  std::unique_ptr<Maximizer<T>> maximizer_ = nullptr;

  NoiseManagementType nm_type_ = NoiseManagementType::None;
  int size_ = 0;
  CudaContext *context_ = nullptr;
  bool const_set_if_ = false;
};

} // namespace RPU
