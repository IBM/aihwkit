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

#include "noise_manager.h"

#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <random>

#include "cuda_math_util.h"
#include "cuda_util.h"
#include <cub/cub.cuh>

#include "io_iterator.h"

namespace RPU {

/****************************************************************************************************************/
/* NoiseManager */
/******************************************************************************************************************/

template <typename T>
NoiseManager<T>::NoiseManager(CudaContext *c, int size)
    : size_(size), context_(c), const_set_if_(false) {
  // initialize for m_batch=1
  dev_scale_values_ = RPU::make_unique<CudaArray<float>>(context_, 1);

  amaximizer_ = RPU::make_unique<Maximizer<T>>(context_, size, true);
  maximizer_ = RPU::make_unique<Maximizer<T>>(context_, size, false);

  context_->synchronize();
}

template <typename T>
template <typename InputIteratorT>
void NoiseManager<T>::compute(
    InputIteratorT dev_input,
    const NoiseManagementType &nm_type,
    const IOMetaParameter<T> &io,
    int m_batch,
    bool trans,
    bool is_test) {
  // does not check for positive m_batch!
  nm_type_ = nm_type;

  switch (nm_type_) {

  case NoiseManagementType::None: {
    return;
  }
  case NoiseManagementType::Constant: {
    if (m_batch > dev_scale_values_->getSize()) {
      dev_scale_values_ = RPU::make_unique<CudaArray<float>>(context_, m_batch);
      const_set_if_ = false;
    }
    if (!const_set_if_) {
      dev_scale_values_->setConst(io.nm_thres > 0 ? (float)io.nm_thres : (float)1.0);
      const_set_if_ = true;
    }
    return;
  }

  case NoiseManagementType::Max: {
    this->maximizer_->compute(dev_input, m_batch, trans);
    if (io.nm_thres > 0) {
      this->maximizer_->saturateAbove(io.nm_thres);
    }

    return;
  }

  case NoiseManagementType::AbsMax: {
    this->amaximizer_->compute(dev_input, m_batch, trans);
    if (io.nm_thres > 0) {
      this->amaximizer_->saturateAbove(io.nm_thres);
    }

    return;
  }

  default:
    RPU_FATAL("Noise management type not implemented.");
  }
}

template <typename T> float *NoiseManager<T>::getScaleValues() const {
  switch (nm_type_) {
  case NoiseManagementType::None:
    return nullptr;
  case NoiseManagementType::Constant:
    return dev_scale_values_->getData();
  case NoiseManagementType::AbsMax:
    return amaximizer_->getMaxValues();
  case NoiseManagementType::Max:
    return maximizer_->getMaxValues();
  default:
    RPU_FATAL("Noise management type not implemented.");
  }
};

#define ARGS1(NUM_T) , const NoiseManagementType &, const IOMetaParameter<NUM_T> &, int, bool, bool
#define ARGS2 , int, bool

template class NoiseManager<float>;
RPU_GEN_IITER_TEMPLATES(float, void, NoiseManager<float>::compute, ARGS1(float));

#ifdef RPU_USE_DOUBLE
template class NoiseManager<double>;
RPU_GEN_IITER_TEMPLATES(double, void, NoiseManager<double>::compute, ARGS1(double));
#endif

#undef ARGS1
#undef ARGS2

} // namespace RPU
