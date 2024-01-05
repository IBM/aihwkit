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

#include "chopped_weight_output.h"

#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <random>

#include "cuda_math_util.h"
#include "cuda_util.h"
#include "io_iterator.h"

namespace RPU {

__global__ void kernelCreateWeightOutputSignalsUBLMBO64(
    uint32_t *weight_output_signals,
    const int m_batch,
    const int batch_start,
    const int every,
    const kagg_t *Kc_values) {

  // only for ublm == true needed
  // weight_output_signals: need to be of size: getBo64Batch(m_batch)
  // weight_output_signals: need to be ZEROED before running this kernel.

  RPU_CUDA_1D_KERNEL_LOOP(i_batch, m_batch) {
    // since end position will be coded (but Kc has the start pos),
    // here we check whether the previous batch index is a weight_output index

    // Kc[m_batch] is sum of all and Kc[0] is always zero
    kagg_t Kc_end = Kc_values[i_batch + 1] - 1;

    int iB = Kc_end >> 5;             // this is the index where the last bit pos is
    int last_bit_pos = Kc_end & 0x1f; // this is pos of last bit
    bool weight_output_if = (every > 0) ? ((i_batch + batch_start + 1) % every) == 0 : false;

    uint32_t bitwise = 0;
    if (weight_output_if) {
      // printf("WO SIGNAL [BS %d, EV %d]: B %d (bit pos %d)\n", batch_start, every, (int)i_batch,
      // last_bit_pos);
      bitwise = (uint32_t)0x00000001 << last_bit_pos;
    }
    atomicOr(&weight_output_signals[iB], bitwise);
  }
}

/****************************************************************************************************************/
/* Chopped Weight Output */

/*******************************************************************************/

// copy construcutor
template <typename T>
ChoppedWeightOutput<T>::ChoppedWeightOutput(const ChoppedWeightOutput<T> &other) {
  x_size_ = other.x_size_;
  d_size_ = other.d_size_;
  context_ = other.context_;
  par_ = other.par_;
  current_m_batch_ = other.current_m_batch_;

  cwo_counter_ = other.cwo_counter_;
  nwo_counter_ = other.nwo_counter_;
  swapped_choppers_ = other.swapped_choppers_;
  flexible_in_size_ = other.flexible_in_size_;

  if (other.dev_x_chopper_buffer_1_) {

    dev_x_chopper_buffer_1_ = RPU::make_unique<CudaArray<chop_t>>(*other.dev_x_chopper_buffer_1_);
    dev_x_chopper_buffer_2_ = RPU::make_unique<CudaArray<chop_t>>(*other.dev_x_chopper_buffer_2_);
    dev_d_chopper_buffer_1_ = RPU::make_unique<CudaArray<chop_t>>(*other.dev_d_chopper_buffer_1_);
    dev_d_chopper_buffer_2_ = RPU::make_unique<CudaArray<chop_t>>(*other.dev_d_chopper_buffer_2_);

    context_->synchronize();

    // reproduce swapped state
    if (swapped_choppers_) {
      x_chopper_in_ = dev_x_chopper_buffer_2_->getData();
      x_chopper_out_ = dev_x_chopper_buffer_1_->getData();
      d_chopper_in_ = dev_d_chopper_buffer_2_->getData();
      d_chopper_out_ = dev_d_chopper_buffer_1_->getData();
    } else {
      x_chopper_in_ = dev_x_chopper_buffer_1_->getData();
      x_chopper_out_ = dev_x_chopper_buffer_2_->getData();
      d_chopper_in_ = dev_d_chopper_buffer_1_->getData();
      d_chopper_out_ = dev_d_chopper_buffer_2_->getData();
    }
  }

  // don't copy switching probs or outputs
  x_switching_probs_ = nullptr;
  d_switching_probs_ = nullptr;

  dev_switching_probs_ = nullptr;
  dev_weight_output_signals_ = nullptr;
  weight_outputs_ = nullptr;
  dev_weight_output_in_chopper_ = nullptr;
  dev_weight_output_out_chopper_ = nullptr;
}

// copy assignment
template <typename T>
ChoppedWeightOutput<T> &ChoppedWeightOutput<T>::operator=(const ChoppedWeightOutput<T> &other) {

  ChoppedWeightOutput<T> tmp(other);
  swap(*this, tmp);
  context_->synchronize();
  return *this;
}

// move constructor
template <typename T> ChoppedWeightOutput<T>::ChoppedWeightOutput(ChoppedWeightOutput<T> &&other) {
  *this = std::move(other);
}

// move assignment
template <typename T>
ChoppedWeightOutput<T> &ChoppedWeightOutput<T>::operator=(ChoppedWeightOutput<T> &&other) {

  x_size_ = other.x_size_;
  d_size_ = other.d_size_;
  context_ = other.context_;
  par_ = std::move(other.par_);
  current_m_batch_ = other.current_m_batch_;

  cwo_counter_ = other.cwo_counter_;
  nwo_counter_ = other.nwo_counter_;
  swapped_choppers_ = other.swapped_choppers_;

  dev_x_chopper_buffer_1_ = std::move(other.dev_x_chopper_buffer_1_);
  dev_x_chopper_buffer_2_ = std::move(other.dev_x_chopper_buffer_2_);
  dev_d_chopper_buffer_1_ = std::move(other.dev_d_chopper_buffer_1_);
  dev_d_chopper_buffer_2_ = std::move(other.dev_d_chopper_buffer_2_);

  x_chopper_in_ = other.x_chopper_in_;
  other.x_chopper_in_ = nullptr;

  x_chopper_out_ = other.x_chopper_out_;
  other.x_chopper_out_ = nullptr;

  d_chopper_in_ = other.d_chopper_in_;
  other.d_chopper_in_ = nullptr;

  d_chopper_out_ = other.d_chopper_out_;
  other.d_chopper_out_ = nullptr;

  x_switching_probs_ = other.x_switching_probs_;
  other.x_switching_probs_ = nullptr;

  d_switching_probs_ = other.d_switching_probs_;
  other.d_switching_probs_ = nullptr;

  dev_switching_probs_ = std::move(other.dev_switching_probs_);
  dev_weight_output_signals_ = std::move(other.dev_weight_output_signals_);
  weight_outputs_ = nullptr;
  dev_weight_output_in_chopper_ = std::move(other.dev_weight_output_in_chopper_);
  dev_weight_output_out_chopper_ = std::move(other.dev_weight_output_out_chopper_);
  flexible_in_size_ = other.flexible_in_size_;
  return *this;
};

template <typename T>
ChoppedWeightOutput<T>::ChoppedWeightOutput(CudaContextPtr c, int x_size, int d_size)
    : context_(c), x_size_(x_size), d_size_(d_size){};

template <typename T> void ChoppedWeightOutput<T>::printToStream(std::stringstream &ss) const {
  ss << "\t counter:\t" << getCounter() << std::endl;
  ss << "\t batch start:\t" << getBatchStart() << std::endl;
  ss << "\t val start:\t" << getValStart() << std::endl;
  ss << "\t num wo:\t" << getNumWeightOutputs() << std::endl;
  ss << "\t flexible:\t" << getFlexibleInSize() << std::endl;
  par_.printToStream(ss);
  ss << std::endl;
};

template <typename T>
void ChoppedWeightOutput<T>::dumpExtra(RPU::state_t &extra, const std::string prefix) {
  RPU::state_t state;

  // don't handle maximizers (no states)
  RPU::insert(state, "current_m_batch", current_m_batch_);
  RPU::insert(state, "cwo_counter", cwo_counter_);
  RPU::insert(state, "nwo_counter", nwo_counter_);
  RPU::insert(state, "flexible_in_size", flexible_in_size_);
  RPU::insert(state, "swapped_choppers", swapped_choppers_);
  RPU::insert(state, "dev_switching_probs", dev_switching_probs_);
  RPU::insert(state, "dev_weight_output_out_chopper", dev_weight_output_out_chopper_);
  RPU::insert(state, "dev_weight_output_in_chopper", dev_weight_output_in_chopper_);
  RPU::insert(state, "dev_weight_output_signals", dev_weight_output_signals_);
  RPU::insert(state, "dev_x_chopper_buffer_1", dev_x_chopper_buffer_1_);
  RPU::insert(state, "dev_x_chopper_buffer_2", dev_x_chopper_buffer_2_);
  RPU::insert(state, "dev_d_chopper_buffer_1", dev_d_chopper_buffer_1_);
  RPU::insert(state, "dev_d_chopper_buffer_2", dev_d_chopper_buffer_2_);

  RPU::insertWithPrefix(extra, state, prefix);
}

template <typename T>
void ChoppedWeightOutput<T>::loadExtra(
    const RPU::state_t &extra, const std::string prefix, bool strict) {

  using V = std::vector<T>;
  auto state = RPU::selectWithPrefix(extra, prefix);

  RPU::load(state, "current_m_batch", current_m_batch_, strict);
  RPU::load(state, "cwo_counter", cwo_counter_, strict);
  RPU::load(state, "nwo_counter", nwo_counter_, strict);
  RPU::load(state, "flexible_in_size", flexible_in_size_, strict);
  RPU::load(state, "swapped_choppers", swapped_choppers_, strict);
  RPU::load(this->context_, state, "dev_switching_probs", dev_switching_probs_, strict);
  RPU::load(
      this->context_, state, "dev_weight_output_out_chopper", dev_weight_output_out_chopper_,
      strict);
  RPU::load(
      this->context_, state, "dev_weight_output_in_chopper", dev_weight_output_in_chopper_, strict);
  RPU::load(this->context_, state, "dev_weight_output_signals", dev_weight_output_signals_, strict);
  RPU::load(this->context_, state, "dev_x_chopper_buffer_1", dev_x_chopper_buffer_1_, strict);
  RPU::load(this->context_, state, "dev_x_chopper_buffer_2", dev_x_chopper_buffer_2_, strict);
  RPU::load(this->context_, state, "dev_d_chopper_buffer_1", dev_d_chopper_buffer_1_, strict);
  RPU::load(this->context_, state, "dev_d_chopper_buffer_2", dev_d_chopper_buffer_2_, strict);
}

/******************************************************************************************************************/

template <typename T> int ChoppedWeightOutput<T>::getValStart() const {

  if (par_.every <= 0) {
    return 0;
  }

  if (par_.use_columns) {
    return ((cwo_counter_ - current_m_batch_) / par_.every) % x_size_;
  } else {
    return ((cwo_counter_ - current_m_batch_) / par_.every) % d_size_;
  }
}

template <typename T> int ChoppedWeightOutput<T>::getBatchStart() const {
  // directly BEFORE this batch sub-index the first transfer
  // happens. So 0 means it happend in the last round at the end.

  return par_.every ? (cwo_counter_ - current_m_batch_) % par_.every : 0;
};

template <typename T> int ChoppedWeightOutput<T>::getNumWeightOutputs() const {
  // if batch_start falls on the beginning, it is assumed that the
  // weight output has happened at the end of the previous round
  if (par_.every <= 0) {
    return 0;
  }
  return cwo_counter_ / par_.every - (cwo_counter_ - current_m_batch_) / par_.every;
};

template <typename T> void ChoppedWeightOutput<T>::releaseBuffers() {
  // needs to be called after the makeWeightOutputChoppers to release the shared buffers
  if (weight_outputs_ != nullptr && getNumWeightOutputs() > 0) {
    context_->template releaseSharedBuffer<T>(RPU_BUFFER_CWO);
  }
}

template <typename T>
void ChoppedWeightOutput<T>::makeWeightOutputChoppers(const BitLineMaker<T> *blm) {

  if (!par_.isEnabled() || blm == nullptr) {
    return;
  }
  // prepares the BLM to be used with the CWO pulsed update kernel
  // -- swaps in/out choppers and also advances counter
  // -- is called by makeCounts if chopppers are requested
  // -- generates weight output signals
  // -- generates switching probls
  // -- sets current m_batch and advances counter (BEFORE) the actual computation.

  current_m_batch_ = blm->getCurrentMBatch();

  if (dev_x_chopper_buffer_1_ == nullptr || dev_x_chopper_buffer_1_->getSize() < x_size_) {

    RPU_GET_CUDA_BUFFER(context_, chop_t, dev_x_chopper_buffer_1_, x_size_);
    RPU_GET_CUDA_BUFFER(context_, chop_t, dev_x_chopper_buffer_2_, x_size_);
    RPU_GET_CUDA_BUFFER(context_, chop_t, dev_d_chopper_buffer_1_, d_size_);
    RPU_GET_CUDA_BUFFER(context_, chop_t, dev_d_chopper_buffer_2_, d_size_);

    dev_x_chopper_buffer_1_->setConst(1);
    dev_d_chopper_buffer_1_->setConst(1);
    dev_x_chopper_buffer_2_->setConst(1);
    dev_d_chopper_buffer_2_->setConst(1);

    x_chopper_in_ = dev_x_chopper_buffer_1_->getData();
    x_chopper_out_ = dev_x_chopper_buffer_2_->getData();
    d_chopper_in_ = dev_d_chopper_buffer_1_->getData();
    d_chopper_out_ = dev_d_chopper_buffer_2_->getData();

    swapped_choppers_ = false;
    cwo_counter_ = 0;
    nwo_counter_ = 0;
  }

  // need to swap input and output
  if (swapped_choppers_) {
    x_chopper_in_ = dev_x_chopper_buffer_1_->getData();
    x_chopper_out_ = dev_x_chopper_buffer_2_->getData();
    d_chopper_in_ = dev_d_chopper_buffer_1_->getData();
    d_chopper_out_ = dev_d_chopper_buffer_2_->getData();

    swapped_choppers_ = false;

  } else {
    x_chopper_in_ = dev_x_chopper_buffer_2_->getData();
    x_chopper_out_ = dev_x_chopper_buffer_1_->getData();
    d_chopper_in_ = dev_d_chopper_buffer_2_->getData();
    d_chopper_out_ = dev_d_chopper_buffer_1_->getData();
    swapped_choppers_ = true;
  }

  // update counter BEFORE running the kernel!!
  cwo_counter_ += current_m_batch_;

  if (blm->getCurrentLR() == (T)0.0) {
    return;
  }

  int max_weight_outputs = par_.every > 0 ? (current_m_batch_ + par_.every - 1) / par_.every : 0;
  int sw_size = (x_size_ + d_size_) * max_weight_outputs;
  if (max_weight_outputs > 0 && (par_.in_chop_random || par_.out_chop_prob > (T)0.0)) {
    RPU_GET_CUDA_BUFFER(context_, float, dev_switching_probs_, sw_size);
  }

  x_switching_probs_ = nullptr;
  d_switching_probs_ = nullptr;

  if (max_weight_outputs > 0) {
    int n_weight_outputs = getNumWeightOutputs();
    if (n_weight_outputs > 0) {
      nwo_counter_ += n_weight_outputs; // BEFORE applying
      RPU_GET_CUDA_BUFFER(
          context_, chop_t, dev_weight_output_out_chopper_, n_weight_outputs * getOutSize());
      RPU_GET_CUDA_BUFFER(context_, chop_t, dev_weight_output_in_chopper_, n_weight_outputs);

      if (par_.in_chop_random || par_.out_chop_prob > (T)0.0) {
        context_->randUniform(dev_switching_probs_->getData(), sw_size);

        x_switching_probs_ = dev_switching_probs_->getData();
        d_switching_probs_ = dev_switching_probs_->getData() + x_size_ * max_weight_outputs;
      }
      // out put is done in the correct row/col of a weight matrix
      // (where the non-output rows/cols are random). In case that n_wo
      // is larger than in_size then a second weight matrix is populated
      // and so on
      weight_outputs_ = context_->template getSharedBuffer<T>(RPU_BUFFER_CWO, getWODataSize());
    }
  }

  if (blm->usesBo64() && blm->getCurrentUBLM()) {

    RPU_GET_CUDA_BUFFER(
        context_, uint32_t, dev_weight_output_signals_, blm->getBo64Batch(current_m_batch_));

    // zero signals
    dev_weight_output_signals_->setConst(0);

    // compute weight_output signals
    int nthreads = context_->getNThreads();
    int nblocks = context_->getNBlocks(current_m_batch_);
    kernelCreateWeightOutputSignalsUBLMBO64<<<nblocks, nthreads, 0, context_->getStream()>>>(
        dev_weight_output_signals_->getData(), current_m_batch_, getBatchStart(), par_.every,
        blm->getUmh()->getKcValueData());
  }
}

template class ChoppedWeightOutput<float>;
#ifdef RPU_USE_DOUBLE
template class ChoppedWeightOutput<double>;
#endif
#ifdef RPU_USE_FP16
template class ChoppedWeightOutput<half_t>;
#endif

} // namespace RPU
