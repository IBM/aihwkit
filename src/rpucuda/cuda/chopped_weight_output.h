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

#include "bit_line_maker.h"
#include "cuda_util.h"

namespace RPU {
template <typename T> class BitLineMaker;

template <typename T> struct ChoppedWeightOutputParameter {

  T in_chop_prob = 0.0;    // once applied each chopper/weight output event
                           // for single value. Values applied
                           // sequentially and warped
  T out_chop_prob = 0.0;   // applied once to all out values each
                           // completion of the warping cycle of the in
                           // values
  bool use_columns = true; // true means forward read
  int every = 0;           // chopper/weight output event modulues

  bool in_chop_random = true; // whether apply random chopper for in chopping (otherwise regular)

  void print() const {
    std::stringstream ss;
    printToStream(ss);
    std::cout << ss.str();
  };
  void printToStream(std::stringstream &ss) const {
    ss << "\t in_chop_prob:\t" << in_chop_prob << std::endl;
    ss << "\t out_chop_prob:\t" << out_chop_prob << std::endl;
    ss << "\t every:\t" << every << std::endl;
    ss << "\t use_columns:\t" << std::boolalpha << use_columns << std::endl;
    ss << "\t in_chop_random:\t" << std::boolalpha << in_chop_random << std::endl;
  };

  inline bool isEnabled() const {
    return in_chop_prob > (T)0.0 || out_chop_prob > (T)0.0 || every > 0;
  };
};

template <typename T> class ChoppedWeightOutput {

public:
  explicit ChoppedWeightOutput(CudaContextPtr c, int x_size, int d_size);

  ChoppedWeightOutput(){};

  ~ChoppedWeightOutput() = default;
  ChoppedWeightOutput(const ChoppedWeightOutput<T> &);
  ChoppedWeightOutput<T> &operator=(const ChoppedWeightOutput<T> &);
  ChoppedWeightOutput(ChoppedWeightOutput<T> &&);
  ChoppedWeightOutput<T> &operator=(ChoppedWeightOutput<T> &&);

  friend void swap(ChoppedWeightOutput<T> &a, ChoppedWeightOutput<T> &b) noexcept {
    using std::swap;
    swap(a.context_, b.context_);
    swap(a.x_size_, b.x_size_);
    swap(a.d_size_, b.d_size_);
    swap(a.par_, b.par_);
    swap(a.current_m_batch_, b.current_m_batch_);
    swap(a.cwo_counter_, b.cwo_counter_);
    swap(a.nwo_counter_, b.nwo_counter_);
    swap(a.swapped_choppers_, b.swapped_choppers_);
    swap(a.x_chopper_in_, b.x_chopper_in_);
    swap(a.x_chopper_out_, b.x_chopper_out_);
    swap(a.d_chopper_in_, b.d_chopper_in_);
    swap(a.d_chopper_out_, b.d_chopper_out_);

    swap(a.x_switching_probs_, b.x_switching_probs_);
    swap(a.d_switching_probs_, b.d_switching_probs_);

    swap(a.dev_switching_probs_, b.dev_switching_probs_);
    swap(a.dev_weight_output_signals_, b.dev_weight_output_signals_);
    swap(a.dev_weight_output_in_chopper_, b.dev_weight_output_in_chopper_);
    swap(a.dev_weight_output_out_chopper_, b.dev_weight_output_out_chopper_);

    swap(a.dev_x_chopper_buffer_1_, b.dev_x_chopper_buffer_1_);
    swap(a.dev_x_chopper_buffer_2_, b.dev_x_chopper_buffer_2_);
    swap(a.dev_d_chopper_buffer_1_, b.dev_d_chopper_buffer_1_);
    swap(a.dev_d_chopper_buffer_2_, b.dev_d_chopper_buffer_2_);
    swap(a.flexible_in_size_, b.flexible_in_size_);
  }

  void print() const {
    std::stringstream ss;
    printToStream(ss);
    std::cout << ss.str();
  };
  void printToStream(std::stringstream &ss) const;
  void makeWeightOutputChoppers(const BitLineMaker<T> *blm);
  void releaseBuffers();

  void dumpExtra(RPU::state_t &extra, const std::string prefix);
  void loadExtra(const RPU::state_t &extra, const std::string prefix, bool strict);

  inline void setPar(const ChoppedWeightOutputParameter<T> &par) { par_ = par; };
  inline void setInChopRandom(bool in_chop_random) { par_.in_chop_random = in_chop_random; };
  inline void setInChopProb(T in_chop_prob) { par_.in_chop_prob = in_chop_prob; };
  inline const ChoppedWeightOutputParameter<T> &getPar() const { return par_; };

  inline int getCounter() const { return cwo_counter_; };
  inline int getCurrentMBatch() const { return current_m_batch_; };
  inline void setCounter(uint64_t counter) { cwo_counter_ = counter; };
  inline void setFlexibleInSize(bool flex) { flexible_in_size_ = flex; };
  inline bool getFlexibleInSize() const { return flexible_in_size_; };

  inline int getOutSize() const { return par_.use_columns ? d_size_ : x_size_; };
  inline int getInSize() const { return par_.use_columns ? x_size_ : d_size_; };
  inline int getWODataSize() const {
    return flexible_in_size_ ? getNumWeightOutputs() * getOutSize()
                             : (getNumWeightOutputs() / getInSize() + 1) * x_size_ * d_size_;
  };
  int getBatchStart() const;
  int getValStart() const;
  int getNumWeightOutputs() const;
  inline int getNWOCounter() const { return nwo_counter_; };
  inline void setEvery(int every) {
    // std::cout << "Setting transfer every to " << every << std::endl;
    par_.every = every;
  };
  inline int getEvery() const { return par_.every; };
  inline chop_t *getXChopperInData() const { return x_chopper_in_; };
  inline chop_t *getXChopperOutData() const { return x_chopper_out_; };
  inline chop_t *getDChopperInData() const { return d_chopper_in_; };
  inline chop_t *getDChopperOutData() const { return d_chopper_out_; };
  inline float *getXSwitchingProbData() const {
    return (!par_.in_chop_random && par_.use_columns) ? nullptr : x_switching_probs_;
  };
  inline float *getDSwitchingProbData() const {
    return (!par_.in_chop_random && !par_.use_columns) ? nullptr : d_switching_probs_;
  };

  inline T *getWeightOutputData() const { return weight_outputs_; };

  inline chop_t *getWeightOutputOutChopperData() const {
    return dev_weight_output_out_chopper_ ? dev_weight_output_out_chopper_->getData() : nullptr;
  };
  inline chop_t *getWeightOutputInChopperData() const {
    return dev_weight_output_in_chopper_ ? dev_weight_output_in_chopper_->getData() : nullptr;
  };

  inline uint32_t *getWeightOutputSignalsData() const {
    return dev_weight_output_signals_ ? dev_weight_output_signals_->getData() : nullptr;
  };

  inline void printWeightOutputInChopper() const {
    if (dev_weight_output_in_chopper_) {
      dev_weight_output_in_chopper_->printValues();
    }
  };

private:
  CudaContextPtr context_ = nullptr;
  int x_size_ = 0;
  int d_size_ = 0;
  int current_m_batch_ = 0;
  ChoppedWeightOutputParameter<T> par_;

  uint64_t cwo_counter_ = 0;
  uint64_t nwo_counter_ = 0;
  bool swapped_choppers_ = false;
  bool flexible_in_size_ = false;

  chop_t *x_chopper_in_ = nullptr;
  chop_t *x_chopper_out_ = nullptr;
  chop_t *d_chopper_in_ = nullptr;
  chop_t *d_chopper_out_ = nullptr;
  float *x_switching_probs_ = nullptr;
  float *d_switching_probs_ = nullptr;
  T *weight_outputs_ = nullptr;

  std::unique_ptr<CudaArray<float>> dev_switching_probs_ = nullptr;
  std::unique_ptr<CudaArray<uint32_t>> dev_weight_output_signals_ = nullptr;

  std::unique_ptr<CudaArray<chop_t>> dev_weight_output_in_chopper_ = nullptr;
  std::unique_ptr<CudaArray<chop_t>> dev_weight_output_out_chopper_ = nullptr;
  std::unique_ptr<CudaArray<chop_t>> dev_x_chopper_buffer_1_ = nullptr;
  std::unique_ptr<CudaArray<chop_t>> dev_x_chopper_buffer_2_ = nullptr;
  std::unique_ptr<CudaArray<chop_t>> dev_d_chopper_buffer_1_ = nullptr;
  std::unique_ptr<CudaArray<chop_t>> dev_d_chopper_buffer_2_ = nullptr;
};

} // namespace RPU
