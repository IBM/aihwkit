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
#include "weight_clipper.h"
#include "weight_drifter.h"
#include "weight_modifier.h"
#include "weight_remapper.h"
#include <cfenv>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <unordered_map>

// #pragma STDC FENV_ACCESS ON

#define USE_LOOPED_MATRIX_FORWARD(T)                                                               \
  inline void forwardMatrix(                                                                       \
      const T *X_input, T *D_output, int m_batch, bool x_trans, bool d_trans, bool is_test)        \
      override {                                                                                   \
    RPUAbstract<T>::forwardMatrix(X_input, D_output, m_batch, x_trans, d_trans, is_test);          \
  }

#define USE_LOOPED_MATRIX_BACKWARD(T)                                                              \
  inline void backwardMatrix(                                                                      \
      const T *D_input, T *X_output, int m_batch, bool d_trans = false, bool x_trans = false)      \
      override {                                                                                   \
    RPUAbstract<T>::backwardMatrix(D_input, X_output, m_batch, d_trans, x_trans);                  \
  }

#define USE_LOOPED_MATRIX_UPDATE(T)                                                                \
  inline void updateMatrix(                                                                        \
      const T *X_input, const T *D_input, int m_batch, bool x_trans = false, bool d_trans = false) \
      override {                                                                                   \
    RPUAbstract<T>::updateMatrix(X_input, D_input, m_batch, x_trans, d_trans);                     \
  }

#define USE_ALL_LOOPED_MATRIX_VERSIONS(T)                                                          \
  USE_LOOPED_MATRIX_FORWARD(T);                                                                    \
  USE_LOOPED_MATRIX_BACKWARD(T);                                                                   \
  USE_LOOPED_MATRIX_UPDATE(T)

#define NOT_SUPPORTED RPU_FATAL("Not supported RPU IO Vector type");

namespace RPU {

/******************************************************************************/
/* RPU Abstract */
template <typename T> class RPUSimple;

template <typename T> class RPUAbstract {

public:
  RPUAbstract() : x_size_(0), d_size_(0) { std::fesetround(FE_TONEAREST); };
  explicit RPUAbstract(int x_size, int d_size) : x_size_(x_size), d_size_(d_size) {
    std::fesetround(FE_TONEAREST);
  };
  virtual ~RPUAbstract() = default;

  RPUAbstract(const RPUAbstract<T> &) = default;
  RPUAbstract<T> &operator=(const RPUAbstract<T> &) = default;
  RPUAbstract(RPUAbstract<T> &&) = default;
  RPUAbstract<T> &operator=(RPUAbstract<T> &&) = default;

  friend void swap(RPUAbstract &a, RPUAbstract &b) noexcept {
    using std::swap;
    swap(a.x_size_, b.x_size_);
    swap(a.d_size_, b.d_size_);
    swap(a.learning_rate_, b.learning_rate_);
  }

  void disp() const {
    std::stringstream ss;
    printToStream(ss);
    std::cout << ss.str();
  };
  std::string getDataTypeName() const;
  virtual void printToStream(std::stringstream &ss) const;
  virtual void setLearningRate(T lrate) { learning_rate_ = lrate; };

  virtual void forwardVector(const T *x_input, T *d_output, int x_inc, int d_inc, bool is_test) = 0;
  virtual void backwardVector(const T *d_input, T *x_output, int d_inc = 1, int x_inc = 1) = 0;
  virtual void updateVector(const T *x_input, const T *d_input, int x_inc = 1, int d_inc = 1) = 0;

protected:
  virtual void forwardVectorBias(
      const T *x_input_without_bias, T *d_output, int x_inc, int d_inc, bool is_test) = 0;
  virtual void
  backwardVectorBias(const T *d_input, T *x_output_without_bias, int d_inc = 1, int x_inc = 1) = 0;
  virtual void updateVectorBias(
      const T *x_input_without_bias, const T *d_input, int x_inc = 1, int d_inc = 1) = 0;

  // matrix (batch) versions (expects batch to be second dimensions (first is contiguous))
  // simple implementation that loops the vector version
  virtual void forwardMatrix(
      const T *X_input, T *D_output, int m_batch, bool x_trans, bool d_trans, bool is_test);
  virtual void backwardMatrix(
      const T *D_input, T *X_output, int m_batch, bool d_trans = false, bool x_trans = false);
  virtual void updateMatrix(
      const T *X_input, const T *D_input, int m_batch, bool x_trans = false, bool d_trans = false);

  virtual void forwardMatrixBias(
      const T *X_input_without_bias,
      T *D_output,
      int m_batch,
      bool x_trans,
      bool d_trans,
      bool is_test);
  virtual void backwardMatrixBias(
      const T *D_input,
      T *X_output_without_bias,
      int m_batch,
      bool d_trans = false,
      bool x_trans = false);
  virtual void updateMatrixBias(
      const T *X_input_without_bias,
      const T *D_input,
      int m_batch,
      bool x_trans = false,
      bool d_trans = false);

public:
  int getXSize() const { return x_size_; };
  int getDSize() const { return d_size_; };

  T getLearningRate() const { return learning_rate_; };

  virtual void finishUpdateCalculations(){};
  virtual void finishAllCalculations(){};
  virtual void makeUpdateAsync(){};

protected:
  int x_size_ = 0;
  int d_size_ = 0;
  T learning_rate_ = (T)0.0;
};

template <typename T> struct FlickerParameter {

  int n = 64;                      // pink diffusion
  T r = (T)sqrt(sqrt((float)2.0)); // frequency ratio
  T q = (T)0.5;                    // flip prob
  T h = (T)0.0; // reset time constant: if set to flicker_r^k, reset will mostly effect the first k
                // traps
  bool wreset = false;
  T wreset_tol = (T)1e-5;

  void printToStream(std::stringstream &ss) const;
  void print() const {
    std::stringstream ss;
    printToStream(ss);
    std::cout << ss.str();
  };
};

template <typename T> struct SimpleMetaParameter {

  SimpleMetaParameter() { drift.setSimpleDrift(); }

  T diffusion = (T)0.0;
  T lifetime = (T)0.0;
  T alpha_std = (T)0.0; // one-time relative error when setting alpha scale
  bool use_delayed_update = false;

  FlickerParameter<T> flicker;
  DriftParameter<T> drift;

  virtual void printToStream(std::stringstream &ss) const;
  void print() const {
    std::stringstream ss;
    printToStream(ss);
    std::cout << ss.str();
  };

  RPUSimple<T> *createRPUArray(int x_size, int d_size) {
    auto *rpu = new RPUSimple<T>(x_size, d_size);
    rpu->populateParameter(this);
    rpu->setWeightsUniformRandom(-0.1, 0.1);
    rpu->setLearningRate(0.1);
    return rpu;
  };
};

/******************************************************************************/
/* RPU Simple */

template <typename T> class RPUSimple : public RPUAbstract<T> {

public:
  RPUSimple(){};
  RPUSimple(int x_size, int d_size);
  ~RPUSimple();

  RPUSimple(const RPUSimple<T> &);
  RPUSimple<T> &operator=(const RPUSimple<T> &);
  RPUSimple(RPUSimple<T> &&) noexcept;
  RPUSimple<T> &operator=(RPUSimple<T> &&) noexcept;

  friend void swap(RPUSimple<T> &a, RPUSimple<T> &b) noexcept {

    using std::swap;
    swap(static_cast<RPUAbstract<T> &>(a), static_cast<RPUAbstract<T> &>(b));

    swap(a.rng_, b.rng_);
    swap(a.rw_rng_, b.rw_rng_);
    swap(a.par_, b.par_);

    swap(a.weights_, b.weights_);
    swap(a.shared_weights_if_, b.shared_weights_if_);

    swap(a.weights_buffer_, b.weights_buffer_);
    swap(a.use_delayed_update_, b.use_delayed_update_);

    swap(a.temp_x_vector_bias_, b.temp_x_vector_bias_);
    swap(a.temp_x_matrix_bias_, b.temp_x_matrix_bias_);
    swap(a.temp_tensor_, b.temp_tensor_);

    swap(a.flicker_states_, b.flicker_states_);
    swap(a.flicker_probs_, b.flicker_probs_);

    swap(a.matrix_indices_, b.matrix_indices_);
    swap(a.matrix_indices_set_, b.matrix_indices_set_);

    swap(a.wdrifter_, b.wdrifter_);

    swap(a.wremapper_, b.wremapper_);
    swap(a.wclipper_, b.wclipper_);

    swap(a.fb_weights_, b.fb_weights_);
    swap(a.delta_weights_extern_, b.delta_weights_extern_);

    swap(a.fb_weight_modifier_, b.fb_weight_modifier_);
    swap(a.last_update_m_batch_, b.last_update_m_batch_);

    swap(a.bwd_alpha_, b.bwd_alpha_);
    swap(a.fwd_alpha_, b.fwd_alpha_);
  }

  /*populate parameter is the main entry to set all parameters of
    the RPU. Each RPU type is expected to implement it's sepcifica
    populate parameter routine. It is not virtual because it accepts
    its own meta parameter(s)*/
  void populateParameter(SimpleMetaParameter<T> *p) { par_ = *p; }

  virtual const SimpleMetaParameter<T> &getPar() const { return par_; };

  void dispParameter() const {
    std::stringstream ss;
    printParametersToStream(ss);
    std::cout << ss.str();
  };
  virtual void printParametersToStream(std::stringstream &ss) const;

  void setLearningRate(T lrate) override;

  void printToStream(std::stringstream &ss) const override;

  /* This is to set the random seed. This is currently, however, NOT
     causing all seeds to be set. Some seeds remain random!*/
  virtual void setRandomSeed(unsigned int seed);
  virtual void setWeightsUniformRandom(T min_value, T max_value);

  /* This scales the weights by applying an (digital) output scale
     which represents the abs(max) of the weights*/
  void setWeightsWithAlpha(const T *weightsptr, T assumed_wmax);

  /* setWeights* set the weights perfectly*/
  virtual void setWeights(const T *weightsptr);
  void
  setWeightsAndBias(const T *weightsptr, const T *biasptr, bool real_if = false, int n_loops = 1);
  void setWeightsAndBiasWithAlpha(
      const T *weightsptr, const T *biasptr, T assumed_wmax, bool real_if = false, int n_loops = 1);

  /* setSharedWeights can be used to provide an external weight
     pointer to handle the memory of the weights. Note that weights
     for CPU are always stored in row-major format.*/
  virtual void setSharedWeights(T *weightsptr);
  inline bool getSharedWeightsIf() { return shared_weights_if_; };

  /* access to the CPU weight ptr*/
  inline T **getWeightsPtr() const { return this->weights_; };
  virtual T **getWeights() { return getWeightsPtr(); };

  /* get weights by copying weights to given pointer. Might
     implicitly copy to host (CPU) weights*/
  virtual void getWeights(T *weightsptr) const;

  /* methods to get/set the weights using read-write-verify cycles
     with the current definition of analog forward/update*/
  virtual void getWeightsReal(T *weightsptr) { this->getWeights(weightsptr); };
  virtual void setWeightsReal(const T *weightsptr, int n_loops = 1) {
    this->setWeights(weightsptr);
  };

  /* Returns the DW in place (subtracting the given weights from the
     actual weights and setting the actual weights to the given
     weights). This is for connecting the in-place weight update to
     some other framework that might expect the DW and broadcast
     it. CAUTION: DW cannot be modified and then applied to the W
     again, that would ciircumvent the analog update!*/
  virtual void getAndResetWeightUpdate(T *prev_weights_and_dw_out, T scale = 1.0);
  virtual void applyWeightUpdate(T *dw_and_current_weights_out);

  /* print the weights to stdout*/
  virtual void printWeights(int x_count, int d_count);

  /* Device parameter are parameters per weight element to implement
     systematic noise sources and non-idealities, for instance
     device-to-device variation of the update step size. These are
     interface function to get/set the current device parameters,
     which usually are drawn during instantiation of the RPU object
     based on parameters defining their probabilty distributions. */
  virtual void getDeviceParameterNames(std::vector<std::string> &names) const { names.clear(); };
  virtual void getDeviceParameter(std::vector<T *> &data_ptrs){};
  virtual void setDeviceParameter(const std::vector<T *> &data_ptrs){};

  /* These dumps extra state vectors that are not returned by
     getDeviuceParameters or getWeights*/
  virtual void dumpExtra(RPU::state_t &extra, const std::string prefix);
  virtual void loadExtra(const RPU::state_t &extra, const std::string prefix, bool strict);

  virtual int getHiddenUpdateIdx() const { return 0; };
  virtual void setHiddenUpdateIdx(int idx){};

  /* Decaying the weights once. Alpha can be a factor additionally
     applied to the current decay rate*/
  virtual void decayWeights(bool bias_no_decay);
  virtual void decayWeights(T alpha, bool bias_no_decay);

  /* Clip weights once. Uses the weight clipper to clip weights in
     some manner, only for HWA training.*/
  virtual void clipWeights(const T clip);
  virtual void clipWeights(const WeightClipParameter &wclpar);

  /* Applying a potential reset with given probabilties to a selection of columns */
  virtual void resetCols(int start_col, int n_cols, T reset_prob) {
    RPU_FATAL("Not supported for RPU Simple");
  };

  /* Applying an Gaussian diffusion onto the weights.*/
  virtual void diffuseWeights();

  /* conductance drift */
  virtual void driftWeights(T time_since_last_call);

  /* 1/f pink noise process (flicker noise) */
  virtual void diffuseWeightsPink();
  uint64_t *initFlickerStates();

  /* Remapping the weights. Scales and Biases need to be handled
     from outside for now*/
  virtual void remapWeights(const WeightRemapParameter &wrmpar, T *scales, T *biases = nullptr);

  /* Using stochastic weight averaging, together with remapping [thus the scales/biases]*/
  virtual bool swaWeights(
      const WeightRemapParameter &wrmpar,
      T *swa_weights,
      uint64_t iter,
      T *scales = nullptr,
      T *biases = nullptr);

  /* Modify forward/backward weights (while keeping the update
     weights to a reference). This essentially copies the weight
     matrix and modifies this based on the given parameters
     (e.g. drop connect prob). The modiifiied weight matrix is used
     for the forward and backward pass, but the update is still done
     on the reference weights. Each call a new copied matrix will by
     generated based on the reference weights. Usually, during
     testing, the referenece weiight matrix is used instead (can be
     selected by settiing wmpar appropriately).  */
  virtual void modifyFBWeights(const WeightModifierParameter<T> &wmpar);

  /* Delayed update support. If use_delayed_update is turned on when
     constructing the RPU, then it only uses the buffered weight for
     update. All other weight changes (such as decay or noise,
     forward/backward, etc) are performed on the non-buffered
     (actual) weights. NOTE: it is up to the user to call
     Apply_Delayed_Update after Forward/Backward/Update cycle, which
     copies the buffered weights to the "actual" weights. THIS
     SHOULD BE CALLED BEFORE THE WEIGHT DECAY ETC OPERATORS! */
  bool isDelayedUpdate() const { return getPar().use_delayed_update; };
  void applyDelayedWeights(); // uses Copy_from/to _weight_buffer

  /* If the alpha options are used, then FORWARD AND BACKWARD
     compute a SCALED version of the forward and backward observing
     the setting of fwd_alpha_ and bwd_alpha_. UPDATE uses a LR
     scaling of 1/bwd_alpha */

  T getAlphaLearningRate() const;

  void setFwdAlpha(const T fwd_alpha, bool with_noise = true);
  void setBwdAlpha(const T bwd_alpha, bool with_noise = true);
  void setAlphaScale(const T alpha);
  inline T getFwdAlpha() { return fwd_alpha_; };
  inline T getBwdAlpha() { return bwd_alpha_; };

  /* public interface for allowing computing of DW. Need to be set
     before calling update. Note: ONLY useful for HWA training, not
     for analog training. This can be called before an update is
     made to use a the given weight pointer as weight storage. beta
     is the GEMM beta: W = alpha*X*D + beta*W. Thus setting beta=1
     means that only DW is stored in w */
  virtual void setDeltaWeights(T *dw_extern);
  virtual T *getDeltaWeights() const { return delta_weights_extern_[0]; };

  virtual void setVerbosityLevel(int verbose){};

  /* public interfaces for forward/backward/update. Format is
     expected in x-major order. However, the batch dimension comes
     first iif x_trans or d_trans is set to true */
  void forward(
      const T *X_input,
      T *D_output,
      bool bias = false,
      int m_batch = 1,
      bool x_trans = false,
      bool d_trans = false,
      bool is_test = false);
  void backward(
      const T *D_input,
      T *X_output,
      bool bias = false,
      int m_batch = 1,
      bool d_trans = false,
      bool x_trans = false);
  void update(
      const T *X_input,
      const T *D_input,
      bool bias = false,
      int m_batch = 1,
      bool x_trans = false,
      bool d_trans = false);

  /* public interfaces for forward/backward/update with additional
     3rd dimension. trans means that the major order is (lowest
     first) m_batch, x, dim3, otherwise x, m_batch, dim3. Correct
     permutations of dim3 are applied*/
  void forwardTensor(
      const T *X_input, T *D_output, bool bias, int m_batch, int dim3, bool trans, bool is_test);
  void backwardTensor(const T *D_input, T *X_output, bool bias, int m_batch, int dim3, bool trans);
  void
  updateTensor(const T *X_input, const T *D_input, bool bias, int m_batch, int dim3, bool trans);

  /* Indexed interfaces can be used to implement fast convolutions on GPU*/
  FORCE_INLINE void setMatrixIndices(int *indices) {
    this->matrix_indices_set_ = true;
    this->matrix_indices_ = indices;
  };

  FORCE_INLINE int *getMatrixIndices() {
    if (this->matrix_indices_set_) {
      return this->matrix_indices_;
    } else {
      RPU_FATAL("Matrix indices not set yet!");
    }
  };
  FORCE_INLINE bool hasMatrixIndices() { return this->matrix_indices_set_; };

  virtual void forwardIndexed(
      const T *X_input,
      T *D_output,
      int total_input_size,
      int m_batch,
      int dim3,
      bool trans,
      bool is_test);

  virtual void backwardIndexed(
      const T *D_input, T *X_output, int total_output_size, int m_batch, int dim3, bool trans);
  virtual void updateIndexed(
      const T *X_input, const T *D_input, int total_input_size, int m_batch, int dim3, bool trans);

  virtual void forwardIndexedSlice(
      const T *X_input,
      T *D_output,
      int total_input_size,
      int m_batch,
      int dim3,
      bool trans,
      int m_batch_slice,
      const int *batch_indices,
      bool is_test);
  virtual void backwardIndexedSlice(
      const T *D_input,
      T *X_output,
      int total_output_size,
      int m_batch,
      int dim3,
      bool trans,
      int m_batch_slice,
      const int *batch_indices);
  virtual void updateIndexedSlice(
      const T *X_input,
      const T *D_input,
      int total_input_size,
      int m_batch,
      int dim3,
      bool trans,
      int m_batch_slice,
      const int *batch_indices);

  virtual ContextPtr getContext() const { return nullptr; };
  virtual bool hasInternalContext() const { return true; };
  virtual std::vector<uint64_t> getPulseCounters() const { return std::vector<uint64_t>(); }

protected:
  /* for specialized forward/backward. To be used in any forward pass
     Note: we do not test with specialized forward/backward
     weights. This is in most cases what one wants, as specialized
     FB weights is meant to be a kind of hardware-aware training and
     during testing one usually assumes that one uses inference
     hardware (which does not support FB weights modifier). Also eg
     drop connections during testing makes no sense.

     However, sometimes we might be interested in test-error with
     special FB weights, eg in case of discretized weights. In this
     case the user needs to explicitely use enable_during_test */
  T **getFBWeights(bool is_test) const;

  /* This is called from the Update routines to check which weight
     is used for calculation. If dw is defined, then it will use the
     DW mode, meaning that it will write into delta_weights the DW
     and keep the weights. For HW RPU models that might include
     first using W and then writing the difference to dW

     In case of delayed update is also returns the buffered weights
     instead of the "actual" weight. Combination of external weights
     and buffers are not possible */
  T **getUpWeights();

  /* for beta GEMM during update. 0 means W=DW, 1 means W += DW */
  T getUpBeta() const;

  /* This is to enable an additional weight buffer (for "delayed" update)*/
  virtual void copyWeightsFromBuffer();
  virtual void copyWeightsToBuffer();
  inline T **getWeightsBuffer() const { return weights_buffer_; };

  /* when overriding copy methods below, _Matrix_Bias can be used in derived */
  virtual T *copyToMatrixBiasBuffer(const T *X_input_without_bias, int m_batch, bool x_trans);
  virtual void
  copyFromMatrixBiasBuffer(T *X_input_without_bias, int m_batch, bool x_trans, T *bias_buffer);
  virtual void releaseMatrixBiasBuffer(){};
  virtual T *getMatrixBiasBuffer(int m_batch);
  void forwardMatrixBias(
      const T *X_input_without_bias,
      T *D_output,
      int m_batch,
      bool x_trans,
      bool d_trans,
      bool is_test) override;
  void backwardMatrixBias(
      const T *D_input,
      T *X_output_without_bias,
      int m_batch,
      bool d_trans = false,
      bool x_trans = false) override;
  void updateMatrixBias(
      const T *X_input_without_bias,
      const T *D_input,
      int m_batch,
      bool x_trans = false,
      bool d_trans = false) override;

  /* when overriding copy methods below, _Vector_Bias can be used in derived */
  virtual T *copyToVectorBiasBuffer(const T *x_input_without_bias, int x_inc);
  virtual void copyFromVectorBiasBuffer(T *x_output_without_bias, int x_inc);
  virtual T *getVectorBiasBuffer() { return temp_x_vector_bias_.data(); };

  void forwardVector(const T *x_input, T *d_output, int x_inc, int d_inc, bool is_test) override;
  void backwardVector(const T *d_input, T *x_output, int d_inc = 1, int x_inc = 1) override;
  void updateVector(const T *x_input, const T *d_input, int x_inc = 1, int d_inc = 1) override;

  void forwardVectorBias(
      const T *x_input_without_bias, T *d_output, int x_inc, int d_inc, bool is_test) override;
  void backwardVectorBias(
      const T *d_input, T *x_output_without_bias, int d_inc = 1, int x_inc = 1) override;
  void updateVectorBias(
      const T *x_input_without_bias, const T *d_input, int x_inc = 1, int d_inc = 1) override;

  void forwardMatrix(
      const T *X_input, T *D_output, int m_batch, bool x_trans, bool d_trans, bool is_test)
      override;
  void backwardMatrix(
      const T *D_input,
      T *X_output,
      int m_batch,
      bool d_trans = false,
      bool x_trans = false) override;
  void updateMatrix(
      const T *X_input,
      const T *D_input,
      int m_batch,
      bool x_trans = false,
      bool d_trans = false) override;

  /* only these need to be overloaded for the Tensor interface */
  virtual void getTensorBuffer(T **x_tensor_ptr, T **d_tensor_ptr, int m_batch, int dim3);
  virtual void
  permute132(T *out_tensor, const T *in_tensor, int dim1, int dim2, int dim3, bool bias2);

  /* indexed interface: no need to overload, if not some performance benefits possible*/
  virtual void copyIndexedInput(
      T *out_tensor,
      const T *src_tensor,
      const int total_input_size,
      const int *indices,
      const int size,
      const int m_batch,
      const int dim3,
      const bool trans,
      const int m_batch_slice = 0,
      const int *batch_indices = nullptr);

  virtual void copyIndexedOutput(
      T *out_tensor,
      const T *src_tensor,
      const int total_output_size,
      const int *indices,
      const int size,
      const int m_batch,
      const int dim3,
      const bool trans,
      const int m_batch_slice = 0,
      const int *batch_indices = nullptr);

  virtual void copySliceInput(
      T *out_tensor,
      const T *src_tensor,
      const int size,
      const int m_batch,
      const int dim3,
      const bool trans,
      const int m_batch_slice,
      const int *batch_indices);

  virtual void copySliceOutput(
      T *out_tensor,
      const T *src_tensor,
      const int size,
      const int m_batch,
      const int dim3,
      const bool trans,
      const int m_batch_slice,
      const int *batch_indices);

  virtual void setZero(T *v, const int size);

private:
  void alpha_warning() {
    DEBUG_CALL(
        static int alpha_warning_count = 0; alpha_warning_count++; if (alpha_warning_count < 25) {
          std::cout
              << "Warning: setting weights with alpha scale. Note that alpha scale is NOT "
                 "respected "
                 "when getting weights, saving or export, where thus wrong weights are obtained."
              << std::endl;
        })
  }

public:
  std::mutex mutex_;

protected:
  std::shared_ptr<RNG<T>> rng_ = nullptr;
  std::shared_ptr<RealWorldRNG<T>> rw_rng_ = nullptr;
  T **weights_ = nullptr;
  T **weights_buffer_ = nullptr;
  T **fb_weights_ = nullptr;

  int last_update_m_batch_ = 1;
  bool use_delayed_update_ = false;

private:
  std::vector<T *> delta_weights_extern_;

  void initialize(int x_sz, int d_sz);

  SimpleMetaParameter<T> par_;

  std::vector<T> temp_x_vector_bias_;
  std::vector<T> temp_x_matrix_bias_;
  std::vector<T> temp_tensor_;
  std::vector<uint64_t> flicker_states_;
  std::vector<T> flicker_probs_;

  std::unique_ptr<WeightDrifter<T>> wdrifter_ = nullptr;
  std::unique_ptr<WeightRemapper<T>> wremapper_ = nullptr;
  std::unique_ptr<WeightClipper<T>> wclipper_ = nullptr;
  std::unique_ptr<WeightModifier<T>> fb_weight_modifier_ = nullptr;

  int *matrix_indices_ = nullptr;
  bool matrix_indices_set_ = false;

  T fwd_alpha_ = 1.0;
  T bwd_alpha_ = 1.0;

  bool shared_weights_if_ = false;
};

}; // namespace RPU
