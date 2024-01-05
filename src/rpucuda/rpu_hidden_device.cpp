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

#include "rpu_hidden_device.h"
#include "math_util.h"

namespace RPU {

/********************************************************************************
 * HiddenStepRPUDevice<T>
 *********************************************************************************/

template <typename T> void HiddenStepRPUDevice<T>::printDP(int x_count, int d_count) const {

  if (x_count < 0 || x_count > this->x_size_)
    x_count = this->x_size_;

  if (d_count < 0 || d_count > this->d_size_)
    d_count = this->d_size_;

  for (int i = 0; i < d_count; ++i) {
    for (int j = 0; j < x_count; ++j) {
      std::cout.precision(5);
      std::cout << i << "," << j << ": ";
      std::cout << "[<" << this->w_max_bound_[i][j] << ",";
      std::cout << this->w_min_bound_[i][j] << ">,<";
      std::cout << this->w_scale_up_[i][j] << ",";
      std::cout << this->w_scale_down_[i][j] << ">,<";
      std::cout << hs_scale_up_[i][j] << ",";
      std::cout << hs_scale_down_[i][j] << ">,";
      std::cout << hidden_weights_[i][j] << "]";
      std::cout << std::endl;
    }
  }
}

template <typename T>
void HiddenStepRPUDevice<T>::populate(
    const HiddenStepRPUDeviceMetaParameter<T> &p, RealWorldRNG<T> *rng) {

  PulsedRPUDevice<T>::populate(p, rng);
  auto &par = getPar();

  T up_down = par.hs_up_down;
  T up_bias = up_down > (T)0.0 ? (T)0 : up_down;
  T down_bias = up_down > (T)0.0 ? -up_down : (T)0.0;
  T up_down_std = par.hs_up_down_dtod;

  for (int j = 0; j < this->x_size_; ++j) {
    for (int i = 0; i < this->d_size_; ++i) {

      T gain = ((T)1.0 + par.hs_dw_min_dtod * rng->sampleGauss());
      T r = up_down_std * rng->sampleGauss();
      hs_scale_up_[i][j] = (up_bias + gain + r) * par.hs_dw_min;
      hs_scale_down_[i][j] = (down_bias + gain - r) * par.hs_dw_min;
      hidden_weights_[i][j] = (T)0.0;
    }
  }
}

/*********************************************************************************/
/*UPDATE*/

template <typename T>
inline void update_once(
    T &w,
    int &sign,
    T &hw,
    T &min_bound,
    T &max_bound,
    T &scale_down,
    T &scale_up,
    T &hs_scale_down,
    T &hs_scale_up,
    const T &dw_min_std,
    const T &hs_dw_min_std,
    RNG<T> *rng) {

  T hs_dw = sign > 0 ? -hs_scale_down : hs_scale_up;
  hw += hs_dw_min_std ? ((T)1.0 + hs_dw_min_std * rng->sampleGauss()) * hs_dw : hs_dw;
  if (hw < (T)-1.0) {
    hw = (T)0.0;
    T dw =
        dw_min_std > (T)0.0 ? ((T)1.0 + dw_min_std * rng->sampleGauss()) * scale_down : scale_down;
    w -= dw;
    w = MAX(w, min_bound);
  } else if (hw > (T)1.0) {
    hw = (T)0.0;
    T dw = dw_min_std > (T)0.0 ? ((T)1.0 + dw_min_std * rng->sampleGauss()) * scale_up : scale_up;
    w += dw;
    w = MIN(w, max_bound);
  }
}

template <typename T>
void HiddenStepRPUDevice<T>::doSparseUpdate(
    T **weights, int i, const int *x_signed_indices, int x_count, int d_sign, RNG<T> *rng) {
  T *scale_down = this->w_scale_down_[i];
  T *scale_up = this->w_scale_up_[i];
  T *w = weights[i];
  T *min_bound = this->w_min_bound_[i];
  T *max_bound = this->w_max_bound_[i];
  T *hw = hidden_weights_[i];
  T *hs_scale_down = hs_scale_down_[i];
  T *hs_scale_up = hs_scale_up_[i];

  const auto &par = getPar();

  PULSED_UPDATE_W_LOOP(update_once(
                           w[j], sign, hw[j], min_bound[j], max_bound[j], scale_down[j],
                           scale_up[j], hs_scale_down[j], hs_scale_up[j], par.dw_min_std,
                           par.hs_dw_min_std, rng););
}

template <typename T>
void HiddenStepRPUDevice<T>::doDenseUpdate(T **weights, int *coincidences, RNG<T> *rng) {

  T *scale_down = this->w_scale_down_[0];
  T *scale_up = this->w_scale_up_[0];
  T *w = weights[0];
  T *min_bound = this->w_min_bound_[0];
  T *max_bound = this->w_max_bound_[0];
  T *hw = hidden_weights_[0];
  T *hs_scale_down = hs_scale_down_[0];
  T *hs_scale_up = hs_scale_up_[0];

  const auto &par = getPar();

  PULSED_UPDATE_W_LOOP_DENSE(update_once(
                                 w[j], sign, hw[j], min_bound[j], max_bound[j], scale_down[j],
                                 scale_up[j], hs_scale_down[j], hs_scale_up[j], par.dw_min_std,
                                 par.hs_dw_min_std, rng););
}

template class HiddenStepRPUDevice<float>;
#ifdef RPU_USE_DOUBLE
template class HiddenStepRPUDevice<double>;
#endif
#ifdef RPU_USE_FP16
template class HiddenStepRPUDevice<half_t>;
#endif

} // namespace RPU
