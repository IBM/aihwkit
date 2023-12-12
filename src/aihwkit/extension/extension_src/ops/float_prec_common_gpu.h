#pragma once
#include "float_prec_common.h"

namespace aihwkit {
namespace detail {

template <int EL, int ML, bool saturate_to_inf, bool rounding>
__device__ __forceinline__ float cudaFPrecCast(const float x) {
  FLOATPREC_INIT(EL, ML);
  FLOATPREC_BODY(x, y, saturate_to_inf, rounding);
  return y;
}

template <> __device__ __forceinline__ float cudaFPrecCast<6, 9, false, false>(const float x) {
  FLOATPREC_EL6_ML9_0_0(x, y);
  return y;
}

} // namespace detail
} // namespace aihwkit
