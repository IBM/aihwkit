#pragma once
#include <cstdint>

#define FLOATPREC_INIT(EL, ML)                                                                     \
  const uint32_t bias_el = (1 << (EL - 1)) - 1;                                                    \
  const uint32_t sat_el = (1 << EL) - 1;                                                           \
  const uint32_t sat_exp = (uint32_t)((sat_el + 127 - bias_el) << 23);                             \
  const uint32_t highest_not_needed_bit = 1 << (22 - ML);                                          \
  const uint32_t valid_msk = ~((highest_not_needed_bit << 1) - 1);                                 \
  uint32_t needs_up_round = 0;

#define FLOATPREC_BODY(X, Y, SATURATE, ROUNDING)                                                   \
  uint32_t x_int = ((union {                                                                       \
                     float f;                                                                      \
                     uint32_t i;                                                                   \
                   }){X})                                                                          \
                       .i;                                                                         \
  if (ROUNDING) {                                                                                  \
    needs_up_round = (x_int & highest_not_needed_bit) << 1;                                        \
  }                                                                                                \
  bool overflow_if = ((0x7F800000 & x_int) == 0x7F800000);                                         \
  x_int &= valid_msk;                                                                              \
  x_int = overflow_if ? (x_int & (uint32_t)0xFF800000) : x_int + needs_up_round;                   \
  int f32_exp = ((x_int & (uint32_t)0x7F800000) >> 23) - 127 + bias_el;                            \
  if (SATURATE) {                                                                                  \
    x_int =                                                                                        \
        f32_exp >= (int)sat_el ? ((x_int | (uint32_t)0x7F800000) & (uint32_t)0xFF800000) : x_int;  \
  } else {                                                                                         \
    x_int = f32_exp >= (int)sat_el ? (sat_exp | (x_int & ~((uint32_t)0x7F800000))) : x_int;        \
  }                                                                                                \
  x_int = f32_exp <= 0 ? 0 : x_int;                                                                \
  float Y = ((union {                                                                              \
              uint32_t i;                                                                          \
              float f;                                                                             \
            }){x_int})                                                                             \
                .f;

// special version for DL16FP. no rounding. no sat to inf
#define FLOATPREC_EL6_ML9_0_0(X, Y)                                                                \
  uint32_t x_int = ((union {                                                                       \
                     float f;                                                                      \
                     uint32_t i;                                                                   \
                   }){X})                                                                          \
                       .i;                                                                         \
  x_int = ((x_int & 0x7fffffff) > 0x4f7fc000)   ? (0x4f7fc000 | (x_int & 0x8000000))               \
          : ((x_int & 0x7fffffff) < 0x30800000) ? 0x00000000                                       \
                                                : x_int & 0xffffc000;                              \
  float Y = ((union {                                                                              \
              uint32_t i;                                                                          \
              float f;                                                                             \
            }){x_int})                                                                             \
                .f;

namespace caffe2 {
namespace detail {

template <int EL, int ML, bool saturate_to_inf, bool rounding>
inline float hostFPrecCast(const float x) {
  FLOATPREC_INIT(EL, ML);
  FLOATPREC_BODY(x, y, saturate_to_inf, rounding);
  return y;
}

template <> inline float hostFPrecCast<6, 9, false, false>(const float x) {
  FLOATPREC_EL6_ML9_0_0(x, y);
  return y;
}

} // namespace detail
} // namespace caffe2
