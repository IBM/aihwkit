/**
 * (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
 *
 * Licensed under the MIT license. See LICENSE file in the project root for details.
 */
#pragma once

#include <torch/extension.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <utility_functions.h>

namespace py = pybind11;

#ifdef RPU_USE_FP16
namespace pybind11 {
namespace detail {
template <> struct type_caster<half_t> : public type_caster_base<half_t> {

public:
  bool load(handle src, bool convert) {
    if (py::isinstance<py::float_>(src) || py::isinstance<py::int_>(src)) {
      this->value = new half_t(py::cast<float>(src));
      // std::cerr << "converted " << py::cast<float>(src) << " to " << (float)*((half_t *)
      // this->value) << std::endl;
      return true;
    }

    return false;
  }
  static handle cast(half_t src, return_value_policy policy, handle parent) {
    return type_caster<float>::cast(float(src), policy, parent);
  }
};
} // namespace detail
} // namespace pybind11
#endif

#if RPU_USE_FP16
#define DEFAULT_DTYPE                                                                              \
  ((std::is_same<half_t, T_RPU>::value)                                                            \
       ? torch::kFloat16                                                                           \
       : ((std::is_same<double, T_RPU>::value) ? torch::kFloat64 : torch::kFloat32))
#else
#define DEFAULT_DTYPE ((std::is_same<double, T_RPU>::value) ? torch::kFloat64 : torch::kFloat32)
#endif

#define DEFAULT_TENSOR_OPTIONS auto default_options = torch::TensorOptions().dtype(DEFAULT_DTYPE)

void declare_utils(py::module &m_devices, py::module &m_tiles);

template <typename T, typename T_RPU>
void declare_rpu_tiles(py::module &m, std::string type_name_add);

template <typename T> void declare_rpu_devices(py::module &m, std::string type_name_add);

#ifdef RPU_USE_CUDA
template <typename T, typename T_RPU>
void declare_rpu_tiles_cuda(py::module &m, std::string type_name_add, bool add_utilities = false);
#endif
