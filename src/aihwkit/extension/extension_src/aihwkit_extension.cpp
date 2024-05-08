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

#include "aihwkit_extension.h"
#include "ops/float_prec_op.h"
#include "ops/thevenin_equiv_op.h"

namespace py = pybind11;

PYBIND11_MODULE(aihwkit_extension, m) {
  m.doc() = "Bindings for the AIHWKIT extensions.";
  auto m_ops = m.def_submodule("ops");

  m_ops.def(
      "float_precision_cast", &aihwkit::floatPrecisionCast, py::arg("x_input"), py::arg("exponent"),
      py::arg("mantissa"), py::arg("saturate_to_inf"),
      R"pbdoc(
           fake-casts FP32 number to variable mantissa / exponent. The underlying number format remains FP32

           Args:
               x_input: input tensor
               exponent: number of bits used for the exponent
               mantissa: number of bits used for the mantissa
               saturate_to_inf: whether to set it to infinity if saturated or to perform clipping

           Returns:
               y_output: fake casts tensor of the same size
           )pbdoc");
  m_ops.def(
      "thevenin_equiv", &aihwkit::theveninEquiv, py::arg("x_input"), py::arg("gp_values"),
      py::arg("gm_values"), py::arg("r_s"), py::arg("t_max"), py::arg("time_steps"),
      R"pbdoc(
           Helper function for computing the thevenin equivalent for  time based IR drop

           Args:
               x_input: input tensor
               gp_values: Positive conductance values (in muS)
               gm_values: Negative conductance values (in muS)
               r_s: Series resistance
               t_max: max time (in x units)
               time_steps: number of time steps

           Returns:
               y_output: [vth_rd, rht_3d]
           )pbdoc");
}
