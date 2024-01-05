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

#include "rpu_base.h"
#include "utility_functions.h"

namespace py = pybind11;

PYBIND11_MODULE(rpu_base, m) {
  m.doc() = "Bindings for the RPU simulator.";

  // Devices module.
  auto m_devices = m.def_submodule("devices");
  m_devices.doc() = "Bindings for the simulator devices and parameters.";

  // Tiles module.
  auto m_tiles = m.def_submodule("tiles");
  m_tiles.doc() = "Bindings for the simulator analog tiles.";

  /// for backward compatiblilty we keep the (strange) structure (most
  /// enums in devices, some in tiles)
  declare_utils(m_devices, m_tiles);

  // FLOAT (default) has no extra sub-module (for legacy)
  declare_rpu_devices<float>(m_devices, "");
  declare_rpu_tiles<float, float>(m_tiles, "");
#ifdef RPU_USE_CUDA
  declare_rpu_tiles_cuda<float, float>(m_tiles, "", true);
#endif

  // DOUBLE (separate submodules)
#ifdef RPU_USE_DOUBLE
  auto m_devices_double = m_devices.def_submodule("double", "Double data type");
  declare_rpu_devices<double>(m_devices_double, "");

  auto m_tiles_double = m_tiles.def_submodule("double", "Double data type");
  declare_rpu_tiles<double, double>(m_tiles_double, "");
#ifdef RPU_USE_CUDA
  declare_rpu_tiles_cuda<double, double>(m_tiles_double, "");
#endif
#endif

  // FLOAT16  (separate submodules)
#ifdef RPU_USE_CUDA
#ifdef RPU_USE_FP16
#ifdef RPU_BFLOAT_AS_FP16
  // half_t is now actually bfloat16
  auto m_devices_bfloat16 = m_devices.def_submodule("bfloat16", "Bfloat16 data type");
  declare_rpu_devices<half_t>(m_devices_bfloat16, "");

  auto m_tiles_bfloat16 = m_tiles.def_submodule("bfloat16", "Bfloat16 data type");
  declare_rpu_tiles<at::BFloat16, half_t>(m_tiles_bfloat16, "");
  declare_rpu_tiles_cuda<at::BFloat16, half_t>(m_tiles_bfloat16, "");

#else
  auto m_devices_half = m_devices.def_submodule("half", "Half (Float16) data type");
  declare_rpu_devices<half_t>(m_devices_half, "");

  auto m_tiles_half = m_tiles.def_submodule("half", "Half (Float16) data type");
  declare_rpu_tiles<at::Half, half_t>(m_tiles_half, "");
  declare_rpu_tiles_cuda<at::Half, half_t>(m_tiles_half, "");

#endif
#endif
#endif

  // Cuda utilities.
  auto m_cuda = m.def_submodule("cuda");
  m_cuda.doc() = "CUDA utilities.";

  m_cuda.def(
      "is_compiled",
      [] {
#ifdef RPU_USE_CUDA
        return true;
#else
        return false;
#endif
      },
      R"pbdoc(
    Return whether aihwkit was compiled with CUDA support.
    )pbdoc");
}
