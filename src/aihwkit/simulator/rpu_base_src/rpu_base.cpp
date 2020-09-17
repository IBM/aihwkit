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

#include "rpu_base.h"

namespace py = pybind11;

PYBIND11_MODULE(rpu_base, m) {
  m.doc() = "Bindings for the RPU simulator.";

  // Tiles module.
  auto m_tiles = m.def_submodule("tiles");
  m_tiles.doc() = "Bindings for the simulator analog tiles.";
  declare_rpu_tiles(m_tiles);

  // Devices module.
  auto m_devices = m.def_submodule("devices");
  m_devices.doc() = "Bindings for the simulator devices and parameters.";
  declare_rpu_devices(m_devices);

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

#ifdef RPU_USE_CUDA
  declare_rpu_tiles_cuda(m_tiles);
#endif
}
