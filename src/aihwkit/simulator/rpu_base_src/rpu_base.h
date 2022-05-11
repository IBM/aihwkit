/**
 * (C) Copyright 2020, 2021 IBM. All Rights Reserved.
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
#include "rpu.h"
#include "rpu_buffered_transfer_device.h"
#include "rpu_constantstep_device.h"
#include "rpu_expstep_device.h"
#include "rpu_linearstep_device.h"
#include "rpu_mixedprec_device.h"
#include "rpu_mixedprec_device_base.h"
#include "rpu_onesided_device.h"
#include "rpu_piecewisestep_device.h"
#include "rpu_powstep_device.h"
#include "rpu_pulsed.h"
#include "rpu_simple_device.h"
#include "rpu_transfer_device.h"
#include "rpu_vector_device.h"
#include "weight_clipper.h"
#include "weight_modifier.h"

#ifdef RPU_USE_CUDA
#include "cuda_util.h"
#include "rpucuda.h"
#include "rpucuda_pulsed.h"
#include <ATen/cuda/CUDAContext.h>
#endif

#include <torch/extension.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

#ifdef RPU_USE_DOUBLE
typedef double T;
#else
typedef float T;
#endif

namespace py = pybind11;

void declare_rpu_tiles(py::module &m);
void declare_rpu_devices(py::module &m);
#ifdef RPU_USE_CUDA
void declare_rpu_tiles_cuda(py::module &m);
#endif
