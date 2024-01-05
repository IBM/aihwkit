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
#include "rpu_dynamic_transfer_device.h"
#include "rpu_forward_backward_pass.h"
#include "rpu_pulsed_meta_parameter.h"
#include "rpu_vector_device.h"
#include "weight_clipper.h"
#include "weight_modifier.h"
#include "weight_remapper.h"

void declare_utils(py::module &m_devices, py::module &m_tiles) {

  py::class_<RPU::WeightClipParameter>(m_tiles, "WeightClipParameter")
      .def(py::init<>())
      .def_readwrite("fixed_value", &RPU::WeightClipParameter::fixed_value)
      .def_readwrite("sigma", &RPU::WeightClipParameter::sigma)
      .def_readwrite("type", &RPU::WeightClipParameter::type);

  py::class_<RPU::WeightRemapParameter>(m_tiles, "WeightRemapParameter")
      .def(py::init<>())
      .def_readwrite("remapped_wmax", &RPU::WeightRemapParameter::remapped_wmax)
      .def_readwrite("max_scale_range", &RPU::WeightRemapParameter::max_scale_range)
      .def_readwrite("max_scale_ref", &RPU::WeightRemapParameter::max_scale_ref)
      .def_readwrite("type", &RPU::WeightRemapParameter::type);

  py::enum_<RPU::WeightModifierType>(m_tiles, "WeightModifierType")
      .value("Copy", RPU::WeightModifierType::Copy)
      .value("Discretize", RPU::WeightModifierType::Discretize)
      .value("MultNormal", RPU::WeightModifierType::MultNormal)
      .value("AddNormal", RPU::WeightModifierType::AddNormal)
      .value("DiscretizeAddNormal", RPU::WeightModifierType::DiscretizeAddNormal)
      .value("DoReFa", RPU::WeightModifierType::DoReFa)
      .value("Poly", RPU::WeightModifierType::Poly)
      .value("PCMNoise", RPU::WeightModifierType::PCMNoise)
      .value("ProgNoise", RPU::WeightModifierType::ProgNoise)
      .value("DropConnect", RPU::WeightModifierType::DropConnect)
      .value("None", RPU::WeightModifierType::Copy);

  py::enum_<RPU::WeightRemapType>(m_tiles, "WeightRemapType")
      .value("None", RPU::WeightRemapType::None)
      .value("LayerwiseSymmetric", RPU::WeightRemapType::LayerwiseSymmetric)
      .value("ChannelwiseSymmetric", RPU::WeightRemapType::ChannelwiseSymmetric);

  py::enum_<RPU::WeightClipType>(m_tiles, "WeightClipType")
      .value("None", RPU::WeightClipType::None)
      .value("FixedValue", RPU::WeightClipType::FixedValue)
      .value("LayerGaussian", RPU::WeightClipType::LayerGaussian)
      .value("AverageChannelMax", RPU::WeightClipType::AverageChannelMax);

  py::enum_<RPU::BoundManagementType>(m_devices, "BoundManagementType")
      .value("None", RPU::BoundManagementType::None)
      .value("Iterative", RPU::BoundManagementType::Iterative)
      .value("IterativeWorstCase", RPU::BoundManagementType::IterativeWorstCase);

  py::enum_<RPU::VectorDeviceUpdatePolicy>(m_devices, "VectorUnitCellUpdatePolicy")
      .value("All", RPU::VectorDeviceUpdatePolicy::All)
      .value("SingleFixed", RPU::VectorDeviceUpdatePolicy::SingleFixed)
      .value("SingleSequential", RPU::VectorDeviceUpdatePolicy::SingleSequential)
      .value("SingleRandom", RPU::VectorDeviceUpdatePolicy::SingleRandom);

  py::enum_<RPU::NoiseManagementType>(m_devices, "NoiseManagementType")
      .value("None", RPU::NoiseManagementType::None)
      .value("AbsMax", RPU::NoiseManagementType::AbsMax)
      .value("AbsMaxNPSum", RPU::NoiseManagementType::AbsMaxNPSum)
      .value("Max", RPU::NoiseManagementType::Max)
      .value("Constant", RPU::NoiseManagementType::Constant)
      .value("AverageAbsMax", RPU::NoiseManagementType::AverageAbsMax);

  py::enum_<RPU::OutputWeightNoiseType>(m_devices, "WeightNoiseType")
      .value("None", RPU::OutputWeightNoiseType::None)
      .value("AdditiveConstant", RPU::OutputWeightNoiseType::AdditiveConstant)
      .value("PCMRead", RPU::OutputWeightNoiseType::PCMRead);

  py::enum_<RPU::PulseType>(m_devices, "PulseType")
      .value("None", RPU::PulseType::None)
      .value("StochasticCompressed", RPU::PulseType::StochasticCompressed)
      .value("Stochastic", RPU::PulseType::Stochastic)
      .value("NoneWithDevice", RPU::PulseType::NoneWithDevice)
      .value("MeanCount", RPU::PulseType::MeanCount)
      .value("DeterministicImplicit", RPU::PulseType::DeterministicImplicit);

  py::enum_<RPU::AnalogMVType>(m_devices, "AnalogMVType")
      .value("Ideal", RPU::AnalogMVType::Ideal)
      .value("OnePass", RPU::AnalogMVType::OnePass)
      .value("PosNegSeparate", RPU::AnalogMVType::PosNegSeparate)
      .value("PosNegSeparateDigitalSum", RPU::AnalogMVType::PosNegSeparateDigitalSum);
};
