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

void declare_rpu_devices(py::module &m) {

  using AbstractParam = RPU::AbstractRPUDeviceMetaParameter<T>;
  using PulsedParam = RPU::PulsedRPUDeviceMetaParameter<T>;
  using ConstantStepParam = RPU::ConstantStepRPUDeviceMetaParameter<T>;

  /*
   * Trampoline classes for allowing inheritance.
   */
  class PyAbstractParam : public AbstractParam {
  public:
    std::string getName() const override {
      PYBIND11_OVERLOAD_PURE(std::string, AbstractParam, getName, );
    }
    AbstractParam *clone() const override {
      PYBIND11_OVERLOAD_PURE(AbstractParam *, AbstractParam, clone, );
    }
    RPU::DeviceUpdateType implements() const override {
      PYBIND11_OVERLOAD_PURE(RPU::DeviceUpdateType, AbstractParam, implements, );
    }
    RPU::AbstractRPUDevice<T> *
    createDevice(int x_size, int d_size, RPU::RealWorldRNG<T> *rng) override {
      PYBIND11_OVERLOAD_PURE(
          RPU::AbstractRPUDevice<T> *, AbstractParam, createDevice, x_size, d_size, rng);
    }
  };

  class PyPulsedParam : public PulsedParam {
  public:
    std::string getName() const override { PYBIND11_OVERLOAD(std::string, PulsedParam, getName, ); }
    PulsedParam *clone() const override {
      PYBIND11_OVERLOAD_PURE(PulsedParam *, PulsedParam, clone, );
    }
    RPU::DeviceUpdateType implements() const override {
      PYBIND11_OVERLOAD(RPU::DeviceUpdateType, PulsedParam, implements, );
    }
    RPU::PulsedRPUDevice<T> *
    createDevice(int x_size, int d_size, RPU::RealWorldRNG<T> *rng) override {
      PYBIND11_OVERLOAD_PURE(
          RPU::PulsedRPUDevice<T> *, PulsedParam, createDevice, x_size, d_size, rng);
    }
  };

  class PyConstantStepParam : public ConstantStepParam {
  public:
    std::string getName() const override {
      PYBIND11_OVERLOAD(std::string, ConstantStepParam, getName, );
    }
    ConstantStepParam *clone() const override {
      PYBIND11_OVERLOAD(ConstantStepParam *, ConstantStepParam, clone, );
    }
    RPU::DeviceUpdateType implements() const override {
      PYBIND11_OVERLOAD(RPU::DeviceUpdateType, ConstantStepParam, implements, );
    }
    RPU::ConstantStepRPUDevice<T> *
    createDevice(int x_size, int d_size, RPU::RealWorldRNG<T> *rng) override {
      PYBIND11_OVERLOAD(
          RPU::ConstantStepRPUDevice<T> *, ConstantStepParam, createDevice, x_size, d_size, rng);
    }
  };

  /*
   * Python class definitions.
   */
  py::class_<RPU::SimpleMetaParameter<T>>(m, "FloatingPointTileParameter")
      .def(py::init<>())
      .def(
          "create_array", &RPU::SimpleMetaParameter<T>::createRPUArray, py::arg("x_size"),
          py::arg("d_size"))
      .def(
          "__str__",
          [](RPU::SimpleMetaParameter<T> &self) {
            std::stringstream ss;
            self.printToStream(ss);
            return ss.str();
          })
      // Properties from this class.
      .def_readwrite("diffusion", &RPU::SimpleMetaParameter<T>::diffusion)
      .def_readwrite("lifetime", &RPU::SimpleMetaParameter<T>::lifetime);

  py::class_<RPU::PulsedMetaParameter<T>>(m, "AnalogTileParameter")
      .def(py::init<>())
      .def(
          "create_array",
          [](RPU::PulsedMetaParameter<T> &self, int n_cols, int n_rows, ConstantStepParam *dp) {
            return self.createRPUArray(n_cols, n_rows, dp);
          })
      .def(
          "__str__",
          [](RPU::PulsedMetaParameter<T> &self) {
            std::stringstream ss;
            self.printToStream(ss);
            return ss.str();
          })
      .def_readwrite("forward_io", &RPU::PulsedMetaParameter<T>::f_io)
      .def_readwrite("backward_io", &RPU::PulsedMetaParameter<T>::b_io)
      .def_readwrite("update", &RPU::PulsedMetaParameter<T>::up);

  py::class_<AbstractParam, PyAbstractParam, RPU::SimpleMetaParameter<T>>(
      m, "AbstractResistiveDevicesParameter")
      .def(py::init<>());

  py::class_<PulsedParam, PyPulsedParam, AbstractParam>(m, "PulsedResistiveDeviceParameter")
      .def(py::init<>())
      // Properties from this class.
      .def_readwrite("corrupt_devices_prob", &PulsedParam::corrupt_devices_prob)
      .def_readwrite("corrupt_devices_range", &PulsedParam::corrupt_devices_range)
      .def_readwrite("diffusion_dtod", &PulsedParam::diffusion_dtod)
      .def_readwrite("dw_min", &PulsedParam::dw_min)
      .def_readwrite("dw_min_dtod", &PulsedParam::dw_min_dtod)
      .def_readwrite("dw_min_std", &PulsedParam::dw_min_std)
      .def_readwrite("enforce_consistency", &PulsedParam::enforce_consistency)
      .def_readwrite("lifetime_dtod", &PulsedParam::lifetime_dtod)
      .def_readwrite("perfect_bias", &PulsedParam::perfect_bias)
      .def_readwrite("reset", &PulsedParam::reset)
      .def_readwrite("reset_dtod", &PulsedParam::reset_dtod)
      .def_readwrite("reset_std", &PulsedParam::reset_std)
      .def_readwrite("up_down", &PulsedParam::up_down)
      .def_readwrite("up_down_dtod", &PulsedParam::up_down_dtod)
      .def_readwrite("w_max", &PulsedParam::w_max)
      .def_readwrite("w_max_dtod", &PulsedParam::w_max_dtod)
      .def_readwrite("w_min", &PulsedParam::w_min)
      .def_readwrite("w_min_dtod", &PulsedParam::w_min_dtod);

  py::class_<ConstantStepParam, PyConstantStepParam, PulsedParam>(
      m, "ConstantStepResistiveDeviceParameter")
      .def(py::init<>());

  py::class_<RPU::PulsedUpdateMetaParameter<T>>(m, "AnalogTileUpdateParameter")
      .def(py::init<>())
      .def_readwrite("fixed_bl", &RPU::PulsedUpdateMetaParameter<T>::fixed_BL)
      .def_readwrite("desired_bl", &RPU::PulsedUpdateMetaParameter<T>::desired_BL)
      .def_readwrite("pulse_type", &RPU::PulsedUpdateMetaParameter<T>::pulse_type)
      .def_readwrite("res", &RPU::PulsedUpdateMetaParameter<T>::res)
      .def_readwrite("sto_round", &RPU::PulsedUpdateMetaParameter<T>::sto_round)
      .def_readwrite("update_management", &RPU::PulsedUpdateMetaParameter<T>::update_management)
      .def_readwrite(
          "update_bl_management", &RPU::PulsedUpdateMetaParameter<T>::update_bl_management);

  py::class_<RPU::IOMetaParameter<T>>(m, "AnalogTileInputOutputParameter")
      .def(py::init<>())
      .def_readwrite("bm_test_negative_bound", &RPU::IOMetaParameter<T>::bm_test_negative_bound)
      .def_readwrite("bound_management", &RPU::IOMetaParameter<T>::bound_management)
      .def_readwrite("inp_bound", &RPU::IOMetaParameter<T>::inp_bound)
      .def_readwrite("inp_noise", &RPU::IOMetaParameter<T>::inp_noise)
      .def_readwrite("inp_res", &RPU::IOMetaParameter<T>::inp_res)
      .def_readwrite("inp_sto_round", &RPU::IOMetaParameter<T>::inp_sto_round)
      .def_readwrite("is_perfect", &RPU::IOMetaParameter<T>::is_perfect)
      .def_readwrite("max_bm_factor", &RPU::IOMetaParameter<T>::max_bm_factor)
      .def_readwrite("max_bm_res", &RPU::IOMetaParameter<T>::max_bm_res)
      .def_readwrite("nm_thres", &RPU::IOMetaParameter<T>::nm_thres)
      .def_readwrite("noise_management", &RPU::IOMetaParameter<T>::noise_management)
      .def_readwrite("out_bound", &RPU::IOMetaParameter<T>::out_bound)
      .def_readwrite("out_noise", &RPU::IOMetaParameter<T>::out_noise)
      .def_readwrite("out_res", &RPU::IOMetaParameter<T>::out_res)
      .def_readwrite("out_scale", &RPU::IOMetaParameter<T>::out_scale)
      .def_readwrite("out_sto_round", &RPU::IOMetaParameter<T>::out_sto_round)
      .def_readwrite("w_noise", &RPU::IOMetaParameter<T>::w_noise)
      .def_readwrite("w_noise_type", &RPU::IOMetaParameter<T>::w_noise_type);

  /**
   * Helper enums.
   **/
  py::enum_<RPU::BoundManagementType>(m, "BoundManagementType")
      .value("None", RPU::BoundManagementType::None)
      .value("Iterative", RPU::BoundManagementType::Iterative);

  py::enum_<RPU::NoiseManagementType>(m, "NoiseManagementType")
      .value("None", RPU::NoiseManagementType::None)
      .value("AbsMax", RPU::NoiseManagementType::AbsMax)
      .value("Constant", RPU::NoiseManagementType::Constant)
      .value("Max", RPU::NoiseManagementType::Max);

  py::enum_<RPU::OutputWeightNoiseType>(m, "OutputWeightNoiseType")
      .value("None", RPU::OutputWeightNoiseType::None)
      .value("AdditiveConstant", RPU::OutputWeightNoiseType::AdditiveConstant);

  py::enum_<RPU::PulseType>(m, "PulseType")
      .value("None", RPU::PulseType::None)
      .value("StochasticCompressed", RPU::PulseType::StochasticCompressed)
      .value("Stochastic", RPU::PulseType::Stochastic)
      .value("NoneWithDevice", RPU::PulseType::NoneWithDevice)
      .value("MeanCount", RPU::PulseType::MeanCount);
}
