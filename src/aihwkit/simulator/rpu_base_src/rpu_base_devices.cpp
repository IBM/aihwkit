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

#include "rpu.h"
#include "rpu_base.h"
#include "rpu_buffered_transfer_device.h"
#include "rpu_chopped_transfer_device.h"
#include "rpu_constantstep_device.h"
#include "rpu_dynamic_transfer_device.h"
#include "rpu_expstep_device.h"
#include "rpu_linearstep_device.h"
#include "rpu_mixedprec_device.h"
#include "rpu_mixedprec_device_base.h"
#include "rpu_onesided_device.h"
#include "rpu_piecewisestep_device.h"
#include "rpu_powstep_device.h"
#include "rpu_powstep_reference_device.h"
#include "rpu_pulsed.h"
#include "rpu_simple_device.h"
#include "rpu_softbounds_reference_device.h"
#include "rpu_transfer_device.h"
#include "rpu_vector_device.h"

#define NAME(S) (S + type_name_add).c_str()

template <typename T> void declare_rpu_devices(py::module &m, std::string type_name_add) {

  using AbstractParam = RPU::AbstractRPUDeviceMetaParameter<T>;
  using SimpleParam = RPU::SimpleRPUDeviceMetaParameter<T>;
  using PulsedBaseParam = RPU::PulsedRPUDeviceMetaParameterBase<T>;
  using PulsedParam = RPU::PulsedRPUDeviceMetaParameter<T>;
  using ConstantStepParam = RPU::ConstantStepRPUDeviceMetaParameter<T>;
  using LinearStepParam = RPU::LinearStepRPUDeviceMetaParameter<T>;
  using SoftBoundsParam = RPU::SoftBoundsRPUDeviceMetaParameter<T>;
  using ExpStepParam = RPU::ExpStepRPUDeviceMetaParameter<T>;
  using VectorParam = RPU::VectorRPUDeviceMetaParameter<T>;
  using OneSidedParam = RPU::OneSidedRPUDeviceMetaParameter<T>;
  using TransferParam = RPU::TransferRPUDeviceMetaParameter<T>;
  using MixedPrecParam = RPU::MixedPrecRPUDeviceMetaParameter<T>;
  using PowStepParam = RPU::PowStepRPUDeviceMetaParameter<T>;
  using PowStepReferenceParam = RPU::PowStepReferenceRPUDeviceMetaParameter<T>;
  using PiecewiseStepParam = RPU::PiecewiseStepRPUDeviceMetaParameter<T>;
  using BufferedTransferParam = RPU::BufferedTransferRPUDeviceMetaParameter<T>;
  using ChoppedTransferParam = RPU::ChoppedTransferRPUDeviceMetaParameter<T>;
  using DynamicTransferParam = RPU::DynamicTransferRPUDeviceMetaParameter<T>;
  using SoftBoundsReferenceParam = RPU::SoftBoundsReferenceRPUDeviceMetaParameter<T>;

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
  class PySimpleParam : public SimpleParam {
  public:
    std::string getName() const override {
      PYBIND11_OVERLOAD_PURE(std::string, SimpleParam, getName, );
    }
    SimpleParam *clone() const override {
      PYBIND11_OVERLOAD_PURE(SimpleParam *, SimpleParam, clone, );
    }
    RPU::DeviceUpdateType implements() const override {
      PYBIND11_OVERLOAD_PURE(RPU::DeviceUpdateType, SimpleParam, implements, );
    }
    RPU::SimpleRPUDevice<T> *
    createDevice(int x_size, int d_size, RPU::RealWorldRNG<T> *rng) override {
      PYBIND11_OVERLOAD_PURE(
          RPU::SimpleRPUDevice<T> *, SimpleParam, createDevice, x_size, d_size, rng);
    }
  };

  class PyPulsedBaseParam : public PulsedBaseParam {
  public:
    std::string getName() const override {
      PYBIND11_OVERLOAD_PURE(std::string, PulsedBaseParam, getName, );
    }
    PulsedBaseParam *clone() const override {
      PYBIND11_OVERLOAD_PURE(PulsedBaseParam *, PulsedBaseParam, clone, );
    }
    RPU::DeviceUpdateType implements() const override {
      PYBIND11_OVERLOAD_PURE(RPU::DeviceUpdateType, PulsedBaseParam, implements, );
    }
    RPU::PulsedRPUDeviceBase<T> *
    createDevice(int x_size, int d_size, RPU::RealWorldRNG<T> *rng) override {
      PYBIND11_OVERLOAD_PURE(
          RPU::PulsedRPUDeviceBase<T> *, PulsedBaseParam, createDevice, x_size, d_size, rng);
    }
    T calcWeightGranularity() const override {
      PYBIND11_OVERLOAD(T, PulsedBaseParam, calcWeightGranularity, );
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
    T calcWeightGranularity() const override {
      PYBIND11_OVERLOAD(T, PulsedParam, calcWeightGranularity, );
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
    T calcWeightGranularity() const override {
      PYBIND11_OVERLOAD(T, ConstantStepParam, calcWeightGranularity, );
    }
  };

  class PyLinearStepParam : public LinearStepParam {
  public:
    std::string getName() const override {
      PYBIND11_OVERLOAD(std::string, LinearStepParam, getName, );
    }
    LinearStepParam *clone() const override {
      PYBIND11_OVERLOAD(LinearStepParam *, LinearStepParam, clone, );
    }
    RPU::DeviceUpdateType implements() const override {
      PYBIND11_OVERLOAD(RPU::DeviceUpdateType, LinearStepParam, implements, );
    }
    RPU::LinearStepRPUDevice<T> *
    createDevice(int x_size, int d_size, RPU::RealWorldRNG<T> *rng) override {
      PYBIND11_OVERLOAD(
          RPU::LinearStepRPUDevice<T> *, LinearStepParam, createDevice, x_size, d_size, rng);
    }
    T calcWeightGranularity() const override {
      PYBIND11_OVERLOAD(T, LinearStepParam, calcWeightGranularity, );
    }
  };

  class PySoftBoundsParam : public SoftBoundsParam {
  public:
    std::string getName() const override {
      PYBIND11_OVERLOAD(std::string, SoftBoundsParam, getName, );
    }
    SoftBoundsParam *clone() const override {
      PYBIND11_OVERLOAD(SoftBoundsParam *, SoftBoundsParam, clone, );
    }
    T calcWeightGranularity() const override {
      PYBIND11_OVERLOAD(T, SoftBoundsParam, calcWeightGranularity, );
    }
  };

  class PyExpStepParam : public ExpStepParam {
  public:
    std::string getName() const override {
      PYBIND11_OVERLOAD(std::string, ExpStepParam, getName, );
    }
    ExpStepParam *clone() const override {
      PYBIND11_OVERLOAD(ExpStepParam *, ExpStepParam, clone, );
    }
    RPU::DeviceUpdateType implements() const override {
      PYBIND11_OVERLOAD(RPU::DeviceUpdateType, ExpStepParam, implements, );
    }
    RPU::ExpStepRPUDevice<T> *
    createDevice(int x_size, int d_size, RPU::RealWorldRNG<T> *rng) override {
      PYBIND11_OVERLOAD(
          RPU::ExpStepRPUDevice<T> *, ExpStepParam, createDevice, x_size, d_size, rng);
    }
    T calcWeightGranularity() const override {
      PYBIND11_OVERLOAD(T, ExpStepParam, calcWeightGranularity, );
    }
  };

  class PyVectorParam : public VectorParam {
  public:
    std::string getName() const override { PYBIND11_OVERLOAD(std::string, VectorParam, getName, ); }
    VectorParam *clone() const override { PYBIND11_OVERLOAD(VectorParam *, VectorParam, clone, ); }
    RPU::DeviceUpdateType implements() const override {
      PYBIND11_OVERLOAD(RPU::DeviceUpdateType, VectorParam, implements, );
    }
    RPU::VectorRPUDevice<T> *
    createDevice(int x_size, int d_size, RPU::RealWorldRNG<T> *rng) override {
      PYBIND11_OVERLOAD(RPU::VectorRPUDevice<T> *, VectorParam, createDevice, x_size, d_size, rng);
    }
    T calcWeightGranularity() const override {
      PYBIND11_OVERLOAD(T, VectorParam, calcWeightGranularity, );
    }
  };

  class PyOneSidedParam : public OneSidedParam {
  public:
    std::string getName() const override {
      PYBIND11_OVERLOAD(std::string, OneSidedParam, getName, );
    }
    OneSidedParam *clone() const override {
      PYBIND11_OVERLOAD(OneSidedParam *, OneSidedParam, clone, );
    }
    RPU::DeviceUpdateType implements() const override {
      PYBIND11_OVERLOAD(RPU::DeviceUpdateType, OneSidedParam, implements, );
    }
    RPU::OneSidedRPUDevice<T> *
    createDevice(int x_size, int d_size, RPU::RealWorldRNG<T> *rng) override {
      PYBIND11_OVERLOAD(
          RPU::OneSidedRPUDevice<T> *, OneSidedParam, createDevice, x_size, d_size, rng);
    }
    T calcWeightGranularity() const override {
      PYBIND11_OVERLOAD(T, OneSidedParam, calcWeightGranularity, );
    }
  };

  class PyTransferParam : public TransferParam {
  public:
    std::string getName() const override {
      PYBIND11_OVERLOAD(std::string, TransferParam, getName, );
    }
    TransferParam *clone() const override {
      PYBIND11_OVERLOAD(TransferParam *, TransferParam, clone, );
    }
    RPU::DeviceUpdateType implements() const override {
      PYBIND11_OVERLOAD(RPU::DeviceUpdateType, TransferParam, implements, );
    }
    RPU::TransferRPUDevice<T> *
    createDevice(int x_size, int d_size, RPU::RealWorldRNG<T> *rng) override {
      PYBIND11_OVERLOAD(
          RPU::TransferRPUDevice<T> *, TransferParam, createDevice, x_size, d_size, rng);
    }
    T calcWeightGranularity() const override {
      PYBIND11_OVERLOAD(T, TransferParam, calcWeightGranularity, );
    }
  };

  class PyMixedPrecParam : public MixedPrecParam {
  public:
    std::string getName() const override {
      PYBIND11_OVERLOAD(std::string, MixedPrecParam, getName, );
    }
    MixedPrecParam *clone() const override {
      PYBIND11_OVERLOAD(MixedPrecParam *, MixedPrecParam, clone, );
    }
    RPU::DeviceUpdateType implements() const override {
      PYBIND11_OVERLOAD(RPU::DeviceUpdateType, MixedPrecParam, implements, );
    }
    RPU::MixedPrecRPUDevice<T> *
    createDevice(int x_size, int d_size, RPU::RealWorldRNG<T> *rng) override {
      PYBIND11_OVERLOAD(
          RPU::MixedPrecRPUDevice<T> *, MixedPrecParam, createDevice, x_size, d_size, rng);
    }
  };

  class PyPowStepParam : public PowStepParam {
  public:
    std::string getName() const override {
      PYBIND11_OVERLOAD(std::string, PowStepParam, getName, );
    }
    PowStepParam *clone() const override {
      PYBIND11_OVERLOAD(PowStepParam *, PowStepParam, clone, );
    }
    RPU::DeviceUpdateType implements() const override {
      PYBIND11_OVERLOAD(RPU::DeviceUpdateType, PowStepParam, implements, );
    }
    RPU::PowStepRPUDevice<T> *
    createDevice(int x_size, int d_size, RPU::RealWorldRNG<T> *rng) override {
      PYBIND11_OVERLOAD(
          RPU::PowStepRPUDevice<T> *, PowStepParam, createDevice, x_size, d_size, rng);
    }
    T calcWeightGranularity() const override {
      PYBIND11_OVERLOAD(T, PowStepParam, calcWeightGranularity, );
    }
  };

  class PyPowStepReferenceParam : public PowStepReferenceParam {
  public:
    std::string getName() const override {
      PYBIND11_OVERLOAD(std::string, PowStepReferenceParam, getName, );
    }
    PowStepReferenceParam *clone() const override {
      PYBIND11_OVERLOAD(PowStepReferenceParam *, PowStepReferenceParam, clone, );
    }
    RPU::DeviceUpdateType implements() const override {
      PYBIND11_OVERLOAD(RPU::DeviceUpdateType, PowStepReferenceParam, implements, );
    }
    RPU::PowStepReferenceRPUDevice<T> *
    createDevice(int x_size, int d_size, RPU::RealWorldRNG<T> *rng) override {
      PYBIND11_OVERLOAD(
          RPU::PowStepReferenceRPUDevice<T> *, PowStepReferenceParam, createDevice, x_size, d_size,
          rng);
    }
    T calcWeightGranularity() const override {
      PYBIND11_OVERLOAD(T, PowStepReferenceParam, calcWeightGranularity, );
    }
  };

  class PyPiecewiseStepParam : public PiecewiseStepParam {
  public:
    std::string getName() const override {
      PYBIND11_OVERLOAD(std::string, PiecewiseStepParam, getName, );
    }
    PiecewiseStepParam *clone() const override {
      PYBIND11_OVERLOAD(PiecewiseStepParam *, PiecewiseStepParam, clone, );
    }
    RPU::DeviceUpdateType implements() const override {
      PYBIND11_OVERLOAD(RPU::DeviceUpdateType, PiecewiseStepParam, implements, );
    }
    RPU::PiecewiseStepRPUDevice<T> *
    createDevice(int x_size, int d_size, RPU::RealWorldRNG<T> *rng) override {
      PYBIND11_OVERLOAD(
          RPU::PiecewiseStepRPUDevice<T> *, PiecewiseStepParam, createDevice, x_size, d_size, rng);
    }
    T calcWeightGranularity() const override {
      PYBIND11_OVERLOAD(T, PiecewiseStepParam, calcWeightGranularity, );
    }
  };

  class PyBufferedTransferParam : public BufferedTransferParam {
  public:
    std::string getName() const override {
      PYBIND11_OVERLOAD(std::string, BufferedTransferParam, getName, );
    }
    BufferedTransferParam *clone() const override {
      PYBIND11_OVERLOAD(BufferedTransferParam *, BufferedTransferParam, clone, );
    }
    RPU::DeviceUpdateType implements() const override {
      PYBIND11_OVERLOAD(RPU::DeviceUpdateType, BufferedTransferParam, implements, );
    }
    RPU::BufferedTransferRPUDevice<T> *
    createDevice(int x_size, int d_size, RPU::RealWorldRNG<T> *rng) override {
      PYBIND11_OVERLOAD(
          RPU::BufferedTransferRPUDevice<T> *, BufferedTransferParam, createDevice, x_size, d_size,
          rng);
    }
    T calcWeightGranularity() const override {
      PYBIND11_OVERLOAD(T, BufferedTransferParam, calcWeightGranularity, );
    }
  };

  class PyChoppedTransferParam : public ChoppedTransferParam {
  public:
    std::string getName() const override {
      PYBIND11_OVERLOAD(std::string, ChoppedTransferParam, getName, );
    }
    ChoppedTransferParam *clone() const override {
      PYBIND11_OVERLOAD(ChoppedTransferParam *, ChoppedTransferParam, clone, );
    }
    RPU::DeviceUpdateType implements() const override {
      PYBIND11_OVERLOAD(RPU::DeviceUpdateType, ChoppedTransferParam, implements, );
    }
    RPU::ChoppedTransferRPUDevice<T> *
    createDevice(int x_size, int d_size, RPU::RealWorldRNG<T> *rng) override {
      PYBIND11_OVERLOAD(
          RPU::ChoppedTransferRPUDevice<T> *, ChoppedTransferParam, createDevice, x_size, d_size,
          rng);
    }
    T calcWeightGranularity() const override {
      PYBIND11_OVERLOAD(T, ChoppedTransferParam, calcWeightGranularity, );
    }
  };

  class PyDynamicTransferParam : public DynamicTransferParam {
  public:
    std::string getName() const override {
      PYBIND11_OVERLOAD(std::string, DynamicTransferParam, getName, );
    }
    DynamicTransferParam *clone() const override {
      PYBIND11_OVERLOAD(DynamicTransferParam *, DynamicTransferParam, clone, );
    }
    RPU::DeviceUpdateType implements() const override {
      PYBIND11_OVERLOAD(RPU::DeviceUpdateType, DynamicTransferParam, implements, );
    }
    RPU::DynamicTransferRPUDevice<T> *
    createDevice(int x_size, int d_size, RPU::RealWorldRNG<T> *rng) override {
      PYBIND11_OVERLOAD(
          RPU::DynamicTransferRPUDevice<T> *, DynamicTransferParam, createDevice, x_size, d_size,
          rng);
    }
    T calcWeightGranularity() const override {
      PYBIND11_OVERLOAD(T, DynamicTransferParam, calcWeightGranularity, );
    }
  };

  class PySoftBoundsReferenceParam : public SoftBoundsReferenceParam {
  public:
    std::string getName() const override {
      PYBIND11_OVERLOAD(std::string, SoftBoundsReferenceParam, getName, );
    }
    SoftBoundsReferenceParam *clone() const override {
      PYBIND11_OVERLOAD(SoftBoundsReferenceParam *, SoftBoundsReferenceParam, clone, );
    }
    RPU::DeviceUpdateType implements() const override {
      PYBIND11_OVERLOAD(RPU::DeviceUpdateType, SoftBoundsReferenceParam, implements, );
    }
    RPU::SoftBoundsReferenceRPUDevice<T> *
    createDevice(int x_size, int d_size, RPU::RealWorldRNG<T> *rng) override {
      PYBIND11_OVERLOAD(
          RPU::SoftBoundsReferenceRPUDevice<T> *, SoftBoundsReferenceParam, createDevice, x_size,
          d_size, rng);
    }
    T calcWeightGranularity() const override {
      PYBIND11_OVERLOAD(T, SoftBoundsReferenceParam, calcWeightGranularity, );
    }
  };

  /*
   * Python class definitions.
   */
  py::class_<RPU::SimpleMetaParameter<T>>(m, NAME("FloatingPointTileParameter"))
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
      .def_readwrite("lifetime", &RPU::SimpleMetaParameter<T>::lifetime)
      .def_readwrite("drift", &RPU::SimpleMetaParameter<T>::drift);

  py::class_<RPU::PulsedMetaParameter<T>>(m, NAME("AnalogTileParameter"))
      .def(py::init<>())
      .def(
          "create_array", [](RPU::PulsedMetaParameter<T> &self, int n_cols, int n_rows,
                             AbstractParam *dp) { return self.createRPUArray(n_cols, n_rows, dp); })
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

  py::class_<RPU::PulsedUpdateMetaParameter<T>>(m, NAME("AnalogTileUpdateParameter"))
      .def(py::init<>())
      .def_readwrite("fixed_bl", &RPU::PulsedUpdateMetaParameter<T>::fixed_BL)
      .def_readwrite("desired_bl", &RPU::PulsedUpdateMetaParameter<T>::desired_BL)
      .def_readwrite("d_res_implicit", &RPU::PulsedUpdateMetaParameter<T>::d_res_implicit)
      .def_readwrite("pulse_type", &RPU::PulsedUpdateMetaParameter<T>::pulse_type)
      .def_readwrite("res", &RPU::PulsedUpdateMetaParameter<T>::res)
      .def_readwrite("sto_round", &RPU::PulsedUpdateMetaParameter<T>::sto_round)
      .def_readwrite("um_reg_scale", &RPU::PulsedUpdateMetaParameter<T>::um_reg_scale)
      .def_readwrite("um_grad_scale", &RPU::PulsedUpdateMetaParameter<T>::um_grad_scale)
      .def_readwrite("d_sparsity", &RPU::PulsedUpdateMetaParameter<T>::d_sparsity)
      .def_readwrite("update_management", &RPU::PulsedUpdateMetaParameter<T>::update_management)
      .def_readwrite(
          "update_bl_management", &RPU::PulsedUpdateMetaParameter<T>::update_bl_management)
      .def_readwrite("x_res_implicit", &RPU::PulsedUpdateMetaParameter<T>::x_res_implicit);

  py::class_<RPU::IOMetaParameter<T>>(m, NAME("AnalogTileInputOutputParameter"))
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
      .def_readwrite("nm_assumed_wmax", &RPU::IOMetaParameter<T>::nm_assumed_wmax)
      .def_readwrite("nm_decay", &RPU::IOMetaParameter<T>::nm_decay)
      .def_readwrite("nm_thres", &RPU::IOMetaParameter<T>::nm_thres)
      .def_readwrite("noise_management", &RPU::IOMetaParameter<T>::noise_management)
      .def_readwrite("out_bound", &RPU::IOMetaParameter<T>::out_bound)
      .def_readwrite("out_noise", &RPU::IOMetaParameter<T>::out_noise)
      .def_readwrite("out_noise_std", &RPU::IOMetaParameter<T>::out_noise_std)
      .def_readwrite("out_res", &RPU::IOMetaParameter<T>::out_res)
      .def_readwrite("out_scale", &RPU::IOMetaParameter<T>::out_scale)
      .def_readwrite("out_sto_round", &RPU::IOMetaParameter<T>::out_sto_round)
      .def_readwrite("w_noise", &RPU::IOMetaParameter<T>::w_noise)
      .def_readwrite("w_noise_type", &RPU::IOMetaParameter<T>::w_noise_type)
      .def_readwrite("ir_drop", &RPU::IOMetaParameter<T>::ir_drop)
      .def_readwrite("ir_drop_g_ratio", &RPU::IOMetaParameter<T>::ir_drop_Gw_div_gmax)
      .def_readwrite("out_nonlinearity", &RPU::IOMetaParameter<T>::out_nonlinearity)
      .def_readwrite("out_nonlinearity_std", &RPU::IOMetaParameter<T>::out_nonlinearity_std)
      .def_readwrite("mv_type", &RPU::IOMetaParameter<T>::mv_type)
      .def_readwrite("inp_asymmetry", &RPU::IOMetaParameter<T>::inp_asymmetry)
      .def_readwrite("out_asymmetry", &RPU::IOMetaParameter<T>::out_asymmetry)
      .def_readwrite("slope_calibration", &RPU::IOMetaParameter<T>::slope_calibration)
      .def_readwrite("r_series", &RPU::IOMetaParameter<T>::r_series)
      .def_readwrite("v_offset_std", &RPU::IOMetaParameter<T>::v_offset_std)
      .def_readwrite("v_offset_w_min", &RPU::IOMetaParameter<T>::v_offset_w_min)
      .def_readwrite("w_read_asymmetry_dtod", &RPU::IOMetaParameter<T>::w_read_asymmetry_dtod);

  py::class_<RPU::DriftParameter<T>>(m, NAME("DriftParameter"))
      .def(py::init<>())
      .def_readwrite("nu", &RPU::DriftParameter<T>::nu)
      .def_readwrite("nu_dtod", &RPU::DriftParameter<T>::nu_dtod)
      .def_readwrite("nu_std", &RPU::DriftParameter<T>::nu_std)
      .def_readwrite("wg_ratio", &RPU::DriftParameter<T>::wg_ratio)
      .def_readwrite("g_offset", &RPU::DriftParameter<T>::g_offset)
      .def_readwrite("w_offset", &RPU::DriftParameter<T>::w_offset)
      .def_readwrite("nu_k", &RPU::DriftParameter<T>::nu_k)
      .def_readwrite("log_g0", &RPU::DriftParameter<T>::logG0)
      .def_readwrite("t_0", &RPU::DriftParameter<T>::t0)
      .def_readwrite("reset_tol", &RPU::DriftParameter<T>::reset_tol)
      .def_readwrite("w_noise_std", &RPU::DriftParameter<T>::w_read_std);

  // device params
  py::class_<AbstractParam, PyAbstractParam, RPU::SimpleMetaParameter<T>>(
      m, NAME("AbstractResistiveDeviceParameter"))
      .def(py::init<>())
      .def_readwrite("construction_seed", &AbstractParam::construction_seed);

  py::class_<PulsedBaseParam, PyPulsedBaseParam, AbstractParam>(
      m, NAME("PulsedBaseResistiveDeviceParameter"))
      .def(py::init<>());

  py::class_<SimpleParam, PySimpleParam, AbstractParam>(m, NAME("IdealResistiveDeviceParameter"))
      .def(py::init<>())
      .def_readwrite("reset_std", &SimpleParam::reset_std)
      .def("__str__", [](SimpleParam &self) {
        std::stringstream ss;
        self.printToStream(ss);
        return ss.str();
      });

  py::class_<PulsedParam, PyPulsedParam, AbstractParam>(m, NAME("PulsedResistiveDeviceParameter"))
      .def(py::init<>())
      // Properties from this class.
      .def_readwrite("corrupt_devices_prob", &PulsedParam::corrupt_devices_prob)
      .def_readwrite("corrupt_devices_range", &PulsedParam::corrupt_devices_range)
      .def_readwrite("diffusion_dtod", &PulsedParam::diffusion_dtod)
      .def_readwrite("dw_min", &PulsedParam::dw_min)
      .def_readwrite("dw_min_dtod", &PulsedParam::dw_min_dtod)
      .def_readwrite("dw_min_dtod_log_normal", &PulsedParam::dw_min_dtod_log_normal)
      .def_readwrite("dw_min_std", &PulsedParam::dw_min_std)
      .def_readwrite("enforce_consistency", &PulsedParam::enforce_consistency)
      .def_readwrite("lifetime_dtod", &PulsedParam::lifetime_dtod)
      .def_readwrite("perfect_bias", &PulsedParam::perfect_bias)
      .def_readwrite("reset", &PulsedParam::reset)
      .def_readwrite("reset_std", &PulsedParam::reset_std)
      .def_readwrite("reset_dtod", &PulsedParam::reset_dtod)
      .def_readwrite("up_down", &PulsedParam::up_down)
      .def_readwrite("up_down_dtod", &PulsedParam::up_down_dtod)
      .def_readwrite("w_max", &PulsedParam::w_max)
      .def_readwrite("w_max_dtod", &PulsedParam::w_max_dtod)
      .def_readwrite("w_min", &PulsedParam::w_min)
      .def_readwrite("w_min_dtod", &PulsedParam::w_min_dtod)
      .def_readwrite("count_pulses", &PulsedParam::count_pulses)
      .def("__str__", [](PulsedParam &self) {
        std::stringstream ss;
        self.printToStream(ss);
        return ss.str();
      });

  py::class_<ConstantStepParam, PyConstantStepParam, PulsedParam>(
      m, NAME("ConstantStepResistiveDeviceParameter"))
      .def(py::init<>())
      .def(
          "__str__",
          [](ConstantStepParam &self) {
            std::stringstream ss;
            self.printToStream(ss);
            return ss.str();
          })
      .def(
          "calc_weight_granularity", &ConstantStepParam::calcWeightGranularity,
          R"pbdoc(
        Calculates the granularity of the weights (typically ``dw_min``)

        Returns:
           float: weight granularity
        )pbdoc");

  py::class_<LinearStepParam, PyLinearStepParam, PulsedParam>(
      m, NAME("LinearStepResistiveDeviceParameter"))
      .def(py::init<>())
      .def_readwrite("gamma_up", &LinearStepParam::ls_decrease_up)
      .def_readwrite("gamma_down", &LinearStepParam::ls_decrease_down)
      .def_readwrite("gamma_up_dtod", &LinearStepParam::ls_decrease_up_dtod)
      .def_readwrite("gamma_down_dtod", &LinearStepParam::ls_decrease_down_dtod)
      .def_readwrite("allow_increasing", &LinearStepParam::ls_allow_increasing_slope)
      .def_readwrite("mean_bound_reference", &LinearStepParam::ls_mean_bound_reference)
      .def_readwrite("write_noise_std", &LinearStepParam::write_noise_std)
      .def_readwrite("apply_write_noise_on_set", &LinearStepParam::apply_write_noise_on_set)
      .def_readwrite("mult_noise", &LinearStepParam::ls_mult_noise)
      .def_readwrite("reverse_up", &LinearStepParam::ls_reverse_up)
      .def_readwrite("reverse_down", &LinearStepParam::ls_reverse_down)
      .def_readwrite("reverse_offset", &LinearStepParam::ls_reverse_offset)
      .def(
          "__str__",
          [](LinearStepParam &self) {
            std::stringstream ss;
            self.printToStream(ss);
            return ss.str();
          })
      .def(
          "calc_weight_granularity", &LinearStepParam::calcWeightGranularity,
          R"pbdoc(
        Calculates the granularity of the weights (typically ``dw_min``)

        Returns:
           float: weight granularity
        )pbdoc");

  py::class_<SoftBoundsParam, PySoftBoundsParam, LinearStepParam>(
      m, NAME("SoftBoundsResistiveDeviceParameter"))
      .def_readwrite("mult_noise", &SoftBoundsParam::ls_mult_noise)
      .def_readwrite("write_noise_std", &SoftBoundsParam::write_noise_std)
      .def_readwrite("apply_write_noise_on_set", &SoftBoundsParam::apply_write_noise_on_set)
      .def_readwrite("reverse_up", &SoftBoundsParam::ls_reverse_up)
      .def_readwrite("reverse_down", &SoftBoundsParam::ls_reverse_down)
      .def_readwrite("reverse_offset", &SoftBoundsParam::ls_reverse_offset)
      .def(py::init<>())
      .def(
          "__str__",
          [](SoftBoundsParam &self) {
            std::stringstream ss;
            self.printToStream(ss);
            return ss.str();
          })
      .def(
          "calc_weight_granularity", &SoftBoundsParam::calcWeightGranularity,
          R"pbdoc(
        Calculates the granularity of the weights (typically ``dw_min``)

        Returns:
           float: weight granularity
        )pbdoc");

  py::class_<ExpStepParam, PyExpStepParam, PulsedParam>(m, NAME("ExpStepResistiveDeviceParameter"))
      .def(py::init<>())
      .def_readwrite("A_up", &ExpStepParam::es_A_up)
      .def_readwrite("A_down", &ExpStepParam::es_A_down)
      .def_readwrite("gamma_up", &ExpStepParam::es_gamma_up)
      .def_readwrite("gamma_down", &ExpStepParam::es_gamma_down)
      .def_readwrite("a", &ExpStepParam::es_a)
      .def_readwrite("b", &ExpStepParam::es_b)
      .def_readwrite("write_noise_std", &ExpStepParam::write_noise_std)
      .def_readwrite("apply_write_noise_on_set", &ExpStepParam::apply_write_noise_on_set)
      .def_readwrite("dw_min_std_add", &ExpStepParam::dw_min_std_add)
      .def_readwrite("dw_min_std_slope", &ExpStepParam::dw_min_std_slope)
      .def(
          "__str__",
          [](ExpStepParam &self) {
            std::stringstream ss;
            self.printToStream(ss);
            return ss.str();
          })
      .def(
          "calc_weight_granularity", &ExpStepParam::calcWeightGranularity,
          R"pbdoc(
        Calculates the granularity of the weights (typically ``dw_min``)

        Returns:
           float: weight granularity
        )pbdoc");

  py::class_<VectorParam, PyVectorParam, PulsedBaseParam>(m, NAME("VectorResistiveDeviceParameter"))
      .def(py::init<>())
      .def_readwrite("gamma_vec", &VectorParam::gamma_vec)
      .def_readwrite("update_policy", &VectorParam::update_policy)
      .def_readwrite("first_update_idx", &VectorParam::first_update_idx)
      .def(
          "append_parameter",
          [](VectorParam &self, RPU::AbstractRPUDeviceMetaParameter<T> &dp) {
            return self.appendVecPar(dp);
          },
          py::arg("parameter"),
          R"pbdoc(
           Adds a pulsed base device parameter to the vector device.
           )pbdoc")
      .def(
          "__str__",
          [](VectorParam &self) {
            std::stringstream ss;
            self.printToStream(ss);
            return ss.str();
          })
      .def(
          "calc_weight_granularity", &VectorParam::calcWeightGranularity,
          R"pbdoc(
        Calculates the granularity of the weights (typically ``dw_min``)

        Returns:
           float: weight granularity
        )pbdoc");

  py::class_<OneSidedParam, PyOneSidedParam, VectorParam>(
      m, NAME("OneSidedResistiveDeviceParameter"))
      .def(py::init<>())
      .def_readwrite("refresh_every", &OneSidedParam::refresh_every)
      .def_readwrite("refresh_forward", &OneSidedParam::refresh_io)
      .def_readwrite("refresh_update", &OneSidedParam::refresh_up)
      .def_readwrite("refresh_upper_thres", &OneSidedParam::refresh_upper_thres)
      .def_readwrite("refresh_lower_thres", &OneSidedParam::refresh_lower_thres)
      .def_readwrite("units_in_mbatch", &OneSidedParam::units_in_mbatch)
      .def_readwrite("copy_inverted", &OneSidedParam::copy_inverted)
      .def(
          "__str__",
          [](OneSidedParam &self) {
            std::stringstream ss;
            self.printToStream(ss);
            return ss.str();
          })
      .def(
          "calc_weight_granularity", &OneSidedParam::calcWeightGranularity,
          R"pbdoc(
        Calculates the granularity of the weights (typically ``dw_min``)

        Returns:
           float: weight granularity
        )pbdoc");

  py::class_<TransferParam, PyTransferParam, VectorParam>(
      m, NAME("TransferResistiveDeviceParameter"))
      .def(py::init<>())
      .def_readwrite("gamma", &TransferParam::gamma)
      .def_readwrite("transfer_every", &TransferParam::transfer_every)
      .def_readwrite("no_self_transfer", &TransferParam::no_self_transfer)
      .def_readwrite(
          "transfer_every_vec", &TransferParam::transfer_every_vec) // can this be filled?
      .def_readwrite("units_in_mbatch", &TransferParam::units_in_mbatch)
      .def_readwrite("n_reads_per_transfer", &TransferParam::n_reads_per_transfer)
      .def_readwrite("with_reset_prob", &TransferParam::with_reset_prob)
      .def_readwrite("random_selection", &TransferParam::random_selection)
      .def_readwrite("transfer_columns", &TransferParam::transfer_columns)
      .def_readwrite("transfer_lr", &TransferParam::transfer_lr)
      .def_readwrite("fast_lr", &TransferParam::fast_lr)
      .def_readwrite("transfer_lr_vec", &TransferParam::transfer_lr_vec)
      .def_readwrite("scale_transfer_lr", &TransferParam::scale_transfer_lr)
      .def_readwrite("transfer_forward", &TransferParam::transfer_io)
      .def_readwrite("transfer_update", &TransferParam::transfer_up)
      .def(
          "__str__",
          [](TransferParam &self) {
            std::stringstream ss;
            self.printToStream(ss);
            return ss.str();
          })
      .def(
          "calc_weight_granularity", &TransferParam::calcWeightGranularity,
          R"pbdoc(
        Calculates the granularity of the weights (typically ``dw_min``)

        Returns:
           float: weight granularity
        )pbdoc");

  py::class_<MixedPrecParam, PyMixedPrecParam, SimpleParam>(
      m, NAME("MixedPrecResistiveDeviceParameter"))
      .def(py::init<>())
      .def_readwrite("transfer_every", &MixedPrecParam::transfer_every)
      .def_readwrite("n_rows_per_transfer", &MixedPrecParam::n_rows_per_transfer)
      .def_readwrite("random_row", &MixedPrecParam::random_row)
      .def_readwrite("granularity", &MixedPrecParam::granularity)
      .def_readwrite("compute_sparsity", &MixedPrecParam::compute_sparsity)
      .def_readwrite("n_x_bins", &MixedPrecParam::n_x_bins)
      .def_readwrite("n_d_bins", &MixedPrecParam::n_d_bins)
      .def_readwrite("stoc_round_d", &MixedPrecParam::stoc_round_d)
      .def_readwrite("stoc_round_x", &MixedPrecParam::stoc_round_x)
      .def_readwrite("transfer_lr", &MixedPrecParam::transfer_lr)
      .def(
          "set_device_parameter",
          [](MixedPrecParam &self, const RPU::AbstractRPUDeviceMetaParameter<T> &dp) {
            return self.setDevicePar(dp);
          },
          py::arg("parameter"),
          R"pbdoc(
           Set a pulsed base device parameter of a mixed precision device.
           )pbdoc")
      .def("__str__", [](MixedPrecParam &self) {
        std::stringstream ss;
        self.printToStream(ss);
        return ss.str();
      });

  py::class_<PowStepParam, PyPowStepParam, PulsedParam>(m, NAME("PowStepResistiveDeviceParameter"))
      .def(py::init<>())
      .def_readwrite("pow_gamma", &PowStepParam::ps_gamma)
      .def_readwrite("pow_gamma_dtod", &PowStepParam::ps_gamma_dtod)
      .def_readwrite("pow_up_down", &PowStepParam::ps_gamma_up_down)
      .def_readwrite("pow_up_down_dtod", &PowStepParam::ps_gamma_up_down_dtod)
      .def_readwrite("write_noise_std", &PowStepParam::write_noise_std)
      .def_readwrite("apply_write_noise_on_set", &PowStepParam::apply_write_noise_on_set)
      .def(
          "__str__",
          [](PowStepParam &self) {
            std::stringstream ss;
            self.printToStream(ss);
            return ss.str();
          })
      .def(
          "calc_weight_granularity", &PowStepParam::calcWeightGranularity,
          R"pbdoc(
        Calculates the granularity of the weights (typically ``dw_min``)

        Returns:
           float: weight granularity
        )pbdoc");

  py::class_<PowStepReferenceParam, PyPowStepReferenceParam, PulsedParam>(
      m, NAME("PowStepReferenceResistiveDeviceParameter"))
      .def(py::init<>())
      .def_readwrite("pow_gamma", &PowStepReferenceParam::ps_gamma)
      .def_readwrite("pow_gamma_dtod", &PowStepReferenceParam::ps_gamma_dtod)
      .def_readwrite("pow_up_down", &PowStepReferenceParam::ps_gamma_up_down)
      .def_readwrite("pow_up_down_dtod", &PowStepReferenceParam::ps_gamma_up_down_dtod)
      .def_readwrite("reference_std", &PowStepReferenceParam::reference_std)
      .def_readwrite("reference_mean", &PowStepReferenceParam::reference_mean)
      .def_readwrite("subtract_symmetry_point", &PowStepReferenceParam::subtract_symmetry_point)
      .def_readwrite("n_estimation_steps", &PowStepReferenceParam::n_estimation_steps)
      .def(
          "__str__",
          [](PowStepReferenceParam &self) {
            std::stringstream ss;
            self.printToStream(ss);
            return ss.str();
          })
      .def(
          "calc_weight_granularity", &PowStepReferenceParam::calcWeightGranularity,
          R"pbdoc(
        Calculates the granularity of the weights (typically ``dw_min``)

        Returns:
           float: weight granularity
        )pbdoc");

  py::class_<PiecewiseStepParam, PyPiecewiseStepParam, PulsedParam>(
      m, NAME("PiecewiseStepResistiveDeviceParameter"))
      .def(py::init<>())
      .def_readwrite("piecewise_up", &PiecewiseStepParam::piecewise_up_vec)
      .def_readwrite("piecewise_down", &PiecewiseStepParam::piecewise_down_vec)
      .def_readwrite("write_noise_std", &PiecewiseStepParam::write_noise_std)
      .def_readwrite("apply_write_noise_on_set", &PiecewiseStepParam::apply_write_noise_on_set)
      .def(
          "__str__",
          [](PiecewiseStepParam &self) {
            std::stringstream ss;
            self.printToStream(ss);
            return ss.str();
          })
      .def(
          "calc_weight_granularity", &PiecewiseStepParam::calcWeightGranularity,
          R"pbdoc(
        Calculates the granularity of the weights (typically ``dw_min``)

        Returns:
           float: weight granularity
        )pbdoc");

  py::class_<BufferedTransferParam, PyBufferedTransferParam, TransferParam>(
      m, NAME("BufferedTransferResistiveDeviceParameter"))
      .def(py::init<>())
      .def_readwrite("thres_scale", &BufferedTransferParam::thres_scale)
      .def_readwrite("momentum", &BufferedTransferParam::momentum)
      .def_readwrite("step", &BufferedTransferParam::step)
      .def_readwrite("forget_buffer", &BufferedTransferParam::forget_buffer)
      .def("__str__", [](BufferedTransferParam &self) {
        std::stringstream ss;
        self.printToStream(ss);
        return ss.str();
      });

  py::class_<ChoppedTransferParam, PyChoppedTransferParam, BufferedTransferParam>(
      m, NAME("ChoppedTransferResistiveDeviceParameter"))
      .def(py::init<>())
      .def_readwrite("in_chop_prob", &ChoppedTransferParam::in_chop_prob)
      .def_readwrite("in_chop_random", &ChoppedTransferParam::in_chop_random)
      .def_readwrite("out_chop_prob", &ChoppedTransferParam::out_chop_prob)
      .def_readwrite("auto_scale", &ChoppedTransferParam::auto_scale)
      .def_readwrite(
          "correct_gradient_magnitudes", &ChoppedTransferParam::correct_gradient_magnitudes)
      .def_readwrite("auto_momentum", &ChoppedTransferParam::auto_momentum)
      .def_readwrite("auto_granularity", &ChoppedTransferParam::auto_granularity)
      .def_readwrite("no_buffer", &ChoppedTransferParam::no_buffer)
      .def_readwrite("buffer_granularity", &ChoppedTransferParam::buffer_granularity)
      .def("__str__", [](ChoppedTransferParam &self) {
        std::stringstream ss;
        self.printToStream(ss);
        return ss.str();
      });

  py::class_<DynamicTransferParam, PyDynamicTransferParam, ChoppedTransferParam>(
      m, NAME("DynamicTransferResistiveDeviceParameter"))
      .def(py::init<>())
      .def_readwrite("tail_weightening", &DynamicTransferParam::tail_weightening)
      .def_readwrite("buffer_cap", &DynamicTransferParam::buffer_cap)
      .def_readwrite(
          "experimental_correct_accumulation",
          &DynamicTransferParam::experimental_correct_accumulation)
      .def("__str__", [](DynamicTransferParam &self) {
        std::stringstream ss;
        self.printToStream(ss);
        return ss.str();
      });

  py::class_<SoftBoundsReferenceParam, PySoftBoundsReferenceParam, PulsedParam>(
      m, NAME("SoftBoundsReferenceResistiveDeviceParameter"))
      .def(py::init<>())
      .def_readwrite("slope_up_dtod", &SoftBoundsReferenceParam::slope_up_dtod)
      .def_readwrite("slope_down_dtod", &SoftBoundsReferenceParam::slope_down_dtod)
      .def_readwrite("reference_mean", &SoftBoundsReferenceParam::reference_mean)
      .def_readwrite("reference_std", &SoftBoundsReferenceParam::reference_std)
      .def_readwrite("subtract_symmetry_point", &SoftBoundsReferenceParam::subtract_symmetry_point)
      .def_readwrite("write_noise_std", &SoftBoundsReferenceParam::write_noise_std)
      .def_readwrite(
          "apply_write_noise_on_set", &SoftBoundsReferenceParam::apply_write_noise_on_set)
      .def_readwrite("mult_noise", &SoftBoundsReferenceParam::mult_noise)
      .def(
          "__str__",
          [](SoftBoundsReferenceParam &self) {
            std::stringstream ss;
            self.printToStream(ss);
            return ss.str();
          })
      .def(
          "calc_weight_granularity", &SoftBoundsReferenceParam::calcWeightGranularity,
          R"pbdoc(
        Calculates the granularity of the weights (typically ``dw_min``)

        Returns:
           float: weight granularity
        )pbdoc");
};

#undef NAME

template void declare_rpu_devices<float>(py::module &, std::string);
#ifdef RPU_USE_DOUBLE
template void declare_rpu_devices<double>(py::module &, std::string);
#endif
#ifdef RPU_USE_FP16
template void declare_rpu_devices<half_t>(py::module &, std::string);
#endif
