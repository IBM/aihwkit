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

#include "rpu_base.h"

void declare_rpu_devices(py::module &m) {

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
  using PiecewiseStepParam = RPU::PiecewiseStepRPUDeviceMetaParameter<T>;
  using BufferedTransferParam = RPU::BufferedTransferRPUDeviceMetaParameter<T>;
  using JARTv1bParam = RPU::JARTv1bRPUDeviceMetaParameter<T>;

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

  class PyJARTv1bParam : public JARTv1bParam {
  public:
    std::string getName() const override {
      PYBIND11_OVERLOAD(std::string, JARTv1bParam, getName, );
    }
    JARTv1bParam *clone() const override {
      PYBIND11_OVERLOAD(JARTv1bParam *, JARTv1bParam, clone, );
    }
    RPU::DeviceUpdateType implements() const override {
      PYBIND11_OVERLOAD(RPU::DeviceUpdateType, JARTv1bParam, implements, );
    }
    RPU::JARTv1bRPUDevice<T> *
    createDevice(int x_size, int d_size, RPU::RealWorldRNG<T> *rng) override {
      PYBIND11_OVERLOAD(
          RPU::JARTv1bRPUDevice<T> *, JARTv1bParam, createDevice, x_size, d_size, rng);
    }
    T calcWeightGranularity() const override {
      PYBIND11_OVERLOAD(T, JARTv1bParam, calcWeightGranularity, );
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
      .def_readwrite("lifetime", &RPU::SimpleMetaParameter<T>::lifetime)
      .def_readwrite("drift", &RPU::SimpleMetaParameter<T>::drift);

  py::class_<RPU::PulsedMetaParameter<T>>(m, "AnalogTileParameter")
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

  py::class_<RPU::PulsedUpdateMetaParameter<T>>(m, "AnalogTileUpdateParameter")
      .def(py::init<>())
      .def_readwrite("fixed_bl", &RPU::PulsedUpdateMetaParameter<T>::fixed_BL)
      .def_readwrite("desired_bl", &RPU::PulsedUpdateMetaParameter<T>::desired_BL)
      .def_readwrite("d_res_implicit", &RPU::PulsedUpdateMetaParameter<T>::d_res_implicit)
      .def_readwrite("pulse_type", &RPU::PulsedUpdateMetaParameter<T>::pulse_type)
      .def_readwrite("res", &RPU::PulsedUpdateMetaParameter<T>::res)
      .def_readwrite("sto_round", &RPU::PulsedUpdateMetaParameter<T>::sto_round)
      .def_readwrite("update_management", &RPU::PulsedUpdateMetaParameter<T>::update_management)
      .def_readwrite(
          "update_bl_management", &RPU::PulsedUpdateMetaParameter<T>::update_bl_management)
      .def_readwrite("x_res_implicit", &RPU::PulsedUpdateMetaParameter<T>::x_res_implicit);

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
      .def_readwrite("nm_assumed_wmax", &RPU::IOMetaParameter<T>::nm_assumed_wmax)
      .def_readwrite("nm_decay", &RPU::IOMetaParameter<T>::nm_decay)
      .def_readwrite("nm_thres", &RPU::IOMetaParameter<T>::nm_thres)
      .def_readwrite("noise_management", &RPU::IOMetaParameter<T>::noise_management)
      .def_readwrite("out_bound", &RPU::IOMetaParameter<T>::out_bound)
      .def_readwrite("out_noise", &RPU::IOMetaParameter<T>::out_noise)
      .def_readwrite("out_res", &RPU::IOMetaParameter<T>::out_res)
      .def_readwrite("out_scale", &RPU::IOMetaParameter<T>::out_scale)
      .def_readwrite("out_sto_round", &RPU::IOMetaParameter<T>::out_sto_round)
      .def_readwrite("w_noise", &RPU::IOMetaParameter<T>::w_noise)
      .def_readwrite("w_noise_type", &RPU::IOMetaParameter<T>::w_noise_type)
      .def_readwrite("ir_drop", &RPU::IOMetaParameter<T>::ir_drop)
      .def_readwrite("ir_drop_g_ratio", &RPU::IOMetaParameter<T>::ir_drop_Gw_div_gmax);

  py::class_<RPU::DriftParameter<T>>(m, "DriftParameter")
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
      m, "AbstractResistiveDeviceParameter")
      .def(py::init<>())
      .def_readwrite("construction_seed", &AbstractParam::construction_seed);

  py::class_<PulsedBaseParam, PyPulsedBaseParam, AbstractParam>(
      m, "PulsedBaseResistiveDeviceParameter")
      .def(py::init<>());

  py::class_<SimpleParam, PySimpleParam, AbstractParam>(m, "IdealResistiveDeviceParameter")
      .def(py::init<>())
      .def("__str__", [](SimpleParam &self) {
        std::stringstream ss;
        self.printToStream(ss);
        return ss.str();
      });

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
      .def_readwrite("w_min_dtod", &PulsedParam::w_min_dtod)
      .def("__str__", [](PulsedParam &self) {
        std::stringstream ss;
        self.printToStream(ss);
        return ss.str();
      });

  py::class_<ConstantStepParam, PyConstantStepParam, PulsedParam>(
      m, "ConstantStepResistiveDeviceParameter")
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
      m, "LinearStepResistiveDeviceParameter")
      .def(py::init<>())
      .def_readwrite("gamma_up", &LinearStepParam::ls_decrease_up)
      .def_readwrite("gamma_down", &LinearStepParam::ls_decrease_down)
      .def_readwrite("gamma_up_dtod", &LinearStepParam::ls_decrease_up_dtod)
      .def_readwrite("gamma_down_dtod", &LinearStepParam::ls_decrease_down_dtod)
      .def_readwrite("allow_increasing", &LinearStepParam::ls_allow_increasing_slope)
      .def_readwrite("mean_bound_reference", &LinearStepParam::ls_mean_bound_reference)
      .def_readwrite("write_noise_std", &LinearStepParam::write_noise_std)
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
      m, "SoftBoundsResistiveDeviceParameter")
      .def_readwrite("mult_noise", &SoftBoundsParam::ls_mult_noise)
      .def_readwrite("write_noise_std", &SoftBoundsParam::write_noise_std)
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

  py::class_<ExpStepParam, PyExpStepParam, PulsedParam>(m, "ExpStepResistiveDeviceParameter")
      .def(py::init<>())
      .def_readwrite("A_up", &ExpStepParam::es_A_up)
      .def_readwrite("A_down", &ExpStepParam::es_A_down)
      .def_readwrite("gamma_up", &ExpStepParam::es_gamma_up)
      .def_readwrite("gamma_down", &ExpStepParam::es_gamma_down)
      .def_readwrite("a", &ExpStepParam::es_a)
      .def_readwrite("b", &ExpStepParam::es_b)
      .def_readwrite("write_noise_std", &ExpStepParam::write_noise_std)
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

  py::class_<JARTv1bParam, PyJARTv1bParam, PulsedParam>(m, "JARTv1bResistiveDeviceParameter")
      .def(py::init<>())
      .def_readwrite("write_noise_std", &JARTv1bParam::real_write_noise_std)
      .def_readwrite("alpha0", &JARTv1bParam::alpha0)
      .def_readwrite("alpha1", &JARTv1bParam::alpha1)
      .def_readwrite("alpha2", &JARTv1bParam::alpha2)
      .def_readwrite("alpha3", &JARTv1bParam::alpha3)
      .def_readwrite("beta0", &JARTv1bParam::beta0)
      .def_readwrite("beta1", &JARTv1bParam::beta1)
      .def_readwrite("c0", &JARTv1bParam::c0)
      .def_readwrite("c1", &JARTv1bParam::c1)
      .def_readwrite("c2", &JARTv1bParam::c2)
      .def_readwrite("c3", &JARTv1bParam::c3)
      .def_readwrite("d0", &JARTv1bParam::d0)
      .def_readwrite("d1", &JARTv1bParam::d1)
      .def_readwrite("d2", &JARTv1bParam::d2)
      .def_readwrite("d3", &JARTv1bParam::d3)
      .def_readwrite("f0", &JARTv1bParam::f0)
      .def_readwrite("f1", &JARTv1bParam::f1)
      .def_readwrite("f2", &JARTv1bParam::f2)
      .def_readwrite("f3", &JARTv1bParam::f3)
      .def_readwrite("g0", &JARTv1bParam::g0)
      .def_readwrite("g1", &JARTv1bParam::g1)
      .def_readwrite("h0", &JARTv1bParam::h0)
      .def_readwrite("h1", &JARTv1bParam::h1)
      .def_readwrite("h2", &JARTv1bParam::h2)
      .def_readwrite("h3", &JARTv1bParam::h3)
      .def_readwrite("j0", &JARTv1bParam::j_0)
      .def_readwrite("k0", &JARTv1bParam::k0)
      .def_readwrite("T0", &JARTv1bParam::T0)
      .def_readwrite("un", &JARTv1bParam::un)
      .def_readwrite("Ndiscmax", &JARTv1bParam::Ndiscmax)
      .def_readwrite("Ndiscmin", &JARTv1bParam::Ndiscmin)
      .def_readwrite("Nplug", &JARTv1bParam::Nplug)
      .def_readwrite("a", &JARTv1bParam::a)
      .def_readwrite("ny0", &JARTv1bParam::ny0)
      .def_readwrite("dWa", &JARTv1bParam::dWa)
      .def_readwrite("Rth0", &JARTv1bParam::Rth0)
      .def_readwrite("rdet", &JARTv1bParam::rdet)
      .def_readwrite("lcell", &JARTv1bParam::lcell)
      .def_readwrite("ldet", &JARTv1bParam::ldet)
      .def_readwrite("Rtheff_scaling", &JARTv1bParam::Rtheff_scaling)
      .def_readwrite("RseriesTiOx", &JARTv1bParam::RseriesTiOx)
      .def_readwrite("R0", &JARTv1bParam::R0)
      .def_readwrite("Rthline", &JARTv1bParam::Rthline)
      .def_readwrite("alphaline", &JARTv1bParam::alphaline)
      .def_readwrite("read_voltage", &JARTv1bParam::read_voltage)
      .def_readwrite("pulse_voltage_SET", &JARTv1bParam::pulse_voltage_SET)
      .def_readwrite("pulse_voltage_RESET", &JARTv1bParam::pulse_voltage_RESET)
      .def_readwrite("pulse_length", &JARTv1bParam::pulse_length)
      .def_readwrite("base_time_step", &JARTv1bParam::base_time_step)
      .def_readwrite("Ndisc_min_bound", &JARTv1bParam::Ndisc_min_bound)
      .def_readwrite("Ndisc_max_bound", &JARTv1bParam::Ndisc_max_bound)
      .def_readwrite("Ndiscmax_dtod", &JARTv1bParam::Ndiscmax_dtod)
      .def_readwrite("Ndiscmax_dtod_upper_bound", &JARTv1bParam::Ndiscmax_dtod_upper_bound)
      .def_readwrite("Ndiscmax_dtod_lower_bound", &JARTv1bParam::Ndiscmax_dtod_lower_bound)
      .def_readwrite("Ndiscmin_dtod", &JARTv1bParam::Ndiscmin_dtod)
      .def_readwrite("Ndiscmin_dtod_upper_bound", &JARTv1bParam::Ndiscmin_dtod_upper_bound)
      .def_readwrite("Ndiscmin_dtod_lower_bound", &JARTv1bParam::Ndiscmin_dtod_lower_bound)
      .def_readwrite("ldet_dtod", &JARTv1bParam::ldet_dtod)
      .def_readwrite("ldet_dtod_upper_bound", &JARTv1bParam::ldet_dtod_upper_bound)
      .def_readwrite("ldet_dtod_lower_bound", &JARTv1bParam::ldet_dtod_lower_bound)
      .def_readwrite("rdet_dtod", &JARTv1bParam::rdet_dtod)
      .def_readwrite("rdet_dtod_upper_bound", &JARTv1bParam::rdet_dtod_upper_bound)
      .def_readwrite("rdet_dtod_lower_bound", &JARTv1bParam::rdet_dtod_lower_bound)
      .def_readwrite("Ndiscmax_std", &JARTv1bParam::Ndiscmax_std)
      .def_readwrite("Ndiscmax_ctoc_upper_bound_old", &JARTv1bParam::Ndiscmax_ctoc_upper_bound_old)
      .def_readwrite("Ndiscmax_ctoc_lower_bound_old", &JARTv1bParam::Ndiscmax_ctoc_lower_bound_old)
      .def_readwrite("Ndiscmax_ctoc_upper_bound", &JARTv1bParam::Ndiscmax_ctoc_upper_bound)
      .def_readwrite("Ndiscmax_ctoc_lower_bound", &JARTv1bParam::Ndiscmax_ctoc_lower_bound)
      .def_readwrite("Ndiscmin_std", &JARTv1bParam::Ndiscmin_std)
      .def_readwrite("Ndiscmin_ctoc_upper_bound_old", &JARTv1bParam::Ndiscmin_ctoc_upper_bound_old)
      .def_readwrite("Ndiscmin_ctoc_lower_bound_old", &JARTv1bParam::Ndiscmin_ctoc_lower_bound_old)
      .def_readwrite("Ndiscmin_ctoc_upper_bound", &JARTv1bParam::Ndiscmin_ctoc_upper_bound)
      .def_readwrite("Ndiscmin_ctoc_lower_bound", &JARTv1bParam::Ndiscmin_ctoc_lower_bound)
      .def_readwrite("ldet_std", &JARTv1bParam::ldet_std)
      .def_readwrite("ldet_std_slope", &JARTv1bParam::ldet_std_slope)
      .def_readwrite("ldet_ctoc_upper_bound_old", &JARTv1bParam::ldet_ctoc_upper_bound_old)
      .def_readwrite("ldet_ctoc_lower_bound_old", &JARTv1bParam::ldet_ctoc_lower_bound_old)
      .def_readwrite("ldet_ctoc_upper_bound", &JARTv1bParam::ldet_ctoc_upper_bound)
      .def_readwrite("ldet_ctoc_lower_bound", &JARTv1bParam::ldet_ctoc_lower_bound)
      .def_readwrite("rdet_std", &JARTv1bParam::rdet_std)
      .def_readwrite("rdet_std_slope", &JARTv1bParam::rdet_std_slope)
      .def_readwrite("rdet_ctoc_upper_bound_old", &JARTv1bParam::rdet_ctoc_upper_bound_old)
      .def_readwrite("rdet_ctoc_lower_bound_old", &JARTv1bParam::rdet_ctoc_lower_bound_old)
      .def_readwrite("rdet_ctoc_upper_bound", &JARTv1bParam::rdet_ctoc_upper_bound)
      .def_readwrite("rdet_ctoc_lower_bound", &JARTv1bParam::rdet_ctoc_lower_bound)
      .def(
          "__str__",
          [](JARTv1bParam &self) {
            std::stringstream ss;
            self.printToStream(ss);
            return ss.str();
          })
      .def(
          "calc_weight_granularity", &JARTv1bParam::calcWeightGranularity,
          R"pbdoc(
        Calculates the granularity of the weights (typically ``dw_min``)

        Returns:
           float: weight granularity
        )pbdoc");

  py::class_<VectorParam, PyVectorParam, PulsedBaseParam>(m, "VectorResistiveDeviceParameter")
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

  py::class_<OneSidedParam, PyOneSidedParam, VectorParam>(m, "OneSidedResistiveDeviceParameter")
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

  py::class_<TransferParam, PyTransferParam, VectorParam>(m, "TransferResistiveDeviceParameter")
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

  py::class_<MixedPrecParam, PyMixedPrecParam, SimpleParam>(m, "MixedPrecResistiveDeviceParameter")
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

  py::class_<PowStepParam, PyPowStepParam, PulsedParam>(m, "PowStepResistiveDeviceParameter")
      .def(py::init<>())
      .def_readwrite("pow_gamma", &PowStepParam::ps_gamma)
      .def_readwrite("pow_gamma_dtod", &PowStepParam::ps_gamma_dtod)
      .def_readwrite("pow_up_down", &PowStepParam::ps_gamma_up_down)
      .def_readwrite("pow_up_down_dtod", &PowStepParam::ps_gamma_up_down_dtod)
      .def_readwrite("write_noise_std", &PowStepParam::write_noise_std)
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

  py::class_<PiecewiseStepParam, PyPiecewiseStepParam, PulsedParam>(
      m, "PiecewiseStepResistiveDeviceParameter")
      .def(py::init<>())
      .def_readwrite("piecewise_up", &PiecewiseStepParam::piecewise_up_vec)
      .def_readwrite("piecewise_down", &PiecewiseStepParam::piecewise_down_vec)
      .def_readwrite("write_noise_std", &PiecewiseStepParam::write_noise_std)
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
      m, "BufferedTransferResistiveDeviceParameter")
      .def(py::init<>())
      .def_readwrite("thres_scale", &BufferedTransferParam::thres_scale)
      .def_readwrite("momentum", &BufferedTransferParam::momentum)
      .def_readwrite("step", &BufferedTransferParam::step)
      .def(
          "__str__",
          [](BufferedTransferParam &self) {
            std::stringstream ss;
            self.printToStream(ss);
            return ss.str();
          })
      .def(
          "calc_weight_granularity", &BufferedTransferParam::calcWeightGranularity,
          R"pbdoc(
        Calculates the granularity of the weights (typically ``dw_min``)

        Returns:
           float: weight granularity
        )pbdoc");

  /**
   * Helper enums.
   **/
  py::enum_<RPU::BoundManagementType>(m, "BoundManagementType")
      .value("None", RPU::BoundManagementType::None)
      .value("Iterative", RPU::BoundManagementType::Iterative)
      .value("IterativeWorstCase", RPU::BoundManagementType::IterativeWorstCase)
      .value("Shift", RPU::BoundManagementType::Shift);

  py::enum_<RPU::VectorDeviceUpdatePolicy>(m, "VectorUnitCellUpdatePolicy")
      .value("All", RPU::VectorDeviceUpdatePolicy::All)
      .value("SingleFixed", RPU::VectorDeviceUpdatePolicy::SingleFixed)
      .value("SingleSequential", RPU::VectorDeviceUpdatePolicy::SingleSequential)
      .value("SingleRandom", RPU::VectorDeviceUpdatePolicy::SingleRandom);

  py::enum_<RPU::NoiseManagementType>(m, "NoiseManagementType")
      .value("None", RPU::NoiseManagementType::None)
      .value("AbsMax", RPU::NoiseManagementType::AbsMax)
      .value("AbsMaxNPSum", RPU::NoiseManagementType::AbsMaxNPSum)
      .value("Max", RPU::NoiseManagementType::Max)
      .value("Constant", RPU::NoiseManagementType::Constant)
      .value("AverageAbsMax", RPU::NoiseManagementType::AverageAbsMax);

  py::enum_<RPU::OutputWeightNoiseType>(m, "WeightNoiseType")
      .value("None", RPU::OutputWeightNoiseType::None)
      .value("AdditiveConstant", RPU::OutputWeightNoiseType::AdditiveConstant)
      .value("PCMRead", RPU::OutputWeightNoiseType::PCMRead);

  py::enum_<RPU::PulseType>(m, "PulseType")
      .value("None", RPU::PulseType::None)
      .value("StochasticCompressed", RPU::PulseType::StochasticCompressed)
      .value("Stochastic", RPU::PulseType::Stochastic)
      .value("NoneWithDevice", RPU::PulseType::NoneWithDevice)
      .value("MeanCount", RPU::PulseType::MeanCount)
      .value("DeterministicImplicit", RPU::PulseType::DeterministicImplicit);
}
