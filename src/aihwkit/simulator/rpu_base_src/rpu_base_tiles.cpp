/**
 * (C) Copyright 2020, 2021, 2022 IBM. All Rights Reserved.
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

#define CHECK_CPU(x) TORCH_CHECK(x.device() == torch::kCPU, #x " must be a CPU tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_TORCH_INPUT(x)                                                                       \
  CHECK_CPU(x);                                                                                    \
  CHECK_CONTIGUOUS(x)

void declare_rpu_tiles(py::module &m) {
  using Class = RPU::RPUSimple<T>;
  using ClassPulsed = RPU::RPUPulsed<T>;

  py::class_<RPU::WeightModifierParameter>(m, "WeightModifierParameter")
      .def(py::init<>())
      .def_readwrite("std_dev", &RPU::WeightModifierParameter::std_dev)
      .def_readwrite("res", &RPU::WeightModifierParameter::res)
      .def_readwrite("sto_round", &RPU::WeightModifierParameter::sto_round)
      .def_readwrite("dorefa_clip", &RPU::WeightModifierParameter::dorefa_clip)
      .def_readwrite("pdrop", &RPU::WeightModifierParameter::pdrop)
      .def_readwrite("enable_during_test", &RPU::WeightModifierParameter::enable_during_test)
      .def_readwrite("copy_last_column", &RPU::WeightModifierParameter::copy_last_column)
      .def_readwrite("rel_to_actual_wmax", &RPU::WeightModifierParameter::rel_to_actual_wmax)
      .def_readwrite("assumed_wmax", &RPU::WeightModifierParameter::assumed_wmax)
      .def_readwrite("coeff0", &RPU::WeightModifierParameter::coeff0)
      .def_readwrite("coeff1", &RPU::WeightModifierParameter::coeff1)
      .def_readwrite("coeff2", &RPU::WeightModifierParameter::coeff2)
      .def_readwrite("type", &RPU::WeightModifierParameter::type);

  py::enum_<RPU::WeightModifierType>(m, "WeightModifierType")
      .value("Copy", RPU::WeightModifierType::Copy)
      .value("Discretize", RPU::WeightModifierType::Discretize)
      .value("MultNormal", RPU::WeightModifierType::MultNormal)
      .value("AddNormal", RPU::WeightModifierType::AddNormal)
      .value("DiscretizeAddNormal", RPU::WeightModifierType::DiscretizeAddNormal)
      .value("DoReFa", RPU::WeightModifierType::DoReFa)
      .value("Poly", RPU::WeightModifierType::Poly);

  py::class_<RPU::WeightClipParameter>(m, "WeightClipParameter")
      .def(py::init<>())
      .def_readwrite("fixed_value", &RPU::WeightClipParameter::fixed_value)
      .def_readwrite("sigma", &RPU::WeightClipParameter::sigma)
      .def_readwrite("type", &RPU::WeightClipParameter::type);

  py::enum_<RPU::WeightClipType>(m, "WeightClipType")
      .value("None", RPU::WeightClipType::None)
      .value("FixedValue", RPU::WeightClipType::FixedValue)
      .value("LayerGaussian", RPU::WeightClipType::LayerGaussian)
      .value("AverageChannelMax", RPU::WeightClipType::AverageChannelMax);

  py::class_<Class>(
      m, "FloatingPointTile",
      R"pbdoc(
    Floating point tile.

    Args:
        x_size: ``X`` size of the tile.
        d_size: ``D`` size of the tile.
    )pbdoc")
      .def(py::init<int, int>(), py::arg("x_size"), py::arg("d_size"))
      .def(
          "__str__",
          [](Class &self) {
            std::stringstream ss;
            self.printParametersToStream(ss);
            self.printToStream(ss);
            return ss.str();
          })
      .def(
          "get_info",
          [](Class &self) {
            std::stringstream ss;
            self.printParametersToStream(ss);
            self.printToStream(ss);
            return ss.str();
          })
      .def(
          "get_brief_info",
          [](Class &self) {
            std::stringstream ss;
            self.printToStream(ss);
            return ss.str();
          })
      .def(
          "get_learning_rate", &Class::getLearningRate,
          R"pbdoc(
           Return the tile learning rate.

           Returns:
               float: the tile learning rate.
           )pbdoc")
      .def(
          "get_x_size", &Class::getXSize,
          R"pbdoc(
           Return the tile input dimensions (x-size).

           Returns:
               int: the tile number of columns (including bias if available)
           )pbdoc")
      .def(
          "get_d_size", &Class::getDSize,
          R"pbdoc(
           Return the tile output dimensions (d-size).

           Returns:
               int: the tile number of rows
           )pbdoc")
      .def(
          "set_learning_rate", &Class::setLearningRate, py::arg("learning_rate"),
          R"pbdoc(
           Set the tile learning rate.

           Set the tile learning rate to ``-learning_rate``. Please note that the learning
           rate is always taken to be negative (because of the meaning in gradient descent) and
           positive learning rates are not supported.

           Args:
               learning_rate: the desired learning rate.
           )pbdoc")

      .def(
          "get_weights",
          [](Class &self) {
            // Build the buffer.
            py::array_t<T> weights = py::array_t<T>({self.getDSize(), self.getXSize()});
            py::buffer_info weights_buffer = weights.request();

            // Call RPU function.
            self.getWeights((T *)weights_buffer.ptr);
            return weights;
          },
          R"pbdoc(
           Return the exact tile weights.

           Return the tile weights by producing and exact copy.

           Note:
               This is **not** hardware realistic, and is used for debug purposes only.

           Returns:
               ndarray: the ``[d_size, x_size]`` weight matrix.
           )pbdoc")

      .def(
          "set_weights",
          [](Class &self, py::array_t<T> weights) {
            // Validate the weights dimensions.
            if (weights.ndim() != 2 || weights.shape(0) != self.getDSize() ||
                weights.shape(1) != self.getXSize()) {
              throw std::runtime_error(
                  "Invalid weights dimensions: expected [" + std::to_string(self.getDSize()) + "," +
                  std::to_string(self.getXSize()) + "] array");
            }

            // Build the buffer.
            py::buffer_info weights_buffer = weights.request();

            // Call RPU function.
            return self.setWeights((T *)weights_buffer.ptr);
          },
          py::arg("weights"),
          R"pbdoc(
           Set the tile weights exactly.

           Set the tile weights to the exact values of the ``weights`` parameter.

           Note:
               This is **not** hardware realistic, and is used for debug purposes only.

           Args:
               weights: ``[d_size, x_size]`` weight matrix.
           )pbdoc")

      .def(
          "get_weights_realistic",
          [](Class &self) {
            // Build the buffer.
            py::array_t<T> weights = py::array_t<T>({self.getDSize(), self.getXSize()});
            py::buffer_info weights_buffer = weights.request();

            // Call RPU function.
            self.getWeightsReal((T *)weights_buffer.ptr);
            return weights;
          },
          R"pbdoc(
           Return the tile weights.

           Return the tile weights by using the forward pass. This is the hardware realistic
           version of reading out the weights.

           Returns:
               ndarray: the ``[d_size, x_size]`` weight matrix.
           )pbdoc")

      .def(
          "set_weights_realistic",
          [](Class &self, py::array_t<T> weights, int n_loops = 10) {
            // Validate the weights dimensions.
            if (weights.ndim() != 2 || weights.shape(0) != self.getDSize() ||
                weights.shape(1) != self.getXSize()) {
              throw std::runtime_error(
                  "Invalid weights dimensions: expected [" + std::to_string(self.getDSize()) + "," +
                  std::to_string(self.getXSize()) + "] array");
            }

            // Build the buffer.
            py::buffer_info weights_buffer = weights.request();

            // Call RPU function.
            return self.setWeightsReal((T *)weights_buffer.ptr, n_loops);
          },
          py::arg("weights"), py::arg("n_loops") = 10,
          R"pbdoc(
           Set the tile weights by using the forward/update pass.

           Set the tile weights to the ``weights`` parameter by using the forward/update
           pass. This is the hardware realistic version for handling setting of the weights.

           Args:
               weights: ``[d_size, x_size]`` weight matrix.
               n_loops: number of times the columns of the weights are set in a closed-loop manner.
                   A value of ``1`` means that all columns in principle receive enough pulses to
                   change from ``w_min`` to ``w_max``.
           )pbdoc")
      .def(
          "set_shared_weights",
          [](Class &self, torch::Tensor weights) {
            CHECK_TORCH_INPUT(weights);
            if (weights.dim() != 2 || weights.size(0) != self.getDSize() ||
                weights.size(1) != self.getXSize()) {
              throw std::runtime_error(
                  "Invalid weights dimensions: expected [" + std::to_string(self.getDSize()) + "," +
                  std::to_string(self.getXSize()) + "] array");
            }
            std::lock_guard<std::mutex> lock(self.mutex_);
            return self.setSharedWeights(weights.data_ptr<T>());
          },
          py::arg("weights"))
      .def(
          "set_delta_weights",
          [](Class &self, torch::Tensor delta_weights) {
            CHECK_TORCH_INPUT(delta_weights);

            if (delta_weights.dim() != 2 || delta_weights.size(0) != self.getDSize() ||
                delta_weights.size(1) != self.getXSize()) {
              throw std::runtime_error(
                  "Invalid delta weights dimensions: expected [" + std::to_string(self.getDSize()) +
                  "," + std::to_string(self.getXSize()) + "] array");
            }
            std::lock_guard<std::mutex> lock(self.mutex_);
            return self.setDeltaWeights(delta_weights.data_ptr<T>());
          },
          py::arg("delta_weights"))
      .def(
          "reset_delta_weights",
          [](Class &self) {
            std::lock_guard<std::mutex> lock(self.mutex_);
            return self.setDeltaWeights(nullptr);
          })
      .def(
          "get_shared_weights_if", &Class::getSharedWeightsIf,
          R"pbdoc(
           Returns whether weight is shared.
           )pbdoc")
      .def(
          "get_parameters", &Class::getPar,
          R"pbdoc(
           Returns the current meta parameter structure.
           )pbdoc")
      .def(
          "set_weights_uniform_random",
          [](Class &self, float min_value, float max_value) {
            std::lock_guard<std::mutex> lock(self.mutex_);
            self.setWeightsUniformRandom(min_value, max_value);
          },
          py::arg("min_value"), py::arg("max_value"),
          R"pbdoc(
           Sets weights uniformlay in the range ``min_value`` to ``max_value``.

           Args:
               min_value: lower bound of uniform distribution
               max_value: upper bound
           )pbdoc")
      .def(
          "decay_weights",
          [](Class &self, float alpha = 1.0) {
            std::lock_guard<std::mutex> lock(self.mutex_);
            self.decayWeights(alpha, false);
          },
          py::arg("alpha") = 1.0,
          R"pbdoc(
           Decays the weights::

              W *= (1 - alpha / life_time)

           An analog tile will have possible non-ideal version of this decay.

           Args:
               alpha: decay scale
           )pbdoc")
      .def(
          "drift_weights",
          [](Class &self, float time_since_last_call) {
            std::lock_guard<std::mutex> lock(self.mutex_);
            self.driftWeights(time_since_last_call);
          },
          py::arg("time_since_last_call"),
          R"pbdoc(
           Drift weights according to a power law::

              W = W0*(delta_t/t0)^(-nu_actual)

           Applies the weight drift to all unchanged weight elements
           (judged by ``reset_tol``) and resets the drift for those
           that have changed (nu is not re-drawn, however). Each
           device might have a different version of this drift.

           Args:
               time_since_last_call: This is the time between the calls (``delta_t``),
                   typically the time to process a mini-batch for the
                   network.
           )pbdoc")
      .def(
          "clip_weights",
          [](Class &self, ::RPU::WeightClipParameter &wclip_par) {
            std::lock_guard<std::mutex> lock(self.mutex_);
            self.clipWeights(wclip_par);
          },
          py::arg("weight_clipper_params"),
          R"pbdoc(
           Clips the weights for use of hardware-aware training.

           Several clipping types are available, see ``WeightClipParameter``.

           Args:
               weight_clipper_params: parameters of the clipping.
           )pbdoc")
      .def(
          "modify_weights",
          [](Class &self, ::RPU::WeightModifierParameter &wmpar) {
            std::lock_guard<std::mutex> lock(self.mutex_);
            self.modifyFBWeights(wmpar);
          },
          py::arg("weight_modifier_params"),
          R"pbdoc(
           Modifies the weights in forward and backward (but not update) pass for use of hardware-aware training.

           Several modifier types are available, see ``WeightModifierParameter``.

           Args:
               weight_modifier_params: parameters of the modifications.
           )pbdoc")
      .def(
          "diffuse_weights",
          [](Class &self) {
            std::lock_guard<std::mutex> lock(self.mutex_);
            self.diffuseWeights();
          },
          R"pbdoc(
           Diffuse the weights.

           Diffuse the weights::

              W += diffusion_rate * Gaussian noise

           An analog tile will have a possible non-ideal version of this diffusion.
           )pbdoc")
      .def(
          "reset_columns",
          [](Class &self, int start_col, int n_cols, T reset_prob) {
            std::lock_guard<std::mutex> lock(self.mutex_);
            return self.resetCols(start_col, n_cols, reset_prob);
          },
          py::arg("start_column_idx") = 0, py::arg("num_columns") = 1, py::arg("reset_prob") = 1.0,
          R"pbdoc(
           Resets the weights with device-to-device and cycle-to-cycle
           variability (depending on device type), typically::

              W_ij = xi*reset_std + reset_bias_ij

           Args:
               start_col_idx: a start index of columns (``0..x_size-1``)
               num_columns: how many consecutive columns to reset (with circular warping)
               reset_prob: individual probability of reset.
           )pbdoc")
      .def(
          "forward",
          [](Class &self, const torch::Tensor &x_input_, bool bias = false, bool x_trans = false,
             bool d_trans = false, bool is_test = false) {
            auto x_input = x_input_.contiguous();
            CHECK_TORCH_INPUT(x_input);

            if (x_input.dim() < 1) {
              throw std::runtime_error(
                  "Invalid x_input dimensions: expected at least 1 dimensional array");
            }
            int in_size = x_trans ? x_input.size(0) : x_input.size(-1);
            int expected_in_size = self.getXSize() - (bias ? 1 : 0);
            int m_batch = x_input.numel() / in_size;
            int out_size = self.getDSize();

            // Validate the x_input dimensions.
            if (in_size != expected_in_size) {
              std::string shape_str = x_trans ? ("[*, " + std::to_string(expected_in_size) + "]")
                                              : ("[" + std::to_string(expected_in_size) + ",*]");
              throw std::runtime_error(
                  "Invalid x_input dimensions: expected " + shape_str + " array");
            }

            // Build the buffers.
            std::vector<int64_t> dims(x_input.sizes().begin(), x_input.sizes().end());
            if (d_trans) {
              dims[0] = out_size;
            } else {
              dims[dims.size() - 1] = out_size;
            }
            torch::Tensor d_output = torch::empty(dims, x_input.options());

            // Call RPU function.
            std::lock_guard<std::mutex> lock(self.mutex_);
            self.forward(
                x_input.template data_ptr<T>(), d_output.template data_ptr<T>(), bias, m_batch,
                x_trans, d_trans, is_test);
            return d_output;
          },
          py::arg("x_input"), py::arg("bias") = false, py::arg("x_trans") = false,
          py::arg("d_trans") = false, py::arg("is_test") = false,
          R"pbdoc(
           Compute the dot product (forward pass).

           Compute the dot product:
           .. math:

               \mathbf{y} = W\mathbf{x} [+ \mathbf{b}]

           where :math:`\mathbf{x}` is the input and :math:`W` is the ``[d_size, x_size]``
           current weight matrix. If ``bias`` is True, then it is assumes that a
           bias row is added to the analog tile weights.  The input :math:`\mathbf{x}` is
           then  expected to be of size ``x_size -1`` , as internally it will be
           expanded by a 1, to match the bias row in the tile weights.

           An analog tile will have a possible non-ideal version of this forward pass.

           Args:
               x_input: ``[N,*, x_size (- 1)]`` input :math:`\mathbf{x}` torch::Tensor.
               bias: whether to use bias.
               x_trans: whether the ``x_input`` matrix is transposed. That is of size ``[x_size (- 1), *, N]``
               d_trans: whether the ``d`` matrix is transposed.
               is_test: whether inference (true) mode or training (false)

           Returns:
               torch::tensor: ``[N, *, d_size]`` or ``[d_size, *, N]`` matrix.
           )pbdoc")

      .def(
          "backward",
          [](Class &self, const torch::Tensor &d_input_, bool bias = false, bool d_trans = false,
             bool x_trans = false) {
            auto d_input = d_input_.contiguous();
            CHECK_TORCH_INPUT(d_input);

            if (d_input.dim() < 1) {
              throw std::runtime_error(
                  "Invalid d_input dimensions: expected at least 1 dimensional array");
            }
            int in_size = d_trans ? d_input.size(0) : d_input.size(-1);
            int expected_in_size = self.getDSize();
            int m_batch = d_input.numel() / in_size;
            int out_size = self.getXSize() - (bias ? 1 : 0);

            // Validate the d_input dimensions.
            if (in_size != expected_in_size) {
              std::string shape_str = d_trans ? ("[*, " + std::to_string(expected_in_size) + "]")
                                              : ("[" + std::to_string(expected_in_size) + ", *]");
              throw std::runtime_error(
                  "Invalid d_input dimensions: expected " + shape_str + " array");
            }

            // Build the buffers.
            std::vector<int64_t> dims(d_input.sizes().begin(), d_input.sizes().end());
            if (x_trans) {
              dims[0] = out_size;
            } else {
              dims[dims.size() - 1] = out_size;
            }
            torch::Tensor x_output = torch::empty(dims, d_input.options());

            // Call RPU function.
            std::lock_guard<std::mutex> lock(self.mutex_);
            self.backward(
                d_input.template data_ptr<T>(), x_output.template data_ptr<T>(), bias, m_batch,
                d_trans, x_trans);
            return x_output;
          },
          py::arg("d_input"), py::arg("bias") = false, py::arg("d_trans") = false,
          py::arg("x_trans") = false,
          R"pbdoc(
           Compute the transposed dot product (backward pass).

           Compute the transposed dot product:
           .. math:

               \mathbf{y} = W\mathbf{d}

           where :math:`\mathbf{d}` is the input and :math:`W` is the  current
           weight matrix (of size ``[d_size, x_size]``).

           An analog tile will have a possible non-ideal version of this backward pass.

           Args:
               d_input: ``[N, *,  d_size]`` input :math:`\mathbf{d}` torch::Tensor.
               bias: whether to use bias.
               d_trans: whether the ``d_input`` matrix is transposed. That is of size ``[d_size, *, N]``
               x_trans: whether the ``x`` output matrix is transposed.

           Returns:
               torch::Tensor: ``[N, *, x_size (-1)]`` or ``[x_size (-1), *, N]`` torch::Tensor.
           )pbdoc")

      .def(
          "update",
          [](Class &self, const torch::Tensor &x_input_, const torch::Tensor &d_input_,
             bool bias = false, bool x_trans = false, bool d_trans = false) {
            auto d_input = d_input_.contiguous();
            auto x_input = x_input_.contiguous();

            CHECK_TORCH_INPUT(d_input);
            CHECK_TORCH_INPUT(x_input);

            if ((x_input.dim() < 1) || (d_input.dim() < 1)) {
              throw std::runtime_error(
                  "Invalid x_input/d_input dimensions: expected at least 1 dimensional array");
            }

            int in_size = x_trans ? x_input.size(0) : x_input.size(-1);
            int expected_in_size = self.getXSize() - (bias ? 1 : 0);
            int m_batch = x_input.numel() / in_size;

            int out_size = d_trans ? d_input.size(0) : d_input.size(-1);
            int expected_out_size = self.getDSize();
            int m_batch_from_d = d_input.numel() / out_size;

            // Validate the x_input dimensions.
            if (in_size != expected_in_size) {
              std::string shape_str = x_trans ? ("[*, " + std::to_string(expected_in_size) + "]")
                                              : ("[" + std::to_string(expected_in_size) + ", *]");
              throw std::runtime_error(
                  "Invalid x_input dimensions: expected " + shape_str + " array");
            }
            // Validate the d_input dimensions.
            if (out_size != expected_out_size) {
              std::string shape_str = d_trans ? ("[*, " + std::to_string(expected_out_size) + "]")
                                              : ("[" + std::to_string(expected_out_size) + ", *]");
              throw std::runtime_error(
                  "Invalid d_input dimensions: expected " + shape_str + " array");
            }

            if (m_batch != m_batch_from_d) {
              throw std::runtime_error(
                  "Invalid x_input or d_input dimensions: batch dimensions mismatch!");
            }

            // Call RPU function.
            std::lock_guard<std::mutex> lock(self.mutex_);
            self.update(
                x_input.template data_ptr<T>(), d_input.template data_ptr<T>(), bias, m_batch,
                x_trans, d_trans);
          },
          py::arg("x_input"), py::arg("d_input"), py::arg("bias"), py::arg("d_trans") = false,
          py::arg("x_trans") = false,
          R"pbdoc(
           Compute an n-rank update.

           Compute an n-rank update:
           .. math:

               W \leftarrow W - \lambda \mathbf{x}\mathbf{d}^T

           where :math:`\lambda` is the learning rate.

           An analog tile will have a possible non-ideal version of this update pass.

           Note:
               The learning rate is always positive, and thus scaling is negative.

           Args:
               x_input: ``[N, *, x_size (-1)]`` input :math:`\mathbf{x}` torch::Tensor.
               d_input: ``[N, *, d_size]`` input :math:`\mathbf{d}` torch::Tensor.
               bias: whether to use bias.
               x_trans: whether the ``x_input`` matrix is transposed, ie. ``[x_size (-1), *, N]``
               d_trans: whether the ``d`` matrix is transposed, ie. ``[d_size, *, N]``
           )pbdoc")
      .def(
          "forward_indexed",
          [](Class &self, const torch::Tensor &x_input_, const torch::Tensor &d_tensor_,
             bool is_test = false) {
            auto x_input = x_input_.contiguous();
            auto d_tensor = d_tensor_.contiguous();
            CHECK_TORCH_INPUT(x_input);
            CHECK_TORCH_INPUT(d_tensor);

            int N = x_input.size(0); // batch
            int d_image_size = ((d_tensor.numel() / d_tensor.size(0)) / d_tensor.size(1));

            // Call RPU function.
            std::lock_guard<std::mutex> lock(self.mutex_);
            self.forwardIndexed(
                x_input.template data_ptr<T>(), d_tensor.template data_ptr<T>(), x_input.numel(),
                d_image_size, N, true, is_test);
            return d_tensor;
          },
          py::arg("x_input"), py::arg("d_tensor"), py::arg("is_test") = false,
          R"pbdoc(
           Compute the dot product using an index matrix (forward pass).

           Caution:
               Internal use for convolutions only.

           Args:
               x_input: 4D or 5D torch::tensor in order N,C,(D),H,W
               d_tensor: torch:tensor with convolution dimensions
               is_test: whether inference (true) mode or training (false)

           Returns:
               d_output: 4D 5D torch::tensor in order N, C, (d_depth,) d_height, d_width
           )pbdoc")
      .def(
          "backward_indexed",
          [](Class &self, const torch::Tensor &d_input_, const torch::Tensor &x_tensor_) {
            auto d_input = d_input_.contiguous();
            auto x_tensor = x_tensor_.contiguous();
            CHECK_TORCH_INPUT(d_input);
            CHECK_TORCH_INPUT(x_tensor);

            int N = d_input.size(0); // batch
            int d_image_size = ((d_input.numel() / d_input.size(0)) / d_input.size(1));

            // Call RPU function.
            std::lock_guard<std::mutex> lock(self.mutex_);
            self.backwardIndexed(
                d_input.template data_ptr<T>(), x_tensor.template data_ptr<T>(), x_tensor.numel(),
                d_image_size, N, true);
            return x_tensor;
          },
          py::arg("d_input"), py::arg("x_tensor"),
          R"pbdoc(
           Compute the dot product using an index matrix (backward pass).

           Caution:
               Internal use for convolutions only.

           Args:
               d_input: 4D torch::tensor in order N,C,H,W
               x_tensor: torch:tensor with convolution dimensions

           Returns:
               x_output: 4D (5D) torch::tensor in order N,C, (x_depth,) x_height, x_width
           )pbdoc")
      .def(
          "update_indexed",
          [](Class &self, const torch::Tensor &x_input_, const torch::Tensor &d_input_) {
            auto x_input = x_input_.contiguous();
            auto d_input = d_input_.contiguous();
            CHECK_TORCH_INPUT(x_input);
            CHECK_TORCH_INPUT(d_input);

            int N = d_input.size(0); // batch
            int d_image_size = d_input.numel() / (d_input.size(0) * d_input.size(1));

            // Call RPU function.
            std::lock_guard<std::mutex> lock(self.mutex_);
            self.updateIndexed(
                x_input.template data_ptr<T>(), d_input.template data_ptr<T>(), x_input.numel(),
                d_image_size, N, true);
          },
          py::arg("x_input"), py::arg("d_input"),
          R"pbdoc(
           Compute the dot product using an index matrix (backward pass).

           Caution:
               Internal use for convolutions only.

           Args:
               x_input: 4D torch::tensor input in order N,C,H,W
               d_input: 4D torch::tensor (grad_output) in order N,C,oH,oW
           )pbdoc")
      .def(
          "set_matrix_indices",
          [](Class &self, const torch::Tensor &indices) {
            CHECK_CONTIGUOUS(indices);
            std::lock_guard<std::mutex> lock(self.mutex_);
            self.setMatrixIndices(indices.data_ptr<int>());
          },
          py::arg("indices"),
          R"pbdoc(
           Sets the index vector for the ``*_indexed`` functionality.

           Caution:
               Internal use only.

           Args:
               indices: int torch::Tensor
          )pbdoc")
      .def(
          "get_hidden_parameter_names",
          [](Class &self) {
            std::vector<std::string> v;
            self.getDeviceParameterNames(v);
            return v;
          },
          R"pbdoc(
           Get the hidden parameters of the tile.

           Returns:
               list: list of hidden parameter names.
          )pbdoc")
      .def(
          "get_hidden_parameters",
          [](Class &self) {
            std::vector<std::string> v;
            self.getDeviceParameterNames(v);

            // TODO choose correct tensor options (CPU, float32) probably standard though
            if (!v.size()) {
              return torch::empty({0});
            }
            torch::Tensor hidden_parameters =
                torch::empty({(int)v.size(), self.getDSize(), self.getXSize()});

            std::vector<T *> data_ptrs(v.size());
            size_t size = self.getDSize() * self.getXSize();
            for (size_t i = 0; i < v.size(); i++) {
              data_ptrs[i] = hidden_parameters.data_ptr<T>() + i * size;
            }
            self.getDeviceParameter(data_ptrs);

            return hidden_parameters;
          },
          R"pbdoc(
           Get the hidden parameters of the tile.

           Returns:
               3D tensor: Each 2D slice tensor is of size [d_size, x_size] (in row-major order)
                   corresponding to the parameter name.
          )pbdoc")
      .def(
          "set_hidden_parameters",
          [](Class &self, const torch::Tensor &hidden_parameters_) {
            auto hidden_parameters = hidden_parameters_.contiguous();
            CHECK_TORCH_INPUT(hidden_parameters);

            std::vector<std::string> v;
            self.getDeviceParameterNames(v);

            if (!v.size()) {
              return;
            }

            if (hidden_parameters.dim() != 3 || (size_t)hidden_parameters.size(0) != v.size() ||
                (size_t)hidden_parameters.size(1) != (size_t)self.getDSize() ||
                (size_t)hidden_parameters.size(2) != (size_t)self.getXSize()) {
              throw std::runtime_error("Hidden parameter shape mismatch!");
            }

            std::vector<T *> data_ptrs(v.size());
            size_t size = self.getDSize() * self.getXSize();
            for (size_t i = 0; i < v.size(); i++) {
              data_ptrs[i] = hidden_parameters.data_ptr<T>() + i * size;
            }
            std::lock_guard<std::mutex> lock(self.mutex_);
            self.setDeviceParameter(data_ptrs);
          },
          R"pbdoc(
           Sets the hidden parameters of the tile.

           Args:
               3D tensor: Each 2D slice tensor is of size [d_size, x_size] (in row-major order)
                   corresponding to the parameter name.
          )pbdoc")
      .def(
          "set_hidden_update_index", &Class::setHiddenUpdateIdx,
          R"pbdoc(
           Set the updated device index (in case multiple devices per cross-point).

           Note:
              Only used for vector unit cells, so far. Ignored in other cases.

           Args:
               idx: index of the (unit cell) devices
           )pbdoc")
      .def(
          "get_hidden_update_index", &Class::getHiddenUpdateIdx,
          R"pbdoc(
           Get the current device index that is updated (in case multiple devices per cross-point).

           Args:
               idx: index of the (unit cell) devices, returns 0 in all other cases.
           )pbdoc");

  py::class_<ClassPulsed, Class>(
      m, "AnalogTile",
      R"pbdoc(
    Analog tile.

    Args:
        x_size: ``X`` size of the tile.
        d_size: ``D`` size of the tile.
    )pbdoc")
      .def(py::init<int, int>(), py::arg("x_size"), py::arg("d_size"))
      .def("get_parameters", &ClassPulsed::getMetaPar);
}
