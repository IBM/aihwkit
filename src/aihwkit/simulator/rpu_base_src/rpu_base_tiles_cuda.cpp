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

#ifdef RPU_USE_CUDA
#include "cuda.h"
#include "rpu_base.h"

#define CHECK_CUDA(x)                                                                              \
  TORCH_CHECK(                                                                                     \
      x.device().type() == torch::kCUDA, #x " must be a CUDA tensor. got ", x.device().type(),     \
      " versus ", torch::kCUDA)
#define CHECK_CUDA_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_TORCH_CUDA_INPUT(x)                                                                  \
  CHECK_CUDA(x);                                                                                   \
  CHECK_CUDA_CONTIGUOUS(x)

void declare_rpu_tiles_cuda(py::module &m) {
  using Class = RPU::RPUCudaSimple<T>;
  using ClassPulsed = RPU::RPUCudaPulsed<T>;

  /*
   * Helper bindings.
   */
  py::class_<cudaStream_t>(m, "cudaStream_t");

  /*
   * RPU definitions.
   */

  py::class_<Class, RPU::RPUSimple<T>>(
      m, "CudaFloatingPointTile",
      R"pbdoc(
    Floating point tile (CUDA).

    Args:
        tile: existing ``FloatingPointTile`` that will be copied.
    )pbdoc")
      .def(
          py::init([](RPU::RPUSimple<T> &rpu) {
            return std::unique_ptr<Class>(new Class(at::cuda::getCurrentCUDAStream(), rpu));
          }),
          py::arg("cpu_tile"))
      .def(
          "set_shared_weights",
          [](Class &self, torch::Tensor weights) {
            CHECK_TORCH_CUDA_INPUT(weights);
            // weight is d major for CUDA !
            if (weights.dim() != 2 || weights.size(1) != self.getDSize() ||
                weights.size(0) != self.getXSize()) {
              throw std::runtime_error(
                  "Invalid weights dimensions: expected [" + std::to_string(self.getXSize()) + "," +
                  std::to_string(self.getDSize()) + "] array");
            }
            if (weights.device().index() != self.getGPUId()) {
              throw std::runtime_error(
                  "Weights need to be on the same cuda device: expected " +
                  std::to_string(self.getGPUId()) + ", got " +
                  std::to_string(weights.device().index()));
            }

            std::lock_guard<std::mutex> lock(self.mutex_);
            return self.setSharedWeights(weights.data_ptr<T>());
          },
          py::arg("weights"))
      .def(
          "set_delta_weights",
          [](Class &self, torch::Tensor delta_weights) {
            CHECK_TORCH_CUDA_INPUT(delta_weights);
            // weight is d major for CUDA !
            if (delta_weights.dim() != 2 || delta_weights.size(1) != self.getDSize() ||
                delta_weights.size(0) != self.getXSize()) {
              throw std::runtime_error(
                  "Invalid delta weights dimensions: expected [" + std::to_string(self.getXSize()) +
                  "," + std::to_string(self.getDSize()) + "] array");
            }
            if (delta_weights.device().index() != self.getGPUId()) {
              throw std::runtime_error(
                  "Delta weights need to be on the same cuda device: expected " +
                  std::to_string(self.getGPUId()) + ", got " +
                  std::to_string(delta_weights.device().index()));
            }

            std::lock_guard<std::mutex> lock(self.mutex_);
            return self.setDeltaWeights(delta_weights.data_ptr<T>());
          },
          py::arg("delta_weights"))
      .def(
          "remap_weights",
          [](Class &self, ::RPU::WeightRemapParameter &wrmpar, torch::Tensor scales) {
            CHECK_TORCH_CUDA_INPUT(scales);
            if ((scales.numel() != self.getDSize()) || scales.dim() != 1) {
              throw std::runtime_error(
                  "Invalid scales dimensions: expected [" + std::to_string(self.getDSize()) +
                  "] tensor");
            }
            std::lock_guard<std::mutex> lock(self.mutex_);
            self.remapWeights(wrmpar, scales.data_ptr<T>());
            return scales;
          },
          py::arg("weight_remap_params"), py::arg("scales"),
          R"pbdoc(
           Remaps the weights for use of hardware-aware training.

           Several remap types are available, see ``WeightRemapParameter``.

           Args:
               weight_remap_params: parameters of the remapping.
               scales: scales that will be used and updated during remapping

           Returns:
               torch::tensor: ``[d_size]`` of scales

           )pbdoc")

      .def(
          "forward",
          [](Class &self, const torch::Tensor &x_input_, bool bias = false, bool x_trans = false,
             bool d_trans = false, bool is_test = false, bool non_blocking = false) {
            auto x_input = x_input_.contiguous();
            CHECK_TORCH_CUDA_INPUT(x_input);

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
            self.finishUpdateCalculations();
            std::lock_guard<std::mutex> lock(self.mutex_);
            self.setExternalStream(at::cuda::getCurrentCUDAStream());
            self.forward(
                x_input.template data_ptr<T>(), d_output.template data_ptr<T>(), bias, m_batch,
                x_trans, d_trans, is_test);

            if (!non_blocking) {
              self.finishAllCalculations();
            }
            self.releaseExternalStream();
            return d_output;
          },
          py::arg("x_input"), py::arg("bias") = false, py::arg("x_trans") = false,
          py::arg("d_trans") = false, py::arg("is_test") = false, py::arg("non_blocking") = false,
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
               non_blocking: whether to not sync the cuda execution

           Returns:
               torch::tensor: ``[N, *, d_size]`` or ``[d_size, *, N]`` matrix.
           )pbdoc")

      .def(
          "backward",
          [](Class &self, const torch::Tensor &d_input_, bool bias = false, bool d_trans = false,
             bool x_trans = false, bool non_blocking = false) {
            auto d_input = d_input_.contiguous();
            CHECK_TORCH_CUDA_INPUT(d_input);

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
            self.finishUpdateCalculations();
            std::lock_guard<std::mutex> lock(self.mutex_);
            self.setExternalStream(at::cuda::getCurrentCUDAStream());
            self.backward(
                d_input.template data_ptr<T>(), x_output.template data_ptr<T>(), bias, m_batch,
                d_trans, x_trans);

            if (!non_blocking) {
              self.finishAllCalculations();
            }
            self.releaseExternalStream();
            return x_output;
          },
          py::arg("d_input"), py::arg("bias") = false, py::arg("d_trans") = false,
          py::arg("x_trans") = false, py::arg("non_blocking") = false,
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
               non_blocking: whether to not sync the cuda execution

           Returns:
               torch::Tensor: ``[N, *, x_size (-1)]`` or ``[x_size (-1), *, N]`` torch::Tensor.
           )pbdoc")

      .def(
          "update",
          [](Class &self, const torch::Tensor &x_input_, const torch::Tensor &d_input_,
             bool bias = false, bool x_trans = false, bool d_trans = false,
             bool non_blocking = false) {
            auto d_input = d_input_.contiguous();
            auto x_input = x_input_.contiguous();

            CHECK_TORCH_CUDA_INPUT(d_input);
            CHECK_TORCH_CUDA_INPUT(x_input);

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
            self.finishUpdateCalculations();
            std::lock_guard<std::mutex> lock(self.mutex_);
            self.setExternalStream(at::cuda::getCurrentCUDAStream());
            self.update(
                x_input.template data_ptr<T>(), d_input.template data_ptr<T>(), bias, m_batch,
                x_trans, d_trans);

            if (!non_blocking) {
              self.finishAllCalculations();
            }
            self.releaseExternalStream();
          },
          py::arg("x_input"), py::arg("d_input"), py::arg("bias"), py::arg("d_trans") = false,
          py::arg("x_trans") = false, py::arg("non_blocking") = false,
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
               non_blocking: Whether to not sync the cuda execution
           )pbdoc")
      .def(
          "forward_indexed",
          [](Class &self, const torch::Tensor &x_input_, const torch::Tensor &d_tensor_,
             bool is_test = false, bool non_blocking = false) {
            auto x_input = x_input_.contiguous();
            auto d_tensor = d_tensor_.contiguous();
            CHECK_TORCH_CUDA_INPUT(x_input);
            CHECK_TORCH_CUDA_INPUT(d_tensor);

            int N = x_input.size(0); // batch
            int d_image_size = ((d_tensor.numel() / d_tensor.size(0)) / d_tensor.size(1));

            // Call RPU function.
            self.finishUpdateCalculations();
            std::lock_guard<std::mutex> lock(self.mutex_);
            self.setExternalStream(at::cuda::getCurrentCUDAStream());
            self.forwardIndexed(
                x_input.template data_ptr<T>(), d_tensor.template data_ptr<T>(), x_input.numel(),
                d_image_size, N, true, is_test);

            if (!non_blocking) {
              self.finishAllCalculations();
            }
            self.releaseExternalStream();
            return d_tensor;
          },
          py::arg("x_input"), py::arg("d_tensor"), py::arg("is_test") = false,
          py::arg("non_blocking") = false,
          R"pbdoc(
           Compute the dot product using an index matrix (forward pass).

           Caution:
               Internal use for convolutions only.

           Args:
               x_input: 4D torch::tensor in order N,C,H,W
               d_tensor: torch:tensor with convolution dimensions
               is_test: whether inference (true) mode or training (false)
               non_blocking: Whether to not sync the cuda execution

           Returns:
               d_output: 4D torch::tensor in order N,C,d_height,d_width
           )pbdoc")
      .def(
          "backward_indexed",
          [](Class &self, const torch::Tensor &d_input_, const torch::Tensor &x_tensor_,
             bool non_blocking = false) {
            auto d_input = d_input_.contiguous();
            auto x_tensor = x_tensor_.contiguous();
            CHECK_TORCH_CUDA_INPUT(d_input);
            CHECK_TORCH_CUDA_INPUT(x_tensor);

            int N = d_input.size(0); // batch
            int d_image_size = ((d_input.numel() / d_input.size(0)) / d_input.size(1));

            // Call RPU function.
            self.finishUpdateCalculations();
            std::lock_guard<std::mutex> lock(self.mutex_);
            self.setExternalStream(at::cuda::getCurrentCUDAStream());
            self.backwardIndexed(
                d_input.template data_ptr<T>(), x_tensor.template data_ptr<T>(), x_tensor.numel(),
                d_image_size, N, true);

            if (!non_blocking) {
              self.finishAllCalculations();
            }
            self.releaseExternalStream();
            return x_tensor;
          },
          py::arg("d_input"), py::arg("x_tensor"), py::arg("non_blocking") = false,
          R"pbdoc(
           Compute the dot product using an index matrix (backward pass).

           Caution:
              Internal use for convolutions only.

           Args:
              d_input: 4D torch::tensor in order N,C,H,W
              x_tensor: torch:tensor with convolution dimensions
              non_blocking: Whether to not sync the cuda execution

           Returns:
              x_output: 4D torch::tensor in order N,C,x_height,x_width
           )pbdoc")
      .def(
          "update_indexed",
          [](Class &self, const torch::Tensor &x_input_, const torch::Tensor &d_input_,
             bool non_blocking = false) {
            auto x_input = x_input_.contiguous();
            auto d_input = d_input_.contiguous();
            CHECK_TORCH_CUDA_INPUT(x_input);
            CHECK_TORCH_CUDA_INPUT(d_input);

            int N = d_input.size(0); // batch
            int d_image_size = d_input.numel() / (d_input.size(0) * d_input.size(1));

            // Call RPU function.
            self.finishUpdateCalculations();
            std::lock_guard<std::mutex> lock(self.mutex_);
            self.setExternalStream(at::cuda::getCurrentCUDAStream());
            self.updateIndexed(
                x_input.template data_ptr<T>(), d_input.template data_ptr<T>(), x_input.numel(),
                d_image_size, N, true);

            if (!non_blocking) {
              self.finishAllCalculations();
            }
            self.releaseExternalStream();
          },
          py::arg("x_input"), py::arg("d_input"), py::arg("non_blocking") = false,
          R"pbdoc(
           Compute the dot product using an index matrix (backward pass).

           Caution:
               Internal use for convolutions only.

           Args:
               x_input: 4D torch::tensor input in order N,C,H,W
               d_input: 4D torch::tensor (grad_output) in order N,C,oH,oW
               non_blocking: Whether to not sync the cuda execution

           )pbdoc");

  py::class_<ClassPulsed, RPU::RPUCudaSimple<T>>(
      m, "CudaAnalogTile",
      R"pbdoc(
    Analog tile (CUDA).

    Args:
        tile: existing ``AnalogTile`` that will be copied.
    )pbdoc")
      .def(
          py::init([](RPU::RPUPulsed<T> &rpu) {
            // TODO: why does directly passing a stream is a problem?
            return std::unique_ptr<ClassPulsed>(
                new ClassPulsed(at::cuda::getCurrentCUDAStream(), rpu));
          }),
          py::arg("tile"))
      .def("get_parameters", &ClassPulsed::getMetaPar);
}

#endif
