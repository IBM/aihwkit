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
            // TODO: why does directly passing a stream gets
            return std::unique_ptr<Class>(new Class(at::cuda::getCurrentCUDAStream(), rpu));
          }),
          py::arg("cpu_tile"))
      .def(
          "forward",
          [](Class &self, const torch::Tensor &x_input_, bool bias = false, bool x_trans = false,
             bool d_trans = false, bool is_test = false) {
            auto x_input = x_input_.contiguous();
            CHECK_TORCH_CUDA_INPUT(x_input);

            if (x_input.dim() != 2) {
              throw std::runtime_error("Invalid x_input dimensions: expected 2 dimensional array");
            }

            int in_size = x_trans ? x_input.size(0) : x_input.size(1);
            int expected_in_size = self.getXSize() - (bias ? 1 : 0);
            int m_batch = x_trans ? x_input.size(1) : x_input.size(0);
            int out_size = self.getDSize();

            // Validate the x_input dimensions.
            if (in_size != expected_in_size) {
              std::string shape_str = x_trans ? ("[_, " + std::to_string(expected_in_size) + "]")
                                              : ("[" + std::to_string(expected_in_size) + ",_]");
              throw std::runtime_error(
                  "Invalid x_input dimensions: expected " + shape_str + " array");
            }

            self.finishUpdateCalculations();
            std::lock_guard<std::mutex> lock(self.mutex_);
            self.setStream(at::cuda::getCurrentCUDAStream()); // TODO: better way to get the stream?

            // Build the buffers.
            int dim0 = d_trans ? out_size : m_batch;
            int dim1 = d_trans ? m_batch : out_size;
            torch::Tensor d_output = torch::empty({dim0, dim1}, x_input.options());

            // Call RPU function.
            self.forward(
                x_input.template data_ptr<T>(), d_output.template data_ptr<T>(), bias, m_batch,
                x_trans, d_trans, is_test);

            return d_output;
          },
          py::arg("x_input"), py::arg("bias") = false, py::arg("x_trans") = false,
          py::arg("d_trans") = false, py::arg("is_test") = false,
          R"pbdoc(
           Compute the dot product (forward pass).

           Compute the dot product::

               Y = X * W (+ b)

           where ``X`` is the input and ``W`` is the ``[d_size, x_size]``
           current weight matrix. If ``bias`` is True, then it is assumes that a
           bias row is added to the analog tile weights.  The input ``X`` is
           then  expected to be of size ``x_size -1`` , as internally it will be
           expanded by a 1, to match the bias row in the tile weights.

           An analog tile will have a possible non-ideal version of this forward pass.

           Args:
               x_input: ``[N, x_size (- 1)]`` torch::tensor.

           Returns:
               torch::tensor: ``[N, d_size]`` torch::tensor.
           )pbdoc")

      .def(
          "backward",
          [](Class &self, const torch::Tensor &d_input_, bool bias = false, bool d_trans = false,
             bool x_trans = false) {
            auto d_input = d_input_.contiguous();
            CHECK_TORCH_CUDA_INPUT(d_input);

            if (d_input.dim() != 2) {
              throw std::runtime_error("Invalid d_input dimensions: expected 2 dimensional array");
            }

            int in_size = d_trans ? d_input.size(0) : d_input.size(1);
            int expected_in_size = self.getDSize();
            int m_batch = d_trans ? d_input.size(1) : d_input.size(0);
            int out_size = self.getXSize() - (bias ? 1 : 0);

            // Validate the d_input dimensions.
            if (in_size != expected_in_size) {
              std::string shape_str = d_trans ? ("[_, " + std::to_string(expected_in_size) + "]")
                                              : ("[" + std::to_string(expected_in_size) + ",_]");
              throw std::runtime_error(
                  "Invalid d_input dimensions: expected " + shape_str + " array");
            }

            self.finishUpdateCalculations();
            std::lock_guard<std::mutex> lock(self.mutex_);
            self.setStream(at::cuda::getCurrentCUDAStream());

            // Build the buffers.
            int dim0 = x_trans ? out_size : m_batch;
            int dim1 = x_trans ? m_batch : out_size;
            torch::Tensor x_output = torch::empty({dim0, dim1}, d_input.options());

            // Call RPU function.
            self.backward(
                d_input.template data_ptr<T>(), x_output.template data_ptr<T>(), bias, m_batch,
                d_trans, x_trans);
            return x_output;
          },
          py::arg("d_input"), py::arg("bias") = false, py::arg("d_trans") = false,
          py::arg("x_trans") = false,
          R"pbdoc(
           Compute the transposed dot product (backward pass).

           Compute the transposed dot product::

               Y = D * W'

           where ``D`` is the input and ``W'`` is the ``[d_size, x_size]`` transposed current
           weight matrix.

           An analog tile will have a possible non-ideal version of this backward pass.

           Args:
               d_input: ``[N, d_size]`` torch::tensor.

           Returns:
               torch::tensor: ``[N, x_size (-1)]`` torch::tensor.
           )pbdoc")

      .def(
          "update",
          [](Class &self, const torch::Tensor &x_input_, const torch::Tensor &d_input_,
             bool bias = false, bool x_trans = false, bool d_trans = false) {
            auto d_input = d_input_.contiguous();
            auto x_input = x_input_.contiguous();

            CHECK_TORCH_CUDA_INPUT(d_input);
            CHECK_TORCH_CUDA_INPUT(x_input);

            if ((x_input.dim() != 2) || (d_input.dim() != 2)) {
              throw std::runtime_error(
                  "Invalid x_input/d_input dimensions: expected 2 dimensional array");
            }

            int in_size = x_trans ? x_input.size(0) : x_input.size(1);
            int expected_in_size = self.getXSize() - (bias ? 1 : 0);
            int m_batch = x_trans ? x_input.size(1) : x_input.size(0);

            int out_size = d_trans ? d_input.size(0) : d_input.size(1);
            int expected_out_size = self.getDSize();
            int m_batch_from_d = d_trans ? d_input.size(1) : d_input.size(0);

            // Validate the x_input dimensions.
            if (in_size != expected_in_size) {
              std::string shape_str = x_trans ? ("[_, " + std::to_string(expected_in_size) + "]")
                                              : ("[" + std::to_string(expected_in_size) + ",_]");
              throw std::runtime_error(
                  "Invalid x_input dimensions: expected " + shape_str + " array");
            }
            // Validate the d_input dimensions.
            if (out_size != expected_out_size) {
              std::string shape_str = d_trans ? ("[_, " + std::to_string(expected_out_size) + "]")
                                              : ("[" + std::to_string(expected_out_size) + ",_]");
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
            self.setStream(at::cuda::getCurrentCUDAStream());

            self.update(
                x_input.template data_ptr<T>(), d_input.template data_ptr<T>(), bias, m_batch,
                x_trans, d_trans);
          },
          py::arg("x_input"), py::arg("d_input"), py::arg("bias"), py::arg("d_trans") = false,
          py::arg("x_trans") = false,
          R"pbdoc(
           Compute an n-rank update.

           Compute an n-rank update::

               W += -LR * D * X'

           where ``LR`` is the learning rate.

           An analog tile will have a possible non-ideal version of this update pass.

           Note:
               The learning rate is always positive, and thus scaling is negative.

           Args:
               x_input: ``[N, x_size (-1)]`` torch::tensor.
               d_input: ``[N, d_size]`` torch::tensor.
           )pbdoc")
      .def(
          "forward_indexed",
          [](Class &self, const torch::Tensor &x_input_, int d_height, int d_width,
             bool is_test = false) {
            auto x_input = x_input_.contiguous();
            CHECK_TORCH_CUDA_INPUT(x_input);

            int N = x_input.size(0); // batch
            int C = self.getDSize(); // out_channel
            int d_image_size = d_width * d_height;

            torch::Tensor d_output = torch::empty({N, C, d_height, d_width}, x_input.options());

            // Call RPU function.
            self.finishUpdateCalculations();
            std::lock_guard<std::mutex> lock(self.mutex_);
            self.setStream(at::cuda::getCurrentCUDAStream());
            self.forwardIndexed(
                x_input.template data_ptr<T>(), d_output.template data_ptr<T>(), x_input.numel(),
                d_image_size, N, true, is_test);
            return d_output;
          },
          py::arg("x_input"), py::arg("d_height"), py::arg("d_width"), py::arg("is_test") = false,
          R"pbdoc(
           Compute the dot product using an index matrix (forward pass).

           Caution:
              Internal use for convolutions only.

           Args:
              x_input: 4D torch::tensor in order N,C,H,W
              d_height: height of output image(s)
              d_width: width of output image(s)
              is_test: whether inference (true) mode or training (false)

           Returns:
              d_output: 4D torch::tensor in order N,C,d_height,d_width
           )pbdoc")
      .def(
          "backward_indexed",
          [](Class &self, const torch::Tensor &d_input_, int x_channel, int x_height, int x_width) {
            auto d_input = d_input_.contiguous();
            CHECK_TORCH_CUDA_INPUT(d_input);

            int N = d_input.size(0); // batch
            int d_image_size = d_input.size(2) * d_input.size(3);
            torch::Tensor x_output =
                torch::empty({N, x_channel, x_height, x_width}, d_input.options());

            // Call RPU function.
            self.finishUpdateCalculations();
            std::lock_guard<std::mutex> lock(self.mutex_);
            self.setStream(at::cuda::getCurrentCUDAStream());
            self.backwardIndexed(
                d_input.template data_ptr<T>(), x_output.template data_ptr<T>(), x_output.numel(),
                d_image_size, N, true);
            return x_output;
          },
          py::arg("d_input"), py::arg("x_channel"), py::arg("x_height"), py::arg("x_width"),
          R"pbdoc(
           Compute the dot product using an index matrix (backward pass).

           Caution:
              Internal use for convolutions only.

           Args:
              d_input: 4D torch::tensor in order N,C,H,W
              x_channel: number of grad_input channels
              x_height: height of grad_input image(s)
              x_width: width of grad_input image(s)

           Returns:
              x_output: 4D torch::tensor in order N,C,x_height,x_width
           )pbdoc")
      .def(
          "update_indexed",
          [](Class &self, const torch::Tensor &x_input_, const torch::Tensor &d_input_) {
            auto x_input = x_input_.contiguous();
            auto d_input = d_input_.contiguous();
            CHECK_TORCH_CUDA_INPUT(x_input);
            CHECK_TORCH_CUDA_INPUT(d_input);

            int N = d_input.size(0); // batch
            int d_image_size = d_input.size(2) * d_input.size(3);

            // Call RPU function.
            self.finishUpdateCalculations();
            std::lock_guard<std::mutex> lock(self.mutex_);
            self.setStream(at::cuda::getCurrentCUDAStream());
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
