# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Tests for torch-based inference tiles."""

import os
from typing import Union
from pytest import mark
from torch import (
    float32,
    float16,
    bfloat16,
    int64,
    autocast,
    flatten,
    cuda,
    Tensor,
    allclose,
    randn,
    tensor,
    manual_seed,
    device,
    clip,
    ones,
    zeros_like,
    clamp,
    cat,
)
from torch import dtype as torch_dtype
from torch.nn.functional import log_softmax
from torch.nn import Linear, Conv2d, Module, NLLLoss, MaxPool2d

from aihwkit.nn.conversion import convert_to_analog
from aihwkit.nn import AnalogLinear, AnalogSequential
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import (
    TorchInferenceRPUConfig,
    InferenceRPUConfig,
    WeightRemapType,
    WeightClipType,
    NoiseManagementType,
    BoundManagementType,
)
from .helpers.decorators import parametrize_over_tiles
from .helpers.testcases import ParametrizedTestCase, SkipTest
from .helpers.tiles import (
    TorchInference,
    TorchInferenceCuda,
    TorchInferenceIRDropT,
    TorchInferenceIRDropTCuda,
)

SKIP_CUDA_TESTS = os.getenv("SKIP_CUDA_TESTS")


@parametrize_over_tiles(
    [TorchInference, TorchInferenceCuda, TorchInferenceIRDropT, TorchInferenceIRDropTCuda]
)
class TorchInferenceTest(ParametrizedTestCase):
    """Tests the basic functionality of FloatingPoint and Analog tiles."""

    def test_storing_and_loading(self):
        """
        Test loading and storing the torch tiles.
        """
        # - Create simple rpu_config
        n_in = 20
        rpu_config = self.get_rpu_config()
        rpu_config.forward.is_perfect = True
        linear = AnalogSequential(AnalogLinear(n_in, 10, bias=True, rpu_config=rpu_config))
        inp = randn((10, n_in))
        if self.use_cuda:
            linear = linear.cuda()
            inp = inp.cuda()

        out_pre = linear(inp)
        linear.load_state_dict(linear.state_dict())
        out_post = linear(inp)
        self.assertTensorAlmostEqual(out_pre, out_post)

        # - Create the same config but for inference tile
        rpu_config = InferenceRPUConfig()
        rpu_config.forward.is_perfect = True
        linear_ref = AnalogSequential(AnalogLinear(n_in, 10, bias=True, rpu_config=rpu_config))
        if self.use_cuda:
            linear_ref = linear_ref.cuda()

        linear_ref.load_state_dict(linear.state_dict(), load_rpu_config=True)
        self.assertTensorAlmostEqual(linear_ref(inp), out_post)

    def test_to_device(self):
        """
        Test moving the new torch based models to and from GPU.
        """
        # - Per default on CPU
        rpu_config = self.get_rpu_config()
        rpu_config.forward.is_perfect = True
        linear = AnalogSequential(AnalogLinear(10, 10, bias=True, rpu_config=rpu_config))
        for param in linear.parameters():
            self.assertEqual(param.device, device("cpu"))
        # - Move to GPU
        if self.use_cuda:
            linear = linear.cuda()
            for param in linear.parameters():
                self.assertTrue("cuda" in str(param.device))
        # - Move back to CPU
        linear = linear.cpu()
        for param in linear.parameters():
            self.assertEqual(param.device, device("cpu"))

    def test_set_and_get_weights(self):
        """
        Test integrity of weights after set and get.
        """
        rpu_config = self.get_rpu_config()
        linear = AnalogLinear(10, 10, bias=True, rpu_config=rpu_config)
        if self.use_cuda:
            linear = linear.cuda()
        weights = randn(10, 10)
        linear.set_weights(weights)
        weights_read, _ = linear.get_weights()
        self.assertTensorAlmostEqual(weights, weights_read)
        # - Compare to cpp-based tile
        linear_ref = AnalogLinear(10, 10, rpu_config=InferenceRPUConfig())
        linear_ref.set_weights(weights)
        if self.use_cuda:
            linear_ref = linear_ref.cuda()
        weights_read, _ = linear_ref.get_weights()
        self.assertTensorAlmostEqual(weights, weights_read)

    def test_grad_behavior(self):
        """
        Test correct grad and scale of grad compared to cpp-based tile under clipping.
        """

        def set_discretize(rpu_config: TorchInferenceRPUConfig):
            rpu_config.forward.is_perfect = False
            if rpu_config.forward.ir_drop > 0:
                raise SkipTest("Inp res < 0 not supported")
            rpu_config.forward.out_noise = 0.0
            rpu_config.forward.inp_bound = 1.0
            rpu_config.forward.out_bound = 1e6
            rpu_config.forward.inp_res = -1
            rpu_config.forward.out_res = -1
            rpu_config.forward.noise_management = NoiseManagementType.NONE
            rpu_config.forward.bound_management = BoundManagementType.NONE
            return rpu_config

        class ClippedLinear(Linear):
            """
            Simple clipping linear layer
            """

            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)

            def forward(self, input):  # pylint: disable=redefined-builtin
                x = clip(input, -1.0, 1.0)
                return super().forward(x)

        rpu_config = set_discretize(self.get_rpu_config())

        linear = AnalogLinear(10, 10, bias=False, rpu_config=rpu_config)
        linear_ref = ClippedLinear(10, 10, bias=False)
        linear_ref.weight.data = linear.get_weights()[0]
        inp = randn(50, 10)

        if self.use_cuda:
            linear = linear.cuda()
            linear_ref = linear_ref.cuda()
            inp = inp.cuda()

        out = linear(inp).mean()
        out_ref = linear_ref(inp).mean()
        out.backward()
        out_ref.backward()
        self.assertTensorAlmostEqual(linear_ref.weight.grad, linear.analog_module.tile.weight.grad)

    def test_input_range_grad(self):
        """
        Tests the gradient of the input ranges.
        """
        manual_seed(0)

        def set_discretize(rpu: Union[TorchInferenceRPUConfig, InferenceRPUConfig]):
            rpu.forward.out_noise = 0.0
            rpu.forward.inp_bound = 1.0
            rpu.forward.out_bound = 1e6
            rpu.forward.inp_res = 2**8 - 2
            rpu.forward.out_res = -1
            rpu.forward.noise_management = NoiseManagementType.NONE
            rpu.forward.bound_management = BoundManagementType.NONE
            rpu.pre_post.input_range.enable = True
            rpu.pre_post.input_range.manage_output_clipping = False
            rpu.pre_post.input_range.decay = 0.0
            rpu.pre_post.input_range.init_value = 2.0
            rpu.pre_post.input_range.init_from_data = False
            return rpu

        rpu_config = set_discretize(self.get_rpu_config())
        rpu_config_ref = set_discretize(InferenceRPUConfig())
        linear = AnalogLinear(in_features=256, out_features=256, bias=False, rpu_config=rpu_config)
        linear_ref = AnalogLinear(
            in_features=256, out_features=256, bias=False, rpu_config=rpu_config_ref
        )
        linear_ref.load_state_dict(linear.state_dict(), load_rpu_config=False)

        if self.use_cuda:
            linear = linear.cuda()
            linear_ref = linear_ref.cuda()

        for _ in range(10):
            inp = randn((3, 20, 256))

            if self.use_cuda:
                inp = inp.cuda()

            out = linear(inp).mean()
            out.backward()
            out_ref = linear_ref(inp).mean()
            out_ref.backward()
            self.assertTensorAlmostEqual(
                linear.analog_module.input_range.grad, linear_ref.analog_module.input_range.grad
            )
            linear.analog_module.input_range.grad = None
            linear_ref.analog_module.input_range.grad = None


@mark.parametrize("inp_bound", [0.1, 1.0, 10.0])
@mark.parametrize("out_bound", [0.1, 1.0, 10.0])
@mark.parametrize("inp_res", [2**2, 2**4, -1])
@mark.parametrize("out_res", [2**2, 2**4, -1])
@mark.parametrize("bm_type", [BoundManagementType.ITERATIVE, BoundManagementType.NONE])
def test_discretization_behavior(inp_bound, out_bound, inp_res, out_res, bm_type):
    """
    Test the discretization of the tile.

    Args:
        inp_bound (float): Input bound.
        out_bound (float): Output bound.
        inp_res (float): Input resolution.
        out_res (float): Output resolution.
        bm_type (BoundManagementType): Type of BoundManagement.
    """

    def set_discretize(rpu: Union[TorchInferenceRPUConfig, InferenceRPUConfig]):
        rpu.forward.out_noise = 0.0
        rpu.forward.inp_bound = inp_bound
        rpu.forward.out_bound = out_bound
        rpu.forward.inp_res = inp_res
        rpu.forward.out_res = out_res
        rpu.forward.noise_management = NoiseManagementType.NONE
        rpu.forward.bound_management = bm_type
        return rpu

    rpu_config = set_discretize(TorchInferenceRPUConfig())
    rpu_config_ref = set_discretize(InferenceRPUConfig())
    in_d = 1
    out_d = 1
    linear = AnalogLinear(in_features=in_d, out_features=out_d, bias=False, rpu_config=rpu_config)
    linear.analog_module.tile.weight.data = tensor(1.0).view((1, 1))
    linear_ref = AnalogLinear(
        in_features=in_d, out_features=out_d, bias=False, rpu_config=rpu_config_ref
    )
    linear_ref.load_state_dict(linear.state_dict())
    inp = 10.0 * ones((1, in_d))
    out = linear(inp)
    out_ref = linear_ref(inp)
    assert allclose(out, out_ref, atol=1e-5)


@mark.parametrize("is_training", [True, False])
def test_iterative_linear(is_training: bool):
    """
    Test that checks the gradient of the input range when
    we pass through different inputs without calling
    .backward(), i.e. until we compute the gradients.

    Args:
        is_training: Whether the model is in training mode.
    """

    def populate_rpu(rpu_config: InferenceRPUConfig):
        rpu_config.forward.bound_management = BoundManagementType.NONE
        rpu_config.forward.noise_management = NoiseManagementType.NONE
        rpu_config.forward.out_noise = 0.0
        rpu_config.pre_post.input_range.enable = True
        rpu_config.pre_post.input_range.init_value = 1.0
        rpu_config.forward.is_perfect = True
        rpu_config.pre_post.input_range.init_from_data = 0
        rpu_config.pre_post.input_range.decay = 0.0
        return rpu_config

    cuda_linear = AnalogLinear(256, 256, bias=False, rpu_config=populate_rpu(InferenceRPUConfig()))
    torch_linear = AnalogLinear(
        256, 256, bias=False, rpu_config=populate_rpu(TorchInferenceRPUConfig())
    )
    normal_linear = Linear(256, 256, bias=False)
    if is_training:
        cuda_linear = cuda_linear.train()
        torch_linear = torch_linear.train()
    else:
        cuda_linear = cuda_linear.eval()
        torch_linear = torch_linear.eval()
    torch_linear.set_weights(cuda_linear.get_weights()[0])
    normal_linear.weight.data = torch_linear.get_weights()[0]
    inp = randn((3, 256))
    cuda_out = cuda_linear(inp).mean()
    cuda_out.backward()
    torch_out = torch_linear(inp).mean()
    torch_out.backward()
    inp.requires_grad = True
    inp.retain_grad()
    normal_out = normal_linear(inp).mean()
    normal_out.backward()
    normal_grad = zeros_like(torch_linear.analog_module.input_range)
    normal_grad += clamp((inp >= 1.0) * inp.grad.clone(), min=None, max=0.0).sum()
    normal_grad -= clamp((inp <= -1.0) * inp.grad.clone(), min=0.0, max=None).sum()
    torch_ir_grad = torch_linear.analog_module.input_range.grad.clone()
    cuda_ir_grad = cuda_linear.analog_module.input_range.grad.clone()
    assert allclose(normal_grad, torch_ir_grad, atol=1e-5)
    assert allclose(normal_grad, cuda_ir_grad, atol=1e-5)
    # zero out the ir grads
    for param in torch_linear.parameters():
        param.grad = None
    for param in cuda_linear.parameters():
        param.grad = None
    iter_cuda_out = cat([cuda_linear(inp[i]) for i in range(3)]).mean()
    iter_torch_out = cat([torch_linear(inp[i]) for i in range(3)]).mean()
    assert allclose(iter_torch_out, iter_cuda_out, atol=1e-5)
    iter_cuda_out.backward()
    iter_torch_out.backward()
    assert allclose(normal_grad, cuda_linear.analog_module.input_range.grad, atol=1e-5)
    assert allclose(normal_grad, torch_linear.analog_module.input_range.grad, atol=1e-5)


@mark.parametrize(
    "remap_type", [WeightRemapType.CHANNELWISE_SYMMETRIC, WeightRemapType.LAYERWISE_SYMMETRIC]
)
@mark.parametrize("clip_type", [WeightClipType.FIXED_VALUE, WeightClipType.LAYER_GAUSSIAN])
@mark.parametrize("sigma", [0.1, 2.0, 10.0])
@mark.parametrize("fixed_value", [0.1, 0.5, 1.0])
def test_remapping_and_clipping(
    remap_type: WeightRemapType, sigma: float, clip_type: WeightClipType, fixed_value: float
):
    """
    Test remapping and clipping of weights.

    Args:
        remap_type: Type of remapping type.
        sigma: Number of std's for gaussian clipping.
        clip_type: Type of clipping.
        fixed_value: Fixed value for fixed-value-based clipping.
    """
    # - This allows us to see the clipping on the analog tile weights
    if clip_type == WeightClipType.FIXED_VALUE:
        remap_type = WeightRemapType.NONE
    rpu_config = TorchInferenceRPUConfig()
    rpu_config.forward.bound_management = BoundManagementType.NONE
    rpu_config.forward.noise_management = NoiseManagementType.NONE
    rpu_config.forward.out_noise = 0.0
    rpu_config.remap.type = remap_type
    rpu_config.clip.type = clip_type
    rpu_config.clip.sigma = sigma
    if rpu_config.clip.type == WeightClipType.FIXED_VALUE:
        rpu_config.clip.fixed_value = fixed_value
    else:
        # fixed value will be clipped
        rpu_config.clip.fixed_value = 0.0
    rpu_config.mapping.weight_scaling_omega = 1.0

    linear = AnalogLinear(256, 256, bias=False, rpu_config=rpu_config)
    weights = randn((256, 256))
    linear.set_weights(weights.clone())
    opt = AnalogSGD(linear.parameters(), lr=0.0)
    inp = ones((3, 256))
    out = linear(inp).mean()
    out.backward()
    opt.step()
    if rpu_config.clip.type == WeightClipType.LAYER_GAUSSIAN:
        if weights.std() * rpu_config.clip.sigma <= weights.abs().max():
            # Clipping will happpen
            assert allclose(
                (weights.std() * rpu_config.clip.sigma) / linear.get_weights()[0].abs().max(),
                tensor(1.0),
            )
    elif rpu_config.clip.type == WeightClipType.FIXED_VALUE:
        if rpu_config.clip.fixed_value <= rpu_config.mapping.weight_scaling_omega:
            # - Should be clipped to fixed value
            assert linear.analog_module.tile.weight.abs().max() == rpu_config.clip.fixed_value
    if remap_type == WeightRemapType.LAYERWISE_SYMMETRIC:
        assert linear.analog_module.tile.weight.abs().max() == 1.0
    elif remap_type == WeightRemapType.CHANNELWISE_SYMMETRIC:
        assert (linear.analog_module.tile.weight.abs().max(1)[0] == 1.0).all()


@mark.parametrize("inp_bound", [0.1, 1.0, 10.0])
@mark.parametrize("out_bound", [0.1, 1.0, 10.0])
@mark.parametrize("inp_res", [2**2, 2**4, -1])
@mark.parametrize("out_res", [2**2, 2**4, -1])
def test_discretization(inp_res: int, out_res: int, inp_bound: float, out_bound: float):
    """
    Test discretized forward (input and output).

    Args:
        inp_res: Input resolution.
        out_res: Output resolution.
        inp_bound: Input bound.
        out_bound: Output bound.
    """

    def set_discretize(rpu: Union[TorchInferenceRPUConfig, InferenceRPUConfig]):
        rpu.forward.out_noise = 0.0
        rpu.forward.inp_bound = inp_bound
        rpu.forward.out_bound = out_bound
        rpu.forward.inp_res = inp_res
        rpu.forward.out_res = out_res
        rpu.forward.noise_management = NoiseManagementType.NONE
        rpu.forward.bound_management = BoundManagementType.NONE
        return rpu

    rpu_config_torch = set_discretize(TorchInferenceRPUConfig())
    rpu_config = set_discretize(InferenceRPUConfig())
    linear_torch = AnalogLinear(
        in_features=256, out_features=256, bias=False, rpu_config=rpu_config_torch
    )
    linear = AnalogLinear(in_features=256, out_features=256, bias=False, rpu_config=rpu_config)
    linear.load_state_dict(linear_torch.state_dict())
    inp = randn((3, 256))
    out = linear_torch(inp)
    out_ref = linear(inp)
    assert allclose(out, out_ref, atol=1e-5)


@mark.parametrize(
    "nm_type",
    [
        NoiseManagementType.NONE,
        NoiseManagementType.ABS_MAX,
        NoiseManagementType.CONSTANT,
        NoiseManagementType.MAX,
    ],
)
@mark.parametrize("bm_type", [BoundManagementType.NONE, BoundManagementType.ITERATIVE])
@mark.parametrize("nm_thres", [0.0, 2.0])
def test_noise_and_bound_management(
    nm_type: NoiseManagementType, bm_type: BoundManagementType, nm_thres: float
):
    """
    Test noise and bound management

    Args:
        nm_type: noise management type
        bm_type: bound management type
        nm_thres: noise management threshold
    """

    def set_bm_nm(
        rpu: Union[TorchInferenceRPUConfig, InferenceRPUConfig]
    ) -> Union[TorchInferenceRPUConfig, InferenceRPUConfig]:
        """Set the rpu config."""
        rpu.forward.out_noise = 0.0
        rpu.forward.inp_bound = 1
        rpu.forward.out_bound = 5
        rpu.forward.inp_res = 127
        rpu.forward.out_res = 127
        rpu.forward.nm_thres = nm_thres
        rpu.forward.noise_management = nm_type
        rpu.forward.bound_management = bm_type
        return rpu

    rpu_config_torch = set_bm_nm(TorchInferenceRPUConfig())
    rpu_config = set_bm_nm(InferenceRPUConfig())
    linear_torch = AnalogLinear(
        in_features=256, out_features=256, bias=False, rpu_config=rpu_config_torch
    )
    linear = AnalogLinear(in_features=256, out_features=256, bias=False, rpu_config=rpu_config)
    linear.load_state_dict(linear_torch.state_dict())
    inp = randn((10, 256))
    out = linear_torch(inp)
    out_ref = linear(inp)
    assert allclose(out, out_ref, atol=1e-5)


@mark.parametrize("dtype", [float32, float16, bfloat16])
@mark.parametrize("device_type", [device("cpu"), device("cuda")])
def test_dtype(dtype: torch_dtype, device_type: device):
    """
    Test whether there is some implicit casting of the inputs.

    Args:
        dtype: Torch dtype
        device_type: Torch device

    Raises:
        SkipTest: If cuda not found or disabled
    """
    if SKIP_CUDA_TESTS or not cuda.is_available():
        raise SkipTest("CUDA not available")

    rpu_config_torch = TorchInferenceRPUConfig()
    rpu_config_torch.forward.is_perfect = True

    linear_torch = AnalogLinear(
        in_features=256, out_features=256, bias=False, rpu_config=rpu_config_torch
    )
    linear_torch = linear_torch.to(device=device_type, dtype=dtype)
    inp = randn((10, 256)).to(device=device_type, dtype=dtype)
    out: Tensor
    out = linear_torch(inp)  # pylint: disable=not-callable
    assert out.dtype == dtype


@mark.parametrize("dtype", [float32, float16, bfloat16])
@mark.parametrize("device_type", [device("cpu"), device("cuda")])
@mark.parametrize("use_autocast", [True, False])
def test_low_prec_training(dtype: torch_dtype, device_type: device, use_autocast: bool):
    """
    Test training with and without autocast.

    Args:
        dtype: Torch dtype
        device_type: Torch device
        use_autocast: Whether to use autocast

    Raises:
        SkipTest: If cuda not found or disabled
    """
    if SKIP_CUDA_TESTS or not cuda.is_available():
        raise SkipTest("CUDA not available")

    class Net(Module):
        """Test network."""

        def __init__(self):
            super().__init__()
            self.conv1 = Conv2d(1, 32, 3, 1)
            self.conv2 = Conv2d(32, 64, 3, 1)
            self.pool = MaxPool2d(2, 2)
            self.fc1 = Linear(9216, 128)
            self.fc2 = Linear(128, 10)

        def forward(self, x):
            """Test forward."""
            x = self.conv1(x)
            x = self.pool(self.conv2(x))
            x = flatten(x, 1)
            x = self.fc1(x)
            x = self.fc2(x)
            output = log_softmax(x, dim=1)
            return output

    def get_data_and_labels():
        """Get data and labels."""
        return randn(10, 1, 28, 28), tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 0], dtype=int64)

    model = Net()
    rpu_config = TorchInferenceRPUConfig()
    rpu_config.forward.is_perfect = True
    rpu_config.mapping.max_input_size = 0
    rpu_config.mapping.max_output_size = 0
    model = convert_to_analog(model, rpu_config)
    nll_loss = NLLLoss()

    model = model.to(device=device_type, dtype=dtype)
    optimizer = AnalogSGD(model.parameters(), lr=0.1)
    model = model.train()

    data, target = get_data_and_labels()
    data = data.to(device=device_type, dtype=dtype)
    target = target.to(device=device_type)

    # Modify: For CPU with unsupported dtype (e.g. torch.float32), skip autocast to avoid warning.
    if use_autocast and not (device_type.type == "cpu" and dtype == float32):
        # Runs the forward pass with autocasting.
        with autocast(device_type=str(device_type), dtype=dtype):
            output: Tensor
            output = model(data)
            loss: Tensor
            loss = nll_loss(output, target)
        loss.backward()
        optimizer.step()
    else:
        optimizer.zero_grad()
        output: Tensor
        output = model(data)
        loss: Tensor
        loss = nll_loss(output.float(), target)
        loss.backward()
        optimizer.step()
