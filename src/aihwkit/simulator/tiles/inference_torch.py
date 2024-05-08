# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""High level analog tiles (inference)."""

from typing import Optional, Any, Dict, Tuple, TYPE_CHECKING

from torch import Tensor, zeros_like, clamp
from torch.autograd import no_grad, Function

from aihwkit.simulator.tiles.inference import InferenceTileWithPeriphery
from aihwkit.simulator.tiles.torch_tile import TorchSimulatorTile
from aihwkit.simulator.tiles.module import TileModule
from aihwkit.simulator.tiles.base import SimulatorTileWrapper

from aihwkit.exceptions import TorchTileConfigError, AnalogBiasConfigError, ArgumentError
from aihwkit.simulator.parameters.enums import WeightRemapType, WeightClipType

if TYPE_CHECKING:
    from torch.nn import BackwardHook
    from aihwkit.simulator.configs import TorchInferenceRPUConfig
    from aihwkit.simulator.parameters import InputRangeParameter


class InputRangeForward(Function):
    """
    Enable custom input range gradient computation using torch's autograd.
    """

    # pylint: disable=abstract-method, redefined-builtin, arguments-differ

    @staticmethod
    def forward(
        ctx: Any, x_input: Tensor, input_range: Tensor, ir_params: "InputRangeParameter"
    ) -> Tensor:
        ctx.save_for_backward(x_input, input_range)
        ctx.ir_params = ir_params
        return x_input

    @staticmethod
    def backward(ctx: Any, d_output: Tensor) -> Tuple[Tensor, Tensor, None]:
        x_input, input_range = ctx.saved_tensors
        ir_grad = None

        if input_range is not None:
            ir_params = ctx.ir_params

            upper_thres = x_input >= input_range  # pylint: disable=invalid-unary-operand-type
            lower_thres = x_input <= -input_range  # pylint: disable=invalid-unary-operand-type
            ir_grad = zeros_like(input_range)
            ir_grad += clamp(upper_thres * d_output, min=None, max=0.0).sum()
            ir_grad -= clamp(lower_thres * d_output, min=0.0, max=None).sum()

            if ir_params.gradient_relative:
                ir_grad *= input_range
                ir_grad *= ir_params.gradient_scale

            if ir_params.manage_output_clipping:
                raise NotImplementedError
            if ir_params.decay > 0:
                # - We shrink the input range if less than X% of the inputs are clipping.
                # where X is 1-ir_params.input_min_percentage
                percentage = (x_input.abs() < input_range).float().mean()
                ir_grad += (
                    ir_params.decay * input_range * (percentage > ir_params.input_min_percentage)
                )

        return d_output, ir_grad, None


class TorchInferenceTile(TileModule, InferenceTileWithPeriphery, SimulatorTileWrapper):
    """InferenceTile using a torch-based simulator tile (and not a
    tile from RPUCuda).
    """

    supports_indexed: bool = False
    supports_ddp: bool = True

    def __init__(
        self,
        out_size: int,
        in_size: int,
        rpu_config: Optional["TorchInferenceRPUConfig"] = None,
        bias: bool = False,
        in_trans: bool = False,
        out_trans: bool = False,
    ):
        if in_trans or out_trans:
            raise TorchTileConfigError("in/out trans is not supported.")

        if not rpu_config:
            # Import dynamically to avoid import cycles.
            # pylint: disable=import-outside-toplevel
            from aihwkit.simulator.configs import TorchInferenceRPUConfig

            rpu_config = TorchInferenceRPUConfig()

        TileModule.__init__(self)
        SimulatorTileWrapper.__init__(
            self, out_size, in_size, rpu_config, bias, in_trans, out_trans, torch_update=True
        )
        InferenceTileWithPeriphery.__init__(self)

        if self.analog_bias:
            raise AnalogBiasConfigError("Analog bias is not supported for the torch tile")

        # Hooks for input range grad computation. Will not be saved in state_dict
        self._tile_input_grad_hook = None
        self._tile_input = None  # type: Tensor
        self._x_input_grad = None  # type: Tensor
        self._backward_hook_handle = None  # type: BackwardHook

    def _create_simulator_tile(  # type: ignore
        self, x_size: int, d_size: int, rpu_config: "TorchInferenceRPUConfig"
    ) -> "TorchSimulatorTile":
        """Create a simulator tile.

        Args:
            weight: 2D weight
            rpu_config: resistive processing unit configuration

        Returns:
            a simulator tile based on the specified configuration.
        """
        return rpu_config.simulator_tile_class(x_size=x_size, d_size=d_size, rpu_config=rpu_config)

    def _recreate_simulator_tile(  # type: ignore[override]
        self, x_size: int, d_size: int, rpu_config: "TorchInferenceRPUConfig"
    ) -> Any:  # just use Any instead of Union["SimulatorTile", tiles.AnalogTile, ..]
        """Re-create a simulator tile in __setstate__.

        Args:
            x_size: input size
            d_size: output size
            rpu_config: resistive processing unit configuration

        Returns:
            a simulator tile based on the specified configuration.
        """
        return self.tile

    def init_input_processing(self) -> bool:
        """Helper function to initialize the input processing.

        Note:
            This method is called from the constructor.

        Returns:
            whether input processing is enabled

        Raises: ConfigError in case ``manage_output_clipping`` is
            enabled but not supported.
        """
        enable = super().init_input_processing()

        if not enable:
            return False
        return True

    def set_scales(self, scales: Tensor) -> None:
        """Set all scales with a new scale.

        This will set the mapping scales to ``scales`` and set all other scales to 1.

        Args:
            scales: scales to set.
        """

        super().set_scales(scales)

        # - Remove old hook
        if self._backward_hook_handle is not None:
            self._backward_hook_handle.remove()

        def hook(grad: Tensor) -> Tensor:
            return grad / scales.to(grad.device).view(-1, 1) ** 2

        if self.tile.weight.requires_grad:
            self._backward_hook_handle = self.tile.register_weight_hook(hook)

    def pre_forward(
        self, x_input: Tensor, dim: int, is_test: bool = False, ctx: Any = None
    ) -> Tensor:
        """Operations before the actual forward step for pre processing.

        By default, this is an no-op. However, it could be overridden
        in derived tile classes.

        Args:
            x_input: input tensor for the analog MVM of the tile.
            dim: input channel dimension, ie the x_size dimension
            is_test: whether in eval mode
            ctx: torch auto-grad context [Optional]

        Returns:
            Output tensor of the same shape
        """

        # pylint: disable=unused-argument
        if self.input_range is not None:
            x_input = self.apply_input_range(x_input, not is_test)

        x_input = InputRangeForward.apply(
            x_input, self.input_range, self.rpu_config.pre_post.input_range
        )

        if self.input_range is not None:
            x_input = x_input / self.input_range

        return x_input

    def forward(self, x_input: Tensor, tensor_view: Optional[Tuple] = None) -> Tensor:
        """Torch forward function that calls the analog forward"""
        # pylint: disable=arguments-differ

        # Note: this is now called with autograd enabled and thus will
        # not use the BaseTile.backward functionality. It will call
        # the tile.forward pass internally
        self.tile.set_config(self.rpu_config)  # to allow on-the-fly changes
        out = self.joint_forward(x_input, is_test=not self.training)

        if tensor_view is None:
            tensor_view = self.get_tensor_view(out.dim())
        out = self.apply_out_scaling(out, tensor_view)

        if self.digital_bias:
            return out + self.bias.view(*tensor_view)
        return out

    @no_grad()
    def post_update_step(self) -> None:
        """
        Clip and remap weights after weights have been updated.
        """
        if hasattr(self.rpu_config, "clip") and self.rpu_config.clip.type != WeightClipType.NONE:
            self.tile.clip_weights(self.rpu_config.clip)

        if hasattr(self.rpu_config, "remap") and self.rpu_config.remap.type != WeightRemapType.NONE:
            scales = self.get_scales()
            scales = self.tile.remap_weights(self.rpu_config.remap, scales)
            self.set_scales(scales)

    def get_forward_parameters(self) -> Dict[str, Tensor]:
        """Get the additional parameters generated for the forward pass.

        Returns:
            Dictionary of the forward parameters set.
        """
        dic = {}
        if self.tile.out_noise_values is not None:
            dic["out_noise_values"] = self.tile.out_noise_values
        return dic

    def set_forward_parameters(
        self, dic: Optional[Dict[str, Tensor]] = None, **kwargs: Dict[str, Tensor]
    ) -> None:
        """Set the additional parameters generated for the forward pass.

        Currently only ``out_noise_values`` is implemented.

        Args:
            dic: dictionary of parameters to set (from :meth:`get_forward_parameter`)
            kwargs: parameter names can alternatively given directly as keywords

        Raises:
            ArgumentError: If size are mismatched or keyword unknown
        """
        if dic is None:
            dic = kwargs
        par_lst = ["out_noise_values"]
        for par in par_lst:
            if par in dic:
                current_value = getattr(self.tile, par)
                new_value = dic[par]
                if not isinstance(new_value, Tensor):
                    raise ArgumentError(f"{par} type mismatch. Expected tensor!")
                if current_value is None or current_value.size() != new_value.size():
                    raise ArgumentError(f"{par} size mismatch!")

                setattr(self.tile, par, new_value.reshape(*current_value.shape))
        if set(par_lst) != set(list(dic.keys())):
            raise ArgumentError("Unknown parameter keys given!")
