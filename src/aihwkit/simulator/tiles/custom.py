# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""High level analog tiles (floating point)."""

from typing import Optional, Tuple, Type, Any
from dataclasses import dataclass, field

from torch import Tensor, zeros, float32, randn_like, rand_like
from torch.autograd import no_grad
from torch.nn import Module

from aihwkit.exceptions import AnalogBiasConfigError, TileError
from aihwkit.simulator.tiles.base import SimulatorTileWrapper, SimulatorTile
from aihwkit.simulator.tiles.module import TileModule
from aihwkit.simulator.tiles.torch_tile import AnalogMVM
from aihwkit.simulator.tiles.periphery import TileWithPeriphery
from aihwkit.simulator.tiles.functions import AnalogFunction
from aihwkit.simulator.tiles.array import TileModuleArray
from aihwkit.simulator.parameters.pre_post import PrePostProcessingRPU
from aihwkit.simulator.parameters.mapping import MappableRPU
from aihwkit.simulator.parameters.helpers import _PrintableMixin
from aihwkit.simulator.parameters.io import IOParameters


class CustomSimulatorTile(SimulatorTile, Module):
    """Custom Simulator Tile for analog training.

    To implement specialized SGD algorithms here the forward /
    backward / update are explicitly defined without using
    auto-grad.

    When not overriden, forward and backward use the analog forward
    pass of the
    :class:`~aihwkit.simulator.tiles.torch_tile.TorchSimulatorTile`.

    Update is in floating point but adds optionally noise to the
    gradient.

    """

    # pylint: disable=attribute-defined-outside-init

    def __init__(self, x_size: int, d_size: int, rpu_config: "CustomRPUConfig", bias: bool = False):
        Module.__init__(self)
        self.x_size = x_size
        self.d_size = d_size
        self.learning_rate = 0.1

        if bias:
            raise AnalogBiasConfigError("Analog bias is not supported for TorchSimulatorTile")

        AnalogMVM.check_support(rpu_config.forward)
        AnalogMVM.check_support(rpu_config.backward)
        self.set_config(rpu_config)

        # just buffer to handle device, since do not use auto grad
        self.register_buffer("_analog_weight", zeros(self.d_size, self.x_size, dtype=float32))

    @no_grad()
    def forward(
        self,
        x_input: Tensor,
        bias: bool = False,
        in_trans: bool = False,
        out_trans: bool = False,
        is_test: bool = False,
        non_blocking: bool = False,
    ) -> Tensor:
        """General simulator tile forward.

        Note:
            Ignores additional arguments

        Raises:
            TileError: in case transposed input / output or bias is requested
        """
        if bias or in_trans or out_trans or non_blocking:
            raise TileError("transposed inputs or analog bias not supported")

        return AnalogMVM.matmul(self._analog_weight, x_input, self._fwd_io, False)  # type: ignore

    def backward(
        self,
        d_input: Tensor,
        bias: bool = False,
        in_trans: bool = False,
        out_trans: bool = False,
        non_blocking: bool = False,
    ) -> Tensor:
        """Backward pass.

        Note:
            Ignores additional arguments

        Raises:
            TileError: in case transposed input / output or bias is requested
        """
        if bias or in_trans or out_trans or non_blocking:
            raise TileError("transposed inputs or analog bias not supported")

        return AnalogMVM.matmul(self._analog_weight, d_input, self._bwd_io, True)  # type: ignore

    def update(
        self,
        x_input: Tensor,
        d_input: Tensor,
        bias: bool = False,
        in_trans: bool = False,
        out_trans: bool = False,
        non_blocking: bool = False,
    ) -> Tensor:
        """Update with gradient noise.

        Note:
            Ignores additional arguments

        Raises:
            TileError: in case transposed input / output or bias is requested
        """
        if bias or in_trans or out_trans or non_blocking:
            raise TileError("transposed inputs or analog bias not supported")

        delta_w = d_input.view(-1, d_input.size(-1)).T @ x_input.view(-1, x_input.size(-1))

        if self._update.gradient_noise:
            delta_w += self._update.gradient_noise * randn_like(delta_w)

        self._analog_weight = self._analog_weight - self.learning_rate * delta_w  # type: ignore

    def set_config(self, rpu_config: "CustomRPUConfig") -> None:
        """Updated the configuration to allow on-the-fly changes.

        Args:
            rpu_config: configuration to use in the next forward passes.
        """
        self._fwd_io = rpu_config.forward
        self._bwd_io = rpu_config.backward
        self._update = rpu_config.update

    def set_weights(self, weight: Tensor) -> None:
        """Set the tile weights.

        Args:
            weight: ``[out_size, in_size]`` weight matrix.
        """
        device = self._analog_weight.device
        self._analog_weight = weight.clone().to(device)

    def get_weights(self) -> Tensor:
        """Get the tile weights.

        Returns:
            a tuple where the first item is the ``[out_size, in_size]`` weight
            matrix; and the second item is either the ``[out_size]`` bias vector
            or ``None`` if the tile is set not to use bias.
        """
        return self._analog_weight.data.detach().cpu()

    def get_x_size(self) -> int:
        """Returns input size of tile"""
        return self.x_size

    def get_d_size(self) -> int:
        """Returns output size of tile"""
        return self.d_size

    def get_brief_info(self) -> str:
        """Returns a brief info"""
        return self.__class__.__name__ + f"({self.extra_repr()})"

    def extra_repr(self) -> str:
        """Extra documentation string."""
        return "{}, {}, {}".format(self.d_size, self.x_size, self._analog_weight.device).rstrip()

    def get_learning_rate(self) -> Optional[float]:
        """Get the learning rate of the tile.

        Returns:
           learning rate if exists.
        """
        return self.learning_rate

    def set_learning_rate(self, learning_rate: Optional[float]) -> None:
        """Set the learning rate of the tile.

        No-op for tiles that do not need a learning rate.

        Args:
           learning rate: learning rate to set
        """
        if learning_rate is not None:
            self.learning_rate = learning_rate

    def set_weights_uniform_random(self, bmin: float, bmax: float) -> None:
        """Sets the weights to uniform random numbers.

        Args:
           bmin: min value
           bmax: max value

        Raises:
            TileError: in case bmin >= bmax
        """
        if bmin >= bmax:
            raise TileError("Bmin should be smaller than bmax")
        self.set_weights(rand_like(self.get_weights()) / (bmax - bmin) - bmin)

    def get_meta_parameters(self) -> Any:
        """Returns meta parameters."""
        raise NotImplementedError


class CustomTile(TileModule, TileWithPeriphery, SimulatorTileWrapper):
    r"""Custom tile based on :class:`TileWithPeriphery`.

    Implements a tile with periphery for analog training without the
    RPUCuda engine.

    Raises:
       TileError: in case in-trans / out-trans is used (not supported)
       AnalogBiasConfigError: if analog bias is enabled in the `RPUConfig`
    """

    supports_indexed = False

    def __init__(
        self,
        out_size: int,
        in_size: int,
        rpu_config: Optional["CustomRPUConfig"] = None,
        bias: bool = False,
        in_trans: bool = False,
        out_trans: bool = False,
    ):
        if in_trans or out_trans:
            raise TileError("in/out trans is not supported.")

        if not rpu_config:
            rpu_config = CustomRPUConfig()

        TileModule.__init__(self)
        SimulatorTileWrapper.__init__(
            self,
            out_size,
            in_size,
            rpu_config,  # type: ignore
            bias,
            in_trans,
            out_trans,
            torch_update=True,
        )
        TileWithPeriphery.__init__(self)

        if self.analog_bias:
            raise AnalogBiasConfigError("Analog bias is not supported for the torch tile")

    def _create_simulator_tile(  # type: ignore
        self, x_size: int, d_size: int, rpu_config: "CustomRPUConfig"
    ) -> "SimulatorTile":
        """Create a simulator tile.

        Args:
            weight: 2D weight
            rpu_config: resistive processing unit configuration

        Returns:
            a simulator tile based on the specified configuration.
        """
        return rpu_config.simulator_tile_class(x_size=x_size, d_size=d_size, rpu_config=rpu_config)

    def forward(
        self, x_input: Tensor, tensor_view: Optional[Tuple] = None  # type: ignore
    ) -> Tensor:
        """Torch forward function that calls the analog context forward"""
        # pylint: disable=arguments-differ

        # to enable on-the-fly changes. However, with caution: might
        # change rpu config for backward / update while doing another forward.
        self.tile.set_config(self.rpu_config)

        out = AnalogFunction.apply(
            self.get_analog_ctx(), self, x_input, self.shared_weights, not self.training
        )

        if tensor_view is None:
            tensor_view = self.get_tensor_view(out.dim())
        out = self.apply_out_scaling(out, tensor_view)

        if self.digital_bias:
            return out + self.bias.view(*tensor_view)
        return out

    def post_update_step(self) -> None:
        """Operators that need to be called once per mini-batch.

        Note:
            This function is called by the analog optimizer.

        Caution:
            If no analog optimizer is used, the post update steps will
            not be performed.
        """


@dataclass
class CustomUpdateParameters(_PrintableMixin):
    """Custom parameters for the update"""

    gradient_noise: float = 0.0
    """Adds Gaussian noise (with this std) to the weight gradient. """


@dataclass
class CustomRPUConfig(MappableRPU, PrePostProcessingRPU):
    """Configuration for resistive processing unit using the CustomTile."""

    tile_class: Type = CustomTile
    """Tile class that corresponds to this RPUConfig."""

    simulator_tile_class: Type = CustomSimulatorTile
    """Simulator tile class implementing the analog forward / backward / update."""

    tile_array_class: Type = TileModuleArray
    """Tile class used for mapped logical tile arrays."""

    forward: IOParameters = field(default_factory=IOParameters)
    """Input-output parameter setting for the forward direction."""

    backward: IOParameters = field(default_factory=IOParameters)
    """Input-output parameter setting for the backward direction."""

    update: CustomUpdateParameters = field(default_factory=CustomUpdateParameters)
    """Parameter for the update behavior."""
