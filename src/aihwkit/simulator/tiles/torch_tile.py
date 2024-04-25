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

"""Low level implementation of torch-based tile."""

from typing import Callable, Any, Type, TYPE_CHECKING

from torch import Tensor, zeros, clamp, randn
from torch.nn import Parameter, Module
from torch.autograd import no_grad

from aihwkit.exceptions import TorchTileConfigError, AnalogBiasConfigError
from aihwkit.simulator.tiles.analog_mvm import AnalogMVM
from aihwkit.simulator.tiles.base import SimulatorTile
from aihwkit.simulator.tiles.utils import UniformQuantize
from aihwkit.simulator.parameters.enums import WeightModifierType, WeightClipType, WeightRemapType

from aihwkit.simulator.parameters.inference import (
    WeightModifierParameter,
    WeightClipParameter,
    WeightRemapParameter,
)

if TYPE_CHECKING:
    from aihwkit.simulator.configs.configs import TorchInferenceRPUConfig


class TorchSimulatorTile(SimulatorTile, Module):
    """Torch based tile class.

    Args:
        x_size: input size
        d_size: output size
        rpu_config: resistive processing unit configuration.
    """

    # pylint: disable=abstract-method

    def __init__(
        self,
        x_size: int,
        d_size: int,
        rpu_config: "TorchInferenceRPUConfig",
        bias: bool = False,
        analog_mvm: Type[AnalogMVM] = AnalogMVM,
    ):
        Module.__init__(self)
        self.x_size = x_size
        self.d_size = d_size

        if bias:
            raise AnalogBiasConfigError("Analog bias is not supported for TorchSimulatorTile")

        self.check_rpu_config_support(rpu_config)
        analog_mvm.check_support(rpu_config.forward)

        self._analog_mvm = analog_mvm
        self._f_io = rpu_config.forward
        self._modifier = rpu_config.modifier
        dtype = rpu_config.runtime.data_type.as_torch()
        if self._f_io.out_noise_std > 0:
            out_noise_values = (self._f_io.out_noise * (1.0 + randn((d_size,), dtype=dtype))).abs()
            self.register_buffer("out_noise_values", out_noise_values)
        else:
            self.out_noise_values = None

        # Don't use randn here to avoid changing the seed in
        # comparison to RPUCuda tiles
        self.weight = Parameter(zeros(self.d_size, self.x_size, dtype=dtype))

    def set_weights(self, weight: Tensor) -> None:
        """Set the tile weights.

        Args:
            weight: ``[out_size, in_size]`` weight matrix.
        """
        self.weight.data = weight.clone().to(self.weight.device)

    def get_weights(self) -> Tensor:
        """Get the tile weights.

        Returns:
            a tuple where the first item is the ``[out_size, in_size]`` weight
            matrix; and the second item is either the ``[out_size]`` bias vector
            or ``None`` if the tile is set not to use bias.
        """
        return self.weight.data.detach().cpu()

    def get_x_size(self) -> int:
        """Returns input size of tile"""
        return self.x_size

    def get_d_size(self) -> int:
        """Returns output size of tile"""
        return self.d_size

    def get_brief_info(self) -> str:
        """Returns a brief info"""
        return self.__class__.__name__ + "({})".format(self.extra_repr())

    def extra_repr(self) -> str:
        return "{}, {}, {}".format(self.d_size, self.x_size, self.weight.device).rstrip()

    @no_grad()
    def remap_weights(self, remap: WeightRemapParameter, scales: Tensor) -> Tensor:
        """
        Remap the weights to the specified range and return new scales.

        Args:
            remap: hyper-parameters defining the remapping
            scales: current scale values.

        Raises:
            ConfigError: If WeightRemapType is unknown.

        Returns:
            Tensor: New scales.
        """
        # pylint: disable=arguments-differ

        scaled_weights = self.weight * scales.view(-1, 1)
        if remap.type == WeightRemapType.LAYERWISE_SYMMETRIC:
            new_scale = scaled_weights.abs().max()
            self.weight.data = scaled_weights / new_scale
            return new_scale.view(-1)
        if remap.type == WeightRemapType.CHANNELWISE_SYMMETRIC:
            new_scale, _ = scaled_weights.abs().max(1)
            self.weight.data = scaled_weights / new_scale.view(-1, 1)
            return new_scale

        raise TorchTileConfigError(f"Unknown weight remap type {remap.type}")

    @no_grad()
    def clip_weights(self, clip: WeightClipParameter) -> None:
        """Clip the weights.

        Args:
            clip: parameters specifying the clipping methof and type.

        Raises:
            NotImplementedError: For unsupported WeightClipTypes
            ConfigError: If unknown WeightClipType used.
        """

        if clip.type == WeightClipType.FIXED_VALUE:
            self.weight.data = clamp(self.weight, -clip.fixed_value, clip.fixed_value)
        elif clip.type == WeightClipType.LAYER_GAUSSIAN:
            alpha = self.weight.std() * clip.sigma
            if clip.fixed_value > 0:
                alpha = min(clip.fixed_value, alpha)
            self.weight.data = clamp(self.weight, -alpha, alpha)

        elif clip.type == WeightClipType.AVERAGE_CHANNEL_MAX:
            raise NotImplementedError
        else:
            raise TorchTileConfigError(f"Unknown clip type {clip.type}")

    def set_config(self, rpu_config: "TorchInferenceRPUConfig") -> None:
        """Updated the configuration to allow on-the-fly changes.

        Args:
            rpu_config: configuration to use in the next forward passes.
        """
        self._f_io = rpu_config.forward
        self._modifier = rpu_config.modifier

    def register_weight_hook(self, hook: Callable) -> Any:
        """Register a hook to the weights."""
        return self.weight.register_hook(hook)

    def forward(
        self,
        x_input: Tensor,
        bias: bool = False,
        in_trans: bool = False,
        out_trans: bool = False,
        is_test: bool = False,
        non_blocking: bool = False,
    ) -> Tensor:
        # pylint: disable=too-many-locals, too-many-branches

        if in_trans or out_trans:
            raise TorchTileConfigError("Non-trans MVMs supported only.")

        if not is_test:
            noisy_weights = TorchSimulatorTile.modify_weight(
                self.weight, self._modifier, x_input.shape[0]
            )
        else:
            noisy_weights = self.weight

        return self._analog_mvm.matmul(
            noisy_weights,
            x_input,
            self._f_io,
            False,
            is_test=is_test,
            out_noise_values=self.out_noise_values,
        )

    @staticmethod
    def modify_weight(
        inp_weight: Tensor, modifier: WeightModifierParameter, batch_size: int
    ) -> Tensor:
        """Weight modifier that adds noise to the weights according to rpu config.

        Args:
            inp_weight: Input weights.
            modifier: Noise injection configuration.
            batch_size (int): Batch size.

        Raises:
            TorchTileConfigError: Unsupported/unknown weight modifier type.

        Returns:
            Weights with noise injected.
        """
        per_batch_sample = modifier.per_batch_sample
        target_shape = (batch_size,) + inp_weight.shape if per_batch_sample else inp_weight.shape
        if modifier.type in [WeightModifierType.NONE, WeightModifierType.COPY]:
            return inp_weight

        if modifier.type == WeightModifierType.MULT_NORMAL:
            with no_grad():
                gauss = randn(size=target_shape, device=inp_weight.device, dtype=inp_weight.dtype)
                noise = inp_weight * modifier.std_dev * gauss
            out_weight = inp_weight.clone() + noise
            return out_weight

        assumed_wmax = modifier.assumed_wmax
        if modifier.rel_to_actual_wmax:
            assumed_wmax = inp_weight.abs().max()

        if modifier.type == WeightModifierType.DISCRETIZE:
            # - Discretize the weights on the fly and backprob through them
            out_weight = inp_weight.clone().view(target_shape)
            out_weight = UniformQuantize.apply(
                out_weight, modifier.res, assumed_wmax, modifier.sto_round
            )
        elif modifier.type == WeightModifierType.ADD_NORMAL:
            with no_grad():
                noise = (
                    modifier.std_dev
                    * assumed_wmax
                    * randn(target_shape, device=inp_weight.device, dtype=inp_weight.dtype)
                )
            out_weight = inp_weight.clone() + noise
        else:
            raise TorchTileConfigError(f"Weight modifier {modifier} not supported")
        return out_weight

    @classmethod
    def check_rpu_config_support(cls, rpu_config: "TorchInferenceRPUConfig") -> None:
        """Check the RPUConfig for support with TorchSimulatorTile

        Throws an assertion error when there is an incompatibility
        with the used rpu config and what the current torch tile
        supports

        Args:
            rpu_config: the rpu config to be checked
        Raises:
            TorchTileConfigError: in case a feature is not supported
        """
        # pylint: disable=too-many-branches

        if rpu_config.clip.type == WeightClipType.AVERAGE_CHANNEL_MAX:
            raise TorchTileConfigError("Clip type AVERAGE_CHANNEL_MAX not supported by torch tile")

        if rpu_config.modifier.enable_during_test:
            raise TorchTileConfigError("Modifier noise is currently always off in the torch tile")

        if rpu_config.modifier.copy_last_column:
            raise TorchTileConfigError("Bias is assumed to be in digital only for torch tile")

        if rpu_config.modifier.type in [
            WeightModifierType.DOREFA,
            WeightModifierType.POLY,
            WeightModifierType.PROG_NOISE,
            WeightModifierType.PCM_NOISE,
            WeightModifierType.DISCRETIZE_ADD_NORMAL,
        ]:
            raise TorchTileConfigError(
                "The given modifier noise type is not supported in the torch tile"
            )

        if rpu_config.modifier.pdrop > 0.0:
            raise TorchTileConfigError("The drop-connect is not supported in the torch tile")

        if rpu_config.remap.type not in [
            WeightRemapType.LAYERWISE_SYMMETRIC,
            WeightRemapType.CHANNELWISE_SYMMETRIC,
            WeightRemapType.NONE,
        ]:
            raise TorchTileConfigError("Remapping type not supported.")

        if rpu_config.remap.remapped_wmax != 1.0:
            raise TorchTileConfigError("Remapping to value different from 1.0 not supported.")

        if rpu_config.remap.max_scale_range != 0.0:
            raise TorchTileConfigError("Remap parameter max_scale_range must be 0.0.")

        if rpu_config.remap.max_scale_ref != 0.0:
            raise TorchTileConfigError("Remap parameter max_scale_ref must be 0.0.")
