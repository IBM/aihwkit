# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023 IBM. All Rights Reserved.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""High level analog tiles (inference)."""

# pylint: disable=too-many-ancestors

from typing import Optional, Union, Any, Tuple, List, TYPE_CHECKING

from torch import device as torch_device
from torch import ones, Tensor, float32
from torch.nn import Module
from torch.autograd import no_grad

from aihwkit.exceptions import ConfigError
from aihwkit.simulator.tiles.functions import AnalogFunction
from aihwkit.simulator.tiles.periphery import TileWithPeriphery
from aihwkit.simulator.tiles.module import TileModule
from aihwkit.simulator.tiles.rpucuda import RPUCudaSimulatorTileWrapper
from aihwkit.simulator.tiles.base import BaseTile
from aihwkit.simulator.rpu_base import tiles
from aihwkit.simulator.parameters.base import RPUConfigGeneric
from aihwkit.simulator.parameters.helpers import parameters_to_bindings
from aihwkit.simulator.parameters.enums import WeightModifierType, WeightClipType, WeightRemapType

if TYPE_CHECKING:
    from aihwkit.simulator.configs import InferenceRPUConfig


class InferenceTileWithPeriphery(TileWithPeriphery):
    """Additional (peripheral) functionality for hardware-aware
    training and inference.

    Note:

        Here it is assumed that the training is done in software and
        only the inference pass is done on analog hardware.

    """

    # pylint: disable=abstract-method

    def __init__(self) -> None:
        super().__init__()

        self.drift_baseline = None
        self.drift_readout_tensor = None  # type: Optional[Tensor]
        if isinstance(self, Module):
            self.register_buffer("alpha", ones((1,), dtype=float32))
        else:
            self.alpha = ones((1,), dtype=float32)

        # Helpers.
        self.programmed_weights = None  # type: Optional[Tensor]
        self.nu_drift_list = None  # type: Optional[List[Tensor]]

    def _create_simulator_tile(
        self, x_size: int, d_size: int, rpu_config: RPUConfigGeneric
    ) -> tiles.AnalogTile:
        """Create a simulator tile.

        Args:
            x_size: input size
            d_size: output size
            rpu_config: resistive processing unit configuration

        Returns:
            a simulator tile based on the specified configuration.
        """

        meta_parameter = rpu_config.as_bindings()
        device_parameter = rpu_config.device.as_bindings()

        return meta_parameter.create_array(x_size, d_size, device_parameter)

    @no_grad()
    def init_mapping_scales(self) -> None:
        """Helper function to initialize the mapping scales used to scale the
        weights in digital and determine the conductance conversion.

        Note:
            This method is called from the constructor.
        """
        super().init_mapping_scales()
        if hasattr(self.rpu_config, "remap") and self.rpu_config.remap.type != WeightRemapType.NONE:
            # needs to be always out_size
            mapping_scales = ones(
                (self.out_size,), dtype=float32, device=self.device, requires_grad=False
            )
            self.set_mapping_scales(mapping_scales)

    @no_grad()
    def _forward_drift_readout_tensor(self, reset_if: bool = False) -> Optional[Tensor]:
        """Perform a forward pass using the drift read-out tensor.

        Args:
            reset_if: Will reset the readout tensor, otherwise use the stored one

        Returns:
            Readout tensor if drift compensation is on
        """
        if (
            not hasattr(self.rpu_config, "drift_compensation")
            or self.rpu_config.drift_compensation is None
        ):
            return None

        if self.drift_readout_tensor is None or reset_if:
            self.drift_readout_tensor = (
                self.rpu_config.drift_compensation.get_readout_tensor(self.tile.get_x_size())
                .detach()
                .to(self.device)
            )
            if self.in_trans:
                self.drift_readout_tensor = self.drift_readout_tensor.tranpose(0, 1).clone()
        else:
            self.drift_readout_tensor = self.drift_readout_tensor.to(self.device)

        # We need to take the bias as a common column here, also we do
        # not want to use indexed.
        return self.tile.forward(
            self.drift_readout_tensor, False, self.in_trans, self.out_trans, True, self.non_blocking
        )

    @no_grad()
    def program_weights(self, from_reference: bool = True) -> None:
        """Apply weights noise to the current tile weights and saves these for
        repeated drift experiments.

        This method also establishes the drift coefficients for each
        conductance slice.

        Will also reset the drift readout tensor and compuate a new
        drift compensation baseline

        Args:
            from_reference: Whether to use weights from reference

        Raises:
            ConfigError: in case of ``noise_model`` is not defined in
                the RPUConfig
        """
        # pylint: disable=arguments-differ

        if not hasattr(self.rpu_config, "noise_model"):
            raise ConfigError("Seems that RPUConfig is not of type InferenceRPUConfig.")

        if not from_reference or self.reference_combined_weights is None:
            self.reference_combined_weights = Tensor(self.tile.get_weights())

        (
            self.programmed_weights,
            self.nu_drift_list,
        ) = self.rpu_config.noise_model.apply_programming_noise(self.reference_combined_weights)

        self.tile.set_weights(self.programmed_weights)

        if (
            hasattr(self.rpu_config, "drift_compensation")
            and self.rpu_config.drift_compensation is not None
        ):
            forward_output = self._forward_drift_readout_tensor(True)
            self.drift_baseline = self.rpu_config.drift_compensation.init_baseline(forward_output)

    @no_grad()
    def drift_weights(self, t_inference: float = 0.0) -> None:
        """Programs and drifts the current reference weights.

        The current weight reference is either the current weights or
        the ones at the time when :meth:`initialize_drift_reference`
        was called, which then would overwrite the current weights
        with the drifted ones.

        Args:
            t_inference: Time (in sec) of assumed inference
                time. Programming ends at t=0s.  The rest is waiting time,
                where the devices might drift and accumulate noise. See
                noise model used for details.

        Raises:
            ConfigError: in case of ``noise_model`` is not defined in
                the RPUConfig

        """
        # pylint: disable=arguments-differ, arguments-renamed

        if not hasattr(self.rpu_config, "noise_model"):
            raise ConfigError("Seems that RPUConfig is not of type InferenceRPUConfig.")

        if self.programmed_weights is None:
            self.program_weights()

        drifted_weights = self.rpu_config.noise_model.apply_drift_noise(
            self.programmed_weights, self.nu_drift_list, t_inference
        )
        self.tile.set_weights(drifted_weights)

        if (
            hasattr(self.rpu_config, "drift_compensation")
            and self.rpu_config.drift_compensation is not None
        ):
            forward_output = self._forward_drift_readout_tensor()
            self.alpha = self.rpu_config.drift_compensation.apply(
                forward_output, self.drift_baseline
            ).to(self.device)

    def post_forward(
        self, x_output: Tensor, dim: int, is_test: bool = False, ctx: Any = None
    ) -> Tensor:
        """Operations after the actual forward step for post processing"""

        x_output = super().post_forward(x_output, dim, is_test, ctx)

        if (
            is_test
            and hasattr(self.rpu_config, "drift_compensation")
            and self.rpu_config.drift_compensation is not None
        ):
            # only do drift compensation in eval mode
            return x_output * self.alpha
        return x_output

    @no_grad()
    def post_update_step(self) -> None:
        """Operators that need to be called once per mini-batch.

        In the :class:`~InferenceTile`, the following calls are made
        (if enabled in the ``rpu_config`` settings). First, the post
        update step of the parent is called, then the weight clipping
        is done, subsequently then remapping is done (if enforced),
        and finally the forward-backward weight modifier is
        called. The latter will modify the weights that are used
        during forward and backward (but not update) until the next
        time this function is called.
        """
        super().post_update_step()

        # TODO: make this a little nicer. Now each time bindings are
        # generated, which however has the advantage that parameters
        # could be changed-on-the-fly

        if hasattr(self.rpu_config, "clip") and self.rpu_config.clip.type != WeightClipType.NONE:
            weight_clip_params = parameters_to_bindings(self.rpu_config.clip)
            self.tile.clip_weights(weight_clip_params)

        if hasattr(self.rpu_config, "remap") and self.rpu_config.remap.type != WeightRemapType.NONE:
            weight_remap_params = parameters_to_bindings(self.rpu_config.remap)
            scales = self.get_scales()
            scales = self.tile.remap_weights(weight_remap_params, scales)
            self.set_scales(scales)

        # update the forward / backward modified weights here
        if not hasattr(self.rpu_config, "modifier"):
            return
        if self.rpu_config.modifier.type == WeightModifierType.NONE:
            return
        if (
            self.rpu_config.modifier.type == WeightModifierType.COPY
            and self.rpu_config.modifier.pdrop <= 0.0
        ):
            return
        weight_modify_params = parameters_to_bindings(self.rpu_config.modifier)
        self.tile.modify_weights(weight_modify_params)  # type: ignore

    def cuda(self, device: Optional[Union[torch_device, str, int]] = None) -> "BaseTile":
        self.alpha = self.alpha.cuda(device)
        ret = super().cuda(device)
        return ret

    def cpu(self) -> "BaseTile":
        self.alpha = self.alpha.cpu()
        ret = super().cpu()
        return ret


class InferenceTile(TileModule, InferenceTileWithPeriphery, RPUCudaSimulatorTileWrapper):
    """Tile used for analog inference and hardware-aware training for inference.

    Note:

        This tile uses RPUCuda library with backward and update set to
        perfect.

    Args:
        out_size: output size
        in_size: input size
        rpu_config: resistive processing unit configuration.
        bias: whether to add a bias column to the tile.
        in_trans: Whether to assume an transposed input (batch first)
        out_trans: Whether to assume an transposed output (batch first)
        shared_weights: Whether to keep the weight in torch's memory space

    """

    def __init__(
        self,
        out_size: int,
        in_size: int,
        rpu_config: Optional["InferenceRPUConfig"] = None,
        bias: bool = False,
        in_trans: bool = False,
        out_trans: bool = False,
        shared_weights: bool = True,
    ):
        if not rpu_config:
            # Import `InferenceRPUConfig` dynamically to avoid import cycles.
            # pylint: disable=import-outside-toplevel
            from aihwkit.simulator.configs import InferenceRPUConfig

            rpu_config = InferenceRPUConfig()

        TileModule.__init__(self)
        RPUCudaSimulatorTileWrapper.__init__(
            self,
            out_size,
            in_size,
            rpu_config,
            bias,
            in_trans,
            out_trans,
            shared_weights=shared_weights,
        )
        InferenceTileWithPeriphery.__init__(self)

    def _create_simulator_tile(  # type: ignore
        self, x_size: int, d_size: int, rpu_config: "InferenceRPUConfig"
    ) -> tiles.AnalogTile:
        """Create a simulator tile.

        Args:
            x_size: input size
            d_size: output size
            rpu_config: resistive processing unit configuration

        Returns:
            a simulator tile based on the specified configuration.
        """
        meta_parameter = rpu_config.as_bindings()
        device_parameter = rpu_config.device.as_bindings()

        return meta_parameter.create_array(x_size, d_size, device_parameter)

    def forward(
        self, x_input: Tensor, tensor_view: Optional[Tuple] = None  # type: ignore
    ) -> Tensor:
        """Torch forward function that calls the analog forward"""
        # pylint: disable=arguments-differ

        out = AnalogFunction.apply(
            self.get_analog_ctx(), self, x_input, self.shared_weights, not self.training
        )

        if tensor_view is None:
            tensor_view = self.get_tensor_view(out.dim())
        out = self.apply_out_scaling(out, tensor_view)

        if self.digital_bias:
            return out + self.bias.view(*tensor_view)
        return out
