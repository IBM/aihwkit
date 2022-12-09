# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022 IBM. All Rights Reserved.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""High level analog tiles (inference)."""

from typing import List, Optional, Union, Any, TYPE_CHECKING

from torch import device as torch_device
from torch import ones, zeros, Tensor, float32
from torch.autograd import no_grad

from aihwkit.simulator.tiles.analog import AnalogTile

if TYPE_CHECKING:
    from aihwkit.simulator.configs import InferenceRPUConfig
    from aihwkit.simulator.tiles import BaseTile

# pylint: disable=too-many-instance-attributes


class InferenceTile(AnalogTile):
    """Tile used for analog inference and hardware-aware training for inference.

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
            rpu_config: Optional['InferenceRPUConfig'] = None,
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

        # CAUTION: one cannot save parts of the RPUConfig as
        # properties of the class! This is because those attributes
        # would be restored even if the RPUConfig would be replaced
        # during checkpoint loading

        self.drift_baseline = None
        self.drift_readout_tensor = None  # type: Optional[Tensor]
        self.alpha = ones((1,))

        # Helpers.
        self.reference_combined_weights = None  # type: Optional[Tensor]
        self.programmed_weights = None  # type: Optional[Tensor]
        self.nu_drift_list = None  # type: Optional[List[Tensor]]

        super().__init__(out_size, in_size, rpu_config, bias, in_trans, out_trans)

        if shared_weights:
            self.shared_weights = zeros(out_size, in_size + int(bias),
                                        requires_grad=True)  # type: Tensor
            self.ensure_shared_weights()

    @no_grad()
    def init_mapping_scales(self) -> None:
        """Helper function to initialize the mapping scales used to scale the
        weights in digital and determine the conductance conversion.

        Note:
            This method is called from the constructor.
        """
        # Import `aihwkit.simulator.configs` items dynamically to avoid import cycles.
        # pylint: disable=import-outside-toplevel
        from aihwkit.simulator.configs.utils import WeightRemapType

        super().init_mapping_scales()
        remap = self.rpu_config.remap  # type: ignore
        if remap.type != WeightRemapType.NONE:
            # needs to be always out_size
            mapping_scales = ones((self.out_size, ),
                                  dtype=float32,
                                  device=self.device,
                                  requires_grad=False)
            self.set_mapping_scales(mapping_scales)

    @no_grad()
    def _forward_drift_readout_tensor(self, reset_if: bool = False) -> Optional[Tensor]:
        """Perform a forward pass using the drift read-out tensor.

        Args:
            reset_if: Will reset the readout tensor, otherwise use the stored one

        Returns:
            Readout tensor if drift compensation is on
        """

        if self.rpu_config.drift_compensation is None:
            return None

        if self.drift_readout_tensor is None or reset_if:
            self.drift_readout_tensor = self.rpu_config.drift_compensation.get_readout_tensor(
                self.tile.get_x_size()).detach().to(self.device)
            if self.in_trans:
                self.drift_readout_tensor = self.drift_readout_tensor.tranpose(0, 1).clone()

        # We need to take the bias as a common column here, also we do
        # not want to use indexed.
        return self.tile.forward(self.drift_readout_tensor, False,
                                 self.in_trans, self.out_trans, True)

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
        """

        if not from_reference or self.reference_combined_weights is None:
            self.reference_combined_weights = Tensor(self.tile.get_weights())

        self.programmed_weights, self.nu_drift_list = \
            self.rpu_config.noise_model.apply_programming_noise(
                self.reference_combined_weights)

        self.tile.set_weights(self.programmed_weights)

        if self.rpu_config.drift_compensation is not None:
            forward_output = self._forward_drift_readout_tensor(True)
            self.drift_baseline = self.rpu_config.drift_compensation.init_baseline(forward_output)

    @no_grad()
    def drift_weights(
            self,
            t_inference: float = 0.0
    ) -> None:
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
        """
        # pylint: disable=arguments-differ,arguments-renamed

        if self.programmed_weights is None:
            self.program_weights()

        drifted_weights = self.rpu_config.noise_model.apply_drift_noise(
            self.programmed_weights, self.nu_drift_list, t_inference)
        self.tile.set_weights(drifted_weights)

        if self.rpu_config.drift_compensation is not None:
            forward_output = self._forward_drift_readout_tensor()
            self.alpha = self.rpu_config.drift_compensation.apply(
                forward_output,
                self.drift_baseline).to(self.device)

    @no_grad()
    def post_forward(self, x_output: Tensor, dim: int, is_test: bool = False,
                     ctx: Any = None) -> Tensor:
        """Operations after the actual forward step for post processing """

        x_output = super().post_forward(x_output, dim, is_test, ctx)

        if not is_test or self.rpu_config.drift_compensation is None:
            return x_output

        # only do drift compensation in eval mode
        return x_output * self.alpha

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
        # Import `aihwkit.simulator.configs` items dynamically to avoid import cycles.
        # pylint: disable=import-outside-toplevel
        from aihwkit.simulator.configs.helpers import parameters_to_bindings
        from aihwkit.simulator.configs.utils import (
            WeightClipType, WeightModifierType, WeightRemapType
        )

        super().post_update_step()

        # TODO: make this a little nicer. Now each time bindings are
        # generated, which however has the advantage that parameters
        # could be changed-on-the-fly

        if self.rpu_config.clip.type != WeightClipType.NONE:
            weight_clip_params = parameters_to_bindings(self.rpu_config.clip)
            self.tile.clip_weights(weight_clip_params)

        if self.rpu_config.remap.type != WeightRemapType.NONE:
            weight_remap_params = parameters_to_bindings(self.rpu_config.remap)
            scales = self.get_scales()
            scales = self.tile.remap_weights(weight_remap_params, scales)
            self.set_scales(scales)

        # update the forward / backward modified weights here
        if (self.rpu_config.modifier.type != WeightModifierType.COPY or
                self.rpu_config.modifier.pdrop > 0.0):
            weight_modify_params = parameters_to_bindings(self.rpu_config.modifier)
            self.tile.modify_weights(weight_modify_params)

    def cuda(
            self,
            device: Optional[Union[torch_device, str, int]] = None
    ) -> 'BaseTile':
        """Return a copy of this tile in CUDA memory.

        Args:
            device: CUDA device

        Returns:
            Self with the underlying C++ tile moved to CUDA memory.

        Raises:
            CudaError: if the library has not been compiled with CUDA.
        """
        super().cuda(device)

        self.alpha = self.alpha.cuda(device)
        self.shared_weights.data = zeros(self.tile.get_x_size(),
                                         self.tile.get_d_size(),
                                         requires_grad=True).cuda(device)
        self.ensure_shared_weights()

        return self
