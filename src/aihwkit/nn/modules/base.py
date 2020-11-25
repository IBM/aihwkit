# -*- coding: utf-8 -*-

# (C) Copyright 2020 IBM. All Rights Reserved.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Base class for analog Modules."""

from typing import Any, Dict, List, Optional, Tuple, Union

from torch import device as torch_device
from torch import Tensor
from torch.nn import Module

from aihwkit.exceptions import ModuleError
from aihwkit.simulator.configs import (
    FloatingPointRPUConfig, InferenceRPUConfig, SingleRPUConfig,
    UnitCellRPUConfig
)
from aihwkit.simulator.tiles import (
    AnalogTile, BaseTile, FloatingPointTile, InferenceTile
)


class AnalogModuleBase(Module):
    """Base class for analog Modules.

    Base ``Module`` for analog layers that use analog tiles. When subclassing,
    please note:

    * the ``_setup_tile()`` method is expected to be called by the subclass
      constructor, and it does not only create a tile, but also sets some
      instance attributes that are needed by the analog features (optimizer
      and others).
    * the ``weight`` and ``bias`` Parameters are not guaranteed to be in
      sync with the tile weights and biases during the lifetime of the instance,
      for performance reasons. The canonical way of reading and writing
      weights is via the ``set_weights()`` and ``get_weights()`` as opposed
      to using the attributes directly.
    """
    # pylint: disable=abstract-method

    TILE_CLASS_FLOATING_POINT: Any = FloatingPointTile
    TILE_CLASS_ANALOG: Any = AnalogTile
    TILE_CLASS_INFERENCE: Any = InferenceTile

    def _setup_tile(
            self,
            in_features: int,
            out_features: int,
            bias: bool,
            rpu_config: Optional[
                Union[FloatingPointRPUConfig, SingleRPUConfig,
                      UnitCellRPUConfig, InferenceRPUConfig]] = None,
            realistic_read_write: bool = False
    ) -> BaseTile:
        """Create an analog tile and setup this layer for using it.

        Create an analog tile to be used for the basis of this layer operations,
        and setup additional attributes of this instance that are needed for
        using the analog tile.

        Note:
            This method also sets the following attributes, which are assumed
            to be set by the rest of the methods:
            * ``self.use_bias``
            * ``self.realistic_read_write``
            * ``self.in_features``
            * ``self.out_features``

        Args:
            in_features: input vector size (number of columns).
            out_features: output vector size (number of rows).
            rpu_config: resistive processing unit configuration.
            bias: whether to use a bias row on the analog tile or not.
            realistic_read_write: whether to enable realistic read/write
               for setting initial weights and read out of weights.

        Returns:
            An analog tile with the requested parameters.
        """
        # pylint: disable=attribute-defined-outside-init
        # Default to constant step device if not provided.
        if not rpu_config:
            rpu_config = SingleRPUConfig()

        # Setup the analog-related attributes of this instance.
        self.use_bias = bias
        self.realistic_read_write = realistic_read_write
        self.in_features = in_features
        self.out_features = out_features

        # Create the tile.
        if isinstance(rpu_config, FloatingPointRPUConfig):
            tile_class = self.TILE_CLASS_FLOATING_POINT
        elif isinstance(rpu_config, InferenceRPUConfig):
            tile_class = self.TILE_CLASS_INFERENCE
        else:
            tile_class = self.TILE_CLASS_ANALOG  # type: ignore

        return tile_class(
            out_features, in_features, rpu_config, bias=bias  # type: ignore
        )

    def set_weights(
            self,
            weight: Tensor,
            bias: Optional[Tensor] = None,
            force_exact: bool = False
    ) -> None:
        """Sets the weight (and bias) with given Tensors.

        This uses an realistic write if the property ``realistic_read_write``
        of the layer is set, unless it is overwritten by ``force_exact``.

        Note:
            This is the recommended way for setting the weight/bias matrix of
            the analog tile, as it will correctly store the weights into the
            internal memory. Directly writing to ``self.weight`` and
            ``self.bias`` might yield wrong results as they are not always in
            sync with the analog tile Parameters, for performance reasons.

        Args:
            weight: weight matrix
            bias: bias vector
            force_exact: forces an exact write to the analog tiles
        """
        shape = [self.out_features, self.in_features]
        weight = weight.clone().reshape(shape)

        realistic = self.realistic_read_write and not force_exact
        self.analog_tile.set_weights(weight, bias, realistic=realistic)

        self._sync_weights_from_tile()

    def get_weights(
            self,
            force_exact: bool = False
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Get the weight (and bias) tensors.

        This uses an realistic read if the property ``realistic_read_write`` of
        the layer is set, unless it is overwritten by ``force_exact``.

        Note:
            This is the recommended way for setting the weight/bias matrix from
            the analog tile, as it will correctly fetch the weights from the
            internal memory. Accessing ``self.weight`` and ``self.bias`` might
            yield wrong results as they are not always in sync with the
            analog tile library, for performance reasons.

        Args:
            force_exact: forces an exact read to the analog tiles

        Returns:
            tuple: weight matrix, bias vector
        """
        realistic = self.realistic_read_write and not force_exact
        return self.analog_tile.get_weights(realistic=realistic)

    def _sync_weights_from_tile(self) -> None:
        """Update the layer weight and bias from the values on the analog tile.

        Update the ``self.weight`` and ``self.bias`` Parameters with an
        exact copy of the internal analog tile weights.
        """
        tile_weight, tile_bias = self.get_weights(force_exact=True)
        self.weight.data[:] = tile_weight.reshape(self.weight.shape)
        if self.use_bias:
            self.bias.data[:] = tile_bias.reshape(self.bias.shape)  # type: ignore

    def _sync_weights_to_tile(self) -> None:
        """Update the tile values from the layer weights and bias.

        Update the internal tile weights with an exact copy of the values of
        the ``self.weight`` and ``self.bias`` Parameters.
        """
        self.set_weights(self.weight, self.bias, force_exact=True)

    def _load_from_state_dict(
            self,
            state_dict: Dict,
            prefix: str,
            local_metadata: Dict,
            strict: bool,
            missing_keys: List[str],
            unexpected_keys: List[str],
            error_msgs: List[str]) -> None:
        """Copies parameters and buffers from `state_dict` into only this
        module, but not its descendants.

        This method is a specialization of ``Module._load_from_state_dict``
        that takes into account the extra ``analog_tile_state`` key used by
        analog layers.
        """
        key = '{}analog_tile_state'.format(prefix)
        if key in state_dict:
            analog_state = state_dict.pop(key)
            self.analog_tile.__setstate__(analog_state)
        elif strict:
            missing_keys.append(key)

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys,
            unexpected_keys, error_msgs)

    def state_dict(
            self,
            destination: Any = None,
            prefix: str = '',
            keep_vars: bool = False
    ) -> Dict:
        """Returns a dictionary containing a whole state of the module."""
        self._sync_weights_from_tile()
        # TODO: this will also pickle the resistive device. Problematic?  we
        # could also just save hidden_pars and weights. However, in any case the
        # state_dict will not reflect the model.parameters() any more, which
        # might get tricky. In any case, internal hidden weights need to be
        # saved to reconstruct a meaningful analog tile

        analog_state = self.analog_tile.__getstate__()
        current_state = super().state_dict(destination, prefix, keep_vars)
        current_state['{}analog_tile_state'.format(prefix)] = analog_state
        return current_state

    def cuda(
            self,
            device: Optional[Union[torch_device, str, int]] = None
    ) -> 'AnalogModuleBase':
        """Moves all model parameters, buffers and tiles to the GPU.

        This also makes associated parameters and buffers different objects. So
        it should be called before constructing optimizer if the module will
        live on GPU while being optimized.

        Arguments:
            device (int, optional): if specified, all parameters will be
                copied to that GPU device

        Returns:
            This layer with its parameters, buffers and tiles in GPU.
        """
        # pylint: disable=attribute-defined-outside-init
        # Note: this needs to be an in-place function, not a copy
        super().cuda(device)
        self.analog_tile = self.analog_tile.cuda(device)  # type: BaseTile
        self.set_weights(self.weight, self.bias)
        return self

    def drift_analog_weights(self, t_inference: float = 0.0) -> None:
        """(Program) and drift the analog weights.

        Args:
            t_inference: assumed time of inference (in sec)

        Raises:
            ModuleError: if the layer is not in evaluation mode.
        """
        if self.training:
            raise ModuleError('drift_analog_weights can only be applied in '
                              'evaluation mode')

        if isinstance(self.analog_tile, InferenceTile):
            self.analog_tile.drift_weights(t_inference)

    def program_analog_weights(self) -> None:
        """Program the analog weights.

        Raises:
            ModuleError: if the layer is not in evaluation mode.
        """
        if self.training:
            raise ModuleError('program_analog_weights can only be applied in '
                              'evaluation mode')

        if isinstance(self.analog_tile, InferenceTile):
            self.analog_tile.program_weights()
