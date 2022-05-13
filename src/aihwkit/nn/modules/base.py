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

"""Base class for analog Modules."""
from typing import (
    Any, Dict, List, Optional, Tuple, NamedTuple, Union,
    Generator, TYPE_CHECKING
)

from torch import Tensor, no_grad, ones, float32
from torch.nn import Module, Parameter
from torch import device as torch_device

from aihwkit.exceptions import ModuleError
from aihwkit.simulator.configs.configs import (
    FloatingPointRPUConfig, InferenceRPUConfig, SingleRPUConfig,
    UnitCellRPUConfig, DigitalRankUpdateRPUConfig
)
from aihwkit.simulator.configs.utils import MappingParameter
from aihwkit.simulator.tiles import InferenceTile
from aihwkit.optim.context import AnalogContext

if TYPE_CHECKING:
    from aihwkit.simulator.tiles import BaseTile
    from collections import OrderedDict

RPUConfigAlias = Union[FloatingPointRPUConfig, SingleRPUConfig,
                       UnitCellRPUConfig, InferenceRPUConfig,
                       DigitalRankUpdateRPUConfig]


class AnalogModuleBase(Module):
    """Base class for analog Modules.

    Base ``Module`` for analog layers that use analog tiles. When subclassing,
    please note:

    * the :meth:`_setup_tile()` method is expected to be called by the subclass
      constructor, and it does not only create a tile.
    * :meth:`register_analog_tile` needs to be called for each created analog tile
    * this module does *not* call torch's ``Module`` init as the child is
      likely again derived from Module
    * the ``weight`` and ``bias`` Parameters are not guaranteed to be in
      sync with the tile weights and biases during the lifetime of the instance,
      for performance reasons. The canonical way of reading and writing
      weights is via the :meth:`set_weights()` and :meth:`get_weights()` as opposed
      to using the attributes directly.
    * the ``BaseTile`` subclass that is created is retrieved from the
      ``rpu_config.tile_class`` attribute.

    Args:
        in_features: input vector size (number of columns).
        out_features: output vector size (number of rows).
        bias: whether to use a bias row on the analog tile or not.
        realistic_read_write: whether to enable realistic read/write
            for setting initial weights and during reading of the weights.
        mapping: Configuration of the hardware architecture (e.g. tile size).
    """
    # pylint: disable=abstract-method, too-many-instance-attributes
    ANALOG_CTX_PREFIX: str = 'analog_ctx_'
    ANALOG_SHARED_WEIGHT_PREFIX: str = 'analog_shared_weights_'
    ANALOG_STATE_PREFIX: str = 'analog_tile_state_'
    ANALOG_OUT_SCALING_ALPHA_PREFIX: str = 'analog_out_scaling_alpha_'

    def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool,
            realistic_read_write: bool = False,
            mapping: Optional[MappingParameter] = None,
    ) -> None:
        # pylint: disable=super-init-not-called
        self._analog_tile_counter = 0
        self._registered_helper_parameter = []  # type: list
        self._load_rpu_config = True

        if mapping is None:
            mapping = MappingParameter()

        self.use_bias = bias
        self.digital_bias = bias and mapping.digital_bias
        self.analog_bias = bias and not mapping.digital_bias
        self.realistic_read_write = realistic_read_write
        self.in_features = in_features
        self.out_features = out_features

    def register_analog_tile(self, tile: 'BaseTile', name: Optional[str] = None) -> None:
        """Register the analog context of the tile.

        Note:
            Needs to be called at the end init to register the tile
            for the analog optimizers.

        Args:
            tile: tile to register
            name: Optional tile name used as the parameter name
        """

        if name is None:
            name = str(self._analog_tile_counter)

        ctx_name = self.ANALOG_CTX_PREFIX + name

        if ctx_name not in self._registered_helper_parameter:
            self._registered_helper_parameter.append(ctx_name)
        self.register_parameter(ctx_name, tile.get_analog_ctx())

        if tile.shared_weights is not None:
            if not isinstance(tile.shared_weights, Parameter):
                tile.shared_weights = Parameter(tile.shared_weights)
            par_name = self.ANALOG_SHARED_WEIGHT_PREFIX + str(self._analog_tile_counter)
            self.register_parameter(par_name, tile.shared_weights)

            if par_name not in self._registered_helper_parameter:
                self._registered_helper_parameter.append(par_name)

        mapping = tile.rpu_config.mapping
        if mapping.learn_out_scaling_alpha:
            if tile.out_scaling_alpha is None:
                tile.out_scaling_alpha = Parameter(ones([1], device=tile.device, dtype=float32))
            elif not isinstance(tile.out_scaling_alpha, Parameter):
                tile.out_scaling_alpha = Parameter(tile.out_scaling_alpha)
            par_name = self.ANALOG_OUT_SCALING_ALPHA_PREFIX + str(self._analog_tile_counter)
            self.register_parameter(par_name, tile.out_scaling_alpha)

            if par_name not in self._registered_helper_parameter:
                self._registered_helper_parameter.append(par_name)

        self._analog_tile_counter += 1

    def unregister_parameter(self, param_name: str) -> None:
        """Unregister module parameter from parameters.

        Raises:
            ModuleError: In case parameter is not found
        """
        param = getattr(self, param_name, None)
        if not isinstance(param, Parameter):
            raise ModuleError(f"Cannot find parameter {param_name} to unregister")
        param_data = param.detach().clone()
        delattr(self, param_name)
        setattr(self, param_name, param_data)

    def analog_tiles(self) -> Generator['BaseTile', None, None]:
        """ Generator to loop over all registered analog tiles of the module """
        for param in self.parameters():
            if isinstance(param, AnalogContext):
                yield param.analog_tile

    def named_analog_tiles(self) -> Generator[Tuple[str, 'BaseTile'], None, None]:
        """ Generator to loop over all registered analog tiles of the module with names. """
        for name, param in self.named_parameters():
            if isinstance(param, AnalogContext):
                new_name = name.split(self.ANALOG_CTX_PREFIX)[-1]
                yield (new_name, param.analog_tile)

    def get_analog_tile_devices(self) -> List[Optional[Union[torch_device, str, int]]]:
        """ Return a list of the devices used by the analog tiles.

        Returns:
            List of torch devices
        """
        return [d.device for d in self.analog_tiles()]

    def analog_tile_count(self) -> int:
        """Return the number of registered tiles.

        Returns:
           Number of registered tiles

        """
        return getattr(self, '_analog_tile_counter', 0)

    def _setup_tile(
            self,
            rpu_config: RPUConfigAlias,
    ) -> 'BaseTile':
        """Create a single analog tile with the given RPU configuration.

        Create an analog tile to be used for the basis of this layer
        operations, while using the attributes ``(in_features,
        out_features, bias)`` given to this instance during init.

        After tile creation, the tile needs to be registered using
        :meth:`register_analog_tile`.

        Args:
            rpu_config: resistive processing unit configuration.

        Returns:
            An analog tile with the requested parameters.

        """
        # pylint: disable=protected-access

        # Create the tile.
        return rpu_config.tile_class(self.out_features, self.in_features, rpu_config,
                                     bias=self.analog_bias)

    def set_weights(
            self,
            weight: Tensor,
            bias: Optional[Tensor] = None,
            force_exact: bool = False,
            remap_weights: bool = True,
            weight_scaling_omega: float = None
    ) -> None:
        """Set the weight (and bias) values with given tensors.

        This uses an realistic write if the property ``realistic_read_write``
        of the layer is set, unless it is overwritten by ``force_exact``.

        If ``weight_scaling_omega`` is larger than 0, the weights are set in a
        scaled manner (assuming a digital output scale). See
        :meth:`~aihwkit.simulator.tiles.base.BaseTile.set_weights_scaled`
        for details.

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
            remap_weights: Whether to rescale the given weight matrix
                and populate the digital output scaling factors as
                specified in the configuration
                :class:`~aihwkit.configs.utils.MappingParameter`. A
                new ``weight_scaling_omega`` can be given. Note that
                this will overwrite the existing digital out scaling
                factors.
            weight_scaling_omega: The weight scaling omega factor (see
                :class:`~aihwkit.configs.utils.MappingParameter`). If
                given explicitly here, it will overwrite the value in
                the mapping field.

        Raises:
            ModuleError: in case of multiple defined analog tiles in the module

        """
        shape = [self.out_features, self.in_features]
        weight = weight.clone().reshape(shape)

        realistic = self.realistic_read_write and not force_exact

        analog_tiles = list(self.analog_tiles())
        if len(analog_tiles) != 1:
            raise ModuleError("AnalogModuleBase.set_weights only supports a single tile.")
        analog_tile = analog_tiles[0]

        if remap_weights:
            omega = weight_scaling_omega
            if omega is None:
                omega = analog_tile.rpu_config.mapping.weight_scaling_omega

            analog_tile.set_weights_scaled(
                weight, bias if self.analog_bias else None,
                realistic=realistic,
                weight_scaling_omega=omega)
        else:
            analog_tile.set_weights(weight, bias if self.analog_bias else None,
                                    realistic=realistic)

        if bias is not None and self.digital_bias:
            with no_grad():
                self.bias.data[:] = bias[:]

        self._sync_weights_from_tile()

    def get_weights(
            self,
            force_exact: bool = False,
            apply_out_scales: bool = True,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Get the weight (and bias) tensors.

        This uses an realistic read if the property ``realistic_read_write`` of
        the layer is set, unless it is overwritten by ``force_exact``. It
        scales the analog weights by the digital alpha scale if
        ``weight_scaling_omega`` is positive (see
        :meth:`~aihwkit.simulator.tiles.base.BaseTile.get_weights_scaled`).

        Note:
            This is the recommended way for setting the weight/bias matrix from
            the analog tile, as it will correctly fetch the weights from the
            internal memory. Accessing ``self.weight`` and ``self.bias`` might
            yield wrong results as they are not always in sync with the
            analog tile library, for performance reasons.

        Args:
            force_exact: Forces an exact read to the analog tiles

            apply_out_scales: Whether to return the weights with the
                (digital) output scaling factors applied. Note the
                "logical" weights of the layer which the DNN is
                effectively using are those with the output scales
                applied. If ``apply_out_scales`` is set to False, then
                only the weight values that is programmed onto the
                crossbar array are returned, without applying the
                digital scales.

        Returns:
            tuple: weight matrix, bias vector

        Raises:
            ModuleError: in case of multiple defined analog tiles in the module

        """
        analog_tiles = list(self.analog_tiles())
        if len(analog_tiles) != 1:
            raise ModuleError("AnalogModuleBase.get_weights only supports a single tile.")
        analog_tile = analog_tiles[0]

        realistic = self.realistic_read_write and not force_exact
        if apply_out_scales:
            weight, analog_bias = analog_tile.get_weights_scaled(realistic=realistic)
        else:
            weight, analog_bias = analog_tile.get_weights(realistic=realistic)

        digital_bias = None
        if self.digital_bias:
            with no_grad():
                digital_bias = self.bias.data.clone().detach().cpu()

        if (digital_bias is not None) and (analog_bias is not None):
            bias = digital_bias + analog_bias
        elif digital_bias is not None:
            bias = digital_bias
        else:
            bias = analog_bias
        return weight, bias

    def _sync_weights_from_tile(self) -> None:
        """Update the layer weight and bias from the values on the analog tile.

        Update the ``self.weight`` and ``self.bias`` Parameters with an
        exact copy of the internal analog tile weights.
        """
        tile_weight, tile_bias = self.get_weights(force_exact=True)  # type: Tuple[Tensor, Tensor]

        self.weight.data[:] = tile_weight.reshape(self.weight.shape)
        if self.analog_bias:
            with no_grad():
                self.bias.data[:] = tile_bias.reshape(self.bias.shape)

    def _sync_weights_to_tile(self) -> None:
        """Update the tile values from the layer weights and bias.

        Update the internal tile weights with an exact copy of the values of
        the ``self.weight`` and ``self.bias`` Parameters.
        """
        self.set_weights(self.weight, self.bias if self.analog_bias else None,
                         force_exact=True)

    def _set_load_rpu_config_state(self, load_rpu_config: bool = True) -> None:
        self._load_rpu_config = load_rpu_config

    def load_state_dict(self,  # pylint: disable=arguments-differ
                        state_dict: 'OrderedDict[str, Tensor]',
                        strict: bool = True,
                        load_rpu_config: bool = True) -> NamedTuple:
        """Specializes torch's ``load_state_dict`` to add a flag whether to
        load the RPU config from the saved state.

        Args:
            state_dict: see torch's ``load_state_dict``
            strict: see torch's ``load_state_dict``
            load_rpu_config: Whether to load the saved RPU
                config or use the current RPU config of the model.

                Caution:

                    If ``load_rpu_config=False`` the RPU config can
                    be changed from the stored model. However, the user has to
                    make sure that the changed RPU config makes sense.

                    For instance, changing the device type might
                    change the expected fields in the hidden
                    parameters and result in an error.
        Returns:
            see torch's ``load_state_dict``

        Raises: ModuleError: in case the rpu_config class mismatches
            for ``load_rpu_config=False``.
        """
        self._set_load_rpu_config_state(load_rpu_config)
        return super().load_state_dict(state_dict, strict)

    def _load_from_state_dict(
            self,
            state_dict: Dict,
            prefix: str,
            local_metadata: Dict,
            strict: bool,
            missing_keys: List[str],
            unexpected_keys: List[str],
            error_msgs: List[str]) -> None:
        """Copy parameters and buffers from `state_dict` into only this
        module, but not its descendants.

        This method is a specialization of ``Module._load_from_state_dict``
        that takes into account the extra ``analog_tile_state`` key used by
        analog layers.

        Raises:
            ModuleError: in case the rpu_config class mismatches.
        """

        for name, analog_tile in list(self.named_analog_tiles()):
            key = prefix + self.ANALOG_STATE_PREFIX + name
            if key not in state_dict:  # legacy
                key = prefix + 'analog_tile_state'

            if key in state_dict:
                analog_state = state_dict.pop(key).copy()

                if not self._load_rpu_config:
                    if analog_tile.rpu_config.__class__ != analog_state['rpu_config'].__class__:
                        raise ModuleError("RPU config mismatch during loading: "
                                          "Tried to replace "
                                          f"{analog_state['rpu_config'].__class__.__name__} "
                                          f"with {analog_tile.rpu_config.__class__.__name__}")
                    analog_state['rpu_config'] = analog_tile.rpu_config
                analog_tile.__setstate__(analog_state)

            elif strict:
                missing_keys.append(key)

        # update the weight / analog bias (not saved explicitly)
        self._sync_weights_from_tile()

        # remove helper parameters.
        rm_keys = []
        for par_name in self._registered_helper_parameter:
            key = prefix + par_name
            if key in state_dict:
                state_dict.pop(key)
                rm_keys.append(key)

        super()._load_from_state_dict(
           state_dict, prefix, local_metadata, strict, missing_keys,
           unexpected_keys, error_msgs)

        # remove the missing keys of the helper parameters
        for key in rm_keys:
            missing_keys.remove(key)

    def state_dict(
            self,
            destination: Any = None,
            prefix: str = '',
            keep_vars: bool = False
    ) -> Dict:
        """Return a dictionary containing a whole state of the module."""
        self._sync_weights_from_tile()

        current_state = super().state_dict(destination, prefix, keep_vars)

        for name, analog_tile in self.named_analog_tiles():
            analog_state = analog_tile.__getstate__()
            analog_state_name = prefix + self.ANALOG_STATE_PREFIX + name
            current_state[analog_state_name] = analog_state

        return current_state

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
        for analog_tile in self.analog_tiles():
            if isinstance(analog_tile, InferenceTile):
                analog_tile.drift_weights(t_inference)

    def program_analog_weights(self) -> None:
        """Program the analog weights.

        Raises:
            ModuleError: if the layer is not in evaluation mode.
        """
        if self.training:
            raise ModuleError('program_analog_weights can only be applied in '
                              'evaluation mode')
        for analog_tile in self.analog_tiles():
            if isinstance(analog_tile, InferenceTile):
                analog_tile.program_weights()

    def extra_repr(self) -> str:
        """Set the extra representation of the module.

        Returns:
            A string with the extra representation.
        """
        output = super().extra_repr()
        if self.realistic_read_write:
            output += ', realistic_read_write={}'.format(self.realistic_read_write)
        if self.analog_bias:
            output += ', analog bias'
        if self.digital_bias:
            output += ', digital bias'

        return output
