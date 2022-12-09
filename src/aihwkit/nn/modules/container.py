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

"""Analog Modules that contain children Modules."""

from typing import (
    Callable, Optional, Union, Any, NamedTuple, Tuple, TYPE_CHECKING, Generator
)
from collections import OrderedDict

from torch import device as torch_device
from torch.nn import Sequential

from aihwkit.exceptions import ModuleError, TileError
from aihwkit.nn.modules.base import AnalogModuleBase

if TYPE_CHECKING:
    from torch import Tensor  # pylint: disable=ungrouped-imports


class AnalogSequential(Sequential):
    """An analog-aware sequential container.

    Specialization of torch ``nn.Sequential`` with extra functionality for
    handling analog layers:

    * correct handling of ``.cuda()`` for children modules.
    * apply analog-specific functions to all its children (drift and program
      weights).

    Note:
        This class is recommended to be used in place of ``nn.Sequential`` in
        order to correctly propagate the actions to all the children analog
        layers. If using regular containers, please be aware that operations
        need to be applied manually to the children analog layers when needed.
    """
    # pylint: disable=abstract-method

    def _apply_to_analog(self, fn: Callable) -> 'AnalogSequential':
        """Apply a function to all the analog layers in this module.

        Args:
            fn: function to be applied.

        Returns:
            This module after the function has been applied.
        """
        for module in self.analog_modules():
            fn(module)

        return self

    def apply_to_analog_modules(self, fn: Callable) -> 'AnalogSequential':
        """Apply a function to all the analog modules.

        Args:
            fn: function to be applied.

        Returns:
            This module after the function has been applied.
        """
        for module in self.analog_modules():
            fn(module)

        return self

    def apply_to_analog_tiles(self, fn: Callable) -> 'AnalogSequential':
        """Apply a function to all the analog tiles of all layers in this module.

        Example::

            model.apply_to_analog_tiles(lambda tile: tile.reset())

        This would reset each analog tile in the whole DNN looping
        through all layers and all tiles that might exist in a
        particular layer.

        Args:
            fn: function to be applied.

        Returns:
            This module after the function has been applied.

        """
        for module in self.analog_modules():
            for analog_tiles in module.analog_tiles():
                fn(analog_tiles)
        return self

    def analog_modules(self) -> Generator[AnalogModuleBase, None, None]:
        """Generator over analog modules only"""
        for module in self.modules():
            if isinstance(module, AnalogModuleBase):
                yield module

    def named_analog_modules(self) -> Generator[Tuple[str, AnalogModuleBase], None, None]:
        """Generator over analog modules only"""
        for name, module in self.named_modules():
            if isinstance(module, AnalogModuleBase):
                yield name, module

    def cpu(
            self
    ) -> 'AnalogSequential':
        super().cpu()

        self._apply_to_analog(lambda m: m.cpu())

        return self

    def cuda(
            self,
            device: Optional[Union[torch_device, str, int]] = None
    ) -> 'AnalogSequential':
        super().cuda(device)

        self._apply_to_analog(lambda m: m.cuda(device))

        return self

    def to(
            self,
            device: Optional[Union[torch_device, str, int]] = None
    ) -> 'AnalogSequential':
        """Move and/or cast the parameters, buffers and analog tiles.

        Note:
            Please be aware that moving analog layers from GPU to CPU is
            currently not supported.

        Args:
            device: the desired device of the parameters, buffers and analog
                tiles in this module.

        Returns:
            This module in the specified device.
        """
        # pylint: disable=arguments-differ
        device = torch_device(device)

        super().to(device)

        if device.type == 'cuda':
            self._apply_to_analog(lambda m: m.cuda(device))
        elif device.type == 'cpu':
            self._apply_to_analog(lambda m: m.cpu())

        return self

    def get_analog_tile_device(self) -> Union[torch_device, str, int]:
        """ Return the devices used by the analog tiles.

        Returns:
            device: torch device

        Raises:
            TileError: in case the model is on non-unique devices
        """

        devices = []
        dev_name = []

        self._apply_to_analog(
            lambda mod: devices.extend(mod.get_analog_tile_devices()))

        for name in devices:
            dev_name.append(str(name))

        if len(set(dev_name)) > 1:
            raise TileError("Torch device is not unique")

        device = devices[0]

        return device

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
        # pylint: disable=protected-access
        self._apply_to_analog(lambda m: m._set_load_rpu_config_state(load_rpu_config))
        return super().load_state_dict(state_dict, strict)

    def prepare_for_ddp(self) -> None:
        """Adds ignores to avoid broadcasting the analog tile states in case of
        distributed training.

        Note:
            Call this function before the mode is converted with DDP.

        Important:
            Only InferenceTile supports DDP.

        Raises:

            ModuleError: In case analog tiles are used that do not
                support data-parallel model, ie. all analog training
                tiles.
        """
        # pylint: disable=attribute-defined-outside-init
        exclude_list = []
        for module in self.modules():
            if isinstance(module, AnalogModuleBase):
                for analog_tile in module.analog_tiles():
                    if analog_tile.shared_weights is None:
                        raise ModuleError("DDP is only supported with shared weights"
                                          "(e.g. InferenceTile)")
                exclude_list += [module.ANALOG_CTX_PREFIX, module.ANALOG_STATE_PREFIX]
        exclude_list = list(set(exclude_list))
        params = self.state_dict().keys()
        exclude_params = []
        for param in params:
            for word in exclude_list:
                if word in param and word not in exclude_params:
                    exclude_params.append(param)
                    break
        self._ddp_params_and_buffers_to_ignore = exclude_params

    def remap_analog_weights(self, weight_scaling_omega: Optional[float] = 1.0) -> None:
        """Remap the analog weights and set the digital out scales.

        Caution:

            This should typically *not* be called for analog training
            unless realistic_read_write is set. In this case, it would
            perform a full re-write of the weights. However,
            typically, this this method is intended to correct the
            mapping for hardware-aware trained models before doing the
            inference with programmed weights.

        Args:

            weight_scaling_omega: The optional value to remap the
                weight max to. If None it will take the value set
                initially in the ``RPUConfig.mapping``. Defaults to 1.0.

        """

        self._apply_to_analog(lambda m: m.remap_weights(
            weight_scaling_omega=weight_scaling_omega))

    def drift_analog_weights(self, t_inference: float = 0.0) -> None:
        """(Program) and drift all analog inference layers of a given model.

        Args:
            t_inference: assumed time of inference (in sec)

        Raises:
            ModuleError: if the layer is not in evaluation mode.
        """
        if self.training:
            raise ModuleError('drift_analog_weights can only be applied in '
                              'evaluation mode')

        self._apply_to_analog(lambda m: m.drift_analog_weights(t_inference))

    def program_analog_weights(self) -> None:
        """Program all analog inference layers of a given model.

        Raises:
            ModuleError: if the layer is not in evaluation mode.
        """
        if self.training:
            raise ModuleError('program_analog_weights can only be applied in '
                              'evaluation mode')

        self._apply_to_analog(lambda m: m.program_analog_weights())

    @classmethod
    def from_digital(cls, module: Sequential,  # pylint: disable=unused-argument
                     *args: Any,
                     **kwargs: Any) -> 'AnalogSequential':
        """Construct AnalogSequential in-place from Sequential."""
        return cls(OrderedDict(mod for mod in module.named_children()))
