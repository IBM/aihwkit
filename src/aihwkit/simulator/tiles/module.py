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

"""Tile module base."""

# pylint: disable=too-few-public-methods, abstract-method, bad-super-call

from typing import Dict, List, Optional, Union, Any, Callable, NamedTuple
from collections import OrderedDict

from torch import Tensor, dtype
from torch.nn import Module
from torch import device as torch_device

from aihwkit.exceptions import TileModuleError
from aihwkit.simulator.tiles.base import AnalogTileStateNames, BaseTile, TileModuleBase
from aihwkit.optim.context import AnalogContext


class TileModule(Module, TileModuleBase):
    """Class of all tiles based on ``torch.Module``.

    A TileModule class inherits from three base classes::

        class MyTile(TileModule, MyTile(BaseTile), MySimulatorTileWrapper)

    Assuming this structure, here utility methods are defined that
    help to go through all these classes, such as ``cuda``.
    """

    def __init__(self) -> None:
        super().__init__()

        self.load_rpu_config = True  # whether to load the rpu_config from the state dict
        self.strict_rpu_config_check = True

        self.use_state_dict_hooks = False
        if not hasattr(self, "_register_state_dict_hook"):
            self.use_state_dict_hooks = False

        if self.use_state_dict_hooks:
            self._register_state_dict_hook(TileModule._state_dict_hook)
            self._register_load_state_dict_pre_hook(
                TileModule._load_state_dict_pre_hook, with_module=True
            )
            self.register_load_state_dict_post_hook(TileModule._load_state_dict_post_hook)

    def set_load_rpu_config_state(
        self, load_rpu_config: Optional[bool], strict_rpu_config_check: Optional[bool] = None
    ) -> None:
        """Sets the behavior of when using ``load_state_dict``.

        Caution:
            If ``load_rpu_config=False`` the RPU config can
            be changed from the stored model. However, the user has to
            make sure that the changed RPU config makes sense.

            For instance, changing the device type might
            change the expected fields in the hidden
            parameters and result in an error.

        Args:
            load_rpu_config: Whether to load the saved RPU
            config or use the current RPU config of the model.


            strict_rpu_config_check: Whether to check and throw an
                error if the current ``rpu_config`` is not of the same
                class type when setting ``load_rpu_config`` to
                False. In case of ``False`` the user has to make sure
                that the ``rpu_config`` are compatible.
        """
        if load_rpu_config is not None:
            self.load_rpu_config = load_rpu_config
        if strict_rpu_config_check is not None:
            self.strict_rpu_config_check = strict_rpu_config_check

    def extra_repr(self) -> str:
        """Set the extra representation of the module.

        Returns:
            A string with the extra representation.
        """
        if not hasattr(self, "tile") or isinstance(self.tile, Module):
            return ""
        return self.tile.get_brief_info().rstrip()

    def __getstate__(self) -> Dict:
        # pylint: disable=no-member
        state = {}
        if hasattr(super(Module, self), "__getstate__"):
            state = super(Module, self).__getstate__()
        # if isinstance(self, BaseTile) and hasattr(super(BaseTile, self), '__getstate__'):
        #     state.update(super(BaseTile, self).__getstate__())  # type: ignore
        return state

    def __setstate__(self, state: Dict) -> None:
        # pylint: disable=no-member

        if hasattr(super(Module, self), "__setstate__"):
            # The TileWrapper is handling all the attributes
            super(Module, self).__setstate__(state)
        else:
            Module.__setstate__(self, state)

        # update parameter IDs
        for name in self._parameters:  # type: ignore
            self._parameters[name] = getattr(self, name)  # type: ignore

        for name in self._buffers:
            self._buffers[name] = getattr(self, name)

        for name in self._modules:
            self._modules[name] = getattr(self, name)

    def _apply_without_context(self, fn: Callable) -> None:
        """Loops through parameters (excluding the tile module's own AnalogContext)."""
        _parameters = self._parameters  # type: ignore
        self._parameters = OrderedDict()

        for name, param in _parameters.items():
            if isinstance(param, AnalogContext):
                if param.analog_tile == self:
                    continue
            self._parameters[name] = param
        Module._apply(self, fn)
        self._parameters = _parameters

    def _apply(self, fn: Callable) -> None:
        """Delegates to the module level.

        This avoids looping through parameters which would cause
        unlimited recursions.

        Raises
            TileModuleError in case the fucntion does not exist on Module level
        """
        try:
            fn(self)
        except Exception as exception:
            raise TileModuleError(
                "Applied function is not supported for TileModule: {}".format(str(exception))
            ) from exception

    def is_floating_point(self) -> bool:
        """Dummy for .to to work."""
        return True

    def cuda(self, device: Optional[Union[torch_device, str, int]] = None) -> "TileModule":
        """Return a copy of this tile in CUDA memory.

        Args:
            device: CUDA device

        Returns:
            Self with the underlying C++ tile moved to CUDA memory.

        Raises:
            CudaError: if the library has not been compiled with CUDA.
        """
        # handle the SimulatorTileWrapper
        if hasattr(super(Module, self), "cuda"):
            super(Module, self).cuda(device)
        if isinstance(self, BaseTile) and hasattr(super(BaseTile, self), "cuda"):
            super(BaseTile, self).cuda(device)  # type: ignore

        # at the end. shared weight might be updated above which might
        # yeild issues if the tile is not first updated
        self._apply_without_context(lambda t: t.cuda(device))
        return self

    def cpu(self) -> "TileModule":
        """Return a copy of this tile in CUDA memory.

        Returns:
            Self with the underlying C++ tile moved to CUDA memory.

        Raises:
            CudaError: if the library has not been compiled with CUDA.
        """
        if hasattr(super(Module, self), "cpu"):
            super(Module, self).cpu()
        if isinstance(self, BaseTile) and hasattr(super(BaseTile, self), "cpu"):
            super(BaseTile, self).cpu()  # type: ignore

        self._apply_without_context(lambda t: t.cpu())
        return self

    def to(self, *args: Any, **kwargs: Any) -> "TileModule":
        """Move analog tile module to a device.

        Note:
            Please be aware that moving analog tiles from GPU to CPU is
            currently not supported.

        Caution:
            Other tensor conversions than moving the device to CUDA,
            such as changing the data type are not supported for analog
            tiles and will be simply ignored.

        Returns:
            This module in the specified device.
        """

        device = None
        if "device" in kwargs:
            device = kwargs["device"]
        elif len(args) > 0 and not isinstance(args[0], (Tensor, dtype)):
            device = torch_device(args[0])

        if device is not None:
            device = torch_device(device)
            if device.type == "cuda":
                self.cuda(device)
            else:
                self.cpu()

        self._apply_without_context(lambda t: t.to(*args, **kwargs))
        return self

    @staticmethod
    def get_analog_state_name(prefix: str) -> str:
        """Returns the analog state name."""
        return prefix + AnalogTileStateNames.ANALOG_STATE_NAME

    @staticmethod
    def _state_dict_hook(
        analog_tile: "TileModule", state_dict: Dict, prefix: str, local_metadata: Dict
    ) -> None:
        # pylint: disable=unused-argument

        analog_state_name = TileModule.get_analog_state_name(prefix)
        analog_state = analog_tile.__getstate__()

        # don't return those hidden Module attributes:
        for key in list(analog_state.keys()).copy():
            if key.startswith("_"):
                analog_state.pop(key, None)

        state_dict[analog_state_name] = analog_state

    def state_dict(  # pylint: disable=arguments-differ
        self, destination: Dict, prefix: str = "", keep_vars: bool = False
    ) -> None:
        """Overload to add the hooks for pytorch < 1.12."""

        Module.state_dict(self, destination=destination, prefix=prefix, keep_vars=keep_vars)

        if not self.use_state_dict_hooks:
            TileModule._state_dict_hook(self, destination, prefix, {})

    def _load_from_state_dict(
        self,
        state_dict: Dict,
        prefix: str,
        local_metadata: Dict,
        strict: bool,
        missing_keys: List[str],
        unexpected_keys: List[str],
        error_msgs: List[str],
    ) -> None:
        r"""Overloads :meth:`~torch.nn.Module.load_state_dict` to
        include state dict hooks prior to pytorch 1.12
        """

        if not self.use_state_dict_hooks:
            # use the hooks of pytorch >= 1.12 call the hooks
            # explicitely
            TileModule._load_state_dict_pre_hook(
                self,
                state_dict,
                prefix,
                local_metadata,
                strict,
                missing_keys,
                unexpected_keys,
                error_msgs,
            )

        Module._load_from_state_dict(
            self,
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

        if not self.use_state_dict_hooks:
            TileModule._load_state_dict_post_hook(
                self, (missing_keys, unexpected_keys)  # type: ignore
            )

    @staticmethod
    def _load_state_dict_post_hook(
        analog_tile: "TileModule", incompatible_keys: NamedTuple
    ) -> None:
        """Handle some cross-load situation. These are handled on the
        __setstate__ level."""
        # pylint: disable=unused-argument
        missing_keys, unexpected_keys = incompatible_keys
        for key in missing_keys.copy():
            if AnalogTileStateNames.SHARED_WEIGHTS in key:
                missing_keys.remove(key)
        for key in unexpected_keys.copy():
            if ".tile." in key:
                # this needs to be handled in the analog_module level
                unexpected_keys.remove(key)

    @staticmethod
    def _load_state_dict_pre_hook(
        analog_tile: "TileModule",
        state_dict: Dict,
        prefix: str,
        local_metadata: Dict,
        strict: bool,
        missing_keys: List[str],
        unexpected_keys: List[str],
        error_msgs: List[str],
    ) -> None:
        # pylint: disable=unused-argument
        analog_state_name = TileModule.get_analog_state_name(prefix)

        if analog_state_name not in state_dict:
            missing_keys.append(analog_state_name)
            return

        # handle shared weights here (need to be transposed on CUDA)
        # Torch will throw an error if shapes are not compatible
        shared_weights_key = prefix + AnalogTileStateNames.SHARED_WEIGHTS
        if shared_weights_key in state_dict and isinstance(state_dict[shared_weights_key], Tensor):
            state_dict.pop(shared_weights_key, None)

        analog_state = state_dict.pop(analog_state_name).copy()
        if not analog_tile.load_rpu_config:
            if analog_tile.strict_rpu_config_check:
                if not isinstance(
                    analog_tile.rpu_config, type(analog_state["rpu_config"])
                ) and not isinstance(analog_state["rpu_config"], type(analog_tile.rpu_config)):
                    raise TileModuleError(
                        "RPU config mismatch during loading: "
                        "Tried to replace "
                        f"{analog_state['rpu_config'].__class__.__name__} "
                        f"with {analog_tile.rpu_config.__class__.__name__}"
                    )

            if hasattr(analog_state["rpu_config"], "mapping"):
                old_mapping = analog_state["rpu_config"].mapping
                new_mapping = analog_tile.rpu_config.mapping
                if (
                    old_mapping.max_input_size != new_mapping.max_input_size
                    or old_mapping.max_output_size != new_mapping.max_output_size
                    or old_mapping.digital_bias != new_mapping.digital_bias
                    or (old_mapping.out_scaling_columnwise != new_mapping.out_scaling_columnwise)
                ):
                    raise TileModuleError(
                        "MappingParameter mismatch during loading: "
                        "Tried to replace "
                        f"{old_mapping} "
                        f"with {new_mapping}"
                    )
            analog_state["rpu_config"] = analog_tile.rpu_config
        analog_tile.__setstate__(analog_state)
