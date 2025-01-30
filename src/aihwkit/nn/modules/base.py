# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Base class for adding functionality to analog layers."""
from typing import Any, List, Optional, Tuple, NamedTuple, Union, Generator, Callable, TYPE_CHECKING
from collections import OrderedDict

from torch import Tensor
from torch.nn import Parameter
from torch import device as torch_device

from aihwkit.exceptions import ModuleError
from aihwkit.simulator.tiles.module import TileModule
from aihwkit.simulator.tiles.inference import InferenceTileWithPeriphery
from aihwkit.simulator.tiles.base import AnalogTileStateNames
from aihwkit.simulator.parameters.base import RPUConfigBase

if TYPE_CHECKING:
    from aihwkit.inference.noise.base import BaseNoiseModel


class AnalogLayerBase:
    """Mixin that adds functionality on the layer level.

    In general, the defined methods will be looped for all analog tile
    modules and delegate the function.
    """

    IS_CONTAINER: bool = False
    """Class constant indicating whether sub-layers exist or whether
    this layer is a leave node (that is only having tile modules)"""

    # pylint: disable=no-member

    def apply_to_analog_layers(self, fn: Callable) -> "AnalogLayerBase":
        """Apply a function to all the analog layers.

        Note:
            Here analog layers are all sub modules of the current
            module that derive from ``AnalogLayerBase`` (such as
            ``AnalogLinear``) _except_ ``AnalogSequential``.

        Args:
            fn: function to be applied.

        Returns:
            This module after the function has been applied.

        """
        for _, module in self.named_analog_layers():
            fn(module)

        return self

    def apply_to_analog_tiles(self, fn: Callable) -> "AnalogLayerBase":
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
        for _, analog_tile in self.named_analog_tiles():
            fn(analog_tile)
        return self

    def analog_layers(self) -> Generator["AnalogLayerBase", None, None]:
        """Generator over analog layers only.

        Note:
            Here analog layers are all sub modules of the current module that
            derive from ``AnalogLayerBase`` (such as ``AnalogLinear``)
            _except_ ``AnalogSequential``.
        """
        for _, layer in self.named_analog_layers():  # type: ignore
            yield layer

    def named_analog_layers(self) -> Generator[Tuple[str, "AnalogLayerBase"], None, None]:
        """Generator over analog layers only.

        Note:
            Here analog layers are all sub-modules of the current
            module that derive from ``AnalogLayerBase`` (such as
            ``AnalogLinear``) _except_ those that are containers
            (`IS_CONTAINER=True`) such as ``AnalogSequential``.

        """
        for name, layer in self.named_modules():  # type: ignore
            if isinstance(layer, AnalogLayerBase) and not layer.IS_CONTAINER:
                yield name, layer

    def analog_modules(self) -> Generator["AnalogLayerBase", None, None]:
        """Generator over analog layers and containers.

        Note:
            Similar to :meth:`analog_layers` but also returning all
            analog containers
        """
        for layer in self.modules():  # type: ignore
            if isinstance(layer, AnalogLayerBase):
                yield layer

    def named_analog_modules(self) -> Generator[Tuple[str, "AnalogLayerBase"], None, None]:
        """Generator over analog layers.

        Note:
            Similar to :meth:`named_analog_layers` but also returning all
            analog containers
        """
        for name, layer in self.named_modules():  # type: ignore
            if isinstance(layer, AnalogLayerBase):
                yield name, layer

    def analog_tile_count(self) -> int:
        """Returns the number of tiles.

        Caution:

             This is a static number only counted when first called.

        Returns:
             Number of AnalogTileModules in this layer.
        """
        # pylint: disable=attribute-defined-outside-init
        if not hasattr(self, "_analog_tile_counter"):
            self._analog_tile_counter = len(list(self.analog_tiles()))
        return self._analog_tile_counter

    def analog_tiles(self) -> Generator["TileModule", None, None]:
        """Generator to loop over all registered analog tiles of the module"""
        for _, tile in self.named_analog_tiles():
            yield tile

    def named_analog_tiles(self) -> Generator[Tuple[str, "TileModule"], None, None]:
        """Generator to loop over all registered analog tiles of the module with names."""
        for name, module in self.named_modules():  # type: ignore
            if isinstance(module, TileModule):
                yield (name, module)

    def unregister_parameter(self, param_name: str) -> None:
        """Unregister module parameter from parameters.

        Raises:
            ModuleError: In case parameter is not found.
        """
        param = getattr(self, param_name, None)
        if not isinstance(param, Parameter):
            raise ModuleError(f"Cannot find parameter {param_name} to unregister")
        delattr(self, param_name)
        setattr(self, param_name, None)

    def get_analog_tile_devices(self) -> List[Optional[Union[torch_device, str, int]]]:
        """Return a list of the devices used by the analog tiles.

        Returns:
            List of torch devices.
        """
        return [d.device for d in self.analog_tiles()]

    def set_weights(self, weight: Tensor, bias: Optional[Tensor] = None, **kwargs: Any) -> None:
        """Set the weight (and bias) tensors to the analog crossbar.

        Args:
            weight: the weight tensor
            bias: the bias tensor is available
            **kwargs: see tile level,
                e.g. :meth:`~aihwkit.simulator.tiles.analog.AnalogTile.set_weights`

        Raises:
            ModuleError: if not of type TileModule.
        """
        if hasattr(self, "analog_module"):
            return self.analog_module.set_weights(weight, bias, **kwargs)
        raise ModuleError(f"set_weights not implemented for {type(self).__name__} ")

    def get_weights(self, **kwargs: Any) -> Tuple[Tensor, Optional[Tensor]]:
        """Get the weight (and bias) tensors from the analog crossbar.

        Args:
            **kwargs: see tile level,
                e.g. :meth:`~aihwkit.simulator.tiles.analog.AnalogTile.get_weights`.

        Returns:
            tuple: weight matrix, bias vector

        Raises:
            ModuleError: if not of type TileModule.
        """
        if hasattr(self, "analog_module"):
            return self.analog_module.get_weights(**kwargs)
        raise ModuleError(f"get_weights not implemented for {type(self).__name__} ")

    def load_state_dict(
        self,  # pylint: disable=arguments-differ
        state_dict: "OrderedDict[str, Tensor]",
        strict: bool = True,
        load_rpu_config: Optional[bool] = None,
        strict_rpu_config_check: Optional[bool] = None,
    ) -> NamedTuple:
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

            strict_rpu_config_check: Whether to check and throw an
                error if the current ``rpu_config`` is not of the same
                class type when setting ``load_rpu_config`` to
                False. In case of ``False`` the user has to make sure
                that the ``rpu_config`` are compatible.

        Returns:
            see torch's ``load_state_dict``

        Raises:
            ModuleError: in case the rpu_config class mismatches
            or mapping parameter mismatch for
            ``load_rpu_config=False``.

        """
        for analog_tile in self.analog_tiles():
            analog_tile.set_load_rpu_config_state(load_rpu_config, strict_rpu_config_check)
        return super().load_state_dict(state_dict, strict)  # type: ignore

    def prepare_for_ddp(self) -> None:
        """Adds ignores to avoid broadcasting the analog tile states in case of
        distributed training.

        Note:
            Call this function before the mode is converted with DDP.

        Important:
            Only InferenceTile supports DDP.

        Raises:

            ModuleError: In case analog tiles with are used that do not
                support data-parallel model, ie. all RPUCUda training
                tiles.
        """
        # pylint: disable=attribute-defined-outside-init
        exclude_list = []
        for module in self.modules():  # type: ignore
            if isinstance(module, AnalogLayerBase):
                for analog_tile in module.analog_tiles():
                    if not analog_tile.supports_ddp:
                        raise ModuleError(
                            "DDP is only supported with some tiles (e.g. Torch/InferenceTile)"
                        )
                exclude_list += [
                    AnalogTileStateNames.CONTEXT,
                    AnalogTileStateNames.ANALOG_STATE_NAME,
                ]
        exclude_list = list(set(exclude_list))
        params = self.state_dict().keys()  # type: ignore
        exclude_params = []
        for param in params:
            for word in exclude_list:
                if word in param and word not in exclude_params:
                    exclude_params.append(param)
                    break
        self._ddp_params_and_buffers_to_ignore = exclude_params

    def drift_analog_weights(self, t_inference: float = 0.0) -> None:
        """(Program) and drift the analog weights.

        Args:
            t_inference: assumed time of inference (in sec).

        Raises:
            ModuleError: if the layer is not in evaluation mode.
        """
        if self.training:  # type: ignore
            raise ModuleError("drift_analog_weights can only be applied in evaluation mode")
        for analog_tile in self.analog_tiles():
            if isinstance(analog_tile, InferenceTileWithPeriphery):
                analog_tile.drift_weights(t_inference)

    def program_analog_weights(self, noise_model: Optional["BaseNoiseModel"] = None) -> None:
        """Program the analog weights.

        Args:
            noise_model: Optional defining the noise model to be
                used. If not given, it will use the noise model
                defined in the `RPUConfig`.

                Caution:

                    If given a noise model here it will overwrite the
                    stored `rpu_config.noise_model` definition in the
                    tiles.

        Raises:
            ModuleError: if the layer is not in evaluation mode.
        """
        if self.training:  # type: ignore
            raise ModuleError("program_analog_weights can only be applied in evaluation mode")
        for analog_tile in self.analog_tiles():
            analog_tile.program_weights(noise_model=noise_model)

    def extra_repr(self) -> str:
        """Set the extra representation of the module.

        Returns:
            A string with the extra representation.
        """
        output = super().extra_repr()  # type: ignore
        if len(output) == 0:
            # likely Sequential. Keep also silent
            return output
        if not hasattr(self, "_extra_repr_save"):
            # pylint: disable=attribute-defined-outside-init
            self._extra_repr_save = next(self.analog_tiles()).rpu_config.__class__.__name__
        output += ", " + self._extra_repr_save
        return output.rstrip()

    def remap_analog_weights(self, weight_scaling_omega: Optional[float] = 1.0) -> None:
        """Gets and re-sets the weights in case of using the weight scaling.

        This re-sets the weights with applied mapping scales, so that
        the weight mapping scales are updated.

        In case of hardware-aware training, this would update the
        weight mapping scales so that the absolute max analog weights
        are set to 1 (as specified in the ``weight_scaling``
        configuration of
        :class:`~aihwkit.parameters.mapping.MappingParameter`).

        Note:
            By default the weight scaling omega factor is set to 1
            here (overriding any setting in the ``rpu_config``). This
            means that the max weight value is set to 1 internally for
            the analog weights.

        Caution:
            This should typically *not* be called for analog. Use
            ``program_weights`` to re-program.

        Args:
            weight_scaling_omega: The weight scaling omega factor (see
                :class:`~aihwkit.parameters.mapping.MappingParameter`). If
                set to None here, it will take the value in the
                mapping parameters. Default is however 1.0.
        """
        for analog_tile in self.analog_tiles():
            analog_tile.remap_weights(weight_scaling_omega=weight_scaling_omega)

    def replace_rpu_config(self, rpu_config: RPUConfigBase) -> None:
        """Modifies the RPUConfig for all underlying analog tiles.

        Each tile will be recreated, to apply the RPUConfig changes.

        Note:

            Typically, the RPUConfig class needs to be the same
            otherwise an error will be raised.

        Caution:
            If analog tiles have different RPUConfigs, these
            differences will be overwritten

        Args:
            rpu_config: New RPUConfig to apply
        """

        for analog_tile in self.analog_tiles():
            analog_tile.to(rpu_config)
