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

"""High level analog tiles (base)."""
# pylint: disable=too-few-public-methods, abstract-method, too-many-instance-attributes

from collections import OrderedDict
from typing import Optional, Union, Tuple, Any, Dict, List
from copy import deepcopy

from numpy import array
from numpy.typing import ArrayLike

from torch import Tensor, from_numpy, float32, unsqueeze, cat, empty, stack, dtype
from torch import device as torch_device
from torch.cuda import device as cuda_device
from torch.autograd import no_grad

from aihwkit import __version__
from aihwkit.exceptions import TileError
from aihwkit.simulator.parameters.mapping import MappingParameter
from aihwkit.simulator.parameters.base import RPUConfigGeneric
from aihwkit.simulator.parameters.runtime import RuntimeParameter
from aihwkit.simulator.parameters.enums import RPUDataType
from aihwkit.optim.context import AnalogContext


class TileModuleBase:
    """Base class for (logical) tile modules that can be used in
    layers, e.g. array of TileModules.
    """


class AnalogTileStateNames:
    """Class defining analog tile state name constants.

    Caution:
       Do *not* edit. Some names are attribute names of the tile.
    """

    VERSION = "aihwkit_version"
    WEIGHTS = "analog_tile_weights"
    HIDDEN_PARAMETERS = "analog_tile_hidden_parameters"
    HIDDEN_PARAMETER_NAMES = "analog_tile_hidden_parameter_names"
    CLASS = "analog_tile_class"
    LR = "analog_lr"
    SHARED_WEIGHTS = "shared_weights"
    CONTEXT = "analog_ctx"
    OUT_SCALING = "out_scaling_alpha"
    MAPPING_SCALES = "mapping_scales"
    RPU_CONFIG = "rpu_config"
    ANALOG_STATE_PREFIX = "analog_tile_state_"
    ANALOG_STATE_NAME = "analog_tile_state"
    EXTRA = "state_extra"

    @staticmethod
    def get_field_names() -> List[str]:
        """Returns expected field names."""
        return [
            getattr(AnalogTileStateNames, key)
            for key in AnalogTileStateNames.__dict__
            if not key.startswith("_")
        ]


class BaseTile:
    """Base class for tile classes (without ``torch.Module`` dependence)."""

    def joint_forward(self, x_input: Tensor, is_test: bool = False, ctx: Any = None) -> Tensor:
        """Perform the joint forward method.

        Calls first the ``pre_forward``, then the tile forward, and
        finally the ``post_forward`` step.

        Note:

            The full forward pass is not using autograd, thus all pre
            and post functions need to be handled appropriately in the
            pre/post backward functions.

        Args:
            x_input: ``[N, in_size]`` tensor. If ``in_trans`` is set, transposed.
            is_test: whether to assume testing mode.
            ctx: torch auto-grad context [Optional]

        Returns:
            torch.Tensor: ``[N, out_size]`` tensor. If ``out_trans`` is set, transposed.
        """
        raise NotImplementedError

    def backward(self, d_input: Tensor, ctx: Any = None) -> Tensor:
        """Perform the backward pass.

        Args:
            d_input: ``[N, out_size]`` tensor. If ``out_trans`` is set, transposed.
            ctx: torch auto-grad context [Optional]

        Returns:
            torch.Tensor: ``[N, in_size]`` tensor. If ``in_trans`` is set, transposed.
        """
        raise NotImplementedError

    def update(self, x_input: Tensor, d_input: Tensor) -> None:
        """Perform the update pass.

        Calls the ``pre_update`` method to pre-process the inputs.

        Args:
            x_input: ``[..., in_size]`` tensor. If ``in_trans`` is set, ``[in_size, ...]``.
            d_input: ``[..., out_size]`` tensor. If ``out_trans`` is set, ``[out_size, ...]``.

        Returns:
            None
        """
        raise NotImplementedError


class SimulatorTile:
    """Minimal class interface for implementing the simulator tile.

    Note:
        This tile is generated by ``_create_simulator_tile`` in the
        ``SimulatorTileWrapper``.
    """

    def forward(
        self,
        x_input: Tensor,
        bias: bool = False,
        in_trans: bool = False,
        out_trans: bool = False,
        is_test: bool = False,
        non_blocking: bool = False,
    ) -> Tensor:
        """General simulator tile forward."""
        raise NotImplementedError

    def backward(
        self,
        d_input: Tensor,
        bias: bool = False,
        in_trans: bool = False,
        out_trans: bool = False,
        non_blocking: bool = False,
    ) -> Tensor:
        """Backward pass.

        Only needs to be implemented if torch autograd is `not` used.
        """
        raise NotImplementedError

    def update(
        self,
        x_input: Tensor,
        d_input: Tensor,
        bias: bool = False,
        in_trans: bool = False,
        out_trans: bool = False,
        non_blocking: bool = False,
    ) -> Tensor:
        """Update.

        Only needs to be implemented if torch autograd update is `not` used.
        """
        raise NotImplementedError

    def get_brief_info(self) -> str:
        """Returns a brief info"""
        raise NotImplementedError

    def get_weights(self) -> Tensor:
        """Returns the analog weights."""
        raise NotImplementedError

    def set_weights(self, weight: Tensor) -> None:
        """Stets the analog weights."""
        raise NotImplementedError

    def get_x_size(self) -> int:
        """Returns input size of tile"""
        raise NotImplementedError

    def get_d_size(self) -> int:
        """Returns output size of tile"""
        raise NotImplementedError

    def get_hidden_parameters(self) -> Tensor:
        """Get the hidden parameters of the tile.

        Returns:
            Hidden parameter tensor.
        """
        return empty(0, dtype=float32)

    def get_hidden_parameter_names(self) -> List[str]:
        """Get the hidden parameters names.

        Each name corresponds to a slice in the Tensor slice of the
        ``get_hidden_parameters`` tensor.

        Returns:
            List of names.
        """
        return []

    def set_hidden_parameters(self, params: Tensor) -> None:
        """Set the hidden parameters of the tile."""

    def get_learning_rate(self) -> Optional[float]:
        """Get the learning rate of the tile.

        Returns:
           learning rate if exists.
        """

    def set_learning_rate(self, learning_rate: Optional[float]) -> None:
        """Set the learning rate of the tile.

        No-op for tiles that do not need a learning rate.

        Args:
           learning rate: learning rate to set
        """

    def dump_extra(self) -> Optional[Dict]:
        """Dumps any extra states / attributed necessary for
        checkpointing.

        For Tiles based on Modules, this should be normally handled by
        torch automatically.
        """

    def load_extra(self, extra: Dict, strict: bool = False) -> None:
        """Load any extra states / attributed necessary for
        loading from checkpoint.

        For Tiles based on Modules, this should be normally handled by
        torch automatically.

        Note:
            Expects the exact same RPUConfig / device etc for applying
            the states. Cross-loading of state-dicts is not supported
            for extra states, they will be just ignored.

        Args:
            extra: dictionary of states from `dump_extra`.
            strict: Whether to throw an error if keys are not found.
        """

    def set_weights_uniform_random(self, bmin: float, bmax: float) -> None:
        """Sets the weights to uniform random numbers.

        Args:
           bmin: min value
           bmax: max value
        """
        raise NotImplementedError

    def get_meta_parameters(self) -> Any:
        """Returns meta parameters."""
        raise NotImplementedError


class SimulatorTileWrapper:
    """Wrapper base class for defining the necessary tile
    functionality.

    Will be overloaded extended for C++ or for any TorchTile.

    Args:
        out_size: output size
        in_size: input size
        rpu_config: resistive processing unit configuration.
        bias: whether to add a bias column to the tile.
        in_trans: Whether to assume an transposed input (batch first)
        out_trans: Whether to assume an transposed output (batch first)
        handle_output_bound: whether the bound clamp gradient should be inserted
        ignore_analog_state: whether to ignore the analog state when __getstate__ is called
    """

    def __init__(
        self,
        out_size: int,
        in_size: int,
        rpu_config: RPUConfigGeneric,
        bias: bool = True,
        in_trans: bool = False,
        out_trans: bool = False,
        torch_update: bool = False,
        handle_output_bound: bool = False,
        ignore_analog_state: bool = False,
    ):
        self.is_cuda = False
        self.device = torch_device("cpu")
        self.out_size = out_size
        self.in_size = in_size
        self.rpu_config = deepcopy(rpu_config)
        self.in_trans = in_trans
        self.out_trans = out_trans
        self.handle_output_bound = handle_output_bound
        self.ignore_analog_state = ignore_analog_state
        self.shared_weights = None

        # handling the bias
        if hasattr(rpu_config, "mapping"):
            mapping = rpu_config.mapping
        else:
            mapping = MappingParameter()

        self.digital_bias = bias and mapping.digital_bias
        self.use_bias = bias
        self.analog_bias = bias and not mapping.digital_bias

        x_size = self.in_size + 1 if self.analog_bias else self.in_size
        d_size = self.out_size

        self.tile = self._create_simulator_tile(x_size, d_size, rpu_config)

        self.analog_ctx = AnalogContext(self)
        self.analog_ctx.use_torch_update = torch_update

    def get_runtime(self) -> RuntimeParameter:
        """Returns the runtime parameter."""
        if not hasattr(self.rpu_config, "runtime"):
            self.rpu_config.runtime = RuntimeParameter()
        return self.rpu_config.runtime

    def get_data_type(self) -> RPUDataType:
        """Return data_type setting of the RPUConfig"""
        return self.get_runtime().data_type

    def get_dtype(self) -> dtype:
        """Return dtype setting of the RPUConfig"""
        return self.get_runtime().data_type.as_torch()

    def _create_simulator_tile(
        self, x_size: int, d_size: int, rpu_config: "RPUConfigGeneric"
    ) -> Any:  # just use Any instead of Union["SimulatorTile", tiles.AnalogTile, ..]
        """Create a simulator tile.

        Args:
            x_size: input size
            d_size: output size
            rpu_config: resistive processing unit configuration

        Returns:
            a simulator tile based on the specified configuration.
        """
        raise NotImplementedError

    def get_tensor_view(self, ndim: int, dim: Optional[int] = None) -> tuple:
        """Return the tensor view for ndim vector at dim.

        Args:
            ndim: number of dimensions
            dim: the dimension to set to -1

        Returns:
            Tuple of ones with the `dim`` index sets to -1
        """
        if dim is None:
            dim = 0 if self.out_trans else ndim - 1
        tensor_view = [1] * ndim
        tensor_view[dim] = -1
        return tuple(tensor_view)

    def get_forward_out_bound(self) -> Optional[float]:
        """Helper for getting the output bound to correct the
        gradients using the AnalogFunction.
        """
        return None

    def set_verbosity_level(self, verbose: int) -> None:
        """Set the verbosity level.

        Args:
            verbose: level of verbosity
        """

    def get_analog_ctx(self) -> AnalogContext:
        """Return the analog context of the tile to be used in ``AnalogFunction``."""
        return self.analog_ctx

    def get_brief_info(self) -> str:
        """Return short info about the underlying C++ tile."""
        return self.tile.get_brief_info().rstrip()

    def update(self, x_input: Tensor, d_input: Tensor) -> None:
        """Implements tile update (e.g. using pulse trains)."""
        raise NotImplementedError

    def update_indexed(self, x_input: Tensor, d_input: Tensor) -> None:
        """Implements indexed interface to the tile update
        (e.g. using pulse trains)."""
        raise NotImplementedError

    def get_analog_state(self) -> Dict:
        """Get the analog state for the state_dict.

        Excludes the non-analog state names that might be added for
        pickling. Only fields defined in ``AnalogTileStateNames`` are
        returned.
        """
        state = self.__getstate__()
        fields = AnalogTileStateNames.get_field_names()
        rm_fields = []
        for key in state:
            if key not in fields:
                rm_fields.append(key)
        for key in rm_fields:
            state.pop(key)
        return state

    def __getstate__(self) -> Dict:
        """Get the state for pickling.

        This method removes the ``tile`` member, as the binding Tiles are not
        serializable.
        """
        # Caution: all attributes of the tile will be saved.
        current_dict = self.__dict__.copy()

        if getattr(self, "ignore_analog_state", False):
            return current_dict

        SN = AnalogTileStateNames
        current_dict[SN.WEIGHTS] = self.tile.get_weights()
        current_dict[SN.HIDDEN_PARAMETERS] = self.tile.get_hidden_parameters().data
        current_dict[SN.HIDDEN_PARAMETER_NAMES] = self.tile.get_hidden_parameter_names()
        current_dict[SN.CLASS] = type(self).__name__
        current_dict[SN.LR] = self.tile.get_learning_rate()
        current_dict.pop("tile", None)
        current_dict[SN.CONTEXT] = self.analog_ctx.data
        current_dict[SN.EXTRA] = self.tile.dump_extra()
        current_dict[SN.VERSION] = __version__

        # don't save device. Will be determined by loading object
        current_dict.pop("stream", None)
        current_dict.pop("is_cuda", None)
        current_dict.pop("device", None)

        # this is should not be saved.
        current_dict.pop("image_sizes", None)

        return current_dict

    def __setstate__(self, state: Dict) -> None:
        """Set the state after unpickling.

        This method recreates the ``tile`` member, creating a new one from
        scratch, as the binding Tiles are not serializable.

        Caution:
            RPU configs are overwritten by loading the state.

        Note:

           Some RPUCuda (analog training) compounds have some extra
           internal states that should be set if checkpointing to
           continue training. To support this, extra states are
           extracted and stored. However, these are _not_ applied if
           cross-loading is done, e.g. map location is different for
           inference or tile type is changed. It will not throw any
           notice is they are not applied.

        Raises:
            TileError: if tile class does not match or hidden parameters do not match

        """

        # pylint: disable=too-many-locals, too-many-statements, too-many-branches

        if getattr(self, "ignore_analog_state", False) or state.get("ignore_analog_state", False):
            self.__dict__.update(state)
            analog_ctx = self.analog_ctx
        else:
            SN = AnalogTileStateNames
            current_dict = state.copy()
            tile_class = current_dict.pop(SN.CLASS, type(self).__name__)
            analog_lr = current_dict.pop(SN.LR, 0.01)
            analog_ctx = current_dict.pop(SN.CONTEXT, None)
            weights = current_dict.pop(SN.WEIGHTS)
            extra = current_dict.pop(SN.EXTRA, None)

            hidden_parameters = current_dict.pop(SN.HIDDEN_PARAMETERS)
            hidden_parameters_names = current_dict.pop(SN.HIDDEN_PARAMETER_NAMES, [])
            current_dict.pop("analog_alpha_scale", None)  # legacy
            current_dict.pop("image_sizes", None)  # should not be saved

            # legacy
            if "non_blocking" not in current_dict:
                current_dict["non_blocking"] = False

            # do we need to re-create the tile? fix for issue #609
            # see https://github.com/IBM/aihwkit/issues/609
            need_to_recreate = not hasattr(
                self, "rpu_config"
            ) or "TorchInferenceRPUConfig" not in str(self.rpu_config.__class__)

            # Check for tile mismatch
            rpu_config = current_dict.pop("rpu_config")
            if hasattr(self, "rpu_config"):
                # only for state-dict load. Might not yet be defined (in
                # case of pickle load or deepcopy)
                if not self.rpu_config.compatible_with(tile_class):
                    raise TileError(
                        "Error creating tile"
                        f". Possible mismatch between {tile_class} and {type(self).__name__}"
                    )
                # Need to always keep the same tile class
                rpu_config.tile_class = self.rpu_config.tile_class

            self.rpu_config = rpu_config
            self.__dict__.update(current_dict)

            self.device = torch_device("cpu")
            self.is_cuda = False

            # recreate attributes not saved
            # always first create on CPU
            x_size = self.in_size + 1 if self.analog_bias else self.in_size
            d_size = self.out_size

            # Recreate the tile.
            self.tile = (
                self._create_simulator_tile(x_size, d_size, self.rpu_config)
                if need_to_recreate
                else self.tile
            )
            names = self.tile.get_hidden_parameter_names()
            if len(hidden_parameters_names) > 0 and names != hidden_parameters_names:
                # Check whether names match
                raise TileError(
                    "Mismatch with loaded analog state: Hidden parameter structure is unexpected."
                )

            if not isinstance(hidden_parameters, Tensor):
                hidden_parameters = from_numpy(array(hidden_parameters))
            self.tile.set_hidden_parameters(hidden_parameters)
            if not isinstance(weights, Tensor):
                weights = from_numpy(array(weights))
            self.tile.set_weights(weights)

            if analog_lr is not None:
                self.tile.set_learning_rate(analog_lr)

            # finally set the extra stuff (without complaining if keys not
            # found. Note that these extra states are only needed for some
            # tiles (compounds) if training needs to be continued without
            # resetting counters etc.)
            if extra is not None:
                self.tile.load_extra(extra, False)

        # map location should be applied to tensors in state_dict
        self.analog_ctx = AnalogContext(self)

        if analog_ctx is not None:
            # Keep the object ID and device
            to_device = analog_ctx.device
            if self.device != to_device:
                self.analog_ctx = self.analog_ctx.to(to_device)
            self.analog_ctx.set_data(analog_ctx.data)

    @no_grad()
    def post_update_step(self) -> None:
        """Operators that need to be called once per mini-batch.

        Note:
            This function is called by the analog optimizer.

        Caution:
            If no analog optimizer is used, the post update steps will
            not be performed.
        """

    def _combine_weights(
        self, weight: Union[Tensor, ArrayLike], bias: Optional[Union[Tensor, ArrayLike]] = None
    ) -> Tensor:
        """Helper to combines weights and biases

        In any case, a detached cpu weight and bias copy will be returned.

        Args:
            weight: weights without the bias
            bias: The bias vector if available

        Returns:
            combined weights with biases

        Raises:
            ValueError: if the tile has bias but ``bias`` has not been
                specified.
        """
        d_type = self.get_dtype()
        if not isinstance(weight, Tensor):
            weight = from_numpy(array(weight))
        weight = weight.clone().detach().cpu().to(d_type)

        shape = [self.out_size, self.in_size]
        weight = weight.reshape(shape)

        if self.analog_bias:
            # Create a ``[out_size, in_size (+ 1)]`` matrix.
            if bias is None:
                raise ValueError("Analog tile has a bias, but no bias given")

            if not isinstance(bias, Tensor):
                bias = from_numpy(array(bias))

            bias = unsqueeze(bias.clone().detach().cpu().to(d_type), 1)
            return cat((weight, bias), dim=1)
        # Use only the ``[out_size, in_size]`` matrix.
        return weight

    def _separate_weights(self, combined_weights: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        """Helper to separate the combined weights and biases"""
        # Split the internal weights (and potentially biases) matrix.
        if self.analog_bias:
            # combined_weights is [out_size, in_size (+ 1)].
            return Tensor(combined_weights[:, :-1]), Tensor(combined_weights[:, -1])

        return combined_weights, None

    @no_grad()
    def cpu(self) -> "SimulatorTileWrapper":
        """Return a copy of this tile in CPU memory.

        Returns:
            self in case of CPU
        """
        if not self.is_cuda:
            return self

        self.is_cuda = False
        self.device = torch_device("cpu")
        self.analog_ctx.data = self.analog_ctx.data.cpu()
        self.analog_ctx.reset(self)

        return self

    @no_grad()
    def cuda(
        self, device: Optional[Union[torch_device, str, int]] = None
    ) -> "SimulatorTileWrapper":
        """Return a copy of the  tile in CUDA memory.

        Args:
            device: CUDA device

        Returns:
            Self with the underlying C++ tile moved to CUDA memory.

        Raises:
            CudaError: if the library has not been compiled with CUDA.
        """
        device = torch_device("cuda", cuda_device(device).idx)
        self.is_cuda = True
        self.device = device
        self.analog_ctx.data = self.analog_ctx.data.cuda(device)
        self.analog_ctx.reset(self)
        return self

    def get_hidden_parameters(self) -> "OrderedDict":
        """Get the hidden parameters of the tile.

        Returns:
            Ordered dictionary of hidden parameter tensors.
        """
        names = self.tile.get_hidden_parameter_names()
        hidden_parameters = self.tile.get_hidden_parameters().detach_()

        ordered_parameters = OrderedDict()
        for idx, name in enumerate(names):
            ordered_parameters[name] = hidden_parameters[idx].clone()

        return ordered_parameters

    def set_hidden_parameters(self, ordered_parameters: "OrderedDict") -> None:
        """Set the hidden parameters of the tile.

        Caution:
            Usually the hidden parameters are drawn according to the
            parameter definitions (those given in the RPU config). If
            the hidden parameters are arbitrary set by the user, then
            this correspondence might be broken. This might cause problems
            in the learning, in particular, the `weight granularity`
            (usually ``dw_min``, depending on the device) is needed for
            the dynamic adjustment of the bit length
            (``update_bl_management``, see
            :class:`~aihwkit.simulator.parameters.utils.UpdateParameters`).

            Currently, the new ``dw_min`` parameter is tried to be
            estimated from the average of hidden parameters if the
            discrepancy with the ``dw_min`` from the definition is too
            large.

        Args:
            ordered_parameters: Ordered dictionary of hidden parameter tensors.

        Raises:
            TileError: In case the ordered dict keys do not conform
                with the current rpu config tile structure of the hidden
                parameters
        """
        if len(ordered_parameters) == 0:
            return

        hidden_parameters = stack(list(ordered_parameters.values()), dim=0)
        names = self.tile.get_hidden_parameter_names()
        if names != list(ordered_parameters.keys()):
            raise TileError(
                "Mismatch with loaded analog state: Hidden parameter structure is unexpected."
            )

        self.tile.set_hidden_parameters(hidden_parameters)

    def set_learning_rate(self, learning_rate: Optional[float]) -> None:
        """Set the tile learning rate.

        Set the tile learning rate to ``-learning_rate``. Note that the
        learning rate is always taken to be negative (because of the meaning in
        gradient descent) and positive learning rates are not supported.

        Args:
            learning_rate: the desired learning rate.
        """
        raise NotImplementedError

    def get_learning_rate(self) -> float:
        """Return the tile learning rate.

        Returns:
            float: the tile learning rate.
        """
        raise NotImplementedError
