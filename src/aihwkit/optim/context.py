# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Parameter context for analog tiles."""

# pylint: disable=attribute-defined-outside-init

from typing import Optional, Type, Union, Any, TYPE_CHECKING

from torch import dtype, Tensor
from torch._C import DisableTorchFunction
from torch.nn import Parameter
from torch import device as torch_device
from torch.utils._pytree import tree_map

from aihwkit.optim.weight_view import (
    ReadOnlyWeightView,
    raise_if_readonly_write_target,
)

if TYPE_CHECKING:
    from aihwkit.simulator.tiles.base import SimulatorTileWrapper


class AnalogContext(Parameter):
    """Context for analog optimizer.

    If `analog_bias` (which is provided by `analog_tile`) is False,
        `data` has the same meaning as `torch.nn.Parameter`
    If `analog_bias` (which is provided by `analog_tile`) is True,
        The last column of `data` is the `bias` term

    Note: For diagnostic purposes, `AnalogContext` exposes a read-only logical weight view
        through the `data` attribute, which is equivalent to `analog_tile.get_weights()[0]`.
        This allows users to inspect the effective weights.
    Direct tensor reads on ``ctx`` or ``ctx.data``, such as ``size()``, ``norm()`` are
        equivalent to do so on ``ctx.analog_tile.get_weights()[0]``.
        i.e, ctx.data == ctx.analog_tile.get_weights()[0]

    The `data` attribution inherited from `torch.nn.Parameter` stores the raw tile weights,
        i.e., the weights without scaling
    Its public tensor value is a read-only logical weight view: ``physical weights x scaling``
        # Example usage:
        ---
        layer = AnalogLinear(4, 3, bias=False, rpu_config=rpu_config)
        analog_tile = layer.analog_module
        analog_ctx = analog_tile.analog_ctx
        weight = analog_tile.get_weights()[0]

        # The following two lines will print the same value:
        analog_ctx.size()
        analog_ctx.data.size()
        weight.size()
        ---
    Since the changes of both weights and scaling affect the logical weights, 
        we adopt the convetion that this logical view is read-only
    Therefore, in-place operations, such as ``add_``, ``mul_``, etc, are blocked
        ctx.data.add_(1.0)     # RuntimeError
    Use the following update methods instead of
        writing `data` directly in the analog optimizer:
        ---
        analog_ctx.analog_tile.update(...)
        analog_ctx.analog_tile.update_indexed(...)
        ---

    Even though it allows us to access the weights directly, always keep in mind that it is used
        only for diagnostic purposes. To simulate the real reading, call the `read_weights` method
        instead, i.e. given `analog_ctx: AnalogContext`,
        estimated_weights, estimated_bias = analog_ctx.analog_tile.read_weights()
    """

    def __new__(
        cls: Type["AnalogContext"],
        analog_tile: "SimulatorTileWrapper",
        parameter: Optional[Parameter] = None,
    ) -> "AnalogContext":
        # pylint: disable=signature-differs
        if parameter is None:
            weights_ref = analog_tile._get_tile_weights_ref()
            return Parameter.__new__(
                cls,
                data=weights_ref,
                requires_grad=True,
            )
        # analog_tile.tile can come from different classes:
        #   aihwkit.simulator.rpu_base.devices.AnalogTile (C++)
        #   TorchInferenceTile (Python)
        # It stores the raw tile matrix; if analog_tile.analog_bias is True,
        # the last raw column stores the bias.

        parameter.__class__ = cls
        return parameter

    def __init__(
        self, analog_tile: "SimulatorTileWrapper", parameter: Optional[Parameter] = None
    ):  # pylint: disable=unused-argument
        super().__init__()
        self.analog_tile = analog_tile
        self.use_torch_update = False
        self.use_indexed = False
        self.analog_input = []  # type: list
        self.analog_grad_output = []  # type: list
        self.reset(analog_tile)

    @classmethod
    def __torch_function__(
        cls, func: Any, _types: Any, args: Any = (), kwargs: Optional[Any] = None
    ) -> Any:
        kwargs = kwargs or {}
        func_name = getattr(func, "__name__", "")

        def is_readonly(value: Any) -> bool:
            return isinstance(value, (AnalogContext, ReadOnlyWeightView))

        raise_if_readonly_write_target(func_name, args, kwargs, is_readonly)

        def to_logical_tensor(value: Any) -> Any:
            if isinstance(value, AnalogContext):
                return value._logical_data()
            return value

        args = tree_map(to_logical_tensor, args)
        kwargs = tree_map(to_logical_tensor, kwargs)
        return func(*args, **kwargs)

    def __setitem__(self, key: Any, value: Any) -> None:
        """Block direct item assignment."""
        raise RuntimeError(
            "Direct item assignment on analog weights is not allowed. "
            "Use analog_tile.set_weights() instead."
        )

    def _raw_data(self) -> Tensor:
        """Return the internal raw tile backing tensor."""
        with DisableTorchFunction():  # pylint: disable=not-context-manager
            return super().__getattribute__("data")

    def _logical_data(self) -> Tensor:
        """Return logical weights equivalent to ``analog_tile.get_weights()[0]``."""
        raw = self._raw_data()
        try:
            analog_tile = object.__getattribute__(self, "analog_tile")
        except AttributeError:
            return raw

        logical = raw
        if getattr(analog_tile, "analog_bias", False) and raw.dim() >= 2:
            logical = raw[:, : analog_tile.in_size]

        get_scales = getattr(analog_tile, "get_scales", None)
        if get_scales is None:
            return logical

        scales = get_scales()
        if scales is None:
            return logical

        scales = scales.to(device=logical.device, dtype=logical.dtype)
        return logical * scales.view(-1, 1)

    def __getattribute__(self, name: str) -> Any:
        """Intercept public tensor reads that expose the logical view."""
        if name == "grad_fn":
            return None
        if name in ("device", "dtype", "grad", "is_cuda", "is_leaf", "layout"):
            return getattr(self._raw_data(), name)
        if name == "data":
            return ReadOnlyWeightView(self._logical_data())
        if name == "shape":
            return self._logical_data().shape
        if name == "ndim":
            return self._logical_data().ndim
        return super().__getattribute__(name)

    def __setattr__(self, name: str, value: Any) -> None:
        """Block user-level replacement of ``.data``."""
        if name == "data":
            raise RuntimeError(
                "Direct replacement of analog_ctx.data is not allowed. "
                "Use analog_tile.set_weights(new_weight) for programmatic writes."
            )
        super().__setattr__(name, value)

    def _replace_raw_data(self, data: Tensor) -> None:
        """Replace the internal raw ``Parameter.data`` for tile rebinding."""
        if isinstance(data, ReadOnlyWeightView):
            data = data.as_subclass(Tensor)
        with DisableTorchFunction():  # pylint: disable=not-context-manager
            super().__setattr__("data", data)

    # -- existing API ---------------------------------------------------------

    def set_indexed(self, value: bool = True) -> None:
        """Set the context to forward_indexed."""
        self.use_indexed = value

    def get_data(self) -> Tensor:
        """Get a detached logical weight tensor."""
        return self.data.detach()

    def reset(self, analog_tile: Optional["SimulatorTileWrapper"] = None) -> None:
        """Reset the gradient trace and optionally sets the tile pointer."""

        if analog_tile is not None:
            self.analog_tile = analog_tile
            self.analog_tile.analog_ctx = self

        self.analog_input = []
        self.analog_grad_output = []

    def has_gradient(self) -> bool:
        """Return whether a gradient trace was stored."""
        return len(self.analog_input) > 0

    def __copy__(self) -> Parameter:
        """Turn off copying of the pointers. Context will be re-created
        when tile is created"""
        return Parameter(self._raw_data())

    def __deepcopy__(self, memo: Any) -> Parameter:
        """Turn off deep copying. Context will be re-created when tile is created"""
        return Parameter(self._raw_data())

    def cuda(self, device: Optional[Union[torch_device, str, int]] = None) -> "AnalogContext":
        """Move the context to a cuda device.

        Args:
             device: the desired device of the tile.

        Returns:
            This context in the specified device.
        """
        if not self.analog_tile.is_cuda:
            self._replace_raw_data(self.analog_tile._get_tile_weights_ref())
            self.analog_tile = self.analog_tile.cuda(device)
            self.reset(self.analog_tile)
        return self

    def cpu(self) -> "AnalogContext":
        """Move the context to CPU.

        Note:
            This is a no-op for CPU context.

        Returns:
            self
        """
        self._replace_raw_data(self._raw_data().cpu())
        if self.analog_tile is not None and self.analog_tile.is_cuda:
            self.analog_tile = self.analog_tile.cpu()
            self.reset(self.analog_tile)
        return self

    def to(self, *args: Any, **kwargs: Any) -> "AnalogContext":
        """Move analog tiles of the current context to a device.

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
        # pylint: disable=invalid-name
        self._replace_raw_data(self._raw_data().to(*args, **kwargs))
        device = None
        if "device" in kwargs:
            device = kwargs["device"]
        elif len(args) > 0 and not isinstance(args[0], (Tensor, dtype)):
            device = torch_device(args[0])

        if device is not None:
            device = torch_device(device)
            if device.type == "cuda" and not self.analog_tile.is_cuda:
                self.cuda(device)
            elif device.type == "cpu" and self.analog_tile.is_cuda:
                self.cpu()
        return self

    def __repr__(self) -> str:
        return "AnalogContext of " + self.analog_tile.get_brief_info()
