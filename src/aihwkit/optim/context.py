# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Parameter context for analog tiles."""

# pylint: disable=attribute-defined-outside-init

from typing import Optional, Type, Union, Any, List, TYPE_CHECKING

from torch import dtype, Tensor
from torch._C import DisableTorchFunction
from torch.nn import Parameter
from torch import device as torch_device
from torch.utils._pytree import tree_map

from aihwkit.optim.weight_view import (
    ReadOnlyWeightView,
    raise_if_readonly_write_target,
    PlaceholderDataView,
    _PLACEHOLDER_METADATA_FUNCTIONS,
    _raise_placeholder_read_error,
)
from aihwkit.simulator.parameters.enums import AnalogContextDataViewMode

if TYPE_CHECKING:
    from aihwkit.simulator.tiles.base import SimulatorTileWrapper


# Tensor properties that materialize weight values (transpose / conjugate views
# and complex parts). They must honor the data-view mode like their method
# equivalents (``t()``, ``conj()``); otherwise the getset descriptors fall
# through to ``__torch_function__`` as the whitelisted ``__get__`` and silently
# return uninitialized placeholder memory in PLACEHOLDER mode.
_VALUE_VIEW_PROPERTIES = ("T", "mT", "H", "mH", "real", "imag")


class AnalogContext(Parameter):
    """Context for analog optimizer.

    If `analog_bias` (which is provided by `analog_tile`) is False,
        `data` has the same meaning as `torch.nn.Parameter`
    If `analog_bias` (which is provided by `analog_tile`) is True,
        The last column of `data` is the `bias` term

    For diagnostic purposes, `AnalogContext` provides three public data view modes.
    Consider the code:
        ---
        layer = AnalogLinear(4, 3, bias=False, rpu_config=rpu_config)
        analog_tile = layer.analog_module
        analog_ctx = analog_tile.analog_ctx
        weight = analog_tile.get_weights()[0]
        ---
    where `weight` is the logical weight view, which is already ``physical weights x scaling``

    Data view modes are controlled by `analog_ctx.data_view_mode` and the corresponding methods:
        ---
        analog_ctx.enable_placeholder().    # PLACEHOLDER mode (default)
        analog_ctx.enable_data_view().      # DATA_VIEW mode
        analog_ctx.enable_buffer().         # BUFFER mode
        ---

    * PLACEHOLDER (default): only metadata, such as ``size()``, ``shape``.
        Since the RPU conductance values is not directly accessible in physic, the weight values,
            as well as value-based operations, such as ``norm()``, are blocked by default
            Access them raises ``RuntimeError``.
        ---
        # inspect metadata without reading values:
        analog_ctx.size()         # [4, 3]
        analog_ctx.device()       # 'cpu'
        analog_ctx.norm()         # RuntimeError
        ---
    * DATA_VIEW: exposes a read-only logical weight view through the `data` attribute,
        which is equivalent to `analog_tile.get_weights()[0]`.
        This allows users to inspect the effective weights.
        Since the changes of both weights and scaling affect the logical weights,
            we adopt the convetion that this logical view is read-only
        Therefore, in-place operations, such as ``add_``, ``mul_``, etc, are blocked
        ---
        # The following three lines will print the same value:
        analog_ctx.size()
        analog_ctx.data.size()
        weight.size()
        # Accessing values is allowed, but they are read-only:
        analog_ctx.norm()                   # Successfully returns the norm
        analog_ctx.norm() == weight.norm()  # True
        analog_ctx.add_(1.0)                # RuntimeError
        ---
    * BUFFER: exposes a zero-initialized tensor with the logical weight shape through the `data`
        At that mode, `data` is an independent buffer that is not connected to the analog tile.
        It is intended for optimizers with digital auxiliary state,
            such as mixed-precision training or TT-v2.
        ---
        analog_ctx.norm() == weight.norm()  # Typically False, since the buffer is independent
        analog_ctx.add_(1.0)                # Successfully adds 1.0 to the buffer, but does not
                                              affect the analog tile weights
        ---

    To update the internal analog weights, use the following update methods instead of
        writing `data` directly in the analog optimizer:
        ---
        analog_ctx.analog_tile.update(...)
        analog_ctx.analog_tile.update_indexed(...)
        ---

    Caution: Even though DATA_VIEW mode allows us to access the weights directly,
        always keep in mind that it is used only for diagnostic purposes.
        To simulate the real reading, call the `read_weights` method
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
        self._data_view_mode = AnalogContextDataViewMode.PLACEHOLDER
        self._data_buffer = None  # type: Optional[Tensor]
        self.analog_input = []  # type: list
        self.analog_grad_output = []  # type: list
        self.reset(analog_tile)

    @classmethod
    def __torch_function__(
        cls, func: Any, _types: Any, args: Any = (), kwargs: Optional[Any] = None
    ) -> Any:
        kwargs = kwargs or {}
        func_name = getattr(func, "__name__", "")

        if func_name == "requires_grad_" and args and isinstance(args[0], AnalogContext):
            # ``requires_grad_`` toggles the autograd flag, not weight values, so
            # the read-only in-place guard below must not reject it and the data
            # view redirection must not send it to a throwaway placeholder.
            # ``nn.Module.requires_grad_`` calls this on every parameter to
            # (un)freeze a layer, so route it straight to the real Parameter,
            # mirroring the ``requires_grad`` attribute setter.
            target = args[0]
            requested = args[1] if len(args) > 1 else kwargs.get("requires_grad", True)
            target.requires_grad = bool(requested)
            return target

        def is_readonly(value: Any) -> bool:
            # BUFFER mode exposes an independent, writable digital buffer
            # (used by mixed-precision optimizers), so in-place ops are allowed.
            if isinstance(value, AnalogContext):
                return value._get_data_view_mode() != AnalogContextDataViewMode.BUFFER
            return isinstance(value, ReadOnlyWeightView)

        raise_if_readonly_write_target(func_name, args, kwargs, is_readonly)

        def to_public_tensor(value: Any) -> Any:
            if isinstance(value, AnalogContext):
                return value._torch_function_data(func_name)
            return value

        args = tree_map(to_public_tensor, args)
        kwargs = tree_map(to_public_tensor, kwargs)
        return func(*args, **kwargs)

    def __setitem__(self, key: Any, value: Any) -> None:
        """Block direct item assignment."""
        raise RuntimeError(
            "Direct item assignment on analog weights is not allowed. "
            "Use analog_tile.set_weights() instead."
        )

    def __dir__(self) -> List[str]:
        """List attribute names for interactive tab-completion.

        ``Tensor.__dir__`` dispatches through ``__torch_function__`` (this class
        defines one), which in PLACEHOLDER mode routes ``__dir__`` through the
        weight-read guard and raises. Completers (rlcompleter, IPython) swallow
        that error and show nothing. Listing the class and instance attribute
        names directly avoids the value-read dispatch, so ``analog_ctx.<tab>``
        offers the same tensor ops as a plain tensor in every data-view mode.
        """
        keys = set(dir(type(self)))
        keys.update(object.__getattribute__(self, "__dict__"))
        return sorted(keys)

    @staticmethod
    def _coerce_data_view_mode(value: Any) -> AnalogContextDataViewMode:
        """Convert public mode inputs to ``AnalogContextDataViewMode``."""
        if isinstance(value, AnalogContextDataViewMode):
            return value
        if isinstance(value, str):
            for mode in AnalogContextDataViewMode:
                if value == mode.value or value.upper() == mode.name:
                    return mode
        raise ValueError(
            "data_view_mode must be an AnalogContextDataViewMode value, "
            "or one of: placeholder, data_view, buffer."
        )

    def _get_data_view_mode(self) -> AnalogContextDataViewMode:
        """Return the active public data view mode."""
        try:
            return object.__getattribute__(self, "_data_view_mode")
        except AttributeError:
            return AnalogContextDataViewMode.PLACEHOLDER

    @property
    def data_view_mode(self) -> AnalogContextDataViewMode:
        """Return the active public data access mode."""
        return self._get_data_view_mode()

    @data_view_mode.setter
    def data_view_mode(self, value: Any) -> None:
        """Set the active public data access mode."""
        mode = self._coerce_data_view_mode(value)
        self._data_view_mode = mode
        if mode == AnalogContextDataViewMode.BUFFER:
            self._data_buffer = self._new_data_buffer()
        else:
            self._data_buffer = None

    def enable_data_view(self) -> "AnalogContext":
        """Enable read-only logical weight reads for diagnostics."""
        self.data_view_mode = AnalogContextDataViewMode.DATA_VIEW
        return self

    def enable_placeholder(self) -> "AnalogContext":
        """Enable metadata-only placeholder mode."""
        self.data_view_mode = AnalogContextDataViewMode.PLACEHOLDER
        return self

    def enable_buffer(self) -> "AnalogContext":
        """Enable an independent zero-initialized digital data buffer."""
        self.data_view_mode = AnalogContextDataViewMode.BUFFER
        return self

    def _raw_data(self) -> Tensor:
        """Return the internal raw tile backing tensor."""
        with DisableTorchFunction():  # pylint: disable=not-context-manager
            return super().__getattribute__("data")

    def _logical_shape(self) -> Any:
        """Return the logical public weight shape without reading values."""
        raw = self._raw_data()
        try:
            analog_tile = object.__getattribute__(self, "analog_tile")
        except AttributeError:
            return raw.shape

        if getattr(analog_tile, "analog_bias", False) and raw.dim() >= 2:
            return raw[:, : analog_tile.in_size].shape
        return raw.shape

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

    def _placeholder_data(self) -> PlaceholderDataView:
        """Return a metadata-only public placeholder."""
        return PlaceholderDataView(self._raw_data().new_empty(self._logical_shape()))

    def _new_data_buffer(self) -> Tensor:
        """Create a zero digital buffer with the logical public weight shape."""
        return self._raw_data().new_zeros(self._logical_shape())

    def _buffer_data(self) -> Tensor:
        """Return the independent digital buffer."""
        return object.__getattribute__(self, "_data_buffer")

    def _public_data(self) -> Tensor:
        """Return the tensor exposed by the active public data mode."""
        mode = self._get_data_view_mode()
        if mode == AnalogContextDataViewMode.PLACEHOLDER:
            return self._placeholder_data()
        if mode == AnalogContextDataViewMode.DATA_VIEW:
            return ReadOnlyWeightView(self._logical_data())
        if mode == AnalogContextDataViewMode.BUFFER:
            return self._buffer_data()
        raise RuntimeError(f"Unsupported AnalogContext data view mode: {mode}")

    def _torch_function_data(self, func_name: str) -> Tensor:
        """Return the tensor used to dispatch public torch operations."""
        mode = self._get_data_view_mode()
        if mode == AnalogContextDataViewMode.PLACEHOLDER:
            if func_name not in _PLACEHOLDER_METADATA_FUNCTIONS:
                _raise_placeholder_read_error(func_name)
            return self._placeholder_data()
        if mode == AnalogContextDataViewMode.DATA_VIEW:
            return self._logical_data()
        if mode == AnalogContextDataViewMode.BUFFER:
            return self._buffer_data()
        raise RuntimeError(f"Unsupported AnalogContext data view mode: {mode}")

    def __getattribute__(self, name: str) -> Any:
        """Intercept public tensor reads according to ``data_view_mode``."""
        if name == "grad_fn":
            return None
        if name in ("device", "dtype", "is_cuda", "is_leaf", "layout"):
            return getattr(self._raw_data(), name)
        if name == "requires_grad":
            with DisableTorchFunction():  # pylint: disable=not-context-manager
                return self.as_subclass(Tensor).requires_grad
        if name == "grad":
            # Mixed-precision optimizers SET param.grad (mpmixin.prepare_grad) and
            # torch optimizers READ it; both must hit the Parameter's own grad slot.
            # DisableTorchFunction bypasses the data-view dispatch that would
            # otherwise redirect the read to the raw data view and return None.
            with DisableTorchFunction():  # pylint: disable=not-context-manager
                return super().__getattribute__("grad")
        if name == "data":
            return self._public_data()
        if name == "shape":
            return self._logical_shape()
        if name == "ndim":
            return len(self._logical_shape())
        if name in _VALUE_VIEW_PROPERTIES:
            return self._value_view_property(name)
        return super().__getattribute__(name)

    def _value_view_property(self, name: str) -> Any:
        """Return a value-bearing view property honoring the data-view mode.

        ``T`` / ``mT`` / ``H`` / ``mH`` / ``real`` / ``imag`` read weight values,
        so they follow the same rules as ``t()`` / ``conj()``: blocked in
        PLACEHOLDER mode, served from the read-only logical view in DATA_VIEW
        mode, and from the digital buffer in BUFFER mode.
        """
        if self._get_data_view_mode() == AnalogContextDataViewMode.PLACEHOLDER:
            _raise_placeholder_read_error(name)
        return getattr(self._public_data(), name)

    def __setattr__(self, name: str, value: Any) -> None:
        """Block user-level replacement of ``.data``."""
        if name == "data":
            raise RuntimeError(
                "Direct replacement of analog_ctx.data is not allowed. "
                "Use analog_tile.set_weights(new_weight) for programmatic writes."
            )
        if name in ("grad", "requires_grad"):
            # Both must bypass the data-view dispatch: ``grad`` to hit the
            # Parameter's own grad slot, and ``requires_grad`` because its
            # setter would otherwise be routed through ``__torch_function__``
            # (as ``__set__``) and raise in PLACEHOLDER mode. Toggling
            # ``requires_grad`` is how analog layers are frozen/unfrozen.
            with DisableTorchFunction():  # pylint: disable=not-context-manager
                super().__setattr__(name, value)
            return
        super().__setattr__(name, value)

    def _replace_raw_data(self, data: Tensor) -> None:
        """Replace the internal raw ``Parameter.data`` for tile rebinding."""
        if isinstance(data, (ReadOnlyWeightView, PlaceholderDataView)):
            data = data.as_subclass(Tensor)
        with DisableTorchFunction():  # pylint: disable=not-context-manager
            super().__setattr__("data", data)

        if self._get_data_view_mode() != AnalogContextDataViewMode.BUFFER:
            return

        buffer = object.__getattribute__(self, "_data_buffer")
        logical_shape = self._logical_shape()
        if buffer is not None and buffer.shape == logical_shape:
            self._data_buffer = buffer.to(device=data.device, dtype=data.dtype)
        else:
            self._data_buffer = self._new_data_buffer()

    # -- existing API ---------------------------------------------------------

    def set_indexed(self, value: bool = True) -> None:
        """Set the context to forward_indexed."""
        self.use_indexed = value

    def get_data(self) -> Tensor:
        """Get a detached tensor from the active public data view."""
        if self._get_data_view_mode() == AnalogContextDataViewMode.PLACEHOLDER:
            _raise_placeholder_read_error("get_data")
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
