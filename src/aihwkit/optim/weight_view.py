# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Read-only and placeholder tensor views for analog tile weights."""

from typing import Any, Callable, Optional

from torch import Tensor
from torch._C import DisableTorchFunction
from torch.utils._pytree import tree_map


def is_inplace_operation(func_name: str) -> bool:
    """Return whether ``func_name`` follows the PyTorch in-place convention."""
    return func_name.endswith("_") and not func_name.endswith("__")


def raise_readonly_error(func_name: str) -> None:
    """Raise the standard read-only analog weight mutation error."""
    raise RuntimeError(
        f"  In-place operation '{func_name}' is not allowed on analog weights.\n"
        f"  AnalogContext exposes an always-read-only logical weight view.\n"
        f"  Direct writes would bypass the analog tile update semantics.\n"
        f"  Please use the appropriate analog tile API to update weights:\n"
        f"    - For programmatic writes: analog_tile.set_weights(new_weight)\n"
        f"    - For gradient updates:    analog_tile.update(x_input, d_input)"
    )


def raise_if_readonly_write_target(
    func_name: str,
    args: Any,
    kwargs: Any,
    is_readonly: Callable[[Any], bool],
) -> None:
    """Raise if a read-only analog weight view is used as a write target."""
    if is_inplace_operation(func_name) and args and is_readonly(args[0]):
        raise_readonly_error(func_name)

    if "out" not in kwargs:
        return

    def block_out_target(value: Any) -> Any:
        if is_readonly(value):
            raise_readonly_error(func_name)
        return value

    tree_map(block_out_target, kwargs["out"])


class ReadOnlyWeightView(Tensor):
    """A tensor view that blocks in-place mutations on analog weights.

    This class is stateless — it always blocks in-place operations, such as ``add_``,
    ``mul_``, etc, which raise ``RuntimeError``.
    All read operations (``size``, ``norm``, ``sum``, indexing, comparisons,
    etc.) work transparently.
    """

    @staticmethod
    def __new__(cls, data: Tensor) -> "ReadOnlyWeightView":
        """Create a ReadOnlyWeightView sharing storage with ``data`` when possible."""
        if isinstance(data, ReadOnlyWeightView):
            return data
        return Tensor._make_subclass(cls, data)

    @classmethod
    def __torch_function__(
        cls, func: Any, _types: Any, args: Any = (), kwargs: Optional[Any] = None
    ) -> Any:
        kwargs = kwargs or {}
        func_name = getattr(func, "__name__", "")

        def is_readonly(value: Any) -> bool:
            return isinstance(value, ReadOnlyWeightView)

        raise_if_readonly_write_target(func_name, args, kwargs, is_readonly)

        def unwrap(t: Any) -> Any:
            return t.as_subclass(Tensor) if isinstance(t, ReadOnlyWeightView) else t

        args = tree_map(unwrap, args)
        kwargs = tree_map(unwrap, kwargs)
        return func(*args, **kwargs)

    def __setitem__(self, key: Any, value: Any) -> None:
        """Block item assignment (e.g., ``ctx.data[0, 0] = 999``)."""
        raise RuntimeError(
            "Direct item assignment on analog weights is not allowed. "
            "Use analog_tile.set_weights() instead."
        )


_PLACEHOLDER_METADATA_FUNCTIONS = {
    "__get__",
    "__len__",
    "__reduce__",
    "__reduce_ex__",
    "detach",
    "dim",
    "element_size",
    "is_contiguous",
    "nelement",
    "numel",
    "size",
    "storage_offset",
    "stride",
    "type",
    "requires_grad_",
    # Pure metadata queries: they depend only on dtype / device / layout, never
    # on weight values, so they are safe to answer in placeholder mode.
    "ndimension",
    "get_device",
    "data_ptr",
    "dense_dim",
    "sparse_dim",
    "has_names",
    "is_floating_point",
    "is_complex",
    "is_signed",
    "is_conj",
    "is_neg",
    "is_pinned",
    "is_shared",
    "is_inference",
    "is_distributed",
    "_is_view",
    "_is_zerotensor",
}


def _raise_placeholder_read_error(func_name: str) -> None:
    """Raise the standard error for reads in placeholder mode."""
    operation = func_name or "unknown"
    raise RuntimeError(
        "AnalogContext data is in placeholder mode, so operation "
        f"'{operation}' cannot read weight values. Call "
        "analog_ctx.enable_data_view() for diagnostic logical weight reads, "
        "or analog_ctx.enable_buffer() for an independent digital buffer."
    )


class PlaceholderDataView(Tensor):
    """Tensor-shaped metadata placeholder for analog context data.

    The backing tensor is intentionally meaningless. Only metadata operations
    are allowed; value reads raise ``RuntimeError``.
    """

    @staticmethod
    def __new__(cls, data: Tensor) -> "PlaceholderDataView":
        """Create a placeholder with the same tensor metadata as ``data``."""
        if isinstance(data, PlaceholderDataView):
            return data
        return Tensor._make_subclass(cls, data.detach())

    @classmethod
    def __torch_function__(
        cls, func: Any, _types: Any, args: Any = (), kwargs: Optional[Any] = None
    ) -> Any:
        kwargs = kwargs or {}
        func_name = getattr(func, "__name__", "")

        def is_placeholder(value: Any) -> bool:
            return isinstance(value, PlaceholderDataView)

        if func_name.endswith("_") and not func_name.endswith("__"):
            if args and is_placeholder(args[0]):
                _raise_placeholder_read_error(func_name)

        if "out" in kwargs:

            def block_out_target(value: Any) -> Any:
                if is_placeholder(value):
                    _raise_placeholder_read_error(func_name)
                return value

            tree_map(block_out_target, kwargs["out"])

        if func_name not in _PLACEHOLDER_METADATA_FUNCTIONS:
            _raise_placeholder_read_error(func_name)

        if func_name == "detach":
            source = args[0]
            with DisableTorchFunction():  # pylint: disable=not-context-manager
                return PlaceholderDataView(source.as_subclass(Tensor).detach())

        def unwrap(value: Any) -> Any:
            if isinstance(value, PlaceholderDataView):
                return value.as_subclass(Tensor)
            return value

        args = tree_map(unwrap, args)
        kwargs = tree_map(unwrap, kwargs)
        return func(*args, **kwargs)

    def __repr__(self) -> str:
        """Return metadata summary without exposing meaningless backing values."""
        with DisableTorchFunction():  # pylint: disable=not-context-manager
            raw = self.as_subclass(Tensor)
            return (
                f"PlaceholderDataView(shape={raw.shape}, dtype={raw.dtype}, "
                f"device={raw.device})"
            )

    def __setitem__(self, key: Any, value: Any) -> None:
        """Block item assignment on the placeholder."""
        _raise_placeholder_read_error("__setitem__")
