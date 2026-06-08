# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Read-only tensor view for analog tile weights."""

from typing import Any, Callable, Optional

from torch import Tensor
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
