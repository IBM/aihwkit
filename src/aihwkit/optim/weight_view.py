# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Read-only tensor view for analog tile weights."""

from torch import Tensor
from torch.utils._pytree import tree_map


class ReadOnlyWeightView(Tensor):
    """A tensor that shares storage with tile weights but blocks in-place mutations.

    All read operations (``size``, ``norm``, ``sum``, indexing, comparisons, etc.)
    work transparently because this IS a real tensor sharing the same memory.
    In-place write operations raise ``RuntimeError`` with guidance to use the
    correct analog tile API.

    This class is stateless — it always blocks in-place operations. The policy
    of whether to wrap or unwrap is managed by :class:`AnalogContext` via its
    ``readonly`` flag.
    """

    @staticmethod
    def __new__(cls, data: Tensor) -> "ReadOnlyWeightView":
        """Create a ReadOnlyWeightView sharing storage with ``data``."""
        if isinstance(data, ReadOnlyWeightView):
            return data
        return Tensor._make_subclass(cls, data)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        func_name = getattr(func, "__name__", "")

        # PyTorch convention: in-place ops end with single '_' (add_, mul_, ...)
        # Dunder methods (__repr__, __eq__, ...) end with '__' and must pass through
        if func_name.endswith("_") and not func_name.endswith("__"):
            raise RuntimeError(
                f"In-place operation '{func_name}' is not allowed on analog weights. "
                f"Analog weights cannot be modified directly — this would bypass "
                f"the physical constraints of the analog device.\n"
                f"  - For programmatic writes: analog_tile.set_weights(new_weight)\n"
                f"  - For gradient updates:    analog_tile.update(x_input, d_input)\n"
                f"  - To unlock direct access: analog_ctx.readonly = False"
            )

        # Unwrap to plain Tensor so downstream ops don't propagate our subclass
        def unwrap(t):
            return t.as_subclass(Tensor) if isinstance(t, ReadOnlyWeightView) else t

        args = tree_map(unwrap, args)
        kwargs = tree_map(unwrap, kwargs)
        return func(*args, **kwargs)

    def __setitem__(self, key, value):
        """Block item assignment (e.g., ``ctx.data[0, 0] = 999``)."""
        raise RuntimeError(
            "Direct item assignment on analog weights is not allowed. "
            "Use analog_tile.set_weights() instead, "
            "or set analog_ctx.readonly = False to unlock direct access."
        )

    def as_writable(self) -> Tensor:
        """Return the underlying plain Tensor (for internal tile use only).

        This removes the read-only guard.  Only tile internals should call this.
        """
        return self.as_subclass(Tensor)
