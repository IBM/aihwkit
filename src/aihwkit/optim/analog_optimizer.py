# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Analog-aware inference optimizer."""

from types import new_class
from typing import Any, Callable, Dict, Optional, Type

from torch import cat
from torch.optim import Optimizer, SGD, Adam
from torch.autograd import no_grad

from aihwkit.optim.context import AnalogContext


class AnalogOptimizerMixin:
    """Mixin for analog optimizers.

    This class contains the methods needed for enabling analog in an existing
    ``Optimizer``. It is designed to be used as a mixin in conjunction with an
    ``AnalogOptimizer`` or torch ``Optimizer``.
    """

    def regroup_param_groups(self, *_: Any) -> None:
        """Reorganize the parameter groups, isolating analog layers.

        Update the `param_groups` of the optimizer, moving the parameters for
        each analog layer to a new single group.
        """
        # Create the new param groups.
        analog_param_groups = []
        rm_group_lst = []
        for group in self.param_groups:  # type: ignore[has-type]
            rm_lst = []
            for param in group["params"]:
                if isinstance(param, AnalogContext):
                    param.analog_tile.set_learning_rate(
                        self.defaults["lr"]  # type: ignore[attr-defined]
                    )
                    analog_param_groups.append({"params": [param]})
                    rm_lst.append(id(param))

            group["params"] = [p for p in group["params"] if id(p) not in rm_lst]

            if len(group["params"]) == 0:
                rm_group_lst.append(id(group))

        self.param_groups = [
            g for g in self.param_groups if id(g) not in rm_group_lst  # type: ignore[has-type]
        ]

        # Add analog groups.
        for group in analog_param_groups:
            self.add_param_group(group)  # type: ignore[attr-defined]

    @no_grad()
    def step(self, closure: Optional[Callable] = None, **kwargs: Any) -> Optional[float]:
        """Perform an analog-aware single optimization step.

        If a group containing analog parameters is detected, the optimization
        step calls the related RPU controller. For regular parameter groups,
        the optimization step has the same behaviour as ``torch.optim.SGD``.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
            kwargs: additional arguments if any

        Returns:
            The loss, if ``closure`` has been passed as a parameter.
        """
        # pylint: disable=too-many-branches
        # Update non-analog parameters using the given optimizer
        ret = super().step(closure, **kwargs)  # type: ignore[misc]

        # Update analog parameters
        for group in self.param_groups:
            learning_rate = group.get("lr")

            # Use analog_tile object.
            for param in group["params"]:
                if isinstance(param, AnalogContext):
                    # Handle internal analog update.
                    analog_ctx = param
                    analog_tile = analog_ctx.analog_tile

                    if analog_ctx.use_torch_update:
                        # In this case a separate weight parameter exists: do nothing.
                        continue

                    # Call `update` in the tile.
                    if not analog_ctx.has_gradient():
                        # Forward never used.
                        continue

                    # Update learning rate.
                    if learning_rate == 0.0:
                        analog_ctx.reset()
                        continue

                    if learning_rate is not None:
                        analog_tile.set_learning_rate(learning_rate)

                    runtime = analog_tile.get_runtime()
                    if analog_ctx.use_indexed:
                        for x_input, d_input in zip(
                            analog_ctx.analog_input, analog_ctx.analog_grad_output
                        ):
                            analog_tile.update_indexed(
                                (
                                    x_input.to(analog_tile.device)
                                    if runtime.offload_input
                                    else x_input
                                ),
                                (
                                    d_input.to(analog_tile.device)
                                    if runtime.offload_gradient
                                    else d_input
                                ),
                            )
                    else:
                        x_input = cat(
                            analog_ctx.analog_input, axis=-1 if analog_tile.in_trans else 0
                        )
                        d_input = cat(
                            analog_ctx.analog_grad_output, axis=-1 if analog_tile.out_trans else 0
                        )
                        analog_tile.update(
                            x_input.to(analog_tile.device) if runtime.offload_input else x_input,
                            d_input.to(analog_tile.device) if runtime.offload_gradient else d_input,
                        )

                    analog_ctx.reset()

        # Apply post-update step operations (diffuse, decay, etc).
        # (only here because of unknown params order and shared weights)
        for group in self.param_groups:
            for param in group["params"]:
                if isinstance(param, AnalogContext):
                    param.analog_tile.post_update_step()
        return ret

    def set_learning_rate(self, learning_rate: float = 0.1) -> None:
        """Update the learning rate to a new value.

        Update the learning rate of the optimizer, propagating the changes
        to the analog tiles accordingly.

        Args:
            learning_rate: learning rate for the optimizer.
        """
        for param_group in self.param_groups:
            param_group["lr"] = learning_rate
            for param in param_group["params"]:
                if isinstance(param, AnalogContext):
                    # Update learning rate on the tile
                    param.analog_tile.set_learning_rate(learning_rate)


class AnalogOptimizer(AnalogOptimizerMixin, Optimizer):
    """Generic optimizer that wraps an existing ``Optimizer`` for analog inference.

    This class wraps an existing ``Optimizer``, customizing the optimization
    step for triggering the analog update needed for analog tiles. All other
    (digital) parameters are governed by the given torch optimizer. In case of
    hardware-aware training (``InferenceTile``) the tile weight update is also
    governed by the given optimizer, otherwise it is using the internal analog
    update as defined in the ``rpu_config``.

    The ``AnalogOptimizer`` constructor expects the wrapped optimizer class as
    the first parameter, followed by any arguments required by the wrapped
    optimizer.

    Note:
        The instances returned are of a *new* type that is a subclass of:

        * the wrapped ``Optimizer`` (allowing access to all their methods and
          attributes).
        * this ``AnalogOptimizer``.

    Example:
        The following block illustrate how to create an optimizer that wraps
        standard SGD:

        >>> from torch.optim import SGD
        >>> from torch.nn import Linear
        >>> from aihwkit.simulator.configs.configs import InferenceRPUConfig
        >>> from aihwkit.optim import AnalogOptimizer
        >>> model = AnalogLinear(3, 4, rpu_config=InferenceRPUConfig)
        >>> optimizer = AnalogOptimizer(SGD, model.parameters(), lr=0.02)
    """

    SUBCLASSES = {}  # type: Dict[str, Type]
    """Registry of the created subclasses."""

    def __new__(cls, optimizer_cls: Type, *_: Any, **__: Any) -> "AnalogOptimizer":
        subclass_name = "{}{}".format(cls.__name__, optimizer_cls.__name__)

        # Retrieve or create a new subclass, that inherits both from
        # `AnalogOptimizer` and for the specific torch optimizer
        # (`optimizer_cls`).
        if subclass_name not in cls.SUBCLASSES:
            cls.SUBCLASSES[subclass_name] = new_class(subclass_name, (cls, optimizer_cls), {})

        return super().__new__(cls.SUBCLASSES[subclass_name])

    def __init__(
        self, optimizer_cls: Type, *args: Any, **kwargs: Any  # pylint: disable=unused-argument
    ):
        super().__init__(*args, **kwargs)


class AnalogSGD(AnalogOptimizerMixin, SGD):
    """Implements analog-aware stochastic gradient descent."""


class AnalogAdam(AnalogOptimizerMixin, Adam):
    """Implements analog-aware Adam."""
