# -*- coding: utf-8 -*-

# (C) Copyright 2020 IBM. All Rights Reserved.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Analog-aware stochastic gradient descent optimizer."""

from typing import Callable, Optional

from torch import clone, no_grad
from torch.nn import Module
from torch.optim import SGD

from aihwkit.nn.modules.base import AnalogModuleBase


class AnalogSGD(SGD):
    """Implements analog-aware stochastic gradient descent."""

    def regroup_param_groups(self, model: Module) -> None:
        """Reorganize the parameter groups, isolating analog layers.

        Update the `param_groups` of the optimizer, moving the parameters for
        each analog layer to a new single group.

        Args:
            model: model for the optimizer.
        """
        new_param_groups = []

        # Create the new param groups.
        for (_, module) in model.named_modules():
            if isinstance(module, AnalogModuleBase):
                new_param_groups.append({
                    'params': list(module.parameters(recurse=False)),
                    'analog_tile': module.analog_tile
                })
                module.analog_tile.set_learning_rate(self.defaults['lr'])

        # Remove the analog parameters from the main param group, and add
        # the group.
        for param_group in new_param_groups:
            for param in param_group['params']:  # type: ignore
                # Remove the param by its id(), as removing via list.remove()
                # seems to involve comparisons that can lead to errors.
                index = next(
                    i for i, x in enumerate(self.param_groups[0]['params'])
                    if id(x) == id(param))
                self.param_groups[0]['params'].pop(index)
            self.add_param_group(param_group)

        # Cleanup the main parameter group.
        if not self.param_groups[0]['params']:
            self.param_groups.pop(0)

    @no_grad()
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """Performs an analog-aware single optimization step.

        If a group containing analog parameters is detected, the optimization
        step calls the related RPU controller. For regular parameter groups,
        the optimization step has the same behaviour as ``torch.optim.SGD``.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            learning_rate = group['lr']
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            # Use analog_tile object.
            if group.get('analog_tile'):
                analog_tile = group['analog_tile']

                # Update learning rate
                analog_tile.set_learning_rate(learning_rate)

                weights = next(param for param in group['params']
                               if getattr(param, 'is_weight', False))

                analog_tile.update(weights.input, weights.grad_output)
                continue

            for param in group['params']:
                if param.grad is None:
                    continue
                d_p = param.grad
                if weight_decay != 0:
                    d_p = d_p.add(param, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[param]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                param.add_(d_p, alpha=-group['lr'])

        return loss

    def set_learning_rate(self, learning_rate: float = 0.1) -> None:
        """Update the learning rate to a new value.

        Update the learning rate of the optimizer, propagating the changes
        to the analog tiles accordingly.

        Args:
            learning_rate: learning rate for the optimizer.
        """
        for param_group in self.param_groups:
            param_group['lr'] = learning_rate

            if param_group.get('analog_tile'):
                # Update learning rate on the params
                analog_tile = param_group['analog_tile']

                # Update learning rate on the tile
                analog_tile.set_learning_rate(learning_rate)
