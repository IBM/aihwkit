# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021 IBM. All Rights Reserved.
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
from warnings import warn

from torch import clone, no_grad
from torch.nn import Module
from torch.optim import SGD

from aihwkit.nn.modules.base import AnalogModuleBase
from aihwkit.simulator.tiles.base import AnalogContext


class AnalogSGD(SGD):

    """Implements analog-aware stochastic gradient descent."""

    def check_analog_module_devices(self, model: Module) -> None:
        """Checks and moves analog modules to the correct cuda device."""

        # TODO: remove this check and add a .to / .cuda to the AnalogContext
        manual_cuda_move = False  # For showing only one warning.
        for (_, module) in model.named_modules():
            if isinstance(module, AnalogModuleBase):
                analog_tile = module.analog_tile
                analog_ctx = analog_tile.get_analog_ctx()

                # Pytorch module applies everything only to the parameters
                # and buffers, including the device, so we might need to change
                # the device of the module to put the analog layer on the
                # correct device.
                if analog_ctx.device != analog_tile.device:
                    module.cuda(analog_ctx.device)
                    manual_cuda_move = True

        if manual_cuda_move:
            warn('The tiles of the analog layers have been moved to cuda '
                 'manually. Please use `.cuda()` directly on the analog layers '
                 'or `AnalogSequential.cuda()` for automatic handling.')

    def regroup_param_groups(self, model: Module) -> None:
        """Reorganize the parameter groups, isolating analog layers.

        Update the `param_groups` of the optimizer, moving the parameters for
        each analog layer to a new single group.

        Also checks analog modules for the correct CUDA device.

        Args:
            model: model for the optimizer.
        """

        self.check_analog_module_devices(model)

        # Create the new param groups.
        analog_param_groups = []
        rm_group_lst = []
        for group in self.param_groups:  # type: ignore[has-type]
            rm_lst = []
            for param in group['params']:
                if isinstance(param, AnalogContext):

                    param.analog_tile.set_learning_rate(self.defaults['lr'])
                    analog_param_groups.append({
                        'params': [param],
                    })
                    rm_lst.append(id(param))

            group['params'] = [p for p in group['params'] if id(p) not in rm_lst]

            if len(group['params']) == 0:
                rm_group_lst.append(id(group))

        self.param_groups = [g for g in self.param_groups  # type: ignore[has-type]
                             if id(g) not in rm_group_lst]

        # Add analog groups
        for group in analog_param_groups:
            self.add_param_group(group)

    @no_grad()
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """Performs an analog-aware single optimization step.

        If a group containing analog parameters is detected, the optimization
        step calls the related RPU controller. For regular parameter groups,
        the optimization step has the same behaviour as ``torch.optim.SGD``.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.

        Returns:
            The loss, if ``closure`` has been passed as a parameter.
        """
        # pylint: disable=too-many-branches,too-many-locals, protected-access
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
            for param in group['params']:
                if isinstance(param, AnalogContext):
                    # Handle internal analog update
                    analog_ctx = param
                    analog_tile = analog_ctx.analog_tile

                    if analog_ctx.use_torch_update:
                        # in this case a separate weight parameter exists: do nothing
                        continue

                    # Update learning rate
                    analog_tile.set_learning_rate(learning_rate)

                    # Call `update` in the tile.
                    if not analog_ctx.has_gradient():
                        # forward never used
                        continue

                    if analog_ctx.use_indexed:
                        for x_input, d_input in zip(analog_ctx.analog_input,
                                                    analog_ctx.analog_grad_output):
                            analog_tile.update_indexed(x_input, d_input)
                    else:
                        for x_input, d_input in zip(analog_ctx.analog_input,
                                                    analog_ctx.analog_grad_output):
                            analog_tile.update(x_input, d_input)

                    analog_ctx.reset()

                    continue

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

            # Apply post-update step operations (diffuse, decay, etc).
            # (only here because of unknown params order and shared weights)
            for param in group['params']:
                if isinstance(param, AnalogContext):
                    param.analog_tile.post_update_step()

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
            for param in param_group['params']:
                if isinstance(param, AnalogContext):
                    # Update learning rate on the tile
                    param.analog_tile.set_learning_rate(learning_rate)
