# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

# pylint: disable=too-many-instance-attributes

"""High level analog transfer tiles (analog)."""

from typing import Optional, Tuple, List, Dict, Any
from copy import deepcopy

from math import ceil
from torch import Tensor, concatenate, zeros, trunc, eye, rand, ones
from torch.nn import Module
from torch.autograd import no_grad

from aihwkit.exceptions import ConfigError
from aihwkit.simulator.tiles.base import SimulatorTileWrapper, SimulatorTile
from aihwkit.simulator.tiles.analog import AnalogTileWithoutPeriphery
from aihwkit.simulator.tiles.module import TileModule
from aihwkit.simulator.tiles.periphery import TileWithPeriphery
from aihwkit.simulator.tiles.functions import AnalogFunction
from aihwkit.simulator.parameters.base import RPUConfigGeneric, RPUConfigBase
from aihwkit.simulator.parameters.enums import RPUDataType
from aihwkit.simulator.configs.configs import SingleRPUConfig, UnitCellRPUConfig
from aihwkit.simulator.configs.compounds import DynamicTransferCompound, ChoppedTransferCompound


class TransferSimulatorTile(SimulatorTile, Module):
    """SimulatorTile for transfer.

    The RPUCuda library is only used for the single-tile forward / backward / pulsed
    update, however, not for the transfer from the gradient tile to the actual weight
    tile. The transfer part is implemented in python mostly for illustrative purposes and
    to allow for flexible adjustments and development of new algorithms based on the
    Tiki-taka approach.

    Note:
        Only a subset of parameter settings are supported.

    Caution:

        The construction seed that is applied for both tiles when using
        :class:`~aihwkit.simulator.configs.compounds.ChoppedTransferCompound` is here not
        applied, unless given explicitly for the unit cell devices.

    Args:
        out_size: output size
        in_size: input size
        rpu_config: resistive processing unit configuration.
        dtype: data type to use for the tiles.

    Raises:
        ConfigError: in case a setting is not supported.

    """

    def __init__(
        self, x_size: int, d_size: int, rpu_config: "UnitCellRPUConfig", dtype: RPUDataType
    ):
        Module.__init__(self)

        self.x_size = x_size
        self.d_size = d_size
        self.update_counter = 0
        self.chop_counter = 0
        self.transfer_idx = 0
        self.learning_rate = 0.1

        self.m_x = 0.0
        self.m_d = 0.0

        self._transfer_vec = None  # type: Optional[Tensor]

        if not isinstance(rpu_config.device, ChoppedTransferCompound):
            raise ConfigError("Device chould be of type ChoppedTransferCompound")

        self.device_config = rpu_config.device  # type: ChoppedTransferCompound

        cfg = self.device_config
        rpu_config_0 = SingleRPUConfig(
            device=deepcopy(cfg.unit_cell_devices[0]),
            forward=deepcopy(cfg.transfer_forward),
            backward=deepcopy(cfg.transfer_forward),
            update=deepcopy(rpu_config.update),
            tile_class=AnalogTileWithoutPeriphery,
        )
        rpu_config_1 = SingleRPUConfig(
            device=deepcopy(cfg.unit_cell_devices[1]),
            forward=deepcopy(rpu_config.forward),
            backward=deepcopy(rpu_config.backward),
            update=deepcopy(cfg.transfer_update),
            tile_class=AnalogTileWithoutPeriphery,
        )

        self.grad_tile = rpu_config_0.tile_class(d_size, x_size, rpu_config_0)
        self.weight_tile = rpu_config_1.tile_class(d_size, x_size, rpu_config_1)
        self.from_weight_granularity = rpu_config_0.device.as_bindings(
            dtype
        ).calc_weight_granularity()
        self.to_weight_granularity = rpu_config_1.device.as_bindings(
            dtype
        ).calc_weight_granularity()

        lr_w = self.device_config.step * self.to_weight_granularity
        self.weight_tile.set_learning_rate(lr_w)

        transfer_columns = self.device_config.transfer_columns
        self.t_in_size = self.x_size if transfer_columns else self.d_size
        self.t_out_size = self.d_size if transfer_columns else self.x_size

        hidden_weight = zeros([self.t_in_size, self.t_out_size], dtype=dtype.as_torch())
        self.register_buffer("hidden_weight", hidden_weight)

        if isinstance(cfg, DynamicTransferCompound):
            past_mean_weight = zeros([self.t_in_size, self.t_out_size], dtype=dtype.as_torch())
            self.register_buffer("past_mean_weight", past_mean_weight)
            reference_weight = zeros([self.t_in_size, self.t_out_size], dtype=dtype.as_torch())
            self.register_buffer("reference_weight", reference_weight)

        chopper = ones([self.x_size], dtype=dtype.as_torch())
        self.register_buffer("chopper", chopper)

        # set auto-scale base scale
        granularity = rpu_config.update.desired_bl * self.from_weight_granularity
        self.lr_a_auto_scale = cfg.fast_lr * granularity

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

        return self.weight_tile.tile.forward(
            x_input, bias, in_trans, out_trans, is_test, non_blocking
        )

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
        return self.weight_tile.tile.backward(d_input, bias, in_trans, out_trans, non_blocking)

    def update(
        self,
        x_input: Tensor,
        d_input: Tensor,
        bias: bool = False,
        in_trans: bool = False,
        out_trans: bool = False,
        non_blocking: bool = False,
    ) -> Tensor:
        """Transfer update."""

        # pylint: disable=too-many-branches, too-many-statements
        # pylint: disable=attribute-defined-outside-init, too-many-locals

        cfg = self.device_config
        if in_trans or out_trans or bias:
            raise ConfigError("Trans or bias not supported ")
        if not cfg.n_reads_per_transfer == 1:
            raise ConfigError("Only 1 read per transfer supported")
        # just assume m batch for now
        if cfg.transfer_every < 0:
            raise ConfigError("Auto transfer every not supported")

        m_batch = 1
        if x_input.dim() > 1:
            m_batch = x_input.size(int(in_trans))

        transfer_every = cfg.transfer_every
        if cfg.units_in_mbatch:
            transfer_every = m_batch * transfer_every
        transfer_every = int(ceil(transfer_every))

        if isinstance(cfg, DynamicTransferCompound) and cfg.experimental_correct_accumulation:
            # also not supported is buffer_cap
            raise ConfigError("Correct accumulation is not supported")

        # dynamically adjust learning rates

        if cfg.auto_scale:
            lr_a = self.lr_a_auto_scale
            tau = (1.0 - cfg.auto_momentum) / m_batch

            m_x = x_input.abs().max().item()
            m_d = d_input.abs().max().item()

            self.m_x = (1 - tau) * self.m_x + tau * m_x  # type: ignore
            self.m_d = (1 - tau) * self.m_d + tau * m_d  # type: ignore

            if self.m_x > 0.0 and self.m_d > 0.0:
                lr_a /= self.m_x * self.m_d

        else:
            lr_a = cfg.fast_lr

        # gradient accumulation
        self.grad_tile.set_learning_rate(lr_a)

        if self.chopper is not None:
            x_input *= self.chopper

        self.grad_tile.tile.update(x_input, d_input, bias, in_trans, out_trans, non_blocking)
        self.update_counter += m_batch

        # handle transfer (Note that it will only update once for the m_batch!)
        rest_count = ((self.update_counter - m_batch) % transfer_every) + m_batch
        if rest_count >= transfer_every:
            if rest_count >= 2 * transfer_every:
                raise ConfigError(
                    "Multiple transfers within batch not supported"
                    " for the python transfer implementation."
                )

            # transfer learning rate lr_h
            buffer_granularity = cfg.buffer_granularity if cfg.buffer_granularity > 0.0 else 1.0
            if cfg.auto_granularity > 0.0:
                period = self.t_in_size * transfer_every
                buffer_granularity *= self.from_weight_granularity * cfg.auto_granularity / period
            else:
                buffer_granularity *= self.from_weight_granularity

            if cfg.correct_gradient_magnitudes:
                buffer_granularity *= self.to_weight_granularity / self.from_weight_granularity
                lr_h = self.learning_rate / lr_a / buffer_granularity
            else:
                lr_h = self.learning_rate / buffer_granularity

            chop_period = int(round(1.0 / cfg.in_chop_prob)) if cfg.in_chop_prob > 0 else 1

            if self._transfer_vec is None:
                # construct on the fly
                self._transfer_vec = eye(self.t_in_size, dtype=x_input.dtype, device=x_input.device)

            self.transfer_idx = (self.transfer_idx + 1) % self.t_in_size

            k = self.transfer_idx

            # read gradients
            if cfg.transfer_columns:
                omega = self.grad_tile.tile.forward(
                    self._transfer_vec[k], False, in_trans, out_trans, False, non_blocking
                )
            else:
                omega = self.grad_tile.tile.backward(
                    self._transfer_vec[k], False, in_trans, out_trans, non_blocking
                )

            # update hidden buffer
            if isinstance(cfg, DynamicTransferCompound):
                beta = min(cfg.tail_weightening / chop_period, 1)
                self.hidden_weight[k] += lr_h * self.chopper[k] * (omega - self.reference_weight[k])
                self.past_mean_weight[k] = (1 - beta) * self.past_mean_weight[k] + beta * omega
            else:
                self.hidden_weight[k] += lr_h * self.chopper[k] * omega

            # compute update weight
            write_values = -trunc(self.hidden_weight[k])  # negative because of update LR
            if cfg.forget_buffer:
                self.hidden_weight[k][write_values != 0] = cfg.momentum
            else:
                self.hidden_weight[k] += write_values * (1 - cfg.momentum)

            # handle chopper
            switch_chopper = False
            if cfg.in_chop_prob:
                if cfg.in_chop_random:
                    switch_chopper = rand(1).item() < cfg.in_chop_prob
                else:
                    if k == 0:
                        self.chop_counter = (self.chop_counter + 1) % chop_period
                    switch_chopper = self.chop_counter == 0

            if switch_chopper:
                self.chopper[k] = -self.chopper[k]

            # write to weight
            if not write_values.abs().sum() == 0.0:
                if cfg.transfer_columns:
                    self.weight_tile.tile.update(
                        self._transfer_vec[k],
                        write_values,
                        False,
                        in_trans,
                        out_trans,
                        non_blocking,
                    )
                else:
                    self.weight_tile.tile.update(
                        write_values,
                        self._transfer_vec[k],
                        False,
                        in_trans,
                        out_trans,
                        non_blocking,
                    )

            # additional compute for AGAD
            if isinstance(cfg, DynamicTransferCompound) and switch_chopper:
                self.reference_weight[k] = self.past_mean_weight[k]

    def get_brief_info(self) -> str:
        """Returns a brief info"""
        return self.__class__.__name__ + "({})".format(self.extra_repr())

    def get_weights(self) -> Tensor:
        """Returns the analog weights."""

        return self.weight_tile.tile.get_weights()

    def set_weights(self, weight: Tensor) -> None:
        """Stets the analog weights."""
        self.weight_tile.tile.set_weights(weight)

    def get_x_size(self) -> int:
        """Returns input size of tile"""
        return self.weight_tile.tile.get_x_size()

    def get_d_size(self) -> int:
        """Returns output size of tile"""
        return self.weight_tile.tile.get_d_size()

    def get_hidden_parameters(self) -> Tensor:
        """Get the hidden parameters of the tile.

        Returns:
            Hidden parameter tensor.
        """
        values_0 = self.grad_tile.tile.get_hidden_parameters()
        values_1 = self.weight_tile.tile.get_hidden_parameters()
        lst = [
            values_0,
            values_1,
            self.grad_tile.tile.get_weights()[None, :, :],
            self.weight_tile.tile.get_weights()[None, :, :],
        ]
        for name, buffer in self.named_buffers():
            if "weight" in name:
                buf = buffer.clone().cpu()
                if self.device_config.transfer_columns:
                    buf = buf.T
                lst.append(buf[None, :, :])
        return concatenate(lst, axis=0)

    def get_hidden_parameter_names(self) -> List[str]:
        """Get the hidden parameters names.

        Each name corresponds to a slice in the Tensor slice of the
        ``get_hidden_parameters`` tensor.

        Returns:
            List of names.
        """
        names = self.grad_tile.tile.get_hidden_parameter_names()
        names_0 = [n + "_0" for n in names]

        names = self.weight_tile.tile.get_hidden_parameter_names()
        names_1 = [n + "_1" for n in names]

        lst = names_0 + names_1 + ["fast_weight", "slow_weight"]
        for name, _ in self.named_buffers():
            if "weight" in name:
                lst.append(name)
        return lst

    def set_hidden_parameters(self, params: Tensor) -> None:
        """Set the hidden parameters of the tile."""
        names = self.get_hidden_parameter_names()
        n_base_params_0 = len(self.grad_tile.tile.get_hidden_parameter_names())
        n_base_params_1 = len(self.weight_tile.tile.get_hidden_parameter_names())
        values_0 = params[:n_base_params_0, :, :]
        values_1 = params[n_base_params_0 : n_base_params_0 + n_base_params_1, :, :]

        self.grad_tile.tile.set_hidden_parameters(values_0)
        self.weight_tile.tile.set_hidden_parameters(values_1)

        self.grad_tile.tile.set_weights(params[names.index("fast_weight")])
        self.weight_tile.tile.set_weights(params[names.index("slow_weight")])

        for name, buffer in self.named_buffers():
            if name in names:
                weight = params[names.index(name), :, :]
                if self.device_config.transfer_columns:
                    weight = weight.T
                buffer.data = weight

    def get_learning_rate(self) -> Optional[float]:
        """Get the learning rate of the tile.

        Returns:
           learning rate if exists.
        """
        return self.learning_rate

    def set_learning_rate(self, learning_rate: Optional[float]) -> None:
        """Set the learning rate of the tile.

        No-op for tiles that do not need a learning rate.

        Args:
           learning rate: learning rate to set
        """
        if learning_rate is None:
            return
        self.learning_rate = learning_rate

    def post_update_step(self) -> None:
        """Operators that need to be called once per mini-batch.

        Note:
            This function is called by the analog optimizer.

        Caution:
            If no analog optimizer is used, the post update steps will
            not be performed.
        """
        self.grad_tile.post_update_step()
        self.weight_tile.post_update_step()

    def dump_extra(self) -> Optional[Dict]:
        """Dumps any extra states / attributed necessary for
        checkpointing.

        For Tiles based on Modules, this should be normally handled by
        torch automatically.
        """
        return {
            "weight_tile": self.weight_tile.tile.dump_extra(),
            "grad_tile": self.grad_tile.tile.dump_extra(),
        }

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

        Raises:
            RuntimeError: in case keys are wrong
        """

        if "grad_tile" not in extra or "weight_tile" not in extra:
            raise RuntimeError("Wrong keys")
        self.weight_tile.load_extra(extra["weight_tile"], strict)
        self.grad_tile.load_extra(extra["grad_tile"], strict)

    def set_weights_uniform_random(self, bmin: float, bmax: float) -> None:
        """Sets the weights to uniform random numbers.

        Args:
           bmin: min value
           bmax: max value
        """
        self.weight_tile.tile.set_weights_uniform_random(bmin, bmax)

    def get_meta_parameters(self) -> Any:
        """Returns meta parameters."""
        return self.weight_tile.tile.get_meta_parameters()


class TorchTransferTile(TileModule, TileWithPeriphery, SimulatorTileWrapper):
    r"""Transfer tile for in-memory gradient accumulation algorithms.

    This is a (mostly) python re-implemetation of the
    :class:`~aihwkit.simulator.tiles.analog.AnalogTile` with
    :class:`~aihwkit.simulator.configs.compounds.ChoppedTransferCompound`` that is using the
    C++ RPUCuda library.

    Here only a subset of the parameters are implemented. However, all
    :class:`~aihwkit.simulator.configs.compounds.ChoppedTransferCompound`` as well as
    :class:`~aihwkit.simulator.configs.compounds.DynamicTransferCompound`` are implemented
    here.

    Thus, TTv2, c-TTv2, and AGAD learning algorithms are implemented here.

    Note:

        This implementation is for instructive use mostly. The C++ implementation has
        large speed advantage if the batch size is large and transfer is done multiple
        times per batch. For the torch implementation, at `transfer_every` needs to be
        larger or same size as the batch size, so that only one transfer is made per
        batch.

    Caution:

        When using ``model.analog_tiles()`` generators, this parent
        tile as well as the children tiles will be looped over, which
        might cause in e.g. getting the same weight twice. This is
        because ``TorchTransferTile`` is registered separately as an
        ``TileModule`` to support the periphery, while internally two
        additional tiles are instantiated.

    Usage::

        rpu_config = build_config('agad', device_config)
        # use the torch implementation tile instead of the default RPUCuda with AnalogTile
        rpu_config.tile_class = TorchTransferTile


    Args:
        out_size: output vector size of the tile, ie. the dimension of
            :math:`\mathbf{y}` in case of :math:`\mathbf{y} =
            W\mathbf{x}` (or equivalently the dimension of the
            :math:`\boldsymbol{\delta}` of the backward pass).
        in_size: input vector size, ie. the dimension of the vector
            :math:`\mathbf{x}` in case of :math:`\mathbf{y} =
            W\mathbf{x}`).

        rpu_config: resistive processing unit configuration. This has to be of type
            :class:`~aihwkit.simulator.configs.configs.UnitCellRPUConfig` with a device
            compound derived from
            :class:`~aihwkit.simulator.configs.compounds.ChoppedTransferCompound``.

        bias: whether to add a bias column to the tile, ie. :math:`W`
            has an extra column to code the biases. This is not supported here.
        in_trans: Whether to assume an transposed input (batch first). Not supported
        out_trans: Whether to assume an transposed output (batch first). Not supported

    Raises:
        ConfigError: if one of the not supported cases is used.

    """

    supports_indexed: bool = False
    supports_ddp: bool = False

    def __init__(
        self,
        out_size: int,
        in_size: int,
        rpu_config: RPUConfigGeneric,
        bias: bool = False,
        in_trans: bool = False,
        out_trans: bool = False,
    ):
        TileModule.__init__(self)
        SimulatorTileWrapper.__init__(
            self, out_size, in_size, rpu_config, bias, in_trans, out_trans, ignore_analog_state=True
        )
        TileWithPeriphery.__init__(self)

    def _create_simulator_tile(
        self, x_size: int, d_size: int, rpu_config: RPUConfigGeneric
    ) -> TransferSimulatorTile:
        """Create a simulator tile.

        Args:
            x_size: input size
            d_size: output size
            rpu_config: resistive processing unit configuration

        Returns:
            a simulator tile based on the specified configuration.
        """

        if not isinstance(rpu_config, UnitCellRPUConfig):
            raise ConfigError("Expect an UnitCellRPUConfig.")

        return TransferSimulatorTile(x_size, d_size, rpu_config, dtype=self.get_data_type())

    def forward(
        self, x_input: Tensor, tensor_view: Optional[Tuple] = None  # type: ignore
    ) -> Tensor:
        """Torch forward function that calls the analog forward"""
        # pylint: disable=arguments-differ

        out = AnalogFunction.apply(
            self.get_analog_ctx(), self, x_input, self.shared_weights, not self.training
        )

        if tensor_view is None:
            tensor_view = self.get_tensor_view(out.dim())
        out = self.apply_out_scaling(out, tensor_view)

        if self.digital_bias:
            return out + self.bias.view(*tensor_view)
        return out

    @no_grad()
    def post_update_step(self) -> None:
        """Operators that need to be called once per mini-batch.

        Note:
            This function is called by the analog optimizer.

        Caution:
            If no analog optimizer is used, the post update steps will
            not be performed.
        """
        self.tile.post_update_step()

    def replace_with(self, rpu_config: RPUConfigBase) -> None:
        """Replacing the current `RPUConfig` is not supported.

        Args:
            rpu_config: New `RPUConfig` to check against

        Raises:
            TileModuleError: always
        """
        raise NotImplementedError
