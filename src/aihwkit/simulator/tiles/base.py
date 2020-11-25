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

"""High level analog tiles (base)."""

from collections import OrderedDict
from typing import Dict, Generic, List, Optional, Tuple, TypeVar, Union

from numpy import concatenate, expand_dims
from torch import device as torch_device
from torch import stack, Tensor
from torch.autograd import no_grad

from aihwkit.exceptions import TileError
from aihwkit.simulator.rpu_base import tiles

RPUConfigGeneric = TypeVar('RPUConfigGeneric')


class BaseTile(Generic[RPUConfigGeneric]):
    """Base class for tiles.

    Args:
        out_size: output size
        in_size: input size
        rpu_config: resistive processing unit configuration.
        bias: whether to add a bias column to the tile.
        in_trans: Whether to assume an transposed input (batch first)
        out_trans: Whether to assume an transposed output (batch first)
    """
    # pylint: disable=too-many-instance-attributes

    is_cuda = False

    def __init__(
            self,
            out_size: int,
            in_size: int,
            rpu_config: RPUConfigGeneric,
            bias: bool = True,
            in_trans: bool = False,
            out_trans: bool = False
    ):
        self.out_size = out_size
        self.in_size = in_size
        self.rpu_config = rpu_config
        self.bias = bias
        self.in_trans = in_trans
        self.out_trans = out_trans

        # Only used for indexed.
        self.image_sizes = []  # type: List[int]

        x_size = in_size + 1 if self.bias else in_size
        d_size = out_size

        # Cuda tiles are assumed to init `self.tile` manually.
        if self.is_cuda:
            self.tile = None  # type: Union[tiles.FloatingPointTile, tiles.AnalogTile]
        else:
            self.tile = self._create_simulator_tile(x_size, d_size, rpu_config)
            self.tile.set_learning_rate(0.01)
            self.tile.set_weights_uniform_random(-0.01, 0.01)

        self.device = torch_device('cpu')

    def __getstate__(self) -> Dict:
        """Get the state for pickling.

        This method removes the ``tile`` member, as the binding Tiles are not
        serializable.
        """
        current_dict = self.__dict__.copy()

        current_dict['analog_tile_weights'] = self.tile.get_weights()
        # Store the hidden parameters as a numpy array, as storing it as
        # Tensor causes issues in PyTorch 1.5.
        current_dict['analog_tile_hidden_parameters'] = \
            self.tile.get_hidden_parameters().data.numpy()
        current_dict.pop('tile')
        current_dict.pop('stream', None)

        return current_dict

    def __setstate__(self, state: Dict) -> None:
        """Set the state after unpickling.

        This method recreates the ``tile`` member, creating a new one from
        scratch, as the binding Tiles are not serializable.
        """
        current_dict = state.copy()
        weights = current_dict.pop('analog_tile_weights')
        hidden_parameters = current_dict.pop('analog_tile_hidden_parameters')

        self.__dict__.update(current_dict)

        x_size = self.in_size + 1 if self.bias else self.in_size
        d_size = self.out_size

        # Recreate the tile.
        self.tile = self._create_simulator_tile(x_size, d_size, self.rpu_config)
        self.tile.set_hidden_parameters(Tensor(hidden_parameters))
        self.tile.set_weights(weights)

    def _create_simulator_tile(
            self,
            x_size: int,
            d_size: int,
            rpu_config: RPUConfigGeneric
    ) -> Union[tiles.FloatingPointTile, tiles.AnalogTile]:
        """Create a simulator tile.

        Args:
            x_size: input size
            d_size: output size
            rpu_config: resistive processing unit configuration

        Returns:
            a simulator tile based on the specified configuration.
        """
        raise NotImplementedError

    def set_weights(
            self,
            weights: Tensor,
            biases: Optional[Tensor] = None,
            realistic: bool = False,
            n_loops: int = 10
    ) -> None:
        """Set the tile weights (and biases).

        Sets the internal tile weights to the specified values, and also the
        internal tile biases if the tile was set to use bias (via
        ``self.bias``).

        Note:
            By default this is **not** hardware realistic. You can set the
            ``realistic`` parameter to ``True`` for a realistic transfer.

        Args:
            weights: ``[out_size, in_size]`` weight matrix.
            biases: ``[out_size]`` bias vector. This parameter is required if
                ``self.bias`` is ``True``, and ignored otherwise.
            realistic: whether to use the forward and update pass to
                program the weights iteratively, using
                :meth:`set_weights_realistic`.
            n_loops: number of times the columns of the weights are set in a
                closed-loop manner.
                A value of ``1`` means that all columns in principle receive
                enough pulses to change from ``w_min`` to ``w_max``.

        Returns:
            None.

        Raises:
            ValueError: if the tile has bias but ``bias`` has not been
                specified.
        """
        # Prepare the array expected by the pybind function, appending the
        # biases row if needed.
        weights_numpy = weights.clone().detach().cpu().numpy()

        if self.bias:
            # Create a ``[out_size, in_size (+ 1)]`` matrix.
            if biases is None:
                raise ValueError("Analog tile has a bias, but no bias given")

            biases_numpy = expand_dims(biases.clone().detach().cpu().numpy(), 1)
            combined_weights = concatenate([weights_numpy, biases_numpy], axis=1)
        else:
            # Use only the ``[out_size, in_size]`` matrix.
            combined_weights = weights_numpy

        if realistic:
            return self.tile.set_weights_realistic(combined_weights, n_loops)

        return self.tile.set_weights(combined_weights)

    def get_weights(
            self,
            realistic: bool = False
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Get the tile weights (and biases).

        Gets the tile weights and extracts the mathematical weight
        matrix and biases (if present, by determined by the ``self.bias``
        parameter).

        Note:
            By default this is **not** hardware realistic. Use set
            ``realistic`` to True for a realistic transfer.

        Args:
            realistic: Whether to use the forward pass to
              read out the tile weights iteratively, using
              :meth:`get_weights_realistic`

        Returns:
           a tuple where the first item is the ``[out_size, in_size]`` weight
           matrix; and the second item is either the ``[out_size]`` bias vector
           or ``None`` if the tile is set not to use bias.
        """
        # Retrieve the internal weights (and potentially biases) matrix.
        if realistic:
            combined_weights = self.tile.get_weights_realistic()
        else:
            combined_weights = self.tile.get_weights()

        # Split the internal weights (and potentially biases) matrix.
        if self.bias:
            # combined_weights is [out_size, in_size (+ 1)].
            weights = Tensor(combined_weights[:, :-1])
            biases = Tensor(combined_weights[:, -1])
        else:
            # combined_weights is [out_size, in_size].
            weights = Tensor(combined_weights)
            biases = None

        return weights.to(self.device), biases.to(self.device) if self.bias else None

    def set_learning_rate(self, learning_rate: float) -> None:
        """Set the tile learning rate.

        Set the tile learning rate to ``-learning_rate``. Note that the
        learning rate is always taken to be negative (because of the meaning in
        gradient descent) and positive learning rates are not supported.

        Args:
            learning_rate: the desired learning rate.

        Returns:
            None.
        """
        return self.tile.set_learning_rate(learning_rate)

    def get_learning_rate(self) -> float:
        """Return the tile learning rate.

        Returns:
            float: the tile learning rate.
        """
        return self.tile.get_learning_rate()

    def decay_weights(self, alpha: float = 1.0) -> None:
        """Decays the weights once.

        Args:
           alpha: additional decay scale (such as LR). The base decay
              rate is set during tile init.

        Returns:
            None.
        """
        return self.tile.decay_weights(alpha)

    def diffuse_weights(self) -> None:
        """Diffuses the weights once.

        The base diffusion rate is set during tile init.

        Returns:
            None
        """
        return self.tile.diffuse_weights()

    def reset_columns(
            self,
            start_column_idx: int = 0,
            num_columns: int = 1,
            reset_prob: float = 1.0
    ) -> None:
        r"""Reset (a number of) columns.

        Resets the weights with device-to-device and cycle-to-cycle
        variability (depending on device type), typically:

        .. math::
            W_{ij} = \xi*\sigma_\text{reset} + b^\text{reset}_{ij}

        The reset parameters are set during tile init.

        Args:
            start_column_idx: a start index of columns (0..x_size-1)
            num_columns: how many consecutive columns to reset (with circular warping)
            reset_prob: individual probability of reset.

        Returns:
            None
        """
        return self.tile.reset_columns(start_column_idx, num_columns, reset_prob)

    def cuda(
            self,
            device: Optional[Union[torch_device, str, int]] = None
    ) -> 'BaseTile':
        """Return a copy of this tile in CUDA memory."""
        raise NotImplementedError

    @no_grad()
    def forward(self, x_input: Tensor, is_test: bool = False) -> Tensor:
        """Perform the forward pass.

        Args:
            x_input: ``[N, in_size]`` tensor. If ``in_trans`` is set, transposed.
            is_test: whether to assume testing mode.

        Returns:
            torch.Tensor: ``[N, out_size]`` tensor. If ``out_trans`` is set, transposed.
        """
        # We use no-grad as we do it explicitly in the optimizer.
        return self.tile.forward(x_input, self.bias,
                                 self.in_trans, self.out_trans, is_test)

    def backward(self, d_input: Tensor) -> Tensor:
        """Perform the backward pass.

        Args:
            d_input: ``[N, out_size]`` tensor. If ``out_trans`` is set, transposed.

        Returns:
            torch.Tensor: ``[N, in_size]`` tensor. If ``in_trans`` is set, transposed.
        """
        return self.tile.backward(d_input, self.bias, self.out_trans, self.in_trans)

    def update(self, x_input: Tensor, d_input: Tensor) -> None:
        """Perform the update pass.

        Args:
            x_input: ``[N, in_size]`` tensor. If ``in_trans`` is set, transposed.
            d_input: ``[N, out_size]`` tensor. If ``out_trans`` is set, transposed.

        Returns:
            None
        """
        return self.tile.update(x_input, d_input, self.bias,
                                self.in_trans, self.out_trans)

    def get_hidden_parameters(self) -> OrderedDict:
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

    def set_hidden_parameters(self, ordered_parameters: OrderedDict) -> None:
        """Set the hidden parameters of the tile.

        Args:
            ordered_parameters: Ordered dictionary of hidden parameter tensors.
        """
        if len(ordered_parameters) == 0:
            return

        hidden_parameters = stack(list(ordered_parameters.values()), dim=0)
        self.tile.set_hidden_parameters(hidden_parameters)

    def get_hidden_update_index(self) -> int:
        """Get the current updated device index of the hidden devices.

        Usually this is 0 as only one device is present per
        cross-point for many tile RPU configs. However, some RPU
        configs maintain internally multiple devices per cross-point
        (e.g. :class:`~aihwkit.simulator.config.devices.VectorUnitCell`).

        Returns:
            The next mini-batch updated device index.

        Note:
            Depending on the update and learning policy implemented
            in the tile, updated devices might switch internally as
            well.
        """
        return self.tile.get_hidden_update_index()

    def set_hidden_update_index(self, index: int) -> None:
        """set the current updated hidden device index.

        Usually this is ignored and fixed to 0 as only one device is
        present per cross-point. Other devices, might not allow
        explicit setting as it would interfere with the implemented
        learning However rule. However, some tiles have internally
        multiple devices per cross-point (eg. unit cell) that can be
        chosen depending on the update policy.

        Args:
            index: device index to be updated in the next mini-batch

        Note:
            Depending on the update and learning policy implemented
            in the tile, updated devices might switch internally as
            well.
        """
        self.tile.set_hidden_update_index(index)

    def set_indexed(self, indices: Tensor, image_sizes: List) -> None:
        """Sets the index matrix for convolutions ans switches to
        indexed forward/backward/update versions.

        Args:
            indices : torch.tensor with int indices
            image_sizes: [C_in, H_in, W_in, H_out, W_out] sizes

        Raises:
            ValueError: if ``image_sizes`` does not have valid dimensions.
            TileError: if the tile uses transposition.
        """
        if len(image_sizes) != 5:
            raise ValueError('image_sizes expects 5 sizes [C_in, H_in, W_in, H_out, W_out]')

        if self.in_trans or self.out_trans:
            raise TileError('Transposed indexed versions not supported (assumes NCHW)')

        self.image_sizes = image_sizes
        self.tile.set_matrix_indices(indices)

    @no_grad()
    def forward_indexed(self, x_input: Tensor, is_test: bool = False) -> Tensor:
        """Perform the forward pass for convolutions.

        Args:
            x_input: ``[N, in_size]`` tensor. If ``in_trans`` is set, transposed.
            is_test: whether to assume testing mode.

        Returns:
            torch.Tensor: ``[N, out_size]`` tensor. If ``out_trans`` is set, transposed.

        Raises:
            TileError: if the indexed tile has not been initialized.
        """
        if not self.image_sizes:
            raise TileError('self.image_sizes is not initialized. Please use '
                            'set_indexed()')

        _, _, _, height_out, width_out = self.image_sizes
        return self.tile.forward_indexed(x_input, height_out, width_out, is_test)

    def backward_indexed(self, d_input: Tensor) -> Tensor:
        """Perform the backward pass for convolutions.

        Args:
            d_input: ``[N, out_size]`` tensor. If ``out_trans`` is set, transposed.

        Returns:
            torch.Tensor: ``[N, in_size]`` tensor. If ``in_trans`` is set, transposed.
        """
        channel_in, height_in, width_in, _, _ = self.image_sizes
        return self.tile.backward_indexed(d_input, channel_in, height_in, width_in)

    def update_indexed(self, x_input: Tensor, d_input: Tensor) -> None:
        """Perform the update pass for convolutions.

        Args:
            x_input: ``[N, in_size]`` tensor. If ``in_trans`` is set, transposed.
            d_input: ``[N, out_size]`` tensor. If ``out_trans`` is set, transposed.

        Returns:
            None
        """
        return self.tile.update_indexed(x_input, d_input)

    @no_grad()
    def post_update_step(self) -> None:
        """Operators that need to be called once per mini-batch."""
        if self.rpu_config.device.requires_diffusion():  # type: ignore
            self.diffuse_weights()
        if self.rpu_config.device.requires_decay():  # type: ignore
            self.decay_weights()
