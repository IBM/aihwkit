# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022 IBM. All Rights Reserved.
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
from copy import deepcopy

from torch import (
    Tensor, stack, zeros, as_tensor, cat, unsqueeze, squeeze, ones_like
)
from torch import device as torch_device
from torch import max as torch_max
from torch.nn import Parameter
from torch.autograd import no_grad

from aihwkit.exceptions import TileError
from aihwkit.simulator.rpu_base import tiles
from aihwkit.optim.context import AnalogContext

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
    # pylint: disable=too-many-instance-attributes,too-many-public-methods

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
        self.rpu_config = deepcopy(rpu_config)
        self.bias = bias
        self.in_trans = in_trans
        self.out_trans = out_trans
        self.shared_weights = None  # type: Parameter
        self.out_scaling_alpha = None  # type: Parameter

        # Only used for indexed.
        self.image_sizes = []  # type: List[int]

        x_size = in_size + 1 if self.bias else in_size
        d_size = out_size

        self.tile = self._create_simulator_tile(x_size, d_size, rpu_config)
        self.tile.set_learning_rate(0.01)
        self.tile.set_weights_uniform_random(-0.01, 0.01)

        self.device = torch_device('cpu')
        self.is_cuda = False

        # create analog context
        self.analog_ctx = AnalogContext(self)

    @no_grad()
    def get_analog_ctx(self) -> AnalogContext:
        """Return the analog context of the tile to be used in ``AnalogFunction``."""
        return self.analog_ctx

    @no_grad()
    def ensure_shared_weights(self, shared_weights: Optional[Tensor] = None) -> None:
        """Ensure that the shared_weights is set properly.

        Caution:
           This is only called from analog function.

        No-op if shared weights is not used.
        """
        if shared_weights is not None:
            self.shared_weights.data = shared_weights.data  # type: ignore

        if self.shared_weights is not None:
            self.tile.set_shared_weights(self.shared_weights.data)

    @no_grad()
    def set_delta_weights(self, delta_weights: Optional[Tensor] = None) -> None:
        """Set the weight grad tensor and set the update to.

        No-op if shared weights is not used.
        """
        if self.shared_weights is not None and delta_weights is not None:
            self.tile.set_delta_weights(delta_weights)

    @no_grad()
    def reset_delta_weights(self) -> None:
        """Reset the weight grad tensor to default update behavior (i.e. adding the
        update directly to the weight).

        No-op if shared weights is not used.
        """
        if self.shared_weights is not None:
            self.tile.reset_delta_weights()

    @no_grad()
    def get_brief_info(self) -> str:
        """Return short info about the underlying C++ tile."""
        return self.tile.get_brief_info().rstrip()

    def __getstate__(self) -> Dict:
        """Get the state for pickling.

        This method removes the ``tile`` member, as the binding Tiles are not
        serializable.
        """
        current_dict = self.__dict__.copy()

        current_dict['analog_tile_weights'] = self.tile.get_weights()
        # Store the hidden parameters as a numpy array, as storing it as
        # Tensor causes issues in PyTorch 1.5.
        current_dict['analog_tile_hidden_parameters'] \
            = self.tile.get_hidden_parameters().data.numpy()
        current_dict['analog_tile_hidden_parameter_names'] \
            = self.tile.get_hidden_parameter_names()
        current_dict['analog_tile_class'] = self.__class__.__name__
        current_dict['analog_lr'] = self.tile.get_learning_rate()
        current_dict['shared_weights'] = self.shared_weights
        current_dict.pop('tile', None)

        # don't save device. Will be determined by loading object
        current_dict.pop('stream', None)
        current_dict.pop('is_cuda', None)
        current_dict.pop('device', None)

        return current_dict

    def __setstate__(self, state: Dict) -> None:
        """Set the state after unpickling.

        This method recreates the ``tile`` member, creating a new one from
        scratch, as the binding Tiles are not serializable.

        Caution:
            RPU configs are overwritten by loading the state.

        Raises:
            TileError: if tile class does not match or hidden parameters do not match
        """
        # pylint: disable=too-many-locals

        # Note: self here is NOT initialized! So we need to recreate
        # attributes that were not saved in getstate

        current_dict = state.copy()
        weights = current_dict.pop('analog_tile_weights')
        hidden_parameters = current_dict.pop('analog_tile_hidden_parameters')
        hidden_parameters_names = current_dict.pop('analog_tile_hidden_parameter_names', [])
        alpha_scale = current_dict.pop('analog_alpha_scale', None)
        tile_class = current_dict.pop('analog_tile_class', self.__class__.__name__)
        analog_lr = current_dict.pop('analog_lr', 0.01)
        analog_ctx = current_dict.pop('analog_ctx')
        shared_weights = current_dict.pop('shared_weights')
        shared_weights_if = shared_weights is not None

        self.__dict__.update(current_dict)

        self.device = torch_device('cpu')
        self.is_cuda = False
        # get the current map location from analog_ctx (which is restored)
        to_device = analog_ctx.device

        # recreate attributes not saved
        # always first create on CPU
        x_size = self.in_size + 1 if self.bias else self.in_size
        d_size = self.out_size

        # Recreate the tile.
        # Check for tile mismatch
        if tile_class != self.__class__.__name__:
            raise TileError(
                'Mismatch of tile class: {} versus {}. Can only load analog '
                'state from the same tile class.'.format(self.__class__.__name__, tile_class))

        self.tile = self._create_simulator_tile(x_size, d_size, self.rpu_config)
        names = self.tile.get_hidden_parameter_names()
        if len(hidden_parameters_names) > 0 and names != hidden_parameters_names:
            # Check whether names match
            raise TileError('Mismatch with loaded analog state: '
                            'Hidden parameter structure is unexpected.')
        self.tile.set_hidden_parameters(Tensor(hidden_parameters))
        self.tile.set_weights(weights)

        self.tile.set_learning_rate(analog_lr)

        # re-generate shared weights (CPU)
        if shared_weights_if:
            if not hasattr(self, 'shared_weights'):
                # this is needed when pkl loading
                self.shared_weights = shared_weights

            with no_grad():
                # always new will be populated with set weights.
                self.shared_weights.data = zeros(d_size, x_size, requires_grad=True)
            self.ensure_shared_weights()
        else:
            self.shared_weights = None

        # Regenerate context but keep the object ID
        if not hasattr(self, 'analog_ctx'):  # when loading
            self.analog_ctx = AnalogContext(self, parameter=analog_ctx)
        self.analog_ctx.reset(self)
        self.analog_ctx.set_data(analog_ctx.data)

        if to_device.type.startswith('cuda'):
            self.cuda(to_device)

        if alpha_scale is not None:
            # legacy. We apply the alpha scale instaed of the
            # out_scaling_alpha when loading. The alpha_scale
            # mechansim is now replaced with the out scaling factors
            #
            # Caution: will overwrite the loaded out_scaling_alphas
            # if they would exist also (should not be for old checkpoints)

            self.set_out_scaling_alpha(alpha_scale)

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
        weights_torch = weights.clone().detach().cpu()

        if self.bias:
            # Create a ``[out_size, in_size (+ 1)]`` matrix.
            if biases is None:
                raise ValueError('Analog tile has a bias, but no bias given')

            biases_torch = unsqueeze(biases.clone().detach().cpu(), 1)
            combined_weights = cat((weights_torch, biases_torch), dim=1)
        else:
            # Use only the ``[out_size, in_size]`` matrix.
            combined_weights = weights_torch

        if realistic:
            return self.tile.set_weights_realistic(combined_weights.numpy(), n_loops)

        return self.tile.set_weights(combined_weights.numpy())

    def get_weights(self, realistic: bool = False) -> Tuple[Tensor, Optional[Tensor]]:
        """Get the tile weights (and biases).

        Gets the tile weights and extracts the mathematical weight
        matrix and biases (if present, by determined by the ``self.bias``
        parameter).

        Note:
            By default this is **not** hardware realistic. Use set
            ``realistic`` to True for a realistic transfer.

        Args:
            realistic: Whether to use the forward pass to read out the tile
                weights iteratively, using :meth:`get_weights_realistic`.

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

        return weights, biases if self.bias else None

    def set_weights_scaled(
            self,
            weights: Tensor,
            biases: Optional[Tensor] = None,
            realistic: bool = False,
            n_loops: int = 10,
            weight_scaling_omega: Optional[float] = None
    ) -> None:
        r"""Set the tile weights (and biases) in a scaled fashion.

        Similar to :meth:`set_weights`, however, additionally scales the weights
        by a global scale :math:`\alpha`, that is then applied in digital at the
        output of forward and backward pass, and the learning rate for this tile
        is adjusted accordingly.

        The weights are scaled by :math:`\omega/\max_{ij} |w_{ij}|` and the global
        digital factor :math:`alpha` is set to :math:`\max_{ij} |w_{ij}|/\omega`.

        It can be shown that such a constant factor greatly improves the SNR and
        training accuracy as the full weight range of the analog devices are
        used. See also `Rasch, Gokmen & Haensch (2019)`_ for more details.

        Caution:
            Using ``get_weights`` will now retrieve the true analog weights
            *without* applying the global factor. To get the true weights, use
            ``get_weights`` and scale it by the :math:`\alpha` of this layer
            which can be retrieved by ``get_alpha_scale()``.

        Args:
            weights: ``[out_size, in_size]`` weight matrix.
            biases: ``[out_size]`` bias vector. This parameter is required if
                ``self.bias`` is ``True``, and ignored otherwise.
            realistic: whether to use the forward and update pass to program the
                weights iteratively, using :meth:`set_weights_realistic`.
            n_loops: number of times the columns of the weights are set in a
                closed-loop manner.
                A value of ``1`` means that all columns in principle receive
                enough pulses to change from ``w_min`` to ``w_max``.
            weight_scaling_omega: where the weight max should be mapped in terms of
                the weight range. Note that for ``omega`` larger than
                the maximal weight of the device, weights will get
                clipped for most devices. If this parameter is not
                given, it will default to the ``weight_scaling_omega``
                value set in the
                :class:`~aihwkit.configs.utils.MappingParameter` of the
                ``rpu_config``

        Returns:
            None.

        Raises:
            ValueError: if the tile has bias but ``bias`` has not been
                specified.

        .. _`Rasch, Gokmen & Haensch (2019)`: https://arxiv.org/abs/1906.02698

        """
        # Prepare the array expected by the pybind function, appending the
        # biases row if needed.
        weights_torch = weights.clone().detach().cpu()

        if self.bias:
            # Create a ``[out_size, in_size (+ 1)]`` matrix.
            if biases is None:
                raise ValueError('Analog tile has a bias, but no bias given')

            biases_torch = unsqueeze(biases.clone().detach().cpu(), 1)
            combined_weights = cat((weights_torch, biases_torch), dim=1)
        else:
            # Use only the ``[out_size, in_size]`` matrix.
            combined_weights = weights_torch

        mapping = self.rpu_config.mapping  # type: ignore
        omega = weight_scaling_omega
        if omega is None:
            omega = mapping.weight_scaling_omega

        # Apply the scaling
        if mapping.weight_scaling_omega_columnwise:
            weight_max, _ = torch_max(abs(combined_weights), 1, keepdim=True)
        else:
            weight_max = torch_max(abs(combined_weights)).view(1)

        if omega > 0:
            alpha = weight_max / omega
        elif mapping.learn_out_scaling_alpha:
            alpha = ones_like(weight_max)
        else:
            alpha = None

        if alpha is not None:
            combined_weights = combined_weights / alpha

        self.set_out_scaling_alpha(alpha)

        # update the mapping field
        self.rpu_config.mapping.weight_scaling_omega = omega  # type: ignore

        if realistic:
            return self.tile.set_weights_realistic(combined_weights.numpy(), n_loops)
        return self.tile.set_weights(combined_weights.numpy())

    def get_weights_scaled(self, realistic: bool = False) -> Tuple[Tensor, Optional[Tensor]]:
        """Get the tile weights (and biases) and applies the current alpha
        scale to it.

        Gets the tile weights and extracts the mathematical weight
        matrix and biases (if present, by determined by the ``self.bias``
        parameter).

        Note:
            By default this is **not** hardware realistic. Use set
            ``realistic`` to True for a realistic transfer.

        Args:
            realistic: Whether to use the forward pass to read out the tile
                weights iteratively, using :meth:`get_weights_realistic`.

        Returns:
            tuple: where the first item is the ``[out_size, in_size]`` weight
                matrix; and the second item is either the ``[out_size]`` bias vector
                or ``None`` if the tile is set not to use bias. Both have the alpha
                scale applied.
        """
        weights, biases = self.get_weights(realistic=realistic)

        if self.out_scaling_alpha is None:
            return weights, biases

        alpha = self.out_scaling_alpha.clone().detach().cpu()
        return weights * alpha.view(-1, 1), biases * alpha if self.bias else None

    def get_out_scaling_alpha(self) -> Tensor:
        """Get the out_scaling_alpha used to scale the weights

        Returns:
            tensor: out_scaling_alpha
            """
        return self.out_scaling_alpha

    def set_out_scaling_alpha(self, alpha: Union[Tensor, float]) -> None:
        """Helper function to set the out scaling alpha used to scale the
        weights in digital.

        Args:
            alpha: out scaling alpha scale as a tensor or float
                value (depending on the property set by in the
                :class:`~aihwkit.configs.utils.MappingParameter`
                configurations

        Caution:
            Will not check the correct size of the given alpha.
        """
        if alpha is None:
            self.out_scaling_alpha = None
        elif isinstance(self.out_scaling_alpha, Parameter):
            self.out_scaling_alpha.data = squeeze(as_tensor(alpha)).to(self.device)
        else:
            self.out_scaling_alpha = squeeze(as_tensor(alpha)).to(self.device)

    def apply_out_scaling(self, values: Tensor, tensor_view: Tuple[int, ...] = (-1, )) -> Tensor:
        """Apply the out scaling to the given tensor.

        Args:
            values: tensor to apply the out scaling alphas to.
            tensor_view: view to cast the out scaling alphas before multiplication

        Returns:
            output tensor with applied out scaling factors
        """
        if self.out_scaling_alpha is not None:
            return values * self.out_scaling_alpha.view(*tensor_view)
        return values

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

    @no_grad()
    def decay_weights(self, alpha: float = 1.0) -> None:
        """Decays the weights once according to the decay parameters of the tile.

        Args:
            alpha: additional decay scale (such as LR). The base decay
                rate is set during tile init.

        Returns:
            None.
        """
        return self.tile.decay_weights(alpha)

    @no_grad()
    def drift_weights(self, delta_t: float = 1.0) -> None:
        """Drifts the weights once according to the drift parameters of the
        tile.

        See also :class:`~aihwkit.simulator.configs.utils.DriftParameter`.

        Args:
            delta_t: Time since last drift call.

        Returns:
            None.
        """
        return self.tile.drift_weights(delta_t)

    @no_grad()
    def diffuse_weights(self) -> None:
        """Diffuses the weights once according to the diffusion parameters of
        the tile.

        The base diffusion rate is set during tile init.

        Returns:
            None
        """
        return self.tile.diffuse_weights()

    @no_grad()
    def reset_columns(
            self,
            start_column_idx: int = 0,
            num_columns: int = 1,
            reset_prob: float = 1.0
    ) -> None:
        r"""Reset (a number of) columns according to the reset parameters of the tile.

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

    def cpu(self) -> 'BaseTile':
        """Return a copy of this tile in CPU memory."""
        raise NotImplementedError

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

        Caution:
            Usually the hidden parameters are drawn according to the
            parameter definitions (those given in the RPU config). If
            the hidden parameters are arbitrary set by the user, then
            this correspondence might be broken. This might cause problems
            in the learning, in particular, the `weight granularity`
            (usually ``dw_min``, depending on the device) is needed for
            the dynamic adjustment of the bit length
            (``update_bl_management``, see
            :class:`~aihwkit.simulator.configs.utils.UpdateParameters`).

            Currently, the new ``dw_min`` parameter is tried to be
            estimated from the average of hidden parameters if the
            discrepancy with the ``dw_min`` from the definition is too
            large.

        Args:
            ordered_parameters: Ordered dictionary of hidden parameter tensors.

        Raises:
            TileError: In case the ordered dict keys do not conform
                with the current rpu config tile structure of the hidden
                parameters
        """
        if len(ordered_parameters) == 0:
            return

        hidden_parameters = stack(list(ordered_parameters.values()), dim=0)
        names = self.tile.get_hidden_parameter_names()
        if names != list(ordered_parameters.keys()):
            raise TileError('Mismatch with loaded analog state:'
                            'Hidden parameter structure is unexpected.')

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
        """Set the current updated hidden device index.

        Usually this is ignored and fixed to 0 as only one device is
        present per cross-point. Other devices, might not allow
        explicit setting as it would interfere with the implemented
        learning rule. However, some tiles have internally
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
        """Set the index matrix for convolutions ans switches to
        indexed forward/backward/update versions.

        Args:
            indices : torch.tensor with int indices
            image_sizes: [C_in, H_in, W_in, H_out, W_out] sizes

        Raises:
            ValueError: if ``image_sizes`` does not have valid dimensions.
            TileError: if the tile uses transposition.
        """
        if len(image_sizes) not in (3, 5, 7):
            raise ValueError('image_sizes expects 3, 5 or 7 sizes '
                             '[C_in, (D_in), H_in, (W_in), (D_out), H_out, (W_out)]')

        if self.in_trans or self.out_trans:
            raise TileError('Transposed indexed versions not supported (assumes NC(D)HW)')

        self.image_sizes = image_sizes
        self.tile.set_matrix_indices(indices)

    @no_grad()
    def forward_indexed(self, x_input: Tensor, is_test: bool = False) -> Tensor:
        """Perform the forward pass for convolutions.

        Depending on the input tensor size it performs the forward pass for a
        2D image or a 3D one.

        Args:
            x_input: ``[N, in_size]`` tensor. If ``in_trans`` is set, transposed.
            is_test: whether to assume testing mode.

        Returns:
            torch.Tensor: ``[N, out_size]`` tensor. If ``out_trans`` is set, transposed.

        Raises:
            TileError: if the indexed tile has not been initialized, or if
                ``self.images_sizes`` does not have a valid dimennion.
        """
        if not self.image_sizes:
            raise TileError('self.image_sizes is not initialized. Please use '
                            'set_indexed()')

        n_batch = x_input.size(0)
        channel_out = self.out_size

        if len(self.image_sizes) == 3:
            _, _, height_out = self.image_sizes
            d_tensor = x_input.new_empty((n_batch, channel_out, height_out))
        elif len(self.image_sizes) == 5:
            _, _, _, height_out, width_out = self.image_sizes
            d_tensor = x_input.new_empty((n_batch, channel_out, height_out, width_out))
        elif len(self.image_sizes) == 7:
            _, _, _, _, depth_out, height_out, width_out = self.image_sizes
            d_tensor = x_input.new_empty((n_batch, channel_out, depth_out, height_out, width_out))
        else:
            raise TileError('self.image_sizes length is not 3, 5 or 7')

        return self.tile.forward_indexed(x_input, d_tensor, is_test)

    def backward_indexed(self, d_input: Tensor) -> Tensor:
        """Perform the backward pass for convolutions.

        Depending on the input tensor size it performs the backward pass for a
        2D image or a 3D one.

        Args:
            d_input: ``[N, out_size]`` tensor. If ``out_trans`` is set, transposed.

        Returns:
            torch.Tensor: ``[N, in_size]`` tensor. If ``in_trans`` is set, transposed.

        Raises:
            TileError: if the indexed tile has not been initialized, or if
                ``self.images_sizes`` does not have a valid dimennion.
        """
        if not self.image_sizes:
            raise TileError('self.image_sizes is not initialized. Please use '
                            'set_indexed()')

        n_batch = d_input.size(0)

        if len(self.image_sizes) == 3:
            channel_in, height_in, _ = self.image_sizes
            x_tensor = d_input.new_empty((n_batch, channel_in, height_in))
        elif len(self.image_sizes) == 5:
            channel_in, height_in, width_in, _, _ = self.image_sizes
            x_tensor = d_input.new_empty((n_batch, channel_in, height_in, width_in))
        elif len(self.image_sizes) == 7:
            channel_in, depth_in, height_in, width_in, _, _, _ \
                = self.image_sizes
            x_tensor = d_input.new_empty((n_batch, channel_in, depth_in, height_in, width_in))
        else:
            raise TileError('self.image_sizes length is not 3, 5 or 7')

        return self.tile.backward_indexed(d_input, x_tensor)

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
