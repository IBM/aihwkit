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

"""High level analog tiles."""

from copy import deepcopy
from typing import Dict, Generic, List, Optional, Tuple, TypeVar, Union

from collections import OrderedDict

from numpy import concatenate, expand_dims
from torch import Tensor, stack
from torch.cuda import current_stream, current_device
from torch.cuda import device as cuda_device
from torch import device as torch_device
from torch.autograd import no_grad

from aihwkit.simulator.configs import (
    FloatingPointRPUConfig, SingleRPUConfig, UnitCellRPUConfig, InferenceRPUConfig
)

from aihwkit.simulator.configs.devices import ConstantStepDevice

from aihwkit.simulator.rpu_base import tiles, cuda

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
        """
        # Prepare the array expected by the pybind function, appending the
        # biases row if needed.
        weights_numpy = weights.clone().detach().cpu().numpy()

        if self.bias:
            # Create a ``[out_size, in_size (+ 1)]`` matrix.
            if biases is None:
                raise RuntimeError("Analog tile has a bias, but no bias given!")

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
        """
        return self.tile.decay_weights(alpha)

    def diffuse_weights(self) -> None:
        """Diffuses the weights once.

        The base diffusion rate is set during tile init.
        """
        return self.tile.diffuse_weights()

    def reset_columns(self, start_column_idx: int = 0, num_columns: int = 1,
                      reset_prob: float = 1.0) -> None:
        r"""Reset (a number of) columns.

        Resets the weights with device-to-device and cycle-to-cycle
        variability (depending on device type), typically:

        .. math::
            W_{ij} = \xi*\sigma_\text{reset} + b^\text{reset}_{ij}

        Args:
            start_column_idx: a start index of columns (0..x_size-1)
            num_columns: how many consecutive columns to reset (with circular warping)
            reset_prob: individual probability of reset.

        The reset parameter are set during tile init.
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

    def set_indexed(self, indices: Tensor, image_sizes: List) -> None:
        """Sets the index matrix for convolutions ans switches to
        indexed forward/backward/update versions.

        Args:
            indices : torch.tensor with int indices
            image_sizes: [C_in, H_in, W_in, H_out, W_out] sizes
        """
        if len(image_sizes) != 5:
            raise ValueError('image_sizes expects 5 sizes [C_in, H_in, W_in, H_out, W_out]')

        if self.in_trans or self.out_trans:
            raise ValueError('Transposed indexed versions not supported (assumes NCHW)')

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
        """
        if not self.image_sizes:
            raise ValueError('self.image_sizes is not initialized. Please use '
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
        """
        return self.tile.update_indexed(x_input, d_input)


class FloatingPointTile(BaseTile):
    r"""Floating point tile.

    Implements a floating point or ideal analog tile.

    A linear layer with this tile is perfectly linear, it just uses
    the RPUCuda library for execution.

    **Forward pass**:

    .. math:: \mathbf{y} = W\mathbf{x}

    :math:`W` are the weights, :math:`\mathbf{x}` is the input
    vector. :math:`\mathbf{y}` is output of the vector matrix
    multiplication.  Note that if bias is used, :math:`\mathbf{x}` is
    concatenated with 1 so that the last column of :math:`W` are the
    biases.


    **Backward pass**:

    Typical backward pass with transposed weights:

    .. math:: \mathbf{d'} = W^T\mathbf{d}

    where :math:`\mathbf{d}` is the error
    vector. :math:`\mathbf{d}_o` is output of the backward matrix
    vector multiplication.


    **Weight update**:

    Usual learning rule for back-propagation:

    .. math:: w_{ij} \leftarrow w_{ij} + \lambda d_i\,x_j


    **Decay**:

    .. math:: w_{ij} \leftarrow w_{ij}(1-\alpha r_\text{decay})

    Weight decay can be called by calling the analog tile decay.

    Note:
       ``life_time`` parameter is set during
       initialization. alpha is a scaling factor that can be given
       during run-time.


    **Diffusion**:

    .. math::  w_{ij} \leftarrow w_{ij} +  \xi\;r_\text{diffusion}

    Similar to the decay, diffusion is only done when explicitly
    called. However, the parameter of the diffusion process are
    set during initialization and are fixed for the
    remainder. :math:`\xi` is a standard Gaussian process.

    Args:
        out_size: output vector size of the tile, ie. the dimension of
            :math:`\mathbf{y}` in case of :math:`\mathbf{y} =
            W\mathbf{x}` (or equivalently the dimension of the
            :math:`\boldsymbol{\delta}` of the backward pass).
        in_size: input vector size, ie. the dimension of the vector
            :math:`\mathbf{x}` in case of :math:`\mathbf{y} =
            W\mathbf{x}`).
        rpu_config: resistive processing unit configuration.
        bias: whether to add a bias column to the tile, ie. :math:`W`
            has an extra column to code the biases. Internally, the
            input :math:`\mathbf{x}` will be automatically expanded by
            an extra dimension which will be set to 1 always.
    """

    def __init__(
            self,
            out_size: int,
            in_size: int,
            rpu_config: Optional[FloatingPointRPUConfig] = None,
            bias: bool = False,
            in_trans: bool = False,
            out_trans: bool = False,
    ):
        rpu_config = rpu_config or FloatingPointRPUConfig()
        super().__init__(out_size, in_size, rpu_config, bias, in_trans, out_trans)

    def cuda(
            self,
            device: Optional[Union[torch_device, str, int]] = None
    ) -> 'BaseTile':
        """Return a copy of this tile in CUDA memory."""

        if not cuda.is_compiled():
            raise RuntimeError('aihwkit has not been compiled with CUDA support')

        with cuda_device(device):
            tile = CudaFloatingPointTile(self)

        return tile

    def _create_simulator_tile(
            self,
            x_size: int,
            d_size: int,
            rpu_config: FloatingPointRPUConfig
    ) -> tiles.FloatingPointTile:
        """Create a simulator tile.

        Args:
            x_size: input size
            d_size: output size
            rpu_config: resistive processing unit configuration

        Returns:
            a simulator tile based on the specified configuration.
        """
        meta_parameter = rpu_config.device.as_bindings()

        return meta_parameter.create_array(x_size, d_size)


class AnalogTile(BaseTile):
    r"""Analog tile.

    This analog tile implements an abstract analog tile where many
    cycle-tp-cycle non-idealities and systematic parameter-spreads
    that can be user-defined.

    In general stochastic bit pulse trains are generate during update
    and device materials (or unit cells) at each cross-point are only
    updated if a coincidence of rows and columns pulses.

    Here, a resistive device material is assumed that response with a
    finite step change of its conductance value that is independent of
    its own conductance value.

    In its basic parameter settings it implements the analog RPU tile
    model described in `Gokmen & Vlasov (2016)`_, but with a number of
    enhancements that are adjustable by parameter settings.

    All tile parameters are given in
    :class:`~aihwkit.simulator.parameters.AnalogTileParameters`.

    **Forward pass**:

    In general, the following analog forward pass is computed:

    .. math::

       \mathbf{y} = f_\text{ADC}((W + \sigma_\text{w}\Xi) \otimes
       (f_\text{DAC}( x/\alpha ) +
       \sigma_\text{inp}\,\boldsymbol{\xi}_1 ) +
       \sigma_\text{out}\,\boldsymbol{\xi}_2)\,s_\alpha\,
       s_\text{out}\,\alpha

    where :math:`W` is the weight matrix, :math:`\mathbf{x}` the input
    vector and the :math:`\Xi,\boldsymbol{\xi}_1,\boldsymbol{\xi}_2`
    Gaussian noise variables (with corresponding matrix and vector
    sizes). The :math:`\alpha` is a scale from the noise management
    (see :data:`rpu_types.NoiseManagementTypeMap`). The symbol
    :math:`\otimes` refers to the 'analog' matrix-vector
    multiplication, that might have additional non-linearities.

    :math:`f_\text{Z}` (with `Z` either `ADC` or `DAC`) indicates the
    discretization to a number of equidistant steps between a bound
    value :math:`-b_\text{Z},\ldots,b_\text{Z}` potentially with
    stochastic rounding (SR):

    .. math::

       f_\text{Z}(x) = \text{round}(x\,
       \frac{r_\text{Z}}{2\,b_\text{Z}} +
       \zeta)\frac{2b_\text{Z}}{r_\text{Z}}

    If SR is enabled :math:`\zeta` is an uniform random :math:`\in
    [-0.5,0.5)`. Otherwise :math:`\zeta=0`.  Inputs are clipped below
    :math:`-b_\text{Z}` and above :math:`b_\text{Z}`

    :math:`r_Z` is the resolution of the `ADC` or `DAC`. E.g. for 8
    bit, it would be :math:`1/256`

    Note:
       Typically the resolution is reduced by 2 level, eg. in case of
       8 bits it is set to :math:`1/254` to account for a
       discretization mirror symmetric around zero, including the zero
       and discarding one value.

    The scalar scale :math:`s_\text{out}` can be set by
    ``out_scale``. The scalar scale :math:`s_\alpha` is an additional
    scale that might be use to map weight better to conductance
    ranges.

    For parameters regarding the forward pass behavior, see
    :class:`~aihwkit.simulator.parameters.AnalogTileInputOutputParameters`.


    **Backward pass**:

    Identical to the forward direction except that the transposed
    weight matrix is used.  Same parameters as during the forward pass
    except that bound management is not supported.

    For parameters regarding the backward pass behavior, see
    :class:`~aihwkit.simulator.parameters.AnalogTileInputOutputParameters`.


    **General weight update**:

    The weight update that theoretically needs to be computed is

    .. math:: w_{ij} = w_{ij} + \lambda d_i\,x_j

    thus the outer product of error vector and input vector.

    Although the update depends on the `ResistiveDevice` used, in
    general, stochastic pulse trains of a given length are drawn,
    where the probability of occurrence of an pulse is proportional to
    :math:`\sqrt{\lambda}d_i` and :math:`\sqrt{\lambda}x_j`
    respectively. Then for each cross-point, in case a coincidence of
    column and row pulses occur, the weight is updated one `step`. For
    details, see `Gokmen & Vlasov (2016)`_.

    The amount of how the weight changes per single step might be
    different for the different resistive devices.

    In pseudo code::

        # generate prob number
        p_i  = quantize(A * d_i, res, sto_round)
        q_j  = quantize(B * x_j, res, sto_round)
        sign = sign(d_i)*sign(x_j)

        # generate pulse trains of length BL
        pulse_train_d = gen_pulse_train(p_i, BL) # e.g 101001001
        pulse_train_x = gen_pulse_train(q_j, BL) # e.g 001010010

        for t in range(BL):
            if (pulse_train_x[t]==1) and (pulse_train_d[t]==1)
                update_once(w_ij, direction = sign)

    The probabilities are generated using scaling factors ``A`` and ``B`` that
    are determined by the learning rate and pulse train length ``BL`` (see
    below). ``quantize`` is an optional discretization of the resulting
    probability, to account for limited resolution number in the stochastic
    pulse train generation process on the chip .

    The ``update_once`` functionality is in general dependent on the
    analog tile class.  For `ConstantStep` the step width is
    independent of the actual weight, but has cycle-to-cycle
    variation, device-to-device variation or systematic bias for up
    versus down direction (see below).

    For parameters regarding the update behaviour, see
    :class:`~aihwkit.simulator.parameters.AnalogTileUpdateParameters`.

    Args:
        out_size: output vector size of the tile, ie. the dimension of
            :math:`\mathbf{y}` in case of :math:`\mathbf{y} =
            W\mathbf{x}` (or equivalently the dimension of the
            :math:`\boldsymbol{\delta}` of the backward pass).
        in_size: input vector size, ie. the dimension of the vector
            :math:`\mathbf{x}` in case of :math:`\mathbf{y} =
            W\mathbf{x}`).
        rpu_config: resistive processing unit configuration.
        bias: whether to add a bias column to the tile, ie. :math:`W`
            has an extra column to code the biases. Internally, the
            input :math:`\mathbf{x}` will be automatically expanded by
            an extra dimension which will be set to 1 always.

    .. _Gokmen & Vlasov (2016): https://www.frontiersin.org/articles/10.3389/fnins.2016.00333/full
    """

    def __init__(
            self,
            out_size: int,
            in_size: int,
            rpu_config: Optional[Union[SingleRPUConfig, UnitCellRPUConfig,
                                       InferenceRPUConfig]] = None,
            bias: bool = False,
            in_trans: bool = False,
            out_trans: bool = False,
    ):
        rpu_config = rpu_config or SingleRPUConfig(device=ConstantStepDevice())
        super().__init__(out_size, in_size, rpu_config, bias, in_trans, out_trans)

    def cuda(
            self,
            device: Optional[Union[torch_device, str, int]] = None
    ) -> 'BaseTile':
        """Return a copy of this tile in CUDA memory.

        Args:
            device: CUDA device
        """
        if not cuda.is_compiled():
            raise RuntimeError('aihwkit has not been compiled with CUDA support')

        with cuda_device(device):
            tile = CudaAnalogTile(self)

        return tile

    def _create_simulator_tile(
            self,
            x_size: int,
            d_size: int,
            rpu_config: Union[SingleRPUConfig, UnitCellRPUConfig, InferenceRPUConfig]
    ) -> tiles.AnalogTile:
        """Create a simulator tile.

        Args:
            x_size: input size
            d_size: output size
            rpu_config: resistive processing unit configuration

        Returns:
            a simulator tile based on the specified configuration.
        """
        meta_parameter = rpu_config.as_bindings()
        device_parameter = rpu_config.device.as_bindings()

        return meta_parameter.create_array(x_size, d_size, device_parameter)


class CudaFloatingPointTile(FloatingPointTile):
    """Floating point tile (CUDA).

    Floating point tile that uses GPU for its operation. The instantiation is
    based on an existing non-cuda tile: all the source attributes are copied
    except for the simulator tile, which is recreated using a GPU tile.

    Args:
        source_tile: tile to be used as the source of this tile
    """

    is_cuda = True

    def __init__(self, source_tile: FloatingPointTile):
        if not cuda.is_compiled():
            raise RuntimeError('aihwkit has not been compiled with CUDA support')

        # Create a new instance of the rpu config.
        new_rpu_config = deepcopy(source_tile.rpu_config)

        # Create the tile, replacing the simulator tile.
        super().__init__(source_tile.out_size, source_tile.in_size, new_rpu_config,
                         source_tile.bias, source_tile.in_trans, source_tile.out_trans)
        self.tile = tiles.CudaFloatingPointTile(source_tile.tile)

        # Set the cuda properties
        self.stream = current_stream()
        self.device = torch_device(current_device())

    def cuda(
            self,
            device: Optional[Union[torch_device, str, int]] = None
    ) -> 'CudaFloatingPointTile':
        if self.stream != current_stream(device):
            raise ValueError("Cannot switch streams of existing Cuda tiles")

        return self


class CudaAnalogTile(AnalogTile):
    """Analog tile (CUDA).

    Analog tile that uses GPU for its operation. The instantiation is based on
    an existing non-cuda tile: all the source attributes are copied except
    for the simulator tile, which is recreated using a GPU tile.

    Args:
        source_tile: tile to be used as the source of this tile
    """

    is_cuda = True

    def __init__(self, source_tile: AnalogTile):
        if not cuda.is_compiled():
            raise RuntimeError('aihwkit has not been compiled with CUDA support')

        # Create a new instance of the rpu config.
        new_rpu_config = deepcopy(source_tile.rpu_config)

        # Create the tile, replacing the simulator tile.
        super().__init__(source_tile.out_size, source_tile.in_size, new_rpu_config,
                         source_tile.bias, source_tile.in_trans, source_tile.out_trans)
        self.tile = tiles.CudaAnalogTile(source_tile.tile)

        # Set the cuda properties
        self.stream = current_stream()
        self.device = torch_device(current_device())

    def cuda(
            self,
            device: Optional[Union[torch_device, str, int]] = None
    ) -> 'CudaAnalogTile':
        if self.stream != current_stream(device):
            raise ValueError("Cannot switch CUDA devices of existing Cuda tiles")

        return self
