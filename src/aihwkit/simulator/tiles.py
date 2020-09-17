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

from typing import Dict, Optional, Tuple, Union

from collections import OrderedDict

from numpy import concatenate, expand_dims
from torch import Tensor, stack
from torch.cuda import current_stream, current_device
from torch.cuda import device as cuda_device
from torch import device as torch_device


from aihwkit.simulator.devices import (
    BaseResistiveDevice,
    ConstantStepResistiveDevice,
    FloatingPointResistiveDevice
)
from aihwkit.simulator.rpu_base import tiles, cuda


class BaseTile:
    """Base class for tiles.

    Args:
        out_size: output size
        in_size: input size
        resistive_device: resistive device.
        bias: whether to add a bias column to the tile.
        in_trans: Whether to assume an transposed input (batch first)
        out_trans: Whether to assume an transposed output (batch first)
    """

    is_cuda = False

    def __init__(
            self,
            out_size: int,
            in_size: int,
            resistive_device: BaseResistiveDevice,
            bias: bool = True,
            in_trans: bool = False,
            out_trans: bool = False):
        self.out_size = out_size
        self.in_size = in_size
        self.resistive_device = resistive_device
        self.bias = bias
        self.in_trans = in_trans
        self.out_trans = out_trans

        x_size = in_size + 1 if self.bias else in_size
        d_size = out_size
        self.tile = self.resistive_device.create_tile(x_size, d_size)
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
        self.tile = self.resistive_device.create_tile(x_size, d_size)
        self.tile.set_hidden_parameters(Tensor(hidden_parameters))
        self.tile.set_weights(weights)

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

    def cuda(
            self,
            device: Optional[Union[torch_device, str, int]] = None
    ) -> 'BaseTile':
        """Return a copy of this tile in CUDA memory."""
        raise NotImplementedError

    def forward(self, x_input: Tensor, is_test: bool = False) -> Tensor:
        """Perform the forward pass.

        Args:
            x_input: ``[N, in_size]`` tensor. If ``in_trans`` is set, transposed.
            is_test: whether to assume testing mode.

        Returns:
            torch.Tensor: ``[N, out_size]`` tensor. If ``out_trans`` is set, transposed.
        """
        # Unset the require_grad of the tensor when chaining.
        if x_input.grad_fn:
            x_input = x_input.detach()

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

        resistive_device: resistive device.

        bias: whether to add a bias column to the tile, ie. :math:`W`
          has an extra column to code the biases. Internally, the
          input :math:`\mathbf{x}` will be automatically expanded by
          an extra dimension which will be set to 1 always.

    """

    def __init__(
            self,
            out_size: int,
            in_size: int,
            resistive_device: Optional[FloatingPointResistiveDevice] = None,
            bias: bool = False,
            in_trans: bool = False,
            out_trans: bool = False):
        self.resistive_device: FloatingPointResistiveDevice = (
            resistive_device or FloatingPointResistiveDevice())
        super().__init__(out_size, in_size, self.resistive_device, bias, in_trans, out_trans)

    def cuda(
            self,
            device: Optional[Union[torch_device, str, int]] = None
    ) -> 'BaseTile':
        """Return a copy of this tile in CUDA memory."""

        if not cuda.is_compiled():
            raise RuntimeError('aihwkit has not been compiled with CUDA support')

        with cuda_device(device):
            tile = CudaFloatingPointTile(self.out_size, self.in_size, self.resistive_device,
                                         self.bias, self.in_trans, self.out_trans)
        return tile


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

        resistive_device: resistive device.

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
            resistive_device: Optional[BaseResistiveDevice] = None,
            bias: bool = False,
            in_trans: bool = False,
            out_trans: bool = False):
        self.resistive_device = resistive_device or ConstantStepResistiveDevice()
        super().__init__(out_size, in_size, self.resistive_device, bias, in_trans, out_trans)

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
            tile = CudaAnalogTile(self.out_size, self.in_size, self.resistive_device,
                                  self.bias, self.in_trans, self.out_trans)
        return tile


class CudaFloatingPointTile(FloatingPointTile):
    """Floating point tile (CUDA).

    Args:
        out_size: output vector size of the tile.
        in_size: input vector size of the tile.
        resistive_device: resistive device.
        bias: whether to add a bias column to the tile.
        in_trans: whether to assume a transposed input (batch first)
        out_trans: whether to assume a transposed output (batch first)
    """

    is_cuda = True

    def __init__(
            self,
            out_size: int,
            in_size: int,
            resistive_device: Optional[FloatingPointResistiveDevice] = None,
            bias: bool = False,
            in_trans: bool = False,
            out_trans: bool = False):
        if not cuda.is_compiled():
            raise RuntimeError('aihwkit has not been compiled with CUDA support')

        super().__init__(out_size, in_size, resistive_device, bias, in_trans, out_trans)

        self.tile = tiles.CudaFloatingPointTile(self.tile)
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

    Args:
        out_size: output vector size of the tile.
        in_size: input vector size of the tile.
        resistive_device: resistive device.
        bias: whether to add a bias column to the tile.
        in_trans: whether to assume a transposed input (batch first)
        out_trans: whether to assume a transposed output (batch first)
    """

    is_cuda = True

    def __init__(
            self,
            out_size: int,
            in_size: int,
            resistive_device: Optional[BaseResistiveDevice] = None,
            bias: bool = False,
            in_trans: bool = False,
            out_trans: bool = False):
        if not cuda.is_compiled():
            raise RuntimeError('aihwkit has not been compiled with CUDA support')

        super().__init__(out_size, in_size, resistive_device, bias, in_trans, out_trans)

        self.tile = tiles.CudaAnalogTile(self.tile)
        self.stream = current_stream()
        self.device = torch_device(current_device())

    def cuda(
            self,
            device: Optional[Union[torch_device, str, int]] = None
    ) -> 'CudaAnalogTile':
        if self.stream != current_stream(device):
            raise ValueError("Cannot switch CUDA devices of existing Cuda tiles")

        return self
