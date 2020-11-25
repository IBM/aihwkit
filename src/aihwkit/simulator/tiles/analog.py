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

"""High level analog tiles (analog)."""

from copy import deepcopy
from typing import Optional, Union

from torch import device as torch_device
from torch.cuda import current_device, current_stream
from torch.cuda import device as cuda_device

from aihwkit.exceptions import CudaError
from aihwkit.simulator.configs import (
    InferenceRPUConfig, SingleRPUConfig, UnitCellRPUConfig
)
from aihwkit.simulator.configs.devices import ConstantStepDevice
from aihwkit.simulator.rpu_base import cuda, tiles
from aihwkit.simulator.tiles.base import BaseTile


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
        in_trans: Whether to assume an transposed input (batch first).
        out_trans: Whether to assume an transposed output (batch first).

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

        Returns:
            A copy of this tile in CUDA memory.

        Raises:
            CudaError: if the library has not been compiled with CUDA.
        """
        if not cuda.is_compiled():
            raise CudaError('aihwkit has not been compiled with CUDA support')

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
            raise CudaError('aihwkit has not been compiled with CUDA support')

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
            raise CudaError("Cannot switch CUDA devices of existing Cuda tiles")

        return self
