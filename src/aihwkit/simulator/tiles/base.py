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
# pylint: disable=too-many-lines, wrong-import-position

from collections import OrderedDict
from typing import (
    Dict, Generic, List, Optional,
    Tuple, TypeVar, Union, TYPE_CHECKING, Any
)
from copy import deepcopy
from numpy.typing import ArrayLike
from numpy import array

from torch import (
    Tensor, stack, zeros, as_tensor, cat,
    unsqueeze, squeeze, ones,
    float32, from_numpy, full, clamp,
    zeros_like, eye, randn,
)

from torch import device as torch_device
from torch import max as torch_max
from torch.nn import Parameter
from torch.autograd import no_grad
from torch.linalg import lstsq

from aihwkit.simulator.rpu_base import tiles
from aihwkit.exceptions import TileError, ConfigError
from aihwkit.optim.context import AnalogContext

RPUConfigGeneric = TypeVar('RPUConfigGeneric')

if TYPE_CHECKING:
    from aihwkit.simulator.configs.utils import MappingParameter, InputRangeParameter


class AnalogTileStateNames:  # pylint: disable=too-few-public-methods
    """ Class defining analog tile state name constants.

    Caution:
       Do *not* edit. Some names are attribute names of the tile.
    """

    WEIGHTS = 'analog_tile_weights'
    HIDDEN_PARAMETERS = 'analog_tile_hidden_parameters'
    HIDDEN_PARAMETER_NAMES = 'analog_tile_hidden_parameter_names'
    CLASS = 'analog_tile_class'
    LR = 'analog_lr'
    SHARED_WEIGHTS = 'shared_weights'
    CONTEXT = 'analog_ctx'
    OUT_SCALING = 'out_scaling_alpha'
    MAPPING_SCALES = 'mapping_scales'
    RPU_CONFIG = 'rpu_config'


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
        self.mapping_scales = None  # type: Tensor
        self.input_range = None  # type: Parameter

        # Whether CUDA-calls should be blocking
        self.non_blocking = False

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

        # Helpers.
        self.reference_combined_weights = None  # type: Optional[Tensor]

        # init input / output processing
        self.init_learned_out_scales()
        self.init_mapping_scales()
        self.init_input_processing()

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
        # Caution: all attributes of the tile will be saved.
        current_dict = self.__dict__.copy()

        SN = AnalogTileStateNames
        current_dict[SN.WEIGHTS] = self.tile.get_weights()
        current_dict[SN.HIDDEN_PARAMETERS] \
            = self.tile.get_hidden_parameters().data
        current_dict[SN.HIDDEN_PARAMETER_NAMES] \
            = self.tile.get_hidden_parameter_names()
        current_dict[SN.CLASS] = self.__class__.__name__
        current_dict[SN.LR] = self.tile.get_learning_rate()
        current_dict.pop('tile', None)

        # don't save device. Will be determined by loading object
        current_dict.pop('stream', None)
        current_dict.pop('is_cuda', None)
        current_dict.pop('device', None)

        # this is should not be saved.
        current_dict.pop('image_sizes', None)

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
        # pylint: disable=too-many-locals, too-many-statements, too-many-branches

        # Note: self here is NOT initialized! So we need to recreate
        # attributes that were not saved in getstate
        SN = AnalogTileStateNames
        current_dict = state.copy()
        current_dict.pop('image_sizes', None)  # should not be saved
        weights = current_dict.pop(SN.WEIGHTS)

        hidden_parameters = current_dict.pop(SN.HIDDEN_PARAMETERS)
        hidden_parameters_names = current_dict.pop(SN.HIDDEN_PARAMETER_NAMES, [])
        alpha_scale = current_dict.pop('analog_alpha_scale', None)  # legacy
        tile_class = current_dict.pop(SN.CLASS, self.__class__.__name__)
        analog_lr = current_dict.pop(SN.LR, 0.01)
        analog_ctx = current_dict.pop(SN.CONTEXT)
        shared_weights = current_dict.pop(SN.SHARED_WEIGHTS)
        shared_weights_if = shared_weights is not None

        mapping_scales = current_dict.pop(SN.MAPPING_SCALES, None)
        learned_out_scales = current_dict.pop(SN.OUT_SCALING, None)

        current_dict.pop('noise_model', None)  # legacy
        current_dict.pop('drift_compensation', None)  # legacy

        # legacy
        if 'non_blocking' not in current_dict:
            current_dict['non_blocking'] = False

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
        if not isinstance(weights, Tensor):
            weights = from_numpy(array(weights))
        self.tile.set_weights(weights)

        if not isinstance(hidden_parameters, Tensor):
            hidden_parameters = from_numpy(array(hidden_parameters))
        self.tile.set_hidden_parameters(hidden_parameters)

        self.tile.set_learning_rate(analog_lr)

        # re-generate shared weights (CPU)
        if shared_weights_if:
            if not hasattr(self, SN.SHARED_WEIGHTS):
                # this is needed when pkl loading
                self.shared_weights = shared_weights

            with no_grad():
                # always new will be populated with set weights.
                self.shared_weights.data = zeros(d_size, x_size, requires_grad=True)
            self.ensure_shared_weights()
        else:
            self.shared_weights = None

        # Regenerate context but keep the object ID
        if not hasattr(self, SN.CONTEXT):  # when loading
            self.analog_ctx = AnalogContext(self, parameter=analog_ctx)
        self.analog_ctx.reset(self)
        self.analog_ctx.set_data(analog_ctx.data)

        # set scales
        self.out_scaling_alpha = None
        self.mapping_scales = None
        self.init_mapping_scales()
        self.init_learned_out_scales()

        if self.out_scaling_alpha is None and learned_out_scales is not None:
            if mapping_scales is None:
                mapping_scales = 1.0
            x = learned_out_scales.view(learned_out_scales.numel()).clone()
            mapping_scales = mapping_scales * x
            learned_out_scales = None

        self.set_mapping_scales(mapping_scales)
        self.set_learned_out_scales(learned_out_scales)

        if alpha_scale is not None:
            # legacy. We apply the alpha scale instaed of the
            # out_scaling_alpha when loading. The alpha_scale
            # mechansim is now replaced with the out scaling factors
            #
            # Caution: will overwrite the loaded out_scaling_alphas
            # if they would exist also (should not be for old checkpoints)

            self.set_mapping_scales(alpha_scale)

        if to_device.type.startswith('cuda'):
            self.cuda(to_device)

        if alpha_scale is not None:
            # legacy. We apply the alpha scale instaed of the
            # out_scaling_alpha when loading. The alpha_scale
            # mechansim is now replaced with the out scaling factors
            #
            # Caution: will overwrite the loaded out_scaling_alphas
            # if they would exist also (should not be for old checkpoints)

            self.set_mapping_scales(alpha_scale)

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

    def _combine_weights(self, weights: Union[Tensor, ArrayLike],
                         biases: Optional[Union[Tensor, ArrayLike]] = None) -> Tensor:
        """ Helper to combines weights and biases

        In any case, a detached cpu weight and bias copy will be returned.

        Args:
            weights: weights without the bias
            biases: The bias vector if available

        Returns:
            combined weights with biases

        Raises:
            ValueError: if the tile has bias but ``bias`` has not been
                specified.
        """
        if not isinstance(weights, Tensor):
            weights = from_numpy(array(weights))
        weights = weights.clone().detach().cpu().to(float32)

        if self.bias:
            # Create a ``[out_size, in_size (+ 1)]`` matrix.
            if biases is None:
                raise ValueError('Analog tile has a bias, but no bias given')

            if not isinstance(biases, Tensor):
                biases = from_numpy(array(biases))

            biases = unsqueeze(biases.clone().detach().cpu().to(float32), 1)
            return cat((weights, biases), dim=1)
        # Use only the ``[out_size, in_size]`` matrix.
        return weights

    def _separate_weights(self, combined_weights: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        """ Helper to separate the combined weights and biases
        """
        # Split the internal weights (and potentially biases) matrix.
        if self.bias:
            # combined_weights is [out_size, in_size (+ 1)].
            return Tensor(combined_weights[:, :-1]), Tensor(combined_weights[:, -1])

        return combined_weights, None

    def set_weights(
            self,
            weights: Tensor,
            biases: Optional[Tensor] = None,
            apply_weight_scaling: bool = False,
            weight_scaling_omega: Optional[float] = None
    ) -> None:
        """Set the tile weights (and biases).

        Sets the internal tile weights to the specified values, and also the
        internal tile biases if the tile was set to use bias (via
        ``self.bias``).

        Note:
           This setting is **not** hardware realistic. Use the
           :meth:`program_weights` for a realistic weight transfer.

        Args:
            weights: ``[out_size, in_size]`` weight matrix.
            biases: ``[out_size]`` bias vector. This parameter is required if
                ``self.bias`` is ``True``, and ignored otherwise.
            apply_weight_scaling: Whether to rescale the given weight matrix
                and populate the digital output scaling factors as
                specified in the configuration
                :class:`~aihwkit.configs.utils.MappingParameter`. A
                new ``weight_scaling_omega`` can be given. Note that
                this will overwrite the existing digital out scaling
                factors.
            weight_scaling_omega: The weight scaling omega factor (see
                :class:`~aihwkit.configs.utils.MappingParameter`). If
                given explicitly here, it will overwrite the value in
                the mapping field.

        Returns:
            None.
        """
        self.reference_combined_weights = None
        combined_weights = self._combine_weights(weights, biases)

        if apply_weight_scaling:
            combined_weights = self.apply_weight_scaling(combined_weights,
                                                         weight_scaling_omega)
        return self.tile.set_weights(combined_weights)

    def get_weights(self, apply_weight_scaling: bool = False
                    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Get the tile weights (and biases).

        Gets the tile weights and extracts the mathematical weight
        matrix and biases (if present, by determined by the ``self.bias``
        parameter).

        Note:
             The returned weight is a copy of the internal weights (not a
             pointer) and is always on CPU and detached.

        Note:
             This is **not** a hardware realistic weight readout. Use
            :meth:`read_weights` for a realistic transfer.

        Args:
            apply_weight_scaling: Whether to return the weights with the
                (digital) output scaling factors applied. Note the
                "logical" weights of the layer which the DNN is
                effectively using are those with the output scales
                applied. If ``apply_weight_scaling`` is set to False, then
                only the weight values that is programmed onto the
                crossbar array are returned, without applying the
                digital scales.

        Returns:
            a tuple where the first item is the ``[out_size, in_size]`` weight
            matrix; and the second item is either the ``[out_size]`` bias vector
            or ``None`` if the tile is set not to use bias.

        """
        # Retrieve the internal weights (and potentially biases) matrix.
        combined_weights = self.tile.get_weights()
        weights, biases = self._separate_weights(combined_weights)

        if not apply_weight_scaling:
            return weights, biases

        alpha = self.get_scales()
        if alpha is not None:
            alpha = alpha.detach().cpu()
            return weights * alpha.view(-1, 1), biases * alpha if self.bias else None
        return weights, biases

    @no_grad()
    def program_weights(self, from_reference: bool = True,
                        x_values: Optional[Tensor] = None,
                        learning_rate: float = 0.1,
                        max_iter: int = 10000,
                        tolerance: Optional[float] = 0.01,
                        w_init: Union[float, Tensor] = 0.01) -> None:
        """Programm the target weights into the conductances using the
        pulse update defined.

        Programming is done using the defined tile-update (e.g. SGD)
        and matching inputs (`x_values` by default `eye`).

        Args:

            from_reference: Whether to use weights from reference
                (those that were initally set with `set_weights`) or
                the current weights.
            x_values: Values to use for the read-and verify. If none
                are given, unit-vectors are used
            learning_rate: Learning rate of the optimization
            max_iter: max number of batches for the iterative programming
            tolerance: Stop the iteration loop early if the mean
                output deviation is below this number. Given in
                relation to the max output.
            w_init: initial weight matrix to start from. If given as
                float, weights are set uniform random in `[-w_init,
                w_init]`. This init weight is given directly in
                normalized conductance units and should include the
                bias row if existing.
        """

        if not from_reference or self.reference_combined_weights is None:
            self.reference_combined_weights = self.tile.get_weights()
        target_weights = self.reference_combined_weights

        if x_values is None:
            x_values = eye(self.tile.get_x_size())
        x_values = x_values.to(self.device)
        target_values = x_values @ target_weights.to(self.device).T

        target_max = target_values.abs().max().item()
        if isinstance(w_init, Tensor):
            self.tile.set_weights(w_init)
        else:
            self.tile.set_weights_uniform_random(-w_init, w_init)

        lr_save = self.tile.get_learning_rate()
        self.tile.set_learning_rate(learning_rate)

        for _ in range(max_iter):
            y = self.tile.forward(x_values, False)
            error = y - target_values
            if tolerance is not None and (error.abs().mean().item()
                                          / target_max) < tolerance:
                break
            self.tile.update(x_values, error, False)

        self.tile.set_learning_rate(lr_save)

    @no_grad()
    def read_weights(self,
                     apply_weight_scaling: bool = False,
                     x_values: Optional[Tensor] = None,
                     over_sampling: int = 10
                     ) -> Tuple[Tensor, Optional[Tensor]]:
        """Reads the weights (and biases) in a realistic manner
        by using the forward pass for weights readout.

        Gets the tile weights and extracts the mathematical weight
        matrix and biases (if present, by determined by the ``self.bias``
        parameter).

        The weight will not be directly read, but linearly estimated
        using random inputs using the analog forward pass.

        Note:

            If the tile includes digital periphery (e.g. out scaling),
            these will be applied. Thus this weight is the logical
            weights that correspond to the weights in an FP network.

        Note:
            weights are estimated using the ``lstsq`` solver from torch.

        Args:
            apply_weight_scaling: Whether to rescale the given weight matrix
                and populate the digital output scaling factors as
                specified in the configuration
                :class:`~aihwkit.configs.utils.MappingParameter`. A
                new ``weight_scaling_omega`` can be given. Note that
                this will overwrite the existing digital out scaling
                factors.

            x_values: Values to use for estimating the matrix. If
                not given, inputs are standard normal vectors.

            over_sampling: If ``x_values`` is not given,
                ``over_sampling * in_size`` random vectors are used
                for the estimation

        Returns:
            a tuple where the first item is the ``[out_size, in_size]`` weight
            matrix; and the second item is either the ``[out_size]`` bias vector
            or ``None`` if the tile is set not to use bias.

        """

        if x_values is None:
            x_values = randn(self.in_size * over_sampling, self.in_size,
                             device=self.device,
                             dtype=float32)
        else:
            x_values = x_values.to(self.device)

        y_values = self.forward(x_values, is_test=True)

        # calculate pseudo inverse (with bias, if necessary)
        if self.bias:
            ones_column = ones(self.in_size * over_sampling, 1,
                               device=self.device, dtype=float32)
            x_values = cat([x_values, ones_column], axis=1)

        est_weights = lstsq(x_values, y_values).solution.T.cpu()
        weights, biases = self._separate_weights(est_weights)

        if not apply_weight_scaling:
            # we de-apply (devide) because we want to use the full self.forward above
            alpha = self.get_scales()
            if alpha is not None:
                alpha = alpha.detach().cpu()
                return weights / alpha.view(-1, 1), biases / alpha if self.bias else None
        return weights, biases

    def apply_weight_scaling(
            self,
            combined_weights: Tensor,
            weight_scaling_omega: Optional[float] = None
    ) -> Tensor:
        r"""Set the tile weights (and biases) in a scaled fashion.

        Scales the weights by a layerwise scale or columnwise scale (if
        ``weight_scaling_columnwise`` is set), that is then applied in digital
        at the output of forward and backward pass, and the learning rate for
        this tile is adjusted accordingly.

        If layerwise scale is chosen, weights are scaled by
        :math:`\omega/\max_{ij} |w_{ij}|` and the global digital factor
        :math:`alpha` is set to :math:`\max_{ij} |w_{ij}|/\omega`.

        It can be shown that such a constant factor greatly improves the SNR and
        training accuracy as the full weight range of the analog devices are
        used. See also `Rasch, Gokmen & Haensch (2019)`_ for more details.

        Args:
            combined_weights: ``[d_size, x_size]`` weight matrix.
            weight_scaling_omega: where the weight max should be mapped in terms of
                the weight range. Note that for ``omega`` larger than
                the maximal weight of the device, weights will get
                clipped for most devices. If this parameter is not
                given, it will default to the ``weight_scaling_omega``
                value set in the
                :class:`~aihwkit.configs.utils.MappingParameter` of the
                ``rpu_config``

        Returns:
            scaled weights.


        .. _`Rasch, Gokmen & Haensch (2019)`: https://arxiv.org/abs/1906.02698

        """
        # Prepare the array expected by the pybind function, appending the
        # biases row if needed.
        if not hasattr(self.rpu_config, 'mapping'):
            return combined_weights

        mapping = self.rpu_config.mapping  # type: MappingParameter
        omega = weight_scaling_omega
        if omega is None:
            omega = mapping.weight_scaling_omega

        if omega is not None and omega > 0:
            # Apply the scaling
            if mapping.weight_scaling_columnwise:
                weight_max, _ = torch_max(abs(combined_weights), 1, keepdim=True)
            else:
                weight_max = torch_max(abs(combined_weights)).view(1)

            alpha = weight_max / omega

            alpha[alpha == 0.0] = 1.0
            combined_weights = combined_weights / alpha

            self.set_scales(alpha)
        return combined_weights

    @no_grad()
    def get_mapping_scales(self) -> Optional[Tensor]:
        """Get the scales used for the weight mapping.

        Returns:

           Mapping scales: the vector (or scalar) that is used to
           determine the mapping into (norm) conductance units. These
           scales are used at the output of the analog MVM.
        """
        return self.mapping_scales

    @no_grad()
    def set_mapping_scales(self, mapping_scales: Optional[Union[Tensor, float]]) -> None:
        """Set the scales used for the weight mapping.

        Args:
           mapping_scales: Vector (or scalar) used for the mapping
           of weights into conductance units. This mapping is never in
           the SGD graph but might get initialized when
           ``weight_scaling_omega`` is used or remapping is enforced.
        """

        if mapping_scales is None:
            self.mapping_scales = None
            return

        if isinstance(mapping_scales, float):
            if self.mapping_scales is None:
                self.mapping_scales = ones((1, ),
                                           dtype=float32,
                                           device=self.device,
                                           requires_grad=False)
            self.mapping_scales[:] = mapping_scales
            return

        if isinstance(self.mapping_scales, Tensor) and len(mapping_scales) == 1:
            self.mapping_scales[:] = mapping_scales.to(self.device)
            return

        self.mapping_scales = mapping_scales.flatten().to(self.device)

    @no_grad()
    def init_mapping_scales(self) -> None:
        """Helper function to initialize the mapping scales used to scale the
        weights in digital and determine the conductance conversion.

        Note:
            This method is called from the constructor.
        """
        if not hasattr(self.rpu_config, 'mapping'):
            self.set_mapping_scales(None)
            return

        mapping = self.rpu_config.mapping  # type: MappingParameter
        mapping_scales = None
        if mapping.weight_scaling_omega:
            if mapping.weight_scaling_columnwise:
                mapping_scales = ones((self.out_size, ),
                                      dtype=float32,
                                      device=self.device,
                                      requires_grad=False)
            else:
                mapping_scales = ones((1, ),
                                      dtype=float32,
                                      device=self.device,
                                      requires_grad=False)
        self.set_mapping_scales(mapping_scales)

    @no_grad()
    def init_input_processing(self) -> None:
        """Helper function to initialize the input processing.

        Note:
            This method is called from the constructor.

        Raises: ConfigError in case ``manage_output_clipping`` is
            enabled but not supported.
        """
        self.input_range = None
        if not hasattr(self.rpu_config, 'pre_post'):
            return

        ir_params = self.rpu_config.pre_post.input_range  # type: InputRangeParameter
        if ir_params.enable:
            self.input_range_update_idx = 0
            self.input_range = full((1,), ir_params.init_value, dtype=float32,
                                    device=self.device, requires_grad=True)

            if (ir_params.manage_output_clipping
                    and not ir_params.supports_manage_output_clipping(self.rpu_config)):
                raise ConfigError("RPU Config does not support `manage_output_clipping`.")

    @no_grad()
    def set_scales(self, scales: Union[Tensor, float]) -> None:
        """Set all scales with a new scale.

        This will set the mapping scales to ``scales`` and set all other scales to 1.

        Args:
            scales: scales to set.
        """

        self.set_mapping_scales(scales)
        self.set_learned_out_scales(1.0)

    @no_grad()
    def get_scales(self) -> Optional[Tensor]:
        """ Set all scales with a new scale.

        Returns:
            Scale tensor if any scale exist else None.
        """

        learned_out_scales = self.get_learned_out_scales()
        mapping_scales = self.get_mapping_scales()
        if mapping_scales is None and learned_out_scales is None:
            return None
        if mapping_scales is None:
            return learned_out_scales
        if learned_out_scales is None:
            return mapping_scales
        return mapping_scales * learned_out_scales

    @no_grad()
    def get_learned_out_scales(self) -> Tensor:
        """Get the learned_out_scaled that can be used add an output scale to
        the weights, that is learned.

        Returns:
            tensor: learned_out_scales

        """
        return self.out_scaling_alpha

    @no_grad()
    def init_learned_out_scales(self) -> None:
        """Helper function to initialize the learned out scaling used to scale the
        weights in digital.

        Note:
            This method is called from the constructor.
        """

        if not hasattr(self.rpu_config, 'mapping'):
            return

        mapping = self.rpu_config.mapping  # type: ignore
        if mapping.learn_out_scaling:
            if mapping.out_scaling_columnwise:
                self.out_scaling_alpha = ones((self.out_size, ),
                                              dtype=float32,
                                              device=self.device,
                                              requires_grad=True)
            else:
                self.out_scaling_alpha = ones((1, ),
                                              dtype=float32,
                                              device=self.device,
                                              requires_grad=True)

    @no_grad()
    def set_learned_out_scales(self, alpha: Union[Tensor, float]) -> None:
        """Helper function to set the out scaling alpha used to scale the
        weights in digital.

        Note:
            Will be a no-op in case :meth:`~init_learned_out_scales`
            was not called

        Caution:
            Will not check the correct size of the given alpha.

        Args:
            alpha: out scales as a parameter that is learned.

        """
        if self.out_scaling_alpha is None:
            return

        if isinstance(self.out_scaling_alpha, Parameter):
            self.out_scaling_alpha.data[:] = squeeze(as_tensor(alpha)).to(self.device)
        elif isinstance(self.out_scaling_alpha, Tensor):
            self.out_scaling_alpha[:] = squeeze(as_tensor(alpha)).to(self.device)
        else:
            self.out_scaling_alpha = squeeze(as_tensor(alpha)).to(self.device)

    def apply_out_scaling(self, values: Tensor,
                          tensor_view: Optional[Tuple[int, ...]] = None) -> Tensor:
        """Apply the learned out scaling to the given tensor.

        Args:
            values: tensor to apply scaling to.
            tensor_view: view to cast the out scalings before multiplication

        Returns:
            output tensor with applied out scaling factors
        """
        if self.out_scaling_alpha is not None:
            if tensor_view is None:
                tensor_view = self._get_tensor_view(values.dim(),
                                                    0 if self.out_trans else values.dim() - 1)
            return values * self.out_scaling_alpha.view(*tensor_view)
        return values

    @no_grad()
    def apply_input_range(self, values: Tensor, update_from_data: bool = False) -> Tensor:
        """ Apply the input clipping.

        Args:
            values: tensor to clip
            update_from_data: whether to update from data if applicable

        Returns:
            clipped output tensor
        """
        if self.input_range is None:
            return values

        if update_from_data:
            ir_params = self.rpu_config.pre_post.input_range  # type: ignore
            idx = self.input_range_update_idx
            if idx < ir_params.init_from_data:
                self.input_range.data = (self.input_range.data * idx
                                         + ir_params.init_std_alpha * values.std()
                                         ) / (idx + 1)
                self.input_range_update_idx += 1

        self.input_range.data = self.input_range.data.abs()
        return clamp(values,
                     min=-self.input_range.item(),  # pylint: disable=invalid-unary-operand-type
                     max=self.input_range.item())

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

    @no_grad()
    def reset(
            self,
            reset_prob: float = 1.0
    ) -> None:
        r"""Reset the updated device tile according to the reset parameters of the tile.

        Resets the weights with device-to-device and cycle-to-cycle
        variability (depending on device type), typically:

        .. math::
            W_{ij} = \xi*\sigma_\text{reset} + b^\text{reset}_{ij}

        The reset parameters are set during tile init.

        Args:
            reset_prob: individual probability of reset.

        Returns:
            None
        """
        return self.tile.reset_columns(0, -1, reset_prob)

    def cpu(self) -> 'BaseTile':
        """Return a copy of this tile in CPU memory.

        Returns:
            self in case of CPU

        """
        if not self.is_cuda:
            return self

        state_dict = self.__getstate__()
        for value in state_dict.values():
            if isinstance(value, AnalogContext):
                value.data = value.data.cpu()
        self.__setstate__(state_dict)
        return self

    def cuda(
            self,
            device: Optional[Union[torch_device, str, int]] = None
    ) -> 'BaseTile':
        """Return a copy of this tile in CUDA memory."""
        raise NotImplementedError

    def _get_tensor_view(self, ndim: int, dim: int) -> tuple:
        """Return the tensor view for ndim vector at dim.

        Args:
            ndim: number of dimensions
            dim: the dimension to set to -1

        Returns:
            List of ones with the `dim`` index sets to -1
        """
        tensor_view = [1] * ndim
        tensor_view[dim] = -1
        return tuple(tensor_view)

    @no_grad()
    def pre_forward(self, x_input: Tensor, dim: int,
                    is_test: bool = False, ctx: Any = None) -> Tensor:
        """Operations before the actual forward step for pre processing.

        By default, this is an no-op. However, it could be overridden
        in derived tile classes.

        Args:
            x_input: input tensor for the analog MVM of the tile.
            dim: input channel dimension, ie the x_size dimension
            is_test: whether in eval mode
            ctx: torch auto-grad context [Optional]

        Returns:
            Output tensor of the same shape
        """
        # pylint: disable=unused-argument
        if self.input_range is not None:
            x_input = self.apply_input_range(x_input, not is_test) / self.input_range
        return x_input

    @no_grad()
    def post_forward(self, x_output: Tensor,
                     dim: int,
                     is_test: bool = False,
                     ctx: Any = None) -> Tensor:
        """Operations after the actual forward step for post processing.

        Args:
            x_output:  tensor that is the output from the forward pass of the tile
            dim: output channel dimension, ie the d_size dimension
            is_test: whether in eval mode
            ctx: torch auto-grad context [Optional]

        Returns:
            Output tensor of the same shape
        """
        # pylint: disable=unused-argument
        scale = None
        if self.input_range is not None:
            scale = self.input_range

            # if output clip determines input clip learning
            ir_params = self.rpu_config.pre_post.input_range  # type: ignore
            if ctx is not None and ir_params.manage_output_clipping:
                out_bound = self.rpu_config.forward.out_bound * 0.999  # type: ignore
                output_percentage = (x_output.abs() < out_bound).float().mean()
                ctx.output_percentage = output_percentage

        if self.mapping_scales is not None:
            tensor_view = self._get_tensor_view(x_output.dim(), dim)
            if scale is not None:
                scale = scale * self.get_mapping_scales().view(tensor_view)
            else:
                scale = self.get_mapping_scales().view(tensor_view)

        if scale is not None:
            return x_output * scale
        return x_output

    @no_grad()
    def forward(self, x_input: Tensor, is_test: bool = False, ctx: Any = None) -> Tensor:
        """Perform the forward pass.

        Calls first the ``pre_forward``, then the tile forward, and
        finally the ``post_forward`` step.

        Note:

            The full forward pass is not using autograd, thus all pre
            and post functions need to be handled appropriately in the
            pre/post backward functions.

        Args:
            x_input: ``[N, in_size]`` tensor. If ``in_trans`` is set, transposed.
            is_test: whether to assume testing mode.
            ctx: torch auto-grad context [Optional]

        Returns:
            torch.Tensor: ``[N, out_size]`` tensor. If ``out_trans`` is set, transposed.

        """
        # We use no-grad as we do it explicitly in the optimizer.
        x_input = self.pre_forward(x_input,
                                   0 if self.in_trans else x_input.dim() - 1,
                                   is_test, ctx)
        x_output = self.tile.forward(x_input, self.bias, self.in_trans,
                                     self.out_trans, is_test, self.non_blocking)
        return self.post_forward(x_output,
                                 0 if self.out_trans else x_output.dim() - 1,
                                 is_test, ctx)

    @no_grad()
    def pre_backward(self, d_input: Tensor, dim: int, ctx: Any = None) -> Tensor:
        """Operations before the actual backward step for pre processing.

        By default, this is an no-op. However, it could be overridden
        in derived tile classes.

        Args:
            d_input: The input tensor from to the analog MVM of the tile.
            dim: the dim of the d_size dimension
            ctx: torch auto-grad context [Optional]

        Returns:
            The preprocessed tensor of the same shape
        """
        # pylint: disable=unused-argument
        if self.mapping_scales is not None:
            tensor_view = self._get_tensor_view(d_input.dim(), dim)
            return d_input * self.get_mapping_scales().view(tensor_view)
        return d_input

    @no_grad()
    def post_backward(self, d_output: Tensor, dim: int, ctx: Any = None) -> Tensor:
        """Operations after the actual backward step for post processing.

        Here, the mapping scales are applied if exist.

        Args:
            d_output: The output tensor from the analog MVM of the tile.
            dim: the dim of the x_size dimension
            ctx: torch auto-grad context [Optional]

        Returns:
            The postprocessed tensor of the same shape
        """
        # pylint: disable=unused-argument

        if self.input_range is not None and ctx is not None:
            # compute gradient of the clip
            x_input,  = ctx.saved_tensors
            ir_params = self.rpu_config.pre_post.input_range  # type: ignore

            upper_thres = x_input >= self.input_range
            lower_thres = x_input <= -self.input_range  # pylint: disable=invalid-unary-operand-type

            grad = zeros_like(self.input_range)

            grad += clamp(upper_thres * d_output, min=None, max=0.0).sum()
            grad -= clamp(lower_thres * d_output, min=0.0, max=None).sum()

            if ir_params.gradient_relative:
                grad *= self.input_range
            grad *= ir_params.gradient_scale

            if ir_params.manage_output_clipping:
                output_percentage = getattr(ctx, 'output_percentage', 1.0)
                grad -= (1.0 - output_percentage) * self.input_range * (
                    output_percentage < ir_params.output_min_percentage)

            if ir_params.decay > 0:
                percentage = (x_input.abs() < self.input_range).float().mean()
                grad += ir_params.decay * self.input_range * (
                    percentage > ir_params.input_min_percentage)

            if self.input_range.grad is None:
                self.input_range.grad = grad
            else:
                self.input_range.grad += grad

        return d_output

    @no_grad()
    def backward(self, d_input: Tensor, ctx: Any = None) -> Tensor:
        """Perform the backward pass.

        Args:
            d_input: ``[N, out_size]`` tensor. If ``out_trans`` is set, transposed.
            ctx: torch auto-grad context [Optional]

        Returns:
            torch.Tensor: ``[N, in_size]`` tensor. If ``in_trans`` is set, transposed.
        """
        d_input = self.pre_backward(d_input, 0 if self.out_trans else d_input.dim() - 1,
                                    ctx)
        d_output = self.tile.backward(d_input, self.bias, self.out_trans, self.in_trans,
                                      self.non_blocking)
        return self.post_backward(d_output, 0 if self.in_trans else d_output.dim() - 1,
                                  ctx)

    @no_grad()
    def pre_update(self, x_input: Tensor, x_dim: int,
                   d_input: Tensor, d_dim: int) -> Tuple[Tensor, Tensor]:
        """Operations before the actual update step for pre processing.

        Be default, if the mapping scales are used, the ``d_input``
        will be divided by the mapping scales to compensate for the
        conductance mapping.

        Caution:

            The ``x_input`` and ``d_input`` here are the *original* inputs
            to the ``forward` and ``backward`` methods, thus the
            ``pre_forward`` and ``pre_backward`` function are *not*
            applied, and might need to be applied again here.

        Args:
            x_input: The forward input tensor.
            x_dim: the dim of the x_size dimension of the forward input.
            d_input: The backward (gradient) input tensor.
            d_dim: the dim of the d_size dimension of the backward input.

        Returns:
           Tuple of the preprocessed x_input and d_input tensors of the same shape

        """
        # pylint: disable=unused-argument
        if self.input_range is not None:
            x_input = self.apply_input_range(x_input, False) / self.input_range

        if self.mapping_scales is not None:
            tensor_view = self._get_tensor_view(d_input.dim(), d_dim)
            return x_input, d_input / self.get_mapping_scales().view(tensor_view)

        return x_input, d_input

    @no_grad()
    def update(self, x_input: Tensor, d_input: Tensor) -> None:
        """Perform the update pass.

        Calls the ``pre_update`` method to pre-process the inputs.

        Args:
            x_input: ``[..., in_size]`` tensor. If ``in_trans`` is set, ``[in_size, ...]``.
            d_input: ``[..., out_size]`` tensor. If ``out_trans`` is set, ``[out_size, ...]``.

        Returns:
            None
        """
        x_input, d_input = self.pre_update(x_input,
                                           0 if self.in_trans else x_input.dim() - 1,
                                           d_input,
                                           0 if self.out_trans else d_input.dim() - 1)
        return self.tile.update(x_input, d_input, self.bias,
                                self.in_trans, self.out_trans, self.non_blocking)

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

    def is_indexed(self) -> bool:
        """Returns whether index matrix for convolutions has been set.

        Returns:
           Whether index matrix has been set
        """
        return self.tile.has_matrix_indices()

    def set_indexed(self, indices: Tensor, image_sizes: List) -> None:
        """Set the index matrix for convolutions and switches to
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
    def forward_indexed(self, x_input: Tensor, is_test: bool = False,
                        ctx: Any = None) -> Tensor:
        """Perform the forward pass for convolutions.

        Depending on the input tensor size it performs the forward pass for a
        2D image or a 3D one.

        Args:
            x_input: ``[N, in_size]`` tensor. If ``in_trans`` is set, transposed.
            is_test: whether to assume testing mode.
            ctx: torch auto-grad context [Optional]

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

        x_input = self.pre_forward(x_input, 1, is_test, ctx)
        x_output = self.tile.forward_indexed(x_input, d_tensor, is_test, self.non_blocking)
        return self.post_forward(x_output, 1, is_test, ctx)

    @no_grad()
    def backward_indexed(self, d_input: Tensor, ctx: Any = None) -> Tensor:
        """Perform the backward pass for convolutions.

        Depending on the input tensor size it performs the backward pass for a
        2D image or a 3D one.

        Args:
            d_input: ``[N, out_size]`` tensor. If ``out_trans`` is set, transposed.
            ctx: torch auto-grad context [Optional]

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

        d_input = self.pre_backward(d_input, 1, ctx)
        d_output = self.tile.backward_indexed(d_input, x_tensor, self.non_blocking)
        return self.post_backward(d_output, 1, ctx)

    @no_grad()
    def update_indexed(self, x_input: Tensor, d_input: Tensor) -> None:
        """Perform the update pass for convolutions.

        Calls the ``pre_update`` methods to pre-process the inputs.

        Args:
            x_input: ``[N, in_size]`` tensor. If ``in_trans`` is set, transposed.
            d_input: ``[N, out_size]`` tensor. If ``out_trans`` is set, transposed.

        Returns:
            None
        """
        x_input, d_input = self.pre_update(x_input, 1, d_input, 1)
        return self.tile.update_indexed(x_input, d_input, self.non_blocking)

    @no_grad()
    def post_update_step(self) -> None:
        """Operators that need to be called once per mini-batch.

        Note:
            This function is called by the analog optimizer.

        Caution:

            If no analog optimizer is used, the post update steps will
            not be performed.
        """
        if self.rpu_config.device.requires_diffusion():  # type: ignore
            self.diffuse_weights()
        if self.rpu_config.device.requires_decay():  # type: ignore
            self.decay_weights()
