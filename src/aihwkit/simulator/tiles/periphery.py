# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Base tile with added periphery and common utility methods."""

# pylint: disable=too-many-lines

from typing import Optional, Tuple, Union, Any, List
from numpy import array

from torch import (
    Tensor,
    as_tensor,
    cat,
    ones,
    where,
    from_numpy,
    full,
    clamp,
    zeros_like,
    eye,
    randn,
    zeros,
    logical_or,
)

from torch import device as torch_device
from torch import max as torch_max
from torch.nn import Parameter, Module
from torch.autograd import no_grad
from torch.linalg import lstsq

from aihwkit.exceptions import TileError, ConfigError
from aihwkit.simulator.tiles.base import BaseTile, SimulatorTileWrapper, SimulatorTile

from aihwkit.simulator.parameters.mapping import MappingParameter
from aihwkit.simulator.parameters.pre_post import PrePostProcessingRPU, InputRangeParameter


class TileWithPeriphery(BaseTile, SimulatorTileWrapper):
    """Partial class for tile modules with periphery.

    The function ``joint_forward`` should be called from the
    TileModule level.

    The class also implements the digital bias and adds output scales as
    well as mapping / reading / programming
    functionality. Additionally input range and output scale learning
    is implemented.

    Note:

        This is only a partial class implementation for the
        periphery. All classes inherit from this need to also inherit
        from :class:`~aihwkit.simulator.tiles.module.TileModule`.

        All the module buffers and parameters will be
        handled by the TileModule.

    """

    # pylint: disable=no-member, too-many-public-methods, abstract-method
    # pylint: disable=too-many-instance-attributes
    supports_indexed = True

    def __init__(self) -> None:
        # SimulatorTileWrapper.__init__ is called later. Only included here to
        # make mypy happier
        # pylint: disable=super-init-not-called

        self.out_scaling_alpha = None  # type: Parameter
        self.mapping_scales = None  # type: Tensor
        self.mapping_lr_scale = 1.0
        self.input_range = None  # type: Parameter

        self.image_sizes = []  # type: List[int]

        if self.digital_bias:
            # Note that the bias needs to be handled on the module level
            self.bias = Parameter(zeros(self.out_size, dtype=self.get_dtype()), requires_grad=True)
        else:
            self.bias = None

        # Whether CUDA-calls should be blocking
        self.non_blocking = False

        # Helpers.
        self.reference_combined_weights = None  # type: Optional[Tensor]

        # init input / output processing
        self.init_learned_out_scales()
        self.init_mapping_scales()
        self.init_input_processing()

        if isinstance(self, Module):
            mapping_scales = self.__dict__.pop("mapping_scales")
            self.register_buffer("mapping_scales", mapping_scales)

    @no_grad()
    def set_weights(
        self,
        weight: Tensor,
        bias: Optional[Tensor] = None,
        apply_weight_scaling: bool = True,
        realistic: bool = False,
        weight_scaling_omega: Optional[float] = None,
    ) -> None:
        """Set the tile weights (and biases).

        Sets the internal tile weights (and biases) to the specified
        values.

        Note:

            By default this is **not** a hardware realistic weight
            readout but an exact weight copy of the internal weights.

        Caution:

            By default the peripheral digital scales are applied to
            the weights, so that the weight is scaled (in case
            ``weight_scaling_omega`` is set accordingly).

        Args:
            weight: ``[out_size, in_size]`` weight matrix.
            bias: ``[out_size]`` bias vector. This parameter is required if
                ``self.analog_bias`` is ``True``, and ignored otherwise.
            apply_weight_scaling: Whether to rescale the given weight matrix
                and populate the digital output scaling factors as
                specified in the configuration
                :class:`~aihwkit.simulator.configs.MappingParameter`. A
                new ``weight_scaling_omega`` can be given. Note that
                this will overwrite the existing digital out scaling
                factors.
            realistic: whether to enable realistic write for
                getting the weights. Internally calls
                `program_weights`.
            weight_scaling_omega: The weight scaling omega factor (see
                :class:`~aihwkit.simulator.configs.MappingParameter`). If
                given explicitly here, it will overwrite the value in
                the mapping field.

        """
        self.reference_combined_weights = None

        if bias is not None and self.digital_bias:
            if not isinstance(bias, Tensor):
                bias = from_numpy(array(bias))

            self.bias.data[:] = bias[:].clone().detach().to(self.get_dtype()).to(self.bias.device)
            bias = None

        combined_weights = self._combine_weights(weight, bias)

        if apply_weight_scaling:
            combined_weights = self.apply_weight_scaling(combined_weights, weight_scaling_omega)
        self.tile.set_weights(combined_weights)

        if realistic:
            self.program_weights()

    @no_grad()
    def get_weights(
        self, apply_weight_scaling: bool = True, realistic: bool = False
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Get the tile weights (and biases).

        Gets the tile weights and extracts the mathematical weight
        matrix and biases (if present, by determined by the ``self.analog_bias``
        parameter).

        Note:
             The returned weight is a copy of the internal weights (not a
             pointer) and is always on CPU and detached.

        Note:
             By default tis is **not** a hardware realistic weight readout. Use
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
            realistic: whether to enable realistic read/write for
                getting the weights. Internally calls
                `read_weights`.

        Returns:
            a tuple where the first item is the ``[out_size, in_size]`` weight
            matrix; and the second item is either the ``[out_size]`` bias vector
            or ``None`` if the tile is set not to use bias.

        """
        if realistic:
            return self.read_weights(apply_weight_scaling=apply_weight_scaling)

        # Retrieve the internal weights (and potentially biases) matrix.
        combined_weights = self.tile.get_weights()
        weight, bias = self._separate_weights(combined_weights)

        if self.digital_bias:
            bias = self.bias.detach().cpu()

        if not apply_weight_scaling:
            return weight, bias

        alpha = self.get_scales()
        if alpha is not None:
            alpha = alpha.detach().cpu()
            return (weight * alpha.view(-1, 1), bias * alpha if self.analog_bias else bias)
        return weight, bias

    @no_grad()
    def program_weights(
        self,
        from_reference: bool = True,
        x_values: Optional[Tensor] = None,
        learning_rate: float = 0.1,
        max_iter: int = 10000,
        tolerance: Optional[float] = 0.01,
        w_init: Union[float, Tensor] = 0.01,
    ) -> None:
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
            self.tile.set_weights_uniform_random(-w_init, w_init)  # type: ignore

        lr_save = self.tile.get_learning_rate()  # type: ignore
        self.tile.set_learning_rate(learning_rate)  # type: ignore

        for _ in range(max_iter):
            y = self.tile.forward(x_values, False)
            error = y - target_values
            if tolerance is not None and (error.abs().mean().item() / target_max) < tolerance:
                break
            self.tile.update(x_values, error, False)  # type: ignore

        self.tile.set_learning_rate(lr_save)  # type: ignore

    @no_grad()
    def remap_weights(self, weight_scaling_omega: Optional[float] = 1.0) -> None:
        """Gets and re-sets the weights in case of using the weight scaling.

        This re-sets the weights with applied mapping scales, so that
        the weight mapping scales are updated.

        In case of hardware-aware training, this would update the
        weight mapping scales so that the absolute max analog weights
        are set to 1 (as specified in the ``weight_scaling``
        configuration of
        :class:`~aihwkit.simulator.configs.MappingParameter`).

        Note:
            By default the weight scaling omega factor is set to 1
            here (overriding any setting in the ``rpu_config``). This
            means that the max weight value is set to 1 internally for
            the analog weights.

        Caution:
            This should typically *not* be called for analog. Use
            ``program_weights`` to re-program.

        Args:
            weight_scaling_omega: The weight scaling omega factor (see
                :class:`~aihwkit.simulator.configs.MappingParameter`). If
                set to None here, it will take the value in the
                mapping parameters. Default is however 1.0.
        """
        weight, bias = self.get_weights(apply_weight_scaling=True)
        self.set_weights(
            weight, bias, apply_weight_scaling=True, weight_scaling_omega=weight_scaling_omega
        )

    @no_grad()
    def read_weights(
        self,
        apply_weight_scaling: bool = False,
        x_values: Optional[Tensor] = None,
        over_sampling: int = 10,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Reads the weights (and biases) in a realistic manner
        by using the forward pass for weights readout.

        Gets the tile weights and extracts the mathematical weight
        matrix and biases (if present, by determined by the ``self.analog_bias``
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
                :class:`~aihwkit.simulator.configs.MappingParameter`. A
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

        Raises:
            TileError: in case wrong code usage of TileWithPeriphery
        """
        dtype = self.get_dtype()
        if x_values is None:
            x_values = randn(
                self.in_size * over_sampling, self.in_size, device=self.device, dtype=dtype
            )
        else:
            x_values = x_values.to(self.device)

        # joint forward does only apply the mapping scales
        if not isinstance(self, Module):
            raise TileError("TileWithPeriphery is expected to be part of a Module")

        # forward pass in eval mode
        was_training = self.training
        is_indexed = self.is_indexed()
        self.eval()
        if is_indexed:
            self.analog_ctx.set_indexed(False)
        y_values = self.forward(x_values)
        if was_training:
            self.train()
        if is_indexed:
            self.analog_ctx.set_indexed(True)

        if self.bias is not None:
            y_values -= self.bias

        # calculate pseudo inverse (with bias, if necessary)
        if self.analog_bias:
            ones_column = ones(self.in_size * over_sampling, 1, device=self.device, dtype=dtype)
            x_values = cat([x_values, ones_column], axis=1)

        est_weight = lstsq(x_values, y_values).solution.T.cpu()
        weight, bias = self._separate_weights(est_weight)

        if self.digital_bias:
            bias = self.bias.detach().cpu()

        if not apply_weight_scaling:
            # we de-apply all scales
            alpha = self.get_scales()
            if alpha is not None:
                alpha = alpha.detach().cpu()
                return (weight / alpha.view(-1, 1), bias / alpha if self.analog_bias else bias)
        return weight, bias

    def apply_weight_scaling(
        self, combined_weights: Tensor, weight_scaling_omega: Optional[float] = None
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
                :class:`~aihwkit.simulator.configs.MappingParameter` of the
                ``rpu_config``

        Returns:
            scaled weights.


        .. _`Rasch, Gokmen & Haensch (2019)`: https://arxiv.org/abs/1906.02698

        """
        # Prepare the array expected by the pybind function, appending the
        # biases row if needed.

        if not hasattr(self.rpu_config, "mapping"):
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
        if mapping_scales is None or not hasattr(self.rpu_config, "mapping"):
            self.mapping_scales = None
            return

        if isinstance(mapping_scales, float):
            if self.mapping_scales is None:
                self.mapping_scales = ones((1,), dtype=self.get_dtype(), device=self.device)
                self.mapping_scales[:] = mapping_scales
        elif isinstance(self.mapping_scales, Tensor) and len(mapping_scales) == 1:
            self.mapping_scales[:] = mapping_scales.to(self.device)
        else:
            self.mapping_scales = mapping_scales.flatten().to(self.device)

        if getattr(self.rpu_config.mapping, "weight_scaling_lr_compensation", False):
            self.mapping_lr_scale = 1.0 / self.mapping_scales.mean().item()

    @no_grad()
    def init_mapping_scales(self) -> None:
        """Helper function to initialize the mapping scales used to scale the
        weights in digital and determine the conductance conversion.

        Note:
            This method is called from the constructor.
        """
        if not hasattr(self.rpu_config, "mapping"):
            self.set_mapping_scales(None)
            return

        mapping = self.rpu_config.mapping  # type: MappingParameter
        mapping_scales = None
        if mapping.weight_scaling_omega:
            if mapping.weight_scaling_columnwise:
                mapping_scales = ones(
                    (self.out_size,),
                    dtype=self.get_dtype(),
                    device=self.device,
                    requires_grad=False,
                )
            else:
                mapping_scales = ones(
                    (1,), dtype=self.get_dtype(), device=self.device, requires_grad=False
                )
                self.set_mapping_scales(mapping_scales)

    @no_grad()
    def init_input_processing(self) -> bool:
        """Helper function to initialize the input processing.

        Note:
            This method is called from the constructor.

        Returns:
            whether input processing is enabled

        Raises:
            ConfigError: in case ``manage_output_clipping`` is
                enabled but not supported.
        """
        self.input_range = None

        if not isinstance(self.rpu_config, PrePostProcessingRPU):
            return False  # type: ignore

        ir_params = self.rpu_config.pre_post.input_range  # type: InputRangeParameter
        if ir_params.enable:
            self.input_range_update_idx = Parameter(
                full((1,), 0.0, device=self.device, requires_grad=False)
            )
            if ir_params.learn_input_range:
                self.input_range = Parameter(
                    full(
                        (1,),
                        ir_params.init_value,
                        dtype=self.get_dtype(),
                        device=self.device,
                        requires_grad=True,
                    )
                )
            else:
                input_range = full(
                    (1,),
                    ir_params.init_value,
                    dtype=self.get_dtype(),
                    device=self.device,
                    requires_grad=False,
                )
                if hasattr(self, "input_range") and self.input_range is None:
                    delattr(self, "input_range")
                self.register_buffer("input_range", input_range)  # type: ignore

            if ir_params.manage_output_clipping and not ir_params.supports_manage_output_clipping(
                self.rpu_config
            ):
                raise ConfigError("RPU Config does not support `manage_output_clipping`.")
            return True
        return False

    @no_grad()
    def set_input_range(self, value: Union[Tensor, float]) -> None:
        """Sets the input range.

        Args:
           value: input range value

        Raises:
             ConfigError: in case input range is None
        """
        if self.input_range is None:
            raise ConfigError("Input range is not enabled")
        if isinstance(value, Tensor):
            input_range = value[0].item()
        else:
            input_range = value
        if isinstance(self.input_range, Parameter):
            self.input_range.data[0] = abs(input_range)
        else:
            self.input_range[0] = abs(input_range)

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
        """Get all scales with a new scale.

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

        if not hasattr(self.rpu_config, "mapping"):
            return

        mapping = self.rpu_config.mapping
        if mapping.learn_out_scaling:
            if mapping.out_scaling_columnwise:
                self.out_scaling_alpha = Parameter(
                    ones(
                        (self.out_size,),
                        dtype=self.get_dtype(),
                        device=self.device,
                        requires_grad=True,
                    )
                )
            else:
                self.out_scaling_alpha = Parameter(
                    ones((1,), dtype=self.get_dtype(), device=self.device, requires_grad=True)
                )

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
            self.out_scaling_alpha.data = self.out_scaling_alpha.data.view(-1)
            self.out_scaling_alpha.data[:] = as_tensor(alpha).to(self.device).view(-1)
        elif isinstance(self.out_scaling_alpha, Tensor):
            self.out_scaling_alpha = self.out_scaling_alpha.view(-1)
            self.out_scaling_alpha[:] = as_tensor(alpha).to(self.device).view(-1)
        else:
            self.out_scaling_alpha = as_tensor(alpha).to(self.device).view(-1)

    def apply_out_scaling(
        self, values: Tensor, tensor_view: Optional[Tuple[int, ...]] = None
    ) -> Tensor:
        """Apply the learned out scaling to the given tensor.

        Args:
            values: tensor to apply scaling to.
            tensor_view: view to cast the out scalings before multiplication

        Returns:
            output tensor with applied out scaling factors
        """
        if self.out_scaling_alpha is not None:
            if tensor_view is None:
                tensor_view = self.get_tensor_view(values.dim())
            return values * self.out_scaling_alpha.view(*tensor_view)
        return values

    def apply_input_range(self, values: Tensor, update_from_data: bool = False) -> Tensor:
        """Apply the input clipping.

        Args:
            values: tensor to clip
            update_from_data: whether to update from data if applicable

        Returns:
            clipped output tensor
        """

        if self.input_range is None:
            return values

        if isinstance(self.rpu_config, PrePostProcessingRPU) and update_from_data:
            ir_params = self.rpu_config.pre_post.input_range
            idx = self.input_range_update_idx
            if idx < ir_params.init_from_data:
                std = values.std()
                if std > 0.0:
                    self.input_range.data = (
                        self.input_range.data * idx + ir_params.init_std_alpha * std
                    ) / (idx + 1)
                    self.input_range_update_idx.data += 1

                self.input_range.data = self.input_range.data.abs()

        return clamp(
            values,
            min=-abs(self.input_range.item()),  # pylint: disable=invalid-unary-operand-type
            max=abs(self.input_range.item()),
        )

    def pre_forward(
        self, x_input: Tensor, dim: int, is_test: bool = False, ctx: Any = None
    ) -> Tensor:
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

    def post_forward(
        self, x_output: Tensor, dim: int, is_test: bool = False, ctx: Any = None
    ) -> Tensor:
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

        bound = None
        if self.handle_output_bound:
            bound = self.get_forward_out_bound()  # pylint: disable=assignment-from-none
        if bound is not None and ctx is not None:
            # pylint: disable=invalid-unary-operand-type
            grad_zero_msk = logical_or(x_output >= bound, x_output <= -bound)
            ctx.saved_analog_tensors.append(grad_zero_msk)

        scale = None
        if self.input_range is not None:
            scale = self.input_range

        if self.mapping_scales is not None:
            tensor_view = self.get_tensor_view(x_output.dim(), dim)
            if scale is not None:
                scale = scale * self.get_mapping_scales().view(tensor_view)
            else:
                scale = self.get_mapping_scales().view(tensor_view)

        if scale is not None:
            return x_output * scale
        return x_output

    def joint_forward(self, x_input: Tensor, is_test: bool = False, ctx: Any = None) -> Tensor:
        """Perform the forward pass.

        Calls first the ``pre_forward``, then the tile forward, and
        finally the ``post_forward`` step.

        Caution:

            This will apply the (digital) mapping scales, but
            *not* the learnable out-scales which are handled in the
            forward pass of the module

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
        x_input = self.pre_forward(x_input, 0 if self.in_trans else x_input.dim() - 1, is_test, ctx)
        x_output = self.tile.forward(
            x_input, self.analog_bias, self.in_trans, self.out_trans, is_test, self.non_blocking
        )
        return self.post_forward(
            x_output, 0 if self.out_trans else x_output.dim() - 1, is_test, ctx
        )

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
            tensor_view = self.get_tensor_view(d_input.dim(), dim)
            d_input = d_input * self.get_mapping_scales().view(tensor_view)

        if self.handle_output_bound and ctx is not None:
            zero_grad_msk = (
                None if len(ctx.saved_analog_tensors) < 2 else ctx.saved_analog_tensors[1]
            )
            if zero_grad_msk is not None:
                d_input = where(zero_grad_msk, 0.0, d_input)

        return d_input

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
            x_input = ctx.saved_analog_tensors[0]
            ir_params = self.rpu_config.pre_post.input_range  # type: ignore

            upper_thres = x_input >= self.input_range
            lower_thres = x_input <= -self.input_range  # pylint: disable=invalid-unary-operand-type

            grad = zeros_like(self.input_range)

            grad += clamp(upper_thres * d_output, min=None, max=0.0).sum()
            grad -= clamp(lower_thres * d_output, min=0.0, max=None).sum()

            if ir_params.gradient_relative:
                grad *= self.input_range
                grad *= ir_params.gradient_scale

            zero_grad_msk = (
                None if len(ctx.saved_analog_tensors) < 2 else ctx.saved_analog_tensors[1]
            )
            if ir_params.manage_output_clipping and zero_grad_msk is not None:
                output_percentage = 1.0 - zero_grad_msk.count_nonzero().item() / d_output.numel()
                grad -= (
                    (1.0 - output_percentage)
                    * self.input_range
                    * (output_percentage < ir_params.output_min_percentage)
                )

            if ir_params.decay > 0:
                percentage = (x_input.abs() < self.input_range).float().mean()
                grad += (
                    ir_params.decay
                    * self.input_range
                    * (percentage > ir_params.input_min_percentage)
                )

            if self.input_range.grad is None:
                self.input_range.grad = grad
            else:
                self.input_range.grad += grad

        return d_output

    def backward(self, d_input: Tensor, ctx: Any = None) -> Tensor:
        """Perform the backward pass.

        Args:
            d_input: ``[N, out_size]`` tensor. If ``out_trans`` is set, transposed.
            ctx: torch auto-grad context [Optional]

        Returns:
            torch.Tensor: ``[N, in_size]`` tensor. If ``in_trans`` is set, transposed.
        """
        d_input = self.pre_backward(d_input, 0 if self.out_trans else d_input.dim() - 1, ctx)
        d_output = self.tile.backward(  # type: ignore
            d_input, self.analog_bias, self.out_trans, self.in_trans, self.non_blocking
        )
        return self.post_backward(d_output, 0 if self.in_trans else d_output.dim() - 1, ctx)

    def pre_update(
        self, x_input: Tensor, x_dim: int, d_input: Tensor, d_dim: int
    ) -> Tuple[Tensor, Tensor]:
        """Operations before the actual update step for pre processing.

        By default, if the mapping scales are used, the ``d_input``
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
            tensor_view = self.get_tensor_view(d_input.dim(), d_dim)
            return x_input, d_input / (
                self.get_mapping_scales().view(tensor_view) * self.mapping_lr_scale
            )
        return x_input, d_input

    def update(self, x_input: Tensor, d_input: Tensor) -> None:
        """Perform the update pass.

        Calls the ``pre_update`` method to pre-process the inputs.

        Args:
            x_input: ``[..., in_size]`` tensor. If ``in_trans`` is set, ``[in_size, ...]``.
            d_input: ``[..., out_size]`` tensor. If ``out_trans`` is set, ``[out_size, ...]``.

        Returns:
            None
        """
        x_input, d_input = self.pre_update(
            x_input,
            0 if self.in_trans else x_input.dim() - 1,
            d_input,
            0 if self.out_trans else d_input.dim() - 1,
        )
        return self.tile.update(  # type: ignore
            x_input, d_input, self.analog_bias, self.in_trans, self.out_trans, self.non_blocking
        )

    @no_grad()
    def cuda(self, device: Optional[Union[torch_device, str, int]] = None) -> "BaseTile":
        """Return a copy of this tile in CUDA memory.

        Args:
            device: CUDA device

        Returns:
            Self with the underlying buffers to CUDA memory.
        """
        if self.mapping_scales is not None:
            self.mapping_scales = self.mapping_scales.cuda(device)
        return self

    @no_grad()
    def cpu(self) -> "BaseTile":
        """Return a copy of this tile in CPU memory.

        Returns:
            Self with the underlying buffers moved to CPU memory.

        """
        if self.mapping_scales is not None:
            self.mapping_scales = self.mapping_scales.cpu()
        return self

    def is_indexed(self) -> bool:
        """Returns whether index matrix for convolutions has been set.

        Returns:
            Whether index matrix has been set

        Raises:
            TileError: if `has_matrix_indices` method is not avialable
        """
        if not self.supports_indexed:
            return False
        if not hasattr(self.tile, "has_matrix_indices"):
            raise TileError("Expects to find `has_matrix_indices` for indexed interface.")
        return self.tile.has_matrix_indices()

    def set_indexed(self, indices: Tensor, image_sizes: List) -> None:
        """Set the index matrix for convolutions and switches to
        indexed forward/backward/update versions.

        Args:
            indices : torch.tensor with int indices
            image_sizes: [C_in, H_in, W_in, H_out, W_out] sizes

        Raises:
            ValueError: if ``image_sizes`` does not have valid dimensions
            TileError: if the tile uses transposition or indexed not supported..
        """
        if isinstance(self.tile, SimulatorTile):
            raise TileError("Only RPUCuda simulator tiles support indexed interface")

        if len(image_sizes) not in (3, 5, 7):
            raise ValueError(
                "image_sizes expects 3, 5 or 7 sizes "
                "[C_in, (D_in), H_in, (W_in), (D_out), H_out, (W_out)]"
            )

        if self.in_trans or self.out_trans:
            raise TileError("Transposed indexed versions not supported (assumes NC(D)HW)")
        self.analog_ctx.set_indexed()
        self.image_sizes = image_sizes
        self.tile.set_matrix_indices(indices)

    @no_grad()
    def joint_forward_indexed(
        self, x_input: Tensor, is_test: bool = False, ctx: Any = None
    ) -> Tensor:
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
            raise TileError("self.image_sizes is not initialized. Please use set_indexed()")

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
            raise TileError("self.image_sizes length is not 3, 5 or 7")

        x_input = self.pre_forward(x_input, 1, is_test, ctx)
        x_output = self.tile.forward_indexed(  # type: ignore
            x_input, d_tensor, is_test, self.non_blocking
        )
        return self.post_forward(x_output, 1, is_test, ctx)

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
            raise TileError("self.image_sizes is not initialized. Please use set_indexed()")

        n_batch = d_input.size(0)

        if len(self.image_sizes) == 3:
            channel_in, height_in, _ = self.image_sizes
            x_tensor = d_input.new_empty((n_batch, channel_in, height_in))
        elif len(self.image_sizes) == 5:
            channel_in, height_in, width_in, _, _ = self.image_sizes
            x_tensor = d_input.new_empty((n_batch, channel_in, height_in, width_in))
        elif len(self.image_sizes) == 7:
            channel_in, depth_in, height_in, width_in, _, _, _ = self.image_sizes
            x_tensor = d_input.new_empty((n_batch, channel_in, depth_in, height_in, width_in))
        else:
            raise TileError("self.image_sizes length is not 3, 5 or 7")

        d_input = self.pre_backward(d_input, 1, ctx)
        d_output = self.tile.backward_indexed(d_input, x_tensor, self.non_blocking)  # type: ignore
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
        return self.tile.update_indexed(x_input, d_input, self.non_blocking)  # type: ignore

    def set_learning_rate(self, learning_rate: Optional[float]) -> None:
        """Set the tile learning rate.

        Set the tile learning rate to ``-learning_rate``. Note that the
        learning rate is always taken to be negative (because of the meaning in
        gradient descent) and positive learning rates are not supported.

        Args:
            learning_rate: the desired learning rate.
        """
        if learning_rate is not None:
            self.tile.set_learning_rate(learning_rate * self.mapping_lr_scale)

    def get_learning_rate(self) -> float:
        """Return the tile learning rate.

        Returns:
            float: the tile learning rate.
        """
        return self.tile.get_learning_rate() / self.mapping_lr_scale
