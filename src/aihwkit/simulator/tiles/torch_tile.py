# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023 IBM. All Rights Reserved.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Low level implementation of torch-based tile."""

from typing import Union, Callable, Any, Optional, TYPE_CHECKING
from numbers import Number

from torch import Tensor, zeros, randn_like, clamp, bmm, randn
from torch.nn import Parameter, Module
from torch.autograd import no_grad

from aihwkit.exceptions import TorchTileConfigError, ConfigError, AnalogBiasConfigError
from aihwkit.simulator.tiles.base import SimulatorTile
from aihwkit.simulator.tiles.utils import UniformQuantize
from aihwkit.simulator.parameters.enums import (
    NoiseManagementType,
    BoundManagementType,
    AnalogMVType,
    WeightModifierType,
    WeightNoiseType,
    WeightClipType,
    WeightRemapType,
)
from aihwkit.simulator.parameters.training import IOParameters
from aihwkit.simulator.parameters.inference import (
    WeightModifierParameter,
    WeightClipParameter,
    WeightRemapParameter,
)

if TYPE_CHECKING:
    from aihwkit.simulator.configs.configs import TorchInferenceRPUConfig


class TorchSimulatorTile(SimulatorTile, Module):
    """Torch based tile class.

    Args:
        x_size: input size
        d_size: output size
        rpu_config: resistive processing unit configuration.
    """

    # pylint: disable=abstract-method

    def __init__(
        self, x_size: int, d_size: int, rpu_config: "TorchInferenceRPUConfig", bias: bool = False
    ):
        Module.__init__(self)
        self.x_size = x_size
        self.d_size = d_size

        if bias:
            raise AnalogBiasConfigError("Analog bias is not supported for TorchSimulatorTile")
        TorchSimulatorTile.check_rpu_config_support(rpu_config)

        self._f_io = rpu_config.forward
        self._modifier = rpu_config.modifier
        dtype = rpu_config.runtime.data_type.as_torch()
        if self._f_io.out_noise_std > 0:
            out_noise_values = (self._f_io.out_noise * (1.0 + randn((d_size,), dtype=dtype))).abs()
            self.register_buffer("out_noise_values", out_noise_values)
        else:
            self.out_noise_values = None

        # Don't use randn here to avoid changing the seed in
        # comparison to RPUCuda tiles
        self.weight = Parameter(zeros(self.d_size, self.x_size, dtype=dtype))

    def set_weights(self, weight: Tensor) -> None:
        """Set the tile weights.

        Args:
            weight: ``[out_size, in_size]`` weight matrix.
        """
        self.weight.data = weight.clone().to(self.weight.device)

    def get_weights(self) -> Tensor:
        """Get the tile weights.

        Returns:
            a tuple where the first item is the ``[out_size, in_size]`` weight
            matrix; and the second item is either the ``[out_size]`` bias vector
            or ``None`` if the tile is set not to use bias.
        """
        return self.weight.data.detach().cpu()

    def get_x_size(self) -> int:
        """Returns input size of tile"""
        return self.x_size

    def get_d_size(self) -> int:
        """Returns output size of tile"""
        return self.d_size

    def get_brief_info(self) -> str:
        """Returns a brief info"""
        return self.__class__.__name__ + "({})".format(self.extra_repr())

    def extra_repr(self) -> str:
        return "{}, {}, {}".format(self.d_size, self.x_size, self.weight.device).rstrip()

    @no_grad()
    def remap_weights(self, remap: WeightRemapParameter, scales: Tensor) -> Tensor:
        """
        Remap the weights to the specified range and return new scales.

        Args:
            remap: hyper-parameters defining the remapping
            scales: current scale values.

        Raises:
            ConfigError: If WeightRemapType is unknown.

        Returns:
            Tensor: New scales.
        """
        # pylint: disable=arguments-differ

        scaled_weights = self.weight * scales.view(-1, 1)
        if remap.type == WeightRemapType.LAYERWISE_SYMMETRIC:
            new_scale = scaled_weights.abs().max()
            self.weight.data = scaled_weights / new_scale
            return new_scale.view(-1)
        if remap.type == WeightRemapType.CHANNELWISE_SYMMETRIC:
            new_scale, _ = scaled_weights.abs().max(1)
            self.weight.data = scaled_weights / new_scale.view(-1, 1)
            return new_scale

        raise TorchTileConfigError(f"Unknown weight remap type {remap.type}")

    @no_grad()
    def clip_weights(self, clip: WeightClipParameter) -> None:
        """Clip the weights.

        Args:
            clip: parameters specifying the clipping methof and type.

        Raises:
            NotImplementedError: For unsupported WeightClipTypes
            ConfigError: If unknown WeightClipType used.
        """

        if clip.type == WeightClipType.FIXED_VALUE:
            self.weight.data = clamp(self.weight, -clip.fixed_value, clip.fixed_value)
        elif clip.type == WeightClipType.LAYER_GAUSSIAN:
            alpha = self.weight.std() * clip.sigma
            if clip.fixed_value > 0:
                alpha = min(clip.fixed_value, alpha)
            self.weight.data = clamp(self.weight, -alpha, alpha)

        elif clip.type == WeightClipType.AVERAGE_CHANNEL_MAX:
            raise NotImplementedError
        else:
            raise TorchTileConfigError(f"Unknown clip type {clip.type}")

    def set_config(self, rpu_config: "TorchInferenceRPUConfig") -> None:
        """Updated the configuration to allow on-the-fly changes.

        Args:
            rpu_config: configuration to use in the next forward passes.
        """
        self._f_io = rpu_config.forward
        self._modifier = rpu_config.modifier

    def register_weight_hook(self, hook: Callable) -> Any:
        """Register a hook to the weights."""
        return self.weight.register_hook(hook)

    def forward(
        self,
        x_input: Tensor,
        bias: bool = False,
        in_trans: bool = False,
        out_trans: bool = False,
        is_test: bool = False,
        non_blocking: bool = False,
    ) -> Tensor:
        # pylint: disable=too-many-locals, too-many-branches

        if in_trans or out_trans:
            raise TorchTileConfigError("Non-trans MVMs supported only.")

        if not is_test:
            noisy_weights = TorchSimulatorTile.modify_weight(
                self.weight, self._modifier, x_input.shape[0]
            )
        else:
            noisy_weights = self.weight

        return AnalogMVM.matmul(
            noisy_weights, x_input, self._f_io, False, out_noise_values=self.out_noise_values
        )

    @staticmethod
    def modify_weight(
        inp_weight: Tensor, modifier: WeightModifierParameter, batch_size: int
    ) -> Tensor:
        """Weight modifier that adds noise to the weights according to rpu config.

        Args:
            inp_weight: Input weights.
            modifier: Noise injection configuration.
            batch_size (int): Batch size.

        Raises:
            TorchTileConfigError: Unsupported/unknown weight modifier type.

        Returns:
            Weights with noise injected.
        """
        per_batch_sample = modifier.per_batch_sample
        target_shape = (batch_size,) + inp_weight.shape if per_batch_sample else inp_weight.shape
        if modifier.type in [WeightModifierType.NONE, WeightModifierType.COPY]:
            return inp_weight

        if modifier.type == WeightModifierType.MULT_NORMAL:
            with no_grad():
                gauss = randn(size=target_shape, device=inp_weight.device)
                noise = inp_weight * modifier.std_dev * gauss
            out_weight = inp_weight.clone() + noise
            return out_weight

        assumed_wmax = modifier.assumed_wmax
        if modifier.rel_to_actual_wmax:
            assumed_wmax = inp_weight.abs().max()

        if modifier.type == WeightModifierType.DISCRETIZE:
            # - Discretize the weights on the fly and backprob through them
            out_weight = inp_weight.clone().view(target_shape)
            out_weight = UniformQuantize.apply(
                out_weight, modifier.res, assumed_wmax, modifier.sto_round
            )
        elif modifier.type == WeightModifierType.ADD_NORMAL:
            with no_grad():
                noise = (
                    modifier.std_dev * assumed_wmax * randn(target_shape, device=inp_weight.device)
                )
            out_weight = inp_weight.clone() + noise
        else:
            raise TorchTileConfigError(f"Weight modifier {modifier} not supported")
        return out_weight

    @staticmethod
    def check_rpu_config_support(rpu_config: "TorchInferenceRPUConfig") -> None:
        """Check the RPUConfig for support with TorchSimulatorTile

        Throws an assertion error when there is an incompatibility
        with the used rpu config and what the current torch tile
        supports

        Args:
            rpu_config: the rpu config to be checked

        Raises:
            TorchTileConfigError, ConfigError: in case a feature is not supported
        """
        # pylint: disable=too-many-branches

        AnalogMVM.check_support(rpu_config.forward)

        if rpu_config.clip.type == WeightClipType.AVERAGE_CHANNEL_MAX:
            raise TorchTileConfigError("Clip type AVERAGE_CHANNEL_MAX not supported by torch tile")

        if rpu_config.modifier.enable_during_test:
            raise TorchTileConfigError("Modifier noise is currently always off in the torch tile")

        if rpu_config.modifier.copy_last_column:
            raise TorchTileConfigError("Bias is assumed to be in digital only for torch tile")

        if rpu_config.modifier.type in [
            WeightModifierType.DOREFA,
            WeightModifierType.POLY,
            WeightModifierType.PROG_NOISE,
            WeightModifierType.PCM_NOISE,
            WeightModifierType.DISCRETIZE_ADD_NORMAL,
        ]:
            raise TorchTileConfigError(
                "The given modifier noise type is not supported in the torch tile"
            )

        if rpu_config.modifier.pdrop > 0.0:
            raise TorchTileConfigError("The drop-connect is not supported in the torch tile")

        if rpu_config.remap.type not in [
            WeightRemapType.LAYERWISE_SYMMETRIC,
            WeightRemapType.CHANNELWISE_SYMMETRIC,
            WeightRemapType.NONE,
        ]:
            raise TorchTileConfigError("Remapping type not supported.")

        if rpu_config.remap.remapped_wmax != 1.0:
            raise TorchTileConfigError("Remapping to value different from 1.0 not supported.")

        if rpu_config.remap.max_scale_range != 0.0:
            raise TorchTileConfigError("Remap parameter max_scale_range must be 0.0.")

        if rpu_config.remap.max_scale_ref != 0.0:
            raise TorchTileConfigError("Remap parameter max_scale_ref must be 0.0.")


class AnalogMVM:
    """Torch implementation of (part of) the IO-managed forward /
    backward pass in RPUCuda.
    """

    # pylint: disable=too-many-locals, too-many-branches

    @staticmethod
    def _matmul(weight: Tensor, input_: Tensor, trans: bool = False) -> Tensor:
        """The inner FP GEMM."""
        if weight.ndim == 3:
            if input_.ndim == 2:
                if not trans:
                    return bmm(input_[:, None, :], weight.permute(0, 2, 1))[:, 0, :]
                return bmm(input_[:, None, :], weight)[:, 0, :]
            if not trans:
                return bmm(input_, weight.permute(0, 2, 1))
            return bmm(input_, weight)
        if not trans:
            return input_ @ weight.T
        return input_ @ weight

    @staticmethod
    def matmul(
        weight: Tensor, input_: Tensor, io_pars: IOParameters, trans: bool = False, **fwd_pars: Any
    ) -> Tensor:
        """Noisy, io-managed mat-mul.

        Args:
            weight: weight matrix (``out_size``, ``in_size``)
            input_: activation (m, ``in_size`` / ``out_size``)
            io_pars: Parameter defining the mat-mul nonlinearities
            trans : transpose of the weight (so that ``in_size`` and
                ``out_size`` is transposed).
            fwd_pars: additional parameter dictionary
        Returns:
            Result tensor
        """

        if io_pars.is_perfect:
            return AnalogMVM._matmul(weight, input_, trans)

        nm_scale_values = AnalogMVM._compute_noise_management(
            input_=input_, nm_type=io_pars.noise_management, io_pars=io_pars
        )

        out_size = input_.shape[:-1] + (weight.shape[int(trans)],)
        n_management = io_pars.noise_management != NoiseManagementType.NONE
        b_management = io_pars.bound_management != BoundManagementType.NONE
        if (
            io_pars.inp_noise <= 0.0
            and isinstance(nm_scale_values, Tensor)
            and (nm_scale_values == 0.0).all()
        ):
            # - Shortcut, output would be all zeros
            return zeros(size=out_size, device=input_.device, dtype=input_.dtype())

        if isinstance(nm_scale_values, Tensor):
            # set zeros to 1 to avoid divide-by-zero errors
            nm_scale_values[nm_scale_values <= 0.0] = 1.0

        out_scale = io_pars.out_scale
        scale = 1.0
        scaling = False
        if not b_management:
            # Fast path without bound management
            if n_management:
                scale = scale / nm_scale_values
                scaling = not (isinstance(scale, Number) and scale == 1.0)

            _, output = AnalogMVM._compute_analog_mv(
                weight=weight,
                input_=input_,
                trans=trans,
                scale=scale,
                scaling=scaling,
                io_pars=io_pars,
                **fwd_pars,
            )

            if scaling or out_scale != 1.0:
                output *= out_scale / scale
            return output

        # with bound management
        bound_test_passed = False
        reduction_due_to_bound_management = 0.5

        inp_res = io_pars.inp_res
        if inp_res > 0:
            inp_res = 1 / inp_res if inp_res > 1.0 else inp_res
            inp_res *= 2.0 * io_pars.inp_bound

        bm_round = 0
        while not bound_test_passed:
            bound_test_passed = True
            reduction_due_to_bound_management *= 2.0

            bm_round += 1

            scaling = False
            scale = 1.0

            if n_management:
                scale /= nm_scale_values
                scaling = not (isinstance(scale, Number) and scale == 1.0)

            if b_management:
                scale /= reduction_due_to_bound_management
                scaling = not (isinstance(scale, Number) and scale == 1.0)

            bound_test_passed, output = AnalogMVM._compute_analog_mv(
                weight=weight,
                input_=input_,
                trans=trans,
                scale=scale,
                scaling=scaling,
                io_pars=io_pars,
                **fwd_pars,
            )

            bound_test_passed = bound_test_passed or (
                (reduction_due_to_bound_management > io_pars.max_bm_factor)
                or (
                    (inp_res > 0.0)
                    and (reduction_due_to_bound_management > io_pars.max_bm_res / inp_res)
                )
            )

        # - Final scaling
        if scaling or out_scale != 1.0:
            output *= out_scale / scale
        return output

    @staticmethod
    def _compute_analog_mv(
        weight: Tensor,
        input_: Tensor,
        trans: bool,
        scale: float,
        scaling: bool,
        io_pars: IOParameters,
        **fwd_pars: Any,
    ) -> Tensor:
        """
        Prepare input, perform noisy MVM and finalize output. Takes care of noise/bound
        management and discretization.

        Args:
            weight: Weight tensor.
            input_: Input tensor in format [N, in_size].
            trans: whether to transpose the weight
            scale: Scale for scaling the input.
            scaling: Whether to scale.
            io_pars: forward pass configuration.
            fwd_pars: additional forward parameters

        Returns:
            Whether the bound management test passed and the result.

        Raises:
            NotImplementedError: If choices in the rpu config were made that are not supported.
            ConfigError: If unknown AnalogMVType

        """
        prepared_input = AnalogMVM._prepare_input(
            input_=input_,
            scale=scale,
            scaling=scaling,
            with_asymmetry=io_pars.inp_asymmetry != 0.0,
            io_pars=io_pars,
        )
        if io_pars.mv_type == AnalogMVType.ONE_PASS:
            # - Perform the noisy MVM

            out_values = AnalogMVM._matmul(weight, prepared_input, trans=trans)

            bound_test_passed, finalized_outputs = AnalogMVM._finalize_output(
                out_values=out_values, io_pars=io_pars, **fwd_pars
            )
        elif io_pars.mv_type in [
            AnalogMVType.POS_NEG_SEPARATE,
            AnalogMVType.POS_NEG_SEPARATE_DIGITAL_SUM,
        ]:
            raise NotImplementedError
        else:
            raise ConfigError(f"Unknown AnalogMVType {io_pars.mv_type}")
        return bound_test_passed, finalized_outputs

    @staticmethod
    def _finalize_output(
        out_values: Tensor,
        io_pars: IOParameters,
        out_noise_values: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        """Discretize the output if applicable.

        Args:
            out_values: Finalized input.
            io_pars: IO parameter.
            out_noise_values: if given, individual out noise values
                per output. Otherwise out_noise is used
            kwargs: dummy container
        Returns:
            Finalized output.
        """
        # pylint: disable=unused-argument

        bound_test_passed = True
        bound = io_pars.out_bound if io_pars.out_bound > 0 else float("inf")
        asymmetry_scale = 1.0 - io_pars.out_asymmetry
        res = io_pars.out_res

        if io_pars.out_asymmetry != 0.0:
            out_values[out_values < 0] *= asymmetry_scale
        if io_pars.out_noise > 0.0 or out_noise_values is not None:
            if out_noise_values is None:
                out_values += io_pars.out_noise * randn_like(out_values)
            else:
                out_values += randn_like(out_values) * out_noise_values.view(
                    1, -1, *((1,) * (out_values.ndim - 2))
                )

        # - Discretize
        if res > 0:
            out_values = UniformQuantize.apply(
                out_values, res, io_pars.out_bound, io_pars.out_sto_round
            )
        not_clipping = ((out_values <= bound) & (out_values >= -bound)).all()
        out_values = clamp(out_values, min=-bound, max=bound)
        if io_pars.bound_management != BoundManagementType.NONE:
            bound_test_passed = not_clipping

        return bound_test_passed, out_values

    @staticmethod
    def _prepare_input(
        input_: Tensor, scale: Tensor, scaling: bool, with_asymmetry: bool, io_pars: IOParameters
    ) -> Tensor:
        """Quantization and scaling of the input.

        Args:
            input_: Input to the tile.
            scale: Scaling of the input.
            scaling: Whether to scale or not.
            with_asymmetry: Asymmetrically scale.
            io_pars: forward pass configuration.

        Returns:
            torch.Tensor: Discretized input.
        """
        bound = io_pars.inp_bound if io_pars.inp_bound > 0 else float("inf")
        noise = io_pars.inp_noise
        asymmetry_scale = 1.0 - io_pars.inp_asymmetry
        res = io_pars.inp_res
        scaled_input = input_.clone()

        # - Scale
        if scaling:
            scaled_input *= scale

        # - Discretize
        if io_pars.inp_res > 0:
            scaled_input = UniformQuantize.apply(
                scaled_input, res, io_pars.inp_bound, io_pars.inp_sto_round
            )

        # - Clip between -bound and bound
        if no_grad():
            scaled_input = clamp(scaled_input, min=-bound, max=bound)

        if noise > 0.0:
            # - Apply input noise
            scaled_input += noise * randn_like(scaled_input)

        if with_asymmetry:
            scaled_input[scaled_input < 0] *= asymmetry_scale

        return scaled_input

    @staticmethod
    def _compute_noise_management(
        input_: Tensor, nm_type: NoiseManagementType, io_pars: IOParameters, axis: int = -1
    ) -> Union[float, Tensor]:
        """Returns scale based on noise management strategy by which the input is scaled.

        Args:
            input_: Input tensor.
            nm_type: Noise management type.
            io_pars: parameter specifying the analog forward pass nonidealities
            axis: dimension to take compute the mangement (default -1)
        Raises:
            ConfigError: If NoiseManagementType unknown.

        Returns:
            Scales for the input
        """
        if nm_type == NoiseManagementType.NONE:
            return 1.0
        if nm_type == NoiseManagementType.ABS_MAX:
            abs_max = input_.abs().max(axis=axis, keepdim=True)[0]
            if io_pars.nm_thres > 0.0:
                return clamp(abs_max, max=io_pars.nm_thres)
            return abs_max
        if nm_type == NoiseManagementType.CONSTANT:
            return io_pars.nm_thres if io_pars.nm_thres > 0.0 else 1.0
        if nm_type == NoiseManagementType.MAX:
            _max = input_.max(axis=axis, keepdim=True)[0]
            if io_pars.nm_thres > 0.0:
                return clamp(_max, max=io_pars.nm_thres)
            return _max
        raise ConfigError(f"Unknown NoiseManagementType {nm_type}")

    @staticmethod
    def check_support(io_pars: IOParameters) -> None:
        """Check whether the IO settings are supported.

        Throws an assertion error when there is an incompatibility

        Args:
            io_pars: the IOParameters to be checked

        Raises:
            ConfigError: in case a feature is not supported
        """
        # pylint: disable=too-many-branches

        if io_pars.mv_type != AnalogMVType.ONE_PASS:
            raise ConfigError("Only AnalogMVType.ONE_PASS supported as forward.mv_type")

        if io_pars.bound_management == BoundManagementType.SHIFT:
            raise ConfigError("Shift bound management not supported in torch tile")

        if io_pars.noise_management in [
            NoiseManagementType.AVERAGE_ABS_MAX,
            NoiseManagementType.ABS_MAX_NP_SUM,
        ]:
            raise ConfigError("Special noise mangement types not supported.")

        if io_pars.w_noise > 0.0 or io_pars.w_noise_type != WeightNoiseType.NONE:
            raise ConfigError("forward.w_noise not supported in torch tile")

        if io_pars.ir_drop > 0.0:
            raise ConfigError("IR drop not supported in torch tile")

        if io_pars.out_nonlinearity > 0.0:
            raise ConfigError("S-shaped non-linearity not supported in torch tile")

        if io_pars.slope_calibration > 0.0:
            raise ConfigError("Slope calibration not supported in torch tile")

        if io_pars.v_offset_std > 0.0 or io_pars.v_offset_w_min > 0.0 or io_pars.r_series > 0.0:
            raise ConfigError("Voltage offset or R-series not supported in torch tile")

        if io_pars.w_read_asymmetry_dtod > 0.0:
            raise ConfigError("Device polarity read dependence is not supported in torch tile")
