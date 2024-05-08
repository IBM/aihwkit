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

"""Implementation of analog MVM for torch tiles."""

from typing import Union, Any, Optional
from numbers import Number

from torch import Tensor, zeros, randn_like, clamp, bmm
from torch.autograd import no_grad

from aihwkit.exceptions import ConfigError
from aihwkit.simulator.tiles.utils import UniformQuantize
from aihwkit.simulator.parameters.enums import (
    NoiseManagementType,
    BoundManagementType,
    AnalogMVType,
    WeightNoiseType,
)
from aihwkit.simulator.parameters.io import IOParameters


class AnalogMVM:
    """Torch implementation of (part of) the IO-managed forward /
    backward pass in RPUCuda.
    """

    # pylint: disable=too-many-locals, too-many-branches

    @classmethod
    def _matmul(cls, weight: Tensor, input_: Tensor, trans: bool = False) -> Tensor:
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

    @classmethod
    def matmul(
        cls,
        weight: Tensor,
        input_: Tensor,
        io_pars: IOParameters,
        trans: bool = False,
        is_test: bool = False,
        **fwd_pars: Any,
    ) -> Tensor:
        """Noisy, io-managed mat-mul.

        Args:
            weight: weight matrix (``out_size``, ``in_size``)
            input_: activation (m, ``in_size`` / ``out_size``)
            io_pars: Parameter defining the mat-mul nonlinearities
            trans : transpose of the weight (so that ``in_size`` and
                ``out_size`` is transposed).
            is_test: whether testing or training mode
            fwd_pars: additional parameter dictionary
        Returns:
            Result tensor
        """

        if io_pars.is_perfect or io_pars.mv_type == AnalogMVType.IDEAL:
            return cls._matmul(weight, input_, trans)

        nm_scale_values = cls._compute_noise_management(
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
            return zeros(size=out_size, device=input_.device, dtype=input_.dtype)

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

            _, output = cls._compute_analog_mv(
                weight=weight,
                input_=input_,
                trans=trans,
                scale=scale,
                scaling=scaling,
                is_test=is_test,
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

            bound_test_passed, output = cls._compute_analog_mv(
                weight=weight,
                input_=input_,
                trans=trans,
                scale=scale,
                scaling=scaling,
                is_test=is_test,
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

    @classmethod
    def _compute_analog_mv(
        cls,
        weight: Tensor,
        input_: Tensor,
        trans: bool,
        scale: float,
        scaling: bool,
        is_test: bool,  # pylint: disable=unused-argument
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
            is_test: whether test or training mode
            io_pars: forward pass configuration.
            fwd_pars: additional forward parameters

        Returns:
            Whether the bound management test passed and the result.

        Raises:
            NotImplementedError: If choices in the rpu config were made that are not supported.
            ConfigError: If unknown AnalogMVType

        """
        prepared_input = cls._prepare_input(
            input_=input_,
            scale=scale,
            scaling=scaling,
            with_asymmetry=io_pars.inp_asymmetry != 0.0,
            io_pars=io_pars,
        )
        if io_pars.mv_type == AnalogMVType.ONE_PASS:
            # - Perform the noisy MVM

            out_values = cls._matmul(weight, prepared_input, trans=trans)

            bound_test_passed, finalized_outputs = cls._finalize_output(
                out_values=out_values, io_pars=io_pars, **fwd_pars
            )
        elif io_pars.mv_type in [
            AnalogMVType.POS_NEG_SEPARATE,
            AnalogMVType.POS_NEG_SEPARATE_DIGITAL_SUM,
            AnalogMVType.SPLIT_MODE,
            AnalogMVType.BIT_WISE,
        ]:
            raise NotImplementedError
        else:
            raise ConfigError(f"Unknown AnalogMVType {io_pars.mv_type}")
        return bound_test_passed, finalized_outputs

    @classmethod
    def _finalize_output(
        cls,
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

    @classmethod
    def _prepare_input(
        cls,
        input_: Tensor,
        scale: Tensor,
        scaling: bool,
        with_asymmetry: bool,
        io_pars: IOParameters,
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

    @classmethod
    def _compute_noise_management(
        cls, input_: Tensor, nm_type: NoiseManagementType, io_pars: IOParameters, axis: int = -1
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

    @classmethod
    def check_support(cls, io_pars: IOParameters) -> None:
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
