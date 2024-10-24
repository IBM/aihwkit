# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

# pylint: disable=too-many-locals, too-many-arguments

"""Low level implementation of torch-based tile."""

from typing import Any, Tuple, List, Optional
from math import log2

from torch import (
    Tensor,
    empty,
    zeros,
    sum as torch_sum,
    flip,
    abs as torch_abs,
    sign,
    floor,
    fmod,
    allclose,
    linspace,
    Size,
)
from torch.autograd import no_grad
from torch.nn.functional import pad

from aihwkit.extension import extension_ops  # type: ignore
from aihwkit.inference.converter.conductance import SinglePairConductanceConverter

from aihwkit.exceptions import ConfigError
from aihwkit.simulator.tiles.analog_mvm import AnalogMVM

from aihwkit.simulator.parameters.enums import (
    NoiseManagementType,
    BoundManagementType,
    AnalogMVType,
    WeightNoiseType,
)
from aihwkit.simulator.parameters.io import IOParameters, IOParametersIRDropT


class AnalogMVMIRDropT(AnalogMVM):
    """Torch Perseus implementation of (part of) the IO-managed forward /
    backward pass in RPUCuda.
    """

    # pylint: disable=arguments-differ

    @classmethod
    def _get_res(cls, res: float) -> float:
        """Return resolution as number less than 1

        Args:
            res: resolution specified either less than or greater than 1

        Returns:
            float resolution specified as a number less than 1
        """
        res = 1.0 / res if res > 1.0 else res
        assert res > 0, "resolution is <= 0"
        return res

    @classmethod
    def _interleave_cols_nd(cls, mvm1: Tensor, mvm2: Tensor) -> Tensor:
        """Interleaves two matrices along final dimension to mimic North-South ADcs,
        starting with mvm1. Can now handle n-dimensional matrices, not just 2D.

        Args:
            mvm1: ``[*, batch_size, out_size/2]`` output activations (to south ADCs)
            mvm2: ``[*, batch_size, out_size/2]`` output activations (to north ADCs)

        Returns:
            Column-wise interleaved matrix of output activations that captures
            IR drop in both directions (to north and south ADCs) in a
            symmetric tile design.
        """
        shape = Size(mvm1.shape[0:-1]) + Size([mvm1.shape[-1] + mvm2.shape[-1]])
        mvm = empty(*shape).to(mvm1.device)
        mvm[..., 0::2] = mvm1
        mvm[..., 1::2] = mvm2
        return mvm

    @classmethod
    def _pad_symmetric(
        cls, input_: Tensor, weight: Tensor, phys_input_size: int = 512
    ) -> Tuple[Tensor, Tensor]:
        """Return input_ (activations) and weights symmetrically padded
        with zeros to mimic symmetric ADCs (north and south).
        Can now handle n-dimensional input_, not just 2D.

        Args:
            input_: ``[*, batch_size, in_size]`` input_ (activations).
            weight: ``[in_size, out_size]`` weight matrix.
            phys_input_size: number of hardware tile rows

        Returns:
            Tuple[Tensor] containing the symmetrically 0-padded input_ and weight
        """
        pad1 = int((phys_input_size - weight.shape[0]) / 2)
        if pad1 == 0:
            return input_, weight
        pad2 = phys_input_size - weight.shape[0] - pad1
        input_pad_tuple = (pad1, pad2) + (0, 0) * (input_.dim() - 1)
        input_ = pad(input_, input_pad_tuple, "constant", 0)
        weight = pad(weight, (0, 0, pad1, pad2), "constant", 0)
        return input_, weight

    @classmethod
    def _prepare_inputs(
        cls,
        input_: Tensor,
        scale: Tensor,
        scaling: bool,
        with_asymmetry: bool,
        io_pars: IOParametersIRDropT,
    ) -> List[Tensor]:
        """
        Returns list of activations that will be applied to MVM tile
        depending on the mode of operation. For instance, ONE_PASS
        will simply return a list containing one set of activations
        to be applied. SPLIT_MODE will return a list of two sets
        of activations that will be applied to the MVM tile. This
        results in higher throughput/energy efficiency, but may
        amplify some output noise and sacrifice accuracy. BIT_WISE
        will return a list of activations for each bit, which
        depends on the inp_res.  SPLIT_MODE and BIT_WISE will
        appropriately bit-shift the output results to perform
        the MVM operation correctly. Can handle n-dimensional
        input activations, not just 2D.

        Args:
            input_: ``[*, batch_size, in_size]`` input activations.
            scale: scale for rescaling input activations.
            scaling: whether to rescale input activations.
            with_asymmetry:
            io_pars: forward pass configuration.

        Returns:
            List[Tensor] containing input activations to be sequentially
            applied to the MVM tile.

        Raises:
            NotImplementedError: If choices in the rpu config were made that are not supported.
            ConfigError: If unknown AnalogMVType
        """

        res = cls._get_res(io_pars.inp_res)
        prepared_input = super(AnalogMVMIRDropT, cls)._prepare_input(
            input_, scale, scaling, with_asymmetry, io_pars
        )

        if io_pars.mv_type == AnalogMVType.ONE_PASS:
            prepared_input = [prepared_input]
        elif io_pars.mv_type == AnalogMVType.SPLIT_MODE:
            n_bits = int(log2(1.0 / res + 2))
            if not log2(1.0 / res + 2) % 1 == 0:
                raise ConfigError(
                    f"inp_res={1. / res} must be of form (2**n_bits - 2) "
                    "for AnalogMVType.SPLIT_MODE"
                )

            if not io_pars.split_mode_bit_shift % 1 == 0:
                raise ConfigError(
                    f"split_mode_bit_shift={io_pars.split_mode_bit_shift}"
                    " must be integer"
                )

            if not io_pars.split_mode_bit_shift < n_bits:
                raise ConfigError(
                    f"split_mode_bit_shift={io_pars.split_mode_bit_shift} "
                    f"cannot exceed equivalent bits specified by inp_res={n_bits}."
                )

            int_input = prepared_input / (2 * res)

            upper_bits = sign(int_input) * floor(
                torch_abs(int_input) / (2**io_pars.split_mode_bit_shift)
            )
            prepared_input_msb = upper_bits * (2 * res)

            # +/- remainders
            lower_bits = fmod(int_input, 2**io_pars.split_mode_bit_shift)
            prepared_input_lsb = lower_bits * (2 * res)

            if not allclose(
                (2**io_pars.split_mode_bit_shift) * prepared_input_msb
                + prepared_input_lsb,
                prepared_input,
            ):
                raise ConfigError("Split mode pwm conversion error")

            prepared_input = [prepared_input_lsb, prepared_input_msb]

        elif io_pars.mv_type == AnalogMVType.BIT_WISE:
            int_input = prepared_input / (2.0 * res)
            n_bits = int(log2(1.0 / res + 2)) - 1
            prepared_input = []
            for _ in range(n_bits):
                # fmod for +/- remainders
                lsb = fmod(int_input, 2)
                int_input = sign(int_input) * floor(torch_abs(int_input) / 2)
                prepared_input.append(lsb * (2 * res))

        elif io_pars.mv_type in [
            AnalogMVType.POS_NEG_SEPARATE,
            AnalogMVType.POS_NEG_SEPARATE_DIGITAL_SUM,
        ]:
            raise NotImplementedError
        else:
            raise ConfigError(f"Unknown AnalogMVType {io_pars.mv_type}")
        return prepared_input

    @classmethod
    @no_grad()
    def _thev_equiv(
        cls,
        input_: Tensor,
        g_lst: List[Tensor],
        time_steps: int = 128,
        t_max: float = 1.0,
        segments: int = 8,
        r_s: float = 0.15,
        phys_input_size: int = 512,
        use_extension: bool = True,
    ) -> Tuple[Tensor, Tensor]:
        """Returns the Thevenin voltages and resistances for an entire 2D MVM
        tile as a function of time. First computes the atomic time-varying
        Thevenin equivalents for each unit cell. These are combined into
        synthetic (non-physical) segments, which serve as an approximation to
        limit computational expense. Lastly, the wire series resistance between
        segments is included when collapsing the time-varying segment Thevenin
        equivalents into one time-varying Thevenin equivalent circuit per MVM
        tile column (i.e. each element of out_size). This part is represented by
        the for loop at the end of the method. Largest indices are closest to
        the ADC. Can now handle n-dimensional input activations, not just 2D.

        Args:
            input_: ``[*, batch_size, in_size]`` MVM tile input activations

            g_lst: list of conductances in MVM tile

            t_max: max sim time, beyond which activations all zero. Can cease computation

            segments: Number of synthetic segments for IR drop calculation (approximation).
                Ideally the number of segments matches the number of unit cells per column (i.e.
                in_size). Default value is 8. Increasing beyond 8 for in_size of 512 results in
                diminishing returns for IR drop accuracy calculations while incurring higher
                computational overhead.

                Note:
                    When using the C++ extension segments is always maximized.

            r_s: wire series resistance in units of Ohms

            phys_input_size: max hardware MVM tile rows (need to 0-pad)

            use_extension: Whether to use the C++ extension operator
                for speedup if available

        Returns:
            Tuple[Tensor] containing thevenin voltages vth_3d and rth_3d where both
            have dimensions ``[*, batch_size, out_size/2, time_steps]``.
            Tensor vth_nd is given volts and rth_nd is in units of MOhms.
        """
        syn_rows_per_seg = int(phys_input_size / segments)
        assert syn_rows_per_seg * segments == phys_input_size, (
            "Error: phys_input_size "
            "(%s) must be evenly divisible by number "
            "of segments (%s)" % (str(phys_input_size), str(segments))
        )

        gp_2d, gm_2d = g_lst

        out_sh = Size(input_.shape[0:-1]) + Size([gp_2d.shape[1]]) + Size([time_steps])

        input_ = input_.view(-1, input_.shape[-1])  # make 2d

        if use_extension and extension_ops is not None:
            # use C++ code for speedup if available
            output = extension_ops.thevenin_equiv(
                input_,
                gp_2d.T.contiguous(),
                gm_2d.T.contiguous(),
                r_s,
                t_max,
                time_steps,
            )
            vth_3d = output[0, :, :, :]
            rth_3d = output[1, :, :, :]

            vth_nd = vth_3d.view(*out_sh)  # reshape back to appropriate dimensions
            rth_nd = rth_3d.view(*out_sh)

            return vth_3d, rth_3d

        gp_4d = gp_2d[None, :, :, None]
        gm_4d = gm_2d[None, :, :, None]
        t_4d = linspace(0.0, t_max, time_steps)[None, None, None, :].to(input_.device)
        x_4d = input_[:, :, None, None]

        def sum_segs(g_values: Tensor) -> Tensor:
            if syn_rows_per_seg == 1:
                return g_values
            shape = g_values.shape
            return g_values.view(
                (
                    shape[0],
                    shape[1] // syn_rows_per_seg,
                    syn_rows_per_seg,
                    shape[2],
                    shape[3],
                )
            ).sum(dim=2)

        # pp
        pos_msk = x_4d > t_4d
        neg_msk = x_4d < -t_4d
        g_4d = gp_4d * pos_msk + gm_4d * neg_msk
        g_4d = sum_segs(g_4d)
        gth_4d_segs = g_4d
        vth_4d_segs = 0.6 * g_4d

        # mm
        g_4d = gm_4d * pos_msk + gp_4d * neg_msk
        g_4d = sum_segs(g_4d)
        gth_4d_segs += g_4d
        vth_4d_segs += 0.2 * g_4d

        # zz
        g_4d = (torch_abs(x_4d) <= t_4d) * (gp_4d + gm_4d)
        g_4d = sum_segs(g_4d)

        # regularized to avoid device-by-zero
        gth_4d_segs += g_4d + 1e-12  # atomic Thev equiv conductance [uS]
        vth_4d_segs += 0.4 * g_4d  # atomic Thevenin equivalent resistance [MOhm]
        vth_4d_segs /= gth_4d_segs
        g_4d = None

        # wire resistance depends on segmentation
        rw_segs = 1e-6 * r_s * syn_rows_per_seg

        vth_3d = vth_4d_segs[:, 0, :, :]
        rth_3d = 1.0 / gth_4d_segs[:, 0, :, :]
        for seg in range(1, segments, 1):
            r_1 = rth_3d + rw_segs
            r_2 = 1.0 / gth_4d_segs[:, seg, :, :]
            rth_3d = (r_1 * r_2) / (r_1 + r_2)  # parallel R
            vth_3d = (vth_3d / r_1 + vth_4d_segs[:, seg, :, :] / r_2) * rth_3d
        rth_3d += 0.5 * rw_segs

        vth_nd = vth_3d.view(*out_sh)  # reshape back to appropriate dimensions
        rth_nd = rth_3d.view(*out_sh)

        return vth_nd, rth_nd  # rth_nd in MOhm

    @classmethod
    def _matmul_irdrop(
        cls,
        weight: Tensor,
        input_: Tensor,
        trans: bool,
        io_pars: IOParametersIRDropT,
        ir_drop: float,
        t_max: float,
        time_steps: int,
        phys_input_size: int,
        g_converter: SinglePairConductanceConverter,
        info: Optional[str] = None,  # pylint: disable=unused-argument
    ) -> Tensor:
        """The inner FP GEMM.

        Args:
            weight: ``[in_size, out_size]`` MVM tile weights
            input_: ``[*, batch_size, in_size]`` MVM tile input activations
            trans: whether to transpose the weight
            io_pars: Parameter defining the mat-mul nonlinearities and time-dependent IR drop
            ir_drop: scale of the IR-drop wire resistance
            t_max: max time, beyond which activations all zero. Can cease computation
            phys_input_size: physical size of the tile in input dimension
            g_converter: specifies weight programming scheme which determines conductances
                for Thevenin equivlanet circuit.
            info: info string.

        Returns:
            Tensor with n-dimensional matmul result
        """
        ir_drop_rs = ir_drop * io_pars.ir_drop_rs
        res = cls._get_res(io_pars.inp_res)
        bit_res = io_pars.inp_bound / res

        if ir_drop == 0.0:
            return super(AnalogMVMIRDropT, cls)._matmul(weight, input_, trans)

        if weight.dim() != 2:
            raise ConfigError("Only 2-d weights are supported for time-sliced IR-drop")

        if not trans:
            new_weight = weight.T
        else:
            new_weight = weight

        syn_rows_per_seg = int(phys_input_size / io_pars.ir_drop_segments)
        assert syn_rows_per_seg * io_pars.ir_drop_segments == phys_input_size, (
            "Error: phys_input_size "
            "(%s) must be evenly divisible by number "
            "of segments (%s)" % (str(phys_input_size), str(io_pars.ir_drop_segments))
        )

        input_, new_weight = cls._pad_symmetric(
            input_, new_weight, phys_input_size=phys_input_size
        )
        if g_converter is None:
            g_converter = SinglePairConductanceConverter()  # type: ignore
        g_lst, params = g_converter.convert_to_conductances(new_weight)

        mvm = zeros((input_.shape[0], g_lst[0].shape[1])).to(input_.device)
        # (f1, (gp1, gm1)), (f2, (gp2, gm2), ... , (fn, (gpn, gmn)) # low to highest significance
        for f_factor, g_lst_pair in zip(params.get('f_lst', [1.]),
                                        zip(g_lst[::2], g_lst[1::2])):
            vth_nd, rth_nd = cls._thev_equiv(
                input_,
                [g[:, 0::2] for g in g_lst_pair],  # even cols
                time_steps=time_steps,
                t_max=t_max,
                segments=io_pars.ir_drop_segments,
                r_s=ir_drop_rs,
                phys_input_size=phys_input_size,
            )
            i_out_nd = (vth_nd - io_pars.ir_drop_v_read) / rth_nd  # uA
            mvm_even_col_down_adc = torch_sum(i_out_nd, dim=-1)  # * x batch_size x n_cols/2

            vth_nd, rth_nd = cls._thev_equiv(
                flip(input_, (-1,)),  # flip input
                [flip(g[:, 1::2], (0,)) for g in g_lst_pair],  # odd cols
                time_steps=time_steps,
                t_max=t_max,
                segments=io_pars.ir_drop_segments,
                r_s=ir_drop_rs,
                phys_input_size=phys_input_size,
            )
            i_out_nd = (vth_nd - io_pars.ir_drop_v_read) / rth_nd  # uA
            mvm_odd_col_up_adc = torch_sum(i_out_nd, dim=-1)  # * x batch_size x n_cols/2

            mvm += f_factor * cls._interleave_cols_nd(
                mvm_even_col_down_adc, mvm_odd_col_up_adc
            )  # symmetric ADCs

        mvm /= params['scale_ratio']  # conductance normalization
        mvm /= 0.2  # hardware normalization
        mvm /= bit_res / 2.0  # normalize
        return mvm

    @classmethod
    def _compute_analog_mv(  # type: ignore
        cls,
        weight: Tensor,
        input_: Tensor,
        trans: bool,
        scale: float,
        scaling: bool,
        is_test: bool,
        io_pars: IOParametersIRDropT,
        phys_input_size: int,
        g_converter: SinglePairConductanceConverter,
        **fwd_pars: Any,
    ) -> Tensor:
        """
        Prepare input, perform noisy MVM and finalize output. Takes care of noise/bound
        management and discretization.

        Args:
            weight: Weight tensor.
            input_: Input tensor in format [*, batch_size, in_size].
            trans: whether to transpose the weight
            scale: Scale for scaling the input.
            scaling: Whether to scale.
            io_pars: forward pass configuration.
            ir_drop: scale of the ir drop
            phys_input_size: Physical input size
            g_converter: conductance programming scheme for calculating Thevenin equivalent circuit
            fwd_pars: additional forward parameters

        Returns:
            Whether the bound management test passed and the result.

        Raises:
            NotImplementedError: If choices in the rpu config were made that are not supported.
            ConfigError: If unknown AnalogMVType

        """
        ir_drop = io_pars.ir_drop if is_test else 0.0

        prepared_input = cls._prepare_inputs(
            input_=input_,
            scale=scale,
            scaling=scaling,
            with_asymmetry=io_pars.inp_asymmetry != 0.0,
            io_pars=io_pars,
        )
        res = cls._get_res(io_pars.inp_res)
        bit_res = io_pars.inp_bound / res + 2

        if io_pars.mv_type == AnalogMVType.ONE_PASS:
            # - Perform the noisy MVM
            out_values = cls._matmul_irdrop(
                weight,
                prepared_input[0],
                trans,
                io_pars,
                ir_drop=ir_drop,
                t_max=1.0,
                time_steps=int(
                    (bit_res / 2) * io_pars.ir_drop_time_step_resolution_scale
                ),
                phys_input_size=phys_input_size,
                g_converter=g_converter,
            )
            bound_test_passed, finalized_outputs = cls._finalize_output(
                out_values=out_values, io_pars=io_pars, **fwd_pars
            )

        elif io_pars.mv_type == AnalogMVType.SPLIT_MODE:
            [prepared_input_lsb, prepared_input_msb] = prepared_input

            time_steps = int(2**io_pars.split_mode_bit_shift)
            t_max = (2 * res) * (2**io_pars.split_mode_bit_shift - 1)
            out_values_lsb = cls._matmul_irdrop(
                weight,
                prepared_input_lsb,
                trans,
                io_pars,
                ir_drop=ir_drop,
                t_max=t_max,
                time_steps=int(time_steps * io_pars.ir_drop_time_step_resolution_scale),
                phys_input_size=phys_input_size,
                g_converter=g_converter,
                info="LSB",
            )
            bound_test_passed_lsb, finalized_outputs_lsb = cls._finalize_output(
                out_values=out_values_lsb, io_pars=io_pars, **fwd_pars
            )

            time_steps = 2 ** int(
                log2(bit_res) - io_pars.split_mode_bit_shift - 1
            )  # minus 1 for sign bit
            t_max = (2 * res) * (time_steps - 1)
            out_values_msb = cls._matmul_irdrop(
                weight,
                prepared_input_msb,
                trans,
                io_pars,
                ir_drop=ir_drop,
                t_max=t_max,
                time_steps=int(time_steps * io_pars.ir_drop_time_step_resolution_scale),
                phys_input_size=phys_input_size,
                g_converter=g_converter,
                info="MSB",
            )
            bound_test_passed_msb, finalized_outputs_msb = cls._finalize_output(
                out_values=out_values_msb, io_pars=io_pars, **fwd_pars
            )

            finalized_outputs = (
                finalized_outputs_lsb
                + (2**io_pars.split_mode_bit_shift) * finalized_outputs_msb
            )
            bound_test_passed = bound_test_passed_lsb * bound_test_passed_msb

        elif io_pars.mv_type == AnalogMVType.BIT_WISE:
            finalized_outputs, bound_test_passed = 0.0, True
            for bit_pos, prepared_input_1b in enumerate(prepared_input):
                out_values_1b = cls._matmul_irdrop(
                    weight,
                    prepared_input_1b,
                    trans,
                    io_pars,
                    ir_drop=ir_drop,
                    t_max=2 * res,
                    time_steps=2,
                    phys_input_size=phys_input_size,
                    g_converter=g_converter,
                    info=str(bit_pos),
                )
                bound_test_passed_1b, finalized_outputs_1b = cls._finalize_output(
                    out_values=out_values_1b, io_pars=io_pars, **fwd_pars
                )
                finalized_outputs += (2**bit_pos) * finalized_outputs_1b
                bound_test_passed *= bound_test_passed_1b

        elif io_pars.mv_type in [
            AnalogMVType.POS_NEG_SEPARATE,
            AnalogMVType.POS_NEG_SEPARATE_DIGITAL_SUM,
        ]:
            raise NotImplementedError
        else:
            raise ConfigError(f"Unknown AnalogMVType {io_pars.mv_type}")
        return bound_test_passed, finalized_outputs

    @classmethod
    def check_support(cls, io_pars: IOParameters) -> None:
        """Check whether the IO settings are supported.

        Throws an assertion error when there is an incompatibility

        Args:
            io_pars: the IOParametersDropT to be checked

        Raises:
            ConfigError: in case a feature is not supported
        """

        # pylint: disable=too-many-branches

        if io_pars.mv_type not in [
            AnalogMVType.ONE_PASS,
            AnalogMVType.SPLIT_MODE,
            AnalogMVType.BIT_WISE,
        ]:
            raise ConfigError(
                "Only AnalogMVType.ONE_PASS, "
                "AnalogMVType.SPLIT_MODE, and "
                "AnalogMVType.BIT_WISE supported as forward.mv_type"
            )

        if io_pars.bound_management == BoundManagementType.SHIFT:
            raise ConfigError("Shift bound management not supported in torch tile")

        if io_pars.noise_management in [
            NoiseManagementType.AVERAGE_ABS_MAX,
            NoiseManagementType.ABS_MAX_NP_SUM,
        ]:
            raise ConfigError("Special noise mangement types not supported.")

        if io_pars.w_noise > 0.0 or io_pars.w_noise_type != WeightNoiseType.NONE:
            raise ConfigError("forward.w_noise not supported in torch tile")

        if io_pars.out_nonlinearity > 0.0:
            raise ConfigError("S-shaped non-linearity not supported in torch tile")

        if io_pars.slope_calibration > 0.0:
            raise ConfigError("Slope calibration not supported in torch tile")

        if (
            io_pars.v_offset_std > 0.0
            or io_pars.v_offset_w_min > 0.0
            or io_pars.r_series > 0.0
        ):
            raise ConfigError("Voltage offset or R-series not supported in torch tile")

        if io_pars.w_read_asymmetry_dtod > 0.0:
            raise ConfigError(
                "Device polarity read dependence is not supported in torch tile"
            )
