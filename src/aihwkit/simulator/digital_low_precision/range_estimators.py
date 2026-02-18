# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.

# pylint: skip-file
# type: ignore

import copy
from collections import namedtuple
from enum import Enum

import numpy as np
import torch
from scipy.optimize import minimize_scalar
from torch import nn
from torch.nn import functional as F

from aihwkit.simulator.digital_low_precision.utils import to_numpy


class RangeEstimatorBase(nn.Module):
    def __init__(
        self, per_channel=False, quantizer=None, axis=None, n_groups=None, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.register_buffer("current_xmin", None)
        self.register_buffer("current_xmax", None)
        self.per_channel = per_channel
        self.quantizer = quantizer
        self.axis = axis
        self.n_groups = n_groups

        self.per_group_range_estimation = False
        self.ranges = None

    def forward(self, x):
        """
        Accepts an input tensor, updates the current estimates of x_min and x_max
        and retruns them.
        Parameters
        ----------
        x: Input tensor

        Returns
        -------
        self.current_xmin: tensor
        self.current_xmax: tensor
        """
        raise NotImplementedError()

    def reset(self):
        """
        Reset the range estimator.
        """
        self.current_xmin = None
        self.current_xmax = None

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        xmin_key = prefix + "current_xmin"
        xmax_key = prefix + "current_xmax"
        if xmin_key in state_dict:
            if self.current_xmin is None or state_dict[xmin_key].shape != self.current_xmin.shape:
                self.current_xmin = state_dict[xmin_key]
        if xmax_key in state_dict:
            if self.current_xmax is None or state_dict[xmax_key].shape != self.current_xmax.shape:
                self.current_xmax = state_dict[xmax_key]

        super(RangeEstimatorBase, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def __repr__(self):
        # We overwrite this from nn.Module as we do not want to have submodules such as
        # self.quantizer in the reproduce. Otherwise it behaves as expected for an nn.Module.
        lines = self.extra_repr().split("\n")
        extra_str = lines[0] if len(lines) == 1 else "\n  " + "\n  ".join(lines) + "\n"

        return self._get_name() + "(" + extra_str + ")"


class CurrentMinMaxEstimator(RangeEstimatorBase):
    def __init__(self, percentile=None, *args, **kwargs):
        self.percentile = percentile
        super().__init__(*args, **kwargs)

    def forward(self, x):
        if self.per_group_range_estimation:
            assert self.axis != 0
            x = x.transpose(0, self.axis).contiguous()
            x = x.view(x.size(0), -1)

            ranges = x.max(-1)[0].detach() - x.min(-1)[0].detach()

            if self.ranges is None:
                self.ranges = ranges
            else:
                momentum = 0.1
                self.ranges = momentum * ranges + (1 - momentum) * ranges
            return

        if self.axis is not None:
            if self.axis != 0:
                x = x.transpose(0, self.axis).contiguous()
            x = x.view(x.size(0), -1)

            if self.n_groups is not None:
                ng = self.n_groups
                assert ng > 0 and x.size(0) % ng == 0
                gs = x.size(0) // ng

                # permute
                if self.ranges is not None:
                    i = torch.argsort(self.ranges)
                    Iarr = torch.eye(len(i), device=self.ranges.device)
                    P = Iarr[i]
                    x = P.mm(x)

                x = x.view(ng, -1)
                m = x.min(-1)[0].detach()
                M = x.max(-1)[0].detach()

                m = m.repeat_interleave(gs)
                M = M.repeat_interleave(gs)

                # permute back
                if self.ranges is not None:
                    m = P.T.mv(m)
                    M = P.T.mv(M)

                self.current_xmin = m
                self.current_xmax = M

            else:
                self.current_xmin = x.min(-1)[0].detach()
                self.current_xmax = x.max(-1)[0].detach()

        elif self.per_channel:
            # Along 1st dim
            x_flattened = x.view(x.shape[0], -1)
            if self.percentile:
                data_np = to_numpy(x_flattened)
                x_min, x_max = np.percentile(
                    data_np, (self.percentile, 100 - self.percentile), axis=-1
                )
                self.current_xmin = torch.Tensor(x_min)
                self.current_xmax = torch.Tensor(x_max)
            else:
                self.current_xmin = x_flattened.min(-1)[0].detach()
                self.current_xmax = x_flattened.max(-1)[0].detach()

        else:
            if self.percentile:
                device = x.device
                data_np = to_numpy(x)
                x_min, x_max = np.percentile(data_np, (self.percentile, 100))
                x_min = np.atleast_1d(x_min)
                x_max = np.atleast_1d(x_max)
                self.current_xmin = torch.Tensor(x_min).to(device).detach()
                self.current_xmax = torch.Tensor(x_max).to(device).detach()
            else:
                self.current_xmin = torch.min(x).detach()
                self.current_xmax = torch.max(x).detach()

        return self.current_xmin, self.current_xmax


class AllMinMaxEstimator(RangeEstimatorBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        if self.per_channel:
            # Along 1st dim
            x_flattened = x.view(x.shape[0], -1)
            x_min = x_flattened.min(-1)[0].detach()
            x_max = x_flattened.max(-1)[0].detach()
        else:
            x_min = torch.min(x).detach()
            x_max = torch.max(x).detach()

        if self.current_xmin is None:
            self.current_xmin = x_min
            self.current_xmax = x_max
        else:
            self.current_xmin = torch.min(self.current_xmin, x_min)
            self.current_xmax = torch.max(self.current_xmax, x_max)

        return self.current_xmin, self.current_xmax


class RunningMinMaxEstimator(RangeEstimatorBase):
    def __init__(self, momentum=0.9, *args, **kwargs):
        self.momentum = momentum
        super().__init__(*args, **kwargs)

    def forward(self, x):
        if self.axis is not None:
            if self.axis != 0:
                x = x.transpose(0, self.axis).contiguous()
            x = x.view(x.size(0), -1)

            if self.n_groups is not None:
                ng = self.n_groups
                assert ng > 0 and x.size(0) % ng == 0
                gs = x.size(0) // ng

                x = x.view(ng, -1)
                m = x.min(-1)[0].detach()
                M = x.max(-1)[0].detach()

                x_min = m.repeat_interleave(gs)
                x_max = M.repeat_interleave(gs)

            else:
                x_min = x.min(-1)[0].detach()
                x_max = x.max(-1)[0].detach()

        elif self.per_channel:
            # Along 1st dim
            x_flattened = x.view(x.shape[0], -1)
            x_min = x_flattened.min(-1)[0].detach()
            x_max = x_flattened.max(-1)[0].detach()

        else:
            x_min = torch.min(x).detach()
            x_max = torch.max(x).detach()

        if self.current_xmin is None:
            self.current_xmin = x_min
            self.current_xmax = x_max
        else:
            self.current_xmin = (1 - self.momentum) * x_min + self.momentum * self.current_xmin
            self.current_xmax = (1 - self.momentum) * x_max + self.momentum * self.current_xmax

        return self.current_xmin, self.current_xmax


class OptMethod(Enum):
    grid = 1
    golden_section = 2

    @classmethod
    def list(cls):
        return [m.name for m in cls]


class MSE_Estimator(RangeEstimatorBase):
    def __init__(
        self, num_candidates=100, opt_method=OptMethod.grid, range_margin=0.5, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        assert opt_method in OptMethod

        self.opt_method = opt_method
        self.num_candidates = num_candidates
        self.loss_array = None
        self.max_pos_thr = None
        self.max_neg_thr = None
        self.max_search_range = None
        self.one_sided_dist = None
        self.range_margin = range_margin
        if self.quantizer is None:
            raise NotImplementedError(
                "A Quantizer must be given as an argument to the MSE Range" "Estimator"
            )
        self.max_int_skew = (2**self.quantizer.n_bits) // 4  # for asymmetric quantization

    def loss_fx(self, data, neg_thr, pos_thr, per_channel_loss=False):
        y = self.quantize(data, x_min=neg_thr, x_max=pos_thr)
        temp_sum = torch.sum(((data - y) ** 2).view(len(data), -1), dim=1)
        # if we want to return the MSE loss of each channel separately, speeds up the per-channel
        # grid search
        if per_channel_loss:
            return to_numpy(temp_sum)
        else:
            return to_numpy(torch.sum(temp_sum))

    @property
    def step_size(self):
        if self.one_sided_dist is None:
            raise NoDataPassedError()

        return self.max_search_range / self.num_candidates

    @property
    def optimization_method(self):
        if self.one_sided_dist is None:
            raise NoDataPassedError()

        if self.opt_method == OptMethod.grid:
            # Grid search method
            if self.one_sided_dist or self.quantizer.symmetric:
                # 1-D grid search
                return self._perform_1D_search
            else:
                # 2-D grid_search
                return self._perform_2D_search
        elif self.opt_method == OptMethod.golden_section:
            # Golden section method
            if self.one_sided_dist or self.quantizer.symmetric:
                return self._golden_section_symmetric
            else:
                return self._golden_section_asymmetric
        else:
            raise NotImplementedError("Optimization Method not Implemented")

    def quantize(self, x_float, x_min=None, x_max=None):
        temp_q = copy.deepcopy(self.quantizer)
        # In the current implementation no optimization procedure requires temp quantizer for
        # loss_fx to be per-channel
        temp_q.per_channel = False
        if x_min or x_max:
            temp_q.set_quant_range(x_min, x_max)
        return temp_q(x_float)

    def golden_sym_loss(self, range, data):
        """
        Loss function passed to the golden section optimizer from scipy in case of symmetric
        quantization
        """
        neg_thr = 0 if self.one_sided_dist else -range
        pos_thr = range
        return self.loss_fx(data, neg_thr, pos_thr)

    def golden_asym_shift_loss(self, shift, range, data):
        """
        Inner Loss function (shift) passed to the golden section optimizer from scipy
        in case of asymmetric quantization
        """
        pos_thr = range + shift
        neg_thr = -range + shift
        return self.loss_fx(data, neg_thr, pos_thr)

    def golden_asym_range_loss(self, range, data):
        """
        Outer Loss function (range) passed to the golden section optimizer from scipy in case of
         asymmetric quantization
        """
        temp_delta = 2 * range / (2**self.quantizer.n_bits - 1)
        max_shift = temp_delta * self.max_int_skew
        result = minimize_scalar(
            self.golden_asym_shift_loss,
            args=(range, data),
            bounds=(-max_shift, max_shift),
            method="Bounded",
        )
        return result.fun

    def _define_search_range(self, data):
        self.channel_groups = len(data) if self.per_channel else 1
        self.current_xmax = torch.zeros(self.channel_groups, device=data.device)
        self.current_xmin = torch.zeros(self.channel_groups, device=data.device)

        if self.one_sided_dist or self.quantizer.symmetric:
            # 1D search space
            self.loss_array = np.zeros(
                (self.channel_groups, self.num_candidates + 1)
            )  # 1D search space
            self.loss_array[:, 0] = np.inf  # exclude interval_start=interval_finish
            # Defining the search range for clipping thresholds
            self.max_pos_thr = max(abs(float(data.min())), float(data.max())) + self.range_margin
            self.max_neg_thr = -self.max_pos_thr
            self.max_search_range = self.max_pos_thr
        else:
            # 2D search space (3rd and 4th index correspond to asymmetry where fourth
            # index represents whether the skew is positive (0) or negative (1))
            self.loss_array = np.zeros(
                [self.channel_groups, self.num_candidates + 1, self.max_int_skew, 2]
            )  # 2D search space
            self.loss_array[:, 0, :, :] = np.inf  # exclude interval_start=interval_finish
            # Define the search range for clipping thresholds in asymmetric case
            self.max_pos_thr = float(data.max()) + self.range_margin
            self.max_neg_thr = float(data.min()) - self.range_margin
            self.max_search_range = max(abs(self.max_pos_thr), abs(self.max_neg_thr))

    def _perform_1D_search(self, data):
        """
        Grid search through all candidate quantizers in 1D to find the best
        The loss is accumulated over all batches without any momentum
        :param data: input tensor
        """
        for cand_index in range(1, self.num_candidates + 1):
            neg_thr = 0 if self.one_sided_dist else -self.step_size * cand_index
            pos_thr = self.step_size * cand_index

            self.loss_array[:, cand_index] += self.loss_fx(
                data, neg_thr, pos_thr, per_channel_loss=self.per_channel
            )
            # find the best clipping thresholds
        min_cand = self.loss_array.argmin(axis=1)
        xmin = (
            np.zeros(self.channel_groups) if self.one_sided_dist else -self.step_size * min_cand
        ).astype(np.single)
        xmax = (self.step_size * min_cand).astype(np.single)
        self.current_xmax = torch.tensor(xmax).to(device=data.device)
        self.current_xmin = torch.tensor(xmin).to(device=data.device)

    def _perform_2D_search(self, data):
        """
        Grid search through all candidate quantizers in 1D to find the best
        The loss is accumulated over all batches without any momentum
        Parameters
        ----------
        data:   PyTorch Tensor
        Returns
        -------

        """
        for cand_index in range(1, self.num_candidates + 1):
            # defining the symmetric quantization range
            temp_start = -self.step_size * cand_index
            temp_finish = self.step_size * cand_index
            temp_delta = float(temp_finish - temp_start) / (2**self.quantizer.n_bits - 1)
            for shift in range(self.max_int_skew):
                for reverse in range(2):
                    # introducing asymmetry in the quantization range
                    skew = ((-1) ** reverse) * shift * temp_delta
                    neg_thr = max(temp_start + skew, self.max_neg_thr)
                    pos_thr = min(temp_finish + skew, self.max_pos_thr)

                    self.loss_array[:, cand_index, shift, reverse] += self.loss_fx(
                        data, neg_thr, pos_thr, per_channel_loss=self.per_channel
                    )

        for channel_index in range(self.channel_groups):
            min_cand, min_shift, min_reverse = np.unravel_index(
                np.argmin(self.loss_array[channel_index], axis=None),
                self.loss_array[channel_index].shape,
            )
            min_interval_start = -self.step_size * min_cand
            min_interval_finish = self.step_size * min_cand
            min_delta = float(min_interval_finish - min_interval_start) / (
                2**self.quantizer.n_bits - 1
            )
            min_skew = ((-1) ** min_reverse) * min_shift * min_delta
            xmin = max(min_interval_start + min_skew, self.max_neg_thr)
            xmax = min(min_interval_finish + min_skew, self.max_pos_thr)

            self.current_xmin[channel_index] = torch.tensor(xmin).to(device=data.device)
            self.current_xmax[channel_index] = torch.tensor(xmax).to(device=data.device)

    def _golden_section_symmetric(self, data):
        for channel_index in range(self.channel_groups):
            if channel_index == 0 and not self.per_channel:
                data_segment = data
            else:
                data_segment = data[channel_index]

            self.result = minimize_scalar(
                self.golden_sym_loss,
                args=data_segment,
                bounds=(0.01 * self.max_search_range, self.max_search_range),
                method="Bounded",
            )
            self.current_xmax[channel_index] = torch.tensor(self.result.x).to(device=data.device)
            self.current_xmin[channel_index] = (
                torch.tensor(0.0).to(device=data.device)
                if self.one_sided_dist
                else -self.current_xmax[channel_index]
            )

    def _golden_section_asymmetric(self, data):
        for channel_index in range(self.channel_groups):
            if channel_index == 0 and not self.per_channel:
                data_segment = data
            else:
                data_segment = data[channel_index]

            self.result = minimize_scalar(
                self.golden_asym_range_loss,
                args=data_segment,
                bounds=(0.01 * self.max_search_range, self.max_search_range),
                method="Bounded",
            )
            self.final_range = self.result.x
            temp_delta = 2 * self.final_range / (2**self.quantizer.n_bits - 1)
            max_shift = temp_delta * self.max_int_skew
            self.subresult = minimize_scalar(
                self.golden_asym_shift_loss,
                args=(self.final_range, data_segment),
                bounds=(-max_shift, max_shift),
                method="Bounded",
            )
            self.final_shift = self.subresult.x
            self.current_xmax[channel_index] = torch.tensor(self.final_range + self.final_shift).to(
                device=data.device
            )
            self.current_xmin[channel_index] = torch.tensor(
                -self.final_range + self.final_shift
            ).to(device=data.device)

    def forward(self, data):
        if self.loss_array is None:
            # Initialize search range on first batch, and accumulate losses with subsequent calls

            # Decide whether input distribution is one-sided
            if self.one_sided_dist is None:
                self.one_sided_dist = bool((data.min() >= 0).item())

            # Define search
            self._define_search_range(data)

        # Perform Search/Optimization for Quantization Ranges
        self.optimization_method(data)

        return self.current_xmin, self.current_xmax

    def reset(self):
        super().reset()
        self.loss_array = None


class CrossEntropyEstimator(MSE_Estimator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # per_channel_loss argument is here only to be consistent in definition with other loss fxs
    def loss_fx(self, data, neg_thr, pos_thr, per_channel_loss=False):
        quantized_data = self.quantize(data, neg_thr, pos_thr)
        log_quantized_probs = F.log_softmax(quantized_data, dim=1)
        unquantized_probs = F.softmax(data, dim=1)
        return to_numpy(torch.sum(-unquantized_probs * log_quantized_probs))


class NoDataPassedError(Exception):
    """Raised data has been passed inot the Range Estimator."""

    def __init__(self):
        super().__init__("Data must be pass through the range estimator to be initialized")


RangeEstimatorMap = namedtuple("RangeEstimatorMap", ["value", "cls"])


class RangeEstimators(Enum):
    current_minmax = RangeEstimatorMap(0, CurrentMinMaxEstimator)
    allminmax = RangeEstimatorMap(1, AllMinMaxEstimator)
    running_minmax = RangeEstimatorMap(2, RunningMinMaxEstimator)
    MSE = RangeEstimatorMap(3, MSE_Estimator)
    cross_entropy = RangeEstimatorMap(4, CrossEntropyEstimator)

    @property
    def cls(self):
        return self.value.cls

    @classmethod
    def list(cls):
        return [m.name for m in cls]
