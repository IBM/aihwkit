# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.

# pylint: skip-file
# type: ignore

from torch import nn

from aihwkit.simulator.digital_low_precision.quantization_manager import QuantizationManager
from aihwkit.simulator.digital_low_precision.quantizers import QMethods
from aihwkit.simulator.digital_low_precision.range_estimators import RangeEstimators


def _set_layer_learn_ranges(layer):
    if isinstance(layer, QuantizationManager):
        if layer.quantizer.is_initialized:
            layer.learn_ranges()


def _set_layer_fix_ranges(layer):
    if isinstance(layer, QuantizationManager):
        if layer.quantizer.is_initialized:
            layer.fix_ranges()


def _set_layer_estimate_ranges(layer):
    if isinstance(layer, QuantizationManager):
        # if layer.quantizer.is_initialized:
        layer.estimate_ranges()


def _set_layer_estimate_ranges_train(layer):
    if isinstance(layer, QuantizationManager):
        # if layer.quantizer.is_initialized:
        layer.estimate_ranges_train()


class QuantizedModule(nn.Module):
    """
    Parent class for a quantized module. It adds the basic functionality of switching the module
    between quantized and full precision mode. It also defines the cached parameters and handles
    the reset of the cache properly.
    """

    def __init__(
        self,
        *args,
        method=QMethods.asymmetric_uniform,
        act_method=None,
        n_bits=8,
        n_bits_act=None,
        per_channel_weights=False,
        per_channel_acts=False,
        percentile=None,
        weight_range_method=RangeEstimators.current_minmax,
        weight_range_options=None,
        act_range_method=RangeEstimators.running_minmax,
        act_range_options=None,
        scale_domain="linear",
        **kwargs,
    ):
        kwargs.pop("quant_dict", None)
        kwargs.pop("module_map", None)
        kwargs.pop("quantization_map", None)
        super().__init__(*args, **kwargs)

        self.method = method
        self.act_method = act_method or method
        self.n_bits = n_bits
        self.n_bits_act = n_bits_act if n_bits_act is not None else n_bits
        self.per_channel_weights = per_channel_weights
        self.per_channel_acts = per_channel_acts
        self.percentile = percentile
        self.weight_range_method = weight_range_method
        self.weight_range_options = weight_range_options if weight_range_options else {}
        self.act_range_method = act_range_method
        self.act_range_options = act_range_options if act_range_options else {}
        self.scale_domain = scale_domain

        self.cached_params = None
        self._caching = True

        self.quant_params = None
        self._quant_w = False
        self._quant_a = False

    @property
    def caching(self):
        return self._caching

    @caching.setter
    def caching(self, value: bool):
        self._caching = value
        if not value:
            self.cached_params = None

    def quantized_weights(self):
        self.cached_params = None
        self._quant_w = True

    def full_precision_weights(self):
        self.cached_params = None
        self._quant_w = False

    def quantized_acts(self):
        self._quant_a = True

    def full_precision_acts(self):
        self._quant_a = False

    def quantized(self):
        self.quantized_weights()
        self.quantized_acts()

    def full_precision(self):
        self.full_precision_weights()
        self.full_precision_acts()

    def learn_ranges(self):
        self.apply(_set_layer_learn_ranges)

    def fix_ranges(self):
        self.apply(_set_layer_fix_ranges)

    def estimate_ranges(self):
        self.apply(_set_layer_estimate_ranges)

    def estimate_ranges_train(self):
        self.apply(_set_layer_estimate_ranges_train)

    def train(self, mode=True):
        super().train(mode)
        if mode:
            self.cached_params = None
        return self

    def _apply(self, *args, **kwargs):
        self.cached_params = None
        return super(QuantizedModule, self)._apply(*args, **kwargs)

    def extra_repr(self):
        quant_state = "weight_quant={}, act_quant={}".format(self._quant_w, self._quant_a)
        parent_repr = super().extra_repr()
        return "{},\n{}".format(parent_repr, quant_state) if parent_repr else quant_state


class QuantizedActivation(QuantizedModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        act_qparams = dict(n_bits=self.n_bits_act, scale_domain=self.scale_domain)
        self.activation_quantizer = QuantizationManager(
            qmethod=self.act_method,
            qparams=act_qparams,
            init=self.act_range_method,
            init_params=self.act_range_options,
        )

    def quantize_activations(self, x):
        if self._quant_a:
            return self.activation_quantizer(x)
        else:
            return x

    def forward(self, x):
        return self.quantize_activations(x)


class FP32Acts(nn.Module):
    def forward(self, x):
        return x

    def reset_ranges(self):
        pass
