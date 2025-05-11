# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.

# pylint: skip-file
# type: ignore

from copy import deepcopy

import torch
from torch import nn

from aihwkit.simulator.digital_low_precision.base_quantized_classes import QuantizedModule
from aihwkit.simulator.digital_low_precision.quantization_manager import QuantizationManager
from aihwkit.simulator.digital_low_precision.range_estimators import RangeEstimators
from aihwkit.simulator.digital_low_precision.utils import to_numpy


activations_list = [nn.ReLU, nn.ReLU6, nn.Hardtanh, nn.Sigmoid, nn.Tanh, nn.PReLU, nn.GELU]


class QuantizationHijacker(QuantizedModule):
    """Mixin class that 'hijacks' the forward pass in a module to perform quantization and
    dequantization on the weights and output distributions.

    Usage:
    To make a quantized nn.Linear layer:
    ```
    >>> class QuantLinear(QuantizationHijacker, nn.Linear):
    ...     pass
    ```

    It is vital that QSchemeForwardHijacker is the first parent class, and that the second parent
    class derives from nn.Module, otherwise it will not be reached by a super(., .) call.

    NB: this implementation (for now) assumes that there will always be some training involved,
    e.g. to estimate the activation ranges.
    """

    def __init__(self, *args, activation: nn.Module = None, **kwargs):
        super().__init__(*args, **kwargs)
        if activation:
            assert isinstance(activation, tuple(activations_list))
        self.activation_function = deepcopy(activation) if activation else None

        weight_qparams = dict(n_bits=self.n_bits, scale_domain=self.scale_domain)
        act_qparams = dict(n_bits=self.n_bits_act, scale_domain=self.scale_domain)

        self.activation_quantizer = QuantizationManager(
            qmethod=self.act_method,
            init=self.act_range_method,
            per_channel=self.per_channel_acts,
            qparams=act_qparams,
            init_params=self.act_range_options,
        )

        if self.weight_range_method == RangeEstimators.current_minmax:
            weight_init_params = dict(percentile=self.percentile)
        else:
            weight_init_params = self.weight_range_options
        self.weight_quantizer = QuantizationManager(
            qmethod=self.method,
            init=self.weight_range_method,
            per_channel=self.per_channel_weights,
            qparams=weight_qparams,
            init_params=weight_init_params,
        )
        self.activation_save_target = None
        self.activation_save_name = None

    def forward(self, x, offsets=None):
        weight, bias = self.get_params()
        res = self.run_forward(x, weight, bias, offsets=offsets)
        res = self.quantize_activations(res)
        return res

    def get_params(self):
        if not self.training and self.cached_params:
            return self.cached_params

        weight, bias = self.get_weight_bias()

        if self._quant_w:
            weight = self.weight_quantizer(weight)

        if self._caching and not self.training and self.cached_params is None:
            self.cached_params = (
                torch.Tensor(to_numpy(weight)).to(weight.device),
                torch.Tensor(to_numpy(bias)).to(bias.device) if bias is not None else None,
            )
        return weight, bias

    def get_weight_bias(self):
        bias = None
        if hasattr(self, "bias"):
            bias = self.bias
        return self.weight, bias

    def run_forward(self, x, weight, bias, offsets=None):
        # Performs the actual (e.g., linear) operation of the layer
        raise NotImplementedError()

    def quantize_activations(self, activations):
        """Quantize a single activation tensor or all activations from a layer. I'm assuming that
        we should quantize all outputs for a layer with the same quantization scheme.
        """
        if self.activation_function is not None:
            activations = self.activation_function(activations)

        if self.activation_save_target is not None:
            self.activation_save_target[self.activation_save_name] = activations.data.cpu().numpy()

        if self._quant_a:
            activations = self.activation_quantizer(activations)

            if self.activation_save_target is not None:
                self.activation_save_target[self.activation_save_name + "_Q"] = (
                    activations.data.cpu().numpy()
                )

        return activations
