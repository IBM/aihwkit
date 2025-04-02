# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

# mypy: disable-error-code=attr-defined

"""Quantization configuration parameters"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from torch import nn

from aihwkit.simulator.digital_low_precision.config_utils import (
    CrossEntropyEstimatorParams,
    CurrentMinMaxEstimatorParams,
    MSEEstimatorParams,
    RunningMinMaxEstimatorParams,
)
from aihwkit.simulator.digital_low_precision.range_estimators import RangeEstimators

from .helpers import _PrintableMixin


@dataclass
class BaseQuantConfig(_PrintableMixin):
    """Base class for quantization parameter configuration, that contains necessary
    fields to configure the type of quantization for either activations of weights.
    The user should use the `ActivationQuantConfig` and `WeightQuantConfig` as interfaces
    to these parameters, as they include default initializations and an easier API.
    """

    _range_estimator: RangeEstimators
    """
    The range estimator to be used for quantization range estimation. The supported estimators
    are defined in the Enum `RangeEstimators`. This field should always be accessed through the
    property `range_estimator`.

    NOTE: This field is used in conjuction with the `_range_estimator_params` field to initialize
    each range estimator with each own arguments properly. As each range estimator has different
    arguments, these are defined in the dataclasses `CurrentMinMaxEstimatorParams`,
    `RunningMinMaxEstimatorParams`, `MSEEstimatorParams`, and `CrossEntropyEstimatorParams`.
    When the user initializes the `_range_estimator` with the appropriate enum value, the
    `_range_estimator_params` field will initialize the corresponding proper params class, to
    avoid misconfiguration (see `range_estimator_params.setter`)
    """

    n_bits: int = 0
    """The number of bits for the quantization of the operations.
    If <= 0 is selected, no quantization will be applied. By default 0 (no quantization).
    """

    symmetric: bool = True
    """If True, the quantization will be symmetrical (just scale).
    If False, asymmetric quantization will be used. This option is valid
    only if `n_bits` > 0."""

    _range_estimator_params: Any = field(init=False)
    """
    Defines the parameters of the selected `range_estimator`. Its type Depends on the
    value of the `range_estimator` and thus its initialization is conditional and must
    always be in sync with the type of `range_estimator`. It should always be used
    through the property `range_estimator_params`.
    """

    _range_estimator_to_params: Dict[RangeEstimators, Any] = field(
        default_factory=lambda: {
            RangeEstimators.allminmax: None,
            RangeEstimators.current_minmax: CurrentMinMaxEstimatorParams,
            RangeEstimators.running_minmax: RunningMinMaxEstimatorParams,
            RangeEstimators.MSE: MSEEstimatorParams,
            RangeEstimators.cross_entropy: CrossEntropyEstimatorParams,
        },
        init=False,
    )

    def __post_init__(self) -> None:
        """Initializes the `_range_estimator_params` with the default values according to
        the type of the `range_estimator object.
        """
        self._range_estimator_params = (
            self._range_estimator_to_params[self._range_estimator]()
            if self._range_estimator != RangeEstimators.allminmax
            else None
        )

    @property
    def range_estimator(self) -> RangeEstimators:
        """range_estimator property"""
        return self._range_estimator

    @range_estimator.setter
    def range_estimator(self, new_range_estimator: RangeEstimators) -> None:
        """Change `range_estimator` but also reset `range_estimator_params` object to the default
        configuration for the corresponding type of estimator.
        """
        # Update mode and reset parameters
        self._range_estimator = new_range_estimator
        self._range_estimator_params = (
            self._range_estimator_to_params[new_range_estimator]()
            if self._range_estimator != RangeEstimators.allminmax
            else None
        )

    @property
    def range_estimator_params(self) -> Any:
        """range_estimator_params property"""
        return self._range_estimator_params

    @range_estimator_params.setter
    def range_estimator_params(self, new_params: Any) -> None:
        """Configure the `range_estimator_params` but make sure it's in accordance to the type
        of range_estimator selected, to avoid misconfiguration.
        """
        expected_type = self._range_estimator_to_params[self._range_estimator]
        if expected_type is None and new_params is not None:
            raise TypeError(
                "Range estimator 'allminmax' does not accept a range_estimator params object"
            )
        if not isinstance(new_params, expected_type):
            raise TypeError(
                f"Expected parameters of type {expected_type.__name__}, "
                + f"got {type(new_params).__name__}"
            )
        self._range_estimator_params = new_params


@dataclass
class ActivationQuantConfig(BaseQuantConfig):
    """The quantization config that should be used for activation quantization.
    This class wraps the `BaseQuantConfig` class, by initializing the range_estimator
    object in the `running_minmax` mode.
    """

    def __init__(
        self,
        n_bits: int = 0,
        symmetric: bool = True,
        range_estimator: RangeEstimators = RangeEstimators.running_minmax,
        range_estimator_params: Optional[Any] = None,
    ):
        """Initializes the object by calling the `BaseQuantConfig` init function and overwriting
        the `range_estimator_params` attribute if given by the user. For more details of the fields,
        see also `BaseQuantConfig`.

        Parameters
        ----------
        n_bits : int, optional
            The number of bits for quantization. If <= 0, no quantization is applied, by default 0
        symmetric : bool, optional
            True for symmetric quantization, False for asymmetric, by default True
        range_estimator : RangeEstimators, optional
            The type of range estimator to use for PTQ or QAT with range estimation,
            by default RangeEstimators.running_minmax
        range_estimator_params : Optional[Any], optional
            The parameters to configure the `range_estimator` object. Its type must be in
            accordance with the selected range_estimator (see `BaseQuantConfig` for details).
            If None is given, the default params object, initialized in the `BaseQuantConfig`
            post_init function will be retained, by default None
        """
        super().__init__(n_bits=n_bits, symmetric=symmetric, _range_estimator=range_estimator)
        if range_estimator_params is not None:
            self.range_estimator_params = range_estimator_params


@dataclass
class WeightQuantConfig(BaseQuantConfig):
    """The quantization config that should be used for weight quantization. This class wraps the
    `BaseQuantConfig` class, by initializing the range_estimator object in the `current_minmax`
    mode, and adding the `per_channel` option.
    """

    def __init__(
        self,
        n_bits: int = 0,
        symmetric: bool = True,
        per_channel: bool = False,
        range_estimator: RangeEstimators = RangeEstimators.current_minmax,
        range_estimator_params: Optional[Any] = None,
    ):
        """Initializes the object by calling the `BaseQuantConfig` init function and overwriting
        the `range_estimator_params` attribute if given by the user. For more details of the fields,
        see also `BaseQuantConfig`.

        Parameters
        ----------
        n_bits : int, optional
            The number of bits for quantization. If <= 0, no quantization is applied, by default 0
        symmetric : bool, optional
            True for symmetric quantization, False for asymmetric, by default True
        per_channel : bool, optional
            True for per-channel quantization, False for per-tensor, by default False
        range_estimator : RangeEstimators, optional
            The type of range estimator to use for PTQ or QAT with range estimation,
            by default RangeEstimators.current_minmax
        range_estimator_params : Optional[Any], optional
            The parameters to configure the `range_estimator` object. Its type must be in
            accordance with the selected range_estimator (see `BaseQuantConfig` for details).
            If None is given, the default params object, initialized in the `BaseQuantConfig`
            post_init function will be retained, by default None
        """
        super().__init__(n_bits=n_bits, symmetric=symmetric, _range_estimator=range_estimator)
        if range_estimator_params is not None:
            self.range_estimator_params = range_estimator_params
        self.per_channel = per_channel


@dataclass
class QuantizationConfig(_PrintableMixin):
    """Holds the activation and weight quantization configuration objects for a layer"""

    activation_quant: ActivationQuantConfig = field(default_factory=ActivationQuantConfig)
    """ Configuration for the activation quantization of a layer.

    NOTE: The convention is that this activation quantizer of a layer corresponds to the OUTPUT
    activations and not its inputs activations (these are considered already quantized by the
    previous layer's quantizer). See  `QuantizationMap.input_activation_qconfig_map` and
    `QuantizedInputModule` for ways to define a layer that should also have its inputs quantized.
    """

    weight_quant: WeightQuantConfig = field(default_factory=WeightQuantConfig)
    """ Configuration for the weight quantization of a layer """


@dataclass
class QuantizedModuleConfig(_PrintableMixin):
    """Utility dataclass that pairs a torch Module, aimed to be a quantized implementation of
    a module, with a `QuantizationConfig`. It's used to define the `module_qconfig_map` and
    `instance_qconfig_map` fields of the `QuantizationMap` dataclass"""

    quantized_module: nn.Module
    module_qconfig: QuantizationConfig


@dataclass
class QuantizationMap(_PrintableMixin):
    """This is the datastructure that is consumed by the `convert_to_quantized` function and the
    quantized modules. It defines how to replace a module to a quantized counterpart. It offers
    the capability to define specific quantization options per-layer-instance for maximum
    flexibility but also per-module-type to reduce the definition code. See below for the
    available options and how to use each option.
    """

    default_qconfig: QuantizationConfig = field(default_factory=QuantizationConfig)
    """This is a utility field and it is NOT used during the `convert_to_quantized` call. It exists
    to simplify development code in the case where most layers use the same quantization
    configuration, so that the user defines it here and then shares it when he defines the
    `instance_qconfig_map` and `module_qconfig_map` fields (see `append_default_conversions`
    function for such a use).
    """

    module_qconfig_map: Dict[nn.Module, QuantizedModuleConfig] = field(default_factory=lambda: {})
    """This field defines a map of how to convert various module types to quantized quanterparts. It
    is a dictionary where the keys are a type of a Module (e.g., nn.Linear) and the value is an
    instance of `QuantizedModuleConfig`, which includes a quantized Module to replace the original
    one (e.g., QuantLinear) and a `QuantizationConfig` instance with the parameters for the new
    quantized layer.

    NOTE: This is the lowest priority for the conversions. If an instance of a module is defined
    in the `instance_qconfig_map`, the conversion defined here will be ignored, and the conversion
    defined in the `instance_qconfig_map` will happen. The same applies if an instance of a module
    is included in the `excluded_modules` list.

    Examples
    --------
    >>> # Replace every instance of nn.Linear with QuantLinear and use the default qconfig
    >>> quantization_map.module_qconfig_map[nn.Linear] = QuantizedModuleConfig(
    ...     quantized_module=QuantLinear, module_qconfig=quantization_map.default_qconfig
    ... )
    """

    instance_qconfig_map: Dict[str, QuantizedModuleConfig] = field(default_factory=lambda: {})
    """This field defines a map of how to convert specific instances of modules to quantized
    quanterparts. It is a dictionary where the keys are the string identifier of a layer, as
    it appears in the state dict of the model (e.g., \'block1.linear1\') and the value is an
    instance of `QuantizedModuleConfig`, which includes a quantized Module to replace the
    original one (e.g., QuantLinear) and a `QuantizationConfig` instance with the parameters
    for the new quantized layer.


    NOTE: This takes priority over a possible conversion defined in the `module_qconfig_map`
    for the type of the layer identified by the string identifier (e.g. for QuantLinear layers).
    If a layer is both included in this field and in the `excluded_modules` list (by user error),
    the latter has priority.

    Examples
    --------
    >>> # Replace a specific instance of a Linear layer differently than the others
    >>> quantization_map.instance_qconfig_map[\'block1.linear1\'] = QuantizedModuleConfig(
    ...     quantized_module=QuantLinear, module_qconfig=custom_layer1_qconfig
    ... )
    """

    input_activation_qconfig_map: Dict[str, ActivationQuantConfig] = field(
        default_factory=lambda: {}
    )
    """This field defines a map of the modules that should be wrapped in the `QuantizedInputModule`
    and the parameters of how to quantize the input activations. Since the convention used in this
    database is that each quantized layer quantizes its output activations, there are cases that a
    module could receive unquantized data as inputs (for example if it's the first layer, or if it
    follows a functional call that cannot be quantized with the scheme defined here). For these
    reasons, an arbitrary Module could be wrapped in the `QuantizedInputModule` class to allow for
    full customizability. The modules here are defined based on the string identifier, as appears in
    the state dict of the model, while the value of the dict is an `ActivationQuantConfig` object.

    NOTE: This conversion follows any quantization conversion defined in the `instance_qconfig_map`
    and `module_qconfig_map`. That means that if a quantization for a module is defined in one of
    the other maps and here, the quantized module will be properly wrapped in the
    `QuantizedInputModule`.

    Examples
    --------
    >>> # Quantize the inputs of the first layer of a network
    >>> quantization_map.input_activation_qconfig_map[\'firstblock.firstlayer\'] = (
    ...     ActivationQuantConfig(n_bits=8, symmetric=False)
    ... )
    """

    excluded_modules: List[str] = field(default_factory=lambda: [])
    """This field is a list of modules, identified by their state dict string, to be excluded
    from ANY conversion that is defined for their type or for their instance. This takes precedence
    over all other conversion steps.
    """
