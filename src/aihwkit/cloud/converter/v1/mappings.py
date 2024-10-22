# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

# pylint: disable=no-name-in-module, import-error

"""Mappings for version 1 of the AIHW Composer format."""

from collections import namedtuple
from typing import Any, Dict

from torch.nn import (
    BCELoss,
    BatchNorm2d,
    Conv2d,
    ConvTranspose2d,
    CrossEntropyLoss,
    Flatten,
    LeakyReLU,
    Linear,
    LogSigmoid,
    LogSoftmax,
    MSELoss,
    MaxPool2d,
    NLLLoss,
    ReLU,
    Sigmoid,
    Softmax,
    Tanh,
)
from torchvision.datasets import FashionMNIST, SVHN

from aihwkit.cloud.converter.definitions.onnx_common_pb2 import (  # type: ignore[attr-defined]
    AttributeProto,
)
from aihwkit.cloud.converter.exceptions import ConversionError
from aihwkit.nn import AnalogConv2d, AnalogLinear
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.presets import (
    CapacitorPreset,
    EcRamPreset,
    IdealizedPreset,
    ReRamESPreset,
    ReRamSBPreset,
    PCMPreset,
    MixedPrecisionCapacitorPreset,
    MixedPrecisionEcRamPreset,
    MixedPrecisionIdealizedPreset,
    MixedPrecisionPCMPreset,
    MixedPrecisionReRamESPreset,
    MixedPrecisionReRamSBPreset,
    TikiTakaCapacitorPreset,
    TikiTakaEcRamPreset,
    TikiTakaIdealizedPreset,
    TikiTakaReRamESPreset,
    TikiTakaReRamSBPreset,
)

Type = namedtuple("Type", ["attribute_type", "field", "fn"])

TYPES = {
    int: Type(AttributeProto.AttributeType.INT, "i", lambda x: x),  # type: ignore
    bool: Type(AttributeProto.AttributeType.BOOL, "b", lambda x: x),  # type: ignore
    str: Type(
        AttributeProto.AttributeType.STRING, "s", lambda x: x.encode("utf-8")  # type: ignore
    ),
    float: Type(AttributeProto.AttributeType.FLOAT, "f", lambda x: x),  # type: ignore
}

TYPES_LISTS = {
    int: Type(AttributeProto.AttributeType.INTS, "ints", lambda x: x),  # type: ignore
    bool: Type(AttributeProto.AttributeType.BOOLS, "bools", lambda x: x),  # type: ignore
    str: Type(
        AttributeProto.AttributeType.STRINGS,
        "strings",  # type: ignore
        lambda x: [y.encode("utf-8") for y in x],
    ),
    float: Type(AttributeProto.AttributeType.FLOATS, "floats", lambda x: x),  # type: ignore
}


class Function:
    """Mapping for a function-like entity."""

    def __init__(self, id_: str, args: Dict):
        self.id_ = id_
        self.args = args

    def to_proto(self, source: object, proto_cls: type) -> object:
        """Convert a source object into a destination object."""
        instance = proto_cls(id=self.id_)

        for name, type_ in self.args.items():
            value = self.get_field_value_to_proto(source, name, None)

            argument = AttributeProto(name=name)
            if isinstance(type_, list):
                proto_type = TYPES_LISTS[type_[0]]
                final_value = proto_type.fn(value)

                # Fix arguments where ``Union[T, Sequence[T]]`` is allowed.
                # TODO: generalize, as this check assumes int and 2-size.
                if isinstance(final_value, int):
                    final_value = [final_value, final_value]

                getattr(argument, proto_type.field).extend(final_value)
            else:
                proto_type = TYPES[type_]
                setattr(argument, proto_type.field, proto_type.fn(value))
            argument.type = proto_type.attribute_type
            instance.arguments.append(argument)

        return instance

    def from_proto(self, source: Any, cls: type) -> object:
        """Convert a proto object into a destination object."""
        kwargs = {}

        for argument in source.arguments:
            type_ = self.args[argument.name]

            if isinstance(type_, list):
                proto_type = TYPES_LISTS[type_[0]]
            else:
                proto_type = TYPES[type_]

            new_argument = self.get_argument_from_proto(argument, proto_type.field, None)
            if isinstance(type_, list):
                new_argument[argument.name] = list(new_argument[argument.name])
            kwargs.update(new_argument)

        # handle the weights_scaling_omega legacy
        weight_scaling_omega = kwargs.pop("weight_scaling_omega", None)
        if weight_scaling_omega is not None:
            if "rpu_config" not in kwargs or not hasattr(kwargs["rpu_config"], "mapping"):
                raise ConversionError("Expect Mappable RPUConfig")
            kwargs["rpu_config"].mapping.weights_scaling_omega = weight_scaling_omega

        return cls(**kwargs)

    def get_field_value_to_proto(self, source: Any, field: str, default: Any = None) -> Any:
        """Get the value of a field."""
        return getattr(source, field, default)

    def get_argument_from_proto(self, source: Any, field: str, default: Any = None) -> Dict:
        """Get the value of an argument."""
        return {source.name: getattr(source, field, default)}


class LayerFunction(Function):
    """Mapping for a function-like entity (Layer)."""

    def get_field_value_to_proto(self, source: Any, field: str, default: Any = None) -> Any:
        """Get the value of a field."""
        if field == "bias":
            return getattr(source, "bias", None) is not None

        if field == "weight_scaling_omega":
            return next(source.analog_tiles()).rpu_config.mapping.weight_scaling_omega

        if field == "rpu_config":
            preset_cls = type(next(source.analog_tiles()).rpu_config)
            try:
                return Mappings.presets[preset_cls]
            except KeyError as ex:
                raise ConversionError(
                    "Invalid rpu_config in layer: {} not among the presets".format(preset_cls)
                ) from ex

        return super().get_field_value_to_proto(source, field, default)

    def get_argument_from_proto(self, source: Any, field: str, default: Any = None) -> Dict:
        """Get the value of an argument."""
        if source.name == "rpu_config":
            preset_str = getattr(source, field, default).decode("utf-8")
            try:
                preset = InverseMappings.presets[preset_str]
            except KeyError as ex:
                raise ConversionError(
                    "Invalid rpu_config in layer: {} not among the presets".format(preset_str)
                ) from ex
            return {"rpu_config": preset()}

        return super().get_argument_from_proto(source, field, default)


class Mappings:
    """Mappings between Python entities and AIHW format."""

    # pylint: disable=too-few-public-methods

    datasets = {FashionMNIST: "fashion_mnist", SVHN: "svhn"}

    layers = {
        AnalogConv2d: LayerFunction(
            "AnalogConv2d",
            {
                "in_channels": int,
                "out_channels": int,
                "kernel_size": [int],
                "stride": [int],
                "padding": [int],
                "dilation": [int],
                "bias": bool,
                "rpu_config": str,
                "weight_scaling_omega": float,
            },
        ),
        AnalogLinear: LayerFunction(
            "AnalogLinear",
            {
                "in_features": int,
                "out_features": int,
                "bias": bool,
                "rpu_config": str,
                "weight_scaling_omega": float,
            },
        ),
        BatchNorm2d: LayerFunction("BatchNorm2d", {"num_features": int}),
        Conv2d: LayerFunction(
            "Conv2d",
            {
                "in_channels": int,
                "out_channels": int,
                "kernel_size": [int],
                "stride": [int],
                "padding": [int],
                "dilation": [int],
                "bias": bool,
            },
        ),
        ConvTranspose2d: LayerFunction(
            "ConvTranspose2d",
            {
                "in_channels": int,
                "out_channels": int,
                "kernel_size": [int],
                "stride": [int],
                "padding": [int],
                "output_padding": [int],
                "dilation": [int],
                "bias": bool,
            },
        ),
        Flatten: LayerFunction("Flatten", {}),
        Linear: LayerFunction("Linear", {"in_features": int, "out_features": int, "bias": bool}),
        MaxPool2d: LayerFunction(
            "MaxPool2d",
            {"kernel_size": int, "stride": int, "padding": int, "dilation": int, "ceil_mode": bool},
        ),
    }

    activation_functions = {
        LeakyReLU: Function("LeakyReLU", {"negative_slope": float}),
        LogSigmoid: Function("LogSigmoid", {}),
        LogSoftmax: Function("LogSoftmax", {"dim": int}),
        ReLU: Function("ReLU", {}),
        Sigmoid: Function("Sigmoid", {}),
        Softmax: Function("Softmax", {"dim": int}),
        Tanh: Function("Tanh", {}),
    }

    loss_functions = {
        BCELoss: Function("BCELoss", {}),
        CrossEntropyLoss: Function("CrossEntropyLoss", {}),
        MSELoss: Function("MSELoss", {}),
        NLLLoss: Function("NLLLoss", {}),
    }

    optimizers = {AnalogSGD: Function("AnalogSGD", {"lr": float})}

    presets = {
        ReRamESPreset: "ReRamESPreset",
        ReRamSBPreset: "ReRamSBPreset",
        CapacitorPreset: "CapacitorPreset",
        EcRamPreset: "EcRamPreset",
        IdealizedPreset: "IdealizedPreset",
        PCMPreset: "PCMPreset",
        MixedPrecisionReRamESPreset: "MixedPrecisionReRamESPreset",
        MixedPrecisionReRamSBPreset: "MixedPrecisionReRamSBPreset",
        MixedPrecisionCapacitorPreset: "MixedPrecisionCapacitorPreset",
        MixedPrecisionEcRamPreset: "MixedPrecisionEcRamPreset",
        MixedPrecisionIdealizedPreset: "MixedPrecisionIdealizedPreset",
        MixedPrecisionPCMPreset: "MixedPrecisionPCMPreset",
        TikiTakaReRamESPreset: "TikiTakaReRamESPreset",
        TikiTakaReRamSBPreset: "TikiTakaReRamSBPreset",
        TikiTakaCapacitorPreset: "TikiTakaCapacitorPreset",
        TikiTakaEcRamPreset: "TikiTakaEcRamPreset",
        TikiTakaIdealizedPreset: "TikiTakaIdealizedPreset",
    }


def build_inverse_mapping(mapping: Dict) -> Dict:
    """Create the inverse mapping between Python entities and AIHW Composer formats."""
    return {
        value if not isinstance(value, Function) else value.id_: key
        for key, value in mapping.items()
    }


class InverseMappings:
    """Mappings between AIHW Composer format and Python entities."""

    # pylint: disable=too-few-public-methods

    datasets = build_inverse_mapping(Mappings.datasets)
    layers = build_inverse_mapping(Mappings.layers)
    activation_functions = build_inverse_mapping(Mappings.activation_functions)
    loss_functions = build_inverse_mapping(Mappings.loss_functions)
    optimizers = build_inverse_mapping(Mappings.optimizers)
    presets = build_inverse_mapping(Mappings.presets)
