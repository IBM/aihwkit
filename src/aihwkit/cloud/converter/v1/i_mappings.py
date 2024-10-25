# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

# pylint: disable=no-name-in-module, import-error, too-few-public-methods

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
from torchvision.datasets import FashionMNIST, SVHN  # type: ignore[import]

from aihwkit.simulator.configs import InferenceRPUConfig
from aihwkit.simulator.presets.web import (
    WebComposerInferenceRPUConfig,
    OldWebComposerInferenceRPUConfig,
)

from aihwkit.cloud.converter.definitions.i_onnx_common_pb2 import (  # type: ignore[attr-defined]
    AttributeProto,
)
from aihwkit.cloud.converter.exceptions import ConversionError
from aihwkit.nn import AnalogConv2d, AnalogConv2dMapped, AnalogLinear, AnalogLinearMapped
from aihwkit.optim import AnalogSGD
from aihwkit.cloud.converter.v1.rpu_config_info import RPUconfigInfo


Type = namedtuple("Type", ["attribute_type", "field", "fn"])

# pylint: disable=no-member
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
# pylint: enable=no-member


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

                if isinstance(final_value, int):
                    final_value = [final_value, final_value]

                getattr(argument, proto_type.field).extend(final_value)
            else:
                proto_type = TYPES[type_]
                setattr(argument, proto_type.field, proto_type.fn(value))
            argument.type = proto_type.attribute_type
            instance.arguments.append(argument)

        # TODO:  add to_proto for state_dict

        return instance

    def from_proto(self, source: Any, cls: type, default: Any = None) -> object:
        """Convert a proto object into a destination object."""
        kwargs = {}

        for argument in source.arguments:
            if argument.name not in self.args:
                continue

            type_ = self.args[argument.name]

            if isinstance(type_, list):
                proto_type = TYPES_LISTS[type_[0]]
            else:
                proto_type = TYPES[type_]

            default_ = None if default is None else default.get(argument.name, None)
            new_argument = self.get_argument_from_proto(argument, proto_type.field, default_)
            if isinstance(type_, list):
                new_argument[argument.name] = list(new_argument[argument.name])
            kwargs.update(new_argument)

        # TODO:  add from_proto for state_dict

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
        """Get the value of a field.

        Raises ConversionError
        """
        if field == "bias":
            return getattr(source, "bias", None) is not None
        if field == "rpu_config":
            # preset_cls = type(source.analog_tile.rpu_config)
            analog_tile = next(source.analog_tiles())
            preset_cls = type(analog_tile.rpu_config)
            if preset_cls not in Mappings.presets:
                raise ConversionError(
                    "Invalid rpu_config in layer: " f"{preset_cls} not among the presets"
                )
            return Mappings.presets[preset_cls]
        return super().get_field_value_to_proto(source, field, default)

    def get_argument_from_proto(self, source: Any, field: str, default: Any = None) -> Dict:
        """Get the value of an argument.

        Raises ConversionError
        """
        if source.name == "rpu_config":
            if not isinstance(default, RPUconfigInfo):
                raise ConversionError("Expect an new RPUconfigInfo as default for layer creation.")

            return {"rpu_config": default.create_inference_rpu_config(self.id_)}

        return super().get_argument_from_proto(source, field, default)


class Mappings:
    """Mappings between Python entities and AIHW format."""

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
            },
        ),
        AnalogConv2dMapped: LayerFunction(
            "AnalogConv2dMapped",
            {
                "in_channels": int,
                "out_channels": int,
                "kernel_size": [int],
                "stride": [int],
                "padding": [int],
                "dilation": [int],
                "bias": bool,
                "rpu_config": str,
            },
        ),
        AnalogLinear: LayerFunction(
            "AnalogLinear",
            {"in_features": int, "out_features": int, "bias": bool, "rpu_config": str},
        ),
        AnalogLinearMapped: LayerFunction(
            "AnalogLinearMapped",
            {"in_features": int, "out_features": int, "bias": bool, "rpu_config": str},
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
        InferenceRPUConfig: "InferenceRPUConfig",
        OldWebComposerInferenceRPUConfig: "OldWebComposerInferenceRPUConfig",
        WebComposerInferenceRPUConfig: "WebComposerInferenceRPUConfig",
    }


def build_inverse_mapping(mapping: Dict) -> Dict:
    """Create the inverse mapping between Python entities and AIHW Composer formats."""
    return {
        value if not isinstance(value, Function) else value.id_: key
        for key, value in mapping.items()
    }


class InverseMappings:
    """Mappings between AIHW Composer format and Python entities."""

    datasets = build_inverse_mapping(Mappings.datasets)
    layers = build_inverse_mapping(Mappings.layers)
    activation_functions = build_inverse_mapping(Mappings.activation_functions)
    loss_functions = build_inverse_mapping(Mappings.loss_functions)
    optimizers = build_inverse_mapping(Mappings.optimizers)
    presets = build_inverse_mapping(Mappings.presets)
