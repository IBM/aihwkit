# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

# pylint: disable=no-name-in-module,import-error

"""Converter for `BasicTraining` Experiment."""

from typing import Any, Dict, List

from torch.nn import Module
from torch.nn.modules.loss import _Loss

from aihwkit.cloud.converter.exceptions import ConversionError
from aihwkit.cloud.converter.v1.mappings import InverseMappings, Mappings
from aihwkit.experiments.experiments.training import BasicTraining

from aihwkit.cloud.converter.definitions.input_file_pb2 import (  # type: ignore[attr-defined]
    TrainingInput,
    Dataset,
    Training,
)
from aihwkit.cloud.converter.definitions.common_pb2 import (  # type: ignore[attr-defined]
    LayerOrActivationFunction,
    LossFunctionProto,
    Network,
    LayerProto,
    ActivationFunctionProto,
    OptimizerProto,
    Version,
)
from aihwkit.cloud.converter.definitions.onnx_common_pb2 import (  # type: ignore[attr-defined]
    AttributeProto,
)
from aihwkit.nn import AnalogSequential


class BasicTrainingConverter:
    """Converter for `BasicTraining` Experiment."""

    def to_proto(self, experiment: BasicTraining) -> Any:
        """Convert an `Experiment` to its protobuf representation."""
        version = self._version_to_proto()
        dataset = self._dataset_to_proto(experiment.dataset, experiment.batch_size)
        network = self._model_to_proto(experiment.model)
        training = self._training_to_proto(
            experiment.epochs, experiment.learning_rate, experiment.loss_function
        )

        training_input = TrainingInput(
            version=version, dataset=dataset, network=network, training=training
        )

        return training_input

    def from_proto(self, training_proto: Any) -> Any:
        """Convert a protobuf representation to an `Experiment`."""
        dataset = InverseMappings.datasets[training_proto.dataset.dataset_id]
        model = self._model_from_proto(training_proto.network)
        batch_size = training_proto.dataset.batch_size
        loss_function = InverseMappings.loss_functions[training_proto.training.loss_function.id]
        epochs = training_proto.training.epochs
        learning_rate = training_proto.training.optimizer.arguments[0].f

        return BasicTraining(
            dataset=dataset,
            model=model,
            batch_size=batch_size,
            loss_function=loss_function,
            epochs=epochs,
            learning_rate=learning_rate,
        )

    # Methods for converting to proto.

    @staticmethod
    def _version_to_proto() -> Any:
        return Version(schema=1, opset=1)

    @staticmethod
    def _dataset_to_proto(dataset: type, batch_size: int) -> Any:
        if dataset not in Mappings.datasets:
            raise ConversionError("Unsupported dataset: {}".format(dataset))

        return Dataset(dataset_id=Mappings.datasets[dataset], batch_size=batch_size)

    @staticmethod
    def _model_to_proto(model: Module) -> Any:
        if not isinstance(model, AnalogSequential):
            raise ConversionError("Unsupported model: only AnalogSequential is supported")

        children_types = {type(layer) for layer in model.children()}
        valid_types = set(Mappings.layers.keys()) | set(Mappings.activation_functions.keys())
        if children_types - valid_types:
            raise ConversionError("Unsupported layers: {}".format(children_types - valid_types))

        network = Network()
        for child in model.children():
            child_type = type(child)
            if child_type in Mappings.layers:
                item = LayerOrActivationFunction(
                    layer=Mappings.layers[child_type].to_proto(child, LayerProto)
                )
            else:
                item = LayerOrActivationFunction(
                    activation_function=Mappings.activation_functions[child_type].to_proto(
                        child, ActivationFunctionProto
                    )
                )
            network.layers.extend([item])

        return network

    @staticmethod
    def _training_to_proto(epochs: int, learning_rate: float, loss_function: _Loss) -> Any:
        if loss_function not in Mappings.loss_functions:
            raise ConversionError("Unsupported loss function: {}".format(loss_function))

        # Build optimizer manually.
        optimizer = OptimizerProto(id="AnalogSGD")
        optimizer.arguments.append(
            AttributeProto(
                name="lr", type=AttributeProto.AttributeType.FLOAT, f=learning_rate  # type: ignore
            )
        )

        training = Training(
            epochs=epochs,
            optimizer=optimizer,
            loss_function=Mappings.loss_functions[loss_function].to_proto(
                loss_function(), LossFunctionProto
            ),
        )

        return training

    # Methods for converting from proto.

    @staticmethod
    def _model_from_proto(model_proto: Any) -> Module:
        layers = []
        for layer_proto in model_proto.layers:
            if layer_proto.WhichOneof("item") == "layer":
                layer_cls = InverseMappings.layers[layer_proto.layer.id]
                layer = Mappings.layers[layer_cls].from_proto(layer_proto.layer, layer_cls)
            else:
                layer_cls = InverseMappings.activation_functions[layer_proto.activation_function.id]
                layer = Mappings.activation_functions[layer_cls].from_proto(
                    layer_proto.activation_function, layer_cls
                )

            layers.append(layer)

        return AnalogSequential(*layers)


class BasicTrainingResultConverter:
    """Converter for `BasicTraining` results."""

    # pylint: disable=too-few-public-methods

    def from_proto(self, results: Any) -> Any:
        """Convert a result to its json representation."""
        return {"version": {"schema": 1, "opset": 1}, "epochs": self._epochs_from_proto(results)}

    # Methods for converting from proto.
    @staticmethod
    def _epochs_from_proto(epochs_proto: Any) -> List[Dict]:
        epochs = []
        for epoch in epochs_proto.epochs:
            epoch_dict = {"epoch": epoch.epoch, "metrics": {}}
            for metric in epoch.metrics:
                epoch_dict["metrics"][metric.name] = metric.f

            epochs.append(epoch_dict)

        return epochs
