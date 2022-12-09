# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022 IBM. All Rights Reserved.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Converters for `BasicInferencing` Experiment."""

from typing import Any, Dict, List
# from collections import OrderedDict

from torch.nn import Module, NLLLoss

from aihwkit.simulator.configs import InferenceRPUConfig
from aihwkit.cloud.converter.exceptions import ConversionError
from aihwkit.cloud.converter.v1.i_mappings import (
    InverseMappings, Mappings)

# pylint: disable=no-name-in-module,import-error
from aihwkit.experiments.experiments.inferencing import (  # type: ignore[import]
    BasicInferencing
)
from aihwkit.cloud.converter.definitions.i_input_file_pb2 import (   # type: ignore[attr-defined]
    InferenceInput, Dataset, Inferencing, NoiseModelProto, PCMProto, GenericProto, AnalogProto
)

from aihwkit.cloud.converter.definitions.i_common_pb2 import (   # type: ignore[attr-defined]
    LayerOrActivationFunction, Network, LayerProto,
    ActivationFunctionProto, Version
)
from aihwkit.nn import AnalogSequential

from aihwkit.cloud.converter.v1.analog_info import AnalogInfo
from aihwkit.cloud.converter.v1.noise_model_info import NoiseModelInfo
from aihwkit.cloud.converter.v1.rpu_config_info import RPUconfigInfo


class BasicInferencingConverter:
    """Converter for `BasicInferencing` Experiment."""

    def to_proto(self, experiment: BasicInferencing,
                 analog_info: Dict,
                 noise_model_info: Dict) -> Any:
        """Convert an `Experiment` to its protobuf representation."""

        version = self._version_to_proto()
        dataset = self._dataset_to_proto(experiment.dataset,
                                         experiment.batch_size)
        network = self._model_to_proto(experiment.model,
                                       experiment.weight_template_id)
        inferencing = self._inferencing_to_proto(experiment.inference_repeats,
                                                 experiment.inference_time,
                                                 analog_info,
                                                 noise_model_info,
                                                 )
        return InferenceInput(
            version=version,
            dataset=dataset,
            network=network,
            inferencing=inferencing
        )

    def from_proto(self, protobuf: Any) -> BasicInferencing:
        """Convert a protobuf representation to an `Experiment`."""

        dataset = InverseMappings.datasets[protobuf.dataset.dataset_id]
        layers = protobuf.network.layers
        # build RPUconfig_info to be used when it is instantiated dynamically
        alog_info = AnalogInfo(protobuf.inferencing.analog_info)
        nm_info = NoiseModelInfo(protobuf.inferencing.noise_model_info)
        rc_info = RPUconfigInfo(nm_info, alog_info, layers)

        model = self._model_from_proto(protobuf.network, rc_info)

        batch_size = protobuf.dataset.batch_size
        inference_info = self._inference_info_from_proto(protobuf)
        loss_function = NLLLoss
        weight_template_id = inference_info['weight_template_id']
        inference_repeats = inference_info['inference_repeats']
        inference_time = inference_info['inference_time']

        return BasicInferencing(
            dataset=dataset,
            model=model,
            batch_size=batch_size,
            loss_function=loss_function,
            weight_template_id=weight_template_id,
            inference_repeats=inference_repeats,
            inference_time=inference_time
        )

    # Methods for converting to proto.

    @staticmethod
    def _version_to_proto() -> Any:
        return Version(
            schema=1,
            opset=1
        )

    @staticmethod
    def _dataset_to_proto(dataset: type, batch_size: int) -> Any:
        if dataset not in Mappings.datasets:
            raise ConversionError(f'Unsupported dataset: {dataset}')

        return Dataset(
            dataset_id=Mappings.datasets[dataset],
            batch_size=batch_size
        )

    @staticmethod
    def _model_to_proto(model: Module, weight_template_id: str) -> Any:

        if not isinstance(model, AnalogSequential):
            raise ConversionError('Unsupported model: only AnalogSequential is supported')

        children_types = {type(layer) for layer in model.children()}
        valid_types = set(Mappings.layers.keys()) | set(Mappings.activation_functions.keys())
        if children_types - valid_types:
            raise ConversionError('Unsupported layers: '
                                  f'{children_types - valid_types}')

        # Create a new input_file pb Network object with weight_template_id
        network = Network(weight_template_id=weight_template_id)

        for child in model.children():
            child_type = type(child)
            if child_type in Mappings.layers:
                item = LayerOrActivationFunction(
                    layer=Mappings.layers[child_type].to_proto(child, LayerProto)
                )
            else:
                item = LayerOrActivationFunction(
                    activation_function=Mappings.activation_functions[child_type].to_proto(
                        child, ActivationFunctionProto)
                )
            # pylint: disable=no-member),undefined-variable
            network.layers.extend([item])

        return network

    @staticmethod
    def _noise_model_to_proto(
            noise_model_info: Dict) -> NoiseModelProto:  # type: ignore[valid-type]
        """Creates a protobuf NoiseModelProto object from input dictionaries"""

        model = None
        device_id = noise_model_info['device_id']
        if device_id == 'pcm':
            model = NoiseModelProto(
                pcm=PCMProto(
                    device_id=device_id,
                    programming_noise_scale=noise_model_info['programming_noise_scale'],
                    read_noise_scale=noise_model_info['read_noise_scale'],
                    drift_scale=noise_model_info['drift_scale'],
                    drift_compensation=noise_model_info['drift_compensation'],
                    poly_first_order_coef=noise_model_info['poly_first_order_coef'],
                    poly_second_order_coef=noise_model_info['poly_second_order_coef'],
                    poly_constant_coef=noise_model_info['poly_constant_coef']
                    )
                )
        else:
            model = NoiseModelProto(
                generic=GenericProto(
                    device_id=device_id,
                    programming_noise_scale=noise_model_info['programming_noise_scale'],
                    read_noise_scale=noise_model_info['read_noise_scale'],
                    drift_scale=noise_model_info['drift_scale'],
                    drift_compensation=noise_model_info['drift_compensation'],
                    poly_first_order_coef=noise_model_info['poly_first_order_coef'],
                    poly_second_order_coef=noise_model_info['poly_second_order_coef'],
                    poly_constant_coef=noise_model_info['poly_constant_coef'],
                    drift_mean=noise_model_info['drift_mean'],
                    drift_std=noise_model_info['drift_std'],
                    )
            )
        return model

    @staticmethod
    def rpu_config_info_from_info(analog_info: Dict,
                                  noise_model_info: Dict
                                  ) -> RPUconfigInfo:
        """ Creates RPUconfigInfo """

        # TODO: Remove the "Info" objects: NoiseModelInfo, AnalogInfo. Seems unecessary
        nm_info = NoiseModelInfo(BasicInferencingConverter._noise_model_to_proto(
            noise_model_info))  # type: ignore[name-defined]
        a_info = AnalogInfo(AnalogProto(**analog_info))
        return RPUconfigInfo(nm_info, a_info, None)

    @staticmethod
    def rpu_config_from_info(analog_info: Dict,
                             noise_model_info: Dict,
                             func_id: str = 'id-not-provided'
                             ) -> InferenceRPUConfig:
        """ Creates RPUConfig """
        nm_info = NoiseModelInfo(BasicInferencingConverter._noise_model_to_proto(
            noise_model_info))  # type: ignore[name-defined]
        a_info = AnalogInfo(AnalogProto(**analog_info))
        return RPUconfigInfo(nm_info,
                             a_info, None).create_inference_rpu_config(func_id)

    @staticmethod
    def _inferencing_to_proto(
            inference_repeats: int,
            inference_time: float,
            analog_info: Dict,
            noise_model_info: Dict) -> Inferencing:  # type: ignore[valid-type]
        """Creates protobuf Inferencing object"""

        # Not sure why mypy and pylint cannot find following method, python3 can.
        # pylint: disable=undefined-variable
        nm_info = BasicInferencingConverter._noise_model_to_proto(
            noise_model_info)  # type: ignore[name-defined]
        a_info = AnalogProto(**analog_info)

        return Inferencing(inference_repeats=inference_repeats,
                           inference_time=inference_time,
                           analog_info=a_info,
                           noise_model_info=nm_info
                           )

    # Methods for converting from proto.

    @staticmethod
    def _inference_info_from_proto(
            inference_pb: InferenceInput) -> Dict:  # type: ignore[valid-type]
        """Converts inference_info from protobuf to a dictionary"""

        inferencing = inference_pb.inferencing  # type: ignore[attr-defined]
        network = inference_pb.network  # type: ignore[attr-defined]
        return {
                    'inference_repeats': inferencing.inference_repeats,
                    'inference_time': inferencing.inference_time,
                    'weight_template_id': network.weight_template_id
                }

    @staticmethod
    def _model_from_proto(network: Network,  # type: ignore[valid-type]
                          rc_info: RPUconfigInfo) -> Module:
        layers = []
        for layer_proto in network.layers:  # type: ignore[attr-defined]
            if layer_proto.WhichOneof('item') == 'layer':
                layer_cls = InverseMappings.layers[layer_proto.layer.id]
                layer = Mappings.layers[layer_cls].from_proto(
                    layer_proto.layer, layer_cls, {'rpu_config': rc_info})
            else:
                layer_cls = InverseMappings.activation_functions[layer_proto.activation_function.id]
                layer = Mappings.activation_functions[layer_cls].from_proto(
                    layer_proto.activation_function, layer_cls)

            layers.append(layer)

        return AnalogSequential(*layers)

    @staticmethod
    def _analog_info_from_proto(
            analog_info: AnalogProto) -> Dict:  # type: ignore[valid-type]
        """Converts from protobuf analog_info to a dictionary"""

        return {
                    'output_noise_strength': (
                        analog_info.output_noise_strength  # type: ignore[attr-defined]
                    ),
                    'adc': analog_info.adc,  # type: ignore[attr-defined]
                    'dac': analog_info.dac  # type: ignore[attr-defined]
                }

    @staticmethod
    def _noise_model_from_proto(
            noise_model_info: NoiseModelProto) -> Dict:  # type: ignore[valid-type]
        """Converts from protobuf noise_model to a dictionary"""

        extra = {}
        typ = noise_model_info.WhichOneof('item')  # type: ignore[attr-defined]

        if typ == 'pcm':
            # pcm does not have 2 extra fields
            info = noise_model_info.pcm  # type: ignore[attr-defined]

        elif typ == 'generic':
            info = noise_model_info.generic  # type: ignore[attr-defined]
            # There are 2 extra fields in generic
            extra = {
                'drift_mean': info.drift_mean,
                'drift_std': info.drift_std
            }
        else:
            raise TypeError

        base = {
            'device_id': typ,
            'programming_noise_scale': info.programming_noise_scale,
            'read_noise_scale': info.read_noise_scale,
            'drift_scale': info.drift_scale,
            'drift_compensation': info.drift_compensation,
            'poly_first_order_coef': info.poly_first_order_coef,
            'poly_second_order_coef': info.poly_second_order_coef,
            'poly_constant_coef': info.poly_constant_coef,
        }

        # add the extra fields if any in return value
        return {**base, **extra}


class BasicInferencingResultConverter:
    """Converter for `BasicInverencing` results."""

    # pylint: disable=too-few-public-methods

    def from_proto(self, results: Any) -> Any:
        """Convert a result to its json representation."""

        return {
            'version': {
                'schema': 1,
                'opset': 1
            },
            'epochs': self._epochs_from_proto(results)
        }

    @staticmethod
    def to_json() -> Dict:
        """Converts dictionary and converts it to a output_file compliant json dictionary"""

        version = {
            'version': {
                'schema': 1,
                'opset': 1
            }
        }

        # will probably need to return {**base, **extra}
        #   comes from:
        #   aihw-experience-validator-python/aihwx_validator/converters/json/i_converter.py

        return version

    @staticmethod
    def _epochs_from_proto(epochs_proto: Any) -> List[Dict]:
        epochs = []
        for epoch in epochs_proto.epochs:
            epoch_dict = {
                'epoch': epoch.epoch,
                'metrics': {}
            }
            for metric in epoch.metrics:
                epoch_dict['metrics'][metric.name] = metric.f

            epochs.append(epoch_dict)

        return epochs
