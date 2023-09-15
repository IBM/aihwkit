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

# pylint: disable=too-many-locals

"""Exporting utilities"""

from typing import Optional, Dict, Tuple, Union, OrderedDict
from copy import deepcopy
from collections import OrderedDict as ordered_dict
from csv import writer, reader

from torch import Tensor
from torch.nn import Module, ModuleList

from numpy import array

from aihwkit.exceptions import TileError, FusionExportError
from aihwkit.simulator.tiles.inference import InferenceTile
from aihwkit.inference.converter.base import BaseConductanceConverter
from aihwkit.inference.converter.fusion import FusionConductanceConverter
from aihwkit.inference.noise.fusion import FusionImportNoiseModel
from aihwkit.nn.modules.base import AnalogLayerBase
from aihwkit.simulator.tiles.base import TileModuleBase, SimulatorTile


def _fusion_save_csv(file_name: str, conductance_data: OrderedDict, header: str) -> None:
    """Saves the conductance data to a csv file."""

    layer_names = list(conductance_data.keys())

    with open(file_name, "w", newline="", encoding="utf-8") as file:
        file.write(header)
        csv_writer = writer(file, delimiter=",")
        csv_writer.writerow(layer_names)
        for layer_data in conductance_data.values():
            csv_writer.writerow(layer_data)


def _fusion_load_csv(file_name: str, header: str) -> OrderedDict:
    """Loads the conductance data to a csv file.

    Raises:
        FusionExportError: if the header mismatches
    """

    conductance_data = ordered_dict()
    header_lines = header.split("\n")[:-1]

    with open(file_name, "r", newline="", encoding="utf-8") as file:
        csv_reader = reader(file, delimiter=",")
        for header_line in header_lines:
            tmp = next(csv_reader)
            actual_header_line = ",".join(tmp)
            if actual_header_line != header_line:
                raise FusionExportError("Header line mismatch: ", header_line, actual_header_line)
        layer_names = next(csv_reader)
        for layer_name in layer_names:
            tmp = next(csv_reader)
            conductance_data[layer_name] = [float(x) for x in tmp]
    return conductance_data


def _fusion_get_csv_header(analog_model: Module) -> str:
    """Generates and returns the fusion CSV header."""

    def add_to_header(current_header: str, field: str) -> str:
        current_header += ",".join([str(info[field]) for info in layer_infos.values()]) + "\n"
        return current_header

    layer_infos = ordered_dict()  # type: OrderedDict

    idx = 0
    for layer_name, module in analog_model.named_modules():
        if isinstance(module, (TileModuleBase, SimulatorTile, ModuleList)):
            continue

        if isinstance(module, AnalogLayerBase) and module.IS_CONTAINER:  # type: ignore
            continue

        idx += 1
        if not isinstance(module, AnalogLayerBase):
            continue

        weights = module.get_weights()[0]
        layer_infos[layer_name] = {
            "num_weights": array(weights.size()).prod().item(),
            "class": module.__class__.__name__,
            "id": idx,
            "height": weights.size()[0],
            "width": array(weights.size())[1:].prod(),
        }
    header = ""
    header = add_to_header(header, "id")
    header = add_to_header(header, "class")
    header = add_to_header(header, "num_weights")
    header = add_to_header(header, "height")
    header = add_to_header(header, "width")
    return header


def fusion_export(
    analog_model: Module,
    g_converter: Optional[BaseConductanceConverter] = None,
    file_name: Optional[str] = None,
) -> Tuple[Dict, Dict]:
    """Exports an analog module for inference on the Fusion chip.

    Args:

        analog_model: The analog model to export weights from. It is
            assumed that weights are stored in
            :class:`~aihwkit.simulator.tiles.inference.InferenceTile`

        g_converter: the conductance converter to be used. If non
            given
            :class:`~aihwkit.inference.converter.fusion.FusionConductanceConverter`
            is used.

        file_name: if give, will be used to export the CSV file to
            upload to the composer.

    Raises:
        TileError: if the analog tiles are not derived from InferenceTile

    """
    g_converter = deepcopy(g_converter) or FusionConductanceConverter()
    state_dict = analog_model.state_dict()

    conductance_data = ordered_dict()
    for layer_name, module in analog_model.named_analog_layers():
        layer_data = []
        for analog_tile in module.analog_tiles():
            if not isinstance(analog_tile, InferenceTile):
                raise TileError("Expected an InferenceTile.")

            weights = analog_tile.tile.get_weights()
            target_conductances, _ = g_converter.convert_to_conductances(weights)

            # we do not store params but recreate later. Note that for
            # that the model for import and export has to be exactly
            # the same. Better save the model checkpoint and load it
            # later? or just save the parmater in an extra file ?

            for conductance_matrix in target_conductances:
                layer_data.extend(conductance_matrix.flatten().numpy().tolist())

        conductance_data[layer_name] = layer_data

    if file_name is not None:
        header = _fusion_get_csv_header(analog_model)
        _fusion_save_csv(file_name, conductance_data, header)

    return conductance_data, state_dict


def fusion_import(
    conductance_data: Union[OrderedDict, str],
    analog_model: Module,
    state_dict: Optional[Dict] = None,
    g_converter: Optional[BaseConductanceConverter] = None,
) -> Module:
    """Imports the data from the Fusion chip and sets the model weights.

    Args:

        conductance_data: Either the data dictionary or filename from the
            fuction experiment

        analog_model: The analog model used for export. It will be changed in place.

        state_dict: the state_dict of the model to initialize the same
            weights as when using :func:`fusion_export`

            Caution:

                If not given the weights of the model are simply
                overwritte, however, it is assumed that the weights
                are the very same as during the :func:`fusion_export`
                call and not check is done to ensure it.

        g_converter: The same g-converter used for :func:`fusion_export`

    Returns:
        Model with the given layer conductance data applied. The model
        will be in eval mode.

    Raises:
        TileError: if the analog tiles are not derived from InferenceTile
        FusionExportError: if the header mismatches

    """
    g_converter = deepcopy(g_converter) or FusionConductanceConverter()
    if state_dict is not None:
        analog_model.load_state_dict(state_dict)

    if isinstance(conductance_data, str):
        header = _fusion_get_csv_header(analog_model)
        conductance_data = _fusion_load_csv(conductance_data, header)

    analog_model.eval()

    for layer_name, module in analog_model.named_analog_layers():
        layer_data = conductance_data[layer_name]
        last_value = 0
        for analog_tile in module.analog_tiles():
            if not isinstance(analog_tile, InferenceTile):
                raise TileError("Expected an InferenceTile.")

            weights = analog_tile.tile.get_weights()
            target_conductances, params = g_converter.convert_to_conductances(weights)

            # we do not store params but recreate later. Note that for
            # that the model for import and export has to be exactly
            # the same. Better save the model checkpoint and load it
            # later? or just save the parmater in an extra file ?
            programmed_conductances = []
            for conductance_matrix in target_conductances:
                num_values = conductance_matrix.numel()
                g_mat = Tensor(layer_data[last_value : last_value + num_values]).reshape(
                    *conductance_matrix.size()
                )
                last_value += num_values
                programmed_conductances.append(g_mat)

            fusion_noise = FusionImportNoiseModel(
                g_converter=g_converter, programmed_conductances=programmed_conductances, **params
            )
            analog_tile.program_weights(from_reference=False, noise_model=fusion_noise)

    analog_model.drift_analog_weights()

    return analog_model
