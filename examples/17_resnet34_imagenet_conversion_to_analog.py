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

"""aihwkit example 17: resnet34 CNN conversion to analog.

resnet34 inspired network based on the paper:
https://arxiv.org/abs/1512.03385
"""
# pylint: disable=invalid-name

# Imports from PyTorch.
from torchvision.models import resnet34

# Imports from aihwkit.
from aihwkit.nn.conversion import convert_to_analog_mapped
from aihwkit.simulator.presets import TikiTakaReRamSBPreset
from aihwkit.simulator.configs.utils import MappingParameter

# Example: Load a predefined model from pytorch library and convert to
#          its analog version.

# Load a pytorch model.
model = resnet34()
print(model)

# Define device and chip configuration used in the RPU tile
mapping = MappingParameter(max_input_size=512,  # analog tile size
                           max_output_size=512,
                           digital_bias=True,
                           weight_scaling_omega=0.6)  # whether to use analog or digital bias
# Choose any preset or RPU configuration here
rpu_config = TikiTakaReRamSBPreset(mapping=mapping)

# Convert the model to its analog version.
# this will replace ``Linear`` layers with ``AnalogLinearMapped``
model = convert_to_analog_mapped(model, rpu_config)

# Note: One can also use ``convert_to_analog`` instead to convert
# ``Linear`` to ``AnalogLinear`` (without mapping to multiple tiles)


print(model)
