# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021 IBM. All Rights Reserved.
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
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.simulator.presets import TikiTakaReRamSBPreset

# Device used in the RPU tile
RPU_CONFIG = TikiTakaReRamSBPreset()


def main():
    """Load a predefined model from pytorch library and convert to its analog version."""
    # Load the pytorch model.
    model = resnet34()
    print(model)

    # Convert the model to its analog version.
    model = convert_to_analog(model, RPU_CONFIG, weight_scaling_omega=0.6, digital_bias=True)

    print(model)


if __name__ == '__main__':
    # Execute only if run as the entry point into the program
    main()
