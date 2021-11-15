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

"""Decorators for aihwkit tests."""

from itertools import product
from typing import List, Callable

from parameterized import parameterized_class


def parametrize_over_tiles(tiles: List) -> Callable:
    """Parametrize a TestCase over different kind of tiles.

    Args:
        tiles: list of tile descriptions. The ``TestCase`` will be repeated
            for each of the entries in the list.

    Returns:
        The decorated TestCase.
    """

    def object_to_dict(obj):
        """Convert the public members of an object to a dictionary."""
        ret = {key: value for key, value in vars(obj).items()
               if not key.startswith('_')}
        ret['parameter'] = obj.__name__

        return ret

    def class_name(cls, _, params_dict):
        """Return a user-friendly name for a parametrized test."""
        return '{}_{}'.format(cls.__name__, params_dict['parameter'])

    return parameterized_class([object_to_dict(tile) for tile in tiles],
                               class_name_func=class_name)


def parametrize_over_layers(layers: List, tiles: List, biases: List, digital_biases: List) \
        -> Callable:
    """Parametrize a TestCase over different kind of layers.

    The ``TestCase`` will be repeated for each combination of `layer`, `tile`
    and `bias`.

    Args:
        layers: list of layer descriptions.
        tiles: list of tile descriptions.
        biases: list of bias values.
        digital_biases: list of digital_biases values.

    Returns:
        The decorated TestCase.
    """

    def object_to_dict(layer, tile, bias, digital_bias):
        """Convert the public members of an object to a dictionary."""
        ret = {key: value for key, value in vars(layer).items()
               if not key.startswith('_')}
        ret['parameter'] = '{}_{}_{}'.format(layer.__name__,
                                             tile.__name__,
                                             'Bias' if bias else 'NoBias')
        ret['get_rpu_config'] = tile.get_rpu_config
        ret['bias'] = bias
        ret['digital_bias'] = digital_bias
        ret['tile_class'] = tile

        return ret

    def class_name(cls, _, params_dict):
        """Return a user-friendly name for a parametrized test."""
        return '{}_{}'.format(cls.__name__, params_dict['parameter'])

    return parameterized_class(
        [object_to_dict(layer, tile, bias, digital_bias) for
         layer, tile, bias, digital_bias in product(layers, tiles, biases, digital_biases)],
        class_name_func=class_name)


def parametrize_over_presets(presets: List) -> Callable:
    """Parametrize a TestCase over different kind of presets.

    Note that this decorator expects a list of Presets, as opposed to a list of
    helper objects.

    Args:
        presets: list of presets.

    Returns:
        The decorated TestCase.
    """

    def class_name(cls, _, params_dict):
        """Return a user-friendly name for a parametrized test."""
        return '{}_{}'.format(cls.__name__, params_dict['preset_cls'].__name__)

    return parameterized_class(
        [{'preset_cls': preset} for preset in presets],
        class_name_func=class_name)


def parametrize_over_experiments(models: List) -> Callable:
    """Parametrize a TestCase over different kind of experiments.

    Args:
        models: list of model descriptions. The ``TestCase`` will be
            repeated for each of the entries in the list.

    Returns:
        The decorated TestCase.
    """

    def object_to_dict(obj):
        """Convert the public members of an object to a dictionary."""
        ret = {key: value for key, value in vars(obj).items()
               if not key.startswith('_')}
        ret['parameter'] = obj.__name__

        return ret

    def class_name(cls, _, params_dict):
        """Return a user-friendly name for a parametrized test."""
        return '{}_{}'.format(cls.__name__, params_dict['parameter'])

    return parameterized_class([object_to_dict(model) for model in models],
                               class_name_func=class_name)
