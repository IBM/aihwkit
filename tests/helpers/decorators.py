# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Decorators for aihwkit tests."""

from itertools import product
from functools import partial
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
        ret = {key: value for key, value in vars(obj).items() if not key.startswith("_")}
        ret["parameter"] = obj.__name__

        return ret

    def class_name(cls, _, params_dict):
        """Return a user-friendly name for a parametrized test."""
        return "{}_{}".format(cls.__name__, params_dict["parameter"])

    return parameterized_class([object_to_dict(tile) for tile in tiles], class_name_func=class_name)


def parametrize_over_layers(layers: List, tiles: List, biases: List) -> Callable:
    """Parametrize a TestCase over different kind of layers.

    The ``TestCase`` will be repeated for each combination of `layer`, `tile`
    and `bias`.

    Args:
        layers: list of layer descriptions.
        tiles: list of tile descriptions.
        biases: list of bias values: 'analog', 'digital' or None

    Returns:
        The decorated TestCase.
    """

    def get_rpu_config(tile, digital_bias, *args, **kwargs):
        rpu_config = tile.get_rpu_config(tile, *args, **kwargs)
        rpu_config.mapping.digital_bias = digital_bias
        return rpu_config

    def object_to_dict(layer, tile, bias):
        """Convert the public members of an object to a dictionary."""
        ret = {key: value for key, value in vars(layer).items() if not key.startswith("_")}
        ret["parameter"] = "{}_{}_{}Bias".format(
            layer.__name__, tile.__name__, "No" if bias is None else bias.capitalize()
        )
        digital_bias = bias == "digital"
        analog_bias = bias == "analog"
        ret["get_rpu_config"] = partial(get_rpu_config, tile=tile, digital_bias=digital_bias)
        ret["bias"] = bias is not None
        ret["digital_bias"] = digital_bias
        ret["analog_bias"] = analog_bias
        ret["tile_class"] = tile

        return ret

    def class_name(cls, _, params_dict):
        """Return a user-friendly name for a parametrized test."""
        return "{}_{}".format(cls.__name__, params_dict["parameter"])

    return parameterized_class(
        [object_to_dict(layer, tile, bias) for layer, tile, bias in product(layers, tiles, biases)],
        class_name_func=class_name,
    )


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
        return "{}_{}".format(cls.__name__, params_dict["preset_cls"].__name__)

    return parameterized_class(
        [{"preset_cls": preset} for preset in presets], class_name_func=class_name
    )


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
        ret = {key: value for key, value in vars(obj).items() if not key.startswith("_")}
        ret["parameter"] = obj.__name__

        return ret

    def class_name(cls, _, params_dict):
        """Return a user-friendly name for a parametrized test."""
        return "{}_{}".format(cls.__name__, params_dict["parameter"])

    return parameterized_class(
        [object_to_dict(model) for model in models], class_name_func=class_name
    )
