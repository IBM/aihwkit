# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Tests for linear layer."""

from torch import Tensor
from torch.nn import Module, Sequential

from aihwkit.nn import AnalogSequential
from aihwkit.simulator.tiles.array import TileModuleArray
from aihwkit.simulator.tiles.module import TileModule

from .helpers.decorators import parametrize_over_layers
from .helpers.layers import Linear, LinearMapped, Conv2dMapped
from .helpers.testcases import ParametrizedTestCase
from .helpers.tiles import Inference, TorchInference, TorchInferenceIRDropT, Custom


@parametrize_over_layers(
    layers=[Linear, LinearMapped, Conv2dMapped],
    tiles=[Inference, TorchInference, TorchInferenceIRDropT, Custom],
    biases=["analog", "digital", None],
)
class LayerModuleBaseTest(ParametrizedTestCase):
    """Linear base test."""

    def test_loops(self) -> None:
        """Check using a single layer."""

        model = AnalogSequential(self.get_layer(5, 5), self.get_layer(5, 5))

        self.assertEqual(len(list(model.analog_layers())), 2)
        self.assertEqual(len(list(model.named_analog_layers())), 2)

        self.assertEqual(len(list(model[0].analog_layers())), 1)
        self.assertEqual(len(list(model[0].named_analog_layers())), 1)

        self.assertEqual(len(list(model[1].analog_layers())), 1)
        self.assertEqual(len(list(model[1].named_analog_layers())), 1)

        n_global = len(list(model.named_analog_tiles()))
        n_1 = len(list(model[0].named_analog_tiles()))
        n_2 = len(list(model[1].named_analog_tiles()))
        self.assertEqual(n_global, n_1 + n_2)
        self.assertEqual(n_1, n_2)

        self.assertEqual(n_1, model[0].analog_tile_count())
        self.assertEqual(n_2, model[1].analog_tile_count())
        self.assertEqual(n_global, model.analog_tile_count())

    def check_attribute(self, model: Module, attr: str, value: int) -> None:
        """Checks whether attribute is set"""
        if isinstance(getattr(model[0], "analog_module", None), TileModule):
            self.assertEqual(getattr(model[0].analog_module, attr, 0), value)
            self.assertEqual(getattr(model[1].analog_module, attr, 0), value)
        elif isinstance(getattr(model[0], "analog_module", None), TileModuleArray):
            # assuming TileModuleArray
            self.assertTrue(getattr(model[1].analog_module, attr, -1), -1)
            self.assertTrue(getattr(model[0].analog_module, attr, -1), -1)

            for tile_row in model[0].analog_module.array:
                for tile in tile_row:
                    self.assertEqual(getattr(tile, attr, -1), value)
            for tile_row in model[1].analog_module.array:
                for tile in tile_row:
                    self.assertEqual(getattr(tile, attr, -1), value)
        else:
            # Conv2d has array field
            self.assertTrue(getattr(model[1].array, attr, -1), -1)
            self.assertTrue(getattr(model[0].array, attr, -1), -1)

            for tile_row in model[0].array:
                for tile in tile_row:
                    self.assertEqual(getattr(tile, attr, -1), value)
            for tile_row in model[1].array:
                for tile in tile_row:
                    self.assertEqual(getattr(tile, attr, -1), value)

    def test_apply_analog_tiles(self) -> None:
        """Check apply tiles."""

        model = AnalogSequential(self.get_layer(5, 5), self.get_layer(5, 5))

        def add_attr(tile: TileModule, value: int):
            if not hasattr(tile, "added_attr"):
                tile.added_attr = value
            else:
                tile.added_attr = 0

        model.apply_to_analog_tiles(lambda t: add_attr(t, 1))
        self.check_attribute(model, "added_attr", 1)

    def test_apply_modules_tiles(self) -> None:
        """Check apply tiles."""

        model = AnalogSequential(self.get_layer(5, 5), self.get_layer(5, 5))

        def add_attr(tile: TileModule, value: int) -> None:
            if not hasattr(tile, "added_attr"):
                tile.added_attr = value
            else:
                tile.added_attr = 0

        for module in model.analog_layers():
            module.apply_to_analog_tiles(lambda t: add_attr(t, 1))

        self.check_attribute(model, "added_attr", 1)

    def test_apply_layers(self) -> None:
        """Check apply layers."""

        model = AnalogSequential(self.get_layer(5, 5), self.get_layer(5, 5))

        def add_attr(mod: Module, value: int):
            if not hasattr(mod, "added_attr"):
                mod.added_attr = value
            else:
                mod.added_attr = 0

        model.apply_to_analog_tiles(lambda t: add_attr(t, 1))
        model.apply_to_analog_layers(lambda m: add_attr(m, 3))
        self.check_attribute(model, "added_attr", 1)

        self.assertEqual(getattr(model[0], "added_attr", -1), 3)
        self.assertEqual(getattr(model[1], "added_attr", -1), 3)

    def test_deep_dnn(self) -> None:
        """Check apply modules."""

        class MyMod(Module):
            """Simple class module"""

            # pylint: disable=no-self-argument

            def __init__(this) -> None:
                super().__init__()
                this.weight = self.get_layer(5, 5)

            def forward(this, x_input: Tensor) -> Tensor:
                """Simple forward."""
                return this.weight(x_input)

        model = AnalogSequential(
            self.get_layer(5, 5),
            Sequential(self.get_layer(5, 5)),
            AnalogSequential(self.get_layer(5, 5)),
            MyMod(),
        )

        if "LinearMapped" in type(next(model.analog_layers())).__name__:
            n_tiles = 16
        elif "Conv2dMapped" in type(next(model.analog_layers())).__name__:
            n_tiles = 24
        else:
            n_tiles = 4

        self.assertEqual(len(list(model.analog_tiles())), n_tiles)
        self.assertEqual(model.analog_tile_count(), n_tiles)
        model.apply_to_analog_layers(
            lambda m: self.assertEqual(m.analog_tile_count(), n_tiles // 4)
        )
        self.assertEqual(len(list(model.analog_layers())), 4)

        self.assertEqual(model[2].analog_tile_count(), n_tiles // 4)
        self.assertEqual(len(list(model[2].analog_layers())), 1)
