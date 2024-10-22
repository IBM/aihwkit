# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""TestCases for aihwkit tests."""

import os
from typing import Type
from unittest import SkipTest, TestCase

from numpy.testing import assert_array_almost_equal, assert_raises
from aihwkit.simulator.rpu_base import cuda

SKIP_CUDA_TESTS = os.getenv("SKIP_CUDA_TESTS") or not cuda.is_compiled()


class AihwkitTestCase(TestCase):
    """Test case that contains common asserts and functions for aihwkit."""

    def assertTensorAlmostEqual(self, tensor_a, tensor_b, decimal=4):
        """Assert that two tensors are almost equal."""
        # pylint: disable=invalid-name
        array_a = tensor_a.detach().cpu().numpy()
        array_b = tensor_b.detach().cpu().numpy()
        assert_array_almost_equal(array_a, array_b, decimal=decimal)

    def assertNotAlmostEqualTensor(self, tensor_a, tensor_b, decimal=4):
        """Assert that two tensors are not equal."""
        # pylint: disable=invalid-name
        assert_raises(
            AssertionError, self.assertTensorAlmostEqual, tensor_a, tensor_b, decimal=decimal
        )


class ParametrizedTestCase(AihwkitTestCase):
    """Test case that is parametrized using aihwkit test decorators.

    Base class for aihwkit parametrized tests. This base class should be used
    in combination with the decorators in the ``.decorators`` package
    (``@parametrize_over_tiles``, ``@parametrize_over_layers``).

    The decorators will set the class attributes and methods accordingly at
    runtime for each parametrized test.
    """

    use_cuda: bool = False
    """Determines if the test case requires CUDA support."""

    simulator_tile_class: Type = None
    """rpu_base tile class that is expected to be used internally in this test."""

    first_hidden_field: str = ""
    """First of the hidden fields to check during hidden parameters."""

    bias: bool = False
    """If True, the tiles and layer in this test will use bias."""

    digital_bias: bool = False
    """If True, the tiles and layer in this test will use bias."""

    parameter: str = ""
    """Friendly name of the parameters used in this test."""

    def setUp(self) -> None:
        if self.use_cuda and SKIP_CUDA_TESTS:
            raise SkipTest("not compiled with CUDA support")

        super().setUp()
