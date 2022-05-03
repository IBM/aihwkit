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

"""TestCases for aihwkit tests."""

from typing import Type
from unittest import SkipTest, TestCase

from numpy.testing import assert_array_almost_equal, assert_raises

from aihwkit.simulator.rpu_base import cuda


class AihwkitTestCase(TestCase):
    """Test case that contains common asserts and functions for aihwkit."""

    def assertTensorAlmostEqual(self, tensor_a, tensor_b, decimal=6):
        """Assert that two tensors are almost equal."""
        # pylint: disable=invalid-name
        array_a = tensor_a.detach().cpu().numpy()
        array_b = tensor_b.detach().cpu().numpy()
        assert_array_almost_equal(array_a, array_b, decimal=decimal)

    def assertNotAlmostEqualTensor(self, tensor_a, tensor_b, decimal=6):
        """Assert that two tensors are not equal."""
        # pylint: disable=invalid-name
        assert_raises(AssertionError, self.assertTensorAlmostEqual, tensor_a, tensor_b,
                      decimal=decimal)


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

    first_hidden_field: str = ''
    """First of the hidden fields to check during hidden parameters."""

    bias: bool = False
    """If True, the tiles and layer in this test will use bias."""

    digital_bias: bool = False
    """If True, the tiles and layer in this test will use bias."""

    parameter: str = ''
    """Friendly name of the parameters used in this test."""

    def setUp(self) -> None:
        if self.use_cuda and not cuda.is_compiled():
            raise SkipTest('not compiled with CUDA support')

        super().setUp()
