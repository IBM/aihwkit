# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""TestCases for aihwkit tests."""

import math
import os
from functools import lru_cache
from typing import Type
from unittest import SkipTest, TestCase

from numpy.testing import assert_array_almost_equal, assert_raises
from aihwkit.simulator.rpu_base import cuda

SKIP_CUDA_TESTS = os.getenv("SKIP_CUDA_TESTS") or not cuda.is_compiled()


def _atol_to_decimal(atol):
    """Convert an absolute tolerance to the ``decimal`` parameter used by
    ``numpy.testing.assert_array_almost_equal``.

    The numpy function checks ``abs(a - b) < 1.5 * 10**(-decimal)``,
    so we solve for the largest ``decimal`` that still admits ``atol``.
    """
    if atol <= 0:
        return 6  # effectively exact
    return max(0, -math.ceil(math.log10(atol / 1.5)))


@lru_cache(maxsize=None)
def _probe_cuda_conv3d_tolerance(in_channels, kernel_size, n_trials=10):
    """Measure cuDNN-vs-CUBLAS numerical divergence for Conv3d.

    Uses Xavier-normalized weights so that output magnitude is ~O(1),
    making the measured absolute tolerance directly comparable to test
    scenarios with properly initialised weights.

    Returns the max absolute difference observed across *n_trials*.
    """
    import torch
    import torch.nn.functional as F

    if not torch.cuda.is_available():
        return 0.0

    k = kernel_size
    dot_dim = in_channels * k ** 3
    max_diff = 0.0
    with torch.no_grad():
        for _ in range(n_trials):
            x = torch.randn(3, in_channels, k + 1, k + 2, k + 3, device="cuda")
            w = torch.randn(3, in_channels, k, k, k, device="cuda") / math.sqrt(dot_dim)
            y_cudnn = F.conv3d(x, w, padding=k // 2)  # pylint: disable=not-callable
            with torch.backends.cudnn.flags(enabled=False):
                y_ref = F.conv3d(x, w, padding=k // 2)  # pylint: disable=not-callable
            max_diff = max(max_diff, (y_cudnn - y_ref).abs().max().item())
    return max_diff


@lru_cache(maxsize=None)
def _probe_cuda_rnn_tolerance(input_size, hidden_size, num_layers,
                              bidirectional=False, n_trials=10):
    """Measure cuDNN-vs-non-cuDNN divergence for ``torch.nn.RNN``.

    Returns the max absolute difference observed across *n_trials* on
    a forward pass (no training).
    """
    import torch

    if not torch.cuda.is_available():
        return 0.0

    seq_len = 10 if bidirectional else 3
    max_diff = 0.0
    with torch.no_grad():
        for _ in range(n_trials):
            rnn = torch.nn.RNN(
                input_size, hidden_size, num_layers,
                bidirectional=bidirectional,
            ).cuda()
            x = torch.randn(seq_len, 3, input_size, device="cuda")
            y_cudnn = rnn(x)[0]
            with torch.backends.cudnn.flags(enabled=False):
                y_ref = rnn(x)[0]
            max_diff = max(max_diff, (y_cudnn - y_ref).abs().max().item())
    return max_diff


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

    def get_cuda_decimal(self, base_atol, training_steps=0):
        """Derive a ``decimal`` value for ``assert_array_almost_equal``
        from a measured *base_atol* (the forward-pass tolerance probed at
        test-session start).

        On CPU (``self.use_cuda == False``) the default tight precision
        (``decimal=6``) is returned regardless of the measured tolerance.

        Args:
            base_atol: absolute tolerance measured by one of the ``_probe_*``
                helpers for a single forward pass on the current GPU.
            training_steps: number of forward+backward+update iterations the
                comparison spans.  Use 0 for a pure forward-pass comparison.
                Each step roughly doubles the accumulated error.

        Returns:
            ``decimal`` value suitable for
            ``numpy.testing.assert_array_almost_equal``.
        """
        if not self.use_cuda:
            return 6

        if base_atol <= 0:
            return 6

        # Safety margin: 3× for forward-only.  Each training step
        # compounds the per-layer error (empirically ~2× per step).
        margin = 3.0 * (2.0 ** training_steps)
        return _atol_to_decimal(base_atol * margin)
