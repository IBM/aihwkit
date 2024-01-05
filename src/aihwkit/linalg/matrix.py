# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Defines an analog matrix
"""

from typing import Any, Union, Tuple
from scipy.sparse.linalg import LinearOperator

from numpy import ndarray, dtype

# Imports from PyTorch.
from torch import Tensor, from_numpy
from torch.autograd import no_grad

# Imports from aihwkit.
from aihwkit.simulator.parameters.base import RPUConfigBase


class AnalogMatrix(LinearOperator):
    """Defines an analog matrix for linear algebra.

    Args:
        matrix: matrix to store into the analog crossbar array.
        rpu_config: RPU Config specifying the analog hardware properties.
        realistic: Whether to use realistic writing while storing the
            matrix elements. Otherwise, the elements will be stored
            exactly without write noise(fake write)
        to_kwargs: arguments for the torch .to call such as `device`
    """

    def __init__(
        self,
        matrix: Union[Tensor, ndarray],
        rpu_config: RPUConfigBase,
        realistic: bool = False,
        **to_kwargs: Any,
    ):
        # pylint: disable=super-init-not-called

        out_features, in_features = matrix.shape
        tile_module_class = rpu_config.get_default_tile_module_class(out_features, in_features)

        if isinstance(matrix, ndarray):
            matrix = from_numpy(matrix)

        self._tile = tile_module_class(out_features, in_features, rpu_config, False)
        self._tile = self._tile.to(**to_kwargs)
        self._tile.set_weights(matrix, realistic=realistic)
        self.realistic = realistic

    def cuda(self, *args: Any, **kwargs: Any) -> "AnalogMatrix":
        """Move to GPU."""
        self._tile = self._tile.cuda(*args, **kwargs)
        return self

    def to(self, *args: Any, **kwargs: Any) -> "AnalogMatrix":
        """Move to device, datatype, or rpu_config."""
        # pylint: disable=invalid-name
        self._tile = self._tile.to(*args, **kwargs)
        return self

    def __matmul__(self, matrix: Union[Tensor, ndarray]) -> Union[Tensor, ndarray]:
        output_numpy = False
        if isinstance(matrix, ndarray):
            output_numpy = True
            matrix = from_numpy(matrix).to(self._tile.device)

        with no_grad():
            output = self._tile.forward(matrix)

        if output_numpy:
            return output.cpu().numpy()
        return output

    def __rmatmul__(self, matrix: Union[Tensor, ndarray]) -> Union[Tensor, ndarray]:
        output_numpy = False
        if isinstance(matrix, ndarray):
            output_numpy = True
            matrix = from_numpy(matrix).to(self._tile.device)

        with no_grad():
            output = self._tile.backward(matrix)

        if output_numpy:
            return output.cpu().numpy()
        return output

    def _matmat(self, X: Union[Tensor, ndarray]) -> Union[Tensor, ndarray]:
        return self @ X

    def _matvec(self, x: Union[Tensor, ndarray]) -> Union[Tensor, ndarray]:
        return self @ x

    def _rmatvec(self, x: Union[Tensor, ndarray]) -> Union[Tensor, ndarray]:
        return x @ self

    def _rmatmat(self, X: Union[Tensor, ndarray]) -> Union[Tensor, ndarray]:
        return X @ self

    @property
    def shape(self) -> Tuple[int, int]:
        """Returns shape of analog matrix."""
        return (self._tile.out_size, self._tile.in_size)

    @property
    def dtype(self) -> dtype:
        """Return data type."""
        return self._tile.rpu_config.get_data_type().as_torch()

    def ger(
        self, x_matrix: Union[Tensor, ndarray], d_matrix: Union[Tensor, ndarray], alpha: float = 1.0
    ) -> None:
        """GER (rank update) function performed on the analog matrix.

        Args:
            x_matrix: left input matrix
            d_matrix: right input matrix
            alpha: scale factor
        """
        if isinstance(x_matrix, ndarray):
            x_matrix = from_numpy(x_matrix)

        if isinstance(d_matrix, ndarray):
            d_matrix = from_numpy(d_matrix)

        self._tile.set_learning_rate(alpha)
        self._tile.update(x_matrix, -d_matrix)
