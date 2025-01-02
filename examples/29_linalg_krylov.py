# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

# pylint: disable=invalid-name

"""Krylov with analog preconditioner"""

import numpy as np
import matplotlib.pyplot as plt
from torch import device as torch_device

from pyamg.gallery import poisson
from pyamg.krylov import fgmres
from pyamg.util.linalg import norm
from pyamg import smoothed_aggregation_solver

from aihwkit.linalg import AnalogMatrix
from aihwkit.simulator.presets import ReRamSBPreset
from aihwkit.simulator.rpu_base import cuda


# Check GPU device
DEVICE = torch_device("cuda" if cuda.is_compiled() else "cpu")

# config the hardware properties
rpu_config = ReRamSBPreset()

A = poisson((10, 10)).astype("float32")
n = A.shape[0]
b = np.ones((n,)).astype("float32")
E = np.eye(n).astype("float32")

ml = smoothed_aggregation_solver(A)
M_fp = ml.aspreconditioner() * E

M = AnalogMatrix(M_fp, rpu_config=rpu_config, realistic=False, device=DEVICE)

(x, flag) = fgmres(A, b, M=M, maxiter=6, tol=1e-8)
print(f"{norm(b - A*x):.6}")

plt.clf()

plt.subplot(1, 2, 1)
plt.imshow(M @ E)
plt.title("Analog pre-conditioner")

plt.subplot(1, 2, 2)
plt.imshow(M_fp)
plt.title("Digital pre-conditioner")
