# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.

"""Util conversion function for the quant library"""

import numpy as np
from numpy.typing import NDArray
from torch import Tensor


def to_numpy(tensor: Tensor) -> NDArray:
    """
    Helper function that turns the given tensor into a numpy array.

    Parameters
    ----------
    tensor : torch.Tensor

    Returns
    -------
    tensor : float or np.array
    """
    if isinstance(tensor, np.ndarray):
        return tensor
    if hasattr(tensor, "is_cuda"):
        if tensor.is_cuda:
            return tensor.cpu().detach().numpy()
    if hasattr(tensor, "detach"):
        return tensor.detach().numpy()
    if hasattr(tensor, "numpy"):
        return tensor.numpy()

    return np.array(tensor)
