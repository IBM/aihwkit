# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.

import numpy as np
import torch


def to_numpy(tensor):
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
    if hasattr(tensor, 'is_cuda'):
        if tensor.is_cuda:
            return tensor.cpu().detach().numpy()
    if hasattr(tensor, 'detach'):
        return tensor.detach().numpy()
    if hasattr(tensor, 'numpy'):
        return tensor.numpy()

    return np.array(tensor)
