# -*- coding: utf-8 -*-

# (C) Copyright 2026 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Download helper for SVHN from the Hugging Face hub, for aihwkit tests.

The original host of the dataset (``ufldl.stanford.edu``) used by torchvision
is not always reachable. Importing this module patches
:class:`torchvision.datasets.SVHN` so that the download step fetches the data
from Hugging Face and writes it to the same ``<root>/<split>_32x32.mat`` files
that torchvision would have downloaded. Everything else (class identity, paths,
loading) is left untouched.
"""

import os

import numpy as np
from scipy.io import savemat
from torchvision.datasets import SVHN

HF_REPO = "ufldl-stanford/svhn"
HF_CONFIG = "cropped_digits"


def _download_from_hf(self: SVHN) -> None:
    """Replacement for ``SVHN.download`` that uses Hugging Face."""
    # pylint: disable=import-outside-toplevel
    from datasets import load_dataset

    fpath = os.path.join(self.root, self.filename)
    if os.path.isfile(fpath):
        return

    os.makedirs(self.root, exist_ok=True)
    data = load_dataset(HF_REPO, HF_CONFIG, split=self.split)

    # Same layout as the original ``.mat`` files: X is HxWxCxN, y is Nx1
    # where the digit 0 is labeled 10.
    images = np.stack([np.asarray(img.convert("RGB"), dtype=np.uint8) for img in data["image"]])
    labels = np.asarray(data["label"], dtype=np.uint8)
    labels[labels == 0] = 10
    savemat(fpath, {"X": np.transpose(images, (1, 2, 3, 0)), "y": labels.reshape(-1, 1)})


def _check_integrity(self: SVHN) -> bool:
    """Replacement for ``SVHN._check_integrity``: the md5 of the original files can't match."""
    return os.path.isfile(os.path.join(self.root, self.filename))


SVHN.download = _download_from_hf  # type: ignore[method-assign]
SVHN._check_integrity = _check_integrity  # type: ignore[method-assign]  # pylint: disable=protected-access
