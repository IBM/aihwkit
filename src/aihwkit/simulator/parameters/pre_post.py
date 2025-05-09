# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

# pylint: disable=too-many-instance-attributes

"""Pre-post processing related parameters for resistive processing units."""

from dataclasses import dataclass, field
from typing import Any, Optional

from .helpers import _PrintableMixin
from .io import IOParameters
from .base import RPUConfigBase
from .enums import BoundManagementType, NoiseManagementType


@dataclass
class InputRangeParameter(_PrintableMixin):
    """Parameter related to input range learning"""

    enable: bool = field(default_factory=lambda: False, metadata={"always_show": True})
    """Whether to enable to learn the input range. Note that if enable is
    ``False`` then no clip is applied.

    Note:

        The input bound (``forward.inp_bound``) is assumed to be 1 if
        enabled as the input range already scales the input into to the
        range :math:`(-1, 1)` by dividing the input to the type by
        itself and multiplying the output accordingly.

        Typically, noise and bound management should be set to `NONE`
        for the input range learning as it replaces the dynamic
        managements with a static but learned input bound. However, in
        some exceptional experimental cases one might want to enable
        the management techniques on top of the input range learning,
        so that no error is raised if they are not set to `NONE`.
    """

    learn_input_range: bool = True
    """Whether to learn the input range when enabled.

    Note:

       If not learned, the input range should in general be set
       with some calibration method before training the DNN.

    """

    init_value: float = 3.0
    """Initial setting of the input range in case of input range learning."""

    init_from_data: int = 100
    """Number of batches to use for initialization from data. Set 0 to turn off."""

    init_std_alpha: float = 3.0
    """Standard deviation multiplier for initialization from data."""

    decay: float = 0.001
    """Decay rate for input range learning."""

    input_min_percentage: float = 0.95
    """Decay is only applied if percentage of non-clipped values is above this value.

    Note:

        The added gradient is (in case of non-clipped input
        percentage ``percentage > input_min_percentage``)::

            grad += decay * input_range
    """

    manage_output_clipping: bool = False
    """Whether to increase the input range when output clipping occurs.

    Caution:

        The output bound is taken from the ``forward.out_bound``
        value, which has to exist. Noise and bound management have to
        be set to NONE if this feature is enabled otherwise a
        ``ConfigError`` is raised.

    """

    output_min_percentage: float = 0.95
    """Increase of the input range is only applied if percentage of
    non-clipped output values is below this value.

    Note:

        The gradient subtracted from the input range is (in case of
        ``output_percentage < output_min_percentage``)::

            grad -= (1.0 - output_percentage) * input_range
    """

    gradient_scale: float = 1.0
    """Scale of the gradient magnitude (learning rate) for the input range learning."""

    gradient_relative: bool = True
    """Whether to make the gradient of the input range learning relative to
    the current range value.
    """

    calibration_info: Optional[str] = None
    """ Information field for potential post-training calibrations. """

    def supports_manage_output_clipping(self, rpu_config: Any) -> bool:
        """Checks whether rpu_config supported ``manage_output_clipping``.

        Args:
            rpu_config: RPUConfig to check

        Returns:
            True if supported otherwise False
        """

        if not hasattr(rpu_config, "forward") or rpu_config.forward.is_perfect:
            return False
        if not isinstance(rpu_config.forward, IOParameters):
            return False
        if rpu_config.forward.noise_management != NoiseManagementType.NONE:
            return False
        if rpu_config.forward.bound_management != BoundManagementType.NONE:
            return False
        return True


@dataclass
class PeripheryQuantizationParameter(_PrintableMixin):
    """
    Defines parameters used for the quantization of the periphery operations
    in the `QuantizedTorchInferenceTile`. These parameters will be applied both
    to the affine scales (e.g., per-column weight scaling) but also on the
    bias that is added, whether that happens at the tile level or at the
    array level (see `QuantizedTileModuleArray`).
    """

    n_bits: int = 0
    """The number of bits for the quantization of the periphery operations.
    If <= 0 is selected, no quantization will be applied. By default 0 (no quantization).
    """

    symmetric: bool = True
    """If True, the quantization in the periphery will be symmetrical.
    If False, asymmetric quantization will be used. This option is valid
    only if `n_bits` > 0."""

    learn_quant_params: bool = False
    """If True, the quantization scale (and offset if asymmetrical) will be
    learned during training (or was learned, if loading a checkpoint). If False,
    the scales will be estimated during training or used as trained. By default False."""

    init_learning_after: int = 100
    """If `learn_quant_params` is True, the quantization parameters will
    be estimated (minmax) up to `init_learning_after` batches before switching
    to being learnt. Has to be atleast 1 to initialize the params, by default 100."""


@dataclass
class PrePostProcessingParameter(_PrintableMixin):
    """Parameter related to digital input and output processing, such as input clip
    learning.
    """

    input_range: InputRangeParameter = field(default_factory=InputRangeParameter)


@dataclass
class PrePostProcessingParameterQuant(PrePostProcessingParameter):
    """Parameter related to digital input and output processing, such as input clip
    learning and periphery quantization.
    """

    periph_quant: PeripheryQuantizationParameter = field(
        default_factory=PeripheryQuantizationParameter
    )


@dataclass
class PrePostProcessingRPU(RPUConfigBase, _PrintableMixin):
    """Defines the pre-post parameters and utility factories"""

    pre_post: PrePostProcessingParameter = field(default_factory=PrePostProcessingParameter)
    """Parameter related digital pre and post processing."""
