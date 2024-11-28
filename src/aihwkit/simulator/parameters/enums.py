# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

# pylint: disable=too-many-instance-attributes

"""Utility enumerators for resistive processing units."""

from enum import Enum
from torch import dtype, float32, float64, half

_TORCH_DATA_TYPE_MAP = {"float": float32, "double": float64, "half": half}


class RPUDataType(Enum):
    """Data type for the C++ simulation of the analog tiles.

    Note:

        `FLOAT` is the default. Other data types need special
        compilation options to be enabled.
    """

    FLOAT = "float"
    """Float32 (single) precision format."""

    DOUBLE = "double"
    """Float64 (double) precision format."""

    HALF = "half"
    """Float16 (half) precision format."""

    def as_torch(self) -> dtype:
        """Returns corresponding torch dtype."""
        return _TORCH_DATA_TYPE_MAP[self.value]


class BoundManagementType(Enum):
    """Bound management type.

    In the case ``Iterative`` the MAC is iteratively recomputed with
    inputs iteratively halved, when the output bound was hit.

    Caution:
        Bound management is **only** available for the forward pass. It
        will be ignored when used for the backward pass.
    """

    NONE = "None"
    """No bound management."""

    ITERATIVE = "Iterative"
    r"""Iteratively recomputes input scale set to :math:`\alpha\leftarrow\alpha/2`.

    It iteratively recomputes the bounds up to limit of passes (given by
    ``max_bm_factor`` or ``max_bm_res``).
    """

    ITERATIVE_WORST_CASE = "IterativeWorstCase"
    """Worst case bound management.

    Uses ``AbsMax`` noise management for the first pass and only when output
    bound is hit, the ``AbsMaxNPSum`` for the second. Thus, at most 2 passes
    are computed.
    """

    SHIFT = "Shift"
    """Shift bound management.

    Shifts the output by adding the difference ``output_bound - max_output`` to
    the analog output value. This is only useful to increase the dynamic range
    before the softmax, where the max can be safely.

    Note:
        Shifting needs hardware implementations.
    """


class NoiseManagementType(Enum):
    r"""Noise management type.

    Noise management determines a factor :math:`\alpha` how the input is reduced:

    .. math:: \mathbf{y} = \alpha\;F_\text{analog-mac}\left(\mathbf{x}/\alpha\right)
    """

    NONE = "None"
    """No noise management."""

    ABS_MAX = "AbsMax"
    r"""Use :math:`\alpha\equiv\max{|\mathbf{x}|}`."""

    ABS_MAX_NP_SUM = "AbsMaxNPSum"
    """Assume weight value is constant and given by ``nm_assumed_wmax``.

    Takes a worst case scenario of the weight matrix to calculate the input
    scale to ensure that output is not clipping. Assumed weight value is
    constant and given by ``nm_assumed_wmax``.
    """

    MAX = "Max"
    r"""Use :math:`\alpha\equiv\max{\mathbf{x}}`."""

    CONSTANT = "Constant"
    r"""A constant value (given by parameter ``nm_thres``)."""

    AVERAGE_ABS_MAX = "AverageAbsMax"
    """Moment-based scale input scale estimation.

    Computes the average abs max over the mini-batch and applies ``nm_decay``
    to update the value with the history.

    Note:
        ``nm_decay`` is ``1-momentum`` and always given in mini-batches.
        However, the CUDA implementation does not discount values within
        mini-batches, whereas the CPU implementation does.
    """


class WeightNoiseType(Enum):
    r"""Output weight noise type.

    The weight noise is applied for each MAC computation, while not
    touching the actual weight matrix but referring it to the output.

    .. math:: y_i = \sum_j w_{ij}+\xi_{ij}
    """

    NONE = "None"
    """No weight noise."""

    ADDITIVE_CONSTANT = "AdditiveConstant"
    r"""The :math:`\xi\sim{\cal N}(0,\sigma)` thus all are Gaussian distributed.

    :math:`\sigma` is determined by ``w_noise``.
    """

    PCM_READ = "PCMRead"
    """Output-referred PCM-like read noise.

    Output-referred PCM-like read noise that scales with the amount of current
    generated for each output line and thus scales with both conductance values
    and input strength.

    The same general for is taken as for PCM-like statistical model of the 1/f
    noise during inference, see
    :class:`aihwkit.inference.noise.pcm.PCMLikeNoiseModel`.
    """


class PulseType(Enum):
    """Pulse type."""

    NONE = "None"
    """Floating point update instead of pulses."""

    STOCHASTIC_COMPRESSED = "StochasticCompressed"
    """Generates actual stochastic bit lines.

    Plus and minus pulses are taken in the same pass.
    """

    STOCHASTIC = "Stochastic"
    """Two passes for plus and minus (only CPU)."""

    NONE_WITH_DEVICE = "NoneWithDevice"
    """Floating point like ``None``, but with analog devices (e.g. weight
    clipping)."""

    MEAN_COUNT = "MeanCount"
    """Coincidence based in prob (:math:`p_a p_b`)."""

    DETERMINISTIC_IMPLICIT = "DeterministicImplicit"
    r"""Coincidences are computed in deterministic manner.

    Coincidences are calculated by :math:`b_l x_q d_q` where ``BL`` is the
    desired bit length (possibly subject to dynamic adjustments using
    ``update_bl_management``) and :math:`x_q` and :math:`d_q` are the quantized
    input and error values, respectively, normalized to the range
    :math:`0,\ldots,1`. It can be shown that explicit bit lines exist that
    generate these coincidences.
    """


class WeightModifierType(Enum):
    """Weight modifier type."""

    NONE = "None"
    """No weight modifier. Nothing happens to the weight. """

    DISCRETIZE = "Discretize"
    """Quantize the weights."""

    MULT_NORMAL = "MultNormal"
    """Multiplicative Gaussian noise."""

    ADD_NORMAL = "AddNormal"
    """Additive Gaussian noise."""

    DISCRETIZE_ADD_NORMAL = "DiscretizeAddNormal"
    """First discretize and then additive Gaussian noise."""

    DOREFA = "DoReFa"
    """DoReFa discretization."""

    POLY = "Poly"
    r"""Nth order Polynomial noise model (in terms of the weight value).

    In detail, for the duration of a mini-batch, each weight will be
    added a Gaussian random number with the standard deviation of
    :math:`\sigma_\text{wnoise} (c_0 + c_1 w_{ij}/\omega +
    c_N w_{ij}^N/\omega^N)`, where :math:`omega` is either the actual
    max weight (if ``rel_to_actual_wmax`` is set) or the value
    ``assumed_wmax``.
    """

    PCM_NOISE = "PCMNoise"
    """PCM programming / drift noise model added to the weight matrix
    during training.
    """

    PROG_NOISE = "ProgNoise"
    """Programming noise model added to the weight matrix during
    training. Same as "POLY", except that weights are ensured to keep
    the sign with noise mirrored at zero.
    """

    DROP_CONNECT = "DropConnect"
    """Drop connect.

    Note that if "pdrop > 0" this is applied. It drop connect can be
    applied in conjunction to other modifiers as well.
    """

    COPY = "Copy"
    """Legacy. No explicit weight modifier, however, pdrop is still observed.

    Use DROP_CONNECT or NONE instead.
    """


class WeightClipType(Enum):
    """Weight clipper type."""

    NONE = "None"
    """None."""

    FIXED_VALUE = "FixedValue"
    """Clip to fixed value give, symmetrical around zero."""

    LAYER_GAUSSIAN = "LayerGaussian"
    """Calculates the second moment of the whole weight matrix and clips
    at ``sigma`` times the result symmetrically around zero."""

    AVERAGE_CHANNEL_MAX = "AverageChannelMax"
    """Calculates the abs max of each output channel (row of the weight
    matrix) and takes the average as clipping value for all."""


class WeightRemapType(Enum):
    """Weight clipper type."""

    NONE = "None"
    """None."""

    LAYERWISE_SYMMETRIC = "LayerwiseSymmetric"
    """Remap according to the absolute max of the full weight matrix."""

    CHANNELWISE_SYMMETRIC = "ChannelwiseSymmetric"
    """Remap each column (output channel) in respect to the absolute max."""


class VectorUnitCellUpdatePolicy(Enum):
    """Vector unit cell update policy."""

    ALL = "All"
    """All devices updated simultaneously."""

    SINGLE_FIXED = "SingleFixed"
    """Device index is not changed. Can be set initially and/or updated on
    the fly."""

    SINGLE_SEQUENTIAL = "SingleSequential"
    """Each device one at a time in sequence."""

    SINGLE_RANDOM = "SingleRandom"
    """A single device is selected by random choice each mini-batch."""


class AnalogMVType(Enum):
    """Type of the analog matrix-vector product."""

    IDEAL = "Ideal"
    """FP mat-vec without any non-idealities. Same as setting
    ``is_perfect=True``."""

    ONE_PASS = "OnePass"
    """One pass through the crossbar array for positive and negative inputs."""

    POS_NEG_SEPARATE = "PosNegSeparate"
    """Two passes through the crossbar array for positive and negative
    inputs separately. The output of the two passes are added in
    analog and then passed once through ADC stage (which also applies
    the output noise, range clipping, output non-linearity etc.).

    Caution:
        Only supported for RPUCuda tiles (based on C++ not pure torch)
    """

    POS_NEG_SEPARATE_DIGITAL_SUM = "PosNegSeparateDigitalSum"
    """Two passes through the crossbar array for positive and negative
    inputs separately. The ADC output stage is applied to each pass
    separately and the results are summed in full precision (i.e. in
    digital).

    Caution:
        Only supported for RPUCuda tiles (based on C++ not pure torch)
    """

    SPLIT_MODE = "SplitMode"
    """Split PWM bits into two phases to speedup MAC and increase energy
    efficiency (may sacrifice some accuracy).

    Caution:
        Only supported for ``TorchInferenceRPUConfigIRDropT``
    """

    BIT_WISE = "BitWise"
    """Bit-wise PWM input to speedup MAC and increase energy
    efficiency (may sacrifice some accuracy).

    Caution:
        Only supported for ``TorchInferenceRPUConfigIRDropT``
    """


# legacy
class CountLRFeedbackPolicy(Enum):
    """Depreciated. Legacy."""

    NONE = "None"
    PAST_MOMENTUM = "PastMomentum"
    EXTERN = "Extern"
