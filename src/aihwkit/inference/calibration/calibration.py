# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Calibration for inference."""

from typing import Optional, Dict, Tuple, TYPE_CHECKING
from collections.abc import Iterator
from functools import partial
from enum import Enum

from tqdm import tqdm

from torch import tensor, Tensor, cat, randperm, no_grad
from torch.nn import Module

from aihwkit.exceptions import ConfigError, ArgumentError
from aihwkit.simulator.parameters.enums import NoiseManagementType
from aihwkit.simulator.parameters.pre_post import PrePostProcessingRPU
from aihwkit.simulator.tiles.base import AnalogTileStateNames
from aihwkit.nn.modules.base import AnalogLayerBase

if TYPE_CHECKING:
    from aihwkit.simulator.parameters.utils import IOParameters


class InputRangeCalibrationType(Enum):
    """Input range post-training calibration type.

    Different styles of calibrating the DAC ranges post-training.
    """

    NONE = "None"
    """No Calibration."""

    MOVING_STD = "MovingStd"
    """Computes a moving average of x*standard deviation of the inputs."""

    MOVING_QUANTILE = "MovingQuantile"
    """Computes the moving average of the quantiles. Saves memory."""

    CACHE_QUANTILE = "CacheQuantile"
    """Caches inputs that are then used to compute the Xth quantile for the input range."""

    MAX = "Max"
    """Takes the abs().max() over the inputs."""


def _calibration_pre_forward(
    mod: Module,
    input_args: Tuple,
    calibration_type: InputRangeCalibrationType,
    cache_key: str,
    global_cache: Dict[str, Tensor],
    max_samples: int = 1000,
    ir_quantile: float = 0.99,
) -> None:
    """Caches inputs for calibrating the input ranges.

    Args:
        input_args: Forward inputs.
        calibration_type: type used for calibration
        cache_key: key of global cache
        max_samples: Maximal number of cache samples
    """

    # get rid of entries that are all-zeros
    x_input = input_args[0]
    x_input = x_input.reshape(-1, x_input.size(-1))
    x_input = x_input[~(x_input == 0.0).all(-1)]

    ir_params = mod.rpu_config.pre_post.input_range  # type: ignore
    cache = global_cache[cache_key]
    if calibration_type in [
        InputRangeCalibrationType.CACHE_QUANTILE,
        InputRangeCalibrationType.MAX,
    ]:
        # We need to cache the inputs
        # Add new samples to the cache
        if calibration_type in [InputRangeCalibrationType.CACHE_QUANTILE]:
            cache = cat([cache, x_input.reshape(-1, x_input.size(-1)).clone().detach().cpu()])
            # Shuffle and limit the number
            cache = cache[randperm(cache.size(0))[:max_samples]]
        else:
            # Compute the max
            if cache.numel() == 0:
                cache = x_input.abs().max().detach()
            else:
                cache = max(cache, x_input.abs().max().detach())
    elif calibration_type in [
        InputRangeCalibrationType.MOVING_QUANTILE,
        InputRangeCalibrationType.MOVING_STD,
    ]:
        idx = mod.input_range_update_idx
        val = 0
        if calibration_type == InputRangeCalibrationType.MOVING_QUANTILE:
            val = (
                x_input.abs().max()
                if ir_quantile == 1.0
                else x_input.flatten().quantile(ir_quantile)
            ).item()
        else:
            if idx < max_samples:
                std = x_input.std().item()
                val = ir_params.init_std_alpha * std

        if val > 0:
            old_val = mod.input_range.item()
            new_val = (old_val * idx + val) / (idx + 1)
            mod.set_input_range(new_val)
            mod.input_range_update_idx += 1
    else:
        raise ConfigError(f"Unknown InputRangeCalibrationType {calibration_type}")

    global_cache[cache_key] = cache


@no_grad()
def calibrate_input_ranges(
    model: Module,
    calibration_type: InputRangeCalibrationType,
    dataloader: Iterator,
    quantile: float = 0.99995,
    max_samples: int = 1000,
    std_alpha: Optional[float] = None,
    force_all_layers: bool = True,
    verbose: bool = False,
) -> None:
    """Calibrate the input ranges according to the defined strategy.

    Only tiles that support and have enabled input range learning will
    be calibrated. If noise management is turned on an error is
    raised.

    Note:
        This implementation transiently registers a new `forward_pre_hook`
        on the analog tile level. It assumes that the user has not defined
        any other forward prehooks.

    Args:
        model: The analog model for
            which to calibrate the input ranges.
        calibration_type: Strategy of the calibration. See :class:`~InputRangeCalibrationType`
        dataloader: Iterator that yields the next inputs. Is used like this
            ``x = next(dataloader); model(x)``
        quantile: Quantile used for hard-coded quantile setting.
            Defaults to 0.99995.
        max_samples: Max batch samples to cache in each tile.
            Defaults to 1000.
        std_alpha: Number of standard deviations for moving
            standard deviation strategy. Defaults to ``init_std_alpha`` from RPUConfig
        force_all_layers: Whether to force all layers to be
            (re)-calibrated (default). Otherwise only the layer having
            ``input_range.enable = True`` will be calibrated.
        verbose: Whether to print verbose output.

    Raises:
        ConfigError: If RPUConfig does not support input range learning
        ArgumentError: If non-analog model is given

    """
    # pylint: disable=too-many-statements, too-many-locals, too-many-branches

    if calibration_type == InputRangeCalibrationType.NONE:
        return

    if not isinstance(model, AnalogLayerBase) or not isinstance(model, Module):
        raise ArgumentError("Expect an analog module")

    was_training = model.training
    model = model.eval()
    handles = []
    is_perfect_dic = {}
    cache = {}  # type: Dict[str, Tensor]

    for tile_name, tile in model.named_analog_tiles():
        rpu_config = tile.rpu_config

        if not isinstance(rpu_config, PrePostProcessingRPU) or not hasattr(rpu_config, "forward"):
            continue

        if not force_all_layers and not rpu_config.pre_post.input_range.enable:
            continue

        # Reset / modify the necessary tile fields

        if not rpu_config.pre_post.input_range.enable:
            rpu_config.pre_post.input_range.enable = True
            rpu_config.pre_post.input_range.learn_input_range = False
            tile.init_input_processing()

        rpu_config.pre_post.input_range.init_from_data = 0  # turn off on-the-fly mechanism

        if std_alpha is not None:
            rpu_config.pre_post.input_range.init_std_alpha = std_alpha

        needs_set_state = False

        io_pars = rpu_config.forward  # type: IOParameters
        if io_pars.noise_management != NoiseManagementType.NONE:
            if not force_all_layers:
                raise ConfigError(
                    "Noise management should be turned off for input_range calibration."
                )
            io_pars.noise_management = NoiseManagementType.NONE
            needs_set_state = True

        is_perfect_dic[tile_name] = io_pars.is_perfect
        if (
            "Max" in calibration_type.value or "Cache" in calibration_type.value
        ) and not is_perfect_dic[tile_name]:
            rpu_config.forward.is_perfect = True
            needs_set_state = True

        if needs_set_state:
            # need to recreate tile to apply rpu config changes to tile
            tile_state = tile.__getstate__()
            tile_state[AnalogTileStateNames.RPU_CONFIG] = rpu_config
            tile.__setstate__(tile_state)

        # generate hook
        cache[tile_name] = tensor([])
        hook = partial(
            _calibration_pre_forward,
            ir_quantile=quantile,
            calibration_type=calibration_type,
            cache_key=tile_name,
            global_cache=cache,
            max_samples=max_samples,
        )
        handles.append(tile.register_forward_pre_hook(hook))

    # Pass through the samples
    progress_bar = tqdm if verbose else lambda x: x
    for args, kwargs in progress_bar(dataloader):  # type: ignore
        model(*args, **kwargs)

    # Remove hooks
    for handle in handles:
        handle.remove()

    # now create the input range fields
    for tile_name, tile in model.named_analog_tiles():
        rpu_config = tile.rpu_config
        if not rpu_config.pre_post.input_range.enable:
            rpu_config.pre_post.input_range.enable = True
            rpu_config.pre_post.input_range.learn_input_range = False
            tile.init_input_processing()

    for tile_name, tile in model.named_analog_tiles():
        rpu_config = tile.rpu_config

        if not isinstance(rpu_config, PrePostProcessingRPU) or not hasattr(rpu_config, "forward"):
            continue

        if not force_all_layers and not rpu_config.pre_post.input_range.enable:
            continue

        inputs = cache[tile_name]
        if inputs.numel() == 0:
            if verbose:
                print(f"Warning: Tile {tile_name} cached inputs is empty")
            continue

        input_range = tile.input_range.item()
        # Compute on the cache

        if calibration_type == InputRangeCalibrationType.CACHE_QUANTILE:
            input_range = inputs.flatten().quantile(quantile).item()
        elif calibration_type == InputRangeCalibrationType.MAX:
            input_range = inputs.item()

        # Restore the tile if necessary
        if rpu_config.forward.is_perfect != is_perfect_dic[tile_name]:
            tile_state = tile.__getstate__()
            tile_state[AnalogTileStateNames.RPU_CONFIG].forward.is_perfect = is_perfect_dic[
                tile_name
            ]
            tile.__setstate__(tile_state)

        # set the input range
        tile.set_input_range(input_range)
        if verbose:
            print(f"Calibrated tile {tile_name}: {input_range:.5f}.")

        # Store calibration info
        rpu_config.pre_post.input_range.init_value = tile.input_range.item()
        rpu_config.pre_post.input_range.calibration_info = calibration_type.value

    if was_training:
        model = model.train()
