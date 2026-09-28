# -*- coding: utf-8 -*-

# (C) Copyright 2026 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Regression tests for zero scales and SoftBoundsReference IG updates."""

from typing import Union

import pytest
import torch

from aihwkit.nn import AnalogLinear
from aihwkit.simulator import rpu_base
from aihwkit.simulator.configs import SingleRPUConfig, UnitCellRPUConfig
from aihwkit.simulator.configs.compounds import MixedPrecisionCompound
from aihwkit.simulator.configs.devices import SoftBoundsReferenceDevice
from aihwkit.simulator.parameters.enums import PulseType
from aihwkit.simulator.tiles.analog import AnalogTile


@pytest.fixture(params=["cpu", "cuda"])
def backend(request: pytest.FixtureRequest) -> str:
    """Run on CPU and on CUDA when both the extension and runtime support it."""
    if request.param == "cuda" and (
        not rpu_base.cuda.is_compiled() or not torch.cuda.is_available()
    ):
        pytest.skip("CUDA extension or runtime unavailable")
    return request.param


def make_config(
    dw_min: float = 0.0,
    up_down: float = 0.2,
    subtract_symmetry_point: bool = False,
    slope_dtod: float = 0.0,
    mixed_precision: bool = False,
) -> Union[SingleRPUConfig, UnitCellRPUConfig]:
    """Build the reported noiseless configuration."""
    device = SoftBoundsReferenceDevice(
        dw_min=dw_min,
        w_min=-0.8,
        w_max=0.8,
        up_down=up_down,
        subtract_symmetry_point=subtract_symmetry_point,
        dw_min_dtod=0.0,
        dw_min_std=0.0,
        w_min_dtod=0.0,
        w_max_dtod=0.0,
        up_down_dtod=0.0,
        slope_up_dtod=slope_dtod,
        slope_down_dtod=slope_dtod,
        construction_seed=123,
    )
    if mixed_precision:
        # MP requires a positive transfer granularity independent of dw_min.
        cfg = UnitCellRPUConfig(
            device=MixedPrecisionCompound(
                device=device, granularity=0.001, n_x_bins=0, n_d_bins=0, n_rows_per_transfer=-1
            )
        )
    else:
        cfg = SingleRPUConfig(device=device)
    cfg.forward.is_perfect = True
    cfg.backward.is_perfect = True
    cfg.update.pulse_type = PulseType.DETERMINISTIC_IMPLICIT
    cfg.update.update_management = True
    cfg.update.desired_bl = 31
    return cfg


@pytest.mark.parametrize("subtract_symmetry_point", [False, True])
@pytest.mark.parametrize("slope_dtod", [0.0, 0.2])
def test_zero_step_initialization(
    backend: str, subtract_symmetry_point: bool, slope_dtod: float
) -> None:
    """Zero granularity must produce finite bounds and IG response scales."""
    tile = AnalogTile(
        2,
        2,
        make_config(subtract_symmetry_point=subtract_symmetry_point, slope_dtod=slope_dtod),
        bias=False,
    ).to(backend)
    assert torch.isfinite(tile.tile.get_weights()).all()
    hidden = tile.get_hidden_parameters()
    for values in hidden.values():
        assert torch.isfinite(values).all()
    torch.testing.assert_close(hidden["dwmin_up"], torch.ones(2, 2))
    torch.testing.assert_close(hidden["dwmin_down"], torch.full((2, 2), 0.8))
    if slope_dtod == 0:
        torch.testing.assert_close(hidden["min_bound"], torch.full((2, 2), -0.8))
        torch.testing.assert_close(hidden["max_bound"], torch.full((2, 2), 0.8))
    slope_sum = hidden["dwmin_up"] / hidden["max_bound"] - (
        hidden["dwmin_down"] / hidden["min_bound"]
    )
    reference = (hidden["dwmin_up"] - hidden["dwmin_down"]) / slope_sum
    if not subtract_symmetry_point:
        reference = torch.zeros(2, 2)
    torch.testing.assert_close(hidden["reference"], reference)
    target = torch.tensor([[0.10, -0.20], [0.05, 0.30]])
    tile.tile.set_weights(target)
    torch.testing.assert_close(tile.tile.get_weights(), target)


@pytest.mark.parametrize("mixed_precision", [False, True], ids=["rank", "mp"])
@pytest.mark.parametrize("batch_size", [1, 2, 8, 31, 64, 128, 512])
@pytest.mark.parametrize("pulse_type", [PulseType.DETERMINISTIC_IMPLICIT, PulseType.STOCHASTIC])
def test_infinite_granularity_update(
    backend: str, mixed_precision: bool, batch_size: int, pulse_type: PulseType
) -> None:
    """Rank and MP updates must remain finite and preserve IG learning."""
    cfg = make_config(mixed_precision=mixed_precision)
    cfg.update.pulse_type = pulse_type
    tile = AnalogTile(2, 2, cfg, bias=False).to(backend)
    assert torch.isfinite(tile.tile.get_weights()).all()
    target = torch.tensor([[0.10, -0.20], [0.05, 0.30]])
    tile.tile.set_weights(target)
    torch.testing.assert_close(tile.tile.get_weights(), target)
    tile.tile.set_learning_rate(0.1)
    x = torch.tensor([[0.2, -0.1]], device=backend).repeat(batch_size, 1)
    d = torch.tensor([[0.3, -0.4]], device=backend).repeat(batch_size, 1)
    tile.update(x, d)
    if mixed_precision:
        # At update counter zero MP only accumulates chi. A second batch
        # triggers the transfer to analog weights before checking learning.
        tile.update(x, d)
    updated = tile.tile.get_weights()
    assert torch.isfinite(updated).all()
    assert torch.any(updated != target)
    if batch_size == 1 and not mixed_precision:
        gradient = 0.1 * (d.T @ x).cpu()
        response = torch.where(gradient > 0, 0.8 * (1 + target / 0.8), 1 - target / 0.8)
        torch.testing.assert_close(updated, target - gradient * response)
    tile.update(-x, d)
    assert torch.isfinite(tile.tile.get_weights()).all()
    for values in tile.get_hidden_parameters().values():
        assert torch.isfinite(values).all()


@pytest.mark.parametrize("subtract_symmetry_point", [False, True])
def test_zero_step_layer(backend: str, subtract_symmetry_point: bool) -> None:
    """Layer construction, weight setting, and perfect MVM remain finite."""
    layer = AnalogLinear(
        2, 2, bias=False, rpu_config=make_config(subtract_symmetry_point=subtract_symmetry_point)
    ).to(backend)
    assert torch.isfinite(layer.get_weights()[0]).all()
    target = torch.tensor([[0.10, -0.20], [0.05, 0.30]])
    layer.set_weights(target)
    torch.testing.assert_close(layer.get_weights()[0], target)
    x = torch.tensor([[0.2, -0.1]], device=backend)
    torch.testing.assert_close(layer(x), x @ target.to(backend).T)


@pytest.mark.parametrize("up_down", [-1.0, 0.2, 1.0])
@pytest.mark.parametrize("subtract_symmetry_point", [False, True])
@pytest.mark.parametrize("dw_min", [0.0, 0.001])
def test_directional_scales(
    backend: str, up_down: float, subtract_symmetry_point: bool, dw_min: float
) -> None:
    """Normal and one-sided devices retain bounds in pulsed and IG modes."""
    tile = AnalogTile(
        2,
        2,
        make_config(
            dw_min=dw_min, up_down=up_down, subtract_symmetry_point=subtract_symmetry_point
        ),
        bias=False,
    ).to(backend)
    hidden = tile.get_hidden_parameters()
    for values in hidden.values():
        assert torch.isfinite(values).all()
    torch.testing.assert_close(hidden["min_bound"], torch.full((2, 2), -0.8))
    torch.testing.assert_close(hidden["max_bound"], torch.full((2, 2), 0.8))
    effective_dw_min = 1.0 if dw_min == 0 else dw_min
    scale_up = effective_dw_min * (1 + min(up_down, 0))
    scale_down = effective_dw_min * (1 - max(up_down, 0))
    symmetry_point = (scale_up - scale_down) / (scale_up / 0.8 + scale_down / 0.8)
    reference = symmetry_point if subtract_symmetry_point else 0.0
    torch.testing.assert_close(hidden["reference"], torch.full((2, 2), reference))
    tile.tile.set_weights(torch.zeros(2, 2))
    tile.tile.set_learning_rate(0.1)
    tile.update(
        torch.tensor([[0.2, -0.1]], device=backend), torch.tensor([[0.3, -0.4]], device=backend)
    )
    updated = tile.tile.get_weights()
    assert torch.isfinite(updated).all()
    if not subtract_symmetry_point or abs(up_down) < 1:
        assert torch.any(updated != 0)


@pytest.mark.parametrize("pulse_type", [PulseType.NONE, PulseType.NONE_WITH_DEVICE])
def test_zero_step_floating_point_update(backend: str, pulse_type: PulseType) -> None:
    """The zero-step guard must not suppress explicitly selected FP updates."""
    cfg = make_config()
    cfg.update.pulse_type = pulse_type
    tile = AnalogTile(2, 2, cfg, bias=False).to(backend)
    target = torch.tensor([[0.10, -0.20], [0.05, 0.30]])
    tile.tile.set_weights(target)
    tile.tile.set_learning_rate(0.1)
    x = torch.tensor([[0.2, -0.1]], device=backend)
    d = torch.tensor([[0.3, -0.4]], device=backend)
    tile.update(x, d)
    expected = target - 0.1 * (d.T @ x).cpu()
    torch.testing.assert_close(tile.tile.get_weights(), expected)
