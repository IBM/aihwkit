# -*- coding: utf-8 -*-

# (C) Copyright 2026 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Shared pulsed granularity and separate IG updates in vector unit cells."""

from typing import List, Optional

import pytest
import torch

from aihwkit.simulator.configs import UnitCellRPUConfig
from aihwkit.exceptions import ConfigError
from aihwkit.simulator.configs.compounds import TransferCompound, VectorUnitCell
from aihwkit.simulator.configs.devices import ConstantStepDevice, LinearStepDevice
from aihwkit.simulator.parameters import PulseType, UpdateParameters, VectorUnitCellUpdatePolicy
from aihwkit.simulator.tiles import AnalogTile


@pytest.fixture(params=[False, True], ids=["cpu", "cuda"])
def use_cuda(request: pytest.FixtureRequest) -> bool:
    """Exercise both backends, skipping CUDA only when no GPU is available."""
    if request.param and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return request.param


def _device(dw_min: float) -> ConstantStepDevice:
    return ConstantStepDevice(
        dw_min=dw_min,
        dw_min_dtod=0.0,
        dw_min_std=0.0,
        up_down_dtod=0.0,
        w_min_dtod=0.0,
        w_max_dtod=0.0,
        construction_seed=42,
    )


def _tile(
    dw_min: List[float],
    use_cuda: bool,
    gamma: Optional[List[float]] = None,
    policy: VectorUnitCellUpdatePolicy = VectorUnitCellUpdatePolicy.ALL,
    first_index: int = 0,
    pulse_type: PulseType = PulseType.DETERMINISTIC_IMPLICIT,
) -> AnalogTile:
    config = UnitCellRPUConfig(
        device=VectorUnitCell(
            unit_cell_devices=[_device(value) for value in dw_min],
            gamma_vec=[] if gamma is None else gamma,
            update_policy=policy,
            first_update_idx=first_index,
        ),
        update=UpdateParameters(
            pulse_type=pulse_type,
            # Reserve enough count words for the 100-pulse regression on CUDA.
            desired_bl=100,
            fixed_bl=False,
            update_management=False,
            update_bl_management=False,
            res=0.0,
            x_res_implicit=0.0,
            d_res_implicit=0.0,
        ),
    )
    tile = AnalogTile(1, 1, config, bias=False)
    if use_cuda:
        tile = tile.cuda(torch.device("cuda:0"))
    tile.tile.set_learning_rate(0.1)
    tile.tile.set_weights(torch.zeros(1, 1))
    return tile


def _update(tile: AnalogTile, batch: int = 1) -> None:
    signal = torch.ones(batch, 1, device=tile.device)
    tile.update(signal, signal)


@pytest.mark.parametrize(
    "pulse_type", [PulseType.DETERMINISTIC_IMPLICIT, PulseType.STOCHASTIC_COMPRESSED]
)
def test_mixed_vector_update(use_cuda: bool, pulse_type: PulseType) -> None:
    """The pulsed child takes 100 pulses; the IG child takes one 0.1 update."""
    tile = _tile([0.0, 0.001], use_cuda, pulse_type=pulse_type)
    _update(tile)
    hidden = tile.get_hidden_parameters()
    for index in range(2):
        torch.testing.assert_close(
            hidden[f"hidden_weights_{index}"].cpu(),
            torch.tensor([[-0.1]]),
            atol=1e-6,
            rtol=0.0,
        )
    assert tile.tile.get_weights().item() == pytest.approx(-0.1, abs=1e-6)


@pytest.mark.parametrize(
    "granularities,gamma,expected",
    [
        ([0.001, 0.003], [0.25, 0.75], [-0.04, -0.12]),
        ([0.0, 0.001, 0.003], [0.2, 0.2, 0.6], [-0.1, -0.04, -0.12]),
        ([0.001, 0.001], None, [-0.1, -0.1]),
        ([0.0, 0.0], [0.25, 0.75], [-0.1, -0.1]),
        ([0.0, 0.001], [1.0, 0.0], [-0.1, 0.0]),
        ([0.0, 0.001], [1.0, 1.0], [-0.05, -0.05]),
        ([0.0, 0.0], [1.0, 1.0], [-0.05, -0.05]),
        ([0.001, 0.001], [1.0, 1.0], [-0.05, -0.05]),
        ([0.001, 0.001], [0.0, 0.0], [0.0, 0.0]),
    ],
)
def test_weighted_pulsed_granularity(
    use_cuda: bool,
    granularities: List[float],
    gamma: Optional[List[float]],
    expected: List[float],
) -> None:
    """Weighted pulse steps and mode budgets give the nominal combined update."""
    tile = _tile(granularities, use_cuda, gamma=gamma)
    factors = gamma if gamma is not None else [1.0 / len(granularities)] * len(granularities)
    expected_granularity = sum(abs(factor) * step for factor, step in zip(factors, granularities))
    bindings = tile.rpu_config.device.as_bindings(tile.rpu_config.runtime.data_type)
    assert bindings.calc_weight_granularity() == pytest.approx(expected_granularity, abs=1e-8)
    _update(tile)
    hidden = tile.get_hidden_parameters()
    for index, value in enumerate(expected):
        assert hidden[f"hidden_weights_{index}"].item() == pytest.approx(value, abs=1e-6)
    expected_weight = sum(factor * value for factor, value in zip(factors, expected))
    assert tile.tile.get_weights().item() == pytest.approx(expected_weight, abs=1e-6)


@pytest.mark.parametrize("index", [0, 1])
def test_single_fixed_mixed_vector(use_cuda: bool, index: int) -> None:
    """Only the selected child receives an update, in its own update mode."""
    tile = _tile(
        [0.0, 0.001],
        use_cuda,
        policy=VectorUnitCellUpdatePolicy.SINGLE_FIXED,
        first_index=index,
    )
    _update(tile)
    hidden = tile.get_hidden_parameters()
    for child in range(2):
        expected = -0.1 if child == index else 0.0
        assert hidden[f"hidden_weights_{child}"].item() == pytest.approx(expected, abs=1e-6)
    assert tile.tile.get_weights().item() == pytest.approx(-0.05, abs=1e-6)


@pytest.mark.parametrize(
    "policy",
    [VectorUnitCellUpdatePolicy.SINGLE_SEQUENTIAL, VectorUnitCellUpdatePolicy.SINGLE_RANDOM],
)
def test_selected_child_changes_once(use_cuda: bool, policy: VectorUnitCellUpdatePolicy) -> None:
    """Selection is shared by IG and pulsed dispatch, including CUDA tuning."""
    tile = _tile([0.0, 0.001], use_cuda, policy=policy)
    tile.tile.set_learning_rate(0.01)
    previous = [0.0, 0.0]
    for _ in range(12):
        _update(tile)
        selected = tile.get_hidden_update_index()
        previous[selected] -= 0.01
        hidden = tile.get_hidden_parameters()
        for child in range(2):
            assert hidden[f"hidden_weights_{child}"].item() == pytest.approx(
                previous[child], abs=1e-6
            )


@pytest.mark.parametrize("restore", [False, True])
def test_mixed_vector_batch_and_hidden_parameters(use_cuda: bool, restore: bool) -> None:
    """Restoring hidden response scales must not turn an IG child into a pulsed child."""
    tile = _tile([0.0, 0.001, 0.003], use_cuda, gamma=[0.2, 0.2, 0.6])
    if restore:
        tile.set_hidden_parameters(tile.get_hidden_parameters())
    _update(tile, batch=3)
    assert tile.tile.get_weights().item() == pytest.approx(-0.3, abs=1e-6)


def test_ig_child_preserves_nonlinear_response(use_cuda: bool) -> None:
    """An IG child can use a different response family from its pulsed siblings."""
    tile = _tile([0.0, 0.001], False)
    config = tile.rpu_config
    config.device.unit_cell_devices[0] = LinearStepDevice(
        dw_min=0.0,
        dw_min_dtod=0.0,
        dw_min_std=0.0,
        up_down_dtod=0.0,
        w_min_dtod=0.0,
        w_max_dtod=0.0,
        gamma_up=0.5,
        gamma_down=0.5,
        gamma_up_dtod=0.0,
        gamma_down_dtod=0.0,
    )
    tile = AnalogTile(1, 1, config, bias=False)
    if use_cuda:
        tile = tile.cuda(torch.device("cuda:0"))
    tile.tile.set_learning_rate(0.03)
    tile.tile.set_weights(torch.tensor([[0.2]]))
    _update(tile)
    hidden = tile.get_hidden_parameters()
    assert hidden["hidden_weights_0"].item() == pytest.approx(0.165, abs=1e-6)
    assert hidden["hidden_weights_1"].item() == pytest.approx(0.17, abs=1e-6)
    assert tile.tile.get_weights().item() == pytest.approx(0.1675, abs=1e-6)


def test_nested_ig_vector_update(use_cuda: bool) -> None:
    """Nested vectors propagate the pulsed budget and IG gradient to their leaves."""
    tile = _tile([0.0, 0.001], False)
    config = tile.rpu_config
    config.device.unit_cell_devices = [
        VectorUnitCell(unit_cell_devices=[_device(0.0), _device(0.001)]),
        VectorUnitCell(unit_cell_devices=[_device(0.0), _device(0.001)]),
    ]
    tile = AnalogTile(1, 1, config, bias=False)
    if use_cuda:
        tile = tile.cuda(torch.device("cuda:0"))
    tile.tile.set_learning_rate(0.1)
    tile.tile.set_weights(torch.zeros(1, 1))
    _update(tile)
    assert tile.tile.get_weights().item() == pytest.approx(-0.1, abs=1e-6)


def test_transfer_rejects_nested_ig_vector() -> None:
    """Vector IG support does not bypass the restriction inside transfer compounds."""
    with pytest.raises(ConfigError, match="dw_min=0"):
        TransferCompound(
            unit_cell_devices=[
                VectorUnitCell(unit_cell_devices=[_device(0.0), _device(0.001)]),
                _device(0.001),
            ]
        )
