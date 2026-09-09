#!/usr/bin/env python3
"""Focused verification of the half-select (HS) CPU path.

Confirms three things the existing test_hs_training_cpu.py does NOT:
  1. enable_hs_tracking() engages and get_hs_transition_counts() records transitions.
  2. hs_decay is configurable from Python and reaches the C++ device.
  3. hs_decay materially changes the weight trajectory (1.0 = no decay vs 0.5 = strong decay),
     under identical seed/inputs with tracking enabled.
"""
from __future__ import annotations

import numpy as np
import torch

from aihwkit.simulator.configs import SingleRPUConfig
from aihwkit.simulator.configs.devices import ConstantStepDevice
from aihwkit.simulator.parameters.enums import PulseType
from aihwkit.simulator.tiles import AnalogTile


def make_tile(hs_decay: float, pulse_type: PulseType, seed: int = 7) -> AnalogTile:
    torch.manual_seed(seed)
    device = ConstantStepDevice(dw_min=0.01)
    device.hs_decay = hs_decay
    rc = SingleRPUConfig(device=device)
    rc.update.pulse_type = pulse_type
    rc.update.desired_bl = 31
    tile = AnalogTile(8, 8, rpu_config=rc)
    tile.set_weights(torch.zeros(8, 8))
    return tile


def run_updates(tile: AnalogTile, n: int, track: bool, seed: int = 123):
    raw = tile.tile
    if track:
        raw.enable_hs_tracking()
    gen = torch.Generator().manual_seed(seed)
    # fixed input/error vectors; sign structure drives half-selected transitions
    x = (torch.rand(8, generator=gen) - 0.5).sign()
    d = (torch.rand(8, generator=gen) - 0.5).sign()
    for _ in range(n):
        tile.update(x, d)
    counts = list(raw.get_hs_transition_counts()) if track else None
    return tile.get_weights()[0].clone(), counts, (raw.is_hs_tracking_enabled() if track else False)


def main() -> int:
    print("=== 1. hs_decay configurability ===")
    t = make_tile(0.5, PulseType.HALFSELECTED_STOCHASTIC)
    b = t.rpu_config.device.as_bindings(
        __import__("aihwkit.simulator.parameters.enums", fromlist=["RPUDataType"]).RPUDataType.FLOAT
    )
    print(f"   dataclass hs_decay=0.5 -> bindings hs_decay={b.hs_decay}  "
          f"{'OK' if abs(b.hs_decay - 0.5) < 1e-6 else 'FAIL'}")

    print("\n=== 2. tracking engages, transition counts recorded ===")
    _, counts, enabled = run_updates(make_tile(0.99, PulseType.HALFSELECTED_STOCHASTIC), n=50, track=True)
    total = sum(counts)
    print(f"   is_hs_tracking_enabled: {enabled}")
    print(f"   transition counts (16): {counts}")
    print(f"   total transitions: {total}  {'OK (>0)' if total > 0 else 'FAIL (==0)'}")

    print("\n=== 3. hs_decay materially changes weights (tracking on) ===")
    w_nodecay, _, _ = run_updates(make_tile(1.0, PulseType.HALFSELECTED_STOCHASTIC), n=50, track=True)
    w_decay, _, _ = run_updates(make_tile(0.5, PulseType.HALFSELECTED_STOCHASTIC), n=50, track=True)
    diff = (w_nodecay - w_decay).abs().max().item()
    print(f"   max|W(hs_decay=1.0) - W(hs_decay=0.5)| = {diff:.5f}  "
          f"{'OK (decay changes weights)' if diff > 1e-4 else 'FAIL (no effect)'}")
    print(f"   ||W|| no-decay={w_nodecay.norm().item():.4f}  decay={w_decay.norm().item():.4f}")

    ok = (abs(b.hs_decay - 0.5) < 1e-6) and (total > 0) and (diff > 1e-4)
    print(f"\n{'PASS' if ok else 'FAIL'}: HS CPU path fully verified.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
