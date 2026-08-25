#!/usr/bin/env python3
"""Test script for StochasticStream pulse type implementation."""

import torch
import numpy as np
from aihwkit.simulator.configs import SingleRPUConfig
from aihwkit.simulator.parameters import PulseType
from aihwkit.simulator.tiles import AnalogTile


def test_stochastic_stream_pulse_type():
    """Test the StochasticStream pulse type functionality."""

    # Create RPU config with StochasticStream pulse type
    rpu_config = SingleRPUConfig()
    rpu_config.update.pulse_type = PulseType.STOCHASTIC_STREAM
    rpu_config.update.desired_bl = 64  # Set bit length

    print("Testing StochasticStream pulse type...")
    print(f"Pulse type: {rpu_config.update.pulse_type}")
    print(f"Desired BL: {rpu_config.update.desired_bl}")

    # Create analog tile
    tile = AnalogTile(4, 4, rpu_config=rpu_config)

    # Create test inputs
    x_input = torch.randn(2, 4) * 0.5  # Small values to ensure pulse probabilities < 1
    d_input = torch.randn(2, 4) * 0.5

    print(f"\nInput shapes: x={x_input.shape}, d={d_input.shape}")
    print(f"x_input range: [{x_input.min().item():.3f}, {x_input.max().item():.3f}]")
    print(f"d_input range: [{d_input.min().item():.3f}, {d_input.max().item():.3f}]")

    # Get initial weights and update count
    initial_weights = tile.get_weights()
    if isinstance(initial_weights, tuple):
        initial_weights = initial_weights[0].clone()  # Use first weight matrix
    else:
        initial_weights = initial_weights.clone()
    initial_update_count = tile.update_count

    print(f"\nInitial update count: {initial_update_count}")
    print(f"Initial weight range: [{initial_weights.min().item():.3f}, {initial_weights.max().item():.3f}]")

    # Perform update
    learning_rate = 0.01
    print(f"\nPerforming update with learning rate: {learning_rate}")

    tile.update(x_input, d_input)

    # Check results
    final_weights = tile.get_weights()
    if isinstance(final_weights, tuple):
        final_weights = final_weights[0]  # Use first weight matrix
    final_update_count = tile.update_count
    weight_change = final_weights - initial_weights

    print(f"\nFinal update count: {final_update_count}")
    print(f"Update count increment: {final_update_count - initial_update_count}")
    print(f"Weight change range: [{weight_change.min().item():.6f}, {weight_change.max().item():.6f}]")
    print(f"Weight change mean: {weight_change.mean().item():.6f}")
    print(f"Weight change std: {weight_change.std().item():.6f}")

    # Verify update occurred
    assert final_update_count > initial_update_count, "Update count should have increased"
    assert not torch.allclose(initial_weights, final_weights, atol=1e-7), "Weights should have changed"

    print("\n✓ StochasticStream pulse type test passed!")

    return True


def test_multiple_updates():
    """Test multiple updates to verify consistency."""

    rpu_config = SingleRPUConfig()
    rpu_config.update.pulse_type = PulseType.STOCHASTIC_STREAM
    rpu_config.update.desired_bl = 32

    tile = AnalogTile(3, 3, rpu_config=rpu_config)

    print("\nTesting multiple updates...")

    # Perform multiple updates
    for i in range(5):
        x_input = torch.randn(1, 3) * 0.3
        d_input = torch.randn(1, 3) * 0.3

        initial_count = tile.update_count
        tile.update(x_input, d_input)
        final_count = tile.update_count

        print(f"Update {i+1}: count {initial_count} -> {final_count} (increment: {final_count - initial_count})")

        assert final_count > initial_count, f"Update count should increase in iteration {i+1}"

    print("✓ Multiple updates test passed!")

    return True


def test_batch_update():
    """Test batch update behavior."""

    rpu_config = SingleRPUConfig()
    rpu_config.update.pulse_type = PulseType.STOCHASTIC_STREAM
    rpu_config.update.desired_bl = 16

    tile = AnalogTile(2, 2, rpu_config=rpu_config)

    print("\nTesting batch update...")

    batch_size = 8
    x_input = torch.randn(batch_size, 2) * 0.2
    d_input = torch.randn(batch_size, 2) * 0.2

    initial_count = tile.update_count
    tile.update(x_input, d_input)
    final_count = tile.update_count

    print(f"Batch size: {batch_size}")
    print(f"Update count: {initial_count} -> {final_count}")
    print(f"Expected increment: {batch_size}, Actual increment: {final_count - initial_count}")

    assert final_count - initial_count == batch_size, "Update count should increase by batch size"

    print("✓ Batch update test passed!")

    return True


if __name__ == "__main__":
    try:
        # Run all tests
        test_stochastic_stream_pulse_type()
        test_multiple_updates()
        test_batch_update()

        print("\n" + "="*50)
        print("🎉 All StochasticStream tests passed successfully!")
        print("="*50)

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)