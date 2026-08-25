#!/usr/bin/env python3
"""Compare actual update amounts between StochasticStream and Stochastic pulse types."""

import torch
from aihwkit.simulator.configs import SingleRPUConfig
from aihwkit.simulator.configs.devices import ConstantStepDevice
from aihwkit.simulator.parameters import PulseType
from aihwkit.simulator.tiles import AnalogTile


def test_pulse_type_update_amount(pulse_type, pulse_name, test_vectors):
    """Test actual update amount for a specific pulse type."""

    print(f"\n{'='*60}")
    print(f"Testing {pulse_name}")
    print(f"{'='*60}")

    # Create config
    rpu_config = SingleRPUConfig()
    device = ConstantStepDevice()
    device.dw_min = 0.001

    rpu_config.device = device
    rpu_config.update.pulse_type = pulse_type
    rpu_config.update.desired_bl = 64

    # Create tile
    tile_size = (len(test_vectors['v']), len(test_vectors['u']))
    tile = AnalogTile(tile_size[0], tile_size[1], rpu_config=rpu_config)

    # Set weights to zero
    tile.set_weights(torch.zeros(tile_size))

    # Get test vectors
    u = test_vectors['u']
    v = test_vectors['v']

    print(f"Test vectors: u = {u}, v = {v}")

    # Expected outer product (negative for SGD)
    expected = -torch.outer(v, u)
    print(f"Expected update:\n{expected}")

    # Perform update
    x_input = u.unsqueeze(0)
    d_input = v.unsqueeze(0)

    print(f"\nPerforming update with {pulse_name}...")
    tile.update(x_input, d_input)

    # Get actual weights
    actual_weights = tile.get_weights()
    if isinstance(actual_weights, tuple):
        actual_weights = actual_weights[0]

    print(f"Actual update:\n{actual_weights}")

    # Calculate ratios
    ratio_matrix = actual_weights / expected

    # Get non-zero ratios
    non_zero_mask = expected != 0
    if non_zero_mask.any():
        ratios = ratio_matrix[non_zero_mask]
        avg_ratio = ratios.mean().item()
        std_ratio = ratios.std().item()

        print(f"\nRatio analysis:")
        print(f"  Individual ratios: {ratios}")
        print(f"  Average ratio: {avg_ratio:.6f}")
        print(f"  Std deviation: {std_ratio:.6f}")
        print(f"  Effective learning rate: {avg_ratio:.6f}")

        return {
            'pulse_type': pulse_name,
            'expected': expected,
            'actual': actual_weights,
            'avg_ratio': avg_ratio,
            'std_ratio': std_ratio,
            'ratios': ratios
        }
    else:
        print("No non-zero expected values found!")
        return None


def compare_pulse_types():
    """Compare different pulse types."""

    # Test vectors
    test_vectors = {
        'u': torch.tensor([1.0, 0.5, -0.5]),
        'v': torch.tensor([1.0, -1.0, 0.5])
    }

    # Pulse types to test
    pulse_types = [
        (PulseType.STOCHASTIC_COMPRESSED, "StochasticCompressed"),
        (PulseType.STOCHASTIC_STREAM, "StochasticStream"),
        (PulseType.STOCHASTIC, "Stochastic")
    ]

    results = []

    for pulse_type, pulse_name in pulse_types:
        try:
            result = test_pulse_type_update_amount(pulse_type, pulse_name, test_vectors)
            if result:
                results.append(result)
        except Exception as e:
            print(f"\nError testing {pulse_name}: {e}")

    # Compare results
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}")

    if len(results) >= 2:
        print(f"{'Pulse Type':<25} {'Avg Ratio':<15} {'Std Ratio':<15} {'Effective LR':<15}")
        print("-" * 70)

        for result in results:
            print(f"{result['pulse_type']:<25} {result['avg_ratio']:<15.6f} "
                  f"{result['std_ratio']:<15.6f} {result['avg_ratio']:<15.6f}")

        # Check if all methods have similar scaling
        avg_ratios = [r['avg_ratio'] for r in results]
        ratio_range = max(avg_ratios) - min(avg_ratios)
        ratio_mean = sum(avg_ratios) / len(avg_ratios)

        print(f"\nScaling Analysis:")
        print(f"  Mean effective learning rate: {ratio_mean:.6f}")
        print(f"  Range of ratios: {ratio_range:.6f}")
        print(f"  Relative variation: {ratio_range/ratio_mean*100:.2f}%")

        # Check if close to 0.1
        if abs(ratio_mean - 0.1) < 0.01:
            print(f"  ✓ All methods show ~0.1x scaling (expected)")
        else:
            print(f"  ⚠ Methods show {ratio_mean:.3f}x scaling (not 0.1x)")

    return results


def test_different_input_scales():
    """Test with different input magnitudes to see scaling behavior."""

    print(f"\n{'='*80}")
    print("TESTING DIFFERENT INPUT SCALES")
    print(f"{'='*80}")

    scales = [0.1, 0.5, 1.0, 2.0]
    base_u = torch.tensor([1.0, 1.0])
    base_v = torch.tensor([1.0, -1.0])

    for scale in scales:
        print(f"\n--- Scale factor: {scale} ---")

        test_vectors = {
            'u': base_u * scale,
            'v': base_v * scale
        }

        # Test StochasticStream
        result = test_pulse_type_update_amount(
            PulseType.STOCHASTIC_STREAM,
            f"StochasticStream (scale={scale})",
            test_vectors
        )

        if result:
            print(f"Effective LR at scale {scale}: {result['avg_ratio']:.6f}")


if __name__ == "__main__":
    # Main comparison
    results = compare_pulse_types()

    # Test different scales
    test_different_input_scales()

    print(f"\n{'='*80}")
    print("TEST COMPLETED")
    print(f"{'='*80}")