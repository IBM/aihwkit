#!/usr/bin/env python3
"""Compare StochasticStream vs StochasticCompressed pulse types accuracy."""

import torch
import numpy as np
import matplotlib.pyplot as plt
from aihwkit.simulator.configs import SingleRPUConfig, ConstantStepDevice
from aihwkit.simulator.parameters import PulseType
from aihwkit.simulator.tiles import AnalogTile



def create_tile_with_pulse_type(pulse_type, size=(4, 4), desired_bl=64):
    """Create an analog tile with specified pulse type."""
    rpu_config = SingleRPUConfig()

    # Use PulsedDevice instead of IdealDevice to have more control
    from aihwkit.simulator.configs.devices import PulsedDevice
    device = ConstantStepDevice()
    device.dw_min = 0.001  # Set to 1.0 to get full learning rate effect

    rpu_config.device = device
    rpu_config.update.pulse_type = pulse_type
    rpu_config.update.desired_bl = desired_bl

    # Create tile and initialize weights to zero
    tile = AnalogTile(size[0], size[1], rpu_config=rpu_config)

    # Set weights to zero
    zero_weights = torch.zeros(size)
    tile.set_weights(zero_weights)

    return tile


def test_outer_product_accuracy(u, v, pulse_type, num_trials=10, desired_bl=64):
    """Test how accurately a pulse type implements the outer product update."""

    # Create tile - size should be (output_size, input_size) = (len(v), len(u))
    tile = create_tile_with_pulse_type(pulse_type, size=(len(v), len(u)), desired_bl=desired_bl)

    # Expected weight change (outer product) - SGD uses negative update: W -= lr * d * x^T
    # For IdealDevice, the effective learning rate is controlled by device parameters
    expected_change = -torch.outer(v, u)

    errors = []
    weight_changes = []

    for trial in range(num_trials):
        # Reset tile to zero
        zero_weights = torch.zeros(len(v), len(u))
        tile.set_weights(zero_weights)

        # Get initial weights
        initial_weights = tile.get_weights()
        #print("initial weights : ", initial_weights)
        if isinstance(initial_weights, tuple):
            initial_weights = initial_weights[0]

        # Perform update
        # x_input should match input_size (columns), d_input should match output_size (rows)
        x_input = u.unsqueeze(0)  # [1, len(u)] - matches tile's input dimension
        d_input = v.unsqueeze(0)  # [1, len(v)] - matches tile's output dimension

        tile.update(x_input, d_input)

        # Get final weights
        final_weights = tile.get_weights()
        if isinstance(final_weights, tuple):
            final_weights = final_weights[0]

        #print("final_weights :", final_weights)

        # Calculate actual weight change
        actual_change = final_weights - initial_weights
        weight_changes.append(actual_change.clone())

        # Calculate error
        error = torch.norm(actual_change - expected_change).item()
        errors.append(error)

    return {
        'errors': errors,
        'mean_error': np.mean(errors),
        'std_error': np.std(errors),
        'expected_change': expected_change,
        'actual_changes': weight_changes,
        'mean_actual_change': torch.mean(torch.stack(weight_changes), dim=0)
    }


def run_comparison_experiment():
    """Run comparison experiment between StochasticStream and StochasticCompressed."""

    print("="*60)
    print("Pulse Type Accuracy Comparison Experiment")
    print("="*60)

    # Test vectors
    test_cases = [
        {
            'name': 'Small vectors',
            'u': torch.tensor([0.1, 0.2, -0.1]),
            'v': torch.tensor([0.3, -0.2, 0.1, 0.2])
        },
        {
            'name': 'Medium vectors',
            'u': torch.tensor([0.5, -0.3, 0.2, -0.4, 0.1]),
            'v': torch.tensor([0.2, 0.4, -0.3, 0.1])
        },
        {
            'name': 'Larger vectors',
            'u': torch.randn(6) * 0.3,
            'v': torch.randn(5) * 0.3
        }
    ]

    pulse_types = [
        ('StochasticCompressed', PulseType.STOCHASTIC_COMPRESSED),
        ('StochasticStream', PulseType.STOCHASTIC_STREAM),
        ('NONE_WITH_DEVICE', PulseType.NONE_WITH_DEVICE)
    ]

    bl_values = [32, 64, 128]
    num_trials = 20

    results = {}

    for case in test_cases:
        print(f"\n{case['name']}: u.shape={case['u'].shape}, v.shape={case['v'].shape}")
        print("-" * 50)

        case_results = {}

        for bl in bl_values:
            print(f"\nBit Length (BL): {bl}")

            bl_results = {}

            for pulse_name, pulse_type in pulse_types:
                try:
                    result = test_outer_product_accuracy(
                        case['u'], case['v'], pulse_type,
                        num_trials=num_trials, desired_bl=bl
                    )

                    bl_results[pulse_name] = result

                    print(f"  {pulse_name:20s}: "
                          f"Error = {result['mean_error']:.6f} ± {result['std_error']:.6f}")

                except Exception as e:
                    print(f"  {pulse_name:20s}: ERROR - {e}")
                    bl_results[pulse_name] = None

            case_results[f'BL_{bl}'] = bl_results

        results[case['name']] = case_results

    return results


def analyze_results(results):
    """Analyze and print detailed results."""

    print("\n" + "="*60)
    print("DETAILED ANALYSIS")
    print("="*60)

    for case_name, case_results in results.items():
        print(f"\n{case_name}:")
        print("-" * 40)

        for bl_key, bl_results in case_results.items():
            if bl_results is None:
                continue

            bl = bl_key.split('_')[1]
            print(f"\nBit Length {bl}:")

            stochastic_compressed = bl_results.get('StochasticCompressed')
            stochastic_stream = bl_results.get('StochasticStream')

            if stochastic_compressed and stochastic_stream:
                compressed_error = stochastic_compressed['mean_error']
                stream_error = stochastic_stream['mean_error']

                print(f"  StochasticCompressed: {compressed_error:.6f}")
                print(f"  StochasticStream:     {stream_error:.6f}")

                if compressed_error > 0:
                    ratio = stream_error / compressed_error
                    print(f"  Ratio (Stream/Compressed): {ratio:.3f}")

                    if ratio < 1:
                        print(f"  → StochasticStream is {(1-ratio)*100:.1f}% more accurate")
                    elif ratio > 1:
                        print(f"  → StochasticCompressed is {(ratio-1)*100:.1f}% more accurate")
                    else:
                        print(f"  → Both methods have similar accuracy")


def detailed_single_test():
    """Detailed analysis of a single test case."""

    print("\n" + "="*60)
    print("DETAILED SINGLE TEST ANALYSIS")
    print("="*60)

    # Simple test case
    u = torch.tensor([2.0, 2.0, 2.0])
    v = torch.tensor([-2.0, -2.0, -2.0])

    print(f"Test vectors:")
    print(f"u = {u}")
    print(f"v = {v}")
    print(f"Expected outer product (with SGD negative sign):")
    expected = -torch.outer(v, u)  # Note: v outer u to match tile dimensions
    print(expected)

    for pulse_name, pulse_type in [('StochasticCompressed', PulseType.STOCHASTIC_COMPRESSED),
                                   ('StochasticStream', PulseType.STOCHASTIC_STREAM),
                                   ('None_with_device', PulseType.NONE_WITH_DEVICE)]:

        print(f"\n{pulse_name}:")
        print("-" * 30)

        result = test_outer_product_accuracy(u, v, pulse_type, num_trials=5, desired_bl=64)

        print(f"Mean actual change:")
        print(result['mean_actual_change'])
        print(f"Difference from expected:")
        diff = result['mean_actual_change'] - expected
        print(diff)
        print(f"Mean error: {result['mean_error']:.6f}")


if __name__ == "__main__":
    try:
        # Run main comparison experiment
        results = run_comparison_experiment()

        # Analyze results
        analyze_results(results)

        # Detailed single test
        detailed_single_test()

        print("\n" + "="*60)
        print("🎉 Experiment completed successfully!")
        print("="*60)

        

    except Exception as e:
        print(f"\n❌ Experiment failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)