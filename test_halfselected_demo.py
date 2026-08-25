#!/usr/bin/env python3
"""
Demo script for testing halfselected pulse types with HS state tracking.

This script demonstrates the usage of the new halfselected pulse types:
- HALFSELECTED_STOCHASTIC
- HALFSELECTED_STOCHASTIC_STREAM

It shows how to enable HS tracking, run training, and retrieve HS transition counts.
"""

import torch
import numpy as np
from aihwkit.simulator.parameters.enums import PulseType
from aihwkit.simulator.configs import SingleRPUConfig, ConstantStepDevice
from aihwkit.nn import AnalogLinear

def test_halfselected_pulse_types():
    """Test the halfselected pulse types and HS state tracking."""

    # Device configuration with ConstantStep device
    rpu_config = SingleRPUConfig(
        device=ConstantStepDevice(
            dw_min=0.1,
            up_down=0.0,
            w_min=-1.0,
            w_max=1.0,
        )
    )

    # Test both halfselected pulse types
    pulse_types = [
        PulseType.HALFSELECTED_STOCHASTIC,
        PulseType.HALFSELECTED_STOCHASTIC_STREAM
    ]

    for pulse_type in pulse_types:
        print(f"\n=== Testing {pulse_type.value} ===")

        # Set pulse type in the update configuration
        rpu_config.update.pulse_type = pulse_type
        rpu_config.update.desired_bl = 10

        # Create analog layer
        layer = AnalogLinear(4, 2, rpu_config=rpu_config)

        # Enable HS tracking
        print("Enabling HS tracking...")
        layer.analog_module.tile.enable_hs_tracking()

        # Check if HS tracking is enabled
        is_enabled = layer.analog_module.tile.is_hs_tracking_enabled()
        print(f"HS tracking enabled: {is_enabled}")

        # Reset HS states
        layer.analog_module.tile.reset_hs_states()

        # Generate some training data
        batch_size = 8
        x = torch.randn(batch_size, 4)
        y_target = torch.randn(batch_size, 2)

        # Perform forward and backward passes
        print("Running training steps...")
        optimizer = torch.optim.SGD(layer.parameters(), lr=0.01)

        for epoch in range(5):
            optimizer.zero_grad()

            # Forward pass
            y_pred = layer(x)

            # Compute loss
            loss = torch.nn.functional.mse_loss(y_pred, y_target)

            # Backward pass (this will trigger analog updates)
            loss.backward()
            optimizer.step()

            print(f"  Epoch {epoch+1}: Loss = {loss.item():.4f}")

        # Get HS transition counts
        print("Retrieving HS transition counts...")
        hs_counts = layer.analog_module.tile.get_hs_transition_counts()

        print(f"HS transition counts shape: {hs_counts.shape}")
        print(f"Total HS transitions: {hs_counts.sum().item()}")

        # Print individual transition counts
        transitions = [
            "HS1->HS1", "HS1->HS2", "HS1->HS3", "HS1->HS4",
            "HS2->HS1", "HS2->HS2", "HS2->HS3", "HS2->HS4",
            "HS3->HS1", "HS3->HS2", "HS3->HS3", "HS3->HS4",
            "HS4->HS1", "HS4->HS2", "HS4->HS3", "HS4->HS4"
        ]

        print("Individual transition counts:")
        for i, (transition, count) in enumerate(zip(transitions, hs_counts)):
            if count > 0:
                print(f"  {transition}: {count.item()}")

        # Disable HS tracking
        layer.analog_module.tile.disable_hs_tracking()
        print("HS tracking disabled.")

        print(f"Test completed for {pulse_type.value}")

if __name__ == "__main__":
    print("Testing Halfselected Pulse Types with HS State Tracking")
    print("=" * 60)

    try:
        test_halfselected_pulse_types()
        print("\n✓ All tests completed successfully!")

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()