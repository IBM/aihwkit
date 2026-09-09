#!/usr/bin/env python3
"""
Test HS_decay parameter access from Python.
"""

from aihwkit.simulator.configs import SingleRPUConfig
from aihwkit.simulator.configs.devices import ConstantStepDevice
from aihwkit.simulator.parameters.enums import PulseType
from aihwkit.nn import AnalogLinear

def test_hs_decay_access():
    """Test different ways to access and set hs_decay parameter."""
    print("=== Testing HS_decay Python Access ===")

    # Method 1: Direct device configuration
    print("\n1. Direct device configuration:")
    device = ConstantStepDevice()

    # Check if hs_decay attribute exists
    if hasattr(device, 'hs_decay'):
        print(f"   ✓ device.hs_decay exists: {device.hs_decay}")
        device.hs_decay = 0.95
        print(f"   ✓ Set to 0.95: {device.hs_decay}")
    else:
        print("   ✗ device.hs_decay does not exist")
        print("   Available attributes:", [attr for attr in dir(device) if not attr.startswith('_')])

    # Method 2: Through RPU config
    print("\n2. Through RPU config:")
    rpu_config = SingleRPUConfig(device=device)
    rpu_config.update.pulse_type = PulseType.HALFSELECTED_STOCHASTIC

    if hasattr(rpu_config.device, 'hs_decay'):
        print(f"   ✓ rpu_config.device.hs_decay: {rpu_config.device.hs_decay}")
    else:
        print("   ✗ rpu_config.device.hs_decay does not exist")

    # Method 3: In AnalogLinear layer
    print("\n3. Through AnalogLinear layer:")
    try:
        layer = AnalogLinear(4, 2, rpu_config=rpu_config)

        # Try different access patterns
        access_patterns = [
            "layer.analog_module.tile.rpu_config.device.hs_decay",
            "layer.rpu_config.device.hs_decay",
            "layer.analog_module.rpu_config.device.hs_decay"
        ]

        for pattern in access_patterns:
            try:
                value = eval(pattern)
                print(f"   ✓ {pattern}: {value}")
                break
            except (AttributeError, NameError) as e:
                print(f"   ✗ {pattern}: {e}")

    except Exception as e:
        print(f"   ✗ Error creating layer: {e}")

    # Method 4: Print all device attributes
    print("\n4. All device attributes containing 'hs' or 'decay':")
    all_attrs = dir(device)
    relevant_attrs = [attr for attr in all_attrs if 'hs' in attr.lower() or 'decay' in attr.lower()]

    if relevant_attrs:
        for attr in relevant_attrs:
            try:
                value = getattr(device, attr)
                print(f"   {attr}: {value}")
            except Exception as e:
                print(f"   {attr}: Error accessing - {e}")
    else:
        print("   No attributes found with 'hs' or 'decay'")

    print("\n=== Complete device attributes ===")
    for attr in sorted(all_attrs):
        if not attr.startswith('_'):
            try:
                value = getattr(device, attr)
                print(f"   {attr}: {value}")
            except:
                print(f"   {attr}: <unable to access>")

if __name__ == "__main__":
    test_hs_decay_access()