from typing import Union
import torch
from torch import allclose, randn

from aihwkit.nn import AnalogLinear
from aihwkit.simulator.configs import (
    TorchInferenceRPUConfig,
    InferenceRPUConfig,
    WeightModifierType,
    NoiseManagementType,
    BoundManagementType,
)


def test_weight_modifier(modifier_res: float, wm_type: WeightModifierType):
    """
    Test correctness of discretization.
    """
    # pylint: disable-msg=too-many-locals
    device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_ = torch.device("cpu")

    def populate_rpu(
        rpu_config: Union[TorchInferenceRPUConfig, InferenceRPUConfig],
        modifier_res: float,
        wm_type: WeightModifierType,
    ):
        rpu_config.forward.bound_management = BoundManagementType.NONE
        rpu_config.forward.noise_management = NoiseManagementType.NONE
        rpu_config.noise_model = None
        rpu_config.modifier.type = wm_type
        rpu_config.modifier.res = modifier_res
        rpu_config.modifier.sto_round = False
        rpu_config.forward.inp_res = -1
        rpu_config.forward.out_res = -1
        rpu_config.forward.inp_bound = -1
        rpu_config.forward.out_bound = -1
        rpu_config.forward.out_noise = 0.0
        rpu_config.forward.is_perfect = True
        return rpu_config

    torch.manual_seed(0)
    inp_dim = 256
    out_dim = 255

    tile_weights = torch.randn(inp_dim, out_dim).to(device_)
    inp = randn((1, inp_dim)).to(device_)
    rpu_config_torch = populate_rpu(TorchInferenceRPUConfig(), modifier_res, wm_type)
    rpu_config = populate_rpu(InferenceRPUConfig(), modifier_res, wm_type)

    # One target will be to remove this line
    rpu_config.modifier.res = 2 * (1 / modifier_res if modifier_res > 1.0 else modifier_res)

    linear = AnalogLinear(in_features=inp_dim, out_features=out_dim, bias=False, rpu_config=rpu_config)
    linear_torch = AnalogLinear(
        in_features=inp_dim, out_features=out_dim, bias=False, rpu_config=rpu_config_torch
    )
    linear_torch.set_weights(tile_weights.T)

    # move to device
    linear = linear.to(device_)
    linear_torch = linear_torch.to(device_)

    # load the state dict
    linear.load_state_dict(linear_torch.state_dict(), load_rpu_config=False)

    # post update step generates the weight modification, needed for rpu tile
    linear.analog_module.post_update_step()
    assumed_wmax = rpu_config_torch.modifier.assumed_wmax
    if rpu_config_torch.modifier.rel_to_actual_wmax:
        assumed_wmax = (
            tile_weights.abs().max()
            if wm_type == WeightModifierType.DISCRETIZE
            else tile_weights.abs().amax(0)
        )

    n_states = rpu_config_torch.modifier.res
    n_states = n_states if n_states > 1.0 else 1 / n_states
    res = 2 * (1 / n_states) * assumed_wmax
    quantized_weights = (tile_weights / res).round() * res
    cpp_quantized_weights = tile_weights / res
    cpp_quantized_weights = (
        torch.trunc(cpp_quantized_weights + 0.5 * torch.sign(cpp_quantized_weights)) * res
    )
    # Test if quantization is as expected.
    # pylint: disable=not-callable
    assert allclose(linear_torch(inp), torch.matmul(inp, quantized_weights), atol=1e-4)

    # test if C++ tile is the same
    cpp_out = linear(inp)
    quantized_groundtruth = torch.matmul(inp, cpp_quantized_weights)
    assert allclose(cpp_out, quantized_groundtruth, atol=1e-4)


if __name__ == "__main__":
    test_weight_modifier(2**8 - 2, WeightModifierType.DISCRETIZE_PER_CHANNEL)
    # test_weight_modifier(2**8 - 2, WeightModifierType.DISCRETIZE)