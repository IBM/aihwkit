from torch import manual_seed, rand, mean, tensor, allclose
from torch.nn import Linear

from aihwkit.optim.analog_optimizer import AnalogSGD
from aihwkit.simulator.configs import TorchInferenceRPUConfig
from aihwkit.simulator.configs.utils import (
    BoundManagementType,
    NoiseManagementType,
)
from aihwkit.nn.conversion import convert_to_analog


def train_linear_regression(reload: bool):
    """Train a linear regression model and return the losses."""
    def generate_toy_data(num_samples=100):
        manual_seed(0)
        X = 2 * rand(num_samples, 1)
        y = 4 + 3 * X + rand(num_samples, 1)
        return X, y

    def mean_squared_error(y_true, y_pred):
        return mean((y_true - y_pred) ** 2)

    manual_seed(0)
    num_epochs = 1000
    learning_rate = 0.001
    X, y = generate_toy_data()
    model = Linear(1, 1)

    rpu_config = TorchInferenceRPUConfig()
    rpu_config.forward.bound_management = BoundManagementType.NONE
    rpu_config.forward.noise_management = NoiseManagementType.NONE
    rpu_config.forward.out_noise = 0.0
    rpu_config.pre_post.input_range.enable = True
    rpu_config.pre_post.input_range.init_value = 3.0
    rpu_config.forward.is_perfect = True
    rpu_config.pre_post.input_range.enable = True
    rpu_config.pre_post.input_range.init_from_data = 1000
    rpu_config.pre_post.input_range.learn_input_range = False
    rpu_config.pre_post.input_range.decay = 0.0

    model = convert_to_analog(model, rpu_config)
    optimizer = AnalogSGD(params=model.parameters(), lr=learning_rate)

    losses = []
    for epoch in range(num_epochs):
        predictions = model.forward(X)
        loss = mean_squared_error(y, predictions)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if epoch == 500 and reload:
            sd = model.state_dict()
            optimizer_sd = optimizer.state_dict()
            model = Linear(1, 1)
            model = convert_to_analog(model, rpu_config)
            optimizer = AnalogSGD(params=model.parameters(), lr=learning_rate)
            model.load_state_dict(sd)
            optimizer.load_state_dict(optimizer_sd)
    return tensor(losses)


def test_continue_training():
    """Test if continuing to train still works."""
    losses_false = train_linear_regression(reload=False)
    losses_true = train_linear_regression(reload=True)
    assert allclose(losses_false, losses_true, atol=1e-4)