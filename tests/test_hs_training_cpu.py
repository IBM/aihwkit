# -*- coding: utf-8 -*-

# (C) Copyright 2026 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import os
from time import perf_counter
from typing import Any

import torch
from torch import Tensor

from aihwkit.nn import AnalogLinear, AnalogSequential
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import SingleRPUConfig
from aihwkit.simulator.configs.devices import ConstantStepDevice
from aihwkit.simulator.parameters.enums import PulseType


@dataclass
class RunResult:
    name: str
    history: list[dict[str, float]]
    final_loss: float
    final_accuracy: float
    duration_s: float


@contextmanager
def suppress_native_output() -> Any:
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        os.dup2(devnull.fileno(), 1)
        os.dup2(devnull.fileno(), 2)
        try:
            yield
        finally:
            os.dup2(saved_stdout, 1)
            os.dup2(saved_stderr, 2)
            os.close(saved_stdout)
            os.close(saved_stderr)


def make_synthetic_dataset(
    *,
    input_dim: int = 20,
    train_samples: int = 1000,
    test_samples: int = 200,
    seed: int = 11,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    generator = torch.Generator().manual_seed(seed)

    direction = torch.randn(input_dim, generator=generator)
    direction = direction / direction.norm()

    def _sample_block(n_samples: int) -> tuple[Tensor, Tensor]:
        half = n_samples // 2
        noise = 0.55
        margin = 0.9
        class0 = torch.randn(half, input_dim, generator=generator) * noise - margin * direction
        class1 = (
            torch.randn(n_samples - half, input_dim, generator=generator) * noise
            + margin * direction
        )
        features = torch.cat([class0, class1], dim=0)
        labels = torch.cat(
            [
                torch.zeros(half, dtype=torch.long),
                torch.ones(n_samples - half, dtype=torch.long),
            ],
            dim=0,
        )
        permutation = torch.randperm(n_samples, generator=generator)
        return features[permutation], labels[permutation]

    x_train, y_train = _sample_block(train_samples)
    x_test, y_test = _sample_block(test_samples)
    return x_train, y_train, x_test, y_test


def build_rpu_config(use_hs: bool) -> SingleRPUConfig:
    device = ConstantStepDevice()
    if use_hs and hasattr(device, "hs_decay"):
        setattr(device, "hs_decay", 0.99)
    rpu_config = SingleRPUConfig(device=device)
    rpu_config.update.desired_bl = 31
    rpu_config.update.pulse_type = (
        PulseType.HALFSELECTED_STOCHASTIC if use_hs else PulseType.STOCHASTIC_COMPRESSED
    )
    return rpu_config


def build_model(rpu_config: SingleRPUConfig) -> AnalogSequential:
    torch_nn: Any = getattr(torch, "nn")
    analog_sequential: Any = AnalogSequential
    analog_linear: Any = AnalogLinear
    relu_cls: Any = getattr(torch_nn, "ReLU")
    return analog_sequential(
        analog_linear(20, 64, rpu_config=rpu_config),
        relu_cls(),
        analog_linear(64, 32, rpu_config=rpu_config),
        relu_cls(),
        analog_linear(32, 2, rpu_config=rpu_config),
    )


def evaluate(model: Any, x_data: Tensor, y_data: Tensor) -> tuple[float, float]:
    torch_nn: Any = getattr(torch, "nn")
    criterion_cls: Any = getattr(torch_nn, "CrossEntropyLoss")
    criterion = criterion_cls()
    model.eval()
    with torch.no_grad():
        logits = model(x_data)
        loss = criterion(logits, y_data).item()
        predictions = logits.argmax(dim=1)
        accuracy = (predictions == y_data).float().mean().item()
    return loss, accuracy


def train_model(
    *,
    name: str,
    use_hs: bool,
    x_train: Tensor,
    y_train: Tensor,
    x_test: Tensor,
    y_test: Tensor,
    epochs: int = 6,
    batch_size: int = 64,
    lr: float = 0.05,
    model_seed: int = 123,
) -> RunResult:
    torch.manual_seed(model_seed)
    rpu_config = build_rpu_config(use_hs=use_hs)
    model = build_model(rpu_config)
    torch_nn: Any = getattr(torch, "nn")
    criterion_cls: Any = getattr(torch_nn, "CrossEntropyLoss")
    criterion = criterion_cls()
    analog_sgd: Any = AnalogSGD
    optimizer = analog_sgd(model.parameters(), lr=lr)
    optimizer.regroup_param_groups(model)

    run_start = perf_counter()
    history: list[dict[str, float]] = []

    print(f"\n=== {name} ===")
    print(
        "config: pulse_type=%s hs_decay=%.2f desired_bl=%d"
        % (
            rpu_config.update.pulse_type,
            getattr(rpu_config.device, "hs_decay", float("nan")),
            rpu_config.update.desired_bl,
        )
    )

    with suppress_native_output():
        for epoch in range(1, epochs + 1):
            model.train()
            permutation = torch.randperm(x_train.shape[0])
            epoch_loss_total = 0.0

            for start_idx in range(0, x_train.shape[0], batch_size):
                batch_idx = permutation[start_idx : start_idx + batch_size]
                batch_x = x_train[batch_idx]
                batch_y = y_train[batch_idx]

                optimizer.zero_grad()
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss encountered in {name} at epoch {epoch}")

                loss.backward()
                optimizer.step()

                epoch_loss_total += loss.item() * batch_x.shape[0]

            train_loss = epoch_loss_total / x_train.shape[0]
            test_loss, test_accuracy = evaluate(model, x_test, y_test)
            history.append(
                {
                    "epoch": float(epoch),
                    "train_loss": train_loss,
                    "test_loss": test_loss,
                    "test_accuracy": test_accuracy,
                }
            )

    for record in history:
        print(
            "epoch=%02d train_loss=%.4f test_loss=%.4f test_acc=%.2f%%"
            % (
                int(record["epoch"]),
                record["train_loss"],
                record["test_loss"],
                100.0 * record["test_accuracy"],
            )
        )

    duration_s = perf_counter() - run_start
    final_loss = history[-1]["test_loss"]
    final_accuracy = history[-1]["test_accuracy"]
    print(
        "final: test_loss=%.4f test_acc=%.2f%% duration=%.2fs"
        % (final_loss, 100.0 * final_accuracy, duration_s)
    )

    return RunResult(
        name=name,
        history=history,
        final_loss=final_loss,
        final_accuracy=final_accuracy,
        duration_s=duration_s,
    )


def main() -> None:
    torch.set_num_threads(1)
    x_train, y_train, x_test, y_test = make_synthetic_dataset()

    baseline = train_model(
        name="Baseline (Non-HS)",
        use_hs=False,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
    )
    hs = train_model(
        name="Half Select (HS)",
        use_hs=True,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
    )

    acc_delta = hs.final_accuracy - baseline.final_accuracy
    loss_delta = hs.final_loss - baseline.final_loss

    print("\n=== Final Comparison ===")
    print(
        "baseline_acc=%.2f%% hs_acc=%.2f%% delta=%.2f%%"
        % (100.0 * baseline.final_accuracy, 100.0 * hs.final_accuracy, 100.0 * acc_delta)
    )
    print(
        "baseline_loss=%.4f hs_loss=%.4f delta=%.4f"
        % (baseline.final_loss, hs.final_loss, loss_delta)
    )

    min_expected_accuracy = 0.70
    measurable_delta = 0.005
    assert baseline.final_accuracy >= min_expected_accuracy, "Baseline accuracy is too low."
    assert hs.final_accuracy >= min_expected_accuracy, "HS accuracy is too low."
    assert abs(acc_delta) >= measurable_delta or abs(loss_delta) >= 1e-3, (
        "HS effect is not measurable."
    )

    print("PASS: CPU HS training integration test completed successfully.")


if __name__ == "__main__":
    main()
