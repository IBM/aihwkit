#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
aihwkit 펄스 기반 Weight Update 데모
C++ bit line maker 로직을 Python으로 구현한 MNIST 학습 예제
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from typing import Tuple, List

class PulsedWeightUpdater:
    """
    aihwkit의 C++ bit line maker 로직을 Python으로 구현한 클래스
    gradient를 확률적 펄스로 변환하고 weight를 업데이트
    """

    def __init__(self, desired_bl: int = 10, dw_min: float = 0.001):
        self.desired_bl = desired_bl
        self.dw_min = dw_min
        self.pulse_stats = {
            'total_pulses': 0,
            'zero_pulses': 0,
            'pulse_history': []
        }

    def calculate_bl_ab(self, lr: float) -> Tuple[int, float, float]:
        """
        C++ calculateBlAB 함수 구현
        learning rate와 dw_min에 따라 BL, A, B 계산
        """
        if lr <= 0:
            return 0, 0.0, 0.0

        BL = self.desired_bl
        A = np.sqrt(lr / (self.dw_min * BL))
        B = A

        return BL, A, B

    def generate_stochastic_pulses(self, values: np.ndarray, scale_factor: float,
                                 bl: int) -> Tuple[np.ndarray, dict]:
        """
        C++ generateCounts 함수 구현
        입력 값들을 확률적 펄스로 변환
        """
        n_values = len(values)
        pulses = np.zeros((bl, n_values), dtype=np.int32)
        counts = np.zeros(bl, dtype=np.int32)
        noz_count = 0

        for j, value in enumerate(values):
            # 확률 계산 (PP)
            pp = abs(value) * scale_factor
            pp = min(pp, 1.0)  # 확률은 1을 초과할 수 없음

            if pp == 0.0:
                noz_count += 1
                continue

            # 부호 결정
            sign = 1 if value > 0 else -1
            jplus1_signed = (j + 1) * sign

            # 각 bit line에 대해 확률적으로 펄스 생성
            for k in range(bl):
                if pp > np.random.uniform():
                    pulses[k][counts[k]] = jplus1_signed
                    counts[k] += 1

        stats = {
            'noz_count': noz_count,
            'total_values': n_values,
            'avg_pulses_per_bl': np.mean(counts)
        }

        return pulses, counts, stats

    def update_weights_with_pulses(self, weights: torch.Tensor,
                                 x_grad: torch.Tensor, d_grad: torch.Tensor,
                                 lr: float) -> torch.Tensor:
        """
        펄스 기반 weight 업데이트 수행
        """
        # BL, A, B 계산
        bl, A, B = self.calculate_bl_ab(lr)

        if bl == 0:
            return weights

        # numpy로 변환
        x_vals = x_grad.detach().cpu().numpy()
        d_vals = d_grad.detach().cpu().numpy()

        # 펄스 생성
        x_pulses, x_counts, x_stats = self.generate_stochastic_pulses(x_vals, B, bl)
        d_pulses, d_counts, d_stats = self.generate_stochastic_pulses(d_vals, A, bl)

        # weight 업데이트 계산
        dw_total = np.zeros_like(weights.detach().cpu().numpy())

        for k in range(bl):
            for i in range(x_counts[k]):
                x_idx = abs(x_pulses[k][i]) - 1
                x_sign = 1 if x_pulses[k][i] > 0 else -1

                for j in range(d_counts[k]):
                    d_idx = abs(d_pulses[k][j]) - 1
                    d_sign = 1 if d_pulses[k][j] > 0 else -1

                    dw = x_sign * d_sign * self.dw_min
                    dw_total[d_idx, x_idx] += dw

        # 통계 업데이트
        self.pulse_stats['total_pulses'] += np.sum(x_counts) + np.sum(d_counts)
        self.pulse_stats['zero_pulses'] += x_stats['noz_count'] + d_stats['noz_count']
        self.pulse_stats['pulse_history'].append({
            'x_avg_pulses': x_stats['avg_pulses_per_bl'],
            'd_avg_pulses': d_stats['avg_pulses_per_bl'],
            'bl_used': bl
        })

        # torch tensor로 변환하여 업데이트
        dw_tensor = torch.tensor(dw_total, dtype=weights.dtype, device=weights.device)
        return weights + dw_tensor


class CustomAnalogLinear(nn.Module):
    """
    펄스 기반 업데이트를 사용하는 커스텀 Linear 레이어
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 desired_bl: int = 10, dw_min: float = 0.001):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # 표준 Linear layer 파라미터
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

        # 펄스 업데이터
        self.pulse_updater = PulsedWeightUpdater(desired_bl, dw_min)
        self.last_input = None
        self.last_grad_output = None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # forward pass는 일반 Linear와 동일
        self.last_input = input.detach()
        return torch.nn.functional.linear(input, self.weight, self.bias)

    def pulse_backward(self, lr: float, debug: bool = False):
        """
        펄스 기반 backward pass
        forward pass에서 저장된 input과 weight gradient 사용
        """
        if self.last_input is None or not hasattr(self.weight, 'grad') or self.weight.grad is None:
            return

        # input activations (x)와 weight gradient에서 output gradient (d) 추출
        x_values = self.last_input.mean(dim=0)  # batch 평균

        # weight gradient = x^T @ d 형태이므로, d를 추출하려면
        # 간단한 경우: gradient 자체를 d로 사용 (각 출력 뉴런별)
        d_values = self.weight.grad.mean(dim=1)  # output dimension 평균

        if debug:
            grad_norm = torch.norm(self.weight.grad.data).item()
            weight_norm_before = torch.norm(self.weight.data).item()
            x_norm = torch.norm(x_values).item()
            d_norm = torch.norm(d_values).item()

        # 펄스 기반 weight 업데이트
        with torch.no_grad():
            old_weight = self.weight.data.clone()
            self.weight.data = self.pulse_updater.update_weights_with_pulses(
                self.weight.data, x_values, d_values, lr
            )

        if debug:
            weight_norm_after = torch.norm(self.weight.data).item()
            weight_change_norm = torch.norm(self.weight.data - old_weight).item()
            print(f"    Pulse update - X norm: {x_norm:.6f}, D norm: {d_norm:.6f}, "
                  f"Weight norm: {weight_norm_before:.6f} -> {weight_norm_after:.6f}, "
                  f"Change: {weight_change_norm:.6f}")


class PulseBasedMNISTNet(nn.Module):
    """
    3-layer FC 네트워크 (784 → 256 → 128 → 10) - 펄스 기반 업데이트
    """

    def __init__(self, desired_bl: int = 10, dw_min: float = 0.001):
        super().__init__()

        self.layer1 = CustomAnalogLinear(784, 256, bias=False,
                                       desired_bl=desired_bl, dw_min=dw_min)
        self.sigmoid1 = nn.Sigmoid()

        self.layer2 = CustomAnalogLinear(256, 128, bias=False,
                                       desired_bl=desired_bl, dw_min=dw_min)
        self.sigmoid2 = nn.Sigmoid()

        self.layer3 = CustomAnalogLinear(128, 10, bias=False,
                                       desired_bl=desired_bl, dw_min=dw_min)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = self.sigmoid1(self.layer1(x))
        x = self.sigmoid2(self.layer2(x))
        x = self.log_softmax(self.layer3(x))
        return x

    def get_pulse_stats(self):
        """모든 레이어의 펄스 통계 수집"""
        return {
            'layer1': self.layer1.pulse_updater.pulse_stats,
            'layer2': self.layer2.pulse_updater.pulse_stats,
            'layer3': self.layer3.pulse_updater.pulse_stats
        }


class ManualLinear(nn.Module):
    """
    Manual backward가 가능한 Linear 레이어
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

        self.last_input = None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self.last_input = input.detach()
        return torch.nn.functional.linear(input, self.weight, self.bias)

    def manual_backward(self, lr: float, debug: bool = False, use_xd_direct: bool = False):
        """
        표준 gradient descent 업데이트

        Args:
            use_xd_direct: True이면 x,d로부터 직접 weight 업데이트, False면 기존 gradient 사용
        """
        if self.last_input is None or not hasattr(self.weight, 'grad') or self.weight.grad is None:
            return

        if debug:
            original_grad_norm = torch.norm(self.weight.grad.data).item()
            weight_norm_before = torch.norm(self.weight.data).item()

        with torch.no_grad():
            if use_xd_direct:
                # 주의: 이 방법은 정확하지 않음 - d 값을 올바르게 얻을 수 없음
                # 단지 비교를 위해 구현
                x_values = self.last_input.mean(dim=0)
                d_values = self.weight.grad.mean(dim=1)
                dw = lr * torch.outer(d_values, x_values)
                self.weight.data -= dw

                if debug:
                    dw_norm = torch.norm(dw).item()
                    x_norm = torch.norm(x_values).item()
                    d_norm = torch.norm(d_values).item()
                    print(f"    X,D direct update - X norm: {x_norm:.6f}, D norm: {d_norm:.6f}, dW norm: {dw_norm:.6f}")

                    # gradient 재구성 비교
                    reconstructed_grad = torch.outer(d_values, x_values)
                    grad_diff = torch.norm(self.weight.grad.data - reconstructed_grad).item()
                    print(f"    Grad reconstruction diff: {grad_diff:.6f} (expected to be large)")
            else:
                # 정확한 표준 gradient descent
                self.weight.data -= lr * self.weight.grad.data

        if debug:
            weight_norm_after = torch.norm(self.weight.data).item()
            print(f"    Original grad norm: {original_grad_norm:.6f}")
            print(f"    Weight norm: {weight_norm_before:.6f} -> {weight_norm_after:.6f}")


class StandardMNISTNet(nn.Module):
    """
    3-layer FC 네트워크 (784 → 256 → 128 → 10) - Manual 표준 백프롭
    """

    def __init__(self):
        super().__init__()

        self.layer1 = ManualLinear(784, 256, bias=False)
        self.sigmoid1 = nn.Sigmoid()

        self.layer2 = ManualLinear(256, 128, bias=False)
        self.sigmoid2 = nn.Sigmoid()

        self.layer3 = ManualLinear(128, 10, bias=False)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = self.sigmoid1(self.layer1(x))
        x = self.sigmoid2(self.layer2(x))
        x = self.log_softmax(self.layer3(x))
        return x

    def manual_update(self, lr: float, debug: bool = False, use_xd_direct: bool = False):
        """모든 레이어에 manual gradient descent 적용"""
        self.layer3.manual_backward(lr, debug, use_xd_direct)
        self.layer2.manual_backward(lr, debug, use_xd_direct)
        self.layer1.manual_backward(lr, debug, use_xd_direct)


class PurePyTorchMNISTNet(nn.Module):
    """
    3-layer FC 네트워크 (784 → 256 → 128 → 10) - 순수 PyTorch SGD
    """

    def __init__(self):
        super().__init__()

        self.layer1 = nn.Linear(784, 256, bias=False)
        self.sigmoid1 = nn.Sigmoid()

        self.layer2 = nn.Linear(256, 128, bias=False)
        self.sigmoid2 = nn.Sigmoid()

        self.layer3 = nn.Linear(128, 10, bias=False)
        self.log_softmax = nn.LogSoftmax(dim=1)

        # weight 초기화를 다른 모델들과 동일하게
        with torch.no_grad():
            self.layer1.weight.data = torch.randn_like(self.layer1.weight) * 0.1
            self.layer2.weight.data = torch.randn_like(self.layer2.weight) * 0.1
            self.layer3.weight.data = torch.randn_like(self.layer3.weight) * 0.1

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = self.sigmoid1(self.layer1(x))
        x = self.sigmoid2(self.layer2(x))
        x = self.log_softmax(self.layer3(x))
        return x


def load_mnist_data(batch_size: int = 64) -> Tuple[DataLoader, DataLoader]:
    """MNIST 데이터 로드"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def train_pulse_model(model: PulseBasedMNISTNet, train_loader: DataLoader,
                     epochs: int = 10, lr: float = 0.05) -> List[float]:
    """펄스 기반 모델 학습"""
    criterion = nn.NLLLoss()
    losses = []

    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            # Forward pass와 동시에 intermediate outputs 저장
            x1 = model.layer1(data.view(data.size(0), -1))
            a1 = model.sigmoid1(x1)

            x2 = model.layer2(a1)
            a2 = model.sigmoid2(x2)

            x3 = model.layer3(a2)
            output = model.log_softmax(x3)

            loss = criterion(output, target)

            # Backward pass (manual gradient 계산)
            loss.backward()

            # 각 레이어에 pulse update 적용 (첫 번째 배치에서는 디버깅 출력)
            debug_mode = (epoch == 0 and batch_idx == 0)
            if debug_mode:
                print("  [DEBUG] Pulse-based updates:")
                print("    Layer 3:")
            model.layer3.pulse_backward(lr, debug=debug_mode)
            if debug_mode:
                print("    Layer 2:")
            model.layer2.pulse_backward(lr, debug=debug_mode)
            if debug_mode:
                print("    Layer 1:")
            model.layer1.pulse_backward(lr, debug=debug_mode)

            # gradient 초기화
            model.zero_grad()

            total_loss += loss.item()

            if batch_idx % 200 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}')

        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)
        print(f'Epoch {epoch} - Average Loss: {avg_loss:.6f}')

    return losses


def train_standard_model(model: StandardMNISTNet, train_loader: DataLoader,
                        epochs: int = 10, lr: float = 0.05, use_xd_direct: bool = False) -> List[float]:
    """Manual 표준 백프롭 기반 모델 학습 (펄스 기반과 동일한 방식)"""
    criterion = nn.NLLLoss()
    losses = []

    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            # Forward pass와 동시에 intermediate outputs 저장
            x1 = model.layer1(data.view(data.size(0), -1))
            a1 = model.sigmoid1(x1)

            x2 = model.layer2(a1)
            a2 = model.sigmoid2(x2)

            x3 = model.layer3(a2)
            output = model.log_softmax(x3)

            loss = criterion(output, target)

            # Backward pass (manual gradient 계산)
            loss.backward()

            # 각 레이어에 manual standard update 적용 (첫 번째 배치에서는 디버깅 출력)
            debug_mode = (epoch == 0 and batch_idx == 0)
            if debug_mode:
                update_type = "x,d direct" if use_xd_direct else "gradient"
                print(f"  [DEBUG] Standard backprop updates ({update_type}):")
            model.manual_update(lr, debug=debug_mode, use_xd_direct=use_xd_direct)

            # gradient 초기화
            model.zero_grad()

            total_loss += loss.item()

            if batch_idx % 200 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}')

        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)
        print(f'Epoch {epoch} - Average Loss: {avg_loss:.6f}')

    return losses


def train_pure_pytorch_model(model: PurePyTorchMNISTNet, train_loader: DataLoader,
                            epochs: int = 10, lr: float = 0.05) -> List[float]:
    """순수 PyTorch SGD 기반 모델 학습 (baseline)"""
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    losses = []

    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()

            # Forward pass
            output = model(data)
            loss = criterion(output, target)

            # Backward pass
            loss.backward()

            # Optimizer step
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 200 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}')

        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)
        print(f'Epoch {epoch} - Average Loss: {avg_loss:.6f}')

    return losses


def evaluate_model(model, test_loader: DataLoader) -> float:
    """모델 평가"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

    accuracy = correct / total
    print(f'Test Accuracy: {accuracy:.4f}')
    return accuracy


def visualize_three_way_comparison(pulse_losses: List[float], standard_losses: List[float],
                                 pytorch_losses: List[float], pulse_accuracy: float,
                                 standard_accuracy: float, pytorch_accuracy: float,
                                 pulse_model: PulseBasedMNISTNet):
    """3가지 방법의 결과 비교 시각화"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. Loss 비교
    epochs = range(len(pulse_losses))
    axes[0, 0].plot(epochs, pulse_losses, label='Pulse-based', marker='o', linewidth=2)
    axes[0, 0].plot(epochs, standard_losses, label='Manual Standard', marker='s', linewidth=2)
    axes[0, 0].plot(epochs, pytorch_losses, label='Pure PyTorch', marker='^', linewidth=2)
    axes[0, 0].set_title('Training Loss Comparison')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # 2. 정확도 비교
    methods = ['Pulse-based', 'Manual Standard', 'Pure PyTorch']
    accuracies = [pulse_accuracy, standard_accuracy, pytorch_accuracy]
    colors = ['orange', 'blue', 'green']
    bars = axes[0, 1].bar(methods, accuracies, color=colors)
    axes[0, 1].set_title('Test Accuracy Comparison')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].tick_params(axis='x', rotation=45)
    for bar, acc in zip(bars, accuracies):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{acc:.3f}', ha='center', va='bottom')

    # 3. 펄스 통계
    stats = pulse_model.get_pulse_stats()
    layers = ['layer1', 'layer2', 'layer3']
    total_pulses = [stats[layer]['total_pulses'] for layer in layers]
    axes[0, 2].bar(layers, total_pulses, color='purple')
    axes[0, 2].set_title('Total Pulses per Layer (Pulse-based)')
    axes[0, 2].set_ylabel('Pulse Count')

    # 4. 정확도 차이 (PyTorch 기준)
    acc_diffs = [
        abs(pulse_accuracy - pytorch_accuracy),
        abs(standard_accuracy - pytorch_accuracy),
        0  # PyTorch 자체와의 차이는 0
    ]
    diff_methods = ['Pulse vs PyTorch', 'Manual vs PyTorch', 'PyTorch vs PyTorch']
    axes[1, 0].bar(diff_methods, acc_diffs, color=['red', 'orange', 'gray'])
    axes[1, 0].set_title('Accuracy Difference from Pure PyTorch')
    axes[1, 0].set_ylabel('Absolute Difference')
    axes[1, 0].tick_params(axis='x', rotation=45)

    # 5. Loss 차이 (PyTorch 기준)
    loss_diffs = [
        abs(pulse_losses[-1] - pytorch_losses[-1]),
        abs(standard_losses[-1] - pytorch_losses[-1]),
        0  # PyTorch 자체와의 차이는 0
    ]
    axes[1, 1].bar(diff_methods, loss_diffs, color=['red', 'orange', 'gray'])
    axes[1, 1].set_title('Final Loss Difference from Pure PyTorch')
    axes[1, 1].set_ylabel('Absolute Difference')
    axes[1, 1].tick_params(axis='x', rotation=45)

    # 6. Zero 펄스 비율 (펄스 기반만)
    zero_ratios = [stats[layer]['zero_pulses'] / max(stats[layer]['total_pulses'], 1)
                   for layer in layers]
    axes[1, 2].bar(layers, zero_ratios, color='cyan')
    axes[1, 2].set_title('Zero Pulse Ratio per Layer')
    axes[1, 2].set_ylabel('Ratio')

    plt.tight_layout()
    plt.savefig('three_way_comparison.png', dpi=150)
    plt.show()


if __name__ == "__main__":
    # 파라미터 설정
    DESIRED_BL = 10
    DW_MIN = 0.001
    LEARNING_RATE = 0.05
    EPOCHS = 1
    BATCH_SIZE = 64

    print("=== aihwkit 펄스 기반 vs 표준 백프롭 비교 데모 ===")
    print(f"Parameters: BL={DESIRED_BL}, dw_min={DW_MIN}, lr={LEARNING_RATE}, epochs={EPOCHS}")

    # 데이터 로드
    print("\nLoading MNIST dataset...")
    train_loader, test_loader = load_mnist_data(BATCH_SIZE)

    # =============== 펄스 기반 모델 ===============
    print("\n[1] Creating and training pulse-based model...")
    pulse_model = PulseBasedMNISTNet(desired_bl=DESIRED_BL, dw_min=DW_MIN)
    print(f"Pulse model parameters: {sum(p.numel() for p in pulse_model.parameters())}")

    pulse_losses = train_pulse_model(pulse_model, train_loader, EPOCHS, LEARNING_RATE)
    pulse_accuracy = evaluate_model(pulse_model, test_loader)

    # =============== 표준 백프롭 모델 (Manual) ===============
    print("\n[2] Creating and training manual standard backprop model...")
    standard_model = StandardMNISTNet()
    print(f"Manual standard model parameters: {sum(p.numel() for p in standard_model.parameters())}")

    # x,d 직접 사용 여부 설정 (False로 설정하여 정확한 gradient 사용)
    USE_XD_DIRECT = True

    standard_losses = train_standard_model(standard_model, train_loader, EPOCHS, LEARNING_RATE, use_xd_direct=USE_XD_DIRECT)
    standard_accuracy = evaluate_model(standard_model, test_loader)

    # =============== 순수 PyTorch SGD 모델 ===============
    print("\n[3] Creating and training pure PyTorch SGD model...")
    pytorch_model = PurePyTorchMNISTNet()
    print(f"Pure PyTorch model parameters: {sum(p.numel() for p in pytorch_model.parameters())}")

    pytorch_losses = train_pure_pytorch_model(pytorch_model, train_loader, EPOCHS, LEARNING_RATE)
    pytorch_accuracy = evaluate_model(pytorch_model, test_loader)

    # =============== 결과 비교 ===============
    print("\n" + "="*60)
    print("FINAL COMPARISON RESULTS (3 Methods)")
    print("="*60)
    print(f"[1] Pulse-based Model:")
    print(f"  - Test Accuracy: {pulse_accuracy:.4f}")
    print(f"  - Final Loss: {pulse_losses[-1]:.6f}")

    print(f"\n[2] Manual Standard Backprop Model:")
    print(f"  - Test Accuracy: {standard_accuracy:.4f}")
    print(f"  - Final Loss: {standard_losses[-1]:.6f}")

    print(f"\n[3] Pure PyTorch SGD Model (Baseline):")
    print(f"  - Test Accuracy: {pytorch_accuracy:.4f}")
    print(f"  - Final Loss: {pytorch_losses[-1]:.6f}")

    print(f"\nAccuracy Comparisons:")
    print(f"  - Pulse vs Manual Standard: {abs(pulse_accuracy - standard_accuracy):.4f}")
    print(f"  - Pulse vs Pure PyTorch: {abs(pulse_accuracy - pytorch_accuracy):.4f}")
    print(f"  - Manual Standard vs Pure PyTorch: {abs(standard_accuracy - pytorch_accuracy):.4f}")

    print(f"\nLoss Comparisons:")
    print(f"  - Pulse vs Manual Standard: {abs(pulse_losses[-1] - standard_losses[-1]):.6f}")
    print(f"  - Pulse vs Pure PyTorch: {abs(pulse_losses[-1] - pytorch_losses[-1]):.6f}")
    print(f"  - Manual Standard vs Pure PyTorch: {abs(standard_losses[-1] - pytorch_losses[-1]):.6f}")

    # 펄스 통계
    pulse_stats = pulse_model.get_pulse_stats()
    total_pulses = sum(pulse_stats[layer]['total_pulses'] for layer in ['layer1', 'layer2', 'layer3'])
    print(f"\nPulse Statistics:")
    print(f"  - Total pulses generated: {total_pulses}")
    print(f"  - Average pulses per layer: {total_pulses / 3:.0f}")

    # 결과 시각화
    print("\nGenerating comparison visualization...")
    visualize_three_way_comparison(pulse_losses, standard_losses, pytorch_losses,
                                 pulse_accuracy, standard_accuracy, pytorch_accuracy, pulse_model)
    print("Comparison results saved to 'three_way_comparison.png'")