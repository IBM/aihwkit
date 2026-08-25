#!/usr/bin/env python3
"""
STOCHASTIC_STREAM vs STOCHASTIC_COMPRESSED 상세 비교 분석

MNIST 학습에서 STOCHASTIC_STREAM이 학습이 되지 않는 문제를 분석하기 위한 스크립트
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import copy

from aihwkit.nn import AnalogLinear
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import SingleRPUConfig, ConstantStepDevice
from aihwkit.simulator.parameters import PulseType


def create_test_model(pulse_type):
    """테스트용 단순한 선형 모델 생성"""
    rpu_config = SingleRPUConfig(device=ConstantStepDevice())
    rpu_config.update.pulse_type = pulse_type
    rpu_config.update.desired_bl = 31  # MNIST 예제와 동일
    rpu_config.update.update_bl_management = True
    rpu_config.update.update_management = True

    # 작은 네트워크로 테스트
    model = AnalogLinear(784, 10, bias=True, rpu_config=rpu_config)

    return model


def load_mnist_batch(batch_size=64, shuffle=False, batch_idx=0):
    """MNIST 데이터에서 특정 배치 로드"""
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST('data', download=True, train=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    for i, (images, labels) in enumerate(dataloader):
        if i == batch_idx:
            return images.view(images.shape[0], -1), labels  # Flatten


def analyze_single_update():
    """단일 업데이트에서 상세 분석"""
    print("=== 단일 업데이트 상세 분석 ===")

    # 동일한 데이터로 테스트
    images, labels = load_mnist_batch(batch_size=64, shuffle=False, batch_idx=0)

    # 동일한 초기 가중치로 시작
    torch.manual_seed(42)
    model_stream = create_test_model(PulseType.STOCHASTIC_STREAM)

    torch.manual_seed(42)
    model_compressed = create_test_model(PulseType.STOCHASTIC_COMPRESSED)

    # 초기 가중치 확인
    w_stream_init = model_stream.get_weights()[0].clone()  # 첫 번째는 weight, 두 번째는 bias
    w_compressed_init = model_compressed.get_weights()[0].clone()
    weight_diff = torch.norm(w_stream_init - w_compressed_init).item()
    print(f"초기 가중치 차이: {weight_diff:.10f}")

    if weight_diff > 1e-6:
        print("경고: 초기 가중치가 다릅니다!")
        return

    # 옵티마이저 설정
    optimizer_stream = AnalogSGD(model_stream.parameters(), lr=0.01)
    optimizer_compressed = AnalogSGD(model_compressed.parameters(), lr=0.01)

    criterion = nn.CrossEntropyLoss()

    # STOCHASTIC_STREAM 테스트
    print("\n--- STOCHASTIC_STREAM ---")
    w_before_stream = model_stream.get_weights()[0].clone()

    optimizer_stream.zero_grad()
    output_stream = model_stream(images)
    loss_stream = criterion(output_stream, labels)
    print(f"Loss: {loss_stream.item():.8f}")

    loss_stream.backward()

    # 그래디언트 확인 (analog module의 파라미터들 확인)
    total_grad_norm = 0
    grad_count = 0
    for param in model_stream.parameters():
        if param.grad is not None:
            total_grad_norm += torch.norm(param.grad).item() ** 2
            grad_count += 1

    if grad_count > 0:
        grad_norm_stream = total_grad_norm ** 0.5
        print(f"Gradient norm: {grad_norm_stream:.8f}")
    else:
        grad_norm_stream = 0
        print("No gradients found!")

    optimizer_stream.step()

    w_after_stream = model_stream.get_weights()[0].clone()
    weight_change_stream = torch.norm(w_after_stream - w_before_stream).item()
    print(f"Weight change: {weight_change_stream:.10f}")

    # STOCHASTIC_COMPRESSED 테스트
    print("\n--- STOCHASTIC_COMPRESSED ---")
    w_before_compressed = model_compressed.get_weights()[0].clone()

    optimizer_compressed.zero_grad()
    output_compressed = model_compressed(images)
    loss_compressed = criterion(output_compressed, labels)
    print(f"Loss: {loss_compressed.item():.8f}")

    loss_compressed.backward()

    # 그래디언트 확인 (analog module의 파라미터들 확인)
    total_grad_norm = 0
    grad_count = 0
    for param in model_compressed.parameters():
        if param.grad is not None:
            total_grad_norm += torch.norm(param.grad).item() ** 2
            grad_count += 1

    if grad_count > 0:
        grad_norm_compressed = total_grad_norm ** 0.5
        print(f"Gradient norm: {grad_norm_compressed:.8f}")
    else:
        grad_norm_compressed = 0
        print("No gradients found!")

    optimizer_compressed.step()

    w_after_compressed = model_compressed.get_weights()[0].clone()
    weight_change_compressed = torch.norm(w_after_compressed - w_before_compressed).item()
    print(f"Weight change: {weight_change_compressed:.10f}")

    # 결과 비교
    print(f"\n=== 비교 결과 ===")
    print(f"Loss 차이: {abs(loss_stream.item() - loss_compressed.item()):.10f}")
    print(f"Gradient norm 비율: {grad_norm_stream/grad_norm_compressed:.10f}")
    print(f"Weight change - STREAM: {weight_change_stream:.10f}")
    print(f"Weight change - COMPRESSED: {weight_change_compressed:.10f}")

    if weight_change_compressed > 0:
        ratio = weight_change_stream / weight_change_compressed
        print(f"Weight change 비율 (STREAM/COMPRESSED): {ratio:.10f}")

        if ratio < 0.01:
            print("⚠️  STOCHASTIC_STREAM의 가중치 변화가 매우 작습니다!")
        elif ratio < 0.1:
            print("⚠️  STOCHASTIC_STREAM의 가중치 변화가 작습니다")
        else:
            print("✓ 가중치 변화가 정상적입니다")

    return {
        'stream_change': weight_change_stream,
        'compressed_change': weight_change_compressed,
        'stream_loss': loss_stream.item(),
        'compressed_loss': loss_compressed.item(),
        'grad_norm_stream': grad_norm_stream,
        'grad_norm_compressed': grad_norm_compressed
    }


def analyze_multiple_updates(num_updates=10):
    """여러 업데이트에서 누적 효과 분석"""
    print(f"\n=== {num_updates}회 연속 업데이트 분석 ===")

    # 동일한 초기 조건 설정
    torch.manual_seed(42)
    model_stream = create_test_model(PulseType.STOCHASTIC_STREAM)

    torch.manual_seed(42)
    model_compressed = create_test_model(PulseType.STOCHASTIC_COMPRESSED)

    optimizer_stream = AnalogSGD(model_stream.parameters(), lr=0.01)
    optimizer_compressed = AnalogSGD(model_compressed.parameters(), lr=0.01)

    criterion = nn.CrossEntropyLoss()

    # 결과 저장용
    results = {
        'losses_stream': [],
        'losses_compressed': [],
        'weight_changes_stream': [],
        'weight_changes_compressed': [],
        'cumulative_change_stream': 0,
        'cumulative_change_compressed': 0
    }

    # 초기 가중치 저장
    w_init_stream = model_stream.get_weights()[0].clone()
    w_init_compressed = model_compressed.get_weights()[0].clone()

    for update_idx in range(num_updates):
        # 배치 로드 (매번 다른 배치)
        images, labels = load_mnist_batch(batch_size=64, shuffle=False, batch_idx=update_idx)

        # STOCHASTIC_STREAM 업데이트
        w_before_stream = model_stream.get_weights()[0].clone()

        optimizer_stream.zero_grad()
        output_stream = model_stream(images)
        loss_stream = criterion(output_stream, labels)
        loss_stream.backward()
        optimizer_stream.step()

        w_after_stream = model_stream.get_weights()[0].clone()
        change_stream = torch.norm(w_after_stream - w_before_stream).item()

        # STOCHASTIC_COMPRESSED 업데이트
        w_before_compressed = model_compressed.get_weights()[0].clone()

        optimizer_compressed.zero_grad()
        output_compressed = model_compressed(images)
        loss_compressed = criterion(output_compressed, labels)
        loss_compressed.backward()
        optimizer_compressed.step()

        w_after_compressed = model_compressed.get_weights()[0].clone()
        change_compressed = torch.norm(w_after_compressed - w_before_compressed).item()

        # 결과 기록
        results['losses_stream'].append(loss_stream.item())
        results['losses_compressed'].append(loss_compressed.item())
        results['weight_changes_stream'].append(change_stream)
        results['weight_changes_compressed'].append(change_compressed)

        print(f"Update {update_idx+1}: STREAM loss={loss_stream.item():.6f}, change={change_stream:.8f}")
        print(f"Update {update_idx+1}: COMPRESSED loss={loss_compressed.item():.6f}, change={change_compressed:.8f}")

        ratio = change_stream / change_compressed if change_compressed > 0 else float('inf')
        print(f"Update {update_idx+1}: Ratio = {ratio:.8f}")
        print()

    # 누적 변화량 계산
    total_change_stream = torch.norm(model_stream.get_weights()[0] - w_init_stream).item()
    total_change_compressed = torch.norm(model_compressed.get_weights()[0] - w_init_compressed).item()

    results['cumulative_change_stream'] = total_change_stream
    results['cumulative_change_compressed'] = total_change_compressed

    print(f"=== 누적 결과 ({num_updates} 업데이트) ===")
    print(f"STREAM 총 가중치 변화: {total_change_stream:.8f}")
    print(f"COMPRESSED 총 가중치 변화: {total_change_compressed:.8f}")
    print(f"누적 비율: {total_change_stream/total_change_compressed:.8f}")
    print(f"STREAM 최종 손실: {results['losses_stream'][-1]:.6f}")
    print(f"COMPRESSED 최종 손실: {results['losses_compressed'][-1]:.6f}")

    return results


def visualize_comparison(results):
    """결과 시각화"""
    plt.figure(figsize=(15, 10))

    num_updates = len(results['losses_stream'])
    updates = range(1, num_updates + 1)

    # 손실 함수 비교
    plt.subplot(2, 3, 1)
    plt.plot(updates, results['losses_stream'], 'r-o', label='STOCHASTIC_STREAM', markersize=4)
    plt.plot(updates, results['losses_compressed'], 'b-o', label='STOCHASTIC_COMPRESSED', markersize=4)
    plt.xlabel('Update')
    plt.ylabel('Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.grid(True)

    # 가중치 변화량 비교
    plt.subplot(2, 3, 2)
    plt.plot(updates, results['weight_changes_stream'], 'r-o', label='STOCHASTIC_STREAM', markersize=4)
    plt.plot(updates, results['weight_changes_compressed'], 'b-o', label='STOCHASTIC_COMPRESSED', markersize=4)
    plt.xlabel('Update')
    plt.ylabel('Weight Change Magnitude')
    plt.title('Per-Update Weight Changes')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')

    # 누적 가중치 변화
    plt.subplot(2, 3, 3)
    cumsum_stream = np.cumsum(results['weight_changes_stream'])
    cumsum_compressed = np.cumsum(results['weight_changes_compressed'])
    plt.plot(updates, cumsum_stream, 'r-o', label='STOCHASTIC_STREAM', markersize=4)
    plt.plot(updates, cumsum_compressed, 'b-o', label='STOCHASTIC_COMPRESSED', markersize=4)
    plt.xlabel('Update')
    plt.ylabel('Cumulative Weight Change')
    plt.title('Cumulative Weight Changes')
    plt.legend()
    plt.grid(True)

    # 비율 분석
    plt.subplot(2, 3, 4)
    ratios = [s/c if c > 0 else 0 for s, c in zip(results['weight_changes_stream'], results['weight_changes_compressed'])]
    plt.plot(updates, ratios, 'g-o', markersize=4)
    plt.xlabel('Update')
    plt.ylabel('Ratio (STREAM/COMPRESSED)')
    plt.title('Weight Change Ratio')
    plt.grid(True)
    plt.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Equal change')
    plt.legend()

    # 손실 차이
    plt.subplot(2, 3, 5)
    loss_diff = [abs(s - c) for s, c in zip(results['losses_stream'], results['losses_compressed'])]
    plt.plot(updates, loss_diff, 'm-o', markersize=4)
    plt.xlabel('Update')
    plt.ylabel('Absolute Loss Difference')
    plt.title('Loss Difference (|STREAM - COMPRESSED|)')
    plt.grid(True)
    plt.yscale('log')

    # 효율성 분석
    plt.subplot(2, 3, 6)
    # 손실 감소량 대비 가중치 변화량
    loss_reduction_stream = [results['losses_stream'][0] - loss for loss in results['losses_stream']]
    loss_reduction_compressed = [results['losses_compressed'][0] - loss for loss in results['losses_compressed']]

    efficiency_stream = [lr/wc if wc > 0 else 0 for lr, wc in zip(loss_reduction_stream, results['weight_changes_stream'])]
    efficiency_compressed = [lr/wc if wc > 0 else 0 for lr, wc in zip(loss_reduction_compressed, results['weight_changes_compressed'])]

    plt.plot(updates, efficiency_stream, 'r-o', label='STOCHASTIC_STREAM', markersize=4)
    plt.plot(updates, efficiency_compressed, 'b-o', label='STOCHASTIC_COMPRESSED', markersize=4)
    plt.xlabel('Update')
    plt.ylabel('Loss Reduction / Weight Change')
    plt.title('Update Efficiency')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('detailed_pulse_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


def main():
    """메인 분석 함수"""
    print("STOCHASTIC_STREAM vs STOCHASTIC_COMPRESSED 상세 비교 분석")
    print("=" * 80)

    # 1. 단일 업데이트 분석
    single_result = analyze_single_update()

    # 2. 다중 업데이트 분석
    multi_result = analyze_multiple_updates(num_updates=20)

    # 3. 시각화
    visualize_comparison(multi_result)

    # 4. 종합 분석
    print("\n" + "=" * 80)
    print("종합 분석 결과")
    print("=" * 80)

    if single_result['stream_change'] < 1e-8:
        print("🚨 STOCHASTIC_STREAM에서 가중치가 거의 업데이트되지 않습니다!")
        print("   가능한 원인:")
        print("   - 펄스 생성 로직에 문제가 있을 수 있습니다")
        print("   - 스트림 오버랩이 부족할 수 있습니다")
        print("   - A, B 파라미터 계산에 문제가 있을 수 있습니다")
    elif single_result['stream_change'] / single_result['compressed_change'] < 0.1:
        print("⚠️  STOCHASTIC_STREAM의 업데이트 효율이 낮습니다")
        print(f"   업데이트 비율: {single_result['stream_change'] / single_result['compressed_change']:.6f}")
    else:
        print("✅ 두 방법 모두 정상적으로 동작합니다")

    print("\n분석 완료!")


if __name__ == "__main__":
    main()