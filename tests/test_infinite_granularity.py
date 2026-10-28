# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Tests for infinite-granularity update mode (dw_min=0)."""

import time
from unittest import TestCase, skipIf

import torch
from torch import randn, manual_seed

from aihwkit.exceptions import ConfigError
from aihwkit.simulator.configs.configs import SingleRPUConfig, UnitCellRPUConfig
from aihwkit.simulator.configs.devices import (
    ConstantStepDevice,
    LinearStepDevice,
    SoftBoundsDevice,
    ExpStepDevice,
    PowStepDevice,
    PiecewiseStepDevice,
    SoftBoundsReferenceDevice,
    PowStepReferenceDevice,
    TransferCompound,
    BufferedTransferCompound,
    ChoppedTransferCompound,
    DynamicTransferCompound,
)

SKIP_CUDA = not torch.cuda.is_available()

DW_MIN_CONVERGENCE_SCAN = [0.001, 0.01, 0.1]


IG_DEVICE_CASES = [
    (ConstantStepDevice, {}),
    (LinearStepDevice, {"gamma_up": 0.5, "gamma_down": 0.5}),
    (SoftBoundsDevice, {}),
    (ExpStepDevice, {}),
    (PowStepDevice, {"pow_gamma": 0.5}),
    (
        PiecewiseStepDevice,
        {"piecewise_up": [1.0, 0.5, 1.0], "piecewise_down": [1.0, 0.5, 1.0]},
    ),
    (SoftBoundsReferenceDevice, {}),
    (PowStepReferenceDevice, {"pow_gamma": 0.5}),
]

WRITE_NOISE_DEVICE_CASES = [
    (LinearStepDevice, {"gamma_up": 0.5, "gamma_down": 0.5}),
    (SoftBoundsDevice, {}),
    (ExpStepDevice, {}),
    (PowStepDevice, {"pow_gamma": 0.5}),
    (
        PiecewiseStepDevice,
        {"piecewise_up": [1.0, 0.5, 1.0], "piecewise_down": [1.0, 0.5, 1.0]},
    ),
    (SoftBoundsReferenceDevice, {}),
]


def _make_tile(device_cls, dw_min, lr=0.01, out_size=5, in_size=4, bias=False, **kwargs):
    """Create an AnalogTile with the given device config."""
    from aihwkit.simulator.tiles.analog import AnalogTile

    device = device_cls(dw_min=dw_min, **kwargs)
    rpu_config = SingleRPUConfig(device=device)
    tile = AnalogTile(out_size, in_size, rpu_config, bias=bias)
    tile.tile.set_learning_rate(lr)
    return tile


def _set_weights_and_update(tile, w_init, x, d):
    """Set weights, run one update, return new weights."""
    tile.tile.set_weights(w_init.clone())
    tile.update(x, d)
    return tile.tile.get_weights()


class InfiniteGranularityBasicTest(TestCase):
    """Basic tests that IG mode activates and runs for each device type."""

    def test_all_ig_capable_devices_run(self):
        """All IG-capable concrete device families should run with dw_min=0."""
        manual_seed(1)
        for device_cls, kwargs in IG_DEVICE_CASES:
            with self.subTest(device=device_cls.__name__):
                tile = _make_tile(
                    device_cls, dw_min=0.0, lr=0.02,
                    up_down=0.0, up_down_dtod=0.0,
                    w_max_dtod=0.0, w_min_dtod=0.0,
                    **kwargs,
                )
                w_init = randn(5, 4) * 0.1
                tile.tile.set_weights(w_init.clone())
                tile.update(randn(3, 4), randn(3, 5))
                w_after = tile.tile.get_weights()
                self.assertFalse(torch.isnan(w_after).any())

    def test_constant_step_dw_min_zero(self):
        """ConstantStepDevice with dw_min=0 should produce correct update."""
        manual_seed(42)
        tile = _make_tile(
            ConstantStepDevice, dw_min=0.0, lr=0.1,
            up_down=0.0, up_down_dtod=0.0,
            w_max_dtod=0.0, w_min_dtod=0.0,
        )
        w_init = randn(5, 4) * 0.1
        tile.tile.set_weights(w_init.clone())
        x = randn(3, 4)
        d = randn(3, 5)
        tile.update(x, d)
        w_after = tile.tile.get_weights()

        # Expected: w -= lr * d^T @ x * 1.0 (no variation, beta=0)
        G = d.T @ x
        w_expected = w_init - 0.1 * G
        w_expected = w_expected.clamp(min=-0.6, max=0.6)
        torch.testing.assert_close(w_after, w_expected, atol=1e-5, rtol=1e-5)

    def test_constant_step_zero_initial_weights_match_matmul(self):
        """Zero-initialized ConstantStepDevice IG update should match matmul."""
        manual_seed(42)
        lr = 0.01
        batch_size = 32
        out_size, in_size = 24, 16
        tile = _make_tile(
            ConstantStepDevice, dw_min=0.0, lr=lr,
            out_size=out_size, in_size=in_size,
            w_max=1.0, w_min=-1.0,
            w_max_dtod=0.0, w_min_dtod=0.0,
            up_down_dtod=0.0, dw_min_dtod=0.0, dw_min_std=0.0,
            construction_seed=42,
        )
        w_init = torch.zeros(out_size, in_size)
        x = randn(batch_size, in_size) * 0.1
        d = randn(batch_size, out_size) * 0.1

        w_after = _set_weights_and_update(tile, w_init, x, d)

        w_expected = -lr * (d.T @ x)
        torch.testing.assert_close(w_after, w_expected, atol=1e-9, rtol=1e-6)

    def test_dw_min_nonzero_unchanged(self):
        """dw_min > 0 should still use stochastic update."""
        tile = _make_tile(ConstantStepDevice, dw_min=0.001)
        w_init = randn(5, 4) * 0.1
        tile.tile.set_weights(w_init.clone())
        x = randn(3, 4)
        d = randn(3, 5)
        tile.update(x, d)
        w_after = tile.tile.get_weights()
        # Should have changed (stochastic)
        self.assertFalse(torch.allclose(w_after, w_init, atol=1e-7))

    def test_linear_step_dw_min_zero(self):
        """LinearStepDevice with dw_min=0 should run without errors."""
        tile = _make_tile(
            LinearStepDevice, dw_min=0.0,
            gamma_up=0.5, gamma_down=0.5,
            gamma_up_dtod=0.0, gamma_down_dtod=0.0,
            up_down=0.0, up_down_dtod=0.0,
            w_max_dtod=0.0, w_min_dtod=0.0,
        )
        w_init = randn(5, 4) * 0.1
        tile.tile.set_weights(w_init.clone())
        x = randn(3, 4)
        d = randn(3, 5)
        tile.update(x, d)
        w_after = tile.tile.get_weights()
        self.assertFalse(torch.allclose(w_after, w_init, atol=1e-7))

    def test_soft_bounds_dw_min_zero(self):
        """SoftBoundsDevice with dw_min=0 should run."""
        tile = _make_tile(SoftBoundsDevice, dw_min=0.0)
        w_init = randn(5, 4) * 0.1
        tile.tile.set_weights(w_init.clone())
        tile.update(randn(3, 4), randn(3, 5))

    def test_exp_step_dw_min_zero(self):
        """ExpStepDevice with dw_min=0 should run."""
        tile = _make_tile(ExpStepDevice, dw_min=0.0)
        w_init = randn(5, 4) * 0.1
        tile.tile.set_weights(w_init.clone())
        tile.update(randn(3, 4), randn(3, 5))

    def test_pow_step_dw_min_zero(self):
        """PowStepDevice with dw_min=0 should run."""
        tile = _make_tile(PowStepDevice, dw_min=0.0)
        w_init = randn(5, 4) * 0.1
        tile.tile.set_weights(w_init.clone())
        tile.update(randn(3, 4), randn(3, 5))

    def test_piecewise_step_dw_min_zero(self):
        """PiecewiseStepDevice with dw_min=0 should run."""
        tile = _make_tile(
            PiecewiseStepDevice, dw_min=0.0,
            piecewise_up=[1.0, 0.5, 1.0],
            piecewise_down=[1.0, 0.5, 1.0],
        )
        w_init = randn(5, 4) * 0.1
        tile.tile.set_weights(w_init.clone())
        tile.update(randn(3, 4), randn(3, 5))


class InfiniteGranularityResponseTest(TestCase):
    """Verify response function correctness for specific device types."""

    def test_linear_step_response_formula(self):
        """Verify LinearStepDevice IG update matches the known formula.

        C++ computes slopes as:
          slope_up = -ls_decrease_up * scale_up / w_ref_up
          slope_down = -ls_decrease_down * scale_down / w_ref_down
        With dw_min=0 (effective=1), scale_up=scale_down=1, and
        mean_bound_reference=True: w_ref_up=w_max, w_ref_down=w_min.
        """
        manual_seed(42)
        lr = 0.05
        ls_decrease_up = 0.8
        ls_decrease_down = 0.6
        w_max = 0.6
        w_min = -0.6

        tile = _make_tile(
            LinearStepDevice, dw_min=0.0, lr=lr,
            gamma_up=ls_decrease_up, gamma_down=ls_decrease_down,
            gamma_up_dtod=0.0, gamma_down_dtod=0.0,
            up_down=0.0, up_down_dtod=0.0,
            w_max=w_max, w_min=w_min,
            w_max_dtod=0.0, w_min_dtod=0.0,
        )

        w_init = randn(5, 4) * 0.1
        x = randn(1, 4)  # batch=1 for exact formula match
        d = randn(1, 5)
        w_after = _set_weights_and_update(tile, w_init, x, d)

        # Manually compute expected result matching C++ slope formula
        G = d.T @ x
        scale_up = 1.0  # effective_dw_min=1, gain=1, up_bias=0
        scale_down = 1.0
        slope_up = -ls_decrease_up * scale_up / w_max
        slope_down = -ls_decrease_down * scale_down / w_min

        resp_up = slope_up * w_init + scale_up
        resp_down = slope_down * w_init + scale_down
        resp = torch.where(G > 0, resp_down, resp_up)

        w_expected = w_init - lr * G * resp
        w_expected = w_expected.clamp(min=w_min, max=w_max)

        torch.testing.assert_close(w_after, w_expected, atol=1e-5, rtol=1e-5)


class InfiniteGranularityConvergenceTest(TestCase):
    """Convergence: avg stochastic update stays within the dw_min scale."""

    def _convergence_test(self, device_cls, dw_min, n_repeat=1000, **kwargs):
        manual_seed(0)
        lr = 0.1
        out_size, in_size = 3, 3
        common_kwargs = dict(
            up_down=0.0, up_down_dtod=0.0,
            w_max_dtod=0.0, w_min_dtod=0.0,
            dw_min_dtod=0.0, dw_min_std=0.0,
            **kwargs,
        )

        w_init = randn(out_size, in_size) * 0.1
        x = randn(1, in_size) * 0.3
        d = randn(1, out_size) * 0.3

        tile_ig = _make_tile(
            device_cls, dw_min=0.0, lr=lr,
            out_size=out_size, in_size=in_size, **common_kwargs
        )
        w_ig = _set_weights_and_update(tile_ig, w_init, x, d)

        w_sum = torch.zeros_like(w_init)
        for _ in range(n_repeat):
            tile_s = _make_tile(
                device_cls, dw_min=dw_min, lr=lr,
                out_size=out_size, in_size=in_size, **common_kwargs
            )
            w_s = _set_weights_and_update(tile_s, w_init, x, d)
            w_sum += w_s
        w_avg = w_sum / n_repeat

        err = (w_avg - w_ig).abs().mean().item()
        self.assertLess(
            err, dw_min,
            f"Convergence failed for {device_cls.__name__}: "
            f"dw_min={dw_min:.6f}, err={err:.6f}"
        )

    def test_convergence_constant_step(self):
        for dw_min in DW_MIN_CONVERGENCE_SCAN:
            with self.subTest(dw_min=dw_min):
                self._convergence_test(ConstantStepDevice, dw_min=dw_min)

    def test_convergence_linear_step(self):
        for dw_min in DW_MIN_CONVERGENCE_SCAN:
            with self.subTest(dw_min=dw_min):
                self._convergence_test(
                    LinearStepDevice,
                    dw_min=dw_min,
                    gamma_up=0.5, gamma_down=0.5,
                    gamma_up_dtod=0.0, gamma_down_dtod=0.0,
                )

    def test_convergence_soft_bounds(self):
        for dw_min in DW_MIN_CONVERGENCE_SCAN:
            with self.subTest(dw_min=dw_min):
                self._convergence_test(SoftBoundsDevice, dw_min=dw_min)


class InfiniteGranularityWriteNoiseTest(TestCase):
    """Smoke tests for IG persistent/apparent write-noise path."""

    def _write_noise_test(self, device_cls, cuda=False, **kwargs):
        manual_seed(7)
        tile = _make_tile(
            device_cls, dw_min=0.0, lr=0.01,
            out_size=5, in_size=4,
            write_noise_std=0.01,
            up_down=0.0, up_down_dtod=0.0,
            w_max_dtod=0.0, w_min_dtod=0.0,
            **kwargs,
        )
        if cuda:
            tile = tile.cuda()

        w_init = randn(5, 4) * 0.05
        x = randn(2, 4) * 0.1
        d = randn(2, 5) * 0.1
        tile.tile.set_weights(w_init.clone())
        tile.update(x.cuda() if cuda else x, d.cuda() if cuda else d)

        w_after = tile.tile.get_weights()
        hidden = tile.get_hidden_parameters()
        self.assertIn("persistent_weights", hidden)
        persistent = hidden["persistent_weights"]

        self.assertFalse(torch.isnan(w_after).any())
        self.assertGreater((w_after.cpu() - persistent.cpu()).abs().max().item(), 1e-6)
        self.assertLessEqual(persistent.max().item(), 0.6 + 1e-6)
        self.assertGreaterEqual(persistent.min().item(), -0.6 - 1e-6)

    def test_cpu_write_noise_devices(self):
        for device_cls, kwargs in WRITE_NOISE_DEVICE_CASES:
            with self.subTest(device=device_cls.__name__):
                self._write_noise_test(device_cls, **kwargs)

    @skipIf(SKIP_CUDA, "CUDA not available")
    def test_gpu_write_noise_devices(self):
        for device_cls, kwargs in WRITE_NOISE_DEVICE_CASES:
            with self.subTest(device=device_cls.__name__):
                self._write_noise_test(device_cls, cuda=True, **kwargs)


@skipIf(SKIP_CUDA, "CUDA not available")
class InfiniteGranularityCPUGPUConsistencyTest(TestCase):
    """Verify CPU and GPU produce identical results for IG mode."""

    def _cpu_gpu_test(self, device_cls, **kwargs):
        """Test CPU/GPU consistency with batch_size=1."""
        manual_seed(42)
        lr = 0.05
        out_size, in_size = 8, 6

        tile_cpu = _make_tile(device_cls, dw_min=0.0, lr=lr,
                              out_size=out_size, in_size=in_size, **kwargs)
        tile_gpu = _make_tile(device_cls, dw_min=0.0, lr=lr,
                              out_size=out_size, in_size=in_size, **kwargs)
        tile_gpu = tile_gpu.cuda()

        w_init = randn(out_size, in_size) * 0.1
        x = randn(1, in_size)
        d = randn(1, out_size)

        tile_cpu.tile.set_weights(w_init.clone())
        tile_gpu.tile.set_weights(w_init.clone())

        tile_cpu.update(x, d)
        tile_gpu.update(x.cuda(), d.cuda())

        w_cpu = tile_cpu.tile.get_weights()
        w_gpu = tile_gpu.tile.get_weights()

        torch.testing.assert_close(
            w_cpu, w_gpu, atol=1e-5, rtol=1e-5,
            msg=f"CPU/GPU mismatch for {device_cls.__name__}"
        )

    def test_cpu_gpu_constant_step(self):
        self._cpu_gpu_test(
            ConstantStepDevice,
            up_down=0.0, up_down_dtod=0.0,
            w_max_dtod=0.0, w_min_dtod=0.0,
        )

    def test_cpu_gpu_linear_step(self):
        self._cpu_gpu_test(
            LinearStepDevice,
            gamma_up=0.5, gamma_down=0.5,
            gamma_up_dtod=0.0, gamma_down_dtod=0.0,
            up_down=0.0, up_down_dtod=0.0,
            w_max_dtod=0.0, w_min_dtod=0.0,
        )

    def test_cpu_gpu_soft_bounds(self):
        self._cpu_gpu_test(
            SoftBoundsDevice,
            up_down=0.0, up_down_dtod=0.0,
            w_max_dtod=0.0, w_min_dtod=0.0,
        )

    def test_cpu_gpu_exp_step(self):
        self._cpu_gpu_test(
            ExpStepDevice,
            up_down=0.0, up_down_dtod=0.0,
            w_max_dtod=0.0, w_min_dtod=0.0,
        )

    def test_cpu_gpu_pow_step(self):
        self._cpu_gpu_test(
            PowStepDevice,
            pow_gamma=0.5, pow_gamma_dtod=0.0,
            up_down=0.0, up_down_dtod=0.0,
            w_max_dtod=0.0, w_min_dtod=0.0,
        )

    def test_cpu_gpu_piecewise_step(self):
        self._cpu_gpu_test(
            PiecewiseStepDevice,
            piecewise_up=[1.0, 0.8, 0.6, 0.4, 0.2],
            piecewise_down=[0.2, 0.4, 0.6, 0.8, 1.0],
            up_down=0.0, up_down_dtod=0.0,
            w_max_dtod=0.0, w_min_dtod=0.0,
        )

    def test_cpu_gpu_softbounds_reference(self):
        self._cpu_gpu_test(
            SoftBoundsReferenceDevice,
            up_down=0.0, up_down_dtod=0.0,
            w_max_dtod=0.0, w_min_dtod=0.0,
        )

    def test_cpu_gpu_powstep_reference(self):
        self._cpu_gpu_test(
            PowStepReferenceDevice,
            pow_gamma=0.5, pow_gamma_dtod=0.0,
            up_down=0.0, up_down_dtod=0.0,
            w_max_dtod=0.0, w_min_dtod=0.0,
        )


@skipIf(SKIP_CUDA, "CUDA not available")
class InfiniteGranularityPerformanceTest(TestCase):
    """Performance and memory benchmarks for IG mode."""

    def _benchmark(self, device_cls, dw_min, out_size, in_size, batch_size, n_iter=50, **kwargs):
        """Run n_iter updates and return (total_time, peak_memory_bytes)."""
        rpu_config = SingleRPUConfig(device=device_cls(dw_min=dw_min, **kwargs))
        from aihwkit.simulator.tiles.analog import AnalogTile
        tile = AnalogTile(out_size, in_size, rpu_config, bias=False)
        tile.tile.set_learning_rate(0.01)
        tile = tile.cuda()

        w_init = randn(out_size, in_size) * 0.1
        tile.tile.set_weights(w_init)

        x = randn(batch_size, in_size).cuda()
        d = randn(batch_size, out_size).cuda()

        # Warmup
        for _ in range(5):
            tile.tile.set_weights(w_init)
            tile.update(x, d)
        torch.cuda.synchronize()

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        for _ in range(n_iter):
            tile.tile.set_weights(w_init)
            tile.update(x, d)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        peak_mem = torch.cuda.max_memory_allocated()

        return elapsed, peak_mem

    def test_performance_small_batch(self):
        """IG should be comparable or faster than stochastic for small batch."""
        for batch_size in [1, 8]:
            t_ig, _ = self._benchmark(ConstantStepDevice, dw_min=0.0, out_size=64, in_size=64,
                                      batch_size=batch_size)
            t_st, _ = self._benchmark(ConstantStepDevice, dw_min=0.001, out_size=64, in_size=64,
                                      batch_size=batch_size)
            print(f"  batch={batch_size}: IG={t_ig:.4f}s, stochastic={t_st:.4f}s, "
                  f"ratio={t_ig / t_st:.2f}")

    def test_performance_large_batch(self):
        """IG with large batch should not be excessively slow."""
        for batch_size in [256, 1024, 4096]:
            t_ig, _ = self._benchmark(ConstantStepDevice, dw_min=0.0, out_size=64, in_size=64,
                                      batch_size=batch_size)
            t_st, _ = self._benchmark(ConstantStepDevice, dw_min=0.001, out_size=64, in_size=64,
                                      batch_size=batch_size)
            print(f"  batch={batch_size}: IG={t_ig:.4f}s, stochastic={t_st:.4f}s, "
                  f"ratio={t_ig / t_st:.2f}")

    def test_memory_large_batch(self):
        """IG should not use significantly more memory than stochastic."""
        for batch_size in [1024, 4096, 16384]:
            _, mem_ig = self._benchmark(ConstantStepDevice, dw_min=0.0, out_size=128, in_size=128,
                                        batch_size=batch_size, n_iter=10)
            _, mem_st = self._benchmark(ConstantStepDevice, dw_min=0.001, out_size=128, in_size=128,
                                        batch_size=batch_size, n_iter=10)
            diff_mb = (mem_ig - mem_st) / (1024 * 1024)
            print(f"  batch={batch_size}: IG_mem={mem_ig / (1024 * 1024):.1f}MB, "
                  f"stochastic_mem={mem_st / (1024 * 1024):.1f}MB, diff={diff_mb:.1f}MB")
            # IG should not use significantly more memory than stochastic
            # The IG GEMM allocates input buffers + grad matrix temp (~same as stochastic)
            # Allow up to 20MB extra
            self.assertLess(
                diff_mb, 20.0,
                f"IG uses too much extra memory at batch={batch_size}: "
                f"diff={diff_mb:.1f}MB (IG={mem_ig}, stochastic={mem_st})"
            )


class InfiniteGranularityTransferGuardTest(TestCase):
    """IG mode (dw_min=0) must be rejected inside transfer compounds.

    Transfer compounds scale the transfer learning rate and buffer
    granularity by the sub-device weight granularity (dw_min); a zero
    dw_min would silently disable learning, so it must raise a ConfigError.
    """

    TRANSFER_COMPOUNDS = [
        TransferCompound,
        BufferedTransferCompound,
        ChoppedTransferCompound,
        DynamicTransferCompound,
    ]

    def test_dw_min_zero_rejected_on_construction(self):
        """Constructing a transfer compound with a dw_min=0 sub-device raises."""
        for compound_cls in self.TRANSFER_COMPOUNDS:
            with self.subTest(compound=compound_cls.__name__):
                with self.assertRaises(ConfigError):
                    compound_cls(
                        unit_cell_devices=[
                            ConstantStepDevice(dw_min=0.0),
                            ConstantStepDevice(dw_min=0.001),
                        ]
                    )

    def test_dw_min_zero_on_slow_device_rejected(self):
        """A dw_min=0 on any sub-device (not only the fast one) raises."""
        with self.assertRaises(ConfigError):
            ChoppedTransferCompound(
                unit_cell_devices=[
                    ConstantStepDevice(dw_min=0.001),
                    ConstantStepDevice(dw_min=0.0),
                ]
            )

    def test_nonzero_dw_min_transfer_compound_builds(self):
        """A transfer compound with non-zero dw_min builds a working tile."""
        rpu_config = UnitCellRPUConfig(
            device=ChoppedTransferCompound(
                unit_cell_devices=[
                    ConstantStepDevice(dw_min=0.001),
                    ConstantStepDevice(dw_min=0.001),
                ]
            )
        )
        tile = rpu_config.tile_class(5, 4, rpu_config, bias=False)
        tile.update(randn(3, 4), randn(3, 5))

    def test_single_device_dw_min_zero_still_allowed(self):
        """IG mode remains valid for a plain (non-compound) pulsed device."""
        tile = _make_tile(ConstantStepDevice, dw_min=0.0)
        tile.update(randn(3, 4), randn(3, 5))
