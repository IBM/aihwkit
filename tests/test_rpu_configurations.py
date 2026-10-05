# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Tests for the high level simulator devices functionality."""
from os import environ, getenv
from statistics import median
from sys import version_info
from time import perf_counter
from unittest import SkipTest

from pytest import mark
from torch import cuda, equal, manual_seed

from aihwkit.exceptions import ConfigError
from aihwkit.simulator.configs import SingleRPUConfig
from aihwkit.simulator.configs.devices import (
    PowStepDevice,
    PowStepReferenceDevice,
    SoftBoundsReferenceDevice,
)
from aihwkit.simulator.parameters import IOParameters, UpdateParameters
from aihwkit.simulator.presets.devices import EcRamPresetDevice
from aihwkit.simulator.tiles.analog import AnalogTile

from .helpers.decorators import parametrize_over_tiles
from .helpers.testcases import ParametrizedTestCase
from .helpers.tiles import (
    FloatingPoint,
    Ideal,
    ConstantStep,
    LinearStep,
    ExpStep,
    SoftBounds,
    SoftBoundsPmax,
    PowStep,
    PiecewiseStep,
    Vector,
    OneSided,
    Transfer,
    MixedPrecision,
    FloatingPointCuda,
    IdealCuda,
    ConstantStepCuda,
    LinearStepCuda,
    ExpStepCuda,
    SoftBoundsCuda,
    SoftBoundsPmaxCuda,
    PowStepCuda,
    PiecewiseStepCuda,
    VectorCuda,
    OneSidedCuda,
    TransferCuda,
    MixedPrecisionCuda,
)


@parametrize_over_tiles([FloatingPoint, FloatingPointCuda])
class RPUConfigurationsFloatingPointTest(ParametrizedTestCase):
    """Tests related to resistive processing unit configurations (floating point)."""

    def test_create_array(self):
        """Test creating an array using the mappings to bindings."""
        rpu_config = self.get_rpu_config()

        tile_params = rpu_config.device.as_bindings(rpu_config.runtime.data_type)

        _ = tile_params.create_array(10, 20)

    def test_config_device_parameters(self):
        """Test modifying the device parameters."""
        rpu_config = self.get_rpu_config()

        rpu_config.device.diffusion = 1.23
        rpu_config.device.lifetime = 4.56

        tile = self.get_tile(11, 22, rpu_config).tile

        # Assert over the parameters in the binding objects.
        config = tile.get_meta_parameters()
        self.assertAlmostEqual(config.diffusion, 1.23, places=4)
        self.assertAlmostEqual(config.lifetime, 4.56, places=4)


@parametrize_over_tiles(
    [
        Ideal,
        ConstantStep,
        LinearStep,
        ExpStep,
        SoftBounds,
        SoftBoundsPmax,
        PowStep,
        PiecewiseStep,
        Vector,
        OneSided,
        Transfer,
        MixedPrecision,
        IdealCuda,
        ConstantStepCuda,
        LinearStepCuda,
        ExpStepCuda,
        SoftBoundsCuda,
        SoftBoundsPmaxCuda,
        PowStepCuda,
        PiecewiseStepCuda,
        VectorCuda,
        OneSidedCuda,
        TransferCuda,
        MixedPrecisionCuda,
    ]
)
class RPUConfigurationsTest(ParametrizedTestCase):
    """Tests related to resistive processing unit configurations."""

    def test_create_array(self):
        """Test creating an array using the mappings to bindings."""
        rpu_config = self.get_rpu_config()

        tile_params = rpu_config.as_bindings()
        device_params = rpu_config.device.as_bindings(rpu_config.runtime.data_type)

        _ = tile_params.create_array(10, 20, device_params)

    def test_type_error(self):
        """Test wrong mappings to bindings."""
        if not (version_info[0] >= 3 and version_info[1] > 7):
            raise SkipTest("Only supported for python > 3.7")

        rpu_config = self.get_rpu_config()
        rpu_config.forward.out_bound = True

        with self.assertRaises(ConfigError):
            rpu_config.as_bindings()

    def test_config_device_parameters(self):
        """Test modifying the device parameters."""
        rpu_config = self.get_rpu_config()

        rpu_config.device.diffusion = 1.23
        rpu_config.device.lifetime = 4.56
        rpu_config.device.construction_seed = 192

        # TODO: don't assert over tile.get_meta_parameters() as some of them might
        # not be present.
        _ = self.get_tile(11, 22, rpu_config).tile

    def test_config_tile_parameters(self):
        """Test modifying the tile parameters."""
        rpu_config = self.get_rpu_config()

        rpu_config.forward = IOParameters(inp_noise=0.321)
        rpu_config.backward = IOParameters(inp_noise=0.456)
        rpu_config.update = UpdateParameters(desired_bl=78)

        tile = self.get_tile(11, 22, rpu_config).tile

        # Assert over the parameters in the binding objects.
        config = tile.get_meta_parameters()
        self.assertAlmostEqual(config.forward_io.inp_noise, 0.321)
        self.assertAlmostEqual(config.backward_io.inp_noise, 0.456)
        self.assertAlmostEqual(config.update.desired_bl, 78)

    def test_construction_seed(self):
        """Test the construction seed leads to the same tile values."""
        rpu_config = self.get_rpu_config()

        # Set the seed.
        rpu_config.device.construction_seed = 10

        tile_1 = self.get_tile(3, 4, rpu_config)
        tile_2 = self.get_tile(3, 4, rpu_config)

        hidden_parameters_1 = tile_1.get_hidden_parameters()
        hidden_parameters_2 = tile_2.get_hidden_parameters()

        # Compare old and new hidden parameters tensors.
        for (field, old), (_, new) in zip(hidden_parameters_1.items(), hidden_parameters_2.items()):
            if "weights" in field:
                # exclude weights as these are not governed by construction seed
                continue
            self.assertTrue(old.allclose(new))


@mark.parametrize(
    "device_type",
    [EcRamPresetDevice, SoftBoundsReferenceDevice, PowStepDevice, PowStepReferenceDevice],
)
def test_parallel_device_init_is_seeded(monkeypatch, device_type):
    """Row-local random streams give identical parameters for the same seed."""
    monkeypatch.setenv("AIHWKIT_PARALLEL_DEVICE_INIT", "1")
    config = SingleRPUConfig(device=device_type(construction_seed=123))
    first = AnalogTile(256, 256, config, bias=False).get_hidden_parameters()
    second = AnalogTile(256, 256, config, bias=False).get_hidden_parameters()

    assert first.keys() == second.keys()
    parameters = [name for name in first if "weights" not in name]
    assert parameters
    for name in parameters:
        assert equal(first[name], second[name]), name


@mark.skipif(
    getenv("AIHWKIT_RUN_LARGE_TILE_BENCHMARK") != "1",
    reason="Set AIHWKIT_RUN_LARGE_TILE_BENCHMARK=1 to run the large tile timing test",
)
def test_large_tile_initialization_timing():
    """Manually time a large tile; run this node with ``pytest -s``.

    Defaults: 3072x768 EcRam, three repetitions, CPU construction plus CUDA
    transfer. Override with AIHWKIT_BENCHMARK_DEVICE, _OUT_SIZE, _IN_SIZE,
    _REPEATS, or _CPU_ONLY=1. AIHWKIT_PARALLEL_DEVICE_INIT=1 enables the
    optimized path, whose fixed-seed device samples differ from the old path.
    """
    device_types = {
        "ecram": EcRamPresetDevice,
        "softbounds": SoftBoundsReferenceDevice,
        "powstep": PowStepDevice,
        "powstep-reference": PowStepReferenceDevice,
    }
    device_name = getenv("AIHWKIT_BENCHMARK_DEVICE", "ecram")
    assert device_name in device_types, f"Unknown benchmark device: {device_name}"

    out_size = int(getenv("AIHWKIT_BENCHMARK_OUT_SIZE", "3072"))
    in_size = int(getenv("AIHWKIT_BENCHMARK_IN_SIZE", "768"))
    repeats = int(getenv("AIHWKIT_BENCHMARK_REPEATS", "3"))
    assert min(out_size, in_size, repeats) > 0
    use_cuda = getenv("AIHWKIT_BENCHMARK_CPU_ONLY") != "1"
    if use_cuda:
        assert cuda.is_available(), "CUDA is unavailable; set AIHWKIT_BENCHMARK_CPU_ONLY=1"

    config = SingleRPUConfig(device=device_types[device_name]())
    if use_cuda:
        warmup = AnalogTile(32, 32, config, bias=False).cuda()
        cuda.synchronize()
        del warmup

    cpu_times = []
    cuda_times = []
    total_times = []
    for run in range(repeats):
        manual_seed(1234 + run)
        start = perf_counter()
        tile = AnalogTile(out_size, in_size, config, bias=False)
        cpu_times.append(perf_counter() - start)
        if use_cuda:
            start = perf_counter()
            tile.cuda()
            cuda.synchronize()
            cuda_times.append(perf_counter() - start)
            total_times.append(cpu_times[-1] + cuda_times[-1])
        del tile

    print(
        f"tile={out_size}x{in_size} device={device_name} repeats={repeats} "
        f"parallel_init={environ.get('AIHWKIT_PARALLEL_DEVICE_INIT', '0')} "
        f"omp_threads={environ.get('OMP_NUM_THREADS', 'default')}"
    )
    print(f"CPU construction: {median(cpu_times):.3f} s median; samples={cpu_times}")
    if use_cuda:
        print(f"CUDA transfer: {median(cuda_times):.3f} s median; samples={cuda_times}")
        print(f"CPU + CUDA: {median(total_times):.3f} s median; samples={total_times}")
