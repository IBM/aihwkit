# Building aihwkit with CUDA and verifying the Half-Select (HS) device

This guide covers building the CUDA/GPU port of the half-select (HS) device-update
feature on **WSL2** with an **NVIDIA GTX 1660 (Turing, compute capability sm_75)**,
and how to verify HS on the GPU.

The current dev environment is CPU-only: the system `nvcc` is 10.1 (too old) and the
active venv has `torch 2.10.0+cpu`. The steps below replace both with CUDA-capable
versions. None of this can be validated in the current environment; it is written to
be run on a machine/driver stack that exposes the GTX 1660 to WSL2.

---

## 1. Host-side: NVIDIA driver + WSL2 CUDA passthrough

1. On **Windows** (not inside WSL2) install a recent NVIDIA Windows driver that
   supports WSL2 CUDA (any Game Ready / Studio driver from the last couple of years
   includes the WSL2 CUDA stub). Do **not** install a Linux display driver inside WSL2.
2. Restart WSL2 (`wsl --shutdown` from a Windows shell, then reopen).
3. Verify GPU passthrough works inside WSL2:
   ```bash
   nvidia-smi
   ```
   You should see the GTX 1660 listed with a driver/CUDA version. If this fails, the
   toolkit steps below will not help — fix the driver first.

---

## 2. CUDA Toolkit 11.8 or 12.x inside WSL2

The system `nvcc` (10.1) does not support recent PyTorch or sm_75 toolchains well;
install a modern toolkit **using the WSL-Ubuntu package** (it omits the Linux display
driver, which must stay on the Windows side).

Example for CUDA 12.x (adjust for 11.8 if matching a cu118 torch wheel):
```bash
# Pick ONE toolkit version and keep torch consistent with it (see step 3).
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-4      # or: cuda-toolkit-11-8
```

Set the environment so the *new* `nvcc` is found first (add to `~/.bashrc`):
```bash
export CUDA_HOME=/usr/local/cuda-12.4          # or /usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```
Verify:
```bash
which nvcc && nvcc --version    # must report 11.8 / 12.x, NOT 10.1
```

---

## 3. CUDA-enabled PyTorch matching the toolkit

The build links against the torch that is importable in the active venv. The current
`torch 2.10.0+cpu` must be replaced by a CUDA wheel whose CUDA minor line matches the
toolkit family (cu118 for CUDA 11.8, cu121/cu124 for CUDA 12.x).

The `snn-eprop-analog` venv is **uv-managed**, so install through uv there, e.g.:
```bash
# from the snn-eprop-analog project (uv-managed)
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
```
Or, in a plain venv used for aihwkit:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```
Verify CUDA is visible to torch before building:
```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
# expect something like: 2.x.y+cu121 12.1 True
```

---

## 4. Build aihwkit with CUDA (sm_75)

`RPU_CUDA_ARCHITECTURES=75` matches the GTX 1660 (Turing, sm_75). Building for the
exact arch avoids long multi-arch compiles and JIT fallbacks.

From the repo root (`/home/minsik/aihwkit`):

```bash
python setup.py build_ext -j8 \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_CUDA=ON \
    -DRPU_CUDA_ARCHITECTURES=75 \
    --inplace
```

`--inplace` places the built `rpu_base*.so` next to the sources. If it is not picked
up automatically, copy it into the package:
```bash
cp build/lib*/aihwkit/simulator/rpu_base*.so src/aihwkit/simulator/
```

Alternative (PEP 517) invocation, equivalent flags:
```bash
pip install -e . -C--global-option="build_ext" \
    -C--global-option="-DUSE_CUDA=ON" \
    -C--global-option="-DRPU_CUDA_ARCHITECTURES=75"
```

Notes:
- Make sure the `python`/`pip` used here is the same interpreter that has the CUDA
  torch from step 3 — the CMake torch discovery keys off it.
- If CMake finds the wrong CUDA, pass `-DCUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME`.

---

## 5. Verifying HS on the GPU

This mirrors the CPU test in `tests/test_hs_verify.py`, but on a CUDA tile. Move the
tile to GPU, reach the underlying C++ tile via `analog_tile.tile`, enable tracking,
run updates with a HALFSELECTED pulse type, and check counts + decay effect.

```python
import torch
from aihwkit.simulator.tiles import AnalogTile
from aihwkit.simulator.configs import SingleRPUConfig
from aihwkit.simulator.configs.devices import ConstantStepDevice
from aihwkit.simulator.parameters.enums import PulseType

def make_tile(hs_decay, x_size=8, d_size=8):
    rpu = SingleRPUConfig(device=ConstantStepDevice(hs_decay=hs_decay))
    rpu.update.pulse_type = PulseType.HALF_SELECTED  # HalfselectedStochastic
    analog_tile = AnalogTile(d_size, x_size, rpu)
    analog_tile.cuda()                 # move to GPU
    return analog_tile

analog_tile = make_tile(hs_decay=0.5)
raw = analog_tile.tile                 # underlying C++ RPUCudaPulsed tile
raw.enable_hs_tracking()               # call BEFORE updates (matches CPU semantics)

x = torch.ones(8).cuda()
d = torch.ones(8).cuda()
for _ in range(200):
    analog_tile.update(x, d)

counts = raw.get_hs_transition_counts()   # torch.Tensor of length 16
print("HS transition counts:", counts)
assert counts.sum().item() > 0, "expected non-zero HS transitions"

# hs_decay effect: compare final weights for decay 1.0 vs 0.5 with identical updates
def run(hs_decay):
    t = make_tile(hs_decay=hs_decay)
    t.tile.enable_hs_tracking()
    for _ in range(200):
        t.update(x, d)
    return t.get_weights()[0].clone()

w_no_decay = run(1.0)
w_decay    = run(0.5)
print("mean |w| decay=1.0:", w_no_decay.abs().mean().item())
print("mean |w| decay=0.5:", w_decay.abs().mean().item())
# with decay < 1.0 the HS decay transitions pull magnitudes down -> should differ
assert not torch.allclose(w_no_decay, w_decay)
```

Exact pulse-type enum name: use whatever `PulseType` value maps to
`HalfselectedStochastic` / `HalfselectedStochasticStream` in
`src/aihwkit/simulator/parameters/enums.py` (check that file for the Python name).
The 16-element count vector is laid out as `(prev-1)*4 + (curr-1)` for states HS1..HS4,
i.e. HS1->HS1, HS1->HS2, ..., HS4->HS4.

---

## 6. Known limitations / semantic notes

- **uint32 bit-line path only.** GPU HS is wired only through the 32-bit pulse-count
  path (`PWUKernelParameterBatchSharedFunctorHS`). Implicit-pulse / BO64 paths are not
  HS-aware.
- **ConstantStep only.** Only `ConstantStepDevice` registers the HS kernels
  (`gp_count=2`) and populates `global_par[1] = hs_decay` via `setupHSGlobalParams`.
  `LinearStep` (and other pulsed devices) are **not** HS-wired on GPU.
- **Decay can apply without `enable_hs_tracking()`.** On GPU, whenever the HALFSELECTED
  pulse type + ConstantStep HS kernel is selected, `getUpdateKernels` lazily allocates
  the per-synapse HS-state buffer (`dev_hs_states_`) so the kernel never dereferences a
  null buffer — and the `w *= hs_decay` decay transitions therefore apply even if HS
  tracking was never explicitly enabled. On CPU the decay is gated on
  `enable_hs_tracking()`. **To match CPU behavior and to get non-zero transition
  counts, always call `enable_hs_tracking()` before running updates.**
- **Enable ordering matters.** The diagnostic transition-count buffer (`dev_hs_counts_`)
  is only allocated by `enable_hs_tracking()`, and that call is a no-op if HS state was
  already auto-allocated by a prior update (because `hs_gpu_enabled_` is already true).
  So enable tracking **before** the first update; otherwise counts may stay zero even
  though decay is being applied.
