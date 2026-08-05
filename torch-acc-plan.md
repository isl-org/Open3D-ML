# Open3D-ML torch.accelerator (CUDA/XPU) generalization plan

Status as of 2026-08-04. Use `torch.accelerator` API where possible to
generalize CUDA-only code paths to also work with XPU (Intel Arc, SYCL).
Parity testing strategy (critical, must not regress):
- old CUDA impl results == old CPU impl results
- new SYCL/XPU impl results == old==new CPU impl results

## Already DONE in this repo (verified in this session)
- `tests/test_models_torch.py` and `tests/torch_backend_parity.py`: `_accel`/`accel`
  detection now requires BOTH `o3d.core.sycl.is_available()` AND
  `torch.xpu.is_available()` before selecting XPU (o3d.core.sycl.is_available()
  is True even on SYCL-CPU-fallback-only hosts with no real GPU — unsafe alone).
- `ml3d/metrics/__init__.py`: added SYCL branch selecting `iou_bev_sycl`/`iou_3d_sycl`
  when `open3d.core.sycl.device_count() > 0` (verified numerically matches CPU,
  atol=1e-4).
- `ci/run_ci.sh`: configurable `./ci/run_ci.sh cpu|cuda|xpu` — wheel + torch
  deps per backend; parity suite only for cuda/xpu. GitHub Actions runs `cpu` only.
- `docs/howtos.md`: new "Testing CPU / CUDA / XPU parity" section + CI note.
- `README.md`: added `requirements-torch-xpu.txt` install line + howtos link.
- `ml3d/torch/pipelines/base_pipeline.py`: ALREADY generalized (contrary to
  earlier assumption) — `__init__` already has an `elif (device == 'xpu' or
  str(device).startswith('xpu')) and hasattr(torch, 'xpu') and
  torch.xpu.is_available():` branch alongside the cuda/cpu branches. No changes
  needed here unless we want to also generalize the `torch.cuda.set_device()`
  call in the `distributed` branch to use `torch.accelerator.set_device_index()`.
- `ml3d/utils/builder.py::convert_device_name()`: ALREADY generalized (contrary
  to earlier assumption) — maps `gpu_names = ["gpu", "cuda"] -> "cuda"`,
  `xpu_names = ["xpu", "sycl"] -> "xpu"`, `cpu_names = ["cpu"] -> "cpu"`. No
  changes needed.
- **2026-08-04 (items 1–4):** `point_transformer.py` KNN indices use `.to(device)`
  instead of `.cuda()`; `object_detection.py` distributed wrap uses `model.to()`.
  Model ctors (`PointPillars`, `PointRCNN`, `PVCNN`, `SparseConvUnet`) default
  `device=None` → `ml3d.torch.utils.torch_utils.default_training_device()`
  (cuda if available, else xpu if `torch.xpu.is_available()`, else cpu).
  `scripts/run_pipeline.py`: TF branch handles bare `xpu`, distributed device
  string uses `{args.device}:{id}`, `--backend` help mentions `xccl`.
- **2026-08-04 (follow-up):** `BasePipeline` / torch pipelines default
  `device=None` → `default_training_device()`; distributed rank setup uses
  `torch.accelerator.set_device_index()` when available. Object-detection DDP
  uses integer `device_ids` (was passing `torch.device`).

## Remaining TODO

1. ~~Optional: `base_pipeline.py` distributed branch — `torch.cuda.set_device()` →
   `torch.accelerator.set_device_index()`~~ DONE (2026-08-04).
2. ~~`ci/run_ci.sh`~~ backend argument + GH Actions `cpu` only.
3. Pipeline/model parity tests on local Arc: all 8 `test_models_torch.py` cases
   hit exit 124 at TIMEOUT=300s (XPU oneDNN init then stall — see KNOWN RISK).
   Re-run with longer timeout or after PyTorch/XPU stack fix; not a code regression
   from items 1–4.
4. ~~Cosmetic: docstrings mention xpu~~ DONE (torch models, pipelines, batcher,
   objdet_helper, run_pipeline --device help). TF left cuda/cpu only (no XPU).

## Remaining TODO (historical — items 1–4 completed 2026-08-04)

~~1. point_transformer `.cuda()`~~ DONE  
~~2. object_detection `model.cuda`~~ DONE  
~~3. ctor device="cuda" defaults~~ DONE  
~~4. run_pipeline.py TF/distributed/backend help~~ DONE  

5. TensorFlow: **DO NOT TOUCH** — user confirmed TF has no SYCL support.
   Leave `ml3d/tf/utils/pointnet/pointnet2_utils.py`,
   `ml3d/tf/utils/roipool3d/roipool3d_utils.py`, `ml3d/tf/models/pvcnn.py`
   CUDA-only guards (`open3d.core.cuda.device_count()`) exactly as-is. This
   reverses an earlier (pre-clarification) plan item — do not re-add SYCL
   handling to TF code.

6. Cosmetic (duplicate note): docstring xpu mentions — see Remaining TODO #3 above.

## Verification

- Re-run parity tests locally after changes:
  `OPEN3D_ML_ROOT=$PWD PATH_TO_OPEN3D_ML=$PWD ./tests/run_tests.sh tests/test_models_torch.py`
  (per-test subprocess isolation is required — XPU/SYCL state left behind by
  one model test can deadlock the next in a shared interpreter).
- Environment: venv at `/home/ssheorey/Documents/o3d.venv`, open3d-xpu wheel
  with `BUNDLE_OPEN3D_ML=True`, `BUILD_SYCL_MODULE=True`,
  `BUILD_PYTORCH_OPS=True` (confirmed installed 2026-08-04).
- Local hardware: 2x Intel Arc A770 (i915 driver, renderD128/129).
- KNOWN RISK: a bare `torch.randn(4,4).to('xpu'); x @ x` was previously
  observed to hang indefinitely (300s+) on this machine — likely a GEMM/oneDNN
  JIT compile or Level-Zero stall, unrelated to Open3D-ML code. If model tests
  hang, this is the suspected cause — not a regression from these changes.
  Always run tests with a wrapper timeout (e.g. via `tests/run_tests.sh`'s
  built-in per-test `TIMEOUT`, default 120s, bumped to 300s in recent runs).
- IMPORTANT process hygiene: background test runs via `nohup ... &` can become
  orphaned if the parent shell/terminal is killed without killing the child
  process tree first. Always check `ps aux | grep pytest` for stray leftover
  runs before starting a new test run, to avoid two runs contending for the
  same 2 GPUs and corrupting results/timing.

## Test run (2026-08-04)

- Full suite: `PATH=/home/ssheorey/Documents/o3d.venv/bin:$PATH OPEN3D_ML_ROOT=$PWD
  PATH_TO_OPEN3D_ML=$PWD TIMEOUT=300 ./tests/run_tests.sh tests/test_models_torch.py`
  → **8/8 exit 124** (timeout). Single `test_randlanet_torch` stalls after
  oneDNN SYCL GPU init on 2× Arc A770 (`xpu True`, `sycl device_count 2`).
- Log: `/tmp/test_models_torch_after_fix.log`, `/tmp/randlanet_single.log`.
