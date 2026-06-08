# External Test Gamma-CFAR Architecture Plan

## Goal
Run a GPU-first overnight architecture search on the `external_test` green-excess video. The goal is to improve ROI detection for neurons that fluoresce green, including neurons that are persistent and rarely turn fully off but still brighten when active.

## Current GPU Preflight Status
- Hardware is visible: `NVIDIA GeForce RTX 4070 SUPER` appears in `lspci`.
- CUDA runtime is not usable right now: `nvidia-smi` cannot communicate with the NVIDIA driver.
- Device nodes are missing right now: `/dev/nvidiactl` and `/dev/nvidia0` are absent.
- The overnight runner is intentionally GPU-only. If CuPy/CUDA cannot access a CUDA device, it stops before heavy work. It does not silently fall back to CPU processing.

Do not start the overnight run until `nvidia-smi` works and reports the RTX 4070 SUPER.

## Architecture To Test
This sweep uses a green-excess, multiscale Gamma-CFAR architecture:

1. `source`: use `Outputs/GammaCFAR/external_test/external_test_green_excess.npy`.
2. `highpass`: no temporal high-pass baseline subtraction (`sigma_frames=0.0`) so persistent active neurons are not removed.
3. `smooth`: spatial Gaussian smoothing (`sigma_px=0.8`) on GPU.
4. `score`: robust positive local-z stack on GPU.
5. `cfar_small_ref`: permissive small-reference Gamma-CFAR mask.
6. `cfar_large_ref`: larger-reference Gamma-CFAR mask fused with the small-reference mask.
7. `components`: union the fused CFAR temporal-support footprint with a persistent green projection-blob footprint.
8. `traces/events/activity_states`: extract corrected traces, score transient peaks, and also mark sustained/tonic activity.
9. `rank`: rank ROI candidates for review.

## Sweep Size
The sweep expands to 144 runs:

- `cfar_small_ref.pfa`: `0.02`, `0.04`, `0.08`
- `cfar_small_ref.training_radius_px`: `6`, `8`
- `cfar_large_ref.training_radius_px`: `18`, `24`
- `components.fusion_mode`: `intersection`, `union`
- `components.support_min_frames`: `1`, `15`, `30`
- `components.projection_blob_z`: `1.5`, `2.0`

Each run name is descriptive. Example:

```text
green_roi_mscfar_v3_pfa004_sR06_lR18_union_sup015_pz15
```

This encodes pfa, small reference radius, large reference radius, fusion mode, support frame threshold, and projection threshold.

## Safety Limits
The runner checks these before heavy work:

- At least 150 GiB free disk at the sweep root.
- At least 16 GiB available system RAM.
- CuPy can import, find a CUDA device, allocate a tiny test array, and run a CUDA `uniform_filter` smoke test.

Processing is chunked:

- GPU preprocessing default: 32 frames per chunk.
- GPU CFAR and mask fusion default: 32 frames per chunk.
- CFAR retries smaller chunks of 16 then 8 frames on GPU out-of-memory.
- Heavy array math is CUDA/CuPy. CPU is used for orchestration, JSON, disk I/O, SciPy connected components on 2D maps, and small diagnostic summaries.

## Step 1: Verify GPU Is Ready
Run:

```bash
nvidia-smi
```

Expected: it should list the RTX 4070 SUPER and current VRAM. If it fails, stop and fix the NVIDIA driver/module state first.

Optional device-node check:

```bash
ls -l /dev/nvidiactl /dev/nvidia0
```

Activate the GPU environment before running the Python commands:

```bash
conda activate gpu_pipeline
python -c "import cupy; print(cupy.__version__)"
```

The repository `environment.yml` declares `gpu_pipeline` with NumPy, SciPy, CuPy, CUDA runtime packages, and CUDA-enabled PyTorch. If a different environment is used, it must provide NumPy, SciPy, and CuPy with a working CUDA driver.

## Step 2: Write The Sweep Spec
Run from the repository root:

```bash
python3 tools/prepare_gamma_cfar_workbench_run.py write-green-excess-multiscale-cfar-spec   --dataset-id external_test   --run-id green_excess_multiscale_cfar_v3   --source-npy Outputs/GammaCFAR/external_test/external_test_green_excess.npy   --out Outputs/GammaCFAR/external_test/green_excess_multiscale_cfar_v3.spec.json
```

This command is lightweight. It only writes the JSON spec and validates that the sweep expands to 144 planned runs.

## Step 3: Start The Overnight GPU Run
Run:

```bash
mkdir -p Outputs/GammaCFAR/external_test/green_excess_multiscale_cfar_v3
PYTHONUNBUFFERED=1 nice -n 10 ionice -c2 -n7 python3 tools/prepare_gamma_cfar_workbench_run.py run-green-excess-multiscale-cfar-grid   --spec Outputs/GammaCFAR/external_test/green_excess_multiscale_cfar_v3.spec.json   --sweep-root Outputs/GammaCFAR/external_test/green_excess_multiscale_cfar_v3   --source-npy Outputs/GammaCFAR/external_test/external_test_green_excess.npy   --gpu-cfar-chunk-frames 32   --gpu-preprocess-chunk-frames 32   > Outputs/GammaCFAR/external_test/green_excess_multiscale_cfar_v3/overnight.log 2>&1
```

If the GPU runs out of memory, rerun with smaller chunks:

```bash
PYTHONUNBUFFERED=1 nice -n 10 ionice -c2 -n7 python3 tools/prepare_gamma_cfar_workbench_run.py run-green-excess-multiscale-cfar-grid   --spec Outputs/GammaCFAR/external_test/green_excess_multiscale_cfar_v3.spec.json   --sweep-root Outputs/GammaCFAR/external_test/green_excess_multiscale_cfar_v3   --source-npy Outputs/GammaCFAR/external_test/external_test_green_excess.npy   --gpu-cfar-chunk-frames 16   --gpu-preprocess-chunk-frames 16   > Outputs/GammaCFAR/external_test/green_excess_multiscale_cfar_v3/overnight.log 2>&1
```

## Step 4: Monitor The Run
Watch progress:

```bash
tail -f Outputs/GammaCFAR/external_test/green_excess_multiscale_cfar_v3/overnight.log
```

Check summary while it is running:

```bash
python3 -m json.tool Outputs/GammaCFAR/external_test/green_excess_multiscale_cfar_v3/sweep_summary.json | tail -80
```

Check disk usage:

```bash
du -sh Outputs/GammaCFAR/external_test/green_excess_multiscale_cfar_v3
```

Check GPU load:

```bash
nvidia-smi
```

## Resume Behavior
The runner is resumable at the run level:

- Completed runs with an existing `pipeline_run.json` are skipped.
- Shared preprocessing and shared CFAR masks are reused if present.
- `sweep_summary.json` is rewritten incrementally after each run.

If the machine reboots or the process stops, rerun the same Step 3 command.

## Step 5: Attach Results To The Review App
After the sweep finishes:

```bash
python3 tools/prepare_gamma_cfar_workbench_run.py attach-sweep   --dataset-id external_test   --spec Outputs/GammaCFAR/external_test/green_excess_multiscale_cfar_v3.spec.json   --sweep-root Outputs/GammaCFAR/external_test/green_excess_multiscale_cfar_v3   --app-dir Outputs/NeuronReview/external_test/app   --frame-count 500   --merge-existing
```

Then open or refresh the review app at the local app server.

## Success Criteria
A useful run should show:

- Better diagnostic bright-blob coverage on frames 132, 320, and 402.
- ROI counts that are reviewable, preferably under 300.
- Persistent bright neurons represented through projection-blob or union-source candidates.
- Activity summaries with sustained/tonic intervals, not only spike-like events.
- Fewer obvious vessel/background artifacts than the looser single-CFAR sweeps.

## What To Compare In Review
Prioritize these comparisons:

- `intersection` versus `union` fusion at the same pfa/radii/support settings.
- Small radius `6` versus `8` for compact neuron sensitivity.
- Large radius `18` versus `24` for local-background stability.
- `support_min_frames=1` for brief events versus `15` or `30` for persistent active neurons.
- `projection_blob_z=1.5` versus `2.0` for persistent green ROI recovery versus artifact control.

## Expected Outputs
Main outputs:

- `Outputs/GammaCFAR/external_test/green_excess_multiscale_cfar_v3.spec.json`
- `Outputs/GammaCFAR/external_test/green_excess_multiscale_cfar_v3/sweep_summary.json`
- `Outputs/GammaCFAR/external_test/green_excess_multiscale_cfar_v3/gamma_cfar_grid_brief.md`
- Per-run `pipeline_run.json`, candidate masks, ROI candidates, traces, events, and ranked candidates.

Shared outputs:

- Shared GPU preprocessing arrays under the first run's `artifacts/preprocessing/` folder.
- Shared small/large CFAR masks under `shared_masks/`.
