# Raw Video To Report Workflow

This workflow shows the stable CPU-first path from an input movie to QC,
pipeline artifacts, reports, and exports. It uses a tiny synthetic movie so the
commands can run without lab data, Fiji, a GPU, or internet access.

The same command shape applies to real data once the dataset manifest points to
the real TIFF or NumPy stack.

## 1. Create A Synthetic Dataset

Run this from the repository root:

```bash
python -c "from neurobench.data.synthetic import generate_synthetic_calcium_dataset; generate_synthetic_calcium_dataset(include_impulse_artifact=False).write('Outputs/tutorial_raw_video/fixture', dataset_id='tutorial_synthetic')"
```

This writes:

- `Outputs/tutorial_raw_video/fixture/video.npy`
- `Outputs/tutorial_raw_video/fixture/ground_truth.csv`
- `Outputs/tutorial_raw_video/fixture/dataset_manifest.json`

## 2. Validate And QC The Dataset

```bash
python -m neurobench.cli.main dataset validate Outputs/tutorial_raw_video/fixture/dataset_manifest.json

python -m neurobench.cli.main dataset qc \
  Outputs/tutorial_raw_video/fixture/dataset_manifest.json \
  --output Outputs/tutorial_raw_video/qc
```

Outputs:

- `Outputs/tutorial_raw_video/qc/qc_report.json`
- `Outputs/tutorial_raw_video/qc/qc_report.md`

Use the QC report to check frame count, shape, intensity summaries, saturation
risk, and file provenance before running detectors.

## 3. Create A Pipeline Spec

Write `Outputs/tutorial_raw_video/pipeline_spec.json`:

```json
{
  "schema_version": 1,
  "dataset_id": "tutorial_synthetic",
  "run_id": "tutorial_localz_baseline",
  "pipeline": [
    {
      "id": "source",
      "stage_id": "source_video_import",
      "params": {
        "source": "Outputs/tutorial_raw_video/fixture/video.npy"
      }
    },
    {
      "id": "highpass",
      "stage_id": "temporal_highpass_gaussian",
      "params": {
        "sigma_frames": 2.0
      }
    },
    {
      "id": "score",
      "stage_id": "robust_positive_local_z",
      "params": {
        "epsilon": 0.05
      }
    },
    {
      "id": "components",
      "stage_id": "component_filter",
      "params": {
        "seed_z": 1.5,
        "min_area_px": 3,
        "max_area_px": 100
      }
    },
    {
      "id": "rank",
      "stage_id": "heuristic_priority_v1"
    }
  ],
  "artifacts": {}
}
```

## 4. Validate, Dry-Run, And Execute

```bash
python -m neurobench.cli.main run validate Outputs/tutorial_raw_video/pipeline_spec.json

python -m neurobench.cli.main run dry-run \
  --validate-artifacts \
  Outputs/tutorial_raw_video/pipeline_spec.json

python -m neurobench.cli.main run execute \
  Outputs/tutorial_raw_video/pipeline_spec.json \
  --run-root Outputs/tutorial_raw_video/runs/tutorial_localz_baseline
```

Key outputs:

- `pipeline_run.json`
- `artifacts/preprocessing/highpass_video.npy`
- `artifacts/preprocessing/z_stack.npy`
- `artifacts/candidates/roi_candidates.json`
- `artifacts/candidates/ranked_candidates.json`
- `logs/events.jsonl`

## 5. Generate A Run Report

```bash
python -m neurobench.cli.main report generate \
  Outputs/tutorial_raw_video/runs/tutorial_localz_baseline/pipeline_run.json \
  --output Outputs/tutorial_raw_video/reports/tutorial_localz_baseline
```

Outputs:

- `metrics_report.json`
- `report.md`

The report is intentionally explicit about runtime, artifacts, provenance, and
available scientific metrics. For synthetic tests, object/event metrics can be
added later from ground truth or annotation-derived labels.

## 6. Optional Parameter Sweep

Add this `sweep` block to the pipeline spec:

```json
"sweep": {
  "id": "component_thresholds",
  "parameters": [
    {
      "stage": "components",
      "param": "seed_z",
      "values": [1.4, 1.8]
    },
    {
      "stage": "components",
      "param": "min_area_px",
      "values": [3, 5]
    }
  ]
}
```

Then run:

```bash
python -m neurobench.cli.main run sweep \
  Outputs/tutorial_raw_video/pipeline_spec.json \
  --run-root Outputs/tutorial_raw_video/sweep_component_thresholds
```

Outputs:

- `sweep_summary.json`
- `sweep_report.md`
- one run folder per parameter combination

## 7. Gamma CFAR Cascade Screen

For a small adaptive-background screen on the cropped resting video, use the
example cascaded Gamma CFAR sweep:

```bash
python -m neurobench.cli.main run validate examples/gamma_cfar_cascade_sweep.example.json

python -m neurobench.cli.main run dry-run \
  --validate-artifacts \
  examples/gamma_cfar_cascade_sweep.example.json

python -m neurobench.cli.main run sweep \
  examples/gamma_cfar_cascade_sweep.example.json \
  --run-root Outputs/GammaCFAR/calcium_rest_cropped/gamma_cfar_cascade_grid_v1
```

The architecture is:

```text
source -> temporal high-pass -> spatial Gaussian + robust local-z evidence
       -> small-reference Gamma CFAR -> large-reference Gamma CFAR intersection
       -> component extraction -> local background traces
       -> Kalman event scoring -> heuristic ranking
```

The 12-run grid varies:

- small-reference CFAR `pfa`: `[0.03, 0.06, 0.10]`
- large-reference CFAR `training_radius_px`: `[12, 18]`
- component temporal support: `[20, 25]` frames

The component minimum area is fixed at `20 px` for this pass. This keeps the
screen from being dominated by single-frame impulse speckles while still
allowing active compact footprints that may be smaller than full soma masks.

After execution, create a concise review brief:

```bash
python tools/summarize_gamma_cfar_sweep.py \
  Outputs/GammaCFAR/calcium_rest_cropped/gamma_cfar_cascade_grid_v1 \
  --output Outputs/GammaCFAR/calcium_rest_cropped/gamma_cfar_cascade_grid_v1/gamma_cfar_grid_brief.md \
  --pixel-size-um 0.5
```

The brief reports observable burden and plausibility signals rather than
claiming accuracy: CFAR active fraction, candidate count, temporal support,
median equivalent ROI diameter, fraction of candidates in the expected
`5-10 um` diameter range, Kalman event count, and recommended runs for manual
inspection.

## 8. Optional Comparison Report

After at least two run folders exist:

```bash
python -m neurobench.cli.main report compare \
  Outputs/tutorial_raw_video/sweep_component_thresholds/001_tutorial_localz_baseline__sweep_001/pipeline_run.json \
  Outputs/tutorial_raw_video/sweep_component_thresholds/002_tutorial_localz_baseline__sweep_002/pipeline_run.json \
  --output Outputs/tutorial_raw_video/reports/component_threshold_comparison
```

Outputs:

- `comparison_report.json`
- `comparison_report.md`

## 8. Export Reviewed Results

For real reviewed data, use annotation exports once `review_data.json` and
`annotations.json` exist:

```bash
python tools/export_annotations.py \
  --review-data Outputs/NeuronReview/example/app/review_data.json \
  --annotations Outputs/NeuronReview/example/app/annotations.json \
  --out-dir Outputs/NeuronReview/example/exports/accepted_only \
  --profile accepted_only
```

For inverse-dynamics preparation:

```bash
python tools/export_inverse_dynamics.py \
  --review-data Outputs/NeuronReview/example/app/review_data.json \
  --annotations Outputs/NeuronReview/example/app/annotations.json \
  --out-dir Outputs/NeuronReview/example/exports/inverse_dynamics
```

These exporters write `export_bundle.json` so downstream users can see the
selection policy, alignment status, checksums, and provenance.

## Using A Real TIFF

For a real movie, create or edit a dataset manifest so `paths.raw_video` points
to the TIFF:

```json
{
  "schema_version": 1,
  "dataset_id": "my_sample",
  "name": "my_sample.tif",
  "modality": "calcium",
  "frame_rate_hz": 5.0,
  "pixel_size_microns": 0.5,
  "paths": {
    "raw_video": "Inputs/my_sample.tif",
    "app_dir": "Outputs/NeuronReview/my_sample/app",
    "review_data": "Outputs/NeuronReview/my_sample/app/review_data.json"
  }
}
```

The current local executor supports `.npy` video artifacts directly. TIFF stack
processing for the interactive workbench still uses the Fiji/Groovy workflow
documented in [../NEURON_WORKBENCH.md](../NEURON_WORKBENCH.md).

## Interpretation Notes

- Candidate outputs are proposals for review, not ground truth.
- Low frame-rate calcium imaging marks fluorescence transients, not individual
  action potentials.
- Use `accepted_only` exports for downstream modeling unless the goal is audit
  or debugging.
- Alignment status must be `validated` before behavior-coupled inverse dynamics
  should be treated as synchronized.
