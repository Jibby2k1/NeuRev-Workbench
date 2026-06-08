# Template Grid Workflow

This workflow is the CPU-first preprocessing path for the revised zebrafish
grid dynamics experiments. It keeps the existing neuron/ROI review workbench as
optional QC support while adding a template-aligned 32x32 grid representation.

The current data convention is:

```text
{1,2,3,4,5,6,7,8,9}_{left,right,neutral}.tif
```

Each file is treated as one independent video and one independent fish. Splits
must stay at the video level.

## Scope

This workflow does:

- parse filename-derived `neutral`, `left`, and `right` labels
- build a mean template from one selected reference video
- reject obvious outlier frames before template projection when enabled
- estimate one per-video rigid transform into template coordinates
- apply that transform to every frame
- pool each registered video into a fixed 32x32 grid state sequence
- write JSON metadata, NumPy arrays, TSV summaries, and dashboard PNG previews

This workflow does not implement stimulation metadata, inverse control,
nonrigid registration, optical flow, or transformer temporal models.

## 1. Build The Video Manifest

```bash
python -m neurobench.cli.main video manifest \
  --input-dir Inputs/ZebrafishVideos \
  --pattern "^(?P<index>[1-9])_(?P<label>left|right|neutral)\\.tiff?$" \
  --out Outputs/GridModel/manifest/video_manifest.json
```

The manifest records `video_id`, path, label, index, fish id, condition, frame
count, shape, dtype, and label counts. Unexpected filenames are reported as
warnings unless `--strict` is used.

## 1.5 Safety Preflight

Before any pixel-processing stage on real data, estimate output size and basic
disk/RAM pressure from the manifest metadata:

```bash
python -m neurobench.cli.main video preflight \
  --manifest Outputs/GridModel/manifest/video_manifest.json \
  --out-dir Outputs/GridModel \
  --out Outputs/GridModel/manifest/preflight.json \
  --rows 32 \
  --cols 32 \
  --chunk-size-frames 64
```

Review `warnings` in `preflight.json` before continuing. The preprocessing
commands below are chunked by default; reduce `--chunk-size-frames` to `16` or
`8` if RAM pressure is reported. Do not increase eager-load or grid-output
safety limits unless the preflight estimate and available disk/RAM support it.

## 2. Build The Template

Use `1_neutral` as the default first reference until the team chooses another
reference video.

```bash
python -m neurobench.cli.main template build-from-video \
  --manifest Outputs/GridModel/manifest/video_manifest.json \
  --reference-video-id 1_neutral \
  --projection-kind mean_after_outlier_rejection \
  --max-outlier-fraction 0.05 \
  --chunk-size-frames 64 \
  --out-dir Outputs/GridModel/template
```

Key outputs:

- `template_spec.json`
- `template_projection.npy`
- `template_projection.png`
- `outlier_frame_scores.tsv`
- `outlier_frame_scores.png`

Review the projection and outlier plot before interpreting registration scores.

## 3. Register Videos

```bash
python -m neurobench.cli.main template register-videos \
  --manifest Outputs/GridModel/manifest/video_manifest.json \
  --template Outputs/GridModel/template/template_spec.json \
  --transform-model rigid \
  --rotation-range-deg -10 10 \
  --rotation-step-deg 0.5 \
  --chunk-size-frames 64 \
  --out-dir Outputs/GridModel/registration
```

Each video gets a `registration_result.json` plus source, registered, overlay,
and residual previews. Low correlation, boundary-angle choices, and large blank
fractions are recorded as warnings for dashboard review.

Keep `--allow-uniform-scale` disabled until overlays show systematic size
differences across videos.

## 4. Apply Registration

```bash
python -m neurobench.cli.main template apply-registration \
  --manifest Outputs/GridModel/manifest/video_manifest.json \
  --template Outputs/GridModel/template/template_spec.json \
  --registration-dir Outputs/GridModel/registration \
  --chunk-size-frames 64 \
  --out-dir Outputs/GridModel/registered
```

Registered videos are written as `.npy` memory maps chunk-by-chunk and saved frame-first as:

```text
Outputs/GridModel/registered/<video_id>/registered_video.npy
```

## 5. Generate The 32x32 Grid

```bash
python -m neurobench.cli.main grid generate \
  --template Outputs/GridModel/template/template_spec.json \
  --rows 32 \
  --cols 32 \
  --out Outputs/GridModel/grid/grid_spec_32x32.json
```

The grid covers the full template image in rectangular image coordinates and
contains exactly 1024 deterministic regions.

## 6. Extract Grid States

```bash
python -m neurobench.cli.main grid extract-states \
  --manifest Outputs/GridModel/manifest/video_manifest.json \
  --registered-dir Outputs/GridModel/registered \
  --grid Outputs/GridModel/grid/grid_spec_32x32.json \
  --features mean_intensity \
  --chunk-size-frames 64 \
  --max-grid-state-bytes 1000000000 \
  --out-dir Outputs/GridModel/grid_states
```

Each video writes:

- `grid_states.npz`
- `region_features.tsv`
- `region_summary.json`
- `grid_preview.png`
- `grid_trace_summary.png`

`grid_states.npz` stores `grid_state` as `[T, 32, 32, C]` and `flat_state` as
`[T, 1024, C]`. The MVP channel is `mean_intensity` with per-video robust
percentile normalization.

## 7. Pipeline Shortcut

The same preprocessing path is available as an executable pipeline spec:

```bash
python -m neurobench.cli.main run validate examples/template_grid_32x32_pipeline.example.json

python -m neurobench.cli.main run dry-run \
  --validate-artifacts \
  examples/template_grid_32x32_pipeline.example.json

python -m neurobench.cli.main run execute \
  examples/template_grid_32x32_pipeline.example.json \
  --run-root Outputs/GridModel/runs/template_grid32_preprocess
```

For synthetic validation, the pipeline stages are tested with generated TIFF
fixtures so the path can run without committing real video data.

## Dashboard Review

Open the existing workbench and use the Data, Review, Progress, and Report
pages. The template-grid additions expose:

- manifest label counts
- template projection and outlier plot
- per-video registration scores and warnings
- source/registered/overlay previews
- 32x32 grid overlay toggles
- selected grid-cell row, column, region id, and bbox
- grid-state and model summary panels when downstream artifacts exist

The dashboard reads JSON and PNG previews. It should not load large `.npy`,
`.npz`, or checkpoint files into the browser.
