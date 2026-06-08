# Goal: Template-Aligned 32×32 Grid Dynamics Workbench for Zebrafish Activity Modeling

## 0. Purpose of This File

This is the Codex-facing implementation plan for the revised NeuRev / NeuroBench project direction.

The project is shifting away from individual-neuron detection as the main near-term milestone. The immediate research path is now:

1. Use zebrafish anatomical robustness to build a template-alignment pre-processing workflow.
2. Register each video to a common template using a simple transform: translation + rotation, with optional uniform scale only if needed.
3. Convert each registered video into a fixed `32 × 32` rectangular grid state sequence.
4. Train an autoencoder + recurrent latent predictor:
   - a CNN encoder maps each grid/image frame to a latent code;
   - an RNN predicts the next latent code;
   - a CNN decoder reconstructs the current input and/or the predicted next input.
5. Later, train a classifier on learned latent codes to distinguish `neutral`, `left`, and `right` activity.
6. Do **not** implement inverse control yet. The grid representation should remain compatible with future regional stimulation, but stimulation/control metadata is not available in the current phase.

The current dataset naming convention is:

```text
{1,2,3,4,5,6,7,8,9}_{left,right,neutral}.tif
```

Examples:

```text
1_left.tif
1_right.tif
1_neutral.tif
...
9_left.tif
9_right.tif
9_neutral.tif
```

For this phase, treat each `.tif` as one independent video and one independent fish. Splits must be by video, never by random frames, to avoid leakage.

---

## 1. Non-Negotiable Project Guardrails

1. Do not remove the existing neuron/ROI dashboard functionality. Re-scope it as optional QC and review support.
2. Keep the existing Gamma/CFAR, Fiji/Groovy, annotation, and dashboard workflows functional unless a task explicitly says otherwise.
3. Do not rewrite the repository. Add new modules and stages through the existing pipeline, schema, artifact, and dashboard extension points.
4. Keep CPU-only QC viable. GPU can be optional for model training, but template generation, registration, grid extraction, and dashboard inspection must work without a GPU.
5. Make every new artifact versioned and inspectable: JSON metadata, checksums where practical, preview PNGs, and reproducible parameters.
6. Registration is per video. The template may be created once from a chosen reference video, then every video receives its own transform into template coordinates.
7. The default grid is `32 × 32`, producing 1024 fixed regions. Future work may use finer grids, but 32×32 is the first target.
8. Current-phase modeling has no stimulation/control input. Use only video-derived states and filename-derived labels.
9. Inverse control is explicitly out of scope for this milestone.
10. Transformer-based temporal models are a later experiment only. The MVP should use an RNN/GRU/LSTM because the dataset is likely small.

---

## 2. Locked Team Decisions Incorporated Into This Plan

### 2.1 Template source

Use the mean projection of one selected reference video as the first anatomical template.

Because noisy or motion-corrupted frames may distort the mean, implement optional outlier-frame rejection before computing the final mean. The template projection kind should be named clearly, for example:

```text
mean_after_outlier_rejection
```

Initial default:

```json
{
  "template_source": "single_reference_video",
  "projection_kind": "mean_after_outlier_rejection",
  "outlier_method": "projection_residual_zscore",
  "max_outlier_fraction": 0.05
}
```

### 2.2 Registration unit

Registration is per video, not per fish atlas, per frame, or per stimulus condition. The current videos are already mostly aligned, but registration should improve consistency across the independent fish/video recordings.

Each video gets:

```text
raw .tif video
  -> robust mean projection
  -> transform into reference-template coordinates
  -> registered video
  -> 32×32 grid sequence
```

### 2.3 Transform family

Use a simple rigid/similarity transform progression:

1. Translation only.
2. Translation + rotation.
3. Translation + rotation + optional uniform scale.

Do not implement affine, shear, elastic, nonrigid, or optical-flow registration in the MVP.

Default transform config:

```json
{
  "transform_model": "rigid",
  "allow_translation": true,
  "allow_rotation": true,
  "allow_uniform_scale": false,
  "rotation_range_deg": [-10.0, 10.0],
  "rotation_step_deg": 0.5,
  "scale_range": [1.0, 1.0]
}
```

If real videos show systematic size differences, turn on optional scale with a narrow range first:

```json
{
  "allow_uniform_scale": true,
  "scale_range": [0.95, 1.05],
  "scale_step": 0.01
}
```

### 2.4 Grid representation

Use a `32 × 32` rectangular grid in registered image/template coordinates.

The grid is not an anatomical parcellation. It is a fixed rectangular downsampling/pooling of the registered video into 1024 spatial regions. Anatomy masks may be shown for QC, but the MVP should not drop cells outside the fish unless explicitly configured.

Preferred representation:

```text
registered video:        [T, H, W]
grid state sequence:     [T, 32, 32, C]
flat region sequence:    [T, 1024, C]
```

For the first model, `C = 1` can be a normalized intensity or `dF/F`-like value. Additional channels can be added later.

### 2.5 Stimulation/control input

There is no stimulation log or stimulation metadata for the current phase. Do not build the MVP around stimulation events.

Instead, the grid representation should remain future-compatible with regional stimulation, but all current dynamics models should run with:

```text
control input = absent
```

### 2.6 Activity labels

The class label comes from the filename suffix:

```text
left
right
neutral
```

These labels are for a later classifier trained on latent codes or latent summaries. The autoencoder/RNN dynamics model itself can be trained self-supervised on the video sequence.

### 2.7 Model vocabulary

Use precise internal names instead of `RCNN`, because `R-CNN` usually refers to region-based object detection.

Preferred internal names:

```text
grid_autoencoder
latent_recurrent_predictor
recurrent_grid_autoencoder
grid_latent_dynamics_model
```

Scientific intent:

```text
CNN encoder -> latent code sequence -> RNN predicts next latent code -> CNN decoder reconstructs frame/grid state
```

### 2.8 Model target

The model should support both:

1. reconstructing the current input through the autoencoder;
2. predicting the next latent code and decoding it into the next grid/image state.

First target:

```text
x_t = 32×32 grid frame
z_t = Encoder(x_t)
recon_t = Decoder(z_t)
z_hat_{t+1} = RNN(z_{t-k+1}, ..., z_t)
x_hat_{t+1} = Decoder(z_hat_{t+1})
```

Primary losses:

```text
L_reconstruct = error(recon_t, x_t)
L_predict     = error(x_hat_{t+1}, x_{t+1})
L_latent      = optional error(z_hat_{t+1}, stopgrad(z_{t+1}))
```

### 2.9 Splitting

Split by video. Do not split individual frames randomly across train/validation/test.

Given 27 expected videos, prefer one of these protocols:

1. Stratified train/validation/test split by filename label.
2. Stratified K-fold by video.
3. Leave-one-video-out evaluation for the classifier when data is very small.

For early smoke tests, use synthetic videos and tiny splits, but real-data claims must be video-split.

---

## 3. Current Repository Fit

The uploaded codebase already has useful extension points. Codex should reuse them.

### 3.1 Files and areas to reuse

```text
neurobench/pipeline_catalog.py
neurobench/pipelines/executor.py
neurobench/pipelines/artifacts.py
neurobench/data/video.py
neurobench/data/synthetic.py
neurobench/workbench/server.py
neurobench/workbench/assets/workbench.html
neurobench/workbench/assets/src/*.js
neurobench/exports/inverse_dynamics.py
schemas/*.schema.json
tests/*
```

Important implications:

- New stages belong in `pipeline_catalog.py` first.
- Local execution belongs in `pipelines/executor.py` after algorithms and schemas exist.
- New artifact metadata should go through the existing artifact/run manifest system.
- The dashboard should be extended, not replaced.
- Existing tests around pipeline catalog, executor, and workbench structure should continue passing.

### 3.2 Existing limitation to fix early

The repository already has `VideoStore` support for `.tif/.tiff`, but some executor paths are `.npy`-oriented. Since the real data is `.tif`, Codex should add a single shared video loader for new stages:

```python
def load_video_array(path: str | Path) -> np.ndarray:
    """Return frame-first [T, H, W] video from .npy, .tif, or .tiff."""
```

Use this loader in the template/grid pipeline stages. Intermediate artifacts can still be saved as `.npy` for speed and reproducibility.

---

## 4. Revised Target Workflow

### 4.1 High-level pipeline

```text
raw .tif videos
  -> dataset manifest + filename label parsing
  -> choose one reference video
  -> build template from robust mean of reference video
  -> for each video:
       -> compute robust mean projection
       -> estimate translation + rotation into template coordinates
       -> apply one per-video transform to all frames
       -> generate registered video artifact
       -> pool/downsample into 32×32 grid states
       -> export grid arrays, feature summaries, and dashboard previews
  -> build video-split modeling dataset
  -> train autoencoder reconstruction model
  -> train/evaluate latent recurrent next-code predictor
  -> export predictions and latent codes
  -> train/evaluate latent-code classifier for neutral/left/right
  -> inspect alignment, grid states, reconstructions, predictions, and classification in dashboard
```

### 4.2 MVP data products

The first useful end-to-end product should generate these files for every run:

```text
manifest/video_manifest.json
template/template_spec.json
template/template_projection.npy
template/template_projection.png
registration/<video_id>/registration_result.json
registration/<video_id>/source_projection.png
registration/<video_id>/registered_projection.png
registration/<video_id>/overlay_before_after.png
registered/<video_id>/registered_video.npy
grid/grid_spec_32x32.json
grid/<video_id>/grid_states.npz
grid/<video_id>/grid_preview.png
grid/<video_id>/grid_trace_summary.png
dynamics/dynamics_dataset.json
dynamics/dynamics_arrays.npz
models/autoencoder_run.json
models/latent_rnn_run.json
models/prediction_examples.json
models/latent_codes.npz
classifier/latent_classifier_run.json
```

The dashboard should load JSON and PNG summaries directly. It should not try to load large `.npy`, `.npz`, or `.pt` files into the browser unless a small preview has been generated.

---

## 5. Dataset Manifest and Filename Parsing

Add a dataset manifest step before template registration.

### 5.1 File pattern

Expected input:

```text
*.tif
*.tiff
```

Expected naming regex:

```regex
^(?P<index>[1-9])_(?P<label>left|right|neutral)\.tiff?$
```

### 5.2 Manifest schema

New file:

```text
schemas/video_manifest.schema.json
```

Example:

```json
{
  "schema_version": 1,
  "dataset_id": "zebrafish_left_right_neutral_v1",
  "created_at": "2026-06-01T00:00:00Z",
  "root": "Inputs/ZebrafishVideos",
  "videos": [
    {
      "video_id": "1_left",
      "path": "Inputs/ZebrafishVideos/1_left.tif",
      "index": 1,
      "label": "left",
      "fish_id": "1_left",
      "condition": "left",
      "frame_count": null,
      "height": null,
      "width": null,
      "dtype": null,
      "frame_rate_hz": null,
      "notes": "Each video is treated as a different fish."
    }
  ],
  "label_set": ["left", "right", "neutral"],
  "split_policy": "by_video",
  "extras": {}
}
```

### 5.3 Acceptance criteria

- Parses all files matching `{1..9}_{left,right,neutral}.tif`.
- Rejects or warns on files that do not match the expected pattern.
- Records labels from filenames.
- Records video dimensions and frame counts when possible without loading all frames into memory.
- Produces label counts in the dashboard.
- Guarantees that split builders operate at the video level.

---

## 6. Template Construction From One Reference Video

### 6.1 Purpose

Build the anatomical template from one selected video. This should be a robust mean projection after removing obvious outlier frames.

### 6.2 Suggested implementation module

```text
neurobench/algorithms/template_matching.py
```

Suggested functions:

```python
def load_video_array(path): ...

def robust_frame_projection(
    video,
    *,
    projection_kind="mean",
    sample_stride=1,
    normalize=True,
): ...

def score_frame_outliers(
    video,
    *,
    method="projection_residual_zscore",
    sample_stride=1,
): ...

def select_template_frames(
    video,
    *,
    max_outlier_fraction=0.05,
    z_threshold=3.5,
): ...

def build_template_from_reference_video(
    video,
    *,
    outlier_method="projection_residual_zscore",
    max_outlier_fraction=0.05,
    projection_kind="mean",
): ...
```

### 6.3 Outlier-frame rejection MVP

Use a simple, inspectable method first:

1. Optionally sample frames for speed.
2. Compute an initial robust reference projection, preferably median or trimmed mean for scoring only.
3. For each frame, compute a scalar deviation score:
   - mean absolute residual from reference;
   - global intensity z-score;
   - optional normalized cross-correlation to the reference.
4. Mark frames as outliers if they exceed a configurable z-score threshold.
5. Cap removal at `max_outlier_fraction`, default `0.05`.
6. Compute the final template as the mean of retained frames.

Do not overcomplicate this at first. The dashboard must show:

- number of frames removed;
- frame indices removed;
- before/after mean projection preview;
- outlier score plot.

### 6.4 Template spec

New file:

```text
schemas/template_spec.schema.json
```

Example:

```json
{
  "schema_version": 1,
  "template_id": "template_from_1_neutral_v1",
  "source_video_id": "1_neutral",
  "source_video_path": "Inputs/ZebrafishVideos/1_neutral.tif",
  "created_at": "2026-06-01T00:00:00Z",
  "coordinate_system": {
    "height": 512,
    "width": 512,
    "origin": "top_left",
    "units": "px"
  },
  "projection": {
    "kind": "mean_after_outlier_rejection",
    "path": "template/template_projection.npy",
    "preview_png": "template/template_projection.png",
    "dtype": "float32",
    "sha256": "..."
  },
  "outlier_rejection": {
    "enabled": true,
    "method": "projection_residual_zscore",
    "max_outlier_fraction": 0.05,
    "removed_frame_indices": [12, 58],
    "removed_fraction": 0.004
  },
  "notes": "Initial anatomical template from one reference video mean projection.",
  "extras": {}
}
```

### 6.5 Acceptance criteria

- Template can be built from one `.tif` video.
- Template can be built from one `.npy` video for synthetic tests.
- Outlier rejection can be disabled.
- Removed frames and parameters are recorded.
- The final template projection has deterministic shape and finite values.
- A PNG preview is created for the dashboard.

---

## 7. Per-Video Registration

### 7.1 Purpose

For each video, estimate one transform that maps that video's anatomical projection into the template coordinate system.

Since fish are already mostly aligned, the first implementation should favor small corrections and strong QC rather than complex registration.

### 7.2 Transform types

Implement in this order:

1. `translation`
2. `rigid`: translation + rotation
3. `similarity`: translation + rotation + optional uniform scale

Avoid affine/nonrigid in the current milestone.

### 7.3 Matching method MVP

A robust but simple method is sufficient:

1. Compute source projection from the video, using the same projection/outlier-rejection logic as the template builder.
2. Normalize source and template projections with robust percentiles.
3. For each candidate rotation angle:
   - rotate source projection around its center;
   - estimate translation using phase correlation or normalized cross-correlation;
   - compute a final normalized cross-correlation score.
4. Pick the best rotation + translation.
5. Optionally search a narrow scale range if `allow_uniform_scale=true`.
6. Emit warnings when confidence is low, transform is extreme, or the best angle is at the search boundary.

### 7.4 Registration result schema

New file:

```text
schemas/registration_result.schema.json
```

Example:

```json
{
  "schema_version": 1,
  "video_id": "3_right",
  "template_id": "template_from_1_neutral_v1",
  "registration_scope": "per_video",
  "source_projection": {
    "kind": "mean_after_outlier_rejection",
    "path": "registration/3_right/source_projection.npy",
    "preview_png": "registration/3_right/source_projection.png"
  },
  "transform": {
    "model": "rigid",
    "matrix_3x3": [
      [0.9994, -0.0349, 2.1],
      [0.0349, 0.9994, -1.7],
      [0.0, 0.0, 1.0]
    ],
    "rotation_deg": 2.0,
    "translation_px": [2.1, -1.7],
    "scale": 1.0
  },
  "score": {
    "normalized_cross_correlation": 0.82,
    "confidence": "ok"
  },
  "qc": {
    "warnings": [],
    "blank_fraction_after_warp": 0.03,
    "best_angle_at_boundary": false
  },
  "artifacts": {
    "registered_projection_png": "registration/3_right/registered_projection.png",
    "overlay_before_after_png": "registration/3_right/overlay_before_after.png",
    "residual_png": "registration/3_right/residual.png"
  },
  "extras": {}
}
```

### 7.5 Applying the transform

Apply the same estimated transform to every frame in the video.

Default output:

```text
registered/<video_id>/registered_video.npy
```

The registered video should be frame-first:

```text
[T, H_template, W_template]
```

Use chunked processing when possible:

```json
{
  "chunk_size_frames": 64,
  "interpolation": "linear",
  "fill_value": 0.0,
  "output_dtype": "float32"
}
```

### 7.6 Acceptance criteria

- Known synthetic translation is recovered within `<= 2 px`.
- Known synthetic rotation is recovered within `<= 2 degrees` at default step.
- Registered video has template height/width.
- Registration result records transform, score, warnings, and previews.
- Low-confidence registration does not silently pass without a dashboard-visible warning.
- Per-video results can be compared side by side in the dashboard.

---

## 8. 32×32 Grid State Extraction

### 8.1 Purpose

Convert each registered video into a fixed-size grid sequence that is both model-friendly and future-compatible with regional stimulation.

The grid should be interpreted as spatial pooling/downsampling, not neuron detection.

### 8.2 Grid spec

New file:

```text
schemas/grid_spec.schema.json
```

Default:

```json
{
  "schema_version": 1,
  "grid_id": "grid_32x32_template_from_1_neutral_v1",
  "template_id": "template_from_1_neutral_v1",
  "rows": 32,
  "cols": 32,
  "region_count": 1024,
  "coordinate_system": {
    "height": 512,
    "width": 512,
    "origin": "top_left",
    "units": "px"
  },
  "bounds": "full_template_image",
  "cell_policy": "rectangular_image_coordinates",
  "regions": [
    {
      "region_id": "R00C00",
      "row": 0,
      "col": 0,
      "bbox": [0, 0, 16, 16],
      "center": [8.0, 8.0],
      "pixel_count": 256,
      "anatomy_fraction": null,
      "anatomy_status": "unknown"
    }
  ],
  "extras": {}
}
```

### 8.3 Pooling/downsampling behavior

Codex should support two equivalent outputs:

1. `grid_states`: model-ready grid tensor.
2. `region_features`: flat region-feature tensor for summaries and exports.

Recommended array files:

```text
grid/<video_id>/grid_states.npz
grid/<video_id>/region_features.tsv
grid/<video_id>/region_summary.json
```

`grid_states.npz` should contain:

```text
grid_state: float32[T, 32, 32, C]
flat_state: float32[T, 1024, C]
region_ids: string[1024]
feature_names: string[C]
time_sec: float32[T] or omitted if frame rate unknown
label: string scalar, one of left/right/neutral
video_id: string scalar
```

MVP feature channels:

```text
mean_intensity
```

Recommended next channels:

```text
robust_normalized_intensity
dff
zscore
```

### 8.4 Pooling method

For trace correctness, prefer area-weighted pooling from registered image pixels into 32×32 cells.

For the CNN input, it is acceptable to use image resizing with area interpolation if it matches the pooling behavior closely. Keep the behavior explicit in metadata:

```json
{
  "grid_extraction_method": "area_pooling",
  "normalization": "per_video_robust_percentile",
  "rows": 32,
  "cols": 32
}
```

### 8.5 Acceptance criteria

- Grid has exactly 1024 deterministic region IDs.
- Grid state shape is `[T, 32, 32, C]`.
- Flat region state shape is `[T, 1024, C]`.
- Region bboxes cover the full template image with no gaps.
- Constant input video produces constant grid state.
- Synthetic region activation appears in the expected grid cell.
- Dashboard can draw the 32×32 grid overlay on registered projection/video preview.

---

## 9. Model Design: Autoencoder + Latent RNN

### 9.1 First model: grid autoencoder

Input:

```text
x_t: [B, C, 32, 32]
```

Encoder:

```text
CNN -> latent vector z_t
```

Decoder:

```text
latent vector z_t -> CNN decoder -> reconstruction x_recon_t
```

Initial architecture should be deliberately small:

```json
{
  "model_kind": "grid_autoencoder",
  "input_shape": [1, 32, 32],
  "latent_dim": 32,
  "encoder_channels": [16, 32],
  "decoder_channels": [32, 16],
  "activation": "relu"
}
```

Primary acceptance:

- Forward pass shape is correct.
- Tiny CPU training run finishes.
- Reconstruction examples are exported as dashboard-friendly images/JSON.
- Reconstruction error is compared across videos and labels.

### 9.2 Second model: latent recurrent predictor

Input:

```text
z_window: [B, W, latent_dim]
```

Output:

```text
z_hat_next: [B, latent_dim]
x_hat_next: Decoder(z_hat_next)
```

Recommended recurrent units:

```text
GRU first, LSTM optional
```

Initial architecture:

```json
{
  "model_kind": "latent_gru_predictor",
  "latent_dim": 32,
  "window_frames": 8,
  "recurrent_unit": "gru",
  "hidden_dim": 64,
  "num_layers": 1,
  "prediction_horizon_frames": 1
}
```

Training options:

1. Train autoencoder first, freeze or partially freeze encoder/decoder, then train RNN on latent codes.
2. Later, fine-tune end-to-end if data volume supports it.

MVP should use option 1 because it is easier to debug.

### 9.3 Losses

Autoencoder:

```text
L_ae = MSE(x_recon_t, x_t) + optional MAE(x_recon_t, x_t)
```

Latent predictor:

```text
L_next_image = MSE(Decoder(z_hat_{t+1}), x_{t+1})
L_next_latent = MSE(z_hat_{t+1}, stopgrad(Encoder(x_{t+1})))
L_total = L_next_image + lambda_latent * L_next_latent
```

Start with:

```json
{
  "lambda_latent": 0.1,
  "loss_image": "mse",
  "loss_latent": "mse"
}
```

### 9.4 Baselines

Before any learned recurrent model is considered useful, implement baselines:

1. Persistence baseline:
   ```text
   x_hat_{t+1} = x_t
   ```
2. Moving average baseline:
   ```text
   x_hat_{t+1} = mean(x_{t-k+1}, ..., x_t)
   ```
3. Optional linear latent autoregression:
   ```text
   z_hat_{t+1} = A z_t
   ```

Every learned model report must show whether it beats persistence on validation/test videos.

### 9.5 Metrics

For autoencoder:

```text
reconstruction MSE
reconstruction MAE
optional SSIM on 32×32 grid image
per-video reconstruction error
per-label reconstruction error
```

For latent RNN prediction:

```text
one-step prediction MSE/MAE
rollout MSE for horizons 1, 3, 5, 10
improvement over persistence
per-video prediction error
per-label prediction error
latent prediction error
```

For dashboard:

```text
input grid frame
reconstructed grid frame
true next grid frame
predicted next grid frame
absolute error heatmap
rollout sequence preview
```

### 9.6 Acceptance criteria

- CPU smoke test trains on synthetic grid sequences.
- Output shapes are deterministic.
- Model can save and reload checkpoint.
- Evaluation writes JSON metrics and prediction examples.
- Prediction examples are small enough for the dashboard.
- Learned model is never reported without persistence baseline comparison.

---

## 10. Latent-Code Classifier for `neutral`, `left`, and `right`

### 10.1 Purpose

After the autoencoder and latent recurrent predictor produce meaningful latent codes, train a classifier to identify activity type:

```text
neutral vs left vs right
```

The classifier should use filename labels, not manual annotations.

### 10.2 Inputs

Possible classifier features:

1. Encoder latent codes `z_t`.
2. RNN hidden states `h_t`.
3. Video-level summaries of codes:
   - mean over time;
   - standard deviation over time;
   - temporal slope/energy;
   - pooled hidden state.

MVP:

```text
z_summary = concat(mean_t(z_t), std_t(z_t))
classifier(z_summary) -> {neutral,left,right}
```

### 10.3 Classifier models

Start simple:

1. Logistic regression or linear classifier on latent summaries.
2. Small MLP on latent summaries.
3. Temporal classifier over code sequences only after baseline classifier exists.

### 10.4 Splitting

The classifier must split by video. With 27 expected videos, prefer stratified cross-validation by label:

```text
fold split unit = video
label balance = neutral/left/right
```

Do not randomly split latent frames from the same video across train/test.

### 10.5 Metrics

```text
accuracy
balanced accuracy
confusion matrix
per-class precision/recall
macro F1
per-fold metrics
```

### 10.6 Dashboard outputs

```text
latent_classifier_run.json
confusion_matrix.png
latent_embedding_2d.png  # PCA or UMAP if available, PCA first
per_video_predictions.tsv
```

Acceptance:

- Classifier baseline runs on synthetic or tiny fixture.
- Real-data classifier uses only video-level split.
- Dashboard can display confusion matrix and per-video predicted label.

---

## 11. Transformer Note

A Transformer or attention-based temporal model may be useful later, but it should not be the first implementation.

Reason:

- expected dataset size is small;
- there may only be around 27 videos initially;
- the first challenge is alignment and stable grid representation, not model capacity;
- Transformer models are easier to overfit and harder to validate with limited data.

Allowed future experiment after the RNN baseline is stable:

```text
latent_code_sequence -> small temporal Transformer encoder -> next latent code
```

Do not implement this before:

1. template registration works on real videos;
2. grid states are stable;
3. persistence baseline exists;
4. latent GRU predictor exists;
5. video-split validation is in place.

---

## 12. New Pipeline Stages

Add these to `neurobench/pipeline_catalog.py`. Wire local runners only after schemas and algorithms have tests.

### Stage A: `video_manifest_build`

Purpose: scan a directory of `.tif/.tiff/.npy` files, parse labels, and write a manifest.

Inputs:

```text
input directory or file list
```

Outputs:

```text
video_manifest.json
label_counts.json
```

Default params:

```json
{
  "filename_regex": "^(?P<index>[1-9])_(?P<label>left|right|neutral)\\.tiff?$",
  "split_unit": "video",
  "labels": ["left", "right", "neutral"]
}
```

### Stage B: `template_build_from_video`

Purpose: build a template from one reference video mean projection with optional outlier-frame rejection.

Outputs:

```text
template_spec.json
template_projection.npy
template_projection.png
outlier_frame_scores.tsv
outlier_frame_scores.png
```

Default params:

```json
{
  "reference_video_id": "1_neutral",
  "projection_kind": "mean_after_outlier_rejection",
  "outlier_rejection": true,
  "outlier_method": "projection_residual_zscore",
  "max_outlier_fraction": 0.05,
  "z_threshold": 3.5
}
```

### Stage C: `template_register_video`

Purpose: estimate per-video transform into template coordinates.

Outputs:

```text
registration_result.json
source_projection.npy
source_projection.png
registered_projection.png
overlay_before_after.png
```

Default params:

```json
{
  "transform_model": "rigid",
  "rotation_range_deg": [-10.0, 10.0],
  "rotation_step_deg": 0.5,
  "allow_uniform_scale": false,
  "metric": "normalized_cross_correlation"
}
```

### Stage D: `apply_video_registration`

Purpose: resample all frames of a video into template coordinates.

Outputs:

```text
registered_video.npy
registered_video_summary.json
registered_projection.png
```

Default params:

```json
{
  "interpolation": "linear",
  "fill_value": 0.0,
  "chunk_size_frames": 64,
  "output_dtype": "float32"
}
```

### Stage E: `grid_32x32_generate`

Purpose: generate a 32×32 rectangular grid spec in template coordinates.

Outputs:

```text
grid_spec_32x32.json
grid_overlay.png
```

Default params:

```json
{
  "rows": 32,
  "cols": 32,
  "bounds": "full_template_image",
  "cell_policy": "rectangular_image_coordinates"
}
```

### Stage F: `grid_state_extract`

Purpose: pool each registered video into a 32×32 grid-state sequence.

Outputs:

```text
grid_states.npz
region_features.tsv
region_summary.json
grid_trace_summary.png
```

Default params:

```json
{
  "rows": 32,
  "cols": 32,
  "features": ["mean_intensity"],
  "normalization": "per_video_robust_percentile",
  "pooling": "area"
}
```

### Stage G: `grid_dynamics_dataset_build`

Purpose: build video-split train/validation/test windows for autoencoder and latent RNN.

Outputs:

```text
dynamics_dataset.json
dynamics_arrays.npz
split_manifest.json
```

Default params:

```json
{
  "window_frames": 8,
  "prediction_horizon_frames": 1,
  "split_unit": "video",
  "split_method": "stratified_by_label",
  "train_fraction": 0.7,
  "val_fraction": 0.15,
  "test_fraction": 0.15
}
```

### Stage H: `grid_autoencoder_train`

Purpose: train the 32×32 grid autoencoder.

Outputs:

```text
autoencoder_run.json
autoencoder_checkpoint.pt
autoencoder_metrics.json
reconstruction_examples.json
training_curve.png
```

Default params:

```json
{
  "latent_dim": 32,
  "epochs": 10,
  "batch_size": 32,
  "learning_rate": 0.001,
  "device": "auto",
  "seed": 7
}
```

### Stage I: `latent_rnn_train`

Purpose: train the RNN next-latent-code predictor using encoder latent sequences.

Outputs:

```text
latent_rnn_run.json
latent_rnn_checkpoint.pt
latent_rnn_metrics.json
prediction_examples.json
rollout_examples.json
```

Default params:

```json
{
  "window_frames": 8,
  "latent_dim": 32,
  "recurrent_unit": "gru",
  "hidden_dim": 64,
  "epochs": 10,
  "batch_size": 32,
  "learning_rate": 0.001,
  "prediction_horizon_frames": 1,
  "device": "auto",
  "seed": 7
}
```

### Stage J: `latent_classifier_train`

Purpose: train a classifier on latent code summaries to predict `neutral`, `left`, or `right`.

Outputs:

```text
latent_classifier_run.json
per_video_predictions.tsv
confusion_matrix.png
latent_embedding_2d.png
```

Default params:

```json
{
  "feature_source": "encoder_latent_codes",
  "summary": ["mean", "std"],
  "classifier": "logistic_regression",
  "split_unit": "video",
  "evaluation": "stratified_kfold"
}
```

### Deferred stages

These should remain planned/future, not part of the MVP:

```text
stimulation_plan_import
future_inverse_control_export
temporal_transformer_train
nonrigid_registration
```

---

## 13. Artifact Contracts

### 13.1 `video_manifest.json`

Core fields:

```text
schema_version
dataset_id
root
videos[]
label_set
split_policy
extras
```

Each video:

```text
video_id
path
index
label
fish_id
frame_count
height
width
dtype
frame_rate_hz
```

### 13.2 `template_spec.json`

Core fields:

```text
schema_version
template_id
source_video_id
coordinate_system
projection
outlier_rejection
notes
extras
```

### 13.3 `registration_result.json`

Core fields:

```text
schema_version
video_id
template_id
registration_scope
source_projection
transform
score
qc
artifacts
extras
```

### 13.4 `grid_spec_32x32.json`

Core fields:

```text
schema_version
grid_id
template_id
rows
cols
region_count
coordinate_system
bounds
cell_policy
regions[]
extras
```

### 13.5 `grid_states.npz`

Required arrays:

```text
grid_state: float32[T, 32, 32, C]
flat_state: float32[T, 1024, C]
region_ids: string[1024]
feature_names: string[C]
```

Required metadata:

```text
video_id
label
normalization
source_registered_video
```

### 13.6 `dynamics_dataset.json`

Core fields:

```text
schema_version
dataset_id
grid_id
source_videos[]
array_path
input_shape
latent_shape if known
windowing
splits
normalization
warnings
extras
```

Example:

```json
{
  "schema_version": 1,
  "dataset_id": "zebrafish_grid32_v1",
  "grid_id": "grid_32x32_template_from_1_neutral_v1",
  "source_videos": ["1_left", "1_right", "1_neutral"],
  "array_path": "dynamics/dynamics_arrays.npz",
  "input_shape": [1, 32, 32],
  "windowing": {
    "window_frames": 8,
    "prediction_horizon_frames": 1,
    "stride_frames": 1
  },
  "splits": {
    "split_unit": "video",
    "split_method": "stratified_by_label",
    "train_video_ids": ["1_left", "2_left"],
    "val_video_ids": ["8_left"],
    "test_video_ids": ["9_left"]
  },
  "warnings": [],
  "extras": {}
}
```

### 13.7 `autoencoder_run.json`

Core fields:

```text
schema_version
run_id
model_kind
input_shape
latent_dim
training_config
source_dataset
checkpoint_path
metrics_path
reconstruction_examples_path
seed
device
warnings
extras
```

### 13.8 `latent_rnn_run.json`

Core fields:

```text
schema_version
run_id
model_kind
latent_dim
window_frames
prediction_horizon_frames
recurrent_unit
hidden_dim
source_autoencoder_run
source_dataset
checkpoint_path
metrics_path
prediction_examples_path
baseline_metrics_path
seed
device
warnings
extras
```

### 13.9 `latent_classifier_run.json`

Core fields:

```text
schema_version
run_id
label_set
feature_source
split_unit
split_method
metrics
confusion_matrix_path
per_video_predictions_path
embedding_preview_path
warnings
extras
```

---

## 14. Dashboard Plan

### 14.1 Keep existing pages

Do not create a separate app. Add panels and overlays to the existing dashboard.

### 14.2 Data page additions

Add a `Template / Registration / Grid` inspection section with:

- video manifest summary;
- label counts for `neutral`, `left`, `right`;
- selected reference video;
- template projection preview;
- outlier frame count and score plot;
- per-video registration table;
- source projection / registered projection / overlay preview;
- transform parameters: translation, rotation, optional scale;
- registration score and warnings;
- grid preview over registered projection.

### 14.3 Review page additions

Add overlay toggles:

```text
Template overlay
Registered projection overlay
32×32 grid overlay
Grid cell intensity overlay
Prediction error overlay
```

Add region selection:

- click a grid cell;
- show `region_id`, row, col, bbox;
- show current grid value;
- show trace over time for selected cell;
- show whether that cell is high-error in model predictions.

### 14.4 Experiment Lab additions

Add controls/sweeps for:

```text
reference video choice
outlier rejection enabled/disabled
max outlier fraction
rotation range
rotation step
allow scale yes/no
grid size, default 32×32
normalization method
latent dimension
RNN window length
RNN hidden dimension
classifier type
```

Guardrails:

- mark model sweeps as potentially expensive;
- default to short CPU smoke runs;
- always show split unit: `video`.

### 14.5 Progress page gates

Add gate cards:

1. Manifest parsed.
2. Template built.
3. Outlier rejection reviewed.
4. Per-video registration complete.
5. Registration warnings reviewed.
6. 32×32 grid states generated.
7. Video-split dynamics dataset built.
8. Persistence baseline evaluated.
9. Autoencoder trained/evaluated.
10. Latent RNN trained/evaluated.
11. Latent classifier evaluated.

### 14.6 Report page additions

Add sections:

```text
Dataset manifest
Template construction
Registration summary
Grid extraction summary
Autoencoder reconstruction metrics
Latent RNN prediction metrics
Persistence baseline comparison
Latent classifier metrics
Known limitations
```

### 14.7 Dashboard files likely to change

```text
neurobench/workbench/assets/workbench.html
neurobench/workbench/assets/workbench.css
neurobench/workbench/assets/src/10_state_persistence.js
neurobench/workbench/assets/src/20_review_core.js
neurobench/workbench/assets/src/25_review_controls.js
neurobench/workbench/assets/src/30_architecture_lab.js
neurobench/workbench/assets/src/40_experiment_lab.js
neurobench/workbench/assets/src/60_metrics_report.js
neurobench/workbench/assets/src/70_dataset_qc.js
```

If adding a new source module such as `75_template_grid.js`, update:

```text
tools/build_workbench_assets.py
```

After dashboard edits, run:

```bash
python tools/build_workbench_assets.py
python tools/build_workbench_assets.py --check
python -m pytest -q tests/test_workbench_structure.py tests/test_workbench_assets.py
```

---

## 15. CLI Plan

Add CLI commands without breaking existing commands.

### 15.1 Manifest

```bash
neurobench video manifest \
  --input-dir Inputs/ZebrafishVideos \
  --pattern "^(?P<index>[1-9])_(?P<label>left|right|neutral)\\.tiff?$" \
  --out Outputs/GridModel/manifest/video_manifest.json
```

### 15.2 Template

```bash
neurobench template build-from-video \
  --manifest Outputs/GridModel/manifest/video_manifest.json \
  --reference-video-id 1_neutral \
  --projection-kind mean_after_outlier_rejection \
  --max-outlier-fraction 0.05 \
  --out-dir Outputs/GridModel/template
```

### 15.3 Registration

```bash
neurobench template register-videos \
  --manifest Outputs/GridModel/manifest/video_manifest.json \
  --template Outputs/GridModel/template/template_spec.json \
  --transform-model rigid \
  --rotation-range-deg -10 10 \
  --rotation-step-deg 0.5 \
  --out-dir Outputs/GridModel/registration
```

### 15.4 Apply registration

```bash
neurobench template apply-registration \
  --manifest Outputs/GridModel/manifest/video_manifest.json \
  --template Outputs/GridModel/template/template_spec.json \
  --registration-dir Outputs/GridModel/registration \
  --out-dir Outputs/GridModel/registered
```

### 15.5 Grid extraction

```bash
neurobench grid generate \
  --template Outputs/GridModel/template/template_spec.json \
  --rows 32 \
  --cols 32 \
  --out Outputs/GridModel/grid/grid_spec_32x32.json

neurobench grid extract-states \
  --manifest Outputs/GridModel/manifest/video_manifest.json \
  --registered-dir Outputs/GridModel/registered \
  --grid Outputs/GridModel/grid/grid_spec_32x32.json \
  --features mean_intensity \
  --out-dir Outputs/GridModel/grid_states
```

### 15.6 Dynamics

```bash
neurobench dynamics build-dataset \
  --manifest Outputs/GridModel/manifest/video_manifest.json \
  --grid-states-dir Outputs/GridModel/grid_states \
  --split-unit video \
  --split-method stratified_by_label \
  --window-frames 8 \
  --prediction-horizon-frames 1 \
  --out-dir Outputs/GridModel/dynamics

neurobench dynamics train-autoencoder \
  --dataset Outputs/GridModel/dynamics/dynamics_dataset.json \
  --latent-dim 32 \
  --epochs 10 \
  --out-dir Outputs/GridModel/models/autoencoder_v1

neurobench dynamics train-latent-rnn \
  --dataset Outputs/GridModel/dynamics/dynamics_dataset.json \
  --autoencoder-run Outputs/GridModel/models/autoencoder_v1/autoencoder_run.json \
  --window-frames 8 \
  --hidden-dim 64 \
  --epochs 10 \
  --out-dir Outputs/GridModel/models/latent_rnn_v1
```

### 15.7 Classifier

```bash
neurobench dynamics train-latent-classifier \
  --dataset Outputs/GridModel/dynamics/dynamics_dataset.json \
  --autoencoder-run Outputs/GridModel/models/autoencoder_v1/autoencoder_run.json \
  --labels-from manifest \
  --split-unit video \
  --evaluation stratified_kfold \
  --out-dir Outputs/GridModel/classifier/latent_classifier_v1
```

---

## 16. Example Pipeline Specs

### 16.1 Template-grid preprocessing pipeline

Add:

```text
examples/template_grid_32x32_pipeline.example.json
```

Example:

```json
{
  "schema_version": 1,
  "dataset_id": "zebrafish_grid32_v1",
  "run_id": "template_grid32_preprocess",
  "pipeline": [
    {
      "id": "manifest",
      "stage_id": "video_manifest_build",
      "params": {
        "input_dir": "Inputs/ZebrafishVideos",
        "filename_regex": "^(?P<index>[1-9])_(?P<label>left|right|neutral)\\.tiff?$"
      }
    },
    {
      "id": "template",
      "stage_id": "template_build_from_video",
      "params": {
        "reference_video_id": "1_neutral",
        "projection_kind": "mean_after_outlier_rejection",
        "max_outlier_fraction": 0.05
      }
    },
    {
      "id": "registration",
      "stage_id": "template_register_video",
      "params": {
        "transform_model": "rigid",
        "rotation_range_deg": [-10.0, 10.0],
        "rotation_step_deg": 0.5,
        "allow_uniform_scale": false
      }
    },
    {
      "id": "registered_video",
      "stage_id": "apply_video_registration",
      "params": {
        "chunk_size_frames": 64,
        "output_dtype": "float32"
      }
    },
    {
      "id": "grid",
      "stage_id": "grid_32x32_generate",
      "params": {
        "rows": 32,
        "cols": 32
      }
    },
    {
      "id": "grid_states",
      "stage_id": "grid_state_extract",
      "params": {
        "features": ["mean_intensity"],
        "normalization": "per_video_robust_percentile",
        "pooling": "area"
      }
    }
  ]
}
```

### 16.2 Dynamics model pipeline

Add:

```text
examples/grid_latent_dynamics_pipeline.example.json
```

Example:

```json
{
  "schema_version": 1,
  "dataset_id": "zebrafish_grid32_v1",
  "run_id": "grid32_latent_dynamics_smoke",
  "pipeline": [
    {
      "id": "dataset",
      "stage_id": "grid_dynamics_dataset_build",
      "params": {
        "window_frames": 8,
        "prediction_horizon_frames": 1,
        "split_unit": "video",
        "split_method": "stratified_by_label"
      }
    },
    {
      "id": "autoencoder",
      "stage_id": "grid_autoencoder_train",
      "params": {
        "latent_dim": 32,
        "epochs": 2,
        "batch_size": 8
      }
    },
    {
      "id": "latent_rnn",
      "stage_id": "latent_rnn_train",
      "params": {
        "window_frames": 8,
        "hidden_dim": 64,
        "epochs": 2,
        "batch_size": 8
      }
    },
    {
      "id": "classifier",
      "stage_id": "latent_classifier_train",
      "params": {
        "classifier": "logistic_regression",
        "split_unit": "video"
      }
    }
  ]
}
```

Keep training epochs tiny in example configs so CI and CPU smoke tests remain viable.

---

## 17. Synthetic Fixtures and Tests

Synthetic tests are mandatory because real data labels and alignment landmarks may be unavailable.

### 17.1 Synthetic fixture generator

Add:

```text
neurobench/data/synthetic_fish.py
```

Suggested API:

```python
def generate_synthetic_grid_fish_videos(
    *,
    video_count_per_label=3,
    labels=("left", "right", "neutral"),
    frames=64,
    height=96,
    width=128,
    grid_rows=32,
    grid_cols=32,
    rotation_deg_range=(-5.0, 5.0),
    translation_px_range=(-4.0, 4.0),
    noise_sigma=0.05,
    seed=7,
) -> SyntheticGridFishBundle:
    ...
```

It should generate:

- fish-like anatomical projection;
- noisy videos;
- filename-compatible video IDs;
- known per-video transforms;
- known label-dependent activity patterns;
- optional outlier frames for template tests;
- expected grid-state arrays for validation.

### 17.2 Tests to add

```text
tests/test_video_manifest.py
tests/test_template_building.py
tests/test_template_registration.py
tests/test_grid_state_extraction.py
tests/test_grid_dynamics_dataset.py
tests/test_grid_autoencoder.py
tests/test_latent_rnn.py
tests/test_latent_classifier.py
```

### 17.3 Key test cases

Manifest:

- parses `1_left.tif` correctly;
- rejects or warns on `foo.tif`;
- counts labels;
- enforces video-level split unit.

Template:

- builds mean projection;
- removes synthetic outlier frames;
- records removed frame indices;
- writes template spec and preview PNG.

Registration:

- recovers translation in synthetic data;
- recovers small rotation in synthetic data;
- warns when best rotation is at boundary;
- writes registration result JSON.

Grid:

- creates exactly 1024 regions;
- covers template image without gaps;
- creates `[T, 32, 32, C]` arrays;
- known activated region maps to expected grid cell.

Dynamics dataset:

- windows do not cross video boundaries;
- train/validation/test splits are by video;
- label distribution is recorded;
- no frame leakage across splits.

Autoencoder:

- forward pass shape;
- tiny CPU train run has finite loss;
- checkpoint save/load works;
- reconstruction examples exported.

Latent RNN:

- latent sequence shape;
- predicted latent shape;
- decoded next-frame shape;
- persistence baseline comparison exists;
- rollout examples exported.

Classifier:

- video-level summaries generated;
- split is by video;
- confusion matrix shape is `3×3`;
- per-video predictions written.

---

## 18. Parallel Workstreams

### Workstream A: Manifest, schemas, and contracts

Can start immediately.

Tasks:

- Add `video_manifest.schema.json`.
- Add template, registration, grid, dynamics, model-run, and classifier schemas.
- Add dataclass or pydantic-style model helpers consistent with repository patterns.
- Add examples.

Validation:

```bash
python -m pytest -q tests/test_schema_validation.py tests/test_video_manifest.py
```

### Workstream B: Video I/O and synthetic data

Can start immediately.

Tasks:

- Add shared `.tif/.tiff/.npy` video loader.
- Add synthetic fish/grid video bundle generator.
- Add filename-compatible synthetic data writer.

Validation:

```bash
python -m pytest -q tests/test_synthetic_fish.py tests/test_video_manifest.py
```

### Workstream C: Template construction and registration

Can start after basic video loader exists.

Tasks:

- Build template from one video.
- Implement outlier-frame rejection.
- Implement translation and rigid registration.
- Apply per-video transform.
- Generate registration previews.

Validation:

```bash
python -m pytest -q tests/test_template_building.py tests/test_template_registration.py
```

### Workstream D: 32×32 grid extraction

Can start after template coordinate shape is defined.

Tasks:

- Generate 32×32 grid spec.
- Pool registered videos into grid states.
- Export grid arrays and TSV summaries.
- Add grid previews.

Validation:

```bash
python -m pytest -q tests/test_grid_state_extraction.py
```

### Workstream E: Pipeline integration

Should wait until A-D have core tests.

Tasks:

- Add new stage IDs to catalog.
- Wire local runners incrementally.
- Register artifacts and previews.
- Add example pipeline specs.

Validation:

```bash
python -m pytest -q tests/test_pipeline_catalog.py tests/test_pipeline_executor.py tests/test_template_grid_pipeline_stages.py
```

### Workstream F: Dashboard integration

Can begin with placeholder panels after artifact names are stable.

Tasks:

- Add manifest/template/registration/grid panels.
- Add 32×32 grid overlay.
- Add selected cell trace panel.
- Add model reconstruction/prediction view.
- Add classifier summary view.

Validation:

```bash
python tools/build_workbench_assets.py --check
python -m pytest -q tests/test_workbench_structure.py tests/test_workbench_assets.py
```

### Workstream G: Dynamics dataset and baselines

Can start after grid states exist.

Tasks:

- Build video-split dynamics arrays.
- Implement persistence and moving-average baselines.
- Add metrics export.

Validation:

```bash
python -m pytest -q tests/test_grid_dynamics_dataset.py tests/test_dynamics_baselines.py
```

### Workstream H: Autoencoder and latent RNN

Should start after G is stable.

Tasks:

- Implement grid autoencoder.
- Implement latent GRU predictor.
- Add train/evaluate loops.
- Add checkpointing.
- Add prediction examples.

Validation:

```bash
python -m pytest -q tests/test_grid_autoencoder.py tests/test_latent_rnn.py tests/test_dynamics_training_smoke.py
```

### Workstream I: Latent classifier

Can start after latent codes are exported.

Tasks:

- Build video-level latent summaries.
- Train logistic regression / simple classifier.
- Export confusion matrix and per-video predictions.

Validation:

```bash
python -m pytest -q tests/test_latent_classifier.py
```

---

## 19. Milestone Checkpoints

### Checkpoint 0: Baseline preservation

Run before new feature work:

```bash
python -m pytest -q tests/test_pipeline_catalog.py tests/test_pipeline_executor.py tests/test_workbench_structure.py
```

Exit criteria:

- Existing targeted tests pass.
- Existing failures, if any, are documented before feature changes.

### Checkpoint 1: Manifest and template

Exit criteria:

- `.tif` videos are discovered and parsed.
- Label counts are correct.
- A selected reference video produces a template projection.
- Outlier-frame rejection report is inspectable.

### Checkpoint 2: Per-video registration

Exit criteria:

- Every video has a `registration_result.json`.
- Every video has source/registered/overlay previews.
- Transform parameters are plausible.
- Warnings are visible.

### Checkpoint 3: 32×32 grid extraction

Exit criteria:

- Every registered video has grid states.
- Shape is `[T, 32, 32, C]`.
- Grid preview is visible in the dashboard.
- Selected cell traces can be inspected.

### Checkpoint 4: Pipeline and dashboard integration

Exit criteria:

- Pipeline examples validate.
- Local executor runs synthetic template-grid pipeline.
- Dashboard shows template, registration, and grid outputs.

### Checkpoint 5: Dynamics dataset and baselines

Exit criteria:

- Video-level splits are built.
- Windows do not cross video boundaries.
- Persistence baseline metrics are written.
- Dashboard/report shows baseline metrics.

### Checkpoint 6: Autoencoder

Exit criteria:

- Autoencoder CPU smoke training works.
- Reconstruction examples are exported.
- Reconstruction metrics are per-video and aggregate.

### Checkpoint 7: Latent RNN

Exit criteria:

- Latent codes are exported.
- RNN predicts next latent code.
- Decoded next-frame predictions are exported.
- Model is compared against persistence.

### Checkpoint 8: Latent classifier

Exit criteria:

- Classifier predicts `neutral`, `left`, `right` from latent summaries.
- Evaluation is by video.
- Confusion matrix and per-video predictions are visible.

### Checkpoint 9: Real-data pilot

Exit criteria:

- Run the complete preprocessing path on the real `{1..9}_{left,right,neutral}.tif` dataset.
- Team reviews template choice, registration overlays, and 32×32 grid states.
- Record whether optional scale is necessary.
- Record whether the latent code classifier separates the three conditions better than chance.

---

## 20. Detailed Codex Backlog

### P0: Planning file

#### TG-000: Add this file

Files:

```text
goal.md
```

Acceptance:

- File exists at repo root.
- It reflects the current decisions: one-video mean template, per-video rigid registration, 32×32 grid, no stimulation metadata, latent RNN prediction, split by video.

### P1: Manifest and video loading

#### TG-010: Add shared video loader

Files:

```text
neurobench/data/video.py
neurobench/pipelines/executor.py
tests/test_video_loading.py
```

Acceptance:

- Loads `.npy`, `.tif`, `.tiff` into frame-first arrays.
- Detects unsupported shapes with readable errors.
- Existing `.npy` tests still pass.

#### TG-011: Add video manifest builder

Files:

```text
schemas/video_manifest.schema.json
neurobench/data/video_manifest.py
neurobench/cli/video.py
tests/test_video_manifest.py
examples/video_manifest.example.json
```

Acceptance:

- Parses filename labels.
- Records video IDs and paths.
- Writes JSON manifest.
- Produces label counts.

### P1: Template and registration contracts

#### TG-020: Add template schema/model

Files:

```text
schemas/template_spec.schema.json
neurobench/models/template.py
tests/test_template_models.py
examples/template_spec.example.json
```

Acceptance:

- Validates example.
- Preserves extras.
- Records source video and outlier rejection.

#### TG-021: Add registration schema/model

Files:

```text
schemas/registration_result.schema.json
neurobench/models/registration.py
tests/test_registration_model.py
examples/registration_result.example.json
```

Acceptance:

- Validates rigid transform example.
- Records score, QC warnings, preview paths.

### P1: Template algorithms

#### TG-030: Implement template projection and outlier scoring

Files:

```text
neurobench/algorithms/template_matching.py
tests/test_template_building.py
```

Acceptance:

- Mean projection works.
- Outlier scoring works on synthetic fixture.
- Outlier frame report is written.

#### TG-031: Implement template builder

Files:

```text
neurobench/algorithms/template_matching.py
neurobench/cli/template.py
tests/test_template_building.py
```

Acceptance:

- Builds template from selected video.
- Writes `template_spec.json`, `.npy`, `.png`, and score plot.

#### TG-032: Implement rigid registration

Files:

```text
neurobench/algorithms/template_matching.py
tests/test_template_registration.py
```

Acceptance:

- Recovers synthetic translation/rotation within tolerance.
- Emits low-confidence warnings.
- Emits boundary-angle warnings.

#### TG-033: Implement transform application

Files:

```text
neurobench/algorithms/template_matching.py
neurobench/cli/template.py
tests/test_template_registration.py
```

Acceptance:

- Applies per-video transform to all frames.
- Writes registered video and preview projection.

### P1: Grid contracts and algorithms

#### TG-040: Add grid schema/model

Files:

```text
schemas/grid_spec.schema.json
neurobench/models/grid.py
tests/test_grid_model.py
examples/grid_spec_32x32.example.json
```

Acceptance:

- Validates 32×32 example.
- Region IDs deterministic.
- Region count exactly 1024.

#### TG-041: Implement 32×32 grid generation

Files:

```text
neurobench/algorithms/grid_regions.py
tests/test_grid_state_extraction.py
```

Acceptance:

- Covers full template image.
- Handles dimensions not divisible by 32.
- Writes grid overlay preview.

#### TG-042: Implement grid state extraction

Files:

```text
neurobench/algorithms/grid_regions.py
neurobench/cli/grid.py
tests/test_grid_state_extraction.py
```

Acceptance:

- Writes `grid_states.npz`.
- Writes `region_features.tsv`.
- Constant/synthetic activation tests pass.

### P2: Pipeline integration

#### TG-050: Add stage metadata

Files:

```text
neurobench/pipeline_catalog.py
schemas/architecture_run.schema.json
schemas/pipeline_spec.schema.json
tests/test_pipeline_catalog.py
```

Acceptance:

- New stage IDs appear in catalog.
- Schema accepts new stage IDs.
- Planned stages are clearly marked if not runnable yet.

#### TG-051: Wire preprocessing runners

Files:

```text
neurobench/pipelines/executor.py
tests/test_template_grid_pipeline_stages.py
```

Stages:

```text
video_manifest_build
template_build_from_video
template_register_video
apply_video_registration
grid_32x32_generate
grid_state_extract
```

Acceptance:

- Synthetic end-to-end pipeline runs.
- Artifacts are registered.
- Dashboard summaries are generated.

### P2: Dashboard

#### TG-060: Add manifest/template/registration panels

Files:

```text
neurobench/workbench/assets/workbench.html
neurobench/workbench/assets/src/70_dataset_qc.js
neurobench/workbench/assets/workbench.css
tests/test_workbench_structure.py
```

Acceptance:

- Data page shows manifest counts and template preview.
- Per-video registration table exists.
- Warnings are visible.

#### TG-061: Add 32×32 grid overlay and selected cell panel

Files:

```text
neurobench/workbench/assets/workbench.html
neurobench/workbench/assets/src/20_review_core.js
neurobench/workbench/assets/src/25_review_controls.js
neurobench/workbench/assets/workbench.css
tests/test_workbench_structure.py
```

Acceptance:

- Grid toggle exists.
- Grid overlay does not break ROI overlay.
- Clicking a cell displays region/cell info.

#### TG-062: Add model prediction/reconstruction panels

Files:

```text
neurobench/workbench/assets/src/60_metrics_report.js
neurobench/workbench/assets/src/70_dataset_qc.js
neurobench/workbench/assets/workbench.html
tests/test_workbench_structure.py
```

Acceptance:

- Shows reconstruction examples.
- Shows true next / predicted next / error grid examples.
- Shows persistence comparison.

#### TG-063: Add classifier summary panel

Files:

```text
neurobench/workbench/assets/src/60_metrics_report.js
neurobench/workbench/assets/workbench.html
tests/test_workbench_structure.py
```

Acceptance:

- Shows confusion matrix.
- Shows per-video predictions.
- Shows video-level split warning/status.

### P3: Dynamics dataset and models

#### TG-070: Build video-split dynamics dataset

Files:

```text
schemas/dynamics_dataset.schema.json
neurobench/dynamics/datasets.py
neurobench/cli/dynamics.py
tests/test_grid_dynamics_dataset.py
```

Acceptance:

- Windows do not cross video boundaries.
- Splits are video-level and label-aware.
- Arrays have documented shapes.

#### TG-071: Implement persistence/moving-average baselines

Files:

```text
neurobench/dynamics/baselines.py
neurobench/dynamics/evaluate.py
tests/test_dynamics_baselines.py
```

Acceptance:

- Hand-checked metrics match expected values.
- Baseline metrics written to JSON.

#### TG-072: Implement grid autoencoder

Files:

```text
neurobench/dynamics/models.py
neurobench/dynamics/train.py
tests/test_grid_autoencoder.py
tests/test_dynamics_training_smoke.py
```

Acceptance:

- Forward shape test passes.
- CPU smoke training writes checkpoint and metrics.
- Reconstruction examples exported.

#### TG-073: Implement latent GRU predictor

Files:

```text
neurobench/dynamics/models.py
neurobench/dynamics/train.py
neurobench/dynamics/evaluate.py
tests/test_latent_rnn.py
```

Acceptance:

- Predicts next latent code.
- Decodes next grid frame.
- Compares against persistence.
- Rollout examples exported.

#### TG-074: Implement latent classifier

Files:

```text
neurobench/dynamics/classifier.py
neurobench/cli/dynamics.py
tests/test_latent_classifier.py
```

Acceptance:

- Uses video-level summaries.
- Splits by video.
- Exports confusion matrix and predictions.

### P4: Documentation and real-data pilot

#### TG-080: Add workflow docs

Files:

```text
docs/TEMPLATE_GRID_WORKFLOW.md
docs/GRID_LATENT_DYNAMICS.md
README.md
```

Acceptance:

- Explains full pipeline.
- Includes commands for `.tif` dataset.
- States inverse control and stimulation are future work.

#### TG-081: Real data pilot note

Files:

```text
docs/case_studies/grid32_real_data_pilot.md
```

Acceptance:

- Records chosen reference video.
- Records outlier settings.
- Records registration score summary.
- Records whether optional scale seemed necessary.
- Records model/classifier early results.
- Does not commit raw video data.

---

## 21. Validation Gates and Metrics

### 21.1 Manifest gates

- Expected number of videos found.
- Expected labels found: `left`, `right`, `neutral`.
- Each video has one label.
- Splits are by video.

### 21.2 Template gates

- Template projection finite fraction >= 0.999.
- Outlier removal fraction <= configured cap.
- Removed frames preview/plot exists.
- Team can visually approve template projection.

### 21.3 Registration gates

Synthetic:

- translation error <= 2 px;
- rotation error <= 2 degrees;
- finite output fraction >= 0.85 for mild transforms.

Real data:

- score recorded for every video;
- transform parameters recorded for every video;
- overlay preview exists for every video;
- warnings visible;
- no silent pass on low-confidence registrations.

### 21.4 Grid gates

- Grid shape exactly `32×32`.
- Region count exactly 1024.
- Every video has grid states.
- Frame count in grid states matches registered video.
- Nonfinite fraction reported.

### 21.5 Dynamics dataset gates

- Splits are by video.
- No window crosses video boundary.
- No train/val/test leakage.
- Label distribution is reported.
- Persistence baseline metrics exist.

### 21.6 Autoencoder gates

- Reconstruction metrics finite.
- Reconstruction examples exist.
- Per-video reconstruction errors reported.
- Checkpoint load works.

### 21.7 Latent RNN gates

- Next-code prediction metrics finite.
- Decoded next-grid predictions exist.
- Persistence comparison exists.
- Rollout metrics exist.
- No claim of scientific success unless performance beats persistence on held-out videos or a clearly stated subset.

### 21.8 Classifier gates

- Classifier evaluated by video.
- Confusion matrix exists.
- Balanced accuracy reported.
- Per-video predictions exported.
- Chance-level baseline stated as approximately 33.3% for three balanced classes, adjusted if class counts differ.

---

## 22. Remaining Implementation Questions for the Team

Codex can proceed using defaults, but the team should answer these before the real-data pilot is treated as meaningful.

1. Which exact video should be the first reference template, for example `1_neutral.tif`?
2. Are all videos the same height, width, frame count, and frame rate?
3. Are the `.tif` stacks always frame-first when read by `tifffile`, or do any have channels/pages that need special handling?
4. What is the expected maximum rotation correction: `±5°`, `±10°`, or `±15°`?
5. Should optional scale be disabled until registration previews prove it is needed?
6. Should template outlier rejection remove at most 5% of frames, or should the cap be higher?
7. Should grid state normalization be per video, per dataset, or based only on training videos?
8. Should the first grid model use raw mean intensity, robust normalized intensity, `dF/F`, or multiple channels?
9. What latent dimension should be the first real run: 16, 32, or 64?
10. Should the latent classifier use encoder codes only, RNN hidden states, or both?
11. What is the minimum acceptable dashboard proof for registration: visual overlay only, score threshold, or manually selected landmark check?
12. Should the classifier report be emphasized now, or should it wait until reconstruction/prediction quality is acceptable?

Default choices if no answer is given:

```json
{
  "reference_video_id": "1_neutral",
  "rotation_range_deg": [-10, 10],
  "allow_uniform_scale": false,
  "max_outlier_fraction": 0.05,
  "grid_rows": 32,
  "grid_cols": 32,
  "grid_feature_channels": ["mean_intensity"],
  "normalization": "per_video_robust_percentile",
  "latent_dim": 32,
  "rnn_unit": "gru",
  "split_unit": "video"
}
```

---

## 23. Recommended First Implementation Sprint

Implement this first, in order:

1. Add `goal.md` at repo root.
2. Add video manifest parser for `{1..9}_{left,right,neutral}.tif`.
3. Add shared `.tif/.tiff/.npy` video loader.
4. Add synthetic fish/grid fixture with filename-compatible videos.
5. Add template schema and template builder from one reference video.
6. Add outlier-frame rejection and outlier report.
7. Add translation + rotation registration on synthetic data.
8. Add registered video writer and registration previews.
9. Add 32×32 grid spec generation.
10. Add 32×32 grid state extraction.
11. Add Data/Review dashboard previews for template, registration, and grid.
12. Add video-split dynamics dataset builder.
13. Add persistence baseline.
14. Add autoencoder CPU smoke model.
15. Add latent GRU CPU smoke model.
16. Add latent classifier only after latent codes are being exported.

Do **not** start with Transformer models, nonrigid registration, inverse control, or stimulation input. Those are later extensions after the template-grid representation is validated.

---

## 24. Definition of Done for This Milestone

The milestone is complete when:

1. The repository contains this updated `goal.md`.
2. The real `.tif` filename pattern can be parsed into a manifest.
3. A one-video mean template can be built with optional outlier-frame rejection.
4. Each video can be registered per video using translation + rotation.
5. Registered projection overlays can be inspected in the dashboard.
6. Each registered video can be converted into a `32×32` grid state sequence.
7. Grid states are saved as `.npz` and summarized as dashboard-friendly JSON/PNG/TSV artifacts.
8. Train/validation/test splits are by video.
9. A persistence baseline is computed.
10. A grid autoencoder can train in a CPU smoke test and export reconstructions.
11. A latent GRU can predict next latent codes in a CPU smoke test and export decoded next-state examples.
12. A latent-code classifier can run on video-level code summaries for `neutral`, `left`, and `right`.
13. The dashboard shows template, registration, grid, reconstruction, prediction, and classifier summaries without breaking existing ROI review functionality.
14. Documentation states clearly that inverse control and stimulation metadata are future work, not current claims.
