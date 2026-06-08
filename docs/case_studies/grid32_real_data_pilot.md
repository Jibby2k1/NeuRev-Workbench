# Grid32 Real Data Pilot

This note records the first real-data pilot for the template-aligned 32x32 grid
dynamics workflow. Do not commit raw video data in this file.


## Current Workspace Status

As of 2026-06-02, the exact `{1..9}_{left,right,neutral}.tif` pilot dataset was not found under `Inputs/`, so the real-data preprocessing pilot was not executed in this workspace. This note is ready for the first run once those files are available.

## Safety Preflight

- Preflight JSON: `Outputs/GridModel/manifest/preflight.json`
- Estimated output bytes:
- Available disk bytes:
- Available RAM bytes:
- Chunk size frames: `64`
- Preflight warnings reviewed:
- Decision to proceed:

## Dataset

- Input directory: `Inputs/ZebrafishVideos`
- Expected files: `{1..9}_{left,right,neutral}.tif`
- Video count found:
- Label counts:
- Frame shape consistency:
- Frame count consistency:
- Frame rate:
- Notes:

## Template

- Reference video id: `1_neutral`
- Projection kind: `mean_after_outlier_rejection`
- Outlier method: `projection_residual_zscore`
- Max outlier fraction: `0.05`
- Removed frames:
- Template preview reviewed by:
- Template decision:

## Registration

- Transform model: `rigid`
- Rotation range: `[-10, 10]` degrees
- Rotation step: `0.5` degrees
- Uniform scale enabled: `false`
- Registration score summary:
- Boundary-angle warnings:
- Low-confidence warnings:
- Blank-fraction warnings:
- Overlay review decision:
- Should optional scale be tested next:

## Grid States

- Grid size: `32x32`
- Feature channels: `mean_intensity`
- Normalization: `per_video_robust_percentile`
- Videos with grid states:
- Nonfinite fraction summary:
- Selected-cell trace review:
- Grid-state decision:

## Dynamics

- Window frames: `8`
- Prediction horizon: `1`
- Split unit: `video`
- Split method: `stratified_by_label`
- Train videos:
- Validation videos:
- Test videos:
- Persistence baseline:
- Moving-average baseline:
- Autoencoder reconstruction metrics:
- Latent RNN prediction metrics:
- Improvement over persistence:

## Classifier

- Feature source: encoder latent codes
- Summary: mean and standard deviation over time
- Classifier:
- Evaluation:
- Accuracy:
- Balanced accuracy:
- Macro F1:
- Confusion matrix path:
- Per-video predictions path:
- Better than chance:

## Limitations

- No stimulation/control metadata is used.
- Inverse control is out of scope for this milestone.
- Registration uses one per-video rigid/similarity transform only.
- The 32x32 grid is a rectangular pooling representation, not an anatomical
  parcellation.
- Scientific claims should wait for reviewed registration overlays and
  held-out-video metrics.

## Follow-Ups

- Reference video change needed:
- Scale search needed:
- Grid feature change needed:
- Latent dimension change needed:
- Additional QC needed:
