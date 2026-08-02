# Spon Ca Burst spatial ICA architecture screen

## Purpose

This workflow tests whether spatial patch ICA benefits from dense,
translation-shared application after the accepted Parzen Innovation
background/residual stage. It is a checkpointed C1--C3 screen:

1. `c1_patch_fastica`: spatial patch PCA/FastICA, overlap-add stride 5, and
   quiet-calibrated Wiener component shrinkage.
2. `c2_dense_convolutional_fastica`: the identical learned filters and
   shrinkage applied at every spatial translation.
3. `c3_dense_convolutional_parzen`: the identical dense filter bank with
   bounded noisy-Parzen posterior shrinkage replacing Wiener shrinkage.

This design isolates application geometry before changing the marginal model.
It is not yet noisy-Parzen Infomax filter learning. Grouped morphology and
causal temporal filters remain gated on this spatial checkpoint.

## Data and coordinate contract

The source is the memory-mapped Spon Ca Burst cache. The review interval is UI
frames 1800--2359 inclusive, with UI frames 1800--1899 as quiet calibration.
UI indices are one-based and inclusive; NumPy intervals are zero-based and
half-open. Label coordinates use `x=column`, `y=row`.

The preflight writes a projection overlay. The 79 sparse label rows describe 27
known identities; pixels outside known labels remain unknown rather than
negative.

## Model contract

The default model samples 30,000 label-free 11-by-11 spatial patches and retains
12 PCA dimensions before symmetric log-cosh FastICA. Quiet-only patches
calibrate one scale per component. The exploratory fit uses all review frames
without labels and is therefore transductive, not a held-burst generalization
result.

The Parzen dictionary uses 48 centers, 24 fixed at zero, bandwidth 0.5, noise
variance 1, and a 4,096-point lookup over standardized values from -20 to 20.

## Resource and execution contract

The manifest is
`examples/spon_ca_burst_spatial_ica_screen.example.json`. Preflight and run are:

```bash
.venv-neurobench/bin/python -m neurobench.cli.main experiment \
  parzen-spatial-ica preflight \
  --config examples/spon_ca_burst_spatial_ica_screen.example.json

.venv-neurobench/bin/python -m neurobench.cli.main experiment \
  parzen-spatial-ica run \
  --config examples/spon_ca_burst_spatial_ica_screen.example.json
```

Run only after a matching ready preflight. Never overwrite either the completed
output root or a `.partial` root. The default CUDA batch is one frame and the
manifest caps RAM, GPU memory, disk headroom, and output size.

## Outputs

The completed root is
`Outputs/HierarchicalParzenICA/spon_ca_burst_spatial_ica_screen_v1`.
It contains:

- `REPORT.md`, `metrics.json`, and the resolved/preflight contracts;
- a compressed model and signed analysis-filter TIFF under `model/`;
- one positive-signal TIFF and one signed-remainder TIFF per lane; and
- atomic progress, checkpoint, and completion metadata.

Signal TIFFs share one display scale. Each remainder has a symmetric
variant-specific detail scale recorded in its metadata. Visual review of both
is required before scientific acceptance.

## Advancement gate

Do not advance automatically to noisy-Parzen Infomax optimization, grouped
morphology, or causal temporal convolution. First require:

- no material timing or shape loss;
- acceptable signal leakage in the remainder;
- an improvement that persists across seeds and held-burst fits; and
- no regression in fixed-budget recall, candidate burden, and exact
  semi-synthetic preservation.
