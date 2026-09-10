# ICA repository audit for PC-MITL-ICA phase 1

## Current implementation map

| Concern | Maintained location | Current contract |
| --- | --- | --- |
| Two-frame centering/whitening | `neurobench/algorithms/pairwise_separation.py::center_and_whiten_2d` | Input `[2,N]`; full supplied calibration array is fitted; eigen floor is explicit. |
| CS-Parzen information potential | `neurobench/algorithms/pairwise_separation.py::cs_parzen_objective` | Public input `[N,2]`; exact blockwise Gaussian kernels; optional nonnegative weights; float64 accumulation. |
| CS-Parzen rotation fit | `neurobench/algorithms/pairwise_separation.py::fit_cs_parzen_ica` | Input `[2,N]`; bounded coarse/refined 2-D angle search; separate screen/confirm arrays supported. |
| Label-free two-frame fit | `neurobench/experiments/unsupervised_ica_eval/core.py` | Input `[N,2]`; FastICA plus analytic derivative/PCA/random controls; frozen representation hash. |
| Patch/pair sampling and guarded run | `neurobench/experiments/pairwise_separation/` | Frame-first video becomes aligned previous/current samples; preflight and new output root required. |
| Detection and sparse-positive metrics | `neurobench/experiments/pairwise_separation/evaluation.py`, `neurobench/metrics/sparse_detection.py` | Common threshold/pooling/NMS; unmatched candidates remain unknown. |
| Trace audit | `neurobench/experiments/unsupervised_ica_eval/traces.py` | Raw and processed matched-support summaries; complete real-video audit media are not generated here. |

## Orientation and fitting boundaries

The numerical CS implementation uses `[N,2]`, while its optimizer and the
legacy whitening function use `[2,N]`. The label-free evaluation wrapper uses
`[N,2]` externally and transposes explicitly. PC-MITL-ICA therefore adopts
`[N,q]` at every new public boundary and documents any legacy transpose at the
adapter.

Existing full-video workflows use explicit calibration/screen/confirmation
partitions, but the low-level whitening helper fits whatever array it receives;
leakage prevention belongs to the caller. The phase-1 baseline makes contiguous
train/validation/test indices explicit and fits mean/covariance on train only.

## Defaults and determinism

The maintained CS fitter defaults to bandwidth `0.35`, coarse angle step `3°`,
refinement half-width `3°`, refinement step `0.25°`, and float64 accumulation.
It is a deterministic grid search. The existing two-frame FastICA defaults to
500 iterations, tolerance `1e-6`, and an explicit seed. Full patch dimensions,
stride, crop, threshold, and pooling are manifest-owned rather than global ICA
defaults.

## Documentation discrepancies and missing artifacts

- The proposal's generic `src/neurev/ica/` tree does not match this repository;
  maintained experiment code belongs under `neurobench/experiments/` and shared
  numerical code under `neurobench/algorithms/`.
- Current CS-Parzen is a specialized two-output, blockwise estimator rather
  than a general `q`-component objective.
- Raw and processed trace summaries exist, but the full real-video scientific
  audit is intentionally incomplete for the old label-free Stage A.
- There was no immutable configuration/fingerprint binding config, data ID,
  and commit for the proposed matrix-TC comparison before this phase.

## Safe integration point

The new matrix objective is isolated in
`neurobench/experiments/unsupervised_ica_eval/matrix_itl.py`. It does not alter
or silently replace the current CS-Parzen path. Promotion into the general
pairwise runner is blocked until synthetic gates pass and a separate real-video
configuration is explicitly authorized.
