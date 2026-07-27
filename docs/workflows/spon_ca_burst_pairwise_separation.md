# Spon Ca Burst Pairwise Source Separation

## Status and scope

This workflow implements the design in
`docs/developer/PAIRWISE_SOURCE_SEPARATION_IMPLEMENTATION_BRIEF.md`. It compares
five explicitly named lanes: fixed binary difference, quiet-fitted adaptive
difference, two-by-two InfoMax ICA, bounded CS-Parzen ICA, and constrained
shared-background NMF. Raw Direct remains an external validity anchor; these
methods do not redefine it.

Implementation and tiny synthetic validation are complete. No full Spon Ca
Burst result exists yet, and no performance conclusion should be inferred from
the synthetic tests. A full local run requires explicit user selection.

## Scientific contract

- Frame indices in the UI/manifest are one-based and inclusive. Array intervals
  are zero-based and half-open; coordinates are `x=column`, `y=row`.
- Every lane uses the same causal spatial/EMA preprocessing, quiet-only
  calibration, temporal pooling, NMS, six-pixel primary match radius, and sparse
  known-positive interpretation.
- Unmatched candidates are unknown, not false positives. Known-label candidate
  fraction is only a lower bound.
- ICA is a two-observation source-separation diagnostic, not motion correction.
  If the covariance or activity component is unresolved, the run records the
  reason and omits continuous/mask artifacts for that ICA lane.
- Shared-background NMF must report when it is effectively equivalent to the
  positive adaptive residual rather than claim novelty from a renamed result.
- Resolved lanes write `continuous_activity_signed.tif`, where negative values
  are below mid-gray, zero is mid-gray, and positive values are above mid-gray;
  `positive_z.tif` shows one-sided standardized activity, and `binary_mask.tif`
  shows the final thresholded output. Scaling is fixed per stack at a sampled
  99.5th percentile and recorded in the first TIFF page description.

## Implementation map

- Reusable numerical methods: `neurobench/algorithms/pairwise_separation.py`
- Strict version-1 manifest: `neurobench/experiments/pairwise_separation/config.py`
- Sampling and preflight: `sampling.py`, `preflight.py`
- Fit, evaluation, atomic artifacts, orchestration: `fitting.py`,
  `evaluation.py`, `artifacts.py`, `runner.py`
- Shared sparse-label semantics: `neurobench/metrics/sparse_detection.py`
- Example: `examples/spon_ca_burst_pairwise_separation.example.json`

## Guarded commands

Preflight is a write to an explicit, new artifact directory so its resolved
configuration and label projection cannot be confused with another run:

```bash
.venv-neurobench/bin/python -m neurobench.cli.main experiment pairwise-separation preflight \
  --config examples/spon_ca_burst_pairwise_separation.example.json \
  --artifact-dir Outputs/PairwiseSeparation/preflight_v1
```

After inspecting `preflight.json` and the projection overlay, a separately
authorized run consumes that exact preflight:

```bash
.venv-neurobench/bin/python -m neurobench.cli.main experiment pairwise-separation run \
  --config examples/spon_ca_burst_pairwise_separation.example.json \
  --preflight-dir Outputs/PairwiseSeparation/preflight_v1
```

Both destinations refuse collisions. Do not reuse or overwrite a completed
output root.

## Current validation

The focused tests cover fixed/adaptive difference direction, whitening and
degeneracy, ICA orientation, bounded CS-kernel blocks, NMF nonnegativity and
monotonicity diagnostics, strict config validation, explicit preflight output,
lazy CLI construction, conditional ICA masks, deterministic sparse-label
matching, legacy evaluation parity, and a tiny end-to-end artifact run.

## First full Spon result, July 27

The explicitly selected run at
`Outputs/PairwiseSeparation/spon_ca_burst_pairwise_separation_v1` completed all
five lanes in 29.1 seconds and wrote 5.1 GiB of collision-safe artifacts. It
reproduced Raw Direct exactly at `0.605615942` mean burst recall (49/79 pooled,
232 event candidates).

| Lane | Mean recall | Known matches | Event candidates |
| --- | ---: | ---: | ---: |
| Raw Direct anchor | 0.6056 | 49/79 | 232 |
| Fixed binary difference | 0.1333 | 10/79 | 29 |
| Adaptive binary difference | 0.1333 | 10/79 | 29 |
| InfoMax tanh ICA | 0.3832 | 30/79 | 161 |
| CS-Parzen ICA | 0.1333 | 10/79 | 24 |
| Shared-background NMF | 0.0000 | 0/79 | 0 |

Adaptive quiet gain was `1.000087`, so its practical equivalence to fixed
subtraction is expected. InfoMax was the strongest pairwise lane but did not
beat Raw Direct; it reached the 500-iteration cap without the strict convergence
criterion, so its component and TIFF remain exploratory. CS-Parzen converged
but behaved similarly to subtraction after thresholding. The configured NMF
penalty collapsed activity to zero, making its review TIFF intentionally blank.

Each resolved method directory contains three complementary views:

- `continuous_activity_signed.tif`: signed processing effect; dark is negative,
  mid-gray is zero, and bright is positive;
- `positive_z.tif`: one-sided quiet-standardized activity before the final
  binary decision;
- `binary_mask.tif`: thresholded 0/255 display mask.

Unmatched candidates remain unknown, not false positives. The reported
known-label candidate fractions are therefore lower bounds and not precision.
The next valid checkpoint is visual review of the signed and positive-z InfoMax
stacks, followed by bounded convergence/stability diagnostics rather than a
wide hyperparameter search.

The bounded follow-up in
`docs/workflows/spon_ca_burst_pairwise_feature_fusion.md` tested the continuous
outputs as auxiliary Raw Direct features. Additive and learned scalar fusion
preserved recall but added candidates; soft gating traded candidates for known
matches and did not advance.
