# Spon Ca Burst hierarchical Parzen/noisy-ICA workflow

Last updated: 2026-07-29.

## Status and authorization

This workflow is under implementation. W0 interfaces, the W1 pure-array core,
guarded W2 Stage 1, generated evaluation, bounded real-data architecture
visuals, and a scalar Stage-2 posterior-denoising ablation are implemented.
Patchwise noisy ICA, accepted-model streaming updates, and real-time benchmarks
are not yet implemented.

The generated scientific gate did not authorize general real-data progression.
The user subsequently authorized the bounded CPU visual applications and the
noise-convolved Parzen posterior signal/noise diagnostic on Spon UI frames
1800--2359. These remain diagnostic evidence, not passed Stage-1 or complete
Stage-2 scientific gates. GPU execution remains unauthorized.

Read these contracts before changing this workflow:

- `docs/developer/HIERARCHICAL_PARZEN_NOISY_ICA_IMPLEMENTATION_BRIEF.md`;
- `docs/research/HIERARCHICAL_PARZEN_NOISY_ICA.md`;
- `docs/research/HIERARCHICAL_PARZEN_VISUALS_AND_METRICS.md`;
- `docs/developer/HIERARCHICAL_PARZEN_STAGE1_CHECKPOINT_REPORT.md`;
- `docs/developer/HIERARCHICAL_PARZEN_STAGE1_GUARDED_SYNTHETIC_REPORT.md`;
- `docs/workflows/spon_ca_burst_stage1_architecture_visuals.md`;
- `docs/developer/HIERARCHICAL_PARZEN_STAGE1_ARCHITECTURE_VISUAL_REPORT.md`;
- `docs/developer/HIERARCHICAL_PARZEN_ICA_IMPLEMENTATION_MAP.md`.

## Implemented behavior

`neurobench/algorithms/hierarchical_parzen_ica.py` is filesystem-free and
implements:

- ordinary or robust two-observation centering and stable whitening;
- clean Gaussian-Parzen log density, responsibilities, and negative score;
- Gaussian-noise-convolved density and score;
- posterior clean-source mean and variance;
- log-domain underflow protection and leave-one-out responsibilities;
- bounded deterministic Parzen dictionary initialization and updates;
- symmetric decorrelation and bounded stochastic score-function updates;
- sign/permutation component alignment;
- robust first/second derivative-energy diagnostics;
- projected noise variance bounds; and
- exact four-channel B/S/A/N closure diagnostics.

`neurobench/experiments/hierarchical_parzen_ica/config.py` loads the supplied
version-1 example with strict unknown-field rejection and freezes the initial
method, frame, geometry, dictionary, evaluation, visualization, real-time, and
resource contracts.

`neurobench/experiments/hierarchical_parzen_ica/stage1.py` implements:

- aligned two-frame observations and bounded quiet-prefix fitting;
- fixed common/difference and robust adaptive-gain references;
- compatibility with the bounded batch CS-Parzen reference;
- the bounded stochastic Parzen-score lane with stable artifact method IDs;
- label-free staticness classification with unresolved and degenerate
  no-subtraction fallbacks; and
- causal frozen-model inference against the prior reconstructed background,
  preserving sustained event amplitude instead of emitting only a derivative.

`neurobench/experiments/hierarchical_parzen_ica/safety.py` and
`evaluation.py` implement affine feedback gating, reference anchoring, bounded
fallbacks, collision-safe generated-matrix artifacts, and explicit gate reports.
`synthetic.py` implements all 12 required Stage-1 B/S/A/N fixture classes.
`architecture_lanes.py`, `architecture_config.py`, and `architecture_visuals.py`
implement teacher-forced, raw-recursive, quiet-fixed-point, and bounded
reference-plus-Parzen-innovation lanes with a strict read-only preflight and
separate real-data background/dynamics TIFFs.
`neurobench/metrics/hierarchical_separation.py` retains the focused Stage-1
background/signal leakage and exact-closure metrics.

## Current validation

Run the implemented checkpoint with:

```bash
.venv-neurobench/bin/python -m pytest -q \
  tests/test_hierarchical_parzen_algorithms.py \
  tests/test_hierarchical_parzen_config.py \
  tests/test_hierarchical_parzen_stage1.py \
  tests/test_hierarchical_parzen_safety.py \
  tests/test_hierarchical_parzen_synthetic.py \
  tests/test_hierarchical_parzen_architecture_lanes.py \
  tests/test_hierarchical_parzen_architecture_visuals.py
```

The current tests validate numerical density normalization, score finite
differences, noisy Gaussian convolution, scalar posterior conditioning,
log-domain stability, leave-one-out handling, whitening degeneracy, dictionary
determinism/bounds, demixer decorrelation/determinism, component tracking,
noise projection, closure, strict manifest fields, frozen scientific
guards, all four Stage-1 lanes, quiet-only calibration, explicit unresolved and
degenerate behavior, sustained slow-ramp preservation, adaptive illumination
gain, affine feedback rejection, reference fallback, all 12 generated fixture
classes, collision safety, and deterministic multi-seed artifact generation.

The deterministic quiet-plus-ramp fixture reports `0.000493`
signal-residual NMSE, `0.000493` signal leakage into background, and `5.62e-08`
maximum closure error. The multiplicative-drift fixture estimates `1.035047`
for a true gain of `1.035`; adaptive residual RMS is `0.0443` times the fixed
reference. These are deterministic unit fixtures, not Spon results.

Those values describe the reference lanes only. The batch CS-Parzen recursive
lane fails catastrophically on the same easy fixture (`2.77e17`
signal-residual NMSE), while the deliberately short stochastic smoke fit does
not converge and yields `0.6776` signal-residual NMSE. See the explicit
checkpoint report for the fitted feedback coefficients and full interpretation.

The guarded follow-up executed 240 generated combinations: 12 cases, five
seeds, and four methods. All combinations completed and the numerical gate
passed with maximum closure error below `5.98e-08`. The scientific gate failed.
Adaptive median signal NMSE was `0.03039`; guarded batch was `0.15365`; guarded
stochastic was `0.05687`. Stochastic improved all five similar-persistence
cases but lost all 15 other signal comparisons. All methods incorrectly
resolved every equal-staticness challenge.

## Gates

| Gate | Status | Evidence |
| --- | --- | --- |
| C0 implementation integrity | Partial | 47 focused hierarchical tests pass across the current posterior/signal-split scope; generated and real-data outputs are collision-safe and the split closes exactly. |
| C1 Stage-1 identifiability | Partial | Numerical stability passes across 240 guarded generated combinations. Scientific validity fails: adaptive remains the general reference, learned gains are case-specific, motion/clipping preservation worsens, and unresolved classification fails. |
| C2 Stage-2 numerical stability | Partial posterior-only | All 16 scalar posterior combinations were finite and monotone; both 560-frame splits closed exactly. Patchwise covariance, demixing, overlap-add, and component qualification remain unimplemented. |
| C3 decomposition validity | Partial | Generated B/S/A/N reporting exists; the scientific gate fails and semi-synthetic Spon injection has not run. |
| C4 real signal preservation | Failed as standalone detector | The 185-effective-operator follow-up completed. Cross-fitted selection reached 0.364648 mean recall versus Raw Direct 0.605616; paired ROI-identity interval was entirely negative. |
| C5 downstream utility | Failed standalone; feature fusion remains open | Current innovation reached 0.329969 primary-threshold recall and 0.640580 fixed-budget recall versus Raw Direct 0.657246. Use as a feature alongside Raw Direct, not a replacement input. |
| C6 real-time candidacy | Partial | The four CPU visual lanes completed in 57.15 seconds for 559 outputs, but this includes TIFF I/O and is not a per-frame latency benchmark. |

The guarded generated matrix and real-data diagnostics are not evidence that hierarchical separation detects neurons or isolates pure measurement noise. The full-field detector checkpoint is documented in `docs/workflows/spon_ca_burst_stochastic_architecture_grid.md`. The authorized scalar posterior checkpoint is documented in `docs/workflows/spon_ca_burst_noisy_parzen_signal_split.md`. Next audit the signal/noise TIFFs for neural leakage, then pursue Raw Direct feature fusion or semi-synthetic injection. Full patchwise noisy ICA remains gated.
