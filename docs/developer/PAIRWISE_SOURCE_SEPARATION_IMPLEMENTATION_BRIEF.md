# Pairwise source separation and fast binary derivative

Implementation brief: 2026-07-27.

Status: implementation specification. This document authorizes repository code,
unit tests, synthetic fixtures, preflight, and bounded CPU smoke tests. It does
**not** authorize a long Spon Ca Burst run, a GPU run, replacement of an existing
scientific lane, or modification/deletion of completed `Outputs/` evidence.

## Executive directive

Implement a stage-gated NeuRev experiment that tests the professor's proposed
relationship:

```text
principled pairwise source separation
    -> background-like and activity-like components
    -> frame subtraction as an equal-background approximation
    -> robust positive thresholding as the fast practical method
```

The first implementation must provide five explicit lanes:

1. fixed one-sided binary frame difference;
2. adaptive gain-corrected binary frame difference;
3. a Bell-style InfoMax reference ICA implementation;
4. Cauchy-Schwarz-divergence ICA for two observations;
5. a constrained nonnegative pair decomposition with a shared background and a
   sparse positive innovation.

Do not hide these methods behind one opaque "ICA" option. Every lane must write
its fitted parameters, assumptions, diagnostics, runtime, masks, candidates, and
metrics separately.

The implementation is an experiment, not a silent rewrite of the current
frame-difference or activity-gate workflows. Existing outputs and numerical
anchors must remain reproducible.

## Required reading before editing

Read these files in order:

1. `AGENTS.md`;
2. `docs/workflows/spon_ca_burst_frame_derivatives.md`;
3. `docs/workflows/spon_ca_burst_multihypothesis_cfar.md`;
4. `neurobench/experiments/frame_difference.py`;
5. `neurobench/experiments/smoothed_frame_difference.py`;
6. `neurobench/experiments/activity_gate_benchmark.py`;
7. `neurobench/cli/experiment.py`;
8. `neurobench/algorithms/motion.py`;
9. `tests/test_frame_difference.py` and
   `tests/test_activity_gate_benchmark.py`.

Use `.venv-neurobench/bin/python` for repository commands.

## Current repository truth

The current signed derivative implementation computes

```text
D[t] = I[t] - I[t-k]
```

for fixed lags and writes continuously valued signed diagnostic TIFFs. The
smoothed derivative implementation can normalize the derivative by a per-pixel
quiet-period MAD and apply a symmetric deadband, but it does not implement the
professor's exact one-sided binary rule.

The current activity-gate benchmark is also important negative evidence:

- Raw Direct reproduces mean known-label recall `0.605615942`;
- multiplying intensity by derivative energy was too selective for slowly
  evolving calcium activity;
- the strongest causal lane in that benchmark improved ranking through causal
  smoothing and artifact attenuation, not through a hard derivative-energy
  gate;
- unmatched event candidates remain unknown because the labels are sparse.

Therefore, the new binary derivative must initially be treated as an onset mask,
candidate feature, or review lane. It must not be declared a complete cleaned
calcium image or a replacement detector merely because it produces sparse
frames.

## Scientific model and terminology

For pixel/sample index `p`, construct the two-observation vector

```text
x_t(p) = [I[t-k, p], I[t, p]]^T.
```

The intended aggregate source model is

```text
x_t(p) = A_t [B_t(p), S_t(p)]^T + n_t(p),
```

where:

- `B_t` is persistent anatomy, illumination, static fluorescence, and other
  background-like structure;
- `S_t` is a sparse activity-like innovation;
- `A_t` is a two-by-two mixing matrix;
- `n_t` is residual noise/model error.

Two observations can recover at most two aggregate components. Do not describe
this as separating every neuron independently.

If the same background enters both observations equally, a background-null row
must be proportional to `[-1, 1]`, giving ordinary subtraction. With a gain
change, the corresponding fast approximation is

```text
D_alpha[t] = I[t] - alpha * I[t-k].
```

The practical binary rule is

```text
z[t, p] = (D_alpha[t, p] - quiet_center[p]) /
          max(quiet_scale[p], quiet_scale_floor)
M[t, p] = 1 if z[t, p] >= tau else 0.
```

The primary rule is one-sided and positive. Do not use `abs(z)` in the primary
lane. Negative changes may be retained as diagnostics but are not positive
firing detections.

### Naming requirements

Use precise method identifiers:

- `fixed_binary_difference`;
- `adaptive_binary_difference`;
- `infomax_tanh_ica`;
- `cs_parzen_ica`;
- `shared_background_nmf`.

Do not call the tanh natural-gradient reference "exact Bell ICA." Report it as a
Bell-style or InfoMax reference. Do not call the Cauchy-Schwarz estimator an
exact continuous-density result; it is a bounded Parzen estimate. Do not call a
known-label candidate fraction precision.

## Non-negotiable implementation constraints

1. Preserve all existing user changes and ignored data under `Inputs/` and
   `Outputs/`.
2. Refuse an existing output root. Never overwrite a completed experiment.
3. UI frame indices are one-based and inclusive. NumPy intervals are zero-based
   and half-open. Convert once at the boundary and record both conventions.
4. Coordinates use `x=column`, `y=row`.
5. Every label-driven preflight must write a label projection overlay.
6. Sparse positives do not define exhaustive negatives. Unmatched candidates
   are `unknown`, not automatic false positives.
7. Configure OpenMP/BLAS thread limits before importing heavy numerical modules.
8. Use bounded chunks, deterministic seeds, atomic JSON, progress JSONL, and
   explicit RAM/disk/output estimates.
9. Do not add `scikit-learn` solely for ICA or NMF. Use NumPy/SciPy and the
   repository's existing dependencies.
10. Do not run an all-pixel quadratic Parzen calculation. ICA fitting must use a
    declared, deterministic, bounded sample and chunked kernel blocks.
11. Do not use centered Savitzky-Golay smoothing in any lane described as
    causal or real-time.
12. Record `motion_correction: false` when no registration is used. Never imply
    that ICA removes geometric motion.
13. Do not implement unconstrained rank-two NMF on a two-column image matrix.
    That admits the trivial factorization `W=X, H=I` and does not test the stated
    shared-background hypothesis.
14. Labels may be used for evaluation and outer-fold organization, but not to
    choose ICA component signs, component order, Parzen bandwidth, gain, or
    quiet threshold on the held-out burst.
15. Completion of a run is not scientific success. Keep implementation gates and
    scientific advancement gates separate.

## Required repository additions

Create the following package structure unless an equally clean current package
route is discovered during implementation:

```text
neurobench/
├── algorithms/
│   └── pairwise_separation.py
├── experiments/
│   └── pairwise_separation/
│       ├── __init__.py
│       ├── config.py
│       ├── preflight.py
│       ├── sampling.py
│       ├── fitting.py
│       ├── evaluation.py
│       ├── artifacts.py
│       └── runner.py
└── metrics/
    └── sparse_detection.py
```

Add:

```text
examples/spon_ca_burst_pairwise_separation.example.json
tests/test_pairwise_separation_algorithms.py
tests/test_pairwise_separation_config.py
tests/test_pairwise_separation_runner.py
tests/test_pairwise_separation_cli.py
docs/workflows/spon_ca_burst_pairwise_separation.md
```

The workflow document is written after the implementation and must report what
actually exists. Do not copy this implementation brief into the workflow file
and present planned behavior as completed behavior.

### Ownership

`neurobench/algorithms/pairwise_separation.py` owns reusable numerical
operations only:

- robust difference calibration;
- gain estimation;
- centering/whitening;
- InfoMax fitting;
- Parzen Cauchy-Schwarz divergence and angle search;
- constrained pair NMF updates;
- deterministic component orientation/selection.

The experiment package owns:

- manifest validation;
- Spon frame and label contracts;
- sampling policies;
- resource preflight;
- orchestration;
- per-method application to a review interval;
- candidate extraction/evaluation;
- artifacts and reports.

The CLI must stay thin and lazy-load the experiment package.

## Public numerical interfaces

Implement typed, documented interfaces equivalent to the following. Exact
names may vary only when repository conventions make another name clearer.

```python
@dataclass(frozen=True)
class QuietDifferenceStats:
    center: np.ndarray
    scale: np.ndarray
    scale_floor: float

@dataclass(frozen=True)
class Whitening2D:
    mean: np.ndarray
    covariance: np.ndarray
    whitening: np.ndarray
    dewhitening: np.ndarray
    eigenvalues: np.ndarray
    condition_number: float
    identifiable: bool

@dataclass(frozen=True)
class SeparationFit:
    method_id: str
    demixing: np.ndarray
    mixing: np.ndarray | None
    objective: float | None
    converged: bool
    iterations: int
    activity_component: int | None
    activity_sign: int | None
    diagnostics: dict[str, Any]
```

Required functions:

```python
quiet_difference_stats(...)
fixed_difference(...)
estimate_quiet_gain(...)
adaptive_difference(...)
center_and_whiten_2d(...)
fit_infomax_tanh_ica(...)
cs_parzen_independence(...)
fit_cs_parzen_ica(...)
fit_shared_background_nmf(...)
orient_and_select_activity_component(...)
apply_linear_separation(...)
```

Every function must validate shapes and finite values. The reusable algorithm
module must not read files or know Spon-specific frame numbers.

## Manifest contract

Create a schema-validated dataclass loader for a manifest with this initial
shape:

```json
{
  "schema_version": 1,
  "experiment_id": "spon_ca_burst_pairwise_separation_v1",
  "source_video": "../Outputs/GammaCFAR/spon_ca_burst_3_hindbrain_to_tail_488_20ms/spon_ca_burst_3_hindbrain_to_tail_488_20ms.npy",
  "source_tiff": "../Inputs/Spon Ca Burst/3 hindbrain to tail 488 20ms.tif",
  "labels_tsv": "../Inputs/Spon Ca Burst/labels/labels_normalized.tsv",
  "output_dir": "../Outputs/PairwiseSeparation/spon_ca_burst_pairwise_separation_v1",
  "frames": {
    "review_start_ui": 1800,
    "review_end_ui": 2359,
    "quiet_start_ui": 1800,
    "quiet_end_ui": 1899,
    "frame_period_ms": 20.0
  },
  "preprocessing": {
    "lag_frames": 1,
    "spatial_sigma_px": 1.0,
    "temporal_mode": "causal_ema",
    "temporal_ema_span_frames": 4.0,
    "motion_mode": "none",
    "run_integer_shift_sensitivity": false,
    "max_shift_px": 2
  },
  "sampling": {
    "primary_policy": "uniform_anatomy",
    "screen_samples": 1024,
    "confirm_samples": 4096,
    "screen_angle_step_degrees": 3.0,
    "refine_half_width_degrees": 3.0,
    "refine_angle_step_degrees": 0.25,
    "pairwise_diagnostic_frames_ui": [1900, 2000, 2100, 2200, 2300],
    "seed": 20260727
  },
  "methods": {
    "fixed_binary_difference": {
      "enabled": true
    },
    "adaptive_binary_difference": {
      "enabled": true,
      "alpha_min": 0.8,
      "alpha_max": 1.2,
      "trim_fraction": 0.1,
      "refinement_iterations": 3
    },
    "infomax_tanh_ica": {
      "enabled": true,
      "max_iterations": 500,
      "learning_rate": 0.01,
      "tolerance": 1e-7,
      "initial_angles_degrees": [0, 15, 30, 45, 60, 75]
    },
    "cs_parzen_ica": {
      "enabled": true,
      "bandwidth": 0.35,
      "kernel_block_rows": 256
    },
    "shared_background_nmf": {
      "enabled": true,
      "activity_l1": 0.05,
      "max_iterations": 100,
      "tolerance": 1e-6
    }
  },
  "thresholding": {
    "z_thresholds": [2.0, 2.5, 3.0, 3.5, 4.0],
    "primary_z_threshold": 3.0,
    "one_sided_positive": true,
    "minimum_component_pixels": [1, 3, 5],
    "write_binary_tiff": true
  },
  "evaluation": {
    "binary_temporal_pool": "occupancy",
    "nms_distance_px": 6,
    "primary_match_radius_px": 8,
    "match_radii_px": [4, 6, 8, 10],
    "quiet_false_peaks_per_map": 1.0,
    "capacity_reference_lane": "raw_direct",
    "candidate_review_rows": 240
  },
  "resources": {
    "cpu_threads": 4,
    "frame_chunk": 32,
    "kernel_block_rows": 256,
    "max_ram_mib": 6144,
    "min_free_disk_mib": 8192,
    "max_output_mib": 8192
  }
}
```

Validation requirements:

- `schema_version` must be exactly `1`;
- review and quiet intervals must be inside the video;
- quiet must contain at least 50 defined derivative frames;
- `lag_frames >= 1` and shorter than the quiet interval;
- the primary threshold must occur in `z_thresholds`;
- `one_sided_positive` must be true in version 1;
- sample counts must be bounded, unique in purpose, and no larger than the
  available valid sample population;
- `confirm_samples >= screen_samples`;
- angle steps must exactly tile or safely bound the sign/permutation-unique
  interval `[0, 90)` degrees;
- `kernel_block_rows <= confirm_samples`;
- resource caps must be positive and CPU threads must remain within the existing
  experiment CLI limit;
- output collision is an error for `run` and a reported condition for an
  explicitly read-only inspection mode only.

Unknown manifest keys should be rejected in version 1. Do not silently ignore a
misspelled scientific parameter.

## CLI contract

Add the following command group to `neurobench/cli/experiment.py` without
importing NumPy/SciPy at parser construction time:

```bash
.venv-neurobench/bin/python -m neurobench.cli.main experiment pairwise-separation preflight \
  --config examples/spon_ca_burst_pairwise_separation.example.json

.venv-neurobench/bin/python -m neurobench.cli.main experiment pairwise-separation run \
  --config examples/spon_ca_burst_pairwise_separation.example.json
```

Optional bounded actions may be added only when they reduce risk:

```bash
... experiment pairwise-separation synthetic-smoke --output-dir <new-root>
... experiment pairwise-separation report --run-dir <completed-root>
```

`run` must refuse an existing destination. `report` must be read-only with
respect to scientific arrays and may regenerate only explicitly declared report
artifacts into a new report destination.

## Wave 0 — preserve shared evaluation semantics

Before adding new methods, isolate the reusable sparse-detection semantics that
are currently embedded in the activity-gate benchmark.

### Tasks

1. Add `neurobench/metrics/sparse_detection.py` with deterministic helpers for:
   - temporal map pooling;
   - local-maximum/NMS peak extraction;
   - quiet-calibrated thresholds;
   - one-to-one radius matching;
   - capacity-matched candidate selection;
   - known-label recall summaries;
   - explicit unknown-candidate records.
2. Preserve compatibility wrappers in
   `neurobench/experiments/activity_gate_benchmark.py` while moving behavior to
   the public helper.
3. Add regression tests proving that the Raw Direct lane still reproduces
   `0.605615942` mean recall on the existing fixture/contract and that candidate
   ordering is unchanged.
4. Preserve the current interpretation string: known-label candidate fraction
   is a lower bound only, not precision.

### Wave 0 acceptance

- existing activity-gate benchmark tests pass unchanged or with only import-path
  updates;
- the numerical Raw Direct anchor is unchanged to the existing stored precision;
- no existing report field is removed or renamed;
- the new helper has focused synthetic tests for tie ordering and one-to-one
  matching.

Do not continue if the baseline changes. Diagnose parity first.

## Wave 1 — professor's fast practical approximation

Implement the fixed and adaptive binary derivative lanes before ICA or NMF.

### Shared causal preprocessing

The primary fast lane is:

1. spatial Gaussian smoothing with the configured `sigma`;
2. causal temporal EMA with the configured span;
3. lagged difference;
4. quiet robust calibration;
5. positive one-sided threshold.

For a span `s`, use

```text
ema_alpha = 2 / (s + 1)
F[0] = spatial(I[0])
F[t] = ema_alpha * spatial(I[t]) + (1 - ema_alpha) * F[t-1].
```

The first `lag_frames` derivative frames are undefined. Store them as zero in the
canonical binary mask and mark them undefined in metadata. Never use future
frames.

The raw unsmoothed fixed difference may be retained as a diagnostic lane, but it
must not be silently substituted for the declared primary preprocessing.

### Quiet robust calibration

For each pixel, estimate from defined quiet derivatives:

```text
center[p] = median(D_quiet[:, p])
scale[p] = 1.4826 * median(abs(D_quiet[:, p] - center[p]))
```

Compute the scale floor from the configured percentile of strictly positive
scales. Record:

- zero-scale fraction;
- floor value;
- scale median and p95;
- finite/invalid pixel counts.

Use the same center and scale for all later review frames. Do not normalize each
frame independently.

### Fixed binary difference

Compute:

```text
D_fixed[t] = F[t] - F[t-k]
z_fixed[t] = (D_fixed[t] - center_fixed) / scale_fixed
M_fixed[t] = uint8(z_fixed[t] >= tau)
```

Canonical scientific mask values are exactly `0` and `1`. A viewer TIFF may use
`0` and `255`, but its embedded metadata must state that it is a visualization
encoding of the canonical binary mask.

Write a separate positive-score array such as `max(z, 0)` for diagnostics and
candidate tie-breaking. Do not describe that array as binary.

### Adaptive gain-corrected difference

Estimate one fixed gain from quiet frame pairs and freeze it for the primary
adaptive lane.

For each quiet pair, estimate a bounded robust slope on valid anatomy pixels:

```text
alpha = sum(x0 * x1) / sum(x0 * x0)
```

Then repeat for the configured number of refinement iterations:

1. compute residual `r = x1 - alpha*x0`;
2. discard the configured largest absolute-residual fraction;
3. recompute the slope on retained samples;
4. clip to `[alpha_min, alpha_max]`.

Use the median accepted quiet-pair slope as `alpha_quiet`. Record the complete
quiet-pair slope distribution and rejected/degenerate pair count.

The primary adaptive derivative is

```text
D_adaptive[t] = F[t] - alpha_quiet * F[t-k].
```

An optional per-frame causal gain sensitivity may be implemented later, but it
must be a distinct method ID and may not replace the frozen calibration lane in
version 1.

### Binary temporal score maps

For a temporal evaluation window `T_w`, the primary binary score map is
occupancy:

```text
A[p] = sum_t M[t, p] / number_of_defined_frames.
```

Rank equal occupancy scores deterministically using, in order:

1. maximum positive `z` within the window;
2. `y` coordinate;
3. `x` coordinate.

Record that positive `z` is only a tie-breaker. Also write positive-z temporal
pooling as a diagnostic comparison, not as the primary binary method.

### Optional morphology sensitivities

For each configured minimum connected-component size, create a sensitivity lane
that removes smaller two-dimensional components per frame. The unfiltered
`minimum_component_pixels=1` lane is primary and directly represents the
professor's threshold rule.

Do not introduce erosion, dilation, temporal linking, or a learned morphology
filter in version 1.

### Wave 1 acceptance

Implementation acceptance:

- binary arrays contain only `0/1`;
- output frame alignment exactly matches the source review interval;
- quiet statistics are fixed across later frames;
- causal preprocessing has no future-frame access;
- fixed and adaptive lanes are deterministic across repeated runs;
- adaptive gain improves reconstruction on a synthetic gain-drift fixture;
- per-frame application timing is measured after calibration.

Scientific advancement gate:

- do not replace Raw Direct unless, at the Raw Direct per-burst candidate budget,
  the method's mean known-label recall is no more than `0.02` below Raw Direct
  and its held-out quiet behavior is no worse; or a completed manual candidate
  review demonstrates a meaningful precision improvement;
- a sparse-looking TIFF is not sufficient evidence.

## Wave 2 — two-observation ICA

### Sampling policy

A uniform full-field sample will be dominated by static background, while a
fully activity-enriched sample changes the target density. Implement both but
make `uniform_anatomy` primary.

Build the primary anatomy sampling mask from quiet data only:

- finite pixels;
- not saturated in the quiet median;
- inside robust quiet-intensity bounds recorded in preflight;
- exclude invalid borders introduced by optional registration.

Sample pair/pixel observations uniformly without replacement using the manifest
seed. Do not use labels to choose pixels.

An `activity_enriched_sensitivity` policy may reserve a declared fraction for
large positive fixed derivatives, but all reports must label it as a changed
sampling distribution and a sensitivity result.

### Global pooled fit and bounded pairwise diagnostics

Implement two fitting scopes:

1. `global_pooled`: fit one stable demixing rule from bounded pair/pixel samples
   distributed across the review interval, then apply it to all review pairs;
2. `per_pair_diagnostic`: fit only the explicitly configured diagnostic frames
   to measure angle/component stability and computational cost.

Do not fit a separate quadratic-density model for every frame in the complete
stack in version 1.

### Centering and whitening

For `X` with shape `[2, N]`:

1. subtract the two-channel sample mean;
2. estimate the symmetric covariance;
3. eigendecompose it;
4. floor eigenvalues at `max_eigenvalue * 1e-6` unless the manifest later
   exposes a stricter validated value;
5. construct whitening and dewhitening matrices;
6. report the raw and floored condition numbers.

If the effective rank is less than two or the condition number exceeds `1e8`,
mark the fit `unidentifiable`. The runner must preserve the diagnostic and fall
back to the fixed/adaptive difference output for that pair rather than emitting
an invented ICA component.

After whitening, the remaining two-dimensional orthogonal transform has one
sign/permutation-unique angle in `[0, 90)` degrees. Use that fact for bounded
searches and diagnostics.

### Bell-style InfoMax reference

Implement a deterministic two-output natural-gradient reference using a
super-Gaussian tanh/logistic score. A suitable update is:

```text
Y = W @ Z
Phi = tanh(Y)
G = I - (Phi @ Y.T) / N
W_next = W + learning_rate * G @ W
W_next = symmetric_decorrelate(W_next)
```

Requirements:

- initialize from each configured angle, not from uncontrolled random matrices;
- use symmetric row decorrelation every iteration;
- converge on both objective change and demixing-matrix change;
- retain every restart result;
- select the best converged restart by the declared InfoMax contrast;
- write nonconvergence instead of silently returning the last iterate as a
  successful fit;
- validate sign/permutation equivalence on synthetic mixtures.

The method ID is `infomax_tanh_ica`. Reports must state that it is the standard
reference lane, while Cauchy-Schwarz ICA is the proposed information-theoretic
lane.

### Cauchy-Schwarz Parzen independence criterion

For standardized outputs `u` and `v`, construct one-dimensional Gaussian Parzen
kernel matrices `K_u` and `K_v` using a fixed declared bandwidth. Compute the
bounded sample estimates:

```text
V_joint = mean(K_u * K_v)
V_product = mean(K_u) * mean(K_v)
V_cross = mean(row_mean(K_u) * row_mean(K_v))
D_CS = -log(V_cross / sqrt(V_joint * V_product)).
```

Clamp only for numerical safety and record every clamp. Use the same kernel
normalization consistently. The objective is minimized when the estimated joint
density most closely factorizes into the estimated marginal product under this
criterion.

Implementation requirements:

- calculate kernel rows in blocks; never allocate an unbounded full-field Gram
  matrix;
- screen angles over `[0, 90)` at the configured coarse step using
  `screen_samples`;
- refine only around the best coarse angle using the configured half-width and
  step;
- confirm the winning angle and immediate neighbors using `confirm_samples`;
- write `objective_by_angle.tsv` and the exact sample identities/seeds;
- report the fixed-difference direction in the same whitened angle coordinate;
- report the angular difference between learned and subtraction directions;
- report objective values at identity, sum/difference, InfoMax, and CS-optimal
  directions.

The Parzen bandwidth is fixed before label evaluation. Do not optimize bandwidth
on held-out known-label recall.

### Deterministic component orientation and selection

ICA outputs are sign- and order-ambiguous. Preserve both recovered components,
then select the activity-like component without labels.

For each component:

1. calculate correlation with the fixed signed derivative on the fit sample;
2. orient the sign so that this correlation is nonnegative;
3. calculate positive skewness and upper-tail fraction;
4. calculate absolute correlation with the sum/common image;
5. calculate quiet robust scale and positive excursion fraction.

Select lexicographically:

1. largest derivative correlation;
2. if within `0.05`, largest positive skewness;
3. if still tied, smallest absolute common-image correlation;
4. final deterministic tie: lower component index.

If neither component has derivative correlation at least `0.1` and positive
skewness greater than zero, mark component selection `unresolved`. Keep both
components and do not manufacture a binary activity mask for that fit.

Write all selection statistics. Never choose the component with higher held-out
label recall.

### ICA binary masks

For a resolved activity component, estimate quiet center/MAD from its quiet
outputs and apply the same one-sided threshold contract used by the fast lanes.
This makes thresholding comparable across methods.

Also retain the continuous component. The binary mask is not the only scientific
artifact.

### Wave 2 acceptance

Synthetic acceptance:

- recover a known non-orthogonal two-source mixture up to sign/permutation with
  absolute source correlation at least `0.90` at the declared noise level;
- recover the background-null/difference direction in the equal-background
  special case within `5` degrees after whitening;
- CS divergence at the selected angle is no greater than at identity and the
  fixed sum direction;
- degenerate jointly Gaussian/rank-one cases are flagged rather than presented
  as identified sources.

Real-data implementation acceptance:

- all fits store means, covariance, whitening, demixing, component-selection
  statistics, objective, convergence, sample IDs, and runtime;
- repeated deterministic fits reproduce the same angles and outputs;
- pairwise diagnostic fits report angle stability and failure counts;
- full-interval application is linear/chunked after the global fit.

Scientific interpretation gate:

- if the learned activity direction is within `2` degrees of subtraction and the
  continuous maps correlate above `0.99`, report that subtraction is an
  excellent approximation; do not call ICA a failure;
- if ICA differs materially, it advances only if the difference is stable across
  subsamples/restarts and improves held-out ranking or reviewed selectivity;
- unstable fit angles or unresolved components stop expansion of the ICA search.

## Wave 3 — constrained nonnegative pair decomposition

Do not run vanilla NMF on `[frame_0, frame_1]`. Implement the stated biological
constraint directly:

```text
I0 ~= B
I1 ~= alpha * B + S
B >= 0
S >= 0
```

with objective

```text
0.5 * ||I0 - B||^2
+ 0.5 * ||I1 - alpha*B - S||^2
+ lambda_activity * sum(S).
```

Use the quiet-calibrated `alpha_quiet` from the adaptive-difference lane and
freeze it in the primary NMF lane. Normalize frames with a fixed quiet-derived
intensity scale before applying the dimensionless activity penalty.

A vectorized alternating update is sufficient for version 1:

```text
B <- max((I0 + alpha*(I1 - S)) / (1 + alpha^2), 0)
S <- max(I1 - alpha*B - lambda_activity, 0)
```

Repeat until the configured relative objective tolerance or iteration cap.
Record the complete objective sequence, monotonicity violations, reconstruction
error, nonnegative constraint checks, and convergence state.

This model may reduce almost exactly to a positive adaptive residual. That is a
valid theoretical result. Add the following explicit equivalence diagnostic:

- correlation between `S` and `max(D_adaptive, 0)`;
- normalized mean absolute difference;
- candidate overlap and metric equality.

If correlation exceeds `0.995` and candidate/metric differences are negligible,
stop. Document that the constrained pair NMF collapses to the fast positive
residual under the chosen assumptions. Do not respond with a blind parameter
sweep.

Background spatial regularization, total variation, and multi-frame/windowed NMF
are deferred. Add them only after the pair model demonstrates residual structure
that cannot be explained by adaptive subtraction.

### Wave 3 acceptance

- all returned `B` and `S` values are nonnegative within numerical tolerance;
- objective is nonincreasing or every violation is explicitly recorded and
  treated as nonconvergence;
- reconstruction and activity sparsity are reported;
- synthetic positive-innovation fixtures are recovered;
- a negative-only change does not become a positive activity source;
- the trivial `W=X, H=I` solution is impossible by construction.

## Wave 4 — paired benchmark and artifacts

### Anchors

Every complete benchmark must include:

- `raw_direct` with exact current semantics;
- `causal_artifact_only` if its current reusable preprocessing can be invoked
  without changing its contract;
- all enabled new methods;
- fixed difference direction and common/sum component diagnostics.

Raw Direct must reproduce `0.605615942` before any comparative result is trusted.
If it does not, stop the run and mark the experiment invalid.

### Outer-fold and threshold rules

- preserve the four-burst evaluation organization;
- derive thresholds from quiet windows only;
- use the same NMS distance and match radii for every lane;
- evaluate each method at its quiet-calibrated threshold;
- additionally capacity-match every method to the Raw Direct per-burst candidate
  count;
- do not select a method or threshold using the burst it is being reported on;
- report all preregistered thresholds, not only the best label-performing one.

### Required metrics

Scientific ranking metrics:

- mean and pooled known-label recall;
- recall by burst and match radius;
- total event candidates;
- held-out quiet peaks per map;
- capacity-matched recall;
- known-label candidate fraction labeled explicitly as a lower bound;
- fold wins/losses versus Raw Direct;
- paired ROI-identity bootstrap difference where the existing contract applies.

Separation diagnostics:

- learned angle and difference-angle delta;
- InfoMax and CS objectives;
- component correlation/skew/tail/common-image statistics;
- fit convergence and restart/subsample stability;
- NMF reconstruction/sparsity/equivalence metrics;
- fixed versus adaptive quiet residual variance.

Mask diagnostics:

- quiet and burst nonzero fractions;
- connected-component size distributions;
- temporal occupancy distributions;
- positive/negative derivative fractions;
- static-artifact-region occupancy where the existing artifact mask is
  available.

Runtime/resource metrics:

- calibration fit time;
- per-frame transform time after calibration;
- binary threshold time;
- candidate-map/NMS time;
- median, p95, p99, and maximum;
- peak RSS and output bytes.

Only the fixed/adaptive binary path has a version-1 real-time target. Its complete
post-calibration preprocessing plus mask generation should remain below the
20 ms frame period at p95 on the development PC. ICA/NMF fitting is offline; the
application of a frozen two-by-two demixer should nevertheless be timed
separately because it may be suitable for real-time use after calibration.

### Manual review queue

Write a deterministic stratified candidate queue with the configured row cap.
Include, when available:

- Raw-only candidates;
- fixed/adaptive-only candidates;
- ICA-only candidates;
- NMF-only candidates;
- consensus candidates;
- held-out quiet candidates;
- high-motion/high-artifact candidates;
- candidates near known labels and candidates far from all known labels.

Required fields:

```text
candidate_id
lane
frame_or_burst_id
score
x_px
y_px
matched_known_label
nearest_known_label_px
source_stratum
review_status
review_label
review_note
interpretation
```

Unreviewed entries have `review_status=unreviewed` and
`interpretation=unknown_candidate`.

## Motion sensitivity

The primary version-1 comparison is unregistered to remain comparable with the
current derivative evidence. Preflight must still estimate bounded integer
shifts on a sampled frame set using `neurobench.algorithms.motion` and report the
shift distribution.

When `run_integer_shift_sensitivity` is true, create a separate registered
sensitivity preprocessing lane using the existing integer-shift helper. Record
introduced invalid borders and exclude them consistently from fitting and
metrics.

Do not claim that a two-by-two ICA matrix removes translation, rotation,
deformation, or paired motion edges. If motion sensitivity materially changes
results, registration becomes a prerequisite research question rather than an
ICA tuning parameter.

## Artifact contract

A successful run root must contain at least:

```text
config.resolved.json
preflight.json
run_state.json
progress.jsonl
input_manifest.json
label_projection_overlay.png
sampling/
  primary_samples.npz
  sample_manifest.json
methods/
  <method_id>/
    fit.json
    parameters.npz
    diagnostics.json
    timing.json
    continuous_activity.npy
    binary_mask.npy
    binary_mask.tif                 # only when enabled
    candidate_maps.npz
    objective_by_angle.tsv          # ICA where applicable
candidates/
  candidate_peaks.tsv
  candidate_review_queue.tsv
metrics.json
experiment_summary.json
report.md
figures/
  method_comparison_montage.png
  objective_by_angle.png
  recall_candidate_tradeoff.png
  quiet_burst_occupancy.png
```

Large arrays may be memory-mapped or compressed, but their dtype, shape, axes,
frame alignment, normalization, undefined-leading-frame behavior, checksum, and
method ID must be declared in metadata.

Write to `.partial` files and atomically rename only after validation. On failure,
preserve `run_state.json`, progress, and bounded error information; do not rename
partial scientific arrays as completed outputs.

### `fit.json` minimum fields

```text
schema_version
experiment_id
method_id
fit_scope
status
source_video_sha256
sample_manifest_sha256
lag_frames
preprocessing
sample_count
seed
mean
covariance
whitening
demixing
mixing
objective_name
objective_value
converged
iterations
activity_component
activity_sign
component_selection
runtime_seconds
warnings
```

NMF may replace whitening/demixing fields with explicit background/activity
parameters while retaining the common identity, status, sample, and runtime
fields.

## Preflight contract

Preflight is read-only except for an explicitly supplied new preflight artifact
directory. It must verify:

- source video/TIFF/labels/design paths and checksums;
- source shape/dtype/axes and frame intervals;
- label coordinates and frame projection;
- output collision;
- expected binary, continuous, TIFF, candidate, and diagnostic bytes;
- available disk and RAM against configured caps;
- sample population and deterministic sample feasibility;
- estimated quadratic kernel work:
  `angle_count * sample_count^2` and block memory;
- CPU thread bounds;
- motion diagnostic cost;
- exact method and threshold counts;
- whether every enabled method has a valid configuration;
- that the current source includes enough quiet pairs after lagging.

The preflight JSON must state which tasks are permitted:

```text
synthetic_smoke_ready
cpu_run_ready
full_spon_run_requires_explicit_user_selection
```

A passed preflight is not permission to launch the full Spon run.

## Tests

### Algorithm tests

Create deterministic synthetic fixtures covering:

1. equal shared background plus sparse positive activity;
2. gain drift with known `alpha != 1`;
3. non-orthogonal mixing of two independent non-Gaussian sources;
4. rank-one/near-singular adjacent frames;
5. jointly Gaussian sources where ICA is not uniquely identifiable;
6. positive onset followed by slow calcium-like persistence;
7. negative-only decay;
8. one-pixel translation producing paired edges;
9. saturated and zero-MAD pixels;
10. constrained NMF equivalence to positive adaptive residual.

Required assertions include:

- subtraction cancels exactly equal background in the noiseless fixture;
- adaptive gain reduces residual error relative to fixed subtraction in the gain
  fixture;
- quiet MAD flooring avoids divide-by-zero and remains deterministic;
- binary output contains only `0/1` and is one-sided;
- whitening yields identity covariance within tolerance when identifiable;
- InfoMax and CS-ICA recover sources up to sign/permutation at the declared
  synthetic noise level;
- CS divergence is finite, deterministic, and block-size invariant within
  tolerance;
- component orientation is label-free and deterministic;
- unidentifiable cases return an explicit state;
- NMF outputs are nonnegative and do not activate on negative-only changes.

### Runner/integration tests

Use a tiny TIFF/NPY fixture and small label table. Verify:

- config path resolution;
- preflight byte/work estimates;
- output collision refusal;
- one-based/zero-based frame alignment;
- atomic artifacts and no leftover successful `.partial` files;
- interrupted/failed state recording;
- method disabling/enabling;
- exact candidate ordering;
- report fields and interpretation wording;
- CLI lazy loading and exit codes;
- deterministic repeated-run equality in separate new output roots.

### Existing regression suite

Run at minimum:

```bash
.venv-neurobench/bin/python -m pytest -q \
  tests/test_frame_difference.py \
  tests/test_smoothed_frame_difference.py \
  tests/test_activity_gate_benchmark.py \
  tests/test_pairwise_separation_algorithms.py \
  tests/test_pairwise_separation_config.py \
  tests/test_pairwise_separation_runner.py \
  tests/test_pairwise_separation_cli.py
```

Then run the complete suite before handoff:

```bash
.venv-neurobench/bin/python -m pytest -q
```

Do not weaken or delete an existing test to make the new workflow pass.

## Reporting contract

`report.md` must be answer-first and contain:

1. **Validity** — source identity, Raw Direct reproduction, preflight status,
   and any failed method fits;
2. **Theoretical finding** — learned directions, CS/InfoMax objectives, and
   whether subtraction is recovered as the background-null approximation;
3. **Practical finding** — fixed/adaptive binary-mask behavior and timing;
4. **Detection comparison** — recall, quiet behavior, candidate burden, and
   capacity-matched results;
5. **NMF finding** — whether constrained NMF is distinct from adaptive positive
   residual;
6. **Motion sensitivity** — measured shifts and registered sensitivity when run;
7. **Unknowns** — sparse labels, unreviewed candidates, and unresolved
   components;
8. **Decision** — advance, retain as auxiliary, document equivalence, or stop;
9. **Next valid experiment** — one bounded action, not a blind sweep.

The report must never state "ICA cleaned the image" unless a declared quantitative
and reviewed criterion establishes what "cleaned" means. Prefer:

- "activity-like component";
- "background attenuation";
- "known-label recall";
- "unreviewed candidate";
- "binary onset mask."

## Scientific decision gates

Use these statuses in `experiment_summary.json`:

```text
invalid_baseline
implementation_only
fixed_difference_supported
adaptive_difference_supported
ica_equivalent_to_difference
ica_promising_requires_review
ica_unstable_stop
nmf_equivalent_to_adaptive_residual
nmf_promising_requires_review
no_method_advances
```

Decision precedence:

1. If Raw Direct does not reproduce, status is `invalid_baseline`.
2. If code/synthetic tests pass but the full benchmark/manual review has not run,
   status is `implementation_only`.
3. If ICA is stable but equivalent to subtraction, use
   `ica_equivalent_to_difference`; this is a theoretically useful result.
4. If ICA is unstable across bounded restarts/subsamples, use
   `ica_unstable_stop`; do not widen the search.
5. If NMF collapses to adaptive positive residual, use
   `nmf_equivalent_to_adaptive_residual`; do not add arbitrary NMF ranks.
6. A method is `promising_requires_review` only when it improves a preregistered
   ranking/selectivity criterion without violating held-out quiet behavior, and
   unmatched candidates still require review.
7. No automatic method replacement occurs in this experiment.

## Parallel-safe implementation work packages

The following work may proceed concurrently only when agents avoid overlapping
files.

### Package A — numerical algorithms

Owns:

- `neurobench/algorithms/pairwise_separation.py`;
- `tests/test_pairwise_separation_algorithms.py`.

Delivers all pure-array operations and synthetic source-recovery tests.

### Package B — config and preflight

Owns:

- `neurobench/experiments/pairwise_separation/config.py`;
- `neurobench/experiments/pairwise_separation/preflight.py`;
- `tests/test_pairwise_separation_config.py`;
- the example manifest.

Delivers strict loading, path/frame/resource validation, and estimates.

### Package C — evaluation extraction

Owns:

- `neurobench/metrics/sparse_detection.py`;
- compatibility changes in `activity_gate_benchmark.py`;
- exact regression tests for current metrics.

This package must land and pass parity before the new runner relies on it.

### Package D — runner, artifacts, report, and CLI

Owns the remaining experiment package, CLI registration, integration tests, and
workflow documentation. It depends on A, B, and C public interfaces.

Before integration, reconcile dataclass/function signatures explicitly. Do not
solve merge conflicts by duplicating near-identical metric or algorithm code.

## Ordered implementation checklist for Codex

1. Inspect the working tree and current branch. Preserve all user changes.
2. Read the required files and record the current Raw Direct contract.
3. Implement Wave 0 metric extraction and prove exact parity.
4. Implement pure fixed/adaptive algorithms and synthetic tests.
5. Implement strict config/preflight and example manifest.
6. Implement InfoMax reference, CS-Parzen objective/search, component selection,
   and synthetic identifiability tests.
7. Implement constrained shared-background NMF and equivalence diagnostics.
8. Implement runner, artifacts, binary/continuous outputs, timing, and failure
   recovery.
9. Add the thin CLI group.
10. Run synthetic and tiny integration tests only.
11. Run the focused regression suite, then the full test suite.
12. Write the truthful workflow document describing implemented behavior and
    commands.
13. Update `AGENTS.md`, `docs/CODEBASE_NAVIGATION.md`, and
    `docs/developer/README.md` so the new workflow is discoverable.
14. Do **not** launch the full Spon experiment. End with the exact preflight
    command and state that the long run still requires explicit user selection.

## Definition of done

The implementation task is complete only when:

- all five declared lanes exist behind one manifest-driven experiment;
- the professor's exact positive binary threshold rule is represented directly;
- adaptive subtraction is an explicit, independently reported lane;
- InfoMax and CS-ICA are implemented without a new heavy dependency;
- CS kernels are sample- and memory-bounded;
- ICA ambiguity/unidentifiability is handled explicitly;
- constrained NMF cannot select the trivial identity factorization;
- the existing Raw Direct numerical anchor is preserved;
- all artifacts are collision-safe, atomic, aligned, and auditable;
- synthetic, focused, and full repository tests pass;
- no full Spon/GPU run was launched without a new explicit user instruction;
- the handoff states what was implemented, what was only smoke-tested, and the
  exact command for the next authorized preflight.
