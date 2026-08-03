# Codex implementation specification: event-balanced two-frame CS-Parzen ICA

## 0. Mission

Extend the existing two-frame Cauchy-Schwarz (CS) Parzen ICA experiment so that rare annotated neural-event observations can receive controlled statistical mass during fitting **without physically duplicating frames** and without changing the untouched, naturally sparse evaluation distribution.

The implementation must answer one narrow scientific question:

> Does increasing the declared event mass in the training empirical distribution cause a stable, held-out-generalizing departure from the derivative-like global ICA solution?

This is a diagnostic experiment, not a claim of neural/background source separation.

---

## 1. Required scientific constraints

1. Preserve the current causal preprocessing contract:
   - spatial Gaussian smoothing, `sigma_px = 1.0`;
   - causal EMA, `alpha = 0.4`;
   - adjacent-frame observation `x = [P[t-1], P[t]]`.
2. Reproduce the existing unweighted CS-Parzen result before introducing weighting.
3. Implement repetition as nonnegative sample weights, not duplicated arrays or duplicated video frames.
4. Treat a sample as a **pixel-time pair**, not an entire frame.
5. Keep whitening fitted from the natural training distribution in the primary experiment.
6. Never treat unlabeled pixels as confirmed negatives.
7. Split by complete burst interval before constructing event weights or augmented sample pools.
8. Evaluate at natural prevalence. No weighting, oversampling, or copied samples are allowed in validation/test evaluation.
9. Preserve current sparse-positive reporting language:
   - known-label recall is identifiable;
   - candidate count is identifiable;
   - precision is not identifiable from sparse-positive labels.
10. Defer spatially varying or time-varying angle fields until the global weighted experiment passes the stage gate in Section 15.

---

## 2. Source packet and baseline contract

Use the uploaded packet as the regression contract:

- `CHATGPT_HANDOFF.md`
- `METHOD_REFERENCE.md`
- `preliminary_metrics.json`
- `source_evidence/fit.json`
- `source_evidence/real_data_metrics.json`

Baseline facts that must be reproduced within numerical tolerance:

| Quantity | Expected value |
|---|---:|
| Movie shape | `2359 x 340 x 573` in `T,Y,X` |
| Review interval | UI `1800-2359`, inclusive |
| Quiet interval | UI `1800-1899`, inclusive |
| Frame period | `20 ms` |
| Gaussian sigma | `1 px` |
| EMA alpha | `0.4` |
| Parzen bandwidth | `0.35` |
| Screen samples | `1024` |
| Confirmation samples | `4096` |
| Coarse angle step | `3 deg` |
| Refinement angle step | `0.25 deg` |
| Kernel block rows | `256` |
| Objective evaluations | `58` |
| Converged | `true` |
| Learned-direction cosine to derivative | `0.999999917349` |
| Sampled `corr(Y,D)` | `0.997828106997` |
| CS-Parzen known-label mean recall | `0.133333333` |
| CS-Parzen total event candidates | `24` |

Do not change baseline defaults while adding weighting.

---

## 3. Pre-implementation repository discovery

The zip is an evidence packet, not the source tree. Before editing code, locate the existing implementation by searching the repository for these exact strings:

```text
cs_parzen_ica
kernel_block_rows
objective_by_angle
bandwidth 0.35
activity_component
fixed_binary_difference
adaptive_binary_difference
```

Then identify:

1. the causal preprocessing implementation;
2. the sample-index generator;
3. whitening and rotation code;
4. CS-Parzen objective implementation;
5. coarse-to-fine angle optimizer;
6. component orientation logic;
7. detection/evaluation pipeline;
8. JSON and figure writers;
9. existing unit/integration tests.

**Do not create a parallel replacement pipeline** if the current code can be extended cleanly. Add a weighted path while retaining the exact unweighted path.

If the current structure has no reasonable separation of concerns, create logical modules equivalent to:

```text
parzen/
  cs_objective.py             # existing/unweighted and new weighted objective
  sample_weights.py           # event-mass construction and diagnostics
experiments/
  event_weighted_pairwise_ica.py
configs/
  event_weighted_pairwise_ica_standard.yaml
tests/
  test_weighted_cs_objective.py
  test_event_weight_construction.py
  test_event_split_integrity.py
```

Adapt names to the repository's conventions.

---

## 4. Method overview

For every valid pixel-time sample `n = (t, y, x)`:

```text
R_t = raw fluorescence
P_t = EMA_0.4(G_sigma=1 * R_t)
x_n = [P_(t-1)(y,x), P_t(y,x)]^T
z_n = Q_nat (x_n - mu_nat)
y_n(theta) = R(theta) z_n
```

The primary weighted empirical distribution is

```text
P_alpha = (1 - alpha) P_natural + alpha P_event
```

where:

- `P_natural` is sampled from training frames at natural prevalence;
- `P_event` contains event-support samples from training bursts only;
- each training burst receives equal total mass within `P_event`;
- `alpha = 0` must be exactly the current unweighted baseline.

The fitted angle is

```text
theta_star(alpha) = argmin_theta D_CS(
    p_joint_alpha_theta,
    p_marginal1_alpha_theta * p_marginal2_alpha_theta
)
```

using the existing bounded angle search.

---

## 5. Data structures

### 5.1 Sample index

Represent sample identity independently from sample values.

```python
@dataclass(frozen=True)
class PairSampleIndex:
    frame_ui: int          # current frame t, one-based UI convention
    y: int                 # row
    x: int                 # column
    event_id: int | None   # None for ordinary natural samples
    stratum: str           # natural | event_frame | event_roi
    phase: str | None      # onset | peak | decay | None
```

Maintain a vectorized representation for production code, but preserve enough metadata to audit leakage and per-event mass.

### 5.2 Weighted sample batch

```python
@dataclass
class WeightedPairBatch:
    samples: np.ndarray       # shape [N, 2], float32 or float64
    weights: np.ndarray       # shape [N], float64, finite and >= 0
    indices: PairSampleIndexTable
    weight_sum: float
    weight_ess: float
    per_event_mass: dict[int, float]
```

Validate:

```text
samples.ndim == 2
samples.shape[1] == 2
weights.shape == (N,)
all(weights >= 0)
allfinite(samples, weights)
sum(weights) > 0
```

Normalize weights internally only where required. Persist both raw and normalized summaries.

---

## 6. Training distribution construction

### 6.1 Natural pool

Construct the natural pool from training frames only, after removing:

- the held-out burst interval;
- a temporal guard around the held-out interval;
- invalid leading lag frames;
- user-declared bad/outlier frames, if an existing mask is available.

Keep the current sampling strategy and seed for `alpha=0`. Do not silently change spatial or temporal sampling.

### 6.2 Event pool: equal mass per burst

Let the training event IDs be `J_train`. For event `j`, build an index set `E_j`.

The event empirical distribution is:

```text
P_event = mean_j P(E_j)
```

Therefore each event receives total normalized mass `1 / len(J_train)` regardless of:

- burst duration;
- number of labeled rows;
- ROI area;
- number of sampled points.

Within event `j`, assign uniform mass unless phase balancing is explicitly enabled.

### 6.3 Required event modes

Implement two distinct modes.

#### `frame_balanced`

Event support includes sampled pixel-time pairs from annotated burst frames, independent of distance from a label.

Purpose: faithfully test the user's original whole-frame repetition idea and expose global event-time structure.

#### `roi_balanced`

Event support includes pixel-time pairs within a disk around a sparse-positive annotation during its associated burst interval.

Default radius:

```yaml
event_roi_radius_px: 3
```

The radius must be configurable and logged. Use the existing coordinate convention `x = column`, `y = row`.

Do not infer negatives outside the ROI.

### 6.4 Fixed sample support across alpha

For each outer fold, weighting mode, and random seed:

1. draw the natural sample indices once;
2. draw the event sample indices once, with equal caps per training event;
3. reuse those exact indices for every `alpha`;
4. change only the mixture weights.

This prevents Monte Carlo resampling noise from being misinterpreted as an alpha-dependent angle shift.

Conservative standard caps:

```yaml
event_screen_max_samples_per_event: 128
event_confirmation_max_samples_per_event: 512
```

With three training events, the largest standard screen set is approximately `1024 + 3*128 = 1408` unique samples and the largest confirmation set is approximately `4096 + 3*512 = 5632`, before overlap merging. If fewer unique event samples exist, use all of them. Never fill a quota by duplicating indices.

The natural subset must be the same one used by the `alpha=0` regression lane. Event rows may be appended with zero weight at `alpha=0`, which preserves the baseline objective exactly while keeping sample support fixed across the sweep.

### 6.5 Optional phase balancing

Do not enable in the standard profile. When enabled, divide each burst into declared phases:

```text
onset | peak | decay
```

Each phase receives a configurable share of that event's mass. Phase rules must be deterministic and recorded.

---

## 7. Event-mass parameterization

Implement `alpha` directly.

```yaml
alpha_grid: [0.0, 0.01, 0.025, 0.05, 0.10, 0.20, 0.35]
```

For an intuitive report, also compute the repetition-equivalent factor:

```text
r_equiv(alpha) = alpha * N_natural / ((1 - alpha) * N_event)
```

This value is descriptive only. Never materialize repeated samples.

### 7.1 Merge overlapping sample identities

A positive event sample may also be present in the natural pool. Construct mixture weights by sample identity:

```text
w_total(index) = (1 - alpha) * w_natural(index)
               + alpha * w_event(index)
```

Then merge identical indices and sum their weights. This exactly represents the mixture measure without redundant rows.

### 7.2 Weight effective sample size

Compute:

```text
weight_ess = (sum(w)^2) / sum(w^2)
```

Report both:

```text
weight_ess
weight_ess_fraction = weight_ess / N_unique
```

This measures concentration due to weighting; do not call it the autocorrelation-adjusted temporal effective sample size.

---

## 8. Whitening modes

### 8.1 `natural_fixed` — required default

For every outer fold:

1. fit `mu_nat` and `Q_nat` from the fold's natural training pool;
2. freeze them for all `alpha` and both event modes;
3. vary only the weighted higher-order CS objective.

This isolates the effect of event weighting on the ICA rotation.

### 8.2 `weighted` — secondary ablation

Optionally compute weighted mean/covariance and whitening for each `alpha`.

This changes both second-order and higher-order geometry and must be reported as a separate lane. Do not mix its results with the primary lane.

### 8.3 Numerical safeguards

Retain existing covariance regularization. Log:

```text
covariance
condition_number
eigenvalues
regularization or eigenvalue floor
```

Fail clearly if the whitening matrix is non-finite.

---

## 9. Weighted CS-Parzen objective

### 9.1 API contract

Extend the current objective to accept optional weights:

```python
def cs_parzen_objective(
    y: np.ndarray,                  # [N, 2]
    bandwidth: float,
    *,
    weights: np.ndarray | None = None,
    block_rows: int = 256,
    accumulator_dtype: np.dtype = np.float64,
) -> CSObjectiveResult:
    ...
```

Required behavior:

- `weights=None` is exactly equivalent to all-ones weights;
- multiplying every weight by a positive scalar leaves the objective unchanged;
- zero-weight samples contribute nothing;
- negative or non-finite weights raise a validation error.

### 9.2 Exact weighted information-potential terms

Let `w` be nonnegative, `W = sum(w)`, and let `K1`, `K2` be Gaussian overlap-kernel matrices for output dimensions 1 and 2, using the same bandwidth convention as the existing code.

Compute:

```text
V_joint = w^T (K1 elementwise_mul K2) w / W^2
V_m1    = w^T K1 w / W^2
V_m2    = w^T K2 w / W^2
V_prod  = V_m1 * V_m2
V_cross = w^T ((K1 w) elementwise_mul (K2 w)) / W^3
D_CS    = -log(V_cross / sqrt(V_joint * V_prod))
```

Preserve the existing numerical-clamp convention. Log each clamp.

### 9.3 Blockwise implementation

Do not construct resident `N x N` matrices in the standard implementation.

For row blocks `B`:

```python
for start in range(0, N, block_rows):
    stop = min(start + block_rows, N)
    yb = y[start:stop]
    wb = w[start:stop]

    K1 = gaussian_overlap(yb[:, 0, None] - y[:, 0][None, :], h)
    K2 = gaussian_overlap(yb[:, 1, None] - y[:, 1][None, :], h)

    k1w = K1 @ w
    k2w = K2 @ w

    joint_num += dot(wb, (K1 * K2) @ w)
    m1_num    += dot(wb, k1w)
    m2_num    += dot(wb, k2w)
    cross_num += dot(wb, k1w * k2w)
```

Use:

- float32 for temporary kernel blocks unless baseline parity requires float64;
- float64 for all scalar accumulators;
- no autograd in the first implementation;
- the existing bounded angle grid, not gradient descent.

### 9.4 Semantic parity test

On a small toy dataset with integer weights, compare:

1. weighted objective on unique samples;
2. unweighted objective on explicitly repeated samples.

They must match within a strict tolerance after accounting for the existing kernel normalization convention.

---

## 10. Angle search and canonical orientation

Retain the current coarse-to-fine search:

```yaml
coarse_step_deg: 3.0
refine_step_deg: 0.25
screen_samples: 1024
confirmation_samples: 4096
kernel_block_rows: 256
bandwidth: 0.35
```

Use the same seed `20260727` for baseline parity, then allow a configurable experiment seed.

To avoid component-order ambiguity, preserve a canonical output convention:

1. identify the output most aligned with the common direction on the natural training pool;
2. identify the orthogonal innovation output;
3. orient the innovation output so its correlation with `D_t = P_t - P_(t-1)` is positive;
4. report the effective observation-space direction for both outputs.

Do not select a component using held-out labels.

---

## 11. Cross-validation and leakage prevention

### 11.1 Outer folds

Use leave-one-burst-out validation across the four annotated intervals.

For held-out event `j`:

```text
train events = all events except j
test event   = j
```

### 11.2 Guard interval

Default:

```yaml
heldout_guard_frames: 10
```

Remove the held-out interval plus guard frames from:

- natural training sample generation;
- natural whitening;
- event pool construction;
- bandwidth-selection data, if bandwidth is later tuned.

This protects against EMA carryover and adjacent-frame correlation leakage.

### 11.3 Quiet calibration

The existing quiet interval may be reused for robust score calibration if it is event-disjoint. It must be described as a quiet reference, not guaranteed negative ground truth.

### 11.4 Split-first rule

The implementation order must be:

```text
split -> mask/guard -> construct natural/event pools -> assign weights -> fit
```

Never build a global oversampled pool and split afterward.

---

## 12. Standard experiment matrix

Run the following first:

| Lane | Weight mode | Whitening | Alpha grid |
|---|---|---|---|
| `natural` | none | natural fixed | `[0.0]` |
| `frame_balanced` | event frame | natural fixed | full grid |
| `roi_balanced` | event ROI | natural fixed | full grid |
| `roi_balanced_weighted_whitening` | event ROI | weighted | `[0.0, 0.05, 0.10, 0.20]` |

Do not implement phase balancing, local angle fields, or discriminative CS terms in the first pull request unless the required lanes are complete and tested.

---

## 13. Evaluation outputs

### 13.1 Per-fit metrics

Persist at least:

```json
{
  "fold_id": 1,
  "heldout_event_id": 1,
  "weight_mode": "roi_balanced",
  "whitening_mode": "natural_fixed",
  "alpha": 0.10,
  "repeat_equivalent": 0.0,
  "unique_sample_count": 0,
  "weight_sum": 0.0,
  "weight_ess": 0.0,
  "weight_ess_fraction": 0.0,
  "per_event_mass": {},
  "angle_degrees": 0.0,
  "angle_shift_from_alpha0_degrees": 0.0,
  "objective_weighted_train": 0.0,
  "objective_natural_holdout": 0.0,
  "cosine_to_derivative": 0.0,
  "cosine_to_common": 0.0,
  "correlation_to_fixed_derivative": 0.0,
  "known_label_recall": 0.0,
  "candidate_count": 0,
  "precision_identified": false,
  "converged": true,
  "objective_evaluations": 0,
  "runtime_seconds": 0.0,
  "peak_rss_mb": 0.0,
  "peak_vram_mb": 0.0
}
```

Use `null` for unavailable values rather than fabricated zeros in real output.

### 13.2 Aggregate plots

Generate deterministic figures:

1. `angle_shift_vs_alpha.png`
2. `derivative_cosine_vs_alpha.png`
3. `train_and_holdout_objective_vs_alpha.png`
4. `weight_ess_vs_alpha.png`
5. `known_label_recall_vs_candidate_count.png`
6. `fold_angle_stability.png`
7. `frame_vs_roi_weighting_comparison.png`
8. `representative_weighted_outputs.png` for declared frames only

Every plot must distinguish folds and include aggregate median/range or mean/confidence interval. Do not imply statistical confidence from four folds without clearly labeling the small sample count.

### 13.3 Video output

Do not generate a full diagnostic video for every `alpha`. This is resource-heavy and visually redundant.

Standard profile:

- render videos only for `alpha = 0`, the best stable nonzero `alpha`, and one high-alpha stress case;
- reuse fixed display bounds where scientifically meaningful;
- include the fitted angle and weight diagnostics in the video manifest.

---

## 14. Hardware and resource policy

Target the user's local workstation conservatively:

- Intel i9-class CPU;
- approximately 78 GiB RAM;
- RTX 4070 SUPER with 12 GiB VRAM;
- Ubuntu;
- full movie available through memory mapping.

### 14.1 Standard profile

```yaml
compute_device: cpu
max_parallel_folds: 2
max_worker_processes: 8
screen_samples: 1024
confirmation_samples: 4096
kernel_block_rows: 256
kernel_dtype: float32
accumulator_dtype: float64
cache_preprocessed_movie: memmap
render_all_videos: false
bootstrap_replicates: 0
```

Expected policy targets:

- peak process RAM below 16 GiB per active fold;
- combined local run below approximately 32 GiB RAM;
- VRAM usage negligible in CPU mode;
- no full `N x N` kernel matrix retained;
- standard maximum unique kernel set approximately `5632` before overlap merging;
- no full float64 duplicate of the movie.

### 14.2 Optional GPU mode

GPU acceleration may be added only after CPU correctness tests pass.

Requirements:

- process one angle and one kernel block at a time;
- cap peak VRAM below 4 GiB in the standard profile;
- explicitly free temporary tensors between fits;
- preserve float64 scalar accumulation where possible;
- verify CPU/GPU objective parity.

Do not use GPU multiprocessing across folds on a single 12 GiB card.

### 14.3 Extended profile

Only after standard results exist:

```yaml
confirmation_samples: 8192
bootstrap_replicates: 8
max_parallel_folds: 1
```

Do not exceed `8192` confirmation samples without a measured scaling report. The objective is quadratic in sample count.

---

## 15. Stage gates

### Gate A — weighted objective correctness

Pass all tests in Section 17 and reproduce `alpha=0` baseline.

### Gate B — global event-weight study

Complete all four folds for frame-balanced and ROI-balanced modes across the standard alpha grid.

Proceed only if all artifacts are generated and no leakage is detected.

### Gate C — spatial extension eligibility

Do **not** implement a spatial angle field unless the global study shows all of the following default engineering criteria:

1. median nonzero angle shift of at least `1.0 degree` at one moderate alpha (`<= 0.20`);
2. shift direction is consistent in at least three of four held-out folds;
3. the shift is not exclusive to frame-balanced weighting;
4. held-out known-label recall/candidate behavior is not materially degraded;
5. weight ESS remains at least `20%` of unique sample count;
6. the result survives at least one independent sample seed.

These are configurable engineering gates, not universal scientific thresholds. Record any overrides.

### Gate D — first spatial model

If Gate C passes, begin only with:

```text
theta(p) = theta_global + delta_grid(p)
```

using a coarse `4 x 4` correction grid, bilinear interpolation, global shrinkage, and natural fixed whitening. Do not begin with one independent angle per pixel.

---

## 16. Configuration schema

Create a versioned YAML configuration similar to:

```yaml
schema_version: 1
experiment_id: spon_ca_burst_event_weighted_cs_parzen_v1

source:
  movie_path: null
  labels_path: null
  axes: TYX
  ui_one_based: true
  review_interval_ui: [1800, 2359]
  quiet_interval_ui: [1800, 1899]
  burst_intervals_ui:
    1: [2003, 2026]
    2: [2040, 2063]
    3: [2122, 2149]
    4: [2254, 2300]

preprocessing:
  gaussian_sigma_px: 1.0
  ema_alpha: 0.4
  motion_correction: none

sampling:
  seed: 20260727
  screen_samples: 1024
  confirmation_samples: 4096
  heldout_guard_frames: 10
  event_roi_radius_px: 3
  event_screen_max_samples_per_event: 128
  event_confirmation_max_samples_per_event: 512
  reuse_sample_indices_across_alpha: true
  merge_duplicate_indices: true
  equal_mass_per_event: true
  phase_balancing: false

weighting:
  modes: [frame_balanced, roi_balanced]
  alpha_grid: [0.0, 0.01, 0.025, 0.05, 0.10, 0.20, 0.35]

whitening:
  primary_mode: natural_fixed
  run_weighted_ablation: true
  covariance_regularization: existing_baseline

parzen:
  bandwidth: 0.35
  kernel_block_rows: 256
  kernel_dtype: float32
  accumulator_dtype: float64

angle_search:
  range_degrees: [0.0, 90.0]
  coarse_step_degrees: 3.0
  refine_step_degrees: 0.25

compute:
  device: cpu
  max_parallel_folds: 2
  max_worker_processes: 8
  max_peak_ram_gb: 32
  max_peak_vram_gb: 4

outputs:
  root_dir: artifacts/event_weighted_cs_parzen_v1
  render_selected_videos_only: true
  selected_video_alphas: [0.0, auto_best_stable, 0.35]
```

Resolve actual source paths through the repository's current data configuration. Do not hard-code private absolute paths in committed files.

---

## 17. Tests

### 17.1 Unit tests

1. **Uniform-weight parity**
   - `weights=None` equals `weights=np.ones(N)`.
2. **Scale invariance**
   - `objective(w) == objective(c*w)` for positive `c`.
3. **Duplication equivalence**
   - integer-weight objective equals explicit-replication objective on a small dataset.
4. **Zero-weight exclusion**
   - adding zero-weight points does not alter the objective.
5. **Validation**
   - negative, NaN, infinite, or all-zero weights fail clearly.
6. **Block parity**
   - several block sizes produce the same objective.
7. **Rotation periodicity/sign convention**
   - equivalent rotations map to canonical reported angles/components.
8. **ESS formula**
   - uniform and concentrated toy cases match closed-form results.
9. **Per-event mass**
   - events receive equal total mass despite unequal sample counts.
10. **Duplicate-index merge**
    - merged and unmerged mixture representations produce the same objective.

### 17.2 Integration tests

1. Reproduce the uploaded `alpha=0` fit within tolerances.
2. Verify no held-out event index appears in training metadata.
3. Verify guard frames are excluded.
4. Run a two-alpha, one-fold smoke experiment end-to-end.
5. Confirm deterministic JSON/plots for a fixed seed.
6. Confirm natural evaluation sample counts do not change with alpha.
7. Confirm frame-balanced and ROI-balanced pools differ as expected.
8. Measure and record peak RAM; fail or warn above configured budget.

### 17.3 Suggested tolerances

Use baseline-specific tolerances rather than exact bit equality:

```text
objective absolute tolerance: 1e-8 to 1e-6, depending on dtype path
angle tolerance: 0.25 degree or tighter
cosine-to-derivative tolerance: 1e-6
candidate/recall output: exact, assuming unchanged upstream code
```

Document any looser tolerance and its cause.

---

## 18. Failure modes and required diagnostics

### Event weighting produces no angle movement

This is a valid scientific result. Persist the full alpha path and conclude that the two-frame model remains derivative-like under event emphasis.

### Angle moves only at high alpha

Flag `weight_concentration_warning` when:

```text
weight_ess_fraction < 0.20
```

Do not select that alpha as the preferred model without explicit override.

### Frame weighting moves but ROI weighting does not

Report a likely global event-time effect. Generate motion/illumination diagnostics if existing tools support them; do not label the effect neural.

### Weighted objective improves but held-out metrics degrade

Flag probable overfitting or bandwidth mismatch. Do not tune alpha on the held-out event.

### Fold directions disagree

Report circular angle dispersion and refrain from averaging raw angles across wrapping boundaries.

### Covariance becomes ill-conditioned in weighted-whitening ablation

Keep the natural-fixed primary result. Log the condition number and abort only the affected ablation.

---

## 19. Deliverables

The implementation is complete only when it produces:

1. code changes extending the existing CS-Parzen objective with weights;
2. configuration files for smoke and standard profiles;
3. unit and integration tests;
4. one reproducible baseline report showing `alpha=0` parity;
5. all four leave-one-burst-out standard runs;
6. aggregate JSON/CSV tables;
7. the eight figures listed in Section 13.2;
8. selected diagnostic videos only;
9. a machine-readable manifest containing:
   - source hashes;
   - git commit;
   - configuration hash;
   - random seeds;
   - split definitions;
   - resource measurements;
   - scientific status string.

Use the scientific status:

```text
diagnostic_event_weighting_study_not_validated_source_separation
```

---

## 20. Explicit non-goals for the first implementation

Do not add:

- one angle per pixel;
- multiscale spatial fields;
- time-varying demixing;
- neural networks;
- synthetic noise augmentation;
- discriminative event-vs-quiet CS term;
- automatic bandwidth learning;
- motion correction changes;
- new candidate detection heuristics;
- full-video rendering for every alpha.

These changes would make it impossible to attribute results specifically to event weighting.

---

## 21. Final Codex execution order

1. Discover the current implementation and report the relevant files.
2. Add weighted-objective unit tests before production edits.
3. Implement weighted blockwise CS terms.
4. Prove uniform-weight and duplication parity.
5. Add event-pool and mixture-weight construction.
6. Add split-integrity tests.
7. Reproduce `alpha=0` baseline.
8. Run a one-fold smoke study for frame and ROI modes.
9. Inspect angle paths, ESS, runtime, and memory.
10. Run the four-fold standard study.
11. Generate aggregate artifacts.
12. Write a concise results note that distinguishes:
    - measured facts;
    - interpretations;
    - unsupported claims;
    - whether Gate C was met.

Do not proceed to local ICA automatically. Stop after reporting the global weighted study and its stage-gate result.
