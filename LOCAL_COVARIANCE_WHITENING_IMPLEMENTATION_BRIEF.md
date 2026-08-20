# Local Covariance Whitening for Spon Ca Burst

Implementation and experiment brief for Codex.

**Repository:** `Jibby2k1/NeuRev-Workbench`
**Repository snapshot reviewed:** `0972f08dca7a34bd786e513d7e2a6d0fe03b63cd`
**Proposed repository destination:** `docs/developer/LOCAL_COVARIANCE_WHITENING_IMPLEMENTATION_BRIEF.md`
**Status:** documentation-only specification
**Date:** 2026-08-15

## 1. Authorization boundary

This brief authorizes:

- repository code for CPU numerical references;
- deterministic unit tests;
- synthetic and semi-synthetic fixtures;
- manifest/schema work;
- read-only preflight;
- bounded tiny smoke tests on generated data;
- report, table, and preview-figure generation;
- a collision-safe local output root for preflight/smoke artifacts.

This brief does **not** authorize:

- a full Spon Ca Burst run;
- a GPU run;
- use of `--authorize-full-spon`;
- modification or reinterpretation of completed `Outputs/` roots;
- overwriting any prior experiment;
- automatic promotion of a whitening lane into the maintained detector;
- a claim that a whitened coordinate is a biological source or a cleaned movie.

A full real-data or GPU run requires explicit user selection, a new experiment ID/output root, source and resource preflight, and compliance with the scientific-audit contract.

## 2. Executive directive

Implement a stage-gated experiment that asks one precise question:

> After the current causal joint MSLN representation, does reproducible locally varying off-diagonal covariance remain, and does quiet-fitted shrinkage whitening improve held-out covariance calibration and sparse-event evidence beyond diagonal and global controls without distorting event morphology or temporal dynamics?

The first implementation must whiten a **small feature bank of existing MSLN contexts**, not raw spatial patches.

Primary feature vector:

```text
phi[r,t] = [
    Z_compact_s5_g1_t15[r,t],
    Z_broad_s15_g3_t23[r,t],
    Z_broad_s15_g3_t31[r,t],
]
```

Primary factorial:

```text
identity
global_diagonal
local_diagonal
      vs.
global_full_zca
local_full_zca
```

This design separates:

1. additional local scale correction from full covariance whitening;
2. global covariance modeling from spatially varying covariance modeling;
3. covariance value from general nonlinear or ICA effects.

Do not begin with full patch ZCA, unrestricted spatiotemporal covariance, adaptive online covariance, or a new broad ICA sweep.

## 3. Required reading before editing

Read in this order:

1. `AGENTS.md`
2. `docs/workflows/SCIENTIFIC_AUDIT_OUTPUT_STANDARD.md`
3. `docs/workflows/spon_ca_burst_msln_msica.md`
4. `docs/research/msln_msica_joint_residual_v2_package/README.md`
5. `docs/research/msln_msica_joint_residual_v2_package/PAPER_AND_SLIDES.md`
6. `examples/spon_ca_burst_joint_msln_residual_sweep_v2.example.json`
7. `neurobench/algorithms/multiscale_local_normalization.py`
8. `neurobench/algorithms/multiscale_subspace.py`
9. `neurobench/algorithms/pairwise_separation.py`
10. `neurobench/algorithms/quiet_calibration.py`
11. `neurobench/metrics/sparse_detection.py`
12. `neurobench/experiments/msln_msica/joint_sweep.py`
13. `neurobench/experiments/msln_msica/artifacts.py`
14. focused tests for the files above.

Use `.venv-neurobench/bin/python` for repository commands.

## 4. Current repository truth to preserve

Every generated report must state the following before interpreting new results:

- Current causal joint MSLN is a scalar studentized residual using a prior-frame spatial annulus, protected spatial core, temporal guard, and quiet-fitted scale floor.
- The maintained MSLN code does not construct a multivariate local vector or force a multivariate covariance matrix to the identity.
- The review-leading v2 lane was `joint_s15_g3_t31_g1` persistence, with 58/79 sparse known-positive matches at a fixed allocation of 58 candidates per burst; the historical Raw Direct anchor was 49/79 from 232 quiet-threshold proposals, and the protocols are not identical.
- Broad ICA directions were bootstrap-unstable and do not establish biological source identity.
- Raw -> MSICA -> MSLN did not improve the broad control in v3, and the planned cascade stopped when the raw-MSICA stability gate failed.
- The v4 five-seed energy ensemble was more stable than individual ICA source directions, but its 52/79 label-free rank-1 result remains provisional.
- Sparse annotations define known positives only. Unmatched candidates are unknown, not false positives.
- The current recording is development data because prior choices have already been influenced by its labels and results. A new run on the same recording is not an independent confirmation.
- Completion of a run is not scientific success.

## 5. Terminology contract

Use these exact names in code, manifests, artifacts, and reports:

- `msln_feature_bank`
- `covariance_calibration_support`
- `local_covariance_fit`
- `global_covariance_fit`
- `diagonal_standardized_features`
- `zca_whitened_features`
- `mahalanobis_energy`
- `empirical_quiet_surprise`
- `unresolved_tile_mask`
- `tile_blend_weight`
- `heldout_whiteness_metrics`
- `temporal_prewhitened_innovation` (later gated family)

Do not emit an artifact named only `cleaned`, `signal`, `noise`, `source`, or `denoised`.

The raw movie and signed MSLN maps remain immutable scientific references.

## 6. Scientific model

For a declared MSLN feature vector `phi[r,t] in R^d`, estimate quiet/background statistics in spatial region `tau`:

```text
mu_tau     = E[phi[r,t] | r in tau, t in quiet_fit]
Sigma_tau  = Cov[phi[r,t] | r in tau, t in quiet_fit]
```

Use shrinkage:

```text
Sigma_tilde_tau = (1 - alpha_tau) * Sigma_tau
                + alpha_tau * trace(Sigma_tau) / d * I
```

and ZCA whitening:

```text
W_tau = U_tau @ diag(max(lambda_tau, floor)^(-1/2)) @ U_tau.T

y[r,t] = W_tau @ (phi[r,t] - mu_tau)
q[r,t] = sum(y[r,t] ** 2)
```

The intended property is conditional and empirical:

```text
Cov[y | tile=tau, heldout_quiet] ~= I
```

Do not claim independence or Gaussianity. Use the chi-square distribution of `q` only as a diagnostic reference. Operational scoring uses empirical quiet-tail calibration.

## 7. V1 scope and non-goals

### 7.1 V1 primary scope

Implement:

1. a residual-covariance audit over selected MSLN contexts;
2. global diagonal and global full-ZCA controls;
3. overlapping tile-wise diagonal and full-ZCA transforms;
4. OAS covariance as primary, fixed ridge as a deterministic control, and Ledoit-Wolf as an optional comparison;
5. empirical quiet-tail calibration of Mahalanobis energy;
6. deterministic synthetic and semi-synthetic fixtures;
7. held-out quiet-block evaluation;
8. label-free lane freezing;
9. protected sparse-positive evaluation and scientific-audit integration hooks;
10. optional downstream ICA comparison only behind a passed representation gate.

### 7.2 Explicit non-goals for V1

Do not implement as primary lanes:

- full per-pixel spatial-patch ZCA;
- full spatiotemporal patch covariance;
- online/adaptive covariance updates;
- label-trained tile selection, feature weighting, or covariance estimation;
- neural denoising;
- reconstruction of a cleaned movie;
- source naming;
- a large feature-bank or tile-size hyperparameter sweep;
- a new ICA objective.

A temporal AR-prewhitening family may be implemented after the V1 cross-context numerical reference is complete, but it must remain a separate family and cannot be silently combined with feature-bank whitening.

## 8. Proposed repository changes

Create or modify only the following maintained paths unless repository navigation indicates a better existing owner.

```text
neurobench/algorithms/local_covariance_whitening.py
neurobench/experiments/msln_msica/local_whitening_program.py
neurobench/reports/local_whitening.py
neurobench/metrics/whiteness.py
examples/spon_ca_burst_local_whitening_v1.example.json
schemas/spon_ca_burst_local_whitening_v1.schema.json   # only if public schema practice requires it
docs/workflows/spon_ca_burst_local_whitening.md
tests/test_local_covariance_whitening.py
tests/test_local_whitening_program.py
tests/test_whiteness_metrics.py
```

Prefer extending existing artifact, sparse-detection, and scientific-audit helpers over duplicating them.

Do not create a CUDA module in the first patch. Establish the float64 CPU fitting reference and float32 application contract first.

## 9. Core data structures

### 9.1 `WhiteningFeatureBank`

```python
@dataclass(frozen=True)
class WhiteningFeatureBank:
    feature_ids: tuple[str, ...]
    values: np.ndarray             # [T, H, W, D], float32
    valid_frames: np.ndarray       # [T], bool
    source_contexts: tuple[dict[str, object], ...]
    diagnostics: dict[str, object]
```

Validation:

- `D == len(feature_ids)`;
- finite float32 values;
- all source arrays align in TYX and valid-frame support;
- no display-clipped or squared array may enter the feature bank;
- feature order is deterministic and recorded.

### 9.2 `CovarianceEstimatorConfig`

```python
@dataclass(frozen=True)
class CovarianceEstimatorConfig:
    method: Literal["oas", "ledoit_wolf", "fixed_ridge", "diagonal"]
    ridge_ratio: float
    eigenvalue_floor_ratio: float
    maximum_condition_number: float
    center: Literal["mean", "median"]
```

Primary defaults:

```text
method                       = oas
eigenvalue_floor_ratio       = 1e-5
maximum_condition_number     = 1e4
center                       = mean
```

`median` is a gated robust-center ablation. Do not call median centering a robust covariance estimator.

### 9.3 `SpatialTileConfig`

```python
@dataclass(frozen=True)
class SpatialTileConfig:
    tile_height: int
    tile_width: int
    stride_y: int
    stride_x: int
    blend: Literal["hann", "triangular", "uniform"]
    boundary_mode: Literal["crop", "reflect"]
    minimum_raw_samples: int
    maximum_fit_samples: int
    spatial_subsample: int
    temporal_subsample: int
```

Primary defaults, subject to source dimensions:

```text
tile_height / tile_width = 64
stride_y / stride_x      = 32
blend                     = hann
boundary_mode             = crop
maximum_fit_samples       = 65536
spatial_subsample         = 2
temporal_subsample        = 1
```

The preflight must adjust or reject the tile geometry if it does not cover the field exactly under the declared boundary mode. Do not silently change it during `run`.

### 9.4 `LocalCovarianceFit`

```python
@dataclass(frozen=True)
class LocalCovarianceFit:
    fit_id: str
    feature_ids: tuple[str, ...]
    tile_bounds_yx: tuple[int, int, int, int] | None
    mean: np.ndarray               # [D], float64
    sample_covariance: np.ndarray  # [D, D], float64
    covariance: np.ndarray         # regularized [D, D], float64
    whitening: np.ndarray          # ZCA [D, D], float64
    eigenvalues_raw: np.ndarray    # [D], float64
    eigenvalues_regularized: np.ndarray
    shrinkage: float
    condition_number: float
    effective_rank: float
    sample_count: int
    block_count: int
    resolved: bool
    unresolved_reason: str | None
    diagnostics: dict[str, object]
```

Never return an arbitrary identity transform under a successful status. An unresolved tile must be explicit and must use the declared fallback policy.

### 9.5 `LocalWhiteningResult`

```python
@dataclass(frozen=True)
class LocalWhiteningResult:
    zca_features: np.ndarray       # [T, H, W, D], float32
    mahalanobis_energy: np.ndarray # [T, H, W], float32
    quiet_surprise: np.ndarray     # [T, H, W], float32
    blend_weight: np.ndarray       # [H, W], float32
    unresolved_tile_mask: np.ndarray
    valid_frames: np.ndarray
    diagnostics: dict[str, object]
```

## 10. Numerical algorithms

### 10.1 Feature-bank construction

Add a small helper that loads or computes declared MSLN contexts using the maintained `causal_joint_msln` implementation.

Requirements:

- use signed scientific MSLN arrays only;
- preserve full float32 values;
- intersect valid-frame masks;
- record each context's scale floor and causal/reference support;
- cache arrays only under the new collision-safe output root;
- never read a visualization-normalized array as scientific input.

### 10.2 Covariance sample gathering

Gather samples as rows `[N, D]` from `quiet_fit` frames and a declared spatial region.

Requirements:

- deterministic lexicographic or seeded subsampling;
- no random pixel-frame split presented as independent validation;
- fit/selection/holdout partitions are contiguous temporal blocks;
- record raw sample count and blocked effective sample diagnostics;
- reject nonfinite values;
- preserve feature order;
- use float64 for fitting.

### 10.3 Covariance estimation

Implement:

```python
fit_covariance(
    samples: np.ndarray,
    config: CovarianceEstimatorConfig,
) -> LocalCovarianceFit
```

Primary estimator: `sklearn.covariance.OAS` or an exact repository-owned numerical equivalent.

Controls:

- diagonal covariance from per-feature variances;
- fixed ridge shrinkage toward `trace(S) / D * I`;
- Ledoit-Wolf if dependency/version behavior is stable.

After estimation:

1. symmetrize exactly;
2. eigendecompose with `np.linalg.eigh`;
3. floor eigenvalues by `max(abs_max * floor_ratio, float64_eps)`;
4. compute condition number and effective rank;
5. mark unresolved if condition exceeds the cap before/after declared regularization rules;
6. construct ZCA whitening `U @ diag(lambda^-1/2) @ U.T`;
7. verify `W @ Sigma @ W.T` against identity.

Do not use a matrix square root implementation whose orientation or numerical convention is implicit.

### 10.4 Tile fitting and application

Implement separate fit and apply phases:

```python
fits = fit_tiled_feature_whiteners(feature_bank, quiet_fit_mask, tile_config, estimator_config)
result = apply_tiled_feature_whiteners(feature_bank, fits, tile_config, quiet_calibration)
```

Rules:

- fit every tile before applying to event/review frames;
- no adaptation during primary inference;
- apply ZCA in bounded frame chunks;
- blend transformed feature channels using deterministic nonnegative tile weights;
- normalize by accumulated blend weight;
- derive energy from the final blended feature vector, not by blending independently normalized display energies;
- ensure every valid pixel has positive blend weight;
- preserve an unresolved mask and fallback provenance.

Primary unresolved fallback hierarchy:

```text
local full unresolved
    -> matching global full fit
    -> local diagonal if full fit is the only unresolved part
    -> identity only when explicitly recorded and no valid covariance fit exists
```

The fallback is part of the resolved manifest and report.

### 10.5 Quiet-tail calibration

Fit the empirical survival function of `mahalanobis_energy` on a calibration quiet block that is disjoint from covariance fitting.

```text
quiet_surprise = -log10(max(empirical_survival(q), probability_floor))
```

Requirements:

- monotone interpolation;
- deterministic tie handling;
- probability floor recorded;
- calibration block distinct from fit and held-out assessment;
- optional chi-square QQ/KS diagnostic only; no forced parametric threshold.

### 10.6 Whiteness metrics

Create `neurobench/metrics/whiteness.py` with at least:

```python
covariance_identity_error(covariance) -> float
normalized_off_diagonal_energy(covariance) -> float
maximum_absolute_correlation(covariance) -> float
effective_rank(eigenvalues) -> float
fit_holdout_covariance_error(fit_covariance, holdout_covariance) -> float
temporal_autocorrelation_summary(values, max_lag) -> dict[str, object]
tile_boundary_discontinuity(values, tile_geometry) -> dict[str, float]
```

Definitions must be in docstrings and report metadata. Unit tests must use analytically known matrices.

### 10.7 Optional temporal-prewhitening family

Only after the cross-context CPU reference is complete, add a separate family:

```text
Z[r,t] = sum_{k=1}^p a[tile,k] * Z[r,t-k] + innovation[r,t]
```

Primary orders: `p in [1, 2, 4, 6]`.

Requirements:

- fit only on quiet support;
- compare global versus local coefficients;
- shrink local coefficients toward a global fit;
- reject/project unstable AR polynomials;
- record invalid causal prefix;
- preserve both carrier and innovation;
- measure event width and integrated amplitude;
- do not combine with cross-context whitening in the primary V1 result.

## 11. Experiment manifest

Create `examples/spon_ca_burst_local_whitening_v1.example.json` with an exact top-level contract similar to:

```json
{
  "schema_version": 1,
  "experiment_id": "spon_ca_burst_local_whitening_v1",
  "source": {
    "movie_path": "../Outputs/GammaCFAR/spon_ca_burst_3_hindbrain_to_tail_488_20ms/spon_ca_burst_3_hindbrain_to_tail_488_20ms.npy",
    "labels_path": "../Inputs/Spon Ca Burst/labels/labels_normalized.tsv",
    "axes": "TYX",
    "ui_one_based": true,
    "review_interval_ui": [1800, 2359],
    "quiet_interval_ui": [1800, 1899],
    "burst_intervals_ui": {
      "1": [2003, 2026],
      "2": [2040, 2063],
      "3": [2122, 2149],
      "4": [2254, 2300]
    }
  },
  "feature_bank": {
    "context_ids": [
      "joint_s5_g1_t15_g1",
      "joint_s15_g3_t23_g1",
      "joint_s15_g3_t31_g1"
    ],
    "scientific_array": "signed_msln",
    "feature_order_frozen": true
  },
  "quiet_partition": {
    "fit_fraction": 0.50,
    "calibration_fraction": 0.25,
    "holdout_fraction": 0.25,
    "mode": "contiguous_blocks"
  },
  "covariance": {
    "primary_estimator": "oas",
    "control_estimators": ["fixed_ridge"],
    "ridge_ratios": [0.05],
    "eigenvalue_floor_ratio": 1e-5,
    "maximum_condition_number": 10000.0,
    "modes": [
      "identity",
      "global_diagonal",
      "global_full_zca",
      "local_diagonal",
      "local_full_zca"
    ]
  },
  "tiles": {
    "primary_size_px": 64,
    "sensitivity_sizes_px": [32, 128],
    "overlap_fraction": 0.5,
    "blend": "hann",
    "boundary_mode": "crop",
    "maximum_fit_samples": 65536,
    "spatial_subsample": 2,
    "temporal_subsample": 1
  },
  "screen": {
    "freeze_without_spatial_labels": true,
    "maximum_primary_lanes": 1,
    "maximum_diagnostic_lanes": 2,
    "selection_terms": [
      "heldout_whiteness",
      "quiet_tail_stability",
      "block_candidate_recurrence",
      "carrier_morphology_consistency",
      "tile_boundary_score"
    ]
  },
  "evaluation": {
    "candidate_budgets": [20, 40, 58, 80, 100],
    "guardrail_budget": 58,
    "nms_distance_px": 6,
    "match_radius_px": 6,
    "unlabeled_candidates": "unknown",
    "winner_basis": "label_free_freeze_then_sparse_positive_guardrail"
  },
  "scientific_audit": {
    "enabled": true
  },
  "compute": {
    "device": "cpu",
    "cpu_threads": 4,
    "workers": 1,
    "frame_chunk": 8,
    "maximum_peak_ram_gb": 16
  },
  "outputs": {
    "root_dir": "../Outputs/LocalWhitening/spon_ca_burst_local_whitening_v1",
    "representative_frames_ui": [1900, 2003, 2040, 2122, 2254],
    "fps": 10.0
  }
}
```

Validation must reject unknown keys and silently altered defaults.

The 50/25/25 quiet split above is a default proposal. Preflight must verify that every partition contains enough valid frames and samples. It must not use spatial labels to alter the partition.

## 12. Program stages and CLI

Implement one module with explicit stages:

```bash
.venv-neurobench/bin/python -m neurobench.experiments.msln_msica.local_whitening_program \
  preflight --config examples/spon_ca_burst_local_whitening_v1.example.json

.venv-neurobench/bin/python -m neurobench.experiments.msln_msica.local_whitening_program \
  synthetic --config examples/spon_ca_burst_local_whitening_v1.example.json

.venv-neurobench/bin/python -m neurobench.experiments.msln_msica.local_whitening_program \
  covariance-audit --config examples/spon_ca_burst_local_whitening_v1.example.json

.venv-neurobench/bin/python -m neurobench.experiments.msln_msica.local_whitening_program \
  run --config examples/spon_ca_burst_local_whitening_v1.example.json \
  --authorize-full-spon

.venv-neurobench/bin/python -m neurobench.experiments.msln_msica.local_whitening_program \
  summarize --output-root Outputs/LocalWhitening/spon_ca_burst_local_whitening_v1
```

Behavior:

- `preflight` is read-only with respect to source data and refuses an existing output root;
- `synthetic` requires no real-data authorization;
- `covariance-audit` may inspect the declared quiet source only if explicitly selected by the user; otherwise keep a generated/tiny mode;
- `run` requires an exact matching preflight and `--authorize-full-spon`;
- `summarize` performs no refitting;
- `--resume` may resume only an exact manifest/input fingerprint;
- no stage overwrites a completed artifact.

Do not invent a `gpu-preflight` command until a separate CUDA implementation exists.

## 13. Stage gates

### Gate G0: residual covariance is worth modeling

Advance from covariance audit only when all are true:

1. post-MSLN off-diagonal covariance is reproducible across at least two held-out quiet blocks;
2. one of the following holds beyond block-bootstrap uncertainty:
   - full covariance improves held-out whiteness over diagonal covariance;
   - local covariance improves held-out whiteness over global covariance;
3. covariance estimates are resolved under the declared conditioning cap.

Suggested diagnostic defaults:

```text
median max_abs_correlation > 0.15
OR
median normalized_off_diagonal_energy > 0.10
```

These are preregistered defaults, not theoretical universal thresholds.

A failed G0 ends the program with a report stating that current MSLN already captures the useful tested second-order structure.

### Gate G1: synthetic event preservation

Primary local full-ZCA lane must:

- reduce held-out covariance identity error relative to local diagonal;
- preserve event morphology correlation >= 0.90;
- preserve amplitude-ranking Spearman correlation >= 0.95;
- produce no nonfinite values;
- remain within the condition-number cap;
- avoid a material increase in empirically calibrated null extremes;
- explicitly fail or fall back under declared contamination/covariance-shift limits.

### Gate G2: real quiet-block generalization

Primary local full-ZCA lane must:

- reduce held-out whiteness error by at least 25% relative to the better of local diagonal and global full in a majority of quiet blocks;
- produce stable ZCA matrices across block bootstraps;
- pass tile-boundary and blend-coverage checks;
- have a bounded unresolved-tile fraction;
- fit within the declared CPU RAM/runtime envelope.

If only local diagonal passes, conclude that additional local standardization is useful but full whitening is unsupported.

### Gate G3: label-free lane freeze

Freeze exactly one primary lane and at most two diagnostics using only:

- held-out quiet whiteness;
- empirical tail stability;
- block candidate recurrence;
- morphology consistency with the signed carrier;
- tile/estimator sensitivity;
- compute/resource constraints.

Write the freeze decision and all rejected alternatives before loading spatial labels for evaluation.

### Gate G4: protected sparse-positive utility

Treat the current recording as exploratory development data.

A lane may advance to ICA comparison only if it is Pareto-useful across:

- fixed-budget known-positive recall;
- nearest-candidate distance/rank;
- candidate burden or null calibration;
- morphology and temporal preservation;
- fit/block/tile stability.

A protected-best alternative selected after labels is a ceiling and cannot replace the frozen primary result.

### Gate G5: ICA increment

Compare the existing MSLN -> MSICA control with MSLN -> local whitening -> MSICA using identical pair samples and ICA settings.

Advance only if local whitening gives at least one of:

- >=25% lower ICA-angle circular standard deviation;
- materially lower component-swap/ambiguity rate;
- materially higher cross-bootstrap map correlation;

with no degradation in representation-level utility.

A different ICA angle or lower objective alone is not success.

## 14. Synthetic fixture matrix

Build deterministic fixtures with known covariance and event truth.

### 14.1 Background/noise families

```text
iid_gaussian
poisson_gaussian
spatial_gaussian_kernel
ar1_temporal
ar4_temporal
separable_spatiotemporal
nonseparable_moving_edge
heteroscedastic_field
broad_drift
motion_like_translation
saturation_artifact
covariance_shift_train_to_test
```

### 14.2 Signal families

```text
compact_transient       # 3-9 px, 1-8 frames
broad_sustained
amplitude_ladder
multiple_overlapping
boundary_crossing
slow_rise_fast_decay
fast_rise_slow_decay
```

### 14.3 Stress axes

```text
signal_amplitude
event_diameter
event_duration
quiet_fit_contamination_fraction: [0, 0.01, 0.05, 0.10, 0.20]
correlation_strength
covariance_shift
sample_count
tile_size
```

Do not run the full Cartesian product by default. Use a deterministic compact covering design and report the exact combinations.

## 15. Evaluation outputs

Every completed stage writes small machine-readable indices first.

```text
<root>/
  resolved_config.json
  input_fingerprints.json
  status.json
  progress.json
  summary.json
  llm_context.json
  artifact_index.json
  validation.json
  covariance_audit/
    global_covariances.json
    tile_covariance_index.csv
    eigenvalue_summary.csv
    heldout_whiteness.csv
    bootstrap_stability.csv
    covariance_atlas_preview.png
  synthetic/
    fixture_manifest.json
    metrics.csv
    failure_cases.csv
    representative_figures/
  fits/
    global/*.npz
    local/tile_<id>.npz
    fit_index.csv
  representations/
    <lane>/zca_features.npy
    <lane>/mahalanobis_energy.npy
    <lane>/quiet_surprise.npy
    <lane>/unresolved_tile_mask.npy
    <lane>/diagnostics.json
  screening/
    lane_metrics.csv
    freeze_decision.json
  evaluation/
    candidate_budget_curves.csv
    sparse_positive_metrics.json
  scientific_audit/
    ... standard three-section contract ...
  REPORT.md
```

Large arrays remain chunked/memory-mapped and are written atomically. Reports reference arrays by stable IDs and checksums.

## 16. Required plots

Create, at minimum:

1. pre/post covariance and correlation matrices for global and representative local tiles;
2. covariance eigenvalue spectra and shrinkage coefficients;
3. map of condition number, effective rank, and unresolved tiles;
4. held-out whiteness by tile and quiet block;
5. global-vs-local and diagonal-vs-full factorial comparison;
6. empirical null-tail QQ/survival plots;
7. synthetic event morphology and trace comparisons;
8. tile-boundary seam diagnostics;
9. candidate-budget curves for the frozen lane and controls;
10. if ICA runs, angle/bootstrap and cross-map correlation plots.

Use fixed scales across compared lanes. Do not infer scientific utility from a per-frame normalized video.

## 17. Scientific-audit integration

A full real-data run must implement `docs/workflows/SCIENTIFIC_AUDIT_OUTPUT_STANDARD.md`.

Model videos must show the exact stage sequence:

```text
Raw
-> selected signed MSLN feature channels
-> ZCA-whitened feature channels or a declared compact summary
-> Mahalanobis energy
-> empirical quiet surprise
-> temporally pooled detection map
```

Display rules:

- raw and signed scientific arrays use fixed declared movie-wide scales;
- signed channels show zero explicitly;
- energy/surprise use zero-based scales;
- no display-normalized array enters detection;
- expert markers and model markers remain section-pure;
- unmatched model candidates remain unknown.

## 18. Tests

### 18.1 Numerical unit tests

Test:

- exact centering and covariance on small matrices;
- diagonal whitening;
- ZCA whitening of a known SPD covariance;
- invariance of Mahalanobis energy to invertible linear reparameterization within tolerance;
- OAS/fixed-ridge conditioning;
- eigenvalue-floor behavior;
- unresolved-fit behavior;
- deterministic sample gathering;
- overlap weights sum to positive values everywhere;
- tile blending reproduces a global transform when all tile fits are identical;
- empirical survival calibration monotonicity and tie handling;
- float32 apply vs float64 reference tolerance.

Suggested numerical tolerances:

```text
known SPD identity error      <= 1e-8 in float64
float32 application max error <= 2e-5 on bounded fixtures
blend coverage minimum         > 0
```

### 18.2 Scientific fixture tests

Test that:

- identity/diagonal/full controls remain distinct;
- full whitening beats diagonal only when off-diagonal covariance exists;
- local whitening beats global only under spatial covariance nonstationarity;
- an event excluded from calibration remains detectable;
- calibration contamination causes measurable self-whitening;
- shrinkage limits low-eigenvalue amplification;
- covariance shift is reported rather than hidden;
- patch-boundary events do not produce seams above tolerance.

### 18.3 Workflow tests

Test:

- exact manifest validation;
- source/UI coordinate projection overlay;
- output-root collision refusal;
- preflight/run fingerprint matching;
- atomic failure status;
- deterministic resume;
- no full-Spon run without explicit authorization;
- scientific-audit default-on behavior;
- report/index agreement.

Run focused tests first, then the repository suite appropriate to the changed surface.

## 19. Resource and safety rules

- CPU reference first: float64 fitting, float32 arrays/application.
- One worker and one tile/lane at a time by default.
- Maximum four CPU threads in the example manifest.
- Bound sample gathering; never materialize all pixel-frame samples when a deterministic cap is sufficient.
- Process frames in chunks and write arrays atomically.
- Refuse an existing output root.
- Preserve `Inputs/`, completed `Outputs/`, and user changes.
- Verify free RAM/disk and active processes before any long run.
- A future CUDA implementation requires CPU/GPU parity, observed VRAM peak below the manifest cap, and a separate explicit user-authorized run.

## 20. Implementation sequence for Codex

Execute these tasks in order. Stop at a failed gate rather than widening scope.

### Task 1: repository orientation and design note

- Read required files.
- Confirm owners/helpers for artifacts, reports, sparse detection, and scientific audit.
- Write a short implementation plan in the new workflow doc.
- Do not change scientific code yet.

### Task 2: whiteness metrics

- Implement `neurobench/metrics/whiteness.py`.
- Add analytical unit tests.
- Confirm no dependency on Spon data.

### Task 3: covariance fit reference

- Implement dataclasses and global covariance fit/apply.
- Add diagonal, fixed-ridge, and OAS.
- Add exact ZCA numerical tests and unresolved behavior.

### Task 4: tiled fit/apply

- Implement deterministic tile enumeration, overlap weights, fit index, application, and blending.
- Add seam, coverage, and global-equivalence tests.

### Task 5: synthetic fixture program

- Implement compact deterministic fixtures and the G0/G1 metrics.
- Generate a tiny report and representative figures.
- Do not access Spon source data.

### Task 6: manifest and preflight

- Add exact schema validation, source fingerprints, quiet partitions, resource estimates, output collision checks, and label projection overlay.
- Preflight may create only the new output root and small metadata/previews.

### Task 7: covariance-audit stage

- Wire the three fixed MSLN contexts.
- Compute global/local covariance diagnostics on declared quiet blocks.
- Implement G0 and G2 reports.
- No spatial-label use.

### Task 8: representation and label-free screen

- Apply frozen fits to the review interval.
- Generate energy/surprise arrays and lane diagnostics.
- Freeze one primary and at most two diagnostic lanes before spatial labels.

### Task 9: protected evaluation and audit hooks

- Reuse sparse-detection metrics and fixed candidate budgets.
- Generate the full scientific-audit package.
- Keep protected ceilings separate from the frozen result.

### Task 10: optional ICA increment

- Only implement/run if the representation gate passes and the user explicitly selects the full real-data continuation.
- Reuse exact existing two-frame ICA settings and pair indices.
- Compare stability and utility, not only objective values.

### Task 11: documentation and final validation

- Complete workflow documentation, API docstrings, command examples, and interpretation rules.
- Run focused tests and the appropriate repository suite.
- Validate JSON/CSV/report consistency and all generated media.

## 21. Definition of done

The V1 implementation is complete only when:

- the CPU numerical reference is deterministic and tested;
- global/local and diagonal/full controls are implemented distinctly;
- synthetic fixtures demonstrate expected success and failure behavior;
- preflight is collision-safe and source-fingerprinted;
- every fit records conditioning, shrinkage, support, and fallback provenance;
- held-out quiet whiteness is evaluated on contiguous blocks;
- a lane cannot be frozen using spatial labels;
- sparse-positive evaluation preserves unknown-candidate semantics;
- the scientific-audit contract is integrated for full runs;
- no completed output root is modified;
- no full Spon/GPU run occurs without explicit user authorization;
- the report can conclude either `advance` or `stop` without requiring a wider sweep.

## 22. Expected scientific outcomes

The program supports four legitimate conclusions:

1. **Stop: no residual covariance.** Current MSLN already removes the useful tested second-order structure.
2. **Local diagonal only.** Additional spatially varying scale correction helps, but full whitening is unnecessary.
3. **Global full only.** Off-diagonal covariance matters, but one global covariance model is sufficient.
4. **Local full ZCA passes.** Spatially varying off-diagonal covariance is reproducible and useful; proceed to protected evaluation and, only then, ICA-stability testing.

Any of these outcomes is scientifically informative. Do not force the experiment toward outcome 4.
