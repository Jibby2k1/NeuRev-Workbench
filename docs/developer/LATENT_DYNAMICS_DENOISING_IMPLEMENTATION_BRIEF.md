# Stable latent-dynamics denoising before temporal feature extraction

Implementation brief: 2026-07-27.

Status: documentation-only implementation specification. This brief authorizes
repository code, unit tests, synthetic fixtures, read-only preflight, and bounded
CPU smoke tests. It does **not** authorize a full Spon Ca Burst run, a GPU run,
replacement of an existing scientific lane, modification of completed
`Outputs/`, or automatic promotion of a denoised result into the workbench.

## Executive directive

Implement a stage-gated NeuRev experiment with the following scientific order:

```text
raw movie
    -> explicit observation/baseline model
    -> stable latent fluorescence estimate
    -> uncertainty and residual diagnostics
    -> ordinary and model-aware temporal differences
    -> existing candidate extraction/evaluation
```

The primary target is the denoised latent fluorescence trajectory, not a binary
event mask and not a maximally independent output. Differencing, ICA-like
directions, CFAR, and event models are downstream feature extractors. They may be
useful only after the latent trajectory is shown to preserve real dynamics while
reducing measurement noise.

The first implementation must keep five objects distinct:

1. `observation`: the measured intensity after declared calibration;
2. `latent_state`: the estimated denoised fluorescence trajectory;
3. `state_difference`: a lagged difference of the latent trajectory;
4. `dynamic_drive`: the part of the latent state not predicted by its own stable
   dynamics;
5. `observation_residual`: measured intensity minus the reconstructed
   observation.

Do not collapse these into one artifact named `cleaned`, `activity`, or
`innovation`.

## Why this is the next experiment

The repository already establishes three important facts.

1. Adjacent-frame InfoMax and CS-Parzen fits recovered directions nearly
   collinear with `[-1, 1]`; for this two-observation problem, pairwise ICA
   behaved as a temporal derivative. See
   `docs/research/PAIRWISE_ICA_AS_TEMPORAL_DERIVATIVE.md`.
2. Fixed/adaptive differences and pairwise ICA did not improve Raw Direct under
   the tested additive and soft-gating fusions. They remain useful timing/change
   evidence, not replacement images. See
   `docs/workflows/spon_ca_burst_pairwise_feature_fusion.md`.
3. The historical `kalman_positive_residual_stack` is an asymmetric adaptive
   baseline/EMA. A later lane applied positive clipping, Gaussian smoothing, and
   quiet-MAD whitening before a local contrast detector and failed. That result
   rejects that composite preprocessing lane; it is not a test of a fitted
   latent state-space model.

The missing experiment is therefore not “another ICA.” It is:

> Can an explicit, stable temporal observation model estimate a latent
> fluorescence trajectory whose dynamics are more repeatable than the raw
> movie, without erasing slow calcium activity or hallucinating events?

If so, ordinary differences and model-aware residuals can be evaluated as
features of that denoised trajectory.

## Required reading before editing

Read these files in order:

1. `AGENTS.md`;
2. `docs/workflows/spon_ca_burst_frame_derivatives.md`;
3. `docs/research/PAIRWISE_ICA_AS_TEMPORAL_DERIVATIVE.md`;
4. `docs/workflows/spon_ca_burst_pairwise_feature_fusion.md`;
5. `docs/workflows/spon_ca_burst_learnable_contrast.md`;
6. `neurobench/algorithms/background.py`;
7. `neurobench/algorithms/pairwise_separation.py`;
8. `neurobench/experiments/learnable_contrast/diagnostic.py`;
9. `neurobench/metrics/sparse_detection.py`;
10. `neurobench/cli/experiment.py`;
11. the focused tests for the files above.

Use `.venv-neurobench/bin/python` for repository commands.

## Current repository truth that must remain visible

- Raw Direct is the structural/amplitude anchor and reproduces mean held-out
  known-label recall `0.605615942`.
- The causal artifact-only proposal retained `58/79` known labels but increased
  candidate burden; it is not established as a precision improvement.
- Pairwise ICA and CS-Parzen empirically rediscovered the derivative direction.
- Multiplicative derivative-energy gates removed slowly evolving calcium
  activity.
- The prior “Kalman spatiotemporal” learned-contrast lane scored zero after a
  compound preprocessing path. Do not cite this as evidence that Kalman
  filtering or smoothing is intrinsically invalid.
- Sparse labels do not provide exhaustive negatives. Unmatched event candidates
  remain `unknown`.
- No current method establishes a uniquely correct denoised movie.

Every new report must reproduce these facts before presenting new results.

## Scientific contract

For pixel or sample index `p`, use the declared observation decomposition

\[
X_t(p) = G_t(p) + S_t(p) + N_t(p) + A_t(p),
\]

where:

- `G_t` is baseline, illumination, persistent anatomy, and slow drift;
- `S_t` is the latent fluorescence signal of interest;
- `N_t` is stochastic measurement noise;
- `A_t` is unmodeled artifact, including motion or saturation.

The first version may estimate `G_t` with a frozen quiet baseline plus an
optional robust global gain/offset correction. It must **not** claim that a
temporal state model removes geometric motion. Record `motion_correction: false`
when registration is not used.

After the declared baseline/gain correction, model the signed residual

\[
r_t(p) = s_t(p) + \epsilon_t(p).
\]

Do not positive-clip `r_t` before state estimation. Positive amplitude is a
downstream view:

\[
a_t(p) = [\widehat{s}_t(p)]_+.
\]

### Stable AR(1) reference model

The mandatory first model is

\[
s_t(p) = \gamma s_{t-1}(p) + u_t(p), \qquad
r_t(p) = s_t(p) + \epsilon_t(p),
\]

with

\[
0 \leq \gamma \leq 1-\varepsilon,\qquad
u_t \sim \mathcal N(0,q),\qquad
\epsilon_t \sim \mathcal N(0,r).
\]

Here `u_t`, not `nu_t`, is the state drive; implementations should use the
symbol/name consistently even if a document renderer cannot distinguish them.
This is deliberately simple. It provides:

- a causal Kalman-filter estimate;
- an offline Rauch--Tung--Striebel smoother estimate;
- posterior uncertainty;
- a prediction error;
- and a model-aware temporal residual.

It is a reference model, not a claim that calcium dynamics are exactly Gaussian
or first-order.

### The central relation to differencing

From the AR(1) model,

\[
u_t = s_t - \gamma s_{t-1}.
\]

Ordinary differencing is the special case

\[
\Delta s_t = s_t-s_{t-1},
\]

which corresponds to setting `gamma = 1`. The model-aware drive uses
`[-gamma, 1]`, while the pairwise ICA experiment recovered approximately
`[-1, 1]`.

Keep the following coefficients separate:

- `alpha_gain`: compensates frame-to-frame illumination gain in an observation;
- `gamma_decay`: predicts persistence of the latent fluorescence state.

They are not interchangeable even though both appear in two-frame subtraction.

### Required named outputs

Use precise names:

- `latent_filter_mean`: causal posterior state mean;
- `latent_smoother_mean`: offline posterior state mean;
- `latent_posterior_variance`: compact time/tile covariance artifact;
- `filter_innovation`: \(r_t-\widehat{s}_{t|t-1}\);
- `filter_innovation_z`: innovation normalized by its predictive standard
  deviation;
- `smoother_residual`: \(r_t-\widehat{s}_{t|T}\);
- `state_difference_lag_k`: \(\widehat{s}_t-\widehat{s}_{t-k}\);
- `dynamic_drive`: \(\widehat{s}_t-\gamma\widehat{s}_{t-1}\);
- `positive_dynamic_drive`: positive part of `dynamic_drive`.

In Kalman terminology, “innovation” normally means the measurement prediction
error. Use `dynamic_drive`, not `innovation`, for
\(\widehat{s}_t-\gamma\widehat{s}_{t-1}\).

## Hypotheses

Test the following in order.

### H1: denoising

A stable latent estimate reduces quiet-period measurement noise and synthetic
reconstruction error while preserving injected and labeled event amplitude,
duration, and timing.

### H2: dynamics

Differences computed from the latent estimate have a higher event-to-quiet
separation and better fold/perturbation stability than the same differences
computed from raw or legacy-EMA inputs.

### H3: model-aware differencing

`dynamic_drive` improves onset evidence relative to ordinary lag-1 differencing
when the latent state has nontrivial persistence.

### H4: downstream utility

At the same quiet-calibrated candidate burden, one or more latent-derived
features improve held-out known-label recall or reduce candidates without recall
loss. Failure of H4 does not invalidate a denoiser that passes H1--H3; it means
the tested detector/fusion did not exploit it.

## Method ladder

Implement and report explicit lanes. Never hide them behind a generic
`denoised` option.

### Anchors

1. `raw_direct`: frozen current baseline.
2. `raw_difference_lag1`.
3. `raw_difference_lag4`.
4. `legacy_asymmetric_ema`: the existing
   `kalman_positive_residual_stack`, preserved under its historical API but
   described accurately in new reports.
5. `legacy_smoothed_difference`: offline reference only, when the completed
   artifact is available.

### Mandatory state-space lanes

6. `stable_ar1_filter`: causal state estimate.
7. `stable_ar1_smoother`: offline state estimate using the same fitted model.
8. `stable_ar1_filter_drive`.
9. `stable_ar1_smoother_drive`.
10. lag-1 and lag-4 differences of both filter and smoother states.

### Gated extensions

11. `stable_ar2_smoother`: only after AR(1) numerical and scientific gates pass.
12. `robust_ar1_smoother`: Huber/Student-t observation reweighting, only after a
    Gaussian reference exists.
13. `local_dynamic_factor_smoother`: patchwise spatial factor model, only after
    scalar/tilewise outputs and evaluation are trustworthy.
14. external PMD or self-supervised denoising imports as comparison evidence.
15. OASIS-style sparse nonnegative dynamics only on accepted/candidate traces or
    after a spatial demixing stage; do not silently run trace deconvolution on
    every raw pixel and call the result a denoised movie.

## Parameter fitting and leakage rules

The denoiser is unsupervised.

- Known event coordinates and burst identities may be used for final evaluation,
  outer-fold organization, and visualization only.
- They must not choose `gamma`, `q`, `r`, noise floors, tile rank, robust
  thresholds, stopping epoch, or denoiser lane.
- Estimate a per-pixel quiet center and robust scale only from the declared quiet
  interval.
- A conservative initial observation-noise estimate is

  \[
  \widehat r(p) \approx \frac{1}{2}
  \operatorname{Var}_{\rm robust}(r_t(p)-r_{t-1}(p))
  \]

  on quiet frames. Record that slow quiet dynamics can make this an overestimate.
- Normalize by the quiet scale before fitting shared dynamics; restore physical
  units in output artifacts.
- Fit shared AR parameters on a deterministic, bounded sample stratified across
  the field and quiet-intensity/noise strata.
- Use deterministic contiguous temporal validation blocks for predictive
  likelihood. Overlapping frames from the same block may not cross
  train/validation boundaries.
- If bounded EM is implemented, initialize it from a declared grid and constrain
  every update. Save the full likelihood and parameter history.
- Never choose the denoiser by maximizing entropy of its output.

### Mandatory AR(1) stability parameterization

Do not rely on post-update clipping as the only stability mechanism. Use a
bounded parameterization such as

\[
\gamma=(1-\varepsilon)\operatorname{sigmoid}(\theta),
\]

or fit a bounded decay time

\[
\gamma = \exp(-\Delta t/\tau),\qquad
\tau_{\min}\leq\tau\leq\tau_{\max}.
\]

Declare a nonzero stability margin `epsilon`.

Constrain

\[
q_{\min}\leq q\leq q_{\max},\qquad
r_{\min}\leq r\leq r_{\max}.
\]

Reject fits that sit on multiple bounds without an explicit diagnostic.

## Numerical stability requirements

1. Use float64 for parameter fitting, likelihood accumulation, and scalar
   covariance recursions. Dense scientific outputs may be float32.
2. Use the Joseph covariance update or an algebraically equivalent
   positive-semidefinite form.
3. Floor every predictive variance before division or logarithms.
4. Check every covariance for finiteness and nonnegativity.
5. The RTS backward gain must use a floored predicted covariance.
6. Record minimum/maximum filter gain, predictive variance, posterior variance,
   log-likelihood increment, and standardized innovation.
7. Reject NaN/Inf, negative variance beyond tolerance, covariance collapse,
   unbounded state magnitude, and likelihood decrease beyond the declared
   numerical tolerance.
8. Test long constant, impulse, ramp, and pure-noise sequences.
9. The smoother is offline and noncausal. Never expose it as a real-time lane.
10. Report filter latency and smoother look-ahead semantics separately.
11. For AR(2), parameterize stable poles or verify the companion matrix with a
    strict Schur margin. Do not optimize unconstrained coefficients.
12. For a patchwise dynamic-factor extension, require a stability certificate
    for every latent transition matrix and overlap-add tiles with a declared
    window.

## Recommended repository additions

Create this structure unless current conventions make an equally clear route
preferable:

```text
neurobench/
├── algorithms/
│   └── latent_dynamics.py
├── experiments/
│   └── latent_dynamics/
│       ├── __init__.py
│       ├── config.py
│       ├── preflight.py
│       ├── noise.py
│       ├── fitting.py
│       ├── filtering.py
│       ├── features.py
│       ├── synthetic.py
│       ├── evaluation.py
│       ├── artifacts.py
│       └── runner.py
└── metrics/
    └── latent_signal.py
```

Add:

```text
examples/spon_ca_burst_latent_dynamics.example.json
tests/test_latent_dynamics_algorithms.py
tests/test_latent_dynamics_config.py
tests/test_latent_dynamics_synthetic.py
tests/test_latent_dynamics_runner.py
tests/test_latent_dynamics_cli.py
docs/workflows/spon_ca_burst_latent_dynamics.md
```

Register a thin, lazy-loaded CLI:

```text
neurobench experiment latent-dynamics preflight
neurobench experiment latent-dynamics synthetic
neurobench experiment latent-dynamics run
neurobench experiment latent-dynamics feature-benchmark
```

`preflight` must require a new explicit artifact directory. `run` must require
the reviewed matching preflight. A full Spon run remains unauthorized until the
user explicitly selects it.

## Public numerical interfaces

Implement typed, documented interfaces equivalent to:

```python
@dataclass(frozen=True)
class QuietNoiseModel:
    center: np.ndarray
    scale: np.ndarray
    scale_floor: float
    difference_variance: np.ndarray
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class StableAR1:
    gamma: float
    process_variance: float
    observation_variance: float
    initial_variance: float
    stability_margin: float
    fit_status: str
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class StateSpaceResult:
    filter_mean: np.ndarray
    filter_variance: np.ndarray
    predicted_mean: np.ndarray
    predicted_variance: np.ndarray
    innovation: np.ndarray
    innovation_variance: np.ndarray
    log_likelihood: float
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class SmootherResult:
    mean: np.ndarray
    variance: np.ndarray
    lag_covariance: np.ndarray | None
    diagnostics: dict[str, Any]
```

Required pure-array functions:

```python
estimate_quiet_noise(...)
stable_ar1_from_decay(...)
validate_stable_ar1(...)
kalman_filter_ar1(...)
rts_smoother_ar1(...)
fit_shared_ar1_grid(...)
fit_shared_ar1_em(...)            # optional until grid reference passes
state_difference(...)
dynamic_drive(...)
standardized_filter_innovation(...)
smoother_observation_residual(...)
```

Every function must validate shapes, axes, dtypes, finite values, and parameter
bounds. `neurobench/algorithms/latent_dynamics.py` must not read files or know
Spon-specific frame numbers.

The implementation must support a vectorized `[T, N]` interface and a bounded
tile/chunk wrapper. It must not allocate a full pixel-by-pixel covariance matrix.

## Example manifest contract

Create a schema-validated configuration with this initial shape:

```json
{
  "schema_version": 1,
  "experiment_id": "spon_ca_burst_latent_dynamics_v1",
  "source_video": "../Outputs/GammaCFAR/spon_ca_burst_3_hindbrain_to_tail_488_20ms/spon_ca_burst_3_hindbrain_to_tail_488_20ms.npy",
  "labels_tsv": "../Inputs/Spon Ca Burst/labels/labels_normalized.tsv",
  "output_dir": "../Outputs/LatentDynamics/spon_ca_burst_latent_dynamics_v1",
  "frames": {
    "review_start_ui": 1800,
    "review_end_ui": 2359,
    "quiet_start_ui": 1800,
    "quiet_end_ui": 1899,
    "frame_period_ms": 20.0
  },
  "preprocessing": {
    "baseline_mode": "quiet_median",
    "signed_residual": true,
    "gain_mode": "none",
    "motion_mode": "none",
    "quiet_scale_floor_percentile": 10.0
  },
  "fit": {
    "sample_pixels": 4096,
    "sample_seed": 20260727,
    "temporal_validation_blocks": 5,
    "stability_epsilon": 0.001,
    "decay_time_ms_grid": [40, 80, 160, 320, 640, 1280],
    "process_to_observation_grid": [0.01, 0.03, 0.1, 0.3, 1.0],
    "parameter_mode": "bounded_grid"
  },
  "application": {
    "tile_height": 64,
    "tile_width": 64,
    "write_filter_mean": true,
    "write_smoother_mean": true,
    "write_dense_residuals": false
  },
  "features": {
    "lags": [1, 4],
    "write_dense_features": false,
    "write_selected_tiffs": true,
    "positive_views": true
  },
  "evaluation": {
    "primary_match_radius_px": 6,
    "match_radii_px": [4, 6, 8],
    "quiet_false_peaks_per_map": 1.0,
    "capacity_reference_lane": "raw_direct",
    "synthetic_seeds": [7, 13, 19, 29, 37]
  },
  "resources": {
    "cpu_threads": 2,
    "max_ram_mib": 4096,
    "min_free_disk_mib": 4096,
    "max_output_mib": 3072
  }
}
```

Unknown fields must fail validation. Resolve all paths relative to the manifest.
The example must remain a small specification; it must not authorize execution.

## Artifact contract

A completed model run must write:

```text
config.resolved.json
preflight.json
run_state.json
progress.jsonl
resource_summary.json
fit/
    sample_manifest.json
    candidate_models.tsv
    selected_model.json
    parameter_history.tsv
    predictive_likelihood.tsv
    stability.json
noise/
    quiet_noise_summary.json
    quiet_center.npy
    quiet_scale.npy
states/
    filter_mean.npy               # only when enabled
    smoother_mean.npy             # only when enabled
    filter_variance_by_time.npy
    smoother_variance_by_time.npy
features/
    feature_manifest.json
    pooled_candidate_maps.npz
    selected_review_tiffs/        # only preregistered views
diagnostics/
    residual_summary.json
    innovation_summary.json
    quiet_autocorrelation.tsv
    event_preservation.tsv
    perturbation_stability.tsv
evaluation/
    metrics.json
    lane_summary.tsv
    known_matches.tsv
    unmatched_candidates.tsv
report.md
```

Avoid dense duplicate arrays. If parameters are shared after quiet
normalization, posterior variance is time- or tile-indexed and need not be
duplicated for every pixel. Derive differences and dynamic drive in chunks from
the stored state unless a specific dense artifact is preregistered.

Large arrays must declare shape, axes, dtype, frame alignment, units,
normalization, model ID, source checksum, and whether the result is causal.
Write `.partial` arrays and atomically rename after validation.

## Synthetic and falsification suite

Synthetic validation is mandatory before real-data application. Include:

1. constant latent signal plus white noise;
2. stable AR(1) signal with known parameters;
3. stable AR(2) signal when that lane exists;
4. positive transient with known onset, rise, decay, and amplitude;
5. slow ramp/plateau that must not be erased;
6. two events separated by a short interval;
7. pure noise with no latent event;
8. impulsive outliers;
9. heteroscedastic noise;
10. illumination gain/offset drift;
11. one-pixel translation edge or synthetic motion artifact;
12. model mismatch between injected and fitted decay.

For image fixtures, inject soma-like rings/disks into real quiet-background
frames with known spatial footprints. Preserve a noise-free ground truth.

Required synthetic metrics:

- latent NMSE and correlation;
- amplitude bias;
- onset and peak-time error;
- decay-time bias;
- quiet false-event count;
- residual autocorrelation;
- standardized-innovation mean, variance, and tail rate;
- uncertainty interval coverage;
- filter/smoother difference;
- parameter error and stability margin;
- derivative/drive event-to-quiet separation.

Mandatory falsification expectations:

- A noise-free identity case remains unchanged within tolerance.
- Pure noise does not produce stable positive activity.
- A slow real ramp is not classified entirely as baseline/noise.
- A sharp event is not shifted outside the declared timing tolerance.
- A smoother may use future evidence but must be labeled noncausal.
- A synthetic motion edge is not described as neural activity.
- Exact observation nulls or missing information are not “recovered”; any
  completion must be labeled prior/model-conditioned.

## Real-data evaluation

### Denoising evaluation comes first

Before running the existing detector, report:

1. quiet-period variance and temporal autocorrelation before/after;
2. residual autocorrelation and spatial structure;
3. standardized filter-innovation calibration;
4. event-window amplitude, rise, duration, and area preservation at known
   coordinates;
5. latent-state consistency across nearby pixels around each known coordinate;
6. parameter and output stability across deterministic samples, temporal blocks,
   and bounded perturbations;
7. disagreement between causal filter and offline smoother;
8. visual panels containing raw, latent, difference, dynamic drive, and residual
   views with one fixed scale.

Noise reduction alone is insufficient. A method fails if it obtains a smooth
movie by attenuating real event amplitude, extending events backward in time,
or moving structure into the residual.

### Feature benchmark comes second

Use the existing Raw Direct temporal pooling, quiet-only threshold calibration,
six-pixel NMS, one-to-one matching, and candidate-cap logic. Compare:

- Raw Direct;
- raw lag-1 and lag-4 differences;
- legacy asymmetric EMA;
- causal latent amplitude;
- offline latent amplitude;
- causal/offline latent lag-1 and lag-4 differences;
- causal/offline positive dynamic drive;
- standardized filter innovation.

Do not initially fuse features. First determine whether each feature is useful
on its own under identical evaluation. Only a feature that passes its
preregistered gate may enter a later bounded fusion initialized exactly at Raw
Direct.

Report known matches and unmatched candidates separately. Do not call the
known-label candidate fraction precision.

## Advancement gates

### C0: implementation integrity

- exact Raw Direct reproduction;
- exact frame/coordinate contracts;
- finite deterministic outputs;
- collision-safe artifacts;
- all unit/synthetic tests pass.

Failure stops all scientific interpretation.

### C1: numerical stability

Across all synthetic seeds and the real preflight sample:

- strict stable-pole margin;
- no NaN/Inf;
- no negative covariance beyond tolerance;
- no covariance collapse;
- bounded gains and state magnitude;
- deterministic rerun agreement.

Failure stops real-data execution.

### C2: denoising validity

Advance a latent lane only if it:

- improves synthetic latent NMSE over raw and legacy EMA;
- reduces quiet noise without materially attenuating injected events;
- preserves onset and duration within preregistered tolerances;
- does not increase quiet false-event rate;
- and passes residual/innovation diagnostics in a majority of seeds and
  perturbations.

### C3: real signal preservation

Advance to feature benchmarking only if known-coordinate amplitude, temporal
area, and onset summaries remain within declared preservation bounds and the
result is stable across temporal blocks/parameter perturbations.

### C4: downstream feature value

Advance a feature to fusion only if, at the same quiet-calibrated operating
point, it either:

- improves mean leave-one-burst-out known-label recall and wins at least three of
  four bursts; or
- reduces candidate burden by at least 20% with no known-label recall loss.

This gate authorizes a later fusion experiment, not replacement of Raw Direct.

### C5: real-time consideration

A causal lane may be considered for streaming only after offline scientific
gates pass and p50/p95/p99 per-frame latency, initialization, drift, and
recalibration behavior are measured against the 20 ms frame deadline. The RTS
smoother is never a real-time candidate.

## Unit and integration tests

At minimum, test:

- AR(1) stability parameterization at extreme raw parameters;
- exact scalar Kalman results against a hand-calculated sequence;
- filter covariance positivity;
- RTS smoothing against a dense Gaussian-conditioning reference on a tiny
  sequence;
- vectorized results against independent scalar loops;
- chunk/tile invariance;
- float32 output versus float64 reference tolerance;
- deterministic grid/EM fitting;
- no label access during fitting;
- difference and dynamic-drive identities;
- `gamma=1` relation between dynamic drive and ordinary difference where
  explicitly permitted in a test fixture;
- undefined leading frames for each lag;
- causal versus noncausal metadata;
- config unknown-field rejection;
- output collision refusal;
- partial-file cleanup behavior;
- exact Raw Direct reproduction;
- sparse-positive `unknown` semantics;
- CLI lazy import and thread-environment behavior.

Do not use only reconstruction loss as a test oracle.

## Resource and write-safety requirements

1. Preserve all user changes and ignored data under `Inputs/` and `Outputs/`.
2. Refuse an existing output root.
3. Set OpenMP/BLAS limits before importing heavy numerical modules.
4. Use deterministic bounded samples and tiles.
5. Never allocate `[pixels, pixels]` covariance.
6. Use memory maps for dense state output.
7. Preflight must estimate each dense artifact independently and enforce the
   configured output cap.
8. Use atomic JSON, `.partial` arrays, progress JSONL, and resource heartbeats.
9. A preflight is read-only except for its explicit new artifact directory.
10. Do not launch the full Spon run or any GPU job without explicit user
    selection.
11. Do not delete or rewrite completed pairwise, frame-difference,
    learnable-contrast, or causal-proposal evidence.
12. Completion is not scientific success; preserve failed lanes and stop reasons.

## Implementation order for Codex

1. Add pure NumPy/SciPy scalar and vectorized AR(1) filtering/smoothing with
   exhaustive unit tests.
2. Add synthetic generators and falsification tests.
3. Add strict config/preflight and resource estimates.
4. Add bounded parameter grid fitting with no labels.
5. Add chunked state application and compact artifact writing.
6. Add latent feature generation in streaming chunks.
7. Add denoising metrics and report generation.
8. Add the Raw Direct-compatible feature benchmark.
9. Run only tiny synthetic and tiny-array smoke tests.
10. Write `docs/workflows/spon_ca_burst_latent_dynamics.md` from implemented
    behavior. Do not copy this planned specification and call it completed.
11. Stop and report the exact command/preflight artifact required for a full
    Spon run. Wait for explicit user selection before executing it.
12. Consider AR(2), robust observations, local dynamic factors, PMD, or
    self-supervised denoising only after the mandatory AR(1) reference is
    scientifically interpretable.

## Definition of done

The implementation wave is complete when:

- the new numerical interfaces and tests pass;
- synthetic ground-truth results expose both successes and failures;
- preflight is collision-safe and resource-bounded;
- a tiny smoke run emits the complete artifact schema;
- the feature benchmark exactly reproduces Raw Direct on its anchor lane;
- the workflow document distinguishes latent state, ordinary difference,
  dynamic drive, filter innovation, and observation residual;
- no full Spon/GPU run has been launched without explicit selection;
- and the final handoff states which gates are passed, failed, or not yet run.

The scientific objective is not to produce the smoothest movie. It is to recover
a stable, uncertainty-aware latent trajectory that preserves biologically useful
dynamics, then determine whether ordinary or model-aware differencing extracts
features more reliably than differencing the noisy observation.
