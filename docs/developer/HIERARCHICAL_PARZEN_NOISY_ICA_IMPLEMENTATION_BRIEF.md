# Hierarchical Parzen ICA and Noisy Parzen ICA for NeuRev

Implementation brief: 2026-07-29.

Status: documentation-only implementation specification. This brief authorizes
repository code, unit tests, deterministic synthetic/semi-synthetic fixtures,
read-only preflight, plot generation, and bounded tiny smoke tests. It does not
authorize a full Spon Ca Burst run, replacement of an existing scientific lane,
modification of completed `Outputs/`, or a closed-loop deployment.

## Executive directive

Implement a stage-gated, amplitude-preserving hierarchical source-separation
experiment:

```text
measured movie
    -> Stage 1: Parzen common/background separation
    -> amplitude-preserving dynamic residual
    -> Stage 2: local noisy Parzen ICA
    -> structured neural signal + structured artifact + measurement noise
```

The implementation must produce an explicit closure:

\[
Y \approx \widehat B + \widehat S + \widehat A + \widehat N,
\]

where:

- `background_like` (`B`) is persistent or broad common structure;
- `structured_neural_signal` (`S`) is accepted localized dynamic structure;
- `structured_artifact` (`A`) is coherent non-neural residual such as motion or
  saturation;
- `measurement_noise` (`N`) is the remaining approximately unstructured residual.

The user-facing three-way description may summarize `A + N` as residual
uncertainty, but the scientific artifacts must keep `A` and `N` separate.

## Required reading before editing

Read in order:

1. `AGENTS.md`;
2. `docs/workflows/spon_ca_burst_pairwise_separation.md`;
3. `docs/research/PAIRWISE_ICA_AS_TEMPORAL_DERIVATIVE.md`;
4. `docs/workflows/spon_ca_burst_pairwise_feature_fusion.md`;
5. `docs/workflows/spon_ca_burst_latent_dynamics.md`;
6. `docs/workflows/spon_ca_burst_representation_benchmark.md`;
7. `docs/research/DENOISE_THEN_DIFFERENCE.md`;
8. `neurobench/algorithms/pairwise_separation.py`;
9. `neurobench/algorithms/latent_dynamics.py`;
10. `neurobench/algorithms/representation_benchmark.py`;
11. `neurobench/metrics/sparse_detection.py`;
12. `tools/build_representation_detection_diagnostics.py`;
13. the focused tests for those modules.

Use `.venv-neurobench/bin/python` for repository commands.

## Current repository truth to preserve

Every report must reproduce these facts before interpreting new results:

- pairwise InfoMax and CS-Parzen recovered directions nearly collinear with the
  temporal derivative;
- derivative/ICA evidence did not improve Raw Direct under the tested additive
  and soft-gating fusions;
- offline AR(1) smoother amplitude improved known-label recall while difference,
  dynamic-drive, and innovation lanes underperformed;
- the selected shared AR(1) decay landed at the upper grid boundary and has not
  passed all event-preservation/perturbation checks;
- amplitude PCA rank 8 has only a provisional two-match fixed-budget advantage;
- rank-64 spatial ICA did not converge reliably and must not be used as evidence
  of stable high-rank separation;
- sparse positive labels do not define exhaustive negatives;
- ordinary false-positive, true-negative, precision, and specificity metrics are
  unavailable without additional review;
- completion of a run is not scientific success.

## Scientific model

For frame `t` and pixel `p`, use the conceptual decomposition

\[
Y_t(p)=B_t(p)+S_t(p)+A_t(p)+N_t(p).
\]

This is not identifiable from one movie without assumptions. The hierarchy makes
those assumptions explicit:

### Stage 1 assumptions

- a substantial background-like contribution is common or slowly changing
  between aligned nearby frames;
- the background subspace is lower-dimensional than the full movie;
- background-like components tend to be temporally persistent and spatially
  broad;
- Stage 1 may remain unresolved when slow signal and background are not
  distinguishable.

### Stage 2 assumptions

- after Stage 1, structured signal occupies a low-dimensional local subspace;
- measurement noise statistics can be estimated from quiet residuals;
- accepted neural components are localized/coherent and stable;
- structured artifacts can be distinguished from measurement noise through
  motion, morphology, saturation, and residual diagnostics;
- noise is additive in the Stage-2 model and is not expected to emerge as a
  uniquely identifiable ordinary ICA component.

## Terminology contract

Use these exact scientific names:

- `observation`;
- `stage1_background`;
- `stage1_dynamic_residual`;
- `stage1_differential_component`;
- `stage1_background_confidence`;
- `stage2_noisy_output`;
- `stage2_posterior_source`;
- `structured_neural_signal`;
- `structured_artifact`;
- `measurement_noise`;
- `closure_residual`.

Do not use one ambiguous artifact named `cleaned`, `activity`, `signal`, or
`noise` without the stage and definition.

## Stage 1: common/background Parzen ICA

### Observation construction

For lag `k`, aligned pixel samples are

\[
\mathbf x_t(p)=
\begin{bmatrix}
I_{t-k}(p)\\
I_t(p)
\end{bmatrix}.
\]

The initial Spon reference uses `k=1`. Optional lag-4 and three-frame embeddings
are gated ablations, not defaults.

Support optional robust gain correction:

\[
I_t(p)\approx \alpha_t I_{t-k}(p)+\text{change}.
\]

Keep `alpha_gain` distinct from any latent decay coefficient used elsewhere.

### Centering and whitening

Implement stable two-dimensional centering/whitening:

\[
\mathbf z=Q(\mathbf x-\boldsymbol\mu),
\]

with:

- float64 fitting;
- robust or ordinary covariance as explicit lanes;
- eigenvalue floors;
- condition number and effective rank;
- `identifiable=false` when the second eigenvalue is below the declared floor;
- no arbitrary output component when the fit is unresolved.

### Stage-1 method lanes

Implement separate method IDs:

1. `fixed_common_difference_reference`;
2. `adaptive_gain_common_difference`;
3. `batch_cs_parzen_pairwise`;
4. `stochastic_parzen_score_pairwise`;
5. `stochastic_minibatch_cs_parzen_pairwise` (optional gated lane).

Do not hide all methods under a single `ica` option.

### Batch CS-Parzen reference

Retain compatibility with the existing bounded angle-search reference. The
Cauchy-Schwarz divergence between joint and product marginals is

\[
D_{\mathrm{CS}}(p,q)
=-\log
\frac{\left(\int p q\right)^2}
{\left(\int p^2\right)\left(\int q^2\right)}.
\]

Use bounded deterministic samples and chunked kernel blocks. Never materialize
an all-pixel pairwise kernel matrix.

### Stochastic Parzen score-function lane

For component `j`, maintain a bounded dictionary `c[j,m]` and bandwidth `h[j]`:

\[
\widehat p_j(y)=
\frac1M\sum_m \mathcal N(y;c_{jm},h_j^2).
\]

Define posterior dictionary weights

\[
\alpha_{jm}(y)=
\frac{\mathcal N(y;c_{jm},h_j^2)}
{\sum_\ell\mathcal N(y;c_{j\ell},h_j^2)},
\]

and negative score

\[
\psi_j(y)=
\frac{y-\sum_m\alpha_{jm}(y)c_{jm}}{h_j^2}.
\]

Use a natural-gradient-style update with declared sign convention:

\[
W^+=
\operatorname{decorrelate}
\left(
W+\eta[I-\boldsymbol\psi(\mathbf y)\mathbf y^\top]W
\right).
\]

Required safeguards:

- symmetric decorrelation with eigenvalue floor;
- gradient norm clipping;
- maximum angle change per accepted update;
- rollback on nonfinite objective/weights;
- separate fast inference and slow adaptation rates;
- deterministic dictionary seeding;
- dictionary size, minimum separation, replacement policy, and age recorded;
- fixed initial bandwidth from calibration; adaptive bandwidth only as a gated
  two-timescale lane;
- compare update direction against numerical finite differences and batch
  objective on tiny fixtures.

### Optional stochastic mini-batch CS-Parzen lane

A more literal stochastic CS-divergence lane may use PyTorch/autograd on bounded
mini-batches and dictionaries. It must:

- share the same demixer parameterization;
- use blockwise kernels;
- report objective variance across batches;
- never be described as the exact full-sample divergence;
- remain a comparison, not replace the score-function reference by default.

### Sign and permutation tracking

Because Stage 1 updates over time, implement deterministic component tracking:

- match current components to previous components by absolute map correlation
  and mixing-vector cosine;
- orient signs using continuity and derivative/common references;
- record every sign flip, swap, unresolved interval, and fallback;
- reject abrupt swaps beyond the configured confidence threshold;
- use the last accepted model during unresolved streaming intervals.

### Static/background component score

For component `j`, calculate robust normalized derivative energies:

\[
E_1(j)=
\frac{\operatorname{median}(\Delta y_j)^2}
{\operatorname{Var}(y_j)+\epsilon},
\qquad
E_2(j)=
\frac{\operatorname{median}(\Delta^2 y_j)^2}
{\operatorname{Var}(y_j)+\epsilon}.
\]

Also calculate:

- cosine to `[1, alpha_gain]` in observation coordinates;
- cosine to `[-alpha_gain, 1]`;
- spatial total variation/high-frequency mass;
- spatial support fraction;
- global-intensity correlation;
- event-versus-quiet modulation for reporting only, never held-out component
  selection;
- stability across blocks.

Use a declared weighted score. The initial component selector must not use labels.
Return:

```text
background_component: 0 | 1 | null
background_confidence: float
classification_status: resolved | unresolved | degenerate
classification_terms: {...}
```

### Background reconstruction

Given

\[
\mathbf y=WQ(\mathbf x-\boldsymbol\mu),
\]

compute the inverse mapping. For orthogonal `W` after whitening,

\[
\widehat{\mathbf x}^{(B)}
=
\boldsymbol\mu+Q^\dagger W^\top
\mathbf y^{(B)},
\]

where only the selected background component is retained. The current-frame
background is the second observation coordinate. Define

\[
R_t=I_t-\widehat B_t.
\]

Required outputs:

- background reconstruction;
- amplitude-preserving residual;
- differential component as diagnostic;
- reconstruction closure;
- background-confidence map/series;
- component identity metadata.

Do not positive-clip before Stage 2. Positive views are downstream artifacts.

### Conservative leakage controls

Provide explicit ablations:

- exact background subtraction;
- confidence-weighted subtraction;
- structural floor preserving a small background fraction;
- no-subtraction fallback when unresolved.

Select among them on semi-synthetic leakage/event-preservation criteria, not real
held-out labels.

## Stage 2: local noisy Parzen ICA

### Patch geometry

Use overlapping patches:

```text
residual_patch.shape = [T, P]
```

where `P = patch_height * patch_width`. Initial reference:

- patch sizes 16, 24, or 32 px;
- 50% overlap;
- Hann/Tukey overlap-add window;
- deterministic raster order;
- no patch may allocate a full-field covariance;
- edge padding and valid-mask semantics must be explicit.

### Quiet noise estimation

Estimate noise from Stage-1 residual quiet frames. Support:

1. `diagonal_robust`;
2. `diagonal_shrinkage`;
3. `low_rank_plus_diagonal` (gated).

At minimum record:

- per-pixel center and variance;
- first-difference variance;
- intensity-conditioned variance bins;
- temporal ACF;
- local spatial covariance;
- covariance condition number;
- shrinkage coefficient;
- quiet block stability.

Do not assume the residual is already noise. Structured quiet artifacts must be
flagged.

### Noise-corrected signal subspace

For a patch:

\[
\widehat\Sigma_s=
\Pi_{\succeq0}
(\widehat\Sigma_r-\widehat\Sigma_n).
\]

Rank candidates are selected from positive eigenmodes with explicit thresholds:

- eigenvalue above noise floor;
- stable across temporal blocks;
- bounded by configured maximum rank;
- minimum explained structured energy;
- no rank selected when all modes are below the floor.

Required lanes:

- ordinary PCA whitening;
- noise-corrected whitening;
- optional generalized-eigenvalue whitening.

The noise-agnostic lane is a required control.

### Stage-2 ICA observations

Let `Q` project/whiten the retained signal subspace:

\[
\widetilde{\mathbf r}_t=Q\mathbf r_t,
\qquad
\mathbf y_t=W\widetilde{\mathbf r}_t.
\]

Maintain the projected noise covariance

\[
\Sigma_{\widetilde n}=Q\Sigma_nQ^\top.
\]

For output row `w_j`,

\[
\nu_j^2=\mathbf w_j^\top
\Sigma_{\widetilde n}\mathbf w_j.
\]

Floor and cap `nu_j^2`; record when bounds are active. The initial stochastic
implementation may stop-gradient through `nu_j^2` within each update and refresh
it afterward. A full-autograd variance-gradient lane is optional and must be
compared on tiny fixtures.

### Noise-convolved Parzen model

Represent latent source `s_j` by

\[
\widehat p_{s_j}(s)=
\frac1M\sum_m\mathcal N(s;c_{jm},h_j^2).
\]

The observed output density is

\[
\widehat p_{y_j}(y)=
\frac1M\sum_m
\mathcal N(y;c_{jm},h_j^2+\nu_j^2).
\]

The negative score is

\[
\psi_j(y)=
\frac{y-\overline c_j(y)}{h_j^2+\nu_j^2},
\qquad
\overline c_j(y)=\sum_m\alpha_{jm}(y)c_{jm}.
\]

Use the same bounded natural-gradient/decorrelation framework as Stage 1, but
with component-specific noise broadening.

### Posterior source denoising

For each dictionary component,

\[
\mu_{jm}(y)=
\frac{\nu_j^2c_{jm}+h_j^2y}{h_j^2+\nu_j^2}.
\]

The posterior source mean is

\[
\widehat s_j(y)=\sum_m\alpha_{jm}(y)\mu_{jm}(y).
\]

Required output variants:

- noisy demixed output;
- posterior source mean;
- optional posterior variance;
- score-function value;
- shrinkage amount;
- dictionary responsibility entropy.

### Dictionary contract

Implement a bounded dictionary class with:

```python
@dataclass(frozen=True)
class ParzenDictionaryConfig:
    maximum_centers: int
    minimum_center_separation: float
    bandwidth: float
    bandwidth_min: float
    bandwidth_max: float
    update_rate: float
    replacement_policy: str
    warmup_samples: int
    seed: int
```

Dictionary updates use posterior source estimates, not raw noisy outputs. Support:

- deterministic reservoir;
- farthest-center replacement;
- optional weighted coreset;
- age and usage counts;
- freeze after calibration;
- rollback/checkpoint.

Do not update dictionaries using held-out labels.

### Reconstruction

Map posterior sources back to patch coordinates:

\[
\widehat{\mathbf r}^{(S)}_t
=Q^\dagger W^{-1}\widehat{\mathbf s}_t.
\]

For orthogonal `W`, use `W^T`; otherwise use a checked pseudoinverse. Record
condition number and reconstruction closure. Combine patches with normalized
overlap-add.

The preliminary residual is

\[
\mathbf e_t=\mathbf r_t-\widehat{\mathbf r}^{(S)}_t.
\]

Do not immediately call `e_t` noise.

### Neural-signal qualification

For each component, compute:

- top spatial mass and support fraction;
- total variation;
- annularity/ring score;
- compactness and connected-component count;
- local temporal coherence;
- event-versus-quiet robust SNR;
- recurrence across temporal blocks;
- seed/component stability;
- correlation with raw spatial gradients;
- correlation with estimated motion fields;
- saturation overlap;
- posterior denoising gain;
- residual reduction.

Use a declared rule returning:

```text
accepted_neural
structured_artifact
unresolved
```

The first implementation must include a manual-review export for unresolved and
artifact components.

### Measurement-noise qualification

The remaining residual may be labeled `measurement_noise` only if it passes:

- temporal ACF norm bound;
- spatial correlation beyond optical scale bound;
- low event-triggered residual energy;
- low correlation with image gradients/motion;
- intensity-conditioned variance calibration;
- stable quiet statistics across blocks;
- no large coherent connected components.

Otherwise retain a structured-artifact or unresolved residual channel.

## Optional alternating refinement

After Stage 1 and Stage 2 pass synthetic gates, implement one optional refinement
lane:

1. reconstruct accepted structured signal;
2. refit Stage-1 background from `observation - accepted_signal`;
3. recompute residual;
4. refit Stage-2 signal once;
5. preserve all original and refined channels separately.

No unbounded iteration in v1. Report whether leakage improves or merely moves
energy between channels.

## Public numerical interfaces

Implement typed pure-array interfaces equivalent to:

```python
@dataclass(frozen=True)
class WhiteningResult:
    mean: np.ndarray
    covariance: np.ndarray
    whitening: np.ndarray
    dewhitening: np.ndarray
    eigenvalues: np.ndarray
    condition_number: float
    effective_rank: int
    identifiable: bool
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class ParzenDictionaryState:
    centers: np.ndarray
    ages: np.ndarray
    usage: np.ndarray
    bandwidth: np.ndarray
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class DemixingFit:
    method_id: str
    demixing: np.ndarray
    converged: bool
    iterations: int
    objective: float | None
    gradient_norm: float | None
    update_count: int
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class Stage1Result:
    background: np.ndarray
    dynamic_residual: np.ndarray
    differential_component: np.ndarray
    closure_residual: np.ndarray
    background_component: int | None
    confidence: float
    classification_status: str
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class NoiseModel:
    covariance: np.ndarray
    model_kind: str
    intensity_bins: np.ndarray | None
    variance_by_bin: np.ndarray | None
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class SignalSubspace:
    basis: np.ndarray
    eigenvalues: np.ndarray
    rank: int
    projected_noise_covariance: np.ndarray
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class NoisyParzenPosterior:
    noisy_output: np.ndarray
    posterior_mean: np.ndarray
    posterior_variance: np.ndarray | None
    score: np.ndarray
    projected_noise_variance: np.ndarray
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class Stage2PatchResult:
    structured_reconstruction: np.ndarray
    residual: np.ndarray
    components: np.ndarray
    component_maps: np.ndarray
    component_classes: tuple[str, ...]
    diagnostics: dict[str, Any]
```

Required pure-array functions:

```python
center_and_whiten_2d(...)
gaussian_parzen_log_density(...)
gaussian_parzen_score(...)
parzen_responsibilities(...)
symmetric_decorrelate(...)
fit_batch_cs_parzen_2d(...)
fit_stochastic_parzen_ica(...)
track_demixing_components(...)
component_derivative_energy(...)
component_staticness_score(...)
reconstruct_selected_component(...)
stage1_background_residual(...)

estimate_patch_noise_model(...)
noise_corrected_subspace(...)
projected_noise_variance(...)
noisy_parzen_log_density(...)
noisy_parzen_score(...)
noisy_parzen_posterior_mean(...)
fit_local_noisy_parzen_ica(...)
qualify_structured_component(...)
overlap_add_patches(...)
residual_noise_diagnostics(...)
decomposition_closure(...)
```

All functions must validate axes, shapes, dtypes, finite values, parameter bounds,
and conditioning. The numerical module must not read files.

## Recommended repository layout

```text
neurobench/
├── algorithms/
│   └── hierarchical_parzen_ica.py
├── experiments/
│   └── hierarchical_parzen_ica/
│       ├── __init__.py
│       ├── config.py
│       ├── preflight.py
│       ├── sampling.py
│       ├── stage1.py
│       ├── noise.py
│       ├── stage2.py
│       ├── dictionaries.py
│       ├── streaming.py
│       ├── synthetic.py
│       ├── evaluation.py
│       ├── visuals.py
│       ├── artifacts.py
│       └── runner.py
└── metrics/
    └── hierarchical_separation.py
```

Add:

```text
examples/spon_ca_burst_hierarchical_parzen_noisy_ica.example.json
tests/test_hierarchical_parzen_algorithms.py
tests/test_hierarchical_parzen_config.py
tests/test_hierarchical_parzen_stage1.py
tests/test_hierarchical_parzen_stage2.py
tests/test_hierarchical_parzen_synthetic.py
tests/test_hierarchical_parzen_runner.py
tests/test_hierarchical_parzen_cli.py
docs/workflows/spon_ca_burst_hierarchical_parzen_noisy_ica.md
```

The workflow document must be written from implemented behavior and actual
results. Do not copy this plan and present it as completed work.

## CLI contract

Add a lazy-loaded experiment group:

```text
neurobench experiment hierarchical-parzen-ica preflight
neurobench experiment hierarchical-parzen-ica synthetic
neurobench experiment hierarchical-parzen-ica semi-synthetic
neurobench experiment hierarchical-parzen-ica tiny-smoke
neurobench experiment hierarchical-parzen-ica run
neurobench experiment hierarchical-parzen-ica report
neurobench experiment hierarchical-parzen-ica realtime-benchmark
```

Required behavior:

- `preflight`: read-only except for a new explicit artifact directory;
- `synthetic`: no repository data required;
- `semi-synthetic`: requires explicitly selected quiet source frames but no full
  experiment run;
- `tiny-smoke`: generated small arrays only;
- `run`: requires an identical reviewed preflight and a new output root;
- `report`: reads completed artifacts only;
- `realtime-benchmark`: frozen inference by default; adaptation is a separate
  explicit flag.

## Strict manifest contract

Use the included example as the initial schema. Unknown fields fail validation.
Resolve paths relative to the manifest. Validate:

- frame and coordinate contracts;
- method IDs;
- patch geometry and overlap;
- dictionary bounds;
- rank and covariance bounds;
- output estimates;
- deterministic seeds;
- no existing destination;
- source and label checksums;
- thread/GPU resource settings;
- requested visualization count.

## Artifact contract

A completed run must write:

```text
config.resolved.json
preflight.json
run_state.json
progress.jsonl
resource_summary.json
input_manifest.json

stage1/
    fit_summary.tsv
    block_fits.jsonl
    selected_model.json
    component_tracking.tsv
    staticness_terms.tsv
    update_history.tsv
    background.npy                 # when enabled
    dynamic_residual.npy           # when enabled
    differential.npy               # optional diagnostic
    closure_summary.json

stage2/
    patch_manifest.tsv
    noise_model.json
    noise_covariance_summary.npz
    subspace_summary.tsv
    fit_summary.tsv
    dictionary_manifest.tsv
    component_manifest.tsv
    accepted_signal.npy            # when enabled
    structured_artifact.npy        # when enabled
    measurement_noise.npy           # when enabled and qualified
    closure_summary.json

metrics/
    decomposition_metrics.json
    leakage_matrix.tsv
    stage1_metrics.tsv
    stage2_metrics.tsv
    noise_metrics.tsv
    event_preservation.tsv
    component_stability.tsv
    detection_metrics.json
    candidate_peaks.tsv
    shared_failure.tsv
    latency.tsv

figures/
    decomposition_sheets/
    stage1_geometry.png
    stage1_angle_tracking.png
    stage1_staticness_scatter.png
    noise_spectrum_rank.png
    parzen_density_diagnostics.png
    accepted_component_gallery.png
    artifact_component_gallery.png
    leakage_matrix.png
    residual_acf_psd.png
    noise_variance_calibration.png
    event_preservation.png
    recall_candidate_tradeoff.png
    fixed_budget_recall.png
    shared_failure_matrix.png
    stability_summary.png
    realtime_latency.png

review/
    diagnostic_tiffs/
    unresolved_components.tsv
    artifact_components.tsv
    shared_miss_review.tsv
    workbench_suggestions.json

report.md
RESULTS_INDEX.md
```

Dense arrays are optional and must be individually estimated in preflight.
Prefer compact maps, selected TIFFs, component factors, and reconstructable
artifacts. Use memory maps and `.partial` atomic completion for dense arrays.

Every scientific array must declare:

- axes;
- shape;
- dtype;
- units;
- normalization;
- frame alignment;
- stage and method ID;
- source checksum;
- causal status;
- whether labels influenced fitting;
- closure/reconstruction relationship.

## Synthetic and semi-synthetic suite

Synthetic validation is mandatory before real data.

### Stage-1 fixtures

1. static background + white noise;
2. gain-scaled static background;
3. linear background drift;
4. nonlinear slow drift;
5. static background + fast event;
6. static background + slow ramp/plateau event;
7. background and signal with similar persistence;
8. pure noise;
9. one-pixel translation edge;
10. saturation/clipping;
11. heteroscedastic noise;
12. unresolved equal-staticness components.

### Stage-2 fixtures

1. one localized non-Gaussian source + Gaussian noise;
2. two independent localized sources;
3. correlated/synchronous sources;
4. one annular source;
5. overlapping sources;
6. Gaussian source counterexample;
7. impulsive artifact + signal;
8. motion-edge component + signal;
9. Poisson-Gaussian noise;
10. correlated read noise;
11. rank overestimate/underestimate;
12. bandwidth and dictionary mismatch;
13. source/noise variance extremes;
14. patch-boundary source;
15. component permutation/sign challenge.

### Semi-synthetic fixtures

Inject known annular or membrane-like events into real quiet Spon frames with:

- multiple SNR levels;
- multiple rise/decay times;
- slow plateaus;
- recurrence across windows;
- overlapping neurons;
- edge and center locations;
- background drift;
- controlled translations;
- Poisson-Gaussian and impulsive noise.

Retain exact true `B`, `S`, `A`, and `N` arrays.

## Metrics and visualization

Implement the complete contract in
`docs/research/HIERARCHICAL_PARZEN_VISUALS_AND_METRICS.md`.

At minimum, no scientific decision may be made without:

- closure error;
- B/S/A/N leakage matrix;
- event amplitude/area/onset/duration preservation;
- residual temporal and spatial correlation;
- component stability;
- detection performance at quiet-calibrated and fixed-budget policies;
- mandatory visual decomposition sheets;
- real-time latency when a causal lane is considered.

## Experiment matrix

### Stage 1 screen

| Factor | Values |
|---|---|
| method | fixed, adaptive gain, batch CS-Parzen, stochastic score Parzen |
| lag | 1, 4 (lag 4 gated) |
| covariance | ordinary, robust |
| dictionary size | 32, 64, 128 |
| bandwidth multiplier | 0.5, 1, 2 |
| adaptation | frozen, slow block, slow online |
| subtraction | exact, confidence weighted |

Use a bounded fractional design initially. Do not cross every factor blindly.

### Stage 2 screen

| Factor | Values |
|---|---|
| patch size | 16, 24, 32 |
| overlap | 50% |
| noise model | diagonal robust, diagonal shrinkage |
| subspace | ordinary PCA, noise-corrected PCA |
| max rank | 2, 4, 8 |
| ICA | ordinary Parzen, noisy Parzen posterior |
| dictionary size | 32, 64, 128 |
| bandwidth multiplier | 0.5, 1, 2 |
| adaptation | frozen, slow block |
| qualification | none control, morphology/coherence rule |

The first scientific comparison must isolate:

1. noise-corrected whitening effect;
2. noise-convolved Parzen effect;
3. posterior source denoising effect;
4. component qualification effect.

### Required anchors

- Raw Direct;
- fixed/adaptive difference;
- existing batch CS-Parzen pairwise;
- latent filter amplitude;
- latent smoother amplitude;
- amplitude PCA rank 8;
- noise-agnostic local ICA;
- optional PMD/CNMF external comparison when available.

## Advancement gates

### C0 — implementation integrity

- all focused tests pass;
- deterministic reruns match;
- Raw Direct anchor reproduces;
- frames/coordinates are correct;
- artifact writes are collision-safe;
- closure arithmetic is exact within tolerance;
- no labels enter fitting.

Failure stops all interpretation.

### C1 — Stage-1 numerical stability

- finite whitening/demixing;
- covariance condition within bound or explicit unresolved status;
- bounded angle updates;
- no silent sign/permutation switches;
- deterministic dictionary behavior;
- objective/update direction validated on tiny fixtures.

### C2 — Stage-1 scientific validity

Advance Stage 1 only if, on semi-synthetic cases:

- neural leakage into background is below the preregistered limit;
- background leakage into residual improves over fixed/adaptive subtraction;
- slow event amplitude and temporal area are preserved;
- motion edges are not interpreted as background success;
- unresolved cases are surfaced;
- improvement is stable across seeds and temporal blocks.

### C3 — Stage-2 numerical stability

- PSD noise-corrected covariance;
- stable rank selection;
- finite projected noise variance;
- bounded dictionaries/bandwidths;
- demixer orthogonality/condition within bound;
- no component explosion/collapse;
- patch overlap-add closure passes.

### C4 — Stage-2 decomposition validity

Advance Stage 2 only if:

- structured-signal NMSE/correlation improves over noise-agnostic ICA;
- B/S/A/N leakage improves;
- final residual has less temporal/spatial/event structure;
- injected event amplitude/timing/morphology are preserved;
- artifacts are not relabeled as noise;
- performance is stable across seeds, patch offsets, and nearby ranks.

### C5 — real-data preservation

Advance to downstream detection only if:

- known-coordinate amplitude, area, onset, duration, and localization stay within
  declared bounds;
- shared failures are categorized as representation, NMS, threshold, matching,
  annotation, or no-evidence failures;
- visual review shows no systematic signal absorption into background/noise;
- residual diagnostics remain credible.

### C6 — downstream value

Advance a lane to later fusion only if, under identical policies, it either:

- improves mean held-out known-label recall and wins at least three of four
  bursts; or
- reduces candidate burden by at least 20% with no known-label recall loss.

This gate does not establish ordinary precision.

### C7 — real-time candidacy

A causal frozen lane may be considered for streaming only if:

- total inference p95 is below 20 ms on the native Spon field with headroom;
- adaptation is slower and independently bounded;
- unresolved/unstable updates freeze or roll back;
- p99, memory, initialization, and drift behavior are reported;
- no noncausal smoother or centered temporal filter is included.

## Unit and integration tests

At minimum test:

### Core Parzen mathematics

- density integrates/normalizes numerically on simple fixtures;
- score matches finite differences;
- noisy density equals clean density convolved with Gaussian for analytic cases;
- posterior mean matches scalar Gaussian conditioning;
- responsibilities sum to one;
- log-domain evaluation avoids underflow;
- dictionary replacement is deterministic;
- bandwidth/noise floors work;
- leave-one-out behavior avoids self-kernel collapse.

### Stage 1

- exact common/difference directions;
- gain-adjusted directions;
- reconstruction in observation coordinates;
- background subtraction closure;
- derivative/second-derivative energies;
- staticness selection and unresolved margin;
- sign/permutation continuity;
- batch and stochastic update agreement on tiny mixtures;
- near-singular covariance rejection;
- slow neural ramp leakage test;
- motion-edge counterexample.

### Stage 2

- noise covariance estimators;
- PSD projection of `Sigma_r - Sigma_n`;
- rank recovery on known fixtures;
- projected noise variance;
- noisy Parzen score/posterior;
- source reconstruction;
- overlap-add invariance;
- patch-boundary source;
- Gaussian-source nonidentifiability diagnostic;
- correlated-source degradation;
- artifact qualification;
- final residual diagnostics.

### Workflow

- strict config and unknown-field rejection;
- preflight collision refusal;
- label projection overlay;
- no label access during fitting;
- output estimates and caps;
- atomic partial cleanup;
- CLI lazy imports and thread settings;
- generated report/figure presence;
- sparse-positive semantics;
- Raw Direct anchor;
- deterministic tiny smoke artifact tree.

## Parallel-safe implementation plan

### W0 — interface freeze and repository audit

- confirm current main head and current anchors;
- freeze dataclasses, method IDs, artifact names, and config schema;
- add no scientific implementation yet.

### W1 — Parzen numerical core

- clean/noisy densities;
- score functions;
- posterior means;
- dictionaries;
- decorrelation;
- finite-difference tests.

### W2 — Stage 1

- sampling;
- whitening;
- batch and stochastic lanes;
- component tracking;
- staticness classification;
- reconstruction;
- Stage-1 synthetic tests.

### W3 — Stage 2

- patching;
- noise models;
- noise-corrected subspace;
- noisy Parzen ICA;
- posterior reconstruction;
- component qualification;
- overlap-add.

### W4 — fixtures

- synthetic B/S/A/N generators;
- semi-synthetic injection into quiet frames;
- parameterized SNR, kinetics, overlap, drift, motion, and noise.

### W5 — metrics

- closure;
- leakage;
- event preservation;
- noise diagnostics;
- stability;
- downstream detector adapters.

### W6 — visuals/reports

- mandatory figures;
- selected TIFFs;
- component galleries;
- shared-failure matrix;
- concise report and results index.

### W7 — workflow shell

- config;
- preflight;
- resources;
- artifacts;
- CLI;
- run-state/progress;
- tiny smoke.

### W8 — real-time benchmark

- frozen inference path;
- slow adaptation path;
- latency/memory;
- freeze/rollback tests.

Integration dependencies:

```text
W0 -> W1
W1 -> W2 and W3
W2/W3 -> W4/W5/W6
W4/W5/W6 -> W7
W7 -> tiny smoke
all scientific gates -> W8
```

## Implementation order for Codex

1. Audit current files and write a short implementation map.
2. Freeze the public interfaces and strict manifest.
3. Implement W1 with exhaustive pure-array tests.
4. Implement Stage-1 fixed/batch/stochastic references.
5. Implement Stage-1 synthetic report and visual contract.
6. Implement noise models and noise-corrected subspace.
7. Implement noisy Parzen score and posterior source estimates.
8. Implement local patch reconstruction and overlap-add.
9. Implement component qualification and artifact/noise separation.
10. Implement synthetic and semi-synthetic leakage reports.
11. Add config, preflight, runner, artifacts, and CLI.
12. Run only generated tiny smoke and bounded synthetic/semi-synthetic tests.
13. Write the workflow document from actual behavior.
14. Stop with exact commands and required preflight for a real-data run.
15. Do not execute the full Spon run without explicit user selection.

## Definition of done

This implementation wave is complete when:

- the numerical identities and score gradients are verified;
- Stage 1 can reconstruct background and preserve residual amplitude;
- Stage 2 explicitly models noise and produces posterior source estimates;
- synthetic/semi-synthetic B/S/A/N closure and leakage are reported;
- mandatory visuals and plots are generated;
- residual noise is qualified rather than assumed;
- a tiny smoke run emits the complete artifact contract;
- current NeuRev anchors remain reproducible;
- no completed output is overwritten;
- no full real-data run occurs without explicit authorization;
- the final handoff identifies passed, failed, and not-run gates without
  overstating scientific success.
