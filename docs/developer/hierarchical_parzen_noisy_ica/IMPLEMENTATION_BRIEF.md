# Hierarchical Parzen ICA and Noisy Parzen ICA

Implementation brief: 2026-07-29.

Status: documentation-only implementation specification. Code, tests, synthetic
fixtures, preflight, report builders, and tiny smoke tests are authorized. Any
full Spon Ca Burst run, long GPU job, or downstream detector replacement requires
explicit user selection and a new collision-safe output root.

## 1. Executive directive

Implement a reversible two-stage NeuRev experiment:

```text
observed movie X
    -> Stage 1 temporal Parzen ICA
    -> reconstructed background B_hat
       + amplitude-preserving residual R = X - B_hat
    -> Stage 2 local noisy Parzen ICA
    -> structured dynamic signal S_hat
       + noise/artifact candidate N_hat = R - S_hat
```

The complete output must satisfy

\[
X=\widehat B+\widehat S+\widehat N+E_{\mathrm{closure}}
\]

within numerical tolerance. Scientific success additionally requires correct
attribution, preserved neural dynamics, a noise-like residual, stable results,
and useful downstream evidence.

Do not implement this as a generic function called `hierarchical_ica` that hides
all assumptions. Every stage, component assignment, covariance estimate,
posterior source estimate, fallback, and diagnostic must remain separately
inspectable.

## 2. Required repository context

Read before editing:

1. `AGENTS.md`;
2. `docs/research/HIERARCHICAL_PARZEN_NOISY_ICA.md`;
3. `docs/workflows/spon_ca_burst_pairwise_separation.md`;
4. `docs/research/PAIRWISE_ICA_AS_TEMPORAL_DERIVATIVE.md`;
5. `docs/workflows/spon_ca_burst_pairwise_feature_fusion.md`;
6. `docs/workflows/spon_ca_burst_latent_dynamics.md`;
7. `docs/workflows/spon_ca_burst_representation_benchmark.md`;
8. `neurobench/algorithms/pairwise_separation.py`;
9. `neurobench/algorithms/latent_dynamics.py`;
10. `neurobench/algorithms/representation_benchmark.py`;
11. `neurobench/metrics/sparse_detection.py`;
12. `neurobench/cli/experiment.py`;
13. focused tests for those modules.

Use `.venv-neurobench/bin/python` for repository commands. Preserve all ignored
local data and completed outputs.

## 3. Current scientific anchors

Every report must state the current anchors before new results:

- adjacent-frame InfoMax and CS-Parzen approximately rediscovered subtraction;
- pairwise derivative/ICA fusion did not improve Raw Direct under the tested
  formulations;
- Raw Direct and all historical frame/index conventions remain frozen anchors;
- the offline latent smoother improved known-label amplitude recovery but
  increased candidate burden and is noncausal;
- amplitude PCA rank 8 produced a small provisional fixed-budget gain;
- rank-64 spatial ICA did not converge reliably;
- sparse-positive labels do not define exhaustive negatives;
- completion of a run is not scientific success.

The new experiment is not authorized to reinterpret those results by changing
their metrics, files, or output roots.

## 4. Scientific terminology

Use the following names exactly in code, configuration, artifacts, and reports:

- `observation`: calibrated measured movie;
- `stage1_background`: reconstructed background-like contribution;
- `stage1_residual`: observation minus Stage-1 background;
- `stage2_structured_signal`: reconstructed accepted structured sources;
- `stage2_noise_candidate`: Stage-1 residual minus structured signal;
- `closure_residual`: observation minus the three reconstructed channels;
- `stage1_component`: an aggregate temporal ICA component;
- `stage2_source`: a local structured source under the noisy ICA model;
- `noise_model`: quiet-derived additive-noise model;
- `posterior_source_mean` and `posterior_source_variance`;
- `assignment_status`: `resolved`, `ambiguous`, or `fallback`.

Do not call the Stage-1 non-background ICA output `dynamic_signal`. Do not call
`stage2_noise_candidate` measurement noise until it passes the residual contract.

## 5. Stage 1 model

### 5.1 Input

For one spatial sample `p` and current frame `t`, support:

```text
pair embedding:   [I[t-1,p], I[t,p]]
triple embedding: [I[t-2,p], I[t-1,p], I[t,p]]
```

Mandatory v1 uses the pair embedding. The triple embedding is an explicit
ablation and may not silently replace v1.

Use a deterministic bounded spatial sample for fitting. Sampling policies:

- `uniform_field` mandatory;
- `quiet_intensity_stratified` recommended;
- `exclude_saturated` mandatory when a saturation mask exists;
- labels unavailable to fitting.

### 5.2 Preprocessing

Implement reusable operations for:

- robust per-coordinate centering;
- covariance estimation;
- eigenvalue flooring;
- whitening and dewhitening;
- declared condition-number and effective-rank checks;
- optional global gain/offset normalization;
- optional motion-corrected input only when an external registered input is
  explicitly supplied.

No method may imply that ICA removes geometric motion.

### 5.3 Batch CS-Parzen reference

Reuse or generalize the repository's bounded Cauchy-Schwarz Parzen independence
criterion. It must support dimensions 2 and 3 and bounded block evaluation.

Return:

```python
@dataclass(frozen=True)
class Stage1Fit:
    embedding_dim: int
    mean: np.ndarray
    covariance: np.ndarray
    whitening: np.ndarray
    dewhitening: np.ndarray
    demixing: np.ndarray
    mixing: np.ndarray
    objective: float
    converged: bool
    iterations: int
    seed: int
    bandwidth: float
    condition_number: float
    diagnostics: dict[str, Any]
```

### 5.4 Stochastic Parzen reference

Implement a bounded stochastic information-gradient lane only after the batch
reference passes.

Required controls:

- deterministic seed;
- bounded dictionary or replay buffer;
- batch size;
- update interval;
- learning-rate schedule;
- symmetric decorrelation after every declared number of updates;
- maximum matrix/angle change per update;
- gradient clipping;
- objective and gradient norm telemetry;
- burn-in and frozen-evaluation periods;
- fallback to last valid demixer.

The stochastic lane must reproduce the batch direction on deterministic tiny
fixtures within a declared angular tolerance before dataset use.

### 5.5 Component assignment

Implement `score_background_components(...)` with no label access.

For each component, return at least:

```python
@dataclass(frozen=True)
class BackgroundComponentScore:
    component: int
    first_difference_energy: float
    second_difference_energy: float
    global_intensity_correlation: float
    low_spatial_frequency_fraction: float
    spatial_support_fraction: float
    event_modulation_proxy: float
    motion_edge_correlation: float | None
    composite_score: float
```

The assignment result must contain:

```python
@dataclass(frozen=True)
class BackgroundAssignment:
    selected_components: tuple[int, ...]
    status: str
    score_margin: float
    signs: tuple[int, ...]
    permutation: tuple[int, ...]
    diagnostics: dict[str, Any]
```

Mandatory rules:

- use derivative energy, not derivative mean;
- use both temporal and spatial evidence;
- record all score terms;
- reject a forced assignment when the top-two margin is below the configured
  threshold;
- preserve continuity against the preceding valid fit in streaming mode;
- labels and event centers may be used only for later evaluation, not assignment.

### 5.6 Background reconstruction

Implement reconstruction in observation coordinates. For every fitted sample and
full-frame application, verify:

```text
embedded observation approximately equals reconstructed sum of components
```

Extract the current-coordinate contribution of the selected background
components. Write:

- `stage1_background`;
- `stage1_residual`;
- `stage1_reconstruction_error`;
- component-wise current-coordinate contributions.

The amplitude residual is the only Stage-1 movie passed to Stage 2.

## 6. Stage 2 noisy Parzen ICA

### 6.1 Local patch model

For overlapping patch vector `r_t`:

\[
r_t=A s_t+n_t.
\]

Support:

- fixed patch sizes;
- overlap-add windows;
- bounded component rank;
- deterministic patch traversal;
- border handling;
- optional active-patch skipping based only on unsupervised energy/coherence;
- component continuity across adjacent analysis windows.

Global high-rank ICA is an ablation, not the production target.

### 6.2 Quiet noise estimation

Implement:

```python
@dataclass(frozen=True)
class PatchNoiseModel:
    center: np.ndarray
    covariance: np.ndarray
    inverse_sqrt: np.ndarray
    eigenvalues: np.ndarray
    model_kind: str
    quiet_frames: int
    diagnostics: dict[str, Any]
```

Mandatory v1 noise models:

1. `diagonal_robust`;
2. `diagonal_plus_low_rank` as a gated extension.

Diagnostics must include:

- variance range and floors;
- temporal ACF summary;
- spatial ACF summary;
- covariance condition number;
- intensity/variance relation;
- quiet holdout likelihood or calibration error;
- fraction of covariance energy represented by the selected model.

### 6.3 Noise-corrected signal subspace

Implement

\[
\widehat\Sigma_s=
\operatorname{PSD}(\widehat\Sigma_r-\widehat\Sigma_n).
\]

Return the selected signal eigenvectors and all rejected/floored eigenvalues.
Rank selection must be bounded by configuration and may use:

- positive noise-corrected eigenvalue threshold;
- parallel-analysis or quiet-null threshold as a gated extension;
- stability across temporal blocks.

Do not choose rank from known labels.

### 6.4 Parzen source model

Implement a per-source Gaussian mixture/Parzen prior:

```python
@dataclass(frozen=True)
class ParzenSourceModel:
    centers: np.ndarray
    bandwidth: float
    projected_noise_variance: float
    weights: np.ndarray
    diagnostics: dict[str, Any]
```

Required numerical functions:

```python
noisy_parzen_log_density(...)
noisy_parzen_responsibilities(...)
noisy_parzen_score(...)
noisy_parzen_posterior_mean(...)
noisy_parzen_posterior_variance(...)
```

Evaluate mixtures with log-sum-exp. Enforce finite positive bandwidth and noise
variance floors. Emit the effective observed bandwidth
`sqrt(bandwidth**2 + projected_noise_variance)` but keep its factors separate.

### 6.5 Dictionary policy

Mandatory v1:

- bounded centers selected deterministically from calibration outputs;
- centers frozen during the first batch comparison;
- optional posterior-mean dictionary refresh only after the fixed-dictionary
  reference passes;
- no uncontrolled append-only dictionary.

For streaming adaptation, use reservoir, quantile, clustering, or exponential
prototype updates with a hard maximum count. Record occupancy, replacement rate,
center drift, and effective sample age.

### 6.6 Noisy stochastic demixer

Implement:

```python
@dataclass(frozen=True)
class NoisyParzenICAFit:
    mean: np.ndarray
    signal_whitening: np.ndarray
    signal_dewhitening: np.ndarray
    demixing: np.ndarray
    mixing: np.ndarray
    source_models: tuple[ParzenSourceModel, ...]
    converged: bool
    iterations: int
    seed: int
    objective_history: tuple[float, ...]
    gradient_norm_history: tuple[float, ...]
    diagnostics: dict[str, Any]
```

Support three modes:

- `batch_reference`;
- `mini_batch`;
- `streaming_frozen_dictionary`.

The first scientific run compares all three on synthetic data, but only the batch
reference and a bounded mini-batch lane are required for the first real preflight.

### 6.7 Source acceptance and reconstruction

A Stage-2 component is accepted as structured signal only when it satisfies a
declared combination of:

- reproducibility across seeds/windows;
- spatial compactness or coherent local support;
- nontrivial posterior SNR;
- temporal structure above quiet-noise expectation;
- low motion-edge correlation;
- bounded uncertainty;
- positive improvement to semi-synthetic source reconstruction.

Component acceptance must be label-blind for the denoising experiment. Known
labels may be used after freezing to evaluate preservation and detection.

Reconstruct accepted components, overlap-add with a declared window, and write:

- `stage2_structured_signal`;
- `stage2_noise_candidate`;
- posterior uncertainty summaries;
- patch disagreement maps;
- accepted/rejected component manifests.

## 7. Optional reversible refinement

Implement only after the no-refinement reference passes.

One bounded iteration may update Stage-1 background using the structured signal:

```text
B_refined = trust_region_background_update(X - S_hat, B_initial)
R_refined = X - B_refined
refit Stage 2 once
```

Constraints:

- no labels;
- maximum relative background change;
- no reduction in semi-synthetic signal preservation;
- report before/after attribution and closure;
- never iterate to unconstrained convergence in v1.

## 8. Proposed repository layout

```text
neurobench/
├── algorithms/
│   ├── hierarchical_parzen.py
│   └── noisy_parzen_ica.py
├── experiments/
│   └── hierarchical_parzen_noisy_ica/
│       ├── __init__.py
│       ├── config.py
│       ├── preflight.py
│       ├── sampling.py
│       ├── stage1.py
│       ├── noise.py
│       ├── stage2.py
│       ├── reconstruction.py
│       ├── synthetic.py
│       ├── evaluation.py
│       ├── figures.py
│       ├── artifacts.py
│       └── runner.py
├── metrics/
│   └── decomposition.py
└── reports/
    └── hierarchical_parzen.py
```

Add:

```text
examples/spon_ca_burst_hierarchical_parzen_noisy_ica.example.json
tests/test_hierarchical_parzen_algorithms.py
tests/test_noisy_parzen_density.py
tests/test_noisy_parzen_ica.py
tests/test_hierarchical_parzen_config.py
tests/test_hierarchical_parzen_synthetic.py
tests/test_hierarchical_parzen_runner.py
tests/test_hierarchical_parzen_cli.py
docs/workflows/spon_ca_burst_hierarchical_parzen_noisy_ica.md
```

Register thin lazy-loaded commands:

```text
neurobench experiment hierarchical-parzen synthetic
neurobench experiment hierarchical-parzen preflight
neurobench experiment hierarchical-parzen smoke
neurobench experiment hierarchical-parzen run
neurobench experiment hierarchical-parzen report
```

`run` must require an identical reviewed preflight. `report` must be read-only.

## 9. Strict manifest behavior

- unknown fields fail;
- all paths resolve relative to the manifest;
- frame intervals are validated at the boundary;
- output roots must not exist;
- preflight writes only to a new explicit preflight directory;
- configuration equality between preflight and run is exact after canonical
  serialization;
- labels are listed as evaluation-only inputs;
- every stochastic lane declares all seeds;
- every dense artifact has an explicit write flag and estimated size;
- full Spon and GPU execution remain explicit selections.

The example manifest supplied with this package is the starting schema.

## 10. Artifact contract

A completed scientific run must write:

```text
config.resolved.json
preflight.json
input_manifest.json
run_state.json
progress.jsonl
resource_summary.json
stage1/
    fit.json
    objective.tsv
    component_scores.tsv
    assignment.json
    component_contributions.npz
    background.npy                 # when enabled
    residual.npy                   # when enabled
    closure_summary.json
noise/
    model.json
    covariance.npz
    temporal_acf.tsv
    spatial_acf.tsv
    intensity_variance.tsv
stage2/
    patch_manifest.tsv
    fits/
    accepted_components.tsv
    rejected_components.tsv
    posterior_summary.tsv
    patch_disagreement.npy         # when enabled
    structured_signal.npy          # when enabled
    noise_candidate.npy            # when enabled
    closure_summary.json
evaluation/
    synthetic_metrics.tsv
    leakage_matrices.npz
    signal_preservation.tsv
    residual_validity.tsv
    stability.tsv
    detection_lane_summary.tsv
    known_matches.tsv
    unmatched_candidates.tsv
    latency.tsv
figures/
    *.png
representative_tiffs/
    *.tif
report.md
RESULTS_INDEX.md
```

Large arrays must record:

- shape and axes;
- dtype;
- frame alignment;
- units and normalization;
- causal/noncausal status;
- source checksum;
- model/fit ID;
- display scale for TIFFs;
- whether the artifact is scientific or display-only.

Use `.partial` outputs and atomic rename. Preserve failed run state and bounded
error information.

## 11. Numerical stability requirements

### Stage 1

- float64 covariance, whitening, objective accumulation, and demixer updates;
- eigenvalue floors relative to the largest eigenvalue;
- explicit condition-number limit;
- symmetric decorrelation with PSD floors;
- finite objective and gradient checks;
- angle/matrix update cap;
- deterministic restart selection;
- unresolved assignment fallback;
- reconstruction closure validation.

### Stage 2

- PSD-projected noise-corrected covariance;
- rank-zero outcome is valid and must not be forced to one;
- log-domain Gaussian mixtures;
- positive floors for source bandwidth and projected noise variance;
- responsibility sums verified against one;
- posterior variance nonnegative and finite;
- demixer condition and orthogonality diagnostics;
- bounded dictionary size;
- overlap-add denominator floors;
- exact local and global closure checks.

### Scientific stability

- multiple seeds and temporal blocks;
- nearby bandwidth/noise/rank/patch perturbations;
- component matching before comparison;
- subspace angles and matched correlations;
- candidate Jaccard and known-event preservation;
- report failure rates and unresolved rates.

A smooth objective is not enough. A result is scientifically unstable when small
changes move neural energy among background, signal, and noise channels.

## 12. Write and resource safety

1. Preserve user changes and ignored data under `Inputs/` and `Outputs/`.
2. Never overwrite completed outputs.
3. Set OpenMP/BLAS limits before heavy imports.
4. Use memory maps and bounded spatial chunks.
5. Never allocate a full pixels-by-pixels covariance.
6. Estimate every optional dense output before execution.
7. Record RAM, disk, CPU, GPU, and latency telemetry.
8. Use deterministic seeds and atomic metadata.
9. Full-video TIFF generation is conditional and preregistered.
10. A preflight passing does not itself authorize a full run.

## 13. Minimum public numerical interfaces

```python
robust_temporal_embedding(...)
center_and_whiten_embedding(...)
fit_cs_parzen_demixer(...)
fit_stochastic_parzen_demixer(...)
score_background_components(...)
assign_background_components(...)
reconstruct_current_coordinate(...)
estimate_patch_noise_model(...)
noise_corrected_signal_subspace(...)
fit_noisy_parzen_source_model(...)
noisy_parzen_responsibilities(...)
noisy_parzen_score(...)
noisy_parzen_posterior_mean(...)
noisy_parzen_posterior_variance(...)
fit_noisy_parzen_ica(...)
select_structured_components(...)
overlap_add_reconstruction(...)
decomposition_closure(...)
attribution_leakage_matrix(...)
residual_validity_metrics(...)
```

Every function validates shapes, finite values, axes, bounds, and deterministic
behavior. Pure algorithm modules must not read files or know Spon-specific frame
numbers.

## 14. Definition of done for the implementation wave

The first implementation wave is complete only when:

- all pure numerical functions and tests pass;
- batch Stage 1 reproduces the known derivative/common-mode geometry on tiny
  fixtures;
- stochastic Stage 1 agrees with batch within the declared angular tolerance;
- noisy Parzen posterior means/variances match analytic or numerical references;
- Stage 2 improves source recovery over ordinary ICA in at least the declared
  noisy synthetic regimes;
- decomposition closure passes;
- figures and tables are generated from real artifacts, not placeholders;
- preflight is strict, collision-safe, and resource bounded;
- a tiny smoke run emits the complete artifact tree;
- the workflow document is written from implemented behavior;
- no full Spon/GPU run has occurred without explicit selection;
- the handoff states which gates passed, failed, or remain unrun.

The implementation objective is not to maximize entropy or visual smoothness. It
is to produce stable, inspectable, quantitatively validated attribution of movie
energy to background, structured dynamics, and residual noise/artifact.
