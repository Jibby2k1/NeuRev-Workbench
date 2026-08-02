# Codex Goal: Hierarchical Parzen ICA and Noisy Parzen ICA for NeuRev

Implement the stage-gated hierarchical source-separation program specified by:

1. `docs/developer/HIERARCHICAL_PARZEN_NOISY_ICA_IMPLEMENTATION_BRIEF.md`;
2. `docs/research/HIERARCHICAL_PARZEN_NOISY_ICA.md`;
3. `docs/research/HIERARCHICAL_PARZEN_VISUALS_AND_METRICS.md`;
4. `examples/spon_ca_burst_hierarchical_parzen_noisy_ica.example.json`.

Treat the current `main` branch and all completed `Inputs/` and `Outputs/` artifacts as immutable evidence. Read `AGENTS.md`, the pairwise-separation workflow, the pairwise-feature-fusion workflow, the latent-dynamics workflow, the representation-benchmark workflow, and the sparse-label diagnostic tooling before editing.

## Scientific objective

Estimate four explicit channels from a measured movie:

```text
background_like
structured_neural_signal
structured_artifact
measurement_noise
```

and provide the top-level closure:

```text
observation ~= background_like + structured_neural_signal
             + structured_artifact + measurement_noise
```

The user-facing three-way interpretation may aggregate artifact with residual uncertainty, but the implementation must keep structured artifact separate internally so that motion or saturation is not mislabeled as noise or neural signal.

## Mandatory method hierarchy

### Stage 1 — Background reconstruction

Implement explicit lanes:

1. fixed common/difference reference;
2. existing batch CS-Parzen pairwise reference;
3. stochastic Parzen score-function ICA;
4. optional stochastic mini-batch CS-Parzen objective.

For every lane:

- construct aligned two-frame observations;
- center and stably whiten with eigenvalue floors;
- fit a bounded two-dimensional demixer;
- track sign and permutation continuously;
- classify the background-like component using a declared staticness score;
- allow `unresolved` classification when the score margin is insufficient;
- reconstruct the background contribution in observation coordinates;
- produce an amplitude-preserving current-frame residual;
- measure signal leakage into background and background leakage into residual.

Do not pass only a derivative component to Stage 2.

### Stage 2 — Structured signal under additive noise

On overlapping residual patches:

- estimate quiet noise covariance;
- support diagonal, diagonal-shrinkage, and optional low-rank-plus-diagonal noise models;
- subtract/project the noise covariance before signal-subspace rank selection;
- fit bounded low-rank local ICA;
- implement a Gaussian-noise-convolved Parzen source density;
- use the resulting score function in a stochastic/natural-gradient update;
- maintain a bounded Parzen dictionary using posterior source estimates;
- compute posterior clean source means;
- reconstruct structured signal and final residual;
- qualify components by spatial localization, annularity, temporal coherence, event-versus-quiet evidence, seed/window stability, and motion-edge correlation;
- send rejected structured components to `structured_artifact`, not automatically to measurement noise.

### Optional refinement

After both stages pass synthetic gates, allow one bounded alternating-refinement pass:

```text
background <- refit using observation - accepted_signal
signal     <- refit using observation - updated_background
```

Never overwrite the original Stage-1 and Stage-2 outputs; refinement is a separate lane.

## Required implementation routes

Prefer:

```text
neurobench/algorithms/hierarchical_parzen_ica.py
neurobench/experiments/hierarchical_parzen_ica/
    __init__.py
    config.py
    preflight.py
    sampling.py
    stage1.py
    noise.py
    stage2.py
    dictionaries.py
    streaming.py
    synthetic.py
    evaluation.py
    visuals.py
    artifacts.py
    runner.py
neurobench/metrics/hierarchical_separation.py
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

The pure numerical module must not read files or know Spon-specific frame numbers.

## CLI contract

Add a thin lazy-loaded group:

```text
neurobench experiment hierarchical-parzen-ica preflight
neurobench experiment hierarchical-parzen-ica synthetic
neurobench experiment hierarchical-parzen-ica semi-synthetic
neurobench experiment hierarchical-parzen-ica tiny-smoke
neurobench experiment hierarchical-parzen-ica run
neurobench experiment hierarchical-parzen-ica report
neurobench experiment hierarchical-parzen-ica realtime-benchmark
```

`preflight` must require a new explicit artifact directory. `run` must require an identical reviewed preflight. Full real-data execution remains unauthorized until explicitly selected.

## Non-negotiable constraints

- Preserve user changes and ignored local data.
- Never overwrite a completed output root.
- Use `.venv-neurobench/bin/python`.
- Set BLAS/OpenMP limits before numerical imports.
- Keep UI frames one-based/inclusive and NumPy intervals zero-based/half-open.
- Keep `x=column`, `y=row`.
- Labels may evaluate outputs but may not select demixing directions, dictionaries, bandwidths, noise covariance, ranks, or component identities on held-out data.
- Unmatched event candidates remain `unknown`.
- No all-pixel quadratic Parzen calculation.
- Use bounded deterministic samples, blockwise kernels, finite dictionaries, memory maps, atomic metadata, progress JSONL, and explicit RAM/disk estimates.
- Every output must declare axes, units, normalization, causal status, source checksum, stage, and model ID.
- Do not call a residual measurement noise until whiteness, spatial-correlation, event-locking, and intensity-variance diagnostics pass.

## Parallel-safe work packages

The following may proceed concurrently after shared interfaces are frozen:

- **W1:** core Gaussian-kernel, dictionary, score-function, decorrelation, and posterior-mean mathematics;
- **W2:** Stage-1 pairwise reconstruction, staticness classification, sign/permutation tracking;
- **W3:** Stage-2 noise covariance, noise-corrected rank selection, local patch geometry, overlap-add;
- **W4:** synthetic and semi-synthetic B/S/A/N fixtures;
- **W5:** metrics and leakage analysis;
- **W6:** mandatory figures, TIFFs, galleries, and report generation;
- **W7:** config, preflight, resource estimation, artifacts, and CLI;
- **W8:** streaming/frozen-inference latency benchmark.

Integration order:

```text
W1 -> W2/W3 -> W4 -> W5/W6 -> W7 -> tiny smoke -> W8
```

## Required gates

1. **C0 implementation integrity:** deterministic tests, exact closure, collision safety, Raw Direct anchor reproduction.
2. **C1 Stage-1 identifiability:** background classification resolved with adequate margin and low signal leakage on synthetic fixtures.
3. **C2 Stage-2 numerical stability:** finite outputs, stable decorrelation, bounded dictionaries, no covariance/rank collapse.
4. **C3 decomposition validity:** semi-synthetic leakage matrix passes declared limits and residual noise diagnostics improve.
5. **C4 real signal preservation:** amplitude, temporal area, onset, duration, morphology, and spatial localization are preserved.
6. **C5 downstream utility:** fixed-budget or quiet-calibrated known-label performance improves without misleading precision claims.
7. **C6 real-time candidacy:** frozen inference meets the 20 ms Spon frame budget at p95 with drift/fallback behavior measured.

Do not proceed to the next scientific stage merely because code completed.

## Definition of done for this implementation wave

- all pure-array and config tests pass;
- synthetic and semi-synthetic reports contain the complete required figure set;
- Stage 1 and Stage 2 each have at least one stable reference and one stochastic Parzen lane;
- decomposition closure and leakage are measured, not assumed;
- noise is evaluated through residual diagnostics rather than visual smoothness;
- a tiny smoke run emits the full artifact tree;
- Raw Direct and current repository anchors remain reproducible;
- `docs/workflows/spon_ca_burst_hierarchical_parzen_noisy_ica.md` reports only implemented behavior;
- the final handoff states exact passed, failed, and not-run gates;
- no full Spon/GPU run is launched without explicit authorization.
