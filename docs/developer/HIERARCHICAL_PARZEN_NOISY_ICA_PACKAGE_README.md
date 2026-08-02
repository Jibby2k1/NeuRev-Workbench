# NeuRev Hierarchical Parzen / Noisy-ICA Package

This package specifies a new NeuRev research program for separating a calcium-imaging movie into:

```text
measured movie
    -> background-like reconstruction
    -> amplitude-preserving dynamic residual
    -> structured neural signal
    -> measurement-noise residual
```

The proposed hierarchy is:

1. **Stage 1: common-mode Parzen ICA**
   - Fit a two-observation ICA model to aligned frame pairs.
   - Identify the background-like component using derivative energy, second-derivative energy, mixing direction, spatial broadness, and global-intensity correlation.
   - Reconstruct the background contribution and subtract it from the current frame.
   - Preserve the resulting amplitude residual; do not pass only a derivative component to Stage 2.

2. **Stage 2: local noisy Parzen ICA**
   - Form overlapping residual patches as multichannel observations.
   - Estimate observation-noise covariance from quiet Stage-1 residuals.
   - Perform noise-corrected subspace selection.
   - Fit bounded local Parzen ICA with an explicit additive-noise model.
   - Compute posterior denoised source amplitudes and reconstruct structured signal.
   - Keep structured artifacts separate from stochastic measurement noise when possible.

3. **Evaluation and real-time path**
   - Validate decomposition with semi-synthetic data where true background, signal, and noise are known.
   - Preserve Raw Direct, latent smoother amplitude, amplitude PCA rank 8, existing pairwise methods, and other current NeuRev anchors.
   - Require visual decompositions, leakage matrices, residual diagnostics, event-preservation plots, detection Pareto plots, component-stability reports, and latency distributions.
   - Train or adapt slowly; use frozen or slowly adapting projections for real-time inference.

## Package contents

- `docs/developer/HIERARCHICAL_PARZEN_NOISY_ICA_CODEX_GOAL.md` — the imported Codex goal.
- `docs/research/HIERARCHICAL_PARZEN_NOISY_ICA.md` — concise repository-facing scientific explanation.
- `docs/developer/HIERARCHICAL_PARZEN_NOISY_ICA_IMPLEMENTATION_BRIEF.md` — detailed implementation contract.
- `docs/research/HIERARCHICAL_PARZEN_VISUALS_AND_METRICS.md` — mandatory metrics, figures, plots, and report layouts.
- `docs/research/overleaf/hierarchical_parzen_noisy_ica_main.tex` — non-conflicting Overleaf entrypoint.
- `docs/research/overleaf/hierarchical_parzen_noisy_ica.tex` — canonical mathematical write-up.
- `examples/spon_ca_burst_hierarchical_parzen_noisy_ica.example.json` — proposed strict manifest.
- `AGENTS.md` and `docs/CODEBASE_NAVIGATION.md` — integrated routing entries.
- `docs/developer/HIERARCHICAL_PARZEN_ICA_IMPLEMENTATION_MAP.md` — live ownership and progress map.

## Repository assumptions used

The package is grounded in the current NeuRev repository state through commit
`89bb730197978d393acf25d8af8e4050aafae1bd` and the following established results:

- adjacent-frame InfoMax and CS-Parzen separation recovered an effective derivative direction;
- fixed and adaptive differences were useful as change evidence but not as replacement images;
- offline AR(1) smoother amplitude improved known-label recall, while latent differences and dynamic-drive features underperformed;
- amplitude PCA rank 8 produced a small provisional fixed-budget gain;
- high-rank ICA became unstable and rank-64 FastICA did not converge reliably;
- sparse labels identify known positives but do not identify exhaustive negatives.

## Scope and authorization

This is a **documentation and implementation specification**. It authorizes:

- repository code;
- unit tests;
- deterministic synthetic and semi-synthetic fixtures;
- read-only preflight;
- tiny CPU/GPU smoke tests;
- plot/report generation on synthetic fixtures.

It does **not** by itself authorize:

- a full Spon Ca Burst run;
- replacement of current scientific lanes;
- modification or deletion of completed `Outputs/` evidence;
- promotion of an unsupervised component to a neuron label;
- a closed-loop or stimulation deployment.

Any real-data run must use a new experiment ID and output root, pass a reviewed preflight, and preserve current sparse-positive semantics.
