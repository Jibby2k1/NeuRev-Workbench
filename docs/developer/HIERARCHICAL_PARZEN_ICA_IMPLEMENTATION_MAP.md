# Hierarchical Parzen ICA implementation map

Last updated: 2026-07-29.

This is the concise ownership and progress map for
`docs/developer/HIERARCHICAL_PARZEN_NOISY_ICA_IMPLEMENTATION_BRIEF.md`. It does
not replace that scientific contract.

## Current implementation wave

| Work package | Canonical ownership | Status |
| --- | --- | --- |
| W0 interfaces and manifest | `neurobench/algorithms/hierarchical_parzen_ica.py`, `neurobench/experiments/hierarchical_parzen_ica/config.py` | Public result/config interfaces and strict v1 manifest implemented; preflight and CLI not yet implemented. |
| W1 Parzen numerical core | `neurobench/algorithms/hierarchical_parzen_ica.py` | Initial clean/noisy density, score, posterior, dictionary, decorrelation, stochastic-update, tracking, and closure functions implemented with deterministic tests. |
| W2 Stage 1 | `neurobench/experiments/hierarchical_parzen_ica/stage1.py`, `safety.py` | Guarded frozen-model routes implemented. Reference initialization, affine feedback bounds, learned-fraction anchoring, and convergence fallback eliminate recursive explosions. Accepted-model streaming tracking remains. |
| W3 Stage 2 | `signal_noise_config.py`, `signal_noise_split.py`, future patchwise `noise.py`/`stage2.py` | Scalar noisy-Parzen posterior screening and exact signal/noise TIFF reconstruction are implemented. Noise-corrected local subspaces, ICA, overlap-add, and qualification remain. |
| W4 fixtures | `neurobench/experiments/hierarchical_parzen_ica/synthetic.py` | All 12 required Stage-1 synthetic fixture classes implemented with exact B/S/A/N closure and five-seed execution. Semi-synthetic Spon injection remains unauthorized. |
| W5 metrics | `neurobench/metrics/hierarchical_separation.py`, `neurobench/experiments/hierarchical_parzen_ica/evaluation.py` | Multi-seed Stage-1 background, signal, artifact, closure, feedback, convergence, and fallback reporting implemented. Full Stage-2 B/S/A/N reporting remains. |
| W6 visuals and reports | `architecture_visuals.py`, `signal_noise_split.py`, reports | Real-data Stage-1 background/dynamics plus conservative and balanced posterior signal/noise TIFFs are implemented. Full patchwise decomposition sheets remain. |
| W7 workflow shell | architecture and signal/noise configs/runners, `neurobench/cli/experiment.py` | Strict read-only-preflighted, collision-safe CPU routes are implemented for architecture, grid, and posterior-split diagnostics. The full patchwise Stage-2 runner remains unimplemented. |
| W8 streaming benchmark | future `streaming.py` | Not implemented. |

## Frozen interfaces and method IDs

The pure numerical module owns the public dataclasses requested by the brief:
whitening, dictionary state/config, demixing fit, Stage-1 result, noise model,
signal subspace, noisy posterior, and Stage-2 patch result.

The strict manifest currently requires these Stage-1 method IDs:

```text
fixed_common_difference_reference
adaptive_gain_common_difference
batch_cs_parzen_pairwise
stochastic_parzen_score_pairwise
```

It requires the Stage-2 control pair:

```text
ordinary_parzen_ica
noisy_parzen_ica_posterior
```

These identifiers must remain stable in artifacts and reports. Any schema
change requires a version increment and migration note rather than silent
reinterpretation.

## W2 checkpoint evidence

See
`docs/developer/HIERARCHICAL_PARZEN_STAGE1_CHECKPOINT_REPORT.md`
for the initial unguarded failure and
`docs/developer/HIERARCHICAL_PARZEN_STAGE1_GUARDED_SYNTHETIC_REPORT.md`
for the 240-combination guarded follow-up. The real-data state-architecture
comparison is in
`docs/developer/HIERARCHICAL_PARZEN_STAGE1_ARCHITECTURE_VISUAL_REPORT.md`.

The deterministic Stage-1 checkpoint now covers:

1. aligned `[I(t-k), I(t)]` construction with explicit zero-based output frames;
2. quiet-prefix-only whitening, gain estimation, and component classification;
3. fixed, adaptive-gain, bounded batch CS-Parzen, and bounded stochastic
   score-Parzen method IDs;
4. label-free staticness selection with explicit unresolved and rank-degenerate
   no-subtraction paths;
5. observation-coordinate reconstruction using the previously reconstructed
   background state rather than the previous raw frame, so sustained events do
   not collapse to a derivative-only residual;
6. exact closure, slow-ramp preservation, illumination-gain, and motion-edge
   counterexample tests.

On the deterministic quiet-plus-slow-ramp fixture, the fixed reference produces
`0.000493` signal-residual NMSE, `0.000493` signal leakage into background, and
`5.62e-08` maximum closure error. On the multiplicative-drift fixture, adaptive
gain estimates `1.035047` from a true `1.035` and its residual RMS is `0.0443`
times the fixed lane. These are unit-fixture results, not evidence on Spon.

The learned lanes do not pass this checkpoint. In the direct four-lane fixture,
batch CS-Parzen produces `2.77e17` signal-residual NMSE because its recursive
background coefficient is `6.12445`. The three-iteration stochastic smoke fit
does not converge and produces `0.6776` signal-residual NMSE. “Finite output”
is therefore only a software-smoke result, not scientific success.

The guarded matrix passes numerical stability but fails scientific validity.
Adaptive remains the best general reference. Guarded stochastic Parzen improves
all five similar-persistence cases but loses all 15 other signal comparisons;
batch raw feedback is rejected in 57/60 runs. The next checkpoint is improved
unresolved classification, event-preservation evaluation, and accepted-model
tracking.

The user-authorized real-data visual comparison subsequently fit one fully
learned stochastic model and applied teacher-forced, raw-recursive,
quiet-fixed-point, and bounded reference-plus-innovation architectures to UI
frames 1800--2359. Both explicit fixed-point lanes preserved background spatial
contrast through the interval; raw recurrence retained 85.8%. This is rollout
evidence, not neuron-detection evidence. Stage 2 and GPU execution remain
unauthorized.
