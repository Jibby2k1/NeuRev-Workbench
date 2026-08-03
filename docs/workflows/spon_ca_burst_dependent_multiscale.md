# Spon Ca Burst dependent multiscale demixing

## Current scope

W0--W5 are implemented: strict configuration and dependency semantics; deterministic 5/7/15-pixel views; local-PCA provider/fallback handling; joint quiet-noise diagnostics; reversible structural baselines; matrix-Renyi group dependence with declared nuisance residualization; and a 15-fixture, three-seed, three-lane generated gate. Fixed-scale real-data diagnostic videos are also implemented.

The W5 gate did not authorize scientific advancement: C1 passed, C2 and C3 failed, C4 was not qualified, and C5 remained diagnostic only. W6 semi-synthetic injection and the W7 patchwise scientific run are not complete. The accepted carrier remains scientific trace authority, and every residual remains noise_candidate. See docs/research/SPON_CA_BURST_DEPENDENT_MULTISCALE_REAL_DIAGNOSTIC_V1_RESULTS.md.

## Population-preserving W5b

The patchwise population-preserving revision passes C1 and C2 but fails morphology-specific C3. Aggregate preservation medians pass, but broad neural activity remains attenuated while broad drift and motion-crossing cases are amplified. W6 and W7 therefore remain blocked. See docs/research/SPON_CA_BURST_DEPENDENT_MULTISCALE_POPULATION_W5B_RESULTS.md.

The preferred visual review is the six-panel grayscale artifact at Outputs/HierarchicalParzenICA/spon_ca_burst_dependent_multiscale_grayscale_review_v1/grayscale_decomposition_review.mp4. Signed grayscale uses black for negative, mid-gray for zero, and white for positive.

## Confirmation-authority W5c

The quiet-calibrated W5c experiment compared the W5b reference, coherence-confirmed authority, carrier-constrained authority, and a combined primary lane against the patchwise orthogonal baseline. The primary lane reduced overall median leakage but failed broad-neural attribution and morphology-specific preservation; C2 and C3 failed, so W6 remains blocked. See docs/research/DEPENDENT_MULTISCALE_DEMIXING_METHODS_AND_MEETING_BRIEF_2026_08_02.md.

## Generated baseline

```bash
.venv-neurobench/bin/python -m neurobench.cli.main \
  experiment dependent-multiscale synthetic \
  --output-dir /tmp/neurev-dependent-multiscale-synthetic
```

This runs only small generated fixtures. The destination and its `.partial`
sibling must both be absent.

For an artifact-free test:

```bash
.venv-neurobench/bin/python -m neurobench.cli.main \
  experiment dependent-multiscale smoke
```

## Real-data guard and diagnostic lane

The preflight command writes a collision-safe, read-only-source audit. A ready audit does not authorize execution by itself. The scientific W7 run remains gated because W5 C2/C3 failed. After explicit user selection, run may be used only as the marked full-frame failure-analysis diagnostic described in the results note; it does not claim patchwise W7 completion or carrier replacement.

The preferred completed diagnostic is Outputs/HierarchicalParzenICA/spon_ca_burst_dependent_multiscale_real_v3. Its fixed-scale video is visuals/full_interval_decomposition_diagnostic.mp4.

Sparse unmatched candidates remain unknown, not negative. UI frame intervals are one-based and inclusive; array intervals are zero-based and half-open; coordinates are x=column, y=row.
