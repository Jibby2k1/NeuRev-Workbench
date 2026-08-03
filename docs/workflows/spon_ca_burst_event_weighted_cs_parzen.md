# Spon Ca Burst event-balanced CS-Parzen ICA

## Scope and status

This workflow implements the global event-mass diagnostic specified in
`Event_Balanced_CS_Parzen_ICA_CODEX_SPEC.md`. It asks whether controlled
training mass on annotated event observations produces a stable,
held-out-generalizing departure from the derivative-like global two-frame
CS-Parzen ICA solution.

This is not validated neural/background source separation. Unmatched candidates
remain unknown, precision is not identified by the sparse-positive labels, and
completion is not scientific success. Spatial or time-varying angle fields are
outside this workflow and are not launched even when the machine-readable Gate
C result passes.

## Implementation map

- Weighted blockwise information potentials and the legacy-compatible angle
  optimizer: `neurobench/algorithms/pairwise_separation.py`
- Strict YAML contract: `neurobench/experiments/event_weighted_cs_parzen/config.py`
- Pixel-time identities, equal event mass, overlap merging, and ESS:
  `sample_weights.py`
- Split-first natural, frame-balanced, and ROI-balanced pools: `sampling.py`
- Natural-fixed and weighted-whitening fits: `fitting.py`
- Resource/label projection audit: `preflight.py`
- Checkpointed fold/alpha execution and natural-prevalence evaluation:
  `runner.py`
- Aggregate tables, eight figures, Gate C, and results note: `artifacts.py`

The standard profile contains 76 fits: one natural lane, seven frame-balanced
alphas, seven ROI-balanced alphas, and four weighted-whitening ablations in each
of four held-out burst folds. The smoke profile bounds this to one fold and two
alphas.

## Guarded commands

Preflight reads sources and writes only a new audit directory:

```bash
.venv-neurobench/bin/python -m neurobench.cli.main experiment event-weighted-cs-parzen preflight \
  --config examples/spon_ca_burst_event_weighted_cs_parzen.standard.yaml \
  --artifact-dir Outputs/PairwiseSeparation/event_weighted_preflight_v1
```

Inspect `preflight.json`, `config.resolved.json`, and
`label_projection_overlay.png`. A full Spon run is separately authorized and
must use a new output root:

```bash
.venv-neurobench/bin/python -m neurobench.cli.main experiment event-weighted-cs-parzen run \
  --config examples/spon_ca_burst_event_weighted_cs_parzen.standard.yaml \
  --preflight-dir Outputs/PairwiseSeparation/event_weighted_preflight_v1 \
  --authorize-full-spon
```

An interrupted matching output can use `--resume`; completed output roots are
never overwritten. The smoke profile is still real Spon data and therefore also
requires `--authorize-full-spon`. Generated-only integration coverage does not.

## Scientific and artifact contracts

- Causal preprocessing is frozen at spatial Gaussian sigma 1 px and EMA alpha
  0.4.
- Splits and ten-frame guards are applied before pools or weights are built.
- Natural and event identities are drawn once per fold/mode/seed and reused at
  every alpha.
- Training bursts receive equal event mass even when their durations or ROI
  support differ.
- Whitening is fitted on natural training samples in the primary lanes.
- Holdout objective, candidate count, and known-label recall use natural
  prevalence with no copied or weighted samples.
- Every fit records identities, per-event mass, ESS, angle path, whitening
  diagnostics, runtime, RSS, and sparse-positive semantics.
- Aggregate artifacts include JSON/CSV, eight deterministic figures, a
  machine-readable manifest, `RESULTS.md`, and `stage_gate.json`.
- Scientific status is
  `diagnostic_event_weighting_study_not_validated_source_separation`.

## Current validation and standard result

Generated-only tests cover weighted/unweighted parity, weight-scale invariance,
integer repetition equivalence, zero-weight exclusion, invalid weights, kernel
block invariance, equal event mass, duplicate merge, ESS, held-out guards,
frame/ROI support differences, strict config loading, preflight projection, and
a one-fold two-alpha end-to-end artifact run.

The authorized standard CPU sweep completed all 76 fits in
`Outputs/PairwiseSeparation/spon_ca_burst_event_weighted_cs_parzen_v1`. All fits
converged and the immutable alpha-zero parity gate passed. Natural weighting
recovered 15/79 known labels with 84 candidates. The strongest exploratory
weighted-whitening lane, alpha 0.10, recovered 33/79 with 235 candidates, but
the result was fold-dependent and precision remains unidentified. Primary
ROI-balanced weighting did not preserve held-out recall, no moderate alpha met
every preregistered Gate C criterion, and the profile had only one sample seed.
Gate C therefore failed and no spatial extension was launched.

See `docs/research/CHATGPT_CCFAR_ICA_FEATURE_ENGINEERING_HANDOFF.md` for the
concise result interpretation and proposed continuous-CFAR ICA feature-
engineering follow-up.
