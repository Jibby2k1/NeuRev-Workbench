# Codex Execution Plan

This plan is optimized for parallel implementation while preserving one
scientific integration path. Work packages may run concurrently only after their
interfaces and fixtures are frozen.

## Operating constraints

- Work on a new branch.
- Preserve all existing user changes and ignored `Inputs/`/`Outputs/` data.
- Never overwrite a completed output root.
- Use `.venv-neurobench/bin/python`.
- Set CPU thread environment before heavy imports.
- Do not launch a full Spon or GPU run without explicit user selection.
- Every package must have deterministic tests before integration.
- Keep labels unavailable to all separation fitting and component selection.

## Dependency graph

```text
W0 contracts/fixtures
  ├── W1 Stage-1 batch algorithms
  ├── W2 noisy Parzen density primitives
  ├── W3 metrics and figures
  └── W4 config/preflight skeleton

W1 -> W5 Stage-1 stochastic lane
W2 -> W6 Stage-2 batch/mini-batch ICA
W1 + W2 + W4 -> W7 runner/artifacts
W3 + W7 -> W8 synthetic reports and tiny smoke
W5 + W6 + W8 -> W9 workflow documentation and handoff
```

## W0: freeze contracts and deterministic fixtures

Owner scope:

```text
neurobench/experiments/hierarchical_parzen_noisy_ica/synthetic.py
tests/fixtures or focused test helpers
```

Tasks:

1. Add exact pair/triple temporal-mixture fixtures.
2. Add scalar Gaussian-Parzen convolution fixtures with analytic posterior
   means/variances.
3. Add multichannel noisy ICA fixtures with known mixing, source, and noise.
4. Add tiny image patches with known background, annular signal, noise, and
   artifact channels.
5. Define axis, dtype, sign, permutation, scale, and closure conventions.
6. Add deterministic source/component matching helpers for tests.

Acceptance:

- fixtures produce identical checksums across runs;
- every true channel is separately available;
- no label file or Spon path is required;
- tests fail before algorithm implementation but fixture tests pass.

## W1: Stage-1 batch algorithms

Owner scope:

```text
neurobench/algorithms/hierarchical_parzen.py
tests/test_hierarchical_parzen_algorithms.py
```

Tasks:

1. Generalize centering/whitening to embedding dimensions 2 and 3.
2. Generalize bounded CS-Parzen objective/rotation search where appropriate.
3. Add deterministic orthogonal parameterization for 3D or use a bounded
   optimizer with explicit retraction/decorrelation.
4. Implement first/second difference energies and spatial assignment metrics.
5. Implement unresolved/fallback component assignment.
6. Reconstruct current-coordinate component contributions.
7. Test exact closure and common/difference reference directions.

Acceptance:

- pair fixture recovers common/difference directions up to allowed ambiguity;
- triple fixture recovers the known aggregate subspace;
- closure within tolerance;
- forced ambiguous assignment is impossible;
- no I/O in algorithm module.

## W2: noisy Parzen density primitives

Owner scope:

```text
neurobench/algorithms/noisy_parzen_ica.py
tests/test_noisy_parzen_density.py
```

Tasks:

1. Implement stable Gaussian mixture log density.
2. Implement noise-convolved responsibilities.
3. Implement analytic score function.
4. Implement posterior source mean and variance.
5. Support weighted centers and vectorized samples/components.
6. Add finite floors and log-sum-exp.
7. Verify by numerical quadrature and finite differences.

Acceptance:

- responsibilities sum to one;
- analytic score matches numerical derivative;
- posterior moments match quadrature;
- zero-noise limit returns the Parzen source model;
- large-noise limit shrinks appropriately toward prior centers;
- tests cover extreme bandwidth/noise ratios.

## W3: metrics, matching, and figure contracts

Owner scope:

```text
neurobench/metrics/decomposition.py
neurobench/reports/hierarchical_parzen.py
tests/test_decomposition_metrics.py
tests/test_hierarchical_parzen_figures.py
```

Tasks:

1. Implement closure metrics.
2. Implement NMSE/correlation/scale-aligned metrics.
3. Implement attribution leakage matrices.
4. Implement event amplitude/area/timing/spatial preservation metrics.
5. Implement temporal/spatial residual ACF and PSD summaries.
6. Implement intensity-noise calibration.
7. Implement seed/block/component stability matching.
8. Build figures with exact filenames from the metrics contract.
9. Write figure manifest mapping each PNG to source data.

Acceptance:

- synthetic perfect decomposition yields identity leakage matrix;
- swapped/leaky channels produce expected matrix entries;
- all figures render from tiny test data;
- no figure performs scientific calculations independently of metrics modules;
- fixed-scale montage test verifies shared display limits.

## W4: strict config and preflight

Owner scope:

```text
neurobench/experiments/hierarchical_parzen_noisy_ica/config.py
neurobench/experiments/hierarchical_parzen_noisy_ica/preflight.py
examples/spon_ca_burst_hierarchical_parzen_noisy_ica.example.json
tests/test_hierarchical_parzen_config.py
```

Tasks:

1. Implement schema-versioned dataclasses.
2. Reject unknown fields and invalid bounds.
3. Resolve paths relative to manifest.
4. Validate source shape, frame indices, quiet interval, labels, and output
   collisions.
5. Estimate RAM/disk/GPU for every enabled dense/visual artifact.
6. Write label projection overlay for later evaluation.
7. Record labels unavailable to fitting.
8. Require explicit preflight destination.

Acceptance:

- canonical serialization exact;
- changed config cannot reuse preflight;
- existing output root rejected;
- invalid stochastic/dictionary/rank settings rejected;
- preflight does no dataset processing beyond bounded validation/overlay.

## W5: Stage-1 stochastic Parzen lane

Depends on W1 and W0.

Owner scope:

```text
neurobench/algorithms/hierarchical_parzen.py
focused stochastic tests
```

Tasks:

1. Add bounded dictionary/replay state.
2. Add stochastic information-gradient update.
3. Add symmetric decorrelation/retraction.
4. Add update angle/matrix cap and fallback.
5. Add objective/gradient/dictionary telemetry.
6. Add batch-versus-stochastic comparison fixture.
7. Add window-to-window sign/permutation tracking.

Acceptance:

- converges to batch reference direction within declared tolerance on matched
  fixture;
- stays finite on long constant/noise sequences;
- component assignment does not flicker on stationary input;
- fallback activates under injected degeneracy;
- runtime scales with bounded dictionary/batch, not full history.

## W6: Stage-2 noisy ICA and patch reconstruction

Depends on W2 and W0.

Owner scope:

```text
neurobench/algorithms/noisy_parzen_ica.py
neurobench/experiments/hierarchical_parzen_noisy_ica/noise.py
neurobench/experiments/hierarchical_parzen_noisy_ica/stage2.py
neurobench/experiments/hierarchical_parzen_noisy_ica/reconstruction.py
tests/test_noisy_parzen_ica.py
```

Tasks:

1. Estimate diagonal robust quiet covariance.
2. Implement optional diagonal-plus-low-rank covariance behind a flag.
3. Implement PSD noise-corrected signal subspace.
4. Implement ordinary Parzen ICA baseline.
5. Implement batch noisy Parzen ICA.
6. Implement bounded mini-batch/stochastic lane.
7. Implement posterior-clean center initialization and optional one-pass refresh.
8. Implement component acceptance scores.
9. Implement overlap-add reconstruction and disagreement maps.
10. Implement rank-zero and unresolved patch behavior.

Acceptance:

- noise-corrected covariance tests pass;
- noisy model improves predefined source NMSE over ordinary ICA in matched
  noisy fixtures;
- patch closure exact;
- rank-zero patch emits zero signal and full residual without failure;
- overlap-add of an identity fixture reproduces the input;
- accepted-source stability reported across seeds.

## W7: orchestration and artifacts

Depends on W1, W2, W4, and stable interfaces from W3/W6.

Owner scope:

```text
neurobench/experiments/hierarchical_parzen_noisy_ica/
  stage1.py
  evaluation.py
  artifacts.py
  runner.py
neurobench/cli/experiment.py
tests/test_hierarchical_parzen_runner.py
tests/test_hierarchical_parzen_cli.py
```

Tasks:

1. Add lazy CLI group and thread configuration.
2. Validate matching preflight/config.
3. Fit Stage 1 on bounded samples.
4. Apply Stage 1 in chunks and write optional dense artifacts.
5. Fit noise model and Stage 2 patch models.
6. Apply Stage 2 and overlap-add.
7. Write closure, metrics, figures, tables, and selected TIFFs.
8. Run single-feature detection only after denoising evaluation is produced.
9. Emit progress/resource heartbeats and atomic run state.
10. Preserve partial diagnostics on scientific gate failure while never marking
    incomplete arrays complete.

Acceptance:

- tiny end-to-end smoke emits full artifact tree;
- output collision refused;
- partial files cleaned on exception;
- exact accounting identity verified;
- report states gates honestly;
- `report` command performs no scientific recomputation.

## W8: synthetic/semi-synthetic experiment and report

Depends on W3 and W7.

Tasks:

1. Run all deterministic scalar/vector fixtures.
2. Run bounded synthetic matrix across seeds.
3. Run tiny semi-synthetic image injections.
4. Compare batch/stochastic Stage 1.
5. Compare ordinary/noise-corrected/noisy Parzen Stage 2.
6. Produce all required plots and leakage matrices.
7. Create a concise result report with failures, not only winners.
8. Stop if G0-G3 fail.

Commands should resemble:

```bash
.venv-neurobench/bin/python -m neurobench.cli.main experiment \
  hierarchical-parzen synthetic \
  --config examples/spon_ca_burst_hierarchical_parzen_noisy_ica.example.json \
  --output-dir /tmp/neurev-hierarchical-parzen-synthetic-v1
```

No real Spon execution is authorized by this work package.

## W9: final workflow and handoff

Depends on W8.

Tasks:

1. Write `docs/workflows/spon_ca_burst_hierarchical_parzen_noisy_ica.md` from
   implemented behavior.
2. Update API reference/navigation.
3. State exact passed, failed, and unrun gates.
4. Provide the exact preflight command for a future real run.
5. List expected RAM/disk/GPU and visual outputs.
6. Stop before any dataset-scale command.

## Integration order

Codex should integrate in this sequence:

```text
W0
W1 + W2 + W3 + W4 in parallel
W5 + W6 in parallel
W7
W8
W9
```

Do not merge parallel work by copying overlapping implementations. Resolve one
owner per public symbol and run all focused tests after each integration.

## Focused validation commands

Expected examples:

```bash
.venv-neurobench/bin/python -m pytest -q \
  tests/test_hierarchical_parzen_algorithms.py \
  tests/test_noisy_parzen_density.py \
  tests/test_noisy_parzen_ica.py \
  tests/test_decomposition_metrics.py

.venv-neurobench/bin/python -m pytest -q \
  tests/test_hierarchical_parzen_config.py \
  tests/test_hierarchical_parzen_runner.py \
  tests/test_hierarchical_parzen_cli.py
```

Also run the relevant existing pairwise, latent-dynamics, representation, sparse
metric, CLI, and artifact tests to prevent regression.

## Stop conditions

Stop and report rather than expanding the search when any of the following occur:

- Stage-1 neural leakage exceeds the preregistered limit;
- background assignment is frequently ambiguous;
- noisy Stage 2 does not beat ordinary ICA on matched synthetic noise;
- leakage matrix is not diagonally dominant;
- residual remains event locked or spatially coherent;
- component stability fails;
- outputs are visually attractive but metrics fail;
- the best result requires nonconvergent high-rank ICA;
- runtime exceeds the resource envelope;
- a requested action would overwrite completed evidence;
- a full Spon/GPU run has not been explicitly selected.

## Final handoff template

```text
Implementation integrity: PASS / FAIL
Stage-1 validity: PASS / FAIL / NOT RUN
Stage-2 noisy-source validity: PASS / FAIL / NOT RUN
End-to-end attribution: PASS / FAIL / NOT RUN
Real signal preservation: PASS / FAIL / NOT RUN
Detection utility: PASS / FAIL / NOT RUN
Real-time eligibility: PASS / FAIL / NOT RUN

Best supported claim:
Primary failure mode:
Exact output root:
Tests executed:
Resources measured:
Next justified action:
Prohibited next action:
```
