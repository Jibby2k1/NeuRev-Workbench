# Spon Ca Burst ICA and whitening evaluation

## Purpose and authority

This workflow executes the conditional-factorial program specified in
`docs/research/ICA_WHITENING_HYPERPARAMETER_EVALUATION_PLAN.md`. It evaluates
temporal, spatial, and joint spatiotemporal ICA while treating whitening
geometry, covariance scope, support, strength, and regularization as explicit
factors.

The example manifest is
`examples/spon_ca_burst_ica_whitening_evaluation_v1.example.json`. It resolves
the frozen movie and sparse labels through `NEUROBENCH_DATA_ROOT` while keeping
program artifacts under the active implementation checkout. The source-data
checkout is an input authority, not the Git implementation authority.

This workflow does not establish precision, specificity, one-to-one biological
source identity, or cross-recording generalization. Unmatched candidates remain
unknown.

## Frozen v1 design

The v1 manifest uses a compatibility-filtered Cartesian product over:

- temporal, spatial, and joint spatiotemporal ICA;
- FastICA log-cosh and bounded HSIC pairwise-rotation objectives;
- no, spatial, temporal, both separable orders, and joint whitening;
- global-quiet and regional-quiet covariance fits;
- centered and causal temporal semantics where applicable;
- ranks 2 and 4; and
- 3- and 5-sample spatial/temporal supports.

Every compatible cell contains deterministic boundary/canonical anchors and a
32-point scrambled Sobol interior. Three paired screen seeds are retained. The
ready preflight reports 912 valid cells and 106,080 fits with design SHA-256
`ddfab0c2753d77fd37792ed8ab021c92560d7f7cb9104c699e636d718deb0eab`.

The design is intentionally large. Execution is resumable and writes one
compact checkpoint record per completed fit. Full matrices are deferred to
frozen finalists so the screen remains within its output contract.

## Guarded commands

Use the maintained environment and explicitly select the data authority:

```bash
export NEUROBENCH_DATA_ROOT='/path/to/NeuRev-Workbench-data-authority'
export MPLCONFIGDIR=/tmp/neurev-mpl
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
```

Create a non-colliding preflight:

```bash
.venv-neurobench/bin/python -m neurobench.experiments.ica_whitening_evaluation \
  preflight \
  --config examples/spon_ca_burst_ica_whitening_evaluation_v1.example.json \
  --artifact-dir Outputs/ICAWhiteningEvaluation/preflight_spon_ca_burst_ica_whitening_evaluation_v1
```

Run a bounded, stratified smoke:

```bash
.venv-neurobench/bin/python -m neurobench.experiments.ica_whitening_evaluation \
  synthetic \
  --config examples/spon_ca_burst_ica_whitening_evaluation_v1.example.json \
  --preflight-dir Outputs/ICAWhiteningEvaluation/preflight_spon_ca_burst_ica_whitening_evaluation_v1 \
  --limit 72
```

Run or resume the full truth-known stage by omitting `--limit`. Never delete or
overwrite an existing completed root. A matching run manifest and input/design
hashes are required for resumption.

## Stage gates

The first biological-label-sealed stage uses isolated, overlapping,
synchronous, and source-free truth-known movies. It records numerical health,
source recovery, crosstalk, trace preservation, null abstention, and compact
frequency/space-time response summaries.

Every model family must contain at least one fit satisfying all frozen gates:

- convergence fraction at least 0.8;
- mean truth-source absolute correlation at least 0.7;
- mean absolute crosstalk at most 0.25;
- trace preservation at least 0.7;
- numerical resolution under conditioning/effective-rank guards; and
- correct abstention on the source-free fixture.

If any family lacks a passing fit, the program stops before real-data label
evaluation. Smoke outcomes are diagnostic and cannot weaken the full-run gate.

If all families pass, freeze the label-free selection rule before opening
real-data label metrics. The complete label-response surface is exploratory;
only nested or label-free-selected finalists may enter protected confirmation.

## Scientific audit

The manifest enables the repository scientific-audit standard. Truth-known and
other unlabeled stages use a frozen candidate-surrogate Model section and mark
Expert annotations `not_applicable`. A promoted real-data finalist must produce
the complete expert-only, model-only, and matched-comparison evidence set under
`docs/workflows/SCIENTIFIC_AUDIT_OUTPUT_STANDARD.md`.

Computational completion, gate advancement, audit completion, within-recording
scientific support, and independent-recording confirmation are separate states.
