# Center–Whiten–ICA Architecture Benchmark

Status: preregistered implementation contract
Date: 2026-08-15

## Question

For Spon Ca Burst, which parts of the pipeline

```text
Raw -> centering bank -> whitening bank -> ICA rotation -> activity evidence
```

are useful, redundant, or harmful when every architecture uses identical
samples, event windows, candidate budgets, and protected evaluation?

This is an architecture ablation. A whitened coordinate is not a cleaned movie
or biological source, and ICA completion is not source identification.

## Fixed sample-axis contract

Every pixel-frame sample has a three-channel multiscale context vector. The
channel count and sample IDs remain fixed across compared architectures.

Two center families are compared:

- `joint_residual`: raw minus a causal prior-frame spatial-annulus mean;
- `joint_msln`: the same numerator divided by its causal local scale with a
  quiet-fitted floor.

This makes the MSLN question explicit: local studentization is a center-family
factor rather than an assumed prerequisite for whitening.

The four frozen three-context banks are:

1. `reference_diverse`: `s5/g1/t15`, `s15/g3/t23`, `s15/g3/t31`;
2. `compact_temporal`: `s5/g1/t9`, `s5/g1/t15`, `s5/g1/t31`;
3. `broad_temporal`: `s15/g3/t9`, `s15/g3/t23`, `s15/g3/t31`;
4. `spatial_diverse_t15`: `s5/g1/t15`, `s9/g3/t15`, `s15/g3/t15`.

The current frame, protected spatial core, and most recent temporal guard frame
are excluded in every causal context.

## Whitening bank

Primary screen modes:

- `none`: explicit negative/control lane;
- `ica_native_global`: conventional ICA centering/PCA-whitening control;
- `global_diagonal_quiet`;
- `global_full_oas_quiet`;
- `global_full_fixed_ridge_0p05_quiet`;
- `local_diagonal_quiet_t64`;
- `local_full_oas_quiet_t64`.

Only finalists receive tile-size sensitivity at 32 and 128 pixels, fixed-ridge
sensitivity at 0.01 and 0.10, and a median-center ablation. Covariance fitting
uses the quiet-fit block only. Empirical energy calibration uses a disjoint
quiet-calibration block, and quiet evaluation uses held-out blocks.

## ICA bank

External-whitening lanes run an ICA rotation with internal whitening disabled.
The `ica_native_global` control uses the conventional internal global PCA
whitening. Primary ICA choices are symmetric FastICA with:

- nonlinearities `logcosh`, `exp`, and `cube`;
- seeds 7, 13, and 19;
- maximum iterations 300 and 800;
- tolerances `1e-4` and `1e-5`.

The first screen uses the compact covering set rather than the complete
Cartesian product:

```text
logcosh: seeds 7/13/19, 300 iterations, tolerance 1e-5
exp:     seed 7,       300 iterations, tolerance 1e-5
cube:    seed 7,       300 iterations, tolerance 1e-5
logcosh sensitivity: seed 7, 800 iterations, tolerances 1e-4 and 1e-5
```

Identity rotation is retained as a no-ICA representation control. All ICA fits
use the same deterministic natural-prevalence sample IDs. No spatial label or
ROI membership enters fitting, tuning, or finalist selection.

## Staged search

### Stage A — center-bank screen

Evaluate all eight center-family/context-bank combinations before covariance
or ICA widening. Retain at most four by a Pareto rule over held-out quiet
calibration, event/quiet contrast, signed-carrier morphology, temporal width,
and compute cost. Burst intervals may define event time windows; spatial labels
remain unopened.

### Stage B — whitening screen

Evaluate the seven primary whitening modes within retained center banks.
Retain at most six center/whitening combinations. A mode with unresolved
support is not silently rescued: fallback fraction and fallback identity are
selection metrics.

### Stage C — ICA screen

Evaluate the compact ICA covering set plus identity rotation. Freeze at most
four complete architectures using:

- quiet covariance/tail calibration;
- event/quiet contrast;
- component-map compactness and recurrence;
- cross-seed matched-map correlation;
- convergence, component ambiguity, and effective demixing condition;
- morphology and temporal preservation;
- runtime, host RAM, and VRAM.

The screen writes JSON/CSV/checkpoints only. The explicit screening audit
opt-out reason is:

```text
User requested a metrics-only first-pass architecture screen on 2026-08-15;
full diagnostics are mandatory for the frozen finalists.
```

### Stage D — protected finalist evaluation

Only after `freeze_decision.json` exists may spatial labels be loaded. Evaluate
the frozen primary and up to three diagnostic architectures with the exact
existing NMS/matching and candidate-budget contract. Report fixed-budget known
positive recall, rank and nearest distance, candidate burden, per-burst
consistency, and the label-informed best ceiling separately.

### Stage E — scientific audit

Generate and validate the complete three-section scientific audit for every
frozen finalist. Expert and model sections remain marker-pure. Videos show Raw,
centered feature channels, whitened channels/declared compact summary, ICA
components, activity evidence, and the pooled map used for ranking.

## Interpretation controls

- Full-rank global whitening followed by ideal ICA is linearly equivalent to
  ICA-native whitening. Differences primarily diagnose regularization,
  leakage, finite-sample behavior, or optimization—not a new source model.
- Local whitening is not globally invertible as one matrix and may change the
  ICA problem materially, but its unresolved/fallback support must be exposed.
- `joint_residual` versus `joint_msln` isolates local studentization.
- Identity rotation distinguishes whitening utility from ICA utility.
- Unmatched candidates remain unknown, not false positives.
- This recording is development data; protected metrics are exploratory and
  not independent confirmation.

## Resource contract

- Local CUDA only unless a later user request selects UF DSI; never Bala.
- One process, one center bank, and one whitening lane at a time.
- Maximum four CPU threads, one worker, 4 GiB experiment VRAM, and bounded
  host-memory limits enforced outside the process.
- Atomic row checkpoints and deterministic resume after every architecture.
- New collision-safe output roots; completed roots are immutable.
- Preflight checks source hashes, the absence of label inputs, RAM/disk/GPU
  headroom, exact combination counts, and CPU/CUDA parity before the screen.

## Implementation and execution

The strict manifest is
`examples/spon_ca_burst_center_whiten_ica_v1.example.json`. The maintained
entry point is:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv-neurobench/bin/python tools/run_center_whiten_ica_benchmark.py \
  preflight examples/spon_ca_burst_center_whiten_ica_v1.example.json

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv-neurobench/bin/python tools/run_center_whiten_ica_benchmark.py \
  screen examples/spon_ca_burst_center_whiten_ica_v1.example.json
```

`screen` runs and resumes Stages A-C only. It writes scalar JSON/CSV metrics,
row checkpoints, retained-lane records, and `freeze_decision.json`; it does not
load spatial labels or render diagnostic media. A completed screen therefore
has scientific status `protected_evaluation_not_run`, not scientific success.
Protected evaluation and the complete audit are separate post-freeze steps.
