# Fish Intent and Inverse-Control Tooling Roadmap

## Objective

Develop small, auditable tools that turn the repository's existing detection,
alignment, export, dynamics, and latency capabilities into one stage-gated
workflow:

`measurement → intent → action-conditioned dynamics → constrained control`.

The implementation should remain manifest-driven, resumable, and easy for
future agents to inspect. Avoid a single end-to-end script that hides dataset,
split, metric, or safety decisions.

## Reuse before adding code

| Existing capability | Reuse |
|---|---|
| `neurobench.metrics.detection` | object matching, precision/recall, duplicate/split/merge analysis |
| `neurobench.metrics.event_quality` | event precision/recall and timing quality |
| `neurobench.exports.behavior_alignment` | frame/time mapping, sync diagnostics, resampling warnings |
| `neurobench.exports.inverse_dynamics` | accepted trace/event export and alignment bundle |
| Workbench manual ROI and missed-neuron tools | annotation correction, hard positives, ROI history |
| `neurobench.realtime.latency` and `neurobench.online` | latency percentiles, budget metadata, online contracts |
| Dynamics manifests, supervisors, summaries | atomic experiments, resume behavior, hardware-aware execution |

The first engineering task is integration and schema extension, not replacing
these components.

## Proposed package boundaries

### 1. `neurobench/benchmarks/activation/`

Responsibilities:

- build spatial-temporal annotation tiles;
- record coverage as `exhaustive`, `candidate_review`, or `sparse_positive`;
- freeze train/validation/test group assignments;
- stratify random, disagreement, low-SNR, motion, vessel, and edge tiles;
- emit a versioned benchmark manifest with content hashes.

Suggested commands:

- `neurobench activation benchmark-build`
- `neurobench activation benchmark-audit`

### 2. `neurobench/metrics/activation_benchmark.py`

Extend, rather than duplicate, existing detection/event metrics with:

- threshold sweeps;
- precision-recall and FROC curves;
- calibration and expected calibration error;
- stratified metrics by coverage, session, SNR, motion, and registration
  quality;
- group bootstrap confidence intervals;
- explicit suppression of precision when labels are non-exhaustive.

Suggested command:

- `neurobench activation benchmark-evaluate`

### 3. `neurobench/intent/`

Modules:

- `schema.py`: trial, movement-onset, direction, window, coordinate, and
  uncertainty contracts;
- `dataset.py`: causal pre-onset window export;
- `features.py`: population, spatial, temporal, and spatial-temporal
  representations;
- `experiments.py`: matched ablations I0–I5;
- `controls.py`: label, coordinate, and temporal permutations plus
  motion-leakage checks;
- `report.py`: group-level metrics, calibration, lead-time curves, and failure
  gates.

Suggested commands:

- `neurobench intent dataset-build`
- `neurobench intent run`
- `neurobench intent report`

### 4. `neurobench/control/`

Modules:

- `action_schema.py`: requested/measured action, target, amplitude, duration,
  timestamps, latency, clipping, interlocks, and no-action;
- `transitions.py`: state-action-next-state tables and support diagnostics;
- `system_id.py`: simple and probabilistic action-conditioned models;
- `simulator.py`: rollout interface with uncertainty;
- `mpc.py`: constrained action selection and abstention;
- `safety.py`: action envelope, watchdog, rejection reasons, and fallback;
- `report.py`: counterfactual support, calibration, simulator efficacy, and
  safety results.

Suggested commands:

- `neurobench control dataset-build`
- `neurobench control system-id`
- `neurobench control simulate`
- `neurobench control shadow`

## Shared schemas

Every derived example should carry provenance:

- source manifest and content hash;
- fish/session/video/trial IDs;
- detector and registration artifact IDs;
- raw timestamps and mapped timestamps;
- split assignment;
- annotation-coverage state;
- data-quality and exclusion reasons;
- code/config version and seed.

Control transitions additionally require requested and measured action fields.
Missing action is not equivalent to no stimulation; the schema must distinguish
`unknown`, `not logged`, and an observed `no_action`.

Store compact records in Parquet or another typed table format and keep a small
JSON manifest beside them. Large tensors can be referenced by path and slice
rather than copied into each record.

## Experiment contracts

Every experiment directory should contain:

- `config.json` with all hyperparameters and feature definitions;
- `dataset_manifest.json` and split hashes;
- `status.json` written atomically;
- `metrics.json` with metric definitions and valid denominators;
- `predictions.parquet` or a documented compact equivalent;
- `quality.json` with alignment, missingness, and coverage summaries;
- `runtime.json` with CPU/GPU memory and latency percentiles;
- `README.md` that states the estimand, conclusion, and limitations.

Failed runs retain their config and traceback. Resumes skip only experiments
whose completion contract is satisfied.

## Stage-gate report

Add one read-only command:

`neurobench program stage-gate-report --manifest <path>`

It should never infer readiness from file existence alone. It evaluates
machine-readable criteria and returns:

- `pass`, `fail`, or `insufficient_evidence` for each gate;
- exact supporting artifacts and denominators;
- blockers and recommended next acquisition/experiment;
- a warning when a downstream experiment uses an upstream artifact that did
  not pass its gate.

The report should cover:

1. activation benchmark coverage and robustness;
2. intent data readiness and held-out evidence;
3. stimulation identifiability and action support;
4. transition-model calibration;
5. simulator/controller safety;
6. online latency and drift.

## Hardware-aware runner

Reuse the dynamics runner's successful patterns:

- probe CPU, RAM, GPU memory, and current load before launch;
- use conservative batch-size defaults and bounded dataloader workers;
- sample peak GPU/RAM and reduce the next job after OOM;
- run one GPU-heavy experiment at a time;
- keep CPU-only evaluation/backfills separate;
- checkpoint atomically and support time limits;
- leave a configurable safety margin rather than targeting 100% memory;
- log latency p50/p95/p99 for anything intended for online use.

The runner should optimize useful throughput, not instantaneous utilization.
High utilization is acceptable; memory exhaustion, swapping, thermal
instability, and an unresponsive workstation are not.

## Implementation sequence

### T0 — contracts and benchmark integrity

- add annotation-coverage enum and benchmark manifest;
- add precision suppression for non-exhaustive labels;
- add threshold-sweep, FROC, calibration, and group-bootstrap metrics;
- test empty labels, sparse positives, duplicate predictions, split/merge, and
  mixed coverage.

### T1 — intent dataset and baselines

- extend behavior alignment into movement-onset trials;
- implement leakage guards and group splits;
- build I0–I4 representations and controls;
- report per-class, calibration, lead-time, and group-bootstrap metrics.

### T2 — action data and system identification

- finalize action schema and stimulation-log importer;
- validate requested-versus-measured actions and timestamp residuals;
- build transition support diagnostics;
- implement descriptive and regularized action-conditioned baselines.

### T3 — simulation and control

- implement probabilistic transition interface;
- implement constrained MPC with abstention;
- add offline rollout, perturbation, uncertainty, and safety tests;
- add shadow-mode logging without issuing stimulation.

### T4 — integrated operations

- implement stage-gate report;
- add online drift and detector/calibration monitoring;
- require explicit deployment configuration and hardware interlocks for any
  command-capable adapter.

## Test strategy

- Unit-test schemas, timestamps, leakage guards, split isolation, metrics, and
  safety constraints.
- Use synthetic fixtures where spatial-only, temporal-only, confounded, and
  action-dependent ground truth are known.
- Add integration tests that run tiny CPU experiments through each manifest.
- Assert that sparse-positive benchmarks cannot emit ordinary precision.
- Assert that intent test groups never enter fitting, threshold selection, or
  detector selection.
- Assert that the controller cannot issue out-of-envelope actions and abstains
  on unsupported state-action inputs.
- Keep hardware and live stimulation adapters mocked in CI.

## LLM-efficiency rules

- Put stable operating context in short index documents and link to detailed
  experiment artifacts.
- Prefer typed schemas and explicit enums over prose-only conventions.
- Keep commands discoverable through one CLI tree and `--help`.
- Keep generated summaries concise; link raw tables rather than embedding
  thousands of rows in Markdown.
- Include an `experiment_id` and a one-paragraph conclusion in every artifact
  README so agents can route without reading large logs.
- Separate current status from historical narrative.
- Do not duplicate metrics or alignment implementations across packages.

## Definition of done for the next development milestone

The next milestone is complete when:

1. an activation benchmark can state whether precision is measurable and can
   produce PR/FROC curves when it is;
2. an aligned intent dataset can run I0–I4 with leakage controls and group
   splits;
3. an action schema and readiness audit expose exactly which stimulation fields
   are missing;
4. one stage-gate report links all evidence without claiming control readiness;
5. all new paths have tiny CPU integration tests and documented manifests.

Related documents:

- [Program roadmap](../research/FISH_INVERSE_CONTROL_ROADMAP.md)
- [Activation robustness plan](../research/NEURAL_ACTIVATION_DETECTION_ROBUSTNESS.md)
- [Intent and inverse-control plan](../research/LEFT_RIGHT_INTENT_AND_CONTROL_PLAN.md)

