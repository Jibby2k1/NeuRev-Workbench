# AGENTS

Operational routing for coding agents in NeuRev Workbench. Keep this file short:
load only the workflow context needed for the current task.

## Always

- Project root: `/home/jibby2k1/UF Dropbox/CNEL/State Analysis (Fish)/NeuRev-Workbench`.
- Use `.venv-neurobench/bin/python` for Python commands.
- Preserve user changes and ignored local data under `Inputs/` and `Outputs/`.
- Never delete archived experiment logs or overwrite completed output roots.
- Before a long run, verify inputs, output collision, RAM/disk headroom, active
  processes, GPU memory, and a read-only preflight.
- Prefer bounded chunks, explicit thread counts, atomic metadata, progress
  heartbeats, and resumable/idempotent outputs.
- Do not restart the stopped grid128 Stage A sweep or launch its Stage B plan
  unless the user explicitly selects that GPU job.

## Current GPU state

The historical grid128 Stage A sweep is stopped at manifest index `477 / 972`.
Its former PID is not running. The default grid-dynamics follow-up is the
validated 57-experiment Stage B manifest, but it still requires explicit user
selection. Inspect live processes and GPU telemetry rather than assuming state
from documentation.

For any grid128 sweep, comparison, Stage B, active-cell, or shared-horizon task,
read `docs/developer/GRID128_EXPERIMENT_HANDOFF.md` first. The primary root is
`Outputs/GridModel/060126_crop512_grid128_max_v1`.

## Spon Ca Burst workflows

Local source data and labels:

```text
Inputs/Spon Ca Burst/3 hindbrain to tail 488 20ms.tif
Inputs/Spon Ca Burst/3 hindbrain to tail 488 20ms.xlsx
Inputs/Spon Ca Burst/labels/labels_normalized.tsv
Inputs/Spon Ca Burst/labels/label_summary.json
```

Memory-mapped cache:

```text
Outputs/GammaCFAR/spon_ca_burst_3_hindbrain_to_tail_488_20ms/spon_ca_burst_3_hindbrain_to_tail_488_20ms.npy
```

Read the matching workflow before changing or running an experiment:

- `docs/workflows/spon_ca_burst_soma_excitation.md`
- `docs/workflows/spon_ca_burst_learnable_contrast.md`

First guarded CUDA run: `Outputs/LearnableContrast/spon_ca_burst_v1_cuda_guarded`.
Its gate is `do_not_advance`; direct residual is the held-out recall baseline to beat.

Spatiotemporal factorial v2: `Outputs/LearnableContrast/spon_ca_burst_spatiotemporal_factorial_v2`.
All 64 fits completed. Stabilized scaling improved learned recall to `0.2051`;
initialization jitter was secondary; the Kalman spatiotemporal learned cells
scored `0.0`; the gate stopped masked/final stages by design.

Learnable raw-direct v3: `Outputs/LearnableContrast/spon_ca_burst_learnable_direct_tuning_v3`.
All 36 screen fits completed. Every cumulative variant and learning rate tied
frozen direct at `0.6056` mean held-out recall and won `0/4` bursts, so the
conditional confirmation/masked/final stages did not run. The next justified
test is full-field quiet hard-negative mining, not a wider blind parameter sweep.

UI frames are one-based and inclusive; NumPy intervals are zero-based and
half-open. Coordinates use `x=column`, `y=row`. Every new label-driven run must
write a projection overlay in preflight. Unlabeled event pixels remain unknown,
not negative.

## Intent and inverse-control work

Before changing activation benchmarks, left/right intent experiments,
stimulation schemas, action-conditioned dynamics, simulators, or controllers,
read `docs/programs/fish_inverse_control/README.md` and
`docs/research/FISH_INVERSE_CONTROL_ROADMAP.md`. Route focused work through:

- `docs/research/NEURAL_ACTIVATION_DETECTION_ROBUSTNESS.md`
- `docs/research/LEFT_RIGHT_INTENT_AND_CONTROL_PLAN.md`
- `docs/developer/FISH_CONTROL_TOOLING_ROADMAP.md`
- `examples/fish_control_program.example.json`

Use `neurobench program fish-control audit` for dependency, gate, input,
resource, and exact combination-count checks. An audit result never authorizes a
GPU run or stimulation.

Do not treat sparse-positive labels as exhaustive negatives, passive intent as a
causal action effect, or action-free forward dynamics as an invertible
controller. Any command-capable deployment requires an explicit safety envelope,
measured-action logging, interlocks, and a passed shadow-mode gate.

## Repository map

- `neurobench/`: maintained implementation.
- `neurobench/experiments/`: resource-bounded workflows.
- `neurobench/cli/`: lazy-loaded CLI groups.
- `examples/`: small reproducible manifests.
- `docs/workflows/`: scientific workflow contracts.
- `docs/developer/`: detailed handoffs and architecture notes.
- `tests/`: focused regression and workflow tests.
- `Inputs/`: ignored local source data.
- `Outputs/`: ignored generated artifacts.

Use `docs/CODEBASE_NAVIGATION.md` for ownership guidance and `VISIONS.md`
selectively for broader goals.

## Interpretation

Completion is not scientific success. Prefer held-out evidence, comparison to
an appropriate baseline, fold/seed consistency, active-region metrics, and
visual review. Keep known matches, unmatched candidates, and manually accepted
scientific positives distinct.
