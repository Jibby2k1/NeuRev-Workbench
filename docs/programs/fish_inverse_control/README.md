# Fish Intent and Inverse-Control Program

This directory is the canonical entry point for work that begins with neural
activation measurement and ends with safe inverse control of left/right fish
behavior.

## Current decision

The next high-impact work is **not another broad model sweep**. It is:

1. build an exhaustively reviewed activation panel so precision becomes
   measurable;
2. compare frozen direct, CFAR, learned, structured-background, and
   deconvolution pipelines on that panel;
3. audit and freeze causal pre-movement intent trials;
4. run matched spatial-only, temporal-only, and spatiotemporal intent
   ablations;
5. collect measured, safely exploratory stimulation transitions before fitting
   an inverse model;
6. evaluate uncertainty-gated MPC in simulation before shadow mode.

The validated portfolio contains **8 experiments and 68 future compute jobs**.
At the current checkpoint, five experiments are blocked by prerequisites and
three require manual data/annotation work. No compute or GPU job is ready to
launch.

## Start here

| Need | Document or artifact |
|---|---|
| Understand the whole causal program | [Roadmap](../../research/FISH_INVERSE_CONTROL_ROADMAP.md) |
| Understand why detection precision is unresolved | [Activation robustness](../../research/NEURAL_ACTIVATION_DETECTION_ROBUSTNESS.md) |
| Design the left/right and control experiments | [Intent and control plan](../../research/LEFT_RIGHT_INTENT_AND_CONTROL_PLAN.md) |
| See what the repository has already established | [Experiment history](EXPERIMENT_HISTORY.md) |
| See the primary-literature grounding | [Research grounding](RESEARCH_GROUNDING.md) |
| See exact experiment counts and sequencing | [Experiment portfolio](EXPERIMENT_PORTFOLIO.md) |
| Implement packages, schemas, and runners | [Tooling roadmap](../../developer/FISH_CONTROL_TOOLING_ROADMAP.md) |
| Understand the workspace changes | [Workspace organization](WORKSPACE_ORGANIZATION.md) |
| Inspect current machine-readable readiness | [Generated audit](audit/program_audit.md) |
| Open the current portable analysis | [Experiment-program report](../../reports/fish_control_program_v1/report.html) |
| Open the earlier roadmap analysis | [Roadmap report](../../reports/fish_inverse_control_roadmap/report.html) |

## Machine-readable authority

The program definition is
`examples/fish_control_program.example.json` and is validated by
`schemas/fish_control_program.schema.json`.

Run a read-only audit:

```bash
.venv-neurobench/bin/python -m neurobench.cli.main program fish-control audit \
  --manifest examples/fish_control_program.example.json \
  --out-dir docs/programs/fish_inverse_control/audit
```

The audit checks:

- exact factor-grid/repeat counts;
- dependency cycles and missing dependencies;
- stage-gate status;
- required local inputs;
- output collisions;
- CPU, RAM, and GPU memory envelopes;
- live free-disk and active GPU-job limits when path checks are enabled;
- ownership and resumable status of existing `resume_atomic` output roots;
- manual-action and explicit-approval boundaries.

An existing `resume_atomic` root must contain `program_run.json` with schema
version 1, matching program and experiment IDs, and status `failed` or `stopped`.
Missing, mismatched, running, completed, or cancelled markers block readiness.

It does not launch experiments, authorize stimulation, or convert a planned
experiment into scientific evidence.

## Authority and history

- This hub and the machine-readable program manifest are the current planning
  authority for activation-to-control work.
- `AGENTS.md` remains authoritative for operational safety and the stopped
  grid128 sweep.
- `docs/developer/GRID128_EXPERIMENT_HANDOFF.md` remains authoritative for
  grid128-specific history and launch rules.
- Root-level generic plan files are retained as historical records. They should
  not override this program's stage gates.

