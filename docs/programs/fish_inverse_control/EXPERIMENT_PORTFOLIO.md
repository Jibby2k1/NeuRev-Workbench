# High-Impact Experiment Portfolio

The canonical machine-readable definition is
`examples/fish_control_program.example.json`. This document explains why its
experiment count and ordering are intentionally small.

## Portfolio summary

| Priority | ID | Purpose | Compute jobs | Current readiness |
|---:|---|---|---:|---|
| 1 | `fc00` | exhaustive activation annotation panel | 0 | manual action |
| 2 | `fc01` | frozen detector tournament | 6 | blocked by `fc00` |
| 3 | `fc02` | structured-background + PU refinement | 12 | blocked and approval-gated |
| 4 | `fc03` | causal intent dataset readiness | 0 | manual action |
| 5 | `fc04` | spatial/temporal intent ablation | 24 | blocked by intent data |
| 6 | `fc05` | stimulation/action logging readiness | 0 | manual action |
| 7 | `fc06` | action-conditioned linear system ID | 8 | blocked by action data and approval |
| 8 | `fc07` | uncertainty-gated MPC simulation | 18 | blocked by system ID and approval |
|  | **Total** | 8 experiments | **68** | no compute job currently ready |

The three zero-job entries are not missing designs. They are human/data
checkpoints that prevent compute from laundering an invalid dataset into a
model result.

## FC00 — precision-enabling annotation

Target at least 120 spatial-temporal tiles stratified across:

- active events;
- quiet periods;
- high motion;
- bright/structured background;
- overlapping cells;
- edges and low-SNR regions;
- disagreements among current detectors.

Every tile records `exhaustive`, `candidate_review`, or `sparse_positive`
coverage. Use two reviewers on a subset and adjudicate disagreements. This is
the highest-information action because it makes the program's missing primary
metric—precision—identifiable.

## FC01 — six frozen evaluations

Compare:

1. raw direct;
2. Gamma CFAR;
3. learned contrast v2;
4. direct-tuned v3;
5. CaImAn/CNMF;
6. annular structured background plus deconvolution.

Threshold curves are computed within each evaluation rather than multiplied
into separate training jobs. Select operating points on validation groups and
open the sealed test panel only after method and threshold logic are frozen.

## FC02 — twelve confirmatory GPU fits

Factor grid:

- architecture: amplitude + annular background, amplitude + soft CFAR;
- loss: coverage-masked focal, non-negative PU;
- seeds: 3.

This is 2 × 2 × 3 = 12 fits. It directly tests the two plausible explanations
for v3's tie: background structure and sparse-positive risk. It preserves raw
amplitude and stops after the screen if neither branch improves validation
precision/recall.

## FC03 — causal intent-data audit

Before modeling:

- review the 1,118-frame manual interval, reversed range, and workbook title;
- derive trial-level movement onset and direction;
- record timing uncertainty;
- define causal pre-onset windows;
- freeze held-out fish/session groups;
- quantify class balance and missingness.

If these cannot be satisfied, the correct result is a new acquisition plan, not
a larger classifier.

## FC04 — twenty-four matched intent cells

Representations:

1. behavior only;
2. population amount;
3. spatial laterality;
4. coordinate-free temporal history;
5. spatiotemporal neural activity;
6. spatiotemporal activity plus behavior.

Windows:

1. -500 to -250 ms;
2. -250 to -100 ms;
3. -100 to -40 ms;
4. post-onset leakage diagnostic.

Six representations × four windows = 24 primary cells. Start with regularized
linear models. Nonlinear confirmation is a later gate, not part of the 24.

Required controls:

- coordinate shuffle and left/right mirror;
- temporal circular shift;
- label permutation;
- activity-count matching;
- stimulus-only and behavior-history-only baselines;
- post-onset diagnostic.

## FC05 — action-data audit

Freeze requested and measured action, target, amplitude, duration, waveform,
command/hardware timestamps, latency, sham, clipping, interlock, recent-action,
and explicit no-action fields. Missing action is never recoded as no-action.

## FC06 — eight controlled system-ID cells

Models:

1. action-free ridge;
2. state-action ridge;
3. controlled linear state-space;
4. input preferential subspace.

Targets:

1. left/right response;
2. next neural state.

Four models × two targets = 8 cells. Withhold fish/session and stimulation
levels. Add no-action, sham, contralateral, off-target, and action-permutation
controls.

## FC07 — eighteen simulation jobs

Transition models:

1. controlled linear ensemble;
2. probabilistic neural ensemble.

Controllers:

1. fixed safe;
2. randomized safe;
3. uncertainty-gated MPC.

Two models × three controllers × three repeats = 18 jobs. Test latency,
dropout, registration drift, action saturation, and distribution shift. This
stage cannot issue real stimulation commands.

## Hardware schedule

Hardware snapshot:

- Intel Core i9-14900K, 32 logical CPUs;
- 78 GiB RAM;
- RTX 4070 SUPER, 12,282 MiB GPU memory;
- approximately 2 TiB free disk at planning time.

Default experiment envelope:

- at most 16 CPU threads;
- at most 48 GiB RAM;
- one GPU-heavy job at a time;
- GPU target 8,500 MiB and hard limit 9,600 MiB;
- reserve 2,200 MiB for the desktop/runtime;
- checkpoint atomically and resume by completed job IDs;
- batch 2 or lower for grid-like recurrent models, based on the OOM history.

The machine was already running other desktop/test workloads during planning,
so no compute job was launched.

## Decision order

FC00, FC03, and FC05 may progress as parallel manual/data workstreams. Compute
remains sequential inside each branch:

`FC00 → FC01 → FC02`  
`FC03 → FC04`  
`FC05 → FC06 → FC07`

The control branch may develop schemas and synthetic tests early, but it cannot
claim readiness until its gates pass.

