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
- Before designing, running, or completing any new experiment, read
  `docs/workflows/SCIENTIFIC_AUDIT_OUTPUT_STANDARD.md`.
- Scientific-audit outputs are default-on. A run is not audit-complete until
  expert-only and model-only full-field videos, close-ups and traces, the
  figure/table-only matched comparison, detection metadata, LLM context index,
  and report pass validation. Read the small JSON/CSV index before large media.
  Only an explicit user opt-out with a recorded reason may suppress the set;
  unlabeled runs use a frozen candidate-surrogate Model section and mark Expert
  as not applicable.
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

- `docs/workflows/spon_ca_burst_representation_benchmark.md`
- `docs/workflows/spon_ca_burst_soma_excitation.md`
- `docs/workflows/spon_ca_burst_learnable_contrast.md`
- `docs/workflows/spon_ca_burst_frame_derivatives.md`
- `docs/workflows/spon_ca_burst_pairwise_separation.md`
- `docs/workflows/spon_ca_burst_event_weighted_cs_parzen.md`
- `docs/workflows/spon_ca_burst_pairwise_feature_fusion.md`
- `docs/workflows/spon_ca_burst_hierarchical_parzen_noisy_ica.md`
- `docs/workflows/spon_ca_burst_multiscale_information.md`
- `docs/workflows/spon_ca_burst_scientific_feature_audit.md`
- `docs/research/PAIRWISE_ICA_AS_TEMPORAL_DERIVATIVE.md`
- `docs/developer/PAIRWISE_SOURCE_SEPARATION_IMPLEMENTATION_BRIEF.md`
- `docs/research/DENOISE_THEN_DIFFERENCE.md`
- `docs/developer/LATENT_DYNAMICS_DENOISING_IMPLEMENTATION_BRIEF.md`

The completed v1 representation run evaluated 36 fits. Amplitude PCA rank 8
led fixed-budget neuron ID at 54/79 known matches versus Raw Direct 52/79;
this two-match gain remains provisional. Rank-16 ICA was stable but did not win
at fixed budget; rank-64 ICA did not converge. Read the workflow before follow-up.

First guarded CUDA run: `Outputs/LearnableContrast/spon_ca_burst_v1_cuda_guarded`.
Its gate is `do_not_advance`; direct residual is the held-out recall baseline to beat.

Spatiotemporal factorial v2: `Outputs/LearnableContrast/spon_ca_burst_spatiotemporal_factorial_v2`.
All 64 fits completed. Stabilized scaling improved learned recall to `0.2051`;
initialization jitter was secondary; the Kalman spatiotemporal learned cells
scored `0.0`; the gate stopped masked/final stages by design.

Learnable raw-direct v3: `Outputs/LearnableContrast/spon_ca_burst_learnable_direct_tuning_v3`.
All 36 screen fits completed. Every cumulative variant and learning rate tied
frozen direct at `0.6056` mean held-out recall and won `0/4` bursts, so the
conditional confirmation/masked/final stages did not run.

Morphology-aware CFAR v4-v6 is documented in
`docs/workflows/spon_ca_burst_multihypothesis_cfar.md`. The v6 top-two bounded
gate reached `0.3294` mean cross-fitted recall (27/79 known matches, 53
candidates), versus nested fixed selection `0.3158` (26/79, 59 candidates).
Its `+0.0135` gain missed the predeclared `+0.02` C2 gate, so bounded-kernel C3
did not run. The next justified step is morphology/neighborhood annotation and
exhaustive review of a fixed candidate panel, not another blind sweep.

Causal proposal program:
`Outputs/FrameDifference/spon_ca_burst_causal_proposal_overnight_v1`. All 1,884
evaluations completed; adaptive causal subtraction retained 58/79 known matches
with 488 candidates versus 745 for its causal reference, while bounded CFAR
fusion failed C2. This work is sidelined pending review of its 206-row fixed
queue; do not restart or widen it without explicit selection.

Pairwise separation is implemented for fixed/adaptive difference, InfoMax ICA,
bounded CS-Parzen ICA, and shared-background NMF. Only synthetic/tiny validation
is authorized by default. Preflight requires an explicit new artifact directory;
a full Spon run requires explicit user selection.

Latent-dynamics denoising now has a stable AR(1) numerical reference, strict
preflight, deterministic synthetic fixtures, chunked CPU runner, artifact
contract, and tiny smoke coverage. Read
`docs/workflows/spon_ca_burst_latent_dynamics.md` before changing or running it.
The completed full CPU run is
`Outputs/LatentDynamics/spon_ca_burst_latent_dynamics_v1`: exact Raw Direct
macro recall was `0.6056` (49/79 matches, 232 candidates); offline smoother
amplitude reached `0.6867` (55/79, 320 candidates) and won 4/4 bursts. The
causal filter reached `0.6540` but won only 2/4. C2/C3 confirmation remains
incomplete, and the smoother is not real-time.
The historical `kalman_positive_residual_stack` remains a legacy asymmetric-EMA
baseline, not a full Kalman model. A full Spon or GPU run still requires
explicit user selection; any additional confirmation run requires a new output
root and explicit selection.

Hierarchical Parzen ICA is specified in
`docs/developer/HIERARCHICAL_PARZEN_NOISY_ICA_IMPLEMENTATION_BRIEF.md` and
`docs/research/HIERARCHICAL_PARZEN_NOISY_ICA.md`. The planned hierarchy
reconstructs a background-like component from adjacent-frame Parzen ICA, passes
an amplitude-preserving residual to local noisy Parzen ICA, and keeps structured
neural signal, structured artifact, and qualified measurement noise separate.
Only code, tests, synthetic/semi-synthetic fixtures, preflight, and tiny smoke
work are authorized by the specification. A full Spon/GPU run requires explicit
selection and a new output root.

The guarded Stage-1 generated matrix is
`Outputs/HierarchicalParzenICA/stage1_guarded_synthetic_multiseed_v1`.
All 240 combinations completed and numerical stability passed, but scientific
validity failed. Adaptive gain remains the general reference; constrained
stochastic Parzen improved all five similar-persistence cases but lost the
other 15 signal comparisons, and batch raw feedback was rejected in 57/60
runs. Read
`docs/developer/HIERARCHICAL_PARZEN_STAGE1_GUARDED_SYNTHETIC_REPORT.md`.
Do not begin Stage 2 or semi-synthetic Spon injection until the failed gate is
addressed.

The completed scientific feature audit is
`Outputs/HierarchicalParzenICA/spon_ca_burst_scientific_feature_audit_v1`.
It evaluated 16 maps and 192 lanes. Family-specific `coherence_w15` improved
budget-20 recall in all four bursts (`0.6053` versus carrier `0.5409`) and is
the primary compact confirmation candidate; lagged recurrence is secondary.
The all-family selector was less stable. Read
`docs/research/SPON_CA_BURST_SCIENTIFIC_FEATURE_AUDIT_V1_RESULTS.md` before
widening this search. Precision still requires exhaustive bounded-field labels.

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
