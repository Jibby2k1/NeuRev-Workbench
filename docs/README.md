# Neurobench Documentation

This directory is organized around the way a reviewer or developer usually uses
the project.

## Start Here

- [How to Use the NeuRev Dashboard](HOW_TO_USE_DASHBOARD.md): concise,
  researcher-facing instructions for opening a dataset, inspecting Raw and
  processed evidence, correcting labels, reviewing changes, and publishing an
  immutable annotation revision.
- [Fish Intent and Inverse-Control Program](programs/fish_inverse_control/README.md):
  current authority, evidence history, literature grounding, exact experiment
  portfolio, machine audit, and implementation routing.
- [Neuron Workbench](NEURON_WORKBENCH.md): local dashboard setup, autosave,
  workflow home, Data, Pipelines, Experiment Lab, Review, Progress, Report,
  exports, and sharing notes.
- [Resting Video Algorithm Brief](RESTING_VIDEO_ALGORITHM_BRIEF.md): concise
  lab-shareable explanation of the current resting-video detector, waveforms,
  event markers, and caveats.
- [Raw Video To Report Workflow](workflows/raw_video_to_report.md): CPU-only
  end-to-end command path from a raw video through QC, pipeline execution,
  reports, sweeps, and exports.
- [Spon Ca Burst Soma Excitation](workflows/spon_ca_burst_soma_excitation.md):
  bounded CPU experiment for quiet-baseline dark soma zones, positive CFAR
  excitation, and frozen model transfer after human frame 1900.
- [Spon Ca Burst Learnable Contrast](workflows/spon_ca_burst_learnable_contrast.md):
  weakly supervised, quiet-cross-fitted guarded contrast training and masked-ROI
  discovery evaluation from sparse burst-window point labels.
- [Template Grid Workflow](TEMPLATE_GRID_WORKFLOW.md): manifest, one-reference
  template construction, per-video rigid registration, and 32x32 grid-state
  extraction for the zebrafish left/right/neutral videos.
- [Grid Latent Dynamics](GRID_LATENT_DYNAMICS.md): video-split dynamics arrays,
  persistence baselines, grid autoencoder, latent GRU predictor, and latent-code
  classifier commands.
- Dataset intake starts with `neurobench dataset intake`, which creates a
  metadata-only manifest and readiness report for local files or future public
  sources such as DANDI/NWB and Figshare-style datasets.

## Dashboard Pages

- [Dashboard User Guide](HOW_TO_USE_DASHBOARD.md): the normal review sequence,
  correction workspace screen map, real-data checklist, and troubleshooting.
- [Pipelines](ARCHITECTURE_LAB.md): compare generated runs, build
  pipeline stacks, configure stage parameters, plan sweeps, and understand
  real-time readiness metadata.
- [Data](DATASET_QC.md): inspect raw and intermediate frame outputs in
  pipeline order, diagnose missing outputs, and review dataset/process warnings.
- [Progress](METRICS_AUDIT.md): track review progress, review burden,
  tuning readiness, robustness examples, validation readiness, and adjudication.
- [Annotation Schema](ANNOTATION_SCHEMA.md): annotation JSON fields, reviewer
  provenance, labels, exports, and settings.

## Methods And Integration

- [Fish Inverse-Control Roadmap](research/FISH_INVERSE_CONTROL_ROADMAP.md):
  authoritative checkpoint map from activation measurement through causal
  left/right intent, action-conditioned system identification, and constrained
  control.
- [Neural Activation Detection Robustness](research/NEURAL_ACTIVATION_DETECTION_ROBUSTNESS.md):
  coverage-aware precision/recall benchmark and detector experiment plan.
- [Left/Right Intent And Control Plan](research/LEFT_RIGHT_INTENT_AND_CONTROL_PLAN.md):
  spatial-versus-temporal intent ablations, leakage controls, action schema,
  system-identification ladder, and deployment gates.
- [Portable Fish Control Experiment Program](reports/fish_control_program_v1/report.html):
  current stage-gated portfolio, historical evidence, primary-research grounding,
  workstation envelope, and recommended next steps.
- [Processing Notes](PROCESSING_NOTES.md): current high-pass, local-z, ROI,
  event, discovery, and robustness rationale.
- [SOTA Integrations](SOTA_INTEGRATIONS.md): Suite2p, PMD, OASIS, and related
  external-tool attachment paths.
- [Inverse Dynamics Export](INVERSE_DYNAMICS_EXPORT.md): downstream export
  contract for accepted ROIs/events and behavior alignment.
- [Test And Experiment Report](TEST_AND_EXPERIMENT_REPORT.md): consolidated
  validation, unit-test, sweep, dashboard, and inverse-control-readiness report.
- [Inverse-Control Discussion Brief](INVERSE_CONTROL_DISCUSSION_BRIEF.md):
  historical context and prompts; use the fish-control program hub for current
  evidence and stage gates.
- [Grid32 Real Data Pilot](case_studies/grid32_real_data_pilot.md): lightweight
  pilot note template for recording real-data template, registration, grid,
  dynamics, and classifier decisions without committing raw videos.

## Developer References

- [Codebase Navigation](CODEBASE_NAVIGATION.md): current package map, task
  routes, entry points, hotspots, and agent notes.
- [Codebase Audit](CODEBASE_AUDIT.md): broader maintainability and
  LLM/human-navigability audit with staged refactor plan.
- [Dashboard Code Audit](DASHBOARD_CODE_AUDIT.md): dashboard families,
  organization risks, and dashboard-specific UX/efficiency plan.
- [Workbench Video and Catalog Refactor](developer/WORKBENCH_VIDEO_CATALOG_REFACTOR.md):
  single-canvas annotation layout, unified App/LLM video lookup, preservation
  guardrails, and prioritized bloat reduction.
- [Adding A Pipeline Stage](developer/adding_pipeline_stage.md): catalog,
  executor, tests, artifacts, and real-time metadata needed for a new stage.
- [Fish Control Tooling Roadmap](developer/FISH_CONTROL_TOOLING_ROADMAP.md):
  proposed benchmark, intent, action-conditioned dynamics, simulator, MPC, and
  stage-gate package boundaries.
- [API Reference](API_REFERENCE.md): generated Python module/class/function
  reference.
- [Long-Term Plan](plan.md): project roadmap and broader research directions.

## Recommended Reading Order

1. Read the [Resting Video Algorithm Brief](RESTING_VIDEO_ALGORITHM_BRIEF.md)
   before presenting the current detector to collaborators.
2. Use [Neuron Workbench](NEURON_WORKBENCH.md) to run or share the dashboard.
3. Use [Pipelines](ARCHITECTURE_LAB.md) and
   [Data](DATASET_QC.md) when changing parameters or comparing runs.
4. Use [Progress](METRICS_AUDIT.md) before tuning thresholds or exporting
   reviewed data.
5. Use [Adding A Pipeline Stage](developer/adding_pipeline_stage.md) when a new
   algorithm needs to become a first-class dashboard component.
6. Use [Template Grid Workflow](TEMPLATE_GRID_WORKFLOW.md) and
   [Grid Latent Dynamics](GRID_LATENT_DYNAMICS.md) for the current
   template-aligned 32x32 grid experiments.
7. Use [Codebase Navigation](CODEBASE_NAVIGATION.md) before broad edits or when
   handing the repository to another coding agent.
8. Use the [Fish Inverse-Control Roadmap](research/FISH_INVERSE_CONTROL_ROADMAP.md)
   as the current authority before designing intent or action-conditioned
   control experiments; use the older
   [Inverse-Control Discussion Brief](INVERSE_CONTROL_DISCUSSION_BRIEF.md) for
   historical meeting context.
