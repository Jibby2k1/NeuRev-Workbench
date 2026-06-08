# Neurobench Documentation

This directory is organized around the way a reviewer or developer usually uses
the project.

## Start Here

- [Neuron Workbench](NEURON_WORKBENCH.md): local dashboard setup, autosave,
  workflow home, Data, Pipelines, Experiment Lab, Review, Progress, Report,
  exports, and sharing notes.
- [Resting Video Algorithm Brief](RESTING_VIDEO_ALGORITHM_BRIEF.md): concise
  lab-shareable explanation of the current resting-video detector, waveforms,
  event markers, and caveats.
- [Raw Video To Report Workflow](workflows/raw_video_to_report.md): CPU-only
  end-to-end command path from a raw video through QC, pipeline execution,
  reports, sweeps, and exports.
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

- [Processing Notes](PROCESSING_NOTES.md): current high-pass, local-z, ROI,
  event, discovery, and robustness rationale.
- [SOTA Integrations](SOTA_INTEGRATIONS.md): Suite2p, PMD, OASIS, and related
  external-tool attachment paths.
- [Inverse Dynamics Export](INVERSE_DYNAMICS_EXPORT.md): downstream export
  contract for accepted ROIs/events and behavior alignment.
- [Grid32 Real Data Pilot](case_studies/grid32_real_data_pilot.md): lightweight
  pilot note template for recording real-data template, registration, grid,
  dynamics, and classifier decisions without committing raw videos.

## Developer References

- [Adding A Pipeline Stage](developer/adding_pipeline_stage.md): catalog,
  executor, tests, artifacts, and real-time metadata needed for a new stage.
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
