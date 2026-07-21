# Test And Experiment Report

This report summarizes the validation work and model experiments completed for
Neurobench through the grid-dynamics workflow. It is intended to support design
decisions for a later inverse-control system: what is already tested, what the
experiments show, and what cannot yet be inferred from the current evidence.

## Executive Summary

The codebase now has broad automated coverage across data ingestion, annotation,
pipeline execution, workbench behavior, metrics, reporting, grid-state
extraction, template registration, latent dynamics, and dashboard visualization.
The current source tree contains 96 test files with 395 collected test functions.

There are two important validation facts:

- The previous overnight validation soak passed: 8,227 command runs, 0 failures,
  1,372 completed cycles, and 316 tests collected at that time.
- A current post-grid local run with `.venv-neurobench/bin/python -m pytest -q`
  produced 389 passed, 4 failed, 2 skipped, 76 subtests passed. The failures are
  listed under "Current Open Test Failures".

The grid-dynamics experiments provide useful forward-prediction evidence. The
strongest and most consistent result is that temporal convolutional predictors
on template-aligned grid states beat split-aware persistence on held-out fish
videos for many configurations. The 128-grid high-resolution temporal-CNN sweep
is the cleanest positive result: 31 of 32 test rows improved over persistence.
The 32-grid large sweep is broader and noisier: 850 of 1,212 rows improved on
test, with the best test improvement around 2.29e-4 MSE.

For inverse control, these results support using the grid/latent models as
candidate forward models, priors, and state estimators. They do not yet establish
causal controllability. The next stage must define an action space, collect or
simulate interventions, and validate closed-loop control objectives against
behavior-aligned outputs.

## Validation Layers

### Unit And Contract Tests

The current test inventory covers 96 files and 395 test functions. The tests are
not just syntax checks; they exercise model contracts, schemas, CLI behavior,
pipeline runners, artifact manifests, report generation, browser assets, and
scientific metrics.

| Area | Representative test files | What is validated |
|---|---|---|
| Data ingestion and video handling | `test_video_loading.py`, `test_video_manifest.py`, `test_crop_video.py`, `test_dataset_intake.py`, `test_dataset_qc.py` | TIFF/NPY/MP4 handling, frame-first shape conventions, crop semantics, dataset manifests, preflight and QC fields. |
| Pipeline execution | `test_pipeline_executor.py`, `test_pipeline_catalog.py`, `test_pipeline_sweeps.py`, `test_pipeline_batch.py`, `test_pipeline_sweep_execution.py` | Stage registry, dry-run normalization, artifact dependency checks, synthetic end-to-end execution, parameter sweeps, success/failure summaries. |
| Filtering and detection | `test_cfar_algorithm.py`, `test_motion_algorithm.py`, `test_prepare_gamma_cfar_workbench_run.py`, `test_cfar_contrast_maps.py` | Gamma CFAR behavior, high-pass/local-z preprocessing, component filters, green-excess pipelines, motion correction, contrast maps. |
| Annotation and review | `test_annotations_model.py`, `test_review_data_model.py`, `test_review_batches.py`, `test_reviewer_provenance.py`, `test_annotation_agreement.py`, `test_compare_annotations_cli.py` | Review schemas, provenance, agreement reports, active-learning batch construction, reviewer-state exports. |
| Export and inverse-dynamics prep | `test_annotation_exports.py`, `test_inverse_dynamics_export.py`, `test_export_bundle_model.py`, `test_behavior_alignment.py` | Control-ready ROI/event selection, trace/event TSV outputs, alignment metadata, bundle manifests and checksums. |
| Metrics and reporting | `test_object_metrics.py`, `test_event_metrics.py`, `test_annotation_metrics.py`, `test_metrics_report_builder.py`, `test_run_comparison_metrics.py`, `test_population_summaries.py` | Object matching, event quality, annotation metrics, report rendering, population/run summaries. |
| Workbench app | `test_workbench_assets.py`, `test_workbench_builder.py`, `test_workbench_browser_smoke.py`, `test_workbench_materialize.py`, `test_workbench_server.py`, `test_workbench_structure.py` | Static asset packaging, browser smoke contract, materialization, server endpoints, workbench HTML/JS structure. |
| Template/grid workflow | `test_grid_model.py`, `test_grid_state_extraction.py`, `test_template_building.py`, `test_template_registration.py`, `test_template_grid_pipeline_stages.py`, `test_template_grid_safety.py` | Grid specs, template creation, rigid registration, safety checks, grid-state artifacts. |
| Dynamics models | `test_dynamics_baselines.py`, `test_grid_autoencoder.py`, `test_latent_rnn.py`, `test_latent_classifier.py`, `test_dynamics_sweep.py`, `test_dynamics_comparison.py`, `test_scalable_temporal_cnn.py`, `test_overnight_sweep.py` | Persistence and array baselines, autoencoder runs, latent GRU/transformer contracts, classifier summaries, temporal-CNN/ConvGRU/ConvLSTM forward shapes and sweep summaries. |
| CLI and docs | `test_cli_main.py`, `test_cli_report.py`, `test_api_reference_generation.py`, `test_docs_workflows.py`, `test_developer_docs.py`, `test_environment_setup.py` | Console script behavior, help text, JSON output contracts, generated API reference consistency, docs/environment synchronization. |
| Runtime and logging | `test_device_abstraction.py`, `test_gpu_smoke.py`, `test_realtime_latency.py`, `test_realtime_stream.py`, `test_online_stage.py`, `test_run_logger.py` | CPU/GPU fallback behavior, streaming and latency accounting, online event detection, run logs and JSONL events. |

### Overnight Soak

The most important historical stability run is:

`Outputs/ValidationRuns/validation_soak_20260518_011311/experiment_brief.md`

Summary:

- Started: 2026-05-18 05:13:11 UTC
- Finished: 2026-05-18 14:13:11 UTC
- Duration: 9 hours
- Command runs: 8,227
- Passed command runs: 8,227
- Failed command runs: 0
- Completed cycles: 1,372
- Test inventory at start: 316 tests

Repeated suites:

| Suite | Runs | Passes | Failures | Purpose |
|---|---:|---:|---:|---|
| `collect_tests` | 1,372 | 1,372 | 0 | Test collection/import stability. |
| `full_pytest` | 1,371 | 1,371 | 0 | Full CPU-safe codebase contract. |
| `pipeline_and_device` | 1,371 | 1,371 | 0 | Pipeline runners, sweeps, device fallback. |
| `workbench_process_lab` | 1,371 | 1,371 | 0 | Workbench, server, asset, and intermediate-export paths. |
| `science_metrics_exports` | 1,371 | 1,371 | 0 | Metrics, reports, annotation exports, inverse-dynamics export. |
| `docs_api_setup` | 1,371 | 1,371 | 0 | Docs, schemas, generated API reference, environments. |

Interpretation: at that point in the project, the scientific and workbench
contracts were stable under repeated execution. The grid-dynamics work landed
after this soak, so the newer grid tests and model outputs need their own full
soak before being treated with the same confidence.

### Current Local Test Run

Command:

```bash
.venv-neurobench/bin/python -m pytest -q
```

Result:

- 389 passed
- 4 failed
- 2 skipped
- 7 warnings
- 76 subtests passed
- Runtime: 33.98 seconds

Current open failures:

| Test | Failure mode | Likely meaning |
|---|---|---|
| `test_api_reference_generation.py::ApiReferenceGenerationTests::test_checked_in_reference_matches_generator_output` | Checked-in API reference differs from generated output. | Documentation needs regeneration after API/signature changes. |
| `test_cfar_contrast_maps.py::CfarContrastMapTests::test_cli_attaches_shared_contrast_artifacts_to_runs` | CLI stdout was empty where JSON was expected. | CLI path or subprocess environment is not emitting the expected JSON contract. |
| `test_cli_main.py::CliMainTests::test_cli_run_dry_run_json_example` | CLI stdout was empty where JSON was expected. | Same class of CLI-output contract issue. |
| `test_cli_report.py::CliReportTests::test_cli_report_compare_requires_two_runs` | Return code was `-11` instead of expected `1`. | Possible segmentation fault or native-library crash in that CLI subprocess path. |

Interpretation: the current tree is mostly healthy but should not be considered
fully green until these failures are fixed. The failures are concentrated around
documentation generation and CLI subprocess behavior, not around the core
grid-dynamics model math. Still, CLI crashes matter for reproducibility and
must be closed before a new release or a trusted automation workflow.

## Experiment Families

### 050126 Calcium Event/ROI Pipeline

Artifacts under:

- `Outputs/HighPass`
- `Outputs/CandidateEventPipeline`
- `Outputs/TemporalCandidateScoring`
- `Outputs/EventPreservingNoiseSuppression`
- `Outputs/NeuronReview`

Purpose:

1. Ingest calcium imaging video.
2. Apply temporal high-pass filtering and local positive-z transforms.
3. Detect candidate events and components using CFAR and related filters.
4. Build review workbench inputs.
5. Export accepted/control-ready traces and events for later inverse-dynamics
   analysis.

What was validated:

- Candidate and event export structures.
- ROI/event review states.
- Trace/event selection logic for inverse-dynamics export.
- Metrics and report generation for candidate quality.

Relevance to inverse control:

- This layer provides a reviewed neural feature table.
- It is good for selecting reliable neural states and event labels.
- It does not yet provide behavioral actuation or causal intervention data.

### 051626 Template-Aligned Grid Smoke Workflow

Artifacts under:

- `Outputs/GridModel/051626`

Purpose:

1. Build a 32x32 grid template.
2. Register resting/left/right videos to a common frame.
3. Extract grid-region traces.
4. Train smoke autoencoders.
5. Train latent GRU predictors.
6. Run an initial latent dynamics sweep.

Representative outputs:

- `grid/grid_spec_32x32.json`
- `registration_translation/*/registration_result.json`
- `dynamics_smoke/dynamics_dataset.json`
- `models/autoencoder_*`
- `models/latent_gru_*`
- `sweeps/latent_dynamics_norm_sweep_v1/sweep_results.tsv`

What this established:

- Template registration and grid extraction can create a common state space
  across videos.
- Latent autoencoders and latent recurrent predictors can be trained on the
  resulting grid-state arrays.
- This was a feasibility/smoke workflow, not the final best model family.

Relevance to inverse control:

- Establishes the state representation `x_t` and optional latent `z_t`.
- Suggests a route to low-dimensional planning.
- Needs behavior alignment and action labels before it can be used for inverse
  control.

### 060126 Crop512 Grid32 Workflow

Artifacts under:

- `Outputs/GridModel/060126_crop512_grid32_v1`

Input videos:

- 11 cropped videos from `Inputs/060126`
- Labels: left, right, rest/resting
- Splits are video-level and stratified by label.

Processing:

1. Crop input videos to a common 512x512 region.
2. Build/register template-aligned projections.
3. Extract 32x32 grid states.
4. Build datasets with window length 8 and different horizons/strides.
5. Train autoencoders for some latent experiments.
6. Train direct pixel-grid dynamics models and latent models.
7. Render dashboard clips comparing target, model prediction, persistence, and
   absolute error.

Important split:

- Test videos include held-out examples such as `5 right`, `7 rest`, and
  `8 left`.
- The split unit is the video, not random frames. This matters because frame
  randomization would overstate generalization.

#### Restricted Sweep

Path:

`Outputs/GridModel/060126_crop512_grid32_v1/cropped32_restricted_sweep_v1/sweep_summary.tsv`

Summary:

- Rows: 177
- Positive test improvement over persistence: 113/177
- Best test improvement: 0.000197904
- Positive validation improvement: 140/177
- Best validation improvement: 0.000722776

Top test models:

| Rank | Model | Family | Dataset | Test improvement |
|---:|---|---|---|---:|
| 1 | `temporal_cnn_w8_s1_h50_residual_mse_hc32_l4_lr1em04_rs0p1000_e35_s7` | temporal CNN | `w8_s1_h50` | 0.000197904 |
| 2 | `temporal_cnn_w8_s1_h50_motion_weighted_huber_hc32_l4_lr1em04_rs0p1000_e35_s7` | temporal CNN | `w8_s1_h50` | 0.000196572 |
| 3 | `temporal_cnn_w8_s1_h50_residual_mse_hc32_l4_lr1em04_rs0p0500_e35_s7` | temporal CNN | `w8_s1_h50` | 0.000182180 |
| 4 | `unet_convgru_w8_s1_h50_motion_weighted_huber_hc32_lr1em04_rs0p1000_e35_s7` | U-Net ConvGRU | `w8_s1_h50` | 0.000174105 |
| 5 | `temporal_cnn_w8_s1_h50_motion_weighted_huber_hc32_l4_lr1em04_rs0p0500_e35_s7` | temporal CNN | `w8_s1_h50` | 0.000171835 |

Interpretation:

- Temporal CNNs were already strong in the restricted search.
- The 50-frame horizon dataset was consistently useful.
- Motion-weighted Huber and residual MSE both produced competitive results.

#### Large Sweep

Path:

`Outputs/GridModel/060126_crop512_grid32_v1/cropped32_large_sweep_v1/sweep_summary.tsv`

Summary:

- Rows: 1,212
- Positive test improvement over persistence: 850/1,212
- Best test improvement: 0.000229475
- Median test improvement: 0.0000468768
- Positive validation improvement: 1,044/1,212
- Best validation improvement: 0.000761287
- Median validation improvement: 0.000263850

Top test models:

| Rank | Model | Family | Dataset | Test improvement |
|---:|---|---|---|---:|
| 1 | `temporal_cnn_w8_s1_h50_residual_mse_hc32_l6_lr1em04_rs0p1000_e50_s13` | temporal CNN | `w8_s1_h50` | 0.000229475 |
| 2 | `temporal_cnn_w8_s1_h50_residual_mse_hc32_l4_lr1em04_rs0p1000_e50_s13` | temporal CNN | `w8_s1_h50` | 0.000224478 |
| 3 | `temporal_cnn_w8_s1_h50_residual_mse_hc64_l4_lr1em04_rs0p1000_e50_s7` | temporal CNN | `w8_s1_h50` | 0.000219380 |
| 4 | `temporal_cnn_w8_s1_h50_motion_weighted_huber_hc32_l6_lr1em04_rs0p1000_e50_s7` | temporal CNN | `w8_s1_h50` | 0.000216267 |
| 5 | `temporal_cnn_w8_s1_h50_motion_weighted_huber_hc64_l4_lr1em04_rs0p1000_e50_s7` | temporal CNN | `w8_s1_h50` | 0.000216051 |

Interpretation:

- The best held-out models are direct temporal CNN pixel/grid predictors.
- The 8-input-frame, stride-1, 50-frame-horizon dataset dominates the top test
  group.
- Validation and test rankings are not identical. Some models with excellent
  validation performance underperform on test, so selection should use
  held-out-video test behavior and visual diagnostics together.

Dashboard videos:

- `comparison_dashboard.html` compares the top selected models.
- The clip collection has 66 MP4 files: 3 selected top models x 11 input videos x
  2 views.
- Each clip contains four panels: target frame, model prediction shifted by the
  forecast horizon, persistence prediction shifted by the same horizon, and
  lag-compensated absolute error.
- The absolute-error panel is display-normalized separately from the intensity
  panels, so bright error means "large relative to this clip's error
  distribution", not necessarily large relative to the full signal range.

### 060126 High-Resolution Temporal-CNN Workflow

Artifacts under:

- `Outputs/GridModel/060126_crop512_highres_temporalcnn_v1`

Purpose:

Evaluate scalable temporal-CNN predictors at higher grid resolutions, including
64-grid and 128-grid variants.

#### Grid64 Scalable Temporal-CNN Sweep

Path:

`Outputs/GridModel/060126_crop512_highres_temporalcnn_v1/sweeps/grid64_scalable_temporalcnn_v1/sweep_summary.tsv`

Summary:

- Rows: 720
- Positive test improvement: 332/720
- Best test improvement: 0.000231597
- Positive validation improvement: 679/720
- Best validation improvement: 0.000779433

Top test model:

`scalable_tcnn_w8_s1_h50_stack_tiny_24x3_residual_mse_lr1em04_rs0p1000_e50_s13`

Interpretation:

- Grid64 can match or slightly exceed the best grid32 test improvement.
- The broader sweep showed weaker median test behavior than grid32, suggesting
  higher resolution increases capacity and noise together.

#### Grid128 Scalable Temporal-CNN Sweep

Path:

`Outputs/GridModel/060126_crop512_highres_temporalcnn_v1/sweeps/grid128_scalable_temporalcnn_v1/sweep_summary.tsv`

Summary:

- Rows: 32
- Positive test improvement: 31/32
- Best test improvement: 0.000321632
- Median test improvement: 0.000256264
- Positive validation improvement: 32/32
- Best validation improvement: 0.000583569

Top test models:

| Rank | Model | Dataset | Test improvement |
|---:|---|---|---:|
| 1 | `scalable_tcnn_w8_s1_h25_stack_small_32x4_motion_weighted_huber_lr1em04_rs0p1000_e50_s7` | `w8_s1_h25` | 0.000321632 |
| 2 | `scalable_tcnn_w8_s1_h25_stack_tiny_24x3_motion_weighted_huber_lr1em04_rs0p0500_e50_s7` | `w8_s1_h25` | 0.000314425 |
| 3 | `scalable_tcnn_w8_s1_h25_stack_tiny_24x3_motion_weighted_huber_lr1em04_rs0p1000_e50_s7` | `w8_s1_h25` | 0.000314013 |
| 4 | `scalable_tcnn_w8_s1_h25_stack_wide_48x4_motion_weighted_huber_lr1em04_rs0p0500_e50_s7` | `w8_s1_h25` | 0.000308508 |
| 5 | `scalable_tcnn_w8_s1_h25_stack_wide_48x4_motion_weighted_huber_lr1em04_rs0p1000_e50_s7` | `w8_s1_h25` | 0.000304955 |

Interpretation:

- Grid128 is currently the most compelling forward-prediction result by
  consistency.
- The tested horizon here is 25 frames, not 50 frames. For control, this may be
  advantageous because shorter horizons reduce compounding uncertainty.
- This result should be expanded before final model choice because only 32 rows
  were tested.

## What The Tests Establish For Inverse Control

The validated methods establish the following components:

1. A state-construction pipeline exists.
   Raw imaging videos can be cropped, registered, and transformed into
   template-aligned grid states.

2. A held-out-video prediction protocol exists.
   Models are evaluated by video-level splits rather than random frame splits.

3. Persistence is a meaningful baseline.
   Improvements are reported as MSE reduction relative to copying the last input
   frame forward.

4. Several model families can predict future grid states.
   Direct temporal CNNs and scalable temporal CNNs are the strongest current
   candidates.

5. Review/export paths exist for control-ready neural features.
   Accepted ROIs, event features, traces, and behavior-alignment metadata can be
   exported.

6. Visual diagnostics exist.
   Dashboard videos show target, shifted model prediction, shifted persistence,
   and lag-compensated error.

These are necessary foundations for inverse control, but they are not sufficient
for closed-loop control.

## What The Tests Do Not Yet Establish

The current work does not yet prove:

- Causality: the models are observational predictors, not intervention models.
- Controllability: no action variable has been learned or validated.
- Closed-loop stability: no controller has been run against real or simulated
  interventions.
- Behavior causation: left/right/rest labels are video-level labels, not
  frame-resolved causal behavior outputs.
- Actuator constraints: no model currently represents stimulation amplitude,
  spatial target, timing, safety envelope, or latency.
- Cross-session robustness: the best models are promising on the current split,
  but more days/fish/sessions are needed before general deployment.

## Inverse-Control Design Implications

The safest near-term inverse-control path is:

1. Use grid128 or grid32 temporal CNNs as forward dynamics models.
2. Add an explicit action/input channel `u_t` once interventions or planned
   stimuli are available.
3. Define the controlled target as either a future grid state, a latent state,
   or a behavior proxy such as left/right probability.
4. Use model predictive control in simulation first:
   repeatedly optimize a short future action sequence under the learned forward
   model, apply the first action, then re-estimate state.
5. Keep a persistence and no-control baseline in every evaluation.
6. Require behavior-aligned validation before claiming control of fish behavior.

Recommended first controller objective:

```text
minimize_u  ||phi(x_{t+H}) - target||^2
          + lambda_action ||u||^2
          + lambda_smooth ||u_t - u_{t-1}||^2
          + lambda_risk artifact_or_saturation_penalty
```

Where:

- `x_t` is the observed grid state.
- `phi` is either identity, a learned latent encoder, or a behavior decoder.
- `u_t` is the stimulation/perturbation/control action.
- `H` should start short, probably 10-25 frames, because the grid128 sweep was
  strongest at 25 frames.

## Recommended Next Tests

Before using these methods for inverse control, add:

1. Green current suite
   Fix the 4 current failures and run the full suite again.

2. Post-grid soak
   Repeat the overnight validation with the 395-test inventory.

3. Dashboard numerical regression
   Add tests that parse rendered selector metadata and verify absolute-error
   alignment, normalization metadata, and panel order.

4. Behavior alignment validation
   Add tests for frame-level behavior traces, sync offsets, and video-to-behavior
   clock drift.

5. Action-conditioned dataset schema
   Add `u_t` action channels, actuator metadata, action latency, and stimulation
   safety constraints to dataset manifests.

6. Causal split protocol
   Evaluate intervention-held-out sessions, not just video-held-out passive
   recordings.

7. Closed-loop simulation tests
   Build a simulated controller using a learned forward model and require it to
   beat persistence/no-control on target-reaching objectives.

8. Model uncertainty tests
   Evaluate ensembles or dropout uncertainty and reject control actions when
   predictive uncertainty is high.

9. Safety tests
   Add hard constraints for stimulation duration, spatial masks, saturation, and
   out-of-distribution states.

## Appendix: Test Inventory By File

| Test file | Count |
|---|---:|
| `test_active_learning.py` | 4 |
| `test_annotation_agreement.py` | 4 |
| `test_annotation_exports.py` | 5 |
| `test_annotation_metrics.py` | 2 |
| `test_annotations_model.py` | 6 |
| `test_api_reference_generation.py` | 4 |
| `test_architecture_runs.py` | 5 |
| `test_artifact_store.py` | 5 |
| `test_attach_pipeline_intermediates.py` | 2 |
| `test_behavior_alignment.py` | 5 |
| `test_candidate_clustering.py` | 3 |
| `test_candidate_feature_table.py` | 3 |
| `test_candidate_ranking.py` | 3 |
| `test_cfar_algorithm.py` | 3 |
| `test_cfar_contrast_maps.py` | 2 |
| `test_chunked_processing.py` | 3 |
| `test_cli_main.py` | 19 |
| `test_cli_report.py` | 3 |
| `test_compare_annotations_cli.py` | 2 |
| `test_correctness_foundations.py` | 14 |
| `test_crop_video.py` | 2 |
| `test_dataset_intake.py` | 2 |
| `test_dataset_qc.py` | 4 |
| `test_developer_docs.py` | 4 |
| `test_device_abstraction.py` | 5 |
| `test_docs_workflows.py` | 5 |
| `test_dynamics_baselines.py` | 2 |
| `test_dynamics_comparison.py` | 1 |
| `test_dynamics_sweep.py` | 1 |
| `test_environment_setup.py` | 3 |
| `test_event_metrics.py` | 4 |
| `test_export_bundle_model.py` | 7 |
| `test_gamma_cfar_sweep_report.py` | 2 |
| `test_gpu_smoke.py` | 2 |
| `test_grid_autoencoder.py` | 1 |
| `test_grid_dynamics_dataset.py` | 3 |
| `test_grid_model.py` | 1 |
| `test_grid_search_summary.py` | 2 |
| `test_grid_state_extraction.py` | 2 |
| `test_input_checksums.py` | 6 |
| `test_intermediate_export.py` | 2 |
| `test_inverse_dynamics_export.py` | 5 |
| `test_latent_classifier.py` | 1 |
| `test_latent_rnn.py` | 5 |
| `test_llm_architecture_planning.py` | 7 |
| `test_manifests_annotations.py` | 4 |
| `test_metrics_report_builder.py` | 3 |
| `test_metrics_report_model.py` | 5 |
| `test_metrics_report_render.py` | 6 |
| `test_models.py` | 8 |
| `test_motion_algorithm.py` | 3 |
| `test_object_metrics.py` | 4 |
| `test_online_stage.py` | 3 |
| `test_overnight_sweep.py` | 11 |
| `test_parameter_hashing.py` | 6 |
| `test_pipeline_batch.py` | 1 |
| `test_pipeline_catalog.py` | 6 |
| `test_pipeline_executor.py` | 12 |
| `test_pipeline_run_models.py` | 7 |
| `test_pipeline_runner_index.py` | 2 |
| `test_pipeline_sweep_execution.py` | 3 |
| `test_pipeline_sweeps.py` | 6 |
| `test_plugin_registry.py` | 5 |
| `test_population_summaries.py` | 5 |
| `test_prepare_gamma_cfar_workbench_run.py` | 16 |
| `test_proposal_analysis.py` | 2 |
| `test_realtime_latency.py` | 3 |
| `test_realtime_stream.py` | 4 |
| `test_registration_model.py` | 1 |
| `test_review_batches.py` | 7 |
| `test_review_data_model.py` | 4 |
| `test_review_roi_sidecars.py` | 1 |
| `test_reviewer_provenance.py` | 7 |
| `test_run_comparison_metrics.py` | 3 |
| `test_run_comparison_report.py` | 3 |
| `test_run_logger.py` | 4 |
| `test_scalable_temporal_cnn.py` | 4 |
| `test_schema_validation.py` | 7 |
| `test_sweep_evidence_report.py` | 2 |
| `test_sweep_visuals.py` | 1 |
| `test_synthetic_data.py` | 5 |
| `test_synthetic_fish.py` | 1 |
| `test_template_building.py` | 2 |
| `test_template_grid_pipeline_stages.py` | 1 |
| `test_template_grid_safety.py` | 3 |
| `test_template_models.py` | 1 |
| `test_template_registration.py` | 2 |
| `test_video_loading.py` | 3 |
| `test_video_manifest.py` | 2 |
| `test_video_store.py` | 4 |
| `test_workbench_assets.py` | 3 |
| `test_workbench_browser_smoke.py` | 1 |
| `test_workbench_builder.py` | 5 |
| `test_workbench_materialize.py` | 2 |
| `test_workbench_server.py` | 11 |
| `test_workbench_structure.py` | 4 |
