# Processing Notes

This repository now has two complementary workflows:

- the original Python grid-search pipeline for Gamma/Kalman-MCC filtering and
  CFAR evaluation
- the Fiji/Groovy neuron-review workflow for practical inspection, ROI review,
  trace denoising, and event annotation

For a concise peer-facing explanation of the current resting-video result,
algorithm details, waveform colors, event markers, and hyperparameter status,
see [RESTING_VIDEO_ALGORITHM_BRIEF.md](RESTING_VIDEO_ALGORITHM_BRIEF.md).

## Current Fiji/Groovy Workflow

The current calcium-video workflow uses the following scripts in order:

1. `tools/temporal_highpass_gaussian.ijm`
   - subtracts a temporal Gaussian background from the raw TIFF
   - writes signed float high-pass stacks for several temporal scales
2. `tools/event_preserving_noise_suppression.groovy`
   - applies median correction, local positive z scoring, and support filters
   - useful as an exploratory baseline
3. `tools/candidate_event_pipeline.groovy`
   - builds permissive spatial candidates using robust local z scores
   - outputs masks, labels, and component tables
4. `tools/temporal_candidate_scoring.groovy`
   - scores candidate components using raw trace support, high-pass support,
     nearby-frame support, and collapse penalties
5. `tools/generate_neuron_review_app.groovy`
   - turns aggregate evidence into stable ROI candidates
   - writes evidence maps for missed-neuron discovery
   - ranks uncovered candidate suggestions by combined evidence
   - extracts raw and local-background-corrected traces
   - applies trace-level robust Kalman baseline estimation
   - writes browser-ready review data
6. `tools/build_neuron_workbench_v2.py`
   - builds the v2 annotation workbench from `review_data.json`
   - accepts manifest-driven paths and attaches architecture-run metadata
   - configures/exports browser assets and manifests, but does not execute
     compute pipelines in-browser
7. `tools/serve_neuron_workbench.py`
   - serves the workbench locally and autosaves annotations

For routine processing, prefer the manifest runner instead of invoking each
stage manually:

```bash
python3 tools/run_neuron_review_pipeline.py \
  --dataset-manifest Outputs/Manifests/calcium_rest_cropped.dataset.json \
  --fiji "$(command -v fiji)"
```

The scripts still default to `calcium_video_2`, but they now accept
manifest-supplied dataset IDs and paths. A new sample writes to dataset-specific
folders such as `Outputs/HighPass/<dataset_id>/`,
`Outputs/CandidateEventPipeline/<dataset_id>/`, and
`Outputs/NeuronReview/<dataset_id>/app/`.

## Denoising Policy

The recommended default is trace-level denoising rather than pixel-video
denoising. Pixel-level smoothing can blur sparse firing events or turn impulse
noise into plausible-looking structure. The workbench therefore:

- finds stable ROI candidates first
- extracts each ROI trace from the raw video
- subtracts a local background/neuropil ring trace
- computes corrected `dF/F`
- estimates a slow baseline with a robust Kalman-style update
- calls candidate firing events from positive innovations

This keeps the sparse event signal visible while still suppressing slow drift
and baseline fluctuations.

## Data And Git Hygiene

Scientific input files and generated review outputs can be large and should
stay outside git:

- `Inputs/*.tif`, nested TIFFs, extracted input folders, and zip files are
  ignored.
- `Outputs/` is ignored.
- Commit scripts, documentation, and reusable source code only.

If a data file is already tracked, do not remove or restore it as part of a
tooling/documentation commit unless that is the explicit goal of the commit.

## Review-Driven Iteration

The intended loop is:

1. Generate candidate ROIs and events.
2. Review in the workbench.
3. Export ROI and event annotations.
4. Use accepted/rejected labels to tune ROI generation, trace denoising, and
   event thresholds.
5. Repeat until accepted ROIs are stable and event calls are usable for the
   inverse-dynamics workflow.

High-impact future robustness work:

- motion/drift correction before ROI extraction
- stronger local-correlation evidence maps
- uncertainty-ranked review queues
- ROI split/merge and footprint brush editing
- side-by-side comparison of trace denoisers and event models through
  Architecture Lab run manifests
- Process Lab summaries for ROI size, trace noise, event density, discovery
  burden, and evidence-map inspection
- selected-ROI crop and event-centered filmstrip review for local context

## Manifest And Schema Layer

The repository now includes a lightweight manifest layer:

- `schemas/dataset_manifest.schema.json`
- `schemas/architecture_run.schema.json`
- `schemas/annotations.schema.json`
- `schemas/review_data.schema.json`
- `examples/dataset_manifest.example.json`
- `examples/architecture_runs.example.json`

Use `tools/create_dataset_manifest.py` to create a dataset manifest for a new
video. The Python workbench builder and server no longer require absolute
machine-local source paths by default.

`tools/build_workbench_index.py` scans `Outputs/NeuronReview/*/app/` and writes
`Outputs/NeuronReview/index.html`, making it possible to cycle through multiple
processed videos from one local landing page.

The browser autosave file migrates to annotation schema v3. Existing labels are
preserved, while new fields capture trace quality, control readiness, artifact
class, identity grouping, and event timing quality.

The review-data generator now emits a `qc` block with frame-mean brightness
statistics for future rebuilds. The dashboard also computes lightweight QC from
existing ROI, event, noise, and evidence-map fields.

Architecture Lab uses completed architecture-run manifests for comparison and
planned pipeline manifests for Build mode. Planned manifests should capture
dataset references, pipeline family, parameter presets, expected artifacts,
status, and provenance, but they are not evidence that a run has completed.
External tools still perform the computational work; the local browser only
configures, reviews, compares, and exports the resulting metadata.

## Review UI Planning

The Review page is being organized around a stable video/trace workflow with
collapsible supporting sections. The documentation contract for this layout is:

- high-frequency ROI/event labeling controls remain easy to reach
- lower-frequency panels such as advanced scoring, discovery, and audit details
  can collapse without changing saved annotations
- Selected ROI Context remains keyboard-reachable and text-labeled so the crop,
  footprint, local metrics, and event filmstrip have an accessible summary
- layout preferences and collapsed sections are UI state, not scientific labels

Manual test planning for this slice should focus on navigation and data
boundaries: collapsing sections should not change `annotations.json`, selected
ROI context should remain reachable without a mouse, and Architecture Lab Build
mode should export plans without implying browser-side pipeline execution.

## Missed-Neuron Discovery

The workbench now separates ROI review from coverage auditing. Coverage auditing
uses evidence maps and “uncovered” suggestions to show signal that is not
explained by the current ROI set.

The generator writes browser-ready evidence maps under:

```bash
Outputs/NeuronReview/calcium_video_2/app/evidence/
```

The current discovery maps are intentionally lightweight:

- raw mean, max, and temporal standard deviation projections
- robust-z max projection
- z-threshold peak-count projection
- uncovered robust-z score after masking near existing ROIs
- local contrast proxy
- combined discovery score

Suggestions are not treated as final truth. They are review targets with
provenance and artifact cues. The user can promote likely missed neurons or mark
regions as artifacts. Those labels should feed the next tuning pass.

## Annotation Metric Helpers

`neurobench.annotation_metrics.compute_annotation_summary()` preserves the
existing ROI, event, suggestion, trace-quality, control-readiness, and review
burden fields. It also emits triage helpers under `triage_categories` and
`triage_queue_counts` for `strong_neuron`, `possible_missed_neuron`,
`artifact_like`, `merged_cluster`, `weak_trace`, and `needs_event_review`.

These categories are intentionally derived from current annotation and
review-data fields only. They should be used to prioritize review queues and
audit coverage, not as replacement labels in the annotation schema.
