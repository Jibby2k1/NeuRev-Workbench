# PLAN.md — Neuron Annotation, Architecture Lab, and Future Real-Time Inference

## 0. Purpose

This plan is for Codex implementation work in the `Separable-Gamma-CFAR` repository. The goal is to evolve the current calcium-imaging workbench into a high-recall, human-in-the-loop labeling and architecture-evaluation system for zebrafish light-sheet calcium videos.

The near-term target is **clear annotations**. The long-term target is **real-time neural activity inference for inverse dynamics and tail-motion control**.

This plan should be implemented incrementally. Preserve the existing grid-search and Fiji/Groovy workflows while adding manifest-driven, schema-driven, annotation-driven infrastructure around them.

---

## 1. Project context and scientific constraints

### 1.1 Imaging setup

Known constraints from the user:

- Imaging modality: **light-sheet calcium imaging**.
- Indicator: **GCaMP6f fish line**.
- Frame rate: **5 Hz**, so one frame is 200 ms.
- Current neuron size estimate: median around **4.5 µm**; expected soma scale around **3–10 µm**.
- Neurons are often **only noticeable when active**.
- Some neurons have highly variable baseline brightness / standard state.
- Noise and weak SNR are major problems.
- There is slight fish drift / morphing, currently believed to be minor but should be tracked.
- Near-term output: strong annotation dashboard.
- Long-term output: real-time inference suitable for inverse dynamics and future closed-loop fish control.
- Initial control target: **tail motion**; future target may be finer-grained behavioral control.

### 1.2 Consequences for the software design

Because there are currently no labels for the calcium videos, the dashboard must become the label-generation system. It should not assume that any algorithmic output is ground truth.

The system should optimize for:

1. **Neuron existence** — identify stable neuron identities / ROI candidates.
2. **Candidate firing events** — identify plausible GCaMP transients or activity events.
3. **Trace quality** — decide whether an ROI trace is reliable enough for inverse dynamics.
4. **Review efficiency** — favor high recall, but rank candidates so users spend most time on valuable decisions.
5. **Future online compatibility** — keep exported traces/events/parameters compatible with real-time models.

At 5 Hz with GCaMP6f, event timing should be treated as frame-level or low-resolution event timing, not exact spike timing. Deconvolution outputs should be treated as activity features or event probabilities unless independently validated.

---

## 2. Current repository snapshot

### 2.1 Existing workflows

The repository currently has two complementary workflows:

1. **Python grid-search workflow**
   - `main.py`
   - `config.py`
   - `worker.py`
   - `core/filters.py`
   - `core/detection.py`
   - `core/pipelines.py`
   - `evaluation/metrics.py`
   - `evaluation/analysis.py`
   - `reporting/generators.py`
   - `reporting/plotters.py`

2. **Fiji/Groovy + browser review workflow**
   - `tools/temporal_highpass_gaussian.ijm`
   - `tools/event_preserving_noise_suppression.groovy`
   - `tools/candidate_event_pipeline.groovy`
   - `tools/temporal_candidate_scoring.groovy`
   - `tools/generate_neuron_review_app.groovy`
   - `tools/build_neuron_workbench_v2.py`
   - `tools/serve_neuron_workbench.py`
   - `docs/NEURON_WORKBENCH.md`
   - `docs/PROCESSING_NOTES.md`

The Fiji/Groovy workflow already builds review data, ROI candidates, trace events, discovery evidence maps, autosaved `annotations.json`, and TSV exports.

### 2.2 Existing strengths

- Useful local browser workbench already exists.
- ROI-level and event-level labels already exist.
- Discovery suggestions already exist for possible missed neurons.
- The app already supports autosave to `annotations.json`.
- Existing evidence maps include raw mean, raw max, raw temporal standard deviation, robust-z max, peak-count, uncovered score, local contrast proxy, and combined discovery score.
- Existing trace handling includes local background correction and robust Kalman-style baseline / positive-innovation event detection.
- Existing code already separates some core algorithmic pieces: filters, detection, pipelines, metrics, reporting.

### 2.3 Current problems to fix before adding large features

#### P0.1 Metric key mismatch

`evaluation/metrics.py::calculate_froc_point_metrics()` returns uppercase keys:

```python
"TPR", "FPPI", "TP", "FP", "FN"
```

But `worker.py::process_full_task_for_worker()` currently records lowercase keys:

```python
"tpr": float(m.get("tpr", 0.0)),
"fppi": float(m.get("fppi", 0.0)),
```

This can silently produce zeros and break ranking / FROC / AUC logic.

Required fix:

- Normalize metric keys globally.
- Prefer canonical uppercase keys for existing compatibility.
- Add fallback helpers where needed.
- Add unit tests that fail if a 100% TPR synthetic detection is recorded as zero.

#### P0.2 CPU/no-CuPy import issue

`core/filters.py` sets `cp = None` if CuPy is unavailable, but then uses annotations like:

```python
def apply_kalman_mcc_filter_gpu(x_prev: cp.ndarray, ...)
```

When CuPy is unavailable, this can raise an import-time error.

Required fix:

- Add `from __future__ import annotations` at the top of `core/filters.py`, or replace those annotations with strings / `Any`.
- Add a CPU-only import smoke test.

#### P0.3 Pixel-level detections are the wrong metric unit

`utils.extract_pixel_detections()` returns one detection per active pixel. For neuron existence review, an ROI/component should be the detection unit. Pixel-level FPPI can over-penalize large footprints and confuse evaluation.

Required fix:

- Keep `extract_pixel_detections()` for legacy reports.
- Add component-level detection extraction.
- Add object-level matching metrics.

#### P0.4 Hard-coded dataset paths

Several legacy scripts historically assumed:

- a specific developer's home directory
- `calcium_video_2`
- fixed `Inputs/...` TIFF names
- fixed `Outputs/.../calcium_video_2/...` folders

Required fix:

- Introduce dataset/run manifests.
- Add CLI arguments to workbench build/serve scripts.
- Avoid absolute local paths in committed source.

#### P0.5 Inefficient trace modeling in browser JS

`tools/build_neuron_workbench_v2.py` embeds JS where `eventsForRoi(roi)` repeatedly calls `modeledTrace(roi)` and then calls it again inside the event loop for amplitudes.

Required fix:

- Add a client-side trace model cache keyed by ROI ID and current trace parameters.
- Invalidate cache when event threshold, Kalman gain, or spike gain changes.

#### P0.6 JSON encoder converts integer NumPy IDs to float

`utils.NumpyEncoder` currently handles both `np.integer` and `np.floating` by returning `float(obj)`. IDs should stay integers.

Required fix:

```python
if isinstance(obj, np.integer):
    return int(obj)
if isinstance(obj, np.floating):
    return float(obj)
```

#### P0.7 Generated outputs should remain outside git

Continue to avoid committing scientific input TIFFs and generated `Outputs/` data. Commit source, schemas, docs, tests, and small synthetic fixtures only.

---

## 3. Target architecture

The target project should have four layers:

```text
1. Data / manifests
   dataset_config.json, architecture_runs.json, annotations.json, schema files

2. Candidate generation
   current Fiji/Groovy, Python pipelines, local-z, CFAR, denoisers, Suite2p, CaImAn, PMD, OASIS

3. Evaluation
   annotation-driven metrics, component matching, ROI matching, event matching, trace-quality metrics

4. Workbench UI
   Review page, Architecture Lab page, Dataset QC page, Metrics/Audit page
```

The core design rule:

> The UI consumes standardized artifacts. It should not care whether a candidate came from Groovy, Python, Suite2p, CaImAn, DeepCAD-RT, PMD, or another method.

---

## 4. New directory and file structure

Add the following structure while preserving existing files:

```text
schemas/
  dataset_manifest.schema.json
  architecture_run.schema.json
  annotations.schema.json
  review_data.schema.json

neurobench/
  __init__.py
  manifests.py
  schemas.py
  paths.py
  video_io.py
  components.py
  roi_features.py
  trace_models.py
  event_models.py
  background.py
  motion.py
  evidence.py
  metrics.py
  annotation_metrics.py
  synthetic.py

neurobench/integrations/
  __init__.py
  suite2p_import.py
  caiman_import.py
  deepcadrt_runner.py
  pmd_runner.py
  oasis_events.py
  fiola_notes.md

neurobench/workbench/
  __init__.py
  build.py
  server.py
  assets/
    workbench.css
    workbench.js
    architecture_lab.js
    dataset_qc.js
    metrics_audit.js

tools/
  create_dataset_manifest.py
  build_architecture_run.py
  build_architecture_lab.py
  build_neuron_workbench_v2.py       # keep, but refactor to call neurobench.workbench.build
  serve_neuron_workbench.py          # keep, but refactor to call neurobench.workbench.server
  export_annotations.py
  compute_annotation_metrics.py
  generate_synthetic_fixture.py

tests/
  test_import_cpu.py
  test_metric_key_casing.py
  test_component_detections.py
  test_manifest_paths.py
  test_annotation_schema.py
  test_trace_cache_logic.py
  test_synthetic_smoke.py

examples/
  dataset_manifest.example.json
  architecture_runs.example.json
  annotations_v3.example.json
```

Keep existing `core/`, `evaluation/`, and `reporting/` modules for backward compatibility. New work should prefer `neurobench/` modules unless a small targeted fix belongs in an existing file.

---

## 5. Manifest-driven data model

### 5.1 Dataset manifest

Create a dataset manifest schema and an example manifest.

Required fields:

```json
{
  "schema_version": 1,
  "dataset_id": "calcium_video_2",
  "description": "Light-sheet GCaMP6f zebrafish calcium video",
  "modality": "light_sheet",
  "indicator": "GCaMP6f",
  "frame_rate_hz": 5.0,
  "frame_period_sec": 0.2,
  "pixel_size_um": null,
  "soma_diameter_um_range": [3.0, 10.0],
  "estimated_median_soma_diameter_um": 4.5,
  "raw_video": "Inputs/050126/050126/calcium video 2.tif",
  "output_root": "Outputs",
  "review_root": "Outputs/NeuronReview/calcium_video_2",
  "architecture_root": "Outputs/ArchitectureRuns/calcium_video_2",
  "behavior": {
    "tail_motion_file": null,
    "frame_alignment_file": null
  },
  "notes": {
    "visibility": "neurons mostly visible when active",
    "noise": "weak SNR; uneven fluorescence; slight drift/morphing"
  }
}
```

Implementation notes:

- `pixel_size_um` may be unknown. Do not assume 0.5 µm globally unless the manifest says so.
- Treat `soma_diameter_um_range` as a configurable prior, not a hard constraint.
- Add derived pixel-area defaults only when `pixel_size_um` is present.
- Use `Path` resolution relative to the manifest file or repository root. Make this behavior explicit and tested.

### 5.2 Architecture run manifest

Each candidate-generation pipeline should produce a standardized architecture run:

```json
{
  "schema_version": 1,
  "dataset_id": "calcium_video_2",
  "run_id": "current_sigma06_localz_v1",
  "label": "Current sigma06 robust local-z candidate pipeline",
  "created_at": "2026-05-11T00:00:00Z",
  "method_family": "current_localz_cfar",
  "purpose": "candidate_proposal",
  "stages": [
    {
      "name": "temporal_highpass_gaussian",
      "params": {"sigma_frames": 0.6}
    },
    {
      "name": "robust_positive_local_z",
      "params": {"local_radius_px": 11}
    },
    {
      "name": "component_filter",
      "params": {"seed_z": 2.0, "grow_z": 1.1, "min_area_px": 4}
    },
    {
      "name": "trace_event_scoring",
      "params": {"event_threshold_z": 2.4}
    }
  ],
  "artifacts": {
    "raw_video": "...",
    "registered_video": null,
    "denoised_video": null,
    "z_stack": "...",
    "mask_tif": "...",
    "label_tif": "...",
    "rois_json": "rois.json",
    "events_tsv": "events.tsv",
    "traces_npz": "traces.npz",
    "evidence_maps_dir": "evidence/",
    "review_data_json": "review_data.json"
  },
  "provenance": {
    "source_script": "tools/generate_neuron_review_app.groovy",
    "git_commit": null,
    "software_versions": {}
  }
}
```

### 5.3 ROI schema

Create `rois.json` with one object per candidate ROI:

```json
{
  "schema_version": 1,
  "dataset_id": "calcium_video_2",
  "run_id": "current_sigma06_localz_v1",
  "rois": [
    {
      "roi_id": "current_sigma06_localz_v1:roi_000001",
      "local_id": 1,
      "centroid_x": 123.4,
      "centroid_y": 88.2,
      "bbox": [118, 83, 129, 94],
      "area_px": 42,
      "footprint": {
        "encoding": "points",
        "points": [[123, 88], [124, 88]]
      },
      "candidate_type": "functional_roi",
      "provenance": ["current_sigma06_localz_v1"],
      "features": {
        "peak_z": 3.1,
        "trace_snr": 1.8,
        "local_corr_mean": null,
        "background_corr": null,
        "event_count": 4,
        "compactness": null,
        "motion_sensitivity": null
      },
      "events": [
        {
          "frame": 52,
          "time_sec": 10.4,
          "score_z": 2.9,
          "amplitude": 0.042,
          "method": "robust_kalman_positive_innovation"
        }
      ]
    }
  ]
}
```

### 5.4 Annotation schema v3

Extend the current `annotations.json` to preserve existing labels while adding trace quality, artifact classes, identity handling, and control-readiness.

Target shape:

```json
{
  "schema_version": 3,
  "dataset_id": "calcium_video_2",
  "updated_at": "2026-05-11T00:00:00Z",
  "reviewer": null,
  "active_run_ids": ["current_sigma06_localz_v1"],
  "rois": {
    "current_sigma06_localz_v1:roi_000001": {
      "cell_state": "accepted",
      "trace_quality": "usable",
      "control_ready": "maybe",
      "artifact_class": "none",
      "identity_group": "cell_000123",
      "needs_action": null,
      "notes": "Visible when active; weak baseline."
    }
  },
  "events": {
    "current_sigma06_localz_v1:roi_000001:52": {
      "event_state": "accepted",
      "event_type": "weak_transient",
      "timing_quality": "approximate",
      "notes": "Peak within one frame."
    }
  },
  "suggestions": {},
  "edits": {
    "splits": [],
    "merges": [],
    "footprint_edits": []
  },
  "settings": {}
}
```

Allowed ROI `cell_state` values:

```text
accepted | rejected | unsure | duplicate | merged | split_needed | redraw_needed | hidden
```

Allowed `trace_quality` values:

```text
excellent | usable | weak | unusable | unsure
```

Allowed `control_ready` values:

```text
yes | maybe | no | unsure
```

Allowed artifact classes:

```text
none | uneven_background | vessel_or_static_structure | impulse_noise | border_artifact |
saturation_or_bright_blob | motion_artifact | merged_cells | duplicate | out_of_focus_background | other | unsure
```

Allowed event states:

```text
accepted | rejected | unsure
```

Allowed event types:

```text
clear_transient | weak_transient | background_contaminated | motion_correlated |
noise_spike | saturation | decay_tail | other | unsure
```

Migration requirement:

- Add a migration utility from current `annotations.version == 2` to `schema_version == 3`.
- Do not discard existing `accept`, `reject`, `unsure`, notes, discovery suggestion labels, or promoted ROIs.

---

## 6. Dashboard design

The dashboard should remain a local browser app, but it should become multi-page. The user requested a standalone configurable architecture dashboard as a different page in the same application. Implement a tabbed or hash-routed app with these pages:

```text
#/review
#/architecture
#/dataset-qc
#/metrics
```

### 6.1 Page 1 — Review / Label Studio

Purpose: label neuron existence, events, trace quality, artifact class, and control readiness.

Required features:

- Raw video playback.
- Overlay current candidate ROI footprints.
- Event-centered playback: jump to event and display frames `t-5` through `t+10` where available.
- Current ROI trace panel.
- Raw trace, local-background trace, corrected dF/F, event trace, and event threshold.
- Event list for selected ROI.
- ROI labels:
  - accepted
  - rejected
  - unsure
  - duplicate
  - split needed
  - redraw needed
- Trace-quality label:
  - excellent
  - usable
  - weak
  - unusable
  - unsure
- Control-readiness label:
  - yes
  - maybe
  - no
  - unsure
- Artifact-class selector.
- Notes field.
- Suggestion promotion / artifact labeling.
- Keyboard shortcuts preserved from current workbench.
- Export ROI, event, suggestion, and full JSON annotations.

Enhancements over current app:

- Add local crop around selected ROI.
- Add denoised crop if selected architecture run has a denoised video.
- Add evidence crop for current evidence map.
- Add local background / neuropil ring visualization.
- Show which architecture run(s) proposed the selected ROI.
- Show local correlation, SNR, event count, background correlation, and artifact score when available.

### 6.2 Page 2 — Architecture Lab

Purpose: compare candidate-generation pipelines side-by-side.

Required features:

- Load multiple architecture runs for the same dataset.
- Synchronized frame slider across runs.
- Synchronized ROI/candidate selection across runs.
- Side-by-side panes:

```text
Raw | Registered | Denoised | Evidence map | Candidate overlay | Trace/events
```

- Compare run A vs run B vs run C.
- Show run parameters and stage list.
- Show architecture provenance for selected candidates.
- Show ensemble agreement:
  - found by current pipeline only
  - found by DeepCAD-RT + current pipeline
  - found by Suite2p only
  - found by CaImAn only
  - found by many methods
- Show candidate priority queue based on confidence and uncertainty.
- Show accepted/rejected/unsure counts by architecture.
- Allow exporting an architecture comparison report.

Initial architecture options to support:

```text
current_localz_cfar
current_gamma_single_stage
current_gamma_two_stage
current_kalman_mcc_cfar
deepcadrt_denoised_localz
pmd_denoised_localz
suite2p_import
caiman_import
oasis_event_model
```

Implementation can start by comparing only existing current runs, then add importers.

### 6.3 Page 3 — Dataset QC

Purpose: help distinguish software issues from acquisition/hardware issues.

Required panels:

- Framewise mean brightness.
- Framewise max brightness.
- Framewise robust noise estimate.
- Photobleaching / baseline trend.
- Spatial mean fluorescence map.
- Spatial max fluorescence map.
- Spatial temporal-standard-deviation map.
- Saturation map.
- Estimated SNR map.
- Local temporal noise map.
- Drift estimate over time.
- Motion-correction residuals if available.
- Candidate count vs threshold.
- Warning panel:
  - extreme saturation
  - severe drift
  - very low dynamic range
  - high background heterogeneity
  - unstable illumination

This page is important because the hardware system is still improving.

### 6.4 Page 4 — Metrics / Audit

Purpose: evaluate architecture outputs using human annotations.

Before labels exist, this page shows only candidate counts and review progress. After labels accumulate, show:

- Accepted ROIs per architecture.
- Rejected ROIs per architecture.
- Unsure rate per architecture.
- Accepted events per architecture.
- Rejected events per architecture.
- Trace-quality distribution.
- Control-ready neuron count.
- Review burden:

```text
candidate ROIs per accepted ROI
candidate events per accepted event
minutes reviewed per accepted ROI, if timing is tracked
```

- Missed-neuron audit:

```text
promoted discovery suggestions / accepted ROIs
```

- Architecture complementarity:

```text
accepted neurons found only by architecture X
accepted neurons found by multiple architectures
false positives common to architecture X
```

- Exportable summary TSV/CSV.

---

## 7. Candidate-generation approach

The goal is recall-heavy proposal. It is better to overestimate neurons and let users reject than to miss neurons, but missed-neuron discovery must also be supported.

### 7.1 Candidate ensemble

Generate a union of candidates from multiple methods:

1. Existing robust local-z / current Fiji/Groovy pipeline.
2. Existing Gamma/CFAR single-stage and two-stage variants.
3. Existing Kalman-MCC residual + CFAR variants.
4. Local-correlation evidence candidates.
5. Event-triggered average footprint candidates.
6. DeepCAD-RT-denoised video + current detector.
7. PMD-denoised video + current detector.
8. Suite2p imported ROIs.
9. CaImAn/CNMF imported components.
10. Manual discovery suggestions promoted by users.

Merge candidates spatially into identity groups where appropriate, but preserve original architecture provenance.

### 7.2 New evidence maps

Add these evidence maps as soon as possible:

#### Local temporal correlation map

For each pixel, compute the mean/median correlation of its time series with neighboring pixels.

Purpose:

- Real neuron activity tends to be spatially coherent.
- Random noise and impulse noise often have poor local coherence.

Implementation notes:

- Use local windows, e.g. radius 1–3 px.
- Use robust normalization per pixel before correlation.
- Avoid large memory blowups; process in tiles if needed.

#### Event-triggered average footprint map

For candidate events, average local frames around the event peak:

```text
t - 2, t - 1, t, t + 1, t + 2, t + 3, t + 4
```

At 5 Hz, this covers approximately 1.4 seconds. Expose the window in config.

Purpose:

- Real GCaMP transients should often have a spatially compact footprint over several frames.
- Impulse noise often appears as a single-frame speckle.

#### Local background heterogeneity map

Estimate local background mean/median and local variation. Use to flag regions where uneven illumination may produce false candidates.

#### Saturation / bright-blob map

Detect persistent high-intensity regions and saturated pixels.

#### Motion sensitivity map

Compare evidence before vs after simple rigid correction. Candidates that disappear or shift strongly may be motion-related.

### 7.3 Background correction

Because the user reports varying fluorescence across the entire image, add spatial and temporal background options:

- Framewise median subtraction.
- Per-pixel slow baseline estimate.
- Local percentile / rolling-ball background subtraction.
- Local MAD normalization.
- Optional neuropil/background ring subtraction per ROI.

Expose these stages as architecture-run parameters. Do not bake one global background correction into all methods.

### 7.4 Motion correction

Even slight drift can matter under weak SNR. Add conservative motion correction as an optional architecture stage.

Recommended order:

1. Implement simple rigid phase-correlation correction or import Suite2p/CaImAn registered outputs.
2. Record x/y shifts per frame.
3. Add Dataset QC plot of shifts over time.
4. Compare candidates with and without motion correction in Architecture Lab.
5. Add nonrigid correction only if rigid correction is inadequate.

Do not hide raw video. Always allow raw vs registered comparison.

### 7.5 Candidate priority score

Add a heuristic score for review queue ordering:

```text
priority_score =
  + 0.25 * architecture_agreement
  + 0.20 * local_correlation_score
  + 0.20 * event_support_score
  + 0.15 * trace_snr_score
  + 0.10 * compactness_score
  + 0.10 * uncovered_discovery_score
  - 0.20 * background_correlation_score
  - 0.15 * artifact_score
```

Implementation requirement:

- Keep all component scores visible.
- Treat the total as a queue-ordering heuristic, not truth.
- Once labels exist, replace or augment this heuristic with a trained candidate ranker.

---

## 8. Trace and event modeling

### 8.1 Trace extraction

For each ROI, export:

- raw ROI mean trace,
- local background ring trace,
- corrected trace,
- dF/F trace,
- robust baseline trace,
- positive innovation trace,
- event-score trace,
- optional denoised trace,
- optional OASIS deconvolved trace.

Use frame rate from the manifest (`frame_rate_hz = 5.0`). Store `time_sec = frame / frame_rate_hz` or equivalent, with a clear convention for whether frame numbers are 0-based or 1-based.

### 8.2 Event detection

Initial event models:

1. Current robust Kalman positive innovation detector.
2. Peak detection on corrected dF/F.
3. OASIS AR(1) or AR(2) deconvolution.
4. Optional CASCADE or ENS² later, after ROI traces are stable.

At 5 Hz, use event timing quality labels:

```text
precise | approximate | unusable | unsure
```

For GCaMP6f, default decay parameters should be configurable. Start with a broad range and/or estimate per ROI from clear accepted events:

```text
tau_decay_sec candidates: 0.4, 0.7, 1.0, 1.5
sampling_rate_hz: 5.0
```

Do not claim exact spike counts from deconvolution unless validated with electrophysiology or an accepted external ground truth.

### 8.3 Control-readiness

Add `control_ready` as a label and as a derived metric. A neuron should be considered more control-ready if:

- ROI is accepted.
- Trace quality is usable or excellent.
- Events are accepted or mostly accepted.
- Trace has tolerable background contamination.
- Candidate is stable under small parameter changes.
- Candidate is not strongly motion-correlated.
- Timing is adequate for tail-motion modeling.

Export control-ready traces separately for inverse-dynamics experiments.

### 8.4 Export for inverse dynamics

Add an export command:

```bash
python tools/export_annotations.py \
  --dataset-manifest examples/dataset_manifest.example.json \
  --annotations Outputs/NeuronReview/calcium_video_2/app/annotations.json \
  --out Outputs/InverseDynamics/calcium_video_2/
```

Output files:

```text
accepted_rois.tsv
accepted_events.tsv
control_ready_traces.npz
all_labeled_traces.npz
event_features.tsv
roi_features.tsv
annotation_summary.json
```

`event_features.tsv` should include:

```text
dataset_id
roi_id
identity_group
frame
time_sec
event_state
event_type
timing_quality
amplitude
score_z
source_method
trace_quality
control_ready
```

---

## 9. State-of-the-art integrations

Do not block dashboard work on SOTA integration. Build the schema and UI first, then add integrations as optional architecture runs.

### 9.1 Suite2p importer

Rationale:

- Suite2p includes registration, ROI detection, signal extraction, ROI classification, deconvolution, and GUI-style visualization.
- It is primarily associated with two-photon workflows, but its registration / ROI / trace outputs are still useful as a candidate source to compare.

Implementation:

```text
neurobench/integrations/suite2p_import.py
```

Required capabilities:

- Read Suite2p output folder.
- Import `stat.npy`, `iscell.npy`, `F.npy`, `Fneu.npy`, `spks.npy`, and `ops.npy` when present.
- Convert Suite2p ROIs into `rois.json` schema.
- Convert traces into `traces.npz` schema.
- Convert deconvolved activity into `events.tsv` using configurable thresholds.
- Write `architecture_run.json` with method family `suite2p_import`.

CLI:

```bash
python -m neurobench.integrations.suite2p_import \
  --dataset-manifest dataset_manifest.json \
  --suite2p-dir path/to/suite2p/plane0 \
  --run-id suite2p_default_v1 \
  --out Outputs/ArchitectureRuns/calcium_video_2/suite2p_default_v1
```

### 9.2 CaImAn / CNMF importer

Rationale:

- CaImAn supports motion correction, source extraction, deconvolution, and online analysis.
- CaImAn is especially relevant because published CaImAn work applied online processing to zebrafish whole-brain light-sheet data.
- CNMF/CNMF-E-style background modeling may help with uneven fluorescence and low SNR.

Implementation:

```text
neurobench/integrations/caiman_import.py
```

Required capabilities:

- Import CaImAn estimates/components if available.
- Convert spatial components into ROI footprints.
- Convert temporal components into traces.
- Import deconvolved activity if available.
- Preserve CaImAn quality metrics where available.

Do not require CaImAn as a core dependency. Make it an optional integration.

### 9.3 DeepCAD-RT runner/importer

Rationale:

- DeepCAD-RT is highly relevant for low-SNR fluorescence imaging and has zebrafish calcium-imaging demonstrations.
- It is self-supervised and can train from raw noisy videos, which is useful because there are no clean labels.

Implementation:

```text
neurobench/integrations/deepcadrt_runner.py
```

Start as an importer/runner wrapper rather than embedding all DeepCAD code.

Required capabilities:

- Accept raw TIFF and DeepCAD-RT output TIFF.
- Register the denoised TIFF as an architecture artifact.
- Run existing local-z/component detection on denoised video if requested.
- Always preserve raw-video display next to denoised output in the dashboard.

Important warning:

- Denoised video must not become ground truth.
- The UI must always show raw and denoised views side-by-side.
- Labelers should be able to reject denoiser hallucinations or denoiser-suppressed events.

### 9.4 PMD / localized low-rank denoising

Rationale:

- PMD is training-free and explicitly designed to separate low-dimensional signal from temporally uncorrelated noise in functional imaging.
- It is useful as an independent denoising baseline against DeepCAD-RT.

Implementation:

```text
neurobench/integrations/pmd_runner.py
```

Start by supporting imported PMD-denoised TIFFs. A direct runner can be added later if dependencies are stable.

### 9.5 OASIS event model

Rationale:

- OASIS is relevant to online calcium deconvolution and future real-time inference.
- It can produce a trace-level activity estimate from accepted or candidate ROI traces.

Implementation:

```text
neurobench/integrations/oasis_events.py
```

Required capabilities:

- Input: ROI traces from `traces.npz`.
- Config: `frame_rate_hz`, AR model type, tau/decay parameters, thresholding options.
- Output: `events.tsv` and per-frame deconvolved traces.

Example CLI:

```bash
python -m neurobench.integrations.oasis_events \
  --dataset-manifest dataset_manifest.json \
  --architecture-run Outputs/ArchitectureRuns/calcium_video_2/current_sigma06_localz_v1/architecture_run.json \
  --tau-decay-sec 0.7 \
  --run-id oasis_current_sigma06_tau07
```

### 9.6 FIOLA / online path

Do not implement FIOLA in the first pass. Keep it in the plan for future real-time inference.

Near-term work:

- Add notes in `neurobench/integrations/fiola_notes.md`.
- Ensure architecture-run schema can represent online initialization and streaming inference.
- Ensure accepted ROI masks/traces can be exported in a form usable by a future online pipeline.

### 9.7 CASCADE / ENS²

These should be later-stage event-probability integrations, not initial neuron-existence tools.

Reason:

- They depend on trace quality and indicator/model compatibility.
- At 5 Hz with GCaMP6f, exact spike timing is limited by the acquisition rate and indicator kinetics.

Use these for:

- candidate event probability,
- continuous activity features,
- comparison against OASIS and manual event labels.

Do not treat them as ground-truth spike counters.

---

## 10. Evaluation and metrics

### 10.1 Replace ground-truth-only thinking with annotation-driven metrics

The current Python grid search expects a ground-truth CSV. For the current calcium videos, there are no labels. Therefore, the primary evaluation target should be the generated annotations.

Add:

```text
neurobench/annotation_metrics.py
```

Metrics should include:

- Review progress:
  - total candidate ROIs,
  - labeled ROIs,
  - unlabeled ROIs,
  - accepted/rejected/unsure counts,
  - accepted/rejected/unsure event counts.

- Architecture quality:
  - accepted ROI count,
  - rejected ROI count,
  - unsure rate,
  - accepted event count,
  - rejected event count,
  - trace-quality distribution,
  - control-ready count.

- Review burden:
  - candidates per accepted ROI,
  - candidates per control-ready ROI,
  - candidate events per accepted event.

- Missed-neuron discovery:
  - promoted suggestions per architecture,
  - accepted promoted suggestions,
  - missed-neuron rate relative to accepted ROIs.

- Architecture complementarity:
  - accepted ROIs found only by one architecture,
  - accepted ROIs found by multiple architectures,
  - architecture agreement score.

### 10.2 Object-level detection utilities

Add:

```text
neurobench/components.py
```

Functions:

```python
extract_components_2d(mask: np.ndarray, min_area: int = 1) -> list[Component]
extract_component_detections(mask_stack: np.ndarray, min_area: int = 1) -> dict[int, list[dict]]
merge_components_across_architectures(rois_by_run: dict, distance_px: float, iou_threshold: float) -> list[IdentityGroup]
compute_iou(points_a, points_b) -> float
centroid_distance(a, b) -> float
```

Component detection objects should include:

```json
{
  "x": 123.4,
  "y": 88.2,
  "area": 42,
  "bbox": [118, 83, 129, 94],
  "points": [[123, 88], [124, 88]],
  "score": 2.7
}
```

### 10.3 FROC metrics should remain available but fixed

For legacy ground-truth workflows:

- Fix metric casing.
- Add component-level FROC as an option.
- Keep pixel-level FROC only for historical comparability.
- Add tests for uppercase/lowercase compatibility.

### 10.4 Future real-time metrics

Once tail data alignment is available, add:

- neural event to tail-motion latency,
- event-feature predictive utility,
- trace lag correlation with tail motion,
- control-ready neuron subset performance,
- online feasibility estimate:
  - per-frame processing time,
  - latency budget,
  - memory footprint.

---

## 11. Workbench implementation details

### 11.1 Refactor generated app assets

Current `tools/build_neuron_workbench_v2.py` embeds large CSS and JS strings. For maintainability, move static assets into:

```text
neurobench/workbench/assets/
  workbench.css
  workbench.js
  architecture_lab.js
  dataset_qc.js
  metrics_audit.js
```

The Python builder should:

- read `review_data.json`, architecture manifests, and annotations,
- emit `index.html`,
- copy static assets,
- write a small `app_config.json` or embed minimal bootstrap JSON.

Preserve the ability to run without npm/build tooling.

### 11.2 Add CLI args to builder

Change `tools/build_neuron_workbench_v2.py` to accept:

```bash
python tools/build_neuron_workbench_v2.py \
  --dataset-manifest dataset_manifest.json \
  --review-data Outputs/NeuronReview/calcium_video_2/app/review_data.json \
  --architecture-runs Outputs/ArchitectureRuns/calcium_video_2/architecture_runs.json \
  --app-dir Outputs/NeuronReview/calcium_video_2/app
```

Fallback behavior:

- If only `--app-dir` is supplied, read `review_data.json` from that directory.
- If no args are supplied, use relative defaults, not workstation-specific paths.

### 11.3 Server improvements

Refactor `tools/serve_neuron_workbench.py` to use `neurobench.workbench.server`.

Add endpoints:

```text
GET  /annotations.json
PUT  /annotations.json
GET  /app_config.json
GET  /architecture_runs.json
GET  /metrics.json
POST /export/annotations      # optional; can be added later
```

Keep server local and simple. Do not add external web framework unless necessary.

### 11.4 Client-side cache

Implement a JS trace cache:

```javascript
const traceCache = new Map();

function traceCacheKey(roi) {
  return `${roi.id}|${setting('kalmanGain')}|${setting('spikeGain')}`;
}

function modeledTraceCached(roi) {
  const key = traceCacheKey(roi);
  if (!traceCache.has(key)) traceCache.set(key, modeledTrace(roi));
  return traceCache.get(key);
}

function invalidateTraceCache() {
  traceCache.clear();
}
```

Use cached traces in:

- `eventsForRoi`,
- `eventFrames`,
- ROI quality scoring,
- event list rendering,
- trace drawing.

### 11.5 Label widgets

Add UI widgets for:

- `cell_state`,
- `trace_quality`,
- `control_ready`,
- `artifact_class`,
- `event_state`,
- `event_type`,
- `timing_quality`,
- `identity_group`,
- `needs_action`.

Do not remove existing accept/reject/unsure buttons. Map them to `cell_state` or `event_state`.

### 11.6 ROI editing roadmap

Add in stages:

1. Mark ROI as `split_needed`, `merge_needed`, `redraw_needed`.
2. Select multiple ROIs and assign same `identity_group`.
3. Merge selected ROIs into a new virtual ROI.
4. Split ROI by drawing a line or selecting sub-components.
5. Brush add/remove footprint pixels.

Initial implementation can store edit intentions without modifying the source ROI. Actual footprint editing can come later.

---

## 12. CLI examples

### 12.1 Create dataset manifest

```bash
python tools/create_dataset_manifest.py \
  --dataset-id calcium_video_2 \
  --raw-video "Inputs/050126/050126/calcium video 2.tif" \
  --modality light_sheet \
  --indicator GCaMP6f \
  --frame-rate-hz 5 \
  --estimated-median-soma-diameter-um 4.5 \
  --soma-diameter-um-range 3 10 \
  --out datasets/calcium_video_2.dataset.json
```

### 12.2 Build current architecture run from existing review data

```bash
python tools/build_architecture_run.py \
  --dataset-manifest datasets/calcium_video_2.dataset.json \
  --run-id current_sigma06_localz_v1 \
  --method-family current_localz_cfar \
  --review-data Outputs/NeuronReview/calcium_video_2/app/review_data.json \
  --out Outputs/ArchitectureRuns/calcium_video_2/current_sigma06_localz_v1
```

### 12.3 Build multi-page dashboard

```bash
python tools/build_neuron_workbench_v2.py \
  --dataset-manifest datasets/calcium_video_2.dataset.json \
  --architecture-runs Outputs/ArchitectureRuns/calcium_video_2/architecture_runs.json \
  --app-dir Outputs/NeuronReview/calcium_video_2/app
```

### 12.4 Serve dashboard

```bash
python tools/serve_neuron_workbench.py \
  --app-dir Outputs/NeuronReview/calcium_video_2/app \
  --port 8765
```

### 12.5 Compute annotation metrics

```bash
python tools/compute_annotation_metrics.py \
  --dataset-manifest datasets/calcium_video_2.dataset.json \
  --architecture-runs Outputs/ArchitectureRuns/calcium_video_2/architecture_runs.json \
  --annotations Outputs/NeuronReview/calcium_video_2/app/annotations.json \
  --out Outputs/NeuronReview/calcium_video_2/annotation_metrics.json
```

---

## 13. Implementation phases

### Phase 0 — Stabilize existing code

Goal: fix correctness issues before building new dashboard features.

Tasks:

1. Fix metric key casing in `worker.py`.
2. Add uppercase/lowercase fallback helpers where reports consume `TPR`/`FPPI`.
3. Fix `core/filters.py` CuPy annotation import issue.
4. Fix `utils.NumpyEncoder` integer handling.
5. Add component-level detection extraction in `utils.py` or `neurobench/components.py`.
6. Add unit tests:
   - CPU import smoke test,
   - metric casing test,
   - component detection test,
   - JSON encoder ID preservation test.
7. Add `pytest` to a minimal environment or development requirements file.

Acceptance criteria:

- `python -c "import core.filters; import worker"` works without CuPy.
- Synthetic perfect detections produce nonzero/expected TPR.
- Component extraction returns one detection for one blob, not one per pixel.
- Existing grid-search path still runs.

### Phase 1 — Manifests and schemas

Goal: remove hard-coded paths and standardize data.

Tasks:

1. Add `schemas/` files.
2. Add `neurobench/manifests.py`.
3. Add `examples/dataset_manifest.example.json`.
4. Add `examples/architecture_runs.example.json`.
5. Add `tools/create_dataset_manifest.py`.
6. Refactor `tools/build_neuron_workbench_v2.py` CLI path handling.
7. Refactor `tools/serve_neuron_workbench.py` default path handling.
8. Update docs.

Acceptance criteria:

- Workbench can be built with `--app-dir` and `--dataset-manifest`.
- No committed Python script requires a developer-specific home path by default.
- Existing hard-coded Groovy scripts may remain temporarily, but docs should mark them as legacy until refactored.

### Phase 2 — Annotation schema v3

Goal: support labels needed for inverse dynamics and future model training.

Tasks:

1. Add annotation schema v3.
2. Add migration from v2 to v3.
3. Extend browser annotations with:
   - `cell_state`,
   - `trace_quality`,
   - `control_ready`,
   - `artifact_class`,
   - `event_type`,
   - `timing_quality`,
   - `identity_group`,
   - `needs_action`.
4. Update TSV exports.
5. Add `tools/export_annotations.py`.

Acceptance criteria:

- Existing annotations still load.
- Existing accept/reject/unsure buttons still work.
- New labels are autosaved.
- Full JSON export includes new fields.

### Phase 3 — Workbench UI refactor and Review page enhancements

Goal: make the UI maintainable and better for labeling.

Tasks:

1. Move CSS/JS assets out of Python strings.
2. Add hash-based page routing.
3. Keep current review page functional.
4. Add selected-ROI crop panel.
5. Add event-centered playback.
6. Add provenance/evidence card for selected ROI.
7. Add trace cache.
8. Add artifact/trace/control widgets.

Acceptance criteria:

- Review page works at least as well as current v2 workbench.
- Trace/event rendering is faster for large ROI sets.
- User can label trace quality and control readiness.

### Phase 4 — Architecture-run schema and Architecture Lab page

Goal: compare methods side-by-side.

Tasks:

1. Add `build_architecture_run.py` that converts existing `review_data.json` into standardized run artifacts.
2. Add `architecture_runs.json` manifest.
3. Add Architecture Lab page.
4. Support selecting run A / run B.
5. Show synchronized frames and overlays.
6. Show architecture parameters.
7. Show architecture provenance per candidate.
8. Show architecture-specific annotation counts.

Acceptance criteria:

- At least two current pipeline variants can be compared side-by-side.
- Architecture Lab can display raw/evidence/ROI overlays for selected runs.
- Accepted/rejected/unsure counts are grouped by run.

### Phase 5 — Dataset QC page

Goal: expose noise, drift, fluorescence, and hardware-quality indicators.

Tasks:

1. Add `neurobench/evidence.py` functions for QC maps.
2. Add framewise brightness/noise plots.
3. Add spatial mean/max/std/saturation maps.
4. Add drift estimate plot.
5. Add candidate-count-vs-threshold plot.
6. Add warnings.

Acceptance criteria:

- QC page can be generated from a raw TIFF or existing review data.
- User can inspect whether a video has severe brightness drift, saturation, or motion.

### Phase 6 — Local correlation and event-triggered evidence

Goal: improve candidate ranking and missed-neuron discovery.

Tasks:

1. Compute local correlation map.
2. Compute event-triggered average footprint for candidate events.
3. Add these maps to `review_data.json` and/or architecture artifacts.
4. Add ROI features:
   - within-ROI pixel correlation,
   - ROI-vs-background correlation,
   - local correlation mean/max,
   - event-triggered footprint compactness.
5. Add these fields to ROI evidence card.
6. Add these fields to candidate priority score.

Acceptance criteria:

- New evidence maps appear in Review and Architecture Lab.
- Candidate ranking can sort by local correlation or event support.
- Exported ROI features include new fields.

### Phase 7 — Annotation-driven metrics

Goal: quantify review and architecture performance without external labels.

Tasks:

1. Add `neurobench/annotation_metrics.py`.
2. Add `tools/compute_annotation_metrics.py`.
3. Add Metrics/Audit page.
4. Add architecture complementarity metrics.
5. Add review burden metrics.
6. Add exportable summary.

Acceptance criteria:

- Metrics page updates from `annotations.json`.
- Metrics work with no labels, partial labels, and complete labels.
- Metrics can answer: which architecture gives the most accepted/control-ready ROIs per review burden?

### Phase 8 — SOTA importers

Goal: integrate SOTA outputs as competing evidence sources.

Recommended order:

1. DeepCAD-RT output importer / runner wrapper.
2. PMD output importer.
3. Suite2p importer.
4. OASIS event model.
5. CaImAn importer.
6. FIOLA notes / future online preparation.
7. CASCADE/ENS² event-probability integration later.

Acceptance criteria:

- Each integration writes the same architecture-run schema.
- Each integration can be enabled/disabled without breaking the core dashboard.
- Raw video remains visible beside denoised/inferred outputs.

### Phase 9 — Inverse-dynamics export

Goal: prepare clean data for tail-motion modeling.

Tasks:

1. Add behavior/tail alignment fields to dataset manifest.
2. Add export of accepted/control-ready traces.
3. Add event feature table.
4. Add timing metadata.
5. Add placeholder for tail-motion synchronized features.

Acceptance criteria:

- A downstream model can load traces/events and know frame/time alignment.
- Control-ready subset can be exported separately.

### Phase 10 — Online/real-time preparation

Goal: future-proof for closed-loop fish control.

Tasks:

1. Add latency tracking for pipeline stages.
2. Add online-compatible trace extraction API.
3. Add OASIS online mode wrapper if practical.
4. Add architecture-run fields for initialization vs streaming stages.
5. Add benchmark command for per-frame processing time.

Acceptance criteria:

- Offline annotations can seed an online pipeline.
- The code can estimate whether processing is within a 200 ms/frame budget at 5 Hz.

---

## 14. Tests

Add tests with small synthetic fixtures only.

### 14.1 Synthetic video fixture

Generate a tiny video with:

- shape `(T=40, H=64, W=64)`,
- three synthetic neurons,
- low SNR,
- uneven background field,
- one impulse-noise artifact,
- slight drift,
- known event times.

This should not be used as a scientific benchmark. It is for software correctness.

### 14.2 Required tests

```text
test_import_cpu.py
  - core modules import without CuPy.

test_metric_key_casing.py
  - worker/result metrics use canonical keys.
  - calculate_truncated_auc works on canonical points.

test_component_detections.py
  - one connected blob returns one component.
  - two blobs return two components.
  - min_area filters small speckles.

test_manifest_paths.py
  - relative paths resolve correctly.
  - no absolute default path required.

test_annotation_schema.py
  - v2 annotations migrate to v3.
  - accepted/rejected/unsure states map correctly.
  - new fields persist.

test_trace_cache_logic.py
  - modeled traces are computed once per ROI/parameter key in JS-equivalent logic or a Python mirror.

test_synthetic_smoke.py
  - generate synthetic video.
  - build minimal architecture run.
  - build workbench app.
  - compute annotation metrics from a small fake annotations file.
```

---

## 15. Documentation updates

Update or add:

```text
docs/NEURON_WORKBENCH.md
docs/PROCESSING_NOTES.md
docs/ARCHITECTURE_LAB.md
docs/ANNOTATION_SCHEMA.md
docs/SOTA_INTEGRATIONS.md
docs/INVERSE_DYNAMICS_EXPORT.md
```

Docs should explain:

- how to create a dataset manifest,
- how to generate current pipeline outputs,
- how to build the dashboard,
- how to serve and autosave,
- how to export annotations,
- how to interpret trace/event labels,
- how architecture runs are compared,
- why denoised outputs are evidence and not ground truth,
- how to prepare for tail-motion inverse dynamics.

---

## 16. Implementation cautions

1. **Do not make denoised video the only view.** Always show raw video next to denoised/evidence views.
2. **Do not optimize only for visual smoothness.** A smooth denoised movie can suppress weak true events or hallucinate structure.
3. **Do not treat deconvolution as exact spike counts.** Use it as an event/activity feature unless validated.
4. **Do not overfit to `calcium_video_2`.** Everything should accept `dataset_id` and manifest paths.
5. **Do not remove legacy workflows until replacements are tested.** Preserve current Groovy/Fiji scripts while adding manifest-aware versions.
6. **Do not commit large TIFFs or generated outputs.** Use synthetic fixtures for tests.
7. **Do not hide uncertain cases.** The UI should support unsure, weak, split-needed, redraw-needed, and duplicate labels.
8. **Do not assume pixel size.** Read it from manifest or leave physical-size filtering disabled.
9. **Do not assume drift is irrelevant.** Estimate and display drift even if correction is optional.
10. **Do not assume all accepted neurons are control-ready.** Trace quality and control-readiness are separate labels.

---

## 17. Source references for SOTA planning

These references motivated the integration order and should be used to guide implementation decisions, not as mandatory dependencies.

- Suite2p documentation: registration, ROI detection, signal extraction, deconvolution, GUI.  
  https://suite2p.readthedocs.io/

- Suite2p parameters and ROI detection docs.  
  https://suite2p.readthedocs.io/en/latest/parameters/  
  https://suite2p.readthedocs.io/en/latest/roidetection/

- CaImAn documentation: motion correction, CNMF/source extraction, deconvolution, online analysis.  
  https://caiman.readthedocs.io/

- CaImAn paper and zebrafish light-sheet online analysis note.  
  https://elifesciences.org/articles/38173

- DeepCAD-RT paper and project page.  
  https://www.nature.com/articles/s41587-022-01450-8  
  https://cabooster.github.io/DeepCAD-RT/

- OASIS online calcium deconvolution.  
  https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005423  
  https://github.com/j-friedrich/OASIS

- PMD / localized penalized matrix decomposition for functional-imaging denoising.  
  https://arxiv.org/abs/1807.06203

- FIOLA for future online fluorescence imaging analysis.  
  https://pubmed.ncbi.nlm.nih.gov/37679524/

- CASCADE calibrated spike inference.  
  https://github.com/HelmchenLabSoftware/Cascade

---

## 18. First Codex task list

Start here. Do not jump to SOTA integrations before completing these foundations.

### Task A — Fix existing correctness issues

1. Fix `worker.py` metric casing.
2. Fix `core/filters.py` CPU import issue.
3. Fix `utils.NumpyEncoder` integer handling.
4. Add `extract_component_detections`.
5. Add tests for all four.

### Task B — Add manifest support

1. Add `neurobench/manifests.py`.
2. Add `schemas/dataset_manifest.schema.json`.
3. Add `examples/dataset_manifest.example.json`.
4. Update `tools/build_neuron_workbench_v2.py` with CLI args.
5. Update `tools/serve_neuron_workbench.py` defaults.

### Task C — Add annotation schema v3

1. Add `schemas/annotations.schema.json`.
2. Add migration utility.
3. Extend browser-side `defaultAnnotations()`.
4. Add UI widgets for trace quality, artifact class, and control readiness.
5. Update exports.

### Task D — Add Architecture Lab skeleton

1. Add `architecture_runs.json` example.
2. Add `tools/build_architecture_run.py` to convert current `review_data.json` to standardized artifacts.
3. Add a second page in the dashboard.
4. Display at least two runs side-by-side if available.
5. Show ROI provenance and run parameters.

### Task E — Add annotation metrics

1. Add `neurobench/annotation_metrics.py`.
2. Add `tools/compute_annotation_metrics.py`.
3. Add Metrics/Audit page.
4. Add CSV/JSON export.

---

## 19. Definition of done for the next major milestone

The next major milestone is complete when:

1. The app can be built from a dataset manifest rather than hard-coded paths.
2. The app has at least two pages: Review and Architecture Lab.
3. `annotations.json` supports v3 labels for neuron existence, event validity, trace quality, artifact class, and control readiness.
4. Current pipeline outputs can be represented as a standardized architecture run.
5. At least two architecture runs can be compared side-by-side.
6. The Metrics/Audit page can summarize partial annotations.
7. Component-level detection utilities exist.
8. The known metric casing and CPU import bugs are fixed.
9. A small synthetic smoke test can build a minimal app and compute metrics.
10. Documentation explains how to run the workflow from manifest creation through annotation export.
