# Neuron Annotation Workbench

The neuron annotation workbench is a local browser tool for reviewing neuron
ROI candidates and candidate firing events from calcium-imaging videos. It is
designed as a high-convenience annotation surface: large video review, trace
inspection, ROI/event labels, notes, keyboard shortcuts, and autosave.

Use this guide when you want to process a video, open the dashboard, review
candidates, compare parameter settings, or export annotation data for the lab.
For a map of the full documentation set, see [README.md](README.md).

For the current cropped resting-video result and an algorithm-first explanation
of the waveforms, thresholds, and yellow event markers, see
[RESTING_VIDEO_ALGORITHM_BRIEF.md](RESTING_VIDEO_ALGORITHM_BRIEF.md).

## Quick Start

For the current resting crop, the normal local path is:

```bash
python3 tools/run_neuron_review_pipeline.py \
  --dataset-manifest Outputs/Manifests/calcium_rest_cropped.dataset.json \
  --fiji /path/to/Fiji.app/ImageJ-linux64
```

```bash
python3 tools/serve_neuron_workbench.py \
  --root-dir Outputs/NeuronReview \
  --port 8765
```

Then open:

```text
http://127.0.0.1:8765/calcium_rest_cropped/app/index.html
```

If the server is already running on another port, use that port instead. The
dashboard can be opened as a static file, but autosave, generation jobs,
trace-materialization, and `architecture_runs.json` updates require the local
server.

## What To Use When

| Goal | Dashboard Page | Notes |
| --- | --- | --- |
| Decide where to go next | `Home` | Shows the active run, next best action, and workflow cards for the full dashboard. |
| Build or compare pipeline stacks | `Pipelines` | Configure stages, save reusable named architectures, and compare completed/generated runs. |
| Plan threshold sweeps or hand-picked variants | `Experiment Lab` | Select a model instance, inspect recommendation cards, plan sweeps/sets/Optuna studies, and launch the first preview locally. |
| Ask an LLM for architecture ideas | CLI handoff | Generate context/prompt files, validate returned JSON, and import proposals into Pipelines. |
| Tune overlays, filters, labels, and manual ROIs | `Review` | Use `Guided` for routine review, `Standard` for common controls, and `Expert` for manual/editing tools. |
| Inspect intermediate outputs frame-by-frame | `Data` | Synchronized raw/intermediate stage grid. |
| Validate candidate neurons and events | `Review` | Main annotation page with compact transport controls and collapsible review/display tools. |
| Track review burden and readiness | `Progress` | Includes tuning gate, robustness examples, validation, and adjudication. |
| Share current progress | `Review Session` panel or `Report` | Export handoff Markdown/JSON or report Markdown. |

## What It Reviews

The current workbench consumes `review_data.json`, generated from:

1. The raw TIFF video.
2. A robust positive z-score stack.
3. Stable ROI candidates built from aggregate spatial/temporal evidence.
4. ROI traces with local background correction.
5. Trace-level robust Kalman baseline estimates and positive innovation event
   calls.

The default calcium-video output is written under:

```bash
Outputs/NeuronReview/calcium_video_2/app/
```

Generated outputs and scientific input data are not intended to be committed to
git.

## Generate The Review Data

Run the Fiji/Groovy generator after the upstream high-pass and candidate
pipeline outputs exist. Prefer the manifest-driven runner for normal work:

```bash
python3 tools/run_neuron_review_pipeline.py \
  --dataset-manifest Outputs/Manifests/calcium_rest_cropped.dataset.json \
  --fiji /path/to/Fiji.app/ImageJ-linux64
```

This writes:

- `review_data.json`: video metadata, ROI footprints, traces, and event data
- `roi_summary.tsv`: compact ROI summary table
- `discovery_suggestions.tsv`: candidate missed-neuron suggestions from
  uncovered evidence maps
- `evidence/*.png`: projection and discovery maps for coverage auditing
- `frames/frame_###.png`: browser-friendly video frames
- `parameters.txt`: generation parameters

## Build The Workbench UI

The v2 workbench builder is stdlib-only Python. It reads the generated
`review_data.json` and writes `index.html`, `workbench.css`, and
`workbench.js`:

```bash
python3 tools/build_neuron_workbench_v2.py
```

The builder also accepts explicit paths or a dataset manifest:

```bash
python3 tools/build_neuron_workbench_v2.py \
  --dataset-manifest examples/dataset_manifest.example.json
```

Useful explicit arguments:

- `--app-dir`: output directory containing `index.html` and autosaved
  `annotations.json`
- `--review-data`: source `review_data.json`
- `--dataset-manifest`: dataset metadata and standard paths
- `--architecture-runs`: optional standardized run comparison manifest

The builder writes `architecture_runs.json` into the app directory, so the
local server can expose the attached architecture metadata alongside
`annotations.json`.

Static dashboard code now lives in `neurobench/workbench/assets/`. The builder
copies those tracked assets into the app directory, which makes UI changes much
easier to review than editing large embedded Python strings.

The JavaScript source of truth is `neurobench/workbench/assets/src/`.
`neurobench/workbench/assets/workbench.js` is the generated served bundle. Edit
ordered source modules, then rebuild/check the bundle:

```bash
python3 tools/build_workbench_assets.py
python3 tools/build_workbench_assets.py --check
```

The HTML shell is packaged as `neurobench/workbench/assets/workbench.html`; the
legacy builder script is only a CLI wrapper around the package builder.

For an optional real-browser smoke check, run:

```bash
NEUROBENCH_BROWSER_SMOKE=1 python3 -m pytest tests/test_workbench_browser_smoke.py
```

The smoke test builds a tiny workbench, opens it with Firefox headless, and
checks that a browser screenshot is produced. It is opt-in because local desktop
Firefox sessions can interfere with headless screenshot mode on some machines.

The v1 builder configures and exports a static/local browser workbench. It does
not execute image-processing pipelines in the browser. Pipeline runs still
happen through Fiji/Groovy, Python, or imported external-tool outputs; the
browser consumes the resulting JSON, manifests, frames, evidence maps, and
autosaved annotations.


## Lightweight Sweep Review Payloads

Large Gamma CFAR sweeps can keep full `review_rois.json` files for compatibility while also exposing lighter browser payloads:

- `review_rois_summary_file`: ROI geometry, priority, stencil status inputs, and compact event summaries without full traces.
- `review_trace_shards_dir`: one JSON shard per ROI containing trace arrays loaded only when that ROI is selected.
- `stencil_gap_report_file`: rough low-coverage boxes inside the saved anatomy stencil for Review Triage.

Build or refresh these sidecars for an existing app with:

```bash
python3 tools/build_review_roi_sidecars.py --app-dir Outputs/NeuronReview/051626_3_right_50hz/app
```

Export CFAR contrast maps for Data Compare with:

```bash
.venv-neurobench/bin/python tools/export_cfar_contrast_maps.py \
  --app-dir Outputs/NeuronReview/051626_3_right_50hz/app \
  --sweep-root Outputs/GammaCFAR/051626_3_right_50hz/gamma_cfar_cascade_grid_50hz_v2 \
  --all-runs \
  --chunk-frames 10 \
  --normalization-sample-stride 10
```

The Review Overlap and Triage pages prefer the summary payload. The Inspect page loads an ROI trace shard on demand, so changing sweeps no longer requires pulling every trace-heavy ROI record into memory. Data Compare consumes the contrast-map frame sequences directly from `artifacts.intermediates`.

## Dataset Intake For Future Data

Before running heavy processing on a new local or public dataset, create a
metadata-only intake manifest:

```bash
neurobench dataset intake \
  --source-template dandi-nwb \
  --dataset-id example_public_fish \
  --raw-video Inputs/Public/example.nwb \
  --frame-rate-hz 50 \
  --pixel-size-microns 0.5 \
  --out Outputs/Manifests/example_public_fish.dataset.json \
  --report-out Outputs/Manifests/example_public_fish.intake_report.json
```

The intake command does not download large public data. It records expected
paths, frame rate, pixel size, source type, and readiness checks so conversion
or optional dependency gaps are visible before the dashboard is built.

## Process A New Dataset

Use a dataset manifest when cycling through multiple TIFF videos. For example,
to process `calcium_rest_cropped.tif` with the current Fiji/Groovy pipeline:

```bash
python3 tools/create_dataset_manifest.py \
  --out Outputs/Manifests/calcium_rest_cropped.dataset.json \
  --dataset-id calcium_rest_cropped \
  --raw-video "Inputs/050126/050126/calcium_rest_cropped.tif" \
  --app-dir Outputs/NeuronReview/calcium_rest_cropped/app \
  --frame-rate-hz 5.0 \
  --pixel-size-microns 0.5
```

```bash
python3 tools/run_neuron_review_pipeline.py \
  --dataset-manifest Outputs/Manifests/calcium_rest_cropped.dataset.json \
  --fiji /path/to/Fiji.app/ImageJ-linux64
```

The runner executes high-pass filtering, event-preserving denoising, candidate
generation, temporal scoring, review-data generation, proposal/artifact
analysis, the v2 workbench build, and the multi-dataset index build. Use
`--dry-run` to print the exact commands without running Fiji.

## Run With Autosave

Use the local server for persistent annotation saving:

```bash
python3 tools/serve_neuron_workbench.py --port 8765
```

Then open:

```text
http://127.0.0.1:8765/
```

The server supports:

- `GET /`: workbench app
- `GET /annotations.json`: current annotation state
- `GET /architecture_runs.json`: attached architecture-run metadata
- `GET /api/environment`: local Fiji/Python/GPU readiness for generation
- `POST /api/jobs/generate-view`: start a whitelisted local generation job
- `POST /api/jobs/generate-preview`: start the same run-aware generator without
  rebuilding the multi-dataset index
- `POST /api/materialize-traces`: extract traces for manual/edited virtual ROI
  masks from the local raw video and save them into `annotations.json`
- `POST /api/llm-proposals/import`: validate an LLM architecture proposal pack
  and merge planned runs/templates into `architecture_runs.json`; this endpoint
  imports metadata only and does not execute arbitrary commands. In Experiment
  Lab, this powers the **Import Full To Dashboard** and **Import Candidates To
  Dashboard** buttons.
- `GET /api/jobs` and `GET /api/jobs/<job_id>`: generation status and logs
- `PUT /annotations.json`: autosave endpoint

Writes are atomic: the server writes a temporary JSON file and then replaces
`annotations.json`.

When `NEUROBENCH_OWNER_TOKEN` is set in the server environment, generation
endpoints require the dashboard's **Unlock Generation** control. The browser
sends the token as `X-Neurobench-Owner-Token`; the token is never exposed by
`GET /api/environment`. This is the recommended mode before sharing the
dashboard through a tunnel.

Generated pipeline runs are written under the dataset app instead of
overwriting the baseline review data:

```text
Outputs/NeuronReview/<dataset>/app/generated_runs/<run_id>/
```

Each generated run receives its own `review_data.json`, frames, intermediate
frame exports, status, logs, and app URL in `architecture_runs.json`.

For a multi-dataset workbench index, serve the review root instead:

```bash
python3 tools/serve_neuron_workbench.py \
  --root-dir Outputs/NeuronReview \
  --port 8765
```

Then `/` shows all processed dataset dashboards, and autosave writes only to
safe per-dataset files under `*/app/annotations.json` or
`*/app/architecture_runs.json`.

For short-term peer access, run the server locally and expose it with a tunnel.
Use owner-only generation so peers can inspect dashboards without launching
jobs on your machine:

```bash
export NEUROBENCH_OWNER_TOKEN="choose-a-private-token"
python3 tools/serve_neuron_workbench.py \
  --root-dir Outputs/NeuronReview \
  --host 127.0.0.1 \
  --port 8765
```

Then expose `http://127.0.0.1:8765` with Cloudflare Tunnel, ngrok, Tailscale
Funnel, or another tunnel tool and share the generated public URL.

## Suggested Review Workflow

1. Set `Reviewer` before labeling. This stamps new ROI, event, suggestion,
   manual ROI, and split/merge decisions with `reviewer_id`.
2. Start in `Guided` mode with the `Fast triage` workflow preset.
3. Use the `Review Session` panel to confirm autosave, active run, current
   queues, and tuning-readiness progress.
4. Label the `Next annotation batch` queue before broad parameter tuning.
   The first practical tuning gate is about `20` reviewed ROIs and `20`
   reviewed events.
5. Switch to `Missed neuron search` or `Artifact cleanup` when the main queue
   is too noisy or obvious candidates are missing.
6. Use `Data` to inspect raw/intermediate frame outputs when a parameter
   change appears to introduce artifacts.
7. Use `Progress` before exporting. Check reviewer provenance, accepted
   events, control-ready ROIs, robustness examples, and adjudication results.
8. Export a `Review Session` handoff or `Report` Markdown when sharing status
   with a lab-mate.

## User Experience Notes

- Expert controls are intentionally hideable. New reviewers should stay in
  `Guided` mode until they need generation, detector, manual ROI, split/merge,
  or export tools.
- `Home` is now a guided launchpad. It prioritizes one next action and the
  four main workflows instead of showing every metric up front.
- `Experiment Lab` opens as a wizard-first planner: choose a base pipeline,
  choose a strategy, add named sets or search metadata, preview runs, then save
  or generate a first preview. Advisor, LLM, and diagnostics panels live behind
  Expert disclosures.
- `Data` distinguishes true missing frame outputs from summary-only stages. A
  stage such as component extraction, background-ring correction, Kalman event
  scoring, or priority ranking may show a diagnostic summary when no video
  artifact was exported.
- Overlay opacity, selected-ROI fill, outline width, and focus mode are
  workflow preferences. They do not alter the saved scientific labels.

- Review now shows a raw projection strip for quick inspection of raw mean, raw
  max, temporal standard deviation, z-max sample, and candidate overlay without
  leaving the ROI workflow. Click the candidate-overlay thumbnail or open
  `Review > Candidate Overlay` (`#candidate-overlay`) for the full sweep overlay
  view.
- Stencil-aware queues are available in Expert queue controls: `Inside stencil`,
  `Outside stencil`, and `Near stencil edge`. The stencil is an inspection prior
  for review prioritization, not ground-truth segmentation.
- Frame labels include seconds when `frame_rate_hz` is present, so 50 Hz and
  future high-rate datasets can be inspected without mentally converting frames.
- The trace plot is interactive: use the trace window controls and frame marker
  to navigate temporal context while validating event markers.
- Manual ROI and mask editing are review aids for missed candidates. When
  materialized through the local server, their traces are saved with the
  annotation state so they can be revisited.

## Review Layout

The Review page is organized for repeated ROI triage:

- `Theme` can be set to `System`, `Light`, or `Dark`. The setting is saved with
  dashboard preferences and applies across all tabs.
- `Guided` mode is the default low-clutter review surface. It keeps playback,
  video, trace, event timeline, selected ROI context, core ROI/event labels,
  notes, simple navigation, and guided review visible.
- `Standard` mode adds common run-selection and display controls for routine
  parameter inspection.
- `Expert` mode reveals generation controls, detector thresholds, Kalman
  trace parameters, overlay tuning, manual ROI tools, split/merge tools,
  parameter snapshots, export controls, and raw parameters.
- The mode setting is a user workflow preference saved in `annotations.json`;
  it is not a scientific annotation label.
- the `Reviewer` field stores the current reviewer ID in settings and stamps
  subsequent ROI, event, suggestion, virtual ROI, and split/merge edits with
  `reviewer_id` plus `updatedAt`
- the Report page summarizes reviewer-stamped label counts so contribution and
  adjudication coverage can be checked without opening raw JSON
- the Report page also counts reviewed labels missing a reviewer ID by label
  type, so older or unstamped ROI, event, suggestion, and split/merge decisions
  can be revisited before inter-rater comparison
- the `Review Session` panel gives a compact readiness check: reviewer ID,
  autosave status, active run, active queues, tuning-label progress, missing
  reviewer provenance, and export readiness
- `Handoff Markdown` and `Handoff JSON` export the current review state, active
  queues, progress, and suggested next annotation batch for lab-mate review or
  session continuation
- the primary video panel remains the spatial reference for frame playback,
  ROI overlays, and discovery overlays
- the primary transport row stays compact: play, frame slider, fit width,
  fullscreen, and screenshot remain visible while secondary review tools live
  in collapsible groups
- the trace panel stays adjacent to the selected ROI workflow so candidate
  events can be checked against frame context
- the ROI/event controls, labels, queues, filters, and notes are grouped into
  review sections instead of one long control stack
- secondary review sections are collapsible so a reviewer can keep high-use
  labeling controls visible while hiding lower-frequency panels such as
  discovery, advanced scoring, or audit details
- collapse state is a workflow preference, not an annotation label
- the `Next annotation batch` queue ranks unlabeled ROIs that are most useful
  for the first tuning pass, using event support, trace SNR, local coherence,
  artifact cues, and existing priority scores
- `Guided Review` turns that batch into one task at a time, with a prompt,
  reason badges, progress goals, and next/previous task controls
- Each guided task also has task-specific quick-decision buttons. ROI tasks can
  be accepted, rejected, or marked unsure; event tasks can be accepted,
  rejected, or marked unsure; suggestion tasks can be marked missed/artifact/
  unsure or promoted. After a quick decision, the guide advances to the next
  remaining task.
- Event review has its own queue filter and previous/next queue navigation.
  This lets a reviewer move across ROIs for unlabeled, accepted, rejected,
  unsure, high-z, missing-reviewer, or reviewer-specific event labels without
  manually selecting each ROI first. Event `Accept + Next`, `Reject + Next`,
  and `Unsure + Next` follow this event queue, so labeling can continue across
  ROI boundaries.
- Workflow presets configure several controls together for common tasks:
  `Fast triage`, `Event validation`, `Missed neuron search`, `Artifact cleanup`,
  and `Mask editing`.
- The `Shortcuts` button or `?` key opens a compact keyboard reference overlay.
- The `Jump` box accepts ROI IDs, `roi:ID`, `f120`, or `frame:120` for direct
  navigation during review sessions.
- `Undo Label` or `Ctrl/Cmd Z` restores the previous in-session ROI, event, or
  suggestion label change, including guided quick decisions and suggestion
  promotions.
- `Bookmark`, `Open Mark`, and `Delete Mark` provide a small revisit list for
  frames, selected ROIs, selected events, and suggestions within the active run.
- In Expert mode, shift/ctrl/cmd multi-select can label several selected ROIs
  at once with `Accept Selected`, `Reject Selected`, or `Unsure Selected`, in
  addition to the existing group/action/split/merge controls.
- Common compound decisions have one-click presets: `Strong Neuron + Next`
  accepts the ROI, marks trace/control quality, sets high confidence, and adds
  evidence tags; `Artifact ROI + Next` rejects the ROI with artifact metadata;
  `Artifact + Next` rejects the selected event as an artifact and advances
  through the event queue.
- Expert queue controls show the selected ROI's position in the filtered
  queue and provide `Prev Queue` / `Next Queue` buttons for mouse-first review.
- Reviewer-aware queues can show reviewed ROIs missing `reviewer_id`, ROIs
  reviewed by the current `Reviewer`, or ROIs reviewed by someone else.
- Discovery suggestion queues include the same reviewer-aware filters, which
  makes missed-neuron and artifact suggestion provenance easier to audit.
- Discovery suggestion review now has previous/next queue navigation and
  label-and-advance buttons for promote, missed-neuron, duplicate, artifact,
  and unsure decisions. This makes missed-neuron cleanup closer to the ROI and
  event review flow.
- Expert reviewer tools can stamp the current `Reviewer` onto selected
  reviewed labels or all reviewed labels missing reviewer provenance. These
  backfill actions use the same in-session undo stack as label changes.
- `Next Missing Reviewer` jumps directly to the next reviewed ROI, event,
  suggestion, virtual ROI, or split/merge decision that still needs a reviewer
  stamp.
- The Export panel and Report page can download a reviewer provenance audit
  JSON with stamped/missing counts, per-reviewer contribution counts, and a
  machine-readable list of reviewed labels that still lack `reviewer_id`.
- The Export panel can also download the active ROI, event, or suggestion
  queue as TSV. These queue exports preserve the current filters and sort order,
  making it easier to share a focused review worklist or save the current
  handoff state.

The reorganization should not change the saved scientific meaning of
`annotations.json`; it only makes the same review operations easier to scan and
use during long labeling sessions.

Expert mode includes an Autosave Recovery panel. The browser keeps a small
local recovery history of recent annotation states before autosave writes. A
recovery point can be restored back into the current browser/server state or
downloaded as a standalone JSON snapshot.

## Build A Review Batch

Use the annotation-batch helper when you want a concrete labeling worklist
outside the browser:

```bash
python3 tools/build_annotation_batch.py \
  --review-data Outputs/NeuronReview/calcium_rest_cropped/app/review_data.json \
  --annotations Outputs/NeuronReview/calcium_rest_cropped/app/annotations.json \
  --out Outputs/NeuronReview/calcium_rest_cropped/annotation_batch.json \
  --out-dir Outputs/NeuronReview/calcium_rest_cropped/annotation_batch
```

The batch contains prioritized ROIs, event frames, and discovery suggestions,
with short reasons for each item. The first useful tuning milestone is about
`20` reviewed ROIs and `20` reviewed events; before that, parameter comparisons
are still mostly qualitative.

When `--out-dir` is supplied, the helper also writes `review_tasks.tsv` and
`review_task_features.tsv`. Those feature rows are the bridge to a later
active-learning ranker; until enough labels exist, they should be interpreted
as transparent heuristic guidance.

## Generate A Review Report

After or during a review session, build a compact report for lab discussion:

```bash
python3 tools/build_review_report.py \
  --review-data Outputs/NeuronReview/calcium_rest_cropped/app/review_data.json \
  --annotations Outputs/NeuronReview/calcium_rest_cropped/app/annotations.json \
  --out-json Outputs/NeuronReview/calcium_rest_cropped/review_report.json \
  --out-md Outputs/NeuronReview/calcium_rest_cropped/review_report.md
```

The dashboard `Report` tab shows the same summary in-browser and can download
a Markdown version. The report is meant to communicate review progress,
accepted outputs, next candidates to inspect, and recommended next actions.

## Annotation Files

`annotations.json` stores:

- ROI labels: accept, reject, unsure, hidden/deleted
- ROI notes
- v3 ROI fields: neuron state, trace quality, control readiness, artifact
  class, identity group, needs-action flag, confidence, and reason tags
- Event labels: accept, reject, unsure
- Event notes
- v3 event fields: event state, event type, timing quality, confidence, and
  reason tags
- Discovery suggestion labels: promoted, missed neuron, artifact, unsure
- Discovery suggestion confidence and reason tags
- Artifact classes such as vessel/static structure, impulse noise, border
  artifact, saturation/bright blob, or uncertain artifact
- Promoted missed-neuron footprints copied from discovery suggestions
- Intent-first virtual ROI merges in `virtualRois`, with source ROI IDs and
  identity-group metadata
- Manual ROI footprints in `virtualRois`, with `roi_kind` values such as
  `manual_center`, `manual_circle`, or `manual_lasso`
- Review-session action counts for lightweight workflow auditing
- Review settings such as thresholds, display settings, and queue mode

The browser also keeps a localStorage backup. If the app is opened directly as
a file instead of through the server, export TSVs before closing the browser.

## Keyboard Shortcuts

- `Space`: play/pause
- Left/right arrows: previous/next frame
- `j` / `k`: next/previous ROI in the active queue
- `n` / `p`: next/previous event in the selected ROI
- `N` / `P`: next/previous event in the active event queue, across ROIs
- `.` / `,`: next/previous discovery suggestion in the active suggestion queue
- `a`: accept selected ROI
- `r`: reject selected ROI
- `u`: mark selected ROI unsure
- `e`: accept selected event
- `x`: reject selected event
- `g` / `G`: promote the selected discovery suggestion, or promote and advance
- `m` / `M`: mark the selected suggestion as missed, or mark and advance
- `f`: fullscreen video panel

## Export

The workbench exports three TSVs:

- ROI annotations: one row per ROI with label, notes, footprint metadata, and
  event count
- Event annotations: one row per candidate event with ROI ID, frame, label,
  amplitude, and z score
- Discovery annotations: one row per missed-neuron suggestion with promotion,
  artifact, notes, score, and provenance fields
- Full JSON: migrated v3 annotation state, including review settings

Use these exports as the bridge into downstream inverse-dynamics analysis.

For an inverse-dynamics-ready trace/event table:

```bash
python3 tools/export_inverse_dynamics.py \
  --review-data Outputs/NeuronReview/calcium_video_2/app/review_data.json \
  --annotations Outputs/NeuronReview/calcium_video_2/app/annotations.json \
  --out-dir Outputs/NeuronReview/calcium_video_2/inverse_dynamics
```

For file-based exports outside the browser, use:

```bash
python3 tools/export_annotations.py \
  --review-data Outputs/NeuronReview/calcium_video_2/app/review_data.json \
  --annotations Outputs/NeuronReview/calcium_video_2/app/annotations.json \
  --out-dir Outputs/NeuronReview/calcium_video_2/exports
```

For review-burden and annotation-count summaries:

```bash
python3 tools/compute_annotation_metrics.py \
  --review-data Outputs/NeuronReview/calcium_video_2/app/review_data.json \
  --annotations Outputs/NeuronReview/calcium_video_2/app/annotations.json \
  --out Outputs/NeuronReview/calcium_video_2/annotation_metrics.json
```

## Discovery Mode

Discovery mode is meant to address missed neurons. It adds evidence maps and
candidate suggestions that are not already covered by the current ROI set.

Available evidence maps include:

- raw mean projection
- raw max projection
- raw temporal standard deviation
- robust-z max projection
- peak-count projection
- uncovered robust-z score
- local contrast proxy
- local temporal correlation
- event-support map
- combined discovery score

Use the evidence-map overlay to inspect regions where the video has strong
signal but no accepted ROI. Suggestions can be promoted, marked as missed
neurons, marked as artifacts, or left unsure. Promoted suggestions are saved in
`annotations.json`; they become part of the review record and can be used to
tune the next candidate-generation pass.

New review-data builds also attach transparent scoring fields to ROIs and
suggestions: priority score, local correlation, event support, trace SNR,
background correlation, compactness, and artifact risk. The queue can sort by
these fields, but the scores are only review guidance, not ground truth.

## Pipelines

The same generated app now includes a Pipelines page at `#pipelines`
(`#architecture` remains a compatibility alias). It displays standardized architecture-run metadata: run ID,
parameters, ROI/event/suggestion counts, evidence maps, and artifact paths.

The current review output is automatically represented as a baseline run. To
write an explicit run manifest:

```bash
python3 tools/build_architecture_run.py \
  --review-data Outputs/NeuronReview/calcium_video_2/app/review_data.json \
  --out Outputs/ArchitectureRuns/calcium_video_2/architecture_runs.json
```

When two or more runs are attached, Pipelines provides Run A / Run B
selectors and a comparison table for candidate totals, accepted/control-ready
counts, and review-burden metrics.

Pipelines also provides a synchronized A/B Review viewer. It loads two
generated run `review_data.json` files into a browser cache and shows the same
frame from each run side-by-side, including ROI outlines and current-frame
event highlights. This is meant for quick visual comparison of parameter
settings without disturbing the active Review page. The main Review and Data
context changes only when `Use A In Review/Data` or `Use B In Review/Data` is
pressed. `Next Difference` and `Prev Difference` jump to frames where the two
loaded runs have different candidate event counts.

Use `tools/merge_architecture_runs.py` to combine separate method outputs into
one manifest for comparison.

Pipelines also includes a Build mode for configuring planned run
manifests. Build mode is a planning/export surface: it captures dataset
references, proposed pipeline options, output locations, and run metadata so the
configuration can be saved or handed to command-line tooling. In v1 it does not
run Fiji, Python, Suite2p, CaImAn, PMD, OASIS, or any other pipeline inside the
browser. The `Pipeline Identity` section lets a user name the architecture,
assign a stable local ID, add a short description, save it as a reusable local
template, and send that template directly to Experiment Lab.

Planned pipeline manifests should be treated as requested or proposed work
until a real run artifact exists. Once a pipeline has actually executed, attach
its produced architecture-run manifest to compare it with the current baseline.
Review, Pipelines, and Data share the active run selection. If a
selected run has reachable `review_data.json`, Review can load it; if it is
only planned, the app can start a local Generate View job through the server.
Generation is still local and whitelisted: the browser can choose the run and
backend, but it cannot submit arbitrary shell commands. The default backend
uses the existing Fiji/Groovy pipeline and exports QC intermediate frame tiles;
the Python GPU option is shown only as an explicit backend choice and reports
CUDA readiness before starting.

LLM-assisted architecture planning is also local and file-based. Use
`neurobench llm context` to create a provider-neutral context and prompt, then
import LLM-returned JSON with `neurobench llm import-proposals`. Imported
proposals become normal saved architectures and planned runs in
`architecture_runs.json`. The importer enforces known stages, parameter ranges,
concrete sweep step IDs, and a default limit of `4096` parameter combinations
per architecture. Use `neurobench llm run-proposals` for local executable smoke
tests before generating full Review views.

Pipelines includes a Parameter Experiments table. Use it to label
generated runs or sweep outputs as `looks best`, `too noisy`, `too strict`,
`artifact heavy`, or `needs review`, then open the same active run in Review or
Data. These labels are workflow notes stored with dashboard settings,
not scientific ground truth.

## Experiment Lab

The generated app includes an Experiment Lab page at `#experiments`. It is a
planning surface for parameter studies:

- `Model instance` selects a saved architecture template, preset, active run, or
  current Build Pipeline stack as the experiment base.
- `Sweep axes` mode uses the current Build Pipeline stack and configured sweep
  axes to create a cartesian set of planned architecture runs.
- `Named sets` mode creates hand-picked one-parameter variants with readable
  labels for small targeted comparisons.
- `Optuna plan` mode stores study metadata and numeric search-space bounds for
  later optimization tooling. It does not run Optuna in the browser.
- `Save Plan` writes planned runs and optimization-study metadata into
  `architecture_runs.json` through the local server; static mode downloads the
  JSON.
- `Generate First Preview` saves the plan, activates the first planned run, and
  calls the same local generation endpoint used by Pipelines.

The top of Experiment Lab is an `Experiment Command Center`. It summarizes
generated/planned/failed runs, the draft sweep budget, imported LLM proposal
sets, annotation-aware recommendations, the active run's utility score, and the
active run's delta against the current baseline. Use this section first when
deciding whether to label more examples, try high-recall discovery, add artifact
suppression, or generate a preview.

The command center includes an `Evaluation Checklist`, `Decision Matrix`,
`Session Recipe`, `Prioritized Action Queue`, `Parameter Sensitivity` panel,
`Follow-up Planner`, and `Parameter Coverage Map`. The session recipe stores a
short objective, highlights the highest-priority next action, summarizes the
first runs and parameter moves to try, and can be copied or downloaded as
Markdown/JSON for lab discussion or external LLM planning. `LLM Architecture
Request` turns the same state into a provider-neutral prompt with a request
type, extra constraints, current stack, available stage IDs, and the expected
proposal JSON shape. `LLM Proposal Intake` lets you paste returned proposal JSON
for browser-side sanity checks, proposal/sweep summaries, and a copyable local
import command before running the authoritative CLI importer. If the pasted JSON
has errors or warnings, the intake panel can also copy or download a repair
prompt that lists the exact issues, known stage IDs, and expected schema shape
for the next LLM round. The same panel includes proposal triage: a sorted review
table that marks proposals as import candidates, budget-review items, or
repair-first items and can export a short Markdown triage note. Candidate-only
proposal-pack export drops repair-first and budget-review proposals into a
normalized JSON pack for the first local import attempt, plus a Markdown note
showing what was kept and dropped. Import Readiness then shows whether full or
candidate import is the safer next step and exposes copyable commands for both
the full returned pack and the candidate-only pack. Post-Import Plan gives the
follow-up sequence for saving the JSON, importing proposals, reloading the
dashboard, generating a preview, and optionally running the local executable
proposal experiment summary. LLM Proposal Lifecycle tracks each proposal from
pasted/imported to generated, promising, rejected, repair-needed, or discussed,
then converts review outcomes and failure signals into a follow-up LLM prompt.
The action queue merges readiness blockers,
recommendations, sensitivity, follow-up suggestions, and coverage gaps into a
ranked list of concrete next steps with direct actions. Queue items can be
marked `Done`, `Snoozed`, or reopened so stale advice does not dominate the
view. The `Focus` selector switches between `Guided`, `Diagnostics`, and `All`
views so the page can stay compact during routine use. Recent queue state
changes are shown in `Action History` and can be exported as TSV for handoff.
The checklist turns common blockers into visible gates: enough review seed
labels, at least one generated preview, a shortlist, an active-run note,
Data outputs, and a manageable sweep budget. The decision matrix ranks
runs by shortlist state, utility score, and generated readiness, then lets you
label runs directly and export a TSV summary for spreadsheets or meeting notes.
The sensitivity panel groups explicit sweep/named-set parameter changes and
reports the best run, average utility, score spread, and a suggested next move
for each parameter. The follow-up planner converts those signals into small
local refinements that can be added as named sets or a sweep axis. The coverage
map finds tunable numeric parameters in the current stack that have not been
tested yet and can add small probe sets or a sweep axis.

The command center also includes `Lab Share And LLM Handoff`. Use it to add a
short note to the active run, shortlist promising runs, download a concise
Markdown brief for lab discussion, or download a provider-neutral JSON context
for external LLM feedback. The JSON context includes dataset metadata, active
run parameters, utility-score components, baseline deltas, annotation summary,
session recipe, LLM prompt request, recommendations, planned-run budget,
shortlist entries, proposal-intake status, and imported proposal set summaries.

Experiment Lab schedules planned runs for the local, whitelisted generation
backend. It does not execute arbitrary browser-side code.

## Progress

The generated app also includes a Progress page at `#progress`
(`#metrics` remains a compatibility alias). It
summarizes live annotation progress from the current browser/server state:

- ROI, event, and discovery suggestion labels
- trace-quality labels
- control-readiness labels
- candidate burden, such as candidates per accepted ROI/event

Use this page during review sessions to see whether labeling is converging or
whether the candidate generator is creating too much review burden.

Progress also includes three workflow panels:

- `Robustness Example Gallery` jumps to representative strong, uncertain,
  artifact-like, merged-cluster, event-supported, and missed-neuron examples.
- `Validation And Real-Time Readiness` summarizes pipeline validation, 100 Hz
  latency metadata, GPU-sensitive stages, and the synthetic latency smoke-test
  command.
- `Adjudication Comparator` compares two annotation JSON files locally in the
  browser and creates a disagreement queue for final review.

The Review queue also includes v3-specific filters for needs-action ROIs,
control-ready ROIs, problematic traces, priority score, local correlation,
event support, trace SNR, and artifact risk.

## Inter-Rater Comparison

To compare two reviewer annotation files and produce an adjudication queue:

```bash
python3 tools/compare_annotations.py \
  --annotations-a reviewer_a_annotations.json \
  --annotations-b reviewer_b_annotations.json \
  --reviewer-a reviewer_a \
  --reviewer-b reviewer_b \
  --run-id current_review_pipeline \
  --out-dir Outputs/AnnotationAgreement/current_review_pipeline
```

The tool writes:

- `agreement_report.json`: full machine-readable agreement metrics
- `agreement_report.md`: concise human-readable summary
- `disagreement_queue.tsv`: flat conflict queue for adjudication, including
  file-level reviewer names plus any per-item `reviewer_id` and `updatedAt`
  stamps from the dashboard

The agreement report also includes reviewer-provenance coverage for each input
file, so missing `reviewer_id` stamps are visible before adjudication.
Add `--require-reviewer-provenance` when you want the command to write the
reports but exit nonzero if either input has reviewed labels missing
`reviewer_id`.

## Data

The generated app includes a Data page at `#data`
(`#process` remains a compatibility alias). It audits lightweight
video and candidate-set properties such as ROI size, trace noise, event density,
discovery burden, artifact cues, and evidence-map thumbnails.

Data uses one synchronized frame slider for the raw video and any
generated intermediate stage frames declared in `artifacts.intermediates`.
Stages without exported browser-readable frames remain visible as missing-output
tiles.

The Data page also includes `#data-compare`, a two-column comparison view for
inspecting the raw video beside one selected synchronized stage output. It can
switch to an overlap mode for alignment inspection. Gamma CFAR contrast maps are
exposed as `cfar_contrast_map` frame-sequence intermediates and default to the
large-reference CFAR contrast view when available.

Data also includes a Discovery and Artifact Triage section. It surfaces
high-priority missed-neuron candidates and ROI artifact-risk reasons such as
large/merged footprints, low local coherence, background correlation,
elongation, near-border location, and existing artifact labels.

It also includes `Process Decision Support`, which mirrors the active run from
Architecture/Experiment Lab. This card reports readiness, utility score,
missing browser-readable intermediate outputs, and the same annotation-aware
recommendations used in Experiment Lab so that process inspection stays tied to
the architecture being tested.

When available, that triage table is loaded from
`analysis/proposal_analysis.json`, which is produced by:

```bash
python3 tools/build_proposal_analysis.py \
  --review-data Outputs/NeuronReview/calcium_rest_cropped/app/review_data.json \
  --annotations Outputs/NeuronReview/calcium_rest_cropped/app/annotations.json \
  --architecture-runs Outputs/NeuronReview/calcium_rest_cropped/app/architecture_runs.json \
  --run-id current_review_pipeline
```

The same command also writes `artifact_classifier.tsv` and
`missed_neuron_proposals.tsv` for sharing or spreadsheet review.

If `pixel_size_microns` is present in the dataset manifest, the QC page also
reports equivalent ROI diameters in microns. Without that manifest field,
physical-size checks are intentionally disabled.

## Selected ROI Context

The Review page includes a selected-ROI context panel beneath the trace:

- a cropped view around the selected ROI
- the ROI footprint overlaid on the crop
- equivalent ROI diameter in pixels and microns when pixel size is known
- peak score, trace noise, and event count
- priority score, local correlation, event support, background correlation,
  trace SNR, and artifact risk
- an event-centered cropped filmstrip from roughly `t-5` to `t+10`

Use this panel to decide whether a yellow event marker corresponds to a compact
local fluorescence change, shared background motion, or impulse/static artifact.

The Selected ROI Context panel is part of the Review layout rather than a modal
or hover-only affordance. It should remain reachable by keyboard navigation,
expose text labels for the selected ROI and metrics, preserve visible focus
state for controls, and provide non-color cues for event/ROI status wherever
possible. The crop and filmstrip are visual evidence; the adjacent metric text
is the accessible summary that should be used in testing and screen-reader
review.

## Intent-First ROI Editing

The Review page supports shift/ctrl/cmd multi-select in the ROI list or overlay.
Selected ROIs can be assigned the same `identity_group`, marked with a shared
`needs_action`, or saved as a virtual merge. Virtual merges are stored only in
`annotations.json`; the source `review_data.json` footprints remain unchanged.

Expert mode also includes manual ROI creation tools:

- `Center`: click a neuron center to create a circular manual ROI using the
  current manual radius.
- `Circle`: click-drag to define a circular manual ROI.
- `Lasso`: draw a freeform footprint on the video overlay.
- `Missed Neuron Workflow`: draw/select a candidate and store rough or precise
  event windows with `start_frame`, `end_frame`, `precision`, `state`,
  reviewer provenance, and timestamp metadata.

Manual ROIs are saved as annotation-layer `virtualRois`; they do not rewrite
`review_data.json`. They can be selected, labeled, annotated with confidence
and reason tags, and exported. In local-server mode, `Materialize Traces`
extracts raw, background, dF/F, baseline, event, and z traces from the raw
video for unmaterialized manual or edited virtual ROIs, then saves those trace
fields back into `annotations.json`.

Expert mode also includes ROI brush editing. `Brush add` and `Brush erase`
create or update an annotation-layer edited copy of the selected ROI with
`roi_kind: manual_edit` and `provenance: roi_brush_edit`. The original detector
footprint remains unchanged, so mask refinement stays auditable. Brush edits
keep a short per-ROI geometry history for `Undo Mask`, and edited masks that
were copied from detector ROIs can be reset with `Revert To Source`. Any mask
geometry change clears previously materialized trace fields so traces are not
mistakenly reused after footprint edits.

## Troubleshooting

- If autosave is not active, confirm the page URL starts with
  `http://127.0.0.1:8765/`.
- If the video does not appear, regenerate the frame PNGs with
  `generate_neuron_review_app.groovy`.
- If the ROI list is empty, check queue filters such as minimum area, minimum
  events, or hidden/deleted view.
- If smaller neurons appear under-detected, rerun the high-recall Gamma CFAR
  grid or lower `component_min_area_px`, `support_min_frames`, and `seed_z` for
  the component stage before rebuilding review data.
- If you cannot find the candidate overlay, use the `Candidate Overlay` subtab
  under Review or go directly to `#candidate-overlay`.
- If discovery suggestions look too broad, treat them as audit regions rather
  than final ROIs; mark artifacts and promote only visually plausible neurons.
- If annotations appear stale, inspect
  `Outputs/NeuronReview/calcium_video_2/app/annotations.json`.
