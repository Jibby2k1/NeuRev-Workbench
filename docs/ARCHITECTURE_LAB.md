# Architecture Lab

Architecture Lab is the comparison page inside the neuron workbench. It is
available at:

```text
http://127.0.0.1:8765/#architecture
```

The page consumes standardized architecture-run metadata. A run can represent
the current Fiji/Groovy pipeline, a Python CFAR pipeline, or future imports such
as Suite2p, CaImAn, PMD, OASIS, or denoising models.

Suite2p outputs can be converted with `tools/import_suite2p_run.py`,
PMD-denoised videos can be attached with `tools/import_pmd_run.py`, and OASIS
trace outputs can be attached with `tools/import_oasis_run.py`; see
`docs/SOTA_INTEGRATIONS.md`.

Architecture Lab has two related modes:

- Compare mode reads completed architecture-run manifests and reports candidate
  counts, accepted/control-ready counts, review burden, evidence maps, and
  artifact paths.
- Build mode captures a planned run configuration that can be exported as a
  manifest for command-line execution or later attachment.

Build mode now has three practical surfaces:

- the pipeline stack, where stages are ordered and parameterized
- recommended architecture cards, which load common baseline, adaptive CFAR,
  artifact-suppression, high-recall, motion-aware, PMD, Suite2p, and OASIS
  plans
- a component library grouped by import, preprocessing, filtering, artifact,
  ROI, trace, event, ensemble, and ranking roles

Each component card shows what the stage does, why it exists, whether it is
implemented/planned/external, its input/output artifact contract, its tunable
parameters, real-time badges, and the QC outputs the Process Lab page should be
able to inspect.

Architecture Lab, Review, and Process Lab now share an active run selection.
Selecting a completed/generated run can load that run's `review_data.json` when
the file is reachable from the local workbench server. Selecting a planned run
does not change the Review video or pretend outputs exist. When the dashboard
is served locally, Generate View starts a whitelisted local job that runs the
selected Architecture Lab run, updates `architecture_runs.json`, and refreshes
Review/Process Lab when outputs are ready. Generated runs are isolated under
`app/generated_runs/<run_id>/` so a parameter test does not overwrite the
baseline dashboard.

Compare mode also includes a synchronized A/B Review viewer. Choose Run A and
Run B, then load A/B Review to fetch each generated `review_data.json` into a
browser cache. The viewer shows the same frame from both runs side-by-side with
ROI circles and event-near-frame highlights. It is read-only: changing A/B
selection does not replace the main Review page until you explicitly choose
`Use A In Review/QC` or `Use B In Review/QC`. Use `Next Difference` or
`Prev Difference` to jump to frames where the two loaded runs have different
candidate event counts.

Build mode is intentionally not a browser execution engine. In v1, the
workbench can configure, save, and export planned pipeline metadata, but it does
not run Fiji/Groovy, Python, Suite2p, CaImAn, PMD, OASIS, denoising models, or
other compute pipelines in-browser.

## Recommended Workflow

1. Use `Compare` mode to understand the current baseline and any completed
   generated runs.
2. Use `Build Pipeline` only when you want to change the actual sequence of
   stages or their parameters.
3. Keep the stack small for early experiments. Prefer one or two parameter
   changes at a time so Review and Process Lab comparisons remain interpretable.
4. Save planned runs before generation. Planned runs are visible to Review,
   Experiment Lab, and Process Lab, but they remain clearly marked as planned
   until generated artifacts exist.
5. Generate a preview first. Inspect the output in Review and Process Lab
   before launching a full run.

## Architecture Lab vs. Experiment Lab

Use Architecture Lab when the question is "what should this pipeline contain?"
Use Experiment Lab when the question is "which values should we try for this
pipeline?"

- Architecture Lab owns the ordered stage stack, component descriptions,
  parameter meanings, real-time badges, validation messages, and run comparison.
- Saved architectures are stored as reusable local templates in
  `architecture_runs.json` under `saved_pipelines[]`; they are not counted as
  completed/generated runs until materialized as planned/generated run entries.
  The Architecture Library supports edit, experiment, rename, and delete
  actions. Deleting a saved architecture template does not delete planned or
  generated runs that were created from it.
- Experiment Lab reuses that stack to create sweep axes or named parameter
  sets, or to define planning-only Optuna studies, then saves the resulting
  plans for local generation.
- Both pages write through the same `architecture_runs.json` contract, so the
  selected run stays synchronized with Review and Process Lab.

## Create A Run Manifest

Convert the current `review_data.json` into a run manifest:

```bash
python3 tools/build_architecture_run.py \
  --review-data Outputs/NeuronReview/calcium_video_2/app/review_data.json \
  --out Outputs/ArchitectureRuns/calcium_video_2/architecture_runs.json
```

Then build the dashboard with:

```bash
python3 tools/build_neuron_workbench_v2.py \
  --architecture-runs Outputs/ArchitectureRuns/calcium_video_2/architecture_runs.json
```

If no run manifest is supplied, the builder creates an in-memory baseline run
from the current review data. The selected or generated architecture-run
manifest is also written into the app directory as `architecture_runs.json`.

## Planned Pipeline Manifests

Planned pipeline manifests describe intended work before a run exists. They
should include enough information to reproduce or launch the run outside the
browser:

- dataset ID and source manifest path
- planned pipeline family, implementation, and version when known
- parameters and preset names that affect ROI/event generation
- expected output locations for `review_data.json`, evidence maps, traces, and
  architecture-run manifests
- optional `artifacts.proposal_analysis`,
  `artifacts.artifact_classifier_tsv`, and
  `artifacts.missed_neuron_proposals_tsv` entries for generated Process Lab
  triage
- optional `artifacts.intermediates[]` entries for browser-readable stage
  outputs, preferably PNG frame patterns such as
  `intermediates/<run_id>/<stage_id>/frame_%03d.png`
- status such as `planned`, `exported`, `running`, `completed`, or `failed`
- provenance notes, including who configured the plan and when it was exported

Completed run manifests remain the comparison source of truth. A planned
manifest should not be counted as a completed Architecture Lab run until the
external pipeline has produced artifacts and an architecture-run manifest is
attached.

The browser-side Generate View workflow is intentionally constrained. It calls
local server endpoints for predefined project jobs only; it never sends
arbitrary shell commands. The default backend uses the proven Fiji/Groovy
pipeline. A Python GPU backend can be selected explicitly, and the dashboard
reports Torch/CUDA readiness before attempting it.

Run-aware generation currently bridges the implemented Fiji/Groovy parameters
that affect the standard review outputs: temporal high-pass sigma, robust
local-z radius/epsilon, connected-component seed/grow/min/max area, local
background ring radius/weight, and Kalman/event thresholds. Planned or external
stages that do not yet have an executor remain metadata/QC expectations until a
worker is added for them.

## Parameter Sweeps

Build mode can also describe small parameter sweeps without running them in the
browser. A sweep is stored as manifest-level `sweep.parameters`, and each
expanded planned run receives a `sweep` assignment with the exact stage,
parameter, and value used. The dashboard and `tools/build_pipeline_run.py` use
the same plan/export-only contract.

Experiment Lab also supports an `Optuna plan` mode. This v1 integration records
the intended study direction, objective, trial budget, sampler/pruner labels,
and numeric search-space bounds for selected stage parameters. It intentionally
does not add an `optuna` dependency or execute optimization jobs from the
browser yet. The dashboard can duplicate an Optuna plan draft or convert its
numeric bounds into low/mid/high sweep seed values for a small deterministic
first pass.

Recommended first sweeps are narrow and interpretable:

- event threshold, such as `2.0, 2.4, 2.8`
- component seed/grow thresholds
- minimum and maximum ROI area
- Kalman positive-innovation event threshold

Use the exported planned manifest as a run sheet for Fiji/Python execution, then
merge completed runs back into Architecture Lab for comparison.

In the dashboard, saving a sweep expands it into separate planned run IDs.
Generate those run IDs from the Review run selector to produce side-by-side
Review and Process Lab artifacts for each combination.

The Compare view includes a Parameter Experiments table for these generated
runs. It shows ROI/event/suggestion counts, annotation-derived burden when
available, artifact-like queue counts, missed-neuron candidate counts, run
status, and a lightweight reviewer label. Selecting Run A makes that run the
global active run used by Review and Process Lab.

For a standard review-pack starting point, generate grouped planned runs:

```bash
python3 tools/build_sweep_pack.py \
  --dataset-id calcium_rest_cropped \
  --out Outputs/ArchitectureRuns/calcium_rest_cropped/review_pack_v1.json
```

The pack includes permissive, balanced, strict, artifact-suppression, and
high-recall planned runs. These are review plans, not completed detector
outputs. Use them to decide which variants to execute and then compare completed
runs for candidate stability and artifact burden.

## Stage Explanations And 100 Hz Readiness

Build mode reads the shared `neurobench.pipeline_catalog` metadata. Each stage
should include a plain-language description, why it is useful, parameter
explanations, and real-time metadata. The dashboard surfaces this directly in
the stack so parameter choices are not just raw names and numbers.

The current component catalog includes implemented stages for source/review
import, temporal high-pass filtering, event-preserving denoising, spatial
Gaussian smoothing, rigid drift estimation, robust local-z scoring, Gamma CFAR,
adaptive EWMA/Gamma CFAR, component filtering, local background correction,
trace event scoring, Kalman positive-innovation scoring, heuristic ranking, and
external Suite2p/PMD/OASIS imports. Planned components cover flat-field and
photobleach correction, Hampel impulse rejection, trace Kalman smoothing,
local-correlation and event-triggered footprint evidence, background/artifact
maps, soma-scale blob candidates, split/merge suggestions, ensemble/stability
scoring, artifact classification, and active-learning review ranking.

For upcoming 100 Hz samples, use the real-time badges as planning warnings:

- `streaming` means the stage is intended to work frame-by-frame.
- `adaptive` means the stage maintains or updates local statistics online.
- `offline` or `batch` means the stage is evidence/comparison-only for closed
  loop work unless a streaming runner is added.
- The 100 Hz summary assumes a 10 ms/frame budget when the dataset manifest
  reports `frame_rate_hz: 100.0`.

Benchmark candidate streaming stages with:

```bash
python3 tools/benchmark_pipeline_stage.py \
  --stage adaptive_ewma_z \
  --frame-rate-hz 100 \
  --out Outputs/Benchmarks/adaptive_ewma_z_100hz.json
```

## Merge Multiple Runs

When each method writes its own manifest, merge them before building the
dashboard:

```bash
python3 tools/merge_architecture_runs.py \
  --out Outputs/ArchitectureRuns/calcium_video_2/architecture_runs.json \
  Outputs/ArchitectureRuns/calcium_video_2/current_run.json \
  Outputs/ArchitectureRuns/calcium_video_2/suite2p_architecture_runs.json
```

Duplicate `run_id` values fail by default. Add `--replace` only when the later
manifest should intentionally overwrite an earlier run.

## Current Fields

Architecture Lab v1 shows:

- run ID and label
- dataset ID
- ROI, event, suggestion, and frame counts
- review-data and ROI-summary artifact paths
- proposal-analysis, artifact-classifier, and missed-neuron proposal artifact
  paths when present
- evidence-map labels
- paired run comparison for candidate counts, accepted/rejected counts,
  control-ready counts, and review burden when two or more runs are available
- Build-mode component descriptions, parameter explanations, availability
  status, real-time badges, expected QC outputs, and one-click recommended
  architecture presets

Process Lab is tied to the selected architecture run. That page shows the
ordered pipeline context beside frame/evidence navigation so QC warnings can be
interpreted against the exact stack that produced or is expected to produce the
candidate set.

When generated intermediate outputs are attached, Process Lab displays them as a
synchronized stage grid driven by a single frame slider. Missing stage outputs
are shown as placeholders so it is obvious what still needs to be exported.

The companion Metrics/Audit page is available at `#metrics` and summarizes the
current annotation burden and review progress.

## Review/Test Planning

For Build mode and pipeline manifests, documentation and manual testing should
verify:

- exported planned manifests never imply that browser-side execution happened
- completed manifests remain visually distinct from planned/exported work
- the baseline run still appears when no explicit manifest is supplied
- Run A / Run B comparison ignores planned-only entries unless they have
  completed run artifacts
- paths exported from Build mode are relative or manifest-driven where possible
  instead of hard-coded to one workstation
