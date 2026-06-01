# Pipelines

Pipelines is the comparison and pipeline-builder page inside the neuron
workbench. It is available at:

```text
http://127.0.0.1:8765/#pipelines
```

Legacy links to `#architecture` and `#architecture-lab` still open this page.

The page consumes standardized architecture-run metadata. A run can represent
the current Fiji/Groovy pipeline, a Python CFAR pipeline, or future imports such
as Suite2p, CaImAn, PMD, OASIS, or denoising models.

Suite2p outputs can be converted with `tools/import_suite2p_run.py`,
PMD-denoised videos can be attached with `tools/import_pmd_run.py`, and OASIS
trace outputs can be attached with `tools/import_oasis_run.py`; see
`docs/SOTA_INTEGRATIONS.md`.

Pipelines has two related modes:

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

It also includes a `Multi-stage CFAR cascade` preset. This represents the
professor-suggested idea of a small reference-region CFAR followed by a larger
reference-region CFAR as two explicit CFAR stages, not as a hidden composite
block. The default cascade intersects the two masks so compact candidates must
survive both a fine local-background check and a broader background check.

Each component card shows what the stage does, why it exists, whether it is
implemented/planned/external, whether a local Python runner is available, its
input/output artifact contract, its tunable parameters, real-time badges, and
the QC outputs the Data page should be able to inspect. `Implemented`
means the stage is part of the Neurobench model; `local runner` means the
stdlib/Python executor can currently run it for local sweeps.

Pipelines, Review, and Data now share an active run selection.
Selecting a completed/generated run can load that run's `review_data.json` when
the file is reachable from the local workbench server. Selecting a planned run
does not change the Review video or pretend outputs exist. When the dashboard
is served locally, Generate View starts a whitelisted local job that runs the
selected pipeline run, updates `architecture_runs.json`, and refreshes
Review/Data when outputs are ready. Generated runs are isolated under
`app/generated_runs/<run_id>/` so a parameter test does not overwrite the
baseline dashboard.

Compare mode also includes a synchronized A/B Review viewer. Choose Run A and
Run B, then load A/B Review to fetch each generated `review_data.json` into a
browser cache. The viewer shows the same frame from both runs side-by-side with
ROI circles and event-near-frame highlights. It is read-only: changing A/B
selection does not replace the main Review page until you explicitly choose
`Use A In Review/Data` or `Use B In Review/Data`. Use `Next Difference` or
`Prev Difference` to jump to frames where the two loaded runs have different
candidate event counts.

Build mode is intentionally not a browser execution engine. In v1, the
workbench can configure, save, and export planned pipeline metadata, but it does
not run Fiji/Groovy, Python, Suite2p, CaImAn, PMD, OASIS, denoising models, or
other compute pipelines in-browser.


## Run Readiness Badges

Pipeline cards now show compact readiness badges for local executability,
review-data availability, ROI candidates, intermediate outputs, projection
diagnostics, and stencil coverage. A warning badge does not invalidate a run; it
marks what should be generated, attached, or inspected before using the run for
review decisions.

## Recommended Workflow

1. Use `Compare` mode to understand the current baseline and any completed
   generated runs.
2. Use `Build Pipeline` only when you want to change the actual sequence of
   stages or their parameters.
3. Keep the stack small for early experiments. Prefer one or two parameter
   changes at a time so Review and Data comparisons remain interpretable.
4. Save planned runs before generation. Planned runs are visible to Review,
   Experiment Lab, and Data, but they remain clearly marked as planned
   until generated artifacts exist.
5. Generate a preview first. Inspect the output in Review and Data
   before launching a full run.

## Pipelines vs. Experiment Lab

Use Pipelines when the question is "what should this pipeline contain?"
Use Experiment Lab when the question is "which values should we try for this
pipeline?"

- Pipelines owns the ordered stage stack, component descriptions,
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
- Experiment Lab also has an Experiment Command Center that surfaces run queue
  readiness, imported LLM proposal sets, sweep budget warnings, annotation-aware
  next-step recommendations, utility scores, and baseline deltas before more
  compute is launched.
- Session Recipe turns the current objective, highest-priority action, planned
  runs, suggested parameter moves, and safeguards into a compact Markdown/JSON
  handoff for lab discussion or external LLM-assisted architecture planning.
- LLM Architecture Request builds a copyable/downloadable prompt from the
  recipe, current stack, available stage catalog, extra constraints, and the
  expected proposal JSON shape. It supports architecture-feedback, compact
  parameter-search, noise/artifact-control, and 100 Hz real-time request modes.
- LLM Proposal Intake accepts pasted proposal JSON, performs lightweight
  browser checks for required fields, unknown stage IDs, sweep-axis references,
  and combination counts, then exposes the local importer command. When issues
  are found, it can generate a repair prompt with the exact errors, warnings,
  known stage IDs, and expected schema shape. It also triages parsed proposals
  into candidate, budget-review, or repair-first decisions and can export a
  compact Markdown review note. Candidate-only export creates a normalized JSON
  pack containing only import candidates for the first local test pass. Import
  Readiness shows whether full or candidate import is safer and provides
  copyable commands for both paths. When served locally, **Import Full To
  Dashboard** and **Import Candidates To Dashboard** call the matching
  `/api/llm-proposals/import` endpoint to validate and merge the proposal pack
  into `architecture_runs.json` without launching execution. The CLI command is
  still shown for reproducible terminal workflows. Post-Import Plan then gives
  the reload, preview-generation, and optional local experiment-run sequence.
- LLM Proposal Lifecycle persists proposal decisions such as try-next,
  promising, reject, needs-repair, generated, and discussed. It summarizes
  generated outcomes and creates a follow-up prompt from review failure signals.
- Prioritized Action Queue merges checklist blockers, recommendation cards,
  sensitivity signals, follow-up suggestions, and coverage gaps into a ranked
  set of concrete next steps. Queue items can be marked done, snoozed, or
  reopened to keep the next-step surface focused during longer review sessions.
  Action History records recent queue state changes for session handoff.
- The Experiment Command Center has a `Focus` selector with guided,
  diagnostics, and all-panel views, so the page can stay compact while still
  exposing deeper analyses when needed.
- The command center includes an evaluation checklist and decision matrix so
  run selection is driven by explicit gates, utility scores, labels, notes, and
  exportable TSV summaries rather than memory.
- Parameter Sensitivity groups explicit sweep/named-set changes by parameter
  and reports values tested, generated/scored counts, average utility, score
  spread, best run, and a suggested next move.
- Follow-up Planner converts the strongest sensitivity signals into small local
  refinements around the best observed value and can add them back to
  Experiment Lab as named sets or a sweep axis.
- Parameter Coverage Map scans the current pipeline stack for numeric
  parameters that have not been tested yet and can add small probe sets or a
  sweep axis directly from the coverage table.
- The Experiment Command Center can export a concise Markdown brief,
  provider-neutral LLM prompt, proposal-intake summary, and machine-readable
  handoff JSON, so architecture feedback can happen outside the dashboard
  without copying raw dashboard state by hand.
- Both pages write through the same `architecture_runs.json` contract, so the
  selected run stays synchronized with Review and Data.

## LLM-Guided Architecture Proposals

The workbench supports an LLM-assisted planning workflow without embedding a
chat UI or sending data to a provider from the dashboard. The intended flow is:

1. Build a provider-neutral handoff context:

   ```bash
   neurobench llm context \
     --dataset-manifest Outputs/Manifests/calcium_rest_cropped.dataset.json \
     --architecture-runs Outputs/NeuronReview/calcium_rest_cropped/app/architecture_runs.json \
     --objective review_efficiency \
     --max-combinations 4096 \
     --context-out Outputs/ArchitectureRuns/calcium_rest_cropped/llm_context.json \
     --prompt-out Outputs/ArchitectureRuns/calcium_rest_cropped/llm_prompt.md
   ```

2. Give the prompt/context to an LLM and ask it to return JSON matching
   `schemas/llm_architecture_proposal.schema.json`.
3. Import the returned proposals:

   ```bash
   neurobench llm import-proposals \
     Outputs/ArchitectureRuns/calcium_rest_cropped/llm_proposals.json \
     --architecture-runs Outputs/NeuronReview/calcium_rest_cropped/app/architecture_runs.json \
     --out Outputs/NeuronReview/calcium_rest_cropped/app/architecture_runs.json \
     --validation-report Outputs/ArchitectureRuns/calcium_rest_cropped/llm_validation_report.json
   ```

Imported proposals become saved architecture templates, planned runs, and an
experiment record in the same `architecture_runs.json` used by Pipelines,
Experiment Lab, Review, and Data. The importer rejects unknown
stages, duplicate step IDs, out-of-range parameters, ambiguous sweep references,
and per-architecture sweeps larger than the configured combination budget
(`4096` by default).

The LLM should reference sweep axes by concrete step ID. This matters for
architectures with repeated stages, such as:

```json
{"stage": "cfar_small_ref", "param": "pfa", "values": [0.01, 0.03]}
```

For local executable tests of LLM proposal packs, use:

```bash
neurobench llm run-proposals \
  Outputs/ArchitectureRuns/calcium_rest_cropped/llm_proposals.json \
  --run-root Outputs/ArchitectureRuns/calcium_rest_cropped/llm_runs \
  --ground-truth-csv Inputs/annotations/resting_crop_ground_truth.csv
```

This runner only executes stages marked with `local runner` in the component
catalog. When a ground-truth CSV is provided, the summary adds object
precision/recall, event-onset recall, burden counts, and runtime so proposal
sets can be compared before committing to full Review generation. Full
Review-app generation still uses the dashboard's local Generate View path and
the existing Fiji/Python bridge.

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
  `artifacts.missed_neuron_proposals_tsv` entries for generated Data
  triage
- optional `artifacts.intermediates[]` entries for browser-readable stage
  outputs, preferably PNG frame patterns such as
  `intermediates/<run_id>/<stage_id>/frame_%03d.png`
- status such as `planned`, `exported`, `running`, `completed`, or `failed`
- provenance notes, including who configured the plan and when it was exported

Completed run manifests remain the comparison source of truth. A planned
manifest should not be counted as a completed pipeline run until the
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
merge completed runs back into Pipelines for comparison.

In the dashboard, saving a sweep expands it into separate planned run IDs.
Generate those run IDs from the Review run selector to produce side-by-side
Review and Data artifacts for each combination.

The Compare view includes a Parameter Experiments table for these generated
runs. It shows ROI/event/suggestion counts, annotation-derived burden when
available, artifact-like queue counts, missed-neuron candidate counts, run
status, and a lightweight reviewer label. Selecting Run A makes that run the
global active run used by Review and Data.

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

Pipelines v1 shows:

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

Data is tied to the selected pipeline run. That page shows the
ordered pipeline context beside frame/evidence navigation so QC warnings can be
interpreted against the exact stack that produced or is expected to produce the
candidate set.

When generated intermediate outputs are attached, Data displays them as a
synchronized stage grid driven by a single frame slider. Missing stage outputs
are shown as placeholders so it is obvious what still needs to be exported.

The companion Progress page is available at `#progress` and summarizes the
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
