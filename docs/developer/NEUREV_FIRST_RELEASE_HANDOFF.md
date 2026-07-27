# NeuRev first-release implementation handoff

Implementation snapshot: 2026-07-25.

This handoff records what the Wave 0–2 implementation actually delivers from
`docs/NeuRev_Codex_Implementation_Brief.docx`, how to recover or compare a
historical app, and what remains deferred. It is an implementation handoff,
not a claim of scientific success or completed browser acceptance.

The normal user path is:

```text
Datasets → Annotate → Results
```

Pipeline construction, experiments, method comparison, detector controls, raw
parameters, and LLM planning remain available behind **Research Tools**. A
prepared frame stack and an annotation app are not evidence that a detector
run completed: annotation-only apps receive an empty architecture-run catalog
unless a real catalog is supplied or already installed.

## Non-negotiable preservation contract

- Existing `review_data.json`, `annotations.json`, and
  `architecture_runs.json` are read-only by default.
- Existing annotation bytes are rewritten only when
  `workbench build --migrate-annotations` is explicitly selected.
- An existing architecture-run catalog is preserved byte-for-byte unless a
  replacement is explicitly supplied.
- Historical `Outputs/` roots, completed scientific outputs, and archived run
  logs are not bulk rebuilt or deleted.
- Unknown modality, indicator, frame rate, pixel size, fish, session, and view
  metadata remain unknown until supplied by a trusted source.
- Sparse positives and unlabeled event pixels remain unknown, not negative.
- Completion of an import, QC, frame-rendering, or annotation job is not
  scientific detector success.

## Wave 0 — baseline, recovery, and explicit migration

### Capture a preservation baseline

Capture every catalog-discovered app plus legacy `Outputs/NeuronReview/*/app`
apps:

```bash
.venv-neurobench/bin/python -m neurobench.cli.main workbench baseline \
  --root . \
  --output .neurobench/baselines/wave0-before.json
```

To lock only selected apps, repeat `--app-dir` as needed:

```bash
.venv-neurobench/bin/python -m neurobench.cli.main workbench baseline \
  --root . \
  --app-dir Outputs/NeuronReview/<dataset>/app \
  --app-dir Outputs/GammaCFAR/<dataset>/app \
  --output .neurobench/baselines/wave0-before.json
```

Relative app and baseline paths resolve under `--root`; paths outside that
root are rejected. Capture fails if an app is missing any required protected
file:

- `review_data.json`
- `annotations.json`
- `architecture_runs.json`
- `index.html`
- `workbench.css`
- `workbench.js`

`dataset_manifest.generated.json` is protected and hashed when present, but it
is optional because valid historical apps can predate that file. The baseline
also records catalog identity and capability state. It does not hash raw video
pixels, frame directories, screenshots, or complete experiment trees.

The baseline stable identity excludes `captured_at` and the top-level
`workspace_root` capture field. Writing is atomic and refuses to replace an
existing baseline unless `--overwrite` is explicitly supplied.

### Verify or compare baselines

Verify the stored app set and catalog against the current workspace:

```bash
.venv-neurobench/bin/python -m neurobench.cli.main workbench baseline \
  --root . \
  --verify .neurobench/baselines/wave0-before.json
```

The command exits nonzero and reports changed paths when protected bytes or
stable catalog identity differ. `--app-dir` is intentionally unavailable for
verification; verification reuses the app set stored in the baseline.

Compare two stored captures while ignoring capture timestamps:

```bash
.venv-neurobench/bin/python -m neurobench.cli.main workbench baseline \
  --root . \
  --diff \
    .neurobench/baselines/wave0-before.json \
    .neurobench/baselines/wave0-after.json
```

Add `--json` to capture, verify, or diff for a machine-readable report. Use
`--overwrite` only with `--output`, and only when deliberately replacing an
audit record.

### Serve current code without rebuilding an archive

The default mode renders the packaged HTML/CSS/JavaScript in memory and leaves
the app directory untouched:

```bash
.venv-neurobench/bin/python -m neurobench.cli.main workbench serve \
  --app-dir Outputs/NeuronReview/<dataset>/app \
  --asset-mode current \
  --host 127.0.0.1 \
  --port 8765
```

Use installed mode only for compatibility or before/after comparison:

```bash
.venv-neurobench/bin/python -m neurobench.cli.main workbench serve \
  --app-dir Outputs/NeuronReview/<dataset>/app \
  --asset-mode installed
```

`installed` serves the archived `index.html`, `workbench.css`, and
`workbench.js` bytes and warns when their marker or CSS/JavaScript bytes are
stale or altered relative to the package. `current` does not make the
installed files current; it only changes what the server returns. Inspect the
distinction with:

```bash
.venv-neurobench/bin/python -m neurobench.cli.main workbench status \
  --app-dir Outputs/NeuronReview/<dataset>/app \
  --json
```

Server construction and read-only GETs do not create a job directory. The
durable store is initialized only when a job-backed mutation needs it.

### Rebuild only when an installed upgrade is intended

`workbench build` is the explicit installed-asset mutation path. It renders
and validates first, stages files beside the app, publishes CSS/JavaScript
before `index.html`, and uses `index.html` as the existing-app commit point.
Preflight/render failure leaves the existing app unchanged.

```bash
.venv-neurobench/bin/python -m neurobench.cli.main workbench build \
  --review-data Outputs/NeuronReview/<dataset>/app/review_data.json \
  --app-dir Outputs/NeuronReview/<dataset>/app
```

That command preserves an existing annotation file and run catalog by default.
Schema migration is a separate, explicit operation:

```bash
.venv-neurobench/bin/python -m neurobench.cli.main workbench build \
  --review-data Outputs/NeuronReview/<dataset>/app/review_data.json \
  --app-dir Outputs/NeuronReview/<dataset>/app \
  --migrate-annotations
```

Before any historical rebuild, capture and verify a baseline, retain the
before file, and inspect the after diff. Prefer `serve --asset-mode current`
when the goal is only to use the new interface.

### Deterministic task migration

Python and browser migration preserve unknown scientific fields and map the
legacy review-workflow preset as follows:

| Legacy preset | `annotationTask` |
| --- | --- |
| `fast_triage`, `validate_neurons` | `neuron_validation` |
| `event_validation` | `event_validation` |
| `missed_neuron_search`, `find_missed_neurons` | `missed_neuron_search` |
| `artifact_cleanup`, `clean_artifacts` | `artifact_resolution` |
| `mask_editing` | `signal_background` |
| missing or unrecognized | `neuron_validation` |

An already valid `annotationTask` wins. `researchToolsEnabled` is migrated as
an independent boolean. Old layouts are moved to one annotation-safe canvas
unless a later layout version explicitly records another choice. Explicit
schema migration also removes embedded `cfar_regions.edit_history`; the CFAR
regions themselves and unrelated scientific fields remain preserved.

## Wave 1 — annotation-first shell

The packaged app has four primary destinations:

1. **Datasets** shows catalog records, truthful lifecycle/capability states,
   local registration, upload, metadata resolution, QC, promotion, processing,
   and label-preview actions.
2. **Annotate** supplies task-specific normal-mode shells for neuron
   validation, missed-neuron search, event validation, artifact resolution,
   exhaustive-tile presentation, and signal/background marking.
3. **Results** combines review progress, import/job status, scientific-run
   availability, and one selector for the existing annotation/report exports.
4. **Research Tools** retains legacy data/QC, pipeline, experiment, report,
   comparison, and planning surfaces.

The normal neuron decision is **Neuron**, **Not neuron**, or **Unsure**.
Choosing a decision does not silently advance; **Save + Next** is a separate
action. Normal queue labels are **Next**, **Needs attention**, **Reviewed**,
and **All**. ROI and CFAR controls are opened contextually; CFAR controls remain
hidden outside the signal/background task unless Research Tools is enabled.

The task shells are not a Wave 4 benchmark contract. In particular,
`exhaustive_tile` is presently a UI task choice without spatial task bounds,
coverage declarations, reviewer assignment, adjudication, or benchmark-truth
export.

The Results export selector consolidates the existing ROI, event, suggestion,
split/merge, annotation JSON, provenance, handoff, and report downloads. It
does not yet implement the complete project/benchmark/lab-handoff export
contract from the brief.

Focused Node/Python tests cover task decisions, routing, migration, dataset
qualification, and owner-aware mutation headers. No real browser acceptance
run has been completed for this snapshot; do not report the browser workflow
as accepted based only on those tests.

## Wave 2 — plug-and-play ingestion

### Supported and unsupported sources

The implemented first-release source set is:

| Source | Status | Notes |
| --- | --- | --- |
| `.tif`, `.tiff` | Supported | Video metadata inspection and bounded normal processing. |
| `.npy` | Supported | Video metadata inspection; the existing video loader provides memory-mapped behavior where applicable. |
| `.csv`, `.tsv` | Supported | Streamed row iteration for label-table preview and reconciliation. |
| `.xlsx` | Supported when `openpyxl` is installed | Read-only workbook iteration with the same row limit. |
| NeuRev `.json` | Supported for four native contracts | A 64 MiB hard cap is checked before payload read. Only schema-valid `review_data`, annotations v3, `architecture_runs`, and `export_bundle` objects are recognized. |
| OME-TIFF, NWB, HDF5, DANDI, ZIP/folder | Deferred | No first-release adapter interface has landed. |

NeuRev JSON support is deliberately not a generic JSON loader. Unknown or mixed
document shapes, invalid UTF-8/JSON, duplicate object keys, schema failures, and
oversized sources are rejected. Import metadata keeps only the recognized
payload kind, declared dataset ID, checksum, byte count, and bounded count
summary; it does not embed the payload or follow paths declared inside it.

Local registration accepts existing files only under configured `Inputs/` or
`Outputs/` roots and does not copy them. Browser uploads are bounded, stream to
an exclusive `.partial` destination under `Inputs/<dataset_id>/`, checksum the
source, fsync, and atomically rename on success. Failed partial uploads are
cleaned up and receive a durable failed import record when enough identity is
available.

Every import records an immutable import/dataset identity, source mode and
role, original and destination path, SHA-256/byte count, observed metadata,
warnings, generated artifacts, revision, and lifecycle state. Missing
scientific metadata remains `null` until supplied. Only an explicitly promoted
video may create the canonical generated dataset manifest or enter normal
processing; a label table is never fabricated as `raw_video`.

### Truthful lifecycle

The shared state set is:

```text
uploaded
  ├─ video → metadata_needed → qc_ready → processing → ready
  │                                                → annotation_in_progress → complete
  ├─ labels → qc_ready → processing → complete
  └─ NeuRev JSON → qc_ready → preview + explicit confirmation → processing → complete

active non-final state → failed
failed → metadata_needed or qc_ready after explicit retry/repair
```

Registration usually completes bounded inspection immediately, so a video
normally first appears as `metadata_needed`; label tables and recognized
NeuRev JSON first appear as `qc_ready`.
`uploaded` remains a valid transient state, not a promise that QC or processing
has run. Dataset-qualified routes reject mismatched dataset IDs. Mutating
routes use the local owner-token contract when configured.

### Durable import jobs and publication

QC, normal processing, label reconciliation, and confirmed NeuRev JSON
publication use atomic JSON job records
under the app's `.neurobench/jobs` directory. Jobs record status, stage,
progress, inputs, outputs, errors, and a bounded log tail. On restart,
previously queued/running records are marked `stopped`; they are explained, not
silently shown as completed and not automatically resumed.

QC samples a bounded frame set. Normal processing verifies source identity,
requires completed QC and explicit primary-video promotion, refuses collisions
with protected app artifacts, stages output, and publishes a bounded browser
frame set. This durable store does not yet cover every research-generation or
trace-materialization path, and it does not implement separate `job.json`,
`progress.json`, and `log.txt` files for every job.

### Label reconciliation is separate from native annotations

Label preview is required before confirmation. A confirmed CSV/TSV/XLSX
import publishes:

```text
app/external_labels/<import_id>.json
app/external_labels/<import_id>.overlay.svg
```

The JSON artifact keeps every source row under a stable key such as
`row_00000002`. Each entry retains the original `source_row`, source row
number, normalized mapped values, and reconciliation classifications/status.
The summary reports total, matched, unmatched, duplicate, and rejected rows.
The overlay makes coordinates and reconciliation status visually inspectable.

Reconciliation requires native `review_data.json` ROI identities, rechecks the
source checksum, and never rewrites `annotations.json` or the canonical dataset
manifest. Imported external labels remain distinct from native decisions; they
are not stored under `annotations.externalLabels`.

The reconciliation algorithm is isolated in
`neurobench/workbench/label_reconciliation.py`: it accepts source/review/import
values and returns bounded artifact bytes, overlay bytes, and a summary without
owning locks, job state, or filesystem publication. The server worker owns
those orchestration concerns and publishes only after the pure step succeeds.

### NeuRev JSON remains external to canonical app state

NeuRev JSON preview revalidates the source checksum, recognized payload kind,
schema, count summary, and any declared dataset identity. Confirmation must be
the literal JSON boolean `true`. The durable import job rechecks the checksum,
copies the original bytes losslessly with a streaming checksum, and publishes
exclusively to:

```text
app/external_neurev/<import_id>.json
```

The external copy retains source whitespace and key order exactly. Confirmation
does not merge or replace `review_data.json`, `annotations.json`,
`architecture_runs.json`, or a canonical dataset manifest. Catalog/API/UI
records show a distinct `neurev_json` attachment with its recognized payload
kind and counts. A declared dataset ID must normalize to the target dataset ID;
cross-dataset attachment is rejected.

## Wave 0–2 delivery ledger

`Delivered` means implemented and covered by focused automated checks.
`Partial` identifies a concrete landed subset plus its remaining boundary.

| ID | Status | Actual first-release result |
| --- | --- | --- |
| CX-000 | Partial | Content hashes, representative historical-app discovery, preservation tests, and HTTP no-write checks landed; screenshot fixtures and manual browser comparisons did not. |
| CX-001 | Delivered | Deterministic legacy-preset-to-task mapping and explicit installed annotation migration landed. |
| CX-002 | Partial | Shared `ready`, `import_only`, `planned`, `blocked`, and `unavailable` states feed catalog/API/UI contracts. Per-method builder/executor parity is not yet complete. |
| CX-100 | Delivered | Datasets / Annotate / Results / Research Tools product shell landed while legacy research routes remain reachable. |
| CX-101 | Delivered | `annotationTask` and independent `researchToolsEnabled` state landed with deterministic legacy migration. |
| CX-102 | Partial | Six contextual task shells landed; exhaustive coverage and benchmark semantics remain Wave 4 work. |
| CX-103 | Delivered | Three-way neuron decision and separate Save + Next action are runtime-tested. |
| CX-104 | Delivered | ROI/CFAR panels are contextual in normal mode; Research Tools can still expose advanced controls. |
| CX-105 | Delivered | Four normal queue labels landed; specialized queues remain in the advanced implementation. |
| CX-106 | Delivered | Results is the normal progress/report destination; legacy report machinery remains underneath. |
| CX-107 | Partial | Existing exports are consolidated in Results, but benchmark truth/project/lab-handoff parity is incomplete. |
| CX-108 | Delivered | User-facing shell copy uses NeuRev while package and CLI names remain `neurobench`. |
| CX-200 | Delivered | Versioned import schema/records, immutable identity, checksums, revisions, warnings, artifacts, normalized path-safe dataset IDs, and collision refusal landed. |
| CX-201 | Delivered | Streamed bounded upload supports TIFF/NPY/CSV/TSV/XLSX plus four explicitly recognized native NeuRev JSON contracts. JSON has a 64 MiB pre-read cap, strict validation, preview, explicit confirmation, checksum revalidation, and lossless external publication. |
| CX-202 | Delivered | Allowed-root local registration without copying landed. |
| CX-203 | Partial | Browser upload/register, metadata, QC, promotion, process, label, and NeuRev JSON preview/confirmation actions landed; no real-browser end-to-end acceptance has been recorded. |
| CX-204 | Delivered | Import, catalog, and generated review data preserve unknown scientific metadata instead of fabricating defaults. |
| CX-205 | Delivered | Previewed, row-preserving label mapping/reconciliation and visual overlay landed without replacing native annotations. |
| CX-206 | Delivered | Shared lifecycle states and guarded transitions landed in records, API, and dataset cards. |
| CX-207 | Deferred | Future-format adapter interface is not implemented. |

## Waves 3–5 delivered/deferred matrix

Later-wave status is explicit so a visible placeholder or inherited legacy
feature is not mistaken for a completed contract.

| ID | Status | Delivered subset / remaining work |
| --- | --- | --- |
| CX-300 | Partial | Import QC, normal processing, and label jobs persist atomic records and recover active jobs as `stopped`. Conversion, all generation, and trace materialization are not unified, and the requested per-job file trio is absent. |
| CX-301 | Partial | Normal annotation hides backend controls and offers one preparation action. Broader Research Tools backend truth/fall-through semantics still require audit and cleanup. |
| CX-302 | Deferred | `generate_neuron_review_app` planning-versus-execution naming/behavior was not resolved as part of this release. |
| CX-303 | Partial | Shared capability states and guarded normal actions landed. Full stage/recipe filtering across every research catalog path is not complete. |
| CX-304 | Partial | Import metadata, dimensions, checksums, warnings, upload-size/disk gates, and bounded QC are exposed. A unified dependency/RAM/disk/output preflight summary for every runner is deferred. |
| CX-305 | Partial | The normal import flow uses goal-oriented QC and Prepare manual annotation actions; modality recipe selection and full raw-parameter separation remain deferred. |
| CX-400 | Deferred | `annotationTask` is a UI preference, not the required bounds/coverage/reviewer/adjudication task schema. |
| CX-401 | Deferred | The exhaustive-tile UI choice does not implement bounded playback, completion checks, or an exhaustive coverage declaration. |
| CX-402 | Deferred | Existing reviewer provenance/queues do not constitute primary/secondary assignment and adjudication workflow. |
| CX-403 | Deferred | No separate active-annulus, soma-extent, inner-region, and outer-background morphology schema landed. |
| CX-404 | Deferred | No coverage manifest or benchmark-truth export with held-out grouping and adjudication landed. |
| CX-500 | Partial | A separate product-shell source module, an explicit 12-module bundle manifest, focused runtime tests, and a pure label-reconciliation module landed. The large review module and HTTP import/job orchestration remain cross-domain hot spots. |
| CX-501 | Deferred | CSS has not been split into ordered source modules. |
| CX-502 | Deferred | The app still embeds substantial review data and the stage catalog in generated HTML; a lightweight lazy bootstrap has not landed. |
| CX-503 | Deferred | Media generation has not been unified under `neurobench.media`. |
| CX-504 | Deferred | Gamma preparation remains in the compatibility script and has not been split into package modules. |
| CX-505 | Partial | First-release import QC samples bounded frames, but all generic and research QC paths have not been converted to one chunked contract. |
| CX-506 | Partial | Normal import processing renders a bounded browser frame set and refuses protected-output collisions; repository-wide dense-output opt-in policy is deferred. |

## Verification and acceptance boundary

Run generated-asset consistency, syntax, and focused behavior checks:

```bash
.venv-neurobench/bin/python tools/build_workbench_assets.py --check
node --check neurobench/workbench/assets/workbench.js
.venv-neurobench/bin/python -m pytest -q \
  tests/test_workbench_assets.py \
  tests/test_workbench_builder.py \
  tests/test_workbench_baseline.py \
  tests/test_workbench_cli.py \
  tests/test_workbench_server.py \
  tests/test_dataset_imports.py \
  tests/test_workbench_import_flow.py \
  tests/test_workbench_jobs.py \
  tests/test_label_reconciliation.py \
  tests/test_workbench_product_shell_runtime.py \
  tests/test_workbench_structure.py \
  tests/test_annotations_model.py \
  tests/test_dataset_catalog.py
```

Do not regenerate the bundle merely to run verification; use `--check` first.
Run the full suite after focused failures are resolved.

Automated coverage includes preservation, in-memory asset rendering, asset
tamper/source-manifest detection, no-write current-mode HTTP serving,
upload/register safety,
import transitions, source mutation rejection, row-preserving label artifacts,
lossless external NeuRev JSON publication, arbitrary/mixed/oversized JSON
rejection, bounded/schema-valid/identity-bound import sidecars, native
manifest/annotation preservation, durable job recovery, task decisions,
routing, annotation-tool mutual exclusion, run-bucket preservation, and
owner-aware mutations.

The following manual acceptance remains open:

- a real browser upload/register → metadata → QC → promote → process →
  annotate → export path;
- a browser comparison of current versus installed mode on non-empty
  historical Spon and resting-calcium apps;
- upload/processing restart behavior observed through the UI;
- keyboard-only neuron and event review;
- screenshot baselines and a 20-candidate normal-mode usability pass.

Until those checks are recorded, describe Waves 0–2 as implemented with the
limitations above, not as browser-accepted or production-certified.

No part of this release authorizes the stopped Grid128 sweep, a new GPU run,
stimulation, controller deployment, or causal left/right claims. Those remain
subject to the repository's scientific workflow and safety gates.
