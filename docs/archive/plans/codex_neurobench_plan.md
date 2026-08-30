# Codex Implementation Plan: Research-Grade Neuroimaging Discovery and Analysis Platform

## 0. How Codex Should Use This Document

This document is a Codex-facing execution plan for evolving the `Separable-Gamma-CFAR-main` repository into a research-grade neuroimaging discovery, annotation, analysis, and export platform for noisy zebrafish calcium/voltage imaging videos.

It is intentionally written as an implementation playbook, not as a product pitch. Codex should use it to plan small, reviewable changes. Do **not** attempt to implement the full plan in one pass.

### 0.1 Global implementation rules

1. **Do not rewrite the repository from scratch.** Preserve current working behavior, especially the existing Gamma/CFAR grid-search workflow, Fiji/Groovy review workflow, workbench server, annotation tooling, manifests, importers, and tests.
2. **Run the test suite before and after meaningful changes.** Current inspected baseline: `pytest -q` passes with 54 tests, 37 subtests, and two `np.trapz` deprecation warnings from `evaluation/metrics.py`.
3. **Prefer staged migration.** Introduce stable interfaces and compatibility wrappers before moving large code blocks.
4. **Keep raw scientific data and generated outputs out of git.** Commit source, schemas, docs, tests, and tiny synthetic fixtures only.
5. **Treat public artifacts as contracts.** Dataset manifests, pipeline specs, run manifests, annotations, review data, metrics reports, and export bundles must be versioned and validated.
6. **Keep experimental flexibility.** Experimental methods are allowed, but they should be isolated behind stable artifact interfaces.
7. **Do not hide algorithmic uncertainty.** Workbench suggestions, ranking, triage, and active-learning outputs are recommendations, not ground truth.
8. **Favor inspectability over cleverness.** Every candidate, event, score, report, and export should be traceable to input data, parameters, code version, and artifacts.
9. **Make CPU-only onboarding reliable first.** GPU acceleration and streaming inference are future-facing; they must not become dependencies for basic usage.
10. **When existing docs conflict with current code, inspect code and tests first.** The existing `docs/plan.md` contains valuable historical context but some early P0 items appear to have already been fixed in the current codebase.

### 0.2 Recommended Codex workflow for each task

For every task, Codex should follow this procedure:

```text
1. Inspect relevant files and tests.
2. Identify current behavior and current public interfaces.
3. Make the smallest useful change.
4. Add or update tests.
5. Run targeted tests.
6. Run full pytest if the change touches shared code.
7. Update docs/examples only when behavior or workflow changed.
8. Summarize changed files, behavior preserved, and tests run.
```

### 0.3 Stop conditions

Stop and ask for human review if a change requires any of the following:

- changing raw-data assumptions without a synthetic or documented fixture;
- deleting existing tools rather than wrapping/deprecating them;
- changing annotation schema semantics without a migration path;
- removing Fiji/Groovy workflow support;
- changing public output filenames used by docs/tests without compatibility aliases;
- adding large new dependencies without a clear CPU-only fallback;
- converting browser workbench logic in a way that cannot be tested by smoke tests.

---

## 1. Current Baseline Codex Should Assume

### 1.1 Repository snapshot

The repository currently contains these major areas:

```text
Separable-Gamma-CFAR-main/
├── README.md
├── config.py
├── data_loader.py
├── main.py
├── utils.py
├── worker.py
├── environment.yml
├── core/
│   ├── detection.py
│   ├── filters.py
│   └── pipelines.py
├── evaluation/
│   ├── analysis.py
│   └── metrics.py
├── reporting/
│   ├── generators.py
│   └── plotters.py
├── neurobench/
│   ├── annotations.py
│   ├── annotation_metrics.py
│   ├── architecture_runs.py
│   ├── manifests.py
│   ├── online.py
│   ├── pipeline_catalog.py
│   ├── proposal_analysis.py
│   ├── review_batches.py
│   ├── review_reports.py
│   ├── sweep_packs.py
│   ├── integrations/
│   └── workbench/assets/
├── schemas/
├── docs/
├── examples/
├── tests/
└── tools/
```

### 1.2 Current implemented strengths

The project is already more than a one-off detector. It currently supports:

- Gamma-kernel feature extraction and CFAR-style thresholding.
- Kalman-MCC feature extraction.
- Single-stage and two-stage Gamma processing paths.
- Pixel-level detection extraction and component-level detection extraction.
- FROC-style metrics and grid-search reporting.
- Review workbench generation and local workbench serving.
- Autosaved annotations and annotation schema migration.
- Review batches and review reports.
- Proposal analysis for next annotation targets.
- Dataset manifests and architecture-run manifests.
- Pipeline catalog with implemented and planned stages.
- Sweep packs.
- External importer scaffolds for Suite2p, PMD, and OASIS.
- Inverse dynamics export scaffold.
- Early online/adaptive EWMA-Z logic.
- Tests for annotation metrics, manifests, annotations, architecture runs, pipeline catalog, proposal analysis, review batches, sweep packs, online stage, workbench server, and workbench structure.

### 1.3 Items already partially fixed in current code

The existing historical `docs/plan.md` lists several P0 issues. Codex should verify before redoing them. Current inspected code indicates these are already addressed or partly addressed:

| Historical issue | Current state to verify |
|---|---|
| Metric uppercase/lowercase mismatch | `evaluation.metrics.metric_value()` exists, and `worker.py` records canonical uppercase plus lowercase compatibility copies. |
| CPU/no-CuPy annotation import issue | `core/filters.py` already uses `from __future__ import annotations`. |
| NumPy integer JSON encoding | `utils.NumpyEncoder` now returns `int(obj)` for `np.integer`. |
| Component-level detections | `utils.extract_component_detections()` exists and is tested. |

Do not spend a task reimplementing these unless tests show regression.

### 1.4 Current high-priority outstanding issues

The most concrete outstanding issues are:

1. `evaluation/metrics.py` still uses `np.trapz`; replace with `np.trapezoid`.
2. `environment.yml` contains an absolute `prefix`; replace docs with portable CPU/GPU/dev setup files.
3. `main.py` contains duplicated family-wise/combined reporting logic; extract reusable reporting functions.
4. CLI entrypoints are scattered across `tools/*.py`; introduce one `neurobench` command surface.
5. Dataset, pipeline, run, artifact, candidate, annotation, metrics, and export objects are not yet canonical Python models.
6. Pipeline catalog metadata is more mature than executable pipeline-stage registry.
7. Workbench builder still contains a large embedded fallback asset path; packaged assets should become canonical.
8. Run provenance and artifact registration are not yet first-class.
9. Object/event-level scientific metrics need to become primary alongside pixel-level FROC.
10. Documentation needs to distinguish stable workflows, experimental workflows, legacy scripts, and case studies.

---

## 2. North-Star Architecture

### 2.1 Platform goal

Build a modular, reproducible, inspectable, human-in-the-loop research platform for noisy zebrafish calcium/voltage imaging workflows.

The platform should cover:

```text
Raw video
  → dataset manifest
  → dataset QC
  → preprocessing and candidate proposal
  → review workbench
  → annotation and triage
  → metrics and comparison
  → reports
  → export bundle
  → downstream behavior / inverse dynamics analysis
```

### 2.2 Design principles

| Principle | Codex implementation implication |
|---|---|
| Stable data contracts | Add schemas/models before adding more UI/process complexity. |
| Human-in-the-loop | Treat annotations as first-class data, not side effects. |
| High recall with triage | Candidate generation can over-detect, but ranking/review burden metrics must exist. |
| Inspectability | Every output should link to producer stage, parameters, run ID, and artifact path. |
| Reproducibility | Every run writes a manifest, logs, warnings, parameter hash, and artifact table. |
| Extensibility | New detectors/denoisers/importers plug into shared stage/artifact interfaces. |
| CPU-first | New users should complete synthetic workflow without GPU or Fiji. |
| GPU/online-ready | Design abstractions now, defer heavy implementation. |
| Practical maintainability | Prefer small modules, clear CLI, clear tests, and simple schemas. |

### 2.3 Stable public artifact families

Codex should converge the repository around these artifacts:

```text
DatasetManifest
PipelineSpec
PipelineRun
ArtifactRecord
CandidateNeuron
CandidateEvent
ReviewData
Annotation
ReviewBatch
MetricsReport
ExportBundle
```

Every artifact should have:

```text
schema_version
id
source/provenance
paths or payload
created_at
optional warnings
optional extras
```

---

## 3. Target Repository Structure

Codex should migrate toward this structure gradually. Do not move everything in one pass.

```text
Separable-Gamma-CFAR-main/
├── pyproject.toml
├── README.md
├── LICENSE
├── environment.cpu.yml
├── environment.gpu.yml
├── requirements-dev.txt
├── configs/
│   ├── gamma_cfar_high_recall.yml
│   ├── kalman_mcc_baseline.yml
│   └── default_review_pipeline.yml
├── examples/
│   ├── manifests/
│   ├── pipelines/
│   ├── architecture_runs/
│   └── synthetic/
├── schemas/
│   ├── dataset_manifest.schema.json
│   ├── pipeline_spec.schema.json
│   ├── pipeline_run.schema.json
│   ├── artifact_record.schema.json
│   ├── candidate_neuron.schema.json
│   ├── candidate_event.schema.json
│   ├── annotations.schema.json
│   ├── review_batch.schema.json
│   ├── review_data.schema.json
│   ├── metrics_report.schema.json
│   └── export_bundle.schema.json
├── neurobench/
│   ├── cli/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── dataset.py
│   │   ├── run.py
│   │   ├── workbench.py
│   │   ├── review.py
│   │   ├── metrics.py
│   │   ├── report.py
│   │   ├── export.py
│   │   ├── importers.py
│   │   └── benchmark.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   ├── pipeline.py
│   │   ├── artifacts.py
│   │   ├── candidates.py
│   │   ├── annotations.py
│   │   ├── metrics.py
│   │   └── exports.py
│   ├── validation/
│   │   ├── __init__.py
│   │   ├── schemas.py
│   │   ├── paths.py
│   │   └── compatibility.py
│   ├── data/
│   │   ├── io.py
│   │   ├── video.py
│   │   ├── manifests.py
│   │   ├── checksums.py
│   │   └── qc.py
│   ├── algorithms/
│   │   ├── gamma.py
│   │   ├── cfar.py
│   │   ├── kalman_mcc.py
│   │   ├── denoise.py
│   │   ├── motion.py
│   │   ├── normalization.py
│   │   ├── components.py
│   │   └── traces.py
│   ├── pipelines/
│   │   ├── catalog.py
│   │   ├── specs.py
│   │   ├── executor.py
│   │   ├── stages.py
│   │   ├── artifacts.py
│   │   ├── grid_search.py
│   │   └── review_pipeline.py
│   ├── discovery/
│   │   ├── proposal_analysis.py
│   │   ├── ranking.py
│   │   ├── clustering.py
│   │   ├── active_learning.py
│   │   └── triage.py
│   ├── review/
│   │   ├── annotations.py
│   │   ├── batches.py
│   │   ├── agreement.py
│   │   └── reports.py
│   ├── metrics/
│   │   ├── detection.py
│   │   ├── event_quality.py
│   │   ├── annotation.py
│   │   ├── run_comparison.py
│   │   └── summaries.py
│   ├── reports/
│   │   ├── generators.py
│   │   ├── plots.py
│   │   ├── render.py
│   │   └── templates/
│   ├── workbench/
│   │   ├── builder.py
│   │   ├── server.py
│   │   ├── assets/
│   │   └── templates/
│   ├── exports/
│   │   ├── annotations.py
│   │   ├── inverse_dynamics.py
│   │   ├── behavior_alignment.py
│   │   └── bundles.py
│   ├── integrations/
│   │   ├── suite2p.py
│   │   ├── pmd.py
│   │   ├── oasis.py
│   │   └── registry.py
│   ├── realtime/
│   │   ├── online.py
│   │   ├── stream.py
│   │   ├── latency.py
│   │   └── state.py
│   └── logging/
│       ├── run_logger.py
│       └── events.py
├── core/
│   └── compatibility wrappers during migration
├── evaluation/
│   └── compatibility wrappers during migration
├── reporting/
│   └── compatibility wrappers during migration
├── workflows/
│   └── fiji/
├── scripts/
│   ├── legacy/
│   └── compatibility/
├── tools/
│   └── thin wrappers around neurobench.cli during migration
├── docs/
│   ├── quickstart.md
│   ├── installation.md
│   ├── concepts/
│   ├── workflows/
│   ├── reference/
│   ├── developer/
│   ├── case_studies/
│   └── archive/
└── tests/
    ├── unit/
    ├── integration/
    ├── e2e/
    ├── fixtures/
    └── browser_smoke/
```

### 3.1 Why this structure helps

- `algorithms/` separates reusable scientific functions from CLI/workbench code.
- `pipelines/` provides a stable stage executor and artifact store.
- `models/` and `schemas/` make artifact contracts explicit.
- `review/` owns human annotation logic independent of the browser UI.
- `workbench/` owns only build/serve/assets/templates.
- `metrics/` computes scientific measures; `reports/` renders them.
- `exports/` defines downstream artifact formats.
- `integrations/` isolates external tools.
- `realtime/` can mature without destabilizing offline batch workflows.
- `core/`, `evaluation/`, `reporting/`, and `tools/` can remain as compatibility layers until migration is complete.

---

## 4. Reduction Plan for Codex

The table below identifies simplifications and removals. Codex should not delete aggressively; mark deprecations first unless the item is clearly duplicate and covered by tests.

| ID | What to reduce | Why it causes friction | Replacement | Removal strategy |
|---|---|---|---|---|
| R-001 | `np.trapz` use in `evaluation/metrics.py` | Produces deprecation warnings and future compatibility risk. | `np.trapezoid` | Safe immediate replacement with test. |
| R-002 | Absolute `prefix` in `environment.yml` | Makes setup machine-specific. | `environment.cpu.yml`, `environment.gpu.yml`, `requirements-dev.txt`, docs. | Keep old env archived or remove prefix; update README. |
| R-003 | Duplicated family-wise/combined report logic in `main.py` | Inconsistent future edits; bloated entrypoint. | `neurobench.pipelines.grid_search` and/or `reporting.grid_search_summary`. | Extract function first; preserve old outputs. |
| R-004 | Script sprawl in `tools/` | Users must memorize many script names and path conventions. | Unified `neurobench` CLI with wrappers. | Gradually convert scripts to wrappers. |
| R-005 | Multiple implicit pipeline spec shapes | Catalog, architecture runs, examples, and scripts can drift. | Canonical `PipelineSpec` model/schema with migration helpers. | Support legacy shapes with warning. |
| R-006 | Loose artifact path conventions | Downstream tools depend on implicit filenames. | `ArtifactStore` and `ArtifactRecord`. | Add registry first; then migrate consumers. |
| R-007 | Pixel-level detections as primary evaluation unit | Pixel FPPI over-penalizes larger footprints and is less biologically meaningful. | Component/neuron/event metrics. | Keep pixel metrics as legacy/debug metrics. |
| R-008 | Embedded fallback workbench CSS/JS strings in builder | Duplicates packaged assets and obscures code. | Packaged files under `neurobench/workbench/assets/`. | Remove after packaging tests and docs update. |
| R-009 | Local dataset names and hardcoded paths in docs/tools | Makes examples nonportable. | Manifest-driven examples and CLI args. | Replace in docs; keep only as commented case-study paths. |
| R-010 | Mixed report-generation surfaces | Reports generated by `evaluation`, `reporting`, `review_reports`, and proposal analysis overlap. | `MetricsReport` + renderers. | Migrate one report family at a time. |
| R-011 | Planned pipeline stages presented too similarly to executable stages | Users may expect unavailable browser-side execution. | Stage availability enum: `executable`, `import_only`, `planned`, `deprecated`. | Update catalog/schema/docs together. |
| R-012 | Docs mixing tutorial, old plan, case study, and reference | Codex and users may follow stale instructions. | `docs/quickstart`, `docs/reference`, `docs/case_studies`, `docs/archive`. | Move stale docs only after adding replacements. |
| R-013 | One-off export names that imply accepted-only data | Can confuse downstream users. | Explicit export profiles: `all_reviewed`, `accepted_only`, `accepted_plus_uncertain`, `inverse_dynamics`. | Add new names; preserve old aliases temporarily. |
| R-014 | Algorithm code interwoven with multiprocessing/reporting | Harder to test and reuse. | `algorithms/` + `pipelines/` modules. | Introduce wrappers; migrate tests; then move. |
| R-015 | Browser-side expensive repeated trace modeling | Review UI can become slow. | Client-side trace model cache keyed by ROI and parameters. | Add cache with invalidation test/smoke check. |

---

## 5. Refactor Plan for Codex

### 5.1 Refactor F-001: Canonical data models

**Current problem**

Dataset, pipeline, run, annotation, candidate, metric, and export concepts are spread across schemas, examples, tools, and implicit dictionaries.

**Target architecture**

Add `neurobench/models/` with dataclass-based models and JSON helpers:

```text
neurobench/models/dataset.py       DatasetManifest
neurobench/models/pipeline.py      PipelineSpec, PipelineRun
neurobench/models/artifacts.py     ArtifactRecord, ArtifactStore metadata
neurobench/models/candidates.py    CandidateNeuron, CandidateEvent
neurobench/models/annotations.py   Annotation, ReviewBatch
neurobench/models/metrics.py       MetricsReport
neurobench/models/exports.py       ExportBundle
```

Use dataclasses plus explicit validation functions first. Avoid adding Pydantic unless project maintainers approve the dependency.

**Affected files**

- `neurobench/manifests.py`
- `neurobench/architecture_runs.py`
- `neurobench/annotations.py`
- `neurobench/review_batches.py`
- `neurobench/review_reports.py`
- `schemas/*.schema.json`
- `examples/*.json`
- `tools/*.py`

**Migration strategy**

1. Add models without removing existing functions.
2. Add `from_dict`, `to_dict`, `load_json`, `write_json` helpers.
3. Validate with existing schemas.
4. Update one CLI/tool at a time to use models internally.
5. Preserve existing output shape unless a schema version bump is introduced.

**Tests required**

- `test_dataset_manifest_model_roundtrip`
- `test_pipeline_spec_model_roundtrip`
- `test_pipeline_run_model_roundtrip`
- `test_annotation_model_roundtrip`
- `test_invalid_model_validation_messages`

---

### 5.2 Refactor F-002: Schema validation module

**Current problem**

Schemas exist but validation is not uniformly exposed as a reusable API or CLI behavior.

**Target architecture**

Add:

```text
neurobench/validation/schemas.py
neurobench/validation/paths.py
neurobench/validation/compatibility.py
```

Primary APIs:

```python
def schema_path(schema_name: str) -> Path: ...
def load_schema(schema_name: str) -> dict: ...
def validate_dict(payload: Mapping[str, Any], schema_name: str) -> None: ...
def validate_json(path: Path, schema_name: str) -> dict: ...
def validation_error_summary(exc: Exception) -> str: ...
```

**Affected files**

- `schemas/`
- `examples/`
- `tools/create_dataset_manifest.py`
- `tools/build_architecture_run.py`
- future CLI modules

**Migration strategy**

1. Add schema loader with repository-relative lookup.
2. Add tests on all existing examples.
3. Update tools to call validation before writing and after writing.
4. Add CLI validation commands.

**Tests required**

- `test_validate_existing_examples`
- `test_validate_json_missing_file`
- `test_validate_json_bad_schema_name`
- `test_validation_error_summary_contains_field_path`

---

### 5.3 Refactor F-003: Unified CLI

**Current problem**

Useful functions are available as many scripts in `tools/`, but the user has no coherent command surface.

**Target architecture**

Add `pyproject.toml` with:

```toml
[project.scripts]
neurobench = "neurobench.cli.main:main"
```

CLI groups:

```text
neurobench dataset create|validate|qc
neurobench run validate|execute|batch|sweep|compare|inspect
neurobench workbench build|serve|index
neurobench review batch|report|agreement
neurobench metrics compute
neurobench report generate
neurobench import suite2p|pmd|oasis
neurobench export annotations|inverse-dynamics|bundle
neurobench benchmark stage
neurobench validate dataset|pipeline|run|annotations|review-data|export
```

Use `argparse` first to avoid dependency churn.

**Affected files**

- New `neurobench/cli/`
- `tools/*.py`
- README/docs/tests

**Migration strategy**

1. Add `neurobench --help` only.
2. Add `dataset validate` and `validate dataset` as first real commands.
3. Wrap existing scripts rather than moving logic immediately.
4. Keep existing `tools/*.py` scripts working.
5. Add tests for CLI help and basic validation commands.

**Tests required**

- `test_cli_help`
- `test_cli_dataset_validate_example`
- `test_cli_invalid_command_exits_nonzero`
- `test_legacy_script_still_works`

---

### 5.4 Refactor F-004: Pipeline executor and stage registry

**Current problem**

`neurobench/pipeline_catalog.py` contains rich metadata, but executable behavior is scattered across Python scripts, Fiji/Groovy tools, and browser planning modes.

**Target architecture**

Add:

```text
neurobench/pipelines/catalog.py
neurobench/pipelines/specs.py
neurobench/pipelines/stages.py
neurobench/pipelines/executor.py
neurobench/pipelines/artifacts.py
```

Core stage protocol:

```python
class PipelineStage:
    stage_id: str
    availability: str
    input_artifacts: list[ArtifactSpec]
    output_artifacts: list[ArtifactSpec]

    def validate_params(self, params: dict) -> None:
        ...

    def run(self, context: PipelineContext) -> StageResult:
        ...
```

Initial executable stages should be small and CPU-safe:

- `dataset_qc`
- `gamma_cfar_candidate_proposal` or a synthetic/mock candidate stage for tests
- `component_filter`
- `review_data_build`

**Affected files**

- `neurobench/pipeline_catalog.py`
- `core/pipelines.py`
- `tools/run_neuron_review_pipeline.py`
- `tools/build_pipeline_run.py`
- `tools/benchmark_pipeline_stage.py`

**Migration strategy**

1. Keep current catalog metadata.
2. Add a stage registry with a few executable Python stages.
3. Mark planned-only stages clearly.
4. Add dry-run pipeline validation.
5. Add synthetic E2E pipeline execution.
6. Later migrate Fiji/Groovy stages into backend wrappers.

**Tests required**

- `test_stage_registry_lists_executable_stages`
- `test_pipeline_spec_dry_run`
- `test_pipeline_executor_missing_artifact_error`
- `test_synthetic_pipeline_run_writes_manifest`

---

### 5.5 Refactor F-005: Artifact store and run layout

**Current problem**

Outputs and intermediate files are useful but loosely named. Consumers infer paths rather than reading a formal manifest.

**Target architecture**

Every run should have:

```text
run_root/
├── pipeline_run.json
├── logs/
│   ├── run.log
│   └── stages/
├── artifacts/
│   ├── raw_qc/
│   ├── preprocessing/
│   ├── candidates/
│   ├── traces/
│   ├── review_data/
│   ├── metrics/
│   └── reports/
├── workbench/
└── exports/
```

Add `ArtifactRecord` fields:

```json
{
  "artifact_id": "candidate_events.v1",
  "kind": "candidate_events",
  "path": "artifacts/candidates/events.json",
  "schema": "candidate_event.schema.json",
  "producer_stage": "gamma_cfar_candidate_proposal",
  "created_at": "...",
  "sha256": "...",
  "summary": {}
}
```

**Affected files**

- New `neurobench/pipelines/artifacts.py`
- New schema `artifact_record.schema.json`
- `tools/run_neuron_review_pipeline.py`
- workbench build tools
- export tools

**Migration strategy**

1. Add `ArtifactRecord` and `ArtifactStore` without changing old outputs.
2. Start registering artifacts generated by new CLI commands.
3. Update workbench/report/export commands to prefer `pipeline_run.json` artifact table.
4. Preserve old path args as fallback.

**Tests required**

- `test_artifact_record_schema`
- `test_artifact_store_registers_checksum`
- `test_artifact_store_rejects_missing_file`
- `test_pipeline_run_contains_artifacts`

---

### 5.6 Refactor F-006: Algorithm module separation

**Current problem**

Core algorithms are partly reusable but still coupled to legacy import paths and reporting/grid-search behavior.

**Target architecture**

Gradually expose algorithm modules:

```text
neurobench/algorithms/gamma.py
neurobench/algorithms/cfar.py
neurobench/algorithms/kalman_mcc.py
neurobench/algorithms/components.py
neurobench/algorithms/denoise.py
neurobench/algorithms/motion.py
neurobench/algorithms/traces.py
```

Keep compatibility wrappers in `core/`.

**Affected files**

- `core/filters.py`
- `core/detection.py`
- `core/pipelines.py`
- `utils.py`
- `worker.py`
- `tests/test_correctness_foundations.py`

**Migration strategy**

1. Add new modules that import or wrap existing functions.
2. Move pure functions first, not high-level workflows.
3. Add equivalence tests.
4. Change internal imports gradually.
5. Leave `core/` wrappers until a later major cleanup.

**Tests required**

- `test_algorithm_imports_cpu_only`
- `test_component_detections_equivalent_after_move`
- `test_gamma_pipeline_equivalent_after_move`
- `test_cfar_threshold_equivalent_after_move`

---

### 5.7 Refactor F-007: Workbench builder/server separation

**Current problem**

Workbench generation, static assets, data shaping, server endpoints, and job-generation logic are spread across large scripts and assets.

**Target architecture**

```text
neurobench/workbench/builder.py
neurobench/workbench/server.py
neurobench/workbench/assets/workbench.js
neurobench/workbench/assets/workbench.css
neurobench/workbench/templates/
```

Rules:

- Builder validates `ReviewData` before writing HTML.
- Server handles safe local file serving and annotation save endpoints.
- Assets live as package files, not fallback strings.
- Generation jobs remain clearly marked local/experimental.

**Affected files**

- `tools/build_neuron_workbench_v2.py`
- `tools/serve_neuron_workbench.py`
- `neurobench/workbench/assets/workbench.js`
- `neurobench/workbench/assets/workbench.css`
- `schemas/review_data.schema.json`

**Migration strategy**

1. Extract asset loading function.
2. Add package-data test.
3. Add builder function called by old script.
4. Add server function called by old script.
5. Remove embedded fallback after tests and packaging work.

**Tests required**

- `test_workbench_assets_packaged`
- `test_build_workbench_outputs_html`
- `test_server_rejects_path_traversal`
- `test_annotation_save_roundtrip`

---

## 6. Codex Roadmap by Phase

## Phase 0: Audit, Cleanup, and Stabilization

### Goal

Clean up immediate warnings, portability issues, and duplicated legacy code without changing scientific behavior.

### Tasks

1. Replace `np.trapz` with `np.trapezoid`.
2. Add or update test to assert truncated AUC still computes correctly.
3. Create portable environment files:
   - `environment.cpu.yml`
   - `environment.gpu.yml`
   - `requirements-dev.txt`
4. Remove or neutralize absolute `prefix` from setup docs.
5. Extract duplicate reporting logic from `main.py`.
6. Update README to describe stable, experimental, and legacy workflows.
7. Archive or rename stale planning docs only after README/quickstart replacement exists.
8. Add a “current repository status” note to docs.
9. Run full tests and record any warnings.

### Affected files/modules

```text
evaluation/metrics.py
main.py
reporting/generators.py
reporting/plotters.py
environment.yml
README.md
docs/plan.md
docs/quickstart.md          new
docs/archive/               new
tests/
```

### Expected outputs

- Test suite passes with no `np.trapz` warning.
- Portable setup files exist.
- `main.py` is shorter and uses reusable report-generation helper(s).
- README provides a project map and conservative quickstart.
- Existing behavior preserved.

### Acceptance criteria

- `pytest -q` passes.
- No `np.trapz` deprecation warning.
- CPU-only import smoke tests pass.
- `main.py` output filenames remain compatible.
- README does not require local absolute paths.

### Suggested tests

```text
test_truncated_auc_uses_trapezoid_behavior
test_grid_search_report_helper_preserves_output_paths
test_environment_files_have_no_absolute_prefix
test_readme_references_neurobench_quickstart
```

---

## Phase 1: Repository Architecture and Interface Consolidation

### Goal

Introduce canonical models, validation helpers, and the first unified CLI commands while preserving existing scripts.

### Tasks

1. Add `pyproject.toml` with editable-install metadata and console script.
2. Add `neurobench/cli/main.py` with `neurobench --help`.
3. Add `neurobench/validation/schemas.py`.
4. Add schema validation tests for all current examples.
5. Add `DatasetManifest` and `PipelineSpec` models.
6. Add `PipelineRun` and `ArtifactRecord` model stubs.
7. Add CLI commands:
   - `neurobench dataset validate`
   - `neurobench validate dataset`
   - `neurobench run validate`
8. Keep old scripts working.
9. Update docs to show both new CLI and legacy fallback.

### Affected files/modules

```text
pyproject.toml
neurobench/cli/
neurobench/models/
neurobench/validation/
schemas/
examples/
tools/create_dataset_manifest.py
tools/build_pipeline_run.py
tests/
```

### Expected outputs

- `neurobench --help` works after `pip install -e .`.
- Existing examples validate through CLI.
- Models roundtrip current example JSON.
- Legacy scripts still run.

### Acceptance criteria

- CLI smoke tests pass.
- Existing manifest and pipeline examples validate.
- Invalid examples produce field-specific errors.
- No public output format is broken.

### Suggested tests

```text
test_cli_help
test_cli_dataset_validate_example
test_validate_all_examples
test_dataset_manifest_model_roundtrip
test_pipeline_spec_model_roundtrip
test_legacy_create_dataset_manifest_script_still_works
```

---

## Phase 2: Reliable Batch Processing and Reproducible Runs

### Goal

Make processing runs reproducible, inspectable, and batch-capable.

### Tasks

1. Implement `PipelineRun` model fully.
2. Implement `ArtifactStore`.
3. Add run-root layout helper.
4. Add parameter hashing.
5. Add input checksums.
6. Add structured run logging.
7. Add stage registry and dry-run executor.
8. Convert one small CPU-safe synthetic pipeline into executable stages.
9. Create tiny synthetic fixture.
10. Add `neurobench run execute` for synthetic/default pipeline.
11. Add `neurobench run batch` after single-run flow is stable.

### Affected files/modules

```text
neurobench/models/pipeline.py
neurobench/models/artifacts.py
neurobench/pipelines/executor.py
neurobench/pipelines/stages.py
neurobench/pipelines/artifacts.py
neurobench/data/checksums.py
neurobench/logging/run_logger.py
neurobench/cli/run.py
tests/fixtures/synthetic_dataset/
tests/e2e/
```

### Expected outputs

```text
run_root/
├── pipeline_run.json
├── logs/
├── artifacts/
└── workbench/
```

### Acceptance criteria

- Every new run writes valid `pipeline_run.json`.
- Every generated artifact is registered with path, kind, producer stage, and checksum.
- Batch runner records success/failure per dataset.
- Synthetic E2E run completes in CI.

### Suggested tests

```text
test_pipeline_run_manifest_created
test_artifact_store_registers_checksum
test_run_parameter_hash_is_stable
test_pipeline_executor_dry_run
test_synthetic_pipeline_e2e
test_batch_runner_records_failure_without_losing_successes
```

---

## Phase 3: Workbench UX and Annotation/Review Improvements

### Goal

Make candidate review faster, clearer, and more reliable while preserving current browser workbench behavior.

### Tasks

1. Strengthen `review_data.schema.json`.
2. Add `ReviewData` model or validation helper.
3. Add annotation reason tags and confidence labels if not already schema-supported.
4. Add search/filter state to workbench data model.
5. Add guided review queues:
   - unreviewed high score
   - uncertain
   - likely artifact
   - possible missed neuron
   - needs second reviewer
6. Add client-side trace model caching.
7. Ensure keyboard shortcuts are documented and tested by smoke tests.
8. Extract workbench builder logic into `neurobench/workbench/builder.py`.
9. Extract server logic into `neurobench/workbench/server.py`.
10. Remove embedded fallback assets once package-data test passes.

### Affected files/modules

```text
schemas/review_data.schema.json
schemas/annotations.schema.json
neurobench/annotations.py
neurobench/review_batches.py
neurobench/workbench/assets/workbench.js
neurobench/workbench/assets/workbench.css
tools/build_neuron_workbench_v2.py
tools/serve_neuron_workbench.py
tests/test_workbench_structure.py
tests/test_workbench_server.py
```

### Expected outputs

- Validated workbench data.
- Faster trace/event browsing.
- Better review queues.
- Structured annotation metadata.
- Package-based assets.

### Acceptance criteria

- Workbench still builds from existing fixtures.
- Annotation save/load roundtrip preserves new fields.
- Trace cache invalidates when trace parameters change.
- Search/filter logic handles empty and malformed cases safely.
- Server path-safety tests still pass.

### Suggested tests

```text
test_review_data_schema_accepts_current_fixture
test_annotation_reason_tags_confidence_roundtrip
test_review_batch_uncertain_queue
test_workbench_assets_packaged
test_workbench_trace_cache_hooks_present
test_workbench_server_save_annotations_roundtrip
```

---

## Phase 4: Metrics, Reporting, and Scientific Auditability

### Goal

Make outputs scientifically interpretable and audit-ready.

### Tasks

1. Add object-level matching metrics.
2. Add event-level timing/quality metrics.
3. Add annotation agreement metrics for multi-reviewer workflows.
4. Define `MetricsReport` schema/model.
5. Add report renderer for Markdown and JSON summary.
6. Add comparison report across runs.
7. Add provenance appendix to reports.
8. Add warning/limitation sections to reports.
9. Preserve legacy grid-search reports.

### Affected files/modules

```text
evaluation/metrics.py
evaluation/analysis.py
neurobench/annotation_metrics.py
neurobench/review_reports.py
neurobench/proposal_analysis.py
neurobench/metrics/
neurobench/reports/
schemas/metrics_report.schema.json
reporting/
tests/
```

### Expected outputs

```text
metrics_report.json
report.md
report.html                  optional after Markdown stable
comparison_report.md
annotation_agreement_report.json
```

### Acceptance criteria

- Pixel, object, event, annotation, runtime, and provenance metrics are separated.
- Metrics report validates against schema.
- Synthetic ground truth produces expected object/event metrics.
- Reports can be generated from `pipeline_run.json`.

### Suggested tests

```text
test_object_level_matching_iou_and_centroid
test_event_timing_metrics
test_annotation_agreement_binary_labels
test_metrics_report_schema
test_report_contains_provenance_and_warnings
test_legacy_grid_search_reports_still_generate
```

---

## Phase 5: Discovery, Active Learning, and Comparison Tools

### Goal

Reduce review burden and improve candidate discovery across pipelines.

### Tasks

1. Add candidate feature table extraction.
2. Add transparent candidate ranking.
3. Add triage suggestion model:
   - likely true
   - likely artifact
   - uncertain
   - possible missed neuron
   - possible merge
   - possible split
4. Add candidate clustering by spatial overlap and temporal correlation.
5. Add active review batch generator.
6. Add false-positive summary by artifact tag and score bin.
7. Add false-negative/missed-neuron candidate analysis.
8. Add pipeline comparison dashboard/report.
9. Add sweep execution and Pareto summary.

### Affected files/modules

```text
neurobench/discovery/
neurobench/proposal_analysis.py
neurobench/review_batches.py
neurobench/sweep_packs.py
neurobench/architecture_runs.py
neurobench/metrics/run_comparison.py
neurobench/workbench/assets/workbench.js
schemas/review_batch.schema.json
schemas/candidate_neuron.schema.json
schemas/candidate_event.schema.json
```

### Expected outputs

- Ranked candidate list.
- Review batches from ranking/uncertainty.
- Candidate clusters.
- False-positive/false-negative reports.
- Pipeline comparison reports.
- Sweep summaries.

### Acceptance criteria

- Ranking is deterministic.
- Ranking includes explanation fields.
- Clustering is reproducible and links candidate IDs.
- Active batch has higher uncertainty/high-value concentration than random on synthetic fixture.
- Comparison report identifies consensus and unique candidates.

### Suggested tests

```text
test_candidate_feature_table_schema
test_candidate_ranking_is_deterministic
test_candidate_ranking_explanations_present
test_candidate_clustering_spatial_overlap
test_active_batch_prioritizes_uncertain_candidates
test_false_positive_summary_links_candidates
test_pipeline_comparison_identifies_unique_candidates
test_sweep_pareto_summary
```

---

## Phase 6: Performance, GPU Acceleration, and Online Inference

### Goal

Prepare the platform for large datasets, optional GPU acceleration, and future near-real-time inference without destabilizing CPU workflows.

### Tasks

1. Add `VideoStore` abstraction.
2. Add chunked video iteration with overlap.
3. Add memory budget estimator.
4. Add device abstraction:
   - `cpu`
   - `cuda`
   - `auto`
5. Add optional GPU tests guarded by availability.
6. Add CPU/GPU equivalence tests on small arrays.
7. Expand online inference path into `realtime/` module.
8. Add streaming frame source and latency metrics.
9. Add online/offline comparison on synthetic fixture.

### Affected files/modules

```text
neurobench/data/video.py
neurobench/algorithms/
neurobench/pipelines/executor.py
neurobench/realtime/
neurobench/online.py
tools/benchmark_pipeline_stage.py
tests/
```

### Expected outputs

- Chunked processing support.
- Memory warnings before expensive runs.
- Optional GPU execution path.
- Online synthetic stream demo/test.
- Latency report.

### Acceptance criteria

- Chunked and unchunked synthetic outputs match within tolerance.
- CPU fallback works when GPU is unavailable.
- GPU tests skip cleanly when GPU unavailable.
- Online stream produces bounded-latency detections on fixture.

### Suggested tests

```text
test_video_store_iter_chunks
test_chunked_processing_equivalence
test_memory_budget_warning
test_device_auto_falls_back_to_cpu
test_gpu_optional_smoke
test_online_stream_latency_report
test_online_offline_synthetic_agreement
```

---

## Phase 7: Inverse Dynamics Export and Downstream Integration

### Goal

Make final outputs useful for behavior/control modeling and downstream scientific analysis.

### Tasks

1. Define `ExportBundle` schema/model.
2. Add export profile selection:
   - `accepted_only`
   - `accepted_plus_uncertain`
   - `all_reviewed`
   - `inverse_dynamics`
3. Extend inverse-dynamics export with:
   - frame/time mapping
   - alignment status
   - behavior file metadata
   - sync diagnostics
   - resampling/interpolation policy
4. Add downstream loader helper.
5. Add export report.
6. Add docs workflow.

### Affected files/modules

```text
tools/export_annotations.py
tools/export_inverse_dynamics.py
neurobench/exports/
schemas/export_bundle.schema.json
docs/INVERSE_DYNAMICS_EXPORT.md
docs/workflows/inverse_dynamics_export.md
tests/
```

### Expected outputs

```text
export_bundle.json
accepted_traces.tsv
accepted_events.tsv
neuron_metadata.tsv
alignment_report.json
export_report.md
checksums.json
```

### Acceptance criteria

- Export bundle validates.
- Export selection policy is explicit.
- Alignment status is explicit:
  - `not_provided`
  - `provided_unvalidated`
  - `validated`
  - `failed`
- Downstream loader can read bundle without workbench internals.
- Export report includes provenance and limitations.

### Suggested tests

```text
test_export_bundle_schema
test_export_annotations_accepted_only
test_inverse_dynamics_export_alignment_status
test_export_checksums
test_downstream_loader_roundtrip
test_export_report_contains_provenance
```

---

## 7. Detailed Prioritized Task Board

| ID | Priority | Category | Task | Rationale | Files/Modules | Dependencies | Acceptance Criteria |
|---|---:|---|---|---|---|---|---|
| NB-000 | P0 | Audit | Add a short current-state audit note or test log to docs. | Prevents Codex from following stale historical issues blindly. | `docs/`, `README.md` | None | Note states current test baseline and known warnings. |
| NB-001 | P0 | Metrics | Replace `np.trapz` with `np.trapezoid`. | Removes current deprecation warning. | `evaluation/metrics.py`, tests | None | `pytest -q` passes with no trapz warning. |
| NB-002 | P0 | Tests | Add/confirm CPU-only import smoke test. | Protects non-GPU onboarding. | `tests/test_correctness_foundations.py`, `core/filters.py` | None | Test passes when CuPy/GPU unavailable. |
| NB-003 | P0 | Setup | Add `environment.cpu.yml`. | Portable CPU setup is required for onboarding. | repo root | None | No absolute prefix; includes pytest/jsonschema/numpy/scipy/torch CPU-compatible guidance. |
| NB-004 | P0 | Setup | Add `environment.gpu.yml`. | GPU users need optional setup without blocking CPU users. | repo root | NB-003 | No absolute prefix; clearly optional. |
| NB-005 | P0 | Setup | Add `requirements-dev.txt`. | Simplifies editable development/test setup. | repo root | None | `pip install -r requirements-dev.txt` supports tests in a clean env. |
| NB-006 | P0 | Docs | Rewrite README project map. | Reduces confusion about legacy vs stable vs experimental paths. | `README.md` | NB-003 | README has install, quickstart, module map, tests, and known limitations. |
| NB-007 | P0 | Cleanup | Extract duplicate grid-search report logic from `main.py`. | Reduces maintenance risk. | `main.py`, `reporting/` | NB-001 | Existing report outputs preserved by regression test. |
| NB-008 | P0 | Validation | Add schema loader/validator helper. | Schemas become executable contracts. | `neurobench/validation/schemas.py`, `schemas/` | None | Existing examples validate; invalid fixture fails clearly. |
| NB-009 | P0 | CLI | Add `pyproject.toml` and `neurobench --help`. | Creates stable command surface. | `pyproject.toml`, `neurobench/cli/main.py` | NB-005 | Editable install exposes `neurobench`; help exits 0. |
| NB-010 | P0 | CLI | Add `neurobench dataset validate`. | First useful CLI command. | `neurobench/cli/dataset.py`, validation | NB-008, NB-009 | Example manifest validates through CLI. |
| NB-011 | P0 | Models | Add `DatasetManifest` model. | Stable dataset intake object. | `neurobench/models/dataset.py` | NB-008 | Roundtrip example JSON. |
| NB-012 | P0 | Models | Add `PipelineSpec` model. | Stable pipeline configuration object. | `neurobench/models/pipeline.py` | NB-008 | Roundtrip example pipeline spec. |
| NB-013 | P0 | Models | Add `PipelineRun` model skeleton. | Foundation for reproducible runs. | `neurobench/models/pipeline.py`, schema | NB-012 | Minimal run manifest validates. |
| NB-014 | P0 | Testing | Add tiny synthetic dataset fixture generator. | Enables CI E2E without lab data. | `tests/fixtures/`, `neurobench/data/` | NB-011 | Fixture generates small known video/events. |
| NB-015 | P1 | Artifacts | Add `ArtifactRecord` and `ArtifactStore`. | Standardizes intermediate files. | `neurobench/models/artifacts.py`, `neurobench/pipelines/artifacts.py` | NB-013 | Artifacts register with checksum and schema. |
| NB-016 | P1 | Provenance | Add run layout helper. | Makes outputs reproducible and navigable. | `neurobench/pipelines/artifacts.py` | NB-015 | Creates `pipeline_run.json`, `logs/`, `artifacts/`. |
| NB-017 | P1 | Provenance | Add parameter hashing. | Enables exact run comparison. | `neurobench/pipelines/specs.py` | NB-012 | Stable hash independent of dict key ordering. |
| NB-018 | P1 | Provenance | Add input file checksums. | Supports scientific auditability. | `neurobench/data/checksums.py` | NB-011 | SHA256 helper tested on fixture. |
| NB-019 | P1 | Pipelines | Add stage registry. | Connects catalog to executable stages. | `neurobench/pipelines/stages.py` | NB-012 | Registry lists executable/planned/import-only stages. |
| NB-020 | P1 | Pipelines | Add dry-run pipeline validator. | Catches bad specs before expensive processing. | `neurobench/pipelines/executor.py` | NB-019 | Missing stage/params/artifacts fail clearly. |
| NB-021 | P1 | Pipelines | Add minimal synthetic executable pipeline. | Creates first reproducible E2E run. | `neurobench/pipelines/`, tests/e2e | NB-014, NB-020 | Synthetic run writes valid manifest and artifacts. |
| NB-022 | P1 | CLI | Add `neurobench run validate`. | Pipeline specs become user-facing. | `neurobench/cli/run.py` | NB-020 | Valid/invalid specs handled clearly. |
| NB-023 | P1 | CLI | Add `neurobench run execute` for synthetic/simple pipeline. | First real execution command. | `neurobench/cli/run.py` | NB-021 | Run command writes run root. |
| NB-024 | P1 | Workbench | Extract packaged asset loader. | Reduces fallback duplication. | `tools/build_neuron_workbench_v2.py`, `neurobench/workbench/` | NB-009 | Asset package test passes. |
| NB-025 | P1 | Workbench | Extract builder module. | Makes workbench generation testable. | `neurobench/workbench/builder.py` | NB-024 | Old script delegates to builder; output smoke test passes. |
| NB-026 | P1 | Workbench | Extract server module. | Makes local server testable and reusable. | `neurobench/workbench/server.py`, `tools/serve_neuron_workbench.py` | NB-025 | Existing server path-safety tests pass. |
| NB-027 | P1 | Workbench | Add trace model cache in JS. | Improves review performance. | `neurobench/workbench/assets/workbench.js` | NB-024 | Cache invalidation hooks present and smoke-tested. |
| NB-028 | P1 | Review | Strengthen annotation reason/confidence schema. | Increases review data value. | `schemas/annotations.schema.json`, `neurobench/annotations.py` | NB-008 | Roundtrip preserves confidence/tags. |
| NB-029 | P1 | Review | Add guided review queue API. | Reduces annotation burden. | `neurobench/review_batches.py` | NB-028 | Queue deterministic and tested. |
| NB-030 | P1 | Metrics | Add object-level candidate matching. | Neuron discovery needs object metrics. | `neurobench/metrics/detection.py`, `evaluation/metrics.py` | NB-014 | Synthetic object metrics match expected. |
| NB-031 | P1 | Metrics | Add event-level timing metrics. | Activity analysis needs event metrics. | `neurobench/metrics/event_quality.py` | NB-014 | Known event timings produce expected errors. |
| NB-032 | P1 | Reports | Add `MetricsReport` schema/model. | Report outputs become structured. | `schemas/metrics_report.schema.json`, `neurobench/models/metrics.py` | NB-030, NB-031 | Metrics report validates. |
| NB-033 | P1 | Reports | Add Markdown report renderer. | Human-readable audit output. | `neurobench/reports/render.py` | NB-032 | Synthetic report includes QC/run/metrics/provenance. |
| NB-034 | P2 | Data | Add dataset QC command. | Catches drift/saturation/memory issues early. | `neurobench/data/qc.py`, CLI | NB-014 | QC report generated from synthetic fixture. |
| NB-035 | P2 | Algorithms | Add robust local normalization/CFAR stage. | Improves heterogeneous fluorescence handling. | `neurobench/algorithms/cfar.py` | NB-019 | Background-gradient synthetic test improves FP rate. |
| NB-036 | P2 | Algorithms | Add motion/drift estimation stage. | Detects morphology shifts and acquisition drift. | `neurobench/algorithms/motion.py` | NB-034 | Shifted synthetic video produces expected shift estimate. |
| NB-037 | P2 | Discovery | Add candidate feature table. | Foundation for ranking/triage/active learning. | `neurobench/discovery/ranking.py` | NB-030 | Feature table validates and is deterministic. |
| NB-038 | P2 | Discovery | Add explainable candidate ranking. | Prioritizes human review. | `neurobench/discovery/ranking.py` | NB-037 | Ranking includes reason/explanation fields. |
| NB-039 | P2 | Discovery | Add candidate clustering. | Flags duplicates/split/merge cases. | `neurobench/discovery/clustering.py` | NB-037 | Overlapping candidates cluster correctly. |
| NB-040 | P2 | Discovery | Add active review batch generator. | Reduces manual review burden. | `neurobench/discovery/active_learning.py` | NB-038 | Batch prioritizes uncertain/high-value items. |
| NB-041 | P2 | Comparison | Add run comparison metrics. | Supports algorithm selection. | `neurobench/metrics/run_comparison.py` | NB-032 | Consensus/unique candidates identified. |
| NB-042 | P2 | Sweeps | Add first-class sweep execution/reporting. | Reproducible tuning. | `neurobench/sweep_packs.py`, `neurobench/pipelines/` | NB-041 | Sweep creates runs and comparison report. |
| NB-043 | P2 | Review | Add annotation agreement/adjudication report. | Supports multi-reviewer lab workflows. | `neurobench/review/agreement.py` | NB-028 | Disagreement batch generated. |
| NB-044 | P2 | Export | Define `ExportBundle`. | Stable downstream contract. | `schemas/export_bundle.schema.json`, `neurobench/models/exports.py` | NB-032 | Export bundle validates. |
| NB-045 | P2 | Export | Convert annotation export to profiles. | Avoids accepted/all-reviewed ambiguity. | `tools/export_annotations.py`, `neurobench/exports/annotations.py` | NB-044 | `accepted_only` export tested. |
| NB-046 | P2 | Export | Extend inverse-dynamics export. | Supports behavior/control modeling. | `tools/export_inverse_dynamics.py`, `neurobench/exports/inverse_dynamics.py` | NB-044 | Alignment status and frame/time mapping included. |
| NB-047 | P2 | Docs | Add raw-video-to-report tutorial. | Onboards non-experts. | `docs/workflows/raw_video_to_report.md` | NB-023, NB-033 | Tutorial uses only relative paths/synthetic data. |
| NB-048 | P2 | Docs | Add developer guide for pipeline stages. | Helps future method extension. | `docs/developer/adding_pipeline_stage.md` | NB-019 | Guide includes tested minimal stage example. |
| NB-049 | P3 | Performance | Add `VideoStore` abstraction. | Prepares memory-aware large video processing. | `neurobench/data/video.py` | NB-034 | Chunk iterator tested. |
| NB-050 | P3 | Performance | Add chunked processing equivalence test. | Prevents boundary artifacts. | pipelines/algorithms tests | NB-049 | Chunked/un-chunked synthetic outputs match. |
| NB-051 | P3 | GPU | Add device abstraction. | CPU/GPU code paths become explicit. | `neurobench/pipelines/executor.py`, algorithms | NB-049 | `auto` falls back to CPU. |
| NB-052 | P3 | GPU | Add optional GPU smoke tests. | Enables acceleration without requiring GPU. | tests | NB-051 | Tests skip cleanly when unavailable. |
| NB-053 | P3 | Realtime | Create `realtime/stream.py`. | Future online inference foundation. | `neurobench/realtime/` | NB-049 | Synthetic frame source tested. |
| NB-054 | P3 | Realtime | Add latency report. | Measures online feasibility. | `neurobench/realtime/latency.py` | NB-053 | Bounded latency report on fixture. |
| NB-055 | P3 | Plugins | Add plugin registry for stages/importers. | External methods integrate cleanly. | `neurobench/integrations/registry.py` | NB-019 | Plugin stage can register and validate. |
| NB-056 | P3 | UX | Add visual split/merge editing. | Improves complex neuron morphology review. | workbench JS/CSS, annotation schema | NB-039 | Split/merge decision exports cleanly. |
| NB-057 | P3 | Analysis | Add population time-series summaries. | Supports neuroscience analysis beyond detection. | `neurobench/metrics/summaries.py` | NB-044 | Event raster/correlation summaries generated. |
| NB-058 | P3 | Behavior | Add behavior alignment diagnostics. | Required for inverse dynamics. | `neurobench/exports/behavior_alignment.py` | NB-046 | Sync/resampling diagnostics tested. |
| NB-059 | P3 | Reports | Add HTML report renderer. | Better shareable output. | `neurobench/reports/render.py` | NB-033 | HTML generated from same report model. |
| NB-060 | P3 | Docs | Add API reference generation. | Improves long-term maintainability. | docs tooling | NB-009 | Reference generated without internet or raw data. |

---

## 8. First 10 Codex Work Packages

These are the recommended first 10 task prompts to give Codex. Each work package should be implemented separately.

### Work Package 1 — Remove current metrics warning

```text
Inspect evaluation/metrics.py and tests around truncated AUC/FROC metrics. Replace np.trapz with np.trapezoid while preserving behavior. Add or update a test that verifies calculate_truncated_auc returns the same expected value for a small hand-computable FROC curve. Run targeted tests and full pytest.
```

Acceptance:

- `pytest -q` passes.
- No `np.trapz` warning remains.
- No metric output key behavior changes.

---

### Work Package 2 — Add portable development environment files

```text
Inspect environment.yml, README.md, and tests to infer required dependencies. Add environment.cpu.yml, environment.gpu.yml, and requirements-dev.txt without absolute prefixes. Preserve the existing environment.yml if needed, but update docs to recommend the portable CPU environment for onboarding. Do not add heavyweight new dependencies.
```

Acceptance:

- New env files have no absolute `prefix`.
- README points to CPU setup first.
- Tests still pass.

---

### Work Package 3 — Add pyproject and CLI skeleton

```text
Add pyproject.toml with project metadata and a console script named neurobench pointing to neurobench.cli.main:main. Implement a minimal argparse-based CLI with --help and subcommand placeholders. Add tests that call the CLI help entrypoint without requiring raw data, Fiji, or GPU.
```

Acceptance:

- `pip install -e .` exposes `neurobench`.
- `neurobench --help` exits 0.
- Test coverage added.

---

### Work Package 4 — Add schema validation helpers

```text
Create neurobench/validation/schemas.py with schema_path, load_schema, validate_dict, validate_json, and validation_error_summary helpers. Use existing schemas and examples. Add tests that validate current examples and produce useful errors for a deliberately invalid manifest.
```

Acceptance:

- Current examples validate.
- Invalid example fails with a field-specific message.
- No existing tool behavior breaks.

---

### Work Package 5 — Add dataset validation CLI

```text
Implement neurobench dataset validate using the schema validation helper. Keep tools/create_dataset_manifest.py working. Add CLI tests for validating examples/dataset_manifest.example.json and for failure on a malformed temporary manifest.
```

Acceptance:

- `neurobench dataset validate examples/dataset_manifest.example.json` succeeds.
- Malformed manifest exits nonzero.
- Legacy script remains usable.

---

### Work Package 6 — Add DatasetManifest and PipelineSpec models

```text
Create neurobench/models/dataset.py and neurobench/models/pipeline.py with dataclass-based DatasetManifest and PipelineSpec models. Include from_dict, to_dict, load_json, write_json, and validate methods. Roundtrip existing examples without changing their public JSON shape.
```

Acceptance:

- Existing examples roundtrip.
- Models call schema validation.
- No Pydantic or large new dependency added.

---

### Work Package 7 — Add PipelineRun and ArtifactRecord foundations

```text
Extend neurobench/models/pipeline.py with PipelineRun and create neurobench/models/artifacts.py with ArtifactRecord. Add JSON schemas if absent. Include run_id, dataset_id, pipeline_spec_id, status, timestamps, parameter_hash, environment, code, artifacts, metrics, warnings, and logs. Add minimal validation tests.
```

Acceptance:

- Minimal valid run manifest passes.
- Invalid run missing IDs fails clearly.
- ArtifactRecord includes path/kind/producer/checksum fields.

---

### Work Package 8 — Extract main.py report duplication

```text
Inspect main.py reporting flow. Extract duplicated family-wise and combined report generation into reusable helper functions without changing output paths or filenames. Add a regression-style test or smoke test that verifies helper dispatch can run on a tiny synthetic all_results structure. Keep main.py as the entrypoint.
```

Acceptance:

- `main.py` has one report-generation path for shared logic.
- Existing reporting APIs are preserved.
- Tests pass.

---

### Work Package 9 — Add tiny synthetic fixture generator

```text
Add a tiny synthetic dataset fixture generator for tests. It should create a small video array with a few known candidate neuron/event locations and optional simple artifacts such as background gradient or drift. Store generated data under tests/fixtures or generate in pytest tmp_path. Do not commit large binary outputs.
```

Acceptance:

- Fixture is tiny and CI-safe.
- It includes known event coordinates/timing.
- It can support future object/event metrics tests.

---

### Work Package 10 — Add ArtifactStore and run layout helper

```text
Implement neurobench/pipelines/artifacts.py with ArtifactStore and a run layout helper that creates pipeline_run.json, logs/, artifacts/, workbench/, and exports/ directories. Add checksum registration for files. Add tests using tmp_path.
```

Acceptance:

- ArtifactStore registers file path, kind, schema, producer stage, checksum, and summary.
- Missing file registration fails clearly.
- Run layout is deterministic.

---

## 9. Core Data Object Definitions

These are minimal target definitions. Codex should keep them simple and versioned. Fields can include optional `extras` dictionaries for research flexibility.

### 9.1 DatasetManifest

```json
{
  "schema_version": "1.0",
  "dataset_id": "zfish_001",
  "name": "Example zebrafish recording",
  "species": "Danio rerio",
  "sample": {
    "animal_id": "fish_001",
    "age_dpf": 6,
    "preparation": "light-sheet calcium imaging"
  },
  "acquisition": {
    "modality": "calcium",
    "frame_rate_hz": 10.0,
    "pixel_size_um": [1.0, 1.0],
    "z_planes": 1,
    "channels": ["fluorescence"]
  },
  "paths": {
    "raw_video": "data/raw_video.tif",
    "behavior": null,
    "sync": null,
    "ground_truth": null
  },
  "checksums": {
    "raw_video_sha256": "..."
  },
  "notes": "",
  "created_at": "...",
  "created_by": "...",
  "extras": {}
}
```

### 9.2 PipelineSpec

```json
{
  "schema_version": "1.0",
  "pipeline_spec_id": "gamma_cfar_high_recall_v1",
  "dataset_id": "zfish_001",
  "description": "High-recall Gamma/CFAR candidate proposal",
  "stages": [
    {
      "stage_id": "temporal_highpass_gaussian",
      "params": {"sigma_frames": 15}
    },
    {
      "stage_id": "gamma_cfar",
      "params": {"kernel": "fast_calcium", "target_recall": "high"}
    },
    {
      "stage_id": "component_filter",
      "params": {"min_area_px": 4, "max_area_px": 500}
    }
  ],
  "execution": {
    "device": "auto",
    "num_workers": 4,
    "memory_budget_gb": 16
  },
  "output_root": "outputs/zfish_001/gamma_cfar_high_recall_v1",
  "extras": {}
}
```

### 9.3 PipelineRun

```json
{
  "schema_version": "1.0",
  "run_id": "run_2026_05_13_001",
  "dataset_id": "zfish_001",
  "pipeline_spec_id": "gamma_cfar_high_recall_v1",
  "status": "completed",
  "created_at": "...",
  "completed_at": "...",
  "parameter_hash": "...",
  "code": {
    "git_commit": "...",
    "git_dirty": false,
    "package_version": "..."
  },
  "environment": {
    "python": "...",
    "platform": "...",
    "device": "cpu"
  },
  "artifacts": [],
  "metrics": {},
  "warnings": [],
  "logs": [],
  "extras": {}
}
```

### 9.4 ArtifactRecord

```json
{
  "schema_version": "1.0",
  "artifact_id": "candidate_events.v1",
  "kind": "candidate_events",
  "path": "artifacts/candidates/events.json",
  "schema": "candidate_event.schema.json",
  "producer_stage": "gamma_cfar",
  "created_at": "...",
  "sha256": "...",
  "summary": {},
  "extras": {}
}
```

### 9.5 CandidateEvent

```json
{
  "schema_version": "1.0",
  "event_id": "event_000001",
  "candidate_neuron_id": "roi_000012",
  "run_id": "run_2026_05_13_001",
  "frame_start": 120,
  "frame_peak": 127,
  "frame_end": 145,
  "time_start_s": 12.0,
  "time_peak_s": 12.7,
  "time_end_s": 14.5,
  "scores": {
    "temporal_snr": 5.2,
    "local_z": 4.8,
    "event_quality": 0.87
  },
  "source": {
    "stage_id": "gamma_cfar",
    "algorithm": "gamma_kernel_cfar"
  },
  "status": "unreviewed",
  "extras": {}
}
```

### 9.6 CandidateNeuron

```json
{
  "schema_version": "1.0",
  "candidate_neuron_id": "roi_000012",
  "run_id": "run_2026_05_13_001",
  "centroid_xy": [148.2, 93.7],
  "bbox_xywh": [140, 86, 18, 20],
  "area_px": 87,
  "mask": {
    "encoding": "rle",
    "data": "..."
  },
  "scores": {
    "spatial_compactness": 0.72,
    "background_contrast": 2.1,
    "artifact_score": 0.08,
    "candidate_quality": 0.81
  },
  "events": ["event_000001", "event_000002"],
  "source_runs": ["run_2026_05_13_001"],
  "status": "unreviewed",
  "extras": {}
}
```

### 9.7 Annotation

```json
{
  "schema_version": "1.0",
  "annotation_id": "ann_000001",
  "subject_type": "candidate_neuron",
  "subject_id": "roi_000012",
  "run_id": "run_2026_05_13_001",
  "reviewer": "reviewer_a",
  "created_at": "...",
  "updated_at": "...",
  "label": "accepted",
  "confidence": "high",
  "reason_tags": ["active_neuron", "clear_trace"],
  "notes": "",
  "supersedes": null,
  "extras": {}
}
```

### 9.8 ReviewBatch

```json
{
  "schema_version": "1.0",
  "review_batch_id": "batch_uncertain_001",
  "dataset_id": "zfish_001",
  "run_ids": ["run_2026_05_13_001"],
  "selection_policy": {
    "type": "uncertainty_ranked",
    "params": {"max_items": 100}
  },
  "candidate_ids": ["roi_000012", "roi_000031"],
  "assigned_to": ["reviewer_a"],
  "status": "open",
  "created_at": "...",
  "extras": {}
}
```

### 9.9 MetricsReport

```json
{
  "schema_version": "1.0",
  "metrics_report_id": "metrics_run_001",
  "dataset_id": "zfish_001",
  "run_ids": ["run_2026_05_13_001"],
  "created_at": "...",
  "metrics": {
    "pixel_level": {},
    "object_level": {},
    "event_level": {},
    "annotation": {},
    "runtime": {}
  },
  "figures": [],
  "warnings": [],
  "provenance": {
    "pipeline_run_paths": ["pipeline_run.json"]
  },
  "extras": {}
}
```

### 9.10 ExportBundle

```json
{
  "schema_version": "1.0",
  "export_bundle_id": "export_inverse_dynamics_001",
  "dataset_id": "zfish_001",
  "run_ids": ["run_2026_05_13_001"],
  "created_at": "...",
  "profile": "inverse_dynamics",
  "selection": {
    "annotation_labels": ["accepted"],
    "include_uncertain": false
  },
  "files": {
    "traces": "accepted_traces.tsv",
    "events": "accepted_events.tsv",
    "neurons": "neuron_metadata.tsv",
    "alignment": "alignment_report.json"
  },
  "alignment": {
    "status": "not_provided",
    "frame_rate_hz": 10.0,
    "timebase": "imaging_frames"
  },
  "checksums": {},
  "provenance": {},
  "extras": {}
}
```

---

## 10. Default User Workflow to Build Toward

Codex should make this workflow progressively real. Early phases may use synthetic data only.

### Step 1: Create or validate dataset manifest

```bash
neurobench dataset create \
  --raw-video data/fish001/raw_video.tif \
  --name fish001_session01 \
  --frame-rate-hz 10 \
  --modality calcium \
  --output manifests/fish001_session01.json

neurobench dataset validate manifests/fish001_session01.json
```

### Step 2: Run dataset QC

```bash
neurobench dataset qc manifests/fish001_session01.json \
  --output outputs/fish001_session01/qc
```

Expected QC outputs:

```text
qc_report.json
qc_report.md
summary_stats.json
drift_estimate.tsv
warnings.json
```

### Step 3: Execute high-recall candidate pipeline

```bash
neurobench run execute configs/gamma_cfar_high_recall.yml \
  --dataset manifests/fish001_session01.json \
  --output outputs/fish001_session01/runs/gamma_cfar_high_recall
```

Expected run outputs:

```text
pipeline_run.json
logs/run.log
artifacts/candidates/candidate_neurons.json
artifacts/candidates/candidate_events.json
artifacts/review_data/review_data.json
```

### Step 4: Build and serve workbench

```bash
neurobench workbench build \
  --run outputs/fish001_session01/runs/gamma_cfar_high_recall/pipeline_run.json

neurobench workbench serve \
  outputs/fish001_session01/runs/gamma_cfar_high_recall/workbench
```

Workbench should show:

- raw video;
- enhanced/evidence view;
- ROI overlay;
- trace/event panel;
- candidate score/explanation;
- review queue;
- annotation controls;
- provenance/run metadata;
- warnings and next action.

### Step 5: Review candidates

Inside workbench:

```text
Accept / reject / uncertain / artifact
Set confidence
Add reason tags
Flag needs-second-reviewer
Flag possible split/merge
Flag possible missed neuron
Save annotations
```

### Step 6: Generate reports

```bash
neurobench review report \
  --run outputs/fish001_session01/runs/gamma_cfar_high_recall/pipeline_run.json \
  --annotations outputs/fish001_session01/runs/gamma_cfar_high_recall/workbench/annotations.json

neurobench report generate \
  --run outputs/fish001_session01/runs/gamma_cfar_high_recall/pipeline_run.json \
  --annotations outputs/fish001_session01/runs/gamma_cfar_high_recall/workbench/annotations.json \
  --output outputs/fish001_session01/reports/final_report
```

### Step 7: Compare pipelines if needed

```bash
neurobench run sweep configs/sweeps/gamma_cfar_thresholds.yml \
  --dataset manifests/fish001_session01.json \
  --output outputs/fish001_session01/sweeps/gamma_cfar_thresholds

neurobench run compare \
  outputs/fish001_session01/runs/*/pipeline_run.json \
  --annotations outputs/fish001_session01/annotations/merged_annotations.json \
  --output outputs/fish001_session01/reports/pipeline_comparison
```

### Step 8: Export accepted traces/events

```bash
neurobench export inverse-dynamics \
  --run outputs/fish001_session01/runs/gamma_cfar_high_recall/pipeline_run.json \
  --annotations outputs/fish001_session01/runs/gamma_cfar_high_recall/workbench/annotations.json \
  --output outputs/fish001_session01/exports/inverse_dynamics
```

### Step 9: Archive reproducible bundle

```bash
neurobench export bundle \
  --dataset manifests/fish001_session01.json \
  --runs outputs/fish001_session01/runs/*/pipeline_run.json \
  --reports outputs/fish001_session01/reports \
  --exports outputs/fish001_session01/exports \
  --output outputs/fish001_session01/fish001_session01_reproducible_bundle
```

Final bundle:

```text
fish001_session01_reproducible_bundle/
├── dataset_manifest.json
├── pipeline_runs/
├── annotations/
├── metrics_reports/
├── final_report.md
├── final_report.html
├── exports/
├── checksums.json
└── README.md
```

---

## 11. Processing and Algorithm Development Plan

### 11.1 Default processing pipeline

The target high-recall pipeline should be:

```text
raw video
  → validation and QC
  → optional crop/downsample
  → photobleach/background correction
  → drift/motion estimate
  → optional motion correction
  → denoising
  → local normalization
  → Gamma/CFAR candidate proposal
  → component extraction
  → trace extraction
  → temporal event scoring
  → spatial footprint scoring
  → candidate ranking
  → review_data artifact
```

### 11.2 Denoising stages

Prioritize simple, inspectable denoising first:

| Stage | Purpose | Priority | Acceptance |
|---|---|---:|---|
| Temporal high-pass | Remove slow drift/baseline. | P1 | Synthetic transient preserved within tolerance. |
| Event-preserving suppression | Reduce noise without erasing events. | P1 | Raw and denoised views saved side-by-side. |
| Robust outlier filter | Suppress impulsive artifacts. | P2 | Artifact fixture improves without large event loss. |
| PMD import/use | Use external denoised outputs. | P2 | Imported PMD artifact validates. |
| Learned denoiser plugin | Future research. | P3 | Plugin output registered as artifact. |

Never make denoised video the only workbench view. Always preserve raw context.

### 11.3 Motion and drift handling

Add two separate concepts:

1. **Motion estimation / drift QC**
   - Always safe to run.
   - Produces shift trace, drift score, and warnings.

2. **Motion correction**
   - Optional stage.
   - Starts with rigid XY correction.
   - Later supports nonrigid/local correction.

Acceptance tests:

- synthetic shifted video estimates known shift;
- drift warning triggers over threshold;
- corrected artifact is registered;
- workbench can display drift trace.

### 11.4 Local normalization and CFAR improvements

Enhance CFAR-like detection with:

- robust local median/MAD;
- ring-based local background;
- spatially varying threshold maps;
- exclusion masks;
- anisotropic windows;
- target false-alarm-rate parameter;
- saved maps for background, scale, threshold, evidence.

Acceptance:

- spatial-gradient synthetic fixture produces fewer false positives than global thresholding;
- threshold maps are saved as artifacts;
- parameters appear in run manifest.

### 11.5 Gamma-kernel event detection

Keep Gamma kernels as a central algorithm family. Add:

- frame-rate-aware kernel definitions;
- kernel bank support;
- documented rise/decay/duration semantics;
- event score outputs:
  - peak response;
  - integrated response;
  - onset sharpness;
  - decay fit;
  - local z-score;
- best-kernel metadata per event.

Acceptance:

- synthetic events with known transient shapes are detected;
- kernel parameters appear in run manifest;
- event records include source stage and score fields.

### 11.6 Candidate proposal generation

Candidate proposal should union multiple evidence sources:

- Gamma/CFAR detections;
- robust local-z detections;
- Kalman positive innovation;
- event-triggered footprints;
- temporal correlation;
- Suite2p/PMD/OASIS imports.

Each candidate should record source methods.

Acceptance:

- duplicate proposals merge by spatial overlap and event timing;
- source provenance is preserved;
- review workbench can display source methods.

### 11.7 Temporal and spatial scoring

Temporal scores:

```text
temporal_snr
baseline_noise
local_z
rise_time
decay_time
event_duration
event_isolation
repeated_activation_count
trace_stability
photobleach_sensitivity
```

Spatial scores:

```text
area_px
compactness
eccentricity
edge_proximity
local_background_contrast
footprint_stability
neighbor_overlap
motion_sensitivity
saturation_overlap
artifact_score
```

Scores should be visible in workbench and reports.

---

## 12. Discovery and Human-in-the-Loop Plan

### 12.1 Candidate ranking

Start with transparent heuristic ranking, not black-box ML.

Candidate ranking fields:

```json
{
  "candidate_id": "roi_000012",
  "rank_score": 0.83,
  "rank_reasons": [
    "high temporal_snr",
    "compact footprint",
    "unreviewed",
    "found by multiple sources"
  ],
  "uncertainty_score": 0.42,
  "artifact_suspicion": 0.08
}
```

### 12.2 Review queues

Workbench should expose queues:

```text
All unreviewed
High confidence likely true
Likely artifact
Uncertain
Needs second reviewer
Possible missed neuron
Possible split/merge
Run-unique candidates
Consensus candidates
```

### 12.3 Active learning stages

Stage 1: heuristic active review.

- Pick uncertain candidates.
- Pick high-score rejected-like artifacts.
- Pick low-score accepted-like weak events.
- Pick clusters with conflicting labels.

Stage 2: lightweight supervised ranker after enough annotations.

- Logistic regression or simple tree model only after dependency review.
- Inputs are transparent feature table columns.
- Outputs include probability and explanation.

Stage 3: batch optimization.

- Balance uncertainty, diversity, expected scientific value, and reviewer burden.

Acceptance:

- active batch beats random on uncertainty/high-value enrichment in synthetic or labeled fixture;
- explanations are shown and exported.

### 12.4 False-positive and false-negative analysis

False-positive analysis:

- rejected rate by score bin;
- artifact tags by source stage;
- spatial heatmap of rejected candidates;
- motion/background correlation.

False-negative analysis:

- manually marked missed neurons;
- candidates found only by other runs/importers;
- high local activity regions with no accepted candidate;
- review suggestions from uncovered evidence map.

Acceptance:

- proposal analysis report links every item to reviewable candidate IDs or spatial regions.

---

## 13. Metrics and Reporting Plan

### 13.1 Metric levels

Keep metrics separated:

| Level | Purpose |
|---|---|
| Pixel-level | Legacy algorithm debugging and FROC continuity. |
| Component/neuron-level | Biological discovery and review burden. |
| Event-level | Activity timing/quality. |
| Annotation-level | Review progress and reviewer agreement. |
| Run-level | Runtime, memory, warnings, provenance completeness. |
| Export-level | Downstream readiness and selection policy. |

### 13.2 Object-level metrics

Implement:

```text
spatial_iou
centroid_distance
object_precision
object_recall
duplicate_rate
split_rate
merge_rate
review_burden_per_accepted_neuron
unique_accepted_candidates_by_run
```

### 13.3 Event-level metrics

Implement:

```text
event_precision
event_recall
onset_timing_error_frames
peak_timing_error_frames
duration_error_frames
amplitude_distribution
event_snr
event_isolation
```

### 13.4 Annotation agreement

Support:

```text
Cohen kappa for two reviewers
Fleiss kappa for multiple reviewers
agreement by confidence
agreement by artifact tag
disagreement queue generation
adjudication status
```

### 13.5 Report contents

Every run report should include:

```text
Title
Dataset metadata
Run metadata
Pipeline parameter summary
Input checksums
Stage summaries
Artifact table
QC warnings
Detection metrics
Review summary
Annotation agreement, if available
Candidate examples/crops, if available
Figures
Export summary
Limitations
Reproducibility appendix
```

Reports should initially support Markdown and JSON. HTML can come later.

---

## 14. Reproducibility and Provenance Plan

### 14.1 Required run manifest fields

```json
{
  "run_id": "...",
  "dataset_id": "...",
  "pipeline_spec_id": "...",
  "schema_version": "1.0",
  "created_at": "...",
  "completed_at": "...",
  "status": "completed",
  "input_artifacts": [],
  "output_artifacts": [],
  "parameters": {},
  "parameter_hash": "...",
  "code": {
    "git_commit": "...",
    "git_dirty": true,
    "package_version": "..."
  },
  "environment": {
    "python": "...",
    "platform": "...",
    "packages": {}
  },
  "execution": {
    "device": "cpu",
    "num_workers": 4,
    "memory_budget_gb": 16
  },
  "logs": [],
  "warnings": []
}
```

### 14.2 Structured logs

Use JSONL or structured log records for run/stage logs:

```json
{
  "timestamp": "...",
  "run_id": "...",
  "stage_id": "...",
  "level": "INFO",
  "message": "...",
  "data": {}
}
```

CLI output can remain human-readable.

### 14.3 Error messages

User-facing validation errors should include:

```text
file path
schema or command
field path
expected value/type
observed problem
suggested fix
```

Example:

```text
Dataset manifest validation failed:
  file: manifests/fish001.json
  field: acquisition.frame_rate_hz
  problem: missing required value
  fix: add frame_rate_hz or pass --frame-rate-hz when creating the manifest
```

---

## 15. Testing Matrix

### 15.1 Test categories

| Category | Purpose | Examples |
|---|---|---|
| Unit | Small pure functions | Gamma kernels, CFAR maps, component extraction. |
| Schema | Public JSON contracts | DatasetManifest, PipelineSpec, PipelineRun, annotations. |
| CLI | User-facing commands | `neurobench dataset validate`, `neurobench run validate`. |
| Synthetic E2E | Complete workflow | Synthetic raw video → run → artifacts → report. |
| Workbench smoke | UI build/server contracts | HTML generation, save endpoint, path safety. |
| Integration | External importer/exporter | Suite2p/PMD/OASIS import fixtures. |
| Performance | Runtime/memory bounds | Small benchmark stage. |
| Scientific regression | Expected detection behavior | Synthetic transient recall and FP control. |

### 15.2 Minimum tests before each phase completes

```text
Phase 0: full pytest, no deprecation warnings.
Phase 1: CLI help + schema validation + model roundtrip.
Phase 2: synthetic run writes valid run manifest/artifacts.
Phase 3: workbench build/server smoke + annotation roundtrip.
Phase 4: object/event metrics + metrics report schema.
Phase 5: ranking/clustering/comparison deterministic tests.
Phase 6: chunked equivalence + CPU fallback + online latency fixture.
Phase 7: export bundle schema + downstream loader roundtrip.
```

---

## 16. Documentation and Onboarding Plan

### 16.1 Docs structure

```text
docs/
├── quickstart.md
├── installation.md
├── concepts/
│   ├── datasets.md
│   ├── pipelines.md
│   ├── candidates.md
│   ├── annotations.md
│   ├── metrics.md
│   └── provenance.md
├── workflows/
│   ├── raw_video_to_report.md
│   ├── annotation_campaign.md
│   ├── compare_pipelines.md
│   ├── import_suite2p_pmd_oasis.md
│   └── inverse_dynamics_export.md
├── reference/
│   ├── cli.md
│   ├── schemas.md
│   ├── pipeline_stages.md
│   └── artifact_formats.md
├── developer/
│   ├── architecture.md
│   ├── adding_pipeline_stage.md
│   ├── adding_metric.md
│   ├── adding_export_format.md
│   └── testing.md
├── case_studies/
│   └── resting_video_algorithm_brief.md
└── archive/
    └── old_plan.md
```

### 16.2 Onboarding quickstart target

```bash
conda env create -f environment.cpu.yml
conda activate neurobench
pip install -e ".[dev]"

neurobench --help
neurobench dataset validate examples/manifests/synthetic_dataset.json
neurobench dataset qc examples/manifests/synthetic_dataset.json
neurobench run execute examples/pipelines/gamma_cfar_high_recall.yml
neurobench workbench serve outputs/synthetic/latest/workbench
neurobench report generate outputs/synthetic/latest/pipeline_run.json
```

The quickstart must not require Fiji, GPU, raw lab TIFFs, or absolute local paths.

---

## 17. Risk Register and Tradeoffs

| Risk | Tradeoff | Mitigation |
|---|---|---|
| Fiji/Groovy workflow is useful but not portable. | Preserve current power while building Python-native path. | Treat Fiji as backend under `workflows/fiji/`; CPU synthetic path first. |
| High recall creates review burden. | Over-detection is scientifically useful only with ranking. | Add review burden metrics, ranking, and active batches. |
| Strict schemas can slow experiments. | Public artifacts need stability. | Allow `extras`; isolate experimental code. |
| GPU dependencies complicate setup. | GPU matters for scale but not onboarding. | CPU-first tests; optional GPU tests; fallback behavior. |
| Online inference distracts from offline correctness. | Real-time is long-term ambition. | Keep `realtime/` separate and synthetic-latency focused. |
| Workbench complexity becomes fragile. | Review UX is central. | Validate data model; add smoke tests; avoid unsupported browser execution claims. |
| External importers have incompatible conventions. | Cross-method comparison is valuable. | Preserve source metadata and conversion report. |
| Stale docs mislead Codex. | Existing docs contain valuable context. | Add docs archive and current-state note. |

---

## 18. Non-Goals for Early Phases

Do not spend Phase 0–2 effort on:

- full browser UI redesign;
- full GPU acceleration;
- real-time closed-loop control;
- learned denoising models;
- nonrigid motion correction;
- full plugin packaging;
- PDF report rendering;
- multi-user authentication;
- replacing Fiji/Groovy completely;
- rewriting JavaScript workbench in a framework;
- moving the entire repository to `src/` in one pass.

---

## 19. Acceptance Criteria for the Platform Milestone

The project reaches the next major platform milestone when:

1. CPU-only setup and tests work from a fresh environment.
2. `neurobench --help` and initial CLI commands work.
3. Existing examples validate through schema helpers.
4. Dataset and pipeline models roundtrip examples.
5. A synthetic dataset can be processed into a valid run root.
6. Every new run writes `pipeline_run.json`.
7. Artifacts are registered with checksums.
8. Workbench builds from validated review data.
9. Annotations roundtrip with labels, confidence, tags, and notes.
10. Object/event metrics exist on synthetic fixtures.
11. A Markdown report includes metrics and provenance.
12. Export bundle schema exists and can represent accepted traces/events.
13. Existing legacy workflows still pass their tests.
14. Stale docs are clearly marked or archived.

---

## 20. Open Questions for Human Maintainers

Codex should not block early work on these, but should preserve flexibility until answers are known.

1. Should the canonical frame index be zero-based everywhere, with adapters for external tools?
2. Are videos always 2D time series, or should volumetric data be first-class soon?
3. What are the expected frame rates and indicator kinetics for default Gamma kernels?
4. Should uncertain candidates be exported by default or only accepted candidates?
5. Which artifact categories are most common in current data: motion, vessels, edge effects, saturation, neuropil/background, or fragments?
6. Should reviewer identity be required for all annotations?
7. What downstream inverse-dynamics loader format is most useful: TSV, HDF5/Zarr, NumPy, or JSON bundle plus TSV?
8. How should behavior sync be represented when there is no explicit sync pulse?
9. What latency target matters for future online inference?
10. Should the browser workbench eventually support geometry editing, or should split/merge remain annotation-only?
11. Should Suite2p/PMD/OASIS imports become equal first-class run types or remain external comparison artifacts?
12. What compute environment matters most: laptop CPU, local GPU workstation, acquisition computer, or cluster?

---

## 21. Final Instruction to Codex

Start with Phase 0 and the first 10 work packages. Keep each change small. Preserve old scripts while adding stable interfaces. Every completed task should include tests and a short note explaining:

```text
What changed
Why it changed
Which files were touched
Which public outputs were preserved
Which tests were run
Known limitations / next task
```

Do not implement later phases until earlier interfaces are stable and tested.
