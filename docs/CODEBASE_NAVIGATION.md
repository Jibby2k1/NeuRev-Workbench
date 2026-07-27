# Codebase Navigation

Last updated: 2026-07-27.

This guide is a fast map for humans and coding agents. It names the stable
entry points first, then points to the implementation modules behind common
tasks. Generated data under `Inputs/` and `Outputs/` is intentionally excluded
from this map.

## Primary Entry Points

| Goal | Start Here | Then Read |
| --- | --- | --- |
| Run CLI workflows | `neurobench/cli/main.py` | The matching file under `neurobench/cli/` |
| Use the normal NeuRev workflow | `neurobench workbench baseline`, then `neurobench workbench serve --asset-mode current` | `docs/developer/NEUREV_FIRST_RELEASE_HANDOFF.md`, `neurobench/workbench/baseline.py`, `neurobench/workbench/server.py` |
| Build or serve the neuron dashboard | `neurobench workbench build/status/serve` | `neurobench/workbench/builder.py`, `neurobench/workbench/server.py`, `docs/developer/WORKBENCH_VIDEO_CATALOG_REFACTOR.md`; installed annotation migration requires `--migrate-annotations` |
| Query datasets and videos for App/LLM use | `neurobench dataset catalog` | `neurobench/data/catalog.py`, `neurobench llm context --dataset-id ...` |
| Run Gamma CFAR candidate workflows | `tools/prepare_gamma_cfar_workbench_run.py` | `neurobench/algorithms/cfar.py`, `neurobench/reports/gamma_cfar_sweep.py`, `docs/workflows/raw_video_to_report.md` |
| Run the Spon dark-soma/excitation case study | `examples/spon_ca_burst_soma_excitation.example.json` | `neurobench/experiments/soma_excitation/`, `docs/workflows/spon_ca_burst_soma_excitation.md` |
| Run or inspect learnable contrast experiments | `docs/workflows/spon_ca_burst_learnable_contrast.md` | `neurobench/experiments/learnable_contrast/`, `neurobench/cli/experiment.py` |
| Develop pairwise temporal source separation | `docs/workflows/spon_ca_burst_pairwise_separation.md` | `neurobench/algorithms/pairwise_separation.py`, `neurobench/experiments/pairwise_separation/` |
| Fuse pairwise/derivative evidence with Raw Direct | `docs/workflows/spon_ca_burst_pairwise_feature_fusion.md` | `neurobench/experiments/pairwise_separation/fusion.py`, `docs/research/PAIRWISE_ICA_AS_TEMPORAL_DERIVATIVE.md` |
| Plan stable latent-dynamics denoising and post-denoising features | `docs/research/DENOISE_THEN_DIFFERENCE.md` | `docs/developer/LATENT_DYNAMICS_DENOISING_IMPLEMENTATION_BRIEF.md`, `docs/research/overleaf/neurev_denoise_then_difference.tex`; implementation is not yet present |
| Audit the fish intent/inverse-control program | `docs/programs/fish_inverse_control/README.md` | `examples/fish_control_program.example.json`, `neurobench/programs/fish_control.py` |
| Add a pipeline stage | `docs/developer/adding_pipeline_stage.md` | `neurobench/pipeline_catalog.py`, `neurobench/pipelines/executor.py`, `tests/test_pipeline_executor.py` |
| Work on template/grid preprocessing | `docs/TEMPLATE_GRID_WORKFLOW.md` | `neurobench/algorithms/template_matching.py`, `neurobench/algorithms/grid_regions.py`, `neurobench/cli/template.py`, `neurobench/cli/grid.py` |
| Work on grid dynamics experiments | `docs/GRID_LATENT_DYNAMICS.md` | `neurobench/dynamics/`, `neurobench/cli/dynamics.py` |
| Add or inspect reports | `neurobench/cli/report.py` | `neurobench/reports/`, `neurobench/review_reports.py`, `docs/TEST_AND_EXPERIMENT_REPORT.md` |
| Understand schemas and artifacts | `schemas/`, `neurobench/models/` | `neurobench/validation/schemas.py`, `examples/` |

## Package Map

| Path | Role |
| --- | --- |
| `neurobench/algorithms/` | Core image/video algorithms such as CFAR, motion, template matching, and grid extraction. |
| `neurobench/cli/` | Argparse command groups. These should stay thin and delegate to package modules. |
| `neurobench/data/` | Dataset manifests and catalogs, bounded import inspection, video loading/cropping, QC, checksums, preflight estimates, and synthetic fixtures. |
| `neurobench/dashboards/` | Dashboard manifest and presentation contracts. |
| `neurobench/discovery/` | Candidate clustering, ranking, and active-learning helpers. |
| `neurobench/dynamics/` | Grid/latent dynamics datasets, models, training, sweeps, reports, comparisons, and supervisors. |
| `neurobench/experiments/` | Focused, manifest-driven case studies with explicit resource and artifact contracts. |
| `neurobench/exports/` | Annotation, behavior-alignment, and inverse-dynamics export contracts. |
| `neurobench/integrations/` | Import adapters for external tools such as Suite2p, PMD, and OASIS. |
| `neurobench/logging/` | Atomic run-state and resource logging helpers. |
| `neurobench/metrics/` | Detection, event-quality, run-comparison, and summary metrics. |
| `neurobench/models/` | Dataclass-like model validation and serialization for public JSON artifacts. |
| `neurobench/pipelines/` | Local pipeline execution, artifacts, devices, specs, stages, and sweeps. |
| `neurobench/programs/` | Stage-gated research-program manifests and read-only readiness audits. |
| `neurobench/realtime/` | Streaming and latency helpers. |
| `neurobench/reports/` | Markdown/JSON report builders and renderers. |
| `neurobench/review/` | Reviewer agreement and provenance utilities. |
| `neurobench/validation/` | JSON schema loading and validation helpers. |
| `neurobench/workbench/` | Browser workbench builder, no-write/current and installed serving, preservation baselines, durable local jobs, assets, intermediate exports, ROI sidecars, and materialization. |

## Non-Package Areas

| Path | Role | Guidance |
| --- | --- | --- |
| `tools/` | User-facing scripts and compatibility wrappers. | New implementation logic should usually move into `neurobench/`; keep scripts as thin CLIs. |
| `scripts/` | Experiment-specific research scripts. | Treat as less stable than package modules. Promote reusable logic into `neurobench/` before relying on it elsewhere. |
| `docs/` | Human and agent documentation. | Prefer adding task-specific guides here instead of long comments in code. |
| `examples/` | Small JSON examples for schemas and workflows. | Keep examples small and runnable in tests. |
| `schemas/` | Public JSON schemas. | Schema changes should include model and validation tests. |
| `tests/` | Regression tests. | Name tests after behavior or module ownership, not just bug IDs. |
| `core/`, `evaluation/`, `reporting/` | Legacy/older pipeline modules. | Check whether a newer `neurobench/` implementation exists before extending these. |

## Common Task Routes

### Open A Generated Dashboard

1. Find the dataset root, often under `Outputs/GammaCFAR/...` or
   `Outputs/NeuronReview/...`.
2. Query the canonical catalog with `neurobench dataset catalog --root .`.
3. Serve the selected app through the canonical command:

```bash
.venv-neurobench/bin/python -m neurobench.cli.main workbench serve \
  --app-dir path/to/app \
  --asset-mode current \
  --host 127.0.0.1 \
  --port 8765
```

`current` is the default and renders packaged assets in memory without
rewriting the selected app. Use `workbench status` to compare installed bytes
with the package, and use `--asset-mode installed` only when the archived UI
itself is the intended comparison target. Capture/verify a Wave 0 baseline
before any explicit historical `workbench build`; annotation rewriting also
requires the separate `--migrate-annotations` flag. See
`docs/developer/NEUREV_FIRST_RELEASE_HANDOFF.md` for the recovery commands and
first-release limitations.

`tools/serve_neuron_workbench.py` remains a compatibility and multi-app-index
wrapper; do not use it as the canonical single-app route.

### Build A New Raw-Video Gamma CFAR Result

1. Convert/prepare data with `tools/prepare_gamma_cfar_workbench_run.py`.
2. Write the sweep spec with the same script.
3. Run the fast grid.
4. Attach the sweep to the workbench app.
5. Build the static workbench with `neurobench workbench build`.

Current refactor target: split that script into `neurobench.gamma_cfar.*`
modules while preserving the existing CLI.

### Run The Spon Soma-Excitation Case Study

1. Read `docs/workflows/spon_ca_burst_soma_excitation.md`.
2. Start from `examples/spon_ca_burst_soma_excitation.example.json`.
3. Run the experiment preflight and inspect frame/resource/checkpoint contracts.
4. Run only the CPU-bounded command; outputs refuse collisions.
5. Treat model results as frozen out-of-domain transfer until manual zones and
   events are reviewed.

Exact implementation routes:

| Concern | File |
| --- | --- |
| CLI and pre-import resource environment | `neurobench/cli/experiment.py` |
| Manifest validation and frame contract | `neurobench/experiments/soma_excitation/config.py` |
| RAM/disk/checkpoint planning | `neurobench/experiments/soma_excitation/preflight.py` |
| Quiet-baseline dark-core anatomy | `neurobench/experiments/soma_excitation/zones.py` |
| Direct positive-residual and local-CFAR lanes | `neurobench/experiments/soma_excitation/detector.py` |
| Frozen batch-1 model evaluation | `neurobench/experiments/soma_excitation/transfer.py` |
| Collision-safe orchestration and reports | `neurobench/experiments/soma_excitation/runner.py` |

The corrective v2 example fixes CPU threads at 2, workers at 1, detector chunks
at 8, transfer batches at 1, and the RAM cap at 1,024 MiB. The CLI lazy-loads
the experiment group and sets OpenMP/BLAS limits before scientific imports;
preflight and the runner provide estimated and live `VmRSS`/`VmHWM` checks.

### Add A Pipeline Stage

1. Add stage metadata and parameters in `neurobench/pipeline_catalog.py`.
2. Add execution logic in `neurobench/pipelines/executor.py` if it is locally
   runnable.
3. Add artifact registration and schema coverage.
4. Add tests under `tests/test_pipeline_executor.py` or a focused test file.
5. Update `docs/developer/adding_pipeline_stage.md` if the stage changes the
   workflow pattern.

### Work On The Workbench UI

1. Edit source files under `neurobench/workbench/assets/src/`. Production
   source order is explicit in `assets/src/bundle_sources.txt`; adding or
   removing a module requires updating that manifest.
2. Rebuild the generated bundle:

```bash
.venv-neurobench/bin/python tools/build_workbench_assets.py
.venv-neurobench/bin/python tools/build_workbench_assets.py --check
```

3. Rebuild only a disposable fixture with
   `neurobench workbench build --review-data ... --app-dir ...`. For a
   historical app, prefer `serve --asset-mode current`; capture and verify a
   baseline before an intentional installed rebuild.
4. Run focused tests such as `tests/test_workbench_assets.py`,
   `tests/test_workbench_builder.py`, `tests/test_workbench_cli.py`, and
   `tests/test_workbench_product_shell_runtime.py`.

Do not edit `neurobench/workbench/assets/workbench.js` directly except for
recovery; it is generated.

## Naming And Ownership Conventions

- Public CLI commands live under `neurobench/cli/` or thin `tools/` wrappers.
- Long-running experiments should write under `Outputs/` and record a manifest
  or summary JSON that points to the generated dashboard/report artifacts.
- Reusable algorithms belong in `neurobench/algorithms/`, not in `tools/`.
- Reusable report/dashboard generation belongs in `neurobench/reports/`,
  `neurobench/workbench/`, or a future `neurobench/dashboards/` package.
- Tests should be close in name to the module or workflow they protect.

## Current Hotspots

| Hotspot | Why It Matters | Preferred Direction |
| --- | --- | --- |
| `tools/prepare_gamma_cfar_workbench_run.py` | Single large script mixing conversion, spec writing, sweep execution, attachment, and reporting. | Extract to package modules, keep CLI compatibility. |
| `neurobench/workbench/assets/src/20_review_core.js` | Large browser module with review rendering and interactions. | Split by review state, drawing, ROI list, trace panel, and event controls. |
| `neurobench/workbench/assets/workbench.css` | One served stylesheet still couples the dataset, annotation, results, and research surfaces. | Introduce ordered CSS source modules while preserving generated-bundle compatibility. |
| `neurobench/workbench/server.py` | Large HTTP module still spans dataset routing, owner authorization, import/job orchestration, processing, and legacy research endpoints. Pure label reconciliation now lives in `label_reconciliation.py`, and sidecar reads use the shared bounded identity validator. | Continue extracting route and job-runner modules behind the tested dataset-qualified API; preserve current-mode no-write startup and compatibility hooks. |
| `neurobench/data/imports.py` | Central safety and lifecycle contract for first-release TIFF/NPY, label-table, and four recognized native NeuRev JSON formats. Generic JSON and future scientific-container adapters remain deliberately absent. | Add future bounded adapters behind the existing import record and transition contract; never infer scientific metadata or accept arbitrary JSON shapes. |
| `neurobench/dynamics/overnight_sweep.py` | Large orchestration module for expensive GPU experiments. | Keep runner behavior stable; extract manifest/progress helpers only with tests. |
| `neurobench/pipeline_catalog.py` and `neurobench/pipelines/executor.py` | Central stage definitions and execution; easy to create hidden coupling. | Keep docs/tests synchronized for every stage. |

## Agent Notes

- Read `AGENTS.md` before touching sweep code or interpreting active experiment
  results.
- Prefer `.venv-neurobench/bin/python` for repository Python commands.
- The workspace can require escalated shell reads/writes because sandboxed
  commands may fail before the shell starts.
- Do not delete archived progress logs or generated experiment evidence unless
  explicitly asked.
