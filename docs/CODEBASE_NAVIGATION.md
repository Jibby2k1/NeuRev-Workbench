# Codebase Navigation

Last updated: 2026-07-13.

This guide is a fast map for humans and coding agents. It names the stable
entry points first, then points to the implementation modules behind common
tasks. Generated data under `Inputs/` and `Outputs/` is intentionally excluded
from this map.

## Primary Entry Points

| Goal | Start Here | Then Read |
| --- | --- | --- |
| Run CLI workflows | `neurobench/cli/main.py` | The matching file under `neurobench/cli/` |
| Build or serve the neuron dashboard | `tools/build_neuron_workbench_v2.py`, `tools/serve_neuron_workbench.py` | `neurobench/workbench/builder.py`, `neurobench/workbench/server.py`, `docs/NEURON_WORKBENCH.md` |
| Run Gamma CFAR candidate workflows | `tools/prepare_gamma_cfar_workbench_run.py` | `neurobench/algorithms/cfar.py`, `neurobench/reports/gamma_cfar_sweep.py`, `docs/workflows/raw_video_to_report.md` |
| Run the Spon dark-soma/excitation case study | `examples/spon_ca_burst_soma_excitation.example.json` | `neurobench/experiments/soma_excitation/`, `docs/workflows/spon_ca_burst_soma_excitation.md` |
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
| `neurobench/data/` | Dataset manifests, video loading/cropping, QC, checksums, preflight estimates, and synthetic fixtures. |
| `neurobench/discovery/` | Candidate clustering, ranking, and active-learning helpers. |
| `neurobench/dynamics/` | Grid/latent dynamics datasets, models, training, sweeps, reports, comparisons, and supervisors. |
| `neurobench/experiments/` | Focused, manifest-driven case studies with explicit resource and artifact contracts. |
| `neurobench/exports/` | Annotation, behavior-alignment, and inverse-dynamics export contracts. |
| `neurobench/integrations/` | Import adapters for external tools such as Suite2p, PMD, and OASIS. |
| `neurobench/metrics/` | Detection, event-quality, run-comparison, and summary metrics. |
| `neurobench/models/` | Dataclass-like model validation and serialization for public JSON artifacts. |
| `neurobench/pipelines/` | Local pipeline execution, artifacts, devices, specs, stages, and sweeps. |
| `neurobench/realtime/` | Streaming and latency helpers. |
| `neurobench/reports/` | Markdown/JSON report builders and renderers. |
| `neurobench/review/` | Reviewer agreement and provenance utilities. |
| `neurobench/validation/` | JSON schema loading and validation helpers. |
| `neurobench/workbench/` | Browser workbench builder, server, assets, intermediate exports, ROI sidecars, and materialization. |

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
2. Look for `dashboard_manifest.json`; if present, use its `serve_command`.
3. Otherwise serve the app directly:

```bash
.venv-neurobench/bin/python tools/serve_neuron_workbench.py \
  --app-dir path/to/app \
  --host 127.0.0.1 \
  --port 8765
```

### Build A New Raw-Video Gamma CFAR Result

1. Convert/prepare data with `tools/prepare_gamma_cfar_workbench_run.py`.
2. Write the sweep spec with the same script.
3. Run the fast grid.
4. Attach the sweep to the workbench app.
5. Build the static workbench with `tools/build_neuron_workbench_v2.py`.

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

1. Edit source files under `neurobench/workbench/assets/src/`.
2. Rebuild the generated bundle:

```bash
.venv-neurobench/bin/python tools/build_workbench_assets.py
.venv-neurobench/bin/python tools/build_workbench_assets.py --check
```

3. Rebuild a fixture app with `tools/build_neuron_workbench_v2.py`.
4. Run focused tests such as `tests/test_workbench_assets.py` and
   `tests/test_workbench_builder.py`.

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
