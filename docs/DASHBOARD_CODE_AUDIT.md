# Dashboard Code Audit

Last updated: 2026-07-07.

This note inventories the dashboard surfaces in the repository and proposes a
low-risk cleanup path. It is intentionally scoped to organization, efficiency,
UX utility, and maintainability; it does not change any runtime paths.

## Current Dashboard Surfaces

| Surface | Primary Purpose | Current Entrypoints | Core Package Code |
| --- | --- | --- | --- |
| Neuron workbench | Review ROI candidates, traces, events, annotations, data intermediates, and run comparisons. | `tools/build_neuron_workbench_v2.py`, `tools/serve_neuron_workbench.py`, `tools/build_workbench_index.py` | `neurobench/workbench/`, `neurobench/workbench/assets/` |
| Gamma CFAR sweep review | Prepare TIFF/MP4 datasets, run CFAR grids, attach generated ROI runs to the workbench, and summarize detection burden. | `tools/prepare_gamma_cfar_workbench_run.py`, `tools/summarize_gamma_cfar_sweep.py`, `tools/export_cfar_contrast_maps.py`, `tools/build_review_roi_sidecars.py` | `neurobench/reports/gamma_cfar_sweep.py`, `neurobench/workbench/cfar_contrast_maps.py`, `neurobench/workbench/roi_payloads.py` |
| Grid dynamics comparison | Compare model predictions, reconstruction examples, sweep summaries, and visual examples for dynamics experiments. | `tools/build_grid64_dashboard_artifacts.py`, `tools/build_grid_dynamics_dashboard_videos.py`, `tools/build_crop512_grid32_dashboard_videos.py`, `tools/build_dynamics_comparison_dashboard.py` | `neurobench/dynamics/comparison.py`, `neurobench/dynamics/report.py` |
| Reports and evidence pages | Generate markdown/JSON scientific summaries and sweep evidence reports. | `tools/build_review_report.py`, `tools/build_sweep_evidence_report.py`, report CLI commands | `neurobench/review_reports.py`, `neurobench/reports/`, `neurobench/cli/report.py` |

## Findings

1. `tools/prepare_gamma_cfar_workbench_run.py` is too large.
   It is over 3,500 lines and mixes dataset conversion, spec writing, CPU/GPU
   sweep execution, ROI payload attachment, brief generation, and run manifest
   metadata. This is the main maintainability risk for Gamma CFAR dashboards.

2. Dashboard CLI wrappers and implementation modules are inconsistently split.
   The neuron workbench has a clean package boundary under `neurobench/workbench`,
   but several grid-dynamics and Gamma CFAR dashboard builders still keep
   substantial logic in `tools/`.

3. The workbench JavaScript is split into ordered source files, but the modules
   are still large.
   The served bundle is generated from `neurobench/workbench/assets/src/*.js`,
   which is good. The largest source files are review core, architecture lab,
   and state persistence; these are candidates for further split by page/store
   responsibility.

4. There are several dashboard artifact schemas but no single dashboard
   registry.
   Workbench apps, Gamma CFAR sweep directories, comparison dashboards, and
   markdown reports each have their own expected files. A small registry module
   would make it easier to discover what dashboard can inspect a given output
   directory.

5. UX utility depends on the local server, but the command handoff is still
   manual.
   Static `index.html` works for simple viewing, but autosave, generated run
   payloads, trace shards, and relative JSON/frame loading are more reliable
   through `tools/serve_neuron_workbench.py`.

## Recommended Organization

Move implementation code toward this package layout while keeping `tools/`
as thin compatibility wrappers:

```text
neurobench/
  dashboards/
    registry.py              # discover dashboards and required artifacts
    manifest.py              # common dashboard/app artifact records
  workbench/
    build.py                 # current builder/server remain here or alias here
    server.py
    assets/
  gamma_cfar/
    dataset.py               # TIFF/MP4 conversion and starter review data
    specs.py                 # sweep spec builders
    runners.py               # CPU/GPU fast-grid execution
    attach.py                # workbench attachment and sidecars
    reports.py               # wrappers around gamma_cfar_sweep reporting
  dynamics/
    dashboards.py            # grid dashboard copy/update/video selector helpers
```

`tools/prepare_gamma_cfar_workbench_run.py` should become a CLI shim that imports
from `neurobench.gamma_cfar.*`. Existing commands should remain valid.

## UX Improvements

1. Add a dataset-level dashboard launcher command:

```bash
.venv-neurobench/bin/python -m neurobench.cli.main dashboard open \
  Outputs/GammaCFAR/spon_ca_burst_3_hindbrain_to_tail_488_20ms
```

It should find `app/index.html`, start the workbench server on an available
port, and print the exact URL.

2. Add a generated `dashboard_manifest.json` to each processed dataset root.
   It should list the app, source data, self-template, sweep brief, attached
   architecture runs, and recommended inspection run.

3. Surface CFAR caveats in the workbench home/report panel.
   For example, this run found 80-82 compact candidates with median equivalent
   diameter about 1.6 um at 0.5 um/px, so the UI should make clear that these
   are candidates requiring visual confirmation, not ground-truth neurons.

4. Add a first-run review route.
   A URL hash such as `#review?run=gamma_cfar...__sweep_001` would reduce clicks
   when opening a new sweep result.

## Efficiency Improvements

1. Reuse shared intermediate frame exports across sweep runs.
   The current Gamma CFAR attach path already writes shared intermediate
   references. Keep this pattern explicit in the refactor so large frame
   sequences are not duplicated.

2. Prefer compact ROI summaries plus per-ROI trace shards for large sweeps.
   Full `review_rois.json` files should remain for compatibility but should not
   be the primary browser payload.

3. Add dashboard smoke checks for generated CFAR apps.
   Existing workbench browser tests cover a small fixture. Add a cheaper
   JSON-level test that verifies attached sweep runs, sidecar references, and
   self-template metadata without launching a browser.

4. Keep large `.npy` arrays out of browser paths.
   Dashboards should read JSON summaries and PNG/JPEG frame previews; source
   arrays should remain provenance artifacts for CLI and reproducible analysis.

## Suggested Next Steps

1. Add `dashboard_manifest.json` generation for the new Gamma CFAR dataset root.
2. Extract the dataset conversion and sweep spec builders from
   `tools/prepare_gamma_cfar_workbench_run.py` into package modules.
3. Add a `dashboard status/open` CLI that prints or serves the right dashboard
   for a dataset root.
4. Add regression tests around CFAR workbench attachment and self-template
   metadata.
5. Then split the largest workbench JavaScript source files by page-level
   responsibilities.
