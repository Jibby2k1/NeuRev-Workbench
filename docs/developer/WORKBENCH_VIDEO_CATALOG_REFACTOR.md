# Workbench Layout, Video Catalog, and Bloat Refactor

Last updated: 2026-07-22.

## Outcome

Review now has one annotated video canvas by default and a clearly labeled
`Compare raw + overlay` presentation mode. ROI creation/editing and CFAR
foreground/background labeling live in an always-visible annotation dock; they
are no longer hidden by Guided or Standard mode.

Dataset, labeled-video, App, index, server, and LLM lookup now share
`neurobench.data.catalog`. The catalog is a compatibility layer over existing
artifacts, not a second source of scientific truth.

## Spon Ca Burst Diagnosis

The current Spon review payload describes one source stack:

- source: `Inputs/Spon Ca Burst/3 hindbrain to tail 488 20ms.tif`
- frame shape: `573 x 340`
- frame count: `2359`
- frame rate: `50 Hz`

The former two-pane Review display did not represent two input recordings. The
raw pane and annotated pane used the same frame URL. Saved
`reviewSideBySide: true` made that comparison layout look like two videos, while
saved `uiMode: guided` hid both annotation panels.

Do not crop or split this TIFF based on the former UI. If an acquisition is
confirmed to contain physical subviews, declare optional `video.views[]` bounds
in native full-frame coordinates. Existing ROI and CFAR coordinates remain in
the native frame.

## Canonical Query Surface

Find review apps and labeled video collections:

```bash
.venv-neurobench/bin/python -m neurobench.cli.main dataset catalog --root . --query spon --json
.venv-neurobench/bin/python -m neurobench.cli.main dataset catalog --root . --query left --llm
```

Build an LLM handoff by dataset identity rather than hand-copying paths:

```bash
.venv-neurobench/bin/python -m neurobench.cli.main llm context \
  --dataset-id spon_ca_burst_3_hindbrain_to_tail_488_20ms \
  --catalog-root . \
  --json
```

The LLM command uses a compact stage catalog by default. Pass
`--catalog-detail full` only when parameter documentation and all stage metadata
are actually needed.

Use the same identity for App operations:

```bash
.venv-neurobench/bin/python -m neurobench.cli.main workbench status \
  --dataset-id spon_ca_burst_3_hindbrain_to_tail_488_20ms --catalog-root .

.venv-neurobench/bin/python -m neurobench.cli.main workbench build \
  --dataset-manifest Outputs/GammaCFAR/spon_ca_burst_3_hindbrain_to_tail_488_20ms/dataset_manifest.json

.venv-neurobench/bin/python -m neurobench.cli.main workbench serve \
  --dataset-id spon_ca_burst_3_hindbrain_to_tail_488_20ms --catalog-root . --port 8765
```

Server-backed apps expose:

- `GET /api/dataset`: the canonical record for the served app;
- `GET /api/datasets`: the bounded workspace catalog;
- existing job and environment endpoints remain unchanged.

If a source is explicitly declared as a physical composite, optional
`video.views[]` records expose native-full-frame bounds. The Review toolbar
offers a view-focus selector, outlines those bounds without rewriting
coordinates, and constrains CFAR flood operations to the active view. The
current Spon record has no declared logical views because its former two-pane
appearance was the same source frame duplicated as raw and overlay.

## Durable Contracts

| Contract | Purpose | Catalog treatment |
| --- | --- | --- |
| `dataset_manifest.json` | One dataset/source plus App and experiment paths | Dataset identity and primary source |
| `video_manifest.json` | Labeled video collection, including left/right/neutral fish videos | `videos[]`, labels, dimensions, and split policy |
| `review_data.json` | Browser-ready frame sequence and scientific review payload | Dimensions, frame pattern, counts, and optional logical views |
| `dashboard_manifest.json` | Human entrypoint and serving instructions | Dashboard and App link |
| `architecture_runs.json` | Attached detector/model runs | Preserved independently from UI assets |
| `annotations.json` | Human labels and UI state | Migrated without changing scientific labels |

Supported frame patterns now agree on general printf widths such as `%04d`,
brace widths such as `{frame:05d}` and `{frame05}`, and unpadded `{frame}`.
UI frames remain one-based; array indexing remains zero-based.

## Data-Preservation Guardrail

The workbench builder previously synthesized and overwrote
`architecture_runs.json` whenever no explicit run path was passed. An asset-only
rebuild could therefore erase attached run metadata. The builder now:

1. loads and preserves the app's existing run catalog by default;
2. replaces it only when an explicit different catalog is supplied;
3. synthesizes a baseline only for a new app with no catalog;
4. preserves and migrates `annotations.json` independently.

The Spon catalog was reconstructed from retained sweep summaries, 60 pipeline
manifests, and 61 intact review-artifact directories. The restored manifest has
62 unique runs: baseline, 36 Gamma-CFAR runs, 12 grayscale-projection runs, 12
Kalman-residual runs, and the 300-ROI soma-first review run. No experiment was
rerun.

One historical soma run predates current strict pipeline metadata rules. LLM
context generation retains its scientific summary with an explicit
`legacy_pipeline_metadata` warning; proposal import and new/modified run
pipelines remain strictly validated.

CFAR undo snapshots are session-only compact bitsets, not persisted copies of
full masks. Server autosave uses compact JSON and local recovery history is
byte-bounded so repeated flood edits do not multiply annotation payload size.

## Bloat and Redundancy Audit

High-impact findings, in priority order:

1. A global JavaScript `pointInPolygon` collision broke free-form lasso
   selection; incompatible Review and QC helpers silently replaced each other.
   They are now namespaced and regression-tested. A similar `cleanTsv` collision
   was removed.
2. Concrete CLI commands formerly imported every command registrar. Even
   `workbench --help` loaded roughly 610 MiB because dynamics/Torch modules were
   imported. All concrete commands now lazy-load only their registrar.
3. Eight generated workbench bundles under relevant Outputs contained six
   historical asset hashes. Historical apps must remain pinned unless a user
   explicitly runs `workbench build`; never bulk-overwrite Outputs.
4. `20_review_core.js` is about 3,500 lines, the generated bundle about 13,000
   lines, the CSS about 3,300 lines, and the Gamma-CFAR preparation tool about
   3,550 lines. These are the main structural split targets.
5. Three experiment scripts duplicate the same FFmpeg PNG-to-MP4 encoder, and
   several pipelines independently implement frame rendering and normalization.
6. `generate_neuron_review_app` is described as a full local pipeline stage,
   while its generic executor currently writes only a `manifest_only` record.
7. An installed package module still imports reusable dynamics code from a
   research script, which is not included by package discovery.

## Safe Next Refactors

These should be separate, behavior-preserving changes with focused tests:

1. Split Review JavaScript into canvas geometry, ROI editing, trace/event, and
   export modules; split CSS into ordered generated sources.
2. Extract shared stable-window frame rendering and PNG-sequence MP4 encoding
   under `neurobench.media`; preserve FPS, codec, CRF, frame order, and paths.
3. Split `prepare_gamma_cfar_workbench_run.py` into dataset, specs, runners,
   summaries, and attachment modules while keeping its CLI compatible.
4. Make the Python builder the only UI emitter; limit the Groovy workflow to
   scientific review-data generation.
5. Align the pipeline catalog's review-app stage with executor behavior, or
   rename it to an explicit planning stage.
6. Move script-owned directional-model classes into `neurobench.dynamics` and
   keep the script as a wrapper.
7. Add compact builder-emitted App manifests so catalog discovery need not read
   trace-heavy `review_data.json` files.

Do not combine those extractions with the stopped Grid128 sweep, alter archived
progress logs, delete historical Outputs, or silently change display
normalization.

## Verification

Focused checks for this refactor:

```bash
.venv-neurobench/bin/python tools/build_workbench_assets.py --check
node --check neurobench/workbench/assets/workbench.js
.venv-neurobench/bin/python -m pytest \
  tests/test_cfar_roi_annotation.py \
  tests/test_workbench_builder.py \
  tests/test_workbench_server.py \
  tests/test_dataset_catalog.py \
  tests/test_pipeline_runner_index.py \
  tests/test_llm_architecture_planning.py
```

Observed final validation on 2026-07-22:

- complete repository suite: `590 passed, 2 skipped`;
- packaged and installed Spon asset version: `b4449740fc04`;
- Spon architecture catalog: `62` unique runs, preserved byte-for-byte;
- Spon annotation state: preserved byte-for-byte and valid against the current schema;
- live App/API: correct `573 x 340 x 2359` source at `50 Hz`, annotation dock present,
  and `11` bounded catalog records;
- duplicated Grid128 video manifests: record, per-video, and LLM views all retain
  the declared `50 Hz` rate.
