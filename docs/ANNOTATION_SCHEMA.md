# Annotation Schema

`annotations.json` now migrates to schema version 3. The migration preserves
existing review labels while adding fields needed for inverse-dynamics and
future model training.

## ROI Fields

Each ROI annotation may include:

- `state`: legacy UI label, one of `accept`, `reject`, `unsure`, or empty
- `cell_state`: canonical state, one of `accepted`, `rejected`, `unsure`, or
  empty
- `trace_quality`: `good`, `weak`, `noisy`, `unusable`, or empty
- `control_ready`: `yes`, `maybe`, `no`, or empty
- `artifact_class`: free artifact/category label
- `identity_group`: optional grouping for duplicate/split/merged identities
- `needs_action`: review/edit flag such as `split_needed` or `redraw_needed`
- `confidence`: `high`, `medium`, `low`, or empty
- `reason_tags`: short reviewer rationale tags such as `compact`, `manual`,
  `artifact_risk`, or `needs_second_review`
- `reviewer_id`: reviewer initials/name copied from the dashboard reviewer
  field when the label was last edited
- `updatedAt`: timestamp for the most recent dashboard edit to that annotation
- `notes`
- `deleted`

## Virtual ROI Fields

Intent-first edits and manual ROI footprints are stored in `virtualRois`. A
virtual ROI may include:

- `id`: virtual ROI ID such as `VM_4_7`
- `roi_kind`: `virtual_merge`, `manual_center`, `manual_circle`,
  `manual_lasso`, `manual_edit`, or another annotation-layer ROI kind
- `source_roi_ids`: source ROI IDs included in the merge
- `provenance`: origin of the virtual footprint, such as `manual_overlay` or
  `roi_brush_edit`
- `points`, `bbox`, `area`, `centroidX`, and `centroidY` for manual or generated
  virtual footprints
- optional materialized trace fields: `rawTrace`, `backgroundTrace`,
  `dffTrace`, `baselineTrace`, `eventTrace`, `zTrace`, `events`, `noiseSigma`,
  `traceSnr`, `backgroundCorrelation`, and `trace_materialization`
- `edit_history`: optional bounded list of previous geometry snapshots used by
  browser-side mask undo/revert controls
- `identity_group`: shared neuron identity/group label
- label fields such as `cell_state`, `trace_quality`, `control_ready`,
  `artifact_class`, `confidence`, `reason_tags`, `reviewer_id`, `updatedAt`,
  and `notes`

Virtual ROIs do not rewrite source footprints in `review_data.json`. They are
annotation-layer instructions for downstream export and future rebuilds.

## CFAR Foreground And Background Regions

Any source or virtual ROI annotation may include a separate `cfar_regions`
object. These masks describe which pixels should behave as foreground and
background evidence for CFAR-style detector development; they do not replace
or mutate the ROI footprint.

- `schema_version`: currently `1`
- `foreground_points`: unique `[x, y]` image-pixel coordinates
- `background_points`: unique `[x, y]` image-pixel coordinates
- `reference_frames`: sorted frame indices on which the spatial masks were
  reviewed
- `provenance`: `manual_cfar_feature_annotation` for workbench edits
- `updatedAt`: time of the most recent mask edit
- `last_edit`: frame, target class, tool, reviewer, and any flood-fill settings
  associated with the latest operation

CFAR undo history is session-only and held in compact in-memory bitsets. It is
not a durable scientific field and is stripped during annotation migration;
only the current masks and latest-edit provenance are persisted.

Foreground and background are mutually exclusive: adding a pixel to one class
removes it from the other. The masks are spatial labels, while
`reference_frames` records the temporal evidence used to make them. The full
point sets remain in `annotations.json` and `annotations_v3.json`; ROI TSV
exports summarize them as `cfar_foreground_px`, `cfar_background_px`, and
`cfar_reference_frames` so tabular exports do not expand into one column per
pixel.

## Event Fields

Each event annotation may include:

- `state`: legacy UI label
- `event_state`: canonical state, one of `accepted`, `rejected`, `unsure`, or
  empty
- `event_type`: clear, weak, slow, artifact, or future labels
- `timing_quality`: `clear_frame`, `ambiguous`, `slow_transient`, or empty
- `confidence`: `high`, `medium`, `low`, or empty
- `reason_tags`: short reviewer rationale tags
- `reviewer_id`
- `updatedAt`
- `notes`

## Suggestion Fields

Discovery suggestion annotations may include:

- `state`: `promoted`, `missed`, `artifact`, `unsure`, or empty
- `artifact_class`: artifact/category label when rejected as artifact
- `confidence`: `high`, `medium`, `low`, or empty
- `reason_tags`: rationale tags such as `manual`, `duplicate`, or
  `needs_second_review`
- `reviewer_id`
- `updatedAt`
- `notes`

## Migration

The browser and `neurobench.annotations.migrate_annotations_v3()` both map
legacy labels automatically:

```text
accept -> accepted
reject -> rejected
unsure -> unsure
```

Use `tools/export_annotations.py` to write a migrated `annotations_v3.json` plus
ROI and event TSV files. The dashboard's browser-side TSV exports also include
`reviewer_id` and `updatedAt` for ROI, event, suggestion, and split/merge rows.

## Run-Scoped Labels

The workbench preserves legacy top-level `rois`, `events`, `suggestions`,
`promotedRois`, and `virtualRois` as the active working copy. It also mirrors
those maps under `runs[run_id]` so labels from one architecture run are not
mixed into another run with different ROI IDs or event calls. Older
single-run `annotations.json` files are migrated into the baseline run when the
dashboard opens.

## Review Bookmarks

`bookmarks` is a lightweight browser/workbench revisit list. Each entry stores:

- `id`, `label`, and `createdAt`
- `runId`
- `frame`
- optional `roiId`, `eventFrame`, and `suggestionId`

Bookmarks are navigation aids. They do not count as scientific labels.

## Review Stats

The browser records lightweight review-session metadata in `reviewStats`:

- `sessionStartedAt`
- `lastActionAt`
- `actions`, a count of review operations such as ROI accept/reject, event
  labels, suggestion labels, and v3 field changes

These fields are for workflow auditing only. They are not scientific labels.

## Reviewer Provenance

The Review toolbar includes a `Reviewer` field. When set, new ROI, event,
suggestion, virtual ROI, and split/merge edits are stamped with `reviewer_id`
and `updatedAt`. This is lightweight provenance for lab workflows and
inter-rater comparison; reviewer identity is still configurable per exported
annotation file when using `tools/compare_annotations.py`.

For existing annotation files, use the offline backfill helper to stamp reviewed
labels that are missing reviewer provenance:

```bash
python3 tools/backfill_reviewer_ids.py \
  --annotations Outputs/NeuronReview/calcium_rest_cropped/app/annotations.json \
  --reviewer-id reviewer_a \
  --out Outputs/NeuronReview/calcium_rest_cropped/app/annotations_reviewer_a.json
```

Use `--run-id RUN_ID` for one run-scoped bucket, `--all-runs` for every run
bucket, or `--in-place` only when intentionally modifying the source file. Use
`--dry-run --summary-json reviewer_backfill_summary.json` to inspect how many
items would be stamped before writing a modified annotation file.

To audit reviewer coverage without choosing a reviewer ID:

```bash
python3 tools/backfill_reviewer_ids.py \
  --annotations Outputs/NeuronReview/calcium_rest_cropped/app/annotations.json \
  --audit-only \
  --summary-json reviewer_provenance_audit.json
```

The dashboard can also export a browser-side reviewer provenance audit from the
Export panel or Report page. That audit is intended for quick lab handoff: it
contains stamped/missing counts by label type, per-reviewer contribution counts,
and the reviewed labels that still lack `reviewer_id`.

## Review UI State

Some Review-page behavior is intentionally outside the scientific annotation
schema:

- Review layout section order
- collapsed/expanded state for Review panels
- selected tab or Architecture Lab mode
- transient selected ROI/event IDs
- zoom, playback, and overlay visibility preferences unless explicitly saved in
  review settings

These values may be useful browser preferences, but they should not be treated
as ROI, event, discovery, or virtual-ROI labels.

The Selected ROI Context panel is also derived state. Its crop, footprint
overlay, filmstrip, and metric readouts are computed from `review_data.json`,
current frame/event selection, and dataset metadata such as
`pixel_size_microns`. Accessibility labels and visible text in that panel should
summarize the selected ROI context, but they do not introduce new annotation
fields by themselves.

## Pipeline Build Metadata

Architecture Lab Build mode exports planned pipeline configuration separately
from `annotations.json`. Planned pipeline manifests are not annotation records
and should not be merged into ROI, event, discovery, or virtual-ROI maps.

Use `annotations.json` for reviewer decisions. Use dataset, architecture-run,
or planned-pipeline manifests for data provenance, pipeline parameters,
expected output paths, and run status. In v1, planned-pipeline metadata means
the browser configured or exported a run plan; it does not mean the pipeline was
executed in-browser.
