# Single-Reviewer Annotation Correction Dashboard MVP

This is the implementation and scientific contract. Researchers operating the
dashboard should start with
[How to Use the NeuRev Dashboard](../HOW_TO_USE_DASHBOARD.md), then return here
for revision schemas, API behavior, acceptance criteria, and delivery status.

Status: implementation specification
Date: 2026-08-06

## Implementation Status

Slices 1 and 2 and the first Slice 3 vertical cut are implemented. Public schema/model contracts cover annotation
revisions, append-only operations, and identity/affine view mappings. Workbench
can create a complete revision root through collision-refusing staged
persistence and load it with cross-file validation. A deterministic three-frame
Raw/MSICA/MSLN fixture exercises canonical coordinates and intensity semantics.

Slice 2 adds a read-only Correct labels subpage with matched, missed, and
model-only-unknown queues; synchronized Raw/MSICA/MSLN selection; a canonical
coordinate inspector; and exact-pixel, expert-ROI-mean, and linked-model-mean
traces. Its feedback-revised layout uses a 50/50 split: vertically stacked Raw
and processed videos with spatial tools on the left, and one consolidated
review panel on the right containing the queue, compact selection summary,
synchronized Raw/processed ROI-neighborhood close-ups, and time-series
diagnostics. The visual fixture is
examples/annotation_correction_slice2.example.json.

Slice 3 now provides authenticated draft create/load/list/append endpoints,
optimistic revision-token conflicts with current-snapshot responses, atomic
projection and operation-log replacement, and expert-only Move, Resize,
Tombstone/Restore, Notes, Undo/Redo, browser/API autosave, and JSON draft
export controls. Model geometry remains immutable. Undo and redo append
compensating operations rather than deleting history.

The second Slice 3 control set adds explicit Link/Unlink, immutable-proposal
promotion into a new expert ROI, and non-overlapping one-based inclusive event
interval edits with trace bands.

Slice 4 adds an attributable change-review surface, server-constructed draft
forks, and immutable publication. Forking copies the validated current
projection into a new draft with a fresh operation history and explicit parent;
publishing validates the expected token and creates a separate read-only child
without rewriting the working draft. Scientific reevaluation and audit
generation remain later explicit gates.

The real-data adapter is implemented for the completed multi-lag MSICA v5 audit
at `Outputs/NeuronReview/spon_ca_burst_multilag_msica_v5_annotation_correction_v2`; the v1 visual revision remains preserved.
It preserves the verified 50/50 correction screen and attaches UI frames
1800–2359 through validated lazy frame patterns. Compact per-ROI Raw/MSICA/MSLN
pixel and mean traces replace embedded full-stage arrays. The bridge imports 27
expert identities (79 occurrences), 156 frozen model identities (232
occurrences), and 18 unambiguous identity-level correspondences; the one
supported-but-ambiguous model identity remains explicitly unlinked rather than
forcing a false one-to-one relation. Browser QA must assert
`review-correction-mode`, the visible correction workspace, five correction
canvases, and the absence of a legacy-screen fallback.

## Outcome

Extend the existing NeuRev Workbench Review surface with a focused label-correction workspace. Do not create a parallel dashboard or replace the current video review, traces, ROI tools, autosave, queues, and exports.

The MVP supports one researcher correcting expert annotations while inspecting synchronized Raw and processed representations and auditing frozen model proposals. Its durable result is an immutable annotation revision plus an explicitly regenerated scientific audit.

## Core Job

Inspect an expert or model ROI in Raw and processed views, verify its spatial and temporal evidence, correct the expert label if warranted, and publish a traceable revision without confusing an unlabeled proposal with a negative.

Optimize repeated ROI decisions. Moving to the next item preserves frame, zoom, trace cursor, and selected representation.

## Scientific Invariants

These are release blockers:

1. Source videos, completed experiment outputs, imported expert annotations, and model proposals are immutable inputs.
2. Drafts record append-only operations. Removing a label creates a tombstone.
3. Promoting a model proposal and linking expert/model geometry are separate, explicit operations.
4. Unmatched model candidates are `unknown`, never automatically negative or false positive.
5. UI coordinates use `x = column`, `y = row`; UI frames are one-based and inclusive; NumPy intervals are zero-based and half-open.
6. Canonical geometry is stored in source-video pixels. Processed-view selection uses a declared transform and is read-only if invalid.
7. Saving or publishing labels never silently changes model selection, metrics, videos, or reports. Reevaluation is a separate explicit job.
8. New label-driven outputs retain projection-overlay preflight and the standard Expert Annotations / Model Annotations / Comparison audit contract.
9. Published revisions and completed output roots are never overwritten.

## Screen Layout

### Context bar

Show dataset, frozen run, active annotation revision and parent, reviewer, Draft/Published state, autosave state, undo/redo, Review changes, and Publish revision. Visually separate frozen-run identity from label revision.

### Review queue

The left rail provides stable queues:

- Matched expert labels
- Missed expert labels
- Model-only unknown candidates
- All expert labels
- All model proposals
- Recently edited

Rows show ID, status, event count, draft-change marker, and trace availability. Filters may narrow by burst/event, confidence, review state, or identifier. Membership derives from the frozen operating point and selected revision.

### Synchronized viewer

The center defaults to Raw beside one processed representation. The second panel may show representations supplied by the frozen run, including MSICA, signed MSLN, MSLN-squared frame evidence, or burst-pooled evidence.

Panels share frame, playback, pan, zoom, and cursor. Frame-slider input is rendered directly through a request-animation-frame throttle, and neighboring lazy frame assets are prefetched after selection settles. The overlay defaults to the selected expert/model correspondence; separately named options expose the selected expert, selected model, no annotations, all experts, all models, or both complete sets.

Markers cannot rely only on color: expert geometry uses a circle and model geometry a square/crosshair. When both appear, retain green expert, orange model, and yellow match semantics over grayscale image content.

### ROI inspector

The right rail shows canonical coordinates, geometry, source view, UI and zero-based frames, event intervals, linked proposal, confidence, review state, and notes. Context actions are Move, Resize, Link, Unlink, Promote proposal, Tombstone, Restore, and Revert this edit.

Changing selection during an incomplete geometry edit requires Apply or Discard.

### Trace dock

The fixed-height bottom dock follows the frame cursor. Hover reports frame-local values, click/drag changes the synchronized video frame, wheel input zooms the horizontal time window around the pointer, and reset restores the supplied interval. It includes:

- exact selected-pixel intensity;
- expert ROI mean;
- linked model ROI mean when applicable;
- Raw and processed-stage traces;
- expert event bands and current-frame marker.

Legends state whether each signal is signed, normalized, squared, pooled, or amplitude preserving. Initial traces may be deterministically downsampled; selection loads or materializes the full-resolution shard.

### Model-only pending-label mode

A model-only `annotationCorrection` payload declares `mode: model_only`,
empty `expert_rois` and `matches`, and explicit expert/comparison states
of `not_applicable_pending_labels`. The UI defaults to the model proposal
queue and selected-model overlay. An empty revision remains editable so proposal
promotion can create the first expert ROI with provenance. Package generation
is owned by `neurobench/workbench/model_proposals.py` and the
`workbench model-proposal-package` CLI.


## Primary Flow

1. **Open session.** Select dataset, frozen experiment, base published revision, and reviewer ID. Create or resume one draft; show checksums and warnings.
2. **Select evidence.** Choose a queue item or marker in any view. All viewers, inspector fields, and traces resolve to one canonical location. Empty-space clicks move only the probe cursor.
3. **Inspect.** Scrub/play the event, switch processed representation, and compare pixel and ROI traces. Focus event changes only the viewing window.
4. **Correct.** Primary tools are Select, Exact Pixel/Center, Circle, Move, and Resize. Existing lasso and mask brush remain Expert-mode options, not first-gate requirements. Snap to model peak is explicit and records its proposal.
5. **Save draft.** Applying an edit appends an operation, updates the projection, and saves atomically. Undo/redo append compensating operations.
6. **Review and publish.** Present creates, moves, resizes, tombstones, restores, links, promotions, and event edits. Publish creates an immutable child revision.
7. **Reevaluate.** Offer separate durable jobs to recompute frozen-model matching/metrics and to generate a new scientific-audit root.

## Interaction States

| State | Required behavior |
| --- | --- |
| Idle | Viewer usable; no creation tool armed. |
| Selected | One canonical ROI or probe drives all views and traces. |
| Editing new | Preview distinct and excluded from exports. |
| Editing existing | Before geometry remains visible until apply/discard. |
| Saving | Prevent or queue a second mutation. |
| Saved | Show new revision token and timestamp. |
| Trace materializing | Spatial review remains available with bounded progress. |
| Invalid transform | Processed view is read-only and names missing mapping. |
| Conflict | Stop writes, preserve local operations, offer reload or fork. |
| Publishing | Lock mutations through success or recoverable failure. |
| Reevaluating | Durable progress survives browser closure. |

Shortcuts cover queue navigation, play/pause, frame navigation, Select, Center, Circle, Apply, Discard, Undo, Redo, and Focus event and appear in Help.

## Coordinate and View Contract

Every selectable representation declares a stable view ID, source video ID, time/y/x shape, source-to-view transform, frame mapping, and intensity semantics. The MVP accepts identity and explicit affine mappings.

Stored centroids may be floating point; renderers declare rounding. Displays use `x, y`, arrays use `[y, x]`, and frames show both `UI frame N` and `index N-1`.

## Revision Data Contract

Retain schema-v3 compatibility while adding an immutable revision envelope. Any schema-v4 migration is explicit, never a silent rewrite of archived apps.

Revision metadata includes revision ID, parent ID, draft/published state, reviewer ID, frozen run ID, source-annotation checksum, created/updated times, monotonic revision token, and operation count.

Each operation contains stable operation and target IDs, type, before/after values, evidence view, UI frame, source coordinates, reviewer, timestamp, and expected revision token. Types are create, move, resize, tombstone, restore, link, unlink, promote, edit-notes, and edit-event-interval.

ROI projections add canonical point/circle/mask geometry, source view and frame, linked proposal ID, confidence/review state, tombstone provenance, and a stable ID across revisions.

```text
annotation_revisions/
  ann_<id>/
    revision.json
    annotations.json
    operations.jsonl
    exports/
```

Writes use a temporary sibling and atomic rename. Published directories are application-read-only. Exports name both annotation revision and frozen run.

## Local API

| Method and route | Purpose |
| --- | --- |
| `GET /api/annotation-revisions` | List immutable revisions and resumable drafts. |
| `POST /api/annotation-revisions` | Create a complete validated revision root (migration/bootstrap). |
| `GET /api/annotation-revisions/{id}` | Load revision, labels, and view contract. |
| `POST /api/annotation-revisions/{id}/operations` | Append with expected token. |
| `POST /api/annotation-revisions/{id}/fork` | Server-construct a fresh draft child from the validated projection. |
| `POST /api/annotation-revisions/{id}/publish` | Validate token and create an immutable published child. |
| `POST /api/annotation-revisions/{id}/materialize-traces` | Build trace shards. |
| `POST /api/annotation-revisions/{id}/reevaluate` | Recompute matching/metrics. |
| `POST /api/annotation-revisions/{id}/scientific-audit` | Generate new audit root. |
| `GET /api/jobs/{id}` | Read durable heartbeat, result, or failure. |

Preserve `GET/PUT /annotations.json` during migration. New UI writes use revision operations; the legacy file is a compatibility projection.

Reject out-of-bounds geometry, invalid intervals, duplicate stable IDs, missing transforms, stale tokens, and existing publish/output targets.

## Matching Semantics

Matching derives from annotation revision, frozen run, spatial threshold, temporal criterion, and operating point; it is not expert truth. Distinguish matched expert, missed expert, unmatched model unknown, explicitly promoted proposal, and manually linked correspondence.

## Resource Behavior

Editing is CPU-light and does not reserve the GPU. Use existing video assets, lazy trace shards, bounded frame caches, and deterministic display downsampling. Trace and audit work runs as bounded durable jobs.

Before any GPU reevaluation or rendering job, verify inputs/checksums, output collision, disk/RAM headroom, active processes, and GPU memory, then display the job plan. Label editing itself never triggers GPU work.

## LLM-Efficient Handoff

Every draft and publication writes `annotation_session_context.json` with dataset/run/revision/checksums, matching policy, queue and operation counts, warnings, recent ROI summaries, and direct paths to tables, traces, figures, videos, and reports. It states that unmatched proposals are unknown and identifies the annotation revision used for every metric.

This is the first context for agents; they do not need to scan videos or recursively inspect an output tree to understand session status.

## Implementation Ownership

| Concern | Primary implementation |
| --- | --- |
| Revision state, autosave, token | `assets/src/10_state_persistence.js` |
| Linked selection/rendering | `assets/src/20_review_core.js` |
| Geometry tools | `assets/src/24_cfar_mask_annotation.js` |
| Controls, undo/redo, trace jobs | `assets/src/25_review_controls.js` |
| Correction workspace/queues | `assets/src/80_review_subpages.js` |
| Revision and job API | `neurobench/workbench/server.py` |
| Trace extraction | `neurobench/workbench/materialize.py` |
| Derived matching | `neurobench/workbench/label_reconciliation.py` |
| Public validation | `schemas/annotations.schema.json`, `neurobench/models/annotations.py` |
| Audit generation | `neurobench/reports/scientific_audit.py` |

Asset paths are relative to `neurobench/workbench/`. Never edit generated `assets/workbench.js`; rebuild it from the source manifest.

## Delivery Slices

1. Revision/operation contracts, transforms, collision tests, and tiny Raw/MSICA/MSLN fixture.
2. Read-only queues, linked viewers, probe, inspector, and trace dock.
   The correction workspace defaults to a viewport-fit 50/50 split: Raw and
   processed views divide the left column vertically, while bounded review
   sections scroll internally on the right. Below 760 px it falls back to a
   single-column layout so controls remain usable.
3. Draft edits, autosave, concurrency, undo/redo, reload, and exports.
4. Change review, immutable publication, and compatibility projection.
5. Lazy real-data frame/trace loading, audit-table bridge, explicit
   reevaluation, audit generation, durable jobs, and LLM context.

Each slice remains runnable. Provenance and collision safety are not deferred.

## MVP Acceptance Criteria

1. Raw and identity-mapped processed selection yields identical canonical coordinates, frame, and trace cursor.
2. Invalid transforms disable processed-view editing.
3. Center/circle creation, move/resize, and event edits survive reload exactly.
4. Undo/redo and tombstone/restore survive reload and remain in history.
5. Model linking/promotion is explicit and records proposal and evidence view.
6. Unmatched proposals render and export as unknown.
7. Stale tokens cannot overwrite newer edits; local work can reload or fork.
8. Publish creates an immutable child with reviewer, checksum, timestamps, and validation; duplicate target publication is refused.
9. Original annotations, frozen run, archived metrics, and completed audits remain byte-for-byte unchanged.
10. Pixel/ROI traces align with the cursor and disclose intensity semantics.
11. Reevaluation names the frozen run and published revision.
12. Regenerated audits contain expert-only close-ups/videos/traces, model-only sequential-stage close-ups/videos/traces, and only required matched figures and nearest-candidate trace comparisons in Comparison.
13. Image content is grayscale; annotation colors are reinforced by shape.
14. The LLM context resolves primary artifacts and status without video review.
15. Browser closure does not cancel durable jobs.

## Verification

- Model/schema tests for revisions, operations, geometry, frames, and tokens.
- Server tests for atomic saves, immutable publish, collision refusal, legacy compatibility, and job recovery.
- Browser runtime tests for linked selection, transforms, incomplete edits, overlay isolation, shortcuts, and queues.
- Trace tests against direct NumPy pixel/mask means.
- Tiny end-to-end draft, reload, publish, reevaluate, and audit test.
- Hash regression proving archived annotations and experiment fixtures unchanged.

## Deferred

Multi-reviewer concurrency, assignment, agreement, conflicts, adjudication, consensus labels, remote collaboration, automatic retraining/model selection, and treating sparse annotations as exhaustive negatives are out of MVP scope. Reviewer, parent-revision, and operation provenance remain in the contract for future extension.

## Definition of Done

A researcher can correct expert labels from Raw or processed evidence, inspect aligned traces and coordinates, publish an immutable revision, and deliberately generate a comparable audit package without overwriting the experiment or conflating model proposals with expert truth.
