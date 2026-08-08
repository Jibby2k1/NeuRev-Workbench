# How to Use the NeuRev Dashboard

This is the researcher-facing guide for opening the NeuRev Workbench, reviewing
Raw and processed calcium-imaging evidence, correcting expert ROIs, and
publishing a traceable annotation revision.

> Start here if your goal is to inspect results or correct labels. Developers
> changing dashboard code should use the
> [single-reviewer implementation specification](developer/SINGLE_REVIEWER_ANNOTATION_DASHBOARD_MVP.md)
> after reading this guide.

## At a Glance

| If you want to… | Go to… |
| --- | --- |
| Open an existing dataset | [Start the dashboard](#1-start-the-dashboard) |
| Understand the screen | [Screen map](#2-screen-map) |
| Review a candidate or expert ROI | [Inspect evidence](#3-inspect-spatial-and-temporal-evidence) |
| Correct an expert label | [Edit expert annotations](#4-edit-expert-annotations) |
| Promote a model proposal | [Link or promote](#5-link-or-promote-model-proposals) |
| Review everything you changed | [Review and publish](#6-review-and-publish) |
| Recover from a conflict | [Draft recovery](#7-drafts-recovery-and-conflicts) |
| Prepare a real-data review | [Real-data checklist](#8-real-data-checklist) |
| Diagnose a problem | [Troubleshooting](#9-troubleshooting) |

## The Safe Review Sequence

1. Open the correct dataset and frozen run.
2. Confirm coordinate, frame, and intensity semantics.
3. Inspect Raw and processed spatial evidence.
4. Inspect the pixel and ROI traces.
5. Correct expert annotations only when the evidence supports a change.
6. Review the append-only change list.
7. Publish an immutable child revision.
8. Reevaluate metrics and generate a scientific audit as separate explicit jobs.

Publishing labels does **not** silently rerun the model, replace an experiment,
or update historical metrics.

## 1. Start the Dashboard

### Find an existing app

Dashboard apps normally live under `Outputs/NeuronReview/<dataset>/app/`. If the
path is unknown, inspect the dataset catalog:

```bash
.venv-neurobench/bin/python -m neurobench.cli.main dataset catalog \
  --root .
```

### Check the app before serving

```bash
.venv-neurobench/bin/python -m neurobench.cli.main workbench status \
  --app-dir Outputs/NeuronReview/<dataset>/app \
  --json
```

Check the reported dataset identity, annotation path, and asset status before
opening the app.

### Serve current dashboard code

```bash
.venv-neurobench/bin/python -m neurobench.cli.main workbench serve \
  --app-dir Outputs/NeuronReview/<dataset>/app \
  --asset-mode current \
  --host 127.0.0.1 \
  --port 8765
```

Open `http://127.0.0.1:8765/`.

Use `--asset-mode current` for normal review. It serves the maintained dashboard
without rewriting the archived app. Use `--asset-mode installed` only when the
historical UI itself is the object of comparison.

### Open the correction workspace

Use the Review navigation and choose annotation correction, or open:

```text
http://127.0.0.1:8765/#annotation-correction
```

The correction workspace requires an `annotationCorrection` contract in
`review_data.json`. If that contract is absent, the standard Workbench pages
still function, but revisioned correction controls will not appear.

> **Real-data correction app:** the frozen Spon Ca Burst multi-lag MSICA v5
> audit is attached at
> `Outputs/NeuronReview/spon_ca_burst_multilag_msica_v5_annotation_correction_v2/app`.
> It uses the exact correction workspace shown below, lazy Raw/MSICA/MSLN frame
> patterns for UI frames 1800–2359, and compact per-ROI traces. Full stage arrays
> are not embedded in `review_data.json`.
>
> Serve it with:
>
> ```bash
> .venv-neurobench/bin/python -m neurobench.cli.main workbench serve \
>   --app-dir Outputs/NeuronReview/spon_ca_burst_multilag_msica_v5_annotation_correction_v2/app \
>   --asset-mode installed --host 127.0.0.1 --port 8877
> ```
>
> Then open `http://127.0.0.1:8877/#annotation-correction`. The payload must
> report the frozen lane, 27 expert identities, 156 model identities, and lazy
> frame patterns for `raw`, `msica`, and `msln`; otherwise stop rather than
> accepting a fallback Review screen.

## 2. Screen Map

The correction workspace preserves a 50/50 scientific-review split.

| Area | Purpose |
| --- | --- |
| Left, top | Raw video in source coordinates |
| Left, bottom | Selected processed representation |
| Left toolbar | Selection/highlight tool, playback, frame, processed stage, and overlay |
| Right, review queue | Choose matched experts, missed experts, model-only unknowns, or all ROIs |
| Right, selection summary | Stable ID, coordinates, event information, links, notes, and edit controls |
| Right, ROI neighborhood | Synchronized Raw and processed close-ups |
| Right, trace dock | Raw and processed pixel/ROI time series with the current-frame cursor |
| Right, Review changes | Attributable operation history, draft fork, and immutable publication |

On a desktop-sized screen both halves auto-fit the viewport. Individual
right-hand sections scroll internally. Narrow screens fall back to one column.

## 2A. Review Data Before Expert Labels

Use the [Unlabeled Model Proposal Review workflow](workflows/unlabeled_model_proposal_review.md)
when a frozen run has model candidates but expert labels are still pending. The
generated dashboard starts in **Model proposals** with **Selected model only**,
keeps every proposal unknown, and creates an empty expert annotation revision.

Generate a collision-safe package with:

```bash
.venv-neurobench/bin/python -m neurobench.cli.main workbench \
  model-proposal-package \
  --source-app-dir <completed-model-app> \
  --output-root <new-unused-output-root> \
  --event-source supplied
```

Give experts `BLINDED_EXPERT_TEMPLATE.xlsx` for independent annotation.
Use `MODEL_PROPOSALS_FOR_REVIEW.xlsx` only for explicitly model-assisted
review. The latter preserves the familiar event-block layout and adds durable
model identity, occurrence, score, rank, review, and provenance sheets.

If the new TIFF has not been processed yet, first use the end-to-end
`unlabeled_recording` preflight and run commands in the linked workflow. That
path creates the candidate-bearing app and package together, proposes event
windows explicitly as model-derived timing, and generates the default
model-only scientific audit. Do not call `model-proposal-package` directly on
a raw TIFF; it packages frozen candidates but does not perform inference.


## 3. Inspect Spatial and Temporal Evidence

### Choose the review queue

| Queue | Meaning |
| --- | --- |
| Matched expert | Expert ROI linked to a frozen model proposal |
| Missed expert | Expert ROI without a model correspondence |
| Model-only unknown | Frozen model proposal without an expert label; it is unknown, not negative |
| All expert | Every expert ROI in the current revision |
| All model | Every frozen model proposal |
| Recently edited | Expert ROIs touched in the current draft |

### Select an ROI or pixel

- **Select ROI** chooses the nearest expert circle or model square.
- **Highlight pixel** moves the pixel probe without replacing the selected ROI.
- Raw and processed panels resolve selection back to the same canonical source
  coordinate.

Coordinates are always shown as `x=column, y=row`.

### Scrub the evidence

Use Play/Stop, Previous/Next, or the frame slider. Dragging the slider updates
the canvases continuously without rebuilding the workspace; nearby frames are
prefetched so a short scrub reads like a small animation. Playback uses the same
bounded frame path. Frames in the UI are one-based. The corresponding NumPy
index is `UI frame - 1`.

Use the processed-stage selector to compare the supplied representations. For a
Raw → MSICA → MSLN pipeline, inspect at least Raw, MSICA, and MSLN before
changing a label.

### Choose overlays deliberately

| Overlay | Use |
| --- | --- |
| Selected pair | Default; show the selected ROI and its linked expert/model correspondence |
| Selected expert only | Show only the selected or linked expert ROI |
| Selected model only | Show only the selected or linked frozen model proposal |
| No annotations | Inspect image content without annotation obstruction |
| All expert annotations | Show every expert ROI visible at the frame |
| All model annotations | Show every frozen model proposal visible at the frame |
| All annotations | Show both complete annotation sets for field-level context |

Expert ROIs use green circles. Model proposals use orange squares. The selected
or linked comparison uses yellow. Image content remains grayscale so marker
colors stay legible.

### Read the traces

The trace dock has a fixed height and aligns the current frame across Raw and
processed stages. Hover to read values at a frame. Click or drag within either
lane to move the video and trace cursor together. Use the mouse wheel to zoom
the time axis around the pointer; double-click or choose **Reset view** to show
the complete supplied interval. Confirm:

- whether the exact selected pixel fluctuates at the annotated time;
- whether the ROI mean supports the same event;
- whether Raw and processed traces preserve or suppress the signal;
- whether event bands align with the observed fluctuation.

Intensity semantics are stage-specific. Signed MSLN uses a fixed symmetric scale
with zero at mid-gray. Squared or pooled detection evidence uses a zero-based
scale with zero black. Display normalization is for visualization and does not
replace the scientific values used by the pipeline.

## 4. Edit Expert Annotations

Model geometry is frozen. Direct edits apply only to expert annotations.

Available expert operations are:

- move the center;
- resize the circle;
- save or clear notes;
- add or remove one-based inclusive event intervals;
- link or unlink a frozen model proposal;
- tombstone or restore an ROI.

Apply one logical change at a time. Every accepted change appends an attributable
operation containing the reviewer, target, evidence view, frame, coordinate,
timestamp, and expected revision token.

Undo and Redo append compensating operations. They never erase history.

### Event interval rules

- Start and end are one-based UI frames.
- End is inclusive.
- Intervals must be within the video.
- Intervals for one ROI cannot overlap.

## 5. Link or Promote Model Proposals

A model-only proposal remains unknown unless a reviewer explicitly acts on it.

- **Link** records correspondence between an existing expert ROI and a frozen
  model proposal.
- **Unlink** removes only that declared correspondence.
- **Promote proposal** creates a new expert ROI at the proposal location with
  explicit provenance.

Promotion never converts or moves the frozen model square. It creates a separate
expert circle that can subsequently be reviewed and edited.

## 6. Review and Publish

Before publishing, read the **Review changes** section. It summarizes operation
counts and lists recent changes with target ROI, evidence view, frame, and
reviewer.

### Publish an immutable child

1. Confirm the current draft token and reviewer.
2. Enter a unique child revision ID.
3. Select **Publish immutable child**.
4. Confirm the header now identifies the published revision as read-only.

Publication validates the current server token and writes a separate child
revision. The working draft and source annotations are not rewritten. Reusing an
existing revision ID is refused.

### Fork an editable draft

Use **Fork editable draft** when starting new work from a published revision or
when a separate line of review is needed. The server copies the validated
current projection, records the parent revision, and starts a fresh operation
history at token zero.

### What publication does not do

Publication does not automatically:

- rerun MSICA, MSLN, or candidate selection;
- recompute matching or metrics;
- regenerate videos, close-ups, traces, or comparison figures;
- overwrite a completed scientific-audit root.

Those are deliberate post-publication jobs so results remain attributable to a
specific frozen run and annotation revision.

## 7. Drafts, Recovery, and Conflicts

Edits autosave through the local revision API when it is available. A browser
copy is also retained for recovery, and **Export draft** writes a portable JSON
snapshot.

If a stale revision token is detected:

1. Writes stop.
2. The local draft remains preserved.
3. Export the draft if there is any uncertainty.
4. Reload the server revision or fork a new draft.
5. Reapply only changes that remain scientifically justified.

Never resolve a conflict by overwriting a completed or published revision
directory.

## 8. Real-Data Checklist

Before reviewing a real experiment, verify all of the following.

### Identity and provenance

- Correct source video and dataset ID.
- Correct frozen pipeline run and stage order.
- Source-annotation checksum recorded.
- Reviewer ID recorded.
- No collision with an existing draft, publication, or audit root.

### Spatial and temporal alignment

- A projection overlay confirms label coordinates.
- `x=column, y=row` is respected.
- UI frames are one-based and inclusive.
- Processed views declare a valid source-to-view transform.
- Raw and processed traces share the same time base.

### Scientific interpretation

- Sparse labels are not treated as exhaustive negatives.
- Model-only candidates remain unknown.
- Label-free thresholds and rankings are frozen before label comparison.
- Visualization normalization is distinguished from pipeline values.
- The intended operating point and candidate budget are recorded.

### Resource safety

Label editing itself is CPU-light and should not reserve the GPU. Before any
reevaluation or video-generation job, check input paths, output collision,
available disk and RAM, active processes, GPU memory, bounded thread counts, and
the read-only preflight report.

## 9. Troubleshooting

| Symptom | Check |
| --- | --- |
| Correction workspace is missing | Confirm `annotationCorrection` exists in `review_data.json` |
| Publish/Fork buttons are disabled | Serve through the local Workbench server and confirm the revision API loaded |
| “Conflict” appears | Export the local draft, then reload or fork from the current server revision |
| Processed panel is blank | Confirm the selected view contract and frame arrays/artifacts exist |
| Raw and processed markers disagree spatially | Inspect `source_to_view`; disable processed editing if the transform is invalid |
| Traces do not line up | Confirm frame mapping, time base, and one-based/zero-based conversion |
| Video looks unexpectedly bright | Read the stage intensity semantics and fixed display range |
| Autosave is unavailable | Do not use a static file; start `workbench serve` |
| An old app looks different | Compare `workbench status`; use `current` for maintained UI and `installed` for historical UI |
| Metrics did not change after publication | Expected: launch reevaluation explicitly against the published revision |

## 10. After Review

A scientifically complete post-publication evaluation should produce the
standard three-section audit:

1. Expert Annotations: expert-only full field, close-ups, and traces.
2. Model Annotations: model-only sequential-stage full field, close-ups, and
   traces.
3. Comparison: grayscale matched spatial panels and nearest-candidate trace
   comparisons.

Read the [Scientific Audit Output Standard](workflows/SCIENTIFIC_AUDIT_OUTPUT_STANDARD.md)
before launching those jobs.

## Related Documentation

| Document | Use it for |
| --- | --- |
| [Neuron Workbench](NEURON_WORKBENCH.md) | Building, serving, and understanding the complete dashboard |
| [Annotation Schema](ANNOTATION_SCHEMA.md) | Legacy/current annotation fields and export semantics |
| [Scientific Audit Output Standard](workflows/SCIENTIFIC_AUDIT_OUTPUT_STANDARD.md) | Required videos, figures, traces, reports, and validation |
| [Single-Reviewer Dashboard Specification](developer/SINGLE_REVIEWER_ANNOTATION_DASHBOARD_MVP.md) | Revision contracts, APIs, implementation slices, and acceptance criteria |
| [Codebase Navigation](CODEBASE_NAVIGATION.md) | Implementation ownership and agent routing |

## One-Minute Pre-Publication Check

- Correct dataset and frozen run.
- Correct reviewer and parent revision.
- Raw and processed evidence inspected.
- Pixel and ROI traces inspected.
- Model-only candidates interpreted as unknown.
- Change list reviewed.
- Unique child revision ID chosen.
- Draft exported if any conflict occurred.
- Reevaluation and audit still planned as separate jobs.
