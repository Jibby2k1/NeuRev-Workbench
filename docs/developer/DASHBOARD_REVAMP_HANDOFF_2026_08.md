# NeuRev Dashboard Revamp Handoff — August 2026

Status: implementation and design handoff  
Last updated: 2026-08-08

This note is the compact starting point for continued dashboard work. It
separates behavior already implemented in the maintained Workbench from the
new visual direction captured in Figma. Researchers should still use
[How to Use the NeuRev Dashboard](../HOW_TO_USE_DASHBOARD.md) for normal
operation and
[Single-Reviewer Annotation Correction Dashboard MVP](SINGLE_REVIEWER_ANNOTATION_DASHBOARD_MVP.md)
for the full scientific and persistence contract.

## Implemented and validated

- The annotation-correction workspace preserves a viewport-fit 50/50 split.
- Raw and the selected processed stage are vertically stacked on the left.
- Selection/highlight tools, playback, continuous frame scrubbing, processed
  stage selection, and overlay selection share the left toolbar.
- Overlay choices distinguish selected/linked ROIs from field-wide expert,
  model, or combined annotations.
- The right half consolidates queue, stable ROI identity, summary, synchronized
  neighborhood close-ups, and fixed-height Raw/processed traces.
- Trace interaction supports hover values, click/drag frame changes, horizontal
  zoom, reset, and a synchronized current-frame cursor.
- The completed Spon Ca Burst v5 real-data adapter uses lazy Raw/MSICA/MSLN
  frames for UI frames 1800–2359 and compact per-ROI traces.
- Unlabeled recordings are supported through model-only proposal mode, an
  editable empty expert revision, and proposal workbooks for blinded or
  model-assisted review.
- Scientific audits default to separated Expert Annotations, Model
  Annotations, and Comparison evidence with deterministic, LLM-readable
  metadata.

## Editable visual direction

Figma file:
[NeuRev Dashboard Audit & Experiment Direction](https://www.figma.com/design/FSCKj9XYB1TzmLxktwMQMi)

The file contains three pages:

1. `01 · Current-State Audit` — wide, compressed, and minimum-desktop evidence.
2. `02 · Review Workspace Redesign` — a quieter scientific-workstation shell
   using real microscopy frames and traces.
3. `03 · Experiment Comparison` — a five-arm comparison and common evaluation
   contract for representation/objective studies.

Local capture sources and real copied assets are retained under the ignored
output root
`Outputs/DashboardDesign/neurev_dashboard_figma_board_20260807/`.

## What the concept changes

- Protects the microscopy wells as the visual center of the screen.
- Pins ROI identity and decision actions above the right evidence workspace.
- Uses evidence, stage-sequence, metadata, and queue tabs to control density.
- Separates overlay source (`model`, `expert`, `both`) from overlay scope
  (`selected`, `selected + linked`, `all`).
- Keeps stage/display provenance visible, including signed-MSLN scaling and the
  rule that sparse unlabeled pixels are unknown rather than negative.
- Adds an experiment-level candidate funnel and resource/stability guardrails
  beside biological metrics.

The Figma workspace is a design direction, not a claim that all visual changes
have been merged into the production Workbench. The maintained dashboard code
remains in `neurobench/workbench/assets/`; generated or archived app assets
must not be edited as the source of truth.

## Recommended next implementation slice

1. Translate the pinned ROI header and evidence tabs into the maintained source
   modules without changing revision or annotation semantics.
2. Add viewport regression fixtures at 1920×1080, 1440×900, and 1280×800.
3. Assert that Raw and processed viewers, trace, and close-up remain visible at
   each desktop fixture; allow internal right-panel scrolling rather than page
   overflow.
4. Preserve keyboard navigation, continuous frame scrubbing, and trace
   interaction while reducing toolbar chrome.
5. Add experiment provenance and candidate-funnel data only through explicit
   schema fields; never infer scientific state from filenames or pixels.

## LLM-efficient reading order

1. This handoff.
2. `docs/HOW_TO_USE_DASHBOARD.md` for researcher-visible behavior.
3. `docs/developer/SINGLE_REVIEWER_ANNOTATION_DASHBOARD_MVP.md` for invariants,
   API, schemas, and delivery slices.
4. `docs/workflows/SCIENTIFIC_AUDIT_OUTPUT_STANDARD.md` for required evidence.
5. `neurobench/workbench/assets/src/80_review_subpages.js` and
   `neurobench/workbench/assets/workbench.css` for the maintained correction UI.

Do not recursively inspect generated apps first. Use `review_data.json`,
`annotation_session_context.json`, and scientific-audit `llm_context.json`
before opening large media or embedded HTML.
