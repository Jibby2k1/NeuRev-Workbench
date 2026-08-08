# Scientific Audit Output Standard

This is the default artifact contract for every new NeuRev experiment. These
outputs are primary scientific evidence, not optional presentation material.
They expose temporal behavior, spatial context, missed labels, spurious
candidates, and differences between pipeline stages that aggregate metrics can
hide.

For the interactive review and annotation workflow that precedes publication
and audit generation, see
[How to Use the NeuRev Dashboard](../HOW_TO_USE_DASHBOARD.md).

## Default Policy

Omitting `scientific_audit` from a manifest means the audit is enabled. A run
may omit the complete audit only when the user explicitly requests that
exception and the resolved configuration records a specific `opt_out_reason`.
Individual required outputs cannot be selectively disabled.

When labels exist, audit every unique ROI and every labeled occurrence. When
labels do not exist, freeze the label-free candidate selection first and audit
a declared, reproducible candidate-surrogate panel. Never treat the absence of
labels as permission to omit the audit.

```json
{
  "scientific_audit": {
    "enabled": false,
    "opt_out_reason": "User explicitly requested a metrics-only run on YYYY-MM-DD."
  }
}
```

## Required Three-Section Evidence Set

Every labeled experiment must keep annotation sources visually separated.

### 1. Expert Annotations

- One full-field video containing expert markers only.
- One close-up video and one exact-pixel full-duration trace per unique expert
  ROI. No model marker may appear in this section.
- Videos show synchronized Raw and scientifically relevant processed stages.

### 2. Model Annotations

- One full-field video containing frozen model markers only.
- One close-up video and one exact-pixel full-duration trace per consolidated
  model ROI. No expert marker may appear in this section.
- Videos explicitly show the sequential pipeline, not generically labeled
  caches. For Raw -> MSICA -> MSLN detection this is: Raw input, selected MSICA
  branch, signed MSLN, framewise squared MSLN evidence, and the burst-pooled
  detection map used for candidate ranking.
- Signed MSLN uses a fixed symmetric scale with zero at mid-gray. Squared and
  pooled detection evidence use fixed zero-based scales with zero black.

### 3. Comparison

- Figures and tables only by default: no videos and no close-ups.
- Spatial figures contain exactly two panels: `Raw matched comparison` and
  `MSICA + MSLN matched comparison`.
- Both scientific backgrounds are grayscale. MSLN evidence may affect
  luminance, but must not be green, orange, or another annotation color.
- Produce one trace comparison per expert occurrence against the geometrically
  nearest frozen model candidate from the same event/burst. Record distance,
  candidate rank and score, event correlation, best lag, and the separately
  computed one-to-one match status.
- Preserve aggregate distance-versus-similarity, rank-versus-distance,
  nearest-distance, and model-recurrence diagnostics when applicable.

Across all sections, green is reserved for expert ROIs, orange for model
predictions, and pale yellow for one-to-one match links. Do not use a
red-white-blue scale or colored fluorescence/evidence tint beneath annotations.
Unlabeled experiments use the Model section plus candidate-surrogate traces;
the Expert section is explicitly `not_applicable`, never silently omitted.

## Interpretation Rules

- Coordinates are `x=column`, `y=row`. UI frames are one-based and inclusive;
  NumPy intervals are zero-based and half-open.
- Freeze label-free models, hyperparameters, thresholds, candidate budgets,
  and rankings before using labels for the final audit. Preflight label use is
  limited to geometry validation and projection overlays.
- Unmatched labeled ROIs are misses at the declared comparison operating point.
  Unlabeled candidates remain unknown, not false positives.
- Distinguish the global label-free operating point, any family-specific
  diagnostic comparison, and protected or label-informed ceilings.
- Use fixed spatial crops, intensity scales, marker semantics, frame timing,
  and comparison budgets across architectures. Report any unavoidable
  exception.
- A completed computation is not scientifically audit-complete until this
  artifact set passes inventory and media validation.

## Artifact Layout

```text
<audit_root>/
  REPORT.md
  status.json
  summary.json
  artifact_index.json
  llm_context.json
  validation.json
  1_Expert_Annotations/
    README.md
    videos/*full_field*.mp4
    videos/closeups/roi_<id>.mp4
    figures/traces/roi_<id>.png
    metadata/roi_<id>.json
  2_Model_Annotations/
    README.md
    videos/*full_field*.mp4
    videos/closeups/model_roi_<id>.mp4
    figures/traces/model_roi_<id>.png
    metadata/model_roi_<id>.json
    model_occurrences.csv
  3_Comparison/
    README.md
    spatial_overview.png
    burst_<id>_comparison.png
    trace_comparisons/<occurrence>.png
    nearest_roi_trace_metrics.csv
    expert_model_matches.csv
```

Names may carry experiment-specific prefixes, but the directory roles and
cardinalities are stable. Validate the inventory with
`neurobench.reports.scientific_audit.require_scientific_audit` before marking
the experiment complete.

## LLM-Efficient Audit Index

Agents must inspect `llm_context.json`, `summary.json`, `artifact_index.json`,
and `validation.json` before opening large media. `llm_context.json` is a small,
stable entry point containing the experiment/lane identifier, stage sequence,
operating point, coordinate and frame conventions, section applicability,
expected and observed counts, marker semantics, display scales, key findings,
limitations, and paths to the primary tables and representative artifacts.

Use deterministic filenames and relative paths. Tables contain one row per
occurrence and stable identifiers linking expert ROIs, model ROIs, traces,
videos, and matches. Do not encode required scientific facts only in Markdown,
filenames, or pixels. An LLM should be able to answer inventory, provenance,
operating-point, and match-count questions from the four small JSON files and
CSV headers without decoding a video or recursively scanning the audit root.

## Resource-Safe Production

Audit production follows the same guarded-run rules as the experiment. Use a
new non-colliding output root, reconstruct or load each finalist once, cache
bounded intermediates, render with explicit thread counts, process ROIs
resumably, and write metadata atomically. Generate the full-field video first,
then independently resumable close-ups and figures. Validate every video with
frame count, duration, dimensions, decode checks, and visible annotation-pixel
checks. A renderer failure must leave the scientific run recoverable and the
audit status explicitly incomplete.

## Review Checklist

- Expected and observed expert/model ROI and occurrence counts agree.
- Every applicable expert and model ROI has a section-pure close-up and trace.
- Expert videos contain no model markers; model videos contain no expert
  markers; encoded-frame color checks verify this separation.
- Model videos show the declared sequential stage outputs and the exact
  evidence transformation used for candidate ranking.
- Comparison contains no videos or close-ups, uses only the two standardized
  grayscale spatial panels, and has one nearest-candidate trace per expert
  occurrence.
- Nearest-candidate identity and one-to-one metric assignment remain distinct.
- Raw, intermediate, evidence, and pooled-map panels are synchronized and use
  declared fixed display scales.
- `llm_context.json`, summary, artifact index, validation, reports, and tables
  agree on provenance, operating point, counts, paths, and interpretation.

This audit is the main bridge between algorithmic measurements and defensible
scientific interpretation. New workflow documentation must state how it
implements this standard and identify any explicit user-authorized opt-out.
