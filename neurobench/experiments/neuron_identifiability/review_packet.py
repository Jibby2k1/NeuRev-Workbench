"""Generate a compact author-facing review packet from the provisional run."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from .contracts import atomic_json, atomic_text


def _tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def build_review_packet(run_root: Path) -> Path:
    target = run_root / "review_packet"; target.mkdir(parents=True, exist_ok=True)
    suggestions = _tsv(run_root / "02_label_timing_contract/burst_2_timing_suggestions.tsv")
    sites = _tsv(run_root / "03_trace_atlas/trace_index.tsv")
    traces = np.load(run_root / "03_trace_atlas/traces.npz", allow_pickle=False)["raw"]
    site_index = {row["observation_site_id"]: int(row["site_index"]) for row in sites}
    invalid = []
    pdf = target / "burst_2_timing_review.pdf"
    with PdfPages(pdf) as pages:
        for page_start in range(0, len(suggestions), 10):
            fig, axes = plt.subplots(5, 2, figsize=(11, 14), sharex=True, layout="constrained")
            for ax, row in zip(axes.flat, suggestions[page_start:page_start + 10], strict=False):
                site = row["observation_site_id"]; onset=int(row["suggested_onset_ui"]); peak=int(row["suggested_peak_ui"]); end=int(row["suggested_end_ui"])
                valid = onset <= peak <= end
                if not valid: invalid.append({"observation_id":row["observation_id"],"onset_ui":onset,"peak_ui":peak,"end_ui":end,"reason":"suggested ordering is invalid"})
                frames=np.arange(2025,2076); values=traces[site_index[site],frames-1]
                ax.plot(frames,values,color="black",lw=1); ax.axvspan(2040,2063,color="0.8",alpha=.45,label="original")
                ax.axvline(onset,color="tab:green",ls="--"); ax.axvline(peak,color="tab:orange",ls="--"); ax.axvline(end,color="tab:red",ls="--")
                ax.set_title(f"{row['observation_id']}  suggested {onset}/{peak}/{end}" + ("  INVALID" if not valid else ""),fontsize=9,color="crimson" if not valid else "black")
            for ax in axes[-1]: ax.set_xlabel("UI frame (one-based)")
            fig.suptitle("Burst 2 timing review: black=Raw, gray=original 2040-2063; green/orange/red=onset/peak/end", fontsize=13)
            pages.savefig(fig); plt.close(fig)

    manifest = {
        "schema_version":1,
        "review_order":["D1_roi_010_015_identity","D2_burst_2_timing","D3_claim_language","D4_bounded_field_protocol"],
        "immediate_author_decisions":4,
        "engineering_or_analysis_gates":3,
        "timing_suggestions":len(suggestions),
        "invalid_timing_suggestions":invalid,
        "release_remains_held":True,
    }
    atomic_json(target / "review_manifest.json", manifest)
    atomic_text(target / "AUTHOR_DECISIONS.yaml", """schema_version: 1
run_status: provisional_incomplete
decisions:
  - id: D1_roi_010_015_identity
    status: decided
    choice: merge_with_separate_site_geometry
    notes: 'User confirmed same neuron on 2026-08-22; preserve roi_010 and roi_015 as separate observation sites and geometries.'
  - id: D2_burst_2_timing
    status: decided
    choice: retain_original_2040_2063
    notes: 'User accepted the recommended conservative default on 2026-08-22; automatic per-site suggestions are not adopted.'
  - id: D3_claim_language
    status: decided
    choice: approve_provisional_claim_matrix
    notes: 'User approved all eight claims as written on 2026-08-22; scientific provisional and held statuses remain unchanged.'
  - id: D4_bounded_field_protocol
    status: decided
    choice: remove_precision_aim_optional_single_author_audit
    notes: 'User reported that original labels came from two experts but no two new reviewers are available. Original files do not retain separate expert IDs or exhaustive-field coverage; precision is removed, and generated media remains available for qualitative audit.'
submission_metadata:
  authorship: pending
  ethics: pending
  funding: pending
  conflicts: pending
  data_availability: pending
  code_availability: pending
""")
    atomic_text(target / "REVIEW_PACKET.md", """# Author review packet

This is the shortest path through the remaining decisions. The analysis stays provisional until these items and the separately listed engineering gates are resolved.

## Review now

### D1 — ROI 010 and ROI 015 identity

**Question:** Are these two immutable observation sites measurements of the same canonical neuron?

- `merge_with_separate_site_geometry`: one canonical identity (`roi_010`), while retaining both site IDs and both coordinates in every analysis.
- `separate_neurons`: two canonical identities.
- `unresolved`: safest publication default if the image evidence is insufficient.

Review [the spatial overlay](../01_identity_geometry/roi_010_015_overlay.png) and [the site grouping](../01_identity_geometry/canonical_grouping.tsv). The current analysis never collapses the two geometries.

### D2 — Burst 2 timing

**Recommendation:** retain the original common interval, UI frames 2040–2063, unless you want a manual site-by-site adjudication.

Review [the two-page trace packet](burst_2_timing_review.pdf) and [the suggestion table](../02_label_timing_contract/burst_2_timing_suggestions.tsv). Six automatic suggestions are internally invalid: ROI 006, 008, 015, 016, and 017 have peaks before onset, while ROI 012 has a peak after the proposed end. Therefore the suggestions must not be accepted wholesale.

Available decisions:

- retain the original 2040–2063 interval;
- manually review and enter per-site onset/peak/end values;
- exclude Burst 2 from a primary sensitivity analysis while retaining it descriptively.

### D3 — Scientific claim language

Review [the claim matrix](../manuscript/claim_matrix.json) and [limitations table](../manuscript/tables/limitations_identifiable_claims.tsv).

The key language choices are:

- H2: native-amplitude spatial concentration is supported; normalized specificity remains unresolved.
- H3: observability is repeatable within this recording, not established across recordings.
- H4/H5: neuron-specific morphology and phenotype-based recovery prediction are not promoted.
- H6: two-frame ICA is treated as nearly equivalent to signed temporal difference.
- H7: coherence and lagged recurrence meet the native C3 metric pattern, but promotion remains held.
- H8: precision is unavailable.

### D4 — Precision objective (resolved)

Precision has been removed from the paper's aims. The original labels are credited as user-reported labels from two experts, but the surviving source records only `original_workbook`; it does not preserve separate expert identities or establish that every location in this frozen field was independently reviewed.

The frozen region is `x=[318,510), y=[111,303)` and is intentionally local/enriched. Detector-blind media are available in [bounded_field_media](bounded_field_media/) for an optional single-author qualitative audit. That audit may identify examples or omissions, but it will not produce a precision estimate.

## Work to authorize or defer

These are analysis tasks, not decisions that can be settled by reading the present packet:

1. Run NMS radii 4 and 8 to determine whether compact-lane C3 promotion survives sensitivity.
2. Generate and validate the complete three-section scientific audit before promoting a detector lane.
3. Run nested leave-one-burst-out tensor-rank stability if tensor factors are desired as a claim; otherwise leave them descriptive.

## Submission metadata

Before `manuscript_ready`, supply authorship/order, ethics approval or exemption, funding, conflicts, acknowledgements, data-access language, code archive/identifier, and coauthor approval of the claim matrix.

## How to respond

Fill in [AUTHOR_DECISIONS.yaml](AUTHOR_DECISIONS.yaml), or reply in chat with D1–D4 choices and any claim edits. No decision is assumed from silence.
""")
    atomic_text(target / "CLAIM_LANGUAGE_REVIEW.md", """# Plain-language claim review

You do not need to inspect JSON or verify calculations. For each statement below, mark **approve** if it says what you want the paper to claim, or write replacement wording. The evidence calculations remain source-mapped separately.

## Claims proposed for the paper

1. **Transient activity (H1):** The labeled time intervals contain unusual transient activity in this recording. **Status: provisional support.**
2. **Spatial concentration (H2):** Event amplitude is stronger at labeled centers than matched local controls in native fluorescence units. The normalized spatial-specificity result is unresolved. This is not proof that every signal comes from one neuron. **Status: provisional support with limitation.**
3. **Repeatable observability (H3):** Several site-level amplitude and spatial measures recur across bursts within this recording. We do not claim that this generalizes to other recordings. **Status: provisional support.**
4. **Temporal identity (H4):** The analysis does not establish a reproducible neuron-specific waveform beyond shared tissue activity. **Status: negative/held result.**
5. **Explaining misses (H5):** We cannot yet show that a site's measurement phenotype predicts whether the detector will recover it. **Status: unsupported.**
6. **Two-frame ICA (H6):** The learned two-frame ICA component is nearly equivalent to signed temporal difference and does not establish new source-identifying information. **Status: provisional negative result.**
7. **Compact contextual features (H7):** Coherence and lagged recurrence improve native known-positive recovery, but we will not call them confirmed detector improvements until NMS sensitivity and the scientific audit pass. **Status: promising but held.**
8. **Precision (H8):** Precision is not estimated and has been removed from the paper's aims. The original two-expert positive labels were not preserved as separate exhaustive field reviews, so unlabeled candidates remain unknown rather than false positives. **Status: unavailable by design.**

## Requested response

- `Approve all eight as written`, or
- list the claim numbers you want changed and the intended meaning.

These statements intentionally avoid full-field precision, specificity, false-positive-rate, physical-identifiability, and cross-recording claims.
""")
    return target
