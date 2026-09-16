"""Index frozen success/miss evidence without loading arrays or running a detector.

The card set is an outcome-enriched subset of an existing complete audit.  The
optional framewise join is a separately versioned, descriptive label join; it
does not change the frozen stream or create biological event identities.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
from typing import Any, Iterable


FIXED_NAME = "spon_ca_burst_gamma_ls_fixed_deployment_characterization_v1_20260908_r1"
AUDIT_NAME = "spon_ca_burst_gamma_ls_fixed_deployment_scientific_audit_v1_20260908_r2"
REPLAY_NAME = "spon_ca_burst_gamma_ls_full_recording_q1_stage_replay_v1_20260909_r1"
MODEL_AUDIT_NAME = "spon_ca_burst_gamma_ls_full_recording_q1_model_only_scientific_audit_v1_20260909_r1"
CONTEXT_ID = "support_support_a_h15_g7_n9_m0p5"
TEMPORAL_SEMANTICS = "configured_burst_window_not_per_roi_onset"
CARD_SELECTION_RULE = (
    "all matched occurrences, plus two misses per burst; misses prioritize an "
    "identity matched in another burst, then canonical_roi_id and observation_id"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter="\t" if path.suffix == ".tsv" else ","))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def write_rows(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def is_true(value: Any) -> bool:
    return value is True or str(value).lower() == "true"


def unique_by(rows: Iterable[dict[str, Any]], field: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row[field])
        if key in result:
            raise ValueError(f"Duplicate {field}: {key}")
        result[key] = row
    return result


def select_cards(outcomes: list[dict[str, Any]], misses_per_burst: int = 2) -> list[dict[str, Any]]:
    """Freeze the subset using old outcomes, never a new operator's scores."""
    if misses_per_burst < 1:
        raise ValueError("misses_per_burst must be positive")
    unique_by(outcomes, "observation_id")
    successes = [r for r in outcomes if is_true(r["matched"])]
    successful_ids = {r["canonical_roi_id"] for r in successes}
    selected = [dict(r, pilot_selection_reason="all_existing_matches") for r in successes]
    for burst in sorted({int(r["burst_id"]) for r in outcomes}):
        misses = [r for r in outcomes if int(r["burst_id"]) == burst and not is_true(r["matched"])]
        misses.sort(key=lambda r: (r["canonical_roi_id"] not in successful_ids,
                                   r["canonical_roi_id"], r["observation_id"]))
        if len(misses) < misses_per_burst:
            raise ValueError(f"Burst {burst} has fewer than {misses_per_burst} misses")
        selected.extend(dict(r, pilot_selection_reason="deterministic_miss_stratum")
                        for r in misses[:misses_per_burst])
    return sorted(selected, key=lambda r: (int(r["burst_id"]), r["observation_id"]))


class Sources:
    """Hash small consumed files and verify them against existing artifact seals."""

    def __init__(self) -> None:
        self.records: dict[str, dict[str, Any]] = {}
        self.indexes: dict[Path, dict[str, dict[str, Any]]] = {}

    def attach(self, root: Path, relative: str) -> Path:
        root = root.resolve()
        path = root / relative
        if path.suffix in {".npy", ".npz", ".mp4", ".tif"}:
            raise ValueError("Array and video loading is outside this index-only workflow")
        if path.stat().st_size > 10 * 1024 * 1024:
            raise ValueError(f"Source exceeds the small-file bound: {path}")
        if root not in self.indexes:
            index_path = root / "artifact_index.json"
            index = json.loads(index_path.read_text())
            self.indexes[root] = {a["path"]: a for a in index["artifacts"]}
            self.records[str(index_path)] = {
                "path": str(index_path), "sha256": sha256(index_path),
                "bytes": index_path.stat().st_size, "verification": "index_file_hashed",
            }
        digest = sha256(path)
        expected = self.indexes[root].get(relative)
        if relative != "artifact_index.json" and expected is None:
            raise ValueError(f"Consumed source missing from artifact index: {path}")
        if expected is not None and digest != expected["sha256"]:
            raise ValueError(f"Source SHA-256 mismatch: {path}")
        self.records[str(path)] = {
            "path": str(path), "sha256": digest, "bytes": path.stat().st_size,
            "verification": "matched_source_artifact_index" if expected else "index_file_hashed",
        }
        return path


def local_link(output: Path, target: Path) -> str:
    return html.escape(os.path.relpath(target, output), quote=True)


def validate_fixed_outcomes(outcomes: list[dict[str, str]], experts: list[dict[str, str]]) -> None:
    if len(outcomes) != 79 or len(experts) != 79:
        raise ValueError("Expected exactly 79 original expert occurrences and fixed-state outcomes")
    by_id = unique_by(experts, "observation_id")
    unique_by(outcomes, "observation_id")
    successes = [r for r in outcomes if is_true(r["matched"])]
    if len(successes) != 6 or len({r["canonical_roi_id"] for r in successes}) != 3:
        raise ValueError("Expected six original successes belonging to three identities")
    for row in outcomes:
        if (row["representation"] != "difference_signed" or row["quiet_swap"] != "a_train_b_test"
                or float(row["target_nms_peaks_per_pseudo_burst"]) != 1
                or float(row["nms_distance_px"]) != 6 or int(row["candidate_budget"]) != 58):
            raise ValueError("Mixed frozen operating points in the outcome table")
        expert = by_id[row["observation_id"]]
        if expert["temporal_extent_semantics"] != TEMPORAL_SEMANTICS:
            raise ValueError("Unexpected label timing semantics")
        for field in ("canonical_roi_id", "burst_id", "x_px", "y_px"):
            if row[field] != expert[field]:
                raise ValueError(f"Expert/outcome disagreement: {field}")


def build_cards(fixed: Path, audit: Path, output: Path, sources: Sources) -> dict[str, Any]:
    """Copy fourteen existing comparison figures and attach complete audit evidence."""
    controls: dict[str, Any] = {}
    for name in ("llm_context.json", "summary.json", "validation.json", "inventory.json"):
        controls[name] = json.loads(sources.attach(audit, name).read_text())
    if not controls["validation.json"].get("scientific_audit_complete"):
        raise ValueError("The source full audit is not complete")
    for name in ("frame_index.tsv", "stage_array_manifest.json", "audit_inputs/stage_sources.json",
                 "occupancy_map_index.tsv", "candidate_score_seal.json"):
        sources.attach(fixed, name)
    experts = read_rows(sources.attach(fixed, "audit_inputs/expert_occurrences.tsv"))
    outcomes = read_rows(sources.attach(fixed, "audit_inputs/one_to_one_matches.tsv"))
    validate_fixed_outcomes(outcomes, experts)
    selected = select_cards(outcomes)
    if len(selected) != 14:
        raise ValueError("Expected fourteen selected cards")
    expert_by_id = unique_by(experts, "observation_id")
    nearest_by_id = unique_by(read_rows(sources.attach(
        audit, "3_Comparison/nearest_roi_trace_metrics.csv")), "occurrence_id")
    sources.attach(audit, "3_Comparison/expert_model_matches.csv")
    model_rows = read_rows(sources.attach(audit, "2_Model_Annotations/model_occurrences.csv"))
    model_by_id = unique_by(model_rows, "candidate_id")
    candidates = read_rows(sources.attach(fixed, "candidates_label_sealed.tsv"))
    candidate_by_id = unique_by(candidates, "candidate_id")
    thresholds = read_rows(sources.attach(fixed, "threshold_calibration.tsv"))
    threshold_rows = [r for r in thresholds if r["quiet_swap"] == "a_train_b_test"
                      and float(r["target_nms_peaks_per_pseudo_burst"]) == 1]
    if len(threshold_rows) != 1 or threshold_rows[0]["context_id"] != CONTEXT_ID:
        raise ValueError("Frozen threshold/context not identified uniquely")
    threshold = threshold_rows[0]
    frame_rows = read_rows(fixed / "frame_index.tsv")
    frame_lookup = {int(r["review_frame_zero"]): int(r["source_frame_ui"]) for r in frame_rows}
    for row in frame_rows:
        if int(row["source_frame_zero"]) + 1 != int(row["source_frame_ui"]):
            raise ValueError("Broken one-based source-frame mapping")
    full_coverage = {
        "audit_root": str(audit.resolve()),
        "expert_occurrences": controls["summary.json"]["expert_occurrence_count"],
        "expert_identities": controls["summary.json"]["expert_roi_count"],
        "original_audit_complete": True,
        "binding": "all small source manifests and selected PNGs verified against artifact index",
        "array_and_video_hashes": "declared in attached source indexes; not rehashed or decoded here",
        "original_operating_point": controls["summary.json"]["operating_point"],
    }
    write_json(output / "attached_full_audit.json", full_coverage)
    cards = []
    for row in selected:
        observation = row["observation_id"]
        expert = expert_by_id[observation]
        nearest = nearest_by_id[observation]
        if is_true(nearest["one_to_one_match"]) != is_true(row["matched"]):
            raise ValueError("Nearest-candidate table disagrees with original one-to-one outcome")
        candidate_id = nearest["nearest_candidate_id"]
        candidate = candidate_by_id.get(candidate_id)
        if candidate:
            if (candidate["context_id"] != CONTEXT_ID or candidate["quiet_swap"] != "a_train_b_test"
                    or float(candidate["target_nms_peaks_per_pseudo_burst"]) != 1):
                raise ValueError("Nearest candidate belongs to another operator/readout")
            if frame_lookup[int(candidate["peak_frame_review_zero"])] != int(candidate["peak_source_frame_ui"]):
                raise ValueError("Candidate peak source-frame mapping is inconsistent")
        trace = sources.attach(audit, f"3_Comparison/trace_comparisons/{observation}.png")
        sources.attach(audit, f"1_Expert_Annotations/metadata/roi_{expert['canonical_roi_id']}.json")
        expert_trace = sources.attach(audit, f"1_Expert_Annotations/figures/traces/{expert['canonical_roi_id']}.png")
        copied = output / "cards" / f"{observation}.png"
        copied.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(trace, copied)
        model = model_by_id.get(candidate_id, {})
        metadata = {
            "observation_id": observation, "selection_reason": row["pilot_selection_reason"],
            "outcome": "matched" if is_true(row["matched"]) else "miss",
            "expert": expert, "original_one_to_one_outcome": row,
            "nearest_frozen_candidate_diagnostic": nearest,
            "nearest_candidate_row": candidate, "threshold_calibration": threshold,
            "candidate_peak_threshold_margin_z": (float(candidate["peak_gamma_score_z"]) - float(threshold["threshold_z"])) if candidate else None,
            "expert_threshold_margin_z": None,
            "expert_threshold_margin_status": "not_recomputed_in_index_only_pilot",
            "miss_mechanism": "requires_stage_inspection_and_reviewer_judgment",
            "nms_outcome": "sealed_full_field_burst_occupancy_NMS6; never rerun on crop",
            "exact_stages": ["raw acquired", "conditioned current", "signed adjacent difference", "signed Gamma-LS", "burst occupancy"],
            "source_frame_convention": "UI one-based inclusive; review-array index resolved through frame_index.tsv",
            "timing_claim": "none; best lag is descriptive, no per-ROI onset truth",
            "reference_mean_and_scale": "not persisted; not reconstructed in this pilot",
            "comparison_png": str(copied.relative_to(output)),
            "source_comparison_png": str(trace.resolve()), "source_comparison_png_sha256": sha256(trace),
            "expert_trace": str(expert_trace.resolve()),
            "expert_video": str((audit / "1_Expert_Annotations/videos/closeups" / f"{expert['canonical_roi_id']}.mp4").resolve()),
            "nearest_model_roi_id": model.get("consolidated_model_roi_id"),
            "reviewer_status": "not_reviewed_in_this_run",
            "reviewer_prompts": [
                "Does the raw/conditioned trace show localized activity in the declared burst?",
                "Does temporal differencing attenuate a broad or persistent response?",
                "Could nearby synchronous activity or a moving structure explain the spatial offset?",
                "Is identity association uncertain despite correlated traces?",
                "What evidence, rather than appearance alone, supports a proposed miss mechanism?",
            ],
        }
        write_json(output / "cards" / f"{observation}.json", metadata)
        cards.append(metadata)
    write_json(output / "cards.json", cards)
    write_rows(output / "card_selection.tsv", [{
        "observation_id": c["observation_id"], "canonical_roi_id": c["expert"]["canonical_roi_id"],
        "burst_id": c["expert"]["burst_id"], "outcome": c["outcome"], "selection_reason": c["selection_reason"],
        "source_start_ui": c["expert"]["source_start_ui"], "source_stop_ui": c["expert"]["source_stop_ui"],
        "metadata": f"cards/{c['observation_id']}.json", "comparison_png": c["comparison_png"],
    } for c in cards])
    render_index(cards, output, audit)
    return {
        "card_count": len(cards), "success_card_count": 6, "miss_card_count": 8,
        "success_identity_count": 3, "full_audit_expert_occurrences": 79,
        "full_audit_expert_identity_count": 26, "selection_rule": CARD_SELECTION_RULE,
        "selected_identity_count": len({c["expert"]["canonical_roi_id"] for c in cards}),
        "claim_scope": "outcome_enriched_diagnostic_subset_of_post_selection_within_recording_characterization",
        "new_operator_scored": False, "source_detection_membership_changed": False,
        "new_stage_diagnostics_computed": False, "reviewer_interpretations_recorded": False,
    }


def render_index(cards: list[dict[str, Any]], output: Path, audit: Path) -> None:
    sections = []
    for card in cards:
        obs = card["observation_id"]
        expert, nearest = card["expert"], card["nearest_frozen_candidate_diagnostic"]
        sections.append(f"""<article id="{html.escape(obs)}" data-outcome="{card['outcome']}">
<h2>{html.escape(obs)} · {card['outcome']}</h2>
<p>{html.escape(expert['canonical_roi_id'])}; expert (x,y)=({expert['x_px']}, {expert['y_px']});
burst {expert['burst_id']}, source UI {expert['source_start_ui']}–{expert['source_stop_ui']} inclusive.</p>
<p>Nearest frozen candidate: {html.escape(nearest['nearest_candidate_id'])}; distance {float(nearest['distance_px']):.2f} px;
rank {nearest['candidate_rank']}; one-to-one matched: {nearest['one_to_one_match']}.
Nearest location and assigned identity are separate fields.</p>
<img loading="lazy" src="{card['comparison_png']}" alt="Original audited stage traces for {html.escape(obs)}">
<p><a href="cards/{html.escape(obs)}.json">Complete card metadata</a> ·
<a href="{local_link(output, Path(card['expert_trace']))}">Expert-only full trace</a> ·
<a href="{local_link(output, Path(card['expert_video']))}">Expert-only close-up</a></p>
<p>Review: local activity; attenuation after differencing; nearby activity or motion; identity ambiguity.
Threshold margin at the nearest candidate peak is metadata, not the expert location's threshold margin.
No mechanism has been adjudicated by this index.</p></article>""")
    document = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Gamma-LS success and miss review</title>
<style>body{font:17px/1.5 system-ui,sans-serif;max-width:1100px;margin:auto;padding:24px;color:#202124;background:#fafafa}h1{line-height:1.15}article{background:white;border:1px solid #ddd;padding:20px;margin:24px 0;border-radius:8px}img{width:100%;height:auto}a{color:#174ea6}button{padding:8px 16px;margin:4px}small{color:#555}</style>
<h1>Gamma-LS: 14 existing success and miss cases</h1>
<p>Six matched occurrences from three identities, plus two misses per burst. This outcome-enriched subset is for diagnostic review; it is not a performance or prevalence estimate.</p>
<p>Frozen state: signed difference · h15/g7/n9/mode7.5 · q1 a_train_b_test · burst occupancy · NMS6 · B58.
The complete existing 79-occurrence / 26-identity audit remains attached. This page reuses its figures and adds no scoring.</p>
<p>Burst intervals are broad annotations, not exact neural onsets. Lag and peak frame do not measure biological onset delay. Unmatched candidates remain unknown.</p>
<p>Display the original exact stages and fixed scales. Separate nearest-candidate evidence from one-to-one identity assignment. Reference mean/scale and expert threshold-margin diagnostics were not reconstructed.</p>
<p><a href="source_manifest.json">Source hashes</a> · <a href="cards.json">All card metadata</a> · <a href="card_selection.tsv">Selection table</a> ·
""" + f'<a href="{local_link(output, audit / "REPORT.md")}">Complete source audit</a></p>' + """
<nav><button onclick="show('all')">All 14</button><button onclick="show('matched')">6 matches</button><button onclick="show('miss')">8 misses</button></nav>
""" + "\n".join(sections) + """<script>function show(mode){document.querySelectorAll('article').forEach(x=>x.hidden=mode!=='all'&&x.dataset.outcome!==mode)}</script></html>"""
    (output / "index.html").write_text(document)


def run_framewise_join(base: Path, output: Path, sources: Sources) -> dict[str, Any]:
    """Attach sparse labels to an unchanged stream through the shared site rule.

    This table-level descriptive endpoint has its own version and never
    substitutes for the existing protected or fixed-burst-head metric.
    """
    from .two_stencil_evaluation import evaluate_occurrence_windows

    fixed, replay = base / FIXED_NAME, base / REPLAY_NAME
    replay_summary = json.loads(sources.attach(replay, "summary.json").read_text())
    replay_validation = json.loads(sources.attach(replay, "validation.json").read_text())
    if not replay_validation.get("all_checks_pass"):
        raise ValueError("Frozen operational replay did not pass exact parity checks")
    for name in ("stage_manifest.json", "stage_frame_index.tsv", "ledger_reconciliation.json"):
        sources.attach(replay, name)
    model_audit = base / MODEL_AUDIT_NAME
    for name in ("llm_context.json", "summary.json", "validation.json"):
        sources.attach(model_audit, name)
    proposal_source = sources.attach(replay, "q1_frame_level_proposals_exact_replay.tsv")
    proposals = read_rows(proposal_source)
    experts = read_rows(sources.attach(fixed, "audit_inputs/expert_occurrences.tsv"))
    if len(proposals) != 371 or len(experts) != 79:
        raise ValueError("Expected the frozen 371-row stream and original 79 occurrences")
    unique_by(proposals, "proposal_id")
    unique_by(experts, "observation_id")
    intervals: dict[int, tuple[int, int]] = {}
    for expert in experts:
        burst = int(expert["burst_id"])
        interval = (int(expert["source_start_ui"]), int(expert["source_stop_ui"]))
        if expert["temporal_extent_semantics"] != TEMPORAL_SEMANTICS:
            raise ValueError("Unexpected annotation time semantics")
        if burst in intervals and intervals[burst] != interval:
            raise ValueError("This endpoint requires the same declared interval within each burst")
        intervals[burst] = interval
    if set(intervals) != {1, 2, 3, 4}:
        raise ValueError("Expected all four declared burst windows")
    for row in proposals:
        if (row["context_id"] != CONTEXT_ID or row["representation"] != "difference_signed"
                or float(row["target_nms_peaks_per_calibration_unit"]) != 1
                or row["calibration_burden_unit"] != "nms_peaks_per_nonoverlapping_1s_initialization_block"
                or is_true(row["temporal_linking_applied"])):
            raise ValueError("Operational source is not the frozen frame-level q1 stream")
        if not 101 <= int(row["source_frame_ui"]) <= 2359:
            raise ValueError("Proposal outside the declared application interval")
    result = evaluate_occurrence_windows(proposals, experts, burst_intervals_ui=intervals)
    join_root = output / "framewise_descriptive_join"
    join_root.mkdir()
    shutil.copyfile(proposal_source, join_root / "frozen_frame_level_proposals_371.tsv")
    if sha256(proposal_source) != sha256(join_root / "frozen_frame_level_proposals_371.tsv"):
        raise ValueError("Copied frozen frame stream changed")
    write_json(join_root / "result.json", result)
    for key in ("occurrence_rows", "site_rows", "membership_rows", "burst_summaries"):
        rows = result[key]
        if rows:
            write_rows(join_root / f"{key}.tsv", rows)
        else:
            write_rows(join_root / f"{key}.tsv", [], fields=["no_rows"])
    contract = {
        "endpoint_version": "descriptive_frozen_frame_stream_burst_sites_v1",
        "scope": "new_descriptive_label_join_of_frozen_stream_not_inherited_existing_metric",
        "burst_intervals_ui_inclusive": intervals,
        "site_rule": "within each burst, sort (-score,y,x,source_frame_ui,proposal_id); assign to first retained representative within <=6px or create site; nontransitive",
        "representative_rule": "highest-ranked actual emitted row; no averaging or location recentering",
        "assignment_rule": "score-ranked sites greedily match nearest unassigned positive <=6px; distance ties by observation_id",
        "site_interpretation": "spatial_review_site_not_unique_biological_event_or_neuron",
        "unmatched_candidates": "unknown_not_negative",
        "temporal_scope": TEMPORAL_SEMANTICS,
        "onset_latency_identified": False,
        "new_detector_scores_computed": False,
        "new_source_masks_or_thresholds_fitted": False,
        "all_frozen_frame_rows_preserved": 371,
        "source_frame_stream_sha256": sha256(proposal_source),
        "source_expert_full_audit": str((base / AUDIT_NAME).resolve()),
        "source_operational_model_only_full_audit": str(model_audit.resolve()),
        "new_join_full_scientific_audit_complete": False,
        "audit_boundary": "metadata/table join complete; newly assigned site-specific matched trace/media audit has not been rendered",
        "source_replay_summary": replay_summary,
    }
    write_json(join_root / "contract.json", contract)
    summary = dict(result["summary"], endpoint_version=contract["endpoint_version"],
                   all_frozen_frame_rows_preserved=371,
                   new_join_full_scientific_audit_complete=False,
                   interpretation="new descriptive site endpoint; no event identity, precision, or onset claim")
    write_json(join_root / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_base = Path(__file__).resolve().parents[3] / "Outputs/GammaLSDifference"
    parser.add_argument("--base", type=Path, default=default_base)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing output root: {output}")
    output.mkdir(parents=True)
    write_json(output / "status.json", {"status": "running_index_only", "gpu_used": False})
    sources = Sources()
    summary = build_cards(args.base / FIXED_NAME, args.base / AUDIT_NAME, output, sources)
    summary["framewise_descriptive_join"] = run_framewise_join(args.base, output, sources)
    write_json(output / "source_manifest.json", {
        "consumed_sources": list(sources.records.values()),
        "implementation_path": str(Path(__file__).resolve()), "implementation_sha256": sha256(Path(__file__)),
        "arrays_loaded": False, "videos_decoded": False,
    })
    write_json(output / "summary.json", dict(summary, status="complete_subset_index"))
    write_json(output / "validation.json", {
        "status": "passed_subset_index_contract", "fourteen_unique_cards": True,
        "source_hashes_verified": True, "source_full_audit_attached": True,
        "new_full_scientific_audit_claimed": False, "new_operator_evaluation_claimed": False,
    })
    write_json(output / "llm_context.json", {
        "entrypoint": "summary.json", "review_page": "index.html", "primary_tables": ["card_selection.tsv", "cards.json"],
        "source_manifest": "source_manifest.json", "attached_full_audit": "attached_full_audit.json",
        "separate_descriptive_endpoint": "framewise_descriptive_join/summary.json",
        "limitations": ["outcome-enriched subset", "no new operator scores", "no new biological adjudication", "no exact onset truth"],
    })
    (output / "REPORT.md").write_text(
        "# Frozen Gamma-LS diagnostic review index\n\n"
        "This 14-card index reuses the complete existing fixed-context audit: all six matched occurrences "
        "(three identities), plus two deterministically chosen misses per burst. It changes no detector output "
        "and computes no new score or stage attenuation. Open index.html for the review cards.\n\n"
        "The attached full audit covers all 79 expert occurrences and 26 identities. The pilot selection is "
        "outcome-enriched and cannot estimate recall, precision, or prevalence. Source indexes and selected "
        "PNG bytes are hash-verified; arrays and videos are referenced through their original manifests. "
        "Unmatched candidates are unknown. Configured burst windows do not supply per-neuron onset truth.\n\n"
        "The separate framewise_descriptive_join directory contains a NEW descriptive site endpoint, "
        "using the shared deterministic 6-pixel burst-site consolidation and one-to-one assignment rule. "
        "All 371 original emitted rows are preserved byte for byte. Spatial review sites are not unique "
        "biological events or neurons. The table-level join is complete; a new site-specific matched "
        "trace/media audit has not been rendered, so it is not promoted as a complete new scientific audit.\n"
    )
    write_json(output / "status.json", {"status": "complete_subset_index", "gpu_used": False})
    write_json(output / "artifact_index.json", {"artifacts": [
        {"path": str(p.relative_to(output)), "sha256": sha256(p), "size_bytes": p.stat().st_size}
        for p in sorted(output.rglob("*")) if p.is_file() and p.name != "artifact_index.json"
    ]})
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
