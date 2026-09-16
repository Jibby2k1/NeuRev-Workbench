"""Exact maintained occurrence evaluation with indexed representative lookup.

Only the first-representative spatial lookup differs from the frozen evaluator.
The source evaluator remains unchanged. The optional resume CLI records an
explicit implementation-only backend amendment and requires a bound parity
report before invoking the unchanged study runner. No scientific criterion,
threshold, source stage, annotation, sorting rule or match rule is changed.
"""
from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Any

from . import two_stencil_evaluation as _original

_integer = _original._integer
_proposal_key = _original._proposal_key
MATCH_RADIUS_PX = _original.MATCH_RADIUS_PX
match_peaks_one_to_one = _original.match_peaks_one_to_one
if MATCH_RADIUS_PX != 6.0:
    raise RuntimeError("This exact spatial-hash implementation is bound to the maintained6px radius")


def evaluate_occurrence_windows(
    candidate_rows: Iterable[Mapping[str, Any]],
    positives: Sequence[Mapping[str, Any]],
    *,
    burst_intervals_ui: Mapping[Any, Sequence[int]],
) -> dict[str, Any]:
    """Evaluate frozen frame proposals against broad expert burst occurrences.

    Within each inclusive burst window, proposals are sorted by (-score,y,x,
    source_frame_ui,proposal_id). Each proposal joins the first already accepted
    representative within 6px, otherwise it becomes a new representative. This
    greedy spatial grouping is nontransitive and never averages coordinates.
    Score-ranked representatives are matched to the nearest unassigned expert
    within 6px, with observation-id tie breaking. This measures occurrence-window
    site recall, not frame recall, onset latency, or a count of biological events.
    """
    windows: dict[int, tuple[int, int]] = {}
    for key, bounds in burst_intervals_ui.items():
        burst = _integer(key, "burst_id", minimum=1)
        if burst in windows or len(bounds) != 2:
            raise ValueError("burst windows require unique ids and two inclusive bounds")
        start, stop = (_integer(x, "burst frame bound", minimum=1) for x in bounds)
        if stop < start:
            raise ValueError("burst stop must not precede its start")
        windows[burst] = (start, stop)
    if not windows:
        raise ValueError("at least one burst window is required")
    ordered_windows = sorted(windows.items(), key=lambda item: item[1])
    if any(right[1][0] <= left[1][1] for left, right in zip(ordered_windows, ordered_windows[1:])):
        raise ValueError("burst windows must not overlap")
    labels_by_burst: dict[int, list[dict[str, Any]]] = {b: [] for b in windows}
    observation_ids: set[str] = set()
    for positive in positives:
        row = dict(positive)
        burst = _integer(row["burst_id"], "positive burst_id", minimum=1)
        observation = str(row["observation_id"])
        if burst not in windows or not observation or observation in observation_ids:
            raise ValueError("positive requires a declared burst and unique observation_id")
        for coordinate in ("x_px", "y_px"):
            value = float(row[coordinate])
            if not math.isfinite(value) or value < 0:
                raise ValueError("positive coordinates must be finite and nonnegative")
            row[coordinate] = value
        row.update(burst_id=burst, observation_id=observation)
        observation_ids.add(observation)
        labels_by_burst[burst].append(row)
    groups: dict[int, list[dict[str, Any]]] = {b: [] for b in windows}
    proposal_ids: set[str] = set()
    dimensions: dict[str, set[Any]] = {key: set() for key in (
        "cell_id", "target_proposals_per_frame", "variant_id", "representation", "context_id"
    )}
    total_rows = 0
    for candidate in candidate_rows:
        row = dict(candidate)
        identifier = str(row["proposal_id"])
        if not identifier or identifier in proposal_ids:
            raise ValueError("proposal_id must be unique within the frozen stream")
        proposal_ids.add(identifier)
        row["proposal_id"] = identifier
        for name in ("source_frame_ui", "x_px", "y_px"):
            row[name] = _integer(row[name], name, minimum=1 if name == "source_frame_ui" else 0)
        row["score"] = float(row["score"])
        if not math.isfinite(row["score"]):
            raise ValueError("proposal score must be finite")
        for key, seen in dimensions.items():
            if key in row:
                seen.add(row[key])
                if len(seen) > 1:
                    raise ValueError(f"evaluate one frozen operating point at a time: mixed {key}")
        total_rows += 1
        for burst, (start, stop) in ordered_windows:
            if start <= row["source_frame_ui"] <= stop:
                groups[burst].append(row)
                break
    occurrence_rows: list[dict[str, Any]] = []
    site_rows: list[dict[str, Any]] = []
    membership_rows: list[dict[str, Any]] = []
    burst_summaries: list[dict[str, Any]] = []
    for burst in sorted(windows):
        representatives: list[dict[str, Any]] = []
        memberships: list[list[dict[str, Any]]] = []
        buckets: dict[tuple[int, int], list[int]] = {}
        for row in sorted(groups[burst], key=_proposal_key):
            bucket_x, bucket_y = row["x_px"] // 6, row["y_px"] // 6
            # Only lookup is accelerated. The minimum original representative
            # index, rather than nearest distance or bucket order, preserves the
            # maintained first-representative rule exactly.
            site_index = min((
                index
                for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                for index in buckets.get((bucket_x + dx, bucket_y + dy), ())
                if (row["x_px"] - representatives[index]["x_px"]) ** 2
                + (row["y_px"] - representatives[index]["y_px"]) ** 2 <= MATCH_RADIUS_PX ** 2
            ), default=None)
            if site_index is None:
                site_index = len(representatives)
                representatives.append(row)
                memberships.append([])
                buckets.setdefault((bucket_x, bucket_y), []).append(site_index)
            memberships[site_index].append(row)
        burst_sites = []
        for rank, (representative, members) in enumerate(zip(representatives, memberships), start=1):
            site_id = f"burst_{burst}__site_{rank:05d}"
            site = {
                **representative, "burst_id": burst, "site_id": site_id,
                "site_rank": rank,
                "representative_proposal_id": representative["proposal_id"],
                "member_proposal_count": len(members),
                "member_source_frame_count": len({row["source_frame_ui"] for row in members}),
                "first_member_source_frame_ui": min(row["source_frame_ui"] for row in members),
                "last_member_source_frame_ui": max(row["source_frame_ui"] for row in members),
                "site_interpretation": "spatial_representative_not_unique_event",
            }
            burst_sites.append(site)
            membership_rows.extend({
                "burst_id": burst, "site_id": site_id,
                "proposal_id": row["proposal_id"],
                "source_frame_ui": row["source_frame_ui"],
                "distance_to_representative_px": math.hypot(row["x_px"] - site["x_px"], row["y_px"] - site["y_px"]),
                "is_representative": row["proposal_id"] == site["representative_proposal_id"],
            } for row in members)
        labels = sorted(labels_by_burst[burst], key=lambda row: row["observation_id"])
        peaks = [(row["score"], row["x_px"], row["y_px"]) for row in burst_sites]
        matches, matched_peak_indices = match_peaks_one_to_one(peaks, labels, MATCH_RADIUS_PX)
        by_label = {
            match[0]: (burst_sites[peak_index], match[4])
            for match, peak_index in zip(matches, sorted(matched_peak_indices))
        }
        for label_index, positive in enumerate(labels):
            assigned = by_label.get(label_index)
            nearest = min(burst_sites, key=lambda row: (
                math.hypot(row["x_px"] - positive["x_px"], row["y_px"] - positive["y_px"]),
                row["site_rank"], row["site_id"]), default=None)
            occurrence_rows.append({
                **positive,
                "window_start_frame_ui": windows[burst][0],
                "window_stop_frame_ui": windows[burst][1],
                "matched": assigned is not None,
                "matched_site_id": None if assigned is None else assigned[0]["site_id"],
                "matched_proposal_id": None if assigned is None else assigned[0]["representative_proposal_id"],
                "matched_source_frame_ui": None if assigned is None else assigned[0]["source_frame_ui"],
                "match_distance_px": None if assigned is None else assigned[1],
                "nearest_site_id": None if nearest is None else nearest["site_id"],
                "nearest_site_distance_px": None if nearest is None else math.hypot(nearest["x_px"] - positive["x_px"], nearest["y_px"] - positive["y_px"]),
                "recall_unit": "known_positive_occurrence_within_inclusive_burst_window",
                "onset_latency_identified": False,
            })
        site_rows.extend(burst_sites)
        burst_summaries.append({
            "burst_id": burst, "window_start_frame_ui": windows[burst][0],
            "window_stop_frame_ui": windows[burst][1],
            "emitted_proposal_count_in_window": len(groups[burst]),
            "spatial_representative_count": len(burst_sites),
            "known_positive_count": len(labels), "matched_known_positive_count": len(matches),
            "known_positive_recall": len(matches) / len(labels) if labels else None,
        })
    matched = sum(bool(row["matched"]) for row in occurrence_rows)
    return {
        "occurrence_rows": occurrence_rows, "site_rows": site_rows,
        "membership_rows": membership_rows, "burst_summaries": burst_summaries,
        "summary": {
            "emitted_frame_proposal_count": total_rows,
            "emitted_proposal_count_in_declared_windows": sum(len(rows) for rows in groups.values()),
            "spatial_representative_count_in_declared_windows": len(site_rows),
            "known_positive_count": len(occurrence_rows),
            "matched_known_positive_count": matched,
            "known_positive_occurrence_window_recall": matched / len(occurrence_rows) if occurrence_rows else None,
            "site_consolidation": "score_y_x_frame_id_order_first_representative_within_6px_nontransitive",
            "matching": "score_ranked_representative_to_nearest_unassigned_expert_within_6px_observation_id_ties",
            "evaluation_scope": "descriptive_within_recording_frozen_frame_proposals",
            "precision_identified": False, "onset_latency_identified": False,
            "unique_event_count_identified": False,
            "unmatched_candidates": "unknown_not_negative",
        },
    }


def _binding(path: Path) -> dict[str, Any]:
    path = path.resolve()
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(block)
    return {"path": str(path), "sha256": value.hexdigest(), "size_bytes": path.stat().st_size}


def _json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _first_candidate_frames(path: Path, frame_count: int = 2, maximum_bytes: int = 8 * 1024**2) -> list[dict]:
    """Bounded streaming read of a flat candidate JSON array, never a movie."""
    decoder = json.JSONDecoder()
    buffer = ""
    total = 0
    result = []
    frames: set[int] = set()
    with path.open() as handle:
        while True:
            buffer = buffer.lstrip(" \t\r\n,[")
            if buffer.startswith("]"):
                return result
            try:
                value, end = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                chunk = handle.read(65536)
                total += len(chunk.encode())
                if total > maximum_bytes:
                    raise ValueError("Bounded real-candidate parity read exceeded8MiB")
                if not chunk:
                    if not buffer.strip():
                        return result
                    raise ValueError("Incomplete candidate JSON array")
                buffer += chunk
                continue
            buffer = buffer[end:]
            if not isinstance(value, dict):
                raise ValueError("Expected flat candidate objects")
            frame = int(value["source_frame_ui"])
            if frame not in frames and len(frames) == frame_count:
                return result
            frames.add(frame)
            result.append(value)


def parity_report(output: Path, *, real_prefix: Path | None = None) -> dict[str, Any]:
    """Full dictionary and canonical JSON equality on bounded technical cases."""
    if output.exists():
        raise FileExistsError(output)
    cases = []
    for seed in (11, 73, 20260914):
        rng = random.Random(seed)
        candidates = [dict(proposal_id=f"p{i}", source_frame_ui=rng.randrange(1, 21),
                           score=rng.randrange(-4, 13) / 2,
                           x_px=rng.randrange(160), y_px=rng.randrange(120)) for i in range(700)]
        labels = [dict(observation_id=f"e{i}", burst_id=1 if i < 12 else 2,
                       canonical_roi_id=f"r{i}", x_px=rng.random()*160,
                       y_px=rng.random()*120) for i in range(24)]
        cases.append((f"seed{seed}", candidates, labels, {1: (2, 8), 2: (11, 19)}))
    # Earliest representative wins even when another representative is nearer.
    boundary = [dict(proposal_id=f"b{i}", source_frame_ui=1, score=10-i, x_px=x, y_px=y)
                for i, (x, y) in enumerate(((0, 0), (8, 0), (6, 0), (12, 0),
                                             (6, 6), (6, 12), (0, 6), (0, 0)))]
    cases.append(("bucket_boundaries_nontransitive_duplicates", boundary,
                  [dict(observation_id="e", burst_id=1, canonical_roi_id="r", x_px=6., y_px=0.)], {1: (1, 1)}))
    if real_prefix is not None:
        actual = _first_candidate_frames(real_prefix)
        if not actual:
            raise ValueError("No real candidate rows in bounded prefix sample")
        # Technical geometry parity only. These declared test windows do NOT
        # claim fluorescence truth in the first application frames.
        first, last = min(r["source_frame_ui"] for r in actual), max(r["source_frame_ui"] for r in actual)
        cases.append(("real_first_two_candidate_frames_geometry_only", actual, [], {1: (first, last)}))
    checks = []
    for name, candidates, positives, windows in cases:
        start = time.perf_counter()
        old = _original.evaluate_occurrence_windows(candidates, positives, burst_intervals_ui=windows)
        old_seconds = time.perf_counter() - start
        start = time.perf_counter()
        new = evaluate_occurrence_windows(candidates, positives, burst_intervals_ui=windows)
        new_seconds = time.perf_counter() - start
        if old != new or _canonical(old) != _canonical(new):
            raise RuntimeError(f"Exact occurrence-evaluator parity failed: {name}")
        checks.append(dict(case=name, candidate_count=len(candidates), known_positive_count=len(positives),
                           full_dictionary_equal=True, canonical_JSON_byte_equal=True,
                           complete_output_sha256=hashlib.sha256(_canonical(old)).hexdigest(),
                           original_seconds=old_seconds, indexed_seconds=new_seconds,
                           observed_speed_ratio=old_seconds/new_seconds))
    report = {"status": "PASS", "scope": "implementation parity only; no new scientific metric",
              "backend": _binding(Path(__file__)), "original_evaluator": _binding(Path(_original.__file__)),
              "changes": "only first-representative lookup:6px buckets,9neighbors,inclusive squared distance,min original index",
              "checks": checks, "real_prefix_sample": None if real_prefix is None else {
                  "path": str(real_prefix.resolve()), "maximum_bytes_read": 8*1024**2,
                  "sample_candidate_core_and_metadata_sha256": hashlib.sha256(_canonical(cases[-1][1])).hexdigest(),
                  "sample_is_scientific_evaluation": False}}
    output.parent.mkdir(parents=True, exist_ok=True)
    _json(output, report)
    return report


def resume_evaluation(root: Path, *, parity_path: Path) -> None:
    """Explicit backend amendment, leaving frozen runner and prior metrics intact."""
    from . import spatiotemporal_study as study
    root = root.resolve()
    parity = json.loads(parity_path.read_text())
    if parity["status"] != "PASS" or not parity["checks"] or any(
        not check["full_dictionary_equal"] or not check["canonical_JSON_byte_equal"]
        for check in parity["checks"]
    ):
        raise ValueError("Require a completed exact parity report")
    for record in (parity["backend"], parity["original_evaluator"]):
        if _binding(Path(record["path"])) != record:
            raise ValueError("Parity-checked implementation changed")
    study.load_protocol(root)
    folders = sorted((root / "datasets").iterdir())
    expected = [root / "cells" / folder.name / spec.spec_id / "sealed.json"
                for folder in folders for spec in study.spec_grid()]
    if len(folders) != 17 or len(expected) != 357 or not all(path.is_file() for path in expected):
        raise RuntimeError("Require all357 candidate seals before backend resume")
    provenance_path = root / "evaluation_backend_spatial_hash_v1.json"
    bindings = {"backend": _binding(Path(__file__)), "frozen_runner": _binding(Path(study.__file__)),
                "original_evaluator": _binding(Path(_original.__file__)), "parity_report": _binding(parity_path)}
    if provenance_path.exists():
        provenance = json.loads(provenance_path.read_text())
        if provenance["implementation_bindings"] != bindings:
            raise ValueError("Existing backend amendment has different implementation or parity bindings")
    else:
        completed = sorted((root / "cells").glob("*/*/evaluated.json"))
        previous = [_binding(path) for marker in completed for path in (
            marker, marker.with_name("curves.json"), marker.with_name("operating_metrics.json"))]
        provenance = {"schema": "gamma_st_exact_sparse_backend_amendment_v1",
                      "implementation_bindings": bindings, "all357candidate_seals_present": True,
                      "previously_evaluated_cell_count": len(completed), "prior_numeric_bindings": previous,
                      "change": parity["changes"], "scientific_criteria_changed": False,
                      "operator_or_threshold_rerun": False,
                      "runtime_override": "spatiotemporal_study.evaluate_occurrence_windows only"}
        _json(provenance_path, provenance)
    for record in provenance["prior_numeric_bindings"]:
        if _binding(Path(record["path"])) != record:
            raise ValueError("Previously completed numeric output changed")
    previous = study.evaluate_occurrence_windows
    study.evaluate_occurrence_windows = evaluate_occurrence_windows
    try:
        study.evaluate(root)
    finally:
        study.evaluate_occurrence_windows = previous
    for record in provenance["prior_numeric_bindings"]:
        if _binding(Path(record["path"])) != record:
            raise ValueError("Backend resume changed a previously completed numeric output")
    _json(root / "evaluation_backend_spatial_hash_v1_complete.json", {
        "status": "PASS", "amendment": _binding(provenance_path),
        "evaluation_complete": _binding(root / "evaluation_complete.json"),
        "prior_numeric_outputs_unchanged": True, "scientific_criteria_changed": False})


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("parity")
    check.add_argument("--output", type=Path, required=True)
    check.add_argument("--real-prefix", type=Path)
    resume = commands.add_parser("resume")
    resume.add_argument("--root", type=Path, required=True)
    resume.add_argument("--parity-report", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "parity":
        print(json.dumps(parity_report(args.output, real_prefix=args.real_prefix)), flush=True)
    else:
        resume_evaluation(args.root, parity_path=args.parity_report)


if __name__ == "__main__":
    main()
