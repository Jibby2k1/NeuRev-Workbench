"""Descriptive q=1 stage observations from the sealed two-stencil campaign.

No operator, threshold, NMS, or candidate selection is rerun. The categories
locate an observed absence in the saved readout sequence; they do not identify
noise, motion, neural activity, or a causal mechanism. This supplement does
not replace the campaign's complete scientific-audit artifact set.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

import numpy as np


REPO = Path(__file__).resolve().parents[3]
DEFAULT_SELECTION = REPO / "Outputs/GammaLSDifference/two_stencil_diagnostic_review_v1_20260910_r1/card_selection.tsv"
# Frozen pilot artifact_index.json binds exactly these pre-existing 14 cards.
FROZEN_SELECTION_SHA256 = "8d53a5ec1f1f0fe09ae18fd5d049dd2f18d94ccaf7fe22440410b0d26889a86f"
STAGES = ("X", "A", "M", "sigma", "contrast", "Z")
RADIUS_PX = 6.0
CATEGORIES = (
    "matched",
    "no_matching_disk_pixel_above_threshold",
    "above_threshold_without_nearby_frame_proposal",
    "nearby_frame_proposal_without_nearby_burst_representative",
    "nearby_burst_representative_unmatched",
)
CLAIM_BOUNDARY = (
    "Descriptive within-recording q1 readout diagnostics only. Configured burst "
    "windows are not per-ROI onset labels. Categories do not establish causal "
    "noise/motion explanations, biological precision, false-positive rate, "
    "unique events, external generalization, or a selected winner."
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _write_tsv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, keys, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _verify_file(path: Path, expected: str) -> str:
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"sealed file hash mismatch: {path}")
    return actual


def _matched(value: Any) -> bool:
    if value is True or value == "True":
        return True
    if value is False or value == "False":
        return False
    raise ValueError("matched must be a boolean or the exported True/False string")


def classify_occurrence(*, matched: bool, max_local_z: float, threshold_z: float,
                        nearby_frame_proposal_count: int,
                        nearby_representative_count: int) -> str:
    """Classify saved readout observations with a strict score > tau cutoff.

    A matched occurrence takes precedence. For a miss, the first absent saved
    readout element determines the category. Counts use the same floating
    expert center and inclusive 6px distance as the maintained matcher.
    """
    assigned = _matched(matched)
    if not all(math.isfinite(float(v)) for v in (max_local_z, threshold_z)):
        raise ValueError("scores and threshold must be finite")
    for value in (nearby_frame_proposal_count, nearby_representative_count):
        if isinstance(value, bool) or int(value) != value or value < 0:
            raise ValueError("nearby counts must be nonnegative integers")
    if assigned:
        return CATEGORIES[0]
    if max_local_z <= threshold_z:
        return CATEGORIES[1]
    if nearby_frame_proposal_count == 0:
        return CATEGORIES[2]
    if nearby_representative_count == 0:
        return CATEGORIES[3]
    return CATEGORIES[4]


def matching_disk(shape_yx: Sequence[int], x_px: float, y_px: float,
                  radius_px: float = RADIUS_PX) -> tuple[np.ndarray, np.ndarray]:
    """Return row-major integer pixel centers within an unrounded expert disk."""
    if len(shape_yx) != 2 or any(int(v) != v or v <= 0 for v in shape_yx):
        raise ValueError("shape must contain positive integer height and width")
    height, width = map(int, shape_yx)
    x, y, radius = float(x_px), float(y_px), float(radius_px)
    if not all(math.isfinite(v) for v in (x, y, radius)) or radius <= 0:
        raise ValueError("coordinates and positive radius must be finite")
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError("expert center is outside the image")
    yy, xx = np.mgrid[max(0, math.ceil(y-radius)):min(height, math.floor(y+radius)+1),
                      max(0, math.ceil(x-radius)):min(width, math.floor(x+radius)+1)]
    keep = (xx-x)**2 + (yy-y)**2 <= radius**2
    if not keep.any():
        raise ValueError("matching disk contains no valid image pixel centers")
    return yy[keep], xx[keep]


def occurrence_stage_observations(stages: Mapping[str, Any], *, x_px: float,
                                  y_px: float, source_start_ui: int,
                                  source_stop_ui: int, threshold_z: float,
                                  scale_floor: float) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Read only a burst's exact trace pixel and radius-6 Z disk from memmaps.

    The trace pixel uses floor(coord+0.5), clipped at the last valid pixel.
    The signed peak is the largest Z (never abs(Z)); ties use the earliest
    source frame, and disk ties additionally use row-major pixel order.
    """
    if set(stages) != set(STAGES):
        raise ValueError(f"required stage arrays: {STAGES}")
    shape = tuple(stages["Z"].shape)
    if len(shape) != 3 or any(tuple(value.shape) != shape for value in stages.values()):
        raise ValueError("stages must share one TYX shape")
    start, stop = int(source_start_ui), int(source_stop_ui)
    if (start != source_start_ui or stop != source_stop_ui
            or not 1 <= start <= stop <= shape[0]):
        raise ValueError("source bounds must be valid one-based inclusive integers")
    floor, tau = float(scale_floor), float(threshold_z)
    if not math.isfinite(floor) or floor < 0 or not math.isfinite(tau):
        raise ValueError("scale floor must be finite/nonnegative and tau finite")
    ys, xs = matching_disk(shape[1:], x_px, y_px)
    px = min(shape[2]-1, math.floor(float(x_px)+0.5))
    py = min(shape[1]-1, math.floor(float(y_px)+0.5))
    packet = {name: np.asarray(value[start-1:stop, py, px]).copy() for name, value in stages.items()}
    disk_z = np.asarray(stages["Z"][start-1:stop, ys, xs])
    if not all(np.isfinite(value).all() for value in (*packet.values(), disk_z)):
        raise ValueError("nonfinite value in requested stage observations")
    if np.any(packet["sigma"] < 0):
        raise ValueError("reference standard deviation must be nonnegative")
    peak = int(np.argmax(packet["Z"]))
    disk_t, disk_pixel = np.unravel_index(int(np.argmax(disk_z)), disk_z.shape)
    disk_max = disk_z.max(axis=1)
    packet.update(source_frame_ui=np.arange(start, stop+1, dtype=np.int32),
                  local_disk_max_Z=disk_max, threshold_margin=packet["Z"]-tau,
                  local_disk_threshold_margin=disk_max-tau,
                  sigma_floor_active=packet["sigma"] < floor)
    record = {
        "trace_x_px": px, "trace_y_px": py,
        "trace_rounding": "floor_coord_plus_half_clipped_to_last_valid_pixel",
        "source_start_ui": start, "source_stop_ui": stop,
        "frame_count": stop-start+1, "threshold_z": tau, "scale_floor": floor,
        "rounded_pixel_peak_Z_source_frame_ui": start+peak,
        **{f"{name}_at_rounded_pixel_peak_Z": float(packet[name][peak]) for name in STAGES},
        "rounded_pixel_peak_Z_threshold_margin": float(packet["Z"][peak])-tau,
        "sigma_floor_active_at_rounded_pixel_peak_Z": bool(packet["sigma"][peak] < floor),
        "max_local_disk_Z": float(disk_z[disk_t, disk_pixel]),
        "max_local_disk_Z_threshold_margin": float(disk_z[disk_t, disk_pixel])-tau,
        "max_local_disk_Z_source_frame_ui": start+int(disk_t),
        "max_local_disk_Z_x_px": int(xs[disk_pixel]),
        "max_local_disk_Z_y_px": int(ys[disk_pixel]),
        "matching_disk_pixel_count": len(xs),
        "matching_disk_radius_px": RADIUS_PX,
        "matching_disk_center_is_unrounded": True,
        "suprathreshold_matching_disk_pixel_frame_count": int(np.count_nonzero(disk_z > tau)),
        "source_frames_with_suprathreshold_matching_disk_pixels": int(np.count_nonzero(disk_max > tau)),
    }
    return record, packet


def nearby_readout_observations(occurrence: Mapping[str, Any],
                                candidate_rows: Sequence[Mapping[str, Any]],
                                site_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Inspect already emitted full-field proposals and saved representatives."""
    x, y = float(occurrence["x_px"]), float(occurrence["y_px"])
    start, stop = int(occurrence["source_start_ui"]), int(occurrence["source_stop_ui"])
    burst = int(occurrence["burst_id"])
    candidates = [r for r in candidate_rows if start <= int(r["source_frame_ui"]) <= stop]
    sites = [r for r in site_rows if int(r["burst_id"]) == burst]
    distance = lambda row: math.hypot(float(row["x_px"])-x, float(row["y_px"])-y)
    nearby = [r for r in candidates if distance(r) <= RADIUS_PX]
    nearby_sites = [r for r in sites if distance(r) <= RADIUS_PX]
    return {
        "nearby_frame_proposal_count": len(nearby),
        "nearby_frame_proposal_source_frame_count": len({int(r["source_frame_ui"]) for r in nearby}),
        "nearby_representative_count": len(nearby_sites),
        "nearest_emitted_frame_proposal_distance_px": min(map(distance, candidates), default=None),
        "nearest_burst_representative_distance_px": min(map(distance, sites), default=None),
        "nearby_frame_proposal_ids": json.dumps([r["proposal_id"] for r in nearby]),
        "nearby_burst_representative_ids": json.dumps([r["site_id"] for r in nearby_sites]),
    }


def _verify_join_tables(folder: Path, cell_id: str, candidates, labels) -> tuple[list[dict], list[dict]]:
    # Replay grouping/matching only, on the complete sealed global candidate
    # artifact. No score map, cropped NMS, or threshold selection enters here.
    from .two_stencil_evaluation import evaluate_occurrence_windows
    windows = {int(r["burst_id"]): (int(r["source_start_ui"]), int(r["source_stop_ui"])) for r in labels}
    expected = evaluate_occurrence_windows(candidates, labels, burst_intervals_ui=windows)
    for name in ("occurrence_rows", "site_rows", "membership_rows", "burst_summaries"):
        exported = _read_tsv(folder / f"{name}.tsv")
        expected_rows = [{"operator_cell_id": cell_id, "readout": "Z", "target_proposals_per_frame": 1.0, **r}
                         for r in expected[name]]
        strings = [{k: "" if v is None else str(v) for k, v in row.items()} for row in expected_rows]
        if exported != strings:
            raise ValueError(f"saved q1 {name} differs from sealed candidate/label join: {cell_id}")
    summary = _json(folder / "summary.json")
    if any(summary[key] != value for key, value in expected["summary"].items()):
        raise ValueError(f"q1 summary differs from replayed label join: {cell_id}")
    return expected["occurrence_rows"], expected["site_rows"]


def _campaign_gate(root: Path) -> tuple[dict, dict, dict]:
    protocol, preflight = _json(root/"protocol.json"), _json(root/"preflight.json")
    cells = protocol["cells"]
    if len(cells) != 25 or len({r["cell_id"] for r in cells}) != 25:
        raise ValueError("focused diagnostics require the frozen complete 25-cell matrix")
    seal = _json(root/"campaign_candidate_seal.json")
    if set(seal["seals"]) != {r["cell_id"] for r in cells} or seal["labels_used_for_scoring_or_selection"]:
        raise ValueError("all 25 candidate seals must precede label outcome joins")
    for cell in cells:
        folder = root / "cells" / cell["cell_id"]
        complete = _json(folder/"numeric_complete.json")
        if complete["status"] != "NUMERIC_COMPLETE_AUDIT_PENDING":
            raise ValueError("all 25 numeric joins must be complete")
        if complete["label_join_completed_unix"] < seal["all25sealed_unix"]:
            raise ValueError("label join predates complete campaign candidate seal")
        if complete["label_sha256"] != preflight["label_sha256"]:
            raise ValueError("numeric join label hash differs from preflight")
        _verify_file(folder/"candidate_seal.json", seal["seals"][cell["cell_id"]])
        if complete["candidate_seal_sha256"] != seal["seals"][cell["cell_id"]]:
            raise ValueError("numeric and campaign candidate seal hashes differ")
        scoring = _json(folder/"scoring_complete.json")
        if any(complete[k] != scoring[k] for k in ("stage_paths", "stage_shapes", "stage_sha256", "metadata_sha256", "candidate_seal_sha256")):
            raise ValueError("numeric completion differs from sealed scoring metadata")
    return protocol, preflight, seal


def _render_cards(destination: Path, cells, selected, records, packets) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    chosen = [c for c in cells if c["is_deployed_anchor"] or (
        c["spec"]["reference_family"] == "gamma" and c["spec"]["guard_radius_px"] == 0
        and c["spec"]["support_geometry"] == "square")]
    if len(chosen) != 7:
        raise ValueError("the fixed card comparison requires six Gamma g0 cells and the deployed anchor")
    chosen.sort(key=lambda c: (c["is_deployed_anchor"], c["input_representation"],
                              ("point", "direct", "serial").index(c["spec"]["design"])))
    paths = []
    colors = ("#323232", "#4169a6", "#87529c", "#706345")
    for selected_row in selected:
        observation = selected_row["observation_id"]
        figure, axes = plt.subplots(7, 3, figsize=(18, 18), sharex=True, squeeze=False)
        for index, cell in enumerate(chosen):
            cell_id = cell["cell_id"]
            record, trace = records[(cell_id, observation)], packets[(cell_id, observation)]
            frames = trace["source_frame_ui"]
            for stage, color in zip(("X", "A", "M"), colors):
                axes[index, 0].plot(frames, trace[stage], label=stage, color=color, lw=1)
            for stage, color in zip(("contrast", "sigma"), colors[1:]):
                axes[index, 1].plot(frames, trace[stage], label=stage, color=color, lw=1)
            axes[index, 1].axhline(record["scale_floor"], color="#999999", ls=":", label="sigma floor")
            axes[index, 2].plot(frames, trace["Z"], color=colors[0], label="exact trace pixel Z")
            axes[index, 2].plot(frames, trace["local_disk_max_Z"], color=colors[1], ls="--", label="maximum Z in 6px disk")
            axes[index, 2].axhline(record["threshold_z"], color="#999999", ls=":", label="frozen q1 tau")
            axes[index, 2].set_title(record["diagnostic_category"].replace("_", " "), fontsize=8)
            representation_label = "Current" if cell["input_representation"] == "conditioned_current_frame" else "Signed"
            label = "Deployed signed anchor" if cell["is_deployed_anchor"] else f"{representation_label} / {cell['spec']['design']} Gamma g0"
            axes[index, 0].set_ylabel(label, fontsize=8)
            for axis in axes[index]:
                axis.tick_params(labelsize=8)
                axis.grid(alpha=.2)
                axis.legend(fontsize=6, loc="best", framealpha=.6)
        for axis, title in zip(axes[0], ("Native input, target, reference mean", "Native contrast and reference scale", "Z and frozen threshold")):
            axis.text(.5, 1.28, title, transform=axis.transAxes, ha="center", fontsize=10)
        for axis in axes[-1]:
            axis.set_xlabel("source frame UI (inclusive configured burst window)")
        figure.suptitle(f"{observation}: same frozen diagnostic card across seven q1 readouts\n"
                         "Each row uses its own native units and frozen threshold; y axes are independent. No onset or causal attribution.", fontsize=11)
        figure.tight_layout(rect=(0, .01, 1, .96))
        path = destination / "cards" / f"{observation}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=125)
        plt.close(figure)
        paths.append(str(path.relative_to(destination)))
    return paths


def run_focused_diagnostics(campaign_root, *, selection_path=DEFAULT_SELECTION,
                            render: bool = True) -> dict[str, Any]:
    """Write a new focused_diagnostics/ supplement after all 25 label joins."""
    root, selection_path = Path(campaign_root).resolve(), Path(selection_path).resolve()
    destination = root / "focused_diagnostics"
    if destination.exists():
        raise FileExistsError(f"new focused diagnostics directory required: {destination}")
    protocol, preflight, _ = _campaign_gate(root)
    label_path = Path(preflight["label_source"])
    _verify_file(label_path, preflight["label_sha256"])
    labels = _read_tsv(label_path)
    if len(labels) != 79 or len({r["observation_id"] for r in labels}) != 79:
        raise ValueError("expected 79 unique frozen occurrences")
    _verify_file(selection_path, FROZEN_SELECTION_SHA256)
    selected = _read_tsv(selection_path)
    selected_ids = {r["observation_id"] for r in selected}
    if len(selected) != 14 or len(selected_ids) != 14 or not selected_ids <= {r["observation_id"] for r in labels}:
        raise ValueError("expected the 14 existing unique card IDs from frozen labels")
    by_id = {r["observation_id"]: r for r in labels}
    for row in selected:
        if any(row[k] != by_id[row["observation_id"]][k] for k in ("canonical_roi_id", "burst_id", "source_start_ui", "source_stop_ui")):
            raise ValueError("frozen card identity/window differs from campaign labels")
    # The shared join implementation itself is bound to campaign preflight.
    evaluator = REPO / "neurobench/experiments/gamma_ls_difference/two_stencil_evaluation.py"
    _verify_file(evaluator, preflight["code_sha256"][str(evaluator.relative_to(REPO))])
    destination.mkdir()
    provenance = {"campaign_protocol_sha256": _sha256(root/"protocol.json"),
        "campaign_preflight_sha256": _sha256(root/"preflight.json"),
        "campaign_candidate_seal_sha256": _sha256(root/"campaign_candidate_seal.json"),
        "label_source": str(label_path), "label_sha256": preflight["label_sha256"],
        "selection_path": str(selection_path), "selection_sha256": _sha256(selection_path),
        "code_sha256": _sha256(Path(__file__)), "verified_sources": {}, "verified_cells": {}}
    shutil.copy2(selection_path, destination/"card_selection.tsv")
    shutil.copy2(Path(__file__), destination/Path(__file__).name)
    sources = {}
    for name, source in protocol["inputs"].items():
        provenance["verified_sources"][name] = _verify_file(Path(source["path"]), source["file_sha256"])
        values = np.load(source["path"], mmap_mode="r", allow_pickle=False)
        if list(values.shape) != source["shape_tyx"] or str(values.dtype) != source["dtype"]:
            raise ValueError("input source shape/dtype differs from frozen protocol")
        sources[name] = values
    all_rows, category_rows, card_records, card_packets = [], [], {}, {}
    for cell in protocol["cells"]:
        cell_id = cell["cell_id"]
        folder = root / "cells" / cell_id
        complete = _json(folder/"numeric_complete.json")
        verified = {}
        required_metadata = {str((folder/"operator.json").relative_to(root)),
                             str((folder/"readouts/Z/calibration.json").relative_to(root))}
        if not required_metadata <= set(complete["metadata_sha256"]):
            raise ValueError("operator and q1 calibration must be bound in the scoring seal")
        for path, expected in complete["metadata_sha256"].items():
            verified[path] = _verify_file(root/path, expected)
        operator = _json(folder/"operator.json")
        calibration = _json(folder/"readouts/Z/calibration.json")
        operating = next(r for r in calibration["operating_points"] if r["target_proposals_per_frame"] == 1.0)
        tau, floor = float(operating["threshold_z"]), float(operator["scale_floor"])
        candidate_seal = _json(folder/"candidate_seal.json")
        candidate_item = next(r for r in candidate_seal["candidate_files"] if r["readout"] == "Z" and r["q"] == 1.0)
        q_folder = folder/"readouts/Z/q1"
        candidate_path = q_folder/"candidates.tsv"
        if root/candidate_item["path"] != candidate_path:
            raise ValueError("sealed q1 candidate path differs from expected path")
        verified[candidate_item["path"]] = _verify_file(candidate_path, candidate_item["sha256"])
        candidates = _read_tsv(candidate_path)
        occurrence_rows, site_rows = _verify_join_tables(q_folder, cell_id, candidates, labels)
        summary = _json(q_folder/"summary.json")
        frozen_summary = next(r for r in complete["summary_rows"] if r["readout"] == "Z" and r["target_proposals_per_frame"] == 1.0)
        if summary != frozen_summary or summary["threshold"] != tau or summary["scale_floor"] != floor:
            raise ValueError("q1 summary differs from numeric/calibration seal")
        for path in q_folder.glob("*.tsv"):
            verified[str(path.relative_to(root))] = _sha256(path)
        stages = {"X": sources[cell["input_representation"]]}
        for name in STAGES[1:]:
            path = root/complete["stage_paths"][name]
            verified[complete["stage_paths"][name]] = _verify_file(path, complete["stage_sha256"][name])
            stages[name] = np.load(path, mmap_mode="r", allow_pickle=False)
            if list(stages[name].shape) != complete["stage_shapes"][name] or stages[name].dtype != np.float32:
                raise ValueError("stage shape/dtype differs from saved float32 scoring artifact")
        trajectories, trajectory_rows, cell_rows = {}, [], []
        for index, occurrence in enumerate(occurrence_rows):
            observation = occurrence["observation_id"]
            record, packet = occurrence_stage_observations(stages,
                x_px=float(occurrence["x_px"]), y_px=float(occurrence["y_px"]),
                source_start_ui=int(occurrence["source_start_ui"]), source_stop_ui=int(occurrence["source_stop_ui"]),
                threshold_z=tau, scale_floor=floor)
            nearby = nearby_readout_observations(occurrence, candidates, site_rows)
            assigned = _matched(occurrence["matched"])
            category = classify_occurrence(matched=assigned, max_local_z=record["max_local_disk_Z"],
                threshold_z=tau, nearby_frame_proposal_count=nearby["nearby_frame_proposal_count"],
                nearby_representative_count=nearby["nearby_representative_count"])
            # A matched representative is necessarily an emitted suprathreshold
            # integer pixel in the matching disk. Fail on contradictory artifacts.
            if (assigned and (record["max_local_disk_Z"] <= tau or nearby["nearby_representative_count"] == 0)
                    or nearby["nearby_representative_count"] > 0 and nearby["nearby_frame_proposal_count"] == 0):
                raise ValueError("saved matching and stage/proposal observations are inconsistent")
            prefix = f"occ_{index:03d}"
            row = {"operator_cell_id": cell_id, "input_representation": cell["input_representation"],
                "design": cell["spec"]["design"], "reference_family": cell["spec"]["reference_family"],
                "guard_radius_px": cell["spec"]["guard_radius_px"], "support_geometry": cell["spec"]["support_geometry"],
                "is_deployed_anchor": cell["is_deployed_anchor"], "target_proposals_per_calibration_frame": 1.0,
                **occurrence, **record, **nearby, "diagnostic_category": category,
                "selected_frozen_card": observation in selected_ids, "trajectory_npz_prefix": prefix,
                "trajectory_npz_path": f"trajectories/{cell_id}.npz"}
            cell_rows.append(row)
            trajectories.update({f"{prefix}__{key}": value for key, value in packet.items()})
            trajectory_rows.extend({"observation_id": observation, **{key: value[t].item() for key, value in packet.items()}}
                                   for t in range(len(packet["source_frame_ui"])))
            if observation in selected_ids:
                card_records[(cell_id, observation)] = row
                card_packets[(cell_id, observation)] = packet
        all_rows.extend(cell_rows)
        for category in CATEGORIES:
            category_rows.append({"operator_cell_id": cell_id, "diagnostic_category": category,
                                  "occurrence_count": sum(r["diagnostic_category"] == category for r in cell_rows),
                                  "denominator_known_occurrences": len(cell_rows)})
        trajectory_dir = destination/"trajectories"
        trajectory_dir.mkdir(exist_ok=True)
        np.savez_compressed(trajectory_dir/f"{cell_id}.npz", **trajectories)
        _write_tsv(trajectory_dir/f"{cell_id}.tsv", trajectory_rows)
        _write_tsv(destination/"cells"/f"{cell_id}.tsv", cell_rows)
        provenance["verified_cells"][cell_id] = verified
        _write_json(destination/"provenance.json", provenance)
        print(json.dumps({"focused_diagnostics_cell": cell_id, "occurrences": len(cell_rows), "status": "OBSERVED_FROM_VERIFIED_SAVED_STAGES"}), flush=True)
        del stages, trajectories, trajectory_rows
    _write_tsv(destination/"occurrence_stage_observations.tsv", all_rows)
    _write_tsv(destination/"category_counts.tsv", category_rows)
    cards = _render_cards(destination, protocol["cells"], selected, card_records, card_packets) if render else []
    definitions = {"claim_boundary": CLAIM_BOUNDARY,
        "categories_in_precedence_order": list(CATEGORIES), "cutoff": "strict Z > frozen q1 tau",
        "q_unit": protocol["q_unit"], "matching_disk": "integer pixel centers at Euclidean distance <=6 from original float expert center",
        "trace_pixel": "floor(coord+0.5), clipped to last valid image pixel",
        "peak": "largest signed Z, not absolute magnitude; earliest frame on a tie",
        "disk_peak_ties": "earliest source frame, then row-major pixel order",
        "time_window": "one-based source frames, both endpoints included; configured burst, not onset truth",
        "scale_floor_active": "raw reference sigma < fitted scale floor; epsilon remains part of saved Z denominator",
        "input_units": "native units of each cell's source representation; no cross-representation renormalization",
        "reference_scale_interpretation": "weighted local field dispersion, not sampling uncertainty of target-minus-reference",
        "selection": "same 14 pre-existing cards; all 79 occurrences included in every cell table",
        "proposal_source": "complete saved full-field q1 candidate stream; NMS not rerun",
        "join_integrity": "saved tables checked against shared grouping/matching of sealed candidates and labels",
        "scope": "supplementary stage diagnostic figures/tables; does not replace the complete scientific audit"}
    _write_json(destination/"definitions.json", definitions)
    (destination/"README.md").write_text(
        "# Focused two-stencil stage diagnostics\n\n" + CLAIM_BOUNDARY + "\n\n"
        "Read definitions.json, category_counts.tsv, and occurrence_stage_observations.tsv first. "
        "Each of the 25 cells has all 79 known occurrences. Trajectories retain native X, A, M, sigma, contrast, Z "
        "at the recorded exact image pixel over the inclusive configured burst, plus the maximum Z within "
        "the unrounded radius-6 matching disk. The pixel and disk peaks are distinct observations.\n\n"
        "The five ordered categories describe match, threshold, emitted frame proposal, consolidated "
        "burst representative, and one-to-one assignment observations. An absent nearby frame proposal "
        "can reflect multiple maintained readout rules; no specific cause is inferred. A spatial representative "
        "is a review site, not a biological event or unique neuron. Unmatched candidates remain unknown.\n\n"
        "Cards retain the same 14 previously frozen IDs. Their native-unit y axes are independent and "
        "their q1 thresholds are separately frozen; compare timing and declared scalar margins with that "
        "constraint. The category tables include every occurrence, so card selection cannot change counts.\n")
    report = {"status": "DESCRIPTIVE_STAGE_DIAGNOSTICS_COMPLETE", "cell_count": len(protocol["cells"]),
        "occurrence_observation_count": len(all_rows), "expected_occurrence_observation_count": 25*79,
        "frozen_selected_card_count": len(selected), "rendered_card_count": len(cards), "cards": cards,
        "visual_qa": "pending" if render else "not_rendered", "scientific_promotion": False,
        "causal_attribution_identified": False, "biological_precision_identified": False,
        "scientific_audit_complete": False, "claim_boundary": CLAIM_BOUNDARY}
    report["artifacts"] = [{"path": str(path.relative_to(destination)), "size_bytes": path.stat().st_size,
                            "sha256": _sha256(path)} for path in sorted(destination.rglob("*")) if path.is_file()]
    _write_json(destination/"summary.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="completed 25-cell campaign root")
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION,
                        help="original frozen 14-card TSV or a byte-identical copy")
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    result = run_focused_diagnostics(args.output, selection_path=args.selection, render=not args.no_render)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
