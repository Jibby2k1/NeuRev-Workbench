"""Fixed weak-source diagnostics from sealed crowding scores and assignments.

This reads saved scores/candidates; it never reruns NMS, fits a threshold,
reassigns truth, changes a configuration, or promotes biological claims.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / "Outputs/GammaLSFollowup/followup_20260914_r1"
PLAN = {
    "readouts": ["A", "C", "Z"], "selectors": [13, 3], "seeds": [20260916, 20260917, 20260918],
    "separations_px": [None, 8, 12, 16], "operating_point": "q1", "weak_center_xy": [68, 64],
    "nearby_radius_px": 2, "assignment_radii_px": [2, 6], "center_max_windows_px": [3, 13],
    "expected_cells": 72, "weak_active_frames_per_cell": 246,
    "stages": ["Raw", "Input", "A", "M", "Spread", "C", "Z"],
    "figures": "Three readout-specific trace grids (all four separations and all three seeds), plus one gate-count matrix pooled over the three paired seeds.",
}
GATES = ("center_above_tau", "center_max_eligible_w3", "center_max_eligible_w13",
         "prefix_near2", "selected_near2", "weak_assigned_r2", "weak_assigned_r6")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


class Sources:
    """Fresh byte verification once per distinct input file, with bounded buffers."""
    def __init__(self):
        self.records = {}
        self.stats = {}

    def bind(self, path):
        path = Path(path).resolve()
        stat = path.stat()
        stamp = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if path in self.records:
            require(self.stats[path] == stamp, f"Source changed during diagnostics: {path}")
            return self.records[path]
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        after = path.stat()
        require(stamp == (after.st_size, after.st_mtime_ns, after.st_ctime_ns), f"Source changed while hashing: {path}")
        record = dict(path=str(path), size_bytes=stat.st_size, sha256=digest.hexdigest())
        self.records[path], self.stats[path] = record, stamp
        return record

    def verify(self, expected):
        actual = self.bind(expected["path"])
        require(all(actual[key] == expected[key] for key in ("sha256", "size_bytes")), f"Changed frozen source: {actual['path']}")
        return actual

    def read(self, path):
        self.bind(path)
        return json.loads(Path(path).read_text())

    def unchanged(self):
        for path in self.records:
            self.bind(path)


def _write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _write_tsv(path, rows):
    import csv
    require(bool(rows), "Diagnostic table cannot be empty")
    with Path(path).open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _json_array_rows(path):
    """Stream the full saved prefix without retaining its distant proposals."""
    decoder = json.JSONDecoder()
    with Path(path).open(encoding="utf-8") as stream:
        buffer = ""
        position = 0
        eof = False

        def fill():
            nonlocal buffer, position, eof
            block = stream.read(1024 * 1024)
            eof = not block
            buffer = buffer[position:] + block
            position = 0
            require(len(buffer) <= 32 * 1024 * 1024, "Unexpected oversized prefix row")

        def trim():
            nonlocal position
            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if position < len(buffer) or eof:
                    return
                fill()

        trim()
        require(buffer[position:position + 1] == "[", "Saved candidate prefix must be a JSON array")
        position += 1
        trim()
        if buffer[position:position + 1] == "]":
            position += 1
        else:
            while True:
                trim()
                while True:
                    try:
                        row, offset = decoder.raw_decode(buffer, position)
                        break
                    except json.JSONDecodeError:
                        require(not eof, "Truncated/malformed candidate prefix")
                        fill()
                require(isinstance(row, dict), "Saved candidate prefix rows must be objects")
                position = offset
                yield row
                trim()
                require(position < len(buffer), "Unclosed candidate prefix")
                if buffer[position] == "]":
                    position += 1
                    break
                require(buffer[position] == ",", "Malformed candidate prefix separator")
                position += 1
                trim()
                require(buffer[position:position + 1] != "]", "Trailing comma in candidate prefix")
        require(not buffer[position:].strip(), "Trailing content after candidate prefix")
        for remainder in iter(lambda: stream.read(1024 * 1024), ""):
            require(not remainder.strip(), "Trailing content after candidate prefix")


def center_observations(values, indices, *, x=68, y=64):
    """Exact positive center eligibility before greedy selection, for both windows."""
    require(values.ndim == 3 and 6 <= x < values.shape[2] - 6 and 6 <= y < values.shape[1] - 6,
            "Weak center does not have the declared complete neighborhood")
    center = np.asarray(values[indices, y, x], dtype=np.float64)
    result = {"center": center}
    require(np.isfinite(center).all(), "Nonfinite center score")
    for window in (3, 13):
        radius = window // 2
        patch = np.asarray(values[indices, y - radius:y + radius + 1, x - radius:x + radius + 1])
        require(np.isfinite(patch).all(), "Nonfinite saved score neighborhood")
        maximum = patch.max(axis=(1, 2))
        result[f"local_max_w{window}"] = maximum
        result[f"eligible_w{window}"] = (center > 0) & (center == maximum)
    return result


def _core(row):
    return (str(row["proposal_id"]), int(row["source_frame_ui"]),
            float(row["score"]), float(row["x_px"]), float(row["y_px"]))


def _nearby(rows, frames, *, x=68, y=64):
    result = {frame: dict(count=0, center_retained=False, max_score=None, nearest_distance_px=None) for frame in frames}
    for row in rows:
        frame = int(row["source_frame_ui"])
        if frame not in result:
            continue
        distance2 = (float(row["x_px"]) - x) ** 2 + (float(row["y_px"]) - y) ** 2
        if distance2 <= 4:
            out = result[frame]
            out["count"] += 1
            out["center_retained"] |= distance2 == 0
            out["max_score"] = max(float(row["score"]), out["max_score"] if out["max_score"] is not None else -math.inf)
            distance = math.sqrt(distance2)
            out["nearest_distance_px"] = min(distance, out["nearest_distance_px"] if out["nearest_distance_px"] is not None else math.inf)
    return result


def _assigned(metrics, weak, frames, radius, audit_cores):
    require(metrics["summary"]["match_radius_px"] == radius and
            metrics["summary"]["proposal_count"] == len(audit_cores), "Operating metric radius/count mismatch")
    require(len(metrics["proposal_rows"]) == len(audit_cores) and
            {_core(row) for row in metrics["proposal_rows"]} == set(audit_cores.values()),
            "Operating assignments do not describe the frozen q1 proposals")
    result = {frame: [] for frame in frames}
    for row in metrics["proposal_rows"]:
        if row.get("is_true_positive") and row.get("matched_event_id") == weak["event_id"]:
            frame = int(row["source_frame_ui"])
            require(frame in result, "Weak assignment outside its active window")
            require(float(row["match_distance_px"]) <= radius, "Assignment exceeds its declared match radius")
            result[frame].append(row)
    require(all(len(rows) <= 1 for rows in result.values()), "Duplicate weak assignments in one frame")
    event = [row for row in metrics["event_rows"] if row["event_id"] == weak["event_id"]]
    require(len(event) == 1 and event[0]["active_frame_count"] == len(frames) and
            event[0]["matched_active_frame_count"] == sum(bool(rows) for rows in result.values()),
            "Weak frame assignments do not reconcile the saved event metrics")
    return result


def _expected_cells(protocol):
    regional = [(case, f"{arm}__{method}") for case in protocol["regional_cases"]
                for arm in ("level_X", "level_A", "level_C", "level_Z", "difference_Z")
                for method in ("global", "regional")]
    crowded = [(case["case_id"], f"level_{score}__w{window}") for case in protocol["crowding_cases"]
               for score in PLAN["readouts"] for window in PLAN["selectors"]]
    require(len(regional + crowded) == len(set(regional + crowded)) == 242, "Unexpected frozen study matrix")
    return regional + crowded


def run(root=ROOT, output=None):
    root = Path(root).resolve()
    output = Path(output).resolve() if output is not None else root / "crowding_diagnostics"
    require(not output.exists(), f"Preserve existing diagnostic output: {output}")
    sources = Sources()
    protocol = sources.read(root / "protocol.json")
    preflight = sources.read(root / "preflight.json")
    require(preflight["status"] == "PASS" and preflight["protocol_sha256"] == sources.bind(root / "protocol.json")["sha256"],
            "Frozen protocol/preflight mismatch")
    for record in protocol["code_bindings"]:
        sources.verify(record)
    complete = sources.read(root / "evaluation_complete.json")
    require(complete.get("status") == "PASS" and complete.get("cells") == 242, "All 242 evaluations must finish first")
    evaluated = {}
    for case, arm in _expected_cells(protocol):
        folder = root / "cells" / case / arm
        seal = sources.read(folder / "sealed.json")
        marker = sources.read(folder / "evaluated.json")
        require(seal.get("status") == "SEALED_BEFORE_ACTIVITY_TRUTH_JOIN" and marker.get("status") == "PASS",
                "Every cell must be sealed and evaluated before diagnostic truth access")
        require(Path(marker["seal"]["path"]).resolve() == folder / "sealed.json", "Evaluation points to a different seal")
        sources.verify(marker["seal"])
        evaluated[(case, arm)] = (seal, marker)
    cases = protocol["crowding_cases"]
    require(len(cases) == 12 and {(c["seed"], c["separation_px"]) for c in cases} ==
            {(seed, distance) for seed in PLAN["seeds"] for distance in PLAN["separations_px"]}, "Unexpected crowding cases")
    table, summaries = [], []
    for case in cases:
        name = case["case_id"]
        dataset = root / "datasets" / name
        seal0 = evaluated[(name, "level_A__w13")][0]
        for record in seal0["dataset_bindings"]:
            sources.verify(record)
        meta = sources.read(dataset / "metadata.json")
        experts = sources.read(dataset / "experts.json")
        active = sources.read(dataset / "active.json")
        weak_rows = [r for r in experts if r.get("source_role") == "weak"]
        require(len(weak_rows) == 1, "Expected one declared weak source")
        weak = weak_rows[0]
        require((float(weak["x_px"]), float(weak["y_px"])) == (68., 64.), "Changed weak center")
        frames = list(range(int(weak["source_start_ui"]), int(weak["source_stop_ui"]) + 1))
        require(len(frames) == 246 and frames == list(range(185, 431)), "Changed weak active extent")
        active_weak = [r for r in active if r["event_id"] == weak["event_id"]]
        require(len(active_weak) == 246 and sorted(int(r["source_frame_ui"]) for r in active_weak) == frames and
                all((float(r["x_px"]), float(r["y_px"])) == (68., 64.) for r in active_weak), "Weak active truth mismatch")
        lookup = {int(frame): i for i, frame in enumerate(meta["source_frames_ui"])}
        indices = [lookup[frame] for frame in frames]
        stages = {}
        for stage in PLAN["stages"]:
            record = seal0["stages"][stage]
            sources.verify(record)
            values = np.load(record["path"], mmap_mode="r")
            require(values.shape == (464, 128, 128) and values.dtype == np.float32, f"Unexpected saved stage geometry: {stage}")
            stages[stage] = values
        traces = {stage: np.asarray(values[indices, 64, 68], dtype=np.float64) for stage, values in stages.items()}
        require(all(np.isfinite(trace).all() for trace in traces.values()), "Nonfinite saved stage trace")
        for score in PLAN["readouts"]:
            observations = center_observations(stages[score], indices)
            for window in PLAN["selectors"]:
                arm = f"level_{score}__w{window}"
                seal, marker = evaluated[(name, arm)]
                folder = root / "cells" / name / arm
                for record in seal["dataset_bindings"]:
                    sources.verify(record)
                for stage in PLAN["stages"]:
                    require(seal["stages"][stage] == seal0["stages"][stage], "Selectors/readouts use different native stages")
                require(seal["stages"]["Score"] == seal["stages"][score], "Selected score/stage binding mismatch")
                for key in ("calibration", "prefix", "audit_candidates"):
                    sources.verify(seal[key])
                for record in marker["outputs"]:
                    sources.verify(record)
                operating = sources.read(folder / "calibration.json")
                require(operating["threshold_id"] == "q1" and operating["calibration_method"] == "global" and
                        operating["threshold_frozen_from_calibration_only"] is True and operating["window"] == window,
                        "Diagnostics require the frozen global q1 operating point")
                tau = float(operating["threshold"])
                require(math.isfinite(tau) and tau >= 0, "Invalid frozen cutoff")
                audit = sources.read(seal["audit_candidates"]["path"])
                audit_cores = {str(row["proposal_id"]): _core(row) for row in audit}
                require(len(audit_cores) == len(audit), "Duplicate selected proposal IDs")
                selected_ids = set()
                nearby_prefix_rows = []
                prefix_count = 0
                for row in _json_array_rows(seal["prefix"]["path"]):
                    prefix_count += 1
                    require(math.isfinite(float(row["score"])) and float(row["score"]) > 0, "Nonpositive prefix score")
                    require(int(row["source_frame_ui"]) in meta["application_source_frames_ui"], "Prefix outside application")
                    if float(row["score"]) > tau:
                        identity = str(row["proposal_id"])
                        require(identity not in selected_ids and audit_cores.get(identity) == _core(row),
                                "Final selected proposals do not equal the strict filtered prefix")
                        selected_ids.add(identity)
                    if (float(row["x_px"]) - 68) ** 2 + (float(row["y_px"]) - 64) ** 2 <= 4:
                        x, y = float(row["x_px"]), float(row["y_px"])
                        require(x.is_integer() and y.is_integer() and
                                float(row["score"]) == float(stages[score][lookup[int(row["source_frame_ui"])], int(y), int(x)]),
                                "Nearby saved proposal score differs from its native score array")
                        nearby_prefix_rows.append(row)
                require(selected_ids == set(audit_cores), "Selected proposal absent from its frozen full prefix")
                near_prefix = _nearby(nearby_prefix_rows, frames)
                near_selected = _nearby(audit, frames)
                assignments = {}
                for radius in (2, 6):
                    assignments[radius] = _assigned(sources.read(folder / f"operating_metrics_r{radius}.json"),
                                                     weak, frames, radius, audit_cores)
                local = []
                for index, frame in enumerate(frames):
                    prefix, selected = near_prefix[frame], near_selected[frame]
                    row = dict(case_id=name, seed=case["seed"], separation_px=case["separation_px"], readout=score,
                               selector_window_px=window, arm_id=arm, weak_event_id=weak["event_id"],
                               source_frame_ui=frame, ms_since_sampled_fluorescence_onset=(frame - frames[0]) * 20,
                               weak_center_x_px=68, weak_center_y_px=64, weak_active=True,
                               threshold_q1=tau, center_score=float(observations["center"][index]),
                               center_margin=float(observations["center"][index]) - tau,
                               center_above_tau=bool(observations["center"][index] > tau),
                               center_max_eligible_w3=bool(observations["eligible_w3"][index]),
                               center_max_eligible_w13=bool(observations["eligible_w13"][index]),
                               center_neighborhood_max_w3=float(observations["local_max_w3"][index]),
                               center_neighborhood_max_w13=float(observations["local_max_w13"][index]),
                               prefix_near2=bool(prefix["count"]), prefix_near2_count=prefix["count"],
                               prefix_near2_max_score=prefix["max_score"], prefix_near2_nearest_distance_px=prefix["nearest_distance_px"],
                               center_retained_prefix=prefix["center_retained"],
                               selected_near2=bool(selected["count"]), selected_near2_count=selected["count"],
                               selected_near2_max_score=selected["max_score"], selected_near2_nearest_distance_px=selected["nearest_distance_px"],
                               center_retained_selected=selected["center_retained"],
                               weak_assigned_r2=bool(assignments[2][frame]), weak_assigned_r6=bool(assignments[6][frame]),
                               weak_assignment_r2_proposal_id=assignments[2][frame][0]["proposal_id"] if assignments[2][frame] else None,
                               weak_assignment_r6_proposal_id=assignments[6][frame][0]["proposal_id"] if assignments[6][frame] else None,
                               spread_floor=float(operating["scale_floor"]),
                               spread_floor_active=bool(traces["Spread"][index] < operating["scale_floor"]))
                    row.update({stage: float(trace[index]) for stage, trace in traces.items()})
                    local.append(row)
                summary = dict(case_id=name, seed=case["seed"], separation_px=case["separation_px"], readout=score,
                               selector_window_px=window, arm_id=arm, threshold_q1=tau,
                               weak_active_frame_count=len(local), full_application_prefix_count=prefix_count,
                               full_application_q1_proposal_count=len(audit),
                               **{f"{gate}_frame_count": sum(row[gate] for row in local) for gate in GATES},
                               center_above_tau_without_near2_prefix_count=sum(row["center_above_tau"] and not row["prefix_near2"] for row in local),
                               prefix_near2_without_selected_near2_count=sum(row["prefix_near2"] and not row["selected_near2"] for row in local),
                               weak_assigned_r6_without_near2_selected_count=sum(row["weak_assigned_r6"] and not row["selected_near2"] for row in local))
                require(summary["selected_near2_frame_count"] == summary["weak_assigned_r2_frame_count"],
                        "Disjoint 2px truth disks do not reconcile nearby selection and saved assignment")
                summaries.append(summary)
                table.extend(local)
        del stages
    require(len(summaries) == 72 and len(table) == 72 * 246, "Incomplete diagnostic table")
    pooled = []
    for score in PLAN["readouts"]:
        for distance in PLAN["separations_px"]:
            for window in PLAN["selectors"]:
                selected = [row for row in summaries if row["readout"] == score and row["separation_px"] == distance and row["selector_window_px"] == window]
                require(len(selected) == 3 and {row["seed"] for row in selected} == set(PLAN["seeds"]), "Missing paired seeds")
                pooled.append(dict(readout=score, separation_px=distance, selector_window_px=window,
                                   paired_seed_count=3, weak_active_frame_count=738,
                                   **{f"{gate}_frame_count": sum(row[f"{gate}_frame_count"] for row in selected) for gate in GATES}))
    sources.unchanged()
    output.mkdir(parents=True)
    _write_json(output / "plan.json", PLAN)
    _write_tsv(output / "weak_active_frame_observations.tsv", table)
    _write_tsv(output / "cell_summaries.tsv", summaries)
    _write_tsv(output / "pooled_seed_gate_counts.tsv", pooled)
    _plots(output, table, pooled)
    _write_json(output / "summary.json", dict(status="PASS", diagnostic_cells=72, weak_active_frame_rows=len(table),
        weak_frames_per_cell=246, pooled_seed_groups=24, figure_count=4,
        definitions=dict(exact_center="Saved score at fixed evaluation (x,y)=(68,64); original raw location=(117,113).",
                         center_max_eligibility="Positive saved center score equals the maximum of its full 3x3 or13x13 saved-score patch, before greedy NMS. The fixed center is inside the unchanged6px eligible border.",
                         nearby="At least one saved proposal within inclusive Euclidean2px of the weak center in that same source frame.",
                         prefix="Original complete positive prefix from the cell's seal, after the specified global local-max/greedy selector, before q1 filtering.",
                         final_selected="Exactly the saved audit_candidates rows; verified equal to prefix rows with raw score strictly above frozen q1 tau.",
                         assignment="Weak-source true-positive rows in saved operating_metrics_r2/r6; never recomputed here.",
                         aggregation="Count active weak frames; denominator246 per cell or738 over three paired noise seeds. Frames are not independent replicates.",
                         limitations="Columns are descriptive and not all nested: exact-center exceedance need not produce a nearby peak;6px assignment can occur without a proposal within2px. No cause, preferred configuration, biological identity, real precision, neural onset, or controller performance is inferred."),
        rescoring_performed=False, nms_rerun=False, thresholds_refitted=False, assignments_recomputed=False,
        scientific_promotion=False, visual_qa_complete=False))
    sources.bind(Path(__file__))
    sources.unchanged()
    _write_json(output / "source_manifest.json", dict(status="VERIFIED_SAVED_SOURCES",
        created_utc=datetime.now(timezone.utc).isoformat(), sources=list(sources.records.values()),
        verification="Fresh SHA256 and size for every distinct used source, including full saved stage files and prefix ledgers; buffered reads, read-only mmap extraction. All242 seals/evaluations checked before weak truth access.",
        frozen_parameters=PLAN))
    artifact_sources = Sources()
    artifacts = [artifact_sources.bind(path) for path in sorted(output.rglob("*")) if path.is_file()]
    _write_json(output / "artifact_index.json", dict(artifacts=artifacts))
    return dict(status="PASS", output=str(output), cells=72, frame_rows=len(table), figures=4)


def _plots(output, rows, pooled):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    figures = output / "figures"
    figures.mkdir()
    colors = ("#315b89", "#777777", "#985c8d")
    groups = defaultdict(list)
    for row in rows:
        groups[(row["readout"], row["separation_px"], row["selector_window_px"], row["seed"])].append(row)
    for score in PLAN["readouts"]:
        fig, axes = plt.subplots(4, 2, figsize=(11, 10), sharex=True, sharey=True)
        for row_index, distance in enumerate(PLAN["separations_px"]):
            for col, window in enumerate(PLAN["selectors"]):
                ax = axes[row_index, col]
                for color, seed in zip(colors, PLAN["seeds"]):
                    trace = groups[(score, distance, window, seed)]
                    time = np.asarray([r["ms_since_sampled_fluorescence_onset"] for r in trace]) / 1000
                    ax.plot(time, [r["center_score"] for r in trace], color=color, lw=1.0)
                    ax.axhline(trace[0]["threshold_q1"], color=color, lw=.8, ls="--")
                label = "Weak alone" if distance is None else f"Neighbor {distance}px left"
                ax.set_title(f"{label} | {window}x{window} prefilter", fontsize=10)
                ax.grid(alpha=.2)
                if col == 0:
                    ax.set_ylabel(f"{score} at weak center" + (" (unitless)" if score == "Z" else " (fluorescence units)"), fontsize=9)
                if row_index == 3:
                    ax.set_xlabel("Seconds since first active fluorescence sample", fontsize=9)
        handles = [Line2D([0], [0], color=color, label=str(seed)) for color, seed in zip(colors, PLAN["seeds"])]
        handles += [Line2D([0], [0], color="black", ls="--", label="Setup-frozen q1 cutoff")]
        fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(.5, .035))
        fig.suptitle(f"Fixed weak-center {score}: all three paired seeds and all separations", fontsize=13)
        fig.text(.5, .015, "Same saved score fields for both selectors; separately calibrated cutoffs. Active window only. No NMS rerun or causal attribution.", ha="center", fontsize=8)
        fig.subplots_adjust(left=.105, right=.98, top=.925, bottom=.11, hspace=.4, wspace=.15)
        fig.savefig(figures / f"weak_center_{score}_all_seeds.png", dpi=150)
        plt.close(fig)
    counts = np.asarray([[row[f"{gate}_frame_count"] for gate in GATES] for row in pooled])
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(counts / 738, cmap="Greys", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(GATES)), ["Center\n> cutoff", "Center max\n3x3", "Center max\n13x13", "Prefix\nwithin2px", "Selected\nwithin2px", "Weak assigned\n2px", "Weak assigned\n6px"], fontsize=9)
    labels = [f"{r['readout']} | {'alone' if r['separation_px'] is None else str(r['separation_px'])+'px'} | w{r['selector_window_px']}" for r in pooled]
    ax.set_yticks(range(len(labels)), labels, fontsize=9)
    for y in range(len(pooled)):
        for x in range(len(GATES)):
            ax.text(x, y, f"{counts[y,x]}/738", ha="center", va="center", fontsize=8,
                    color="white" if counts[y,x] > 400 else "black")
    ax.set_title("Distinct diagnostic counts across all three paired seeds\n738 active weak-source frames per row; columns are not a nested loss funnel", fontsize=12)
    fig.subplots_adjust(left=.22, right=.98, top=.91, bottom=.085)
    fig.text(.5, .025, "Full source tables retain each seed. Shade represents count /738; frames are not independent replicates. Saved q1 assignments only.", ha="center", fontsize=8)
    fig.savefig(figures / "weak_gate_counts_pooled_seeds.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.root, args.output), sort_keys=True))
