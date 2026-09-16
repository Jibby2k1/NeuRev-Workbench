"""Resumable, label-sealed execution of the approved two-stencil experiment.

The matrix is fixed before labels enter. Calibration uses scored UI2..100;
application uses UI101..2359. Each cell retains every LS stage and every
proposal at all five frozen cutoffs. Audits are produced separately at q=1.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

from neurobench.algorithms.gamma_two_stencil import (
    build_two_stencil_factorial, two_stencil_local_standardization,
)
from .two_stencil_evaluation import (
    calibrate_tau, evaluate_occurrence_windows, extract_frame_candidates,
    NMS_SEMANTICS, TARGET_PROPOSALS_PER_FRAME,
)

REPO = Path(__file__).resolve().parents[3]
SOURCE_ROOT = REPO / "Outputs/GammaLSDifference/spon_ca_burst_gamma_ls_full_recording_q1_stage_replay_v1_20260909_r1"
LABEL_PATH = REPO / "Outputs/GammaLSDifference/spon_ca_burst_gamma_ls_fixed_deployment_characterization_v1_20260908_r1/audit_inputs/expert_occurrences.tsv"
STAGES = {"A": "target_response", "M": "reference_mean", "sigma": "reference_std", "contrast": "contrast", "Z": "values"}
SNAPSHOT_UI = (50, 100, 1840, 2003, 2014, 2026, 2051, 2135, 2277, 2359)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def write_tsv(path, rows):
    path = Path(path)
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_tsv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def progress(root, **values):
    record = {"updated_unix": time.time(), **values}
    write_json(Path(root) / "progress.json", record)
    print(json.dumps(record, sort_keys=True), flush=True)


def protocol():
    manifest = json.loads((SOURCE_ROOT / "stage_manifest.json").read_text())
    return {
        "schema_version": 1,
        "experiment": "prospective_two_stencil_factorial_v1",
        "source_root": str(SOURCE_ROOT),
        "source_manifest_sha256": sha256(SOURCE_ROOT / "stage_manifest.json"),
        "source_movie": manifest["source_movie"],
        "inputs": {name: {**manifest["arrays"][name], "path": str(SOURCE_ROOT / manifest["arrays"][name]["path"])}
                   for name in ("conditioned_current_frame", "difference_signed")},
        "cells": [asdict(cell) for cell in build_two_stencil_factorial(include_deployed_anchor=True)],
        "calibration_source_ui": [2, 100], "application_source_ui": [101, 2359],
        "cold_start_ui1_excluded_from_calibration": True,
        "scale_floor_percentile": 10.0, "scale_floor_population": "all_positive_reference_std_pixels_UI2_100_numpy_linear",
        "target_proposals_per_frame": list(TARGET_PROPOSALS_PER_FRAME),
        "q_unit": "NMS proposals per calibration score frame; different from older proposals per 1s block",
        "calibration_assumed_event_free": False, "application_rate_controlled": False,
        "nms_semantics": NMS_SEMANTICS,
        "stage_readout_controls": "A_and_contrast_on_direct_gamma_g0_and_serial_gamma_g0_for_both_inputs_plus_deployed_anchor; Z_all25",
        "upstream_conditioning": "unchanged source replay; centered target pooling is additional",
        "snapshot_source_ui": list(SNAPSHOT_UI),
        "frame_interval_ms": 20.0,
        "scientific_audit": {"enabled": True, "media_operating_point_q": 1.0,
            "media_cells": "all25", "other_q_scope": "complete_numeric_proposals_and_occurrence_tables",
            "every_expert_roi_and_occurrence": True, "every_consolidated_model_review_site": True,
            "model_site_identity": "spatial_review_surrogate_not_neuron_or_unique_event"},
        "claim_boundary": "within-recording development comparison; labels do not identify precision, event onset, or external generalization",
    }


def preflight(output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"new output root required: {output}")
    config = protocol()
    if shutil.disk_usage(output.parent if output.parent.exists() else REPO).free < 350 * 1024**3:
        raise RuntimeError("less than350GiB free for full stage and audit artifacts")
    verified = []
    for name, source in {**config["inputs"], "Raw": config["source_movie"]}.items():
        path = Path(source["path"])
        data = np.load(path, mmap_mode="r")
        if list(data.shape) != source["shape_tyx"] or str(data.dtype) != source["dtype"]:
            raise ValueError(f"source shape/dtype mismatch: {name}")
        actual = sha256(path)
        if actual != source["file_sha256"]:
            raise ValueError(f"source hash mismatch: {name}")
        verified.append({"stage": name, "path": str(path), "sha256": actual, "shape": list(data.shape)})
        print(f"verified source {name}", flush=True)
    labels = read_tsv(LABEL_PATH)
    if len(labels) != 79 or len({row["canonical_roi_id"] for row in labels}) != 26:
        raise ValueError("unexpected frozen label inventory")
    height, width = config["inputs"]["difference_signed"]["shape_tyx"][1:]
    if any(not (0 <= float(row["x_px"]) < width and 0 <= float(row["y_px"]) < height) for row in labels):
        raise ValueError("expert coordinate outside source image")
    output.mkdir(parents=True)
    write_json(output / "protocol.json", config)
    source_code = [Path(__file__), Path(__file__).with_name("two_stencil_evaluation.py"),
                   Path(__file__).with_name("evaluation.py"), REPO / "neurobench/metrics/sparse_detection.py",
                   REPO / "neurobench/algorithms/gamma_two_stencil.py"]
    for path in source_code:
        shutil.copy2(path, output / path.name)
    write_json(output / "preflight.json", {"status": "PASS", "sources": verified,
        "labels_geometry_only": True, "label_source": str(LABEL_PATH), "label_sha256": sha256(LABEL_PATH),
        "expert_occurrence_count": 79, "expert_identity_count": 26,
        "code_sha256": {str(p.relative_to(REPO)): sha256(p) for p in source_code},
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "free_disk_gib": shutil.disk_usage(output).free / 1024**3,
        "cpu_affinity": sorted(os.sched_getaffinity(0)), "created_unix": time.time()})
    # Geometry-only overlay is required before label-driven execution.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    raw = np.load(config["source_movie"]["path"], mmap_mode="r")
    projection = np.max(np.asarray(raw[::25], dtype=np.float32), axis=0)
    fig, ax = plt.subplots(figsize=(10, 6))
    lo, hi = np.percentile(projection, [1, 99.5])
    ax.imshow(projection, cmap="gray", vmin=lo, vmax=hi)
    ax.scatter([float(r["x_px"]) for r in labels], [float(r["y_px"]) for r in labels], s=25, facecolors="none", edgecolors="#57cf8a")
    ax.set_title("Preflight geometry only: all79 expert occurrences /26 identities")
    fig.savefig(output / "preflight_projection_overlay.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    progress(output, status="PREFLIGHT_PASS", completed_cells=0, total_cells=25)


def _array(path, shape):
    return np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=shape)


def _run_cell(root, cell, device, chunk_frames):
    import torch
    folder = root / "cells" / cell.cell_id
    if (folder / "scoring_complete.json").exists():
        _verify_scored_cell(root, cell)
        print(f"skip sealed cell {cell.cell_id}", flush=True)
        return
    folder.mkdir(parents=True, exist_ok=True)
    stage_folder = folder / "stages"
    stage_folder.mkdir(exist_ok=True)
    config = json.loads((root / "protocol.json").read_text())
    source = np.load(config["inputs"][cell.input_representation]["path"], mmap_mode="r")
    shape = source.shape
    start_time = time.monotonic()
    # First compute moments on only the declared initialization interval.
    calibration = {name: np.empty((99, *shape[1:]), dtype=np.float32) for name in ("sigma", "contrast", "A")}
    for start in range(1, 100, chunk_frames):
        stop = min(start + chunk_frames, 100)
        result = two_stencil_local_standardization(np.array(source[start:stop]), cell.spec, device=device)
        for name in calibration:
            calibration[name][start - 1:stop - 1] = getattr(result, STAGES[name]).cpu().numpy()
        del result
    positive = calibration["sigma"][calibration["sigma"] > 0]
    if not len(positive):
        raise ValueError("no positive calibration scales")
    floor = float(np.percentile(positive, 10, method="linear"))
    del positive
    spec = replace(cell.spec, scale_floor=floor)
    calibration["Z"] = calibration["contrast"] / (np.maximum(calibration["sigma"], floor) + spec.epsilon)
    controls = cell.is_deployed_anchor or (cell.spec.design in {"direct", "serial"} and cell.spec.reference_family == "gamma" and cell.spec.guard_radius_px == 0)
    readouts = ("Z", "A", "contrast") if controls else ("Z",)
    calibrations = {name: calibrate_tau(calibration[name], source_frames_ui=list(range(2, 101))) for name in readouts}
    for name, frozen in calibrations.items():
        write_json(folder / "readouts" / name / "calibration.json", asdict(frozen))
    write_json(folder / "operator.json", {**asdict(cell), "fitted_spec": asdict(spec), "scale_floor": floor,
        "calibration_population": "UI2..100_reference_std_positive_pixels", "readouts": list(readouts),
        "thresholds_frozen_before_application_and_labels": True})
    del calibration
    arrays = {name: _array(stage_folder / f"{name}.npy", shape) for name in STAGES}
    # Stream complete proposal ledgers: application burden is not bounded by q.
    columns = ["proposal_id", "cell_id", "target_proposals_per_frame", "calibration_burden_unit",
        "threshold_z", "source_frame_ui", "source_time_s", "source_time_basis", "frame_interval_ms",
        "candidate_rank_within_frame", "score", "x_px", "y_px", "biological_status",
        "temporal_linking_applied", "readout", "operator_cell_id"]
    ledgers = {}
    for name in readouts:
        for q in TARGET_PROPOSALS_PER_FRAME:
            path = folder / "readouts" / name / f"q{q:g}" / "candidates.tsv"
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = path.open("w", newline="")
            writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
            writer.writeheader()
            ledgers[name, q] = {"path": path, "handle": handle, "writer": writer, "count": 0}
    last_progress = time.monotonic()
    for start in range(0, len(source), chunk_frames):
        stop = min(start + chunk_frames, len(source))
        result = two_stencil_local_standardization(np.array(source[start:stop]), spec, device=device)
        for name, attribute in STAGES.items():
            arrays[name][start:stop] = getattr(result, attribute).cpu().numpy()
        del result
        for index in range(max(start, 100), stop):
            for name in readouts:
                frozen = calibrations[name]
                minimum = min(frozen.threshold_for(q) for q in TARGET_PROPOSALS_PER_FRAME)
                stream_id = cell.cell_id + "__" + name
                prefix = extract_frame_candidates(arrays[name][index], source_frame_ui=index + 1,
                    threshold_z=minimum, cell_id=stream_id)
                for q in TARGET_PROPOSALS_PER_FRAME:
                    tau = frozen.threshold_for(q)
                    selected = [row for row in prefix if row["score"] > tau]
                    ledgers[name, q]["writer"].writerows({**row, "target_proposals_per_frame": q, "threshold_z": tau,
                        "readout": name, "operator_cell_id": cell.cell_id,
                        "proposal_id": f"{stream_id}__q{q:g}__ui{index+1:06d}__r{rank:05d}",
                        "candidate_rank_within_frame": rank} for rank, row in enumerate(selected, 1))
                    ledgers[name, q]["count"] += len(selected)
        if time.monotonic() - last_progress > 30:
            progress(root, status="SCORING", cell_id=cell.cell_id, source_frames_done=stop, total_source_frames=len(source),
                elapsed_cell_seconds=round(time.monotonic()-start_time, 1),
                peak_gpu_mib=torch.cuda.max_memory_allocated()/1024**2 if device.startswith("cuda") else 0)
            last_progress = time.monotonic()
    for array in arrays.values():
        array.flush()
    snapshot = {name: np.array(array[np.array(SNAPSHOT_UI)-1]) for name, array in arrays.items()}
    np.savez_compressed(folder / "stage_snapshots.npz", source_frames_ui=np.array(SNAPSHOT_UI), **snapshot)
    # Freeze all candidates before any outcome join. Labels are read only below.
    candidate_manifest = []
    for (name, q), ledger in ledgers.items():
        ledger["handle"].flush()
        os.fsync(ledger["handle"].fileno())
        ledger["handle"].close()
        candidate_manifest.append({"readout": name, "q": q, "path": str(ledger["path"].relative_to(root)),
                                   "sha256": sha256(ledger["path"]), "count": ledger["count"]})
    write_json(folder / "candidate_seal.json", {"sealed_unix": time.time(), "candidate_files": candidate_manifest,
               "labels_used_for_scoring_or_selection": False})
    write_json(folder / "scoring_complete.json", {"status": "SCORED_SEALED_AWAITING_CAMPAIGN_LABEL_JOIN", "completed_unix": time.time(),
        "duration_seconds": time.monotonic()-start_time,
        "metadata_sha256": {str(path.relative_to(root)): sha256(path) for path in
            [folder / "operator.json", *[folder / "readouts" / name / "calibration.json" for name in readouts]]},
        "candidate_seal_sha256": sha256(folder / "candidate_seal.json"),
        "stage_shapes": {name: list(array.shape) for name, array in arrays.items()},
        "stage_sha256": {name: sha256(stage_folder/f"{name}.npy") for name in STAGES},
        "stage_paths": {name: str((stage_folder/f"{name}.npy").relative_to(root)) for name in STAGES}})
    progress(root, status="CELL_SCORED_SEALED", cell_id=cell.cell_id, elapsed_cell_seconds=round(time.monotonic()-start_time, 1))


def _verify_scored_cell(root, cell):
    folder = root / "cells" / cell.cell_id
    complete = json.loads((folder / "scoring_complete.json").read_text())
    if sha256(folder / "candidate_seal.json") != complete["candidate_seal_sha256"]:
        raise ValueError("completed candidate seal hash mismatch")
    seal = json.loads((folder / "candidate_seal.json").read_text())
    for path, expected in complete.get("metadata_sha256", {}).items():
        if sha256(root/path) != expected:
            raise ValueError("completed operator/calibration metadata hash mismatch")
    for item in seal["candidate_files"]:
        if sha256(root / item["path"]) != item["sha256"]:
            raise ValueError("completed candidate file hash mismatch")
    for name, path in complete["stage_paths"].items():
        if sha256(root / path) != complete["stage_sha256"][name]:
            raise ValueError("completed stage hash mismatch")


def _evaluate_cell(root, cell):
    folder = root / "cells" / cell.cell_id
    if (folder / "numeric_complete.json").exists():
        return
    # Every matrix cell must be sealed before the first descriptive label join.
    all_cells = build_two_stencil_factorial(include_deployed_anchor=True)
    if any(not (root / "cells" / other.cell_id / "scoring_complete.json").exists() for other in all_cells):
        raise ValueError("all25cells must be sealed before any label outcome join")
    pre = json.loads((root / "preflight.json").read_text())
    if sha256(LABEL_PATH) != pre["label_sha256"]:
        raise ValueError("label source changed since geometry preflight")
    seal = json.loads((folder / "candidate_seal.json").read_text())
    scoring = json.loads((folder / "scoring_complete.json").read_text())
    for path, expected in scoring.get("metadata_sha256", {}).items():
        if sha256(root/path) != expected:
            raise ValueError("sealed operator/calibration metadata hash mismatch")
    if sha256(folder / "candidate_seal.json") != scoring["candidate_seal_sha256"]:
        raise ValueError("candidate seal changed after scoring")
    for source in seal["candidate_files"]:
        if sha256(root / source["path"]) != source["sha256"]:
            raise ValueError("candidate table changed after seal")
    operator = json.loads((folder / "operator.json").read_text())
    floor = operator["scale_floor"]
    readouts = operator["readouts"]
    labels = read_tsv(LABEL_PATH)
    windows = {int(r["burst_id"]): [int(r["source_start_ui"]), int(r["source_stop_ui"])] for r in labels}
    result_rows = []
    for name in readouts:
        frozen = json.loads((folder / "readouts" / name / "calibration.json").read_text())
        for q in TARGET_PROPOSALS_PER_FRAME:
            out = folder / "readouts" / name / f"q{q:g}"
            candidates = read_tsv(out / "candidates.tsv")
            selected_op = next(r for r in frozen["operating_points"] if r["target_proposals_per_frame"] == q)
            evaluation = evaluate_occurrence_windows(candidates, labels, burst_intervals_ui=windows)
            for key in ("occurrence_rows", "site_rows", "membership_rows", "burst_summaries"):
                write_tsv(out / f"{key}.tsv", [{"operator_cell_id": cell.cell_id, "readout": name, "target_proposals_per_frame": q, **row} for row in evaluation[key]])
            summary = {"operator_cell_id": cell.cell_id, "input_representation": cell.input_representation,
                "design": cell.spec.design, "reference_family": cell.spec.reference_family,
                "guard_radius_px": cell.spec.guard_radius_px, "support_geometry": cell.spec.support_geometry,
                "is_deployed_anchor": cell.is_deployed_anchor, "readout": name, "target_proposals_per_frame": q,
                "threshold": selected_op["threshold_z"], "scale_floor": floor,
                "calibration_realized_proposals_per_frame": selected_op["calibration_proposals_per_frame"],
                "application_score_frame_count": 2259,
                "application_proposals_per_frame": len(candidates)/2259,
                **evaluation["summary"]}
            write_json(out / "summary.json", summary)
            result_rows.append(summary)
    write_json(folder / "numeric_complete.json", {**scoring, "status": "NUMERIC_COMPLETE_AUDIT_PENDING",
        "label_join_completed_unix": time.time(), "label_sha256": pre["label_sha256"], "summary_rows": result_rows})
    progress(root, status="CELL_NUMERIC_COMPLETE", cell_id=cell.cell_id)


def run(output, *, device="cuda", chunk_frames=8, cell_ids=None):
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    root = Path(output).resolve()
    if not (root / "preflight.json").exists():
        raise ValueError("run requires a completed preflight in a unique output root")
    if json.loads((root / "protocol.json").read_text()) != protocol():
        raise ValueError("resolved protocol changed since preflight")
    pre = json.loads((root / "preflight.json").read_text())
    for path, expected in pre["code_sha256"].items():
        if sha256(REPO/path) != expected:
            raise ValueError(f"implementation changed since preflight: {path}")
    for source in pre["sources"]:
        if sha256(source["path"]) != source["sha256"]:
            raise ValueError(f"source bytes changed since preflight: {source['stage']}")
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.set_per_process_memory_fraction(.25)
    cells = build_two_stencil_factorial(include_deployed_anchor=True)
    # Anchor first establishes runtime and retained artifacts before full expansion.
    cells = (cells[-1],) + cells[:-1]
    if cell_ids:
        selected = set(cell_ids)
        if selected - {cell.cell_id for cell in cells}:
            raise ValueError("unknown requested cell")
        cells = tuple(cell for cell in cells if cell.cell_id in selected)
    for cell in cells:
        _run_cell(root, cell, device, chunk_frames)
    all_cells = build_two_stencil_factorial(include_deployed_anchor=True)
    sealed_count = sum((root / "cells" / cell.cell_id / "scoring_complete.json").exists() for cell in all_cells)
    if sealed_count < 25:
        progress(root, status="PARTIAL_SCORING_COMPLETE", sealed_cells=sealed_count, total_cells=25, label_join_performed=False)
        return
    seals = {cell.cell_id: sha256(root / "cells" / cell.cell_id / "candidate_seal.json") for cell in all_cells}
    campaign_seal_path = root / "campaign_candidate_seal.json"
    if campaign_seal_path.exists():
        if json.loads(campaign_seal_path.read_text())["seals"] != seals:
            raise ValueError("campaign candidate seals changed after campaign freeze")
    else:
        write_json(campaign_seal_path, {"all25sealed_unix": time.time(),
            "seals": seals, "labels_used_for_scoring_or_selection": False})
    for cell in all_cells:
        _evaluate_cell(root, cell)
    summaries = []
    for cell in build_two_stencil_factorial(include_deployed_anchor=True):
        path = root / "cells" / cell.cell_id / "numeric_complete.json"
        if path.exists():
            summaries.extend(json.loads(path.read_text())["summary_rows"])
    write_tsv(root / "results.tsv", summaries)
    count = len({row["operator_cell_id"] for row in summaries})
    progress(root, status="NUMERIC_COMPLETE_AUDIT_PENDING" if count == 25 else "PARTIAL_NUMERIC_COMPLETE", completed_cells=count, total_cells=25)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "run"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunk-frames", type=int, default=8)
    parser.add_argument("--cell", action="append")
    args = parser.parse_args()
    if args.mode == "preflight":
        preflight(args.output)
    else:
        if not 1 <= args.chunk_frames <= 16:
            parser.error("chunk-frames must be1..16")
        run(args.output, device=args.device, chunk_frames=args.chunk_frames, cell_ids=args.cell)


if __name__ == "__main__":
    main()
