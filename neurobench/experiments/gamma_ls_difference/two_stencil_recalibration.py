"""Uniform calibration correction using unchanged sealed two-stencil stages.

The first sampled-pixel threshold grid underfilled current-frame calibration
budgets. This development replay derives cutoffs from exact NMS peak order
statistics, applies the correction to every arm, and preserves the original run.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np

from neurobench.algorithms.gamma_two_stencil import build_two_stencil_factorial
from .two_stencil_campaign import (
    LABEL_PATH, REPO, TARGET_PROPOSALS_PER_FRAME, _evaluate_cell, _verify_scored_cell,
    progress, sha256, write_json, write_tsv,
)
from .two_stencil_evaluation import extract_frame_candidates
from .two_stencil_exact_calibration import calibrate_tau_exact_nms


def _run_cell(source, root, cell):
    old = source / "cells" / cell.cell_id
    folder = root / "cells" / cell.cell_id
    _verify_scored_cell(source, cell)
    if (folder / "scoring_complete.json").exists():
        _verify_scored_cell(root, cell)
        return
    folder.mkdir(parents=True, exist_ok=True)
    for name in ("stages", "stage_snapshots.npz"):
        link = folder / name
        if (link.exists() or link.is_symlink()) and link.resolve() != (old/name).resolve():
            raise ValueError("partial replay stage link does not point to its verified parent")
        if not link.exists():
            link.symlink_to(old/name, target_is_directory=name == "stages")
    with np.load(old / "stage_snapshots.npz") as packet:
        positions = packet["source_frames_ui"] - 1
        for name in ("A", "M", "sigma", "contrast", "Z"):
            full = np.load(old / "stages" / f"{name}.npy", mmap_mode="r")
            if not np.array_equal(packet[name], full[positions]):
                raise ValueError("parent stage snapshots differ from sealed full-stage arrays")
    operator = json.loads((old / "operator.json").read_text())
    operator["calibration_method"] = "exact_nms_peak_order_statistics"
    operator["stage_source_campaign"] = str(source)
    operator["stage_operator_and_scale_floor_unchanged"] = True
    write_json(folder / "operator.json", operator)
    manifests = []
    start_time = time.monotonic()
    for name in operator["readouts"]:
        array = np.load(folder / "stages" / f"{name}.npy", mmap_mode="r")
        frozen = calibrate_tau_exact_nms(array[1:100], source_frames_ui=list(range(2, 101)))
        write_json(folder / "readouts" / name / "calibration.json", asdict(frozen))
        ledgers = {}
        columns = ["proposal_id", "cell_id", "target_proposals_per_frame", "calibration_burden_unit",
            "threshold_z", "source_frame_ui", "source_time_s", "source_time_basis", "frame_interval_ms",
            "candidate_rank_within_frame", "score", "x_px", "y_px", "biological_status",
            "temporal_linking_applied", "readout", "operator_cell_id"]
        for q in TARGET_PROPOSALS_PER_FRAME:
            path = folder / "readouts" / name / f"q{q:g}" / "candidates.tsv"
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = path.open("w", newline="")
            writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
            writer.writeheader()
            ledgers[q] = {"path": path, "handle": handle, "writer": writer, "count": 0}
        last_progress = time.monotonic()
        stream_id = cell.cell_id + "__" + name
        minimum = min(frozen.threshold_for(q) for q in TARGET_PROPOSALS_PER_FRAME)
        try:
            for index in range(100, len(array)):
                prefix = extract_frame_candidates(array[index], source_frame_ui=index+1,
                                                  threshold_z=minimum, cell_id=stream_id)
                for q, ledger in ledgers.items():
                    tau = frozen.threshold_for(q)
                    selected = [row for row in prefix if row["score"] > tau]
                    ledger["writer"].writerows({**row, "target_proposals_per_frame": q, "threshold_z": tau,
                        "readout": name, "operator_cell_id": cell.cell_id,
                        "proposal_id": f"{stream_id}__q{q:g}__ui{index+1:06d}__r{rank:05d}",
                        "candidate_rank_within_frame": rank} for rank, row in enumerate(selected, 1))
                    ledger["count"] += len(selected)
                if time.monotonic() - last_progress > 30:
                    progress(root, status="RECALIBRATION_REPLAY", cell_id=cell.cell_id,
                             readout=name, source_frames_done=index+1)
                    last_progress = time.monotonic()
        finally:
            for ledger in ledgers.values():
                ledger["handle"].flush()
                os.fsync(ledger["handle"].fileno())
                ledger["handle"].close()
        for q, ledger in ledgers.items():
            manifests.append({"readout": name, "q": q, "path": str(ledger["path"].relative_to(root)),
                              "sha256": sha256(ledger["path"]), "count": ledger["count"]})
    write_json(folder / "candidate_seal.json", {"sealed_unix": time.time(), "candidate_files": manifests,
               "labels_used_for_scoring_or_selection": False})
    old_complete = json.loads((old / "scoring_complete.json").read_text())
    metadata_paths = [folder / "operator.json", *[folder / "readouts" / name / "calibration.json" for name in operator["readouts"]]]
    write_json(folder / "scoring_complete.json", {
        "status": "SCORED_SEALED_AWAITING_CAMPAIGN_LABEL_JOIN", "completed_unix": time.time(),
        "duration_seconds": time.monotonic()-start_time,
        "stage_source_campaign": str(source), "stage_arrays_recomputed": False,
        "candidate_seal_sha256": sha256(folder / "candidate_seal.json"),
        "metadata_sha256": {str(path.relative_to(root)): sha256(path) for path in metadata_paths},
        "stage_shapes": old_complete["stage_shapes"], "stage_sha256": old_complete["stage_sha256"],
        "stage_snapshots_sha256": sha256(old / "stage_snapshots.npz"),
        "stage_paths": old_complete["stage_paths"],
    })
    progress(root, status="CELL_RECALIBRATED_SEALED", cell_id=cell.cell_id,
             elapsed_cell_seconds=round(time.monotonic()-start_time, 1))


def run(source, output):
    source, root = Path(source).resolve(), Path(output).resolve()
    cells = build_two_stencil_factorial(include_deployed_anchor=True)
    if any(not (source / "cells" / cell.cell_id / "numeric_complete.json").exists() for cell in cells):
        raise ValueError("the unchanged parent matrix must be numerically complete")
    pre = json.loads((source / "preflight.json").read_text())
    if sha256(LABEL_PATH) != pre["label_sha256"]:
        raise ValueError("parent label source changed")
    diagnostic = source / "calibration_grid_diagnostic.json"
    if not diagnostic.exists():
        raise ValueError("initialization-only evidence of the grid issue is required")
    config = json.loads((source / "protocol.json").read_text())
    if [asdict(cell) for cell in cells] != config["cells"]:
        raise ValueError("current operator definitions differ from the frozen parent matrix")
    config.update(experiment="two_stencil_factorial_v2_exact_calibration_development_replay",
        calibration_method="exact_nms_peak_order_statistics",
        calibration_amendment={"parent_campaign": str(source), "parent_protocol_sha256": sha256(source / "protocol.json"),
            "initialization_evidence_sha256": sha256(diagnostic),
            "reason": "sampled pixel quantile grid missed attainable calibration counts",
            "applied_uniformly_to_all25_cells_and_all_readouts": True,
            "operator_inputs_kernels_floors_unchanged": True,
            "parent_outcomes_had_been_computed": True,
            "confirmation_status": "development_replay_not_untouched_confirmation"})
    bound_code = [Path(__file__), Path(__file__).with_name("two_stencil_exact_calibration.py"),
        Path(__file__).with_name("two_stencil_evaluation.py"), Path(__file__).with_name("two_stencil_campaign.py"),
        Path(__file__).with_name("evaluation.py"), REPO / "neurobench/metrics/sparse_detection.py",
        REPO / "neurobench/algorithms/gamma_two_stencil.py"]
    code_hashes = {str(path.relative_to(REPO)): sha256(path) for path in bound_code}
    if root.exists():
        if not (root / "protocol.json").exists() or json.loads((root / "protocol.json").read_text()) != config:
            raise ValueError("new root required or original recalibration protocol must match exactly")
        if json.loads((root / "preflight.json").read_text())["code_sha256"] != code_hashes:
            raise ValueError("recalibration implementation changed since replay start")
    else:
        root.mkdir(parents=True)
        write_json(root / "protocol.json", config)
        write_json(root / "preflight.json", {**pre, "created_unix": time.time(),
            "parent_preflight_sha256": sha256(source / "preflight.json"), "code_sha256": code_hashes,
            "cached_stage_hashes_checked_per_cell_before_replay": True,
            "cpu_affinity": sorted(os.sched_getaffinity(0))})
        shutil.copy2(source / "preflight_projection_overlay.png", root / "preflight_projection_overlay.png")
        shutil.copy2(diagnostic, root / diagnostic.name)
        for path in bound_code:
            shutil.copy2(path, root / path.name)
    for cell in cells:
        _run_cell(source, root, cell)
    seals = {cell.cell_id: sha256(root / "cells" / cell.cell_id / "candidate_seal.json") for cell in cells}
    seal_path = root / "campaign_candidate_seal.json"
    if seal_path.exists():
        if json.loads(seal_path.read_text())["seals"] != seals:
            raise ValueError("campaign seal changed")
    else:
        write_json(seal_path, {"all25sealed_unix": time.time(), "seals": seals, "labels_used_for_scoring_or_selection": False})
    for cell in cells:
        _evaluate_cell(root, cell)
    results = []
    for cell in cells:
        results.extend(json.loads((root / "cells" / cell.cell_id / "numeric_complete.json").read_text())["summary_rows"])
    write_tsv(root / "results.tsv", results)
    progress(root, status="NUMERIC_COMPLETE_AUDIT_PENDING", completed_cells=25, total_cells=25,
             calibration_method="exact_nms_peak_order_statistics")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.output)


if __name__ == "__main__":
    main()
