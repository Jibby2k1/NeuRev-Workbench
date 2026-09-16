"""Render readable companion cards from frozen small trajectory artifacts.

Only operator row labels differ from the original focused card rendering.
This module never opens full-recording stage arrays or reruns an operator,
calibration, NMS, matching, or diagnostic classification.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import numpy as np

from . import two_stencil_focused_diagnostics as focused


def rerender_focused_cards(campaign_root, *, output_dir=None):
    root = Path(campaign_root).resolve()
    source = root / "focused_diagnostics"
    destination = Path(output_dir).resolve() if output_dir else root / "focused_diagnostics_readable_v2"
    if destination.exists():
        raise FileExistsError(f"new companion output directory required: {destination}")
    summary = focused._json(source/"summary.json")
    original_qa = focused._json(source/"visual_qa.json")
    focused._verify_file(source/"summary.json", original_qa["source_bindings"]["summary.json"])
    manifest = {row["path"]: row["sha256"] for row in summary["artifacts"]}
    bound = {"summary.json": focused._sha256(source/"summary.json"),
             "visual_qa.json": focused._sha256(source/"visual_qa.json")}
    for name in ("provenance.json", "card_selection.tsv", "occurrence_stage_observations.tsv"):
        bound[name] = focused._verify_file(source/name, manifest[name])
    focused._verify_file(source/"card_selection.tsv", focused.FROZEN_SELECTION_SHA256)
    protocol_hash = focused._json(source/"provenance.json")["campaign_protocol_sha256"]
    focused._verify_file(root/"protocol.json", protocol_hash)
    protocol = focused._json(root/"protocol.json")
    cells = [c for c in protocol["cells"] if c["is_deployed_anchor"] or (
        c["spec"]["reference_family"] == "gamma" and c["spec"]["guard_radius_px"] == 0
        and c["spec"]["support_geometry"] == "square")]
    if len(cells) != 7:
        raise ValueError("companion rendering requires the same seven declared arms")
    cell_ids = {c["cell_id"] for c in cells}
    selection = focused._read_tsv(source/"card_selection.tsv")
    selected_ids = {r["observation_id"] for r in selection}
    all_rows = focused._read_tsv(source/"occurrence_stage_observations.tsv")
    if len(all_rows) != 1975 or len(selection) != 14 or len(selected_ids) != 14:
        raise ValueError("original focused artifact inventory differs from the complete fixed design")
    records = {(r["operator_cell_id"], r["observation_id"]): dict(r) for r in all_rows
               if r["operator_cell_id"] in cell_ids and r["observation_id"] in selected_ids}
    if len(records) != 98:
        raise ValueError("each of 14 cards requires the same seven saved arm observations")
    packets = {}
    for cell in cells:
        cell_id = cell["cell_id"]
        trajectory = f"trajectories/{cell_id}.npz"
        bound[trajectory] = focused._verify_file(source/trajectory, manifest[trajectory])
        with np.load(source/trajectory, allow_pickle=False) as archive:
            for observation in selected_ids:
                row = records[(cell_id, observation)]
                if row["trajectory_npz_path"] != trajectory:
                    raise ValueError("saved trajectory path differs from the declared cell")
                prefix = row["trajectory_npz_prefix"] + "__"
                packet = {name[len(prefix):]: archive[name].copy() for name in archive.files if name.startswith(prefix)}
                for key in ("threshold_z", "scale_floor"):
                    row[key] = float(row[key])
                expected_frames = np.arange(int(row["source_start_ui"]), int(row["source_stop_ui"])+1)
                if not np.array_equal(packet["source_frame_ui"], expected_frames):
                    raise ValueError("saved trajectory differs from the inclusive burst window")
                if any(value.shape != expected_frames.shape or not np.isfinite(value).all() for value in packet.values()):
                    raise ValueError("invalid saved trajectory shape or nonfinite observations")
                if float(packet["Z"].max()) != float(row["Z_at_rounded_pixel_peak_Z"]):
                    raise ValueError("trajectory peak differs from original scalar observation")
                packets[(cell_id, observation)] = packet
    destination.mkdir(parents=True)
    code_paths = (Path(__file__), Path(focused.__file__))
    for path in code_paths:
        shutil.copy2(path, destination/path.name)
    provenance = {
        "source_focused_diagnostics": str(source), "original_artifacts_verified": bound,
        "campaign_protocol_sha256": protocol_hash,
        "render_code_sha256": {path.name: focused._sha256(path) for path in code_paths},
        "change": "Shorter operator row labels only; original trajectories, axes, thresholds, categories and 14-card selection retained.",
        "row_label_definitions": {"Current": "conditioned current frame", "Signed": "signed adjacent difference"},
        "full_stage_arrays_opened": False, "scoring_or_matching_rerun": False,
        "original_artifacts_modified": False, "scientific_promotion": False,
    }
    focused._write_json(destination/"provenance.json", provenance)
    cards = focused._render_cards(destination, cells, selection, records, packets)
    result = {"status": "READABLE_COMPANION_RENDERED_VISUAL_QA_PENDING", "card_count": len(cards),
        "same_frozen_selection": True, "scientific_audit_complete": False, "scientific_promotion": False,
        "cards": [{"path": name, "sha256": focused._sha256(destination/name)} for name in cards],
        "provenance_sha256": focused._sha256(destination/"provenance.json")}
    focused._write_json(destination/"summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="completed campaign root")
    parser.add_argument("--companion-output", type=Path)
    args = parser.parse_args()
    result = rerender_focused_cards(args.output, output_dir=args.companion_output)
    print(result, flush=True)


if __name__ == "__main__":
    main()
