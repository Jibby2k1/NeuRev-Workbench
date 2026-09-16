"""Serial report/audit orchestration after every matrix proposal is sealed."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from .two_stencil_campaign import LABEL_PATH, sha256, write_json, progress


def _audit_revision(value):
    if value is not None and (not value or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in value)):
        raise ValueError("audit revision must be a nonempty lowercase filename token")
    return value


def make_audit_configs(root, *, audit_revision=None):
    from .two_stencil_audit import audit_inventory_plan
    root = Path(root).resolve()
    audit_revision = _audit_revision(audit_revision)
    media_root = root / "scientific_audits"
    config_root = root / "audit_configs"
    if audit_revision is not None:
        media_root /= audit_revision
        config_root /= audit_revision
    protocol = json.loads((root / "protocol.json").read_text())
    preflight = json.loads((root / "preflight.json").read_text())
    if sha256(LABEL_PATH) != preflight["label_sha256"]:
        raise ValueError("label source changed after experiment preflight")
    limits_path = root / "report_artifacts/display_limits.json"
    limits = json.loads(limits_path.read_text())["limits"]
    records = []
    for cell in protocol["cells"]:
        cell_id = cell["cell_id"]
        folder = root / "cells" / cell_id
        complete = json.loads((folder / "numeric_complete.json").read_text())
        if complete.get("stage_snapshots_sha256") and sha256(folder / "stage_snapshots.npz") != complete["stage_snapshots_sha256"]:
            raise ValueError("stage snapshots changed after numerical completion")
        operator = json.loads((folder / "operator.json").read_text())
        candidate_path = folder / "readouts/Z/q1/candidates.tsv"
        if sha256(folder / "candidate_seal.json") != complete["candidate_seal_sha256"]:
            raise ValueError("candidate seal changed after numeric completion")
        seal = json.loads((folder / "candidate_seal.json").read_text())
        summaries = {}
        for q in protocol["target_proposals_per_frame"]:
            q_folder = folder / "readouts/Z" / f"q{q:g}"
            expected_candidates = [item for item in seal["candidate_files"]
                                   if item["readout"] == "Z" and item["q"] == q]
            if len(expected_candidates) != 1:
                raise ValueError(f"q{q:g} must have exactly one sealed Z candidate file")
            expected_candidate = expected_candidates[0]
            if root / expected_candidate["path"] != q_folder / "candidates.tsv" or sha256(q_folder / "candidates.tsv") != expected_candidate["sha256"]:
                raise ValueError(f"q{q:g} candidate table changed after scoring")
            summary = json.loads((q_folder / "summary.json").read_text())
            expected_summaries = [item for item in complete["summary_rows"]
                                  if item["readout"] == "Z" and item["target_proposals_per_frame"] == q]
            if len(expected_summaries) != 1 or summary != expected_summaries[0]:
                raise ValueError(f"q{q:g} numeric summary changed after label join")
            summaries[float(q)] = summary
        for path, expected in complete["metadata_sha256"].items():
            if sha256(root/path) != expected:
                raise ValueError("operator or calibration metadata changed after scoring")
        summary = summaries[1.0]
        representation = cell["input_representation"]
        sources = protocol["inputs"][representation]
        stage_paths = {"Raw": protocol["source_movie"]["path"], "X": sources["path"],
            **{name: str(root/path) for name, path in complete["stage_paths"].items()}}
        stage_hashes = {"Raw": protocol["source_movie"]["file_sha256"], "X": sources["file_sha256"], **complete["stage_sha256"]}
        operating = {**summary, "q": 1.0, "threshold_z": summary["threshold"],
                     "readout": "framewise_Z", "calibration_scope_ui": [2, 100], "application_scope_ui": [101, 2359]}
        config = {
            "output_root": str(media_root / cell_id),
            "stage_paths": stage_paths,
            "signed_stage_keys": [name for name, item in limits[representation].items() if item["signed"]],
            "display_limits": {name: [item["vmin"], item["vmax"]] for name, item in limits[representation].items()},
            "source_frames_ui": list(range(1, 2360)),
            "source_binding": {"cell_id": cell_id, "input_representation": representation,
                "operator": operator, "stage_sha256": stage_hashes,
                "candidates_sha256": sha256(candidate_path), "expert_occurrences_sha256": preflight["label_sha256"],
                "candidate_seal_sha256": complete["candidate_seal_sha256"],
                "campaign_candidate_seal_sha256": sha256(root / "campaign_candidate_seal.json"),
                "display_limits_source": str(limits_path), "display_limits_sha256": sha256(limits_path)},
            "operating_point": operating,
            "fullnumeric_q_paths": {f"{q:g}": str(folder / "readouts/Z" / f"q{q:g}") for q in protocol["target_proposals_per_frame"]},
            "candidates_path": str(candidate_path), "expert_occurrences": str(LABEL_PATH),
            "context_frames": 5, "fullfield_step": 5, "fps": 50.0,
        }
        if audit_revision is not None:
            config["source_binding"]["audit_revision"] = audit_revision
        config_path = config_root / f"{cell_id}.json"
        if config_path.exists() and json.loads(config_path.read_text()) != config:
            raise ValueError("audit configuration changed after it was frozen")
        write_json(config_path, config)
        plan = audit_inventory_plan(candidate_path, LABEL_PATH, config["source_frames_ui"])
        compact = {key: value for key, value in plan.items() if key.endswith("count") or key.endswith("total")}
        records.append({"cell_id": cell_id, "config_path": str(config_path),
                        "audit_revision": audit_revision, **compact})
    inventory_name = "audit_inventory_plan.json" if audit_revision is None else f"audit_inventory_plan_{audit_revision}.json"
    write_json(root / inventory_name, {"states": records, "audit_revision": audit_revision,
        "total_model_review_locations": sum(r["model_roi_count"] for r in records),
        "total_videos": sum(r["expected_video_count"] for r in records),
        "total_trace_figures": sum(r["expected_trace_figure_count"] for r in records),
        "total_closeup_source_frames": sum(r["closeup_video_frame_count_total"] for r in records),
        "audited_q": 1.0, "all25cells": True, "other_q_numeric_only": True})
    return records


def collect_audit_completion(root, records, *, verified_this_run=None):
    """Count only states verified against their frozen configuration and bytes.

    A successful renderer return already verified the full source inputs and
    generated or checked the complete artifact index in this invocation. Reuse
    that evidence rather than rereading those stage arrays a second time. Other
    claimed-complete states use the renderer's complete-only resume path, which
    checks the expected contract and all artifact/source hashes without media
    imports or rendering. A summary flag alone is never sufficient evidence.
    """
    from .two_stencil_audit import run_two_stencil_audit
    root = Path(root).resolve()
    verified_this_run = verified_this_run or {}
    complete = []
    for record in records:
        cell_id = record["cell_id"]
        config = json.loads(Path(record["config_path"]).read_text())
        output = Path(config["output_root"]).resolve()
        expected_root = root / "scientific_audits"
        revision = _audit_revision(record.get("audit_revision"))
        if revision is not None:
            expected_root /= revision
        if output != expected_root / cell_id:
            raise ValueError(f"{cell_id}: frozen audit configuration points to another state/revision")
        path = output / "summary.json"
        if not path.exists():
            continue
        value = json.loads(path.read_text())
        if not value.get("scientific_audit_complete"):
            continue
        status = json.loads((output / "status.json").read_text())
        if status.get("status") != "complete" or not status.get("scientific_audit_complete"):
            raise ValueError(f"{cell_id}: complete summary disagrees with audit status")
        if cell_id in verified_this_run:
            verified = verified_this_run[cell_id]
            evidence = "successful_renderer_return_this_invocation"
        else:
            verified = run_two_stencil_audit(**config)
            evidence = "existing_complete_contract_artifacts_and_sources_reverified"
        if verified != value or not verified.get("scientific_audit_complete"):
            raise ValueError(f"{cell_id}: audit summary differs from verified renderer result")
        complete.append({"cell_id": cell_id, "audit_revision": revision,
                         "output_root": str(output), **value, "completion_verification": evidence,
                         "artifact_index_sha256": sha256(output / "artifact_index.json"),
                         "run_contract_sha256": sha256(output / "run_contract.json")})
    return complete


def audit(root, *, cell_ids=None, audit_revision=None):
    from .two_stencil_audit import run_two_stencil_audit
    root = Path(root).resolve()
    audit_revision = _audit_revision(audit_revision)
    records = make_audit_configs(root, audit_revision=audit_revision)
    all_records = list(records)
    fixed_order = ["deployed_signed_point_gamma_g7_disk",
                   "difference_signed__direct_gamma_g0_square",
                   "difference_signed__serial_gamma_g0_square"]
    records.sort(key=lambda record: (fixed_order.index(record["cell_id"]) if record["cell_id"] in fixed_order else 3, record["cell_id"]))
    if cell_ids:
        if set(cell_ids) - {record["cell_id"] for record in records}:
            raise ValueError("unknown audit cell")
        records = [record for record in records if record["cell_id"] in cell_ids]
    verified_this_run = {}
    for record in records:
        start = time.monotonic()
        progress(root, status="AUDIT_RENDERING", cell_id=record["cell_id"],
                 model_review_locations=record["model_roi_count"], expected_videos=record["expected_video_count"])
        result = run_two_stencil_audit(**json.loads(Path(record["config_path"]).read_text()))
        verified_this_run[record["cell_id"]] = result
        progress(root, status="CELL_AUDIT_COMPLETE", cell_id=record["cell_id"],
                 duration_seconds=round(time.monotonic()-start, 1), observed_videos=result["video_count"])
    complete = collect_audit_completion(root, all_records, verified_this_run=verified_this_run)
    completion = {"q1_scientific_audit_complete": len(complete) == 25, "audit_revision": audit_revision,
        "completed_states": len(complete), "expected_states": 25, "states": complete,
        "other_q_media_audited": False, "external_confirmation_complete": False,
        "biological_precision_identified": False, "winner_selected": False}
    if audit_revision is not None:
        write_json(root / f"audit_completion_{audit_revision}.json", completion)
    write_json(root / "audit_completion.json", completion)
    progress(root, status="Q1_SCIENTIFIC_AUDIT_COMPLETE" if len(complete) == 25 else "PARTIAL_AUDIT_COMPLETE",
             completed_audits=len(complete), total_audits=25)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("report", "audit-preflight", "audit"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cell", action="append")
    parser.add_argument("--audit-revision", help="New namespace for audit configurations, media, and inventory; e.g. encoding_v2")
    arguments = parser.parse_args()
    if arguments.mode == "report":
        from .two_stencil_report import generate_report
        print(json.dumps(generate_report(arguments.output)), flush=True)
    elif arguments.mode == "audit-preflight":
        records = make_audit_configs(arguments.output, audit_revision=arguments.audit_revision)
        print(json.dumps({"cells": len(records), "videos": sum(r["expected_video_count"] for r in records)}), flush=True)
    else:
        audit(arguments.output, cell_ids=arguments.cell, audit_revision=arguments.audit_revision)


if __name__ == "__main__":
    main()
