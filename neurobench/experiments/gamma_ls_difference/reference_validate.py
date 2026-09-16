"""One-pass source/artifact closure for all 189 reference-study q1 audits.

Uses the frozen FileVerifier and completed-audit file lists. The private cache
never survives a call; all observed files and symlink targets are checked again
before publishing PASS. Rendering/decode evidence is inherited by verified hash.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import time

from .followup_validate import (FileVerifier, METADATA, CANDIDATE_FIELDS,
                               _cell_seal, _workers, _write, require, verify_audit_files)
from .reference_media import (BASELINE_ARM, NUMERIC_NAMES, STAGE_NAMES, TOTAL_CELLS,
                              _api, _mutable, common_crowding_limits, matrix, reused)

ALL_STAGES = ("Raw", "Input", "A", "M", "Spread", "C", "Z", "Score")
OP_FIELDS = ("threshold", "threshold_z", "threshold_id", "threshold_frozen_from_calibration_only",
             "setup_source_frames_ui", "scale_floor", "target_proposals_per_frame", "eligible_area_px",
             "application_source_start_ui", "application_source_stop_ui", "application_frame_count")
METRIC_REPRESENTATION_FIELDS = frozenset(("threshold_z", "threshold_label", "no_output_endpoint", "metric_candidate_sha256"))


def _scientific_payload(value):
    """Compare the same selected classifications, despite full-prefix vs filtered input.

The actual native cutoff is checked through calibration/threshold_plan instead.
All match identities, distances, proposal coordinates, scores, frame counts and
delays remain in this comparison. Only metric-input representation fields differ.
"""
    if isinstance(value, dict):
        return {k: _scientific_payload(v) for k, v in value.items() if k not in METRIC_REPRESENTATION_FIELDS}
    if isinstance(value, list):
        return [_scientific_payload(v) for v in value]
    return value


def _event_payload(rows):
    result = {}
    for row in rows:
        key = (row["event_id"], row["threshold_id"], float(row["match_radius_px"]))
        require(key not in result, "Duplicate event/cutoff/radius row")
        result[key] = _scientific_payload(row)
    return result


def _binding_records(value):
    if isinstance(value, dict):
        if "path" in value and "sha256" in value:
            yield value
        else:
            for child in value.values():
                yield from _binding_records(child)
    elif isinstance(value, list):
        for child in value:
            yield from _binding_records(child)


def _same_binding(left, right):
    return (Path(left["path"]).resolve() == Path(right["path"]).resolve()
            and left["sha256"] == right["sha256"]
            and left.get("size_bytes") == right.get("size_bytes"))


def verify_cell(cell, folder, verifier):
    """Check every sealed stage, dataset/setup source, and both-radius curve join."""
    seal, op, candidates = _cell_seal(cell, verifier)
    require(set(ALL_STAGES) <= set(seal["stages"]), "Incomplete scientific stage inventory")
    for record in seal["stages"].values():
        verifier.verify(record)
    require(seal["stages"]["Score"]["sha256"] == seal["stages"]["Z"]["sha256"], "Model Score must be native Z")
    authority = {Path(r["path"]).resolve(): r for r in seal["dataset_bindings"]}
    for name in ("metadata.json", "experts.json", "active.json", "prepared.json"):
        target = (folder / name).resolve()
        require(target in authority, f"Dataset source missing from cell seal: {name}")
        verifier.verify(authority[target])
    prepared = verifier.read_json(folder / "prepared.json")
    require(prepared.get("status") == "PASS", "Dataset preparation is incomplete")
    # This includes full-halo raw/conditioned input, not only displayed crops.
    for record in _binding_records(prepared):
        verifier.verify(record)
    metadata = verifier.read_json(folder / "metadata.json")
    require(metadata["truth_mode"] == "fully_synthetic", "Reference study has synthetic truth only")
    plan = verifier.read_json(cell / "threshold_plan.json")
    ids = [r["threshold_id"] for r in plan]
    require(len(ids) == len(set(ids)) == 10 and "q1" in ids, "Expected ten distinct setup cutoffs")
    curves = verifier.read_json(cell / "curves.json")
    keys = [(r["threshold_id"], float(r["match_radius_px"])) for r in curves["curve_rows"]]
    require(len(keys) == len(set(keys)) == 20
            and set(keys) == {(name, radius) for name in ids for radius in (2., 6.)},
            "Cell curves must include both radii for every setup cutoff")
    evaluated = verifier.read_json(cell / "evaluated.json")
    output_paths = {Path(r["path"]).resolve() for r in evaluated["outputs"]}
    require(all((cell / name).resolve() in output_paths
                for name in ("curves.json", "operating_metrics_r2.json", "operating_metrics_r6.json")),
            "Both operating radii and curves must be sealed by evaluation")
    return seal, op, candidates, curves, metadata


def verify_reuse(new, old, new_cell, old_cell, verifier):
    """Exact scientific-state reuse; case/arm naming metadata may differ."""
    new_seal, new_op, new_rows, new_curves, _ = new
    old_seal, old_op, old_rows, old_curves, _ = old
    for name in ALL_STAGES:
        require(new_seal["stages"][name]["sha256"] == old_seal["stages"][name]["sha256"],
                f"Reused scientific stage differs: {name}")
    require(len(new_rows) == len(old_rows)
            and all(all(a.get(k) == b.get(k) for k in CANDIDATE_FIELDS) for a, b in zip(new_rows, old_rows)),
            "Reused candidate rows differ")
    for key in OP_FIELDS:
        require(new_op[key] == old_op[key], f"Reused operating point differs: {key}")
    require(verifier.read_json(new_cell / "threshold_plan.json") == verifier.read_json(old_cell / "threshold_plan.json"),
            "Reused ten-cutoff plan differs")
    for radius in (2, 6):
        require(_scientific_payload(verifier.read_json(new_cell / f"operating_metrics_r{radius}.json"))
                == _scientific_payload(verifier.read_json(old_cell / f"operating_metrics_r{radius}.json")),
                f"Reused q1 r{radius} metrics differ")
    require(_event_payload(new_curves["event_rows"]) == _event_payload(old_curves["event_rows"]), "Reused all-cutoff event rows differ")
    names = ("threshold_id", "match_radius_px", "threshold", "proposal_count",
             "true_positive_count", "false_positive_count", "false_negative_count", "active_region_frame_count",
             "recovered_event_count", "event_count", "precision", "framewise_sensitivity", "deadline_rows",
             "application_frame_count", "eligible_area_px")
    keyed = lambda curves: {(r["threshold_id"], float(r["match_radius_px"])): r for r in curves["curve_rows"]}
    nr, oldr = keyed(new_curves), keyed(old_curves)
    require(set(nr) == set(oldr) and all(all(nr[k].get(name) == oldr[k].get(name) for name in names) for k in nr),
            "Reused all-cutoff metrics differ")


def verify_inventory(summary, status, validation, inventory, wanted, candidate_count):
    require(status.get("status") == "complete" and status.get("scientific_audit_complete") is True,
            "Audit status is incomplete")
    require(summary.get("scientific_audit_complete") is True and validation.get("status") == "passed"
            and validation.get("failures") == [] and validation.get("scientific_audit_complete") is True
            and inventory.get("complete") is True, "Audit validation is incomplete")
    for actual, expected in (("video_count", "expected_videos"), ("expert_roi_count", "expert_rois"),
                             ("model_roi_count", "model_rois"), ("expert_occurrence_count", "expert_occurrences")):
        require(summary[actual] == wanted[expected], f"Audit inventory differs: {actual}")
    require(summary["model_proposal_count"] == candidate_count
            and summary["comparison_trace_count"] == summary["expert_occurrence_count"],
            "Proposal/comparison inventory differs")


def _validate(root, verifier):
    worker_records = _workers(root, verifier, TOTAL_CELLS)
    preflight = verifier.read_json(root / "preflight.json")
    require(preflight["status"] == "PASS", "Preflight is incomplete")
    verifier.verify(dict(path=str(root / "protocol.json"), sha256=preflight["protocol_sha256"]))
    protocol = verifier.read_json(root / "protocol.json")
    require(protocol["expected_cells"] == TOTAL_CELLS, "Protocol matrix size differs")
    for record in protocol["code_bindings"]:
        verifier.verify(record)
    for key in ("kernel_bindings", "baseline_sources"):
        for record in protocol[key]:
            verifier.verify(record)
    for key in ("baseline_protocol", "baseline_completion"):
        verifier.verify(protocol[key])
    require(verifier.read_json(protocol["baseline_completion"]["path"])["status"] == "PASS", "Baseline is incomplete")
    baseline = Path(protocol["baseline_protocol"]["path"]).resolve().parent
    states = matrix(protocol)
    keys = {(c["case_id"], c["arm_id"]) for c in states}
    reused_keys = {(c["case_id"], c["arm_id"]) for c in states if reused(c)}
    for filename in ("computation_complete.json", "evaluation_complete.json", "datasets_complete.json"):
        value = verifier.read_json(root / filename)
        require(value["status"] == "PASS", f"Incomplete {filename}")
        if filename != "datasets_complete.json":
            require(value["cells"] == TOTAL_CELLS, f"Wrong cell count: {filename}")
        for record in _binding_records(value):
            verifier.verify(record)
    require(verifier.read_json(root / "evaluation_complete.json")["curve_rows"] == 3780, "Expected 3780 numerical curve rows")
    forecast = verifier.read_json(root / "audit_forecast.json")
    fkeys = [(r["case_id"], r["arm_id"]) for r in forecast["cells"]]
    require(forecast["status"] == "PASS" and len(fkeys) == len(set(fkeys)) == TOTAL_CELLS and set(fkeys) == keys,
            "Forecast must cover the exact 189-state matrix")
    verifier.verify(forecast["display_contract"])
    expected = dict(zip(fkeys, forecast["cells"]))
    replication = verifier.read_json(root / "baseline_replication.json")
    rkeys = [(r["case_id"], r["arm_id"]) for r in replication["cells"]]
    require(replication["status"] == "PASS" and len(rkeys) == len(set(rkeys)) == 12 and set(rkeys) == reused_keys
            and all(all(r.get(k) is True for k in ("all_thresholds_equal", "metrics_equal", "stages_equal", "candidates_equal"))
                    for r in replication["cells"]),
            "Central crowding replication must cover exactly twelve states")
    display = verifier.read_json(root / "display_contract.json")
    require(display["protocol_sha256"] == preflight["protocol_sha256"], "Display protocol differs")
    verifier.verify(display["baseline_display"])
    common = common_crowding_limits(verifier.read_json(display["baseline_display"]["path"])["cases"],
                                    sorted({c for c, _ in reused_keys}))
    require(set(display["cases"]) == {c["case_id"] for c in states}, "Display matrix differs")
    for record in display["preview_sources"]:
        verifier.verify(record)
    for case, limits in display["cases"].items():
        require(set(limits) == set(STAGE_NAMES) and limits["Score"] == [-8., 8.], "Incorrect display stage/Score units")
        if case in {c for c, _ in reused_keys}:
            require(limits == common, "Crowding display ranges differ from reused baseline")
        else:
            require(limits == display["exception"]["null_limits"], "Null references must share one display range")
    display_binding = verifier.binding(root / "display_contract.json")
    all_curves, result, fixed_inputs = [], [], {}
    api = _api()
    for index, cell in enumerate(states):
        case, arm = cell["case_id"], cell["arm_id"]
        folder, current = root / "datasets" / case, root / "cells" / case / arm
        new = verify_cell(current, folder, verifier)
        new_seal, new_op, new_candidates, curves, meta = new
        all_curves.extend(curves["curve_rows"])
        fixed = tuple(new_seal["stages"][k]["sha256"] for k in ("Raw", "Input", "A"))
        require(case not in fixed_inputs or fixed_inputs[case] == fixed, "Raw/Input/target A changed across references")
        fixed_inputs[case] = fixed
        audit = root / "audits" / case / arm
        if reused(cell):
            old_cell = baseline / "cells" / case / BASELINE_ARM
            require(audit.resolve(strict=True) == (baseline / "audits" / case / BASELINE_ARM).resolve(strict=True),
                    "Reused audit does not resolve to its original baseline state")
            old = verify_cell(old_cell, baseline / "datasets" / case, verifier)
            verify_reuse(new, old, current, old_cell, verifier)
            owner_cell, owner_seal, owner_op = old_cell, old[0], old[1]
        else:
            require(audit.resolve(strict=True) == audit.absolute(), "New audit unexpectedly aliases another root")
            owner_cell, owner_seal, owner_op = current, new_seal, new_op
        sources = verify_audit_files(audit, verifier)
        summary = verifier.read_json(audit / "summary.json")
        verify_inventory(summary, verifier.read_json(audit / "status.json"), verifier.read_json(audit / "validation.json"),
                         verifier.read_json(audit / "inventory.json"), expected[(case, arm)], len(new_candidates))
        contract = verifier.read_json(audit / "run_contract.json")
        require(contract["operating_point"] == owner_op == summary["operating_point"], "Audit operating point differs")
        require(contract["display_limits"] == display["cases"][case], "Audit display contract differs")
        frames, _, _ = api.frame_sets(meta)
        require(contract["source_frames_ui"] == frames, "Audited source-frame mapping differs")
        sb = contract["source_binding"]
        require(sources["source_binding"] == sb, "Audit source-manifest contract differs")
        require(sb["candidates_sha256"] == owner_seal["audit_candidates"]["sha256"]
                and sb["candidate_seal_sha256"] == verifier.binding(owner_cell / "sealed.json")["sha256"],
                "Audit candidate seal differs")
        require(sb["expert_occurrences_sha256"] == verifier.binding(folder / "experts.json")["sha256"]
                and sb["original_source_offset_xy"] == meta["original_source_offset_xy"]
                and sb["truth_mode"] == meta["truth_mode"], "Audit truth/coordinate mapping differs")
        if not reused(cell):
            require(sb["display_contract_sha256"] == display_binding["sha256"], "New audit display SHA differs")
        require(set(contract["stage_paths"]) == set(STAGE_NAMES), "Every audit needs Raw/Input/A/Score")
        for name in STAGE_NAMES:
            record = owner_seal["stages"][name]
            require(Path(contract["stage_paths"][name]).resolve() == Path(record["path"]).resolve()
                    and sb["stage_sha256"][name] == record["sha256"], "Displayed stage differs from seal")
        displayed_paths = {Path(p).resolve() for p in contract["stage_paths"].values()}
        needed = {k for k in NUMERIC_NAMES if Path(owner_seal["stages"][k]["path"]).resolve() not in displayed_paths}
        require(set(sb["numeric_stage_bindings"]) == needed, "Missing linked scientific stages")
        for name in needed:
            require(_same_binding(sb["numeric_stage_bindings"][name], owner_seal["stages"][name]),
                    "Linked numeric stage differs from seal")
        npaths = {Path(r["path"]).resolve() for r in sources["numeric_state_sources"]}
        require(all((owner_cell / name).resolve() in npaths for name in ("curves.json", "calibration.json")),
                "All-threshold numerical evidence is not attached")
        result.append(dict(cell, reused=reused(cell), summary=summary,
                           metadata_bindings={name: verifier.binding(audit / name) for name in METADATA}))
        if (index + 1) % 9 == 0:
            print(dict(status="AUDITS_VERIFIED", cells=index + 1, total=189,
                       unique_files=verifier.unique_file_count, bytes_hashed=verifier.bytes_hashed), flush=True)
    require(all_curves == verifier.read_json(root / "all_curves.json"), "All-curves aggregate differs from sealed cell rows")
    require(sum(r["summary"]["video_count"] for r in result) == forecast["total_videos"], "Total video count differs")
    return dict(status="PASS", cells=189, new_cells=177, reused_cells=12, audits=result, worker_receipts=worker_records,
                protocol=verifier.binding(root / "protocol.json"), evaluation=verifier.binding(root / "evaluation_complete.json"),
                display_contract=display_binding, forecast=verifier.binding(root / "audit_forecast.json"),
                baseline_replication=verifier.binding(root / "baseline_replication.json"), all_curves=verifier.binding(root / "all_curves.json"),
                reuse_metric_comparison="New metrics classify the complete positive NMS prefix at the native cutoff; baseline metrics classify a prefiltered subset at zero. Comparison excludes only threshold_z/threshold_label/no_output_endpoint/metric_candidate_sha256, while separately checking the exact native ten-cutoff plan and all q1 selected classifications, coordinates, scores, match identities, distances, frame counts and delays. Event/cutoff/radius rows are aligned by identity, not list order.")


def validate(root=None):
    root = _mutable(root if root is not None else _api().ROOT)
    verifier, started = FileVerifier(), time.monotonic()
    code = [verifier.binding(Path(__file__).with_name(name)) for name in
            ("reference_validate.py", "reference_media.py", "followup_validate.py", "followup_audit.py", "two_stencil_audit.py")]
    try:
        result = _validate(root, verifier)
        verifier.assert_unchanged()
        result.update(validated_utc=datetime.now(timezone.utc).isoformat(), validation_code=code,
                      unique_hash_count=verifier.unique_file_count, bytes_hashed=verifier.bytes_hashed,
                      cached_binding_checks=verifier.cache_hits, runtime_seconds=time.monotonic() - started,
                      verification_scope="All189 complete q1 source/artifact inventories, all sealed stages/full-halo dataset sources, both radii/all10 cutoffs, twelve exact central reuses and worker receipts verified. The private one-pass resolved-file SHA cache requires unchanged inode/size/mtime/ctime/alias targets at reuse and closure. Full media decoding, RGB equality and marker purity are inherited from hash-verified per-cell validation, not rerun. Model sites remain review locations.")
        _write(root / "audit_complete.json", result)
        return result
    except Exception as error:
        failure = dict(status="FAIL", cells=0, scientific_audit_complete=False, error_type=type(error).__name__,
                       error=str(error), validation_code=code, unique_hash_count=verifier.unique_file_count,
                       bytes_hashed=verifier.bytes_hashed, runtime_seconds=time.monotonic() - started)
        _write(root / "audit_validation_failure.json", failure)
        if (root / "audit_complete.json").exists():
            _write(root / "audit_complete.json", failure)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=None)
    args = parser.parse_args()
    value = validate(args.root)
    print({k: value[k] for k in ("status", "cells", "new_cells", "reused_cells", "unique_hash_count", "bytes_hashed")}, flush=True)
