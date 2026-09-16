"""Complete q1 media for the reference-radius/order development matrix.

Twelve unchanged central crowding audits are linked to their frozen originals;
every other state uses the existing lossless, inventory-complete audit renderer.
"""
from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path

from .followup_validate import require

TOTAL_CELLS = 189
REUSED_CELLS = 12
BASELINE_ARM = "level_Z__w3"
STAGE_NAMES = ("Raw", "Input", "A", "Score")
NUMERIC_NAMES = ("M", "Spread", "C", "Z")


def _api():
    from . import reference_study
    return reference_study


def _mutable(root):
    root = Path(root).resolve()
    if (root / "completion_manifest.json").exists():
        raise FileExistsError("Completed study roots are immutable")
    return root


def reused(cell):
    return cell.get("reused_audit") is True


def matrix(protocol):
    states = _api().cells(protocol)
    keys = [(c["case_id"], c["arm_id"]) for c in states]
    require(len(keys) == len(set(keys)) == TOTAL_CELLS, "Expected exactly 189 distinct reference states")
    require(len({c["case_id"] for c in states}) == 21, "Expected 21 datasets")
    require(sum(reused(c) for c in states) == REUSED_CELLS, "Expected exactly 12 reused audits")
    by_case = {}
    for cell in states:
        require(cell["study"] == "reference" and cell["input_mode"] == "level"
                and cell["readout"] == "Z" and cell["window"] == 3
                and cell["calibration_method"] == "global", "Reference media semantics differ")
        require(not reused(cell) or cell["arm_id"] == "mean1_n9", "Only the central reference may reuse media")
        by_case.setdefault(cell["case_id"], set()).add(cell["arm_id"])
    expected = {f"mean{mean}_n{n}" for mean in ("2of3", "1", "4of3") for n in (3, 9, 15)}
    require(all(arms == expected for arms in by_case.values()), "Reference grid must be the same nine arms per dataset")
    return states


def common_crowding_limits(existing, cases):
    """Use actual old display values; never round them or infer from new scores."""
    values = [dict(Raw=existing[c]["Raw"], Input=existing[c]["level"]["X"],
                   A=existing[c]["level"]["A"], Score=existing[c]["level"]["Z"]) for c in cases]
    require(len(values) == 12 and all(v == values[0] for v in values),
            "The 12 reused crowding states must have identical frozen display ranges")
    require(values[0]["Score"] == [-8., 8.], "The frozen Z display range differs")
    return copy.deepcopy(values[0])


def extend_null_limits(common, previews):
    """One shared null-only extension, with X/A in the same native units."""
    low, high = common["Raw"]
    level_high = max(common["Input"][1], common["A"][1])
    for row in previews:
        require(all(math.isfinite(float(row[k])) for k in ("raw_low", "raw_high", "input_high", "a_high")),
                "Nonfinite display preview")
        low = min(low, row["raw_low"])
        high = max(high, row["raw_high"])
        level_high = max(level_high, row["input_high"], row["a_high"])
    require(low < high and level_high > 0, "Invalid display extent")
    return dict(Raw=[low, high], Input=[0., level_high], A=[0., level_high], Score=[-8., 8.])


def display(root):
    root = _mutable(root)
    api = _api()
    protocol = api.load(root)
    states = matrix(protocol)
    path = root / "display_contract.json"
    if path.exists():
        saved = api.read(path)
        require(saved["protocol_sha256"] == api.sha256(root / "protocol.json"), "Display protocol changed")
        api.verify(saved["baseline_display"])
        return saved
    import numpy as np
    baseline_display = Path(protocol["baseline_protocol"]["path"]).resolve().parent / "display_contract.json"
    existing = api.read(baseline_display)["cases"]
    crowding = sorted({c["case_id"] for c in states if reused(c)})
    nulls = sorted({c["case_id"] for c in states} - set(crowding))
    require(len(nulls) == 9, "Expected nine new null datasets")
    common = common_crowding_limits(existing, crowding)
    previews, sources = [], []
    for case in nulls:
        seal = api.read(root / "cells" / case / "mean1_n9" / "sealed.json")
        row = dict(case_id=case)
        for name in ("Raw", "Input", "A"):
            record = seal["stages"][name]
            values = np.load(record["path"], mmap_mode="r")[::5]
            require(values.ndim == 3 and values.size > 0, "Display preview needs a nonempty TYX stage")
            sources.append(dict(case_id=case, stage=name, **record))
            if name == "Raw":
                row["raw_low"], row["raw_high"] = map(float, np.percentile(values, [1, 99.8]))
            else:
                row["input_high" if name == "Input" else "a_high"] = float(np.percentile(values, 99.8))
            del values
        previews.append(row)
    null_limits = extend_null_limits(common, previews)
    result = dict(schema_version=1, protocol_sha256=api.sha256(root / "protocol.json"),
                  baseline_display=api.binding(baseline_display),
                  cases={case: copy.deepcopy(common if case in crowding else null_limits)
                         for case in sorted(set(crowding + nulls))},
                  crowding_cases=crowding, null_cases=nulls, preview_sources=sources, previews=previews,
                  preview_rule="Every fifth available source sample, numpy percentiles 1/99.8 for Raw and 99.8 for Input/A; display only, no score or label selection.",
                  exception=dict(applied=null_limits != common,
                      reason="Preserve frozen crowding audit displays exactly; all null datasets share one pooled extension to show nuisance excursions. Compare references within datasets on identical scales.",
                      crowding_limits=common, null_limits=null_limits),
                  rule="Raw/Input/A fixed across all nine references. The 12 crowding datasets share their unchanged baseline ranges; all nine null datasets share the declared range. Input/A have native fluorescence units. Score is direct Z, dimensionless, fixed [-8,8]. Gray pixels are display transforms of float arrays; scientific master RGB composites are lossless.")
    api.write_json(path, result)
    return result


def forecast(root):
    root = _mutable(root)
    api = _api()
    protocol = api.load(root)
    require(api.read(root / "evaluation_complete.json") == dict(status="PASS", cells=189, curve_rows=3780),
            "Evaluate every reference cell and both radii first")
    display(root)
    from .followup_audit import audit_inventory_plan
    rows = []
    for cell in matrix(protocol):
        folder = root / "datasets" / cell["case_id"]
        out = root / "cells" / cell["case_id"] / cell["arm_id"]
        frames, _, _ = api.frame_sets(api.read(folder / "metadata.json"))
        plan = audit_inventory_plan(api.read(out / "audit_candidates.json"), api.read(folder / "experts.json"), frames)
        rows.append(dict(cell, reused=reused(cell), expected_videos=plan["expected_video_count"],
                         expert_rois=plan["expert_roi_count"], model_rois=plan["model_roi_count"],
                         expert_occurrences=plan["expert_occurrence_count"]))
    value = dict(status="PASS", cells=rows, new_cells=177, reused_cells=12,
                 new_videos=sum(r["expected_videos"] for r in rows if not r["reused"]),
                 total_videos=sum(r["expected_videos"] for r in rows), display_contract=api.binding(root / "display_contract.json"))
    api.write_json(root / "audit_forecast.json", value)
    print({k: v for k, v in value.items() if k != "cells"}, flush=True)
    return value


def media(root, worker=0, workers=3):
    root = _mutable(root)
    require(type(worker) is int and type(workers) is int and 0 <= worker < workers <= 3,
            "At most three disjoint integer-indexed media workers")
    api = _api()
    protocol = api.load(root)
    states = matrix(protocol)
    forecast_record = api.read(root / "audit_forecast.json")
    require(forecast_record["status"] == "PASS" and len(forecast_record["cells"]) == TOTAL_CELLS,
            "A complete audit forecast is required")
    api.verify(forecast_record["display_contract"])
    limits = api.read(root / "display_contract.json")["cases"]
    baseline = Path(protocol["baseline_protocol"]["path"]).resolve().parent
    replication = api.read(root / "baseline_replication.json")
    require(replication["status"] == "PASS" and len(replication["cells"]) == REUSED_CELLS,
            "Central crowding replication must pass before reuse")
    from .followup_audit import run_spatiotemporal_audit
    count = 0
    for index, cell in enumerate(states):
        if index % workers != worker:
            continue
        case, arm = cell["case_id"], cell["arm_id"]
        folder, out = root / "datasets" / case, root / "cells" / case / arm
        audit = root / "audits" / case / arm
        if reused(cell):
            old = baseline / "audits" / case / BASELINE_ARM
            contract = api.read(old / "run_contract.json")
            require(contract["display_limits"] == limits[case], "Reused audit display limits differ")
            audit.parent.mkdir(parents=True, exist_ok=True)
            api.link(old, audit)
        else:
            seal = api.read(out / "sealed.json")
            meta = api.read(folder / "metadata.json")
            frames, _, _ = api.frame_sets(meta)
            stages = {k: seal["stages"][k]["path"] for k in STAGE_NAMES}
            source = dict(stage_sha256={k: seal["stages"][k]["sha256"] for k in STAGE_NAMES},
                candidates_sha256=seal["audit_candidates"]["sha256"], expert_occurrences_sha256=api.sha256(folder / "experts.json"),
                candidate_seal_sha256=api.sha256(out / "sealed.json"), display_contract_sha256=api.sha256(root / "display_contract.json"),
                numeric_stage_bindings={k: seal["stages"][k] for k in NUMERIC_NAMES if seal["stages"][k]["path"] not in stages.values()},
                original_source_offset_xy=meta["original_source_offset_xy"], truth_mode=meta["truth_mode"], cell=cell,
                stage_semantics="Raw fluorescence; Input conditioned activity level; A unchanged target average; Score direct unmasked Z=(A-M)/max(reference spread,frozen floor). M/Spread/C/Z arrays linked numerically. Frozen global cutoff after 3x3 local maxima and 6px separation. Model sites are review locations, not inferred neuronal identities; nearest and assigned sites remain distinct.")
            summary = run_spatiotemporal_audit(audit, stage_paths=stages,
                signed_stage_keys=[k for k in STAGE_NAMES if limits[case][k][0] < 0], display_limits=limits[case],
                source_frames_ui=frames, source_binding=source, operating_point=api.read(out / "calibration.json"),
                numeric_threshold_paths={"curves": out / "curves.json", "calibration": out / "calibration.json"},
                candidates_path=out / "audit_candidates.json", expert_occurrences=folder / "experts.json", fps=50.)
            print(dict(status="AUDIT_COMPLETE", worker=worker, case_id=case, arm_id=arm, videos=summary["video_count"]), flush=True)
        count += 1
    result = dict(status="PASS", worker=worker, workers=workers, cells=count)
    api.write_json(root / f"media_worker_{worker}.json", result)
    return result


def validate(root):
    from .reference_validate import validate as verify_all
    return verify_all(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("forecast", "media", "validate"))
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    root = (args.root or _api().ROOT).resolve()
    if args.command == "media":
        media(root, args.worker, args.workers)
    else:
        globals()[args.command](root)


if __name__ == "__main__":
    main()
