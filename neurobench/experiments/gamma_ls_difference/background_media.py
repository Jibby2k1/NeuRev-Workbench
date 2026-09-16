"""Complete q1 media for physical background/noise-matching factorial states.

Seventy-two unchanged noise audits retain their original source contracts.
Logical aliases do not duplicate media or independent evidence.
"""
from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path

from .followup_validate import require

TOTAL_CELLS = 252
REUSED_CELLS = 72
REFERENCE_ARMS = frozenset(("mean2of3_n3", "mean1_n9", "mean4of3_n9"))
STAGE_NAMES = ("Raw", "Input", "A", "Score")
NUMERIC_NAMES = ("M", "Spread", "C", "Z")


def _api():
    from . import background_study
    return background_study


def _mutable(root):
    root = Path(root).resolve()
    if (root / "completion_manifest.json").exists():
        raise FileExistsError("Completed study roots are immutable")
    return root


def reused(cell):
    return cell.get("reused_audit") is True


FACTOR_FIELDS = ("seed", "background", "normalization", "V", "S", "T")


def logical_mapping(protocol):
    """Logical conditions are aliases of physical evidence, never extra replicates."""
    return [dict(row, arm_id=r["arm_id"]) for row in protocol["logical_cases"] for r in protocol["references"]]


def balanced_partitions(rows, workers):
    """Deterministic count-capped greedy scheduling of complete physical states.

    Cost is the frozen inventory's model closeup source-frame count; reused
    audits cost zero rendering frames. This changes scheduling only, not media.
    """
    require(type(workers) is int and 1 <= workers <= 3, "Expected one to three workers")
    keys = [(r["case_id"], r["arm_id"]) for r in rows]
    require(len(keys) == len(set(keys)), "Duplicate scheduling state")
    for row in rows:
        require(type(row["model_closeup_frames"]) is int and row["model_closeup_frames"] >= 0,
                "Invalid model closeup frame estimate")
    capacities = [len(range(w, len(rows), workers)) for w in range(workers)]
    bins = [dict(worker=w, cells=0, estimated_new_model_closeup_frames=0, assigned_keys=[])
            for w in range(workers)]
    cost = lambda row: 0 if row["reused"] else row["model_closeup_frames"]
    for row in sorted(rows, key=lambda r: (-cost(r), r["case_id"], r["arm_id"])):
        chosen = min((b for b in bins if b["cells"] < capacities[b["worker"]]),
                     key=lambda b: (b["estimated_new_model_closeup_frames"], b["cells"], b["worker"]))
        chosen["assigned_keys"].append([row["case_id"], row["arm_id"]])
        chosen["cells"] += 1
        chosen["estimated_new_model_closeup_frames"] += cost(row)
    return bins


def frozen_partition(forecast_record, worker, workers):
    """Reject an edited assignment even when its simple state count still fits."""
    expected = balanced_partitions(forecast_record["cells"], workers)
    require(forecast_record["worker_partitions"][str(workers)] == expected,
            "Forecast worker assignment differs from deterministic inventory schedule")
    require(type(worker) is int and 0 <= worker < workers, "Invalid worker index")
    return expected[worker]


def matrix(protocol):
    states = _api().cells(protocol)
    keys = [(c["case_id"], c["arm_id"]) for c in states]
    cases = {c["case_id"]: c for c in protocol["cases"]}
    arms = {r["arm_id"] for r in protocol["references"]}
    require(len(cases) == len(protocol["cases"]) == protocol["expected_datasets"] == 84
            and len(protocol["references"]) == 3 and arms == REFERENCE_ARMS,
            "Expected 84 physical null datasets and the three fixed references")
    require(len(keys) == len(set(keys)) == protocol["expected_cells"] == TOTAL_CELLS
            and set(keys) == {(c, a) for c in cases for a in arms}, "Incomplete physical background Cartesian matrix")
    require(sum(reused(c) for c in states) == REUSED_CELLS, "Expected exactly 72 reused audits")
    require(protocol["reused_audits"] == REUSED_CELLS and protocol["new_audits"] == TOTAL_CELLS - REUSED_CELLS
            and protocol["expected_curve_rows"] == TOTAL_CELLS * 20, "Declared physical media/numerical counts differ")
    seeds = {c["seed"] for c in cases.values()}
    require(len(seeds) == 3 and all(c["kind"] == "factorial" for c in cases.values()),
            "Expected three seeds and factorial datasets only; no legacy compound")
    expected_logical = {(seed, b, n, v, s, t) for seed in seeds for b in ("flat", "sloped")
                        for n in ("raw", "conditioned") for v in (0, 1) for s in (0, 1) for t in (0, 1)}
    factor_key = lambda row: tuple(row[k] for k in FACTOR_FIELDS)
    expected_physical = {k for k in expected_logical if not (k[2] == "conditioned" and k[4:] == (0, 0))}
    require({factor_key(c) for c in cases.values()} == expected_physical,
            "Physical matrix must deduplicate exactly conditioned S=T=0")
    logical = protocol["logical_cases"]
    require(len(logical) == len({factor_key(c) for c in logical}) == 96
            and {factor_key(c) for c in logical} == expected_logical, "Incomplete 96-condition logical design")
    for row in logical:
        target = row["canonical_case_id"]
        require(target in cases, "Logical alias has no physical dataset")
        expected = list(factor_key(row))
        if row["S"] == row["T"] == 0:
            expected[2] = "raw"
        require(factor_key(cases[target]) == tuple(expected), "Logical alias targets the wrong physical condition")
    for cell in states:
        case = cases[cell["case_id"]]
        require(Path(cell["case_id"]).name == cell["case_id"] and cell["case_id"] not in (".", ".."),
                "Unsafe physical case identifier")
        require(cell["study"] == "background" and cell["input_mode"] == "level"
                and cell["readout"] == "Z" and cell["window"] == 3
                and cell["calibration_method"] == "global", "Background media semantics differ")
        expected_reuse = case["background"] == "sloped" and case["normalization"] == "raw"
        require(reused(cell) == expected_reuse == case["reused_dataset"], "Incorrect exact-reuse eligibility")
        require(cell["case_kind"] == case["kind"] and all(cell[k] == case[k] for k in FACTOR_FIELDS),
                "Cell/dataset factor metadata differ")
        if expected_reuse:
            require(cell["baseline_case_id"] == case["baseline_case_id"] == cell["case_id"]
                    and cell["baseline_arm"] == cell["arm_id"], "Reuse must preserve the exact old case and arm")
        else:
            require(cell["baseline_case_id"] is None and cell["baseline_arm"] is None,
                    "New background states cannot name a reused audit")
    return states


def reused_case_limits(existing, cases):
    """Preserve each original display exactly, even when old cases differ."""
    result = {case: copy.deepcopy(existing[case]) for case in cases}
    require(bool(result), "Reused display inventory is empty")
    for limits in result.values():
        require(set(limits) == set(STAGE_NAMES) and limits["Score"] == [-8., 8.],
                "The frozen display stage or Z scale differs")
        for name, bounds in limits.items():
            require(len(bounds) == 2 and all(math.isfinite(float(v)) for v in bounds)
                    and bounds[0] < bounds[1], f"Invalid frozen display extent: {name}")
        require(limits["Input"][0] == limits["A"][0] == 0., "Expected zero-based level-stage display")
    return result


def pooled_reuse_limits(case_limits):
    """A declared envelope initializes the new shared range; originals stay exact."""
    require(bool(case_limits), "Cannot pool empty display inventory")
    values = list(case_limits.values())
    high = max(v[k][1] for v in values for k in ("Input", "A"))
    return dict(Raw=[min(v["Raw"][0] for v in values), max(v["Raw"][1] for v in values)],
                Input=[0., high], A=[0., high], Score=[-8., 8.])


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
        for record in saved["preview_sources"]:
            api.verify({k: record[k] for k in ("path", "sha256", "size_bytes")})
        return saved
    import numpy as np
    baseline_display = Path(protocol["baseline_protocol"]["path"]).resolve().parent / "display_contract.json"
    existing = api.read(baseline_display)["cases"]
    old_cases = sorted({c["case_id"] for c in states if reused(c)})
    new_cases = sorted({c["case_id"] for c in states} - set(old_cases))
    originals = reused_case_limits(existing, old_cases)
    common = pooled_reuse_limits(originals)
    previews, sources = [], []
    for case in new_cases:
        seal = api.read(root / "cells" / case / "mean1_n9" / "sealed.json")
        row = dict(case_id=case)
        for name in ("Raw", "Input", "A"):
            record = seal["stages"][name]
            api.verify(record)
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
                  cases={case: copy.deepcopy(originals[case] if case in old_cases else null_limits)
                         for case in sorted(set(old_cases + new_cases))},
                  reused_cases=old_cases, new_cases=new_cases, preview_sources=sources, previews=previews,
                  preview_rule="Every fifth available source sample, numpy percentiles 1/99.8 for Raw and 99.8 for Input/A; display only, no score or label selection.",
                  exception=dict(applied=any(v != null_limits for v in originals.values()),
                      reason="Preserve each of the 24 reused sloped/raw dataset displays exactly. All 60 new physical datasets share one pooled fixed range. Any old/new range difference is explicit; compare numerical traces rather than apparent brightness across that boundary.",
                      reused_case_limits=originals, pooled_reuse_limits=common, new_factorial_limits=null_limits),
                  rule="Raw/Input/A fixed across all three references and shared across new factorial datasets. All 24 reused datasets preserve their individual immutable display ranges, with any difference recorded explicitly. Input/A are native fluorescence units; Score is direct Z, dimensionless, fixed [-8,8]. Gray pixels are display transforms of float arrays; master RGB composites preserve those display pixels losslessly, not the underlying float arrays.")
    api.write_json(path, result)
    return result


def forecast(root):
    root = _mutable(root)
    api = _api()
    protocol = api.load(root)
    states = matrix(protocol)
    evaluation = api.read(root / "evaluation_complete.json")
    require(evaluation["status"] == "PASS" and evaluation["cells"] == len(states)
            and evaluation["curve_rows"] == protocol["expected_curve_rows"] == len(states) * 20,
            "Evaluate every physical background cell and both radii first")
    display(root)
    from .followup_audit import audit_inventory_plan
    rows = []
    for cell in states:
        folder = root / "datasets" / cell["case_id"]
        out = root / "cells" / cell["case_id"] / cell["arm_id"]
        frames, _, _ = api.frame_sets(api.read(folder / "metadata.json"))
        experts = api.read(folder / "experts.json")
        require(experts == [] and api.read(folder / "active.json") == [], "Background factorial requires explicit empty synthetic truth")
        plan = audit_inventory_plan(api.read(out / "audit_candidates.json"), experts, frames)
        rows.append(dict(cell, reused=reused(cell), expected_videos=plan["expected_video_count"],
                         expert_rois=plan["expert_roi_count"], model_rois=plan["model_roi_count"],
                         expert_occurrences=plan["expert_occurrence_count"],
                         model_closeup_frames=plan["closeup_video_frame_count_total"]))
    reuse_count = sum(r["reused"] for r in rows)
    value = dict(status="PASS", cells=rows, logical_cells=len(logical_mapping(protocol)),
                 logical_to_physical=logical_mapping(protocol), new_cells=len(rows) - reuse_count, reused_cells=reuse_count,
                 new_videos=sum(r["expected_videos"] for r in rows if not r["reused"]),
                 total_videos=sum(r["expected_videos"] for r in rows), display_contract=api.binding(root / "display_contract.json"),
                 worker_partitions={str(n): balanced_partitions(rows, n) for n in (1, 2, 3)},
                 scheduling_rule="Descending estimated new model closeup frames, stable case/arm ties; greedy least-loaded worker with exact count caps (84 each for three workers). Reused states have zero rendering cost. Full states stay indivisible; no ROI/trace/frame coverage reduction.",
                 media_code=media_code(api), protocol=api.binding(root / "protocol.json"),
                 evaluation=api.binding(root / "evaluation_complete.json"))
    api.write_json(root / "audit_forecast.json", value)
    print({k: value[k] for k in ("status", "new_cells", "reused_cells", "logical_cells", "new_videos", "total_videos")}, flush=True)
    print(dict(worker_estimates=[{k: v for k, v in row.items() if k != "assigned_keys"}
                                for row in value["worker_partitions"]["3"]]), flush=True)
    return value


def media_code(api):
    return [api.binding(Path(__file__).with_name(name)) for name in
            ("background_media.py", "followup_audit.py", "two_stencil_audit.py", "two_stencil_evaluation.py")]


def media(root, worker=0, workers=3):
    root = _mutable(root)
    require(type(worker) is int and type(workers) is int and 0 <= worker < workers <= 3,
            "At most three disjoint integer-indexed media workers")
    api = _api()
    protocol = api.load(root)
    states = matrix(protocol)
    forecast_record = api.read(root / "audit_forecast.json")
    require(forecast_record["status"] == "PASS" and len(forecast_record["cells"]) == len(states)
            and [(c["case_id"], c["arm_id"]) for c in forecast_record["cells"]]
                == [(c["case_id"], c["arm_id"]) for c in states],
            "A complete audit forecast is required")
    require(forecast_record["media_code"] == media_code(api), "Media source changed after forecast")
    assignment = frozen_partition(forecast_record, worker, workers)
    for record in forecast_record["media_code"] + [forecast_record["protocol"], forecast_record["evaluation"]]:
        api.verify(record)
    api.verify(forecast_record["display_contract"])
    limits = api.read(root / "display_contract.json")["cases"]
    baseline = Path(protocol["baseline_protocol"]["path"]).resolve().parent
    replication = api.read(root / "baseline_replication.json")
    require(replication["status"] == "PASS" and len(replication["cells"]) == REUSED_CELLS,
            "All original sloped/raw states must pass exact replication before reuse")
    from .followup_audit import run_spatiotemporal_audit
    count = 0
    by_key = {(c["case_id"], c["arm_id"]): c for c in states}
    for key in assignment["assigned_keys"]:
        cell = by_key[tuple(key)]
        case, arm = cell["case_id"], cell["arm_id"]
        folder, out = root / "datasets" / case, root / "cells" / case / arm
        audit = root / "audits" / case / arm
        if reused(cell):
            old = baseline / "audits" / cell["baseline_case_id"] / cell["baseline_arm"]
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
                numeric_stage_bindings={k: seal["stages"][k] for k in NUMERIC_NAMES
                    if Path(seal["stages"][k]["path"]).resolve() not in {Path(p).resolve() for p in stages.values()}},
                original_source_offset_xy=meta["original_source_offset_xy"], truth_mode=meta["truth_mode"], cell=cell,
                stage_semantics="Raw fluorescence; Input conditioned activity level; A unchanged target average; Score direct unmasked Z=(A-M)/max(reference spread,frozen floor). M/Spread/C/Z arrays linked numerically. Frozen global cutoff after 3x3 local maxima and 6px separation. Model sites are review locations, not inferred neuronal identities; nearest and assigned sites remain distinct.")
            summary = run_spatiotemporal_audit(audit, stage_paths=stages,
                signed_stage_keys=[k for k in STAGE_NAMES if limits[case][k][0] < 0], display_limits=limits[case],
                source_frames_ui=frames, source_binding=source, operating_point=api.read(out / "calibration.json"),
                numeric_threshold_paths={"curves": out / "curves.json", "calibration": out / "calibration.json"},
                candidates_path=out / "audit_candidates.json", expert_occurrences=folder / "experts.json", fps=50.)
            print(dict(status="AUDIT_COMPLETE", worker=worker, case_id=case, arm_id=arm, videos=summary["video_count"]), flush=True)
        count += 1
    result = dict(status="PASS", worker=worker, workers=workers, cells=count,
                  assigned_keys=assignment["assigned_keys"],
                  estimated_new_model_closeup_frames=assignment["estimated_new_model_closeup_frames"],
                  media_code=forecast_record["media_code"], forecast=api.binding(root / "audit_forecast.json"))
    api.write_json(root / f"media_worker_{worker}.json", result)
    return result


def validate(root):
    from .background_validate import validate as verify_all
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
