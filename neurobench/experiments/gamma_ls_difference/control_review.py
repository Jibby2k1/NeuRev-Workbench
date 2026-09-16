"""Postprocess sealed control-study predictions; never refit or score a detector.

The completed study supplies all five numeric targets and its q1 local maxima.
This module verifies the original candidate-seal chain and recomputes only the
small spatial grouping/label joins. Dense stage arrays are never opened or
rehashed. Optional plotting is imported only after the read-only checks pass.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

TARGETS = (0.25, 0.5, 1.0, 2.0, 5.0)
FACTORS = ("sigma", "alpha", "representation", "correction", "design", "reference", "guard", "geometry", "readout")
CATEGORIES = ("matched", "no_score_exceedance", "no_retained_frame_candidate_within6",
              "no_representative_within6", "one_to_one_assignment_competition")
SCOPE = "descriptive within-recording development; broad windows are not onset truth; unmatched candidates are unknown, not negative"


def _sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _json(path: Path) -> Any:
    return json.loads(path.read_text())


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["no_rows"], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _bool(value: Any) -> bool:
    if value is True or value == "True": return True
    if value is False or value == "False": return False
    raise ValueError(f"Invalid boolean: {value!r}")


def _integer(value: Any) -> int:
    number = float(value)
    if not math.isfinite(number) or number != int(number):
        raise ValueError(f"Expected integer, received {value!r}")
    return int(number)


def _finite(value: Any) -> float:
    number = float(value)
    if not math.isfinite(number): raise ValueError("Nonfinite diagnostic or count")
    return number


def _valid_arm_id(name: str) -> bool:
    """Allow decimal parameter names, but only as one ordinary path component."""
    return bool(name) and name not in (".", "..") and all(
        character in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-."
        for character in name
    )


def _equal(actual: Any, expected: Any) -> bool:
    """Compare JSON/TSV serialization without treating the string False as true."""
    if isinstance(expected, Mapping):
        return isinstance(actual, Mapping) and set(actual) == set(expected) and all(_equal(actual[k], v) for k, v in expected.items())
    if isinstance(expected, (list, tuple)):
        return isinstance(actual, (list, tuple)) and len(actual) == len(expected) and all(_equal(a, b) for a, b in zip(actual, expected))
    if expected is None: return actual in (None, "", "None", "null")
    if isinstance(expected, bool):
        try: return _bool(actual) == expected
        except ValueError: return False
    if isinstance(expected, int):
        try: return _integer(actual) == expected
        except (ValueError, TypeError): return False
    if isinstance(expected, float):
        try: return math.isclose(_finite(actual), expected, rel_tol=1e-12, abs_tol=1e-9)
        except (ValueError, TypeError): return False
    return str(actual) == str(expected)


def component_pairs(arms: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Declare contrasts from configuration metadata, without viewing outcomes."""
    modern = [dict(arm) for arm in arms if not arm.get("historical", False)]
    names = [str(arm["arm_id"]) for arm in arms]
    if len(set(names)) != len(names): raise ValueError("Duplicate arm identifiers")
    if any(set(FACTORS) - set(arm) for arm in modern): raise ValueError("Incomplete component factor metadata")
    # The direction is always left minus right. Derived source filenames are
    # intentionally excluded; every scientific factor not contrasted is fixed.
    specifications = [
        ("spatial_smoothing", "sigma", 1, 0, "Gaussian sigma1 minus bypass"),
        ("temporal_smoothing", "alpha", .4, 1.0, "EMA alpha0.4 minus bypass"),
        ("temporal_representation", "representation", "difference", "current", "signed difference minus conditioned current"),
        ("target_pooling", "design", "direct", "point", "direct target pooling minus point target"),
        ("reference_construction", "design", "serial", "direct", "serial reference minus direct reference"),
        ("reference_weights", "reference", "gamma", "uniform", "Gamma reference minus uniform reference"),
        ("motion_correction", "correction", "motion", "none", "bounded registration minus uncorrected"),
        ("global_offset", "correction", "global_offset", "none", "global offset subtraction minus uncorrected"),
        ("readout", "readout", "contrast", "A", "contrast minus amplitude"),
        ("readout", "readout", "Z", "contrast", "standardization minus contrast"),
        ("readout", "readout", "Z", "A", "standardization minus amplitude"),
    ]
    pairs = []
    for family, factor, left_value, right_value, description in specifications:
        for left in modern:
            if left[factor] != left_value: continue
            for right in modern:
                if right[factor] != right_value: continue
                if any(left[key] != right[key] for key in FACTORS if key != factor): continue
                pairs.append(dict(comparison_id=f"{family}__{left['arm_id']}__minus__{right['arm_id']}",
                    component=family, left_arm_id=left["arm_id"], right_arm_id=right["arm_id"],
                    direction="left_minus_right", description=description, changed_factors=factor,
                    confounded=False, interpretation="all other declared scientific factors fixed"))
    for left in modern:
        if left["guard"] != 7 or left["geometry"] != "disk": continue
        for right in modern:
            if right["guard"] != 0 or right["geometry"] != "square": continue
            if any(left[key] != right[key] for key in FACTORS if key not in ("guard", "geometry")): continue
            pairs.append(dict(comparison_id=f"guard_and_support__{left['arm_id']}__minus__{right['arm_id']}",
                component="guard_and_support", left_arm_id=left["arm_id"], right_arm_id=right["arm_id"],
                direction="left_minus_right", description="guard7 disk minus guard0 square",
                changed_factors="guard,geometry", confounded=True,
                interpretation="inner exclusion and outer support geometry both change; not an isolated guard effect"))
    identifiers = [row["comparison_id"] for row in pairs]
    if len(set(identifiers)) != len(identifiers): raise ValueError("Duplicate component comparison")
    return sorted(pairs, key=lambda row: row["comparison_id"])


def paired_counts(left_rows: Sequence[Mapping[str, Any]], right_rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    def outcomes(rows):
        result = {}
        for row in rows:
            observation = str(row["observation_id"])
            if observation in result: raise ValueError("Duplicate paired occurrence")
            result[observation] = _bool(row["matched"])
        return result
    left, right = outcomes(left_rows), outcomes(right_rows)
    if set(left) != set(right): raise ValueError("Paired arms must cover the same known occurrences")
    both = sum(left[key] and right[key] for key in left)
    gained = sum(left[key] and not right[key] for key in left)
    lost = sum(not left[key] and right[key] for key in left)
    neither = len(left) - both - gained - lost
    return dict(known_occurrence_count=len(left), both=both, gained=gained, lost=lost, neither=neither,
                left_matched=both+gained, right_matched=both+lost, delta_matched=gained-lost)


def classify_q1_occurrence(occurrence: Mapping[str, Any], diagnostic: Mapping[str, Any],
                           candidates: Sequence[Mapping[str, Any]], sites: Sequence[Mapping[str, Any]],
                           assigned_observation_by_site: Mapping[str, str]) -> dict[str, Any]:
    """Describe a saved strict-cutoff/window output; no NMS or grouping rerun."""
    observation = str(occurrence["observation_id"])
    if str(diagnostic["observation_id"]) != observation: raise ValueError("Diagnostic occurrence differs")
    peak, threshold = _finite(diagnostic["local_peak"]), _finite(diagnostic["threshold"])
    if not math.isclose(_finite(diagnostic["margin"]), peak-threshold, rel_tol=1e-12, abs_tol=1e-9):
        raise ValueError("Saved local-peak margin disagrees with peak minus threshold")
    x, y = float(occurrence["x_px"]), float(occurrence["y_px"])
    start, stop = _integer(occurrence["source_start_ui"]), _integer(occurrence["source_stop_ui"])
    burst = _integer(occurrence["burst_id"])
    distance = lambda row: math.hypot(float(row["x_px"])-x, float(row["y_px"])-y)
    local_rows = [row for row in candidates if start <= _integer(row["source_frame_ui"]) <= stop and distance(row) <= 6]
    local_sites = [row for row in sites if _integer(row["burst_id"]) == burst and distance(row) <= 6]
    if local_sites and not local_rows: raise ValueError("Nearby representative has no emitted frame candidate")
    if local_rows and peak <= threshold: raise ValueError("Nearby emitted candidate contradicts the local score maximum")
    matched = _bool(occurrence["matched"])
    expected_original = "matched" if matched else ("no_local_exceedance" if peak <= threshold else "exceedance_unmatched")
    if diagnostic.get("category") != expected_original: raise ValueError("Saved diagnostic category disagrees with outcome/score")
    if matched:
        matched_id = str(occurrence["matched_site_id"])
        if matched_id not in {str(row["site_id"]) for row in local_sites} or assigned_observation_by_site.get(matched_id) != observation:
            raise ValueError("Matched occurrence lacks its nearby assigned representative")
        category = CATEGORIES[0]
    elif peak <= threshold: category = CATEGORIES[1]
    elif not local_rows: category = CATEGORIES[2]
    elif not local_sites: category = CATEGORIES[3]
    else:
        if any(not assigned_observation_by_site.get(str(row["site_id"])) for row in local_sites):
            raise ValueError("Unmatched occurrence has an unassigned nearby representative")
        category = CATEGORIES[4]
    return dict(observation_id=observation, burst_id=burst, canonical_roi_id=occurrence["canonical_roi_id"],
        x_px=x, y_px=y, source_start_ui=start, source_stop_ui=stop, matched=matched,
        diagnostic_category=category, local_peak=peak, threshold=threshold, margin=peak-threshold,
        nearby_frame_candidate_count=len(local_rows), nearby_representative_count=len(local_sites),
        nearby_candidate_ids=json.dumps(sorted(str(row["proposal_id"]) for row in local_rows)),
        nearby_site_ids=json.dumps(sorted(str(row["site_id"]) for row in local_sites)),
        nearby_site_assigned_observations=json.dumps({str(row["site_id"]): assigned_observation_by_site.get(str(row["site_id"])) for row in local_sites}, sort_keys=True),
        matched_site_id=occurrence.get("matched_site_id", ""), original_diagnostic_category=diagnostic["category"],
        onset_latency_identified=False, unmatched_candidates="unknown_not_negative")


class _Bindings:
    def __init__(self) -> None:
        self.records: dict[str, dict[str, Any]] = {}

    def add(self, path: Path, expected: Mapping[str, Any] | None = None) -> Path:
        path = path.resolve()
        if not path.is_file(): raise FileNotFoundError(path)
        record = dict(path=str(path), sha256=_sha(path), size_bytes=path.stat().st_size)
        if expected and (record["sha256"] != expected["sha256"] or record["size_bytes"] != int(expected["size_bytes"])):
            raise ValueError(f"Changed sealed artifact: {path}")
        previous = self.records.get(str(path))
        if previous and previous != record: raise ValueError(f"Source changed during review: {path}")
        self.records[str(path)] = record
        return path

    def verify(self, record: Mapping[str, Any]) -> Path:
        return self.add(Path(record["path"]), record)


def _report(arms, pairs, summaries, attrition_counts) -> str:
    lines = ["# Component-control review", "", SCOPE + ".", "",
        "The five calibration targets are proposals per calibration score frame. Their application burdens are measured separately over 460 frames (UI1900–2359); equal targets do not imply equal emitted counts.",
        "The paired tables use the same 79 occurrences: both, gained (left only), lost (right only), and neither. They are not independent recordings, precision estimates, or biological event counts.", "",
        "[All five-target paired contrasts](paired_components.tsv) · [Every q1 occurrence](q1_attrition.tsv) · [All five conditional categories](q1_attrition_counts.tsv) · [All 28 numeric curves](operating_points.tsv)", "",
        "| Configuration | q1 matched /79 | Emitted rows | No local exceedance | No nearby frame candidate | No nearby representative | Assignment competition |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    counts = {row["arm_id"]: row for row in attrition_counts}
    lookup = {(row["arm_id"], row["q"]): row for row in summaries}
    for arm in arms:
        name = arm["arm_id"]; row = lookup[(name, 1.0)]; categories = counts[name]
        lines.append(f"| {name} | {row['matched_known_positive_count']} | {row['emitted_frame_proposal_count']} | " + " | ".join(str(categories[key]) for key in CATEGORIES[1:]) + " |")
    lines += ["", f"The metadata declares {len(pairs)} component pairs, each reported at all five targets. All noncontrasted factors are fixed except the explicitly confounded guard7-disk versus guard0-square comparison, which changes both guard and outer support. The historical residual/carrier remain in the full curves; they are not isolated component ablations.", "",
        "The q1 categories are a precedence-ordered partition over each entire inclusive known window. No score exceedance means the saved maximum over integer pixels within 6 pixels of the original expert coordinate is at or below the strict cutoff. Later categories inspect saved full-field candidates, burst representatives, and one-to-one assignments. A nearby representative assigned elsewhere is not detector absence. These are locations of divergence, not causal biological explanations.", "",
        "The original candidate-seal chain, all sealed files, and frozen labels were verified. Grouping/matching and every saved numeric join/summary were recomputed solely to check agreement; predictions, thresholds and files were unchanged. Dense stage arrays and the q1 local maxima were not recomputed. Their numerical verification is inherited from the completed study, with existing stage hashes retained as provenance.", "",
        "Motion scores are already mapped back to original source coordinates. Global-offset subtraction can remove shared neural activity as well as illumination changes; neither correction is selected for deployment here. Discovery coverage within broad intervals is distinct from configured-region activity tracking and control-ready onset latency. Scientific media completion is a separate gate."]
    return "\n".join(lines) + "\n"


def plot_groups(arms: Sequence[Mapping[str, Any]]) -> tuple[list, list]:
    """Fixed, reader-facing plot groups selected only from factor metadata."""
    smoothing = {(0,1.0): "Neither", (1,1.0): "Spatial only", (0,.4): "Temporal only", (1,.4): "Both"}
    conditioning = []
    for representation, input_label in (("current","Current brightness"),("difference","Signed change")):
        for readout, score_label in (("A","without local standardization"),("Z","with Gamma-LS")):
            selected = [(a["arm_id"], smoothing[(a["sigma"],a["alpha"])]) for a in arms
                if not a.get("historical") and a["representation"] == representation and a["readout"] == readout
                and a["design"] == "point" and a["correction"] == "none" and a["reference"] == "gamma" and a["guard"] == 0]
            conditioning.append((f"{input_label}\n{score_label}",selected))
    base = "s1_a0.4_difference_Z"
    operators = [
        ("Target and reference construction", [
            (base,"Point: single-pixel target"),
            ("centered_direct_Z","Direct: target average, original reference"),
            ("centered_serial_Z","Serial: reference uses target averages")]),
        ("What the local score adds", [
            ("centered_direct_A","Centered target average"),
            ("centered_direct_contrast","Target minus reference"),
            ("centered_direct_Z","Locally standardized")]),
        ("Reference weights and guard\nGuarded disk also changes outer support", [
            (base,"Gamma weights, unguarded square"),
            ("uniform_reference_Z","Uniform weights, unguarded square"),
            ("guard7_disk_Z","Gamma weights, guarded disk")]),
        ("Image correction before scoring", [
            (base,"Uncorrected"),
            ("motion_Z","Motion alignment (translation)"),
            ("global_offset_Z","Global brightness-offset removal")]),
    ]
    return conditioning,operators


def _plots(output: Path, arms: Sequence[Mapping[str, Any]], summaries: Sequence[Mapping[str, Any]]) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    curves = {arm["arm_id"]: sorted([row for row in summaries if row["arm_id"] == arm["arm_id"]], key=lambda row: row["q"]) for arm in arms}
    def draw(path, groups, explanation):
        figure, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
        for axis, (title, names) in zip(axes.flat, groups):
            for name, label in names:
                points = curves[name]
                axis.plot([row["emitted_frame_proposal_count"] for row in points], [row["matched_known_positive_count"] for row in points], marker="o", markersize=4, linewidth=1.3, label=label)
            axis.set(title=title, xlabel="Actual emitted frame proposals (460 application frames)", ylabel="Matched known occurrences /79", ylim=(-1, 80))
            axis.set_xlim(left=0)
            axis.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
            displayed = [row for name, _ in names for row in curves[name]]
            if displayed and all(row["emitted_frame_proposal_count"] == 0 and row["matched_known_positive_count"] == 0 for row in displayed):
                axis.set_xlim(0, 1)
                axis.set_xticks([0, 1])
                count_label = {3: "three", 4: "four"}.get(len(names), str(len(names)))
                axis.text(.5, .4, f"All {count_label} configurations:\n0 proposals, 0 matches",
                          transform=axis.transAxes, ha="center", va="center", fontsize=10)
            axis.grid(alpha=.2); axis.legend(fontsize=8)
        figure.suptitle(explanation + "\nAll five calibration targets: 0.25, 0.5, 1, 2, 5 proposals per calibration frame; each curve connects them in order"
            "\nDescriptive same-recording coverage; emitted proposals are not biological events", fontsize=10)
        temporary = path.with_suffix(".partial.png"); figure.savefig(temporary, dpi=140); plt.close(figure); temporary.replace(path)
    conditioning,operators = plot_groups(arms)
    first = output / "conditioning_coverage_burden.png"
    draw(first,conditioning,"Spatial smoothing averages neighboring pixels; temporal smoothing averages current and earlier frames.")
    second = output / "operator_coverage_burden.png"
    draw(second,operators,"Point tests one pixel; direct averages the target; serial also estimates its reference from target averages.")
    return [first,second]


def run_review(root: str | Path, *, plots: bool = False) -> dict[str, Any]:
    """Verify completed frozen outputs and write only CAMPAIGN/review/."""
    root = Path(root).resolve()
    bindings = _Bindings()
    protocol = _json(bindings.add(root / "protocol.json"))
    complete = _json(bindings.add(root / "evaluation_complete.json"))
    arms = protocol["arms"]
    if complete.get("status") != "NUMERICAL_PASS" or len(arms) != 28 or complete.get("arm_count") != 28:
        raise ValueError("All 28 study arms require completed numerical evaluation")
    if tuple(map(float, protocol["target_proposals_per_frame"])) != TARGETS:
        raise ValueError("Expected all five frozen calibration targets")
    if list(protocol["application_ui"]) != [1900,2359]: raise ValueError("Unexpected application interval")
    if complete.get("online_operating_points") != 140 or complete.get("diagnostic_occurrences") != 2212:
        raise ValueError("Incomplete operating-point or local-diagnostic inventory")
    for record in protocol["code_bindings"]: bindings.verify(record)
    labels_path = bindings.verify(protocol["labels"])
    if labels_path != (root / "expert_occurrences.tsv").resolve(): raise ValueError("Label source does not match the frozen study copy")
    labels = _rows(labels_path)
    label_ids = [row["observation_id"] for row in labels]
    if len(labels) != 79 or len(set(label_ids)) != 79 or len({row["canonical_roi_id"] for row in labels}) != 26:
        raise ValueError("Expected the complete 79-occurrence/26-identity cohort")
    windows = {int(key): tuple(map(int, value)) for key, value in protocol["burst_windows"].items()}
    if len(windows) != 4 or any((int(row["source_start_ui"]),int(row["source_stop_ui"])) != windows[int(row["burst_id"])] for row in labels):
        raise ValueError("Inconsistent frozen burst windows")
    seal = _json(bindings.add(root / "candidate_seal.json"))
    if seal.get("status") != "ALL_ARMS_SEALED_BEFORE_OUTCOME_JOIN" or seal.get("arms") != 28 or len(seal["seals"]) != 28:
        raise ValueError("All 28 original candidate seals are required")
    arm_names = [str(arm["arm_id"]) for arm in arms]
    if len(set(arm_names)) != 28 or any(not _valid_arm_id(name) for name in arm_names):
        raise ValueError("Invalid or duplicate arm identifier")
    seals = {}
    stage_bindings = []
    for record in seal["seals"]:
        path = bindings.verify(record); content = _json(path); name = content["arm"]["arm_id"]
        if name in seals or name not in arm_names or path != root / "arms" / name / "sealed.json":
            raise ValueError("Unexpected or duplicate arm seal")
        if content["arm"] != arms[arm_names.index(name)]: raise ValueError("Arm seal disagrees with frozen factor metadata")
        if content["application_ui"] != [1900,2359] or content["calibration_score_ui"] != [1801,1899]:
            raise ValueError("Arm seal interval disagreement")
        records = {}
        for item in content["files"]:
            verified = bindings.verify(item)
            if str(verified) in records: raise ValueError("Duplicate sealed file")
            records[str(verified)] = item
        for required in [root / "arms" / name / "calibration.json", *(root / "arms" / name / f"q{q:g}" / "candidates.tsv" for q in TARGETS)]:
            if str(required) not in records: raise ValueError("Calibration or candidate table is absent from original seal")
        for stage, item in content["stages"].items():
            path = Path(item["path"])
            if not path.is_file() or path.stat().st_size != int(item["size_bytes"]): raise ValueError("Missing or resized sealed stage array")
            stage_bindings.append(dict(arm_id=name, stage=stage, **item, bytes_rehashed=False))
        seals[name] = content
    if set(seals) != set(arm_names): raise ValueError("Candidate-seal arm inventory mismatch")
    diagnostic_rows = _rows(bindings.add(root / "q1_stage_diagnostics.tsv"))
    diagnostics = {(row["arm_id"], row["observation_id"]): row for row in diagnostic_rows}
    if len(diagnostics) != len(diagnostic_rows) or set(diagnostics) != {(name, observation) for name in arm_names for observation in label_ids}:
        raise ValueError("Incomplete or duplicate q1 local diagnostics")
    # The helper is imported only after its original protocol code binding has
    # been checked. It groups saved rows; no dense score, threshold, or NMS runs.
    helper_path = Path(__file__).with_name("two_stencil_evaluation.py").resolve()
    if str(helper_path) not in bindings.records: raise ValueError("The grouping/matching helper must have an original protocol binding")
    from .two_stencil_evaluation import evaluate_occurrence_windows
    summaries, attrition, all_outcomes = [], [], {}
    for arm in arms:
        name = arm["arm_id"]; folder = root / "arms" / name
        calibration = _json(folder / "calibration.json")
        operating = {float(row["target_proposals_per_frame"]): row for row in calibration["operating_points"]}
        if set(operating) != set(TARGETS) or len(calibration["operating_points"]) != 5:
            raise ValueError("Incomplete or duplicate calibration operating points")
        for q in TARGETS:
            tau = _finite(operating[q]["threshold_z"])
            candidates = _rows(folder / f"q{q:g}" / "candidates.tsv")
            for row in candidates:
                if (not 1900 <= _integer(row["source_frame_ui"]) <= 2359 or _finite(row["target_proposals_per_frame"]) != q
                        or _finite(row["threshold_z"]) != tau or not _finite(row["score"]) > tau):
                    raise ValueError("Candidate does not belong to its frozen application operating point")
                if row.get("cell_id",name) != name: raise ValueError("Candidate arm identifier mismatch")
            recomputed = evaluate_occurrence_windows(candidates, labels, burst_intervals_ui=windows)
            for key in ("occurrence_rows", "site_rows", "membership_rows", "burst_summaries"):
                actual = _rows(bindings.add(folder / f"q{q:g}" / f"{key}.tsv"))
                if not _equal(actual, recomputed[key]): raise ValueError(f"Saved {key} differs from the frozen candidate join: {name} q{q:g}")
            summary = _json(bindings.add(folder / f"q{q:g}" / "summary.json"))
            if not _equal(summary, recomputed["summary"]): raise ValueError(f"Saved numeric summary differs from joined candidates: {name} q{q:g}")
            if summary["known_positive_count"] != 79: raise ValueError("Numeric endpoint cohort changed")
            summaries.append(dict(arm_id=name, q=q, **summary,
                application_score_frame_count=460, actual_proposals_per_application_frame=summary["emitted_frame_proposal_count"]/460,
                calibration_threshold=tau, calibration_proposal_count=operating[q]["calibration_proposal_count"]))
            all_outcomes[(name,q)] = recomputed["occurrence_rows"]
            if q == 1:
                assigned = {row["matched_site_id"]: row["observation_id"] for row in recomputed["occurrence_rows"] if row["matched"]}
                for occurrence in recomputed["occurrence_rows"]:
                    diagnostic = diagnostics[(name,occurrence["observation_id"])]
                    if _finite(diagnostic["threshold"]) != tau: raise ValueError("Local diagnostic uses a different cutoff")
                    attrition.append(dict(arm_id=name, q=1.0, **classify_q1_occurrence(occurrence, diagnostic, candidates, recomputed["site_rows"], assigned)))
    online = _rows(bindings.add(root / "online_summary.tsv"))
    expected_online = [{key: value for key,value in row.items() if key not in ("application_score_frame_count","actual_proposals_per_application_frame","calibration_threshold","calibration_proposal_count")} for row in summaries]
    if not _equal(online, expected_online): raise ValueError("Combined online summary differs from all saved candidate joins")
    pairs = component_pairs(arms); paired = []
    summary_by_key = {(row["arm_id"],row["q"]): row for row in summaries}
    for pair in pairs:
        for q in TARGETS:
            left, right = pair["left_arm_id"], pair["right_arm_id"]
            counts = paired_counts(all_outcomes[(left,q)], all_outcomes[(right,q)])
            lsum, rsum = summary_by_key[(left,q)], summary_by_key[(right,q)]
            paired.append(dict(**pair, q=q, **counts, left_emitted_rows=lsum["emitted_frame_proposal_count"],
                right_emitted_rows=rsum["emitted_frame_proposal_count"], delta_emitted_rows=lsum["emitted_frame_proposal_count"]-rsum["emitted_frame_proposal_count"],
                left_actual_proposals_per_frame=lsum["actual_proposals_per_application_frame"], right_actual_proposals_per_frame=rsum["actual_proposals_per_application_frame"],
                left_threshold=lsum["calibration_threshold"], right_threshold=rsum["calibration_threshold"],
                calibration_unit="proposals_per_setup_score_frame", application_frame_count=460,
                scope=SCOPE))
    attrition_counts = [dict(arm_id=name, q=1.0, known_occurrence_count=79,
        **{category:sum(row["diagnostic_category"] == category for row in attrition if row["arm_id"] == name) for category in CATEGORIES}) for name in arm_names]
    output = root / "review"
    if output.exists() and any(output.iterdir()) and not (output / "review_manifest.json").is_file():
        raise FileExistsError("Refusing to overwrite an unrelated review directory")
    output.mkdir(exist_ok=True)
    contract = dict(schema_version=1, reviewer_sha256=_sha(Path(__file__)), inputs=list(bindings.records.values()),
                    dense_stage_bindings=stage_bindings, dense_stage_arrays_rehashed=False,
                    local_maxima_recomputed=False, saved_numeric_joins_recomputed_and_equal=True,
                    predictions_changed=False, scientific_media_completion_claimed=False, scope=SCOPE)
    manifest_path = output / "review_manifest.json"
    if manifest_path.exists():
        previous = _json(manifest_path)
        if previous.get("contract") != contract: raise ValueError("Review sources or postprocessor changed; preserve the existing review directory")
    _write_json(manifest_path, dict(status="WRITING", contract=contract))
    _write_rows(output / "paired_components.tsv", paired)
    _write_rows(output / "q1_attrition.tsv", attrition)
    _write_rows(output / "q1_attrition_counts.tsv", attrition_counts)
    _write_rows(output / "operating_points.tsv", summaries)
    report_path = output / "REVIEW.md"; temporary = report_path.with_suffix(".md.tmp")
    temporary.write_text(_report(arms,pairs,summaries,attrition_counts)); temporary.replace(report_path)
    if plots: _plots(output,arms,summaries)
    artifact_names = ["paired_components.tsv", "q1_attrition.tsv", "q1_attrition_counts.tsv", "operating_points.tsv", "REVIEW.md"]
    artifact_names += [name for name in ("conditioning_coverage_burden.png", "operator_coverage_burden.png") if (output / name).is_file()]
    artifacts = [dict(path=name, sha256=_sha(output / name), size_bytes=(output / name).stat().st_size) for name in artifact_names]
    result = dict(status="PASS", component_pair_count=len(pairs), paired_operating_point_count=len(paired),
        q1_occurrence_count=len(attrition), arm_count=28, online_operating_point_count=140,
        saved_numeric_joins_recomputed_and_equal=True, dense_stage_arrays_rehashed=False,
        output_root=str(output), scientific_audit_complete=False, predictions_changed=False)
    _write_json(manifest_path, dict(**result, contract=contract, artifacts=artifacts))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Completed campaign root; writes its review/ subdirectory")
    parser.add_argument("--plots", action="store_true", help="Render two additional coverage-versus-actual-burden figures")
    args = parser.parse_args()
    print(json.dumps(run_review(args.output, plots=args.plots), sort_keys=True))


if __name__ == "__main__":
    main()
