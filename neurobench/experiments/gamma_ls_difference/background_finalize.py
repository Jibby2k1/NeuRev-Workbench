"""Readiness check and optional final seal for the physical background study.

No detector, fit, truth parsing, rendering, or metric recomputation occurs here.
The default is read-only. --write adds an immutable source capsule and completion
manifest only after numerical, full-media, independent and visual gates pass.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

from .background_media import logical_mapping, matrix, frozen_partition
from .background_validate import verify_worker_assignment
from .followup_validate import FileVerifier, METADATA, require, verify_audit_files

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / "Outputs/GammaLSBackground/background_20260915_r1"
TEST_SCOPES = {"fixtures", "study", "media_validation", "integrity", "report", "finalize"}
NEW_COMPONENTS = ("fixtures", "study", "integrity", "media", "validate", "report", "recount", "finalize")
PAIR_STAGES = ["Input", "A", "M", "Spread", "C", "Z"]


class Evidence:
    def __init__(self):
        self.verifier = FileVerifier()
        self.records = {}

    def bind(self, path):
        record = self.verifier.binding(path)
        self.records[record["path"]] = record
        return record

    def check(self, record, base=None):
        actual = self.verifier.verify(record, root=base)
        self.records[actual["path"]] = actual
        return actual

    def get(self, path):
        require(Path(path).name not in {"experts.json", "active.json"}, "Finalizer does not parse truth files")
        self.bind(path)
        return self.verifier.read_json(path)

    def walk(self, value):
        if isinstance(value, dict):
            if "path" in value and "sha256" in value:
                self.check(value)
            else:
                for child in value.values():
                    self.walk(child)
        elif isinstance(value, list):
            for child in value:
                self.walk(child)

    def same(self, record, path, base=None):
        actual = self.check(record, base)
        expected = self.bind(path)
        require(actual == expected, f"Evidence owner differs: {path}")


def unique_rows(rows, fields, expected, message):
    keys = [tuple(r[k] for k in fields) for r in rows]
    require(len(keys) == len(set(keys)) == len(expected) and set(keys) == set(expected), message)


def projection_representatives(protocol):
    return sorted((dict(case_id=c["case_id"], seed=c["seed"], background=c["background"])
                   for c in protocol["cases"] if c["normalization"] == "raw" and c["V"] == c["S"] == c["T"] == 0),
                  key=lambda r: (r["seed"], r["background"]))


def representative_trace_states(protocol):
    seed = min(c["seed"] for c in protocol["cases"])
    by = {(c["background"], c["normalization"]): c["case_id"] for c in protocol["cases"]
          if c["seed"] == seed and c["V"] == c["S"] == c["T"] == 1}
    return [dict(case_id=by[b, n], arm_id="mean1_n9") for b in ("flat", "sloped") for n in ("raw", "conditioned")] + [
        dict(case_id=by["sloped", "conditioned"], arm_id=a) for a in ("mean2of3_n3", "mean4of3_n9")]


def new_source_paths(repo):
    """Only this study's eight additive components and their matching tests."""
    repo = Path(repo)
    modules = repo / "neurobench/experiments/gamma_ls_difference"
    return [modules / f"background_{name}.py" for name in NEW_COMPONENTS] + [
        repo / "tests" / f"test_background_{name}.py" for name in NEW_COMPONENTS]


def verify_setup_receipts(protocol, prep, paired, integrity):
    cases = {c["case_id"]: c for c in protocol["cases"]}
    arms = {r["arm_id"] for r in protocol["references"]}
    groups = {(c["seed"], c["background"], a) for c in cases.values() for a in arms}
    require(prep["status"] == "PASS" and prep["datasets"] == 84
            and prep["paired_setup_byte_equal"] is True and prep["variance_pairs_early_byte_equal"] is True,
            "Dataset preparation gate failed")
    unique_rows(prep["setup_pairs"], ["case_id"], {(c,) for c in cases}, "Setup parity inventory differs")
    require(all(r["setup_byte_equal"] is True for r in prep["setup_pairs"]), "Setup parity failed")
    require(paired["status"] == "PASS", "Paired calibration is incomplete")
    unique_rows(paired["groups"], ["seed", "background", "arm_id"], groups, "Paired calibration groups differ")
    require(all(r["cases"] == 14 and r["thresholds_equal"] is True and r["floors_equal"] is True
                for r in paired["groups"]), "Paired calibration equality failed")
    require(integrity["status"] == "PASS" and integrity["cells"] == 252 and integrity["datasets"] == 84
            and integrity["source_interval_ui"] == [1, 264] and integrity["numpy_interval"] == [0, 264]
            and integrity["compared_stage_names"] == PAIR_STAGES
            and integrity["all_scoring_seals_preceded_check"] is True
            and integrity["activity_truth_or_metrics_parsed"] is False
            and integrity["detector_rerun"] is False and integrity["empirical_refit"] is False,
            "Variance-prefix integrity scope differs")
    unique_rows(integrity["fixed_target_checks"], ["case_id"], {(c,) for c in cases}, "Fixed-target inventory differs")
    require(all(r["exact_full_file_hash_equal"] is True and set(r["references"]) == arms
                and len(r["A_bindings"]) == 3 and len({b["sha256"] for b in r["A_bindings"]}) == 1
                for r in integrity["fixed_target_checks"]), "Fixed target equality failed")
    unique_rows(integrity["calibration_groups"], ["seed", "background", "arm_id"], groups, "Integrity calibration groups differ")
    for row in integrity["calibration_groups"]:
        expected = {c["case_id"] for c in cases.values() if (c["seed"], c["background"]) == (row["seed"], row["background"])}
        require(len(row["case_ids"]) == 14 and set(row["case_ids"]) == expected
                and row["exact_plans_equal"] is True and row["exact_floors_equal"] is True,
                "Integrity calibration membership differs")
    by = {(c["seed"], c["background"], c["normalization"], c["V"], c["S"], c["T"]): c["case_id"] for c in cases.values()}
    pairs = {(key[0], key[1], key[2], key[4], key[5], case, by[key[:3] + (1,) + key[4:]])
             for key, case in by.items() if key[3] == 0}
    fields = ["seed", "background", "normalization", "S", "T", "V0_case_id", "V1_case_id"]
    require(len(pairs) == 42, "Expected 42 physical variance pairs")
    unique_rows(integrity["raw_pairs"], fields, pairs, "Raw variance-pair inventory differs")
    unique_rows(integrity["stage_pairs"], fields + ["arm_id"], {p + (a,) for p in pairs for a in arms},
                "Stage variance-pair inventory differs")
    for row in integrity["raw_pairs"] + [s for r in integrity["stage_pairs"] for s in r["stages"].values()]:
        require(row["frames_compared"] == 264 and row["bitwise_equal"] is True and row["finite"] is True,
                "Numeric prefix comparison failed")
    for row in integrity["stage_pairs"]:
        require(set(row["stages"]) == set(PAIR_STAGES) and set(row["candidate_streams"]) == {"setup_prefix", "prefix", "audit_candidates"},
                "Incomplete stage/candidate prefix comparison")
        for name, stream in row["candidate_streams"].items():
            require(stream["exact_ordered_records_equal"] is True and stream["fields_removed"] == []
                    and stream["source_interval_ui"] == ([65, 164] if name == "setup_prefix" else [165, 264]),
                    "Candidate prefix parity failed")


def verify_test_receipt(receipt, evidence):
    require(receipt["status"] == "PASS", "Tests are incomplete")
    suites = receipt["suites"]
    required = TEST_SCOPES | ({"independent_recount"} if (REPO / "tests/test_background_recount.py").exists() else set())
    require(len(suites) == len({s["name"] for s in suites}) and required <= {s["name"] for s in suites},
            "Required test scopes are missing or duplicated")
    evidence.walk(receipt)
    total, seen_xml, tested_sources = 0, set(), set()
    for suite in suites:
        xml = Path(suite["xml"]["path"]).resolve()
        require(xml not in seen_xml, "A test XML cannot count as multiple suites")
        seen_xml.add(xml)
        tree = ET.parse(xml)
        cases = list(tree.iter("testcase"))
        require(cases and not any(list(tree.iter(k)) for k in ("failure", "error", "skipped")), "Test XML contains a failed/skipped or empty suite")
        require(len(cases) == suite["test_count"], "Suite test count differs from actual XML")
        # Check summaries as well as testcase elements; never rely on a claimed minimum.
        for node in tree.iter("testsuite"):
            require(all(int(node.get(k, "0")) == 0 for k in ("failures", "errors", "skipped")), "Test XML summary failed")
        require(suite["source"] and suite["tests"], "Test suite must bind source and test files")
        tested_sources.update(Path(b["path"]).resolve() for b in suite["source"] + suite["tests"])
        total += len(cases)
    require(total == receipt["test_count"], "Summed test count differs from verification receipt")
    return total, tested_sources


def verify_independent_counts(independent):
    require(independent["status"] == "PASS" and independent["curve_rows_recounted"] == 5040,
            "Independent numerical recount incomplete")
    require(independent["cells"] == independent["q1_candidate_tables"] == 252
            and independent["setup_q1_maximum_allowed"] == 6,
            "Independent q1 candidate/setup recount incomplete")
    counts = dict(physical_epoch_rows=1260, logical_epoch_rows=1440,
                  physical_threshold_window_rows=5040, logical_threshold_window_rows=5760,
                  paired_rate_contrasts=1800, alias_logical_cells=36)
    require(all(independent["report_checks"].get(k) == value for k, value in counts.items()),
            "Independent report table recount incomplete")


def verify_projection_qa(root, protocol, prep, evidence):
    qa = evidence.get(root / "projection_visual_qa.json")
    expected = projection_representatives(protocol)
    require(qa["status"] == "PASS" and len(expected) == 6, "Projection QA failed")
    unique_rows(qa["reviewed"], ["case_id", "seed", "background"],
                {(r["case_id"], r["seed"], r["background"]) for r in expected}, "Projection representatives differ")
    require(len(qa["covered_case_ids"]) == 84 and set(qa["covered_case_ids"]) == {c["case_id"] for c in protocol["cases"]},
            "Projection QA does not cover all physical cases")
    declared = {r["case_id"]: r for r in prep["projection_records"]}
    require(len(declared) == len(prep["projection_records"]) == 84, "Prepared projection inventory differs")
    for row in qa["reviewed"]:
        require(row["status"] == "PASS", "Projection representative failed")
        evidence.same(row["projection"], declared[row["case_id"]]["figure"]["path"])
        require(row["projection"]["sha256"] == declared[row["case_id"]]["figure"]["sha256"], "Projection QA source differs")
    evidence.same(qa["mapping"], root / "validation/projection_review_mapping.json")
    evidence.check(qa["montage"])
    mapping = evidence.get(qa["mapping"]["path"])
    require(mapping["status"] == "PASS", "Projection review mapping failed")
    wanted = {(r["seed"], r["background"], r["case_id"]) for r in expected}
    unique_rows(mapping["groups"], ["seed", "background", "representative_case_id"], wanted,
                "Projection mapping representatives differ")
    for group in mapping["groups"]:
        members = {c["case_id"] for c in protocol["cases"]
                   if (c["seed"], c["background"]) == (group["seed"], group["background"])}
        require(len(group["case_ids"]) == 14 and set(group["case_ids"]) == members
                and group["exact_setup_projection_equal"] is True, "Projection equality coverage differs")
    evidence.walk(mapping)
    evidence.walk(qa)


def verify_figure_qa(root, report, evidence):
    figures = report["figures"]
    require(report["figure_count"] == len(figures) == len({f["id"] for f in figures}) == 6, "Expected six report figure pairs")
    pngs = {(root / "report" / f["png"]).resolve() for f in figures}
    pdfs = {f["id"]: (root / "report" / f["pdf"]).resolve() for f in figures}
    require(len(pngs) == len(set(pdfs.values())) == 6, "Repeated report figure artifact")
    qa = evidence.get(root / "report/visual_qa.json")
    evidence.same(report["visual_qa_binding"], root / "report/visual_qa.json")
    require(qa["status"] == "PASS" and len(qa["figures"]) == 6
            and {Path(b["path"]).resolve() for b in qa["figures"]} == pngs, "PNG QA does not cover exact report figures")
    evidence.walk(qa)
    pdf = evidence.get(root / "validation/pdf_visual_qa.json")
    require(pdf["status"] == "PASS", "PDF visual QA failed")
    unique_rows(pdf["inspections"], ["figure_id"], {(key,) for key in pdfs}, "PDF QA inventory differs")
    for row in pdf["inspections"]:
        require(row["status"] == "PASS" and row["page"] == 1, "Report PDF inspection failed")
        evidence.same(row["pdf"], pdfs[row["figure_id"]])
        evidence.check(row["render"])
    evidence.walk(pdf)


def verify_trace_qa(root, protocol, evidence):
    qa = evidence.get(root / "representative_trace_visual_qa.json")
    wanted = {(r["case_id"], r["arm_id"]) for r in representative_trace_states(protocol)}
    require(qa["status"] == "PASS" and len(wanted) == 6, "Representative trace QA failed")
    unique_rows(qa["reviewed"], ["case_id", "arm_id"], wanted, "Deterministic trace selection differs")
    for row in qa["reviewed"]:
        require(row["status"] == "PASS", "Representative state review failed")
        audit = root / "audits" / row["case_id"] / row["arm_id"]
        plan = evidence.get(audit / "inventory_plan.json")
        sites = sorted(plan["model_sites"], key=lambda r: r["model_roi_id"])
        evidence.same(row["artifact_index"], audit / "artifact_index.json")
        viewed = {Path(b["path"]).resolve() for b in row["figures"]}
        if sites:
            site = sites[0]["model_roi_id"]
            require(row["review_site"] == site and row["sample_count"] == 464, "Wrong first-site trace or duration")
            model = audit / "2_Model_Annotations"
            required = {model / "figures/traces" / f"{site}.png", model / "videos/closeups" / f"{site}.png"}
            require({p.resolve() for p in required} <= viewed, "Required trace/closeup preview was not reviewed")
            evidence.same(row["metadata"], model / "metadata" / f"{site}.json")
            evidence.same(row["exact_pixel_csv"], model / "exact_pixel_traces" / f"{site}.csv")
            metadata = evidence.get(row["metadata"]["path"])
            require(metadata["all_source_samples_in_trace"] is True and row["pixel_xy"] == metadata["exact_trace_pixel_xy"],
                    "Reviewed exact-pixel trace metadata differ")
        else:
            require(row.get("review_site") is None and row.get("trace_applicable") is False,
                    "Empty model inventory must explicitly mark trace N/A")
            required = {audit / "2_Model_Annotations/videos/model_sequential_full_field.png",
                        audit / "1_Expert_Annotations/videos/expert_sequential_full_field.png"}
            require({p.resolve() for p in required} <= viewed, "Empty-state full fields were not reviewed")
    evidence.walk(qa)


def source_capsule(root, sources, evidence, *, write):
    """No overwrite: an existing capsule must have exactly the same bytes."""
    sources = sorted({Path(p).resolve() for p in sources})
    plans = []
    for source in sources:
        require(source.is_relative_to(REPO), "Source capsule file is outside the repository")
        original = evidence.bind(source)
        dest = root / "source_capsule" / source.relative_to(REPO)
        require(not dest.is_symlink() and dest.resolve().is_relative_to((root / "source_capsule").absolute()),
                "Source capsule cannot alias its original or an external directory")
        if dest.exists():
            require(evidence.bind(dest)["sha256"] == original["sha256"], "Changed source capsule")
        plans.append((source, dest, original))
    anticipated = dict(status="PASS", files=[dict(source=original,
        copy=dict(original, path=str(dest.resolve()))) for source, dest, original in plans])
    manifest_path = root / "source_capsule_manifest.json"
    if manifest_path.exists():
        require(all(dest.exists() for _, dest, _ in plans) and evidence.get(manifest_path) == anticipated,
                "Source capsule manifest differs")
    rows = []
    for source, dest, original in plans:
        if write and not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            with source.open("rb") as src, dest.open("xb") as dst:
                shutil.copyfileobj(src, dst)
        if write:
            copied = evidence.bind(dest)
            require(copied["sha256"] == original["sha256"] and copied["size_bytes"] == original["size_bytes"], "Source copy differs")
            rows.append(dict(source=original, copy=copied))
    if write:
        value = dict(status="PASS", files=rows)
        path = manifest_path
        if path.exists():
            require(evidence.get(path) == value, "Source capsule manifest differs")
        else:
            exclusive_json(path, value)
        evidence.bind(path)
    return len(plans)


def exclusive_json(path, value):
    data = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with Path(path).open("x") as stream:
        stream.write(data)


def finalize(root=ROOT, write=False):
    root = Path(root).resolve()
    require(not (root / "completion_manifest.json").exists(), "Preserve completed background study")
    e = Evidence()
    p = e.get(root / "protocol.json")
    pre = e.get(root / "preflight.json")
    require(pre["status"] == "PASS" and pre["protocol_sha256"] == e.bind(root / "protocol.json")["sha256"], "Preflight binding differs")
    states = matrix(p)
    keys = {(c["case_id"], c["arm_id"]) for c in states}
    for name in ("code_bindings", "kernel_bindings", "baseline_sources"):
        e.walk(p[name])
    for name in ("baseline_completion", "baseline_protocol", "paper_protocol"):
        e.check(p[name])
    require(e.get(p["baseline_completion"]["path"])["status"] == "PASS", "Baseline completion failed")
    prep = e.get(root / "datasets_complete.json")
    paired = e.get(root / "paired_setup_calibration_check.json")
    integrity = e.get(root / "paired_variance_prefix_check.json")
    verify_setup_receipts(p, prep, paired, integrity)
    e.walk(prep); e.walk(integrity)
    require(set(prep["dataset_bindings"]) == {c["case_id"] for c in p["cases"]}, "Dataset bindings differ")
    for c in p["cases"]:
        if not c["reused_dataset"]:
            receipt = e.get(root / "datasets" / c["case_id"] / "background_prepared.json")
            require(receipt["status"] == "PASS" and receipt["case_id"] == c["case_id"]
                    and receipt["seed"] == c["seed"] and receipt["activity_truth_parsed"] is False
                    and receipt["factors"] == dict(V=c["V"], S=c["S"], T=c["T"], B=c["background"], normalization=c["normalization"]),
                    "Fixture preparation receipt differs")
            e.walk(receipt)
    verify_projection_qa(root, p, prep, e)
    require(e.get(root / "computation_complete.json") == dict(status="PASS", cells=252), "Scoring incomplete")
    require(e.get(root / "evaluation_complete.json") == dict(status="PASS", cells=252, curve_rows=5040), "Evaluation incomplete")
    inventory = e.get(root / "all_scoring_seals.json")
    require(inventory["status"] == "PASS" and inventory["cells"] == 252, "All-scoring-seal inventory incomplete")
    e.same(inventory["protocol"], root / "protocol.json")
    actual_seals = []
    for c in states:
        cell = root / "cells" / c["case_id"] / c["arm_id"]
        seal = e.get(cell / "sealed.json")
        require(seal["status"] == "SEALED_BEFORE_ACTIVITY_TRUTH_JOIN", "Unsealed physical state")
        e.walk(seal); actual_seals.append(e.bind(cell / "sealed.json"))
        evaluated = e.get(cell / "evaluated.json")
        require(evaluated["status"] == "PASS", "Unevaluated physical state")
        e.same(evaluated["seal"], cell / "sealed.json"); e.walk(evaluated)
    require(inventory["seals"] == actual_seals, "Exact scoring-seal inventory differs")
    rows = e.get(root / "all_curves.json")
    require(len(rows) == 5040 and all(r["true_positive_count"] == 0 and r["false_positive_count"] == r["proposal_count"]
            and r["framewise_sensitivity"] is None for r in rows), "Null numerical counts/scope differ")
    e.bind(root / "all_curves.tsv")
    reuse = e.get(root / "baseline_replication.json")
    reused = {(c["case_id"], c["arm_id"]) for c in states if c["reused_audit"]}
    require(reuse["status"] == "PASS" and len(reused) == 72, "Reuse gate failed")
    unique_rows(reuse["cells"], ["case_id", "arm_id"], reused, "Exact reuse matrix differs")
    require(all(all(r[k] is True for k in ("all_thresholds_equal", "metrics_equal", "stages_equal", "candidates_equal"))
                for r in reuse["cells"]), "Baseline scientific replication failed")
    independent = e.get(root / "validation/independent_numeric_check.json")
    verify_independent_counts(independent)
    e.walk(independent)
    e.same(independent["all_curves"], root / "all_curves.json")
    e.same(independent["protocol"], root / "protocol.json")
    e.same(independent["evaluation"], root / "evaluation_complete.json")
    audit = e.get(root / "audit_complete.json")
    require(audit["status"] == "PASS" and (audit["cells"], audit["new_cells"], audit["reused_cells"],
            audit["physical_datasets"], audit["logical_cells"]) == (252, 180, 72, 84, 288)
            and audit["logical_to_physical"] == logical_mapping(p), "Full physical/logical audit is incomplete")
    unique_rows(audit["audits"], ["case_id", "arm_id"], keys, "Full audit matrix differs")
    for name, filename in (("protocol", "protocol.json"), ("evaluation", "evaluation_complete.json"),
                           ("all_curves", "all_curves.json"), ("forecast", "audit_forecast.json"),
                           ("display_contract", "display_contract.json"), ("baseline_replication", "baseline_replication.json")):
        e.same(audit[name], root / filename)
    e.walk(audit["validation_code"])
    require({Path(b["path"]).name for b in audit["validation_code"]} == {"background_validate.py", "background_media.py",
            "followup_validate.py", "followup_audit.py", "two_stencil_audit.py", "two_stencil_evaluation.py"}, "Validator source inventory differs")
    forecast = e.get(root / "audit_forecast.json")
    require(forecast["status"] == "PASS", "Audit forecast incomplete")
    for workers in (1, 2, 3):
        frozen_partition(forecast, 0, workers)
    worker_rows = audit["worker_receipts"]
    require(worker_rows and 1 <= len(worker_rows) <= 3, "Missing media worker receipts")
    unique_rows(worker_rows, ["worker"], {(i,) for i in range(len(worker_rows))}, "Media worker receipt inventory differs")
    require(sum(r["cells"] for r in worker_rows) == 252
            and all(r["workers"] == len(worker_rows) for r in worker_rows), "Media worker coverage differs")
    for worker in audit["worker_receipts"]:
        e.walk(worker)
        receipt = e.get(worker["receipt"]["path"])
        require(receipt["status"] == "PASS", "Media worker incomplete")
        e.same(receipt["forecast"], root / "audit_forecast.json")
        verify_worker_assignment(receipt, forecast)
    videos = traces = new_videos = 0
    for row in audit["audits"]:
        folder = root / "audits" / row["case_id"] / row["arm_id"]
        require(row["reused"] == ((row["case_id"], row["arm_id"]) in reused), "Audit reuse classification differs")
        require(set(row["metadata_bindings"]) == set(METADATA), "Audit metadata inventory differs")
        for name, record in row["metadata_bindings"].items():
            e.same(record, folder / name)
        verify_audit_files(folder, e.verifier)
        summary, status, validation = (e.get(folder / name) for name in ("summary.json", "status.json", "validation.json"))
        require(summary == row["summary"] and summary["scientific_audit_complete"] is True
                and summary["expert_roi_count"] == summary["expert_occurrence_count"] == 0
                and summary["video_count"] == summary["model_roi_count"] + 2
                and status["status"] == "complete" and status["scientific_audit_complete"] is True
                and validation["status"] == "passed" and validation["scientific_audit_complete"] is True
                and validation["failures"] == [], "Full null audit metadata failed")
        videos += summary["video_count"]; traces += summary["model_roi_count"]
        new_videos += 0 if row["reused"] else summary["video_count"]
    require(videos == forecast["total_videos"] and new_videos == forecast["new_videos"], "Full media forecast differs")
    report = e.get(root / "report/manifest.json")
    require(all(report[k] is True for k in ("numerical_complete", "scientific_audit_complete", "visual_qa_complete"))
            and report["physical_cell_count"] == 252 and report["logical_cell_count"] == 288, "Report completion gate failed")
    for record in report["artifacts"]:
        e.check(record, root / "report")
    e.walk(report["inputs"]); e.walk(report["reporting_code"]); e.check(report["reporter"])
    e.same(report["audit_verification"]["aggregate"], root / "audit_complete.json")
    verify_figure_qa(root, report, e); verify_trace_qa(root, p, e)
    tests, tested_sources = verify_test_receipt(e.get(root / "verification.json"), e)
    for name in ("resource_preflight.json", "validation/media_launch_resources.json", "document_link_validation.json"):
        receipt = e.get(root / name); require(receipt["status"] == "PASS", f"Required receipt failed: {name}"); e.walk(receipt)
    runtime = e.get(root / "validation/runtime_environment.json")
    require(runtime["status"] in {"RECORDED", "PASS"}, "Runtime environment is unrecorded")
    e.walk(runtime)
    execution = e.get(root / "validation/execution_receipts.json")
    require(execution["status"] == "PASS" and execution["logs"], "Execution logs are unbound")
    e.walk(execution)
    e.bind(root / "audit_validation.log")
    new_sources = new_source_paths(REPO)
    require({f.resolve() for f in new_sources} <= tested_sources, "Verification receipt omits current background source/test files")
    sources = new_sources + [Path(b["path"]) for b in p["code_bindings"]]
    source_count = source_capsule(root, sources, e, write=write)
    e.verifier.assert_unchanged()
    result = dict(status="PASS", completed_utc=datetime.now(timezone.utc).isoformat(), numerical_complete=True,
        scientific_artifact_audit_complete=True, report_visual_qa_complete=True, source_capsule_complete=bool(write),
        counts=dict(cells=252, logical_cells=288, datasets=84, logical_datasets=96, alias_cells=36, references=3,
                    curve_rows=5040, new_audits=180, reused_audits=72, videos=videos, new_videos=new_videos,
                    reused_videos=videos-new_videos, model_roi_traces=traces, expert_occurrences=0,
                    focused_tests=tests, report_figure_pairs=6, manual_trace_reviews=6, setup_projection_representatives=6,
                    setup_projection_covered_datasets=84, source_capsule_files=source_count),
        claim_boundaries=p["scope"], logical_to_physical=logical_mapping(p),
        validation_scope="Fresh one-pass source, numerical, report, and every indexed audit artifact/source SHA check. Existing lossless decode/marker/geometry evidence is inherited through verified hashes. No truth rows parsed, no detector/refit/metric/media rerun, no biological or independent confirmation. Logical aliases are physical evidence reuse.",
        finalization_code=[e.bind(Path(__file__)), e.bind(Path(__file__).with_name("background_media.py")),
                           e.bind(Path(__file__).with_name("background_validate.py")), e.bind(Path(__file__).with_name("followup_validate.py"))],
        unique_hash_count=e.verifier.unique_file_count, bytes_hashed=e.verifier.bytes_hashed,
        evidence_bindings=sorted(e.records.values(), key=lambda b: b["path"]))
    e.verifier.assert_unchanged()
    if write:
        exclusive_json(root / "completion_manifest.json", result)
    print(dict(status="PASS" if write else "READY", counts=result["counts"], evidence_files=len(e.records)), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    finalize(args.root, args.write)


if __name__ == "__main__":
    main()
