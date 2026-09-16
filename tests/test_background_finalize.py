"""Tiny metadata/file fixtures for final closure; no campaign data or rendering."""
import copy
import json
from pathlib import Path

import pytest

from neurobench.experiments.gamma_ls_difference import background_finalize as final


def protocol():
    cases = []
    for seed in (16, 17, 18):
        for background in ("flat", "sloped"):
            for normalization in ("raw", "conditioned"):
                for v in (0, 1):
                    for s in (0, 1):
                        for t in (0, 1):
                            if normalization == "conditioned" and s == t == 0:
                                continue
                            cases.append(dict(case_id=f"{seed}_{background}_{normalization}_{v}{s}{t}", seed=seed,
                                              background=background, normalization=normalization, V=v, S=s, T=t))
    return dict(cases=cases, references=[dict(arm_id=a) for a in ("mean2of3_n3", "mean1_n9", "mean4of3_n9")])


def setup_receipts(p):
    arms = [r["arm_id"] for r in p["references"]]
    prep = dict(status="PASS", datasets=84, paired_setup_byte_equal=True, variance_pairs_early_byte_equal=True,
                setup_pairs=[dict(case_id=c["case_id"], setup_byte_equal=True) for c in p["cases"]])
    groups = [dict(seed=seed, background=b, arm_id=a, cases=14, thresholds_equal=True, floors_equal=True)
              for seed in (16, 17, 18) for b in ("flat", "sloped") for a in arms]
    paired = dict(status="PASS", groups=groups)
    integrity = dict(status="PASS", cells=252, datasets=84, source_interval_ui=[1, 264], numpy_interval=[0, 264],
                     compared_stage_names=final.PAIR_STAGES, all_scoring_seals_preceded_check=True,
                     activity_truth_or_metrics_parsed=False, detector_rerun=False, empirical_refit=False,
                     fixed_target_checks=[dict(case_id=c["case_id"], references=arms,
                         A_bindings=[dict(sha256="same") for a in arms], exact_full_file_hash_equal=True) for c in p["cases"]],
                     calibration_groups=[], raw_pairs=[], stage_pairs=[])
    for group in groups:
        ids = [c["case_id"] for c in p["cases"] if (c["seed"], c["background"]) == (group["seed"], group["background"])]
        integrity["calibration_groups"].append(dict(group, case_ids=ids, exact_plans_equal=True, exact_floors_equal=True))
    for c in p["cases"]:
        if c["V"]:
            continue
        high = next(d for d in p["cases"] if d["V"] == 1 and all(d[k] == c[k] for k in ("seed", "background", "normalization", "S", "T")))
        row = {k:c[k] for k in ("seed", "background", "normalization", "S", "T")}
        row.update(V0_case_id=c["case_id"], V1_case_id=high["case_id"])
        equality = dict(frames_compared=264, bitwise_equal=True, finite=True)
        integrity["raw_pairs"].append(dict(row, **equality))
        for arm in arms:
            streams = {name: dict(exact_ordered_records_equal=True, fields_removed=[],
                source_interval_ui=[65, 164] if name == "setup_prefix" else [165, 264])
                for name in ("setup_prefix", "prefix", "audit_candidates")}
            integrity["stage_pairs"].append(dict(row, arm_id=arm, stages={s:dict(equality) for s in final.PAIR_STAGES}, candidate_streams=streams))
    return prep, paired, integrity


def test_complete_setup_and_prefix_inventories_are_exact():
    p = protocol()
    prep, paired, integrity = setup_receipts(p)
    final.verify_setup_receipts(p, prep, paired, integrity)
    assert [len(integrity[k]) for k in ("fixed_target_checks", "calibration_groups", "raw_pairs", "stage_pairs")] == [84, 18, 42, 126]


@pytest.mark.parametrize("kind", ["duplicate_target", "wrong_pair", "missing_stage", "not_equal", "truth_read", "wrong_window", "changed_setup"])
def test_prefix_gates_reject_same_count_swaps_or_failed_parity(kind):
    p = protocol()
    prep, paired, x = setup_receipts(p)
    if kind == "duplicate_target":
        x["fixed_target_checks"][-1] = x["fixed_target_checks"][0]
    elif kind == "wrong_pair":
        x["raw_pairs"][0]["V1_case_id"] = x["raw_pairs"][1]["V1_case_id"]
    elif kind == "missing_stage":
        x["stage_pairs"][0]["stages"].pop("C")
    elif kind == "not_equal":
        x["stage_pairs"][0]["stages"]["C"]["bitwise_equal"] = False
    elif kind == "truth_read":
        x["activity_truth_or_metrics_parsed"] = True
    elif kind == "wrong_window":
        x["stage_pairs"][0]["candidate_streams"]["prefix"]["source_interval_ui"] = [165, 265]
    else:
        prep["setup_pairs"][0]["setup_byte_equal"] = False
    with pytest.raises(ValueError):
        final.verify_setup_receipts(p, prep, paired, x)


def test_deterministic_visual_selection_is_independent_of_case_order():
    p = protocol()
    projections, traces = final.projection_representatives(p), final.representative_trace_states(p)
    assert len(projections) == len(traces) == 6
    assert len({r["case_id"] for r in projections}) == 6
    assert len({(r["case_id"],r["arm_id"]) for r in traces}) == 6
    assert {r["arm_id"] for r in traces} == {a["arm_id"] for a in p["references"]}
    p["cases"].reverse()
    assert final.projection_representatives(p) == projections
    assert final.representative_trace_states(p) == traces


def _tests_receipt(tmp_path, monkeypatch):
    monkeypatch.setattr(final, "REPO", tmp_path)
    e = final.Evidence()
    source = tmp_path / "source.py"; source.write_text("# tiny fixture\n")
    suite_rows = []
    for name in sorted(final.TEST_SCOPES):
        xml = tmp_path / f"{name}.xml"
        xml.write_text('<testsuites><testsuite tests="1" failures="0" errors="0" skipped="0"><testcase name="one"/></testsuite></testsuites>')
        suite_rows.append(dict(name=name, test_count=1, xml=e.bind(xml), source=[e.bind(source)], tests=[e.bind(source)]))
    return dict(status="PASS", test_count=len(suite_rows), suites=suite_rows)


def test_count_is_parsed_from_actual_xml_without_invented_minimum(tmp_path, monkeypatch):
    receipt = _tests_receipt(tmp_path, monkeypatch)
    count, sources = final.verify_test_receipt(receipt, final.Evidence())
    assert count == 6 and sources == {(tmp_path / "source.py").resolve()}


@pytest.mark.parametrize("kind", ["failed", "claimed_count", "total", "duplicate_xml", "summary_only", "independent_missing"])
def test_test_receipt_rejects_failed_or_double_counted_evidence(tmp_path, monkeypatch, kind):
    receipt = _tests_receipt(tmp_path, monkeypatch)
    if kind in ("failed", "summary_only"):
        xml = Path(receipt["suites"][0]["xml"]["path"])
        xml.write_text('<testsuite tests="1" failures="1"><testcase name="one">' + ('<failure/>' if kind == "failed" else '') + '</testcase></testsuite>')
        receipt["suites"][0]["xml"] = final.Evidence().bind(xml)
    elif kind == "claimed_count":
        receipt["suites"][0]["test_count"] += 1
    elif kind == "total":
        receipt["test_count"] += 1
    elif kind == "duplicate_xml":
        receipt["suites"][1]["xml"] = receipt["suites"][0]["xml"]
    else:
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests/test_background_recount.py").write_text("# current scope\n")
    with pytest.raises(ValueError):
        final.verify_test_receipt(receipt, final.Evidence())


def test_capsule_readiness_does_not_write_then_copy_is_hash_exact_and_immutable(tmp_path, monkeypatch):
    repo = tmp_path / "repo"; repo.mkdir()
    root = repo / "Outputs/run"; root.mkdir(parents=True)
    source = repo / "source.py"; source.write_text("print('frozen')\n")
    monkeypatch.setattr(final, "REPO", repo)
    assert final.source_capsule(root, [source], final.Evidence(), write=False) == 1
    assert list(root.iterdir()) == []
    final.source_capsule(root, [source], final.Evidence(), write=True)
    copied = root / "source_capsule/source.py"
    assert copied.read_bytes() == source.read_bytes()
    before = (root / "source_capsule_manifest.json").read_bytes()
    final.source_capsule(root, [source], final.Evidence(), write=True)
    assert (root / "source_capsule_manifest.json").read_bytes() == before
    copied.write_text("changed\n")
    with pytest.raises(ValueError, match="Changed source capsule"):
        final.source_capsule(root, [source], final.Evidence(), write=False)
    assert copied.read_text() == "changed\n"


def test_capsule_cannot_symlink_original(tmp_path, monkeypatch):
    root = tmp_path / "run"; root.mkdir()
    source = tmp_path / "source.py"; source.write_text("frozen\n")
    monkeypatch.setattr(final, "REPO", tmp_path)
    (root / "source_capsule").mkdir()
    (root / "source_capsule/source.py").symlink_to(source)
    with pytest.raises(ValueError, match="cannot alias"):
        final.source_capsule(root, [source], final.Evidence(), write=True)


def test_truth_parsing_and_completed_root_mutations_are_forbidden(tmp_path):
    (tmp_path / "experts.json").write_text("[]")
    with pytest.raises(ValueError, match="does not parse truth"):
        final.Evidence().get(tmp_path / "experts.json")
    (tmp_path / "completion_manifest.json").write_text('{"status":"PASS"}')
    before = {p.name:p.read_bytes() for p in tmp_path.iterdir()}
    for write in (False, True):
        with pytest.raises(ValueError, match="Preserve completed"):
            final.finalize(tmp_path, write=write)
    assert {p.name:p.read_bytes() for p in tmp_path.iterdir()} == before


def test_exclusive_manifest_cannot_overwrite_existing_record(tmp_path):
    path = tmp_path / "seal.json"
    final.exclusive_json(path, dict(status="PASS"))
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        final.exclusive_json(path, dict(status="OTHER"))
    assert path.read_bytes() == before


def test_ready_check_rejects_stale_capsule_manifest_without_writes(tmp_path, monkeypatch):
    root = tmp_path / "run"; root.mkdir()
    source = tmp_path / "source.py"; source.write_text("frozen\n")
    monkeypatch.setattr(final, "REPO", tmp_path)
    final.source_capsule(root, [source], final.Evidence(), write=True)
    path = root / "source_capsule_manifest.json"
    final_record = json.loads(path.read_text()); final_record["files"] = []
    path.write_text(json.dumps(final_record))
    before = {str(p.relative_to(root)):p.read_bytes() for p in root.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="manifest differs"):
        final.source_capsule(root, [source], final.Evidence(), write=False)
    assert {str(p.relative_to(root)):p.read_bytes() for p in root.rglob("*") if p.is_file()} == before


def _figure_receipts(root):
    out = root / "report"; out.mkdir()
    validation = root / "validation"; validation.mkdir()
    e = final.Evidence(); figures, inspections = [], []
    for i in range(6):
        stem = f"figure{i}"
        png, pdf, render = out / f"{stem}.png", out / f"{stem}.pdf", validation / f"{stem}.png"
        png.write_bytes(b"synthetic PNG fixture" + bytes([i])); pdf.write_bytes(b"synthetic PDF fixture" + bytes([i])); render.write_bytes(b"test rendered preview" + bytes([i]))
        figures.append(dict(id=stem, png=png.name, pdf=pdf.name))
        inspections.append(dict(figure_id=stem, status="PASS", page=1, pdf=e.bind(pdf), render=e.bind(render)))
    (out / "visual_qa.json").write_text(json.dumps(dict(status="PASS", figures=[e.bind(out / f["png"]) for f in figures])))
    (validation / "pdf_visual_qa.json").write_text(json.dumps(dict(status="PASS", inspections=inspections)))
    return dict(figure_count=6, figures=figures, visual_qa_binding=e.bind(out / "visual_qa.json"))


@pytest.mark.parametrize("kind", ["duplicate_pdf", "changed_pdf", "missing_png", "wrong_page"])
def test_visual_completion_needs_each_exact_current_png_and_pdf(tmp_path, kind):
    report = _figure_receipts(tmp_path)
    final.verify_figure_qa(tmp_path, report, final.Evidence())
    path = tmp_path / "validation/pdf_visual_qa.json"
    pdf = json.loads(path.read_text())
    if kind == "duplicate_pdf":
        pdf["inspections"][-1] = pdf["inspections"][0]
        path.write_text(json.dumps(pdf))
    elif kind == "changed_pdf":
        (tmp_path / "report/figure0.pdf").write_bytes(b"changed after review")
    elif kind == "missing_png":
        qa = tmp_path / "report/visual_qa.json"
        record = json.loads(qa.read_text()); record["figures"].pop()
        qa.write_text(json.dumps(record)); report["visual_qa_binding"] = final.Evidence().bind(qa)
    else:
        pdf["inspections"][0]["page"] = 2; path.write_text(json.dumps(pdf))
    with pytest.raises(ValueError):
        final.verify_figure_qa(tmp_path, report, final.Evidence())


def test_independent_recount_accepts_additional_diagnostics_but_requires_all_agreed_counts():
    receipt = dict(status="PASS", cells=252, curve_rows_recounted=5040, q1_candidate_tables=252, setup_q1_maximum_allowed=6,
        report_checks=dict(physical_epoch_rows=1260, logical_epoch_rows=1440, physical_threshold_window_rows=5040,
            logical_threshold_window_rows=5760, paired_rate_contrasts=1800, alias_logical_cells=36,
            rate_contrast_windows=["early", "late", "application", "settled_early", "settled_late"], primary_late_counts=[]))
    final.verify_independent_counts(receipt)
    receipt["report_checks"]["alias_logical_cells"] = 35
    with pytest.raises(ValueError, match="report table recount"):
        final.verify_independent_counts(receipt)


def test_source_inventory_contains_only_eight_study_components_and_matching_tests(tmp_path):
    tests = tmp_path / "tests"; tests.mkdir()
    (tests / "test_background_residual.py").write_text("# pre-existing unrelated test\n")
    expected = {"fixtures", "study", "integrity", "media", "validate", "report", "recount", "finalize"}
    paths = final.new_source_paths(tmp_path)
    assert len(paths) == len(set(paths)) == 16
    assert {p.name for p in paths} == ({f"background_{name}.py" for name in expected}
                                    | {f"test_background_{name}.py" for name in expected})
    assert tests / "test_background_residual.py" not in paths
