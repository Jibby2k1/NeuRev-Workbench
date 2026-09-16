"""Focused source/inventory/reuse gates using real tiny sealed JSON fixtures."""
import copy
import hashlib
import json

import pytest

from neurobench.experiments.gamma_ls_difference.background_validate import (
    ALL_STAGES, FileVerifier, _workers,
    baseline_audit_records, original_media_owner, validate, verify_cell, verify_inventory, verify_reuse,
    verify_worker_assignment,
)
from neurobench.experiments.gamma_ls_difference.background_media import balanced_partitions


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True))
    return binding(path)


def binding(path):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), size_bytes=path.stat().st_size)


def fixture(root):
    cell, folder = root / "cell", root / "dataset"
    cell.mkdir(parents=True)
    folder.mkdir()
    stages = {}
    for name in ALL_STAGES:
        path = cell / f"{name}.bin"
        path.write_bytes(("stage:Z" if name == "Score" else f"stage:{name}").encode())
        stages[name] = binding(path)
    full = folder / "input_source.bin"
    full.write_bytes(b"full halo input source")
    write(folder / "metadata.json", dict(truth_mode="fully_synthetic", original_source_offset_xy=[49, 49]))
    write(folder / "experts.json", [])
    write(folder / "active.json", [])
    write(folder / "prepared.json", dict(status="PASS", input=binding(full)))
    candidates = [dict(proposal_id="fixed_0001", source_frame_ui=3, x_px=6, y_px=6, score=3.,
                       threshold_z=2., candidate_rank_within_frame=1)]
    write(cell / "audit_candidates.json", candidates)
    write(cell / "prefix.json", candidates)
    write(cell / "setup_prefix.json", [])
    plan = [dict(threshold_id=f"q{i}", threshold=float(i + 1)) for i in range(10)]
    write(cell / "threshold_plan.json", plan)
    write(cell / "calibration.json", dict(threshold_id="q1", threshold=2., threshold_z=2.,
        threshold_frozen_from_calibration_only=True, setup_source_frames_ui=[1, 2], scale_floor=.1,
        target_proposals_per_frame=.1, eligible_area_px=16, setup_proposal_budget=1, setup_proposal_count=1,
        setup_budget_per_reference_area_frame=1., reference_area_px=160, application_source_start_ui=3,
        application_source_stop_ui=4, application_frame_count=2))
    seal = dict(status="SEALED_BEFORE_ACTIVITY_TRUTH_JOIN", stages=stages,
        dataset_bindings=[binding(folder / n) for n in ("metadata.json", "experts.json", "active.json", "prepared.json")],
        prefix=binding(cell / "prefix.json"), setup_prefix=binding(cell / "setup_prefix.json"),
        audit_candidates=binding(cell / "audit_candidates.json"), threshold_plan=binding(cell / "threshold_plan.json"),
        calibration=binding(cell / "calibration.json"))
    write(cell / "sealed.json", seal)
    curves = dict(curve_rows=[dict(p, match_radius_px=r, proposal_count=1, true_positive_count=0,
                                  false_positive_count=1, event_count=0, active_region_frame_count=0, metric_candidate_sha256="fixed")
                             for p in plan for r in (2., 6.)], event_rows=[])
    write(cell / "curves.json", curves)
    for radius in (2, 6):
        write(cell / f"operating_metrics_r{radius}.json", dict(summary=dict(proposal_count=1), event_rows=[]))
    reseal_evaluation(cell)
    return cell, folder, full


def reseal_evaluation(cell):
    write(cell / "evaluated.json", dict(status="PASS", seal=binding(cell / "sealed.json"),
        outputs=[binding(cell / n) for n in ("curves.json", "operating_metrics_r2.json", "operating_metrics_r6.json")]))


def test_complete_cell_verifies_full_halo_and_all_scientific_stages(tmp_path):
    cell, folder, full = fixture(tmp_path)
    verifier = FileVerifier()
    result = verify_cell(cell, folder, verifier)
    verifier.assert_unchanged()
    assert len(result[3]["curve_rows"]) == 20
    assert full.resolve() in verifier._files
    assert all((cell / f"{name}.bin").resolve() in verifier._files for name in ALL_STAGES)
    # A second identical state check does not read large files twice.
    before = verifier.bytes_hashed
    verify_cell(cell, folder, verifier)
    assert verifier.bytes_hashed == before


@pytest.mark.parametrize("kind", ["halo", "spread", "score"])
def test_changed_numeric_or_full_input_source_is_rejected(tmp_path, kind):
    cell, folder, full = fixture(tmp_path)
    (full if kind == "halo" else cell / ("Spread.bin" if kind == "spread" else "Score.bin")).write_bytes(b"changed")
    with pytest.raises(ValueError, match="mismatch"):
        verify_cell(cell, folder, FileVerifier())


def test_missing_radius_cannot_pass_even_with_updated_evaluation_receipt(tmp_path):
    cell, folder, _ = fixture(tmp_path)
    curves = json.loads((cell / "curves.json").read_text())
    curves["curve_rows"] = [r for r in curves["curve_rows"] if r["match_radius_px"] == 6.]
    write(cell / "curves.json", curves)
    reseal_evaluation(cell)
    with pytest.raises(ValueError, match="both radii"):
        verify_cell(cell, folder, FileVerifier())


def test_candidate_path_cannot_bypass_its_seal(tmp_path):
    cell, folder, _ = fixture(tmp_path)
    frozen = cell / "sealed_candidates.json"
    frozen.write_bytes((cell / "audit_candidates.json").read_bytes())
    seal = json.loads((cell / "sealed.json").read_text())
    seal["audit_candidates"] = binding(frozen)
    write(cell / "sealed.json", seal)
    reseal_evaluation(cell)
    with pytest.raises(ValueError, match="path differs.*audit_candidates"):
        verify_cell(cell, folder, FileVerifier())


def test_null_truth_cannot_gain_experts_under_an_updated_seal(tmp_path):
    cell, folder, _ = fixture(tmp_path)
    write(folder / "experts.json", [dict(event_id="not_null")])
    seal = json.loads((cell / "sealed.json").read_text())
    seal["dataset_bindings"] = [binding(folder / n) for n in ("metadata.json", "experts.json", "active.json", "prepared.json")]
    write(cell / "sealed.json", seal)
    reseal_evaluation(cell)
    with pytest.raises(ValueError, match="empty synthetic truth"):
        verify_cell(cell, folder, FileVerifier())


def test_exact_baseline_reuse_all_stages_and_candidate_ids(tmp_path):
    nc, nf, _ = fixture(tmp_path / "new")
    oc, of, _ = fixture(tmp_path / "old")
    verifier = FileVerifier()
    new, old = verify_cell(nc, nf, verifier), verify_cell(oc, of, verifier)
    verify_reuse(new, old, nc, oc, verifier)
    changed = copy.deepcopy(new)
    changed[0]["stages"]["A"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="stage differs: A"):
        verify_reuse(changed, old, nc, oc, verifier)
    changed = copy.deepcopy(new)
    changed[2][0]["proposal_id"] = "renamed_candidate"
    with pytest.raises(ValueError, match="candidate rows differ"):
        verify_reuse(changed, old, nc, oc, verifier)


def test_native_metric_cutoff_change_cannot_hide_behind_identical_counts(tmp_path):
    nc, nf, _ = fixture(tmp_path / "new")
    oc, of, _ = fixture(tmp_path / "old")
    changed = json.loads((nc / "operating_metrics_r6.json").read_text())
    changed["summary"]["threshold_z"] = 0.
    write(nc / "operating_metrics_r6.json", changed)
    reseal_evaluation(nc)
    verifier = FileVerifier()
    with pytest.raises(ValueError, match="r6 metrics differ"):
        verify_reuse(verify_cell(nc, nf, verifier), verify_cell(oc, of, verifier), nc, oc, verifier)


def test_prefix_hash_change_prevents_media_reuse_even_with_same_selected_candidates(tmp_path):
    nc, nf, _ = fixture(tmp_path / "new")
    oc, of, _ = fixture(tmp_path / "old")
    verifier = FileVerifier()
    new, old = verify_cell(nc, nf, verifier), verify_cell(oc, of, verifier)
    changed = copy.deepcopy(new)
    changed[0]["prefix"]["sha256"] = "a" * 64
    with pytest.raises(ValueError, match="bytes differ: prefix"):
        verify_reuse(changed, old, nc, oc, verifier)


def test_null_inventory_has_no_experts_but_keeps_model_media(tmp_path):
    summary = dict(scientific_audit_complete=True, video_count=5, expert_roi_count=0,
                   model_roi_count=3, expert_occurrence_count=0, comparison_trace_count=0, model_proposal_count=9)
    status = dict(status="complete", scientific_audit_complete=True)
    validation = dict(status="passed", scientific_audit_complete=True, failures=[])
    wanted = dict(expected_videos=5, expert_rois=0, model_rois=3, expert_occurrences=0)
    verify_inventory(summary, status, validation, dict(complete=True, expert_applicable=False), wanted, 9)
    with pytest.raises(ValueError, match="video_count"):
        verify_inventory(dict(summary, video_count=4), status, validation, dict(complete=True, expert_applicable=False), wanted, 9)


def test_three_workers_cover_252_exactly(tmp_path):
    for worker in range(3):
        write(tmp_path / f"media_worker_{worker}.json", dict(status="PASS", worker=worker, workers=3, cells=84))
        (tmp_path / f"media_worker_{worker}.log").write_text("completed\n")
    assert [r["cells"] for r in _workers(tmp_path, FileVerifier(), 252)] == [84, 84, 84]
    write(tmp_path / "media_worker_2.json", dict(status="PASS", worker=2, workers=3, cells=83))
    with pytest.raises(ValueError, match="full partition"):
        _workers(tmp_path, FileVerifier(), 252)


def test_completed_root_never_revalidated_or_rewritten(tmp_path):
    write(tmp_path / "completion_manifest.json", dict(status="PASS"))
    write(tmp_path / "audit_complete.json", dict(status="PASS", frozen=True))
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    with pytest.raises(FileExistsError, match="immutable"):
        validate(tmp_path)
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


def test_worker_receipt_requires_exact_assigned_keys_not_only_count():
    rows = [dict(case_id=f"case{i:03d}", arm_id="arm", reused=False, model_closeup_frames=i) for i in range(252)]
    bins = balanced_partitions(rows, 3)
    forecast = dict(cells=rows, worker_partitions={"3": bins})
    receipt = dict(bins[0], workers=3)
    verify_worker_assignment(receipt, forecast)
    receipt = copy.deepcopy(receipt)
    receipt["assigned_keys"][0] = bins[1]["assigned_keys"][0]
    with pytest.raises(ValueError, match="exact frozen state assignment"):
        verify_worker_assignment(receipt, forecast)


def test_original_reference_contract_owner_is_preserved_through_noise_baseline(tmp_path):
    nc, nf, _ = fixture(tmp_path / "noise")
    oc, of, _ = fixture(tmp_path / "reference")
    verifier = FileVerifier()
    noise = verify_cell(nc, nf, verifier)
    contract = dict(numeric_threshold_paths=dict(calibration=str(oc / "calibration.json")),
                    source_binding=dict(candidate_seal_sha256=binding(oc / "sealed.json")["sha256"]))
    owner_cell, owner_seal, owner_op = original_media_owner(contract, nc, noise, verifier)
    assert owner_cell == oc.resolve()
    assert owner_seal == json.loads((oc / "sealed.json").read_text())
    assert owner_op == json.loads((oc / "calibration.json").read_text())
    # The intermediate noise cell has equivalent scientific bytes but its own seal.
    assert binding(oc / "sealed.json")["sha256"] != binding(nc / "sealed.json")["sha256"]
    contract["source_binding"]["candidate_seal_sha256"] = binding(nc / "sealed.json")["sha256"]
    with pytest.raises(ValueError, match="owner seal differs"):
        original_media_owner(contract, nc, noise, verifier)


def test_transitive_media_owner_with_changed_scientific_stage_is_rejected(tmp_path):
    nc, nf, _ = fixture(tmp_path / "noise")
    oc, of, _ = fixture(tmp_path / "reference")
    (oc / "M.bin").write_bytes(b"different scientific reference mean")
    seal = json.loads((oc / "sealed.json").read_text())
    seal["stages"]["M"] = binding(oc / "M.bin")
    write(oc / "sealed.json", seal)
    reseal_evaluation(oc)
    contract = dict(numeric_threshold_paths=dict(calibration=str(oc / "calibration.json")),
                    source_binding=dict(candidate_seal_sha256=binding(oc / "sealed.json")["sha256"]))
    verifier = FileVerifier()
    with pytest.raises(ValueError, match="stage differs: M"):
        original_media_owner(contract, nc, verify_cell(nc, nf, verifier), verifier)


def test_baseline_aggregate_must_be_bound_by_completed_baseline(tmp_path):
    record = write(tmp_path / "audit_complete.json", dict(status="PASS", cells=1,
                   audits=[dict(case_id="case", arm_id="arm")]))
    completion = dict(status="PASS", evidence_bindings=[record])
    assert baseline_audit_records(completion, tmp_path, FileVerifier()) == {("case", "arm"): dict(case_id="case", arm_id="arm")}
    with pytest.raises(ValueError, match="does not bind"):
        baseline_audit_records(dict(status="PASS", evidence_bindings=[]), tmp_path, FileVerifier())
    write(tmp_path / "audit_complete.json", dict(status="PASS", cells=0, audits=[]))
    with pytest.raises(ValueError, match="mismatch"):
        baseline_audit_records(completion, tmp_path, FileVerifier())
