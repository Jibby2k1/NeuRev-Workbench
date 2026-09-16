"""Small scientific integration checks for the reference-only study runner."""
import math
from pathlib import Path

import numpy as np
import pytest

from neurobench.experiments.gamma_ls_difference import reference_study as study
from neurobench.experiments.gamma_ls_difference.spatiotemporal_metrics import (
    evaluate_framewise, seal_candidates,
)


def _protocol(kind="null", references=("mean2of3_n3",)):
    return dict(cases=[dict(case_id="toy", kind=kind, seed=20260916, separation_px=None)],
                references=[dict(arm_id=arm) for arm in references])


def test_only_exact_crowding_anchor_reuses_audit():
    p = _protocol("crowding", (study.ANCHOR, "mean1_n3"))
    p["cases"].append(dict(case_id="null", kind="null", seed=20260916))
    cells = study.cells(p)
    assert len(cells) == 4
    assert [(c["case_id"], c["arm_id"]) for c in cells if c["reused_audit"]] == [("toy", study.ANCHOR)]
    assert all((c["input_mode"], c["readout"], c["window"], c["calibration_method"]) ==
               ("level", "Z", 3, "global") for c in cells)


def test_anchor_returns_exact_existing_stages_without_convolution(tmp_path, monkeypatch):
    state = dict(stages={"A": {"path": "unchanged-A", "sha256": "original"}}, scale_floor=2.)
    monkeypatch.setattr(study, "prepare_mode", lambda *args: state)
    def forbidden(*args, **kwargs):
        raise AssertionError("Exact anchor must not rebuild reference or target stages")
    monkeypatch.setattr(study, "iter_chunks", forbidden)
    assert study.score_reference(tmp_path, "toy", study.ANCHOR) is state


def test_reference_uses_fixed_A_and_only_warm_setup_spread_for_floor(tmp_path, monkeypatch):
    folder = tmp_path / "datasets/toy"
    folder.mkdir(parents=True)
    shape = (166, 14, 14)
    a = np.full(shape, 7., np.float32)
    np.save(folder / "A.npy", a)
    np.save(folder / "level.npy", np.zeros(shape, np.float32))
    # The chunk iterator is replaced by exact controlled stage fields here.
    np.save(folder / "level_source.npy", np.zeros((166, 1, 1), np.float32))
    state = dict(stages={"A": study.binding(folder / "A.npy"),
                         "X": study.binding(folder / "level.npy"),
                         "Raw": study.binding(folder / "level.npy")}, scale_floor=999.)
    monkeypatch.setattr(study, "prepare_mode", lambda *args: state)
    (tmp_path / "kernels").mkdir()
    w = np.ones((3, 3), np.float64); w[1, 1] = 0; w /= w.sum()
    np.save(tmp_path / "kernels/mean1_n3.npy", w)
    def chunks(source, kernels, **kwargs):
        np.testing.assert_array_equal(kernels.reference[0], w)
        for start in range(0, 166, 8):
            stop = min(166, start + 8)
            spread = np.asarray([1000. if t < 64 else 2. if t < 164 else 10000.
                                 for t in range(start, stop)])[:, None, None]
            full = (stop - start, 112, 112)
            yield start, stop, dict(A=np.full(full, -999.), M=np.ones(full),
                                   variance=np.broadcast_to(spread ** 2, full))
    monkeypatch.setattr(study, "iter_chunks", chunks)
    result = study.score_reference(tmp_path, "toy", "mean1_n3")
    assert result["scale_floor"] == 2.
    assert result["stages"]["A"] == state["stages"]["A"]
    assert result["target_exactly_reused"]
    np.testing.assert_array_equal(np.load(result["stages"]["C"]["path"]), np.full(shape, 6, np.float32))
    z = np.load(result["stages"]["Z"]["path"])
    np.testing.assert_array_equal(z[64:164], np.full((100, 14, 14), 3., np.float32))
    np.testing.assert_allclose(z[:64], .006)
    np.testing.assert_allclose(z[164:], .0006)
    np.testing.assert_array_equal(np.load(folder / "A.npy"), a)


def test_all_cells_must_be_sealed_before_any_activity_truth_join(tmp_path, monkeypatch):
    p = _protocol(references=("mean1_n3", "mean1_n15"))
    monkeypatch.setattr(study, "load", lambda root: p)
    first = tmp_path / "cells/toy/mean1_n3"
    first.mkdir(parents=True)
    (first / "sealed.json").write_text("{}")
    def forbidden(path):
        raise AssertionError(f"No cell data/truth should be read before the global seal gate: {path}")
    monkeypatch.setattr(study, "read", forbidden)
    with pytest.raises(RuntimeError, match="Seal all"):
        study.evaluate(tmp_path)
    assert not (first / "evaluated.json").exists()


@pytest.mark.parametrize("radius", [2., 6.])
@pytest.mark.parametrize("tau", [0., 8., 10., math.inf])
def test_global_threshold_prefix_matches_fresh_filtered_assignment(radius, tau):
    prefix = [dict(proposal_id="high", source_frame_ui=1, x_px=24, y_px=20, score=10.),
              dict(proposal_id="low", source_frame_ui=1, x_px=17, y_px=20, score=8.),
              dict(proposal_id="late", source_frame_ui=2, x_px=25, y_px=20, score=9.)]
    events = [dict(event_id=f"e{i}", canonical_roi_id=f"r{i}", source_start_ui=1,
                   source_stop_ui=2, x_px=x, y_px=20) for i, x in enumerate((22, 26))]
    active = [dict(e, source_frame_ui=t) for t in (1, 2) for e in events]
    full = evaluate_framewise(seal_candidates(prefix, source_frames_ui=[1, 2]), active, events,
                              threshold_z=tau, match_radius_px=radius)
    filtered = [r for r in prefix if r["score"] > tau]
    fresh = evaluate_framewise(seal_candidates(filtered, source_frames_ui=[1, 2]), active, events,
                               threshold_z=0., match_radius_px=radius)
    fields = ("proposal_count", "true_positive_count", "false_positive_count", "false_negative_count",
              "recovered_event_count", "duplicate_near_active_region_count", "precision", "framewise_sensitivity")
    assert {k: full["summary"][k] for k in fields} == {k: fresh["summary"][k] for k in fields}
    assert full["proposal_rows"] == fresh["proposal_rows"]
    assert [(r["event_id"], r["first_matched_source_frame_ui"]) for r in full["event_rows"]] == [
           (r["event_id"], r["first_matched_source_frame_ui"]) for r in fresh["event_rows"]]


def test_frozen_baseline_sources_are_verified_on_protocol_load(tmp_path):
    sources = []
    for name in ("baseline_completion.json", "baseline_protocol.json", "baseline_source.json"):
        study.write_json(tmp_path / name, dict(status="PASS"))
        sources.append(study.binding(tmp_path / name))
    p = dict(code_bindings=[], kernel_bindings=[], baseline_completion=sources[0],
             baseline_protocol=sources[1], baseline_sources=[sources[2]])
    study.write_json(tmp_path / "protocol.json", p)
    study.write_json(tmp_path / "preflight.json", dict(protocol_sha256=study.sha256(tmp_path / "protocol.json")))
    assert study.load(tmp_path) == p
    (tmp_path / "baseline_source.json").write_text('{"status":"changed"}')
    with pytest.raises((RuntimeError, ValueError)):
        study.load(tmp_path)


@pytest.mark.parametrize("null_truth", [False, True])
def test_tiny_scoring_evaluation_null_semantics_and_resume_hashes(tmp_path, monkeypatch, null_truth):
    folder = tmp_path / "datasets/toy"
    folder.mkdir(parents=True)
    meta = dict(truth_mode="fully_synthetic", source_frames_ui=list(range(1, 167)),
                setup_source_frames_ui=list(range(65, 165)), application_source_frames_ui=[165, 166])
    event = dict(event_id="weak", observation_id="weak", canonical_roi_id="weak", source_role="weak",
                 burst_id=1, source_start_ui=165, source_stop_ui=165, x_px=15, y_px=15)
    study.write_json(folder / "metadata.json", meta)
    study.write_json(folder / "experts.json", [] if null_truth else [event])
    study.write_json(folder / "active.json", [] if null_truth else [dict(event, source_frame_ui=165)])
    study.write_json(folder / "prepared.json", dict(status="PASS"))
    score = np.zeros((166, 30, 30), np.float32)
    score[:164, 15, 15] = 3.
    score[164:, 15, 15] = 10.
    np.save(folder / "score.npy", score)
    state = dict(stages={k: study.binding(folder / "score.npy") for k in ("A", "M", "Spread", "C", "Z", "X", "Raw")}, scale_floor=1.)
    p = _protocol("null" if null_truth else "crowding")
    monkeypatch.setattr(study, "load", lambda root: p)
    monkeypatch.setattr(study, "score_reference", lambda *args: state)
    # Only the cell list is reduced for this bounded harness. The prepared
    # receipt retains the production cardinality expected by run().
    bs = [study.binding(folder / n) for n in ("metadata.json", "experts.json", "active.json", "prepared.json")]
    study.write_json(tmp_path / "datasets_complete.json", dict(status="PASS", datasets=21, dataset_bindings={"toy": bs}))
    study.run(tmp_path)
    out = tmp_path / "cells/toy/mean2of3_n3"
    calibration = study.read(out / "calibration.json")
    assert calibration["setup_source_frames_ui"] == list(range(65, 165))
    assert calibration["threshold"] == 3.
    assert calibration["application_source_start_ui"] == 165
    assert calibration["application_source_stop_ui"] == 166
    study.evaluate(tmp_path)
    rows = study.read(out / "curves.json")["curve_rows"]
    assert len(rows) == 20
    assert len({r["threshold_id"] for r in rows}) == 10
    q1 = [r for r in rows if r["threshold_id"] == "q1"]
    assert {r["match_radius_px"] for r in q1} == {2., 6.}
    assert all(r["threshold_z"] == 3. and r["proposal_count"] == 2 for r in q1)
    if null_truth:
        assert all(r["event_count"] == 0 and r["active_region_frame_count"] == 0 and
                   r["false_positive_count"] == 2 and r["precision"] == 0. and
                   r["framewise_sensitivity"] is None and r["event_window_coverage"] is None for r in q1)
    else:
        assert all(r["event_count"] == 1 and r["true_positive_count"] == 1 and
                   r["false_positive_count"] == 1 and r["recovered_event_count"] == 1 for r in q1)
    assert all(r["proposal_count"] == 0 and r["precision"] is None for r in rows if r["threshold_id"] == "no_output")
    study.evaluate(tmp_path)  # unchanged numerical resume is allowed
    curves = study.read(out / "curves.json")
    curves["curve_rows"][0]["proposal_count"] += 1
    study.write_json(out / "curves.json", curves)
    with pytest.raises((RuntimeError, ValueError)):
        study.evaluate(tmp_path)


def test_completed_root_is_not_reopened(tmp_path):
    (tmp_path / "completion_manifest.json").write_text("{}")
    with pytest.raises(RuntimeError, match="Preserve completed"):
        study.guard_open(tmp_path)
