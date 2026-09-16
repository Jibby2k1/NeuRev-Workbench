"""Source-contract checks and tiny complete generic-score scientific audits."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil

import pytest

from neurobench.experiments.gamma_ls_difference import control_audit as module


def _fixture(tmp_path: Path, *, candidates: bool = True) -> dict:
    import numpy as np
    frames = list(range(1800, 1812))
    raw = np.full((12, 40, 48), 4.125000000123, dtype=np.float64)
    score = np.zeros(raw.shape, dtype=np.float32)
    score[3:8, 20, 20] = [0, 2, 3, 2, 0]
    score[7, 20, 21] = 2
    paths = {}
    for name in ("Raw", "Input", "Residual", "Score"):
        path = tmp_path / f"{name}.npy"
        np.save(path, raw if name in ("Raw", "Input") else score)
        paths[name] = path
    candidate_rows = [{"proposal_id": "p1", "x_px": 20, "y_px": 20,
                       "source_frame_ui": 1805, "score": 3, "threshold": 1,
                       "target_proposals_per_frame": 1},
                      {"proposal_id": "p2", "x_px": 21, "y_px": 20,
                       "source_frame_ui": 1807, "score": 2, "threshold": 1,
                       "target_proposals_per_frame": 1}] if candidates else []
    path = tmp_path / "candidates.tsv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, delimiter="\t", fieldnames=["proposal_id", "x_px", "y_px", "source_frame_ui", "score", "threshold", "target_proposals_per_frame"])
        writer.writeheader()
        writer.writerows(candidate_rows)
    labels = [{"canonical_roi_id": "roi_001", "observation_id": "b1_roi1",
               "burst_id": 1, "source_start_ui": 1803, "source_stop_ui": 1808,
               "x_px": 20.25, "y_px": 20.25,
               "temporal_extent_semantics": "configured_burst_window_not_per_roi_onset"},
              {"canonical_roi_id": "roi_001", "observation_id": "b2_roi1",
               "burst_id": 2, "source_start_ui": 1809, "source_stop_ui": 1811,
               "x_px": 21.25, "y_px": 20.25,
               "temporal_extent_semantics": "configured_burst_window_not_per_roi_onset"}]
    numeric = {}
    for q in (0.25, 0.5, 1, 2, 5):
        numeric_path = tmp_path / f"q{q}.json"
        numeric_path.write_text(json.dumps({"target_proposals_per_frame": q}))
        numeric[str(q)] = numeric_path
    return {"root": tmp_path / "audit", "stage_paths": paths,
            "display_limits": {name: ([-5, 5] if name in ("Residual", "Score") else [0, 10]) for name in paths},
            "signed_stage_keys": ["Residual", "Score"], "source_frames_ui": frames,
            "candidates_path": path, "expert_occurrences": labels,
            "operating_point": {"target_proposals_per_frame": 1, "threshold": 1,
                                "application_source_start_ui": 1803,
                                "application_source_stop_ui": 1811,
                                "readout_name": "signed residual"},
            "source_binding": {"stage_sha256": {name: module._digest(path) for name, path in paths.items()},
                               "candidates_sha256": module._digest(path),
                               "expert_occurrences_sha256": module._canonical_digest(labels)},
            "fullnumeric_q_paths": numeric, "context_frames": 1}


def test_threshold_is_required_and_candidate_aliases_must_agree():
    assert module._frozen_threshold([{"threshold_z": 2}], {"threshold": 2}) == 2
    assert module._frozen_threshold([], {"threshold_z": -1}) == -1
    for value in (None, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite frozen threshold"):
            module._frozen_threshold([], {"threshold": value})
    with pytest.raises(ValueError, match="Candidate threshold differs"):
        module._frozen_threshold([{"threshold": 2, "threshold_z": 3}], {"threshold": 2})


def test_burst_windows_cannot_silently_replace_one_another():
    a = {"burst_id": 1, "source_start_ui": 10, "source_stop_ui": 20}
    assert module._burst_intervals([a, dict(a)]) == {1: (10, 20)}
    with pytest.raises(ValueError, match="share the declared"):
        module._burst_intervals([a, {**a, "source_stop_ui": 21}])


@pytest.mark.parametrize("damage,expected", [
    ("missing_stage", "Raw/Input/Score"), ("wrong_order", "pipeline order"),
    ("missing_stage_seal", "seal every input stage"),
    ("missing_candidate_seal", "seal the candidate file"),
    ("stage_bytes_changed", "sealed SHA-256"),
    ("candidate_seal_changed", "candidates differs"),
    ("asymmetric_signed_scale", "symmetric"),
    ("non_q1", "operating point must be q1"),
    ("missing_numeric_q1", "including q1"),
])
def test_contract_rejects_invalid_inputs_before_media(tmp_path, monkeypatch, damage, expected):
    kwargs = _fixture(tmp_path)
    if damage == "missing_stage": kwargs["stage_paths"].pop("Input")
    elif damage == "wrong_order": kwargs["stage_paths"] = dict(reversed(list(kwargs["stage_paths"].items())))
    elif damage == "missing_stage_seal": kwargs["source_binding"]["stage_sha256"].pop("Score")
    elif damage == "missing_candidate_seal": kwargs["source_binding"].pop("candidates_sha256")
    elif damage == "stage_bytes_changed": kwargs["stage_paths"]["Score"].write_bytes(b"changed")
    elif damage == "candidate_seal_changed": kwargs["source_binding"]["candidates_sha256"] = "0" * 64
    elif damage == "asymmetric_signed_scale": kwargs["display_limits"]["Score"] = [-1, 5]
    elif damage == "non_q1": kwargs["operating_point"]["target_proposals_per_frame"] = 2
    elif damage == "missing_numeric_q1": kwargs["fullnumeric_q_paths"].pop("1")
    monkeypatch.setattr(module, "_write_video", lambda *a, **k: pytest.fail("invalid source reached renderer"))
    with pytest.raises(ValueError, match=expected):
        module.run_control_audit(**kwargs)
    status = kwargs["root"] / "status.json"
    if status.exists(): assert json.loads(status.read_text())["scientific_audit_complete"] is False


def test_score_binding_is_exact_and_respects_application_interval():
    import numpy as np
    scores = np.zeros((3, 2, 2), dtype=np.float32)
    scores[1, 1, 1] = 2
    candidate = {"source_frame_ui": 11, "x_px": 1, "y_px": 1, "score": 2}
    index = {10: 0, 11: 1, 12: 2}
    module._verify_candidate_score_binding([candidate], scores, index, 1, {})
    with pytest.raises(ValueError, match="differs"):
        module._verify_candidate_score_binding([{**candidate, "score": 2.0000001}], scores, index, 1, {})
    with pytest.raises(ValueError, match="does not exceed"):
        module._verify_candidate_score_binding([candidate], scores, index, 2, {})
    with pytest.raises(ValueError, match="outside the declared application"):
        module._verify_candidate_score_binding([candidate], scores, index, 1, {"application_source_start_ui": 12})


def test_thread_environment_is_restored_on_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "7")
    def fail(*a, **k):
        assert module.os.environ["OMP_NUM_THREADS"] == "1"
        raise RuntimeError("fixture failure")
    monkeypatch.setattr(module, "_run_control_audit", fail)
    with pytest.raises(RuntimeError, match="fixture failure"):
        module.run_control_audit(**_fixture(tmp_path))
    assert module.os.environ["OMP_NUM_THREADS"] == "7"


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg is required")
@pytest.mark.parametrize("has_candidates", [True, False])
def test_tiny_generic_audit_preserves_labels_exact_traces_geometry_and_resume(tmp_path, has_candidates):
    kwargs = _fixture(tmp_path, candidates=has_candidates)
    output = kwargs["root"]
    result = module.run_control_audit(**kwargs)
    assert result["scientific_audit_complete"]
    assert result["expert_roi_count"] == 1
    assert result["expert_occurrence_count"] == 2
    assert result["model_roi_count"] == int(has_candidates)
    assert result["video_count"] == 3 + int(has_candidates)
    assert result["comparison_trace_count"] == 2
    contract = json.loads((output / "run_contract.json").read_text())
    assert contract["stage_sequence"] == ["Raw", "Input", "Residual", "Score"]
    assert contract["cpu_thread_limit"] == 1
    assert contract["media_helper_sha256"] == module._digest(Path(module._base.__file__))
    assert len(contract["inventory_validator_sha256"]) == 64
    context = json.loads((output / "llm_context.json").read_text())
    assert context["comparison_spatial_panels"] == ["Raw matched comparison", "Score matched comparison"]
    assert context["operating_point"]["readout_name"] == "signed residual"
    assert context["coverage"]["q_coverage"]["1"]["full_media_for_this_q"]
    assert not context["coverage"]["q_coverage"]["0.5"]["full_media_for_this_q"]
    trace_path = output / "1_Expert_Annotations/exact_pixel_traces/roi_001.csv"
    traces = list(csv.DictReader(trace_path.open()))
    assert len(traces) == 12
    assert list(traces[0]) == ["source_frame_ui", "Raw", "Input", "Residual", "Score"]
    assert float(traces[0]["Raw"]) == 4.125000000123
    assert int(traces[0]["source_frame_ui"]) == 1800
    assert int(traces[-1]["source_frame_ui"]) == 1811
    expert_metadata = json.loads((output / "1_Expert_Annotations/metadata/roi_001.json").read_text())
    assert len(expert_metadata["coordinate_variants"]) == 2
    metrics = json.loads((output / "3_Comparison/trace_comparisons/b1_roi1.json").read_text())
    assert metrics["comparison_plot_source_frames_ui"] == list(range(1802, 1810))
    assert metrics["correlation_stage"] == "Score"
    assert "event_score_correlation" in metrics and "event_gamma_correlation" not in metrics
    assert metrics["correlation_and_lag_window_ui_inclusive"] == [1803, 1808]
    assert metrics["nearest_assignment_computed_separately"]
    videos = json.loads((output / "video_manifest.json").read_text())["videos"]
    assert all(v["all_frames_rgb_byte_exact"] and v["all_frames_marker_expectations_pass"] for v in videos)
    assert all(v["probe"]["pix_fmt"] == "gbrp" for v in videos)
    full = [v for v in videos if "full_field" in v["path"]]
    assert all(v["source_frames_ui"] == [1800, 1805, 1810, 1811] for v in full)
    assert all(v["panel_image_geometry"]["source_size_wh"] == [48, 40] for v in full)
    assert not list((output / "3_Comparison").rglob("*.mp4"))
    valid = json.loads((output / "validation.json").read_text())
    assert valid["candidate_scores_equal_saved_score_array"]
    assert valid["all_frames_rgb_byte_exact"]
    original = (output / "status.json").stat().st_mtime_ns
    assert module.run_control_audit(**kwargs) == result
    assert (output / "status.json").stat().st_mtime_ns == original
    changed = {**kwargs, "display_limits": {**kwargs["display_limits"], "Score": [-10, 10]}}
    with pytest.raises(ValueError, match="changed source/configuration"):
        module.run_control_audit(**changed)
    kwargs["fullnumeric_q_paths"]["2"].write_text('{"changed":true}')
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        module.run_control_audit(**kwargs)
