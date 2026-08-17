import numpy as np
import json

from neurobench.algorithms.local_covariance_whitening import contiguous_quiet_partitions
from neurobench.experiments.msln_msica.center_whiten_ica_screen import (
    _rank_rows,
    representation_metrics,
    whiten_bank,
)
from neurobench.experiments.msln_msica.center_whiten_ica_benchmark import (
    ArchitectureLane,
    ICAChoice,
    SCREEN_AUDIT_OPTOUT,
)
import neurobench.experiments.msln_msica.center_whiten_ica_screen as screen
from neurobench.algorithms.local_covariance_whitening import WhiteningFeatureBank


def _bank() -> WhiteningFeatureBank:
    rng = np.random.default_rng(17)
    values = rng.normal(size=(30, 8, 9, 3)).astype(np.float32)
    values[20:25, 3:5, 4:6] *= 4
    return WhiteningFeatureBank(
        ("a", "b", "c"), values, np.ones(30, dtype=bool),
        tuple({"scientific_array": "centered_residual_numerator"} for _ in range(3)),
        {"feature_order_frozen": True},
    )


def test_global_whitening_and_label_free_metrics() -> None:
    bank = _bank()
    quiet = np.zeros(30, dtype=bool); quiet[:12] = True
    event = np.zeros(30, dtype=bool); event[20:25] = True
    partitions = contiguous_quiet_partitions(quiet)
    config = {"compute": {"frame_chunk": 2, "maximum_peak_vram_gb": 4}}
    transformed, unresolved, _ = whiten_bank(
        bank, partitions, "global_full_oas_quiet", config,
    )
    metrics = representation_metrics(
        transformed, bank.valid_frames, partitions["holdout"], event,
        quiet_calibration=partitions["calibration"], runtime_seconds=0.1,
        unresolved_fraction=unresolved,
    )
    assert metrics["heldout_covariance_identity_error"] >= 0
    assert metrics["event_quiet_log_contrast"] > 0
    assert unresolved == 0


def test_rank_aggregation_prefers_stronger_and_cheaper_lane() -> None:
    base = {
        "heldout_covariance_identity_error": 1.0,
        "event_quiet_log_contrast": 1.0,
        "event_above_quiet_q99_fraction": 0.5,
        "quiet_q99_calibration_error": 0.01,
        "event_spatial_cohesion": 0.8,
        "event_temporal_recurrence": 0.5,
        "unresolved_fraction": 0.0,
        "runtime_seconds": 2.0,
    }
    weak = {**base, "lane_id": "weak", "event_quiet_log_contrast": 0.1, "runtime_seconds": 4.0}
    strong = {**base, "lane_id": "strong", "heldout_covariance_identity_error": 0.1}
    assert _rank_rows([weak, strong], 1)[0]["lane_id"] == "strong"


def test_screen_is_resumable_and_freezes_before_labels(tmp_path, monkeypatch) -> None:
    movie_path = tmp_path / "movie.npy"
    np.save(movie_path, np.zeros((40, 8, 9), dtype=np.uint16))
    output = tmp_path / "output"; output.mkdir()
    config = {
        "source": {
            "movie_path": str(movie_path), "review_interval_ui": [1, 30],
            "quiet_interval_ui": [1, 12], "burst_intervals_ui": {"1": [21, 25]},
        },
        "screen": {
            "maximum_stage_a": 1, "maximum_stage_b": 1,
            "maximum_finalists": 1, "maximum_ica_samples": 1000,
            "sample_seed": 7,
        },
        "compute": {"frame_chunk": 2, "maximum_peak_vram_gb": 4},
        "outputs": {"root_dir": str(output)},
    }
    (output / "preflight.json").write_text(json.dumps({
        "labels_loaded": False, "audit_opt_out_reason": SCREEN_AUDIT_OPTOUT,
    }))
    monkeypatch.setattr(screen, "load_screen_config", lambda _: config)
    monkeypatch.setattr(screen, "stage_a_lanes", lambda: (("joint_residual", "reference_diverse"),))
    monkeypatch.setattr(screen, "WHITENING_MODES", ("none",))
    monkeypatch.setattr(screen, "stage_b_lanes", lambda retained: ((retained[0][0], retained[0][1], "none"),))
    monkeypatch.setattr(screen, "stage_c_lanes", lambda retained: (
        ArchitectureLane(retained[0][0], retained[0][1], retained[0][2], ICAChoice("identity", 7, 0, 0.0)),
    ))
    monkeypatch.setattr(screen, "build_center_bank", lambda *args: (
        _bank(), np.r_[np.ones(12, dtype=bool), np.zeros(18, dtype=bool)],
        np.r_[np.zeros(20, dtype=bool), np.ones(5, dtype=bool), np.zeros(5, dtype=bool)],
    ))
    first = screen.run_metrics_screen(tmp_path / "ignored.json")
    second = screen.run_metrics_screen(tmp_path / "ignored.json")
    assert first == second
    assert first["labels_loaded"] is False
    assert len(first["finalists"]) == 1
    assert json.loads((output / "stage_c" / "checkpoint.json").read_text())["complete"] is True
