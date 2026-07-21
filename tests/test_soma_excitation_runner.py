from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from neurobench.dynamics.models import TemporalCNNResidual
from neurobench.experiments.soma_excitation.config import (
    CFARConfig,
    DarkZoneConfig,
    DynamicsCheckpoint,
    ResourceLimits,
    SomaExcitationConfig,
)
import neurobench.experiments.soma_excitation.detector as detector_module
import neurobench.experiments.soma_excitation.runner as runner_module
from neurobench.experiments.soma_excitation.preflight import MIB, ResourceBudgetError
from neurobench.experiments.soma_excitation.runner import run_soma_excitation_experiment


def _video() -> np.ndarray:
    frames, height, width = 18, 32, 36
    yy, xx = np.ogrid[:height, :width]
    baseline = 800 + 3 * yy + 2 * xx
    video = np.repeat(baseline[None], frames, axis=0).astype(np.uint16)
    distance = (yy - 16) ** 2 + (xx - 18) ** 2
    video[:, distance <= 3**2] = 150
    video[12:16, (distance > 4**2) & (distance <= 8**2)] = 2400
    return video


def _checkpoint(path: Path) -> Path:
    model = TemporalCNNResidual(
        input_channels=1,
        window_frames=8,
        hidden_channels=1,
        num_blocks=1,
        residual_scale=0.1,
    )
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)
    torch.save(
        {
            "model_state": model.state_dict(),
            "architecture": "temporal_cnn_pixel",
            "input_channels": 1,
            "window_frames": 8,
            "hidden_channels": 1,
            "num_layers": 1,
            "residual_scale": 0.1,
            "prediction_horizon_frames": 2,
            "model_id": "tiny_runner_tcnn",
            "dataset_id": "synthetic_training_domain",
            "normalization_mode": "robust_percentile",
            "grid_size": 128,
            "grid_pooling": "max_intensity",
        },
        path,
    )
    return path


def _config(
    source: Path,
    output: Path,
    checkpoint: Path | None,
) -> SomaExcitationConfig:
    checkpoints = ()
    if checkpoint is not None:
        checkpoints = (
            DynamicsCheckpoint(
                path=str(checkpoint),
                model_id="tiny_runner_tcnn",
                horizon_frames=2,
            ),
        )
    return SomaExcitationConfig(
        source_video=str(source),
        output_dir=str(output),
        experiment_id="synthetic_soma_runner",
        onset_frame_ui=13,
        control_preroll_frames=8,
        end_frame_ui=16,
        frame_rate_hz=50.0,
        resources=ResourceLimits(
            device="cpu",
            worker_count=1,
            chunk_frames=3,
            cpu_threads=1,
            max_ram_mib=1024 if checkpoint is not None else 128,
            max_output_mib=16,
        ),
        cfar=CFARConfig(small_radius_px=1, large_radius_px=6, pfa=0.35),
        dark_zones=DarkZoneConfig(
            inner_sigma=1.0,
            outer_sigma=3.0,
            min_contrast_z=2.0,
            min_distance_px=5.0,
            border_px=7,
            max_zones=4,
            core_radius_px=3.0,
            ring_inner_radius_px=4.0,
            ring_outer_radius_px=8.0,
        ),
        dynamics_checkpoints=checkpoints,
    )


def _directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def test_runner_artifact_contract_two_arms_and_output_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runner_module,
        "_read_process_memory_status",
        lambda: {
            "current_rss_bytes": 240 * MIB,
            "peak_rss_bytes": 320 * MIB,
            "source": "/proc/self/status",
            "peak_guard_available": True,
            "warning": None,
        },
    )
    source = tmp_path / "tiny_spon.npy"
    np.save(source, _video())
    checkpoint = _checkpoint(tmp_path / "tiny_checkpoint.pt")
    output = tmp_path / "experiment"
    config = _config(source, output, checkpoint)

    summary = run_soma_excitation_experiment(config)

    expected = {
        "resolved_config.json",
        "preflight.json",
        "run_state.json",
        "detector_arrays.npz",
        "detector_summary.json",
        "dark_zones.json",
        "review_frames.npz",
        "transfer_results.json",
        "experiment_summary.json",
        "report.md",
    }
    assert {item.name for item in output.iterdir()} == expected
    assert summary["status"] == "completed"
    assert summary["interpretation"]["transfer"].endswith("out-of-distribution.")
    assert "No manual ground truth" in summary["interpretation"]["ground_truth"]
    assert summary["started_at"] <= summary["completed_at"]
    assert {"resolved_config", "preflight", "transfer_results", "experiment_summary"} <= set(
        summary["artifacts"]
    )
    assert json.loads((output / "resolved_config.json").read_text()) == config.to_dict()

    run_state = json.loads((output / "run_state.json").read_text())
    measured_bytes = _directory_size(output)
    assert run_state["status"] == "completed"
    assert run_state["started_at"] == summary["started_at"]
    assert run_state["completed_at"] == summary["completed_at"]
    assert run_state["output_bytes"] == measured_bytes
    assert summary["resources"]["actual_output_bytes"] == measured_bytes
    assert summary["resources"]["observed_current_rss_bytes"] == 240 * MIB
    assert summary["resources"]["observed_peak_rss_bytes"] == 320 * MIB
    assert summary["resources"]["memory_guard_enforced"] is True
    assert summary["resources"]["memory_guard_status"] == "pass"
    assert summary["detector"]["activated_zone_count"] == summary["detector"][
        "cfar_activated_zone_count"
    ]
    assert "positive_residual_signal_global" in summary["detector"]
    assert "positive_residual_signal_ring" in summary["detector"]
    assert [
        item["stage"] for item in summary["resources"]["memory_observations"]
    ] == ["start", "after_detector", "after_transfer", "before_completion"]
    assert run_state["resources"]["observed_current_rss_bytes"] == 240 * MIB
    assert run_state["resources"]["observed_peak_rss_bytes"] == 320 * MIB
    assert run_state["resources"]["memory_guard_status"] == "pass"
    assert measured_bytes <= 16 * 1024 * 1024

    transfer = json.loads((output / "transfer_results.json").read_text())
    assert transfer["status"] == "completed"
    assert transfer["interpretation"]["status"] == "exploratory_out_of_distribution"
    assert transfer["normalization"]["baseline_first_index"] == 4
    assert transfer["normalization"]["baseline_last_index"] == 11
    assert transfer["normalization"]["event_frames_used_for_fit"] is False
    assert transfer["target_range"] == {
        "first_source_index": 12,
        "last_source_index": 15,
        "target_count": 4,
        "control_frames_scored": False,
    }
    assert set(transfer["spatial_arms"]) == {
        "adaptive_full_fov_128",
        "fixed_native_pool4",
    }
    expected_shapes = {
        "adaptive_full_fov_128": [128, 128],
        "fixed_native_pool4": [8, 9],
    }
    for name, arm in transfer["spatial_arms"].items():
        assert len(arm["models"]) == 1
        model_result = arm["models"][0]
        assert model_result["spatial_shape"] == expected_shapes[name]
        assert model_result["control_frames_scored"] is False
        assert model_result["evaluation"]["first_target_index"] == 12
        assert model_result["evaluation"]["last_target_index"] == 15
        assert model_result["evaluation"]["inference_batch_size"] == 1
        assert set(model_result["metrics"]["masked"]) == {"core", "ring"}

    with np.load(output / "detector_arrays.npz") as detector_arrays:
        assert detector_arrays["is_score_frame"].tolist() == [False] * 8 + [True] * 4
    with np.load(output / "review_frames.npz") as review:
        assert review["raw_frames"].shape[0] <= 12
        assert review["raw_frames"].shape[0] == review["source_indices"].shape[0]
        assert np.all(review["source_indices"] >= 12)
        assert np.all(review["source_indices"] < 16)
    assert not list(output.glob("*.tmp"))
    report = (output / "report.md").read_text()
    assert report.index("## Interpretation") < report.index("## Detector result")
    assert "out-of-distribution" in report
    assert "No manual ground truth" in report
    assert (
        "| Direct positive-residual lane | Control mean | Event mean | Difference | Ratio |"
        in report
    )
    assert "| Local residual-CFAR lane | Control fraction | Event fraction | Difference | Ratio |" in report
    assert "Peak RSS: 320.0 MiB / 1024.0 MiB; guard status: pass." in report
    assert "Global delta | High-change delta | Core delta | Ring delta | Positive-change corr." in report
    assert "adaptive_full_fov_128" in report
    assert "fixed_native_pool4" in report


def test_proc_status_reader_parses_rss_and_reports_unavailable(tmp_path: Path) -> None:
    status = tmp_path / "status"
    status.write_text("Name:\tpython\nVmHWM:\t456 kB\nVmRSS:\t123 kB\n")
    parsed = runner_module._read_process_memory_status(status)

    assert parsed == {
        "current_rss_bytes": 123 * 1024,
        "peak_rss_bytes": 456 * 1024,
        "source": str(status),
        "peak_guard_available": True,
        "warning": None,
    }

    unavailable = runner_module._read_process_memory_status(tmp_path / "missing")
    assert unavailable["current_rss_bytes"] is None
    assert unavailable["peak_rss_bytes"] is None
    assert unavailable["peak_guard_available"] is False
    assert "guard unavailable" in unavailable["warning"]


def test_runner_fails_and_records_state_when_peak_rss_exceeds_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runner_module,
        "_read_process_memory_status",
        lambda: {
            "current_rss_bytes": 120 * MIB,
            "peak_rss_bytes": 129 * MIB,
            "source": "/proc/self/status",
            "peak_guard_available": True,
            "warning": None,
        },
    )
    source = tmp_path / "tiny_spon.npy"
    np.save(source, _video())
    output = tmp_path / "over_cap"

    with pytest.raises(ResourceBudgetError, match="Process peak RSS"):
        run_soma_excitation_experiment(_config(source, output, checkpoint=None))

    state = json.loads((output / "run_state.json").read_text())
    resources = state["resources"]
    assert state["status"] == "failed"
    assert state["error_type"] == "ResourceBudgetError"
    assert resources["observed_current_rss_bytes"] == 120 * MIB
    assert resources["observed_peak_rss_bytes"] == 129 * MIB
    assert resources["memory_guard_enforced"] is True
    assert resources["memory_guard_status"] == "failed"
    assert resources["memory_observations"][-1]["stage"] == "after_detector"


def test_runner_refuses_output_collision_without_mutation(tmp_path: Path) -> None:
    source = tmp_path / "tiny_spon.npy"
    np.save(source, _video())
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("user data\n")

    with pytest.raises(FileExistsError, match="Refusing output collision"):
        run_soma_excitation_experiment(_config(source, output, checkpoint=None))

    assert sentinel.read_text() == "user data\n"
    assert {item.name for item in output.iterdir()} == {"keep.txt"}


def test_runner_atomically_records_failure_after_ready_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "tiny_spon.npy"
    np.save(source, _video())
    output = tmp_path / "failed"

    def fail_detector(*args, **kwargs):
        raise RuntimeError("synthetic detector failure")

    monkeypatch.setattr(detector_module, "run_streamed_detector", fail_detector)
    with pytest.raises(RuntimeError, match="synthetic detector failure"):
        run_soma_excitation_experiment(_config(source, output, checkpoint=None))

    state = json.loads((output / "run_state.json").read_text())
    assert state["status"] == "failed"
    assert state["error_type"] == "RuntimeError"
    assert state["error"] == "synthetic detector failure"
    assert state["started_at"] <= state["failed_at"]
    assert (output / "resolved_config.json").is_file()
    assert (output / "preflight.json").is_file()
    assert not (output / "detector_summary.json").exists()
    assert not list(output.glob("*.tmp"))
