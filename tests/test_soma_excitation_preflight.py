import json

import numpy as np
import pytest

from neurobench.experiments.soma_excitation import (
    ConfigValidationError,
    ResourceBudgetError,
    SomaExcitationConfig,
    build_soma_excitation_preflight,
)


def _write_config(tmp_path, *, shape=(2000, 8, 9), resources=None, checkpoints=None):
    video = tmp_path / "source.npy"
    np.save(video, np.zeros(shape, dtype=np.uint8))
    payload = {
        "schema_version": 1,
        "experiment_id": "synthetic_soma_v1",
        "source_video": video.name,
        "output_dir": "result",
        "resources": resources
        or {
            "device": "cpu",
            "worker_count": 1,
            "chunk_frames": 32,
            "cpu_threads": 2,
            "max_ram_mib": 1024 if checkpoints else 64,
            "max_output_mib": 64,
        },
        "dynamics_checkpoints": checkpoints or [],
    }
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_json_config_preflight_maps_ui_frames_and_uses_video_metadata(tmp_path):
    checkpoint = tmp_path / "temporal_cnn.pt"
    checkpoint.write_bytes(b"synthetic checkpoint")
    config_path = _write_config(
        tmp_path,
        checkpoints=[{"path": checkpoint.name, "model_id": "temporal_cnn_leader", "horizon_frames": 5}],
    )

    config = SomaExcitationConfig.load_json(config_path)
    roundtrip_path = tmp_path / "roundtrip.json"
    config.write_json(roundtrip_path)
    assert SomaExcitationConfig.load_json(roundtrip_path).to_dict() == config.to_dict()
    payload = build_soma_excitation_preflight(config)

    assert config.onset_frame_ui == 1900
    assert config.onset_frame_zero == 1899
    assert config.control_start_frame_zero == 1799
    assert payload["source"]["metadata"]["storage_mode"] == "npy_memmap"
    assert payload["frame_bounds"] == {
        "ui_index_base": 1,
        "array_index_base": 0,
        "onset_frame_ui": 1900,
        "onset_frame_zero": 1899,
        "control_start_frame_ui": 1800,
        "control_start_frame_zero": 1799,
        "control_stop_frame_zero_exclusive": 1899,
        "control_frame_count": 100,
        "score_start_frame_ui": 1900,
        "score_start_frame_zero": 1899,
        "score_stop_frame_zero_exclusive": 2000,
        "score_last_frame_ui": 2000,
        "score_frame_count": 101,
        "analysis_start_frame_zero": 1799,
        "analysis_stop_frame_zero_exclusive": 2000,
        "analysis_frame_count": 201,
    }
    assert payload["resources"]["device"] == "cpu"
    assert payload["resources"]["worker_count"] == 1
    assert payload["processing_contract"]["control_frames_are_scored"] is False
    assert payload["dark_zones"]["zone_api_parameters"]["z_threshold"] == 3.0
    assert payload["dark_zones"]["zone_api_parameters"]["ring_outer_radius"] == 10.0
    assert config.dark_zones.to_zone_config().max_zones == 300
    assert payload["dynamics_checkpoints"][0]["model_id"] == "temporal_cnn_leader"
    assert payload["dynamics_checkpoints"][0]["horizon_frames"] == 5
    assert json.loads(json.dumps(payload))["ready"] is True


def test_preflight_accounts_for_scientific_model_runtime_overhead(tmp_path):
    checkpoint = tmp_path / "temporal_cnn.pt"
    checkpoint.write_bytes(b"synthetic checkpoint")
    config_path = _write_config(
        tmp_path,
        shape=(1, 1, 1),
        resources={
            "device": "cpu",
            "worker_count": 1,
            "chunk_frames": 8,
            "cpu_threads": 2,
            "max_ram_mib": 1024,
            "max_output_mib": 256,
        },
        checkpoints=[
            {
                "path": checkpoint.name,
                "model_id": "temporal_cnn_leader",
                "horizon_frames": 2,
            }
        ],
    )
    # _write_config creates its own source; restore the real-data geometry using
    # a sparse NPY so this remains a metadata-only, low-memory test.
    sparse_video = np.lib.format.open_memmap(
        tmp_path / "source.npy",
        mode="w+",
        dtype=np.uint16,
        shape=(2000, 340, 573),
    )
    del sparse_video

    payload = build_soma_excitation_preflight(config_path)
    resources = payload["resources"]

    assert resources["scientific_runtime_overhead_bytes"] == 640 * 1024 * 1024
    assert resources["runtime_guard_headroom_bytes"] == 32 * 1024 * 1024
    assert resources["resolved_chunk_frames"] == 8
    assert 760 * 1024 * 1024 < resources["estimated_peak_ram_bytes"] < 1024 * 1024 * 1024


def test_preflight_reduces_requested_chunk_to_fit_ram_cap(tmp_path):
    config_path = _write_config(
        tmp_path,
        shape=(2000, 64, 64),
        resources={
            "device": "cpu",
            "worker_count": 1,
            "chunk_frames": 128,
            "cpu_threads": 1,
            "max_ram_mib": 17,
            "max_output_mib": 128,
        },
    )

    payload = build_soma_excitation_preflight(config_path)

    resources = payload["resources"]
    assert 1 <= resources["resolved_chunk_frames"] < resources["requested_chunk_frames"]
    assert resources["scientific_runtime_overhead_bytes"] == 0
    assert resources["runtime_guard_headroom_bytes"] == 0
    assert resources["estimated_peak_ram_bytes"] <= 17 * 1024 * 1024
    assert "reduced chunk_frames" in payload["warnings"][0]


@pytest.mark.parametrize(
    ("resources", "message"),
    [
        (
            {
                "device": "cpu",
                "worker_count": 1,
                "chunk_frames": 32,
                "cpu_threads": 1,
                "max_ram_mib": 8,
                "max_output_mib": 128,
            },
            "RAM budget",
        ),
        (
            {
                "device": "cpu",
                "worker_count": 1,
                "chunk_frames": 32,
                "cpu_threads": 1,
                "max_ram_mib": 64,
                "max_output_mib": 1,
            },
            "Estimated output",
        ),
    ],
)
def test_preflight_rejects_resource_budget_failures(tmp_path, resources, message):
    config_path = _write_config(tmp_path, resources=resources)

    with pytest.raises(ResourceBudgetError, match=message):
        build_soma_excitation_preflight(config_path)


def test_preflight_validates_every_checkpoint_path(tmp_path):
    config_path = _write_config(tmp_path, checkpoints=["missing-checkpoint.pt"])

    with pytest.raises(FileNotFoundError, match="Dynamics checkpoint"):
        build_soma_excitation_preflight(config_path)


def test_checkpoint_horizon_is_positive_and_missing_horizon_warns(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    invalid = _write_config(tmp_path, checkpoints=[{"path": checkpoint.name, "horizon_frames": 0}])
    with pytest.raises(ConfigValidationError, match="horizon_frames must be positive"):
        SomaExcitationConfig.load_json(invalid)

    valid = _write_config(tmp_path, checkpoints=[checkpoint.name])
    payload = build_soma_excitation_preflight(valid)
    assert payload["dynamics_checkpoints"][0]["horizon_frames"] is None
    assert any("lacks explicit horizon_frames" in warning for warning in payload["warnings"])


def test_preflight_refuses_output_collision_without_explicit_allow_flag(tmp_path):
    config_path = _write_config(tmp_path)
    (tmp_path / "result").mkdir()

    with pytest.raises(FileExistsError, match="allow_existing_output=True"):
        build_soma_excitation_preflight(config_path)

    payload = build_soma_excitation_preflight(config_path, allow_existing_output=True)
    assert payload["allow_existing_output"] is True
    assert payload["checks"][-1]["detail"] == "existing path explicitly allowed"


@pytest.mark.parametrize(
    "section",
    [
        {"resources": {"device": "cuda"}},
        {"resources": {"worker_count": 2}},
        {"cfar": {"small_radius_px": 8, "large_radius_px": 4}},
        {"dark_zones": {"ring_inner_radius_px": 10, "ring_outer_radius_px": 5}},
    ],
)
def test_config_rejects_unsafe_execution_and_invalid_zone_geometry(tmp_path, section):
    payload = {"source_video": "source.npy", "output_dir": "result", **section}

    with pytest.raises(ConfigValidationError):
        SomaExcitationConfig.from_dict(payload, base_dir=tmp_path)
