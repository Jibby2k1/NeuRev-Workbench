from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import tifffile

from neurobench.experiments.neuron_identifiability.jepa_data import (
    RecordingDescriptor,
    RecordingInventory,
)
from neurobench.experiments.neuron_identifiability.jepa_motion_audit import (
    MotionAuditConfig,
    MotionAuditThresholds,
    assert_json_ready,
    estimate_phase_translation,
    run_jepa_motion_audit,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _inventory(tmp_path: Path, video: np.ndarray, *, recording_number: int = 1) -> RecordingInventory:
    path = tmp_path / "Inputs" / "060126" / f"{recording_number} resting.tif"
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, video, photometric="minisblack", contiguous=True)
    descriptor = RecordingDescriptor(
        recording_id=f"060126_{recording_number:02d}_rest",
        dataset_id="060126",
        recording_number=recording_number,
        behavior="rest",
        uri=f"data://Inputs/060126/{path.name}",
        resolved_path=path,
        sha256=_digest(f"recording-{recording_number}"),
        shape=tuple(int(value) for value in video.shape),
        dtype=video.dtype.name,
    )
    return RecordingInventory(
        dataset_id="060126",
        recordings=(descriptor,),
        descriptor_uri="repo://synthetic_motion_inventory.json",
        descriptor_sha256=_digest("synthetic-motion-inventory"),
    )


def _config(**values: object) -> MotionAuditConfig:
    defaults: dict[str, object] = {
        "frames_per_recording": 6,
        "intensity_spatial_stride": 2,
        "registration_spatial_stride": 1,
        "tile_grid_yx": (2, 2),
        "minimum_registration_extent": 12,
        "require_frozen_060126_inventory": False,
    }
    defaults.update(values)
    return MotionAuditConfig(**defaults)


def _textured_frame(seed: int = 17, shape: tuple[int, int] = (64, 80)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y, x = np.indices(shape)
    texture = 1200.0 + 120.0 * np.sin(x / 4.3) + 90.0 * np.cos(y / 5.1)
    texture += rng.normal(0.0, 35.0, size=shape)
    return np.clip(np.rint(texture), 1, 65_534).astype(np.uint16)


def test_phase_translation_recovers_registration_shift_in_native_pixels() -> None:
    reference = _textured_frame(shape=(96, 112)).astype(np.float32)
    moving = np.roll(np.roll(reference, 3, axis=0), -2, axis=1)

    result = estimate_phase_translation(
        reference,
        moving,
        spatial_stride=1,
        minimum_extent=16,
    )

    assert result["valid"] is True
    # Direction is the correction applied to moving, hence the inverse roll.
    assert result["registration_shift_yx_px"] == pytest.approx([-3.0, 2.0], abs=0.35)
    assert result["magnitude_px"] == pytest.approx(np.hypot(3.0, 2.0), abs=0.35)


def test_phase_translation_rejects_dominant_shift_outside_adjacent_frame_search() -> None:
    reference = _textured_frame(shape=(96, 112)).astype(np.float32)
    moving = np.roll(reference, 18, axis=1)

    result = estimate_phase_translation(
        reference,
        moving,
        spatial_stride=1,
        minimum_extent=16,
        maximum_translation_search_px=8.0,
    )

    assert result["valid"] is False
    assert result["reason"] == "dominant_peak_outside_plausible_adjacent_frame_search"
    assert result["unconstrained_peak_shift_yx_px"] == pytest.approx([0.0, -18.0], abs=0.35)


def test_audit_is_deterministic_portable_and_clear_for_stationary_texture(tmp_path: Path) -> None:
    frame = _textured_frame()
    video = np.repeat(frame[None, :, :], 20, axis=0)
    inventory = _inventory(tmp_path, video)
    config = _config()

    first = run_jepa_motion_audit(inventory, config=config)
    second = run_jepa_motion_audit(inventory, config=config)

    assert first["result_sha256"] == second["result_sha256"]
    assert first["manifest"]["manifest_sha256"] == second["manifest"]["manifest_sha256"]
    assert first["aggregate_gate"]["decision"] == "screen_clear"
    assert first["recordings"][0]["sampling"]["intensity_frame_indices_zero"] == [0, 3, 7, 11, 15, 19]
    assert first["recordings"][0]["sampling"]["adjacent_pair_start_indices_zero"] == [0, 3, 7, 10, 14, 18]
    assert first["recordings"][0]["units"]["temporal"] == "frames"
    assert first["recordings"][0]["units"]["spatial"] == "native_pixels"
    assert first["recordings"][0]["units"]["seconds_available"] is False
    assert first["recordings"][0]["units"]["micrometers_available"] is False
    assert_json_ready(first)
    encoded = json.dumps(first, allow_nan=False)
    assert str(tmp_path) not in encoded
    assert "resolved_path" not in encoded


def test_audit_flags_drift_saturation_and_translation_without_causal_label(tmp_path: Path) -> None:
    base = _textured_frame().astype(np.float32)
    frames = []
    for frame_index in range(20):
        moved = np.roll(base, frame_index, axis=1) - 8.0 * frame_index
        frame = np.clip(np.rint(moved), 0, 65_534).astype(np.uint16)
        frame[:8, :8] = np.iinfo(np.uint16).max
        frames.append(frame)
    inventory = _inventory(tmp_path, np.stack(frames, axis=0))
    config = _config(
        thresholds=MotionAuditThresholds(
            absolute_relative_intensity_drift=0.05,
            high_sensor_rail_fraction=0.005,
            normalized_frame_difference_p95=10.0,
            global_translation_p95_px=0.5,
            tile_median_disagreement_p95_px=10.0,
            minimum_valid_global_pair_fraction=0.5,
            minimum_median_valid_tile_fraction=0.25,
        )
    )

    result = run_jepa_motion_audit(inventory, config=config)
    recording = result["recordings"][0]

    assert result["aggregate_gate"]["decision"] == "review_required"
    assert recording["intensity"]["robust_linear_drift"]["fitted_relative_change_over_sampled_span"] < -0.05
    assert recording["intensity"]["sensor_rail_occupancy"]["high_fraction"] > 0.005
    assert recording["translation"]["global_magnitude_px"]["p95"] > 0.5
    assert {
        "absolute_relative_intensity_drift",
        "high_sensor_rail_fraction",
        "global_translation_p95_px",
    }.issubset(recording["gate"]["triggered_reasons"])
    assert "causal" in recording["gate"]["interpretation"].casefold()
    assert any("No motion correction" in item for item in result["limitations"])
    assert_json_ready(result)


def test_low_texture_registration_is_reviewed_instead_of_reported_as_zero_motion(tmp_path: Path) -> None:
    video = np.full((20, 64, 80), 1000, dtype=np.uint16)
    inventory = _inventory(tmp_path, video)

    result = run_jepa_motion_audit(inventory, config=_config())
    recording = result["recordings"][0]

    assert recording["translation"]["valid_global_pair_fraction"] == 0.0
    assert recording["translation"]["global_magnitude_px"]["p95"] is None
    assert recording["gate"]["checks"]["global_translation_p95_px"]["triggered"] is True
    assert recording["gate"]["checks"]["valid_global_pair_fraction"]["triggered"] is True
    assert recording["gate"]["decision"] == "review_required"
    assert result["aggregate_gate"]["worst_observed_by_check"]["valid_global_pair_fraction"] == 0.0
    assert_json_ready(result)


def test_default_inventory_contract_fails_closed_for_noncanonical_subset(tmp_path: Path) -> None:
    video = np.repeat(_textured_frame()[None, :, :], 8, axis=0)
    inventory = _inventory(tmp_path, video)

    with pytest.raises(ValueError, match="invalid frozen 060126 inventory"):
        run_jepa_motion_audit(inventory)
