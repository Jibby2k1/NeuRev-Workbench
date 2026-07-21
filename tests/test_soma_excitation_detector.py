from __future__ import annotations

import json

import numpy as np
import pytest

import neurobench.experiments.soma_excitation.detector as detector_module
from neurobench.experiments.soma_excitation.config import CFARConfig
from neurobench.experiments.soma_excitation.detector import run_streamed_detector
from neurobench.experiments.soma_excitation.zones import DarkSomaZoneConfig


def _synthetic_video() -> np.ndarray:
    frames, height, width = 20, 64, 64
    yy, xx = np.ogrid[:height, :width]
    baseline = (1000 + yy + 2 * xx).astype(np.uint16)
    video = np.repeat(baseline[None, :, :], frames, axis=0)
    distance = (yy - 32) ** 2 + (xx - 32) ** 2
    video[:, distance <= 3**2] = 200  # Stable dark anatomical core.
    video[8:10] = 65_000  # Deliberately ignored gap, not calibration or score.
    video[14:, (distance > 4**2) & (distance <= 9**2)] = 3500
    return video


def _cfar() -> CFARConfig:
    return CFARConfig(
        small_radius_px=1,
        large_radius_px=12,
        pfa=0.35,
        epsilon=1e-6,
    )


def _zones() -> DarkSomaZoneConfig:
    return DarkSomaZoneConfig(
        inner_sigma=1.0,
        outer_sigma=4.0,
        z_threshold=4.0,
        min_distance=6.0,
        border=13,
        max_zones=1,
        core_radius=3.0,
        ring_inner_radius=4.0,
        ring_outer_radius=9.0,
    )


def _run(path, chunk_frames: int, zone_threshold_method: str = "p99"):
    return run_streamed_detector(
        path,
        control_start=0,
        control_stop=8,
        score_start=10,
        score_stop=20,
        chunk_frames=chunk_frames,
        cfar=_cfar(),
        zone_config=_zones(),
        zone_threshold_method=zone_threshold_method,
    )


def test_late_annulus_onset_frame_mapping_and_lane_semantics(tmp_path) -> None:
    path = tmp_path / "late_burst.npy"
    np.save(path, _synthetic_video())

    result = _run(path, chunk_frames=3)

    assert result.frame_indices.tolist() == [*range(8), *range(10, 20)]
    assert result.ui_frames.tolist() == [value + 1 for value in result.frame_indices]
    assert result.is_score_frame.tolist() == [False] * 8 + [True] * 10
    assert result.summary["frame_ranges"]["score"] == {
        "source_start_index": 10,
        "source_stop_index_exclusive": 20,
        "ui_start_frame": 11,
        "ui_end_frame_inclusive": 20,
        "frame_count": 10,
    }
    assert result.summary["normalization"]["p99"] < 2000  # Gap frames excluded.
    assert (result.summary["dark_zones"]["zones"][0]["y"], result.summary["dark_zones"]["zones"][0]["x"]) == (32, 32)

    for lane in ("raw", "residual"):
        activation = result.summary["zone_activation"][lane]
        assert activation["activated_zone_count"] == 1
        assert activation["zones"][0]["onset_source_index"] == 14
        assert activation["zones"][0]["onset_ui_frame"] == 15
    assert not result.traces["raw_core_fraction"].any()
    assert result.traces["raw_ring_fraction"][12:].min() > 0
    assert result.traces["residual_ring_fraction"][:12].max() == 0
    assert result.traces["residual_ring_fraction"][12:].min() > 0
    assert "dark soma cores are background" in result.summary["cfar"]["raw_evidence"]
    assert result.summary["metrics"]["residual"]["ring_enrichment"]["post_ratio_to_global"] > 1

    assert all(array.ndim <= 2 for array in result.array_payload().values())
    assert all(array.dtype == np.uint32 for array in result.count_maps.values())
    json.dumps(result.summary)


@pytest.mark.parametrize("threshold_method", ["p99", "mean_plus_3sd"])
def test_positive_residual_signal_finds_broad_excitation_without_cfar(
    tmp_path, monkeypatch, threshold_method,
) -> None:
    path = tmp_path / "late_burst.npy"
    np.save(path, _synthetic_video())
    monkeypatch.setattr(
        detector_module, "_cfar_mask",
        lambda evidence, config: np.zeros_like(evidence, dtype=bool),
    )

    result = _run(path, chunk_frames=3, zone_threshold_method=threshold_method)

    assert result.summary["zone_activation"]["residual"][
        "activated_zone_count"] == 0
    activation = result.summary["zone_activation"]["positive_residual_signal"]
    assert activation["activated_zone_count"] == 1
    assert activation["zones"][0]["onset_source_index"] == 14
    assert activation["zones"][0]["onset_ui_frame"] == 15

    metrics = result.summary["metrics"]["positive_residual_signal"]
    assert metrics["ring"]["pre_mean"] == 0.0
    assert metrics["ring"]["post_mean"] > 0.0
    assert metrics["core"]["post_mean"] == 0.0
    assert metrics["ring_enrichment"]["post_ratio_to_global"] > 1.0
    assert result.traces["positive_residual_ring_mean"][:12].max() == 0.0
    assert result.traces["positive_residual_ring_mean"][12:].min() > 0.0
    zone_trace = result.zone_ring_traces["positive_residual_ring_mean"][:, 0]
    assert zone_trace[:12].max() == 0.0
    assert zone_trace[12:].min() > 0.0


def test_chunk_size_invariance_and_bounded_range_requests(tmp_path, monkeypatch) -> None:
    path = tmp_path / "late_burst.npy"
    np.save(path, _synthetic_video().astype(np.float32))
    reference = _run(path, chunk_frames=1)

    calls: list[tuple[int, int, int]] = []
    original = detector_module.iter_video_chunks

    def tracking_iterator(source, **kwargs):
        calls.append(
            (kwargs["start_frame"], kwargs["end_frame"], kwargs["chunk_size"])
        )
        for chunk in original(source, **kwargs):
            assert chunk.frame_count <= kwargs["chunk_size"]
            yield chunk

    monkeypatch.setattr(detector_module, "iter_video_chunks", tracking_iterator)
    chunked = _run(path, chunk_frames=4)

    assert calls == [(0, 8, 4), (0, 8, 4), (10, 20, 4)]
    assert reference.summary["normalization"] == chunked.summary["normalization"]
    assert reference.summary["metrics"] == chunked.summary["metrics"]
    assert reference.summary["zone_activation"] == chunked.summary["zone_activation"]
    for name, expected in reference.array_payload().items():
        np.testing.assert_array_equal(chunked.array_payload()[name], expected)


def test_constant_video_is_safe_and_json_ready(tmp_path) -> None:
    path = tmp_path / "constant.npy"
    np.save(path, np.full((12, 32, 32), 700, dtype=np.uint16))

    result = run_streamed_detector(
        path,
        control_start=0,
        control_stop=4,
        score_start=4,
        score_stop=12,
        chunk_frames=2,
        cfar=CFARConfig(small_radius_px=1, large_radius_px=4, pfa=0.1),
        zone_config=DarkSomaZoneConfig(border=5, max_zones=10),
    )

    assert result.summary["normalization"]["constant_control"] is True
    assert result.summary["dark_zones"]["count"] == 0
    assert not any(array.any() for array in result.count_maps.values())
    assert result.summary["metrics"]["raw"]["global"]["ratio"] is None
    assert result.summary["metrics"]["residual"]["global"]["ratio"] is None
    json.dumps(result.summary)


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"score_start": 3}, "start after control"),
        ({"chunk_frames": 129}, "between 1 and 128"),
        ({"zone_threshold_method": "adaptive"}, "zone_threshold_method"),
    ],
)
def test_resource_and_range_validation(tmp_path, overrides, message) -> None:
    path = tmp_path / "video.npy"
    np.save(path, np.ones((12, 16, 16), dtype=np.uint16))
    arguments = {
        "control_start": 0,
        "control_stop": 4,
        "score_start": 4,
        "score_stop": 12,
        "chunk_frames": 2,
        "cfar": CFARConfig(small_radius_px=1, large_radius_px=3),
        "zone_config": DarkSomaZoneConfig(border=3),
    }
    arguments.update(overrides)

    with pytest.raises(ValueError, match=message):
        run_streamed_detector(path, **arguments)
