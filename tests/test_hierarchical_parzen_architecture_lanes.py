import numpy as np

from neurobench.experiments.hierarchical_parzen_ica.architecture_lanes import (
    ARCHITECTURE_IDS,
    AffineICAReconstruction,
    calibrate_reference_parzen_innovation,
    iter_architecture_frames,
    quiet_median_background,
    refresh_from_half_life,
)


def _constant_movie(value: float = 5.0) -> np.ndarray:
    return np.full((12, 6, 7), value, dtype=np.float64)


def test_teacher_forced_lane_uses_real_previous_frame_and_closes() -> None:
    frames = np.arange(8 * 4 * 5, dtype=float).reshape(8, 4, 5)
    coefficients = AffineICAReconstruction(0.7, 0.2, -3.0)
    baseline = quiet_median_background(frames, 4)
    outputs = list(
        iter_architecture_frames(
            frames,
            "teacher_forced_stochastic",
            coefficients,
            quiet_background=baseline,
        )
    )
    assert [item.output_index_zero for item in outputs] == list(range(1, 8))
    expected = 0.7 * frames[0] + 0.2 * frames[1] - 3.0
    np.testing.assert_allclose(outputs[0].background, expected)
    np.testing.assert_allclose(
        outputs[0].background + outputs[0].dynamics_noise,
        frames[1],
    )


def test_quiet_fixed_point_removes_free_offset_drift() -> None:
    frames = _constant_movie()
    coefficients = AffineICAReconstruction(0.8, 0.1, -1.0)
    baseline = quiet_median_background(frames, 5)
    raw = list(
        iter_architecture_frames(
            frames,
            "raw_stochastic_recurrence",
            coefficients,
            quiet_background=baseline,
        )
    )
    fixed = list(
        iter_architecture_frames(
            frames,
            "quiet_fixed_point_recurrence",
            coefficients,
            quiet_background=baseline,
        )
    )
    assert raw[-1].background.mean() < 2.0
    for item in fixed:
        np.testing.assert_allclose(item.background, 5.0)
        np.testing.assert_allclose(item.dynamics_noise, 0.0)


def test_regularized_innovation_is_quiet_zeroed_and_bounded() -> None:
    frames = _constant_movie()
    frames[7:, 2:4, 3:5] += 4.0
    coefficients = AffineICAReconstruction(0.85, 0.12, -0.8)
    calibration = calibrate_reference_parzen_innovation(
        frames,
        6,
        coefficients,
        frame_period_ms=20.0,
        reference_half_life_seconds=10.0,
        correction_fraction=0.1,
        correction_clip_mad=4.0,
        minimum_correction_limit=0.5,
    )
    outputs = list(
        iter_architecture_frames(
            frames,
            "reference_parzen_innovation",
            coefficients,
            quiet_background=calibration.quiet_background,
            innovation=calibration,
        )
    )
    for item in outputs[:5]:
        np.testing.assert_allclose(item.background, 5.0, atol=1e-6)
    refresh = calibration.reference_refresh
    reference = calibration.quiet_background.copy()
    for item in outputs:
        current = frames[item.output_index_zero]
        reference = (1.0 - refresh) * reference + refresh * current
        correction = item.background - reference
        assert np.max(np.abs(correction)) <= (
            calibration.correction_fraction
            * calibration.correction_limit
            + 1e-6
        )
        np.testing.assert_allclose(
            item.background + item.dynamics_noise,
            current,
            atol=1e-6,
        )


def test_half_life_refresh_and_architecture_contract() -> None:
    refresh = refresh_from_half_life(10.0, 20.0)
    assert 0.001 < refresh < 0.002
    assert ARCHITECTURE_IDS == (
        "teacher_forced_stochastic",
        "raw_stochastic_recurrence",
        "quiet_fixed_point_recurrence",
        "reference_parzen_innovation",
    )
