import numpy as np
import pytest

from neurobench.algorithms.multiscale_local_normalization import (
    JointSTContext,
    SequentialSTContext,
    SpatialMSLNContext,
    TemporalMSLNContext,
    causal_joint_msln,
    robust_center_scale,
    sequential_msln,
    spatial_msln,
    temporal_msln,
)


def test_robust_center_scale_supports_tuple_axes() -> None:
    values = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    center, scale = robust_center_scale(values, axis=(0, 2))
    assert center.shape == (3,)
    expected = np.median(values, axis=(0, 2))
    expected_scale = 1.4826 * np.median(
        np.abs(values - expected[None, :, None]), axis=(0, 2)
    )
    np.testing.assert_allclose(center, expected)
    np.testing.assert_allclose(scale, expected_scale)


def test_spatial_annulus_is_boundary_correct_and_signed() -> None:
    video = np.zeros((2, 5, 5), dtype=np.float32)
    video[:, 2, 2] = 10.0
    result = spatial_msln(
        video,
        SpatialMSLNContext("spatial_5_meanstd", 5, 1),
        scale_floor=1.0,
    )
    assert result.values[0, 2, 2] == pytest.approx(10.0)
    assert result.values[0, 0, 0] < 0
    assert result.diagnostics["reference_count_min"] == 8
    assert result.diagnostics["reference_count_max"] == 24


def test_temporal_is_causal_and_invalid_prefix_is_zero() -> None:
    video = np.arange(8, dtype=np.float32)[:, None, None]
    context = TemporalMSLNContext("temporal_3_meanstd", 3, 1)
    result = temporal_msln(video, context, scale_floor=1.0)
    assert not result.valid_frames[:3].any()
    assert result.valid_frames[3:].all()
    np.testing.assert_array_equal(result.values[:3], 0)
    # At t=3 the reference is frames [0, 1]; frame 2 is the guard.
    assert result.values[3, 0, 0] == pytest.approx(2.5)


def test_sequential_context_propagates_temporal_validity() -> None:
    rng = np.random.default_rng(5)
    video = rng.normal(size=(10, 7, 7)).astype(np.float32)
    spatial = SpatialMSLNContext("spatial_5_meanstd", 5, 1)
    temporal = TemporalMSLNContext("temporal_3_meanstd", 3, 1)
    sequential = SequentialSTContext(
        "st_t3_s5_meanstd",
        spatial.context_id,
        temporal.context_id,
        "temporal_then_spatial",
    )
    result = sequential_msln(
        video,
        sequential,
        spatial_context=spatial,
        temporal_context=temporal,
    )
    assert not result.valid_frames[:3].any()
    assert result.valid_frames[3:].all()
    assert np.isfinite(result.values).all()


def test_joint_spatiotemporal_is_causal_and_preserves_current_interior() -> None:
    video = np.zeros((10, 9, 9), dtype=np.float32)
    video[6, 3:6, 3:6] = 8.0
    context = JointSTContext("joint_s5_g1_t3_g1", 5, 1, 3, 1)
    result = causal_joint_msln(video, context, scale_floor=1.0)

    assert not result.valid_frames[:3].any()
    assert result.valid_frames[3:].all()
    assert np.all(result.values[:6] == 0)
    assert result.values[6, 4, 4] == 8.0
    assert result.diagnostics["current_frame_excluded"] is True
    assert result.diagnostics["reference_frame_count"] == 2


def test_joint_spatiotemporal_future_frames_do_not_change_past() -> None:
    base = np.arange(12, dtype=np.float32)[:, None, None]
    base = np.broadcast_to(base, (12, 7, 7)).copy()
    changed = base.copy()
    changed[9:] += 1000
    context = JointSTContext("joint_s5_g1_t5_g1", 5, 1, 5, 1)
    left = causal_joint_msln(base, context, scale_floor=1.0)
    right = causal_joint_msln(changed, context, scale_floor=1.0)
    np.testing.assert_allclose(left.values[:9], right.values[:9])
