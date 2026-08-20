import numpy as np
import pytest

from neurobench.experiments.msln_msica.center_whiten_ica_benchmark import (
    ICAChoice,
    SCREEN_AUDIT_OPTOUT,
    apply_ica_rotation,
    deterministic_sample_rows,
    fit_ica_rotation,
    load_screen_config,
    stage_a_lanes,
    stage_b_lanes,
    stage_c_lanes,
)


def test_preregistered_stage_sizes_and_caps() -> None:
    stage_a = stage_a_lanes()
    assert len(stage_a) == 8
    retained = stage_a[:4]
    assert len(stage_b_lanes(retained)) == 28
    stage_c = stage_c_lanes(tuple((*item, "none") for item in retained))
    assert len(stage_c) == 32
    assert len({lane.lane_id for lane in stage_c}) == len(stage_c)
    assert "metrics-only" in SCREEN_AUDIT_OPTOUT
    with pytest.raises(ValueError, match="at most four"):
        stage_b_lanes(stage_a[:5])


def test_external_whitening_is_not_repeated_by_fastica() -> None:
    rng = np.random.default_rng(12)
    sources = np.column_stack((
        rng.laplace(size=4000), rng.uniform(-1, 1, 4000),
        rng.standard_t(5, size=4000),
    ))
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    mixed = sources @ q.T
    fit = fit_ica_rotation(
        mixed, ICAChoice("logcosh", 7, 300, 1e-5),
        native_global_whitening=False,
    )
    assert fit.internal_whitening is False
    assert fit.rotation.shape == (3, 3)
    assert apply_ica_rotation(mixed.reshape(20, 20, 10, 3), fit).dtype == np.float32


def test_native_ica_records_internal_global_whitening() -> None:
    rng = np.random.default_rng(4)
    values = rng.laplace(size=(2000, 3)) @ np.asarray([[1, .2, 0], [.1, 2, .2], [0, .3, 1]])
    fit = fit_ica_rotation(
        values, ICAChoice("exp", 7, 300, 1e-5),
        native_global_whitening=True,
    )
    assert fit.internal_whitening is True
    assert np.linalg.norm(fit.mean) > 0

    whitening_only = fit_ica_rotation(
        values, ICAChoice("identity", 7, 0, 0.0),
        native_global_whitening=True,
    )
    transformed = apply_ica_rotation(values, whitening_only)
    np.testing.assert_allclose(np.cov(transformed.T, bias=True), np.eye(3), atol=2e-5)
    assert whitening_only.internal_whitening is True


def test_deterministic_rows_preserve_sample_ids() -> None:
    values = np.arange(10 * 3 * 4 * 3, dtype=np.float32).reshape(10, 3, 4, 3)
    valid = np.asarray([False, True, True, False, True, True, True, False, True, True])
    left, left_ids = deterministic_sample_rows(values, valid, maximum_samples=31, seed=7)
    right, right_ids = deterministic_sample_rows(values, valid, maximum_samples=31, seed=7)
    np.testing.assert_array_equal(left_ids, right_ids)
    np.testing.assert_array_equal(left, right)


def test_example_config_is_strict_metrics_only() -> None:
    config = load_screen_config("examples/spon_ca_burst_center_whiten_ica_v1.example.json")
    assert config["screen"]["save_diagnostics"] is False
    assert config["compute"]["workers"] == 1
    assert "labels_path" not in config["source"]
