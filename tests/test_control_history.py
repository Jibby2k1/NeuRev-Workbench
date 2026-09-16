from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from neurobench.experiments.gamma_ls_difference import control_history as history
from neurobench.experiments.hierarchical_parzen_ica.architecture_lanes import (
    AffineICAReconstruction,
)
from neurobench.experiments.hierarchical_parzen_ica.signal_noise_split import (
    _innovation_residual,
    _quiet_standardization,
)


COEFFICIENTS = AffineICAReconstruction(0.959, 0.033, -2.66)


def _raw() -> np.ndarray:
    rng = np.random.default_rng(20260912)
    result = rng.integers(95, 115, size=(17, 5, 7), dtype=np.uint16)
    result[9:13, 2, 3] += 70
    result[12:, 1, 5] -= 40
    return result


def test_streamed_outputs_exactly_match_legacy_carrier_and_source_alignment(tmp_path: Path) -> None:
    raw = _raw()
    config = SimpleNamespace(
        input_lane={"reference_half_life_seconds": 10.0, "correction_fraction": 0.1, "correction_clip_mad": 4.0},
        frames={"frame_period_ms": 20.0},
    )
    expected, _ = _innovation_residual(raw.astype(np.float32), 8, COEFFICIENTS, config)
    center, scale, _ = _quiet_standardization(expected, 8, 10.0)
    expected_carrier = ((expected - center) / scale).astype(np.float16)
    result = history.build_historical_fixture_stages(
        raw, tmp_path / "stages", quiet_frame_count=8, source_start_ui=1800,
        fixture_coefficients=COEFFICIENTS,
    )
    np.testing.assert_array_equal(np.load(result["paths"]["raw_residual"]), expected)
    np.testing.assert_array_equal(np.load(result["paths"]["carrier_signed"]), expected_carrier)
    assert result["input"]["source_interval_ui_inclusive"] == [1800, 1816]
    assert result["input"]["quiet_interval_ui_inclusive"] == [1800, 1807]
    assert result["causality"]["online_eligible_from_source_ui"] == 1808
    assert not result["causality"]["setup_outputs_are_online_eligible"]
    # The first stored frame is raw[0]-quiet_background; no lag-one page shift.
    np.testing.assert_array_equal(expected[0], (raw[0] - np.median(raw[:8], axis=0)).astype(np.float32))
    assert expected[10, 2, 3] > 0 and expected[-1, 1, 5] < 0


def test_future_perturbation_cannot_change_fit_scale_or_earlier_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw = _raw()
    modified = raw.copy()
    modified[12:] += 500
    fit_inputs = []

    def fit(quiet: np.ndarray, config: SimpleNamespace):
        fit_inputs.append(quiet.copy())
        assert config.stochastic["sample_seed"] == 20260729
        # Deliberately input-dependent fit makes accidental full-view fitting
        # observable, without running the stochastic optimizer in a unit test.
        coefficients = AffineICAReconstruction(0.9, 0.02, float(np.mean(quiet)) * 0.001)
        return coefficients, {"quiet_mean": float(np.mean(quiet)), "labels_used": False}

    monkeypatch.setattr(history, "_fit_raw_stochastic", fit)
    first = history.build_historical_fixture_stages(raw, tmp_path / "first", quiet_frame_count=8)
    second = history.build_historical_fixture_stages(modified, tmp_path / "second", quiet_frame_count=8)
    assert len(fit_inputs) == 2
    for fit_input in fit_inputs:
        np.testing.assert_array_equal(fit_input, raw[:8])
    assert first["fit"] == second["fit"]
    assert first["standardization"] == second["standardization"]
    assert first["input"]["quiet_logical_c_order_bytes_sha256"] == second["input"]["quiet_logical_c_order_bytes_sha256"]
    assert first["input"]["logical_c_order_bytes_sha256"] != second["input"]["logical_c_order_bytes_sha256"]
    for stage in ("raw_residual", "carrier_signed"):
        before, after = np.load(first["paths"][stage]), np.load(second["paths"][stage])
        np.testing.assert_array_equal(before[:12], after[:12])
        assert not np.array_equal(before[12:], after[12:])
    with np.load(first["paths"]["calibration"]) as left, np.load(second["paths"]["calibration"]) as right:
        for name in left.files:
            np.testing.assert_array_equal(left[name], right[name])


def test_constant_fixture_is_finite_zero_and_hashes_bind_actual_files(tmp_path: Path) -> None:
    # Zero is exactly representable throughout the historical mixed-dtype EMA;
    # a nonzero constant can retain tiny legacy roundoff, so do not demand its
    # arithmetic be changed to obtain an artificial exact-constant invariant.
    raw = np.zeros((9, 4, 6), dtype=np.uint16)
    result = history.build_historical_fixture_stages(
        raw, tmp_path / "constant", quiet_frame_count=5,
        fixture_coefficients=AffineICAReconstruction(1.0, 0.0, 0.0),
    )
    for name, item in result["files"].items():
        path = Path(result["paths"][name])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
    for stage in ("raw_residual", "carrier_signed"):
        output = np.load(result["paths"][stage])
        assert np.isfinite(output).all()
        np.testing.assert_array_equal(output, np.zeros(raw.shape))
    assert result["input"]["logical_c_order_bytes_sha256"] == hashlib.sha256(raw.tobytes()).hexdigest()
    assert json.loads((tmp_path / "constant" / "metadata.json").read_text()) == result
    assert result["settings"]["correction_limit"] == 1e-6
    assert result["standardization"]["quiet_scale_floor"] == 1.0
    assert result["cpu_threads"] == 1


def test_production_wrapper_fits_only_first100_with_fixed_ui_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Nonzero setup variation supplies a meaningful historical MAD scale;
    # constant setup followed by a jump legitimately overflows a float16 lane.
    raw = np.broadcast_to(
        (100 + np.arange(560, dtype=np.uint16) % 7)[:, None, None], (560, 2, 3)
    ).copy()
    raw[100:] += 25
    calls = []

    def fit(quiet: np.ndarray, config: SimpleNamespace):
        calls.append(quiet.copy())
        return AffineICAReconstruction(1.0, 0.0, 0.0), {"fixture_fit_spy": True}

    monkeypatch.setattr(history, "_fit_raw_stochastic", fit)
    result = history.build_historical_stages(raw, tmp_path / "production_contract")
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0], raw[:100])
    assert result["input"]["source_interval_ui_inclusive"] == [1800, 2359]
    assert result["input"]["quiet_interval_ui_inclusive"] == [1800, 1899]
    assert result["causality"]["online_eligible_from_source_ui"] == 1900
    assert result["fit"]["fixture_fit_spy"]


def test_float16_overflow_is_rejected_without_committing_output(tmp_path: Path) -> None:
    raw = np.zeros((9, 4, 6), dtype=np.float32)
    raw[-1] = 1e9
    output = tmp_path / "overflow"
    with pytest.warns(RuntimeWarning, match="overflow"):
        with pytest.raises(ValueError, match="float16 carrier overflow"):
            history.build_historical_fixture_stages(
                raw, output, quiet_frame_count=8,
                fixture_coefficients=AffineICAReconstruction(1.0, 0.0, 0.0),
            )
    assert not output.exists()
    assert (tmp_path / "overflow.partial").exists()


def test_collision_refuses_overwrite_and_fit_failure_produces_no_completed_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw = _raw()
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "marker"
    marker.write_text("preserve")
    with pytest.raises(FileExistsError):
        history.build_historical_fixture_stages(raw, output, quiet_frame_count=8)
    assert marker.read_text() == "preserve"
    (tmp_path / "partial_only.partial").mkdir()
    with pytest.raises(FileExistsError):
        history.build_historical_fixture_stages(raw, tmp_path / "partial_only", quiet_frame_count=8)

    def reject(*args, **kwargs):
        raise RuntimeError("raw stochastic fit did not pass gates")

    monkeypatch.setattr(history, "_fit_raw_stochastic", reject)
    with pytest.raises(RuntimeError, match="did not pass gates"):
        history.build_historical_fixture_stages(raw, tmp_path / "failed_fit", quiet_frame_count=8)
    assert not (tmp_path / "failed_fit").exists()


@pytest.mark.parametrize("case", ["shape", "short", "complex", "nan", "bad_period", "bad_source_ui"])
def test_invalid_inputs_fail_without_completed_artifacts(tmp_path: Path, case: str) -> None:
    raw = _raw().astype(np.float32)
    kwargs = {"quiet_frame_count": 8, "fixture_coefficients": COEFFICIENTS}
    if case == "shape":
        raw = raw[0]
    elif case == "short":
        kwargs["quiet_frame_count"] = 3
    elif case == "complex":
        raw = raw.astype(np.complex64)
    elif case == "nan":
        raw[-1, 0, 0] = np.nan
    elif case == "bad_period":
        kwargs["frame_interval_s"] = 0.0
    elif case == "bad_source_ui":
        kwargs["source_start_ui"] = 0
    output = tmp_path / case
    with pytest.raises(ValueError):
        history.build_historical_fixture_stages(raw, output, **kwargs)
    assert not output.exists()


def test_production_refuses_ambiguous_short_view(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="560"):
        history.build_historical_stages(_raw(), tmp_path / "wrong_view")
