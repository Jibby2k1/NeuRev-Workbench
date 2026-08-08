from __future__ import annotations

import numpy as np
import pytest

from neurobench.experiments.pairwise_separation.basic_parzen_diagnostic import (
    _fit_direction,
    _gray,
    _render_frame,
)


def test_fit_direction_reports_observation_geometry() -> None:
    fit = {
        "activity_component": 1,
        "activity_sign": 1,
        "whitening": [[1.0, 0.0], [0.0, 1.0]],
        "demixing": [[1.0, 1.0], [-1.0, 1.0]],
    }
    result = _fit_direction(fit)
    assert np.allclose(result["normalized_effective_direction"], [-2**-0.5, 2**-0.5])
    assert result["absolute_cosine_to_derivative"] == pytest.approx(1.0)
    assert result["absolute_cosine_to_common_direction"] < 1e-12


def test_signed_gray_has_midgray_zero() -> None:
    result = _gray(np.asarray([[-1.0, 0.0, 1.0]], dtype=np.float32), (-1.0, 1.0))
    assert result.tolist() == [[0, 128, 255]]


def test_formula_explicit_frame_is_fixed_1080p() -> None:
    shape = (34, 57)
    zero = np.zeros(shape, dtype=np.float32)
    ramp = np.linspace(0, 1, np.prod(shape), dtype=np.float32).reshape(shape)
    bounds = {
        "raw": (0.0, 1.0),
        "filtered": (0.0, 1.0),
        "fixed": (-1.0, 1.0),
        "parzen": (-1.0, 1.0),
        "z_positive": (0.0, 3.0),
        "residual": (-1.0, 1.0),
    }
    image = _render_frame(
        raw=ramp,
        filtered=ramp,
        fixed=zero,
        parzen=zero,
        z_positive=zero,
        residual=zero,
        bounds=bounds,
        ui_frame=2005,
        review_start_ui=1800,
        burst_id=1,
        rings=((30.0, 20.0),),
        beta=1.0,
        effective_direction=(-2**-0.5, 2**-0.5),
        scale_floor=0.45,
    )
    assert image.size == (1920, 1080)
    assert image.mode == "RGB"
