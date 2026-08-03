import shutil

import numpy as np
import pytest

from neurobench.experiments.hierarchical_parzen_ica.dependent_multiscale_grayscale import (
    _gray,
    write_grayscale_review_video,
)


def test_signed_grayscale_has_documented_zero_midpoint():
    frame = np.array([[-1.0, 0.0, 1.0]], dtype=np.float32)
    rendered = _gray(frame, (-1.0, 1.0))
    np.testing.assert_array_equal(rendered, np.array([[0, 128, 255]], dtype=np.uint8))


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg unavailable")
def test_grayscale_review_has_explicit_palette_and_legend(tmp_path):
    rng = np.random.default_rng(42)
    movie = rng.normal(size=(3, 10, 12)).astype(np.float32)
    channels = {
        "background": 0.2 * movie,
        "structured_signal": 0.5 * movie,
        "structured_artifact": 0.1 * movie,
        "noise_candidate": 0.2 * movie,
    }
    result = write_grayscale_review_video(
        display_observation=movie,
        scientific_carrier=movie,
        channels=channels,
        labels_xy=((4.0, 5.0),),
        review_start_ui=20,
        destination=tmp_path / "gray.mp4",
        fps=2,
    )
    assert result["palette"] == "grayscale_only"
    assert "mid-gray zero" in result["signed_legend"]
    assert result["frames"] == 3
    assert (tmp_path / "gray.mp4").is_file()
