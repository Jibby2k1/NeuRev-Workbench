import shutil

import numpy as np
import pytest

from neurobench.experiments.hierarchical_parzen_ica.dependent_multiscale_figures import (
    write_decomposition_video,
)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg unavailable")
def test_diagnostic_video_is_atomic_fixed_scale_and_aligned(tmp_path):
    rng = np.random.default_rng(30)
    carrier = rng.normal(size=(4, 12, 14)).astype(np.float32)
    channels = {
        "background": 0.2 * carrier,
        "structured_signal": 0.5 * carrier,
        "structured_artifact": 0.1 * carrier,
        "noise_candidate": 0.2 * carrier,
        "closure_residual": np.zeros_like(carrier),
    }
    views = {f"scale_{size}": carrier / size for size in (5, 7, 15)}
    path = tmp_path / "diagnostic.mp4"
    result = write_decomposition_video(
        display_observation=carrier,
        scientific_carrier=carrier,
        channels=channels,
        views=views,
        labels_xy=((7.0, 6.0),),
        review_start_ui=10,
        destination=path,
        fps=2,
    )
    assert path.is_file() and result["frames"] == 4
    assert result["scaling_contract"].startswith("fixed")
    assert not (tmp_path / "diagnostic.partial.mp4").exists()
