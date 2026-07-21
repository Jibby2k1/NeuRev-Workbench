from __future__ import annotations

import numpy as np

from neurobench.algorithms.background import kalman_positive_residual_stack


def test_kalman_positive_residual_preserves_transient_activity(tmp_path):
    video = np.full((20, 6, 7), 100.0, dtype=np.float32)
    video[:, 1, 1] = 300.0
    video[8, 3, 4] = 220.0
    video[9, 3, 4] = 210.0
    source = tmp_path / "video.npy"
    np.save(source, video)

    summary = kalman_positive_residual_stack(
        source,
        tmp_path / "residual",
        baseline_init_frames=3,
        kalman_gain=0.0,
        positive_update_gain=0.001,
        negative_update_gain=0.2,
        chunk_frames=4,
    )

    residual = np.load(summary["residual_npy"], mmap_mode="r")
    assert residual.shape == video.shape
    assert float(residual[8, 3, 4]) > 100.0
    assert float(residual[9, 3, 4]) > 80.0
    assert float(np.median(residual[:, 1, 1])) == 0.0


def test_kalman_positive_residual_writes_projection_artifacts(tmp_path):
    video = np.zeros((4, 3, 3), dtype=np.uint16)
    video[2, 1, 1] = 50
    source = tmp_path / "video.npy"
    np.save(source, video)

    summary = kalman_positive_residual_stack(source, tmp_path / "residual", baseline_init_frames=1)

    assert (tmp_path / "residual" / "background_residual_summary.json").exists()
    assert np.load(summary["artifacts"]["positive_residual_max_projection_npy"]).shape == (3, 3)
    assert np.load(summary["artifacts"]["final_baseline_npy"]).shape == (3, 3)
