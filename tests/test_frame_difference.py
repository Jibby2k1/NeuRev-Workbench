from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile

from neurobench.experiments.frame_difference import FrameDifferenceConfig, run


def test_signed_derivative_tiffs_preserve_alignment_and_zero_midpoint(tmp_path: Path) -> None:
    video = np.asarray(
        [
            [[10, 10], [10, 10]],
            [[12, 8], [10, 10]],
            [[15, 5], [10, 10]],
            [[11, 9], [10, 10]],
            [[20, 0], [10, 10]],
            [[18, 2], [10, 10]],
        ],
        dtype=np.uint16,
    )
    source_video = tmp_path / "source.npy"
    source_tiff = tmp_path / "source.tif"
    np.save(source_video, video)
    tifffile.imwrite(source_tiff, video, photometric="minisblack")
    config = FrameDifferenceConfig(
        experiment_id="test",
        source_video=source_video,
        source_tiff=source_tiff,
        output_dir=tmp_path / "new-parent" / "derivatives",
        lags=(1, 4),
        frame_period_ms=20,
        absolute_percentile=99.5,
        zero_code=32768,
        frame_chunk=2,
        cpu_threads=1,
        max_ram_mib=1,
        min_free_disk_mib=1,
        max_output_mib=10,
    )
    payload = run(config)
    assert payload["status"] == "complete"
    lag1 = tifffile.memmap(config.output_dir / "spon_ca_burst_derivative_lag1.tif")
    lag4 = tifffile.memmap(config.output_dir / "spon_ca_burst_derivative_lag4.tif")
    assert lag1.shape == video.shape == lag4.shape
    assert np.all(lag1[0] == 32768)
    assert np.all(lag4[:4] == 32768)
    assert lag1[1, 0, 0] > 32768
    assert lag1[1, 0, 1] < 32768
    assert lag4[4, 0, 0] > 32768
    assert lag4[4, 0, 1] < 32768
    assert np.all(lag1[:, 1] == 32768)
