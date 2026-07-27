from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile

from neurobench.experiments.smoothed_frame_difference import (
    SmoothedDifferenceConfig,
    run,
)


def test_smoothed_derivatives_write_four_aligned_tiffs_and_remove_cache(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    video = rng.normal(100, 2, size=(20, 8, 10)).astype(np.float32)
    video[10:, 3:6, 4:7] += np.linspace(0, 30, 10)[:, None, None]
    video = np.clip(np.rint(video), 0, 65535).astype(np.uint16)
    source_video = tmp_path / "source.npy"
    source_tiff = tmp_path / "source.tif"
    np.save(source_video, video)
    tifffile.imwrite(source_tiff, video, photometric="minisblack")
    config = SmoothedDifferenceConfig(
        experiment_id="test",
        source_video=source_video,
        source_tiff=source_tiff,
        output_dir=tmp_path / "result",
        lags=(1, 4),
        frame_period_ms=20,
        quiet_start_ui=2,
        quiet_end_ui=9,
        spatial_sigma_px=1,
        temporal_window_frames=7,
        temporal_polyorder=2,
        global_absolute_percentile=99.5,
        quiet_mad_floor_percentile=10,
        quiet_clip_z=5,
        quiet_deadband_z=2.5,
        sample_spatial_stride=2,
        frame_chunk=8,
        cpu_threads=1,
        max_ram_mib=1,
        min_free_disk_mib=1,
        max_output_mib=10,
    )
    payload = run(config)
    assert payload["status"] == "complete"
    assert len(payload["outputs"]) == 4
    assert not (config.output_dir / "smoothed_source.partial.npy").exists()
    assert not list(config.output_dir.glob("*.partial"))
    neutral = {}
    for record in payload["outputs"]:
        stack = tifffile.memmap(record["path"])
        lag = record["lag_frames"]
        assert stack.shape == video.shape
        assert stack.dtype == np.uint16
        assert np.all(stack[:lag] == 32768)
        assert stack.min() < 32768 < stack.max()
        neutral[(lag, record["normalization"])] = record["neutral_fraction"]
    assert neutral[(1, "quiet_mad")] > neutral[(1, "global")]
