from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile

from neurobench.experiments.activity_gated_video import ActivityGateConfig, run


def test_activity_gate_writes_distinct_bounded_review_stacks(tmp_path: Path) -> None:
    rng = np.random.default_rng(17)
    video = rng.normal(300, 3, size=(40, 12, 14)).astype(np.float32)
    video[:, 2, 2] = 4095  # Persistent saturated artifact.
    video[24:31, 6:9, 7:10] += np.linspace(0, 180, 7)[:, None, None]
    video = np.clip(np.rint(video), 0, 4095).astype(np.uint16)
    source_video = tmp_path / "source.npy"
    source_tiff = tmp_path / "source.tif"
    labels_tsv = tmp_path / "labels.tsv"
    np.save(source_video, video)
    tifffile.imwrite(source_tiff, video, photometric="minisblack")
    labels_tsv.write_text(
        "burst_id\tstart_frame_zero\tstop_frame_zero_exclusive\tx_px\ty_px\n"
        "1\t24\t31\t8\t7\n",
        encoding="utf-8",
    )
    config = ActivityGateConfig(
        experiment_id="test_activity_gate",
        source_video=source_video,
        source_tiff=source_tiff,
        labels_tsv=labels_tsv,
        output_dir=tmp_path / "result",
        review_start_ui=10,
        review_end_ui=39,
        quiet_start_ui=10,
        quiet_end_ui=19,
        spatial_sigma_px=1,
        temporal_window_frames=5,
        temporal_polyorder=2,
        derivative_lag_frames=1,
        energy_ema_span_frames=4,
        gate_tau_z=2.5,
        structural_floor=0.2,
        artifact_attenuation=0.7,
        intensity_asinh_gain=5,
        quiet_mad_floor_percentile=10,
        frame_chunk=8,
        cpu_threads=1,
        max_ram_mib=1024,
        min_free_disk_mib=1,
        max_output_mib=10,
    )

    payload = run(config)

    assert payload["status"] == "complete"
    assert [row["variant"] for row in payload["outputs"]] == [
        "strict_gate",
        "floored_gate",
        "artifact_gate",
        "baseline_residual",
    ]
    assert not list(config.output_dir.glob("*.partial"))
    assert (config.output_dir / "run_state.json").read_text().find('"complete"') >= 0
    for record in payload["outputs"]:
        stack = tifffile.memmap(record["path"])
        assert stack.shape == (30, 12, 14)
        assert stack.dtype == np.uint16
    by_name = {row["variant"]: row for row in payload["outputs"]}
    assert (
        by_name["artifact_gate"]["artifact_quiet_median"]
        < by_name["floored_gate"]["artifact_quiet_median"]
    )
    assert payload["artifact_summary"]["artifact_area_ge_0_5"] > 0
