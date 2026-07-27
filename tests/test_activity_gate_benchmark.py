from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile

from neurobench.experiments.activity_gate_benchmark import (
    ActivityGateBenchmarkConfig,
    LANES,
    run,
)


def test_activity_gate_benchmark_reproduces_raw_and_writes_six_lanes(tmp_path: Path) -> None:
    rng = np.random.default_rng(23)
    video = rng.normal(300, 3, size=(1960, 20, 24)).astype(np.float32)
    video[:, 1, 1] = 4095
    windows = ((1900, 1908), (1910, 1918), (1920, 1930), (1940, 1955))
    for start, stop in windows:
        video[start:stop, 9:12, 11:14] += np.linspace(0, 180, stop - start)[:, None, None]
    video = np.clip(np.rint(video), 0, 4095).astype(np.uint16)
    source_video = tmp_path / "source.npy"
    source_tiff = tmp_path / "source.tif"
    labels_tsv = tmp_path / "labels.tsv"
    design = tmp_path / "design.md"
    np.save(source_video, video)
    tifffile.imwrite(source_tiff, video, photometric="minisblack")
    header = "burst_id\tstart_frame_ui\tend_frame_ui\tstart_frame_zero\tstop_frame_zero_exclusive\tpoint_index\troi_identity\tx_px\ty_px\trecurrence_count\n"
    rows = []
    for burst_id, (start, stop) in enumerate(windows, 1):
        rows.append(f"{burst_id}\t{start + 1}\t{stop}\t{start}\t{stop}\t1\troi_001\t12\t10\t4\n")
    labels_tsv.write_text(header + "".join(rows), encoding="utf-8")
    design.write_text("# Test design\n", encoding="utf-8")
    config = ActivityGateBenchmarkConfig(
        experiment_id="test", source_video=source_video, source_tiff=source_tiff,
        labels_tsv=labels_tsv, design_document=design,
        output_dir=tmp_path / "result", preflight_dir=tmp_path / "preflight",
        review_start_ui=1800, review_end_ui=1960, quiet_start_ui=1800, quiet_end_ui=1899,
        spatial_sigma_px=1, offline_temporal_window_frames=7, offline_temporal_polyorder=2,
        causal_temporal_ema_span_frames=4, derivative_lag_frames=1,
        energy_ema_span_frames=4, gate_tau_z=2.5, causal_structural_floors=(0.2, 0.4),
        artifact_attenuation=0.7, intensity_asinh_gain=5, quiet_mad_floor_percentile=10,
        temporal_pool_tau=0.25, nms_distance_px=2, primary_match_radius_px=2,
        match_radii_px=(1, 2, 3), quiet_false_peaks_per_map=1,
        cpu_threads=1, max_ram_mib=1024, min_free_disk_mib=1, max_output_mib=10,
    )

    summary = run(config)

    assert summary["status"] == "complete"
    metrics = __import__("json").loads((config.output_dir / "metrics.json").read_text())
    assert [row["lane"] for row in metrics["lanes"]] == list(LANES)
    assert metrics["lanes"][0]["total_labels"] == 4
    assert (config.preflight_dir / "label_projection_overlay.png").is_file()
    assert (config.output_dir / "candidate_peaks.tsv").is_file()
    assert not list(config.output_dir.glob("*.tmp"))
