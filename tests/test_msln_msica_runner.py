import json
from pathlib import Path

import numpy as np
import pytest

from neurobench.experiments.msln_msica.config import MSLNMSICAConfig
from neurobench.experiments.msln_msica.preflight import preflight
from neurobench.experiments.msln_msica.runner import run, summarize


EXAMPLE = Path(__file__).parents[1] / "examples" / "spon_ca_burst_msln_msica_v1.example.json"


def _tiny_config(tmp_path: Path) -> Path:
    rng = np.random.default_rng(18)
    video = rng.normal(10, .4, size=(36, 21, 21)).astype(np.float32)
    yy, xx = np.mgrid[:21, :21]
    spot = np.exp(-((xx - 10) ** 2 + (yy - 10) ** 2) / 3)
    video[25:28] += 5 * spot
    movie = tmp_path / "movie.npy"; np.save(movie, video)
    labels = tmp_path / "labels.tsv"
    labels.write_text(
        "burst_id\tstart_frame_ui\tend_frame_ui\tstart_frame_zero\tstop_frame_zero_exclusive\tpoint_index\troi_identity\tx_px\ty_px\trecurrence_count\n"
        "1\t26\t28\t25\t28\t1\troi_001\t10\t10\t1\n",
        encoding="utf-8",
    )
    payload = json.loads(EXAMPLE.read_text())
    payload["experiment_id"] = "tiny_msln_msica"
    payload["source"].update({"movie_path": str(movie), "labels_path": str(labels), "baseline_evidence_dir": None, "review_interval_ui": [1, 36], "quiet_interval_ui": [7, 20], "burst_intervals_ui": {"1": [26, 28]}})
    payload["contexts"]["temporal"]["windows_frames"] = [5]
    payload["contexts"]["spatiotemporal"].update({"enabled": False, "pairs": []})
    payload["sampling"].update({"per_context_screen_samples": 48, "per_context_confirmation_samples": 96, "cross_context_max_samples": 256, "bootstrap_replicates": 2, "time_block_length_frames": 4})
    payload["cross_context"]["modes"] = ["identity", "pca", "group_energy"]
    payload["cross_context"]["max_components"] = 4
    payload["cross_context"]["groups"] = {"compact_spatial": ["spatial_5_innovation", "spatial_7_innovation"]}
    payload["compute"].update({"cpu_threads": 1, "frame_chunk": 4, "max_peak_ram_gb": 1, "max_peak_vram_gb": 1})
    payload["outputs"].update({"root_dir": str(tmp_path / "output"), "selected_video_count": 0, "representative_frames_ui": [10, 26]})
    payload["fold_ids"] = [1]
    config_path = tmp_path / "config.json"; config_path.write_text(json.dumps(payload), encoding="utf-8")
    return config_path


def test_tiny_preflight_run_summary_and_collision(tmp_path: Path) -> None:
    config = MSLNMSICAConfig.load(_tiny_config(tmp_path))
    ready = preflight(config)
    assert ready["ready"] and not ready["labels_used_for_fitting"]
    with pytest.raises(FileExistsError):
        preflight(config)
    completed = run(config)
    assert completed["status"] == "complete"
    summary = summarize(config.outputs.root_dir)
    assert summary["status"]["status"] == "complete"
    required = [
        "run_manifest.json", "context_manifest.tsv", "sample_manifest.npz",
        "features/routing/activity_evidence.npy", "features/routing/dominant_context.npy",
        "diagnostics/context_maps_montage.tif", "diagnostics/quiet_ccdf.png",
        "evaluation/synthetic/metrics.json", "stage_gate.json", "RESULTS.md",
    ]
    assert all((config.outputs.root_dir / item).is_file() for item in required)
    assert run(config, resume=True)["status"]["status"] == "complete"
