from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from neurobench.experiments.latent_dynamics import LatentDynamicsConfig, preflight


def write_fixture(tmp_path: Path, *, output_name: str = "run") -> tuple[Path, Path]:
    video = np.random.default_rng(4).normal(100, 2, (120, 40, 41)).astype(np.float32)
    video[104:120, 20, 20] += 12
    np.save(tmp_path / "video.npy", video)
    (tmp_path / "labels.tsv").write_text(
        "burst_id\tstart_frame_ui\tend_frame_ui\tstart_frame_zero\tstop_frame_zero_exclusive\tx_px\ty_px\trecurrence_count\n"
        "1\t105\t108\t104\t108\t20\t20\t1\n"
        "2\t109\t112\t108\t112\t20\t20\t1\n"
        "3\t113\t116\t112\t116\t20\t20\t1\n"
        "4\t117\t120\t116\t120\t20\t20\t1\n", encoding="utf-8"
    )
    payload = {
        "schema_version": 1, "experiment_id": "tiny_latent",
        "source_video": "video.npy", "labels_tsv": "labels.tsv", "output_dir": output_name,
        "frames": {"review_start_ui": 1, "review_end_ui": 120, "quiet_start_ui": 1,
                   "quiet_end_ui": 100, "frame_period_ms": 20.0},
        "preprocessing": {"baseline_mode": "quiet_median", "signed_residual": True,
                          "gain_mode": "none", "motion_mode": "none",
                          "quiet_scale_floor_percentile": 10.0},
        "fit": {"sample_pixels": 32, "sample_seed": 20260727,
                "temporal_validation_blocks": 2, "stability_epsilon": 0.001,
                "decay_time_ms_grid": [40, 80], "process_to_observation_grid": [0.1, 0.3],
                "parameter_mode": "bounded_grid"},
        "application": {"tile_height": 3, "tile_width": 4, "write_filter_mean": True,
                        "write_smoother_mean": True, "write_dense_residuals": False},
        "features": {"lags": [1, 4], "write_dense_features": False,
                     "write_selected_tiffs": False, "positive_views": True},
        "evaluation": {"primary_match_radius_px": 6, "match_radii_px": [4, 6, 8],
                       "quiet_false_peaks_per_map": 1.0, "capacity_reference_lane": "raw_direct",
                       "synthetic_seeds": [7, 13]},
        "resources": {"cpu_threads": 1, "max_ram_mib": 1, "min_free_disk_mib": 1,
                      "max_output_mib": 32},
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    return manifest, tmp_path / output_name


def test_config_resolves_paths_and_rejects_unknown_fields(tmp_path):
    manifest, output = write_fixture(tmp_path)
    config = LatentDynamicsConfig.load(manifest)
    assert config.output_dir == output.resolve()
    payload = json.loads(manifest.read_text())
    payload["fit"]["mystery"] = 1
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Unknown fit fields"):
        LatentDynamicsConfig.load(manifest)


def test_preflight_is_explicit_collision_safe_and_records_contract(tmp_path):
    manifest, _ = write_fixture(tmp_path)
    config = LatentDynamicsConfig.load(manifest)
    artifact = tmp_path / "preflight"
    payload = preflight(config, artifact_dir=artifact)
    assert payload["ready"]
    assert payload["fit"]["labels_available_to_fit"] is False
    assert payload["frames"]["review_zero_half_open"] == [0, 120]
    assert (artifact / "label_projection_overlay.png").is_file()
    with pytest.raises(FileExistsError):
        preflight(config, artifact_dir=artifact)


def test_preflight_refuses_existing_output(tmp_path):
    manifest, output = write_fixture(tmp_path)
    output.mkdir()
    with pytest.raises(RuntimeError):
        preflight(LatentDynamicsConfig.load(manifest), artifact_dir=tmp_path / "preflight")
