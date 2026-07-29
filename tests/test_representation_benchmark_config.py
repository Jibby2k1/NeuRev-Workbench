import json
from pathlib import Path

import pytest

from neurobench.experiments.representation_benchmark.config import (
    RepresentationBenchmarkConfig,
    RepresentationConfigError,
)


def _payload() -> dict:
    return {
        "schema_version": 1, "experiment_id": "tiny",
        "source_video": "video.npy", "labels_tsv": "labels.tsv",
        "output_dir": "output",
        "frames": {
            "review_start_ui": 1, "review_end_ui": 20,
            "quiet_start_ui": 1, "quiet_end_ui": 5, "frame_period_ms": 20,
        },
        "pca": {"inputs": ["quiet_residual"], "ranks": [2, 4]},
        "ica": {
            "inputs": ["quiet_residual"], "ranks": [2], "seeds": [7],
            "fit_sample_pixels": 1024, "max_iterations": 50, "tolerance": 1e-4,
        },
        "autoencoder": {
            "enabled": True, "inputs": ["quiet_residual"], "kinds": ["linear"],
            "ranks": [2], "seeds": [7], "train_pixels": 1024,
            "validation_pixels": 256, "epochs": 2, "batch_size": 64,
            "learning_rate": 0.001, "hidden_width": 8,
        },
        "umap": {
            "enabled_if_available": False, "neighbors": 10, "min_dist": 0.1,
            "sample_pixels": 100, "seed": 7,
        },
        "evaluation": {
            "nms_distance_px": 6, "match_radius_px": 6,
            "quiet_false_peaks_per_map": 1, "fixed_candidates_per_burst": 10,
            "reconstruction_ranks": [2], "representative_rank": 2,
            "component_gallery_count": 2, "write_representative_tiffs": False,
        },
        "resources": {
            "device": "cpu", "cpu_threads": 1, "projection_chunk_pixels": 128,
            "max_ram_mib": 512, "min_free_disk_mib": 128,
            "max_output_mib": 128, "gpu_reserve_mib": 0,
        },
    }


def test_config_loads_and_counts_are_bounded(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(_payload()), encoding="utf-8")
    config = RepresentationBenchmarkConfig.load(path)
    assert config.pca.ranks == (2, 4)
    assert config.resources.device == "cpu"


def test_config_rejects_unknown_fields(tmp_path: Path) -> None:
    payload = _payload()
    payload["surprise"] = True
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RepresentationConfigError):
        RepresentationBenchmarkConfig.load(path)


def test_preflight_json_normalization(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(_payload()), encoding="utf-8")
    config = RepresentationBenchmarkConfig.load(path)
    normalized = json.loads(json.dumps(config.to_dict()))
    assert normalized["pca"]["ranks"] == [2, 4]
