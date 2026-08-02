import json
from pathlib import Path

import pytest

from neurobench.experiments.information_source_separation.conclusive_config import (
    ConclusiveBatchConfig,
    ConclusiveBatchConfigError,
)
from neurobench.experiments.information_source_separation.conclusive_batch import prepare
from neurobench.experiments.information_source_separation.conclusive_stage4 import (
    _configure_gpu_allocation_limit,
)


def _manifest(tmp_path: Path) -> dict:
    methods = []
    for method_id in (
        "raw_direct_reference", "amplitude_pca_reference", "multilag_sobi",
        "full_window_spatial_fastica_reference", "dense_patch_fastica_wiener_reference",
        "kernel_hsic_pairwise_rotation", "knn_mi_pairwise_rotation",
        "group_energy_hsic_isa", "spatial_noisy_parzen_infomax",
        "multistart_consensus", "caiman_cnmf", "caiman_cnmfe",
    ):
        methods.append({"method_id": method_id, "enabled": True,
                        "track": "anchor" if method_id == "raw_direct_reference" else "controlled_input",
                        "configurations": [{}]})
    return {
        "schema_version": 1, "experiment_id": "test", "output_root": "out",
        "scientific_config": "science.json", "source_video": "movie.npy",
        "source_tiff": "movie.tif", "labels_tsv": "labels.tsv",
        "caiman_python": "/tmp/caiman/python",
        "frames": {"review_start_ui": 1800, "review_end_ui": 2359,
                   "quiet_start_ui": 1800, "quiet_end_ui": 1899,
                   "frame_period_ms": 20.0},
        "methods": methods,
        "design": {"development_fixture_count": 72, "confirmation_fixture_count": 312,
                   "semi_synthetic_fixture_count": 135, "confidence_perturbations": 2,
                   "fixed_candidate_budgets": [10,20,40,58,80,100],
                   "manual_review_rows": 320},
        "gates": {"closure_tolerance": 1e-6, "maximum_false_resolution_count": 0,
                  "minimum_identifiable_coverage": 0.8, "minimum_converged_fraction": 0.95,
                  "equivalence_margin": 0.01, "minimum_peak_retention": 0.8,
                  "minimum_area_retention": 0.8, "minimum_waveform_correlation": 0.8,
                  "maximum_timing_error_frames": 2},
        "resources": {"general_cpu_workers": 8, "maximum_caiman_processes": 12,
                      "worker_threads": 1, "gpu_device": "cuda:0",
                      "gpu_allocation_cap_mib": 8192, "minimum_free_gpu_mib": 4096,
                      "rss_soft_cap_mib": 49152, "rss_hard_stop_mib": 57344,
                      "minimum_free_disk_mib": 204800, "maximum_output_mib": 81920,
                      "heartbeat_seconds": 30, "gpu_warning_c": 78, "gpu_stop_c": 83},
    }


def test_conclusive_config_is_strict_and_bounded(tmp_path: Path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(_manifest(tmp_path)))
    config = ConclusiveBatchConfig.load(path)
    assert config.enabled_configuration_count() == 12
    broken = _manifest(tmp_path)
    broken["resources"]["general_cpu_workers"] = 20
    path.write_text(json.dumps(broken))
    with pytest.raises(ConclusiveBatchConfigError):
        ConclusiveBatchConfig.load(path)


def test_prepare_requires_explicit_run_authorization(tmp_path: Path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(_manifest(tmp_path)))
    config = ConclusiveBatchConfig.load(path)
    with pytest.raises(RuntimeError, match="explicit run authorization"):
        prepare(config)


def test_stage4_gpu_limit_is_inert_for_cpu_device(tmp_path: Path):
    path = tmp_path / "manifest.json"
    manifest = _manifest(tmp_path)
    manifest["resources"]["gpu_device"] = "cpu"
    path.write_text(json.dumps(manifest))
    config = ConclusiveBatchConfig.load(path)
    assert _configure_gpu_allocation_limit(config) == {
        "device": "cpu",
        "enabled": False,
        "reason": "non_cuda_device",
    }
