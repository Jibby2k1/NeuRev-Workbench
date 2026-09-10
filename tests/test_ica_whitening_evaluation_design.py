import json
from pathlib import Path

import pytest

from neurobench.experiments.ica_whitening_evaluation import (
    ICAWhiteningConfig,
    ICAWhiteningConfigError,
    build_design,
    design_digest,
    enumerate_cells,
)


def _manifest(tmp_path: Path) -> Path:
    payload = {
        "schema_version": 1,
        "experiment_id": "tiny_ica_whitening",
        "source_video": "video.npy",
        "labels_tsv": "labels.tsv",
        "label_summary": "labels.json",
        "output_dir": "out",
        "frames": {
            "review_start_ui": 1, "review_end_ui": 40,
            "quiet_start_ui": 1, "quiet_end_ui": 10,
            "frame_period_ms": 20.0,
        },
        "design": {
            "master_seed": 17,
            "sobol_points_per_cell": 4,
            "families": ["temporal", "spatial", "joint_spatiotemporal"],
            "objectives": ["fastica_logcosh"],
            "whitening_geometries": ["none", "joint_spatiotemporal"],
            "covariance_scopes": ["global_quiet"],
            "causalities": ["centered", "causal"],
            "ranks": [2, 4],
            "spatial_widths_px": [3, 5],
            "temporal_widths_frames": [3, 5],
            "screen_seeds": [7, 13],
            "confirmation_seeds": [7, 13, 19],
            "maximum_joint_dimension": 128,
        },
        "continuous_bounds": {
            "whitening_exponent": [0.0, 1.0],
            "shrinkage": [0.0001, 0.5],
            "eigen_floor_ratio": [0.000001, 0.01],
            "raw_preserving_blend": [0.0, 1.0],
            "objective_scale": [0.5, 2.0],
        },
        "evaluation": {
            "fixed_candidates_per_burst": 58, "nms_distance_px": 6,
            "match_radius_px": 6, "temporal_pool": "lme0.25",
        },
        "gates": {
            "maximum_condition_number": 1e6,
            "minimum_effective_rank_fraction": 0.25,
            "minimum_converged_fraction": 0.8,
            "minimum_truth_source_correlation": 0.7,
            "maximum_truth_crosstalk": 0.25,
            "minimum_trace_preservation": 0.7,
            "minimum_macro_kpr_delta": 0.0,
        },
        "resources": {
            "device": "cpu", "cpu_threads": 2, "max_ram_mib": 1024,
            "min_free_disk_mib": 100, "max_output_mib": 100,
            "maximum_fit_samples": 256, "application_chunk_size": 64,
        },
        "scientific_audit": {"enabled": True, "opt_out_reason": None},
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_design_is_deterministic_unique_and_conditioned(tmp_path: Path) -> None:
    config = ICAWhiteningConfig.from_json(_manifest(tmp_path))
    cells = enumerate_cells(config)
    assert cells
    assert all(cell.rank <= cell.feature_dimension for cell in cells)
    assert all(
        cell.feature_dimension <= 128
        for cell in cells if cell.family == "joint_spatiotemporal"
    )
    assert all(
        cell.spatial_width_px**2 * cell.temporal_width_frames <= 128
        for cell in cells if cell.whitening_geometry == "joint_spatiotemporal"
    )
    first = build_design(config)
    second = build_design(config)
    assert first == second
    assert design_digest(first) == design_digest(second)
    assert len({row["fit_id"] for row in first}) == len(first)
    assert {row["point_kind"] for row in first} == {"anchor", "sobol"}
    joint_rows = [row for row in first if row["whitening_geometry"] == "joint_spatiotemporal"]
    assert {row["joint_exponent"] for row in joint_rows} >= {0.0, 1.0}


def test_invalid_non_power_of_two_sobol_size_fails(tmp_path: Path) -> None:
    path = _manifest(tmp_path)
    payload = json.loads(path.read_text())
    payload["design"]["sobol_points_per_cell"] = 3
    path.write_text(json.dumps(payload))
    with pytest.raises(ICAWhiteningConfigError, match="power of two"):
        ICAWhiteningConfig.from_json(path)
