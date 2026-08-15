import json
from pathlib import Path

import numpy as np
import pytest

from neurobench.experiments.msln_msica.local_whitening_program import (
    CONTEXT_IDS,
    _frame_bootstrap_identity_errors,
    compare_audits,
    _resume_completed_stage,
    load_config,
    preflight,
    run_full,
    run_synthetic,
)


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "spon_ca_burst_local_whitening_v1.example.json"


def _config_at(tmp_path: Path, *, with_source: bool = False) -> Path:
    payload = json.loads(EXAMPLE.read_text())
    payload["outputs"]["root_dir"] = str(tmp_path / "output")
    if with_source:
        movie = tmp_path / "movie.npy"
        mapped = np.lib.format.open_memmap(movie, mode="w+", dtype=np.float32, shape=(2360, 64, 64))
        mapped[1899] = 1; mapped.flush(); del mapped
        labels = tmp_path / "labels.tsv"
        labels.write_text(
            "burst_id\tstart_frame_ui\tend_frame_ui\tstart_frame_zero\tstop_frame_zero_exclusive\tpoint_index\troi_identity\tx_px\ty_px\trecurrence_count\n"
            "1\t2003\t2026\t2002\t2026\t1\troi_001\t32\t32\t1\n"
        )
        payload["source"]["movie_path"] = str(movie)
        payload["source"]["labels_path"] = str(labels)
    path = tmp_path / "config.json"; path.write_text(json.dumps(payload)); return path


def test_manifest_is_exact_and_default_on() -> None:
    config = load_config(EXAMPLE)
    assert tuple(config["feature_bank"]["context_ids"]) == CONTEXT_IDS
    assert config["scientific_audit"] == {"enabled": True}
    assert config["compute"] == {
        "device": "cpu", "cpu_threads": 4, "workers": 1, "frame_chunk": 8,
        "maximum_peak_ram_gb": 16, "maximum_peak_vram_gb": 4,
    }


def test_frame_bootstrap_is_deterministic_and_detects_whiteness_gap() -> None:
    rng = np.random.default_rng(12)
    correlated = rng.multivariate_normal([0, 0], [[1, .9], [.9, 1]], size=(8, 400)).astype(np.float32)
    whitened = correlated @ np.asarray([[1.0, -0.75], [0.0, .66]], dtype=np.float32)
    indices = np.arange(8)
    left = _frame_bootstrap_identity_errors(correlated[:, :, None, :], indices, seed=7, repetitions=24)
    right = _frame_bootstrap_identity_errors(correlated[:, :, None, :], indices, seed=7, repetitions=24)
    transformed = _frame_bootstrap_identity_errors(whitened[:, :, None, :], indices, seed=7, repetitions=24)
    np.testing.assert_array_equal(left, right)
    assert float(np.median(left - transformed)) > 0


def test_manifest_rejects_unknown_nested_key(tmp_path: Path) -> None:
    payload = json.loads(EXAMPLE.read_text()); payload["tiles"]["surprise"] = 1
    path = tmp_path / "bad.json"; path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="tiles keys differ"):
        load_config(path)


def test_synthetic_stage_never_requires_real_source_and_writes_indices(tmp_path: Path) -> None:
    config = load_config(_config_at(tmp_path))
    summary = run_synthetic(config)
    root = Path(config["outputs"]["root_dir"])
    assert summary["decision"] in {"advance", "stop"}
    fixtures = {row["fixture_id"]: row for row in summary["scientific_fixture_metrics"]}
    assert fixtures["global_off_diagonal"]["full_improvement"] > 0
    assert abs(fixtures["iid_gaussian"]["full_improvement"]) < 0.05
    assert fixtures["covariance_shift_train_to_test"]["reported_covariance_shift"] is True
    assert fixtures["quiet_fit_contamination_0p05"]["self_whitening_ratio"] < 1
    assert fixtures["low_eigenvalue_shrinkage"]["resolved"] is True
    assert (root / "synthetic" / "metrics.csv").is_file()
    assert (root / "synthetic" / "scientific_fixture_metrics.csv").is_file()
    assert summary["fit_artifact_count"] > 0
    assert (root / "synthetic" / "fits" / "fit_index.csv").is_file()
    assert json.loads((root / "status.json").read_text())["stage"] == "synthetic"
    for name in ("summary.json", "llm_context.json", "artifact_index.json", "validation.json", "REPORT.md"):
        assert (root / name).is_file()


def test_preflight_fingerprints_source_and_refuses_collision(tmp_path: Path) -> None:
    config = load_config(_config_at(tmp_path, with_source=True))
    result = preflight(config)
    root = Path(config["outputs"]["root_dir"])
    assert result["source_read_only"] is True
    assert result["quiet_partition_frame_counts"] == {"fit": 50, "calibration": 25, "holdout": 25}
    assert (root / "preflight" / "label_projection_overlay.png").is_file()
    resumed = _resume_completed_stage(config, "preflight")
    assert resumed is not None and resumed["resumed_without_refit"] is True
    with pytest.raises(FileExistsError, match="refuses existing output root"):
        preflight(config)


def test_full_spon_run_requires_explicit_authorization(tmp_path: Path) -> None:
    config = load_config(_config_at(tmp_path))
    with pytest.raises(PermissionError, match="authorize-full-spon"):
        run_full(config, authorize_full_spon=False)


def test_authorization_cannot_bypass_failed_representation_gate(tmp_path: Path) -> None:
    config = load_config(_config_at(tmp_path, with_source=True))
    preflight(config)
    root = Path(config["outputs"]["root_dir"])
    (root / "summary.json").write_text(json.dumps({
        "stage": "covariance-audit", "outcome": "global_full_only", "decision": "stop",
    }))
    with pytest.raises(RuntimeError, match="did not pass the local-full G2"):
        run_full(config, authorize_full_spon=True)


def test_compare_audits_is_collision_safe_and_ranks_lanes(tmp_path: Path) -> None:
    cpu_root = tmp_path / "cpu"; cuda_root = tmp_path / "cuda"
    cpu_root.mkdir(); cuda_root.mkdir()
    rows = []
    for block in ("holdout_1", "holdout_2"):
        for index, lane in enumerate(("identity", "global_diagonal", "global_full_zca", "local_diagonal", "local_full_zca")):
            value = {"global_full_zca": .1, "local_full_zca": .2}.get(lane, 1.0 + index / 10)
            rows.append({
                "quiet_block": block, "lane": lane,
                "heldout_identity_error": value,
                "heldout_max_abs_correlation": value,
                "heldout_off_diagonal_energy": value,
                "tile_boundary_ratio": 1.0,
                "unresolved_pixel_fraction": .5 if lane == "local_full_zca" else 0.0,
            })
    for root, delta in ((cpu_root, 0.0), (cuda_root, 1e-7)):
        adjusted = [{**row, "heldout_identity_error": row["heldout_identity_error"] + delta} for row in rows]
        (root / "summary.json").write_text(json.dumps({
            "stage": "covariance-audit", "outcome": "global_full_only",
            "gate_g2": False, "rows": adjusted,
        }))
    output = tmp_path / "comparison"
    summary = compare_audits(cpu_root, cuda_root, output)
    assert summary["outcomes_match"] is True
    assert summary["lane_ranking_by_mean_heldout_identity_error"][0] == "global_full_zca"
    assert (output / "backend_parity.csv").is_file()
    with pytest.raises(FileExistsError, match="refuses existing"):
        compare_audits(cpu_root, cuda_root, output)
