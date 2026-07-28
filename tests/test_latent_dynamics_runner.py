from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from neurobench.algorithms.latent_dynamics import kalman_filter_ar1
from neurobench.experiments.latent_dynamics import LatentDynamicsConfig, preflight
from neurobench.experiments.latent_dynamics.runner import raw_direct_pool, run
from test_latent_dynamics_config import write_fixture


def test_column_chunks_equal_full_vectorized_filter():
    rng = np.random.default_rng(8)
    observations = rng.normal(size=(30, 11))
    from neurobench.algorithms.latent_dynamics import stable_ar1_from_decay
    model = stable_ar1_from_decay(20, 100, 0.2, 0.7)
    expected = kalman_filter_ar1(observations, model).filter_mean
    actual = np.concatenate([
        kalman_filter_ar1(observations[:, start:start + 3], model).filter_mean
        for start in range(0, observations.shape[1], 3)
    ], axis=1)
    np.testing.assert_array_equal(actual, expected)


def test_raw_direct_anchor_is_exact_and_unknown_semantics_are_explicit():
    values = np.arange(30, dtype=np.float32).reshape(5, 6)
    from neurobench.experiments.learnable_contrast.direct_tuning import _direct_map
    np.testing.assert_array_equal(raw_direct_pool(values), _direct_map(values))


def test_tiny_run_emits_complete_schema_and_no_partials(tmp_path):
    manifest, output = write_fixture(tmp_path)
    config = LatentDynamicsConfig.load(manifest)
    artifact = tmp_path / "preflight"
    preflight(config, artifact_dir=artifact)
    result = run(config, preflight_dir=artifact)
    assert result["status"] == "complete" and result["raw_direct_anchor_passed"]
    required = (
        "config.resolved.json", "preflight.json", "run_state.json", "progress.jsonl",
        "resource_summary.json", "fit/sample_manifest.json", "fit/candidate_models.tsv",
        "fit/selected_model.json", "fit/parameter_history.tsv", "fit/predictive_likelihood.tsv",
        "fit/stability.json", "noise/quiet_noise_summary.json", "noise/quiet_center.npy",
        "noise/quiet_scale.npy", "states/filter_mean.npy", "states/smoother_mean.npy",
        "states/filter_variance_by_time.npy", "states/smoother_variance_by_time.npy",
        "features/feature_manifest.json", "features/pooled_candidate_maps.npz",
        "diagnostics/residual_summary.json", "diagnostics/innovation_summary.json",
        "diagnostics/quiet_autocorrelation.tsv", "diagnostics/event_preservation.tsv",
        "diagnostics/perturbation_stability.tsv", "evaluation/metrics.json",
        "evaluation/lane_summary.tsv", "evaluation/known_matches.tsv",
        "evaluation/unmatched_candidates.tsv", "report.md",
    )
    assert all((output / relative).is_file() for relative in required)
    assert np.load(output / "states/filter_mean.npy").shape == (120, 40, 41)
    state_manifest = json.loads((output / "states/state_manifest.json").read_text())
    assert state_manifest["filter_mean"]["causal"] is True
    assert state_manifest["smoother_mean"]["causal"] is False
    sample_manifest = json.loads((output / "fit/sample_manifest.json").read_text())
    assert sample_manifest["labels_available_to_fit"] is False
    metrics = json.loads((output / "evaluation/metrics.json").read_text())
    assert metrics["sparse_positive_semantics"] == "unmatched_candidates_are_unknown_not_negative"
    assert len(metrics["lanes"]) == 13 and len(metrics["outer_folds"]) == 52
    assert metrics["raw_direct_anchor"]["passed"] is True
    assert 0 <= metrics["raw_direct"]["mean_recall"] <= 1
    assert sum(row["labels"] for row in metrics["outer_folds"] if row["lane"] == "raw_direct") == 4
    assert not list(output.rglob("*.partial"))
    with pytest.raises(FileExistsError):
        run(config, preflight_dir=artifact)


def test_run_rejects_nonmatching_preflight(tmp_path):
    manifest, _ = write_fixture(tmp_path)
    config = LatentDynamicsConfig.load(manifest)
    artifact = tmp_path / "preflight"
    preflight(config, artifact_dir=artifact)
    payload = json.loads((artifact / "config.resolved.json").read_text())
    payload["experiment_id"] = "tampered"
    (artifact / "config.resolved.json").write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="identical resolved config"):
        run(config, preflight_dir=artifact)


def test_failure_removes_partial_files(tmp_path, monkeypatch):
    manifest, output = write_fixture(tmp_path)
    config = LatentDynamicsConfig.load(manifest)
    artifact = tmp_path / "preflight"
    preflight(config, artifact_dir=artifact)
    import neurobench.experiments.latent_dynamics.runner as runner
    monkeypatch.setattr(runner, "_complete_memmap", lambda *args: (_ for _ in ()).throw(RuntimeError("injected")))
    with pytest.raises(RuntimeError, match="injected"):
        runner.run(config, preflight_dir=artifact)
    assert not list(output.rglob("*.partial"))
    assert json.loads((output / "run_state.json").read_text())["status"] == "failed"


def test_selected_review_tiffs_are_materialized(tmp_path):
    manifest, output = write_fixture(tmp_path)
    payload = json.loads(manifest.read_text())
    payload["features"]["write_selected_tiffs"] = True
    manifest.write_text(json.dumps(payload))
    config = LatentDynamicsConfig.load(manifest)
    artifact = tmp_path / "preflight"
    preflight(config, artifact_dir=artifact)
    run(config, preflight_dir=artifact)
    import tifffile
    for name in ("filter_mean.tif", "smoother_mean.tif"):
        path = output / "features" / "selected_review_tiffs" / name
        assert path.is_file() and tifffile.imread(path).shape == (120, 40, 41)
