import json
from pathlib import Path

from neurobench.dynamics.planner import build_adaptive_sweep_plan


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _spec(exp_id: str, kind: str, dataset: str, **params):
    return {"experiment_id": exp_id, "kind": kind, "dataset_key": dataset, "seed": params.pop("seed", 7), "params": params}


def test_build_adaptive_sweep_plan_keeps_controls_and_defers_unstable_pixel_models(tmp_path):
    sweep = tmp_path / "sweep"
    specs = [
        _spec("base_h2_persist", "array_baseline", "w8_s1_h2", baseline_name="persistence", hyperparameter_summary="baseline=persistence"),
        _spec("base_h2_moving", "array_baseline", "w8_s1_h2", baseline_name="moving_average", hyperparameter_summary="baseline=moving_average"),
        _spec("base_h5_persist", "array_baseline", "w8_s1_h5", baseline_name="persistence", hyperparameter_summary="baseline=persistence"),
        _spec("base_h5_moving", "array_baseline", "w8_s1_h5", baseline_name="moving_average", hyperparameter_summary="baseline=moving_average"),
        _spec("linear_h5_delta", "linear_latent", "w8_s1_h5", prediction_target="delta", batch_size=256, hyperparameter_summary="linear delta"),
        _spec("gru_h5_delta_hd64_lr3e5", "latent_gru", "w8_s1_h5", prediction_target="delta", hidden_dim=64, learning_rate=3e-5, batch_size=4, hyperparameter_summary="gru delta hd64 lr3e-5"),
        _spec("gru_h5_abs_hd64_lr3e5", "latent_gru", "w8_s1_h5", prediction_target="absolute", hidden_dim=64, learning_rate=3e-5, batch_size=4, hyperparameter_summary="gru abs hd64 lr3e-5"),
        _spec("gru_h5_delta_hd256_lr1e3", "latent_gru", "w8_s1_h5", prediction_target="delta", hidden_dim=256, learning_rate=1e-3, batch_size=4, hyperparameter_summary="incumbent gru"),
        _spec("xfmr_h5_delta_md64", "latent_transformer", "w8_s1_h5", prediction_target="delta", model_dim=64, num_heads=2, num_layers=2, learning_rate=3e-5, batch_size=4, hyperparameter_summary="xfmr delta"),
        _spec("convgru_h5", "convgru_pixel", "w8_s1_h5", hidden_channels=16, num_layers=1, learning_rate=1e-4, batch_size=4, hyperparameter_summary="convgru small"),
    ]
    _write_json(
        sweep / "sweep_manifest.json",
        {
            "profile": "grid128_sequence_1day",
            "experiment_count": len(specs),
            "datasets": {"w8_s1_h2": {}, "w8_s1_h5": {}},
            "experiments": specs,
        },
    )
    (sweep / "sweep_progress.jsonl").write_text(
        json.dumps({"index": 6, "experiment_count": len(specs), "experiment_id": "gru_h5_delta_hd64_lr3e5", "status": "completed"}) + "\n",
        encoding="utf-8",
    )
    (sweep / "sweep_progress_batch64_oom.jsonl").write_text(
        json.dumps({"index": 10, "experiment_id": "convgru_h5", "kind": "convgru_pixel", "status": "failed", "error": "CUDA out of memory"}) + "\n",
        encoding="utf-8",
    )
    comparison = tmp_path / "comparison"
    _write_json(
        comparison / "results_intelligence.json",
        {
            "best_by_family": {
                "test": {
                    "array_baseline": {"experiment_id": "base_h5_moving", "dataset_key": "w8_s1_h5", "improvement_over_persistence_mse": 0.02, "hyperparameter_summary": "baseline=moving_average"},
                    "latent_gru": {"experiment_id": "gru_h5_delta_hd256_lr1e3", "dataset_key": "w8_s1_h5", "improvement_over_persistence_mse": 0.01, "hyperparameter_summary": "incumbent gru"},
                    "latent_transformer": {"experiment_id": "xfmr_h5_delta_md64", "dataset_key": "w8_s1_h5", "improvement_over_persistence_mse": 0.012, "hyperparameter_summary": "xfmr delta"},
                }
            },
            "family_comparison": {
                "test": {
                    "array_baseline": {"count": 2, "positive_count": 1, "best": {"improvement_over_persistence_mse": 0.02}},
                    "latent_gru": {"count": 3, "positive_count": 2, "best": {"improvement_over_persistence_mse": 0.01}},
                    "latent_transformer": {"count": 1, "positive_count": 1, "best": {"improvement_over_persistence_mse": 0.012}},
                }
            },
            "failure_summary": {"failure_count": 1, "by_kind": {"convgru_pixel": 1}},
        },
    )

    summary = build_adaptive_sweep_plan(sweep_dir=sweep, comparison_dir=comparison, out_dir=tmp_path / "plan", max_experiments=16)
    manifest = json.loads(Path(summary["manifest_path"]).read_text(encoding="utf-8"))
    selected_ids = {item["experiment_id"] for item in manifest["experiments"]}

    assert summary["planned_experiment_count"] < len(specs)
    assert "base_h2_persist" in selected_ids
    assert "base_h5_moving" in selected_ids
    assert "gru_h5_delta_hd64_lr3e5" in selected_ids
    assert "xfmr_h5_delta_md64" in selected_ids
    assert "gru_h5_abs_hd64_lr3e5" not in selected_ids
    assert "convgru_h5" not in selected_ids
    assert summary["deferred_counts"] == {"convgru_pixel": 1}
    assert manifest["progress_summary"]["archived_failure_count"] == 1
    markdown = Path(summary["markdown_path"]).read_text(encoding="utf-8")
    assert "Adaptive Next Sweep Plan" in markdown
    assert "Deferred Families" in markdown
    assert "conservative latent-search neighborhood" in markdown
    assert "--manifest" in summary["suggested_command"]
    assert "next_sweep_manifest.json" in summary["suggested_command"]
    assert "manifest-aware overnight sweep runner" in markdown
