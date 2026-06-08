import json
from pathlib import Path

from neurobench.dynamics.visualize_sweep import generate_sweep_visuals


def test_generate_sweep_visuals_outputs_summary_and_svgs(tmp_path):
    exp = tmp_path / "exp"
    metrics_dir = exp / "residual_pixel_mse"
    metrics_dir.mkdir(parents=True)
    (exp / "experiment_config.json").write_text(json.dumps({
        "experiment_id": "residual_demo",
        "kind": "residual_pixel",
        "dataset_key": "w8_s3_h10",
        "seed": 7,
        "params": {"hidden_dim": 64, "learning_rate": 0.0001, "residual_scale": 0.1},
    }), encoding="utf-8")
    tsv = tmp_path / "sweep_summary.tsv"
    tsv.write_text(
        "rank\texperiment_id\tkind\tdataset_key\tseed\tval_decoded_prediction_mse\tval_persistence_mse\tval_improvement_over_persistence_mse\ttest_decoded_prediction_mse\ttest_persistence_mse\ttest_improvement_over_persistence_mse\tmetrics_path\n"
        f"1\tresidual_demo\tresidual_pixel\tw8_s3_h10\t7\t0.1\t0.2\t0.1\t0.09\t0.1\t0.01\t{metrics_dir / 'concept_metrics.json'}\n",
        encoding="utf-8",
    )

    summary = generate_sweep_visuals(summary_tsv=tsv, out_dir=tmp_path / "visuals", dashboard_prefix="grid")

    assert summary["experiment_count"] == 1
    assert summary["positive_validation_and_test_count"] == 1
    assert (tmp_path / "visuals" / "charts" / "top_validation_improvement.svg").is_file()
    assert (tmp_path / "visuals" / "sweep_visual_summary.json").is_file()
