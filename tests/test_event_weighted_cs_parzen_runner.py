import json
from pathlib import Path

import numpy as np
import yaml

from neurobench.experiments.event_weighted_cs_parzen import (
    EventWeightedCSParzenConfig,
    preflight,
    run,
)


ROOT = Path(__file__).resolve().parents[1]


def test_generated_one_fold_two_alpha_sweep_writes_complete_artifacts(tmp_path):
    rng = np.random.default_rng(71)
    movie = rng.normal(100, 2, size=(120, 12, 14)).astype(np.float32)
    intervals = {1: (70, 74), 2: (82, 86), 3: (94, 98), 4: (108, 112)}
    points = {1: (5, 5), 2: (8, 4), 3: (6, 8), 4: (10, 7)}
    for event, (start, stop) in intervals.items():
        x, y = points[event]
        movie[start - 1 : stop, y - 1 : y + 2, x - 1 : x + 2] += np.linspace(
            0, 35, stop - start + 1
        )[:, None, None]
    movie = np.clip(movie, 0, 4095).astype(np.uint16)
    np.save(tmp_path / "movie.npy", movie)

    header = (
        "burst_id\tstart_frame_ui\tend_frame_ui\tstart_frame_zero\t"
        "stop_frame_zero_exclusive\tpoint_index\troi_identity\tx_px\ty_px\t"
        "recurrence_count\n"
    )
    rows = []
    for event, (start, stop) in intervals.items():
        x, y = points[event]
        rows.append(
            f"{event}\t{start}\t{stop}\t{start - 1}\t{stop}\t1\tr{event}\t"
            f"{x}\t{y}\t1\n"
        )
    (tmp_path / "labels.tsv").write_text(header + "".join(rows), encoding="utf-8")

    raw = yaml.safe_load(
        (ROOT / "examples/spon_ca_burst_event_weighted_cs_parzen.smoke.yaml").read_text()
    )
    raw["experiment_id"] = "generated_event_weighted_smoke"
    raw["source"].update(
        {
            "movie_path": "movie.npy",
            "labels_path": "labels.tsv",
            "baseline_evidence_dir": None,
            "review_interval_ui": [1, 120],
            "quiet_interval_ui": [1, 60],
            "burst_intervals_ui": intervals,
        }
    )
    raw["outputs"]["root_dir"] = "run"
    raw["outputs"]["representative_frames_ui"] = [20, 70]
    raw["sampling"].update(
        {
            "screen_samples": 16,
            "confirmation_samples": 24,
            "event_screen_max_samples_per_event": 2,
            "event_confirmation_max_samples_per_event": 4,
            "heldout_guard_frames": 2,
        }
    )
    raw["angle_search"].update(
        {
            "coarse_step_degrees": 30.0,
            "refine_half_width_degrees": 1.0,
            "refine_step_degrees": 1.0,
        }
    )
    raw["parzen"]["kernel_block_rows"] = 8
    raw["evaluation"]["nms_distance_px"] = 2
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    config = EventWeightedCSParzenConfig.load(config_path)

    audit = preflight(config, artifact_dir=tmp_path / "preflight")
    assert audit["ready"]
    assert (tmp_path / "preflight" / "label_projection_overlay.png").is_file()
    summary = run(config, preflight_dir=tmp_path / "preflight")
    assert summary["status"] == "complete"
    assert summary["fit_count"] == 7
    assert summary["spatial_extension_launched"] is False
    assert (config.outputs.root_dir / "manifest.json").is_file()
    assert (config.outputs.root_dir / "fit_metrics.csv").is_file()
    video_manifest = json.loads(
        (config.outputs.root_dir / "selected_video_manifest.json").read_text()
    )
    assert len(video_manifest["rendered"]) == 2
    assert all(
        (config.outputs.root_dir / row["path"]).is_file()
        for row in video_manifest["rendered"]
    )
    metrics = json.loads((config.outputs.root_dir / "fit_metrics.json").read_text())
    assert {row["natural_evaluation_sample_count"] for row in metrics} == {24}
    assert all(row["precision_identified"] is False for row in metrics)
    assert not list(config.outputs.root_dir.rglob("*.partial"))
    expected_figures = {
        "angle_shift_vs_alpha.png",
        "derivative_cosine_vs_alpha.png",
        "train_and_holdout_objective_vs_alpha.png",
        "weight_ess_vs_alpha.png",
        "known_label_recall_vs_candidate_count.png",
        "fold_angle_stability.png",
        "frame_vs_roi_weighting_comparison.png",
        "representative_weighted_outputs.png",
    }
    assert expected_figures == {
        path.name for path in (config.outputs.root_dir / "figures").glob("*.png")
    }


def test_evidence_backed_gate_a_uses_immutable_full_baseline_contract(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from neurobench.algorithms.pairwise_separation import SeparationFit
    from neurobench.experiments.event_weighted_cs_parzen import runner

    evidence = tmp_path / "evidence"
    (evidence / "source_evidence").mkdir(parents=True)
    expected_objective = 0.01721870304031795
    (evidence / "source_evidence" / "fit.json").write_text(
        json.dumps(
            {
                "objective_value": expected_objective,
                "iterations": 58,
                "diagnostics": {"selected_angle_degrees": 4.0},
            }
        )
    )
    (evidence / "preliminary_metrics.json").write_text(
        json.dumps(
            {
                "direction": {
                    "absolute_cosine_to_derivative": 0.999999917348767
                }
            }
        )
    )
    captured = {}

    def fake_indices(*args, sample_count, **kwargs):
        captured["sample_count"] = sample_count
        return tuple(range(sample_count))

    monkeypatch.setattr(runner, "sample_natural_indices", fake_indices)
    monkeypatch.setattr(
        runner,
        "extract_pair_samples",
        lambda *args, **kwargs: np.zeros((4096, 2), dtype=float),
    )
    fake_whitening = SimpleNamespace()
    monkeypatch.setattr(
        runner,
        "fit_weighted_whitening_2d",
        lambda *args, **kwargs: (np.zeros((2, 4096)), fake_whitening),
    )

    def fake_fit(screen, confirmation, **kwargs):
        captured.update(kwargs)
        captured["screen_samples"] = screen.shape[1]
        captured["confirmation_samples"] = confirmation.shape[1]
        return SeparationFit(
            method_id="cs_parzen_ica",
            demixing=np.eye(2),
            mixing=np.eye(2),
            objective=expected_objective,
            converged=True,
            iterations=58,
            activity_component=None,
            activity_sign=None,
            diagnostics={"selected_angle_degrees": 4.0},
        )

    monkeypatch.setattr(runner, "fit_cs_parzen_ica", fake_fit)
    monkeypatch.setattr(
        runner,
        "canonicalize_fit",
        lambda *args: SimpleNamespace(cosine_to_derivative=0.999999917348767),
    )
    config = SimpleNamespace(
        source=SimpleNamespace(
            review_interval_ui=(1800, 2359),
            baseline_evidence_dir=evidence,
        ),
        sampling=SimpleNamespace(
            confirmation_samples=32,
            screen_samples=16,
            seed=20260727,
        ),
        whitening=SimpleNamespace(eigenvalue_floor_ratio=1e-6),
        parzen=SimpleNamespace(bandwidth=0.35, kernel_block_rows=16),
        angle_search=SimpleNamespace(
            coarse_step_degrees=30.0,
            refine_half_width_degrees=1.0,
            refine_step_degrees=1.0,
        ),
    )
    result = runner._baseline_parity(
        np.zeros((2, 1, 1), dtype=np.float32),
        np.ones((1, 1), dtype=bool),
        config,
    )
    assert result["status"] == "passed"
    assert captured["sample_count"] == 4096
    assert captured["screen_samples"] == 1024
    assert captured["confirmation_samples"] == 4096
    assert captured["block_rows"] == 256
    assert captured["screen_step_degrees"] == 3.0
    assert captured["refine_half_width_degrees"] == 3.0
    assert captured["refine_step_degrees"] == 0.25
