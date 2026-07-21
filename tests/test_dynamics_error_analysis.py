import numpy as np

from neurobench.dynamics.error_analysis import promote_structured_error_metrics, structured_prediction_error_metrics


def test_structured_prediction_error_metrics_separates_active_and_high_change_cells():
    targets = np.asarray(
        [
            [[[[0.0, 0.9], [0.1, 0.8]]]],
            [[[[0.0, 0.2], [0.7, 0.9]]]],
            [[[[0.1, 0.1], [0.1, 0.1]]]],
        ],
        dtype=np.float32,
    )[:, 0]
    last = np.asarray(
        [
            [[0.0, 0.4], [0.1, 0.3]],
            [[0.0, 0.2], [0.2, 0.2]],
            [[0.1, 0.1], [0.1, 0.1]],
        ],
        dtype=np.float32,
    )[:, None]
    pred_diff = np.full_like(targets, 0.1)
    persistence_diff = last - targets
    video_ids = np.asarray(["train", "test", "test"])
    splits = {"train_video_ids": ["train"], "test_video_ids": ["test"]}

    structured = structured_prediction_error_metrics(
        pred_diff=pred_diff,
        persistence_diff=persistence_diff,
        targets=targets,
        last_frames=last,
        video_ids=video_ids,
        splits=splits,
        active_percentile=50,
        top_activity_percent=25,
        high_change_percentile=50,
    )
    metrics = {}
    promote_structured_error_metrics(metrics, structured)

    assert structured["test"]["window_count"] == 2
    assert structured["test"]["active_cell_count"] > 0
    assert structured["test"]["active_cell_improvement_over_persistence_mse"] is not None
    assert structured["thresholds"]["active_threshold"] >= 0.0
    assert "test_active_cell_improvement_over_persistence_mse" in metrics
    assert "test_high_change_improvement_over_persistence_mse" in metrics


def test_prediction_split_metrics_exports_per_video_improvement():
    from neurobench.dynamics.train import _prediction_split_metrics

    decoded_diff = np.asarray(
        [
            [[[0.1, 0.1]]],
            [[[0.2, 0.2]]],
            [[[0.4, 0.4]]],
        ],
        dtype=np.float32,
    )
    persistence_diff = np.asarray(
        [
            [[[0.3, 0.3]]],
            [[[0.1, 0.1]]],
            [[[0.5, 0.5]]],
        ],
        dtype=np.float32,
    )
    latent_diff = np.zeros((3, 2), dtype=np.float32)
    latent_raw_diff = np.zeros((3, 2), dtype=np.float32)
    video_ids = np.asarray(["train_v", "test_a", "test_b"])
    splits = {"train_video_ids": ["train_v"], "test_video_ids": ["test_a", "test_b"]}

    metrics = _prediction_split_metrics(decoded_diff, latent_diff, latent_raw_diff, persistence_diff, video_ids, splits)

    assert metrics["test"]["window_count"] == 2
    assert sorted(metrics["test"]["per_video"]) == ["test_a", "test_b"]
    assert metrics["test"]["per_video"]["test_a"]["window_count"] == 1
    assert metrics["test"]["per_video"]["test_a"]["improvement_over_persistence_mse"] < 0
    assert metrics["test"]["per_video"]["test_b"]["improvement_over_persistence_mse"] > 0
    assert metrics["val"]["per_video"] == {}
