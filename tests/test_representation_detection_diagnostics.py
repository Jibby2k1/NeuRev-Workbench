import numpy as np

from tools.build_representation_detection_diagnostics import (
    FN_COLOR,
    TP_COLOR,
    UNKNOWN_COLOR,
    classify_burst,
    render_overlay,
)


def _label(identity: str, x: float, y: float) -> dict:
    return {
        "burst_id": 1, "roi_identity": identity, "x_px": x, "y_px": y,
        "start_frame_ui": 10, "end_frame_ui": 20, "recurrence_count": 1,
    }


def test_sparse_classification_keeps_unmatched_candidates_unknown() -> None:
    labels = [_label("roi_001", 10, 10), _label("roi_002", 30, 30)]
    result = classify_burst([(4.0, 11, 10), (3.0, 50, 50)], labels, radius=6)
    assert result["matched"] == 1
    assert result["true_positive_labels"][0]["roi_identity"] == "roi_001"
    assert result["false_negative_labels"][0]["roi_identity"] == "roi_002"
    assert result["unmatched_candidates_unknown"] == [
        {"score": 3.0, "x_px": 50, "y_px": 50}
    ]


def test_overlay_uses_distinct_colors_and_legend_strip() -> None:
    labels = [_label("roi_001", 10, 10), _label("roi_002", 30, 30)]
    classified = classify_burst([(4.0, 11, 10), (3.0, 50, 50)], labels, radius=6)
    image = render_overlay(
        np.full((64, 64), 100, dtype=np.uint8),
        frame_ui=15, title="tiny", classification=classified,
    )
    colors = {tuple(value) for value in image.reshape(-1, 3)}
    assert image.shape == (98, 64, 3)
    assert TP_COLOR in colors
    assert FN_COLOR in colors
    assert UNKNOWN_COLOR in colors
