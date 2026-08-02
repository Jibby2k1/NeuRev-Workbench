import numpy as np

from neurobench.experiments.hard_roi_adjudication.review_checklist import (
    timing_suggestion,
)


def test_timing_suggestion_can_identify_a_pre_window_peak() -> None:
    frames = np.arange(2025, 2064)
    trace = np.zeros(len(frames), dtype=np.float64)
    trace[(frames >= 2030) & (frames <= 2038)] = np.asarray(
        [1, 3, 7, 12, 18, 12, 7, 3, 1], dtype=np.float64
    )
    result = timing_suggestion(
        frames, trace, original_start_ui=2040, original_end_ui=2063
    )
    assert result["suggested_peak_ui"] == 2034
    assert result["peak_precedes_original_window"] is True
    assert result["suggested_onset_ui"] <= result["suggested_peak_ui"]
    assert result["suggested_peak_ui"] <= result["suggested_end_ui"]
    assert result["interpretation"] == "reviewer_aid_not_adjudication"
