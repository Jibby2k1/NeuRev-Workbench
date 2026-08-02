import numpy as np

from neurobench.experiments.hard_roi_adjudication.review_checklist_v2 import (
    timing_suggestion_v2,
)


def test_v2_timing_is_restricted_to_search_and_contiguous_peak_segment() -> None:
    frames = np.arange(2025, 2064)
    trace = np.zeros(len(frames), dtype=np.float64)
    trace[0:3] = 50  # unrelated activity outside the allowed early search
    trace[(frames >= 2031) & (frames <= 2037)] = [2, 5, 10, 16, 10, 5, 2]
    result = timing_suggestion_v2(
        frames, trace, original_start_ui=2040, original_end_ui=2063
    )
    assert result["suggested_onset_ui"] >= 2028
    assert result["suggested_peak_ui"] == 2034
    assert result["suggested_onset_ui"] <= result["suggested_peak_ui"]
    assert result["suggested_peak_ui"] <= result["suggested_end_ui"]
    assert result["method"].startswith("v2_contiguous")
