from pathlib import Path

import numpy as np

from neurobench.experiments.hard_roi_adjudication.adjudication import (
    draft_rows,
    label_view,
    load_tsv,
    write_tsv,
)
from neurobench.experiments.hard_roi_adjudication.config import (
    HardRoiAdjudicationConfig,
)
from neurobench.experiments.hard_roi_adjudication.review_pack import (
    _render_review_frame,
)


def _label(burst: int, roi: str, x: float, y: float) -> dict:
    return {
        "burst_id": burst,
        "roi_identity": roi,
        "x_px": x,
        "y_px": y,
        "start_frame_ui": 2003,
        "end_frame_ui": 2026,
    }


def test_config_freezes_target_and_feature_panels() -> None:
    config = HardRoiAdjudicationConfig.load(
        "examples/spon_ca_burst_hard_roi_adjudication_v1.example.json"
    )
    assert {
        "roi_007", "roi_010", "roi_014", "roi_015", "roi_019",
        "roi_020", "roi_023",
    }.issubset(config.review["target_roi_ids"])
    assert set(config.panel_paths()) == {
        "carrier_signed", "coherence_w15", "propagation_lag2_w15",
        "radial_cs_shell", "noise_vst_residual",
    }
    assert all(path.is_absolute() for path in config.panel_paths().values())
    assert all(path.exists() for path in config.panel_paths().values())


def test_draft_encodes_expert_notes_as_provisional_and_merge_is_collapsed() -> None:
    labels = [
        _label(1, "roi_010", 388.694, 138.907),
        _label(1, "roi_015", 389.057, 142.057),
        _label(1, "roi_007", 477.364, 237.223),
    ]
    rows = draft_rows(labels, {"roi_007", "roi_010", "roi_015"})
    by_roi = {row["original_roi_id"]: row for row in rows}
    assert by_roi["roi_015"]["canonical_roi_id"] == "roi_010"
    assert by_roi["roi_015"]["review_status"] == "provisional_expert_note"
    assert by_roi["roi_007"]["include_confirmed"] == "false"

    parsed = []
    for row in rows:
        parsed.append({
            **row,
            "burst_id": int(row["burst_id"]),
            "x_px": float(row["x_px"]),
            "y_px": float(row["y_px"]),
            "original_start_frame_ui": int(row["original_start_frame_ui"]),
            "original_end_frame_ui": int(row["original_end_frame_ui"]),
            "event_onset_ui": None,
            "event_peak_ui": None,
            "event_end_ui": None,
            "include_confirmed": row["include_confirmed"] == "true",
            "include_inclusive": row["include_inclusive"] == "true",
        })
    original = label_view(parsed, "original", "original")
    confirmed = label_view(parsed, "confirmed", "original")
    inclusive = label_view(parsed, "inclusive", "original")
    assert len(original) == 3
    assert len(confirmed) == 1
    assert len(inclusive) == 2
    assert confirmed[0]["roi_identity"] == "roi_010"


def test_table_validation_rejects_pending_required_target(tmp_path: Path) -> None:
    rows = draft_rows([_label(3, "roi_023", 440, 190)], {"roi_023"})
    path = tmp_path / "adjudication.tsv"
    write_tsv(path, rows)
    parsed = load_tsv(path)
    assert parsed[0]["review_status"] == "pending"
    try:
        load_tsv(path, require_adjudicated_targets={"roi_023"})
    except ValueError as exc:
        assert "roi_023" in str(exc)
    else:
        raise AssertionError("pending target was accepted as adjudicated")


def test_review_frame_is_detector_blinded_three_panel_rgb() -> None:
    frame = np.arange(20 * 30, dtype=np.uint16).reshape(20, 30)
    rendered = _render_review_frame(
        frame,
        np.zeros_like(frame),
        frame_ui=2003,
        crop=(2, 3, 28, 19),
        coordinates={"roi_014": (15, 10)},
        display_lo=0,
        display_hi=600,
        difference_hi=300,
        box_half_size_px=4,
    )
    assert rendered.dtype == np.uint8
    assert rendered.ndim == 3 and rendered.shape[2] == 3
    assert rendered.shape[1] >= 3 * 26
    assert rendered.shape[0] % 2 == 0 and rendered.shape[1] % 2 == 0
