from __future__ import annotations

import json
from pathlib import Path

import pytest

from neurobench.review.truth_set import assert_blinded_payload, build_candidate_union, deterministic_region_from_mask, deterministic_second_review_sample


ROOT = Path(__file__).resolve().parents[1]


def candidates() -> list[dict]:
    return json.loads((ROOT / "examples" / "spon_ca_burst_truth_set_v1.example.json").read_text())["candidates"]


def test_candidate_union_is_deterministic_and_deduplicated() -> None:
    first = build_candidate_union(candidates(), spatial_radius_px=1.5, temporal_radius_frames=1, random_seed=5)
    second = build_candidate_union(list(reversed(candidates())), spatial_radius_px=1.5, temporal_radius_frames=1, random_seed=5)
    assert first == second
    assert len(first[0]) == 4


def test_opaque_ids_are_stable_and_public_rows_have_no_source_fields() -> None:
    public, private = build_candidate_union(candidates(), spatial_radius_px=1.5, temporal_radius_frames=1, random_seed=5)
    assert all(item["candidate_id"].startswith("candidate_") for item in public)
    assert set(private) == {item["candidate_id"] for item in public}
    assert_blinded_payload(public)


@pytest.mark.parametrize("payload", [
    {"score": 2.0}, {"rank": 1}, {"lane_id": "secret"}, {"css_class": "fullrank_ica_w17_marker"}, {"filename": "candidate_source_key.json"}, {"label": "Raw Direct"}
])
def test_blinding_audit_rejects_identity_score_rank_filename_or_css(payload: dict) -> None:
    with pytest.raises(ValueError, match="blinding leak"):
        assert_blinded_payload(payload)


def test_second_review_samples_at_least_twenty_percent_per_disposition() -> None:
    rows = [{"subject_id": f"a{i}", "disposition": "accepted"} for i in range(10)] + [{"subject_id": f"r{i}", "disposition": "rejected"} for i in range(5)]
    rows += [{"subject_id": "u", "disposition": "unresolved"}, {"subject_id": "d", "disposition": "accepted", "disagreement": True}]
    selected = deterministic_second_review_sample(rows, seed=7)
    assert "u" in selected and "d" in selected
    assert len([item for item in selected if item.startswith("a")]) >= 2
    assert len([item for item in selected if item.startswith("r")]) >= 1


def test_protected_region_selection_uses_only_mask_seed_and_exclusions() -> None:
    mask = [[1] * 8 for _ in range(6)]
    excluded = [{"x_min": 0, "y_min": 0, "x_max_exclusive": 4, "y_max_exclusive": 3}]
    first = deterministic_region_from_mask(mask, width_px=2, height_px=2, seed=19, excluded_bounds=excluded)
    second = deterministic_region_from_mask(mask, width_px=2, height_px=2, seed=19, excluded_bounds=excluded)
    assert first == second
    assert first["x_min"] >= 4 or first["y_min"] >= 3
