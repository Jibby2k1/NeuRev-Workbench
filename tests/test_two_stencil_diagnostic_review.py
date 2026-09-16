"""Focused checks for provenance and outcome-enriched diagnostic indexing."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from neurobench.experiments.gamma_ls_difference.two_stencil_diagnostic_review import (
    Sources,
    select_cards,
    sha256,
    validate_fixed_outcomes,
)


def outcome(burst: int, identity: int, matched: bool) -> dict[str, str]:
    return {
        "observation_id": f"b{burst:02d}__roi_{identity:03d}",
        "canonical_roi_id": f"roi_{identity:03d}",
        "burst_id": str(burst), "matched": str(matched),
    }


def test_pilot_retains_all_successes_and_pairs_identities_without_score_selection():
    successes = {(1, 3), (3, 1), (3, 3), (3, 5), (4, 3), (4, 5)}
    rows = [outcome(burst, identity, (burst, identity) in successes)
            for burst in range(1, 5) for identity in range(1, 8)]
    selected = select_cards(rows)
    assert len(selected) == 14
    assert selected == select_cards(list(reversed(rows)))
    selected_successes = [r for r in selected if r["matched"] == "True"]
    assert len(selected_successes) == 6
    assert len({r["canonical_roi_id"] for r in selected_successes}) == 3
    for burst in range(1, 5):
        assert sum(r["burst_id"] == str(burst) and r["matched"] == "False" for r in selected) == 2
    burst1_misses = [r["canonical_roi_id"] for r in selected
                     if r["burst_id"] == "1" and r["matched"] == "False"]
    assert burst1_misses == ["roi_001", "roi_005"]
    assert all("score" not in r for r in rows)


def test_pilot_rejects_duplicate_occurrence_and_insufficient_misses():
    one = outcome(1, 1, True)
    with pytest.raises(ValueError, match="Duplicate"):
        select_cards([one, one])
    with pytest.raises(ValueError, match="fewer than"):
        select_cards([one, outcome(1, 2, False)])


def test_consumed_source_must_match_frozen_artifact_index(tmp_path: Path):
    source = tmp_path / "table.tsv"
    source.write_text("id\noriginal\n")
    (tmp_path / "artifact_index.json").write_text(json.dumps({"artifacts": [
        {"path": "table.tsv", "sha256": sha256(source), "size_bytes": source.stat().st_size}
    ]}))
    sources = Sources()
    assert sources.attach(tmp_path, "table.tsv") == source
    source.write_text("id\nmodified\n")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        sources.attach(tmp_path, "table.tsv")
    with pytest.raises(ValueError, match="outside this index-only workflow"):
        sources.attach(tmp_path, "unopened_array.npy")


def test_fixed_state_validation_rejects_head_and_temporal_truth_drift():
    successes = {(1, 3), (3, 1), (3, 3), (3, 5), (4, 3), (4, 5)}
    rows = [outcome(burst, identity, (burst, identity) in successes)
            for burst in range(1, 5) for identity in range(1, 21) if (burst, identity) != (4, 20)]
    experts = []
    for row in rows:
        row.update({"representation": "difference_signed", "quiet_swap": "a_train_b_test",
                    "target_nms_peaks_per_pseudo_burst": "1.0", "nms_distance_px": "6",
                    "candidate_budget": "58", "x_px": "20.1", "y_px": "30.2"})
        experts.append({k: row[k] for k in ("observation_id", "canonical_roi_id", "burst_id", "x_px", "y_px")})
        experts[-1]["temporal_extent_semantics"] = "configured_burst_window_not_per_roi_onset"
    validate_fixed_outcomes(rows, experts)
    rows[0]["quiet_swap"] = "b_train_a_test"
    with pytest.raises(ValueError, match="Mixed frozen"):
        validate_fixed_outcomes(rows, experts)
    rows[0]["quiet_swap"] = "a_train_b_test"
    experts[0]["temporal_extent_semantics"] = "invented_onset_truth"
    with pytest.raises(ValueError, match="timing semantics"):
        validate_fixed_outcomes(rows, experts)
