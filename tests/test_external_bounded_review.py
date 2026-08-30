import csv
import json
import shutil
import zipfile
from pathlib import Path

import pytest

from neurobench.experiments.neuron_identifiability import external_bounded_review as review


def test_coordinate_conversion_respects_header_scale_and_global_origin() -> None:
    assert review.local_to_global(0, 54, header_px=54, media_scale=3, region_x0=318, region_y0=111) == (318, 111)
    assert review.local_to_global(575.5, 629.5, header_px=54, media_scale=3, region_x0=318, region_y0=111) == pytest.approx((509.8333333, 302.8333333))
    with pytest.raises(ValueError):
        review.local_to_global(20, 20, header_px=54, media_scale=3, region_x0=318, region_y0=111)


def test_candidate_randomization_is_deterministic_and_opaque() -> None:
    items = [{"blind_id": f"NC{i:03d}", "detection_site_id": f"dsite_{i:03d}", "video": f"NC{i:03d}.mp4"} for i in range(1, 19)]
    first = review.candidate_randomization(items)
    second = review.candidate_randomization(items)
    assert first == second
    assert [row["blind_review_id"] for row in first] == [f"Q{i:03d}" for i in range(1, 19)]
    assert len({row["blind_id"] for row in first}) == 18


def _contract() -> dict:
    return {"phase_A_clip_ids": ["R01", "R02"], "phase_B_item_ids": ["Q001", "Q002"], "phase_A_matching_radius_px": 6.0,
            "permitted_claim_scope": "bounded", "prohibited_claims": ["external"]}


def _phase_a(reviewer_id: str, shift: float = 0.0) -> dict:
    return {"schema_version": 2, "package_id": review.PACKAGE_ID, "phase": "A_raw_first", "reviewer_id": reviewer_id, "locked": True,
            "coverage": {"R01": True, "R02": True}, "full_region_certified": True,
            "marks": [{"mark_id": "M001", "clip_id": "R01", "x_px_global": 400 + shift, "y_px_global": 150, "class": "distinct_source"}]}


def _phase_b(reviewer_id: str, second_call: str = "uncertain") -> dict:
    return {"schema_version": 2, "package_id": review.PACKAGE_ID, "phase": "B_assisted", "reviewer_id": reviewer_id, "locked": True,
            "phase_order_certified": True, "ratings": {
                "Q001": {"neuron_call": "definite_neuron", "identity_relation": "distinct_source", "raw_visibility": "visible", "ica_visibility": "visible", "ls_visibility": "visible"},
                "Q002": {"neuron_call": second_call, "identity_relation": "unresolved", "raw_visibility": "uncertain", "ica_visibility": "visible", "ls_visibility": "visible"},
            }}


def test_submission_validation_fails_closed_on_coverage_and_phase_order() -> None:
    good_a = _phase_a("A")
    assert review.validate_submission(good_a, _contract()) == []
    bad_a = dict(good_a, coverage={"R01": True})
    assert "Phase A coverage incomplete" in review.validate_submission(bad_a, _contract())
    good_b = _phase_b("A")
    assert review.validate_submission(good_b, _contract()) == []
    bad_b = dict(good_b, phase_order_certified=False)
    assert "Phase B phase-order certification missing" in review.validate_submission(bad_b, _contract())


def test_agreement_analysis_creates_adjudication_queue(tmp_path: Path) -> None:
    private = tmp_path / "private"
    private.mkdir()
    (private / "scoring_contract.json").write_text(json.dumps(_contract()))
    (private / "phase_B_randomization_key.json").write_text(json.dumps({"items": [
        {"blind_review_id": "Q001", "reference_single_reviewer_label": "definite_neuron"},
        {"blind_review_id": "Q002", "reference_single_reviewer_label": "uncertain"},
    ]}))
    payloads = [_phase_a("A"), _phase_a("B", 2.0), _phase_b("A"), _phase_b("B", "unlikely_neuron")]
    paths = []
    for index, payload in enumerate(payloads):
        path = tmp_path / f"submission_{index}.json"
        path.write_text(json.dumps(payload))
        paths.append(path)
    result = review.analyze_submissions(paths, private, tmp_path / "analysis")
    assert result["phase_A"]["spatial_matches"] == 1
    assert result["phase_A"]["median_match_distance_px"] == 2.0
    assert result["phase_B"]["exact_call_agreement"] == 0.5
    assert result["adjudication_items"] == 1


def _write_tsv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader(); writer.writerows(rows)


def test_builder_separates_public_payloads_from_private_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = tmp_path / "run"
    raw = run / "review_packet" / "bounded_field_media"
    raw.mkdir(parents=True)
    media = []
    for burst in (1, 2):
        (raw / f"burst_{burst}_raw_review.mp4").write_bytes(b"raw-video")
        (raw / f"burst_{burst}_mean_projection.png").write_bytes(b"mean")
        (raw / f"burst_{burst}_max_projection.png").write_bytes(b"max")
        media.append({"burst_id": burst, "path": f"burst_{burst}_raw_review.mp4", "first_frame_ui": 10, "last_frame_ui": 20, "frame_count": 11, "fps": 10})
    (raw / "media_manifest.json").write_text(json.dumps({"region": {"x0": 318, "y0": 111, "width": 192, "height": 192}, "media": media}))

    candidates = tmp_path / "candidates"
    candidates.mkdir()
    items = []
    rows = []
    for i in (1, 2):
        blind = f"NC{i:03d}"
        video = f"{blind}.mp4"
        (candidates / video).write_bytes(b"candidate-video")
        items.append({"blind_id": blind, "detection_site_id": f"dsite_{i:03d}", "video": video})
        rows.append({"blind_id": blind, "detection_site_id": f"dsite_{i:03d}", "normalized_label": "definite_neuron", "confidence_1_to_5": "5"})
    (candidates / "manifest.json").write_text(json.dumps({"items": items}))
    _write_tsv(candidates / "user_review_v1.tsv", rows)
    monkeypatch.setattr(review, "_reblind_video", lambda source, destination, blind_id: shutil.copy2(source, destination))

    output = tmp_path / "package"
    result = review.build_package(run, candidates, output)
    assert result["public_private_separation"] is True
    assert (output / "validation.json").is_file()
    public_text = "\n".join(path.read_text(errors="ignore") for path in (output / "public_phase_B_assisted").rglob("*") if path.is_file())
    assert "dsite_" not in public_text
    assert "reference_single_reviewer" not in public_text
    assert "dsite_" in (output / "private_administrator/phase_B_randomization_key.json").read_text()
    with zipfile.ZipFile(output / "NeuRev_external_review_v2_PHASE_A_RAW_FIRST.zip") as archive:
        assert "START_HERE.html" in archive.namelist()
