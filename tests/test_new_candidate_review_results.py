import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / "Outputs/NeuronIdentifiability/new_candidate_roi_review_batch_v1"


def test_user_review_covers_only_the_new_unmatched_batch() -> None:
    manifest = json.loads((PACKET / "manifest.json").read_text())
    with (PACKET / "user_review_v1.tsv").open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))

    expected = {item["blind_id"]: item["detection_site_id"] for item in manifest["items"]}
    observed = {row["blind_id"]: row["detection_site_id"] for row in rows}
    assert manifest["scope"] == "new unmatched candidates only"
    assert manifest["population"]["known_positive_target_sites"] == 0
    assert manifest["population"]["existing_decisions_reopened"] is False
    assert len(rows) == len(expected) == 18
    assert observed == expected
    assert all(row["verbatim_feedback"].strip() for row in rows)
    assert Counter(row["normalized_label"] for row in rows) == {
        "definite_neuron": 9,
        "probable_neuron": 4,
        "uncertain": 4,
        "artifact_or_noise": 1,
    }
