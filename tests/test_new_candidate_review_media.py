from pathlib import Path

from neurobench.experiments.neuron_identifiability.new_candidate_review_media import select_new_sites


ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/"Outputs/NeuronIdentifiability/spon_ca_burst_identifiability_paper_v1_v8"


def test_selection_contains_only_all_18_unmatched_sites()->None:
    rows=select_new_sites(RUN/"detection_profile_taxonomy_v5/detection_site_profiles.tsv",RUN/"detection_profile_taxonomy_v5/detection_occurrence_profiles.tsv",RUN/"detection_class_advanced_extensions_v1/membership_margins.tsv")
    assert len(rows)==18
    assert all(int(row["known_positive_occurrences"])==0 for row in rows)
    assert len({row["detection_site_id"] for row in rows})==18
