from neurobench.experiments.neuron_identifiability.detection_class_advanced_extensions import _collapsed_context, _collapsed_morphology, membership_margins


def test_collapsed_review_categories_are_predeclared_and_distinct():
    assert _collapsed_morphology("localized_center") == "localized boundary"
    assert _collapsed_morphology("no_distinct_boundary") == "ambiguous/non-neuronal boundary"
    assert _collapsed_context("overlapping") == "overlapping"
    assert _collapsed_context("weak_local_contrast") == "challenging/structured context"


def test_membership_margin_reproduces_frozen_assignment():
    taxonomy = {"classification": {"standardized_centroids": {
        "1": {"f": 0.0}, "2": {"f": 2.0}, "3": {"f": 4.0},
    }}}
    # Patch the imported feature tuple for a minimal synthetic contract.
    import neurobench.experiments.neuron_identifiability.detection_class_advanced_extensions as module
    original = module.FEATURES
    module.FEATURES = ("f",)
    try:
        rows = membership_margins([{"detection_occurrence_id": "d", "detection_site_id": "s", "burst_id": "1", "class_id": "2", "z_f": "2.2"}], taxonomy)
    finally:
        module.FEATURES = original
    assert rows[0]["class_id"] == 2
    assert rows[0]["relative_distance_margin"] > 0
