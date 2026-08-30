# Neuron-identifiability experiment package

This package contains the maintained implementation for the Spon Ca Burst
single-recording neuron-identifiability program. Start from the scientific
question, then open only the matching module and test.

## Core routes

| Concern | Modules | Tests or evidence |
| --- | --- | --- |
| Configuration and contracts | `config.py`, `contracts.py`, `identity.py` | `test_neuron_identifiability_contracts.py`, `test_neuron_identifiability_identity.py` |
| Canonical cohort and release | `canonical_v7_extension.py`, `reproducibility_release.py` | canonical-v7 and release tests |
| Trace/media extraction | `trace_extraction.py`, `visual_statistical_atlas.py`, `detector_visual_audit.py`, `bounded_field_media.py` | traces, visual-atlas, bounded-media tests |
| Manuscript orchestration | `runner.py`, `manuscript_exports.py`, `reporting.py`, `cli.py`, `__main__.py` | manuscript and research-story tests |
| Candidate review | `review_packet.py`, `new_candidate_review_media.py` | review-packet and new-candidate tests; `Outputs/NeuronIdentifiability/new_candidate_roi_review_batch_v1/` |
| External team review | `external_bounded_review.py`, `external_review_assets/` | `test_external_bounded_review.py`; staged public ZIPs and private scoring contract |

## Analysis families

| Family | Modules |
| --- | --- |
| Representation and ICA | `representation_confirmation.py`, `cross_neural_ica.py`, `ica_morphology_ablation.py`, `multirepresentation_tensor.py`, `tensor_factor_profiles.py` |
| Feature and recovery analysis | `full_trace_feature_panel.py`, `automated_feature_validation.py`, `deep_dive_suite.py`, `expanded_automated_validation.py` |
| Detection taxonomy and extensions | `detection_profile_taxonomy.py`, `detection_class_extensions.py`, `detection_class_advanced_extensions.py`, `discovery.py` |
| Spatial and identity audits | `spatial_specificity.py`, `nms_sensitivity.py`, `strict_nms_audit.py`, `identity_aware_feature_inspection.py`, `identity_aware_validation_suite.py` |
| Spatial-ICA stability | `spatial_ica_filter_stability.py`, `spatial_ica_reconstruction_stability.py`, `spatial_ica_objective_morphology.py`, `spatial_ica_class_certainty_alignment.py` |
| Automated simulator programs | `major_next_steps.py`, `automated_challenge_suite.py`, `targeted_automated_development.py`, `joint_generative_deblending.py` |

## Current boundary

The canonical cohort remains 50 sites and 106 occurrences. The new candidate
review is single-reviewer and provisional. Do not merge its labels into the
canonical cohort or recompute detector recall without an explicit versioned
truth-set decision.

The next specified package is uncertainty-aware learning. Its authority is
`docs/research/SPON_CA_BURST_UNCERTAINTY_AWARE_LEARNING_PLAN_V1.md`; no module
or result should be inferred to exist until it is implemented and validated.

## Output convention

Use a new collision-safe directory under `Outputs/NeuronIdentifiability/`.
Write `summary.json`, `validation.json`, provenance/input hashes, narrow TSVs,
and `artifact_index.json`. Read those before loading figures or media. Follow
`docs/workflows/SCIENTIFIC_AUDIT_OUTPUT_STANDARD.md` for every new run.
