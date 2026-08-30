# Figure contract

The files beginning with `provisional_` are copied from the uploaded repository
for drafting only. They must not be used as final evidence because they predate
the observation-site/canonical-identity repair.

The protected Codex run should export the following stable filenames:

| Figure | Stable filename | Required contents |
|---|---|---|
| 1 | `fig01_study_and_annotation_contract.pdf` | Recording projection, original sites, canonical mappings, bursts, confidence and field boundary |
| 2 | `fig02_acquisition_qc.pdf` | Mean--variance fit, saturation, motion, field diagnostics and optional-video comparison |
| 3 | `fig03_trace_atlas_examples.pdf` | Three-page frozen hero, confirmed hard-case antihero, and seeded random inclusion; Raw, CS--Parzen ICA, local standardization, and synchronized center traces |
| 4 | `fig04_shared_specific_decomposition.pdf` | Shared template, local controls, variance components and conditional functional/tensor results |
| 5 | `fig05_observability.pdf` | Site-by-burst matrix, rank stability, ICCs and timing/identity sensitivities |
| 6 | `fig06_spatial_phenotypes_failures.pdf` | Radial profiles, center/ring examples, crowding and failure taxonomy |
| 7 | `fig07_detection_implications.pdf` | Frozen-panel budget curves, proposal ceilings, fidelity and bounded-field precision--recall |

The following additional generated PNGs report the post hoc, label-free
detection-event profile analysis. They supplement rather than replace the seven
prespecified figure contracts above.

The evolving diagnostic manuscript additionally uses
`final/fig05_pipeline_diagnostic_audit.png` and
`final/fig06_ica_model_internals.png`, plus
`final/fig07_ls_denominator_diagnostics.png`. Their numbering follows that manuscript
sequence; they do not replace the protected-run filenames in the table above.

| Analysis | Stable filename | Required contents |
|---|---|---|
| Detection-profile overview | `detection_class_overview.png` | Profile-space projection, spatial positions, class counts, and post-freeze known-positive proximity |
| Detection-profile centroids | `detection_class_profiles.png` | Robust-standardized medians for all 11 profile variables |
| Detection-profile representatives | `detection_class_representatives.png` | One centroid-nearest spatial crop and native trace per class |
| Detection-class extensions | `detection_class_extensions.png` | Post-freeze certainty alignment, within-site transitions, recurrence, and lane agreement |
| Advanced class profiling | `detection_class_advanced_extensions.png` | Centroid margins, morphology, consolidated-site field position, and canonical-v7 trajectories |
| Objective spatial-ICA morphology | `spatial_ica_objective_morphology.png` | Protected coordinates versus deterministic translated controls for Raw and spatial ICA |
| Spatial-ICA reconstruction stability | `spatial_ica_reconstruction_stability.png` | Seed and leave-one-burst-out reconstruction, morphology, timing, and recall stability |
| Spatial-ICA class/certainty alignment | `spatial_ica_class_certainty.png` | Protected-site and canonical-v7 class/certainty summaries |
| Canonical-v7 morphology sensitivity | `spatial_ica_canonical_v7_sensitivity.png` | Candidate-assisted dominant-class morphology comparisons |
| Spatial-ICA filter stability | `spatial_ica_filter_stability.png` | Aligned basis-filter gallery documenting the failed component interpretation gate |

Each final figure must have a source-data table, generation command, source hashes,
and an entry in `FIGURE_INDEX.md`. PDF or SVG is preferred for quantitative plots;
raster images must be at journal-appropriate resolution.
