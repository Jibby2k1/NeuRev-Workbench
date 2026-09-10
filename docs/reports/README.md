# Reports and artifact index

Use [Current work](../research/CURRENT_WORK.md) for the cross-program synthesis
and the [research index](../research/README.md) for the full report catalog.

## Current research reports

These Markdown reports explain recorded results and link their provenance.
Local `Outputs/` references may require the research workstation; they are
not portable public evidence capsules. A report's artifact validation does
not establish scientific promotion.

| Report | Reading boundary |
| --- | --- |
| [Gamma-LS final campaign, September 9](../research/SPON_CA_BURST_GAMMA_LS_PAPER_SUCCESS_RESULTS_2026_09_09.md) | Audited proposals and separate protected/independent readouts; failed support and sustained timing gates remain explicit |
| [ICA/whitening synthetic evaluation](../research/ICA_WHITENING_HYPERPARAMETER_EVALUATION_RESULTS.md) | Completed factorial with no passing original recovery configuration |
| [ICA/whitening final real-data results](../research/ICA_WHITENING_REAL_DATA_V1_FINAL_RESULTS.md) | Concluded real-data evidence and completion gates; separate from the Gamma-LS readout |
| [ICA/whitening real-data progress](../research/ICA_WHITENING_REAL_DATA_V1_PROGRESS.md) | Dated snapshot, not a live scheduler or completion assertion |
| [PC-MITL specialized confirmation](../research/PC_MITL_ICA_SPECIALIZED_CONFIRMATION_RESULTS.md) | Locked effect threshold missed; no extension promotion |
| [Envelope morphology](../research/CONTEXTUAL_ENVELOPE_MORPHOLOGY_V1_RESULTS.md) | Temporal operator behavior, separate from detector performance |
| [Feature Atlas v1](../research/SPON_CA_BURST_FEATURE_ATLAS_V1_RESULTS.md) | Exploratory reranking and incomplete model-annotation audit |
| [Uncertainty-aware learning v1.1](../research/UNCERTAINTY_AWARE_FEATURE_LEARNING_V1_1_RESULTS.md) | Engineering completion without a nonlinear advantage or claim promotion |

The earlier [ICA implementation audit](../research/reports/ica_repo_audit.md)
and [PC-MITL phase-1 findings](../research/reports/phase1_findings.md) remain
historical method/provenance notes; the specialized confirmation governs the
later advancement decision.

## Generated analytical readers

Generated readers keep their canonical data and source metadata beside them.

| Report | Canonical source | Generated reader | Status |
|---|---|---|---|
| Fish inverse-control experiment program | `fish_control_program_v1/artifact.json` | [HTML report](fish_control_program_v1/report.html) | verified desktop/mobile |
| Fish neural intent and inverse-control roadmap | `fish_inverse_control_roadmap/artifact.json` | [HTML report](fish_inverse_control_roadmap/report.html) | verified desktop/mobile |
| Automated residual diagnostics for neuron identifiability | [artifact JSON](neuron_identifiability_automated_diagnostics_20260830/artifact.json) | not generated | source-backed technical synthesis; no scientific claim |

Rules:

- edit `artifact.json` or supporting source files, not generated `report.html`;
- rebuild with the repository-approved portable report builder;
- keep source queries and chart notes beside the artifact;
- treat HTML as generated in code review and search;
- never use a screenshot as the only report deliverable.
