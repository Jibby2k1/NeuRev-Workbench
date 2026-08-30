# Figure index

## Figure 3 - Frozen CS-Parzen trace-atlas cases

- Manuscript label: `fig:trace-atlas-cases`
- Stable manuscript asset: `figures/final/fig03_trace_atlas_examples.pdf` (pages 1--3: hero, antihero, random).
- Raster sources: `fig03a_cs_parzen_hero_roi003_b03.png`, `fig03b_cs_parzen_antihero_roi011_b04.png`, and `fig03c_cs_parzen_random_roi010_b03.png`.
- Source data: `figures/final/fig03_trace_atlas_examples_source.tsv`
- Source lane: `delay_embedding__cs_parzen__long__bandwidth-0p25::residual_group::joint_s15_g5_t31_g1`
- Source arrays: `recovery_msica.npy` and `recovery_msln.npy` from `spon_ca_burst_multilag_msica_v5_all_roi_diagnostics/cache/`
- Case rule: frozen hero ROI 003; frozen confirmed hard-case antihero ROI 011; random eligible confirmed four-burst ROI 010 with seed 20260824. Within each site, select the occurrence with maximum native 3x3 center recovery-MSLN value.
- Generation commands: `render_v7_priority_neuron_media.py --ica-source recovery`; then `python scripts/build_fig03_trace_atlas_pdf.py`.
- Interpretation: descriptive representative cases only; no precision, population, or independent-validation claim.
- Raster SHA-256: hero `e2e7403adc89515f4fea918bd3c2092c50d44305744d109213813f5951200aab`; antihero `d61e147f4c42e4add2d6617febf534f9f296db5be11c6ea7826066ae01d3d02c`; random `c213a8cf40f52eeda66c2a411701be2bccf10c1a439c36b1d000a5f5827a87ee`.
- Multipage PDF SHA-256: `d9e8606744488aaa070f9f51478a95801b776d0bb5cc067193ee19af7f8848b3`.

## Figure 4 - Canonical-v7 population trace summary

- Manuscript label: `fig:population-trace-summary`
- Stable manuscript asset: `figures/final/fig04_population_trace_summary.png`.
- Source data: `figures/final/fig04_population_trace_summary_source.tsv`.
- Provenance: `figures/final/fig04_population_trace_summary_provenance.json`.
- Builder: `scripts/build_fig04_population_trace_summary.py`.
- Population: all 106 confirmed canonical-v7 occurrences, grouped into 50 coordinate-defined original sites.
- Display contract: onset-aligned shape-only global/joint-class means and heatmaps; fixed site/class order across Raw, CS-Parzen ICA, and local standardization; stage-native site-median peak distributions kept separate.
- Class contract: all-site joint trace reassessment; Trace Classes T1 and T2 contain 38 and 12 immutable sites, selected from k=2--6 by silhouette times subsample stability with a five-site minimum. T-prefixes distinguish this taxonomy from the paper's frozen three detection-profile classes.
- Experiment outputs: `fig04_trace_taxonomy_experiment.json`, `fig04_trace_taxonomy_site_assignments.tsv`, and `fig04b_trace_taxonomy_stage_agreement.png`.
- Experiment builder: `scripts/run_trace_taxonomy_experiment.py`; run before `scripts/build_fig04_population_trace_summary.py`.
- Persistence outputs: `fig04c_trace_taxonomy_lobo_persistence.png`, `fig04c_trace_taxonomy_lobo_assignments.tsv`, and `fig04c_trace_taxonomy_lobo_summary.json`.
- Persistence builder: `scripts/run_trace_taxonomy_lobo.py`; 25 recurrent sites and 81 held-out assignments, with site-blocked agreement reported both including and excluding burst 2.
- Interpretation: exploratory within-recording trace measurement profiles only; no biological class or multi-recording population claim.

## Figure 5 - Pipeline diagnostic audit

- Manuscript label: `fig:pipeline-diagnostic-audit`.
- Stable manuscript asset: `figures/final/fig05_pipeline_diagnostic_audit.png`.
- Diagnostic table: `figures/final/fig05_pipeline_diagnostic_table.tsv` at occurrence x stage grain.
- Availability ledger: `figures/final/fig05_diagnostic_availability.tsv`.
- Provenance and quality summary: `figures/final/fig05_pipeline_diagnostic_audit.json`.
- Human-readable audit: `PIPELINE_DIAGNOSTIC_AUDIT.md`.
- Builder: `scripts/build_fig05_pipeline_diagnostic_audit.py`.
- Interpretation: observed and derived signal/spatial diagnostics are separated from proxies and complemented by Figures 6--7 model internals and true-denominator reproduction.

## Figure 6 - ICA model internals

- Manuscript label: `fig:ica-model-internals`.
- Stable manuscript asset: `figures/final/fig06_ica_model_internals.png`.
- Machine-readable arrays: `figures/final/fig06_ica_model_internals.npz`.
- Per-site table: `figures/final/fig06_ica_component_energy_by_site.tsv`.
- Provenance summary: `figures/final/fig06_ica_model_internals.json`.
- Builder: `scripts/build_fig06_ica_model_internals.py`.
- Population: 50 unique canonical-v7 observation sites over the aligned 560-frame review interval.
- Interpretation: exposes the six frozen temporal components, demixing/mixing matrices, energy fractions, and numerical embedding inversion residual. It is not a denoising reconstruction test.

## Figure 7 - Local-standardization denominator audit

- Manuscript label: `fig:ls-denominator-audit`.
- Stable manuscript asset: `figures/final/fig07_ls_denominator_diagnostics.png`.
- Machine-readable arrays: `figures/final/fig07_ls_denominator_diagnostics.npz`.
- Per-site table: `figures/final/fig07_ls_denominator_by_site.tsv`.
- Provenance summary: `figures/final/fig07_ls_denominator_diagnostics.json`.
- Builder: `scripts/build_fig07_ls_denominator_diagnostics.py`.
- Population: 50 unique canonical-v7 sites, 560 review frames, and the union of 9,297 pixels in their 15x15 supports.
- Validation: LS center RMSE $6.96\times10^{-7}$ against the saved CUDA output; maximum absolute error $7.63\times10^{-6}$.
- Interpretation: the saved floor was inactive at all audited centers; this does not establish full-field floor inactivity.
- Figure and source SHA-256 values are recorded in `fig04_population_trace_summary_provenance.json` and regenerated with the figure.

## Figure 8 - Full-trace feature panel

- Manuscript label: `fig:full-trace-feature-panel`.
- Stable manuscript asset: `figures/final/fig08_full_trace_feature_panel.png`.
- Importer: `scripts/import_fig08_full_trace_feature_panel.py`.
- Source run: `Outputs/NeuronIdentifiability/spon_ca_burst_full_trace_feature_panel_v2`.
- Interpretation: ceiling-limited known-center event maxima and exploratory
  cross-burst site-rank repeatability; no precision claim.

## Figure 9 - Automated feature validation

- Manuscript label: `fig:automated-feature-validation`.
- Stable manuscript asset: `figures/final/fig09_automated_feature_validation.png`.
- Provenance: `figures/final/fig09_automated_feature_validation.json`.
- Importer: `scripts/import_fig09_automated_feature_validation.py`.
- Source run: `Outputs/NeuronIdentifiability/spon_ca_burst_automated_feature_validation_v1`.
- Population: 106 original geometries at 50 immutable sites; the 102-row
  canonical-collapsed view is a sensitivity only.
- Interpretation: temporal retrieval, displaced unknown controls, exploratory
  grouped recovery modeling, and nonparametric site persistence.

## Figure 10 - Robustness and falsification

- Manuscript label: `fig:robustness-falsification`.
- Stable manuscript asset: `figures/final/fig10_robustness_falsification.png`.
- Importer: `scripts/import_fig09_automated_feature_validation.py`.
- Source run: `Outputs/NeuronIdentifiability/spon_ca_burst_automated_feature_validation_v1`.
- Interpretation: fixed coordinate/timing perturbations, simplified synthetic
  transient sensitivity, and 500-draw site-blocked circular-shift nulls;
  operator validation only.

## Figure 11 - Feature deep dives

- Manuscript label: `fig:feature-deep-dives`.
- Stable manuscript asset: `figures/final/fig11_feature_deep_dives.png`.
- Source run: `Outputs/NeuronIdentifiability/spon_ca_burst_feature_deep_dives_v2`.
- Interpretation: proposal/ranking contrasts, fixed kinetic-bank selection,
  failure taxonomy, and exploratory grouped recovery models.

## Figure 12 - Spatial decay and crowding

- Manuscript label: `fig:spatial-decay-crowding`.
- Stable manuscript asset: `figures/final/fig12_spatial_decay_crowding.png`.
- Interpretation: within-recording spatial-decay and labeled-neighbor proxies;
  sparse labels do not establish full crowding truth.

## Figure 13 - Canonical-v8 identity-aware feature inspection

- Manuscript label: `fig:identity-aware-features`.
- Stable manuscript asset: `figures/final/fig13_identity_aware_feature_inspection_v8.png`.
- Source run: `Outputs/NeuronIdentifiability/spon_ca_burst_identity_aware_feature_inspection_v8_v2`.
- Machine-readable tables: `generated_analysis/identity_aware_feature_inspection_v8/`.
- Population: 94 recovered occurrences, eight identity-collision misses, and
  four identity-clear misses.
- Interpretation: descriptive cross-stage offset stability, extraction,
  morphology, local competition, and neighboring-trace leakage. No classifier
  was fit to the four identity-clear misses, and same-event footprint weighting
  is not prospective recovery evidence.

## Figure 14 - Expanded automated validation

- Manuscript label: `fig:expanded-automated-v3`.
- Stable manuscript asset: `figures/final/fig14_expanded_automated_validation.png`.
- Source run: `Outputs/NeuronIdentifiability/spon_ca_burst_expanded_automated_validation_v3`.
- Machine-readable tables: `generated_analysis/expanded_automated_validation_v3/`.
- Population: 106 occurrences at 50 identity-grouped sites, with 94 recovered
  and 12 missed frozen B58 outcomes.
- Interpretation: ten exploratory non-RL test families covering grouped
  prediction, stability, importance, prospective controls, mixtures,
  metamorphic behavior, abstention, anomaly detection, and graph context.
