# Current research state

## Scope and latest parallel work

This page describes the identity-safe study of 106 occurrences at 50 immutable
sites. The [current repository work map](../../docs/research/CURRENT_WORK.md)
and [generated project story](../../research/generated/PROJECT_STORY.md) cover
the other programs through September 9, 2026.

The separate Gamma-LS campaign now has a completed full-record model-only audit
and a 6/51 sparse-positive result on `15 right` at B58 per one-second block.
The separate ICA/whitening study completed its requested artifact gates and
recovered 20/51 under its own independent block-ranking protocol. These use
different candidate universes and pipelines; they are not a matched comparison,
and neither result closes this manuscript's independent-transfer or population
generalization gates. Read the
[Gamma-LS final report](../../docs/research/SPON_CA_BURST_GAMMA_LS_PAPER_SUCCESS_RESULTS_2026_09_09.md)
and [ICA/whitening final report](../../docs/research/ICA_WHITENING_REAL_DATA_V1_FINAL_RESULTS.md)
for their exact readouts and boundaries.

## Documents to inspect

- `main_technical.tex`: evidence-complete manuscript with full methods and
  interpretation boundaries. `main.tex` remains its canonical journal filename.
- `main_overview.tex`: shorter, plain-language companion using the same results.
- `main_experiment_story.tex`: generated question-design-result-decision history.
- `supplement_main.tex`: extended supplementary material.

Claim status and experiment membership are maintained in the repository-level
`../../research/registry/`. The local `story/research_story.yaml` file is a
generated Overleaf compatibility view. Update canonical records, run
`python -m neurobench.research.registry build`, then run `make story`; use
`make story-check` in validation or continuous integration.

## Established within the current recording

- The immutable canonical-v7 population contains 106 confirmed neuron-burst occurrences
  at 50 immutable coordinate-defined sites across four bursts.
- Canonical v8 is an interpretation layer over that v7 cohort: it adds the
  reviewed identity-aware audit but does not change coordinates, identities, or
  inclusion labels.
- Raw traces contain a strong event-aligned population waveform that is also
  expressed in nearby tissue.
- The all-site trace reassessment identifies two descriptive measurement
  phenotypes, T1 (38 sites) and T2 (12 sites). These are not claimed neuron types.
- The frozen long-delay CS-Parzen temporal transform contains six components at
  lags 0, 1, 2, 4, 8, and 16 frames. Component activations and mixing/demixing
  matrices are exported at all sites.
- Maximum sitewise embedding inversion RMS is 5.69e-12. This validates numerical
  invertibility, not denoising or biological source recovery.
- The true local-standardization denominator was reproduced with the original
  31-frame causal pre-roll and saved global floor. Reproduced center traces agree
  with the saved CUDA output at RMSE 6.96e-7 and maximum error 7.63e-6.
- The global scale floor is 0.371705, the median audited denominator is 0.7648,
  and the floor is inactive at all 50 centers over all 560 review frames.
- Across the complete trace, confirmed-event maxima are nearly ceiling-level at
  known centers: site-weighted mean percentiles are 0.994 for the frozen carrier,
  0.994 for 15-frame coherence, and 0.996 for lag-2 recurrence. No tested feature
  has a strictly positive paired interval versus the carrier.
- The harder frame-level event-versus-quiet test ranks Raw first at site-weighted
  AUC 0.931 [0.906, 0.952], followed by the carrier (0.920), coherence (0.915),
  and lag-2 recurrence (0.905).
- All six compact spatial-challenge features have positive site-bootstrap
  intervals for true-center minus same-field baseline-matched displaced AUC.
  This establishes localized measurement structure, not biological specificity.
- All selected observed feature alignments exceed 500 site-blocked circular
  shifts. Synthetic matched-filter sensitivity rises monotonically with
  injection amplitude; these are operator checks rather than biological truth.
- Of 12 all-lane B58 misses, fixed-neighborhood consensus candidates produced
  two same-identity collisions, six different-identity collisions, and four
  identity-clear hypotheses. Thus eight apparent recentering gains cannot be
  treated as safe coordinate corrections.

## Suggestive but not confirmatory

- Feature Atlas v1 increased augmented macro held-fold SPU-AUC to 0.984392
  from 0.979734 for the retrained existing linear model on the same 1,619
  candidates, with 35/78 versus 34/78 known positives at budget 58. The paired
  global-AUC delta interval [-0.004081, 0.015125] crossed zero; full new-ranking
  audit and scientific promotion remain incomplete. The
  [Atlas result](../../docs/research/SPON_CA_BURST_FEATURE_ATLAS_V1_RESULTS.md)
  motivates a frozen competition-aware proposal-resolution test.

- A blinded single-reviewer pass over all 18 previously unmatched consolidated
  detector sites produced 9 definite-neuron, 4 probable-neuron, 4 uncertain,
  and 1 unlikely/artifact-or-noise call. The resulting 13/18 likely-neuron
  fraction describes a selected candidate batch, not detector precision.
- The candidate-priority score had AUC 0.308 for likely versus
  unresolved/unlikely calls. This is consistent with a review-value score that
  intentionally elevates ambiguity and crowding, but it is unsuitable as a
  neuron-probability score.
- Positive calls included small, discreet, low-SNR, and crescent or possibly
  multi-source appearances. These observations broaden provisional morphology
  hypotheses but do not yet define distinct biological identities.

- T1/T2 trace phenotypes recur sufficiently to support within-recording
  diagnostic use, but they require independent-recording confirmation.
- Spatial and temporal context can improve known-positive ranking in some frozen
  lanes, but sparse positives do not identify full-field precision.
- Representation consensus and multiscale persistence do not improve
  known-center localization, but their median available cross-burst site-rank
  correlations (0.633 and 0.649) motivate independent-transfer tests as
  exploratory observability features.
- Spatial ICA can provide a stable measurement representation even when
  individual filters are not stable enough to name as biological sources.
- A site-grouped role-specific recovery model improves cross-fitted log loss
  from 0.712 to 0.577 and ROC AUC from 0.605 to 0.826, but the site-bootstrap
  log-loss-improvement interval [-0.025, 0.280] crosses zero. Complementarity is
  not confirmed.
- Annular contrast is highly repeatable across sites (nonparametric site
  fraction about 0.87) despite weak event retrieval, showing that repeatability
  can reflect stable nuisance or acquisition structure.
- The identity-clear ROI 19, ROI 22, ROI 25, and ROI 27 candidates remain
  targeted hypotheses; the audit does not promote their coordinates.
- Collision shifts are more directionally consistent across Raw/ICA/LS than
  identity-clear shifts (median cosine 0.972 versus 0.448) and show greater
  neighboring-trace partial correlation (0.474 versus 0.182). These are
  descriptive eight-versus-four comparisons.
- Response-weighted footprint traces increase event-to-pre-event-MAD ratios in
  both groups, but same-event weighting makes this an extraction-sensitivity
  result rather than prospective recovery or identity validation.
- Pre-event-frozen footprint weights retain substantial event peak/MAD (11.98
  collisions; 9.51 identity-clear), providing a less circular extraction test.
- No identity-clear miss has a frozen candidate within eight pixels by budget
  100 in any audited lane. At the frozen outputs these are proposal-absence,
  not merely B58 ranking, cases.
- Multi-neighbor traces explain median 52.3% of collision-candidate variance
  versus 8.8% for clear misses, but exact eight-versus-four tests remain
  descriptive and non-small.
- Expanded identity-grouped prediction reaches AUC 0.833 and log loss 0.498,
  versus AUC 0.611 and log loss 0.690 for carrier alone.
- Variance-stabilized annular context is selected in 97% of 100 site-bootstrap
  sparse fits, followed by multiscale persistence (73%) and lag recurrence
  (39%). This supports observability context, not a neuron-specific annulus.
- Internal abstention reduces error from 20.8% at full coverage to 9.4% at 50%
  coverage. Recovered-only isolation-forest anomaly scoring separates misses at
  AUC 0.814; both require independent confirmation.
- A simple five-neighbor spatial graph has negligible association with recovery
  (-0.009), so distance-only crowding is not an adequate feature model.
- The bounded uncertainty-aware feature-fusion Run B completed 1,619 spatially
  held-out candidate scores (78 known positives and 1,541 unknown candidates).
  Tiny-MLP macro-fold SPU-AUC was 0.97402 versus 0.97410 for linear SPU, with
  MLP-minus-linear interval [-0.00461, 0.00591]; the nonlinear replacement is
  therefore held. The favorable same-union CFAR reranking contrast is useful
  engineering context, not end-to-end detector replacement or scientific
  promotion.

## Still unresolved

- Full-field precision, specificity, and false-positive rate.
- Independent-recording generalization.
- Motion/registration residuals and their contribution to shared waveforms.
- One-to-one biological source identity for ICA or local-standardized outputs.
- Author affiliations, ethics, funding, CRediT roles, acknowledgements, and
  release identifiers required for submission.
- Independent review and adjudication of the 18 new candidate sites, including
  four uncertain and several artifact- or neighbor-confounded cases.

## External review readiness

- A staged portable review package is complete and validated for at least two
  independent reviewers. Phase A contains four Raw-first bounded-field clips;
  Phase B contains 18 re-randomized and re-blinded six-panel candidates.
- Public and private payloads are separated. Phase B is distributed only after
  a locked Phase A submission. Automated scoring produces agreement and an
  adjudication queue, not truth by itself.
- This is implementation readiness only. No external annotations, adjudicated
  bounded truth, or updated detector metric exists yet.

## Primary visual evidence

- Figure 3: synchronized representative Raw/ICA/LS images and traces.
- Figure 4: global and T1/T2 population trace summaries.
- Figure 5: pipeline diagnostic audit and artifact availability.
- Figure 6: temporal-ICA component and invertibility diagnostics.
- Figure 7: true local-standardization denominator audit.
- Figure 8: complete-trace feature localization, carrier deltas, and cross-burst
  site-rank repeatability.
- Figure 9: frame retrieval, displaced controls, grouped recovery modeling, and
  nonparametric site persistence.
- Figure 10: coordinate/timing perturbations, synthetic transient sensitivity,
  and circular-shift falsification.

## Recommended next validation

1. Distribute and complete the staged external bounded review, adjudicate all
   disagreements, and freeze the resulting local truth revision.
2. Run a separately versioned automated robustness package for the frozen
   linear SPU candidate, retaining elastic and bagged-PU sensitivities and
   adding leave-one-burst-out, alternate review-policy, hard-subgroup,
   feature-family-ablation, and source-off/null stress tests. Do not tune the
   held tiny MLP on Run-B outcomes.
3. Jointly review remaining ambiguous identity clusters.
4. Exhaustively expand beyond the enriched region for broader precision estimation.
5. Apply the frozen pipeline and class-assignment rules to an independent recording.
6. Export motion fields and registration residuals.
7. Transfer the frozen carrier/coherence/recurrence and exploratory
   consensus/persistence features without tuning.
8. Preserve the current spatial-context and robust-stopping baseline; defer
   additional deblender complexity until bounded biological identity truth or
   a genuinely independent labelled recording is available.

The completed bounded screen, hold decision, and interpretation boundary are
recorded in
`../../docs/research/UNCERTAINTY_AWARE_FEATURE_LEARNING_V1_1_RESULTS.md`.
The broader legacy learning plan remains
`../../docs/research/SPON_CA_BURST_UNCERTAINTY_AWARE_LEARNING_PLAN_V1.md`.
