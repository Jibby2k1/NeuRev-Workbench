# Spon Ca Burst automated feature validation v1

## Decision summary

The expanded no-human-in-the-loop suite passed its output and population gates
for 106 confirmed occurrences at 50 immutable original geometries. It resolves
the ceiling in the earlier event-maximum percentile by evaluating every event
frame against guarded quiet-reference frames, and it adds spatial displacement,
site-grouped recovery modeling, coordinate/timing perturbations, repeatability,
circular-shift nulls, and synthetic transient sensitivity.

The strongest defensible conclusions are:

1. **Raw amplitude remains the best known-center temporal carrier.** Its
   site-weighted frame AUC was 0.931 [0.906, 0.952], followed by the frozen
   carrier at 0.920 [0.896, 0.941], coherence at 0.915 [0.890, 0.938], and
   lag-2 recurrence at 0.905 [0.878, 0.927]. The task is event-versus-quiet
   reference retrieval, not biological classification.
2. **Spatial context is genuinely informative within the present controls.**
   Every prespecified compact feature scored higher at the labeled center than
   at a deterministic same-field, quiet-baseline-matched displaced coordinate.
   Site-bootstrap lower bounds were positive for all six features. The largest
   paired AUC difference belonged to multiscale persistence (0.217 [0.165,
   0.270]), followed by coherence and lag recurrence. Displaced tissue remains
   biologically unknown, so this establishes localized measurement structure,
   not neuron specificity.
3. **Feature complementarity is promising but not confirmed.** At detector
   budget 58, 94/106 original geometries were recovered by at least one of the
   carrier, coherence, or lag lanes. A site-grouped carrier-only model had
   cross-fitted log loss 0.712 and ROC AUC 0.605; the role-specific model had
   log loss 0.577 and ROC AUC 0.826. However, the site-bootstrap interval for
   log-loss improvement was -0.025 to 0.280, crossing zero. The 102-row
   canonical-collapsed sensitivity was directionally similar (0.618 to 0.474
   log loss), but it is not the primary identity-safe population.
4. **Robustness is feature-dependent.** Raw, carrier, coherence, and lag
   retained occurrence-weighted mean frame AUCs of approximately 0.88--0.90 at
   six-pixel coordinate offsets. Consensus fell to 0.731 and multiscale
   persistence to 0.608. Timing shifts of one to three frames had little effect
   on the four strongest features; ten-frame shifts caused larger degradation,
   especially for persistence.
5. **Repeatability can identify stable nuisance structure.** The annulus lanes
   had the highest nonparametric site variance fractions (about 0.87) and
   burst-pair rank correlations (about 0.84--0.85), despite weak temporal
   retrieval. Repeatability is therefore not interchangeable with event utility
   or biological validity. Carrier combined strong retrieval with median
   burst-pair Spearman 0.817, while coherence had a higher site variance
   fraction but lower median pair correlation.
6. **The operators respond to alignment and calcium-like kinetics as expected.**
   Every observed selected-feature AUC exceeded all 500 site-blocked circular
   shifts (empirical upper-tail p = 1/501). In standardized quiet traces, the
   exponential matched filter rose from mean AUC 0.531 at zero injection to
   0.933 at a four-MAD transient; multiscale persistence rose from 0.517 to
   0.852 and raw LS amplitude from 0.511 to 0.762. This validates operator
   sensitivity, not biological realism.

## Feature-engineering implications

- Keep **Raw or the frozen carrier** as the amplitude-preserving event-strength
  lane at a known coordinate.
- Keep **local coherence** as the primary spatial proposal/ranking context. Its
  strong temporal retrieval and positive displaced-control margin agree with
  its prior early-budget known-positive recall gain.
- Keep **lag-2 recurrence** as secondary, explicitly non-causal temporal
  context. It remains robust but its recovery-model ablation contribution is
  small in this dataset.
- Keep the **exponential matched filter** as a weak-transient kinetic operator,
  not a spike estimator. Its synthetic sensitivity is the clearest reason to
  retain it for transfer tests.
- Treat **center-minus-annulus** as a stable contamination/observability
  diagnostic. Its high repeatability but poor retrieval warns against promoting
  features solely because they are stable.
- Treat **representation consensus and multiscale persistence** as exploratory
  contextual/reliability features. They add candidate recovery information in
  point estimates but have not passed the complementarity uncertainty gate.

## Relationship to established methods

Modern calcium-imaging pipelines commonly separate spatial footprints,
temporal fluorescence, background/neuropil terms, and calcium-event dynamics.
The present division of roles is consistent with that modular structure: Raw
amplitude carries event strength; spatial context constrains localization;
annular signals diagnose contamination; and kinetic filters target transient
shape. The result does not make the NeuRev operators equivalent to CNMF,
CaImAn, Suite2p, or OASIS.

Explicit calcium deconvolution methods fit nonnegative or autoregressive
kinetics and are evaluated against spikes or calibrated simulations. Our
matched filter is deliberately simpler and should remain a comparator until
electrophysiological or equivalent ground truth exists. Likewise, realistic
synthetic frameworks such as NAOMi model anatomy, optics, motion, indicator
dynamics, and sensor noise; the present one-dimensional injections test only
operator behavior. A realistic movie-level simulation is therefore the correct
next synthetic validation, not a broader sweep of the current injection grid.

## Remaining automated tests worth doing

The current suite exhausts the highest-value tests available from the existing
labels and arrays without new review. Further automated work should prioritize:

1. movie-level NAOMi-like synthetic demixing with exact spatial and temporal
   truth;
2. injection into multiple real quiet backgrounds and field/noise strata;
3. frozen transfer to any independently acquired recording, even before labels,
   for acquisition-shift and null-calibration checks;
4. motion-field and registration-residual export when those arrays become
   available;
5. calibrated uncertainty or risk--coverage analysis only after the recovery
   target contains more misses or an exhaustively reviewed bounded field.

No additional automated analysis of this same sparse-positive recording can
identify full-field precision, false-positive rate, or one-to-one biological
source identity.

## Artifacts

- Validated run: `Outputs/NeuronIdentifiability/spon_ca_burst_automated_feature_validation_v1`
- Overview figure: `figures/automated_validation_overview.png`
- Robustness figure: `figures/robustness_falsification.png`
- Primary tables: `tables/temporal_retrieval_summary.tsv`,
  `tables/spatial_displacement_summary.tsv`,
  `tables/recovery_model_summary.tsv`, `tables/reliability_summary.tsv`, and
  `tables/null_calibration.tsv`
- Reproducibility: `preflight.json`, `validation.json`, `artifact_index.json`,
  and `status.json`

The failed first serialization attempt was preserved separately as
`spon_ca_burst_automated_feature_validation_v1.failed-serialization`; it is not
a completed or citable result.
