# Spon Ca Burst automated feature validation

## Status and scope

This workflow is a prespecified, no-human-in-the-loop extension of the
canonical-v7 full-trace feature panel. It evaluates measurement behavior within
the same recording. It is not an independent biological replication and does
not turn off-window activity or displaced tissue into a verified negative
class.

## Immutable population and coordinates

- Primary population: all 106 confirmed original-geometry occurrences in the
  canonical-v7 media manifest, keyed by `observation_id` and immutable
  `original_roi_id + x_int + y_int` geometry.
- Sensitivity population: the 102 canonical-collapsed detector rows. The four
  `roi_015` geometries that canonicalize to `roi_010` are never silently
  discarded from the primary population.
- UI frames are one-based and inclusive. Array slices are zero-based and
  half-open. Coordinates use `x = column`, `y = row`.
- Every event interval plus a 15-frame guard is excluded from the quiet
  reference. Quiet reference frames are unlabeled, not biological negatives.

## Prespecified analyses

1. **Temporal retrieval.** For every occurrence and all 11 frozen panel
   features, compare the event frames with guarded quiet-reference frames using
   frame ROC AUC, average precision, recall in the top 1%, 5%, and 10% of
   eligible frames, and reciprocal rank of the first event frame. Primary
   summaries are means of immutable-site means with site bootstrap intervals.
2. **Spatial displacement.** Compare the true center with a deterministic,
   same-field, quiet-baseline-matched translated coordinate at least 12 pixels
   from every labeled center. The primary estimand is the paired difference in
   temporal ROC AUC for raw amplitude, carrier, coherence, lag-2 recurrence,
   consensus, and persistence. Displaced tissue is an unknown control.
3. **Recovery modeling.** Define recovery as matching at budget 58 by any of
   carrier, coherence, or lag-2 recurrence in the original-geometry,
   adjudicated-timing detector audit. Compare a cross-fitted carrier-only model
   with a prespecified multi-feature model. Site-grouped folds prevent the same
   immutable geometry from appearing in train and test. The 102-row
   canonical-collapsed view is a sensitivity analysis. With few misses, all
   recovery-model conclusions remain exploratory.
4. **Perturbation robustness.** Recompute retrieval after coordinate offsets of
   1, 2, 4, and 6 pixels and event-boundary shifts of -10 to +10 frames. Report
   degradation relative to the unperturbed value; no manual exclusions are
   permitted.
5. **Reliability.** For recurrent geometries, report burst-pair Spearman
   correlations and a transparent nonparametric site variance fraction. This
   is not labeled REML ICC.
6. **Falsification and synthetic sensitivity.** Use site-blocked circular time
   shifts to preserve autocorrelation while breaking event alignment. Separately
   inject normalized calcium-like transients into actual guarded quiet traces
   and evaluate LS amplitude, an exponential matched filter, and multiscale
   persistence across amplitude and decay grids. Injection is an operator
   sensitivity assay, not a biological simulator.

## Gates

- Fail before computation if source hashes differ from the validated v7
  manifest, shapes are not `(560, 340, 573)` after alignment, the detector join
  is not exactly `106 x 3`, or the primary recovery target does not contain both
  classes.
- Never overwrite a completed or partial output root.
- Write only to a `.partial` directory until all expected tables, figures,
  validation checks, and the artifact index exist; then atomically rename it.
- Fixed random seed: `20260827`. Bootstrap count: 2,000. Circular-shift null
  count: 500. Logistic models use fixed L2 regularization; the outcome is never
  used to tune features.

## Interpretation gates

- A feature is complementary only if its cross-fitted addition improves log
  loss and the site-bootstrap interval for the improvement excludes zero.
- Spatial superiority requires a positive site-bootstrap interval for true
  minus displaced AUC; otherwise the result is unresolved.
- A perturbation result describes tolerance of this measurement operator, not
  segmentation or neuron-identity accuracy.
- Null and injection assays validate alignment sensitivity and operator
  behavior only. They cannot establish biological specificity.
