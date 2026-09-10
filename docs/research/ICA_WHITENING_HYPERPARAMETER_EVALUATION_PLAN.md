# ICA and Whitening Hyperparameter Evaluation Plan

**Status:** Execution authorized and active; completion remains gated by exact
coverage, protected confirmation, visual inspection, and independent-recording
confirmation.

## Purpose

Evaluate temporal, spatial, and joint spatiotemporal ICA under a robust,
matched whitening and hyperparameter design. The program has three primary
questions:

1. How does each configuration perform with respect to the sparse known-positive
   labels?
2. What do its selected hyperparameters and learned parameters mean?
3. How stable, signal-preserving, and interpretable are its outputs?

Use one umbrella experiment with a shared fit registry, common folds, paired
seeds, and reusable fitted models. Keep label utility, operator meaning, signal
integrity, and human interpretability as separate claims rather than combining
them into one "best ICA" score.

## Model families

- **Temporal ICA:** adjacent-frame and multi-lag/delay-embedding operators.
- **Spatial ICA:** patch-fitted and dense translation-shared spatial operators.
- **Joint spatiotemporal ICA:** a genuinely learned 3D space-time operator.
- **Separable control:** spatial-to-temporal and temporal-to-spatial cascades.
- **Tensor analysis:** downstream population characterization, not a substitute
  for joint spatiotemporal ICA.

Required matched controls are Raw Direct, signed temporal difference,
rank-matched PCA, a rank-matched random rotation of the whitened subspace, and
the separable construction for joint ICA.

## Whitening as a first-class factor

Record the following independently; do not collapse them into one whitening
label:

1. **Support geometry:** none/control, spatial, temporal, separable
   spatial-to-temporal, separable temporal-to-spatial, or joint spatiotemporal.
2. **Covariance-fit scope:** globally shared and regional/local-block fitted.
   Truly per-location adaptive covariance is not implemented in v1 and must not
   be inferred from the regional scope. Protected finalist refits exclude the
   held-out event interval.
3. **Operator support:** spatial width, temporal width, and joint space-time
   dimensions.
4. **Whitening strength and regularization:** fractional exponent, covariance
   shrinkage, eigenvalue floor, and Raw-preserving blend.
5. **Whitening semantics:** centering, normalization, retained rank, causality,
   boundary handling, and whether ICA performs any additional whitening.

Distinguish a globally fitted local-support convolution from a genuinely local
or adaptive covariance fit. Avoid redundant external whitening followed by an
unrestricted internal ICA re-whitening step. Every fit must export the external
operator, ICA whitening, demixing, and their composite effective transform.

## Experimental design

Use a **conditional Cartesian product** for scientifically meaningful
discrete/ordinal factors:

- ICA family and objective;
- whitening geometry and covariance-fit scope;
- causal versus centered temporal operation;
- ICA rank;
- spatial and temporal support size;
- separable application order; and
- valid whitening/ICA application order.

Exclude structurally invalid combinations rather than encoding them as missing
cells. Joint whitening must obey a preregistered conditioning and resource gate;
its feature dimension is approximately `spatial_width^2 * temporal_width`.

Within each compatible discrete cell, use a scrambled Sobol design for
continuous factors:

- spatial and temporal whitening exponents;
- shrinkage and eigenvalue-floor ratios;
- Raw-preserving blend;
- objective bandwidth/kernel scale;
- regularization and component shrinkage; and
- justified adaptation-rate parameters.

Sample scale parameters on a log scale. Use separate Sobol coordinates for
spatial and temporal parameters in separable models. Add deterministic anchors
because Sobol samples do not guarantee boundaries:

- no and full whitening;
- no and full Raw skip;
- minimum and maximum regularization;
- current canonical/default settings; and
- PCA/ZCA and random-rotation controls.

The initial target is 32 Sobol interior points per compatible cell plus anchors;
64 points may be used for high-dimensional cells after resource preflight.
Use identical folds and paired fit seeds wherever models are comparable.

## Staged execution

1. **Freeze contracts:** data hashes, coordinates, frame intervals, folds,
   candidate construction, model definitions, grids, seeds, and output schema.
2. **Numerical screen:** run the complete discrete design with Sobol interiors
   and anchors; reject only nonfinite, unresolved, or contract-invalid fits.
3. **Truth-known characterization:** measure recovery, crosstalk, reconstruction,
   and abstention on identifiable and deliberately unidentifiable synthetic and
   semi-synthetic mixtures.
4. **Frozen real-data evaluation:** after all fits and selection rules are
   frozen, calculate descriptive label metrics for every valid configuration.
5. **Finalist confirmation:** refit a small Pareto set over additional seeds and
   leave-one-burst-out folds using nested or label-free selection.
6. **Scientific audit:** generate and validate the complete expert-only,
   model-only, and matched-comparison evidence set for promoted finalists.
7. **Independent confirmation:** apply the frozen rule to another recording or
   fish before making cross-recording or biological-source claims.

## Evaluation

### Sparse-label utility

- Known-positive recall at fixed candidate budgets.
- Recall-versus-candidate-burden curves.
- Spatially grouped out-of-fold positive-unlabeled metrics.
- Known-positive rank and reciprocal-rank summaries.
- Localization and peak-frame error.
- Peak and event-area retention.
- Per-burst, canonical-identity, certainty, and morphology-stratum results.

Unmatched candidates remain unknown, not negative. Do not report ordinary
precision, specificity, false-positive rate, or conventional ROC-AUC without an
exhaustively adjudicated bounded field.

### Parameter and response interpretation

For temporal linear operators, export whitening-only, ICA-conditional, and
composite magnitude, phase, and group-delay responses. Summarize DC gain,
frequency centroid, bandwidth, band-energy fractions, effective delay, and
similarity to common-mode and signed-difference kernels.

For spatial operators, export filters and 2D spectra; summarize radial frequency,
orientation, anisotropy, DC suppression, center-surround balance, compactness,
radius, and center offset.

For joint operators, export 3D space-time responses and informative slices;
summarize spatial/temporal bandwidth, group delay, motion/speed sensitivity,
and the residual from the best separable approximation.

Also export covariance eigenvalues before and after regularization, whitening
gain curves, condition number, effective rank, component energy, mixing and
demixing matrices, and exact composite transforms. Interpret individual
components only after permutation/sign alignment and a passed stability gate;
otherwise report permutation-invariant subspace and reconstruction summaries.

### Integrity, stability, and interpretability

- Convergence, finite values, conditioning, eigenvalue-floor activation, and
  analysis/synthesis closure.
- CPU/CUDA parity, runtime, memory, and failure status.
- Raw/output correlation, NMSE, amplitude and area retention, temporal lag,
  centroid displacement, morphology inflation, and residual leakage.
- Approximate real-data SNR using declared event/quiet robust ratios and
  ROI-versus-annulus controls; reserve true SNR, SDR, and source recovery for
  truth-known fixtures.
- Zero-lag, lagged, cross-burst, ROI-annulus, component-residual, and
  frequency-domain coherence measures with fixed windows and null controls.
- Seed, split-half, perturbation, and leave-one-burst-out component/subspace and
  reconstructed-output stability.
- Temporal-shuffle, phase-randomization, spatial-translation, label-shift, and
  rank-deficient-mixture controls.
- Blinded Raw-only versus ICA-only versus Raw-plus-ICA review of boundary
  visibility, neuronal-shape visibility, confidence, review time, and
  disagreement.

## Statistical analysis and selection

- Use complete sites/canonical identities and leakage-connected observations as
  grouping units; do not split by frames or patches.
- Pair seeds and folds across methods and bootstrap at site and burst levels.
- Estimate factorial main effects and interactions for the Cartesian factors.
- Use within-cell response surfaces and rank-sensitivity summaries for continuous
  factors. Formal Sobol indices are not identified by this scrambled point design
  because it is not a Saltelli A/B/A-B construction.
- Report effect sizes, uncertainty, fold consistency, failures, and abstentions;
  do not run an isolated hypothesis test for every configuration.
- Treat the full label-response surface as exploratory. Use nested blocked
  selection or a label-free rule before protected confirmation.
- Select finalists from a Pareto surface spanning label utility, candidate
  burden, preservation, stability, interpretability, and computational cost.

## Required artifacts and completion gates

Each run must write a resolved manifest, fit registry, model/input hashes,
fold/seed identities, numerical status, parameter tables, label metrics,
integrity/stability metrics, compact response summaries, and an LLM-efficient
artifact index. Completed computation is not scientific success.

A configuration may advance only if it is numerically resolved, reproducible,
signal-safe on truth-known controls, and competitive with its matched baseline.
Individual filters or factors require their own stability gate. A promoted
finalist must pass the repository scientific-audit output standard, and no
within-recording result establishes independent neural-source identity or
cross-recording generalization.
