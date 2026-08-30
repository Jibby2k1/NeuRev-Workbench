# Spon Ca Burst uncertainty-aware learning plan v1

Status: specified, not yet run.

## Purpose

Use the existing canonical positives and the new single-reviewer candidate labels to test whether interpretable spatial, temporal, kinetic, recurrence, and neighborhood features support a useful neuron-likeness model. The program must preserve label uncertainty, prevent spatial and identity leakage, and distinguish review priority from neuron likelihood.

This is an exploratory learning program. It is not a replacement for bounded-field truth, independent review, or an independent recording.

## Evidence available at specification time

| Evidence tier | Sites | Occurrences | Permitted use |
| --- | ---: | ---: | --- |
| Canonical confirmed | 50 | 106 | Positive reference; retain immutable identities and site grouping |
| New definite | 9 | 19 | Provisional positive in sensitivity analyses |
| New probable | 4 | 4 | Positive only in inclusive analyses |
| New uncertain | 4 | 5 | Exclude from primary fitting; alternate labels only in sensitivity tests |
| New unlikely/artifact-or-noise | 1 | 1 | Case study, not a sufficient negative class |

The 18 new sites were selected from detector outputs. They cannot be used to estimate full-field precision, and evaluation against the same selection machinery is subject to ascertainment bias.

## Scientific questions

1. Which feature families are stably associated with confident neuron calls?
2. Can positive-unlabeled learning rank candidate neuron likeness without inventing verified negatives?
3. Do conclusions survive alternative treatment of probable and uncertain labels?
4. Does performance survive identity-, neighborhood-, burst-, morphology-, and confidence-grouped holdouts?
5. Are learned scores stable to coordinate, patch, annulus, timing, lag, and noise perturbations?
6. Which features fail specifically for small, low-SNR, crescent, crowded, artifact-adjacent, and single-burst sites?
7. Can review value and neuron likelihood be modeled as separate objectives?

## Work packages

### WP1: Harmonized site table

Create one provenance-preserving table joining immutable site identity, occurrence membership, label tier, reviewer confidence, morphology observations, and frozen feature values. Repeated bursts remain nested within sites. Missing features remain missing and are never silently zero-filled.

Required outputs:

- `site_feature_labels.tsv`
- `occurrence_feature_labels.tsv`
- `feature_dictionary.json`
- `provenance.json`
- validation proving row counts, identity uniqueness, and label-source boundaries

### WP2: Association and stability audit

For each prespecified feature and feature family, report robust effect sizes, exact or site-blocked permutation tests, bootstrap intervals, multiplicity-adjusted exploratory probabilities, missingness, and sign stability. This work package describes association; it does not select a deployable classifier.

### WP3: Positive-unlabeled and positive-only models

Compare simple, inspectable methods:

- robust positive-reference distance;
- one-class SVM;
- isolation forest;
- bagged positive-unlabeled logistic regression;
- non-negative positive-unlabeled risk, if its class-prior assumptions can be stated and stress-tested.

All outputs are neuron-likeness scores. No threshold may be called calibrated biological probability without representative verified negatives.

### WP4: Label-sensitivity benchmark

Fit regularized logistic regression, elastic net, shallow gradient boosting, and a regularized discriminant model under frozen label policies:

- strict: canonical plus new definite are positive; probable and uncertain excluded;
- inclusive: definite and probable are positive; uncertain excluded;
- pessimistic sensitivity: uncertain treated as negative;
- optimistic sensitivity: uncertain treated as positive;
- canonical-only fit with all new candidates held out.

A conclusion is robust only if its direction and practical interpretation survive the prespecified policies.

### WP5: Leakage-resistant validation

Random row-level cross-validation is prohibited. Compare grouped schemes:

- site-grouped folds for repeated occurrences;
- spatial-neighborhood holdout;
- leave-one-burst-out;
- leave-one-morphology-or-limitation-out;
- leave-one-confidence-tier-out;
- leave-crowded-cluster-out.

Any spatially or identity-linked observations must remain in the same fold. Preprocessing, feature selection, and calibration occur within each training fold.

### WP6: Feature ablation and stability selection

Measure the incremental value of amplitude/SNR, kinetics, cross-stage agreement, spatial specificity, morphology, recurrence, and neighborhood/artifact context. Use site bootstraps to report coefficient sign, selection frequency, rank stability, and degradation when each family is removed.

### WP7: Counterfactual robustness

Recompute scores after frozen perturbations of center position, patch radius, annulus definition, trace normalization, burst boundaries, temporal lag, injected noise, and neighbor masking. Desired behavior is smooth degradation around a stable source. Score migration to a nearby stronger source is an identity failure, not successful robustness.

### WP8: Hard-subgroup reporting

Report performance separately for small/discreet, low-SNR, crescent/non-circular, possibly multiple, artifact-adjacent, stronger-neighbor-adjacent, single-burst, and recurrent sites. Small groups require case-level plots and intervals, not broad population claims.

### WP9: Two-score architecture

Maintain two explicitly different targets:

- **Neuron likeness:** similarity to confident neuronal measurements.
- **Review value:** uncertainty, novelty, crowding, disagreement, and expected information gain.

The current candidate-priority score is a review-value prototype. Its AUC of 0.308 for likely versus unresolved/unlikely calls confirms that it must not be relabeled as neuron probability.

### WP10: Simulator-to-biological transfer

Fit feature weights on development simulator families, freeze them, and evaluate once on the biological review set. Separately test biological-positive feature relationships on held-out simulator families. This is a mechanism-transfer diagnostic, not independent biological validation.

## Primary automated gates

1. **Identity gate:** no site or spatial cluster crosses training and evaluation folds.
2. **Label gate:** uncertain sites are excluded from primary fitting and never silently made negative.
3. **Leakage gate:** all transforms and feature selection are fit inside training folds.
4. **Sensitivity gate:** important feature direction is stable under strict and inclusive policies.
5. **Stability gate:** selected features recur across site bootstraps and perturbations.
6. **Baseline gate:** learned scores must improve over frozen amplitude and spatial-context baselines under grouped evaluation.
7. **Abstention gate:** coverage-risk curves must be reported; difficult sites may remain unresolved.
8. **Interpretation gate:** no result is called precision, biological identity, or generalization.

## Recommended implementation sequence

Implement WP1–WP8 as one deterministic `uncertainty_aware_learning_v1` package. Freeze inputs and policies in a manifest, make every work package independently resumable, and write tables before composite figures. WP9 should reuse those outputs. WP10 should remain a separate prospective transfer run so simulator tuning cannot leak into the biological evaluation.

## Decision rules after the run

- Promote a compact feature family only when it is directionally stable across label policies, grouped folds, bootstraps, and perturbations.
- Retain a feature as a review-priority feature when it surfaces difficult or novel cases but does not rank neuron likeness.
- Reject models whose apparent benefit disappears under spatial grouping or uncertain-label sensitivity.
- Stop model expansion if no simple method beats frozen baselines; prioritize new truth rather than increasing model complexity.

## Relationship to reinforcement learning

This program uses supervised sensitivity analysis, positive-unlabeled learning, anomaly detection, and active-review prioritization. Reinforcement learning is not currently justified because there is no validated sequential action, environment transition, and reward contract. A future adaptive acquisition or review-policy problem could introduce that structure, but it should not be retrofitted into the present classification task.
