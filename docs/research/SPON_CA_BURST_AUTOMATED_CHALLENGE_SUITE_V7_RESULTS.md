# Automated challenge suite v7

## Scope

This suite executes all twelve proposed automated test families with exact simulator truth and six seeds per condition. Weak-neighbor recovery uses one-to-one identity assignment, preventing a strong neighboring source from being credited as recovery of the weak identity. Every family has a separate TSV and machine-readable summary.

## Results by test family

1. **Shift-aware null normalization.** Movie-wise robust normalization improved mean challenge F1 from 0.505 to 0.523. This is a positive but modest result.

2. **Identity and footprint suppression.** Point, four-pixel footprint, and trace-aware suppression all produced mean recall 0.514, duplicate rate 0.056, and F1 0.343. The methods were equivalent after local-maximum proposal formation; suppression is not established as solved.

3. **Source-count-agnostic stopping.** BH-FDR stopping selected approximately 9.7, 9.8, and 10.5 proposals for movies containing 2, 4, and 8 sources. Mean F1 was only 0.299. The current empirical-null p-values do not provide acceptable deployable stopping.

4. **Weak-neighbor frontier.** The minimum weak-to-strong amplitude ratio reaching 80% identity-resolved recovery was 1.0 at separations 3, 5, and 7 px, and 0.75 at 10 px. This formalizes the ROI-27-like failure: proximity and amplitude imbalance create a severe identifiability boundary.

5. **Empirical noise spectrum.** Adding AR(1), spatially correlated noise reduced spatial-context pixel AUC from 0.831 to 0.806 and top-eight recall from 0.833 to 0.708. A scalar white-noise mapping is optimistic.

6. **Motion–neuron disentanglement.** Mean F1 was 0.583 without the added motion artifact and 0.556 for both asynchronous and burst-synchronized motion. The tested motion contamination causes modest degradation, but this intervention does not establish full motion robustness.

7. **Morphology interventions.** Mean spatial-context pixel AUC was 0.965 for ellipses, 0.817 for crescents, 0.850 for rings, and 0.963 for fragmented two-lobe footprints. Crescent geometry is the clearest morphology-specific weakness, consistent with the visual miss review.

8. **Severity curves.** Recall first fell below 0.70 at severity 4.0 for colored noise, 1.0 for structured stripes, and 2.0 for impulse transients. Structured background is the earliest tested failure boundary.

9. **OOD detection.** A five-statistic covariance score separated the deliberately shifted challenge movies from baseline with AUC 1.0. This passes the gate but is probably optimistic because the shifts are stylized and strong.

10. **Cross-simulator validation.** Frozen spatial-context thresholds produced F1 0.795 in the primary simulator and 0.949 in an independently implemented Poisson/elliptical simulator. The transfer gate passed, but the second simulator is evidently easier; this does not prove broad realism.

11. **Seed and parameter stability.** Across NMS windows 3, 5, and 7 and matching radii 2, 3, and 4 px, F1 ranged from 0 to 0.5 with SD 0.167. Conclusions are not parameter-robust under compound shift.

12. **Negative controls.** Spatial shuffle and random projections were near chance (AUC 0.500 and 0.484). However, applying the fixed kinetic operator to time-reversed movies retained AUC 0.731. The kinetic score is not direction-selective and fails this falsification criterion.

## Headline gates

Four of five compact headline gates passed: robust null normalization improved F1, duplicates remained below 0.1 in the suppression diagnostic, OOD AUC exceeded 0.8, and cross-simulator F1 exceeded 0.4. The negative-control gate failed because the reversed-time kinetic control remained predictive.

Passing four gates does not mean that ten of twelve problems are solved. Source-count stopping, close weak-neighbor identity, parameter stability, spectral noise realism, and kinetic directionality remain substantive automated-development targets.

## Decision

The next automated development should focus on three components:

1. Parameter-robust source-free stopping, building on the modest robust-normalization gain but replacing current BH p-values.
2. Pre-proposal source separation or joint identity assignment for close, unequal-amplitude neurons; post-proposal suppression alone is insufficient.
3. Directional kinetic features that distinguish causal calcium-like rise/decay from reversed traces.

No result in this suite establishes independent biological validation or bounded-field precision.
