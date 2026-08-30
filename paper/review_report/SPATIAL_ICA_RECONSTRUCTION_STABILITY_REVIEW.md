# Spatial ICA reconstruction stability review

## Result

The corrected signed-signal analysis at `19_spatial_ica_reconstruction_stability_v2` passes all five prespecified gates. It reconstructed the saved reference model, three new sampling/FastICA seeds, and four models fit after excluding one burst window. Seed refits were evaluated across all bursts; leave-one-burst-out fits were evaluated only on the excluded burst.

- Sampled-pixel correlation: 0.99847--0.99957 (gate >= 0.90).
- Median event-map correlation: 0.99832--0.99923 (gate >= 0.90).
- Median morphology-profile Spearman correlation: 0.99481--1.00000 (gate >= 0.70).
- Median labeled-site peak error: 0 frames for every refit (gate <= 1).
- Seed fixed-budget known-positive-recall change: -0.0125 to +0.01087; every excluded-burst recall exactly matched its reference burst (gate absolute change <= 0.05).

The reference fixed-budget mean recall is 0.6713768, exactly restoring the original spatial-screen detection contract. The superseded v1 output positive-clipped the signal before detection and must not be used.

## Interpretation

The dense Wiener reconstruction is highly stable even though its individual rank-12 ICA filters are not. This is mathematically plausible because different bases can span and reconstruct nearly the same signal subspace. The paper may describe spatial ICA as a stable reconstructed representation within this recording, but it must not name individual filters, assign them anatomy, or treat stability as identity, precision, or cross-recording generalization.

The original screen still fails its broader exact semi-synthetic preservation audit, and the human Raw/ICA/combined morphology review remains deferred. Those limitations prevent broad morphology promotion but do not negate this reconstruction-stability result.
