# Spatial ICA class and certainty alignment review

## Result

The post-freeze analysis at `20_spatial_ica_class_certainty_alignment_v3` is complete. It does not refit the three measurement-event classes.

### Protected-v1 class profiles

Fifty-three one-to-one matched observations collapse to 17 immutable observation sites: 7 dominant Class 1, 1 dominant Class 2, and 9 dominant Class 3 sites. With only one Class-2 site, protected three-class inference is prohibited. Raw contrast and boundary sharpness were nominally associated with class but missed BH correction (both q = 0.0588) and partly overlap the Raw spatial-specificity feature used to construct the classes. They are descriptive, not independent validation. No protected spatial-ICA metric approached corrected significance.

### Canonical-v7 class sensitivity

The candidate-assisted sensitivity contains 24 canonical identities: 3 dominant Class 1, 5 Class 2, and 16 Class 3. Spatial-ICA boundary sharpness showed the strongest separation (epsilon-squared = 0.323, nominal p = 0.00485), with medians -0.022, 0.018, and 0.263 for Classes 1--3. It did not survive the 14-test family correction (BH q = 0.0679). No other Raw or spatial-ICA class metric survived correction.

This is suggestive evidence that Class 3 may have a sharper spatial-ICA center-to-boundary transition than Classes 1--2 in the candidate-assisted atlas. It is not validated class separation.

### Certainty sensitivity

At canonical-identity grain, confirmed identities had dominant-class counts 3/2/10 and uncertain identities 0/3/6 for Classes 1/2/3. The effect was moderate (Cramer's V = 0.346) but not significant under identity-label permutation (p = 0.313).

After adjusting each morphology metric for frozen class and burst and aggregating by canonical identity, none of 14 Raw/spatial-ICA morphology tests distinguished 15 confirmed from 9 uncertain identities after BH correction. Therefore uncertainty appears descriptively related to class composition, especially absence of Class 1 and greater Class-2 representation, but no independent objective morphology signature is established.

## Claim boundary

The defensible conclusion is **suggestive profile structure without validated class or certainty separation**. Classes remain measurement-event classes, not neuron types. Candidate-assisted v7 results are sensitivity evidence, not independent validation. No precision, specificity, anatomy, wiring, or causal claim follows.
