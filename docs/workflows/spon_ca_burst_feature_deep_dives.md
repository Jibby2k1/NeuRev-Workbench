# Spon Ca Burst feature deep dives

## Scope

This is a prespecified secondary analysis of the canonical-v7 106-occurrence,
50-geometry population. It introduces no new labels or model proposals. The
frozen detector lanes, event windows, coordinates, and upstream feature values
are reused by hash.

## Analyses

1. Classify every B58 occurrence as recovered by all lanes, partially
   recovered, or missed by all lanes with its exact upstream failure classes.
2. Contrast native-proposal recall gains with ranking gains on the frozen
   common proposal universe at budgets 20, 40, 58, 80, and 100. The contrast is
   descriptive and is not treated as an additive causal decomposition.
3. Relate radius-0-to-6 retrieval decay to distance and counts of other labeled
   centers. Duplicate burst occurrences at an identical site are not spatial
   neighbors.
4. Evaluate a fixed 15-member rise/decay kinetic bank at exact LS centers.
   Kernel selection is leave-one-burst-out.
5. Compare carrier, coherence, lag, and role-specific recovery models over 20
   repeated site-grouped five-fold splits. With only 12 all-lane misses, all
   prediction comparisons remain exploratory.
6. Audit coherence-window and lag variants in the frozen native detector screen
   and quantify lag variation remaining after carrier, coherence, and burst.

## Gates

- The detector join must be exactly `106 x 3`, with 12 all-lane B58 misses.
- Site identities may not cross training and test folds.
- The kinetic bank is fixed at rises `{1,2,4}` and decays `{3,5,10,20,40}`.
- The completed output root is never overwritten; work is written to a partial
  root and atomically promoted after validation.
- Sparse positives do not identify precision, specificity, or false-positive
  rate. Failure classes are pipeline outcomes, not biological mechanisms.
- Because this analysis creates no new proposal or label imagery, it reuses and
  programmatically revalidates the exact frozen detector visual-audit inventory.

## Run

```bash
python -m neurobench.experiments.neuron_identifiability.deep_dive_suite \
  --output-root Outputs/NeuronIdentifiability/spon_ca_burst_feature_deep_dives_v2
```
