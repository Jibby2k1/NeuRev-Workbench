# Spon Ca Burst feature deep dives v2

## Result

The suite passed all population, join, leakage, numerical, figure, and reused
scientific-audit gates for 106 occurrences at 50 immutable geometries.

- At B58, 83 occurrences were recovered by all three quantitative lanes, 11
  by only a subset, and 12 by none. The all-lane misses comprised five uniform
  localization misses, four uniform identity conflicts, one ranking miss, one
  NMS-suppressed miss, and one mixed ranking/NMS miss.
- At B20, coherence improved native-proposal recall over carrier by 0.064, but
  improved common-proposal ranking recall by only 0.011. Lag-2 improved the two
  recall contrasts by 0.081 and 0.011. Most of the early-budget advantage is
  therefore associated with the native-versus-common proposal contrast rather
  than ranking alone, without claiming an additive causal decomposition.
- The best fixed kinetic kernel was a one-frame rise with 20-frame decay
  (site-weighted frame AUC 0.893). It was selected in every leave-one-burst-out
  fold; the unweighted mean across the four held-burst AUCs was 0.915.
- Across 20 repeated site-grouped splits, carrier-only recovery log loss was
  0.710. Carrier plus lag was 0.685; the compact carrier/coherence/lag model was
  0.691; and the broader role-specific model was 0.536. These are exploratory
  point estimates because only 12 occurrences were missed by all lanes.
- Recovered-minus-missed center AUC differences were positive for carrier,
  coherence, and lag, but every site-bootstrap 95% interval crossed zero.
  Temporal signal strength alone therefore does not resolve the failure modes.

## Feature-engineering decision

Retain amplitude carrier, local coherence, lag-2 recurrence, and a simple
calcium-kinetic bank as separate roles. The proposal-universe analysis argues
against treating coherence or lag solely as rerankers. The broad recovery model
is useful for hypothesis generation, but should not become a promoted selector
until it passes bounded-field or independent-recording validation.

The data do not support precision, false-positive, biological source-identity,
or causal propagation claims. The next decisive tests still require exhaustive
bounded-field labels, a new recording, motion fields, or realistic movie-level
simulation.

## Artifacts

- Run: `Outputs/NeuronIdentifiability/spon_ca_burst_feature_deep_dives_v2`
- Overview: `figures/deep_dive_overview.png`
- Spatial analysis: `figures/spatial_decay_crowding.png`
- Primary tables: `failure_taxonomy_summary.tsv`,
  `proposal_ranking_decomposition.tsv`, `kinetic_bank_lobo_selection.tsv`, and
  `recovery_complementarity.tsv`
