# Label-free false-alarm and gated-fusion benchmark v5

## Prospective contract

The generator families were separated before fitting:

- Development and null calibration: baseline, dense neuropil, bleaching.
- Untouched evaluation: nonrigid motion, empirical noise, compound shift.

A balanced logistic gate used carrier, spatial-context, and kinetic scores at the union of their local maxima. Gate labels came only from exact simulator truth in development families. Proposal thresholds did not use source labels: they were selected exclusively from zero-source development movies to target 1, 2, or 4 false alarms per movie. The gate and thresholds were then frozen.

## Untouched-family operating results

| Nominal null target | Model | Proposals | Precision | Recall | F1 | Duplicates/movie |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Spatial context | 3.26 | 0.749 | 0.514 | **0.586** | 0.148 |
| 1 | Gated fusion | 2.74 | **0.791** | 0.449 | 0.528 | 0.389 |
| 2 | Spatial context | 5.17 | 0.549 | **0.590** | **0.5450** | **0.222** |
| 2 | Gated fusion | 4.94 | **0.603** | 0.563 | 0.5441 | 0.870 |
| 4 | Spatial context | 8.46 | 0.412 | **0.694** | 0.493 | **0.037** |
| 4 | Gated fusion | 7.69 | **0.436** | 0.657 | **0.499** | 0.981 |

At the prespecified primary target of two, gating did not improve aggregate F1. It traded recall for precision and promoted many more duplicate candidates. Actual false positives averaged 2.41 per evaluation movie for spatial context and 2.26 for gated fusion, demonstrating imperfect transfer of the nominal development-null rate.

## Family-specific result at target two

| Held-out family | Spatial-context F1 | Gated-fusion F1 | Interpretation |
|---|---:|---:|---|
| Nonrigid motion | **0.744** | 0.676 | Gate degrades a strong spatial backbone and creates duplicates. |
| Empirical noise | 0.566 | **0.626** | Gate's precision improvement is useful in this family. |
| Compound shift | 0.326 | **0.331** | Both remain weak; the difference is small. |

## Decision

The current gate is rejected as a replacement for spatial context. This is a useful prospective negative result: an apparent calibration advantage in the earlier retrospective family analysis does not translate into improved primary F1 after disjoint fitting and label-free threshold selection.

Spatial context remains the frozen proposal backbone. The next automated work should address the two diagnosed failure modes directly: normalize null scores in a shift-aware but label-free manner, and suppress adjacent candidates at an identity or footprint level before applying any fusion score. No fusion weights should be tuned on the three evaluation families reported here.
