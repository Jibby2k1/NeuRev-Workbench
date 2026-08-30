# Targeted automated development v8

## Objective

This prospective package attempted to repair the three principal failures identified by the v7 challenge suite:

1. Parameter-sensitive source-free stopping.
2. Identity-resolved recovery of close unequal sources.
3. Lack of kinetic direction selectivity.

The package uses six seeds, new artifact definitions for stopping, one-to-one identity assignment for weak-source recovery, and separate utility and reversed-control gates for kinetics.

## Parameter-consensus stopping

A robust spatial-context threshold was calibrated from zero-source baseline, dense-neuropil, and bleaching development movies. Evaluation used previously unreported checkerboard, low-frequency flicker, and mixed-artifact definitions.

| Method | Mean F1 | False positives/movie | Match-radius F1 range | Match-radius F1 SD |
|---|---:|---:|---:|---:|
| Single window 5 | 0.5680 | 0.0556 | 0.0417 | 0.0182 |
| Three-window consensus | 0.5680 | 0.0556 | 0.0417 | 0.0182 |

The methods are exactly equivalent because candidates above the conservative source-free threshold already persist across windows 3, 5, and 7. Consensus adds complexity without changing the operating set.

## Pre-proposal rank-two NMF

The close-source challenge varied separation 3, 5, and 7 px; weak-to-strong amplitude ratio 0.2, 0.4, 0.6, and 1.0; and temporal correlation 0 and 0.8. Both methods produced two proposals and were scored with one-to-one identity matching.

| Method | Weak-identity recovery | Mean F1 |
|---|---:|---:|
| Frozen spatial context | **0.354** | **0.580** |
| Rank-two NMF | 0.069 | 0.354 |

NMF fails even though its rank is favorably fixed to the true source count. Unconstrained nonnegative variance decomposition is therefore rejected as the close-source remedy.

## Directional kinetic contrast

| Method | Forward-movie AUC | Reversed-movie AUC | Forward-minus-reversed margin |
|---|---:|---:|---:|
| Current kinetic score | 0.699 | 0.715 | -0.016 |
| Forward minus reversed kernel | 0.477 | 0.523 | -0.045 |

Directional subtraction brings the reversed control near chance, but only by destroying forward detection. Inspection confirmed that this is not a mechanical kernel-orientation error: individual true sources have heterogeneous positive and negative asymmetries, so subtraction cancels useful evidence.

## Decision

Only one of five prespecified gates passed: the reversed directional AUC fell below 0.55. The corresponding forward-utility gate failed, so none of the three proposed replacements is promoted.

- Retain the simpler single-window robust stopping baseline.
- Reject unconstrained rank-two NMF for close-source identity recovery.
- Reject simple forward-minus-reverse kinetic subtraction.

Further automated development should use new conditions and stronger models: a joint spatial-temporal generative deblender with explicit footprint and calcium constraints, and source-specific kinetic likelihood ratios rather than a field-wide subtraction. These v8 evaluation results must not be reused for tuning those replacements.
