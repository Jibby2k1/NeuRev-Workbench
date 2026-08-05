# MSLN/MSICA Broad Cascade v4 Results

## Scope

The completed GPU program tested six architecture families without scientific
early stopping:

1. `Raw -> MSLN1 -> MSICA1 -> MSLN2`;
2. the same original order with `MSICA2` appended;
3. `Raw -> MSICA1 -> MSLN -> MSICA2` per branch;
4. a five-seed switched-order energy ensemble;
5. cross-branch MSICA; and
6. parallel original/switched fusion controls.

The original-order screen included all 30 first-stage contexts, all 900 ordered
context pairs, both persistence and innovation branches, and five MSICA2
bandwidths: 1,800 branch combinations and 9,000 second-stage fits. The other
families covered all 30 contexts. Labels were unavailable during tuning and
three finalists per experiment were frozen by label-free event/quiet contrast.

## Primary and exploratory results

The fixed guardrail uses 58 candidates per burst, or 232 total, against 79
known sparse-positive labels. Raw Direct is the external reference at 49/79
(`0.6056`).

| Architecture | Label-free rank 1 | Protected best of 3 |
| --- | ---: | ---: |
| Original shallow cascade | 39/79 (`0.4937`) | 42/79 (`0.5316`) |
| Original deep cascade | 41/79 (`0.5190`) | 41/79 (`0.5190`) |
| Switched deep cascade | 47/79 (`0.5949`) | 57/79 (`0.7215`) |
| Five-seed switched ensemble | **52/79 (`0.6582`)** | 59/79 (`0.7468`) |
| Cross-branch MSICA | 46/79 (`0.5823`) | 46/79 (`0.5823`) |
| Parallel fusion | 51/79 (`0.6456`) | 53/79 (`0.6709`) |

The label-free rank-1 column is the defensible primary comparison. Protected
best-of-three values are exploratory ceilings after labels were opened and
must not be presented as unbiased winner estimates.

## Interpretation

The original `MSLN -> MSICA -> MSLN [-> MSICA]` cascades did not improve on
Raw Direct. Reordering exposed stronger candidates, but the single-seed
label-free selector did not select the protected high-recall configurations.
Five-seed energy aggregation produced the best label-free result, a provisional
three-match gain. Parallel fusion produced a smaller two-match gain.

Individual ICA fits remained angle/swap unstable. Nevertheless, the three
five-seed finalist energy maps had mean cross-seed correlations of `0.978`,
`0.919`, and `0.889`. Aggregation therefore improved representation-level
stability without establishing stable component identity or biological source
interpretation.

## Diagnostics and validation

- finalist fits: 4,096 screen samples and 16,384 confirmation samples;
- 16 blocked bootstraps and FastICA sensitivity for supported ICA blocks;
- CUDA two-stage parity maximum absolute error: `1.43e-6`;
- worst-context CUDA preflight allocation: 6.71 GB under the 8 GiB cap;
- final artifacts: 18 float32 maps of shape `560 x 340 x 573`;
- videos: six experiment videos plus two architecture-comparison videos, all
  560 frames and 56 seconds at 10 fps;
- focused order/cascade/CUDA suite: 8/8 passed.

Generated maps and videos remain ignored under `Outputs/`. Their local roots
are:

```text
Outputs/HierarchicalParzenICA/spon_ca_burst_msln_msica_cascade_program_v4
Outputs/HierarchicalParzenICA/spon_ca_burst_msln_msica_order_visual_comparison_v1
```

The comparison renderer uses neutral grayscale for raw amplitude,
orange-tinted grayscale for original-order energy, and green-tinted grayscale
for switched-order energy. Derived panels use per-lane quiet-p99 normalization
and one shared fixed display cap; no red/white/blue diverging scale is used.

## Next discussion

See
[`MSLN_MSICA_ALTERNATIVE_OBJECTIVE_EXPERIMENT_BRIEF.md`](MSLN_MSICA_ALTERNATIVE_OBJECTIVE_EXPERIMENT_BRIEF.md)
for the proposed literal objective-function comparison: CS-Parzen, KSG mutual
information, normalized HSIC, matrix-Renyi mutual information, multi-lag
dependence, and a quiet-relative Jensen--Shannon composite.
