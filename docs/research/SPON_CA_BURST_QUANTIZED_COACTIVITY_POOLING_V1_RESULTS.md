# Spon Ca Burst Quantized Coactivity Pooling v1

## Answer

Neither coarse quantization nor threshold-conditioned max/min pooling improved
the protected benchmark. The unchanged float Global Norm (GN) representation
remains the authority. The original hard max/min rule is rejected because it
merged spatial structure and reduced the frozen operating point from 54/79 to
24/79 known matches.

The only plausible follow-up is a soft, causal, high-quantile neighborhood
support term that preserves the center pixel. One such lane reached 56/79 post
hoc with 0.989 correlation to float GN, but the integrity-constrained protected
panel reached only 51/79. This is a hypothesis for independent confirmation,
not evidence of improvement.

## Explicit design

The experiment used the preferred `Raw -> MSICA -> MSLN -> GN` representation
with alpha 0.5 and zero-anchored global movie scaling. It evaluated 325 GPU
lanes:

- one float GN control;
- 24 quantization-only lanes: 8, 16, 32, or 64 levels; uniform,
  square-root-companded, or quiet-CDF bins; memoryless or causal-hysteretic;
- 252 causal neighborhood lanes over quiet thresholds 0.99, 0.999, and 0.9999,
  radii 1, 2, and 3, windows 1, 3, and 5, and support fractions 0.25, 0.5,
  and 0.75;
- 48 combined lanes, testing both quantize-then-support and
  support-then-quantize orderings.

Neighborhood operators included the proposed hard max/min rule as a stress
control, hard or soft upper-quantile support with center-pixel identity below
threshold, and a soft upper-quantile/median alternative. All temporal windows
were causal. Candidate extraction, burst budgets, label geometry, and display
conventions match the preceding GN expert benchmark.

The operating point was frozen using label-free metrics. Expert labels were
then used for fixed-budget reporting and leave-one-burst-out (LOBO) protected
selection. Unmatched detections remain unknown rather than negatives.

## Results

| Evaluation | Known matches at budget 58 |
|---|---:|
| Float GN control | 54/79 |
| Protected all-family LOBO | 54/79 |
| Protected neighborhood LOBO | 54/79 |
| Protected quantization-only LOBO | 52/79 |
| Protected integrity >= 0.95 LOBO | 51/79 |
| Frozen label-free hard-max/min + Q8 lane | 24/79 |
| Post-hoc soft q90/median ceiling | 58/79 |

The post-hoc 58/79 ceiling retained only 0.489 correlation with float GN, so it
does not resolve the signal-integrity concern. The high-integrity soft
q90/identity lane reached 56/79 with 0.989 correlation, but it was selected
after inspecting labels and did not validate as a protected family.

The frozen selector itself failed scientifically: it over-rewarded contrast
and sparsity created by hard extrema. Its 232 burst occurrences collapsed into
only 25 spatial identities, consistent with spatial merging rather than cleaner
cell-specific evidence.

## Decision

Do not promote quantization or hard conditional pooling. Keep float GN as the
canonical representation. If this concept is revisited, pre-register a narrow
independent confirmation of soft q90/identity support, strengthen the
label-free integrity penalty, and evaluate candidate separation and trace
distortion before recall.

Artifacts are under
`Outputs/HierarchicalParzenICA/spon_ca_burst_quantized_coactivity_pooling_v1`.
The annotation-free finalist video is
`scientific_audit/4_Vis_Diagnostics/01_control_and_finalists.mp4`; standard
expert-only, model-only, nearest-candidate trace, and comparison diagnostics are
under `scientific_audit/`.
