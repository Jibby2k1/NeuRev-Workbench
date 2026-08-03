# Dependent multiscale population-preserving W5b results

## Decision

**Do not advance to W6 semi-synthetic Spon injection or W7 real scientific
execution.** The revised patchwise method passes numerical reconstruction and
generated attribution, but fails morphology-specific signal preservation.
Aggregate medians are not sufficient to override that failure.

## Revised method

The frozen diagnostic lane uses 15-pixel patches with 10-pixel stride and
floored-Hann overlap-add. It preserves a transient broad population component
instead of forcing coordinated neural activity to be independent, then moves
only 25% of population-synchronous residual structure from `noise_candidate`
back to `structured_signal`. Original-space closure is conserved, authority is
bounded, and the residual is not renamed measurement noise.

## W5b generated matrix

The collision-safe artifact is
`Outputs/HierarchicalParzenICA/dependent_multiscale_population_w5b_generated_v1`.
It evaluates all 15 required fixtures over seeds 7, 13, and 19 against a
geometry-matched patchwise orthogonal baseline.

- maximum normalized closure: `1.1271e-07` (C1 pass);
- median signal leakage: `0.4026850 -> 0.3431222`;
- relative leakage improvement: `14.79%`;
- median diagonality: `0.4709530 -> 0.4843233`;
- required-case leakage improvements were `14.79%` compact, `29.19%` broad
  neural, `11.02%` correlated neural, `24.98%` motion-crossing, and `22.32%`
  heteroscedastic noise (C2 pass);
- aggregate peak ratio: `1.0122`;
- aggregate area ratio: `1.0625`;
- median/p95 peak-time error: `0 / 1` frames;
- aggregate C3 calculation: pass;
- morphology-subgroup C3 calculation: fail;
- C4: not qualified; C5: diagnostic only.

## Authoritative preservation failure

The subgroup audit exposes opposing errors hidden by the aggregate median:

- broad legitimate neural source: peak `0.5333`, area `0.4424`;
- compact source on broad drift: peak `2.4297`, area `4.0644`;
- motion edge crossing a neuron: peak `1.2087`, area `2.3897`;
- heteroscedastic shot-like noise: peak `1.0984`, area `1.3442`;
- clipping/saturation: peak `1.2900`, area `1.0737`.

Additional bounded diagnostics using low-order temporal trends, artifact guards,
and quiet-calibrated sparse event support did not resolve broad-neural recovery
without amplifying drift or motion. No post-hoc parameter was promoted.

## Grayscale visual correction

The prior red/blue video was deterministic but inadequately explained and hard
to compare with raw grayscale data. The replacement artifact is
`Outputs/HierarchicalParzenICA/spon_ca_burst_dependent_multiscale_grayscale_review_v1/grayscale_decomposition_review.mp4`.

It contains six larger grayscale panels with numeric ranges and a persistent
legend. Signed panels use black for negative, mid-gray for zero, and white for
positive. The neural panel is explicitly positive-only. Sparse-positive labels
use black-backed white rings; all other pixels remain unknown. The video is
560 frames, 10 fps, 56 seconds, 1146 by 570, and has SHA-256
`053ad79f3e28a582ef037454164582a1e29beffdbcdec984efd505a63ce430f9`.

## Next justified direction

The next model revision should use an independent neural confirmation signal,
such as the frozen `coherence_w15` family or the accepted carrier, to constrain
when population authority may activate. It should be designed and falsified on
generated cases before another real-background injection. W6 must remain
blocked until both aggregate and subgroup C3 pass. The accepted carrier remains
scientific trace authority.
