# Contextual-envelope feature pilot v1 results

## Outcome

The bounded pilot completed 360 factorial metric rows across three frozen
sources, five temporal windows, three spatial footprints, and eight feature
forms. All 18 representative MP4s passed full decode. This is validated
engineering output but not a scientific-audit-complete experiment.

The central result is mixed:

- `A*U` is mathematically plausible and retained rank order well, but much of
  its apparent contrast is the `A^2` identity-window effect. Added context
  generally raised quiet upper tails.
- `sqrt(A*U)` preserves the no-pooling identity but visually brightened diffuse
  contextual support and increased active area.
- `A/U` and local range position are informative explanatory maps but are
  noisy or saturated as standalone image features.
- `H=A^2/U` was the cleanest attenuation layer: it retained source ordering
  while suppressing quiet/diffuse activity when the upper envelope was imported.

## Representative 100-ms, 3-by-3 condition

| Source | `U` area inflation | Median agreement | `A*U` Spearman / inversion | `A*U` quiet-p99 delta vs `A^2` | `H` quiet-p99 delta vs `A` |
|---|---:|---:|---:|---:|---:|
| Raw | 1.27x | 0.801 | 0.999 / 0.010 | +0.113 | -0.095 |
| current `ICA -> LS` | 10.85x | 0.402 | 0.996 / 0.024 | +0.051 | -0.020 |
| `TMax5 -> LS -> ICA` | 7.34x | 0.423 | 0.993 / 0.031 | +0.094 | -0.062 |

`H` also reduced the fraction above 0.5 by 0.008, 0.038, and 0.032 for Raw,
current learned evidence, and temporal-envelope evidence, respectively. The
upper envelope itself imported much more support into learned representations
than into Raw.

## Window interpretation

Spatial support was the dominant inflation mechanism. At `w=5,k=1`, upper-
envelope area inflation was 1.09x for Raw, 3.49x for current learned evidence,
and 2.28x for temporal-envelope evidence. Adding `k=3` increased these to
1.27x, 10.85x, and 7.34x. Larger temporal windows added persistence, but after
spatial pooling their incremental inflation was smaller than the spatial jump.

The next bounded comparison should therefore retain:

1. temporal-only `H` at `w=3` and `w=5`;
2. spatiotemporal `H` at `w=5,k=3` as a stronger attenuation/stress arm;
3. `A`, `A^2`, and shuffled or displaced envelopes as mandatory controls;
4. `A/U` and `U-A` as explanations, not primary ranking images.

## Relationship to LS

The pilot supports a family resemblance, not equivalence. LS is causal local
standardized surprise based on a guarded reference mean and standard deviation.
The envelope family uses extrema and present-to-envelope agreement. Both are
context-conditioned pointwise transforms, but LS asks whether the present value
is statistically unusual, whereas `H` asks whether present evidence accounts
for its local upper envelope.

## Claim boundary

No detector was frozen or evaluated. Ordinal retention is not incremental
predictive value, and quiet frames are not verified negatives. The result does
not establish better detection, precision, specificity, neuron identity, or
independent-recording transfer.
