# Temporal contextual-envelope morphology audit v1

## Outcome

Temporal max pooling measurably changes event morphology. The prior
peak-localization tie did not justify discarding pooling: it arose because that
estimand selects frames where the causal upper envelope equals the instantaneous
evidence. When full trace shape is measured, the operators have distinct and
repeatable roles.

The most useful carrier-like result is agreement-attenuated evidence
`H=A^2/U`. Across all three sources, `H` preserves the aligned event anchor to
numerical resolution, increases event-area concentration and core fraction,
and suppresses the post-peak shoulder. The causal upper envelope `U` does the
opposite job: it holds recent maxima, broadens events, and shifts mass into the
post-peak shoulder. These are complementary temporal operations, not evidence
that pooling is uniformly beneficial or harmful.

## Population and audit

- 106 canonical-v7 occurrences at 50 immutable sites;
- Raw, current `ICA -> LS`, and `TMax5 -> LS -> ICA` sources;
- five-frame (100-ms) causal temporal max envelope;
- six operators: `A`, `A2`, `U`, `C=A/U`, `P=A*U`, and `H=A^2/U`;
- 2,000 deterministic site-bootstrap resamples;
- 106 absolute and normalized occurrence panels;
- three fixed-scale six-operator videos.

Validation passed. The run reused validated upstream expert media and created no
detector, candidate coordinates, or model annotations. Scientific promotion is
false.

## Agreement attenuation versus instantaneous evidence

Paired changes are site-mean differences versus `A`; intervals are 95% site
bootstrap intervals.

| Source | Event-area fraction | Core fraction | Post-shoulder fraction | Half-max width |
|---|---:|---:|---:|---:|
| Raw | +0.0087 [0.0066, 0.0107] | +0.0065 [0.0049, 0.0082] | -0.0186 [-0.0210, -0.0163] | -45.2 frames [-73.4, -21.1] |
| ICA-to-LS | +0.0091 [0.0052, 0.0126] | +0.0213 [0.0164, 0.0266] | -0.0166 [-0.0227, -0.0113] | -1.46 frames [-1.97, -1.03] |
| TMax-to-LS-to-ICA | +0.0105 [0.0076, 0.0136] | +0.0220 [0.0170, 0.0273] | -0.0210 [-0.0271, -0.0153] | -3.02 frames [-4.09, -2.18] |

The Raw half-width difference is directionally consistent but should not be
read as a precise calcium-width estimate. Raw positive calibration often leaves
a long connected region above half maximum, producing a broad and variable
reference width. The learned-source widths are more locally interpretable.

## Operator roles

### `U`: contextual upper envelope

`U` retains a recent maximum for up to the declared causal support. It therefore
broadens events and increases post-peak persistence. Relative to `A`, median
half-max width increased by approximately 12 frames in both learned sources.
Its event-area and event-energy fractions decreased because held maxima also
occupy non-event/reference time. This is expected memory behavior, not failure.

### `C=A/U`: envelope agreement

`C` is dimensionless and should be treated as context, not a fluorescence-like
carrier. Its absolute magnitude and its ratio to `A` are not commensurate.
The peak-preservation column is consequently not scientifically interpretable
for `C`; only its temporal pattern and bounded `[0,1]` semantics are useful.

### `P=A*U`: envelope-gated evidence

`P` strongly concentrates area and energy in annotated events and increases
core fraction. It behaves similarly to an amplitude-emphasizing transform,
with recent context reinforcing already-large evidence. This makes it a
promising proposal/ranking channel, but its stronger concentration does not
establish better detection or specificity.

### `H=A^2/U`: agreement-attenuated evidence

`H` is the conservative carrier-like use of pooling. At a new local maximum,
`U=A` and `H=A`; after the peak, `U>A` and `H<A`. The observed atlas matches
that mathematics: anchor preservation followed by faster decay and reduced
post-peak shoulder. This makes `H` a plausible temporal sharpening or
morphology channel without claiming that the underlying calcium decay is
nuisance.

## Decision

Do not discard temporal pooling. Retain three explicit channels with separate
meanings:

1. `U` for short causal memory and temporal tolerance;
2. `P` for context-reinforced proposal/ranking evidence;
3. `H` for peak-preserving, post-peak agreement attenuation.

Do not collapse these into a generic “max-pooled feature.” Their desirable
behavior depends on the product task. Before incorporation into a detector,
the next gate is a bounded-field proposal/specificity comparison that keeps
`A` available and tests whether `P` or `H` reduces false or duplicate
candidates without losing known positives. Separately, biological waveform
analyses should retain `A`, because `H` deliberately alters decay morphology.

