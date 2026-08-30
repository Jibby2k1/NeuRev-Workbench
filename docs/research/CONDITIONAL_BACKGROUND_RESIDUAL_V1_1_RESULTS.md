# Conditional-background residual v1.1 screen results

## Outcome first

The corrected v1.1 engineering screen completed successfully, but the tested
conditional-background residual was not a better input to the frozen
carrier/context/kinetic detector than the native raw movie at this one-seed,
500-step screen budget.

Across the complete 108-fixture, 252-source grid, macro source recall was
`0.18750` for native raw HC, `0.09028` for JEPA-residual HC, and `0.06713` for
the matched frozen-random-provider residual. JEPA residual minus raw was
`-0.09722`, with recording/window/injection-seed grouped 95% interval
`[-0.24306, 0.02784]`. The point estimate was adverse, three of four recording
effects were negative, and the residual amplified rather than suppressed the
registered background diagnostics.

This is a bounded non-scientific engineering screen. It does not formally pass
or fail an experiment-level benefit gate, does not promote `NREV-EXP-0029`,
and supports no scientific claim or evidence capsule.

## Run identity and scope

- Experiment: `NREV-EXP-0029`
- Failed immutable v1 run: `NREV-RUN-EXP-0029-SCREEN-20260830-A`
- Successful v1.1 engineering run:
  `NREV-RUN-EXP-0029-SCREEN-20260830-B`
- Configuration SHA-256:
  `d9928f98d11903f963e38de93537134060b0d3a89785c36bc7568d495537e83b`
- Runner SHA-256:
  `e3b4cb4cebe155b2a46c1c8b4ff810b28ec3bedb5415b5aed1e1097f01c531d9`
- Upstream JEPA seed: `1001`
- Decoder seed: `6201`
- Tile-schedule seed: `6202`
- Grouped-bootstrap seed: `6203`
- Decoder budget: 500 steps and 16,385 trainable parameters per provider arm
- Evaluation: 12 windows, 108 fixtures, 252 exact injected sources, 324
  method-by-fixture result rows

Run B used CUDA bfloat16 execution on an NVIDIA GeForce RTX 4070 SUPER. It
started at `2026-08-30T06:18:33.910042Z` and ended at
`2026-08-30T06:19:23.403796Z`.

## Why run A remains a useful safe failure

Run A stopped before residual endpoint interpretation because v1 supplied
normalized pixels to a raw-HC endpoint whose immutable EXP-0028 anchor had
been evaluated in native raw `float32` units. Four intervention
`RecoveryResult` objects changed only candidate cardinality and unmatched
candidate count; recall, matched identities, and localization were unchanged.

The [run-A failure audit](CONDITIONAL_BACKGROUND_RESIDUAL_V1_RUN_A_FAILURE.md)
showed that native-unit reconstruction matched all 216 parent source-on and
intervention recovery objects. V1.1 corrected only that input domain and kept
numeric tolerance at zero. Run B then reproduced all 108 fixtures and all 216
parent recovery objects exactly before decoder construction or training.

The sequence matters: A is not discarded or reclassified. Its failure shows
that the exact continuity guard caught a real endpoint-domain mismatch, and B
shows that the versioned correction fixed it without weakening the guard.

## Paired recovery results

| Comparison | Macro recall or difference | Grouped 95% interval | Descriptive reading |
| --- | ---: | ---: | --- |
| Native raw HC | `0.18750` | not applicable | Strongest tested endpoint |
| JEPA-residual HC | `0.09028` | not applicable | Below raw at the screen point estimate |
| Random-residual HC | `0.06713` | not applicable | Below both raw and JEPA point estimates |
| JEPA residual minus raw | `-0.09722` | `[-0.24306, 0.02784]` | Adverse point estimate; interval crosses zero |
| Random residual minus raw | `-0.12037` | `[-0.23148, -0.02078]` | Negative throughout the grouped interval |
| JEPA residual minus random residual | `+0.02315` | `[-0.04398, 0.09722]` | Small point advantage; no JEPA-specific interval support |

Source count was averaged within each
recording/background-window/injection-seed cluster before the 1,000-draw
hierarchical bootstrap. Injected sources and source-count conditions were not
resampled as independent biological replicates.

JEPA-minus-raw effects by recording were:

| Recording | Recall difference |
| --- | ---: |
| `060126_10_rest` | `-0.23148` |
| `060126_12_left` | `-0.17593` |
| `060126_15_right` | `-0.01852` |
| Spon Ca Burst recording | `+0.03704` |

The descriptive nonnegative-in-every-recording value was therefore not met.
The single positive Spon effect is not an independent-animal replication and
does not override the aggregate or other-recording results.

As a descriptive micro-recovery diagnostic, stratifying the frozen windows by
their registered background-MAD label did not reveal a favorable JEPA-residual
regime:

| Background-MAD stratum | JEPA recovered / injected sources | Raw recovered / injected sources | Median JEPA source-off RMS ratio | Median JEPA seam ratio |
| --- | ---: | ---: | ---: | ---: |
| Low | `17 / 84` | `21 / 84` | `3.6339838254893495` | `1.698874867465591` |
| Median | `0 / 84` | `13 / 84` | `1.7102741349511534` | `3.4065767263210747` |
| High | `2 / 84` | `11 / 84` | `1.269637223157785` | `2.628523891244171` |

These counts describe exact injected-source recovery within this one frozen
screen; the sources are nested within fixtures and are not independent
biological replicates. The zero-of-84 median-MAD result, together with its
large seam ratio, is especially concerning for this tiled implementation, but
it is not a separately registered subgroup gate.

All methods respected the four-candidate cap. Each residual arm emitted 432
source-on and 432 intervention candidates across the 108 fixtures. Raw HC
emitted 432 source-on candidates and 420 intervention candidates. Native
unmatched candidates remain biologically unknown, not verified false
positives.

## Signal retention did not imply useful residualization

The JEPA residual retained almost all injected signal in the aligned
projection:

| JEPA residual diagnostic, fixture median | Value |
| --- | ---: |
| Aligned retained gain | `0.99908` |
| Predictor absorption | `0.00092` |
| Total signal-error ratio | `0.53120` |
| Orthogonal-distortion ratio | `0.53107` |
| Background RMS ratio | `1.71027` |
| Dynamic-MAD ratio | `1.10933` |
| Seam-to-interior jump ratio | `2.57864` |

Retention and absorption close because

\[
r = x - \hat b,
\]

so the injected source must divide between residual and prediction up to
floating-point error. The maximum projection-closure error was `2.12e-7`, and
the maximum voxelwise pair-closure error was `1.42e-6`; both passed the frozen
`1e-5` engineering limit.

Those closure results validate the accounting, not the usefulness of the
representation. The near-one aligned gain shows that the predictor absorbed
little injected source, but the approximately `0.53` orthogonal distortion
shows that the residual source-on/source-off difference still acquired a large
off-axis error component. Meanwhile, background RMS increased by about 71%,
dynamic MAD increased by about 11%, and the tiling seam diagnostic was much
larger than its interior reference. The combined pattern is consistent with a
conditional predictor that preserves source amplitude but introduces enough
prediction error and patch-boundary structure to degrade the frozen detector.

The decoded movie is therefore not identified as biological background, and
the residual is not identified as neuron-only signal or denoised video.

## Descriptive advancement panel

Only three engineering-triage conditions were favorable: median aligned
retention exceeded `0.90`, median absorption stayed below `0.10`, and the JEPA
point estimate exceeded the random-provider residual. The following conditions
were not met:

- JEPA residual minus raw recall was not at least `+0.02`;
- the JEPA-minus-raw interval lower bound was not positive;
- the JEPA-minus-random interval lower bound was not positive;
- median background RMS ratio was not at most `0.90`; and
- the JEPA-minus-raw effect was not nonnegative in every recording.

These were explicitly nonformal screen thresholds. Their pattern argues
against escalating the same tiled one-layer decoder design unchanged, but it
does not constitute a preregistered scientific rejection of conditional
background learning in general.

## Engineering integrity

The screen's implementation checks passed:

- all 25 parent artifacts were hash-verified before checkpoint loading and
  remained byte-identical afterward at snapshot SHA-256
  `9045cb7f70e6ab805e961a36b00255242377c713c12741151129ad00683d0139`;
- frozen JEPA and random providers remained hash-identical before training,
  after decoder training, and after evaluation;
- the complete 252-source blind-halo geometry and all 64 adversarial target
  tubes passed;
- normalized caches were reused without double normalization;
- raw HC used native units and matched 216 of 216 parent recovery objects;
- all 240 prediction-coverage records were finite, shape-exact at
  `1 x 1 x 32 x 64 x 64`, and covered each voxel exactly once; and
- the 27-artifact, 13,600,885-byte output package matched its artifact index.

These facts establish a technically valid bounded screen. They do not supply
the missing scientific evidence.

## Scientific and publication boundary

The run status is `screen_complete_claim_gates_unresolved`. Numeric execution
completed, but outputs remain scientifically partial because the required
candidate-surrogate full-field videos, close-ups, full-duration traces,
matched comparison figures, and media validation were not produced. The
numeric motion/registration screen `NREV-EXP-0025` is complete, but its usable
motion-field dependency remains unsatisfied because all 12 local fields
required reliability review. Multiple training and decoder seeds,
independent-recording generalization, and biological identity evidence also
remain unresolved.

Accordingly:

- `NREV-EXP-0029` remains `draft`, `not_evaluated`, and evidence tier `none`;
- `scientific_completion=false`;
- `scientific_promotion_allowed=false`;
- no claim record is linked;
- no evidence capsule is created; and
- this self-supervised conditional-prediction experiment is not reinforcement
  learning.

## Decision

Keep the current tiled conditional-residual design on hold. Conditional
prediction remains conceptually plausible, but this implementation's
near-unity aligned amplitude retention co-occurred with no empirical-background
suppression and lower source recovery than raw at the frozen HC endpoint.

Any later learned-background program should be a separately versioned design,
not an in-place widening of this screen. The most important architectural
falsifiers to address would be patch seams, full-field or overlap-add decoding,
prediction-error amplification, and multi-seed stability, followed by the
still-open requirement for a reliable motion estimator or acquisition contract
and the complete scientific audit. Re-running the same one-layer tile decoder
at a larger budget is not justified by this screen alone.

The subsequent frozen, integrity-checked
[rank-displacement and source-off safety diagnostics](CONDITIONAL_BACKGROUND_RESIDUAL_DERIVED_DIAGNOSTICS_V1_RESULTS.md)
show that the JEPA miss pattern is not confined to the
intervention-recovered/source-on-missed category and that a retrospective
source-off no-amplification guardrail admits no residual window. Those
post-screen artifacts strengthen the implementation hold without becoming
registered scientific executions or formal EXP-0029 gates.

The two numeric sentinels requested after Run B are also complete for their
current implementations. [Motion Run E](MOTION_REGISTRATION_CONFOUND_V2_RESULTS.md)
produced no usable local field and zero of 14 multiplicity-supported
associations; [predictor Run B](SOURCE_OFF_PREDICTOR_FEASIBILITY_V1_RESULTS.md)
admitted zero of seven subtractors. Any revisit therefore requires a new
motion/acquisition contract and a separately frozen predictor design rather
than completion of these already executed screens.

## Portable provenance

The full output remains under:

```text
Outputs/NeuronIdentifiability/NREV-EXP-0029/runs/
  NREV-RUN-EXP-0029-SCREEN-20260830-B
```

The exact sanitized `resolved_config.json`, `artifact_index.json`, and
`summary.json` are
retained under
`research/run-provenance/NREV-RUN-EXP-0029-SCREEN-20260830-B/`. Their SHA-256
values are respectively
`1ff575fd0110e4d65e275ce4ced705a3aa47b0d9a83b01d10a2270bbc865218c`
, `34bd983ffdf55cb0941bbd1443adf939f4cb86ac4ffe51a736b2d6d9383cfc1c`,
and `4da57e17f7fc76cfdf0b8bb841c564113fc6987ddaf11d05256e9a77e3e801ba`.
