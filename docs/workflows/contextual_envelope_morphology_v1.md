# Temporal contextual-envelope morphology audit v1

- Registry experiment: `NREV-EXP-0033`
- Parent comparison: `NREV-EXP-0032`
- Status before execution: frozen visual-first within-recording comparison

## Question

What does a 100-ms causal max-pooling envelope change in event morphology when
the estimand is sensitive to the full trace rather than only its winning peak?

## Frozen sources and operators

Reuse the exact aligned inputs, UI frames 1800--2359, and global quiet
calibration from `NREV-EXP-0032` for Raw, current `ICA -> LS`, and
`TMax5 -> LS -> ICA`. For positive instantaneous evidence `A` and the causal
five-frame upper envelope `U`, evaluate:

- `A`: positive instantaneous evidence;
- `A2 = A^2`: amplitude-squared control;
- `U`: contextual upper envelope;
- `C = A/U`: envelope agreement;
- `P = A*U`: envelope-gated evidence;
- `H = A*C = A^2/U`: agreement-attenuated evidence.

Use `0` when both numerator and denominator are zero. No spatial pooling,
shuffling, fitting, threshold tuning, or source-specific window selection is
allowed.

## Alignment, population, and grouping

- Population: all 106 canonical-v7 occurrences at 50 immutable sites.
- Align each occurrence to the maximum of `A` inside its frozen annotated event
  interval. The same anchor is used for every operator within that source.
- Aligned display support: 20 frames before through 40 frames after the anchor.
- Bootstrap unit: immutable site; 2,000 deterministic resamples, seed 20260831.
- Pointwise bands summarize site-bootstrap means and are descriptive, not
  simultaneous confidence bands.
- Quiet windows use the same-duration guarded references from `NREV-EXP-0032`
  and are not verified biological negatives.

## Frozen morphology measurements

For every occurrence and operator, compute:

1. event positive area and squared energy over the annotated interval;
2. equal-duration quiet-reference area and energy, summarized as event fraction;
3. normalized core fraction: area within anchor plus/minus two frames divided by
   area within anchor minus 15 through plus 20 frames;
4. pre-shoulder and post-shoulder fractions over frames -15:-3 and +3:+20;
5. half-maximum width around the aligned anchor, using the contiguous region
   containing the anchor;
6. peak preservation relative to `A` at the shared anchor.

Area and energy are kept separate because squaring changes units and magnitude.
For cross-operator morphology figures, each occurrence trace is divided by its
own positive aligned-window maximum before aggregation. Absolute calibrated
traces remain available in per-occurrence panels.

## Visual outputs

1. Three-source aligned median/mean trace atlas with site-bootstrap bands;
2. source-by-operator morphology metric matrix;
3. paired operator-change distributions versus `A`;
4. one absolute and peak-normalized trace panel for each occurrence;
5. one six-operator comparison video per source over the frozen 560 frames.

Videos use a fixed source-specific display scale derived before rendering and
must not rescale frame by frame.

## Interpretation gates

This audit does not promote a retrieval feature. It may establish that an
operator measurably reshapes the trace if the paired site-bootstrap interval
for at least one prespecified morphology endpoint excludes zero and the visual
atlas shows the same direction without event loss being hidden by normalization.
Any claim of better detection, precision, specificity, identity, or transfer
requires a separate frozen evaluation.

## Scientific boundary

This is a descriptive within-recording operator audit using previously analyzed
events and validated upstream expert media. It creates no detector, candidates,
coordinates, or biological labels. Unmatched activity remains unknown.

