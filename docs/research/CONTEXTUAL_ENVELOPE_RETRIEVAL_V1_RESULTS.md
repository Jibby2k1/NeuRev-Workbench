# Contextual-envelope retrieval comparison v1

## Outcome

No agreement-attenuated feature passed the frozen advance gate. The result is a
useful negative finding: temporal agreement attenuation is a trace-morphology
operator, but the present peak-localization estimand is nearly invariant to it;
adding a 3-by-3 spatial envelope reduces known-center localization.

This comparison is exploratory and within-recording. It does not estimate
full-field precision, specificity, biological identity, or transfer.

## Frozen comparison

The analysis evaluated all 106 canonical-v7 occurrences at 50 immutable sites,
with 5,000 deterministic site-bootstrap resamples. Each of Raw, the current
ICA-to-LS representation, and the temporal-envelope-to-LS-to-ICA representation
was evaluated using positive instantaneous evidence (`A`), amplitude squaring
(`A2`), temporal agreement attenuation at 3 and 5 frames (`H_t3`, `H_t5`),
spatiotemporal agreement attenuation at 5 frames and 3-by-3 pixels
(`H_st5k3`), and matched temporally shuffled or spatially displaced envelope
controls.

## Primary localization result

| Source | A2 | H_t3 | H_t5 | H_st5k3 | Interpretation |
|---|---:|---:|---:|---:|---|
| Raw | 0.9909 | 0.9909 | 0.9909 | 0.9873 | Temporal arms tie A2; spatial arm is lower. |
| ICA-to-LS | 0.8790 | 0.8790 | 0.8791 | 0.8742 | Temporal differences are negligible; spatial arm is lower. |
| Temporal-envelope-to-LS-to-ICA | 0.8579 | 0.8576 | 0.8579 | 0.8243 | Temporal arms tie A2; spatial attenuation materially lowers localization. |

Values are mean event-localization percentiles. No aligned attenuation arm had
a positive site-bootstrap interval against both `A2` and its matched
misalignment control. For the temporal-envelope-to-LS-to-ICA source,
`H_st5k3` was lower than `A2` by 0.0336 (95% interval -0.0655 to -0.0115) and
lower than the displaced-envelope control by 0.0222 (95% interval -0.0522 to
-0.0015).

## Why the temporal comparison ties

For nonnegative instantaneous evidence `A` and a causal upper envelope
`U >= A`, agreement-attenuated evidence is

`H = A(A/U) = A^2/U`.

At a local maximum included in its own pooling window, `U = A`, so `H = A`.
The event-localization metric selects a maximum within each event or quiet
window. It therefore preferentially samples precisely the frames where the
attenuation disappears. Likewise, `A2` is a strictly increasing transform of
nonnegative `A`, so `A` and `A2` have identical within-window maximum ranks.
The near-ties are consequently an estimand/operator interaction, not evidence
that the full traces are identical.

## Secondary morphology measures

Temporal agreement attenuation redistributed energy toward events relative to
`A` while retaining approximately the same event-to-quiet effect as `A`:

- Raw event-energy fraction increased from 0.390 for `A` to 0.408 for `H_t5`,
  compared with 0.570 for `A2`.
- ICA-to-LS increased from 0.432 to 0.444, compared with 0.548 for `A2`.
- Temporal-envelope-to-LS-to-ICA increased from 0.516 to 0.557, compared with
  0.693 for `A2`.

Thus temporal `H` attenuates shoulders and inter-peak signal, but less
aggressively than amplitude squaring. The shuffled temporal control was not
worse, so this analysis found no context-specific retrieval advantage.

## Decision

Retain temporal agreement attenuation as a visual and temporal-morphology
diagnostic, not as an advanced known-center retrieval feature. Do not advance
the 3-by-3 spatiotemporal form for known-center retrieval. If the operator is
studied further, use estimands sensitive to morphology rather than peak rank:
integrated event energy, duration and shoulder suppression, event-versus-quiet
area, or full-field candidate specificity with bounded truth.

## Audit and artifacts

Run `NREV-RUN-EXP-0032-RETRIEVAL-20260831-B` completed with validation passed,
2,226 occurrence-feature rows, and 106 trace-comparison figures. It reused
validated upstream canonical-v7 expert media and introduced no new detector or
model annotations. Scientific promotion remains false. A separate `A.partial`
directory records the successful preflight-only invocation and is not an
analysis result.

