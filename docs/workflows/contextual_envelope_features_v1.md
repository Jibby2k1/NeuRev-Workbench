# Contextual-envelope feature pilot v1

- Registry experiment: `NREV-EXP-0031`
- Status: completed bounded engineering pilot; scientific promotion false
- Terminology: `docs/research/CONTEXTUAL_ENVELOPE_TERMINOLOGY_V1.md`

## Design

Three frozen aligned sources over UI frames 1800--2359 were transformed by one
global per-source quiet calibration into positive instantaneous evidence `A`:
Raw fluorescence, current `ICA -> LS` evidence, and the selected
`TMax5 -> LS -> ICA` temporal-envelope representation. Calibration subtracted
the sampled quiet median, divided by its positive 99.5th-percentile span, and
clipped to `[0,1]`; it was not pixelwise.

For temporal windows `w={1,3,5,9,15}` and spatial footprints `k={1,3,5}`, the
pilot computed the causal contextual upper envelope `U`, matched local minimum,
`A^2`, `A*U`, `sqrt(A*U)`, `A/U`, `A^2/U`, and local range position. The
`w=1,k=1` arm identifies amplitude-only behavior: `A*U=A^2`, while
`sqrt(A*U)=A` and `A^2/U=A` for positive `A`.

The analysis reports sampled Pearson/Spearman association, pair-order
inversion, quiet p99, saturation, active fraction, imported-envelope magnitude,
envelope agreement, upper-envelope area inflation, and fixed-anchor trace
persistence. Eighteen synchronized videos expose the representative
`w=5,k=3` feature fields.

## Audit boundary

This was an operator-semantics pilot, not a frozen detector evaluation. Its
model-only visual set contains fixed review anchors, its Expert section is not
applicable, and its scientific audit remains incomplete. It supplies no
precision, specificity, biological identity, generalization, or scientific
promotion. A claim-bearing follow-up must implement the complete scientific
audit standard and freeze task-specific gates before evaluation.
