# Contextual-envelope terminology v1

## Purpose

This vocabulary separates local standardization, the established local-
correlation feature, and new max-envelope operators. The terms describe
measurement operators, not biological sources.

| Symbol | Preferred term | Definition |
|---|---|---|
| `LS(X)` | causal local standardized surprise | Current value minus its guarded causal spatiotemporal reference mean, divided by the larger of reference standard deviation and the calibrated scale floor. |
| `A` | positive instantaneous evidence | Nonnegative evidence at the present acquisition frame and native coordinate after a declared frozen monotone calibration. |
| `U_wk` | contextual upper envelope | Maximum positive evidence in a declared trailing temporal window and same-size spatial footprint. |
| `C_wk=A/U_wk` | envelope agreement | Fraction of the contextual upper envelope explained by the present frame and native coordinate. |
| `P_wk=A*U_wk` | envelope-gated evidence | Instantaneous evidence weighted by contextual upper-envelope magnitude. |
| `G_wk=sqrt(A*U_wk)` | scale-preserving envelope blend | Geometric blend whose no-pooling arm is the identity. |
| `H_wk=A*C_wk=A^2/U_wk` | agreement-attenuated evidence | Instantaneous evidence suppressed when its envelope is mostly imported from another frame or pixel. |
| `R_wk` | local range position | Present evidence's position between matched local minimum and maximum. |

## Reserved established terms

- **Local coherence** continues to mean the causal pixel-to-neighborhood
  correlation feature (`coherence_w15`). It is not an alias for `A/U`.
- **Carrier** retains its frozen detector-feature meaning.
- **LS/MSLN** means causal local mean/standard-deviation normalization, not
  min--max normalization.
- **Current** means the present acquisition frame. Review playback rate does
  not redefine acquisition time.

## Interpretation boundary

`U` is contextual support, not a localized source value. `C`, `P`, `G`, `H`,
and `R` are deterministic feature fields. None implies neuronal identity,
coherence in a biophysical sense, or improved detection without a separate
evaluation.
