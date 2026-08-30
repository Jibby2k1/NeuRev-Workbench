# Source-on versus intervention rank displacement

This deterministic derived diagnostic rehydrated the frozen EXP-0029 Run-B raw, JEPA-residual, and random-residual score lanes over all 108 exact fixtures / 252 injected sources. It distinguishes paired intervention response from source-on top-four displacement; native competitors remain unknown rather than negatives.

## Classification counts

- `raw`: total source-on recovered `45` (`43` also recovered on the intervention map; `2` source-on-only inconsistent), competition `167`, attenuation/response failure `40`.
- `jepa_residual`: total source-on recovered `19` (`18` also recovered on the intervention map; `1` source-on-only inconsistent), competition `114`, attenuation/response failure `119`.
- `random_residual`: total source-on recovered `18` (`16` also recovered on the intervention map; `2` source-on-only inconsistent), competition `196`, attenuation/response failure `38`.

A source-on miss with intervention recovery is classified as `native_background_competition`; a miss on both maps is `attenuation_or_response_failure`. These are score-map mechanism categories, not biological labels or precision estimates.

## Boundary

The source Run-B scientific audit and motion dependency remain incomplete. This package provides bounded exact-injection diagnostics only; it establishes no neuron identity, false-positive rate, denoising benefit, nuisance robustness, generalization, causal claim, or reinforcement-learning result.
