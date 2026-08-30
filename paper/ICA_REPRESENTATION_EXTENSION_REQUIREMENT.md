# ICA representation-extension requirement

Status: frozen working requirement, 2026-08-24.

The hash-verified frozen two-frame ICA output must be included wherever a
scientifically meaningful representation-matched comparison is made. ICA uses
the same sites, windows, folds, controls, denominators, and multiplicity rules
as the other representations. It is not refit using expert labels.

## Required coverage

- Trace atlas and occurrence metrics: already included.
- Operator identity and representation confirmation: already included.
- Cross-neural zero-lag, lagged, spatial, and robustness analysis: included in
  `12_cross_neural_ica_v1`.
- Functional/tensor decomposition: add the frozen ICA cube beside Raw and
  residual/global-adjusted cubes in the next versioned extension.
- Detection-event classes: preserve the frozen three-class assignments and add
  ICA only as a post-freeze characterization; do not refit or rename classes.
- Measurement profiles and certainty comparisons: add ICA-derived magnitude,
  timing, and cross-burst stability as post-freeze outcomes where denominators
  are complete.
- Figures, tables, and reviewer packets: show ICA beside the matched Raw and
  nuisance-adjusted views whenever space permits; otherwise give a directly
  linked supplementary panel.

## Interpretation boundary

The frozen ICA component has already been shown to be nearly equivalent to a
signed temporal difference. ICA may therefore reveal synchronized temporal
change, but it is not an independently identified neuron, synapse, or causal
source. A difference between ICA and Raw networks is a representation effect,
not evidence that either network is anatomical truth.
