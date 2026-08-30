# Spatial ICA filter stability review

## Result

The filter-stability gate at `18_spatial_ica_filter_stability_v1` fails. All eight rank-12 FastICA fits converged, but the learned filter bank was not sufficiently reproducible across either sampling/ICA seeds or leave-one-burst-window-out fits.

- Across new seeds, mean subspace cosine was 0.866--0.881 (excluding the exact reference replay), below the prespecified 0.90 gate. Median aligned individual-filter absolute correlation was 0.639--0.710, below 0.80.
- Across held-burst exclusions, mean subspace cosine was 0.895--0.918, with one of four folds below 0.90. Median aligned individual-filter correlation was 0.606--0.784, below 0.80 in every fold.

## Interpretation

Individual spatial filters must not be named or assigned neuronal/anatomical meaning. Even a bank-level interpretation is held under the prespecified gate. This does not negate reconstructed-output utility: a non-unique basis can still generate a useful reconstruction, and the separate objective morphology analysis evaluates that output directly.

The subsequent reconstruction-level test passed across three new seeds and four leave-one-burst-window-out fits. The defensible distinction is therefore: **stable reconstructed representation, unstable individual basis elements**.
