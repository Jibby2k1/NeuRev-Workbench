# Joint generative deblending v10

## Frozen model and evaluation boundary

The model was specified before the locked v8 evaluation. It uses two localized nonnegative spatial components, eight alternating spatial/temporal updates, smooth seven-pixel-support footprints, nonnegative frame coefficients, and sparse projection through a fixed calcium kernel.

Development used previously unreported ellipse conditions: distances 4, 6, and 8 px; amplitude ratios 0.3, 0.5, and 0.8; and temporal correlations 0.2 and 0.6. The frozen model was then evaluated once on the locked v8 crescent grid: distances 3, 5, and 7 px; ratios 0.2, 0.4, 0.6, and 1.0; and correlations 0 and 0.8.

## Deblending results

| Population | Method | Weak-identity recovery | Mean F1 | Duplicates |
|---|---|---:|---:|---:|
| Development | Spatial context | **0.407** | **0.648** | 0.065 |
| Development | Constrained generative | 0.389 | 0.639 | 0.065 |
| Locked v8 | Spatial context | **0.521** | **0.635** | 0.021 |
| Locked v8 | Constrained generative | 0.313 | 0.608 | 0.021 |

The generative model fails its development improvement gate and degrades substantially on the locked close-source identity test. Explicit smoothness and calcium projection do not compensate for initialization and factor-identifiability limitations.

## Corrected source-specific kinetic audit

The first internal run computed likelihoods on factors already projected through the forward calcium kernel, which was circular. That run is superseded and is not scientific evidence. In v10, component coefficients are re-extracted from the frozen spatial factors without a kinetic constraint before comparing forward and reversed kernel fits.

Corrected results:

- Direction classification AUC: 0.367.
- Mean forward-component likelihood: 0.102.
- Mean reversed-component likelihood: 0.175.
- Positive forward likelihood: 97.2% of forward components and 91.7% of reversed components.

The likelihood is therefore non-directional and, on average, ranks reversed components above forward components.

## Decision

Only one of six gates passed. The constrained generative model and its source-specific kinetic likelihood are rejected. The locked v8 conditions must not be used to retune this architecture.

The accumulated automated evidence now supports stopping simulator-only model escalation. Spatial context with robust source-free normalization remains the strongest simple proposal baseline, but the decisive next evidence requires bounded biological identity truth or a genuinely independent identity-labelled recording. More elaborate deblenders can otherwise optimize simulator assumptions without resolving the biological uncertainty that motivates the project.
