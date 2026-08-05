# MSLN/MSICA Alternative Objective Experiment Brief

## Discussion question

Can a dependence objective with a better information descriptor produce a
more identifiable and transferable two-frame MSICA separation than the current
CS-Parzen objective, without using neuron coordinates for tuning?

## Current objective and observed limitation

For whitened outputs `y = W x`, the current fit minimizes the Gaussian-Parzen
Cauchy--Schwarz divergence between the joint density and the product of its
marginals:

```text
J_CS(W) = D_CS(p(y1,y2), p(y1)p(y2))
```

This is a legitimate quadratic information-theoretic dependence criterion,
but it is instantaneous, symmetric, and controlled by one global kernel
bandwidth. Marginal standardization removes amplitude from the fit. It does
not explicitly represent temporal dependence, quiet-to-event distributional
change, or whether the angular optimum is identifiable.

The broad cascade result is consistent with that limitation. Individual ICA
fits often had large bootstrap angle dispersion and component swaps, while
five-seed aggregated energy maps were considerably more stable. Thus a low
dependence score can coexist with an unstable component identity.

## Objective families to compare

1. **CS-Parzen reference** -- the existing implementation and bandwidth grid.
2. **KSG Shannon mutual information** -- minimize adaptive-neighborhood MI;
   test neighbor counts `k = 3, 5, 10`.
3. **Normalized HSIC** -- minimize kernel dependence without explicit density
   estimation; test median-bandwidth scales `0.5, 1.0, 2.0`.
4. **Matrix-based Renyi mutual information** -- minimize Gram-spectrum MI;
   test entropy orders approximately `1.01, 1.5, 2.0` and the same bounded
   kernel-scale sensitivity.
5. **Multi-lag information dependence** -- minimize a weighted sum of MI at
   lags `0, 1, 2, 4`, rather than same-sample dependence alone.
6. **Quiet-relative composite** -- combine independence with a bounded reward
   for reproducible quiet-to-review distributional change, preferably
   Jensen--Shannon divergence:

```text
J(W) = I(y1;y2)
       - lambda * max_j JS(p(yj | review), p(yj | quiet))
```

The quiet/review intervals are fixed metadata; neuron coordinates remain
unavailable during fitting and objective selection.

Plain marginal entropy maximization and positive correntropy are not primary
candidates. Earlier local studies found that generic information potential and
positive correntropy were weak unless information was expressed relative to a
quiet distribution.

## Proposed experiment

### Stage 1: objective-surface and recovery sanity checks

- Use deterministic synthetic and semi-synthetic adjacent-frame pairs with
  known persistence, innovation, shared artifact, and heavy-tailed noise.
- Evaluate every objective on the same whitened sample IDs and a dense
  `0--90` degree rotation grid.
- Require finite CPU references, CUDA parity for promoted objectives, recovery
  angle, held-out objective improvement, and resistance to amplitude scaling.

### Stage 2: real-data label-free screen

- Freeze `S5/G1/T31`, `S7/G1/T31`, and `S15/G3/T31` as compact, intermediate,
  and broad contexts.
- Evaluate original and switched block orders, five fitting seeds, and the same
  screen/confirmation sample IDs.
- Select objectives using labels-free diagnostics only.
- Keep Raw Direct, current CS-Parzen, and the five-seed CS energy ensemble as
  immutable references.

### Stage 3: protected evaluation

Open known coordinates only after the objective/configuration freeze. Report
all frozen finalists rather than selecting a new winner from protected recall.
An independent recording is preferred for confirmation.

## Primary diagnostics

- held-out dependence objective and improvement over common/difference axes;
- angular minimum sharpness and separation from near-optimal rotations;
- blocked-bootstrap circular angle dispersion and component-swap fraction;
- five-seed signed-component and energy-map correlations;
- quiet-to-review contrast and burst consistency;
- protected recall curves at budgets `20, 40, 58, 80, 100`;
- CUDA runtime, peak VRAM, numerical clamps, and estimator sensitivity.

## Decision rule

An alternative objective advances only if it improves identifiability, not
merely its own numerical score. A reasonable promotion gate is:

- lower bootstrap angle dispersion or swap rate in at least two of three
  contexts;
- no loss of energy-map agreement or quiet/event contrast;
- consistent direction across at least four of five seeds;
- no protected budget-58 degradation relative to the frozen CS ensemble; and
- bounded CUDA execution below the existing 8 GiB cap.

## Recommended priority

Start with **KSG MI** as the interpretable Shannon reference, **normalized
HSIC** as the non-entropy dependence control, and **multi-lag matrix-Renyi MI**
as the most promising GPU-oriented objective. Add the quiet-relative
Jensen--Shannon composite only after the pure dependence comparison, so any
gain can be attributed to the biological prior rather than the entropy
estimator alone.

The key scientific distinction for discussion is:

```text
better independence estimator
    versus
better objective for neural-event representation
```

Those are related but not equivalent hypotheses and should be tested as
separate factorial dimensions.
