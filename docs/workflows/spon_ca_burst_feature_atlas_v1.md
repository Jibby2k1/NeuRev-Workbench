# Spon Ca Burst Feature Atlas v1 workflow

## Purpose

Feature Atlas v1 tests whether interpretable feature engineering adds ranking
utility beyond the existing 12-feature linear positive-versus-unlabeled model.
It does not change proposal coordinates, labels, event windows, candidate
budget, or the five protected Run-B spatial folds.

## Frozen population and interpretation

- Candidate universe: 1,619 event candidates from
  `NREV-RUN-EXP-0021-SCREEN-20260830-B`.
- Reference roles: 78 canonical-v7 positive anchors and 1,541 unlabeled
  candidates. Unlabeled means unknown, not negative.
- Primary operating point: a global budget of 58 after label-free within-fold
  percentile normalization.
- Primary metric: macro held-fold SPU-AUC.
- Sensitivities: known-positive recall at budget 58, grouped paired bootstrap,
  and leave-one-burst-out SPU-AUC.

## Added feature families

1. **Temporal envelope:** causal `H=A^2/U` peaks at 3- and 5-frame supports,
   plus area agreement and core concentration. Peak features are retained as a
   deliberate estimand check because `H` equals `A` at a new local maximum.
2. **Soma morphology:** scale-normalized LoG responses at 1.5 and 2.5 pixels,
   cross-scale minimum, and an existing center-minus-ring contrast.
3. **Map/source consistency:** alternating-frame carrier-map agreement,
   held-out prediction of the center trace from the surrounding map, and
   alternating-frame stability of the frozen ICA representation.
4. **Nuisance/competition:** guarded-quiet independence from a global field
   trace, local proposal isolation, and carrier margin to proposals within 12
   pixels in the same burst.

The ICA stability feature does not refit ICA in each half. It asks whether the
spatial footprint of the already frozen ICA representation is stable across
alternating event frames. This distinction is required in all conclusions.

## Evaluation matrix

The runner evaluates each new feature alone; retrains the existing 12-feature
linear model; adds each family separately; trains a new-feature-only model; and
trains the full augmented linear model. Every learned score is out of fold with
fixed penalty `0.01`, a 100,000-iteration numerical ceiling, training-only
robust preprocessing, and class-balanced positive-versus-unlabeled ranking
loss. The ceiling was raised from the inherited 50,000 only after the complete
augmented matrix failed to converge; all family-specific and existing-only
models had converged under the original ceiling, and no outcome metric had
been produced or inspected.

No coordinate, burst ID, candidate ID, rank, source count, human description,
or detector outcome is a model input. Geometry-derived local density and score
margin are allowed because they are label-free relational measurements, not
absolute coordinates.

## Scientific-audit boundary

This run is an exploratory analysis on a frozen candidate union. It writes the
small audit indexes, all numerical tables, and comparison figures. Because a
new ranking is produced, full model-only annotation videos and close-ups would
be required before scientific promotion. Until those are rendered and
validated, `scientific_promotion` remains false and the audit is explicitly
incomplete. Existing canonical-v7 expert identities and coordinates are reused
without modification.

## Command

```bash
.venv-neurobench/bin/python -m \
  neurobench.experiments.neuron_identifiability.feature_atlas_v1 \
  --repo-root "$PWD" \
  --data-root "${NEUROBENCH_DATA_ROOT:-.}" \
  --output-root "Outputs/NeuronIdentifiability/spon_ca_burst_feature_atlas_v1_20260905_e"
```

The output root is immutable. Use a new root for any rerun or protocol change.
The original `..._20260905.partial` root records a failed preflight-plus-load
attempt caused by mutating a read-only ICA memmap view; it contains no metrics.
The `..._20260905_b.partial` root records the strict 50,000-iteration augmented
fit failure; it also contains no metrics.
The completed `..._20260905_c` root passed numerical validation but its
leave-one-burst-out figure failed visual QA because labels overlapped. It is
preserved; Run D adds readable labeling plus family-level intervals,
coefficient summaries, and feature-redundancy diagnostics without changing the
candidate, feature, fold, solver, or metric definitions.
Run D's focused leave-one-burst-out plot clipped the leading characters of its
lowest y-axis tick during rasterization. Run E changes only the lower display
limit from 0.965 to 0.970; all numerical definitions remain identical.
