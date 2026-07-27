# Spon Ca Burst Pairwise-Feature Fusion

## Purpose

The adjacent-frame and ICA outputs are activity evidence, not replacement
images or standalone classifiers. This experiment tests whether they improve
Raw Direct decisions or produce a more interpretable original-image view.

For Raw Direct residual `R`, clipped continuous feature `F` in `[0,1]`, and
original image `X`, the two fixed families are:

```text
additive score:  S = R + lambda * quiet_scale(R) * F
soft image gate: Y = X * [floor + (1 - floor) * F]
```

`lambda=0` is the exact Raw Direct initialization. A separate one-parameter
model starts there and tunes only `lambda` in `[0,0.4]` using a learning rate of
`0.001`, 300 epochs, and L2 attraction to zero. Training uses labeled centers
from three bursts and full-field quiet hard negatives; the fourth burst is held
out. Event candidates not matched to known labels remain unknown.

## Frozen design

- Features: fixed difference, adaptive difference, and InfoMax activity.
- Additive weights: `0.05, 0.1, 0.2, 0.4`.
- Soft floors: `0.5, 0.7, 0.85`.
- Fixed comparisons: 22 including Raw Direct.
- Learned fits/evaluations: 12 (three features × four outer bursts).
- Fixed-spec selection: leave one burst out; choose on the other three.
- Primary match radius: 6 pixels with quiet-only calibration.

Only three TIFFs are preregistered to avoid redundant multi-gigabyte output:

- original image with adaptive feature, floor `0.7`;
- original image with InfoMax feature, floor `0.7`;
- Raw residual plus InfoMax feature, `lambda=0.1`.

## Commands

```bash
.venv-neurobench/bin/python -m neurobench.cli.main experiment pairwise-separation fusion-preflight \
  --config examples/spon_ca_burst_pairwise_feature_fusion.example.json \
  --artifact-dir Outputs/PairwiseSeparation/feature_fusion_preflight_v1

.venv-neurobench/bin/python -m neurobench.cli.main experiment pairwise-separation fusion-run \
  --config examples/spon_ca_burst_pairwise_feature_fusion.example.json \
  --preflight-dir Outputs/PairwiseSeparation/feature_fusion_preflight_v1
```

The run refuses existing destinations and consumes the immutable completed
pairwise result. Improvement requires held-out evidence; visual appeal alone is
not a detector-selection criterion.

## Completed result, July 27

`Outputs/PairwiseSeparation/spon_ca_burst_pairwise_feature_fusion_v1`
completed in 27.6 seconds and reproduced Raw Direct exactly at `0.605615942`
(49/79 known matches, 232 candidates).

- All 12 fixed additive fusions retained 49/79. They increased candidates to
  234–249, so none improved the detector.
- The three-feature, four-fold learned scalar fits moved only to
  `lambda=0.0364–0.0460`. They retained 49/79 while producing 233 candidates
  for fixed/adaptive and 235 for InfoMax, versus 232 for Raw Direct.
- InfoMax soft floor `0.85` was the least harmful gate: 47/79 and 198
  candidates. This is a 14.7% candidate reduction with two lost known matches,
  so it does not satisfy a no-recall-loss selectivity gate.
- Floors `0.7` and `0.5` were progressively destructive. At floor `0.7`,
  InfoMax recovered 20/79 and adaptive/fixed 13/79.
- Leakage-safe fixed-spec selection reached `0.580616` mean recall, below Raw
  Direct, because one fold selected the attractive but lossy floor-0.85 gate.

Decision: retain the continuous derivative/ICA value as a visualization,
timing, or optional confidence feature. Do not replace or automatically gate
Raw Direct with the tested formulations. The next justified work is annotation
of persistent-artifact versus useful-structure regions; that supervision could
make attenuation spatially selective instead of asking temporal change alone
to decide what structure should remain.
