# Spon Ca Burst Activity-Feature Utility Study

## Purpose

This workflow asks whether transformations of the accepted signed Parzen
Innovation carrier add useful evidence for sparse neuron detection. It is a
feature study, not a claim that a transformed movie is itself a better
biological signal.

The frozen manifest is
`examples/spon_ca_burst_feature_utility_v1.example.json`.

## Frozen feature bank

The study writes 25 float16 TYX feature arrays:

- signed and even derivative evidence at lags 1, 2, and 4, including positive,
  negative, absolute, power-1.5, log-square, Huber, and bounded-square forms;
- local power-spectral-density signal and correction diagnostics;
- rank-oriented and recall-oriented cross-scale consensus;
- asymmetric component state and innovation;
- causal dynamic-activity and persistent-artifact evidence;
- centered versus membrane morphology crossed with isolated versus crowded
  context;
- continuous CFAR score, local background, local noise, and spatial coherence.

The signed carrier is stored separately. Even nonlinearities never replace it,
because squaring removes polarity and can distort timing. Ten representative
channels are also written as independently normalized 16-bit TIFF stacks for
visual review. Their display normalization is not used by the detector.

## Evaluation design

The program evaluates one carrier baseline, 25 standalone features, three
carrier boosts and three carrier gates per feature, one nonnegative scalar
fusion per feature and held-out burst, and one nonnegative multi-feature fusion
per held-out burst. That is 176 fixed lanes, 100 scalar fits, and 4
multi-feature fits.

Fixed-lane selection and learned fusion are nested by burst: three bursts
determine the choice or weight, and the fourth provides the reported fold
result. Learned weights begin at zero, remain nonnegative, and have total
auxiliary weight at most one.

Primary measurements are quiet-calibrated sparse-label recall and candidate
count, fixed-budget recall at 58 candidates per burst, burst-by-burst wins
against the carrier, redundancy correlations, and synthetic morphology trace
correlation and peak-frame error.

Sparse labels are known positives, not an exhaustive segmentation. Unmatched
event candidates remain unknown, so candidate count is only selectivity
pressure, not measured false-positive count. Quiet full-field maxima are the
defensible hard-negative source.

## Causality and interpretation

Derivative, persistence, morphology, and CFAR-derived channels are causal after
their leading quiet calibration. Asymmetric features are causal after a frozen
offline component fit. Local PSD channels are offline diagnostics and are
excluded from learned online fusion. CFAR background/noise and persistent
artifact score are context diagnostics rather than positive detector evidence.

A feature is promising only if it improves held-out fixed-budget recall or
selectivity consistently, is not merely redundant with a simpler channel, and
retains acceptable synthetic shape/timing behavior. A visually attractive TIFF
is insufficient.

## Commands and outputs

Use the repository CLI:

```bash
.venv-neurobench/bin/python -m neurobench.cli.main \
  experiment feature-utility preflight \
  --config examples/spon_ca_burst_feature_utility_v1.example.json

.venv-neurobench/bin/python -m neurobench.cli.main \
  experiment feature-utility run \
  --config examples/spon_ca_burst_feature_utility_v1.example.json
```

The run requires its matching ready preflight and refuses an existing completed
or partial output root. The completed root is:

```text
Outputs/HierarchicalParzenICA/spon_ca_burst_feature_utility_v1
```

The main review artifacts are `REPORT.md`, `metrics.json`,
`feature_manifest.json`, the JSON/TSV files under `evaluation/`, and the TIFFs
under `feature_tiffs/`. Progress is appended to `progress.jsonl`; the final root
is published only after all arrays, metrics, TIFFs, and metadata are complete.
