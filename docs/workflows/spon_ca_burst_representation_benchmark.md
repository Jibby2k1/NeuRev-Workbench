# Spon Ca Burst PCA, Spatial ICA, and Autoencoder Benchmark

Last updated: 2026-07-28.

## Purpose

This workflow tests broadly understood representation methods on the fixed
Spon Ca Burst review interval while preserving the established neuron-ID
evaluation contract. It is a representation benchmark, not a claim that an
unsupervised component is a neuron.

The primary methods are:

- uncentered PCA/SVD after explicit amplitude or quiet-residual preprocessing;
- full-window spatial FastICA, where pixels are observations and frames are
  features;
- linear autoencoders as an optimization control related to PCA;
- shallow nonlinear autoencoders as a bounded nonlinear comparison; and
- optional UMAP of PCA spatial scores for visualization only.

Adjacent-frame ICA is not reused. That earlier two-observation experiment
recovered a temporal derivative. This workflow instead uses the full 560-frame
review interval.

## Frozen first-run matrix

The example manifest declares 36 fitted representations:

| Family | Inputs | Grid | Fits |
| --- | --- | --- | ---: |
| PCA | amplitude, quiet residual | ranks 8, 16, 32, 64, 128 | 10 |
| Spatial FastICA | amplitude, quiet residual | ranks 16, 32, 64; seeds 7, 13, 19 | 18 |
| Linear autoencoder | quiet residual | ranks 32, 64; seeds 7, 13 | 4 |
| Nonlinear autoencoder | quiet residual | ranks 32, 64; seeds 7, 13 | 4 |

UMAP is an optional post-fit embedding. It is skipped with an explicit status
when `umap-learn` is unavailable and never blocks PCA, ICA, autoencoder, or
neuron-ID results.

## Signal and factor contracts

The review movie has shape `[560, 340, 573]`. Factorization treats each pixel
time course as one observation:

```text
pixel_traces.shape = [194820, 560]
```

The amplitude input divides raw intensity by the quiet 1st-to-99.9th percentile
range and retains persistent spatial structure. The quiet-residual input
subtracts the per-pixel quiet median before applying the same global scale.
There is no hidden per-pixel temporal centering inside PCA.

PCA is computed from the exact full-pixel temporal Gram matrix, with bounded
GPU projection chunks. Spatial ICA rotates a PCA-whitened subspace using
symmetric FastICA and the log-cosh contrast. At a fixed rank, ICA and PCA span
the same reconstruction subspace, so reconstruction NMSE cannot establish that
ICA found better sources. ICA is instead judged by:

- seed convergence and aligned component stability;
- spatial localization;
- event-versus-quiet temporal SNR;
- component-map/trace review; and
- neuron-ID performance of a fixed positive component-evidence construction.

The nonlinear autoencoder's component trace is
`decoder(one_hot) - decoder(zero)`. It is a diagnostic probe and is not claimed
to be an additive decomposition.

## Neuron-ID evaluation

Every evaluated stack uses:

- temperature-0.25 log-mean-exp temporal pooling;
- the established four quiet calibration segments;
- six-pixel NMS and one-to-one label matching;
- quiet calibration at one false peak per map; and
- a second comparison fixed at 58 candidates per burst.

The fixed-budget metric prevents a method from gaining recall only by emitting
more candidates. Raw Direct must reproduce `0.6056159420289855` mean recall or
the run stops as invalid.

Sparse annotations identify known-positive recall and candidate burden.
Unmatched candidates remain unknown, not false positives. Ordinary precision
is not identified.

## Guarded commands

Create a new preflight directory:

```bash
.venv-neurobench/bin/python -m neurobench.cli.main experiment representation preflight \
  --config examples/spon_ca_burst_representation_benchmark.example.json \
  --artifact-dir Outputs/RepresentationBenchmark/preflight_spon_ca_burst_representation_v1
```

After reviewing the projection, resources, combination count, and collision
checks, run the identical manifest:

```bash
.venv-neurobench/bin/python -m neurobench.cli.main experiment representation run \
  --config examples/spon_ca_burst_representation_benchmark.example.json \
  --preflight-dir Outputs/RepresentationBenchmark/preflight_spon_ca_burst_representation_v1
```

Both destinations refuse collisions. A completed output root must never be
overwritten.

## Important outputs

Start with:

- `RESULTS_INDEX.md`;
- `report.md`;
- `experiment_summary.json`;
- `metrics/neuron_id_metrics.json`;
- `metrics/fit_summary.tsv`;
- `figures/fixed_budget_recall.png`;
- `figures/recall_candidate_tradeoff.png`;
- `figures/*components.png`; and
- `representative_tiffs/`.

TIFFs use one fixed display scale across all frames. Signed reconstructions
encode zero at 32768; positive component evidence encodes zero at 0. These
TIFFs are review artifacts, while the JSON/TSV files contain scientific
metrics.

## Completed v1 result

The selected CUDA run at
`Outputs/RepresentationBenchmark/spon_ca_burst_representation_benchmark_v1`
completed all 36 fits and 51 evaluated neuron-ID lanes in 72.5 seconds. Raw
Direct reproduced exactly.

At the fixed 58-candidate-per-burst budget, Raw Direct matched 52/79 known
labels (`0.6572` mean burst recall). The best lane was amplitude PCA rank 8 at
54/79 (`0.6874`), a gain of two known matches. It won three bursts and lost
one. Under quiet-threshold calibration, the same lane matched 56/79 but used
427 candidates versus Raw Direct 49/79 with 232 candidates; this is not a
precision improvement.

Amplitude spatial FastICA rank 16 was highly stable across seeds
(mean aligned absolute component correlations `0.9985` and `0.9993`) and
converged in 69--126 iterations. It reached as high as 59/79 under the looser
quiet threshold with 394 candidates, but only 48--51/79 at fixed budget. Rank
64 ICA hit the 300-iteration cap for every seed and stability fell to roughly
`0.63`; it should not advance.

Linear rank-64 autoencoder reconstruction matched 53/79 at fixed budget for
both seeds, one above Raw Direct. Nonlinear reconstruction achieved high
quiet-threshold recall only by producing 413--594 candidates and was weaker at
fixed budget. Increasing component rank generally reduced component-evidence
performance.

The completed primary output is immutable. The winning-lane TIFF and gallery
are in the separate supplement
`Outputs/RepresentationBenchmark/spon_ca_burst_representation_benchmark_v1_best_visuals`.
The optional 20,000-pixel UMAP visualization is in
`Outputs/RepresentationBenchmark/spon_ca_burst_representation_benchmark_v1_umap`;
it is visualization-only and did not enter neuron-ID selection.
The common-frame comparison sheet is in
`Outputs/RepresentationBenchmark/spon_ca_burst_representation_benchmark_v1_frame_comparison`.

Decision: retain amplitude PCA rank 8 as the only bounded follow-up candidate.
Confirm its two-match fixed-budget gain with temporal-block perturbations and
visual review before any detector replacement.
