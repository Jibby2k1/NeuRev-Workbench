# Spon Ca Burst patch-information v1 results

## Executive result

This completed study is the most direct Information Theoretic Learning (ITL)
test in the current activation-detection program. It evaluates local Gaussian-
Parzen quadratic information potential, Cauchy--Schwarz divergence from a
same-location quiet density, and correntropy on the accepted
quiet-standardized Parzen Innovation carrier.

The main result is promising but narrow:

- leakage-safe selection of standalone Cauchy--Schwarz quiet divergence
  improved mean burst recall from `0.541` to `0.572` at budget 20 and from
  `0.657` to `0.696` at budget 40;
- the selected standalone lane recovered 45/79, 55/79, and 56/79 labels at
  budgets 20, 40, and 58, versus 43/79, 52/79, and 55/79 for the native
  standardized carrier;
- the 7-pixel Cauchy--Schwarz family was selected in all four held-out folds,
  which is stronger parameter stability than most previous feature screens;
- adding all 27 ITL sources raised the optimistic 58-per-source proposal-union
  ceiling from `0.902` (71/79) to `0.919` (72/79), recovering burst-1
  `roi_015`, which was outside the v5 union;
- broad learned fusion did not improve the final ranker. The best nested model
  remained the prior `separation` feature set at `0.460`, `0.604`, and `0.714`
  for budgets 20, 40, and 58. The authoritative v5 nested single-feature result
  remains stronger at budget 58 (`0.734`, 58/79).

The correct decision is to retain one compact Cauchy--Schwarz quiet-divergence
expert as a candidate/proposal lane and test it with real hard-negative labels.
Do not promote the 49-source union or the all-ITL learned ranker.

## Scientific formulation

For a quantized local density `p`, the frozen same-location quiet density `q`,
and Gaussian Parzen interaction matrix `K_sigma`, the tested quantities were

\[
V_2(p)=p^T K_\sigma p,
\qquad H_2(p)=-\log V_2(p),
\]

and

\[
D_{CS}(p,q)=-\log
\frac{(p^T K_\sigma q)^2}
{(p^T K_\sigma p)(q^T K_\sigma q)}.
\]

Local correntropy was the expected Gaussian-kernel similarity between the
center observation and the patch distribution. The common Gaussian density
normalization cancels in `D_CS`.

This is Principe-aligned in its use of nonparametric Parzen densities,
quadratic Renyi information potential, correntropy, and Cauchy--Schwarz
divergence. It is not a reproduction of Parzen Infomax ICA: it does not learn
an ICA unmixing matrix by optimizing an information objective. Instead, it
uses local ITL criteria as label-independent spatial evidence on an existing
Parzen Innovation carrier.

## Frozen design and robustness controls

The feature grid was:

- 13 quiet-standardized intensity centers from -6 to +6 z;
- patch sizes 7, 11, and 15 pixels;
- Parzen bandwidths 0.5, 1.0, and 2.0 z;
- three criteria, producing 27 feature variants;
- standalone, four bounded carrier boosts, and three bounded carrier gates,
  producing 216 fixed lanes;
- budgets 20, 40, 58, 80, and 100, with 20/40 primary and 58 secondary.

The learned audit used 34 ranking features and 49 proposal sources. It tested
four bounded-linear feature sets over learning rates `0.003`, `0.01`, and
`0.03`, L2 values `0.01`, `0.1`, and `1.0`, maximum auxiliary weights `0.25`
and `0.5`, and 300 epochs. This produced 864 inner validation fits and 16
outer refits, or 880 fits total.

Every outer burst was untouched during its hyperparameter selection. Within
the remaining three bursts, each candidate configuration trained on two and
validated on the third. All image statistics and proposals were computed
without labels. Labels entered evaluation and nested selection only.

These counts describe dependent configurations on four bursts, not 1,096
independent scientific trials.

## Fixed-screen results

The table reports leakage-safe, leave-one-burst selection of a configuration
within each lane type.

| Method | Budget 20 | Budget 40 | Budget 58 | Threshold recall | Event candidates |
| --- | ---: | ---: | ---: | ---: | ---: |
| Standardized carrier | 0.541 | 0.657 | 0.703 | 0.726 | 449 |
| ITL standalone | **0.572** | **0.696** | 0.707 | 0.696 | 202 |
| Carrier + ITL boost | 0.564 | 0.670 | **0.716** | 0.710 | 336 |
| Carrier gated by ITL | 0.540 | 0.646 | 0.692 | 0.703 | 417 |

At budget 20, standalone ITL improved three held-out bursts and degraded one:
the exact matches were `[9, 10, 14, 12]` versus `[8, 9, 12, 14]` for the
carrier. At budget 40 it produced `[10, 15, 15, 15]` versus
`[10, 12, 14, 16]`. The four folds are too few for a strong significance
claim, but the direction and selected family justify a focused confirmation.

The lower threshold candidate count for standalone ITL is encouraging
selectivity evidence, not measured precision. Unmatched candidates can be
unlabelled neurons and cannot currently be called false positives.

## Parameter and criterion findings

The strongest post-hoc standalone configurations were all 7-pixel
Cauchy--Schwarz quiet-divergence variants:

| Feature | Budget 20 | Budget 40 | Budget 58 |
| --- | ---: | ---: | ---: |
| `cs_quiet__p7__bw0p5` | **0.605** | 0.707 | 0.718 |
| `cs_quiet__p7__bw1` | 0.580 | **0.708** | **0.730** |
| `cs_quiet__p7__bw2` | 0.594 | 0.707 | 0.729 |

Those values are post-hoc and therefore optimistic. More importantly, the
leakage-safe selector chose 7-pixel Cauchy--Schwarz divergence in every outer
fold: bandwidth 0.5 in three folds and bandwidth 1.0 in the fourth. The useful
signal is therefore stable to bandwidth but strongly favors the smallest
tested spatial support.

Family means across all nine standalone parameterizations were:

| Family | Budget 20 | Budget 40 | Budget 58 |
| --- | ---: | ---: | ---: |
| Cauchy--Schwarz quiet divergence | **0.524** | **0.613** | **0.622** |
| Quadratic information potential | 0.241 | 0.251 | 0.253 |
| Local correntropy | 0.056 | 0.064 | 0.074 |

Quadratic information potential was more useful as a bounded carrier boost:
an 11-pixel, bandwidth-2 lane was selected in three of four boost folds.
Positive correntropy was not useful. In every outer all-ITL or anchor refit,
the correntropy weight was exactly zero; Cauchy--Schwarz received the dominant
ITL weight. A future correntropy test should distinguish similarity from
novelty, for example by testing quiet-relative or centered correntropy loss,
rather than assuming high center-to-neighborhood similarity is positive
activity evidence.

## Nested-ranker results

| Feature set | Budget 20 | Budget 40 | Budget 58 | Threshold recall | Event candidates |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prior separation features | **0.460** | **0.604** | **0.714** | 0.823 | 614 |
| ITL anchor only | 0.438 | **0.604** | 0.689 | 0.904 | 792 |
| All ITL features | 0.438 | **0.604** | 0.661 | **0.920** | 833 |
| Separation + ITL anchor | 0.449 | **0.604** | 0.713 | 0.904 | 742 |

The threshold-recall values for the ITL models are not improvements in
precision: they require 742--833 event candidates. At scarce budgets the ITL
models do not improve ranking. The combined model learned a consistent
Cauchy--Schwarz weight around `0.113--0.120`, a small information-potential
weight around `0.005--0.009`, and zero correntropy, yet still tied or slightly
lost to separation-only ranking.

The likely issue is proposal dilution. Expanding from the v5 union to 49
sources created event tables of 1,342, 1,291, 1,167, and 1,172 candidates.
Useful new proposals exist, but indiscriminately merging all 27 ITL sources
introduces many competitors above the fixed top-k cutoff. More features are
not equivalent to more usable information when the evaluation budget is
scarce.

## Proposal and per-neuron findings

At 58 proposals per source, the augmented union recovered 72/79 observations,
one more than v5. The new recovery is burst-1 `roi_015`, supplied by the
7-pixel Cauchy--Schwarz family.

Seven observations remain outside the augmented proposal union:

- `roi_007` in bursts 2, 3, and 4;
- `roi_017` in burst 2;
- `roi_023` in bursts 3 and 4;
- `roi_027` in burst 4.

At total budget 58, the preferred separation ranker recovered 56/79. Its most
persistent misses remained `roi_007` (4/4), `roi_014` (4/4), `roi_019` (3/3),
`roi_015` (2/4), and `roi_023` (2/2). This preserves the earlier distinction:
`roi_007`/`roi_023` are mainly proposal failures, while `roi_014`/`roi_019`
are proposal-present ranking failures. ITL helped one mixed case but did not
resolve either failure class generally.

## Decision and next checkpoint

The experiment supports a compact next test, not another broad fusion grid:

1. retain the standardized carrier as the immutable scientific trace;
2. add only `cs_quiet__p7` as a separate proposal/ranking expert, with
   bandwidth 0.5 as the frozen default and 1.0 as a sensitivity check;
3. allocate per-source proposal quotas or use late rank fusion so the new
   expert cannot flood the carrier's top candidates;
4. annotate a bounded field exhaustively as neuron, background, artifact, or
   unresolved, including the persistent hard ROIs;
5. evaluate precision-recall and precision at fixed recall, then repeat the
   nested budget-20/40 analysis;
6. test quiet-relative correntropy novelty only as a preregistered ablation;
7. keep timing/amplitude measurements on the carrier rather than the ITL score.

Promotion should require improvement at budget 20 or 40 on at least three of
four held-out bursts and a precision gain on the exhaustively annotated field.
The present standalone Cauchy--Schwarz result meets the first condition at
budget 20 but cannot satisfy the precision condition with current labels.

## Runtime and artifacts

The guarded CUDA run completed in 58.96 seconds with peak resident host memory
of 2,383 MiB. All 27 features, 216 fixed lanes, 880 fits, four outer folds, and
three diagnostic TIFFs completed.

### Complete framewise feature video

The selected default feature, `cs_quiet__p7__bw0p5`, has also been generated
for every frame in the experiment's accepted carrier interval. The output has
560 frames at 340 by 573 pixels corresponding exactly to inclusive UI frames
1800--2359. UI frames 1800--1899 form the frozen quiet calibration; online use
is causal from UI frame 1900 onward.

The raw float16 NPY preserves the exact Cauchy--Schwarz map. The uint16 TIFF
uses one global linear display scale (`black=0.02016`, `white=4.7578`) across
all 560 pages. It does not normalize each frame independently.

On the RTX 4070 SUPER, measured single-frame feature computation was 0.83 ms
median and 1.29 ms p95 against the 20 ms acquisition period. Batched offline
feature generation reached 605 frames/s. These measurements include transfer
from the in-memory standardized carrier and ITL-map computation, but exclude
camera acquisition, upstream Parzen-carrier generation, TIFF compression, and
UI rendering. Therefore the feature map itself has a credible real-time path;
the complete live pipeline still needs an end-to-end latency benchmark.

The original TIFF contains 2,359 raw frames, but the accepted Parzen
Innovation carrier used by this experiment was intentionally constructed only
for frames 1800--2359. Extending the map to raw frames 1--1799 would require
reinitializing and rerunning the upstream causal carrier from frame 1, which is
a new preprocessing condition rather than a faithful visualization of this
completed experiment.

Authoritative root:

```text
Outputs/HierarchicalParzenICA/spon_ca_burst_patch_information_v1
```

Start with `REPORT.md`, `metrics.json`, `evaluation/fixed_screen.json`,
`evaluation/nested_rankers.json`, `evaluation/oracle_coverage.json`,
`evaluation/per_neuron_audit.tsv`, and the three TIFFs under `diagnostics/`.

The full framewise map is under:

```text
Outputs/HierarchicalParzenICA/spon_ca_burst_patch_information_video_v1
```
