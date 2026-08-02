# Spon Ca Burst innovation denoising v3 results

## Bottom line

The v3 program completed all 120 preregistered screening evaluations across
eight new denoising families and eight Pareto mixtures. None passed the full
carrier-relative gate, so the conditional seed refits stopped by design.

This is still a useful architectural result. Three families now occupy
distinct, credible parts of the frontier:

- local PSD-Wiener is the best safe carrier denoiser;
- cross-scale consensus is the best real-label ranking/detection lane;
- asymmetric component dynamics is the closest exact-truth recovery lane.

Constant-weight mixtures did not combine those strengths successfully. The
next innovation should therefore be spatially and temporally conditional
authority, not a larger grid of constant mixture weights.

## Completed design and resources

- Stage A: 96 crop combinations, 12 per family.
- Stage B: 16 full-field semifinals, two per family.
- Family finalists: eight.
- Pareto sources: four.
- Pareto mixtures: eight.
- Verified finalist TIFFs: 20 files, ten signal/remainder pairs.
- TIFF geometry: 560 pages per file, each 340 by 573 pixels.
- Runtime: 397.6 seconds.
- Peak RSS: 11,861 MiB, below the 12,288 MiB cap by about 427 MiB.
- Output size: approximately 3.4 GiB.
- Confirmation: not run because no candidate passed the advancement gate.

The actual RAM peak was close to the explicit cap. Future runs should release
cached Pareto source movies before recomputing family TIFFs or increase the cap
only after a new preflight; the current run completed safely but does not leave
large cap-relative headroom.

## Identity carrier

The unchanged quiet-centered Parzen Innovation carrier scored:

| Recall | Fixed-budget recall | Candidates | Synthetic r | Synthetic peak error |
| ---: | ---: | ---: | ---: | ---: |
| 0.330 | 0.641 | 70 | 0.645 | 2 frames |

Sparse unmatched real candidates remain unknown, not false positives.
Candidate count is a precision-pressure proxy.

## Full-field family finalists

| Family | Variant | Recall | Fixed | Candidates | Peak | Area | Quiet RMS | Synthetic r | Gain | Noise dB | Synthetic error |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Local PSD-Wiener | `local_psd_wiener__03` | 0.479 | 0.659 | 94 | 0.977 | 0.982 | 0.647 | 0.684 | +0.039 | 4.34 | 2 |
| Morphology conditioned | `morphology_conditioned__11` | 0.330 | 0.657 | 70 | 0.998 | 0.994 | 0.854 | 0.652 | +0.007 | 1.33 | 2 |
| Selected NMF | `selected_nmf__01` | 0.437 | 0.641 | 94 | 0.907 | 0.918 | 0.897 | 0.656 | +0.011 | 1.01 | 2 |
| Asymmetric dynamics | `asymmetric_component_dynamics__05` | 0.452 | 0.659 | 89 | 1.030 | 1.124 | 0.649 | **0.694** | **+0.048** | 4.69 | 3.5 |
| Tempered Parzen | `tempered_parzen_posterior__01` | 0.330 | 0.641 | 70 | 1.003 | 1.006 | 0.973 | 0.642 | -0.003 | 0.26 | 2 |
| Graph diffusion | `graph_spatial_diffusion__02` | 0.330 | 0.653 | 71 | 1.000 | 0.999 | 0.956 | 0.644 | -0.002 | 0.38 | 3 |
| Cross-scale consensus | `cross_scale_consensus__10` | **0.536** | 0.682 | 108 | 1.000 | 0.984 | 0.645 | 0.644 | -0.001 | 3.94 | 2 |
| Blind-spot linear | `blindspot_self_supervised__01` | 0.330 | 0.641 | 70 | 0.967 | 0.963 | 0.847 | 0.643 | -0.002 | 1.51 | 2 |

Every finalist met the real peak/area/timing, fixed-recall, and candidate
requirements. Every finalist missed the synthetic-correlation threshold of
0.70. Synthetic timing is reported as a diagnostic but is not the gate that
stopped these candidates.

## The important non-finalist

`cross_scale_consensus__06` deserves separate review even though the
preregistered scalar ranking chose variant 10 as the family TIFF finalist. It
achieved:

- fixed-budget recall 0.711, the best of all 16 Stage B semifinals;
- 65 candidates, fewer than the carrier's 70;
- threshold recall 0.342;
- peak and area retention 1.000 and 0.980;
- quiet RMS 0.473;
- 6.84 dB synthetic noise attenuation; and
- synthetic correlation 0.645, effectively unchanged from the carrier.

This is a strong ranking/filtering result but not an exact-truth recovery
result. It may be useful as a detector-support feature. It should not yet
replace the signal carrier.

## Pareto mixtures

The four selected sources were:

1. `asymmetric_component_dynamics__05`;
2. `blindspot_self_supervised__01`;
3. `cross_scale_consensus__10`;
4. `local_psd_wiener__03`.

The best all-source mixture used weight 0.25 for each correction. It reached
recall 0.453, fixed-budget recall 0.671, 85 candidates, peak/area retention
0.992/1.011, 3.65 dB attenuation, and synthetic correlation 0.676. This was
weaker than the best individual source on every major lead metric.

The best pairwise synthetic correlation was 0.674 for asymmetric dynamics plus
local PSD-Wiener. Constant global weights therefore average incompatible
corrections rather than select the right correction for a pixel and time.

## Morphology-specific finding

The four exact-truth cases remain highly unequal. For local PSD-Wiener,
correlations were:

| Morphology | Correlation | Peak error | Amplitude | Area |
| --- | ---: | ---: | ---: | ---: |
| Center, isolated | 0.335 | 3 | 1.880 | 1.650 |
| Membrane, isolated | 0.611 | 1 | 0.804 | 1.105 |
| Center, crowded | 0.757 | 3 | 0.582 | 0.658 |
| Membrane, crowded | 0.993 | 1 | 0.671 | 0.696 |

The median score hides a decisive weakness: isolated centered sources remain
noise dominated, while crowded membrane sources are already easy. Improving
the two weak cases is more valuable than uniformly strengthening denoising.

The current morphology-conditioned lane barely changed detection because it
used soft evidence to modulate one carrier rather than applying four genuinely
different matched operators. A stronger revision needs distinct center,
annulus, crowded-center, and crowded-annulus experts with independent quiet
calibration and bounded fusion.

## Component-selection finding

The selected-NMF finalist assigned every fitted component a keep probability
of at least 0.5. Its mean keep probability was 0.596 on real data. The proposed
absolute concentration/dynamics thresholds therefore did not actually
separate background from neural components.

The next NMF selector should calibrate each score against a quiet or
time-permuted null, explicitly force or penalize a background allocation, and
report retained and rejected reconstructions separately. A larger grid around
the current absolute thresholds is not justified.

The blind-spot model learned a stable bounded predictor with weight L1 norm
0.706, but its output was nearly detection-inert. A linear isotropic neighbor
predictor is insufficient; any revisit should use morphology-aware or
multi-frame blind spots while retaining a correction bound.

## Real-time interpretation

Full-field finalist compute times include 560 real frames and 128 synthetic
frames, but exclude shared fit and TIFF I/O:

- local PSD-Wiener: 5.19 seconds;
- morphology-conditioned: 7.63 seconds;
- selected NMF: 1.24 seconds;
- asymmetric dynamics: 2.06 seconds;
- tempered Parzen: 3.67 seconds;
- graph diffusion: 1.02 seconds;
- cross-scale consensus: 8.14 seconds;
- fitted blind spot: 0.68 seconds.

These throughput numbers do not prove online readiness. Local PSD-Wiener uses
review-interval spectra and selected NMF uses overlapping windows. The
asymmetric dynamics, fitted blind spot, graph, morphology, and cross-scale
operators are structurally compatible with streaming after calibration.

## Decision

Do not broaden all eight grids. The highest-impact next checkpoint is a
conditional authority model with three inputs:

1. the immutable Parzen Innovation carrier;
2. the local PSD-Wiener correction;
3. the cross-scale ranking score and asymmetric-dynamics confidence.

Authority should be learned or calibrated per pixel/time but constrained to
the correction simplex and trained first on the four-morphology fixture plus
quiet nulls. In parallel, replace the current morphology gate with four
explicit matched experts. Promote only if isolated-center and
isolated-membrane truth improve without degrading real peak/area retention.

## Artifacts

The authoritative root is:

```text
Outputs/HierarchicalParzenICA/spon_ca_burst_innovation_denoising_v3
```

Key files:

- `metrics.json`: complete family, mixture, baseline, and gate record;
- `candidate_comparison.tsv`: concise machine table;
- `stage_b/metrics.json`: both full-field semifinals per family;
- `mixtures/pareto_sources.json`: exact source-selection decision;
- `REPORT.md`: short generated result;
- `finalists/*/signal_positive.tif`: visually normalized candidate signal;
- `finalists/*/remainder_detail.tif`: signed removed/reassigned detail.
