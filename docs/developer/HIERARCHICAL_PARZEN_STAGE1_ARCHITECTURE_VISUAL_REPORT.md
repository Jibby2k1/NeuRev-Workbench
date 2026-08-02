# Hierarchical Parzen Stage-1 architecture visual report

Last updated: 2026-07-29.

## Outcome

Four inference architectures around one fully learned stochastic-Parzen ICA fit
were applied to the real Spon Ca Burst review interval. All four completed and
wrote separate background and signed dynamics-plus-noise TIFFs.

Output:

```text
Outputs/HierarchicalParzenICA/
  spon_ca_burst_stage1_architecture_visuals_v1
```

This is a visual and rollout diagnostic. Labels, Stage 2, neuron detection,
recall, and precision were not used or evaluated.

## Exact run

| Property | Value |
| --- | ---: |
| Source UI frames | 1800--2359 |
| Quiet calibration | 1800--1899 |
| TIFF output UI frames | 1801--2359 |
| Output pages per TIFF | 559 |
| Architectures | 4 |
| TIFFs | 9, including aligned input |
| Device | CPU |
| Numerical threads | 2 |
| Runtime | 57.15 seconds |
| Peak process RAM | 2919.65 MiB |
| Output size | 1704.93 MiB |
| Maximum closure error | `1.22e-4` intensity units |

All nine TIFFs passed independent page-count, shape, dtype, first-page metadata,
and selected quiet/burst-frame checks.

## Shared raw stochastic fit

The raw stochastic fit converged, passed the affine feedback bounds, resolved
the background component, and retained learned fraction 1.0:

```text
P(t) =
  0.9591831001 * previous
  + 0.0333361013 * observation(t)
  - 2.6621502115
```

Unlike the earlier 10%-anchored visual, these four lanes therefore test the
fully learned stochastic direction.

## Regularized innovation controls

| Control | Value |
| --- | ---: |
| Quiet background | Per-pixel median |
| Reference half-life | 10 seconds |
| Per-frame reference refresh | 0.00138533 |
| Parzen correction fraction | 0.1 |
| Quiet correction MAD | 42.8437 |
| Correction clip | 171.375 |
| Maximum applied correction magnitude | 17.1375 |

The quiet bias and correction scale were frozen using only UI frames
1800--1899.

## Rollout measurements

| Architecture | Last/first background spatial SD | Negative background | Quiet dynamics RMS | Post-quiet dynamics RMS |
| --- | ---: | ---: | ---: | ---: |
| Teacher-forced stochastic | 1.04471 | 0 | 71.6245 | 72.2672 |
| Raw stochastic recurrence | 0.858095 | 0 | 212.870 | 261.586 |
| Quiet fixed-point recurrence | 1.01647 | 0 | 52.3298 | 57.2225 |
| Reference plus Parzen innovation | 1.01418 | 0 | 54.1716 | 70.9262 |

The raw recurrence no longer collapses below zero like the earlier 10%-anchored
recurrence, but it still loses about 14.2% of its spatial background contrast
and its background mean falls from 850.9 to 657.5. Its residual energy is much
larger than the other lanes.

Both explicit fixed-point architectures preserve background contrast through
the full interval. This establishes that the earlier visual disappearance was
caused primarily by state architecture and demixer anchoring, not by an
unavoidable property of stochastic Parzen ICA.

The teacher-forced lane also avoids state collapse. Its use of the real previous
frame means a sustained change can move into the reconstructed common source
after onset, so stable background appearance does not prove neural-signal
preservation.

The fixed-point recurrence has the lowest dynamics RMS, but that is not
automatically desirable: low residual energy can mean background suppression or
neural leakage. The innovation lane intentionally leaves more post-quiet
dynamics while limiting its learned background correction.

## Visual review

Every architecture directory contains:

```text
background.tif
dynamics_noise.tif
```

Dynamics/noise is signed. Mid-gray is zero, darker is negative, and brighter is
positive. All architectures share the same display scales:

| Channel | Source limits |
| --- | ---: |
| Background | 155.076 to 4061.704 |
| Dynamics plus noise | -815.745 to 815.745 |

Recommended Fiji slices:

| Region | Slices |
| --- | ---: |
| Quiet example around UI 1850 | 50 |
| Burst 1, UI 2003--2026 | 203--226 |
| Burst 2, UI 2040--2063 | 240--263 |
| Burst 3, UI 2122--2149 | 322--349 |
| Burst 4, UI 2254--2300 | 454--500 |

## Decision

The explicit fixed-point correction is justified and should replace unconstrained
free-offset recursion in later Stage-1 work. The reference-plus-innovation lane
is the strongest architectural candidate because it preserves anatomical
memory while bounding the learned contribution.

No lane advances scientifically until visual review is followed by:

1. quiet-region and labeled-ROI trace comparisons;
2. slow-ramp and plateau leakage measurements on real-quiet injections;
3. motion/saturation artifact controls;
4. fixed-budget neuron-detection evaluation; and
5. confirmation that event activity is not simply absorbed into background.
