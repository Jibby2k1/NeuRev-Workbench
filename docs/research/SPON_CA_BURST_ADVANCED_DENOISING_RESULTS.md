# Spon Ca Burst advanced denoising results

## Bottom line

The corrected v2 program robustly screened all ten requested denoising families,
but none passed the complete preservation gate. Several methods improved one
part of the problem, demonstrating useful directions, but no method yet
simultaneously improved noise, detection, morphology, amplitude, and timing.

The most important result is therefore not a promoted denoiser. It is a clearer
Pareto frontier:

- PSD-Wiener produced the highest known-label threshold recall and strong noise
  attenuation, but greatly increased candidate burden and missed the synthetic
  timing/correlation gate.
- Nonnegative factorization retained real-event amplitude and improved recall,
  but synthetic recovery improved only modestly.
- Nonlocal denoising preserved real-event amplitude almost exactly while
  reducing quiet energy, but did not improve synthetic correlation.
- Component Kalman improved synthetic correlation and attenuated noise, but
  introduced a three-frame synthetic peak error.
- Per-component Parzen ICA produced the best fixed-budget recall, but attenuated
  real peaks slightly beyond the frozen limit and reduced synthetic correlation.

## Corrected full-field finalists

| Family | Recall | Fixed recall | Candidates | Peak | Area | Quiet RMS | Synthetic r | Gain over input | Noise dB | Synthetic peak error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Skip-connected Parzen ICA | 0.330 | 0.641 | 69 | 0.946 | 0.953 | 0.913 | 0.645 | -0.001 | 0.83 | 2 |
| Per-component Parzen ICA | 0.435 | **0.694** | 86 | 0.844 | 0.868 | 0.488 | 0.630 | -0.016 | 7.40 | 2 |
| Multiscale convolutional ICA | 0.342 | 0.659 | 69 | 0.990 | 0.982 | 0.783 | 0.647 | +0.002 | 2.28 | 2 |
| Bounded ICA noise subtraction | 0.330 | 0.641 | 71 | 0.981 | 0.978 | 0.913 | 0.642 | -0.003 | 0.82 | 2 |
| Noise-PSD Wiener | **0.571** | 0.630 | 159 | 0.853 | 0.862 | 0.430 | **0.706** | **+0.060** | 9.07 | 3 |
| Robust low-rank plus sparse | 0.267 | 0.641 | **50** | 0.799 | 0.769 | 0.835 | 0.607 | -0.039 | 1.52 | 2 |
| Nonnegative factorization | 0.502 | 0.657 | 113 | 0.965 | 0.972 | 0.795 | 0.669 | +0.023 | 1.99 | 3 |
| Nonlocal patch denoising | 0.442 | 0.682 | 81 | **0.999** | **0.999** | 0.619 | 0.638 | -0.008 | 4.62 | 2 |
| Component Kalman | 0.439 | 0.646 | 87 | 0.950 | 0.971 | 0.582 | 0.686 | +0.041 | 5.29 | 3 |
| Undecimated wavelet | 0.475 | 0.682 | 83 | 0.486 | 0.484 | **0.274** | 0.567 | -0.078 | **11.44** | 2 |

The unprocessed synthetic fixture has localized median correlation 0.645 and a
two-frame peak error. Synthetic correlation gain is therefore more informative
than correlation alone.

## Family interpretations

### ICA and Parzen variants

The three weak-authority ICA corrections retained real-event shape but changed
little detection behavior. This confirms that small corrections are safe but
not yet useful. Increasing posterior authority improved fixed-budget recall for
the per-component model, but real peak retention fell to 0.844 and synthetic
correlation declined.

Multiscale ICA is the safest ICA formulation tested: peak and area retention
were 0.990 and 0.982. Its unchanged detection and synthetic performance suggest
that the current low-rank Wiener reconstructions do not supply sufficiently
discriminative noise estimates.

### PSD-Wiener

PSD-Wiener is the strongest pure denoising lead. It reached 0.571 threshold
recall, near the historical Raw Direct reference, and improved synthetic
correlation by 0.060 while removing 9.07 dB of exact synthetic noise. The cost
was 159 thresholded candidates and a three-frame synthetic peak error. A future
revision should use a carrier blend and spatially local or anisotropic spectra
rather than one global transfer function.

### Nonnegative factorization

The selected 16-frame, rank-4, 20-iteration factorization with carrier blend
0.25 reached recall 0.502 while preserving peak and area at 0.965 and 0.972.
This is the most balanced source-separation lead. Its modest synthetic gain and
high candidate burden indicate that component selection or explicit background
components are still needed.

### Nonlocal denoising

The gentler radius-1, 5-by-5 patch method retained real-event amplitude almost
exactly and reduced quiet RMS to 0.619. It is the best morphology-preserving
spatial denoiser, but its synthetic correlation was slightly worse than the
unprocessed input. It should be treated as a carrier-preserving preprocessing
candidate, not a detector replacement.

### Component Kalman

The slowest/highest-process-noise component Kalman setting improved synthetic
correlation by 0.041 and removed 5.29 dB of noise while preserving real
amplitude. The three-frame synthetic timing error is the stop condition. A
rise/decay or innovation-gated state model is more justified than stronger
Kalman smoothing.

### Wavelet and robust low-rank methods

Both demonstrated the expected noise/candidate reductions but removed too much
signal. Wavelet shrinkage removed 11.44 dB of synthetic noise but approximately
halved real peak and area. Robust low-rank plus sparse reduced candidates to 50
but also reduced recall and event amplitude. These need explicit carrier skips
or weaker correction authority before further evaluation.

## Evaluator correction

The completed v1 screen inherited a full-frame synthetic trace metric. Noise
outside the injected source could set the apparent peak frame, making purely
spatial denoisers appear to shift time. Version 2 uses four localized matched
spatial projections and is authoritative. Version 1 remains preserved only as
an audit record.

## Checkpoint decision

No finalist met all four preregistered requirements:

1. peak retention at least 0.85;
2. area retention at least 0.85;
3. peak timing error at most one frame; and
4. synthetic correlation at least 0.75.

Consequently, the 120 seed-by-held-burst confirmation strata were recorded but
the stochastic refits were not executed. Running them would violate the staged
stop rule because no method qualified for promotion.

The highest-impact revisions are:

1. carrier-blended local PSD-Wiener;
2. nonnegative factorization with explicit background/component selection;
3. innovation-gated component dynamics; and
4. a carrier skip around wavelet shrinkage.

The authoritative machine-readable results and diagnostic TIFFs are in:

```text
Outputs/HierarchicalParzenICA/spon_ca_burst_advanced_denoising_program_v2
```
