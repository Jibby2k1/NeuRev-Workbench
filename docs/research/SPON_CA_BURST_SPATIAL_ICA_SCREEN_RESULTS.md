# Spon Ca Burst spatial ICA screen results

## Bottom line

Dense translation-shared application produced a small improvement over the
coarse patch lattice without harming median timing. The first component-space
noisy-Parzen posterior was too aggressive: it reduced candidate burden but
substantially attenuated known neural events. None of the three lanes passed
the complete preservation audit, so the result is a useful architecture
checkpoint rather than a promoted method.

## What was tested

All three lanes used the same accepted Parzen Innovation input and the same
spatial ICA fit: 30,000 label-free 11-by-11 patches, rank 12, seed 20260729,
and symmetric log-cosh FastICA. FastICA converged in 96 iterations with final
delta `9.60e-6`. The retained patch subspace explained 34.1% of total patch
variance.

| Lane | Threshold recall | Fixed-budget recall | Event candidates | Peak retention | Area retention | Peak error | Quiet RMS ratio | Synthetic r |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Patch FastICA + Wiener | 0.467 | 0.659 | 81 | 0.972 | 0.949 | 0 | 0.368 | 0.347 |
| Dense convolutional FastICA + Wiener | **0.484** | **0.671** | 91 | **0.975** | **0.952** | 0 | 0.355 | **0.371** |
| Dense convolutional FastICA + Parzen | 0.291 | 0.646 | **74** | 0.468 | 0.533 | 1 | **0.312** | 0.304 |

Threshold recall uses a threshold calibrated to one quiet false peak per quiet
map. Fixed-budget recall asks how many labels occur among the top 58 candidates
per burst, regardless of the calibrated threshold.

## Interpretation

Moving from stride-5 overlap-add to dense stride-1 application increased mean
threshold recall by 1.67 percentage points and fixed-budget recall by 1.25
points. Median peak and area retention also increased slightly, while the quiet
RMS ratio decreased. The cost was ten additional thresholded event candidates.
With sparse labels those extra candidates cannot be called false positives.

The noisy-Parzen posterior cut threshold recall by 19.36 points relative to the
dense Wiener lane and reduced median peak retention from 0.975 to 0.468.
Fixed-budget recall declined much less, from 0.671 to 0.646. That pattern implies
that much of the spatial ranking survived but component amplitudes were
compressed below the quiet-calibrated threshold. The current shared dictionary,
50% zero mass, bandwidth 0.5, and noise variance 1 should therefore not be used
as a final component posterior.

All lanes had poor exact semi-synthetic preservation (`r=0.304--0.371`) and a
nine-frame synthetic peak error. This is the main automatic stop condition.
The dense spatial result is promising enough for visual inspection and a
held-burst/seed confirmation, but it does not justify adding temporal
complexity yet.

## Relation to the intended pipeline

The actual first-pass ordering was:

```text
raw movie
  -> accepted Parzen background / innovation residual
  -> spatial patch ICA fit
  -> patchwise or dense translation-shared application
  -> Wiener or noisy-Parzen component shrinkage
  -> reconstructed signal and exact remainder
```

Thus, Parzen Innovation preprocessing happened before all three lanes. The
third lane also used a noisy-Parzen posterior after ICA demixing. It did not
yet optimize the spatial filters with a Parzen Infomax objective. That objective
remains a separate C3 optimization experiment and should be attempted only
after resolving the attenuation observed here.

## Recommended checkpoint actions

1. Visually compare the dense-Wiener signal and remainder against the patch
   lane, especially thin membranes, crowded cells, and propagation timing.
2. Confirm dense-Wiener across multiple seeds and held-burst fits.
3. If the visual and held-burst checks survive, screen per-component rather
   than shared Parzen dictionaries, lower zero mass, and weaker posterior noise
   assumptions. Preserve a Wiener lane as the fixed reference.
4. Only then train filters under noisy-Parzen Infomax. Add grouped spatial
   morphology next; add causal temporal filters last.

The authoritative machine-readable results are in
`Outputs/HierarchicalParzenICA/spon_ca_burst_spatial_ica_screen_v1/metrics.json`.
The most important visual files are the three `signal_positive.tif` and
`remainder_detail.tif` pairs under that output root.
