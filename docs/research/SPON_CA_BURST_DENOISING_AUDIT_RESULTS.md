# Spon Ca Burst denoising audit: essential results

## Executive result

Eleven variants across seven denoising families completed on the Spon Ca Burst
Parzen Innovation residual. No method passed the deliberately strict combined
audit, and none replaces Raw Direct as a standalone detector. The useful
result is narrower:

- local low-rank methods materially improve Parzen Innovation detection while
  reducing quiet residual energy;
- component-domain Parzen/ICA gives the strongest fixed-budget result and the
  lowest synthetic NMSE, but removes too much labeled amplitude;
- the causal temporal gate is the safest current real-time conservative
  filter;
- Haar-like filtering recovers recall closest to Raw Direct, but shifts and
  distorts event waveforms.

This supports using denoised lanes as auxiliary features or proposals while
retaining the original carrier.

## What ran

The experiment used UI frames 1800–2359, a 100-frame quiet interval, 79 label
rows representing 27 ROI identities, and four bursts. It attempted 11 fixed
variants:

1. frame gamma;
2. robust gamma;
3. quiet Wiener;
4. spatial evidence gate;
5. causal temporal evidence gate;
6. Savitzky–Golay;
7. undecimated Haar-like shrinkage;
8. causal Kalman;
9. local PCA;
10. noise-normalized local PCA;
11. local PCA + FastICA + noisy Parzen.

All 11 completed in 175.0 seconds. Peak RSS was 3.62 GiB. The run wrote and
verified 22 BigTIFF stacks totaling 3.93 GB; every stack has 560 pages at
340×573. Exact signal-plus-remainder closure was zero to float precision for
all methods.

## Comparable detection results

The unfiltered Parzen Innovation residual, recomputed with this exact evaluator,
scored mean recall `0.330`, fixed-budget recall `0.641`, and 70 candidates.
The prior Raw Direct contextual anchor scored `0.606`, `0.657`, and 232
candidates. Raw Direct is not the same input representation, so it is a project
anchor rather than the denoiser ablation baseline.

| Method | Mean recall | Fixed recall | Candidates | Known matches | Quiet RMS | Peak / area retained | Peak error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Unfiltered Parzen Innovation | 0.330 | 0.641 | 70 | 27 | 1.000 | 1.000 / 1.000 | 0 |
| Frame gamma | 0.104 | 0.629 | 24 | 8 | 0.196 | 0.203 / 0.164 | 2 |
| Robust gamma | 0.318 | 0.641 | 64 | 26 | 0.469 | 0.769 / 0.614 | 0 |
| Quiet Wiener | 0.330 | 0.657 | 66 | 27 | 0.688 | 0.970 / 0.933 | 0 |
| Spatial gate | 0.330 | 0.657 | 64 | 27 | 0.607 | 0.978 / 0.952 | 0 |
| Temporal gate | 0.342 | 0.670 | 72 | 28 | 0.560 | 0.962 / 0.942 | 0 |
| Savitzky–Golay | 0.330 | 0.670 | 61 | 27 | 0.591 | 0.978 / 0.997 | 1 |
| Haar-like | 0.559 | 0.676 | 145 | 45 | 0.470 | 0.936 / 0.992 | 4 |
| Kalman | 0.353 | 0.681 | 78 | 29 | 0.359 | 0.732 / 0.786 | 3 |
| Local PCA | 0.464 | 0.717 | 89 | 37 | 0.312 | 0.959 / 0.932 | 1 |
| Noise-normalized PCA | 0.452 | 0.717 | 75 | 36 | 0.309 | 0.954 / 0.927 | 1 |
| Component Parzen/ICA | 0.479 | 0.721 | 92 | 38 | 0.263 | 0.680 / 0.632 | 1 |

Candidate burden is not precision. As a descriptive audit only, known
matches/candidates rises from `0.211` for Raw Direct to `0.310` for Haar,
`0.416` for local PCA, `0.480` for noise-normalized PCA, and `0.413` for
component Parzen/ICA. Unlabeled candidates may be true neurons.

## Interpretation by family

### Pointwise methods

Literal frame min-max gamma is too harsh: quiet energy falls sharply, but it
retains only 20% of median peak amplitude and recall collapses. Robust gamma is
less destructive but offers no detection advantage over the input.

Quiet Wiener is the useful pointwise control. It preserves 97% of peak
amplitude, 93% of area, and exact peak timing while reducing quiet RMS to
0.688. Its mean recall is unchanged, although fixed-budget recall improves by
0.017.

### Spatial and temporal gates

The spatial gate is the most faithful conservative attenuation: peak
retention `0.978`, area `0.952`, waveform correlation effectively `1.0`, no
median peak shift, and quiet RMS `0.607`.

The temporal gate improves mean recall by `0.012` and fixed recall by `0.029`
over unfiltered Parzen Innovation while preserving 96% of peak amplitude and
94% of area. Its compute kernel took 0.70 seconds for 560 frames, or about
1.25 ms/frame. It is causal and is the best current near-real-time candidate.

### Temporal-only methods

Savitzky–Golay is a strong offline visualization smoother: it nearly preserves
area and waveform shape, reduces candidates to 61, and raises fixed recall.
The current centered implementation is not causal.

Haar-like shrinkage provides the highest mean recall outside Raw Direct:
`0.559`, with 45 known matches and 37.5% fewer candidates than Raw Direct.
That benefit comes with waveform correlation `0.850` and a four-frame
(80 ms) median peak error. It is useful as a detector feature, not as a
timing-faithful reconstruction.

The causal Kalman filter suppresses quiet energy well, but its current
320 ms decay removes 27% of peak amplitude and shifts peaks by 60 ms. It needs
a shorter decay or larger process variance before being considered a signal
carrier.

### Local subspaces and component Parzen/ICA

Local PCA gives the strongest balanced offline result: mean recall `0.464`,
fixed recall `0.717`, 89 candidates, 96% peak retention, 93% area retention,
and one-frame median peak error.

Noise normalization lowers candidates from 89 to 75 and keeps fixed recall
unchanged, at the cost of one known match and `0.012` mean recall. It has the
highest descriptive known-match yield (`0.480`) and is the best lane for a
precision-oriented visual audit.

Component Parzen/ICA gives the highest fixed-budget recall (`0.721`), highest
mean recall among the three local methods (`0.479`), lowest quiet RMS
(`0.263`), and lowest synthetic NMSE (`2.026`). Relative to unfiltered Parzen
Innovation, mean recall rises by 45%. However, it retains only 68% of peak
amplitude and 63% of area. It is currently a useful proposal feature, not a
clean signal reconstruction.

The local methods process the entire 560-frame temporal matrix per patch.
Their measured throughput does not make them causal. Real-time deployment
would require a sliding or recursively updated subspace and online component
tracking.

## Why every automatic audit flag failed

The pass rule simultaneously required at least 85% peak and area retention,
waveform correlation at least 0.98, at most one frame of peak error, quiet RMS
at most 0.8, and semi-synthetic correlation at least 0.9.

All methods failed the last condition. Synthetic correlations ranged from
`0.152` to `0.297`; NMSE ranged from `2.026` to `27.963`. The fixture embeds
four morphology types in a real quiet residual crop, so structured residual
activity and artifacts remain in the observed mixture. This is intentionally
more demanding than denoising white Gaussian noise. The failure means none of
the methods cleanly identifies injected truth; it does not mean the detection
trade-offs above are unreal.

The completed root's machine-generated report calls Kalman the strongest
quantitative candidate because its original all-fail tie-break prioritized
synthetic correlation. That is not an overall recommendation. This report
supersedes that scalar ranking, and the runner now reports fixed-budget
detection and synthetic NMSE winners separately for future runs.

## Decision

Do not replace Raw Direct or unfiltered Parzen Innovation with a denoised
video.

For the next visual review, prioritize:

1. `06_noise_normalized_pca` for the most selective local-subspace output;
2. `05_local_pca` for shape preservation;
3. `07_component_parzen_ica` for strongest attenuation and fixed-budget
   detection;
4. `03_temporal_gate` for the real-time path;
5. `04_temporal_haar` for the high-recall auxiliary detector.

Audit both signal and remainder. Any recognizable propagating activity or
neuron morphology in the remainder is evidence of signal leakage. The most
defensible next model experiment is a late fusion of Raw Direct with one
conservative causal gate and one local-subspace proposal score, evaluated with
burst-held-out calibration. It should not train directly on unmatched
candidates as negatives.

## Artifacts

- machine-readable metrics:
  `Outputs/HierarchicalParzenICA/spon_ca_burst_sequential_denoise_audit_v1/metrics.json`;
- concise comparison:
  `Outputs/HierarchicalParzenICA/spon_ca_burst_sequential_denoise_audit_v1/comparison.tsv`;
- all method videos:
  `Outputs/HierarchicalParzenICA/spon_ca_burst_sequential_denoise_audit_v1/methods`;
- implementation manifest:
  `examples/spon_ca_burst_sequential_denoise_audit.example.json`;
- reproducible workflow:
  `docs/workflows/spon_ca_burst_sequential_denoise_audit.md`.
