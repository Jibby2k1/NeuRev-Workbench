# Spon Ca Burst information-theoretic source separation

> [!IMPORTANT]
> **Current status: implementation checkpoint only.** Numerical interfaces,
> truth-aware metrics, generated fixtures, real-background injection tooling,
> staged design, and a tiny CPU smoke are complete. No full generated screen,
> semi-synthetic matrix, Spon benchmark, GPU run, or scientific model selection
> has been authorized or performed.

## Purpose

This program asks whether structured source-separation methods can recover
compact neuronal sources and faithful calcium traces while separating
background, structured artifacts, and measurement noise. Detector proposal and
ranking utility is downstream and cannot substitute for source validity.

The scientific contract is
[`INFORMATION_SOURCE_SEPARATION_BENCHMARK_V1.md`](../developer/INFORMATION_SOURCE_SEPARATION_BENCHMARK_V1.md).

## Implemented checkpoint

### Numerical methods

- PCA-whitened reference;
- multi-lag SOBI with covariance shrinkage and bounded Jacobi diagonalization;
- bounded pairwise normalized-HSIC rotation;
- bounded pairwise kNN mutual-information rotation;
- gated grouped-energy HSIC independent-subspace prototype.

The HSIC and kNN-MI methods are explicitly qualified pairwise-rotation
references. They are not called exact KICA or unrestricted MILCA.

### Truth and metrics

The implementation includes:

- permutation/sign/scale-aligned source matching;
- source correlation, NMSE, and cross-talk;
- trace peak, area, onset, peak-time, and waveform fidelity;
- footprint IoU and centroid error;
- a transparent label-free component evidence score;
- explicit `unresolved` output;
- exact additive B/S/A/N fixture closure;
- real quiet-crop source injection without pretending the native crop has a
  known background/noise decomposition.

Generated cases cover isolated, overlapping, synchronous, correlated,
fast-onset, slow-plateau, similar-persistence, illumination, motion, clipping,
heteroscedastic-noise, pure-noise, and ambiguity conditions.

### Optional external CNMF reference

The CaImAn adapter currently reports:

```text
available=false
fit_authorized=false
fallback_used=false
```

No ordinary NMF substitute is used. Installing and freezing CaImAn requires a
separate explicit decision.

## Frozen example manifest

```text
examples/spon_ca_burst_information_source_separation_v1.example.json
```

The full Cartesian design contains 195 fixtures, 48 configurations, and 9,360
fits. That Cartesian matrix is deliberately not authorized.

The staged design reduces it to:

| Stage | Fixtures | Configurations | Maximum fits |
| --- | ---: | ---: | ---: |
| Screen | 14 | 48 | 672 |
| Confirmation | 195 | at most 7 finalists | 1,365 |
| Total staged ceiling | — | — | 2,037 |

The screen includes seven critique-sensitive cases, two seeds, and the median
SNR. Finalist selection first requires complete finite execution, sufficient
convergence, and correct unresolved behavior. Only then does it rank source
correlation, worst-case recovery, cross-talk, and complexity.

## Read-only preflight

```bash
.venv-neurobench/bin/python \
  -m neurobench.experiments.information_source_separation preflight \
  --config examples/spon_ca_burst_information_source_separation_v1.example.json \
  --output-dir Outputs/InformationSourceSeparation/a_new_output_root
```

The preflight checks source geometry, output and partial-output collisions,
disk headroom, method counts, optional backends, and the CPU/GPU authorization
state. It does not write an artifact.

## Completed tiny smoke

```text
Outputs/InformationSourceSeparation/tiny_smoke_v1
```

The smoke executed eight fits: one configuration from PCA, SOBI, bounded HSIC,
and bounded kNN-MI on one isolated and one ambiguity fixture. It used no GPU,
performed no model selection, and did not run semi-synthetic or real benchmark
evaluation.

On the isolated fixture, scale-aligned temporal source correlation was:

| Method | Mean absolute correlation | Worst correlation | Mean cross-talk |
| --- | ---: | ---: | ---: |
| PCA reference | 0.706 | 0.672 | 0.339 |
| Multi-lag SOBI | 0.738 | 0.669 | 0.348 |
| Bounded HSIC rotation | 0.917 | 0.786 | 0.109 |
| Bounded kNN-MI rotation | 0.950 | 0.908 | 0.175 |

> [!WARNING]
> These are interface-smoke values from one seed, one SNR, one configuration,
> and scale-aligned temporal sources. The HSIC and MI fits did not converge
> within the intentionally shortened two-sweep smoke. None reported unresolved
> in the ambiguity case. The values are not evidence of a winning method or an
> amplitude-faithful reconstruction.

The unresolved failure is actionable: the new label-free qualification module
must be integrated into the staged runner before any screen is eligible.

## Validation

Focused tests cover:

- distinct-temporal-source recovery by SOBI;
- finite, deterministic, qualified HSIC and kNN-MI execution;
- nonlinear-dependence sensitivity;
- exact generated and real-injection closure;
- permutation/sign/scale alignment;
- trace and footprint metrics;
- strict manifest validation;
- explicit unavailable CNMF behavior;
- transparent resolved/unresolved qualification;
- staged fit counts and unresolved-first finalist selection;
- deterministic grouped-HSIC ISA behavior.

## Remaining implementation gates

1. Integrate component qualification and corrected retained-subspace accounting
   into a version-2 smoke runner.
2. Add the existing amplitude PCA and dense FastICA/Wiener references through
   frozen adapters rather than approximations.
3. Implement the screen runner with resumable configuration-level artifacts.
4. Add semi-synthetic source contribution matching in image space.
5. Decide whether to install and pin CaImAn for the CNMF reference.
6. Keep grouped ISA gated until SOBI/HSIC/MI pass generated unresolved checks.
7. Keep spatial noisy-Parzen Infomax gated until attenuation, motion, clipping,
   and ambiguity fixtures pass.
8. Review the 672-fit screen resource estimate before any run authorization.

No final benchmark claim is possible until hard-ROI adjudication and an
exhaustively reviewed bounded field are complete.
