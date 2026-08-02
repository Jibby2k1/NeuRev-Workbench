# Information source-separation implementation checkpoint v1

Date: 2026-08-01

## Decision

The bounded implementation phase is complete enough to make the first generated
screen selectable. The screen has not been launched. Full Spon, semi-synthetic,
GPU, and final benchmark runs remain unauthorized.

## Implemented

- Versioned scientific contract and strict example manifest
- Thirteen exact B/S/A/N fixture families over configurable seeds and SNR
- Real quiet-crop source injection with explicit native-background uncertainty
- Permutation/sign/scale-aware source recovery and cross-talk metrics
- Trace amplitude/area/onset/peak/waveform metrics
- Footprint IoU and centroid metrics
- Multi-lag SOBI with bounded Jacobi joint diagonalization
- Qualified bounded HSIC and kNN-MI pairwise rotation references
- Gated grouped-energy HSIC independent-subspace prototype
- Exact adapters for amplitude PCA, full-window spatial FastICA, and dense
  FastICA/Wiener references
- External CaImAn CNMF availability/version contract with no ordinary-NMF fallback
- Transparent label-free component qualification and unresolved output
- Preregistered screen/confirmation design and unresolved-first finalist rule
- Resumable per-fit screen runner with atomic configuration artifacts
- Read-only tiny and generated-screen preflights
- Seventeen focused passing tests and syntax validation

## Tiny smoke

Output:

```text
Outputs/InformationSourceSeparation/tiny_smoke_v1
```

Eight CPU fits completed. On one isolated fixture, scale-aligned mean temporal
source correlation was 0.706 for PCA, 0.738 for SOBI, 0.917 for bounded HSIC,
and 0.950 for bounded kNN-MI. The information rotators used only two smoke
sweeps and did not converge. Every smoke method forced a decomposition in the
ambiguity fixture. These values validate interfaces and motivate the unresolved
gate; they are not scientific model selection.

## Generated-screen preflight

Proposed root:

```text
Outputs/InformationSourceSeparation/generated_screen_v1
```

The read-only audit passes every resource and collision gate:

| Item | Audited value |
| --- | ---: |
| Screen fixtures | 14 |
| Screen configurations | 48 |
| Screen fits | 672 |
| Estimated peak RAM | 523 MiB |
| Estimated output | 133 MiB |
| Available disk | approximately 1.9 TiB |
| CPU threads | 2 |
| GPU | not requested |

Worst-rank write-free timing probes measured:

- HSIC rank 12, one sweep: 1.57 seconds;
- kNN-MI rank 8, one sweep: 0.30 seconds.

Allowing for ranks, bandwidths/neighbors, up to eight sweeps, fixture loading,
metrics, and atomic artifacts, the conservative screen estimate is roughly
20--30 minutes on the current CPU. Early convergence may reduce it.

## Staged versus Cartesian design

| Design | Fits |
| --- | ---: |
| Full Cartesian screen on all fixtures | 9,360 |
| Preregistered screen | 672 |
| Maximum confirmation after selection | 1,365 |
| Maximum staged total | 2,037 |

The confirmation stage is not implied by authorizing the screen. If no method
passes complete execution, convergence, and unresolved behavior, confirmation
does not run.

## External CNMF decision

CaImAn is not installed. The adapter reports unavailable, unfrozen, and not fit
authorized. Installing it can materially change the environment and requires a
separate explicit decision. The generated screen can run without CNMF, but the
eventual comparison panel is scientifically incomplete until a pinned CNMF
reference is available or explicitly waived.

## Required selections

1. Select or decline the 672-fit CPU-only generated screen.
2. Separately authorize or defer installation and version freezing of CaImAn.

Neither choice authorizes the confirmation matrix, semi-synthetic matrix, full
Spon evaluation, GPU use, or final benchmark.
