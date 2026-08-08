# Spon Ca Burst multi-lag MSICA v5 results

## Decision summary

The experiment tested genuine multi-lag dependence objectives and full delay
embeddings in two architectures:

- Raw -> MSICA.
- Raw -> MSICA -> MSLN.

The completed program supports a qualified next step, not a universal winner.
Raw -> MSICA alone did not beat Raw Direct. After MSLN, the global label-free
selector also missed, but two separately predeclared full-embedding objective
families produced promising family-conditioned results.

At the fixed guardrail of 58 candidates per burst (232 total) and 79 sparse
known-positive labels:

| Result | Known matches |
|---|---:|
| Raw Direct reference | 49/79 |
| Prior switched five-seed ensemble | 52/79 |
| Raw -> MSICA global label-free winner | 39/79 |
| Raw -> MSICA -> MSLN global label-free winner | 44/79 |
| Full-embedding CS-Parzen family champion | 54/79 |
| Full-embedding normalized-HSIC family champion | 54/79 |
| Full-embedding KSG-MI family champion | 53/79 |
| Full-embedding matrix-Renyi family champion | 50/79 |
| Protected best over all frozen lanes | 60/79 |

The 60/79 result is a label-assisted descriptive ceiling and is not deployable.
The 54/79 CS-Parzen and HSIC results were selected without labels within their
predeclared families, but eight objective/formulation strata were inspected.
They are therefore provisional family-conditioned findings requiring
independent-recording confirmation.

## Experiment scope

- 66 parameter-calibration fits.
- 96 expanded lag/profile/weight fits.
- Four objectives: CS-Parzen, KSG mutual information, normalized HSIC, and
  matrix-Renyi mutual information.
- Multi-lag profiles through 16 frames (320 ms).
- Full embeddings through seven coordinates.
- 15 objective-diverse configurations after the positive held-out-gain
  amendment.
- Five real-data resampling seeds and five synthetic seeds per frozen
  configuration.
- All 30 prior MSLN contexts.
- 1,110 Raw -> MSICA -> MSLN lanes.
- Labels excluded from fitting and selection. Preflight used labels only for
  coordinate validation and the required projection overlay.
- Unmatched candidates remain unknown, not negative.

## Objective and stability findings

The two-output multi-lag innovation direction was highly stable across seeds,
but its raw known-label recovery was weak. Full embeddings exposed a useful
residual-subspace energy representation.

Full-embedding normalized HSIC had the cleanest positive held-out-gain
behavior. Full-embedding CS-Parzen was also promising. KSG was mixed across
sampling seeds, and matrix-Renyi embedding was unstable. These diagnostics
argue against pooling every objective under one undifferentiated selector.

The global event/quiet selector was poorly aligned with known-label recovery:
it selected 44/79 even though predeclared family champions reached 54/79. The
next selector should combine held-out dependence gain, seed/map consistency,
and event/quiet contrast.

## Resource and numerical diagnostics

The run used one CUDA worker, four numerical-library threads, CPU-backed output
maps, and eight-frame projection chunks. Live host memory remained bounded.
CUDA projection and MSLN parity errors were below 7.2e-7. Eighteen focused
regression tests passed.

## Recommended confirmation

Freeze two independent-recording confirmation arms:

1. Full-embedding CS-Parzen followed by MSLN.
2. Full-embedding normalized HSIC followed by MSLN.

Freeze each arm's exact family-specific label-free context rule before opening
confirmation labels. Do not promote the 60/79 protected-ceiling lane.

## Local artifacts

The immutable scientific root is:

Outputs/HierarchicalParzenICA/spon_ca_burst_multilag_msica_v5

The authoritative neutral grayscale/orange presentation package is:

Outputs/HierarchicalParzenICA/spon_ca_burst_multilag_msica_v5_deliverables_neutral

It contains three 560-frame videos, 11 event-max maps, three paper/slide
figures, per-experiment concise reports, the conclusive report, a family
champion table, and all 1,110 pipeline lane metrics. Generated Outputs are
ignored by repository policy and are not part of the Git commit.
