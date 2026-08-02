# Information Source Separation Generated Screen GPU v1

## Answer first

The 672-fit generated screen completed, but it does **not** justify generated
confirmation or a Spon Ca Burst benchmark yet.

The information-theoretic rotators recovered resolvable sources better than
amplitude PCA. The best CUDA HSIC configuration (`rank=4`, bandwidth scale
`2.0`) reached `0.8747` mean absolute correlation on resolvable fixtures versus
`0.7666` for PCA rank 8, while reducing mean crosstalk from `0.3892` to
`0.1713`. The best kNN-MI configuration (`rank=6`, `k=5`) reached `0.8600`
with `0.1591` crosstalk.

Those gains are not selectable because every HSIC and kNN-MI configuration
falsely declared both unresolved controls resolved. The current conclusion is:

> HSIC and kNN-MI are promising separators, but their shared component
> qualification/abstention layer is not safe enough to advance them.

## Run contract and integrity

- Completed root:
  `Outputs/InformationSourceSeparation/generated_screen_gpu_v1`
- Fits: `672 / 672`
- Fixtures: 14 (seven cases, two held seeds, median SNR)
- Configurations: 48
- GPU scope: normalized-HSIC objective only, PyTorch float64 on `cuda:0`
- CPU references: PCA, multi-lag SOBI, and kNN-MI
- Real Spon data used for fitting: no
- Semi-synthetic stage: not run
- Generated confirmation: not run
- Output size: 6.2 MiB
- `metrics.json` SHA-256:
  `d0172e7c8205043ed0a96bbb04a0df5ebc63b91bbb56b25973338c48aabdd58d`

The CUDA preflight passed with `1.39e-17` absolute CPU/GPU normalized-HSIC
error against a `1e-10` tolerance. CaImAn `1.13.1` and its CNMF class were
verified in the isolated Python 3.11 environment, but CNMF was not fitted in
this generated screen.

## Resolvable-fixture results

The unresolved controls are excluded from the recovery aggregates in this
table; they contribute only to abstention accuracy.

| Family | Best configuration by resolvable mean | Mean correlation | Worst correlation | Mean crosstalk | Unresolved accuracy |
|---|---|---:|---:|---:|---:|
| PCA | rank 8 | 0.7666 | 0.2616 | 0.3892 | 6/6 across PCA configs |
| SOBI | rank 8, lags 1/2/4/8/15, shrinkage 0.1 | 0.7663 | 0.5857 | 0.3105 | 0/2 for this config |
| CUDA HSIC | rank 4, bandwidth 2.0 | **0.8747** | 0.3695 | 0.1713 | **0/2** |
| kNN-MI | rank 6, k=5 | 0.8600 | 0.3207 | **0.1591** | **0/2** |

Relative to PCA rank 8, CUDA HSIC gains `+0.1081` mean correlation and reduces
crosstalk by `0.2179`. kNN-MI gains `+0.0934` and reduces crosstalk by `0.2301`.
These are generated-fixture effects, not neuron-detection effects.

HSIC led isolated, overlap, and similar-persistence cases. kNN-MI led the
motion-edge and saturation cases. SOBI led the synchronous case. This pattern
supports a genuine complementarity hypothesis rather than a single universally
best contrast.

## Abstention failure

The label-free qualification rule combines spatial compactness, lag-1
persistence, non-Gaussianity, and burst evidence, then requires a minimum top
score and top-two margin. Its behavior was:

- PCA: all 6 unresolved rows correctly abstained;
- SOBI: 18/54 unresolved rows abstained; only short-lag variants were reliable;
- HSIC: 0/18 abstained;
- kNN-MI: 0/18 abstained.

For HSIC, unresolved top scores were `0.8145–0.8428`, well above the fixed
`0.55` threshold. This is not a near-threshold miss. The rotator produces
compact, burst-like components even where the fixture is intentionally not
identifiable, so a morphology/burst score alone cannot represent separation
identifiability.

## Selection audit and correction

The original screen summary included unresolved-control correlations in its
recovery averages. That is methodologically incorrect because recovery is not
a meaningful success criterion where the specification requires abstention.
The completed root is preserved unchanged.

Two non-destructive posthoc audits were written beside it. The final audit is:

```text
Outputs/InformationSourceSeparation/generated_screen_gpu_v1_posthoc_selection_audit_v2
```

It makes two corrections:

1. unresolved controls affect only abstention accuracy;
2. the manifest's frozen `0.01` equivalence margin is used to prefer lower
   rank rather than floating-point differences.

The corrected eligible set is:

- PCA rank 4 (`0.7586` resolvable mean correlation);
- SOBI rank 4, lags 1/2/4, shrinkage 0.02 (`0.7395`);
- SOBI rank 4, lags 1/2/4, shrinkage 0.0 (`0.7394`).

This correction is explicitly posthoc and is not presented as preregistered
confirmation evidence.

## Runtime and numerical behavior

- Total recorded fit time: 273.3 seconds
- CUDA HSIC: 126 fits, 228.8 seconds, median 1.28 seconds
- CPU methods: 546 fits, 44.6 seconds
- HSIC, kNN-MI, and PCA convergence: 100%
- SOBI convergence: 376/378 fits

The screen demonstrates usable CUDA execution and numerical parity; it was not
designed to establish a GPU speedup claim.

## Critical critiques

1. **Only two unresolved rows per configuration.** A perfect 2/2 is weak
   evidence, and 0/2 is decisive only for this specific control. A resolution
   continuum with more independent seeds is required.
2. **The confidence layer measures neural likeness, not identifiability.** It
   needs stability, objective curvature, mixing coherence, or multi-start
   consensus features.
3. **Method-specific confidence calibration can overfit.** Thresholds must be
   calibrated on disjoint generated cases/seeds and evaluated once on a held
   abstention set.
4. **Scale-aligned source correlation is not amplitude fidelity.** Peak/area
   retention and timing gates remain mandatory before scientific use.
5. **Generated truth is not the Spon benchmark.** Sparse known-positive labels
   cannot supply ordinary precision because unlabeled candidates are unknown.
6. **CaImAn is installed but not yet a measured reference.** CNMF requires a
   bounded adapter fit, parameter contract, and its own preflight before it can
   enter any comparison.

## Recommended next experiment

Freeze the unmixing configurations and run an **identifiability/abstention
calibration stage**, not the 1,365-fit confirmation.

The calibration set should span spatial overlap, temporal collinearity,
source-amplitude ratio, SNR, saturation, and motion contamination with disjoint
calibration and evaluation seeds. Candidate confidence features should include
multi-start source stability, bootstrap/subsample stability, mixing-vector
coherence, dependence-objective curvature, and residual dependence.

Primary metrics should be false-resolution rate, unresolved sensitivity,
coverage, selective source-recovery risk, risk-coverage area, worst-case
correlation/crosstalk, and confidence intervals over independent fixtures. Only
after a frozen confidence rule passes should the best HSIC and kNN-MI variants
join PCA and eligible SOBI in generated confirmation. Semi-synthetic Spon,
CaImAn CNMF fitting, and the final fixed-budget Spon benchmark remain later,
separately authorized stages.
