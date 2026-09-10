# ICA and Whitening Hyperparameter Evaluation Results

**Experiment:** `spon_ca_burst_ica_whitening_evaluation_v1`  
**Original automated decision:** `stop_before_real_data`  
**Amended downstream decision:** continue real-data evaluation with the
synthetic result retained as a non-blocking warning  
**Evidence scope:** truth-known computational fixtures only

## Outcome

The exact conditional-factorial screen completed all 106,080 preregistered
fits: 27,936 temporal, 33,600 spatial, and 44,544 joint spatiotemporal. The
four-shard merge found no missing or duplicate fit IDs and one implementation
hash. No family contained a configuration that passed every original numerical
and truth-known gate, so the automated runner stopped before labels. After
inspection of the fixture, including an inactive third source in some 96-frame
cases, the investigator explicitly reclassified this result as a diagnostic
warning rather than a veto. Real-data stages may proceed, while literal source-
recovery and biological-source claims remain restricted.

| Family | Resolved fraction | Best truth correlation | Passing fits |
|---|---:|---:|---:|
| Temporal | 0.229 | 0.598 | 0 |
| Spatial | 0.273 | 0.471 | 0 |
| Joint spatiotemporal | 0.344 | 0.525 | 0 |

The required truth correlation was 0.70. Even after restricting to fits that
passed every other gate, the best values were 0.574 temporal, 0.465 spatial,
and 0.525 joint. This makes the stop decision insensitive to isolated
conditioning, convergence, preservation, crosstalk, or null-abstention
failures.

## Descriptive findings

Across the full design, mean truth correlation was 0.370 temporal, 0.328
spatial, and 0.339 joint. Mean crosstalk remained below the 0.25 ceiling for
all three families, while mean trace preservation was 0.771, 0.755, and 0.797,
respectively. Mean null-unresolved accuracy differed sharply: 0.852 temporal,
0.184 spatial, and 0.086 joint. These are screen-wide descriptions, not
biological performance estimates.

Rank 2 outperformed rank 4 descriptively in truth recovery (0.387 versus
0.292) and convergence (0.997 versus 0.848). No-whitening controls had the
highest screen-wide mean truth correlation (0.389) and perfect trace
preservation by construction. The whitening geometries traded preservation
against decorrelation rather than producing a uniformly superior setting;
none is eligible for promotion from this screen.

The median learned-response summaries were:

- temporal frequency centroid 11.76 Hz and bandwidth 5.18 Hz;
- spatial radial centroid 0.249 cycles/pixel, bandwidth 0.102 cycles/pixel,
  and anisotropy 0.276;
- joint temporal centroid 5.67 Hz, spatial centroid 0.232 cycles/pixel, and
  best rank-1 separable energy fraction 0.9986.

The last value indicates that the learned joint kernels were almost separable
on these fixtures; it does not establish that biological space-time structure
is separable. Median Raw/output correlation was 0.935 temporal, 0.923 spatial,
and 0.956 joint. Median approximate event-to-quiet ratios were 1.370, 1.379,
and 1.315. These are declared approximate fixture integrity metrics, not true
biological SNR.

## Statistical interpretation

The Cartesian factors have complete descriptive main-effect tables, and the
three paired seeds have per-cell dispersion tables. Continuous parameters have
within-compatible-cell Spearman response associations. Formal Sobol
sensitivity indices are not identifiable because the design used scrambled
Sobol coverage without the paired A/B/AB matrices required by variance-
decomposition estimators. No multiplicity-adjusted inferential claim is made
from the association table.

Across 35,360 paired point groups, median seed SD was 0.0058 for truth
correlation, 0.0057 for crosstalk, and 0.0106 for trace preservation (95th
percentiles 0.0191, 0.0210, and 0.0480). The failed recovery gate is therefore
not plausibly explained by one unstable seed in this screen.

## Scientific gates

- **Computational completion:** pass; exact design and merge validated.
- **Truth-known numerical/signal gate:** fail; zero passing fits in every ICA
  family.
- **Biological label utility:** not evaluated in this synthetic run; authorized
  as the next separately frozen real-data stage.
- **Within-recording finalist:** not established.
- **Scientific-audit media:** not applicable to this synthetic run. The enabled
  audit remains mandatory for eligible real-data finalists.
- **Independent confirmation:** not attempted and not established.

## Recommended next experiment

Carry this warning into the real-data grid and separately diagnose why source
recovery saturates below 0.70, especially the near-separability of joint
kernels, rank-4 degradation, conditioning failures, and poor null abstention
outside the temporal family. Do not interpret real-data ICA components as
verified biological sources. If formal Sobol
indices remain desired, preregister a Saltelli-compatible A/B/AB sampling
design rather than treating generic Sobol points as sufficient.

Primary artifacts are under
`Outputs/ICAWhiteningEvaluation/spon_ca_burst_ica_whitening_evaluation_v1/`.
