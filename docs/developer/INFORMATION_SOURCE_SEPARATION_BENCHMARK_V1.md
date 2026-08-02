# Information-theoretic source-separation benchmark v1

Status: bounded implementation specification, 2026-08-01.

## Authorization

This specification authorizes repository code, focused tests, generated and
real-background semi-synthetic fixtures, read-only preflight, and tiny CPU
smoke runs. It does not authorize a full Spon Ca Burst run, a GPU run, package
installation, modification of completed outputs, or evaluation on the frozen
final benchmark.

## Scientific question

Determine whether structured source-separation methods can recover compact
neuronal sources and faithful calcium traces while separating background,
structured artifact, and measurement noise. Downstream proposal and ranking
utility is secondary and must be measured without replacing the scientific
trace with an amplitude-distorting detector score.

## Comparison tracks

The eventual program reports two tracks:

1. `controlled_input`: applicable methods receive the same registered input,
   calibration, sample budget, patch extraction, and outer evaluation.
2. `native_best`: each method receives its scientifically appropriate frozen
   preprocessing, selected without the held-out benchmark.

Every result records the track. Results from different tracks are not treated
as a direct algorithm-only ablation.

## Initial method panel

References:

- `raw_direct_reference`;
- `amplitude_pca_rank8_reference`;
- `local_pca_fastica_wiener_reference`;
- `component_fastica_parzen_reference`.

New candidates:

- `caiman_cnmf_reference_adapter`;
- `multilag_sobi`;
- `kernel_hsic_pairwise_rotation`;
- `knn_mi_pairwise_rotation`;
- `group_energy_isa`;
- `spatial_noisy_parzen_infomax`.

The bounded kernel and k-nearest-neighbor methods must use these qualified
names. They are not presented as exact KICA or an unrestricted reproduction of
MILCA. An unavailable external CNMF backend is an explicit result, not a silent
fallback to ordinary NMF.

## Evaluation stages

### G0 — numerical integrity

- finite outputs and objectives;
- declared convergence or explicit unresolved state;
- deterministic replay for fixed seeds;
- reconstruction closure within the configured tolerance;
- no collapsed, duplicate, or explosive components.

### G1 — generated identifiability

Generated cases cover isolated, overlapping, synchronous, correlated,
fast-onset, slow-plateau, similar-persistence, illumination, motion, clipping,
heteroscedastic-noise, pure-noise, and unresolved mixtures. Selection uses
source truth and never real benchmark labels.

### G2 — real-background semi-synthetic validity

Known compact sources are injected into multiple real quiet Spon crops. The
test reports source recovery, existing-structure leakage, amplitude/timing
fidelity, and failure to abstain. No method advances by succeeding only on
white-noise fixtures.

### G3 — nested sparse-positive utility

Only G0--G2 survivors may enter leave-one-burst-out selection. ROI identity is
grouped. The outer burst is not used for rank, lag, kernel, component, sign,
threshold, or fusion selection.

### G4 — frozen benchmark

Final evaluation requires the hard-ROI adjudication and an exhaustively
reviewed bounded field. It separately reports original/confirmed/inclusive
labels, original/observation timing, proposal/ranking/NMS/localization/temporal
failures, and precision only where exhaustive labels make precision defined.

## Primary metrics

Truth-aware separation metrics:

- permutation/sign/scale-aligned source correlation;
- source normalized mean squared error;
- cross-talk between estimated and nonmatching sources;
- mixing-matrix or subspace error where identifiable;
- spatial footprint intersection-over-union and centroid error;
- peak and area retention;
- onset and peak-frame error;
- neural leakage into background and artifact channels;
- background/artifact leakage into neural channels;
- closure error and residual whiteness;
- correct unresolved/abstention frequency.

Robustness metrics:

- median and worst seed;
- aligned component stability;
- sensitivity to neighboring ranks, lags, patches, and regularization;
- case-family wins/ties/losses;
- runtime, peak RSS, output size, and streaming feasibility.

Detection metrics are downstream only: known-positive recall at budgets 10,
20, 40, 58, 80, and 100; candidate burden; proposal coverage; identical-union
ranking; and decomposed failure class. Unmatched candidates remain unknown.

## Selection rule

Selection is lexicographic:

1. pass numerical integrity;
2. pass amplitude and timing fidelity for a scientific reconstruction;
3. maximize preregistered source recovery on development fixtures;
4. prefer the simpler configuration within the configured equivalence margin;
5. use nested training bursts only for downstream detector-feature selection.

A detector-only auxiliary may fail reconstruction-fidelity gates, but its
artifact name and report must state that it is not the scientific trace.

## Initial implementation order

1. strict manifest and collision-safe preflight;
2. truth matching and fidelity metrics;
3. generated and semi-synthetic fixture contracts;
4. multi-lag SOBI reference;
5. bounded HSIC and kNN-MI rotation references;
6. CNMF backend adapter with explicit availability audit;
7. grouped-source prototype;
8. Parzen Infomax only after the existing attenuation and unresolved failures
   are represented in the fixtures;
9. deterministic tiny smoke report;
10. review of surviving methods before any full-data authorization.

## Non-negotiable critique defenses

- Preserve the original movie, labels, and completed output roots.
- Do not tune on the final benchmark.
- Record every attempted configuration, including failures.
- Do not infer precision from sparse positives.
- Do not use a larger proposal union as evidence of better ranking.
- Do not call a visually smooth movie a valid separation without truth-based
  leakage and fidelity evidence.
- Do not force a component when the assumptions are unidentifiable.
- Do not use labels to choose component sign, permutation, or source count.
- Do not compare native-best methods as if preprocessing were controlled.
- Keep source reconstruction and detector scoring as separate artifacts.
