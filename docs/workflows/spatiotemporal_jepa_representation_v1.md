# Compact spatiotemporal JEPA representation pilot v1

- Status: draft planning hold; one bounded non-scientific screen completed;
  claim-bearing execution is not authorized
- Program: `NREV-PRG-0001`
- Experiment: `NREV-EXP-0028`
- Planned run: `NREV-RUN-EXP-0028-PILOT-01`
- Owner role: research team
- Maintained configuration:
  `examples/spatiotemporal_jepa_representation_v1.example.json`
- Sanitized data inventory:
  `research/data-registry/jepa_raw_video_sources_v1.json`
- Bounded screen result:
  [compact spatiotemporal JEPA screen v1](../research/SPATIOTEMPORAL_JEPA_SCREEN_V1_RESULTS.md)

This protocol tests a compact joint-embedding predictive architecture (JEPA)
as self-supervised representation learning for calcium video. It is not a
V-JEPA-scale foundation model, an action-conditioned world model, or
reinforcement learning. The experiment remains on hold until the registered
preflight, dependency, resource, and execution-authorization gates pass.

## Question and decision

### Question

Does a compact raw-video JEPA improve frozen recovery of exact injected sources
on held-out empirical backgrounds under the identical frozen latent temporal-
change head used by a capacity-matched masked pixel autoencoder and a frozen
random encoder, and relative to the established frozen handcrafted feature
stack, without collapsing or deriving its apparent gain from registered
acquisition nuisances?

### Decision this experiment can change

A passing result may authorize exporting a small, frozen JEPA-derived feature
family for evaluation in `NREV-EXP-0021`. A failed or inconclusive result keeps
the current interpretable carrier/context/kinetic stack as the learning input
and closes this JEPA architecture/configuration without a wider sweep.

The experiment is related to prior automated challenge-boundary work but is not
a source for that already-supported historical claim. It must not promote
`NREV-CLM-0017` (biological source identity) or `NREV-CLM-0018` (full-field
precision). Any eventual claim about a learned representation requires a new,
evidence-backed atomic claim record after validation; this plan does not
pre-register a favorable claim.

### What a supported result would permit

A supported result permits only the following scoped statement:

> Under the identical frozen latent temporal-change head, a compact raw-video
> JEPA improved frozen exact-injection source recovery on the registered
> held-out empirical backgrounds relative to the strongest registered
> comparator.

It may also motivate four or fewer frozen representation summaries, such as
latent temporal change, center-versus-annulus latent contrast, masked prediction
error, or cross-window recurrence. Those summaries must enter
`NREV-EXP-0021` as one predeclared feature family, not as an unconstrained latent
vector.

### What this experiment cannot establish

- Biological neuron identity, neuron probability, or cell type.
- Full-field precision, specificity, or false-positive rate. Unmatched native
  candidates remain unknown.
- Independent-animal generalization. The 11 `060126` recordings may share an
  animal, preparation, date, and acquisition system.
- Denoising, even if a representation suppresses unpredictable noise.
- Causality, action-conditioned dynamics, planning, a world model, or
  reinforcement learning.
- A transferable physical length or time scale while pixel size and frame
  cadence remain unresolved for the `060126` TIFF contract.

## Frozen data contract

### Raw-only self-supervision

The learned JEPA and masked-autoencoder arms receive only read-only, normalized
uint16 grayscale frames from the 11 hash-frozen `060126` TIFFs. The data layer
must not expose ROI coordinates, burst intervals, expert labels, provisional
candidate labels, ICA arrays, local-standardized arrays, detector scores, or
other processed stages. It reads sequential clips through `tifffile.memmap` and
does not copy the raw corpus into the run output.

The frozen handcrafted comparator may compute its previously registered
carrier, spatial-context, and kinetic definitions from raw video. It is an
endpoint comparator only: no handcrafted or processed feature may enter a
learned encoder or its target.

### Recording split

The unit of separation is the complete recording. The split was fixed before
training by holding out the highest-numbered recording in each filename-derived
behavior stratum:

| Split | Recording IDs | Count |
| --- | --- | ---: |
| Training | `060126_01_rest`, `060126_02_left`, `060126_03_right`, `060126_04_rest`, `060126_05_right`, `060126_06_left`, `060126_07_rest`, `060126_08_left` | 8 |
| Validation/evaluation | `060126_10_rest`, `060126_12_left`, `060126_15_right` | 3 |

The split ID is `060126_behavior_stratified_last_trial_v1`. Filename-derived
rest/left/right words are acquisition strata, not neuronal labels. Validation
recordings are never used for parameter fitting, normalization fitting,
checkpoint selection, mask selection, or hyperparameter selection.

The complete Spon Ca Burst recording
`spon_ca_burst_3_hindbrain_to_tail_488_20ms` is excluded from self-supervised
training, normalization, model selection, and gate tuning. It enters only after
the final checkpoints and score definitions are frozen. Its primary use is
paired injection on quiet/background windows; its canonical ROI and burst
labels are secondary diagnostics only.

### Physical metadata boundary

All registered analysis is in native frames and pixels. Pixel size and frame
interval are unresolved in the `060126` inventory and may not be inferred from
the Spon recording, filenames, or prior experiments. Physical-scale or
cross-cadence claims remain blocked until source metadata is independently
registered.

## Frozen representation design

### Clip and token geometry

- Raw clip: 32 contiguous frames by 64 rows by 64 columns.
- Temporal guards: 16 raw frames before and after every sampled clip.
- Tubelet: 4 frames by 8 rows by 8 columns.
- Token grid: 8 by 8 by 8, or 512 tokens per clip.
- Sampling mixture within training recordings: 50% uniform, 25% high temporal
  MAD, and 25% low temporal MAD, selected without label or ROI access.
- Training clip bank: exactly 512 unique label-free clips per training seed,
  comprising 256 uniform, 128 high-temporal-MAD, and 128 low-temporal-MAD
  clips. The training-plan sampling seed is that arm pair's registered training
  seed (`1001`, `1002`, or `1003`), and JEPA and MAE share the resulting bank
  within that seed.
- Validation clip bank: exactly 96 fixed uniform clips, with 32 unique clips
  from each held recording. It is sampled once with seed `2001` and shared
  unchanged across all training seeds; it is diagnostic only and cannot select
  a checkpoint, mask, or hyperparameter.
- Normalization: one robust median/MAD transform fitted only on deterministic
  samples from the eight training recordings and then frozen for every arm and
  held-out source.

Four non-overlapping contiguous 3-D target cuboids of exactly 2 by 4 by 8
tokens are sampled per clip. Their union covers 256 of 512 tokens (50%); each
cuboid spans two temporal tubelets. The context is the exact complement, and
masking occurs before the context encoder. A mask that overlaps, misses the
40%-to-55% registered coverage interval, or exposes target pixels to the
context encoder invalidates the batch.

### Architecture and objectives

The shared architecture is a compact approximately 1–2 million parameter
Conv3D tubelet encoder and local predictor. The specific implementation may use
convolutions rather than a Vision Transformer; the scientific object is masked
latent prediction, not architectural resemblance to a large published model.

| Arm | Objective | Capacity/training contract |
| --- | --- | --- |
| `compact_jepa` | Predict stop-gradient EMA target embeddings at masked tubelets | Shared encoder/predictor, 5,000 updates, three seeds |
| `masked_pixel_autoencoder` | Reconstruct masked raw pixels | Same encoder/predictor initialization and sampled clips/masks; trainable parameter difference at most 1% |
| `frozen_random_encoder` | None | Same seeded encoder architecture, zero updates |
| `frozen_handcrafted_stack` | None | Frozen registered carrier/context/kinetic definitions; no label refit |

### Frozen score-head contract

The primary learned-arm comparison uses one identical, non-trainable score head
for `compact_jepa`, `masked_pixel_autoencoder`, and `frozen_random_encoder`:
`latent_temporal_change_norm`. Each frozen encoder receives the complete
unmasked clip after the same training-only robust normalization. At each latent
spatial position, the head computes the L2 norm across channels of the first
temporal difference, averages that norm over latent time, and bilinearly
upsamples the result to 64 by 64. The head has no learned weights and cannot be
fit to labels, injections, validation outcomes, or exact truth.

JEPA masked-prediction error and MAE masked-reconstruction error are retained as
objective-native secondary lanes with deterministic full mask coverage. They
are not interchangeable measurements: they never enter the primary metric,
never select the strongest primary comparator, and are never compared directly
as though they were the same score head. Any interpretation of those lanes is
within-objective and descriptive.

The handcrafted primary arm retains its frozen carrier, spatial-context, and
kinetic components. The carrier is pixelwise population standard deviation over
time; context is Gaussian-filtered carrier at sigma 1.2 minus sigma 4.0 under
SciPy defaults; kinetic score is the maximum sliding dot product after
pixelwise temporal-median centering with an L2-normalized length-`min(18,T)`
kernel `exp(-t/5) * (1 - exp(-t/1.5))`.

Comparator version
`carrier_context_kinetic_source_off_component_calibration_v1` reproduces the
established `major_next_steps` combined-score semantics on float32 source-off:
its structural calibration API is
`fit_source_off(source_off) -> frozen_callable`, and that callable is applied
to both source-off and source-on. A generic `score_pair` shortcut is not part
of the contract. Each of the three source-off component maps is separately
centered by its spatial median and divided by its normal-consistent spatial MAD
with a `1e-6` minimum scale. Those same three centers and scales are applied
unchanged to the corresponding source-on components, then the three robust-z
maps are summed with unit coefficients. Component definitions, weights, and
calibrations cannot use source-on, labels, or exact truth.

After each arm has formed its combined primary map, the evaluator applies the
common final pair-wide reporting/ranking calibration through the same
fit-source-off-then-freeze interface: source-off combined-map
spatial median and normal-consistent MAD, with a source-off standard-deviation
fallback only if degenerate and numerical lower bound `1e-8`, applied unchanged
to both source-off and source-on. Source-on self-normalization is forbidden at
both calibration stages.

JEPA and masked-autoencoder arms use identical training clips, masks, initial
encoder weights, optimizer, update count, and resource cap within each seed.
There is no hyperparameter sweep and no early stopping. A trainable arm/seed
that does not complete the exact update budget is invalid rather than compared
at an unequal budget.

The shared optimizer is AdamW with learning rate `1e-4`, constant learning-rate
schedule, betas `(0.9, 0.999)`, epsilon `1e-8`, weight decay `0.05`, batch size
eight, and global gradient-norm clipping at `1.0`. These are registered values,
not mutable library defaults; both trainable arms must report them in the
resolved run manifest.

CUDA training uses `bfloat16` autocast with float32 loss accumulation and
float32 metrics. Float16 is forbidden for this pilot because the matched
real-data resource smoke produced non-finite masked-autoencoder gradients.
CPU validation and deterministic reference calculations remain float32. A GPU
without bfloat16 support fails preflight rather than silently changing numeric
precision.

The registered training and training-plan sampling seeds are `1001`, `1002`,
and `1003`; the one shared validation sampling seed is `2001`. This pilot is
fixed at 5,000 optimizer steps and at most four GPU hours per trainable
arm/seed, with a total experiment cap of 24 GPU hours. The implementation has a
hard safety ceiling of 10,000 steps, but that ceiling does not authorize this
run to continue past 5,000. Any later escalation, even below 10,000, requires a
new configuration, run ID, and output root. A cap hit before 5,000 matched
updates yields an incomplete arm rather than a scientific result.

## Primary exact-truth empirical-background endpoint

### Paired fixture design

The primary evaluation operates on paired source-off/source-on raw clips:

```text
source_on = untouched_empirical_background + exact_injected_signal
```

Pair closure must be at most `0.51` float32 ULP, corresponding to one correctly
rounded float32 addition when forming source-on from source-off plus injection.
The maximum absolute residual is still reported, but no scale-independent
absolute threshold is used as a pass/fail gate. Only the injected footprint,
trace, two explicitly named center definitions, and identity are exact truth.
`footprint_peak_centers_yx` stores the integer row/column argmax of each injected
footprint and is the primary matching target because proposals are score-map
maxima. `placement_centers_yx` stores the continuous generative placement
center and is used only for morphology-sensitive secondary recovery and
localization. An ambiguous `centers_yx` field is forbidden. Native biological
structure is not decomposed, and an unmatched proposal is never converted into
a biological negative.

The four held-out background recordings are the three fixed `060126`
validation recordings plus the fully excluded Spon recording. Each contributes
three spatiotemporally disjoint 32-by-64-by-64 crops representing low, median,
and high temporal-MAD strata. Spon crops must lie outside every registered
burst interval. That Spon background selection uses only registered burst-
interval exclusion; ROI identity labels are neither queried nor used. No Spon
clip enters self-supervision.

For every background window, each injection seed (`3101`, `3102`, `3103`) is
crossed with source counts 1, 2, and 4. This yields exactly:

```text
4 recordings x 3 windows x 3 injection seeds x 3 source counts = 108 paired cells
```

Source count is a nested condition, not an independent replicate. The source
generator is frozen as follows:

- amplitude multiplier 1.0 relative to the source-off crop's native
  frame-difference MAD, computed as `1.4826` times the median absolute
  deviation of consecutive-frame differences after per-pixel temporal-median
  centering, with no implicit one-raw-unit or other nonzero floor;
- a zero, non-finite, or otherwise degenerate source-off frame-difference MAD
  aborts the run; automatic replacement is forbidden. A retry requires a newly
  frozen and versioned background-window manifest and a new run ID. A future
  intentional nonzero floor would require a new configuration and run ID and
  explicit per-fixture floor-activation reporting;
- deterministically assigned ellipse and crescent footprints, including crowded
  or overlapping-neighbor cases. Ellipse Gaussian sigmas are 2.5 pixels along
  rotated columns and 1.65 along rotated rows, at an angle uniform on
  `[0,pi)`. A crescent subtracts `0.72` times a Gaussian with sigmas 1.85 and
  1.35, centered at rotated offset `(column=1.15,row=-0.15)`, clips at zero,
  and peak-normalizes. Ellipse versus crescent is the parity of source index
  plus deterministic fixture-pattern seed;
- continuous-uniform center placement within an eight-pixel border. Unforced
  centers require 10-pixel separation. Near-neighbor and overlap fixtures force
  the first pair to distances 6.0 and 3.75 pixels, respectively; multi-source
  crowding mode is `(overlap, near_neighbor, separated)[injection_seed mod 3]`;
- one fixed peak-normalized calcium-like double-exponential kernel with a
  two-frame rise and eight-frame decay. For the registered 32-frame crop,
  onset is discrete-uniform over zero-based frames 2 through 21 inclusive;
  first-impulse amplitude is uniform on `[0.85,1.15)`. With probability 0.5,
  a second impulse is added at an integer lag 7 through 12 inclusive, capped at
  frame 30, with amplitude uniform on `[0.35,0.70)`. The convolved trace is
  peak-normalized before scaling by the source-off frame-difference MAD;
- fixture pattern seed is the first unsigned 64 bits of
  `SHA256(fixture_id|injection_seed)` modulo `2^63`; morphology/kinetics use
  NumPy `default_rng` at that seed plus `1,000,003`;
- no parameter may be tuned from exact-truth recovery.

Every fixture row records the measured source-off frame-difference MAD,
multiplier, resulting raw-unit peak, `amplitude_floor_raw_units=null`, and
`floor_activated=false`. Under this version any nonzero floor activation is a
contract failure, not an unreported numerical convenience.

### Operating point and primary metric

Each frozen arm emits one primary score map for the source-off clip and one for
the source-on clip under the score-head contract above. The source-off spatial
median and scale are fitted once per method/pair and applied unchanged to both
maps; the source-on distribution never calibrates itself. The deployment-like
source-on map and the paired intervention map `source_on_z - source_off_z` are
retained separately.

The primary metric is macro one-to-one recovery of exact injected sources from
the source-on map at a cap of up to four separated local-maximum proposals per
64-by-64 crop, minimum proposal separation two pixels, two-pixel border
exclusion, and a three-pixel match radius. Four is a ceiling, not a quota: if
fewer than four qualifying separated local maxima exist, the method returns
fewer candidates, reports the realized `candidate_count`, and never inserts
low-score filler proposals. Primary one-to-one matching uses
`footprint_peak_centers_yx`; continuous `placement_centers_yx` results are
reported separately by morphology. The primary grouping hierarchy is background
recording, background window, and injection seed; source count remains nested.

The primary comparator is the strongest of the three non-JEPA arms in the same
frozen evaluation table. The MAE and random entries use the identical
`latent_temporal_change_norm` head as JEPA; their objective-native errors are
excluded. The hierarchical bootstrap recomputes all comparator statistics and
reselects the maximum comparator inside every draw. An interval that fixes the
observed winning comparator across draws is forbidden. JEPA passes the primary
benefit gate only if:

1. macro source-on recall improves by at least 0.02 absolute;
2. the 95% grouped-bootstrap interval for the paired recall difference has a
   lower bound strictly above zero;
3. no held-out background recording loses more than 0.02 absolute recall; and
4. the direction remains positive for all three JEPA training seeds.

Leave-one-background-recording-out results are required sensitivity analyses,
not independent-animal replications. A paired sign-flip test over frozen
background-window/injection-seed clusters is secondary because the small number
of top-level recording groups limits inference.

### Secondary metrics

- JEPA deterministic full-coverage masked-prediction error and MAE deterministic
  full-coverage masked-reconstruction error, reported in separate descriptive
  within-objective lanes and never as a common-head comparison.
- Incremental intervention-map one-to-one recall.
- Injected-center score change in source-off background-MAD units.
- Localization error, duplicates, and weak-neighbor recovery.
- Morphology-sensitive recovery and localization relative to continuous
  `placement_centers_yx`, kept separate from primary footprint-peak matching.
- Source-footprint AUC against baseline-matched displaced pixels.
- Effect direction by background recording, temporal-MAD stratum, morphology,
  source count, and seed.
- Fixed B20 and B58 sensitivity at the 106 Spon known-positive occurrences,
  evaluated only after the primary representation and checkpoint freeze.

Known-center Spon sensitivity is diagnostic, not model selection. It cannot
estimate precision, and its near-ceiling carrier localization makes it an
insufficient primary endpoint. The 18 selected candidate-review labels are not
used for training, checkpoint choice, or primary evaluation.

## Nuisance, collapse, and falsification analysis

### Representation-collapse gate

Every trainable seed and held-out recording must have finite embeddings. The
frozen thresholds are:

- finite fraction exactly 1.0;
- mean feature standard deviation at least `1e-4`;
- covariance effective-rank fraction at least 0.10;
- first-principal-component variance fraction at most 0.50;
- primary recall range across JEPA seeds at most 0.05.

Linear centered-kernel alignment, effective rank, spectra, and primary metric
effects are reported across seeds. Crossing any collapse threshold rejects that
seed. Fewer than three valid JEPA seeds makes the primary result inconclusive.

### Nuisance audit

Frozen-encoder linear probes, with recording-aware grouping where applicable,
measure recording identity, filename-derived behavior, global intensity,
temporal position, row/column position, bleaching slope, annular background,
`dtype_rail_fraction`, and motion magnitude once `NREV-EXP-0025` supplies a
motion contract. `dtype_rail_fraction` is exactly the fraction of raw samples
equal to the declared integer dtype maximum. It is not called sensor saturation:
whether the dtype rail corresponds to the sensor rail remains unresolved while
effective sensor bit depth is unknown. High nuisance predictability is reported
rather than hidden; it is not itself evidence of failure or neuronal
specificity.

The scientific benefit gate requires the primary JEPA advantage to retain a
strictly positive grouped interval after nuisance matching or residualization.
If motion fields remain unavailable, the motion component of that gate is
unresolved and no nuisance-robust representation claim may pass.

The required counterfactuals are temporal reversal, within-clip phase
randomization, spatial block shuffling, source-off movies, and
recording-blocked label permutations. A gain that survives only a leakage-prone
random patch split, disappears under recording holdout, or tracks a nuisance
counterfactual materially qualifies or falsifies the proposed benefit.

### Predeclared falsifiers

- JEPA misses the 0.02 primary margin or its grouped interval crosses zero.
- A matched masked pixel autoencoder, random encoder, or frozen handcrafted
  stack equals or exceeds JEPA at the same endpoint.
- The advantage disappears under leave-recording-out or nuisance-residualized
  analysis.
- Embeddings cross a collapse threshold or fewer than three seeds remain valid.
- The effect changes sign across seeds or exceeds the 0.05 seed-range limit.
- Improvement appears only in the secondary near-ceiling Spon known-center
  diagnostic.
- A source-free, phase-randomized, shuffled, or blocked-permutation control
  reproduces the claimed effect.

## Frozen implementation pins

Preflight hashes complete file bytes with SHA-256. The fixture generator is
version `native_background_calcium_injection_v1`; the primary comparator and
evaluator suite is version `jepa_primary_comparator_suite_v1`; the handcrafted
member reports version
`carrier_context_kinetic_source_off_component_calibration_v1`.

| Maintained source | Frozen SHA-256 | Contract role |
| --- | --- | --- |
| `neurobench/experiments/neuron_identifiability/jepa_evaluation.py` | `3bed26ce6ee766f55951a87f8e675d4389c635164438adc04f9fa50310b587c2` | Fixture generator, pair evaluator, matching, hierarchical bootstrap |
| `neurobench/experiments/neuron_identifiability/jepa_training.py` | `4117aa88a06ec0aa4ef111b1bc98bbfda003985b47fd965f24edb9a7b7805f47` | Matched trainer and common latent temporal-change head |
| `neurobench/experiments/neuron_identifiability/jepa_comparators.py` | `54443b4f4e39e8abbba4351b9d9399329b6bb09a54044c0c29f46cc462dd1fa5` | Pair-safe handcrafted comparator |
| `neurobench/experiments/neuron_identifiability/jepa_pilot.py` | `1b3bb2a8873f235f8e03161ea40cac2cef74101d2ba6b7ffe8d9f6021e2197ed` | Score wiring, manifests, and orchestration |
| `neurobench/experiments/neuron_identifiability/major_next_steps.py` | `ac265170e625727a4a9ff8dd5f2e4a437f5cb14a83bb2eeea10f68e9ff2cc7d2` | Legacy combined-score equivalence reference |

Any mismatch blocks execution. A reviewed implementation change requires a
versioned configuration and new run ID rather than silently refreshing a hash.

## Required gates

| ID | Stage | Required | Criterion | Action on failure |
| --- | --- | --- | --- | --- |
| `raw_inventory` | preflight | yes | All 11 `060126` TIFF hashes, shapes, dtype, total frame count, and portable URIs match the committed descriptor. | Stop; do not repair provenance in place. |
| `raw_only_boundary` | preflight | yes | The self-supervised data API cannot expose Spon, labels, ROI/burst metadata, ICA, LS, detector scores, or processed stages. | Stop as leakage. |
| `recording_split` | preflight | yes | The exact 8/3 recording split is disjoint and normalization uses only the eight training recordings. | Stop as leakage. |
| `spon_holdout` | preflight | yes | Spon is absent from self-supervision, normalization, selection, and gate tuning. | Stop as leakage. |
| `clip_bank_contract` | preflight | yes | Each training seed produces 512 unique label-free clips in an exact 256/128/128 uniform/high/low split using that training seed; one seed-2001 validation bank contains 96 unique uniform clips, exactly 32 per held recording, and is shared across training seeds. | Stop and version the sampling plan. |
| `score_head_contract` | preflight | yes | JEPA, MAE, and random use the identical frozen `latent_temporal_change_norm` primary head; objective-native errors remain separate secondary lanes; every calibration uses `fit_source_off(...) -> frozen_callable` applied to off/on; handcrafted components use legacy-equivalent source-off component-wise robust-z followed by the common source-off-only combined-map calibration. | Stop as an endpoint-head or calibration confound. |
| `fixture_implementation_hash` | preflight | yes | The fixture-generator version, complete-file SHA-256, essential constants, and resolved manifest match the maintained configuration. | Stop; review and version any generator change. |
| `comparator_implementation_hash` | preflight | yes | The learned common head, pair evaluator, pair-safe handcrafted comparator, orchestration wiring, and legacy equivalence reference match their registered versions and complete-file SHA-256 values. | Stop; review and version any comparator change. |
| `motion_dependency` | preflight | yes | `NREV-EXP-0025` has an evidence-backed usable motion/residual-field contract. It is currently draft/not-evaluated, so `NREV-EXP-0028` is not ready for claim-bearing execution. | Hold execution authorization and all nuisance-robust claims. |
| `empirical_fixture` | preflight | yes | The 108 paired cells close within `0.51` float32 ULP (one correctly rounded addition), report absolute residual without gating on it, contain both named center fields and all frozen generator factors, and use exactly 1.0 times unfloored source-off frame-difference MAD; a degenerate MAD aborts without replacement. | Stop; retry only with a versioned new window manifest and run ID. |
| `proposal_cap_contract` | analysis | yes | Every method returns only separated local maxima up to cap four, reports actual candidate count, never fills to four, and keeps all three learned seed instances separate. | Invalidate the primary table. |
| `capacity_match` | execution | yes | JEPA and MAE share encoder/predictor initialization, clips, masks, 5,000-update budget, and the exact AdamW contract: LR `1e-4` constant, betas `(0.9, 0.999)`, epsilon `1e-8`, weight decay `0.05`, batch eight, and gradient clip norm `1.0`; trainable parameters differ by at most 1%. | Invalidate comparison. |
| `mask_contract` | execution | yes | Four non-overlapping 2-by-4-by-8-token cuboids cover exactly 50% of tokens, each spans two temporal tubelets, and masking precedes context encoding. | Invalidate affected batches/run. |
| `numerical_precision` | execution | yes | CUDA uses bfloat16 autocast with float32 loss/metrics, all losses and gradients remain finite, and float16 fallback is disabled. | Stop rather than change precision. |
| `collapse` | analysis | yes | All three JEPA seeds pass the frozen finite, variance, rank, and PC-dominance thresholds. | Reject or mark inconclusive. |
| `primary_gain` | analysis | yes | JEPA exceeds the strongest comparator by at least 0.02, grouped 95% lower bound is above zero, and no recording loses more than 0.02. | Do not advance JEPA features. |
| `seed_stability` | analysis | yes | All seed effects are positive and the primary recall range is at most 0.05. | Mark inconclusive. |
| `nuisance_residualized_gain` | analysis | yes | The primary advantage retains a strictly positive grouped interval after registered nuisance matching/residualization. | Do not make a representation-benefit claim. |
| `counterfactuals` | analysis | yes | Source-free and randomized controls do not reproduce the claimed source-recovery effect. | Reject or qualify mechanism. |
| `scientific_audit` | review | yes | The complete applicable audit inventory, media, compact indexes, and validation pass. | Audit incomplete. |
| `publication_boundary` | publication | yes | Sanitized evidence and release candidates contain no raw/private/non-portable payload. | Block publication. |

The native scaffold additionally registers an output-collision gate. A passed
software test, completed optimizer loop, or artifact count is never substituted
for a scientific gate.

## Provenance and resource contract

- Sanitized inventory: `NREV-DATA-JEPA-RAW-VIDEO-SOURCES-V1`; raw hashes are in
  `research/data-registry/jepa_raw_video_sources_v1.json`.
- Configuration:
  `examples/spatiotemporal_jepa_representation_v1.example.json`.
- Dependency lock: `pyproject.toml`.
- Hard experiment dependency: `NREV-EXP-0025`, currently an unsatisfied
  draft/not-evaluated readiness blocker for claim-bearing execution.
- `NREV-EXP-0027` is related future realistic-simulation work, not a hard
  dependency: this experiment's real-background exact-injection fixture is
  self-contained and version/hash gated.
- Output root:
  `Outputs/NeuronIdentifiability/NREV-EXP-0028/runs/NREV-RUN-EXP-0028-PILOT-01`.
- Maximum RAM: 24 GiB; maximum GPU memory: 16 GiB; CPU threads: four;
  minimum free disk: 32 GiB; maximum output: 8 GiB.
- Maximum total training: 24 GPU hours, including all trainable arms and seeds.
- Checkpoint interval: 1,000 optimizer steps; checkpoints and metadata are
  atomic and resumable only under the identical manifest hash.
- Completed output roots are never overwritten. A changed protocol,
  configuration, fixture, seed, or model requires a new run ID and output root.

The planned run record may capture a dirty-state digest because this package is
being scaffolded before commit. That record is not run authorization. Before
execution, code, protocol, configuration, inventory, dependency lock, and input
hashes must be frozen at an inspectable commit or a new planned run record must
capture the exact executable state.

## Required output inventory

The run must write small machine-readable evidence before large media:

```text
<output_root>/
  status.json
  resolved_config.json
  llm_context.json
  artifact_index.json
  validation.json
  REPORT.md
  00_preflight/
    input_inventory.json
    split_manifest.json
    clip_bank_manifest.json
    score_head_manifest.json
    implementation_hash_manifest.json
    normalization_manifest.json
    fixture_manifest.json
    resource_preflight.json
  01_training/
    arm_seed_status.json
    checkpoints/
    learning_curves.tsv
  02_representations/
    collapse_diagnostics.tsv
    seed_stability.tsv
  03_exact_truth_injection/
    paired_cells.tsv
    source_recovery.tsv
    grouped_primary_summary.json
  04_nuisance_and_controls/
    nuisance_probes.tsv
    counterfactuals.tsv
    nuisance_residualized_summary.json
  05_secondary_spon/
    known_positive_b20_b58.tsv
  06_scientific_audit/
    ... standard three-section evidence set ...
```

Every table uses stable recording, background-window, injection-seed,
source-count, source-identity, arm, and training-seed identifiers. Results must
be reconstructible without parsing filenames or decoding videos.

## Scientific audit implementation

The audit is mandatory and implements
`docs/workflows/SCIENTIFIC_AUDIT_OUTPUT_STANDARD.md`; no opt-out is registered.

For the exact-truth primary endpoint, the `1_Expert_Annotations` directory is a
section-pure exact-injection reference rather than a claim of human expert
annotation. Metadata must state
`annotation_source=deterministic_synthetic_injection_truth`. The expected 108
paired cells contain 252 exact injected-source occurrences. Produce a
truth-only synchronized source-off/source-on full-field video plus one close-up,
full-clip trace, and metadata record per injected occurrence. Primary truth
markers use `footprint_peak_centers_yx`; continuous `placement_centers_yx` may
appear only in clearly labeled morphology-sensitive secondary views.

The `2_Model_Annotations` directory contains arm-pure frozen primary-head
predictions for ten primary instances: three seed-specific JEPA, three
seed-specific MAE, three seed-specific random encoders, and one shared
handcrafted comparator. Across 108 cells and up to four local maxima, the
maximum is 4,320 primary proposal occurrences. Actual cardinality is recorded
and may be lower; no filler proposal is created. Seed-specific learned outputs
cannot be collapsed or selected across seeds. Every realized proposal receives
a close-up, full-duration trace, and metadata record. Objective-native JEPA and
MAE error lanes, if rendered, occupy explicitly separate secondary
subdirectories and cannot be pooled with or substituted for the primary-head
proposals. Unmatched proposals are marked unknown. Never display exact-truth
markers in a model-only video.

The label-free validation audit declares Expert `not_applicable`. Before any
media is rendered, freeze a 12-candidate surrogate panel per held-out recording
per arm using only the registered label-free score and deterministic tie-breaks.

The `3_Comparison` directory contains figures and tables only. It keeps nearest
candidate and one-to-one identity assignment separate and reports source-on,
source-off, paired intervention, localization, rank, and recovery. Grayscale
stage backgrounds and green/orange/yellow marker semantics follow the standard.

`llm_context.json`, `summary.json`, `artifact_index.json`, and
`validation.json` must agree on counts, provenance, operating points, section
applicability, unknown-candidate semantics, and claim boundaries. Videos require
frame-count, duration, dimensions, decode, and visible-marker validation before
the run is audit-complete.

## Publication and privacy boundary

Public evidence may contain the protocol, sanitized configuration, portable
inventory and hashes, aggregate tables, small representative figures, model
summaries, validation status, and checksums. Raw TIFF frames, complete learned
checkpoints unless separately approved, private labels, reviewer identities,
workstation paths, and large audit media remain under ignored `Inputs/` or
`Outputs/` or a checksum-addressed external archive.

Any evidence capsule must state that the exact truth covers only injected
sources and that native background candidates remain unknown. Clean-clone
registry, story-currentness, local-link, and publication-boundary checks must
pass before release.

## Review record

- Protocol frozen on: not frozen
- Protocol SHA-256: pending after review
- Code commit: pending executable-state freeze
- Bounded engineering screen: `NREV-RUN-EXP-0028-SCREEN-20260829-B`
  succeeded; the observed primary-gain and collapse values crossed screen-level
  thresholds, while the incomplete experiment-level gates remain unresolved and
  scientific completion is false
- Claim-bearing run authorization: not granted
- Planning decision: `NREV-DEC-0020` holds escalation and claims
- Required next review: protocol/configuration/data split, motion dependency,
  exact fixture generator, resource preflight, and scientific-audit capacity
