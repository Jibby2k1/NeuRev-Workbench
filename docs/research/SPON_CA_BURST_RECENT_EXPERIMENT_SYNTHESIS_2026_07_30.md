# Spon Ca Burst recent experiment synthesis

## Scope

This report steps back across the Spon Ca Burst activation-detection work from
the recent Raw Direct, CFAR, source-separation, denoising, feature-utility, and
nested-ranking studies. It is an architectural synthesis, not a leaderboard:
several studies use different carriers or operating points, and the sparse
labels do not identify ordinary precision.

The common real-data contract is 79 burst-specific positive observations at 27
ROI identities over four bursts. Unless explicitly stated, unmatched event
candidates are unknown. Fixed-budget recall at 58 candidates per burst is the
most comparable ranking endpoint; quiet-threshold recall must always be read
with candidate burden.

## What was actually tested

The recent program is already broad. The major completed screens include:

| Study | Search size | Primary result |
| --- | ---: | --- |
| Learnable guarded contrast v1 | nested CUDA study | 0.133 recall versus Raw Direct 0.606; stopped |
| Spatiotemporal factorial v2 | 64 learned fits | best 0.205; stabilization helped, initialization did not |
| Learnable Raw Direct v3 | 36 fits | all nine variants tied Raw Direct 0.606 |
| Multi-hypothesis CFAR v4–v6 | 24 experts plus nested selective gates | best learned gate 0.329; C3 stopped |
| Pairwise derivative/ICA fusion | 22 fixed comparisons plus 12 fits | no gain over Raw Direct |
| PCA/ICA/autoencoder benchmark | 36 fits, 51 evaluated lanes | amplitude PCA rank 8 fixed recall 0.687 |
| Stochastic architecture grid | 193 rows, 185 effective operators | cross-fitted 0.365; visual innovation lane fixed recall 0.641 |
| Noisy-Parzen split | two 16-cell grids | useful visual split; not a qualified detector |
| Sequential denoise audit | 11 methods | component Parzen/ICA fixed recall 0.721; no method passed all gates |
| Spatial ICA screen | 3 lanes | dense FastICA+Wiener fixed recall 0.671 |
| Advanced denoising v2 | 69 Stage-A and 20 Stage-B combinations | no finalist passed preservation gate |
| Innovation denoising v3 | 96 Stage-A, 16 Stage-B, 8 mixtures | cross-scale variant fixed recall 0.711; no joint-gate pass |
| Feature utility v1 | 176 fixed lanes, 100 scalar fits, 4 multifeature fits | spatial/separation features dominate |
| Innovation ranker v5 | 2,250 inner fits plus 16 outer refits | linear 0.725; nested single feature 0.734 |
| Patch information v1 | 27 ITL features, 216 fixed lanes, 880 nested fits | cross-fitted Cauchy--Schwarz 0.572/0.696 at budgets 20/40 |
| Multiscale patch information v1 | 42 maps, 336 native/same-proposal lanes | native 0.583/0.677 and same-proposal 0.471/0.639/0.759 at budgets 20/40/58 |
| Scientific feature audit v1 | 16 maps, 192 native/right/same-proposal lanes | causal coherence 0.605/0.676/0.722 at budgets 20/40/58 |

These counts are not additive statistical trials: some are model fits, some are
detector lanes, and several share the same four bursts. They demonstrate search
breadth, not independent evidence.

## Comparable performance landmarks

| Method | Quiet-threshold recall | Fixed-budget recall | Candidates | Interpretation |
| --- | ---: | ---: | ---: | --- |
| Raw Direct | 0.606 | 0.657 | 232 | strongest established threshold-recall anchor |
| Archived centered Parzen residual | 0.330 | 0.641 | 70 | selective carrier |
| Amplitude PCA rank 8 | 0.710 | 0.687 | 427 | two-match fixed-budget gain; high burden |
| Dense spatial FastICA + Wiener | 0.484 | 0.671 | 91 | safe spatial-source feature |
| Component Parzen/ICA audit lane | 0.479 | 0.721 | 92 | strong fixed ranking; attenuates amplitude |
| Cross-scale consensus v3 variant 06 | 0.342 | 0.711 | 65 | strong selective ranking/filtering feature |
| Quiet-standardized Parzen carrier | — | 0.703 | — | four-match gain from standardization alone |
| Nested linear innovation ranker | 0.823 | 0.725 | 585 | one same-union match from learning |
| Nested single-feature selection | 0.808 | **0.734** | 524 | strongest nested fixed-budget result |

The last two threshold operating points are not precision improvements: their
candidate burdens are large, and at budgets 20 and 40 they lose to the native
standardized carrier.

The later patch-information study changes the tight-budget result without
changing the budget-58 authority. Cross-fitted 7-pixel Cauchy--Schwarz
divergence from the same-location quiet density reached 0.572 and 0.696 at
budgets 20 and 40, versus 0.541 and 0.657 for the native carrier. It did not
beat the v5 nested single-feature ranker at budget 58. Its full 49-source union
also diluted learned top-k ranking, so only the compact Cauchy--Schwarz expert
is retained for confirmation.

The multiscale follow-up tested 5--15 pixel supports and five fusion concepts.
Standalone performance declined sharply beyond 9 pixels, but broad support was
useful as context: 5/7 agreement and 7-minus-15 contrast were the strongest
held-out fusion families. Native cross-fitted recall reached 0.583/0.677 at
budgets 20/40; on the identical v5 proposal union it reached
0.471/0.639/0.759 at 20/40/58. The gains were not consistent in every burst,
so these formulas require compact confirmation rather than another wide grid.

The scientific feature audit then separated acquisition noise, generative
z-cut morphology, radial information, and causal neighborhood recurrence. A
15-frame local-coherence map was selected in all four family-specific outer
folds and improved budget-20 recall in every burst, reaching
0.605/0.676/0.722 at budgets 20/40/58. Lagged recurrence was non-worse in every
burst at budget 20 and reached 0.589/0.681/0.716. Cross-family selection was
less stable, confirming that a compact frozen recurrence expert is more
justified than a wider learned mixture. The audit also found shot-like
intensity-dependent pair-difference variance and tentative enrichment of
ring-like z-cut fits among missed labels. These results motivate compact
confirmation and phenotype annotation, not a precision claim.

## Development patterns

### 1. Initialization is not the central problem

Initialization jitter was secondary in the spatiotemporal factorial. All nine
low-learning-rate Raw Direct tuning variants tied the frozen detector, and the
residual MLP now ties the bounded linear model. The repeated pattern is not
failure to escape a bad initialization; it is limited supervision and an
objective that does not reliably identify which candidate should occupy the
very top of the list.

### 2. Calibration and normalization have larger effects than model depth

The ranker audit separated a +0.062 fixed-recall gain from quiet per-pixel
standardization, +0.006 from proposal diversity, and +0.017 from linear
fine-tuning. Earlier apparent wins also often changed after evaluator or
synthetic-metric corrections. Future reports must freeze carrier semantics,
NMS, normalization, matching radius, and budget before attributing a gain to an
architecture.

### 3. Spatial context is the most consistent new information

Across PCA, local PSD-Wiener, dense spatial ICA, cross-scale consensus, the
feature bank, and v5 cut-morphology responses, spatial or local-subspace
features repeatedly improve fixed-budget ordering. Pure temporal derivative
features are visually informative but generally weak or timing-distorting as
standalone activation detectors.

The current evidence supports the hypothesis that z-plane cuts, membranes,
crowding, and local background structure create multiple spatial observation
types. It does not yet prove which labeled ROI belongs to which type because
those attributes are not annotated.

The patch-information result further localizes the useful spatial statistic:
same-location quiet-relative density divergence at a 7-pixel support was
stable across all four outer folds. Generic local low entropy and positive
center-to-patch correntropy were not sufficient; quiet-relative distributional
change was the useful criterion.

### 4. Denoising should remain an auxiliary lane

Local PCA, component Parzen/ICA, PSD-Wiener, nonlocal filtering, and wavelets
can reduce quiet energy or improve ranking, but the stronger filters often
attenuate amplitude, area, or timing. The recurring safe architecture is:

```text
immutable activity carrier
  + bounded auxiliary denoising/separation features
  + separate ranking score
```

It is not a single denoised movie used as both scientific reconstruction and
detector score.

### 5. Constant global fusion is saturated

Additive derivative fusion did not beat Raw Direct. Constant Pareto mixtures
in innovation denoising were weaker than their best individual source. Broad
CFAR fusion diluted the strongest center expert. The v5 MLP did not improve on
the linear ranker. More global weights or a wider blind hyperparameter grid are
unlikely to be high-value.

Conditional selection remains plausible, but only with the right supervision:
the system must know when a candidate looks like a center, membrane, crowded
cell, motion edge, persistent artifact, or quiet fluctuation.

### 6. The failures divide into proposal and ranking problems

The v5 per-neuron audit makes this concrete. `roi_007` and `roi_023` are mostly
absent from the entire proposal union. `roi_014` and `roi_019` are present in
the union but repeatedly ranked below budget. The first group needs a new
representation or morphology expert; the second needs better ranking or hard
negative discrimination. A single end-to-end recall number hides this
distinction.

### 7. Precision is now the main measurement bottleneck

Sparse positive labels cannot distinguish a false positive from a real but
unannotated neuron. Candidate count and quiet-field peaks measure selectivity
pressure, not biological precision. Without a bounded exhaustively annotated
field, learned ranking can easily reward real unlabelled activity, artifacts,
or both without us knowing which occurred.

### 8. Visual quality and detector quality are related but not interchangeable

Parzen Innovation and fixed-point backgrounds can look excellent while scoring
below Raw Direct at a quiet threshold. Conversely, some high-recall lanes emit
hundreds of candidates or distort signal timing. Every future promotion needs
both visual remainder review and a preregistered detection operating point.

## The real next experiment

The highest-impact next step is a bounded, exhaustively annotated calibration
field followed by a proposal-versus-ranking ablation. It should proceed in
stages.

### C0 — annotation and phenotype audit

Choose one representative spatial field and annotate every visible candidate
as foreground neuron, background, artifact, or unresolved. For neurons, add:

- center versus membrane appearance;
- isolated versus crowded context;
- approximate footprint or free-form ROI;
- burst activity and onset confidence;
- persistent-bright structure versus nuisance artifact.

Re-audit `roi_007`, `roi_014`, `roi_019`, `roi_023`, and `roi_015` first. This
single annotation effort unlocks real precision-recall, hard-negative mining,
and morphology-conditional evaluation.

### C1 — frozen proposal audit

Run only frozen, label-independent proposal sources:

- quiet-standardized carrier;
- cross-scale rank/recall;
- local PSD signal;
- asymmetric state;
- center and annular cut experts;
- a conservative artifact/motion expert.

Report source-wise and union recall at budgets 10, 20, 40, 58, 80, and 100,
proposal-source overlap, NMS suppression causes, and exact per-neuron recovery.
No learned score should enter this stage.

### C2 — precision-aware ranking

Train a small bounded linear or monotone additive ranker on known neurons and
manually reviewed hard negatives. Compare:

1. standardized carrier on native peaks;
2. standardized carrier on the identical proposal union;
3. one nested selected feature;
4. bounded linear ranking;
5. a compact morphology-conditional mixture of experts.

Use nested burst splits and ROI-identity grouping. Optimize a listwise or
top-k objective at budgets 20 and 40, with budget 58 secondary. Reject a model
that improves 58 while degrading both tighter budgets.

### C3 — timing and scientific-signal guard

For candidates retained by C2, evaluate the ranking score separately from the
scientific trace. Require the carrier trace to preserve peak time, area, and
shape. Denoising and source-separation channels may change ranking but must not
silently replace the trace used for downstream intent or control.

### C4 — causal implementation checkpoint

Profile the promoted frozen feature set and ranker in bounded streaming chunks.
Target the 20 ms frame period with p50/p95/p99 latency, fixed memory, and
fallback behavior. Local PSD and batch ICA require causal approximations before
deployment; cross-scale, asymmetric dynamics, and small convolutions are more
directly streamable.

### Advancement rule

Advance only if the method:

- improves exhaustive-region precision-recall or precision at fixed recall;
- improves known-positive recall at budget 20 or 40 on at least three of four
  held-out bursts;
- does not degrade carrier timing/amplitude beyond the frozen tolerance;
- demonstrates stable behavior across ROI identities and nearby NMS settings;
- remains computationally compatible with a causal path.

## Innovation directions after this checkpoint

If the annotated checkpoint passes, the most credible innovations are:

1. a morphology-conditional matched-filter bank with a bounded gating simplex;
2. listwise ranking that learns which recoverable proposals deserve scarce
   budget, rather than pointwise positive-versus-quiet separation;
3. self-supervised spatial pretraining followed by very small supervised
   calibration, with the carrier retained as a skip;
4. uncertainty-aware abstention for unresolved morphology and motion artifacts;
5. multi-frame causal features that preserve absolute fluorescence while using
   onset/decay evidence as context;
6. hierarchical evaluation that explicitly labels proposal miss, below-budget
   rank, NMS suppression, localization error, and temporal miss.

Only after activation detection has a stable, precision-audited output should
the project treat left/right intent as the next scientific checkpoint. Intent
will need spatial identity and pre-movement timing, not just a better burst
detector. The inverse-control stage then additionally requires action-conditioned
data and cannot be inferred from these passive activation experiments.

## Authoritative artifacts

- Recent synthesis source data:
  `Outputs/HierarchicalParzenICA/spon_ca_burst_innovation_ranker_v5`.
- Exact v5 interpretation:
  `docs/research/SPON_CA_BURST_INNOVATION_RANKER_V5_RESULTS.md`.
- Principe-aligned patch-information evidence:
  `docs/research/SPON_CA_BURST_PATCH_INFORMATION_V1_RESULTS.md`.
- Multiscale patch-information evidence:
  `docs/research/SPON_CA_BURST_MULTISCALE_INFORMATION_V1_RESULTS.md`.
- Feature-utility evidence:
  `docs/research/SPON_CA_BURST_FEATURE_UTILITY_V1_RESULTS.md`.
- Denoising sequence:
  `docs/research/SPON_CA_BURST_DENOISING_AUDIT_RESULTS.md`,
  `docs/research/SPON_CA_BURST_ADVANCED_DENOISING_RESULTS.md`, and
  `docs/research/SPON_CA_BURST_INNOVATION_DENOISING_V3_RESULTS.md`.
- Spatial ICA evidence:
  `docs/research/SPON_CA_BURST_SPATIAL_ICA_SCREEN_RESULTS.md`.
- Earlier Raw Direct, CFAR, and representation contracts:
  `docs/workflows/spon_ca_burst_learnable_contrast.md`,
  `docs/workflows/spon_ca_burst_multihypothesis_cfar.md`, and
  `docs/workflows/spon_ca_burst_representation_benchmark.md`.
