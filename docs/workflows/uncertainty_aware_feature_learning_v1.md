# Uncertainty-aware feature learning v1

Status: frozen for a bounded, non-claim-bearing engineering screen.

Registry experiment: `NREV-EXP-0021`

## Question

Can a tiny nonlinear model combine the already-developed activity features more
usefully than the frozen CFAR and linear/expert combinations when every model
is evaluated on candidate locations it did not train on?

This is a candidate-reranking experiment. It does not replace the frozen
proposal generator, create verified negatives, estimate full-field precision,
or establish that a candidate is a neuron. A positive result would support only
the narrower statement that nonlinear feature interactions improve
positive-versus-unlabeled ranking within this recording under the frozen
spatial holdout. The CFAR comparator is the frozen `cfar_score` reranked over
the same broad candidate union; it is not the end-to-end CFAR proposal and
threshold pipeline, and `cfar_score` is also one of the network inputs.

## Why this is not a repeat of Innovation Ranker v5

Innovation Ranker v5 already compared nested linear and residual-MLP rankers
using the older 79-label sheet and quiet candidates as hard negatives. Its MLP
did not improve the fixed-budget recovery count over the linear ranker. This
experiment changes the question and the safeguards:

- canonical-v7 confirmed occurrences are the positive authority;
- unmatched event candidates remain unknown rather than becoming negatives;
- quiet candidates are source-off null controls only;
- later single-reviewer candidates are a locked conflict/audit panel;
- repeated identities and nearby spatial regions cannot cross outer folds;
- the nonlinear-vs-linear contrast uses the same compact features and fixed
  training contract;
- model outputs are ranking scores, never biological probabilities.

The feature and proposal program was developed on this same recording using an
older 79-label sheet, and the four event maps are pooled over those frozen,
label-derived burst intervals. The present labels do not build the broad
candidate census, choose the five outer folds, or fit preprocessing outside the
training fold, but the evaluation remains a model-only spatial holdout
conditional on oracle temporal windows and a historically label-informed
feature program. It is not unbiased detector generalization.

## Frozen candidate census

The candidate universe is reconstructed with the exact Innovation Ranker v5
map and proposal implementation. The old label sheet fixes only the already
registered event intervals; it supplies no candidate identity to this census:

- 22 proposal sources;
- at most 100 proposals per source;
- 6 px within-source NMS;
- 3 px cross-source deduplication;
- 34 frozen quiet-normalized feature measurements per candidate;
- event counts of 544, 530, 476, and 449 for bursts 1--4 (1,999 total);
- quiet counts of 683, 748, 759, and 748 (2,938 total).

The census is generated before either current label source is joined. Its
source inputs, configuration, implementation files, row counts, feature order,
and output tables are hash checked. Coordinates, proposal order, source count,
detector outcome, reviewer text, and labels are prohibited model inputs.
The frozen input descriptor includes the raw source video, historical event-
window sheet, feature manifest/config/state, every feature array consumed by
the map generator, the structure array, and all three later label/review
authorities. The implementation contract also pins the upstream map,
proposal, label-loading, and sparse-detection modules. All of these authority
bytes are rehashed before final publication of the run.

## Label and exclusion contract

Canonical-v7 confirmed occurrences are greedily matched one-to-one to event
candidates within the same burst and a 6 px radius. A confirmed occurrence
without a match is recorded as an unrecovered proposal-stage miss and is never
silently removed from the denominator.

The later 18-site review was selected from an older v1-label taxonomy. It is not
independent of canonical v7: most reviewed coordinates overlap canonical-v7
confirmed sites. All candidate rows within 6 px of a reviewed site are reserved
from fitting and primary evaluation. Any canonical anchor in that reserve is
also removed from the primary learning cohort but remains visible in the audit
table. The final report must show the exact overlap and conflict counts.

For the remaining event census:

- one matched candidate per confirmed canonical occurrence is positive (`P`);
- candidates within 12 px of any canonical confirmed, inclusive-only, or
  reserved-review coordinate are excluded from the unlabeled pool;
- all other event candidates are unlabeled (`U`), not negative;
- the artifact-or-noise review is a case study, not a negative class;
- quiet candidates are `null_control`, never `U` or negative.

No probable, uncertain, artifact, or quiet row is silently converted to a
verified biological negative.

The base unlabeled spatial units use fixed 12 px by 12 px coordinate cells.
Leakage components merge those cells only as required to keep repeated
canonical identities whole. They are not radius-connected components, so a
chain of nearby candidates cannot percolate into an unbounded group.

## Model-facing features

The primary compact feature set is fixed before scoring:

1. `carrier_signed`
2. `local_psd_signal`
3. `asymmetric_state`
4. `spatial_coherence`
5. `cross_scale_rank`
6. `cross_scale_recall`
7. `cfar_score`
8. `cfar_background`
9. `cfar_noise`
10. `persistent_artifact_score`
11. `cut_center_sigma2p5`
12. `cut_ring_r4p5_t1p25`

The first seven are the frozen v5 `separation` family and define the equal-
weight expert-concatenation baseline. The remaining five give a four-unit
network limited access to background/noise/artifact and center-versus-ring
context. All upstream values retain their frozen quiet-only normalization.
Any additional imputation or scaling is fitted inside each training fold.

Missing values remain missing until fold-local median imputation; an explicit
missingness indicator is added only when a training fold actually contains
missing values. Scaling uses the training-fold median and MAD with an IQR
fallback. Full-census z scores, coordinates, identities, ranks, labels, review
fields, and post-label selection fields are forbidden.

## Outer generalization split

The primary validation comprises five nominal x-coordinate blocks defined from
unreserved positive anchors. Each indivisible positive leakage group is
represented by the median x coordinate of its unreserved positive anchors;
positive groups are sorted and divided to balance positive occurrences while
preserving every canonical identity. Boundaries are midpoints between adjacent
positive-group representatives. An unlabeled-only leakage group is represented
by the median x coordinate of all its eligible rows and assigned to the
corresponding block. The resulting boundaries are frozen before any feature or
model score is inspected.

Whole-group closure may carry individual candidate rows across a nominal
boundary. Such extensions are retained and reported, never split or silently
reassigned. Candidate-row x extents may therefore overlap between adjacent
folds even though group representatives are strictly ordered. For each held
fold, a whole candidate leakage group is purged from training when any member
lies within an inclusive 12 px Euclidean radius of any held test candidate.
The run must verify that no retained train/test candidate pair violates this
geometric guard.

For each outer fold:

- all occurrences of a canonical identity are assigned to one fold;
- the test fold is one contiguous representative-x block plus any explicitly
  reported whole-group closure extensions;
- training rows within 12 px of either held-block boundary are purged;
- reserved-review neighborhoods are absent from all primary folds;
- preprocessing and model fitting use training rows only;
- every eligible event candidate receives exactly one out-of-fold score.

Fold construction may use label state only to balance the number of positive
identity groups. It may not use feature values, model scores, morphology,
review labels, or downstream performance. A secondary leave-one-burst-out
table may be reported, but it cannot replace the spatial primary analysis.

## Fixed methods

All methods receive the same eligible candidate rows.

1. `carrier_signed`: frozen scalar baseline.
2. `cfar_score`: frozen CFAR scalar baseline.
3. `expert_separation_equal_weight`: fold-local robust scaling followed by an
   equal-weight mean over the seven separation features.
4. `positive_reference`: distance to the fold-local positive reference.
5. `linear_spu`: fixed-L2 class-balanced logistic ranking of P versus U.
6. `elastic_linear_spu`: the same fixed linear surrogate with a 50% L1 mix.
7. `bagged_pu_linear`: fixed-count unlabeled bags with a logistic base learner.
8. `tiny_mlp_spu`: 12 inputs, four `tanh` hidden units, one scalar output,
   class-balanced BCE, a deterministic full-batch Adam-style optimizer with
   explicit L2, and ten deterministic seeds. A two-way spatial split inside
   each outer training set selects only the epoch count; outer-test values are
   never used for stopping.

P-versus-U fitting is a ranking surrogate. The U rows are not declared
negative, the sigmoid is not calibrated neuron probability, and no threshold
is a biological decision boundary. Hyperparameters are fixed; no outer-test or
review-panel value may select architecture, regularization, epoch, seed, or
feature.

Before the canonical run, a convergence-only dry check found that the original
0.01 penalty, 1e-8 tolerance, and 5,000-iteration cap failed to converge in all
five folds. No failed final iterate was scored. The numerical contract was
therefore amended before execution by preserving the common 0.01
linear/elastic/PU penalty while changing only the solver tolerance to 1e-6 and
the iteration cap to 50,000. A convergence-only replay on the frozen folds
completed under that contract. No downstream score or rank selected the
amendment. The canonical run must still fail closed if any of its five linear,
five elastic, or fifty bagged-PU base fits does not converge, and it must
persist that completion count without inventing unavailable per-fit iteration
telemetry.

## Primary estimands

Separately fitted folds need not share an absolute score scale. Each method is
therefore evaluated within fold first. For the global fixed-budget sensitivity,
held-fold scores are converted without labels to mid-rank percentiles inside
that fold before concatenation. Raw-score global rankings are sensitivity only.

The primary estimands are computed from out-of-fold scores:

- macro-fold SPU-AUC: mean, across held spatial blocks, of the probability that
  a held known-positive anchor outranks a held unlabeled event candidate;
- known-positive proposal recall at global budgets 20, 58, and 100;
- per-fold known-positive rank distribution and literal recall at
  `K=min(58, held-fold candidate count)`;
- tiny-MLP minus linear-SPU SPU-AUC and recall-at-58 differences;
- tiny-MLP minus strongest frozen baseline differences;
- ten-seed pairwise rank correlation and recall range.

`SPU-AUC` is deliberately named rather than reported as ordinary ROC AUC. The
recall denominator includes unreserved canonical-v7 confirmed occurrences that
were not present in the frozen proposal census; occurrences inside the locked
review reserve are reported separately and cannot re-enter the primary
denominator. Neither metric is precision, specificity, false-positive rate, or
independent-recording generalization.

The primary grouped bootstrap resamples whole leakage components within each
held fold, stratified into positive-containing and U-only components. When a
positive-containing component is sampled, all of its positive and unlabeled
member rows are included jointly. The bootstrap recomputes the candidate-level
positive-versus-unlabeled AUC in each fold and then averages the five fold
values. It therefore respects the merged identity/spatial dependence graph and
targets the macro-fold SPU-AUC contrast rather than a global concatenated-score
sensitivity.

A separate post-hoc association null selects exactly one pre-score
representative per whole leakage component: the smallest stable candidate ID
among positive anchors in each positive-containing component and among
eligible U rows in each U-only component. Unlabeled rows embedded in positive-
containing components do not become separate representatives. It permutes the
fixed representative scores among fixed roles only within held folds,
preserving each fold's component counts. This avoids both splitting dependent
components and the group-maximum order-statistic bias caused by unequal group
sizes. It is explicitly not a refitted label-permutation experiment.

## Locked audit panels

After primary outputs are frozen, one final ensemble trained only on eligible
P/U rows may score:

- the 18 reviewed sites, with canonical-v7 overlap and disagreements visible;
- the 2,938 quiet candidates as source-off null controls;
- prespecified hard cases and proposal-stage misses.

These panels are descriptive. They cannot tune the model, repair a failed
primary comparison, estimate precision, or change the candidate budget.
The review output includes both candidate-grain scores and an 18-site table
with the frozen review label, canonical-v7 overlap, and cross-source conflict
state.

## Screen advance signals

The bounded screen is considered promising only when all of the following are
observed; these are engineering advance signals, not experiment-level
scientific gates:

1. all census, label, identity, exclusion, and fold-integrity checks pass;
2. at least 75 unreserved canonical positive anchors remain, and every fold has
   at least ten positive anchors and ten U spatial groups;
3. the tiny-MLP ten-seed mean-score ensemble SPU-AUC exceeds linear SPU by at
   least 0.03 and the grouped 95% interval for the difference is above zero;
4. the same tiny-MLP ensemble SPU-AUC exceeds the frozen same-union CFAR-score
   rerank by at least 0.03 and the grouped 95% interval for the difference is
   above zero;
5. tiny-MLP recall at budget 58 exceeds the strongest frozen baseline by at
   least 0.03 without any fold losing more than 0.10;
6. the fold-stratified representative-unit permutation p value is at most
   0.05;
7. median pairwise rank correlation across MLP seeds is at least 0.90 and the
   recall-at-58 seed range is at most 0.05;
8. the result is not explained solely by a single burst, spatial block, or
   review-label policy.

Failure of the nonlinear-vs-linear contrast means the small network should not
replace the interpretable fusion layer. A gain over CFAR but not over linear
SPU would support the features or training objective, not nonlinear structure.

## Output and audit contract

The run writes to a new atomic root and preserves:

- exact event and quiet census tables;
- feature dictionary and frozen input/code hashes;
- harmonized candidate/label/exclusion table;
- fold boundaries, group membership, and guard-band exclusions;
- out-of-fold scores for every eligible candidate and method;
- per-fold, aggregate, seed-stability, bootstrap, and permutation tables;
- locked review/conflict and quiet-null tables;
- resolved config, execution provenance, validation, status, artifact index,
  summary, `llm_context.json`, and a concise report.

A 30-minute POSIX process alarm bounds work from public-run entry through all
pre-promotion validation. The runner restores thread limits, freezes and
verifies the complete artifact index, and precomputes its return payload while
the tree is still recoverably partial. It then disarms the alarm immediately
before the single same-filesystem atomic rename; that rename is the commit
point, and no fallible cleanup or validation follows it.

The scientific-audit standard remains enabled. This bounded automated screen
does not create the full three-section media package and must therefore retain
`scientific_audit_complete=false`, `scientific_completion=false`, and
`claim_promotion_allowed=false`. A model may be promoted to a human-review
candidate only through a subsequent run that freezes its scores and renders
the required expert, model, and comparison media.

## Claim boundary

Even a passing engineering screen cannot establish neuron identity, full-field
precision, calibrated probability, causal feature importance, independent-
animal generalization, or replacement of the current scientific detector. It
can justify only a subsequent frozen audit of a small feature-fusion model.
