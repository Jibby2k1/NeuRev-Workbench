# Uncertainty-aware feature learning v1.1 results

## Outcome first

The bounded feature-fusion screen completed as an engineering execution, but
the tiny nonlinear model did **not** improve the primary held-fold ranking
estimand over the matched linear positive-versus-unlabeled model. Tiny-MLP
macro-fold SPU-AUC was `0.9740218518`, versus `0.9740959140` for linear SPU; the
paired whole-leakage-component difference was `-0.0000740622` with descriptive
95% interval `[-0.0046074197, 0.0059124340]`.

The small network therefore should not replace the simpler linear fusion layer
on this evidence. The result is still useful: both learned fusion arms ranked
the frozen same-union candidates much better than the scalar `cfar_score` arm,
and the MLP recovered 34 of 78 known positives at budget 58 versus 14 of 78 for
CFAR reranking. That contrast evaluates feature fusion inside one frozen
candidate union; it is not an end-to-end CFAR detector replacement test, and
`cfar_score` is itself one of the learned models' inputs.

`NREV-EXP-0021` remains `draft`, `not_evaluated`, and evidence tier `none`.
Scientific audit completion and claim promotion are false. No claim or evidence
capsule is created.

## Run sequence

- Failed immutable v1 run:
  `NREV-RUN-EXP-0021-SCREEN-20260830-A`
- Successful v1.1 engineering run:
  `NREV-RUN-EXP-0021-SCREEN-20260830-B`
- Run-B source configuration SHA-256:
  `489e5f448d72fb5325f0531558d5a531eaa31ca8e7ba70362b26e9070a410063`
- Run-B protocol SHA-256:
  `99161431d3ba27192c3ffff85b926fa2a10871e869bd3ad3edb82fb50a4379b7`
- Run-B runner SHA-256:
  `9ee07cfe0549dc357f5adcdb70415983d955bf04bd552ece4e5678cb6632cf25`
- Run-B input-descriptor SHA-256:
  `658b8d0c77ff156f4abb85763f1616eec4e729110ca4d5092ab635f03369e01d`

Run B used CPU execution with one requested numeric thread. It started at
`2026-08-30T17:40:27.018054Z`, ended at
`2026-08-30T17:42:46.733445Z`, and completed in `139.715391` seconds.

## Why Run A remains failed

Run A completed the frozen candidate census, harmonization, and primary model
evaluation, then stopped during paired-bootstrap output assembly. The evaluator
uses `oof_scores` for learned-model aliases and `scores` for frozen scalar
baselines; the v1 assembler incorrectly required `oof_scores` from every
comparator and raised `KeyError` on `carrier_signed`.

No model metric table or final run summary was serialized. Run A is not
resumable and was not repaired in place. Its `.partial` tree remains immutable.
Version 1.1 introduced one validated score accessor, a production-shaped mixed-
schema regression, a new configuration and protocol, and a non-colliding Run-B
root. Candidate, label, feature, fold, solver, seed, bootstrap, permutation,
resource, audit, and interpretation semantics were unchanged.

## Primary ranking and budget results

The primary estimand is the macro mean of within-held-spatial-fold
positive-versus-unlabeled rank AUC. Recall uses the global top-58 operating
point after label-free within-test-fold midrank-percentile normalization.

| Method | Macro-fold SPU-AUC | Known-positive recall @58 |
| --- | ---: | ---: |
| `tiny_mlp_spu` | `0.9740218518` | `34/78 = 0.435897` |
| `linear_spu` | `0.9740959140` | `33/78 = 0.423077` |
| `elastic_linear_spu` | `0.9756013670` | `33/78 = 0.423077` |
| `bagged_pu_linear` | `0.9754961258` | `34/78 = 0.435897` |
| `expert_separation_equal_weight` | `0.9501444137` | `30/78 = 0.384615` |
| `carrier_signed` | `0.9335176617` | `29/78 = 0.371795` |
| `cfar_score` | `0.7646602748` | `14/78 = 0.179487` |

The similar linear, elastic, bagged-PU, and MLP values point toward useful
feature/training-objective signal but no detected nonlinear advantage. The MLP
matches bagged PU at budget 58, while elastic and bagged PU have slightly higher
primary point estimates than the MLP.

## Prespecified paired contrasts

| Tiny MLP minus comparator | Macro SPU-AUC delta | Descriptive whole-component 95% interval |
| --- | ---: | ---: |
| `linear_spu` | `-0.0000740622` | `[-0.0046074197, 0.0059124340]` |
| `expert_separation_equal_weight` | `+0.0238774381` | `[-0.0055430249, 0.0445945088]` |
| `carrier_signed` | `+0.0405041901` | `[0.0064689710, 0.0713861293]` |
| `cfar_score` | `+0.2093615770` | `[0.1207303370, 0.3036814877]` |

The MLP-minus-linear point estimate missed the predeclared `+0.03` descriptive
advance threshold, and its interval lower bound was not above zero. The MLP-
minus-expert interval also crossed zero. The favorable carrier and CFAR
contrasts do not isolate nonlinearity: both the linear model and the network use
the richer feature panel.

The representative whole-component association null returned
`p = 1/2001 = 0.0004997501`. That supports within-screen association between
the fixed MLP score and the P/U roles; it does not show that the MLP is better
than linear, turn U into verified negatives, or establish neuron probability.

## Cohort, dependence, and stability

The primary out-of-fold table contains 1,619 event candidates: 78 canonical-v7
positive anchors and 1,541 unlabeled candidates. U means unknown, not negative.
There were no unreserved proposal-stage misses. The denominator audit also
retains 27 matched anchors and one unmatched confirmed occurrence inside the
locked review reserve, reconciling all 106 confirmed occurrences without
returning reserved labels to model fitting.

Positive-containing leakage-component support by fold was `9/3/3/2/4`. Folds
with only two or three positive-containing components make the cluster
intervals descriptive rather than claim-bearing. The ten MLP seeds had median
pairwise Spearman correlation `0.9942944810`, and their recall-at-58 range was
zero. Stable seeds do not rescue the missing nonlinear contrast.

The 18-site review panel remained locked and descriptive: 10 sites overlap a
confirmed canonical-v7 site and two have cross-source disagreement. The 2,938
quiet candidates are source-off null controls, not negatives.

## Engineering integrity and unresolved signal

Run B passed its frozen input and implementation hashes, exact 24-input and
12-implementation cardinalities, census and harmonizer checks, five-fold
assignment and 12-pixel train/test guard, complete 1,619-row OOF coverage, CPU
thread cap, 30-minute deadline, and atomic publication. All 60 registered
reference solver fits completed: five L2, five elastic, and 50 bagged-PU base
fits at penalty `0.01`, tolerance `1e-6`, and maximum 50,000 iterations.

The complete output index covers 42 artifacts totaling 11,586,631 bytes; every
indexed artifact hash passed independent verification. The engineering
execution passed, but the descriptive advance panel did not. In addition to the
failed nonlinear-versus-linear signals, Run B did not produce the prespecified
leave-one-burst-out and alternate review-policy sensitivities needed to exclude
a single burst, block, or review policy as the explanation.

## Interpretation boundary

This was model-only spatial holdout within one recording. It is conditioned on
legacy label-derived event windows and a feature program historically developed
with labels from the same recording. It does not establish:

- verified biological negatives, precision, specificity, or false-positive rate;
- calibrated neuron probabilities or a biological decision threshold;
- causal feature importance or nonlinear biological structure;
- end-to-end CFAR detector replacement;
- independent-recording or independent-animal generalization; or
- scientific-audit completion or claim promotion.

## Decision and next automated tests

Hold the tiny MLP as a replacement for the interpretable fusion layer, and do
not spend the next cycle on same-recording MLP width, depth, seed, or optimizer
sweeps. Retain `linear_spu` as the simplest confirmation candidate, with
elastic and bagged-PU lanes as frozen sensitivity comparators.

The next useful automated package should keep the candidate census, labels,
feature set, grouping, and model parameters frozen while adding the missing
leave-one-burst-out and review-policy sensitivities, prespecified feature-family
ablations, and source-off/null stress checks. If the simpler fusion remains
stable, freeze it before applying it to a compatible independent recording and
before producing the complete expert/model/comparison media audit. These tests
can evaluate robustness; they still cannot manufacture verified negatives from
unreviewed candidates.

## Portable provenance

The full immutable outputs remain under:

```text
Outputs/NeuronIdentifiability/NREV-EXP-0021/runs/
  NREV-RUN-EXP-0021-SCREEN-20260830-A.partial
  NREV-RUN-EXP-0021-SCREEN-20260830-B
```

Small exact portable Run-A failure artifacts are retained under
`research/run-provenance/NREV-RUN-EXP-0021-SCREEN-20260830-A/`. Exact Run-B
summary, complete-tree index, resolved configuration, report, status,
validation, advance panel, bootstrap, solver completion, and LLM context are
retained under
`research/run-provenance/NREV-RUN-EXP-0021-SCREEN-20260830-B/`. The portable
Run-B provenance derivative omits host-specific library file paths and records
the SHA-256 of the exact full execution provenance that remains in the immutable
output tree.
