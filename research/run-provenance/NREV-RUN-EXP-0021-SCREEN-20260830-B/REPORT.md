# NREV-EXP-0021 bounded feature-learning screen

Run `NREV-RUN-EXP-0021-SCREEN-20260830-B` completed the engineering screen. The descriptive advance verdict is **one or more descriptive engineering signals did not pass**; the scientific audit is still incomplete and claim promotion remains false.

## Answer-first results

The primary metric is the macro mean of within-held-fold positive-versus-unlabeled rank AUC. Recall is reported at the global top-58 operating point after label-free within-fold percentile normalization.

| Method | Primary macro-fold SPU-AUC | Known-positive proposal recall @58 |
|---|---:|---:|
| `carrier_signed` | 0.933518 | 0.371795 |
| `cfar_score` | 0.764660 | 0.179487 |
| `expert_separation_equal_weight` | 0.950144 | 0.384615 |
| `positive_reference` | 0.894641 | 0.307692 |
| `linear_spu` | 0.974096 | 0.423077 |
| `elastic_linear_spu` | 0.975601 | 0.423077 |
| `bagged_pu_linear` | 0.975496 | 0.435897 |
| `tiny_mlp_spu` | 0.974022 | 0.435897 |

Whole-leakage-component paired bootstrap contrasts:

| Prespecified contrast | AUC delta | 95% cluster CI |
|---|---:|---:|
| tiny MLP - `linear_spu` | -0.000074 | [-0.004607, 0.005912] |
| tiny MLP - `carrier_signed` | 0.040504 | [0.006469, 0.071386] |
| tiny MLP - `cfar_score` | 0.209362 | [0.120730, 0.303681] |
| tiny MLP - `expert_separation_equal_weight` | 0.023877 | [-0.005543, 0.044595] |

Positive-containing leakage-component support by fold was 9/3/3/2/4. Because some folds contain only two or three positive components, these 95% cluster-bootstrap intervals are descriptive and non-claim-bearing.

The observed strongest frozen baseline at budget 58 was `expert_separation_equal_weight`; the MLP recall delta was 0.051282. That observed-baseline selection was not repeated inside bootstrap draws, so its corresponding prespecified baseline CI is descriptive. The MLP representative-component null p-value was 0.00049975; median seed rank correlation was 0.994294, and the seed recall@58 range was 0.000000.

## Cohort and denominator

Primary cohort: 78 matched unreserved positive anchors and 1541 unknown event candidates. Unreserved proposal-stage misses: 0; matched review-reserved anchors: 27; review-reserved unmatched confirmed occurrences: 1.

## Applicability and conventions

Stage: `bounded_model_only_spatial_positive_unlabeled_feature_fusion_screen`. Applicability: `within_this_recording_and_frozen_broad_candidate_union_only`. This is model-only spatial holdout within one recording, conditional on legacy label-derived event windows and a historically label-informed feature program. U means unknown, not negative. N/A is used when a denominator or audit is outside this bounded screen. The CFAR arm reranks the same broad union and `cfar_score` is also a fusion input; this is not end-to-end detector replacement or independent-recording generalization.

## Inspect next

Start with `summary.json`, `tables/aggregate_metrics.tsv`, `tables/per_fold_metrics.tsv`, `bootstrap.json`, `advance_signals.json`, and `validation.json`. Reproducibility authority is in `execution_provenance.json`, `resolved_config.json`, and the complete-tree `artifact_index.json`. Locked review and source-off panels are descriptive only.
