# Unsupervised two-frame ICA evaluation

This is the label-free Stage A implementation of
`NEUREV_ICA_UNSUPERVISED_REPRESENTATION_EVAL_PLAN.md`. It reuses the maintained
two-dimensional whitening implementation and deliberately stops before labels,
multi-time-step ICA, or biological interpretation.

```bash
.venv-neurobench/bin/python -m neurobench.experiments.unsupervised_ica_eval \
  --video-npy Outputs/GammaCFAR/spon_ca_burst_3_hindbrain_to_tail_488_20ms/spon_ca_burst_3_hindbrain_to_tail_488_20ms.npy \
  --start 0 --stop 560 --spatial-stride 4 --max-samples 250000 \
  --output-dir Outputs/UnsupervisedICAEval/two_frame_stage_a_v1
```

The output directory must not exist. The command writes `summary.json`,
`frozen_representation.json`, `llm_context.json`, `REPORT.md`, and
`operator_identification.png`. The representation hash binds the complete
component set and the label-free rule that identifies the component closest to
signed temporal difference in observation coordinates. External label evaluation must read that frozen
record and may not change it.

The module also exposes matched-support paired-trace metrics and equal-ROI
candidate-signature summaries for the later external assay. Unmatched candidates
remain unknown. A complete labeled continuation must implement and validate the
full scientific-audit artifact standard; Stage A is not audit-complete evidence.

## Frozen external assay

After copying the exact Stage A artifacts to the declared output root:

```bash
.venv-neurobench/bin/python -m neurobench.experiments.unsupervised_ica_eval.external_assay \
  --video-npy Outputs/GammaCFAR/spon_ca_burst_3_hindbrain_to_tail_488_20ms/spon_ca_burst_3_hindbrain_to_tail_488_20ms.npy \
  --labels Outputs/HardROIAdjudication/spon_ca_burst_hard_roi_adjudication_final_v1/adjudication_final.tsv \
  --frozen Outputs/UnsupervisedICAEval/two_frame_stage_a_v1/frozen_representation.json \
  --output-dir Outputs/UnsupervisedICAEval/two_frame_external_assay_v1
```

This uses labels only after verifying the frozen representation hash. It writes
occurrence-level matched-support metrics, circular-shift comparisons, a Raw
signature heatmap, and compact indexes. Because the labels are sparse positives,
precision and false-positive rate are explicitly not applicable. This bounded
assay is not a substitute for the complete scientific-audit media package.

## Raw signature null/generalization gate

```bash
.venv-neurobench/bin/python -m neurobench.experiments.unsupervised_ica_eval.signature_assay \
  --video-npy Outputs/GammaCFAR/spon_ca_burst_3_hindbrain_to_tail_488_20ms/spon_ca_burst_3_hindbrain_to_tail_488_20ms.npy \
  --labels Outputs/HardROIAdjudication/spon_ca_burst_hard_roi_adjudication_final_v1/adjudication_final.tsv \
  --output-dir Outputs/UnsupervisedICAEval/raw_signature_null_generalization_v1
```

This assay uses deterministic five-fold splits over canonical ROI identity. A
template from training ROIs is scored on held-out ROI templates and compared
with (1) matched windows shifted away from every annotated burst in the same ROI
and (2) the same labeled intervals in a spatial annulus. Uncertainty is
bootstrapped over ROIs. Traces are baseline-median centered and divided by the
pre-event robust scale; they are not peak-aligned or peak-normalized.

## Final ROI-minus-annulus diagnostic

```bash
.venv-neurobench/bin/python -m neurobench.experiments.unsupervised_ica_eval.residual_signature_assay \
  --video-npy Outputs/GammaCFAR/spon_ca_burst_3_hindbrain_to_tail_488_20ms/spon_ca_burst_3_hindbrain_to_tail_488_20ms.npy \
  --labels Outputs/HardROIAdjudication/spon_ca_burst_hard_roi_adjudication_final_v1/adjudication_final.tsv \
  --output-dir Outputs/UnsupervisedICAEval/roi_minus_annulus_signature_v1
```

This final analytic diagnostic subtracts the fixed 3–6 px annulus mean from the
2 px-radius ROI mean. It preserves residual intensity amplitudes and repeats the
held-out and temporal-null gates. Spatial specificity is tested against a fixed
annulus-minus-outer-ring (7–10 px) residual. Failure writes `stop_signature_branch`;
only an S3 result may authorize bounded multi-time-step temporal ICA.
