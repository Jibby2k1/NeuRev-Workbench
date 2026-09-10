# Unsupervised ICA evaluation program

This folder is the durable command/config entry point for
`docs/archive/plans/NEUREV_ICA_UNSUPERVISED_REPRESENTATION_EVAL_PLAN.md`.

- `stage_a_two_frame.json`: label-free canonical fit and operator identification.
- `stage_b_external_assay.json`: frozen-model external evaluation on the canonical
  adjudicated sparse-positive occurrences.
- `stage_c_signature_null_generalization.json`: ROI-held-out Raw signature test
  against same-ROI temporal shifts and matched spatial annuli.
- `stage_d_roi_minus_annulus_signature.json`: final amplitude-preserving local
  residual signature diagnostic and stop/advance gate.

Code lives in `neurobench/experiments/unsupervised_ica_eval/`. Versioned run
artifacts live under `Outputs/UnsupervisedICAEval/`; narrative conclusions live
under `docs/research/`. Stage B does not estimate precision because the labels
are sparse positives rather than exhaustive negatives.

Run the commands in `docs/workflows/unsupervised_two_frame_ica_eval.md`.

The bounded PC-MITL-ICA matrix-Rényi continuation is documented separately in
`docs/research/PC_MITL_ICA_PHASE1_PLAN.md`. It is an additive synthetic phase-1
evaluation and does not replace any configuration above.
Its frozen paired E01/E02 continuation is `pc_mitl_e01_e02.yaml`; results and
decision boundaries are recorded in
`docs/research/PC_MITL_ICA_E01_E02_RESULTS.md`.
The locked follow-up is `pc_mitl_specialized_confirmation.yaml`; its failed
advance gate and stop decision are documented in
`docs/research/PC_MITL_ICA_SPECIALIZED_CONFIRMATION_RESULTS.md`.
