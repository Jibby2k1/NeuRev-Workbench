# Experiment History: Activation to Fish Control

Last reviewed: 2026-07-27.

## Bottom line

The repository has strong engineering evidence and promising passive
forecasting results, but it does not yet have the exhaustive activation truth,
causal pre-movement labels, or action-conditioned transitions needed for inverse
control.

Three recurring patterns explain most recent disappointments:

1. an optimization loss improved while the deployed threshold/NMS decision did
   not;
2. a global or sparse-positive metric was interpreted beyond its valid
   denominator;
3. passive prediction was treated as if it implied intent or controllability.

## Chronological ledger

### Pre-label CFAR exploration

- Completed 36 self-template cascade runs and two 12-run ROI-state sweeps with
  no recorded execution failures.
- Grayscale/projection configurations produced 59–96 candidates and
  2,814–4,529 events; Kalman-residual configurations produced 0–82 candidates
  and 0–4,640 events.
- This established executable pipelines and extreme threshold/preprocessing
  sensitivity. It did not establish precision, recall, or biological validity.

Evidence:

- `Outputs/GammaCFAR/spon_ca_burst_3_hindbrain_to_tail_488_20ms/gamma_cfar_cascade_grid_50hz_self_template_v1/sweep_summary.json`
- `Outputs/GammaCFAR/spon_ca_burst_3_hindbrain_to_tail_488_20ms/grayscale_projection_cfar_roi_state_v1/sweep_summary.json`
- `Outputs/GammaCFAR/spon_ca_burst_3_hindbrain_to_tail_488_20ms/kalman_residual_projection_cfar_roi_state_v1/sweep_summary.json`

### Grid32 passive dynamics, June 4–6

- Completed 1,212 of 1,212 declared experiments.
- 850 of 1,212 improved on persistence on held-out test videos.
- Best test improvement was `0.000229474681`, from an h50 temporal CNN.
- This established passive grid forecasting above persistence. It did not
  establish activation detection, intent, intervention response, or control.

Evidence:
`Outputs/GridModel/060126_crop512_grid32_v1/cropped32_large_sweep_v1/sweep_summary.tsv`.

### Partial high-resolution temporal-CNN run, June 7

- The manifest declares 720 experiments, but only 32 metric rows completed.
- Of those rows, 31 beat persistence; best test improvement was
  `0.000321632309`.
- This supports a promising h25 grid128 neighborhood. It must not be described
  as a completed 32-model design or a completed 720-model sweep.

Evidence:

- `Outputs/GridModel/060126_crop512_highres_temporalcnn_v1/sweeps/grid128_scalable_temporalcnn_v1/sweep_manifest.json`
- `Outputs/GridModel/060126_crop512_highres_temporalcnn_v1/sweeps/grid128_scalable_temporalcnn_v1/sweep_summary.tsv`

### Grid128 Stage A, June 9–13

- Manifest size: 972; stopped at index 477.
- Append-only log: 1,110 records, including 642 duplicated resume skips,
  439 completions, and 29 failures.
- The comparison contains 467 metric rows: 215 positive, 250 negative, and
  2 zero test improvements.
- Best learned/global improvement was `0.000896324986`; best active-cell
  improvement was `0.00145616231`.
- Structured active-cell metrics covered only 151 rows.
- The merged failure audit contains 594 attempts—593 OOM and one stale-code
  `NameError`—including archived retries, not unique specifications.
- Batch sizes 64, 8, and 4 repeatedly exceeded memory; batch 2 was materially
  safer.

This establishes strong passive short-horizon forecasting and metric-dependent
model ranking. It does not establish behavior, intent, or action effects. The
57-experiment Stage B plan was validated but never launched.

Evidence:

- `Outputs/GridModel/060126_crop512_grid128_max_v1/plans/grid128_stage_a_stop_review_v1/stage_a_stop_review.md`
- `Outputs/GridModel/060126_crop512_grid128_max_v1/comparison_grid128_sequence_1day_v1/results_intelligence.json`
- `Outputs/GridModel/060126_crop512_grid128_max_v1/plans/grid128_stage_b_launch_readiness_v1/stage_b_launch_readiness.md`

### Latent interpretation and direction smoke test, June 10–13

- Autoencoder corpus: 17,664 frames, 11 videos, latent dimension 64.
- Frame labels: 6,401 left, 6,486 neutral, 4,777 right.
- Leave-one-video nearest-centroid accuracy was `0.1818`.
- Ridge leave-one-video-out accuracy was `0.3636`, balanced accuracy
  `0.3611`, macro F1 `0.3545`, and majority accuracy `0.3636`.

This is weak whole-video label evidence. It is not a pre-movement intent test:
labels come from video identity, not synchronized movement onsets.

Evidence:

- `Outputs/GridModel/060126_crop512_grid128_max_v1/reports/latent_interpretation_autoencoder128_v1/latent_interpretation_report.md`
- `Outputs/GridModel/060126_crop512_grid128_max_v1/plans/latent_head_smoke_v1/latent_classifier_report.md`

### Manual ROI/spike follow-up, June 29

- Twenty ROIs across two videos: 10 right and 10 left.
- Ninety-nine spike intervals.
- Twelve of 12 temporal-CNN follow-ups completed and improved full-test MSE.
- Best h5 event-window improvement was `0.000946504859`; best h2 was
  `0.000709812052`.

The h5 result is provisional. One left-video interval spans 1,118 frames,
28.7% of all 3,895 annotated spike frames. The import also records a workbook
title mismatch and one omitted reversed range. Re-review and per-video/per-ROI
balanced scoring must precede further conclusions.

Evidence:

- `Outputs/GridModel/060126_crop512_grid128_max_v1/annotations/manual_roi_spikes_v1/manual_roi_spike_annotations.json`
- `Outputs/GridModel/060126_crop512_grid128_max_v1/plans/manual_roi_spike_stage_b_gate_v1/manual_roi_spike_stage_b_gate.md`

### Frozen Spon Ca Burst transfer, July 13

- Processed 560 frames and scored 460.
- Peak RSS was 757.8 MiB under a 1,024 MiB cap.
- Found 300 provisional dark zones: 281 passed direct residual and 90 passed
  local residual CFAR.
- Direct event/control ratio was `1.9951`; local-CFAR ratio was `0.7588`.

This established bounded execution and stronger broad-excitation response in
the direct-amplitude lane. With no manual truth and incomplete cross-domain
provenance, it did not establish detection accuracy.

Evidence:
`Outputs/SomaExcitation/spon_ca_burst_transfer_v2_cpu_guarded/experiment_summary.json`.

### Learnable contrast v1, July 21

- Labels: 79 point-window rows, 27 unique coordinates, four burst windows.
- Learned mean held-out recall: `0.132763975`.
- Direct-residual mean held-out recall: `0.605615942`.
- Learned fold wins: 0.
- Masked Recall@20 was 0.175 for learned versus 0.075 for direct.

The learned score may have complementary proposal-ranking value, but it was much
weaker at known-center recovery. Precision is unidentified because labels are
sparse positives and the unmatched candidate remains unknown.

Evidence:
`Outputs/LearnableContrast/spon_ca_burst_v1_cuda_guarded/experiment_summary.json`.

### Spatiotemporal and initialization factorial v2, July 21

- Eight factor combinations, 32 outer conditions, 64 learned fits.
- Best learned recall: `0.205141477` versus direct `0.605615942`; 0/4 fold
  wins.
- Stabilized scaling added `0.064441` recall with fixed initialization; jitter
  added `0.007937`.
- The tested Kalman + spatial Gaussian + temporal Gaussian + quiet-MAD lane
  reduced direct recall to `0.2937` and every learned Kalman cell scored zero.

This establishes that scaling mattered more than initialization and that this
specific preprocessing/objective combination was harmful. It does not establish
that temporal context is generally harmful.

Evidence:
`Outputs/LearnableContrast/spon_ca_burst_spatiotemporal_factorial_v2/experiment_summary.json`.

### Learnable raw-direct v3, July 21

- Nine cumulative configurations × four outer bursts = 36 fits.
- Every configuration tied frozen direct at `0.605615942` and won 0/4 bursts.
- Pooled recall was 49/79 = `0.620253`; the primary unweighted burst mean was
  `0.605616`.
- Training loss decreased, parameters moved, and candidate counts changed by
  -1 to +4.

The optimizer worked, but the deployed decisions did not improve. The central
unresolved mismatch is that training uses quiet examples at labeled
coordinates, while deployment calibrates full-field quiet NMS peaks.

Evidence:
`Outputs/LearnableContrast/spon_ca_burst_learnable_direct_tuning_v3/validation_report.md`.

### Morphology-aware multi-hypothesis CFAR v4-v6, July 26

- V4 screened 24 fixed experts: two morphologies, three radii, two reference
  estimators, and two temporal pools. The strongest diagnostic expert was the
  radius-8 crowded-center/coherence branch at `0.340839` mean recall (28/79).
- Fusion of all experts was harmful: log-mean-exp reached `0.155538` and max
  fusion `0.034679`. Membrane branches averaged `0.0038`, but morphology types
  are not annotated, so this is not evidence that membrane events are absent.
- Leakage-safe expert selection on three bursts and evaluation on the fourth
  reached `0.315839` (26/79): 3/15, 4/20, 9/21, and 10/23.
- V5 exposed diffuse initialization: its best expert received only 14-17% gate
  mass and the learned result fell to `0.214182` (18/79).
- V6 used a top-two, temperature-0.02 prior and a 10%-bounded contextual
  residual. It reached `0.329374` (27/79) with 53 candidates, versus 59 for
  nested fixed selection. The mean gain was `0.013535`, below the predeclared
  `0.02` C2 gate, and one burst regressed, so kernel-residual training did not
  run.
- Peak CUDA allocation was 337 MiB and peak RSS approximately 1.64 GiB, well
  below the 9,216 MiB GPU and 16,384 MiB RAM caps.

This establishes that scale, robust spatial reference, and causal temporal
coherence can more than double guarded CFAR known-center recall, and that
selective initialization matters. It does not beat Raw Direct (`0.605616`),
identify ordinary precision, or validate membrane/crowding subtypes. The next
high-information step is morphology/neighborhood annotation and exhaustive
review of a fixed candidate panel, not additional blind optimization.

Evidence:
`Outputs/LearnableContrast/spon_ca_burst_multihypothesis_cfar_v4/results.json`,
`Outputs/LearnableContrast/spon_ca_burst_multihypothesis_cfar_v5_selective_gate/results.json`,
and
`Outputs/LearnableContrast/spon_ca_burst_multihypothesis_cfar_v6_sharp_gate/results.json`.

### Causal artifact/baseline proposal program, July 27

- Completed all 1,884 declared evaluations and reproduced both frozen C0
  anchors: Raw Direct `0.605615942` and causal artifact-only `0.734187371`.
- The C1 nested comparison passed (`0.705020` versus `0.594746`, 4/4 burst
  wins). The best recall remained 58/79 rather than exceeding the causal anchor.
- A slow-clipped-EMA fractional method retained 58/79 with 488 candidates,
  compared with 745 for causal artifact-only, making it the most useful
  candidate-yield result.
- Fixed CFAR recovered 34/79 with 65 candidates. Ten-percent CFAR fusion added
  no recall and reduced candidates by only 1.34%; C2 stopped as designed.
- Robustness median and lower quartile were `0.734187`; the worst tested
  photobleach condition was `0.6464`.

The program is temporarily sidelined. Sparse positives do not establish the
precision of its lower-candidate variants; the 206-row frozen review queue needs
foreground/background review before another search. Evidence:
`Outputs/FrameDifference/spon_ca_burst_causal_proposal_overnight_v1`.

### Behavior/action readiness

Behavior alignment and inverse-dynamics export support timestamps, residuals,
gaps, duplicates, resampling, checksums, and explicit alignment states, with
synthetic tests. No real alignment report, inverse-dynamics bundle, behavior
trace, stimulation log, or action log was found in the current local data
inventory. The example manifest leaves all behavior/action paths empty.

This establishes tooling scaffolding only.

## Repeated failure modes

- **Proxy mismatch:** global MSE or training loss improves while the scientific
  decision does not.
- **Incomplete truth:** sparse positives cannot support ordinary precision.
- **Signal suppression:** background/temporal preprocessing can remove broad
  direct-amplitude signal.
- **Leakage:** video identity and post-onset motion can masquerade as intent.
- **Oversubscription:** repeated OOMs support batch-2, one-GPU-job guardrails.
- **Counting ambiguity:** resume skips and archived retries are not unique
  experiments.
- **Provenance gaps:** annotation warnings and missing behavior/action logs
  change interpretation.
- **Stage conflation:** forecasting, detection, intent, causality, and control
  are different estimands.
