# Spon Ca Burst Nested Innovation Ranker

## Purpose

This workflow tests whether complementary spatial, separation, temporal,
cut-morphology, and calibrated-noise views can improve neuron ordering at a
fixed candidate budget without replacing the accepted signed Parzen carrier.
It consumes the completed v1 feature bank and never classifies unmatched event
candidates as negatives.

## Stages

1. Pool the 13 maintained full-resolution feature channels.
2. Generate 21 additional feature maps: signed powers, residual
   heteroscedasticity corrections, onset/energy interactions, artifact
   attenuation, center/annulus cut-morphology responses, crowd context, and
   structural context.
3. Form a spatially deduplicated union from 22 proposal sources.
4. Measure optimistic union coverage at budgets 20, 40, 58, 80, and 100.
5. Screen all 34 candidate features at the common proposal level.
6. Fine-tune 135 carrier-anchored bounded-linear configurations and 120
   carrier-skip residual-MLP configurations.
7. Select hyperparameters with nested leave-one-burst-out evaluation.
8. Refit the selected linear model once and the selected MLP with three
   confirmation seeds for every outer burst.
9. Write budget curves, exact per-neuron recovery audits, and verified
   score/overlay TIFFs.

The exact fit count is 2,250 inner models and 16 outer refits. MLP inner
selection averages two seeds; final confirmation averages three.

## Leakage control

For outer burst \(h\), every hyperparameter is selected using only the other
three bursts. Each candidate configuration is fit on two of those bursts and
validated on the third. Only after selection is it refit on all three and
evaluated on \(h\).

Training uses known positives that have a proposal within six pixels and
full-field quiet hard negatives. Unmatched event candidates remain unknown.
Candidate count is selectivity pressure, not measured precision.

## Model contracts

The linear model has an immutable unit carrier skip:

\[
s(p)=C(p)+\sum_j w_j d_j\phi_j(p),
\qquad w_j\geq 0,\quad \sum_j w_j\leq A.
\]

The MLP also begins at the exact carrier because its final residual layer is
initialized to zero:

\[
s(p)=C(p)+A\tanh\{W_2\,\mathrm{ReLU}(W_1\phi(p)+b_1)+b_2\}.
\]

Maximum residual authority is 0.25 or 0.5. Learning rate, regularization,
hidden width, authority, and feature set are selected only inside nested folds.

## Commands

```bash
.venv-neurobench/bin/python -m neurobench.cli.main \
  experiment innovation-ranker preflight \
  --config examples/spon_ca_burst_innovation_ranker_v5.example.json

.venv-neurobench/bin/python -m neurobench.cli.main \
  experiment innovation-ranker run \
  --config examples/spon_ca_burst_innovation_ranker_v5.example.json

.venv-neurobench/bin/python -m neurobench.cli.main \
  experiment innovation-ranker missed-video \
  --config examples/spon_ca_burst_innovation_ranker_v5.example.json \
  --ranker-root Outputs/HierarchicalParzenICA/spon_ca_burst_innovation_ranker_v5 \
  --output-dir Outputs/HierarchicalParzenICA/spon_ca_burst_innovation_ranker_v5_missed_video_v2
```

The run requires its exact ready preflight and refuses an existing completed or
partial output.

The missed-neuron command is a bounded post-run audit. It streams the raw
review interval directly to two H.264 videos rather than retaining rendered
frames in RAM. It refuses to overwrite an existing output and writes a
machine-readable manifest plus the exact missed observations. Blue boxes mark
currently inactive identities that were missed in at least one burst; red
marks a labeled-active observation missed by the selected recovery field; and
green marks a labeled-active observation of the same identity that was
recovered in another burst. Both videos use one fixed raw-intensity scale so
visibility does not improve artificially through per-frame normalization.

## Primary outputs

```text
Outputs/HierarchicalParzenICA/spon_ca_burst_innovation_ranker_v5/
  REPORT.md
  metrics.json
  evaluation/
    candidate_inventory.json
    feature_screen.json
    oracle_coverage.json
    budget_curves.json
    inner_fine_tuning.json
    nested_rankers.json
    per_neuron_audit.tsv
    recoverable_but_missed.tsv
    not_in_feature_union.tsv
  models/
  diagnostics/
```

The completed v5 missed-neuron audit is written separately:

```text
Outputs/HierarchicalParzenICA/spon_ca_burst_innovation_ranker_v5_missed_video_v2/
  manifest.json
  missed_observations.tsv
  missed_neurons_raw_full.mp4
  missed_neurons_raw_zoom.mp4
```

Diagnostic overlays use green for a candidate matched to a known label, cyan
for an unmatched candidate of unknown scientific status, and red for a known
label missed at the fixed budget. No cyan marker should be called a false
positive without exhaustive annotation.

## Completed v5 result

Version 5 is authoritative. Earlier v1–v3 completed roots and the failed v4
partial are retained as evaluator audit records. V5 excludes zero-evidence NMS
plateaus and distinguishes the archived centered residual from the stored
quiet-standardized carrier.

At 58 candidates per burst, the archived centered residual scored `0.641`, the
quiet-standardized carrier scored `0.703`, the same carrier score on the broad
proposal union scored `0.709`, and nested linear ranking scored `0.725`.
Nested single-feature selection reached `0.734`. The residual MLP tied the
linear model and is not preferred. At budgets 20 and 40, the native
quiet-standardized carrier remained best, so the learned model is not yet a
precision-oriented promotion.

The full interpretation is in
`docs/research/SPON_CA_BURST_INNOVATION_RANKER_V5_RESULTS.md` and the
cross-study synthesis is in
`docs/research/SPON_CA_BURST_RECENT_EXPERIMENT_SYNTHESIS_2026_07_30.md`.
