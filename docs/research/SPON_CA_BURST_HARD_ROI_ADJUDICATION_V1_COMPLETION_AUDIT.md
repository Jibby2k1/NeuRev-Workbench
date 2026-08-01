# Spon Ca Burst hard-ROI adjudication v1 completion audit

Audit date: 2026-08-01

## Outcome

The bounded engineering objective is implemented and validated. The scientific
objective is not yet complete because all 25 target observations still require
human final adjudication. The final non-provisional re-score is correctly gated
and has not been run.

## Requirement audit

| Requirement | State | Evidence |
| --- | --- | --- |
| Preserve original labels | Pass | Current SHA-256 remains `8f008fede5771d83e949d805094292aae7516cc30353dd383da7ff1cf20b077e`; the draft is a separate 79-row TSV. |
| Versioned, collision-safe outputs | Pass | Preflight, review pack, review aid, and exact preview use distinct roots; completed roots are refused. |
| Detector-blinded padded clips | Pass | Four H.264 clips; manifests record `detector_blinded=true` and no outcomes/scores are rendered. |
| Projection overlays | Pass | Four burst overlays are present, including preflight evidence. |
| Observation-level schema | Pass | The 24 fields cover identity, timing, confidence, morphology, context, disposition, inclusion views, provenance, and merges. |
| Required identities covered | Pass | ROI 007, 010, 014, 015, 019, 020, and 023 are included; uncertain burst-2 ROI 008 and 017 are also included. |
| Raw review aids without auto-labeling | Pass | 25 plots plus worksheet; `automatic_adjudication_performed=false`, `detector_outcomes_used=false`. |
| Original/confirmed/inclusive views | Pass | Implemented with canonical-identity collapse. |
| Original/adjudicated timing views | Pass | Implemented; event timing requires a complete ordered triple. |
| Frozen budgets and geometry | Pass | Budgets 20/40/58/80/100, six-pixel matching/NMS, ten-pixel relaxed localization. |
| Separate miss classes | Pass | Matched, identity conflict, ranking, temporal, localization, NMS, and proposal outcomes are emitted per observation at budget 58. |
| Exact quantitative feature path | Pass | Carrier is preserved; coherence and standalone lag recurrence are deterministically reconstructed on CPU. |
| Diagnostic feature limitation | Pass | Radial-shell and VST lanes are marked diagnostic-only because their stored TIFFs are clipped/quantized. |
| No tuning, GPU, or inferred negatives | Pass | Exact preview records all three safeguards explicitly. |
| Reproducible manifest paths | Pass | Every relative path resolves against the manifest; verified from `/tmp`. |
| Focused regression validation | Pass | 11 tests pass; out-of-directory manifest resolution passes. |
| Final human adjudication | Waiting | 11 target rows are provisional expert notes and 14 are pending; zero have accepted observation-specific timing. |
| Final non-provisional re-score | Waiting | The evaluator rejects incomplete target review; no final output root has been created. |

## Exact mechanics landmarks

Original labels with original timing reproduced these macro recalls:

| Lane | B20 | B40 | B58 | B80 | B100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Carrier | 0.540864 | 0.657246 | 0.703080 | 0.715580 | 0.726449 |
| `coherence_w15` | 0.605305 | 0.676449 | 0.722283 | 0.722283 | 0.745057 |
| standalone `propagation_lag2_w15` | 0.621972 | 0.680616 | 0.749819 | 0.761724 | 0.772593 |

The preview is a mechanics check, not evidence that provisional labels are
correct. The standalone lag lane is also distinct from the scientific audit's
cross-fitted all-family selector.

## Human work queue

All 25 target rows must be explicitly finalized, including those that already
contain an expert note:

| ROI | Provisional rows | Pending rows | Required decision |
| --- | ---: | ---: | --- |
| 007 | 1 | 3 | Neuron morphology versus activity-only/unresolved, per burst. |
| 008 | 2 | 0 | Whether burst-2 activity has a defensible cell identity. |
| 010 | 2 | 2 | Identity and canonical relationship to ROI 015. |
| 014 | 2 | 2 | Morphology and per-observation timing, especially early burst 2. |
| 015 | 1 | 3 | Confirm or reject merge into ROI 010 separately by occurrence. |
| 017 | 1 | 0 | Whether burst-2 flash has a defensible cell identity. |
| 019 | 1 | 2 | Subtle activity and early burst-2 timing. |
| 020 | 1 | 0 | Neuron versus localized dot/artifact/unresolved. |
| 023 | 0 | 2 | Morphology/activity in bursts 3 and 4. |

For each row, the reviewer must set final disposition and confidence, canonical
identity/merge reason where relevant, inclusion flags, `review_status`,
`reviewer_id`, and `reviewed_at`. Where timing is changed, onset, peak, and end
must all be supplied. The clips—not the detector audit—should determine these
fields; the raw-trace worksheet may assist timing.

## Release gate

After saving the completed table under a new versioned filename, run:

```bash
.venv-neurobench/bin/python \
  -m neurobench.experiments.hard_roi_adjudication.exact_rescore_cli \
  --config examples/spon_ca_burst_hard_roi_adjudication_v1.example.json \
  --adjudication-tsv /path/to/final_adjudication.tsv \
  --output-dir Outputs/HardROIAdjudication/spon_ca_burst_hard_roi_rescore_final_v1
```

The final interpretation should compare label-view sensitivity and timing-view
sensitivity separately, then examine shifts among proposal, ranking, NMS,
localization, temporal, and identity-conflict categories. It must not report
precision from this targeted sparse-positive panel.
