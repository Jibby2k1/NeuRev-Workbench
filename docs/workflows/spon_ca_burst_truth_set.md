# Spon Ca Burst Truth-Set Workflow

Only deterministic synthetic/tiny-fixture execution is authorized by this workflow. A full Spon or GPU run requires separate, current authorization.

## Commands

```bash
.venv-neurobench/bin/python -m neurobench.cli.main workbench truth-set preflight \
  --manifest examples/spon_ca_burst_truth_set_v1.example.json

.venv-neurobench/bin/python -m neurobench.cli.main workbench truth-set build-package \
  --manifest examples/spon_ca_burst_truth_set_v1.example.json \
  --output-root /tmp/neurev_truth_set_tiny

.venv-neurobench/bin/python -m neurobench.cli.main workbench truth-set audit \
  --truth-set-root /tmp/neurev_truth_set_tiny
```

Preflight is read-only. Build refuses collisions and begins in raw-first mode with no candidate rows in the Review payload. A normal initial A0 result is `incomplete` because no human review has occurred.

## Review sequence

1. Freeze source identity, regions, coverage masks, lane artifacts, candidate construction, budgets 20/40, NMS, matching, ties, code SHA, and the resolved manifest.
2. Review raw and neutral fixed evidence. Add objects/events even when no lane proposed them. Publish the raw-first annotation revision and lock the pass.
3. Reveal only opaque candidates. Disposition every union item as neuron/event, artifact, background, or unresolved; never coerce unresolved to a negative.
4. Select at least 20% independently within accepted/rejected strata and include all unresolved/disagreements. Use the existing agreement and disagreement infrastructure, then adjudicate or retain unresolved explicitly.
5. Unseal once. Any later analysis-policy change creates a successor truth set; prior protected artifacts remain immutable.
6. Run A0. Only an `advance` result permits coverage-authorized protected metrics.

## Blinding and publication

The browser payload contains no lane ID, score, rank, detector-specific filename, color, CSS identifier, or hidden source field. `private/candidate_source_key.json` is generated output and must not be committed or copied into the browser app. Real packages live in ignored, collision-safe roots. Published annotation revisions are referenced, not rewritten.

## Metric interpretation

Known-positive recall, Recall@K, candidate efficiency, acceptance at K, reviewer time, unresolved fraction, agreement, and stability are available in all coverage modes. Precision, AP, false-event, spatial/timing error, duplicate/split/merge, calibration, and abstention metrics require `exhaustive`. Group uncertainty by persistent object identity and burst.

This workflow implements the scientific-audit standard through explicit fingerprints, artifact index, validation, LLM context, report, and immutable evidence roles. Synthetic validation is engineering evidence only and makes no detector-performance claim.
