# Research record templates

These templates are the human-readable source shapes used by
`tools/new_research_experiment.py`. They follow the canonical JSON Schemas in
`research/schemas/`; token values such as `__EXPERIMENT_ID__` are replaced by
the scaffold tool before schema validation.

Use the tool instead of copying these files by hand:

```bash
.venv-neurobench/bin/python tools/new_research_experiment.py \
  --experiment-id NREV-EXP-0100 \
  --decision-id NREV-DEC-0100 \
  --run-id NREV-RUN-EXP-0100-PILOT-01 \
  --program-id NREV-PRG-0001 \
  --title "Frozen transfer on an independent recording" \
  --question "Does the frozen feature panel retain its ranking on an independent recording?" \
  --objective "Estimate transfer performance without refitting feature definitions." \
  --rationale "Independent-recording evidence is an unresolved program boundary." \
  --design-summary "Apply the frozen panel and operating point to a held-out recording." \
  --protocol-path docs/workflows/example_transfer_protocol.md \
  --unit-of-analysis "immutable ROI site within recording" \
  --comparison "frozen panel versus the registered raw-direct comparator" \
  --data-scope "one independent recording; no development-recording refit" \
  --grouping-or-blocking "block by recording and spatial neighborhood" \
  --sample-plan "evaluate every frozen eligible site in the independent recording" \
  --randomization-plan "use registered deterministic seeds for any stochastic fit" \
  --blinding-plan "freeze outputs before independent label access" \
  --analysis-plan "report the prespecified primary metric with grouped uncertainty" \
  --falsifier "The prespecified primary metric does not exceed the comparator gate." \
  --gate transfer analysis "The frozen primary metric exceeds the comparator by the preregistered margin." \
  --config examples/example_transfer.json \
  --input DATASET-TRANSFER-01 data research/data-registry/transfer-01.yaml SHA256 \
  --seed 1001 \
  --output-root Outputs/NeuronIdentifiability/NREV-EXP-0100/runs/NREV-RUN-EXP-0100-PILOT-01
```

The default is a dry run. Review the paths and captured provenance, then repeat
the command with `--write`. The tool validates all three records before writing,
uses exclusive file creation, and updates only the registry index. It refuses
existing record files, existing output roots, absolute paths, `Inputs/` paths,
private-review paths, and input locations that are not durable URLs or committed
sanitized descriptors.

## What each template means

- `native-experiment.template.yaml` records a scientific question, a frozen
  comparison, falsifiers, gates, and interpretation boundaries. It starts as
  `draft` with `not_evaluated` outcome.
- `draft-decision.template.yaml` records a planning hold. It is not a result and
  does not authorize a run, publication, promotion, or scientific claim.
- `planned-run.template.yaml` records exact code, configuration, input,
  environment, randomization, output-root, and validation provenance for a run
  that has not started.
- `experiment-protocol.template.md` is a checklist for the protocol referenced
  by the experiment record. Copy it to `docs/workflows/`, resolve every marker,
  and review it before scaffolding the registry records.
- `evidence-capsule.after-validation.template.json` is deliberately not emitted
  by the planning tool. Create a capsule only after output and scientific-audit
  validation; never pre-populate a favorable outcome.

## Lifecycle after scaffolding

1. Review and freeze the protocol, falsifiers, gates, data split, and resource
   envelope.
2. Record explicit run authorization outside the planning record.
3. Execute into the registered, non-colliding output root.
4. Update the run with observed timestamps, validation checks, and manifest
   hashes; never rewrite a historical run into a different execution.
5. Validate the scientific audit package defined in
   `docs/workflows/SCIENTIFIC_AUDIT_OUTPUT_STANDARD.md`.
6. Publish a sanitized evidence capsule, update claim effects, and replace the
   planning hold with an evidence-backed decision.
7. Run `.venv-neurobench/bin/python -m neurobench.research.registry build`, then
   `check`; generated files are never edited directly.

`visibility` is descriptive metadata, not access control. Anything committed to
a public Git repository is public regardless of that field. Restricted data,
review identities, randomization keys, credentials, raw recordings, and large
outputs must stay outside Git.
