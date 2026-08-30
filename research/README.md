# NeuRev research registry

This directory is the repository-level source of truth for NeuRev's scientific
story. Each fact is recorded once, reviewed in context, and compiled into many
views: the GitHub landing page, documentation, experiment lineage, claim and
evidence matrices, machine navigation, and the Overleaf manuscript.

The registry separates **what was run**, **what happened**, **what the evidence
can support**, and **what the team decided**. A completed computation is not
automatically a validated experiment, a supported claim, or permission to
advance a research program.

## Canonical layout

```text
research/
├── registry/
│   ├── programs/<program-slug>.yaml
│   ├── claims/<claim-id>.yaml
│   ├── experiments/<experiment-id>.yaml
│   ├── decisions/<decision-id>.yaml
│   ├── runs/<run-id>.yaml
│   └── migration/                 # frozen semantic snapshot and compatibility map
├── evidence/<experiment-id>.json
├── schemas/
└── generated/                    # compiled views; never hand edited
```

YAML is used for human-maintained records. Evidence capsules use JSON so they
can be checked, exchanged, and archived without a YAML parser. Both formats are
validated with the JSON Schemas in `research/schemas/`.

Large results remain under `Outputs/<program>/<experiment-id>/runs/<run-id>/`
or in an external archive. The committed capsule contains only portable
metadata, bounded metrics, checksums, claim effects, validation gates, and
links. Raw recordings, reviewer identities, private randomization keys,
credentials, workstation paths, and other restricted material do not belong in
this directory.

## One record, many views

Canonical records are inputs, not presentation files. The registry compiler
joins records by stable IDs and produces disposable views such as:

- current research state and next experiments;
- experiment timeline and lineage graph;
- claim-to-evidence and decision matrices;
- repository and LLM navigation indexes;
- README facts, documentation pages, and manuscript tables or prose.

Generated files must carry a source fingerprint and fail a stale-output check
when their inputs change. To correct a generated statement, edit its canonical
record and rebuild; never patch the generated view.

## Orthogonal scientific state

State is intentionally multidimensional:

| Dimension | Question answered | Controlled examples |
| --- | --- | --- |
| `lifecycle` | Where is the work operationally? | `draft`, `preregistered`, `running`, `computed`, `validated`, `reviewed`, `closed` |
| `outcome` | What did the prespecified comparison find? | `supported`, `rejected`, `mixed`, `inconclusive`, `not_evaluated` |
| `evidence_tier` | What setting produced the evidence? | `descriptive`, `current_recording`, `computational_simulation`, `external_review`, `independent_recording` |
| `review_state` | What scrutiny has the record received? | `not_requested`, `in_review`, `approved`, `adjudication_required`, `adjudicated` |
| `claim_state` | What is the current standing of a claim? | `proposed`, `provisional`, `supported`, `contradicted`, `unresolved`, `retired` |
| decision `action` | What should happen next? | `promote`, `retain`, `hold`, `stop`, `supersede` |

These values must not be collapsed into a single status. Enumeration order is
not a strength ranking: for example, external review of one recording and an
independent recording answer different questions. During migration,
`legacy_status` preserves the original wording for provenance, but generated
views use the controlled dimensions.

`origin.kind` is either `native` or `migrated_legacy`. Native experiments must
register complete design fields, falsifiers, and gates. The migrated v1 story
may omit facts that its source never recorded, but only when it carries the
frozen source path, checksum, migration date, and an explicit `missing_fields`
list. `legacy_status` preserves wording; it never relaxes validation.

## Stable IDs and references

Records use immutable, opaque identifiers:

- `NREV-PRG-####` — program;
- `NREV-CLM-####` — claim;
- `NREV-EXP-####` — experiment;
- `NREV-DEC-####` — decision;
- `NREV-RUN-<TOKEN>` — run;
- `NREV-EVC-<TOKEN>` — evidence capsule.

Renaming a title or moving a rendered view never changes an ID. Relationships
are stored as IDs, not inferred from filenames or prose. Repository paths must
be relative and traversal-free; external artifacts use stable HTTPS or DOI
URLs plus checksums when available.

## Intended lifecycle

```text
question + falsifier
        ↓
freeze protocol and gates
        ↓
run with code, data, configuration, and seed provenance
        ↓
validate outputs and scientific audit package
        ↓
review or adjudicate where required
        ↓
publish a sanitized evidence capsule
        ↓
update claims and record an explicit decision
        ↓
rebuild every generated view
```

Experiment transitions should be monotonic except for an explicit
invalidation. Negative and inconclusive results remain first-class records;
they are never removed merely because they do not advance a method.

## Schema responsibilities

- `program.schema.json` defines scope, boundaries, ownership roles, and program
  lifecycle.
- `claim.schema.json` defines atomic claims, scope, limitations, evidence tier,
  and claim state.
- `experiment.schema.json` defines the question, falsifiers, frozen design,
  gates, lifecycle, outcome, and linked claims.
- `decision.schema.json` records the evidence-backed action and its constraints.
- `run.schema.json` records execution state and reproducible code, input,
  configuration, environment, and artifact provenance.
- `evidence-capsule.schema.json` defines the small public evidence contract that
  links validated runs to metrics, gates, artifacts, and bounded claim effects.

The schemas define structural validity. Cross-record rules—unique IDs, valid
references, legal transitions, evidence/claim consistency, clean-checkout path
resolution, privacy scans, and stale generated outputs—belong in the registry
compiler and CI.

## Build and validation

```bash
# Regenerate GitHub, documentation, navigation, machine, and paper views
.venv-neurobench/bin/python -m neurobench.research.registry build

# Clean-clone-safe schema, cross-link, checksum, and stale-view validation
.venv-neurobench/bin/python -m neurobench.research.registry check

# Research-workstation validation, including ignored metadata-only artifacts
.venv-neurobench/bin/python -m neurobench.research.registry verify-live --check
```

The migrated 19 experiments have evidence capsules but no run records. Their
source did not contain commits, input hashes, configurations, environments,
randomization, or manifests, so the migration deliberately did not invent
them. Future native executions must use the strict run schema.

The registry compiler is a source-checkout tool. Python wheels contain its
implementation for editable development, but do not duplicate the canonical
repository-level records. Run registry commands from an editable NeuRev
checkout; an installed wheel without those records fails with an explicit
checkout-boundary message.
