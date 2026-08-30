# [Experiment title]

- Status: draft
- Program: `NREV-PRG-____`
- Experiment: `NREV-EXP-____`
- Owner role: [role, never private reviewer identity]

## Question and decision

- Question:
- Decision this experiment can change:
- Atomic claim IDs affected:
- What a supported result would permit:
- What it would **not** establish:

## Frozen design

- Unit of analysis:
- Data scope and exclusions:
- Grouping or blocking unit:
- Development, validation, and test separation:
- Comparator and operating point:
- Sample or seed plan:
- Randomization:
- Blinding:
- Primary metric and uncertainty method:
- Secondary diagnostics:

## Falsifiers and gates

| ID | Stage | Required | Criterion | Action on failure |
| --- | --- | --- | --- | --- |
| [gate] | preflight/analysis/review/publication | yes | [criterion] | hold/stop |

List at least one result that would falsify or materially qualify the proposed
claim. A completion count is not a scientific gate.

## Provenance and resources

- Sanitized input identifiers and hashes:
- Configuration path and schema:
- Code commit and dirty-state policy:
- Seeds or explicit non-random rationale:
- CPU/GPU, memory, disk, and wall-time envelope:
- New non-colliding output root:
- Resume and atomic-write behavior:

## Scientific audit

State how this experiment implements
`docs/workflows/SCIENTIFIC_AUDIT_OUTPUT_STANDARD.md`, including section
applicability, expected counts, full-field and close-up media, trace views,
comparison tables, compact LLM index, and validation. If the user explicitly
authorizes an opt-out, record the exact reason here and in resolved
configuration; do not silently omit the audit.

## Publication and privacy boundary

- Content that may enter a public evidence capsule:
- Content that remains under ignored `Inputs/` or `Outputs/`:
- External artifact archive and checksum plan:
- Reviewer identity and private randomization handling:
- Clean-clone checks:

## Review record

- Protocol frozen on:
- Protocol SHA-256:
- Code commit:
- Required approvals or explicit run authorization:
