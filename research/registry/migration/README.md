# v1 research-story migration

This directory preserves the transition from the manuscript-owned v1 story to
the repository-level registry.

- `research_story_v1.yaml` is a normalized YAML reserialization of the v1
  story. It is a semantic snapshot, not a byte-for-byte archive.
- `story_compatibility.yaml` retains only the ordering, document profiles,
  direct evidence paths, and planned-item metadata needed to reproduce the old
  Overleaf view from normalized records.
- `../views.yaml` records the original-source hash, replay source, normalized
  snapshot hash, migration date, and the assertion that the migration made no
  semantic change.

The migration preserves every legacy claim statement, status, scope,
experiment question, design summary, finding, limitation, next decision, and
priority. It does not reconstruct missing execution dates, commits,
configurations, inputs, environments, seeds, preregistered gates, or
experiment-specific falsifiers.

For that reason, the 19 historical experiments have evidence capsules but no
invented run records. Their `origin.kind` is `migrated_legacy` and their
`missing_fields` lists are part of validation. Native future experiments use
`origin.kind: native` and must satisfy the complete design and provenance
contracts.

To prove semantic reconciliation:

```bash
.venv-neurobench/bin/python -m neurobench.research.registry check
```

The compiler rebuilds the paper story from normalized records and fails if it
does not equal the retained semantic snapshot.
