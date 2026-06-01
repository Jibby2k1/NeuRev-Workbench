# Overnight Validation Runs

Use this when the goal is to stress Neurobench before analyzing new lab data.
The runner repeatedly executes a documented test matrix for a fixed duration and
writes a concise brief suitable for lab discussion.

```bash
python3 tools/run_validation_soak.py --duration-hours 9
```

Outputs are written under:

```text
Outputs/ValidationRuns/validation_soak_<timestamp>/
```

Key files:

- `experiment_brief.md`: concise human-readable summary
- `summary.json`: aggregate pass/fail counts and timing
- `events.jsonl`: one machine-readable record per command
- `logs/`: stdout/stderr for every command run
- `metadata.json`: git state, platform, Python path, and test inventory

The matrix covers:

- full repository pytest
- pipeline execution, sweeps, and device fallback
- workbench, Process Lab intermediate export, and CLI attachment
- object/event metrics, reports, annotation exports, inverse-dynamics exports
- documentation, generated API reference, schema validation, and environment setup

The run is a validation soak, not a new scientific result. Its purpose is to
show that the tooling was stable before starting experiments on newly acquired
data.
