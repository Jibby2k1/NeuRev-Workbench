# Spon Ca Burst neuron-identifiability paper workflow

This program is a single-recording, within-recording methodological case study.
It preserves immutable observation-site geometry independently from provisional
canonical-neuron identity and treats unmatched candidates as unknown.

Run with:

```bash
.venv-neurobench/bin/python -m neurobench.experiments.neuron_identifiability run \
  --config examples/spon_ca_burst_neuron_identifiability_paper_v1.example.json \
  --resume --allow-provisional-labels
```

The output root is collision-safe and each stage writes the common atomic
artifact contract. Original-site/original-timing analysis is primary while
identity and timing adjudication remain incomplete. Stages that lack a valid
input or protected validation path must record `complete_degraded` or
`stop_branch`; they must not promote provisional findings.

The scientific-audit output standard remains mandatory for any promoted
detector lane. Sparse-positive labels support known-positive recovery and
candidate burden, not full-field precision, specificity, or false-positive
claims.
