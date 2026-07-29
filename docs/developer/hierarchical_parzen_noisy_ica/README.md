# Codex Package: Hierarchical Parzen / Noisy ICA

Read in this order:

1. `../../research/HIERARCHICAL_PARZEN_NOISY_ICA.md` when installed in the
   repository, or the package-level research note.
2. `IMPLEMENTATION_BRIEF.md` for the scientific and software contract.
3. `EXPERIMENT_AND_METRICS_CONTRACT.md` for mandatory experiments, figures,
   tables, and advancement gates.
4. `CODEX_EXECUTION_PLAN.md` for ordered work packages and stop conditions.
5. `../../../examples/spon_ca_burst_hierarchical_parzen_noisy_ica.example.json`
   for the proposed strict manifest.

## Directive

Implement a reversible, stage-gated experiment:

```text
X
  -> Stage 1 temporal Parzen ICA
  -> B_hat + R
  -> Stage 2 local noisy Parzen ICA
  -> S_hat + N_hat
```

The primary scientific output is not an ICA component. It is the explicit
accounting identity

```text
X approximately equals B_hat + S_hat + N_hat
```

with measured closure, attribution leakage, signal preservation, residual
validity, stability, detection utility, and latency.

## Current authorization

The package authorizes:

- reusable numerical code;
- unit tests;
- deterministic synthetic and semi-synthetic fixtures;
- strict configuration and preflight;
- tiny CPU/GPU smoke tests;
- report and visualization builders.

It does not authorize:

- overwriting any completed `Outputs/` root;
- a full Spon run;
- a long GPU sweep;
- automatic replacement of Raw Direct or the latent smoother;
- calling an unexplained residual `noise` without the required diagnostics.
