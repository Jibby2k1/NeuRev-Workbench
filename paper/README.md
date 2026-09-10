# NeuRev manuscript guide

NeuRev has two distinct paper tracks. Start with the
[current research overview](../docs/research/CURRENT_WORK.md) to see the latest
results and their limits.

| Track | Main question | Repository entry point |
| --- | --- | --- |
| Identity-safe measurement and evaluation | What can sparse-positive calcium-imaging evidence establish about candidates, localization, and biological identity? | [Manuscript source and build guide](overleaf_jnm/README.md), [current research state](overleaf_jnm/CURRENT_RESEARCH_STATE.md), [venue drafts](overleaf_jnm/VENUE_DRAFTS_README.md) |
| Causal Gamma-LS candidate extraction | Can a simple causal front end produce localized event candidates at a manageable burden? | [Final September 9 campaign report](../docs/research/SPON_CA_BURST_GAMMA_LS_PAPER_SUCCESS_RESULTS_2026_09_09.md), [paper outline](../docs/research/SPON_CA_BURST_GAMMA_LS_PAPER_V2_OUTLINE.md), [implementation](../neurobench/experiments/gamma_ls_difference/) |

The identity-safe cohort, protected Gamma-LS burst comparison, separate-recording
block evaluation, and operational framewise proposal ledger have different
denominators and readouts. Keep their estimands separate when revising prose.
ICA/whitening has its own [final result report](../docs/research/ICA_WHITENING_REAL_DATA_V1_FINAL_RESULTS.md).

## Maintained sources and local distributions

`overleaf_jnm/` contains the maintained identity-safe manuscript sources, small
curated figures, and reproducible builders. The JNM and Neuroinformatics venue
drafts remain working manuscripts with open submission gates.

Dated copies under `overleaf_uploads/`, ZIP archives, and local Gamma-LS manuscript
distributions are release payloads. They remain outside Git; preserve their
original manifests and hashes. The local Gamma-LS distribution is not the
identity-safe Overleaf manuscript. Its current scientific authority in this
repository is the September 9 campaign report linked above.

Generate the identity-safe story from canonical records before building:

```bash
.venv-neurobench/bin/python -m neurobench.research.registry build
make -C paper/overleaf_jnm story-check
```

Use the [venue-package builder](overleaf_jnm/scripts/build_venue_overleaf_packages.py)
for dated upload copies. Consult its arguments and the
[venue guide](overleaf_jnm/VENUE_DRAFTS_README.md) before generating a new
distribution. Preserve earlier completed packages.

## Submission boundaries

Neither a compiled draft nor a passing software or artifact check establishes
submission readiness. Author, ethics, acquisition, citation, and release metadata
still require resolution. The Gamma-LS report retains failed support-sufficiency
and strict streaming gates; sparse-positive biological precision and broader
generalization remain unresolved. See the
[identity-safe checklist](overleaf_jnm/submission_checklist.md) and
[publication boundary](../docs/PUBLICATION_BOUNDARY.md).
