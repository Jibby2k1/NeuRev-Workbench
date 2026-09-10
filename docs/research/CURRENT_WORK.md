# Current research work

**Evidence refreshed: 2026-09-09.** Start here for the latest completed work, then follow the linked result reports for exact protocols, estimands, and artifact provenance. The [claim ledger](https://github.com/Jibby2k1/NeuRev-Workbench/blob/codex/neuron-identifiability-paper-20260822/research/generated/CLAIM_LEDGER.md) remains the authority for promoted claims. Dates on older reports describe those snapshots; they do not report live processes.

## Latest result: a simpler Gamma-LS candidate extractor

The [final Gamma-LS campaign report](SPON_CA_BURST_GAMMA_LS_PAPER_SUCCESS_RESULTS_2026_09_09.md) supports a fixed, single-scale causal pipeline:

```text
Acquired frame → Gaussian smoothing → causal EMA → signed difference
              → radial Gamma-LS → empirical threshold → deterministic NMS
```

Signed differencing remains the operational representation. Two-frame ICA did not pass its protected improvement rule; PCA was a control construction. The selected half-width of 15 pixels is an operational context, with support sufficiency unresolved. Max pooling is outside this deployed pipeline.

| Evidence | What the completed work shows | Limit |
| --- | --- | --- |
| Full-recording application | 371 frame-level proposals on 295 of 2,259 eligible frames at calibration target q=1; model-only media audit passed | No temporal linking or biological identification; proposal rows are not unique neurons or events |
| Separate `15 right` recording | 6/51 sparse-positive matches at B58 per one-second block | Calibration and evaluation share the recording; eligibility preflight inspected annotations; unmatched candidates remain unknown |
| Exact-truth simulator | Radial current-frame F1 0.78186 across 108 generated movies at one calibrated null proposal/frame | Active-frame truth favors different endpoints from onset-sensitive differencing; no biological precision estimate |
| Corrected batch timing | Approximately 1,470–2,216 frames/s across batches 1–64 | Replayed, matched workloads; acquisition and downstream control excluded |
| Two paced 1-kHz trials | 212 and 9,327 deadline failures out of 60,000 scheduled frames, with 37 drops in the second trial | Both strict streaming gates failed |

The protected burst-occupancy comparison, one-second block readout, and operational framewise ledger are distinct evaluations. Their recall values and candidate budgets are not interchangeable. See the [three readout definitions](SPON_CA_BURST_GAMMA_LS_PAPER_SUCCESS_RESULTS_2026_09_09.md#canonical-architecture-and-readout-definitions).

## Other completed research

| Workstream | Current conclusion | Read next |
| --- | --- | --- |
| ICA and whitening | The real-data completion audit passed. The selected ICA finalist matched 20/51 positives at B58 for each of three seeds on a frozen common candidate universe | [Final real-data results](ICA_WHITENING_REAL_DATA_V1_FINAL_RESULTS.md), [evaluation design](ICA_WHITENING_HYPERPARAMETER_EVALUATION_PLAN.md) |
| PC-MITL-ICA | A specialized synthetic effect missed the locked minimum improvement rule; retain CS-Parzen and stop the extension | [Confirmation result](PC_MITL_ICA_SPECIALIZED_CONFIRMATION_RESULTS.md), [paired E01/E02 study](PC_MITL_ICA_E01_E02_RESULTS.md) |
| Contextual envelopes | Temporal memory and agreement attenuation alter trace morphology; validation passed without detector promotion | [Morphology audit](CONTEXTUAL_ENVELOPE_MORPHOLOGY_V1_RESULTS.md), [retrieval result](CONTEXTUAL_ENVELOPE_RETRIEVAL_V1_RESULTS.md), [terminology](CONTEXTUAL_ENVELOPE_TERMINOLOGY_V1.md) |
| Feature Atlas v1 | A favorable incremental ranking estimate has an uncertainty interval crossing zero; exploratory | [Atlas report](SPON_CA_BURST_FEATURE_ATLAS_V1_RESULTS.md) |
| Uncertainty-aware learning v1.1 | Tiny nonlinear fusion did not improve the primary held-fold ranking endpoint over matched linear fusion | [Learning result](UNCERTAINTY_AWARE_FEATURE_LEARNING_V1_1_RESULTS.md) |
| Identity-safe evaluation | Canonical-v7 retains 106 occurrences at 50 immutable sites; selected candidate review and recentering audits have bounded meanings | [Current manuscript state](https://github.com/Jibby2k1/NeuRev-Workbench/blob/codex/neuron-identifiability-paper-20260822/paper/overleaf_jnm/CURRENT_RESEARCH_STATE.md), [candidate review](SPON_CA_BURST_NEW_CANDIDATE_REVIEW_V1_RESULTS.md) |

**The ICA 20/51 and Gamma-LS 6/51 figures are not a head-to-head comparison.** Their candidate construction, selection, calibration, and readouts differ. Use each experiment's paired controls and frozen contract to interpret it.

## Documents and reproducibility

- [Result index](README.md): detailed studies and preserved historical snapshots.
- [Workflow index](../workflows/README.md): runnable contracts and focused validation commands.
- [Paper guide](https://github.com/Jibby2k1/NeuRev-Workbench/blob/codex/neuron-identifiability-paper-20260822/paper/README.md): distinct manuscript tracks, source packages, and unresolved submission gates.
- [Registry guide](https://github.com/Jibby2k1/NeuRev-Workbench/blob/codex/neuron-identifiability-paper-20260822/research/README.md): canonical records and generated stories.
- [Publication boundary](../PUBLICATION_BOUNDARY.md): portable source and evidence rules.

Local run roots named in reports identify retained artifacts under ignored `Outputs/`; they are not downloadable public evidence. Read small summaries, validations, and artifact indexes before opening media. Dated paper distributions are preserved locally, while maintained manuscript sources and small curated figures belong in Git.

## Next scientific gates

The immediate priorities are biological adjudication of the exact operational framewise detector, frozen evaluation on additional recordings, and repeated service-time tests that account for tail latency and drops. Any stronger support-sufficiency or ICA claim requires a new prespecified comparison. Artifact validation, software tests, and a pushed branch do not close these scientific or submission gates.
