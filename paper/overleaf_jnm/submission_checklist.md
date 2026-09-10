# Journal submission checklist

This checklist covers the identity-safe working drafts in this directory.
Checked items record the existing package validation; they do not establish
journal submission readiness or validate later source edits automatically.
Use the [venue draft guide](VENUE_DRAFTS_README.md) to rebuild the maintained
sources. The separate [Gamma-LS results and remaining gates](../../docs/research/SPON_CA_BURST_GAMMA_LS_PAPER_SUCCESS_RESULTS_2026_09_09.md)
are not approval evidence for this paper.

## Scientific freeze

- [ ] Identity/geometry repair passes the ROI 010/015 regression test.
- [ ] The protected configuration, label table, temporal windows, spatial supports, nulls, feature panel, candidate budgets, and NMS rules are frozen.
- [ ] All identity-sensitive and timing-sensitive results are regenerated.
- [x] Burst 2 and ROI 010/015 sensitivity analyses are included.
- [x] The bounded-field annotation status is stated accurately.
- [x] Full-field precision, specificity, and false-positive rate are not reported from sparse positives.

## Manuscript values and figures

- [x] `scripts/build_venue_result_facts.py --check` succeeds.
- [x] `macros/venue_results_manifest.json` source-binds every repeated venue-paper fact.
- [ ] `\showprovisionalfalse` is enabled for release; it intentionally remains true in the reviewed working drafts.
- [ ] No `MISSING`, `AUTHOR DECISION`, `CODEX REPLACE`, dagger, or provisional caption remains.
- [x] Every main figure has source hashes and a deterministic generation command.
- [x] Tables, figures, and abstracts use the same generated venue facts and estimand definitions.
- [x] Both upload packages compile and all rendered pages have been inspected for content-affecting warnings.

## Journal files

- [x] Author names and order are recorded as supplied by the user; affiliations and corresponding author remain pending.
- [x] Abstract is no more than 250 words.
- [x] One to seven keywords are supplied.
- [x] Three to five highlights are supplied and each is no more than 85 characters.
- [ ] CRediT statement is complete.
- [ ] Competing-interest statement is complete.
- [ ] Ethics approval and protocol details are complete.
- [ ] Funding and acknowledgements are complete.
- [ ] Data-availability and code-availability statements are complete.
- [ ] Generative-AI disclosure matches actual use and author review.
- [ ] References, DOI values, and author names are verified.

## Reproducibility release

- [ ] Repository release/tag and archive DOI are recorded.
- [ ] Environment lockfile and CPU smoke test are included.
- [ ] Canonical input and annotation hashes are included.
- [x] Source-to-figure and source-to-table/fact manifests are included.
- [ ] The final package excludes private paths, credentials, and unlicensed raw data.
