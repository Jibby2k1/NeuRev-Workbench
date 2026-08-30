# Journal submission checklist

## Scientific freeze

- [ ] Identity/geometry repair passes the ROI 010/015 regression test.
- [ ] The protected configuration, label table, temporal windows, spatial supports, nulls, feature panel, candidate budgets, and NMS rules are frozen.
- [ ] All identity-sensitive and timing-sensitive results are regenerated.
- [ ] Burst 2 and ROI 010/015 sensitivity analyses are included.
- [ ] The bounded-field annotation status is stated accurately.
- [ ] Full-field precision, specificity, and false-positive rate are not reported from sparse positives.

## Manuscript values and figures

- [ ] `scripts/build_results_macros.py --final` succeeds.
- [ ] `result_source_map.json` contains every manuscript macro.
- [ ] `\showprovisionalfalse` is enabled.
- [ ] No `MISSING`, `AUTHOR DECISION`, `CODEX REPLACE`, dagger, or provisional caption remains.
- [ ] Every figure has stable source data and an exact generation command.
- [ ] Tables and figure values agree with `METRICS.json`.
- [ ] Main and supplement compile without warnings that affect content.

## Journal files

- [x] Author names and order are recorded as supplied by the user; affiliations and corresponding author remain pending.
- [ ] Abstract is no more than 250 words.
- [ ] One to seven keywords are supplied.
- [ ] Three to five highlights are supplied and each is no more than 85 characters.
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
- [ ] Source-to-figure and source-to-table manifests are included.
- [ ] The final package excludes private paths, credentials, and unlicensed raw data.
