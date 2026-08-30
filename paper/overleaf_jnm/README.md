# NeuRev Journal of Neuroscience Methods manuscript package

This directory is an Overleaf-compatible working manuscript for a full
Research Article in *Journal of Neuroscience Methods*. It uses Elsevier's
`elsarticle` class and is organized so final analysis outputs can replace all
provisional numbers without editing prose by hand.

The protected single-recording analysis is complete. Current generated analysis
figures, including the three detection-profile taxonomy panels, are included under
`figures/final/`. The narrative and declarations remain
an author-review draft: affiliations, corresponding author, CRediT roles, ethics,
funding, conflicts, acknowledgements, and data/code release identifiers must still
be completed before journal submission.

## Two manuscript versions

- `main_technical.tex` is the evidence-complete version. It delegates to
  `main.tex`, the canonical journal manuscript, and includes the full methods,
  diagnostics, statistical boundaries, and declarations.
- `main_overview.tex` is the relaxed high-level companion. It uses the same
  validated results and figures but emphasizes the research question, current
  evidence, interpretation, and next experiments.
- `main_experiment_story.tex` is a generated question--design--result--decision
  record for each completed experiment and the prioritized next queue.

Both documents are drafts and deliberately preserve unresolved author and
submission metadata. See `CURRENT_RESEARCH_STATE.md` for a concise handoff.

## Research-story build

The repository-level `../../research/registry/` is the source of truth for
claim status, evidence membership, experiment decisions, and next-experiment
priority. `story/research_story.yaml` is a generated compatibility view for the
paper package. Narrative prose remains canonical in the technical manuscript.

```bash
make story        # rebuild the registry view, overview table, and experiment views
make story-check  # fail if generated views are stale or evidence is missing
make diagnostics  # rebuild Figures 4--7, then refresh the story views
```

Generated files are intentionally reviewable and should not be edited by hand:
`STORY_INDEX.md`, `EXPERIMENT_STORY.md`, `generated/story_status_table.tex`, and
`generated/experiment_story.tex`.

## Compile

Set `main.tex` as the Overleaf main document, or run locally:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error main_technical.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error main_overview.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error main_experiment_story.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error supplement_main.tex
```

## Result status

The legacy figures referenced by the current narrative remain visibly provisional.
The protected rerun has generated the current result macros, source maps, tables,
and twelve final analysis figures; those artifacts are included for the authors to
use while revising the narrative.

- `macros/results_macros.tex`
- `paper_results_manifest.json`
- `result_source_map.json`
- stable figure files listed in `figures/README.md`
- generated tables in `tables/`

For a final submission:

1. run `scripts/build_results_macros.py` against the protected output root;
2. resolve all `MISSING`, `AUTHOR DECISION`, and `CODEX REPLACE` markers;
3. set `\showprovisionalfalse` in `macros/provisional.tex`;
4. confirm no daggered values or provisional figure captions remain;
5. complete `submission_checklist.md`.

## Directory structure

- `main.tex`: main article
- `main_technical.tex`: explicit entry point for the evidence-complete article
- `main_overview.tex`: reader-friendly high-level companion
- `sections_overview/`: overview-only narrative sections
- `story/research_story.yaml`: generated paper-compatible registry view
- `generated/`: synchronized LaTeX views built from the story registry
- `supplement_main.tex`: supplementary information
- `sections/`: drafted main-text sections
- `supplement/`: extended methods/results and trace-atlas specification
- `macros/`: notation, draft controls, and generated result values
- `tables/`: manuscript tables
- `figures/`: provisional assets and stable figure contract
- `scripts/`: result-to-LaTeX export utilities
- `references.bib`: source bibliography
- `highlights.txt`: Elsevier highlights file

## Scientific scope

The manuscript is written as an intensive within-recording methods and
identifiability study. It does not claim population-level biological
generalization, exhaustive full-field precision, or recovery of independent
neuronal sources from the current data.
