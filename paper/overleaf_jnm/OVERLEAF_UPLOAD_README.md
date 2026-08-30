# Overleaf upload instructions

1. Upload the contents of the ZIP as a new Overleaf project.
2. Choose the desired main document:
   - `main_technical.tex` (or canonical `main.tex`) for the evidence-complete article;
   - `main_overview.tex` for the shorter high-level research narrative.
   - `main_experiment_story.tex` for the generated experiment-by-experiment record.
3. Use pdfLaTeX and TeX Live 2024 or newer.
4. Compile `supplement_main.tex` separately when exporting the supplement.

The package is compile-oriented but not submission-ready. Red `MISSING` and
`AUTHOR DECISION` markers are intentional and identify information that must be
provided by the authors. `AUTHOR_METADATA.yaml` records the supplied author order
without guessing affiliations or a corresponding author.

The protected-analysis figures are in `figures/final/`, including the label-free
detection-profile taxonomy, post-freeze extensions, and the spatial-ICA morphology,
stability, class, and certainty analyses. The older
`provisional_*.png` files remain because the current narrative still references
them explicitly.

This environment did not contain a TeX engine, so the source and assets were
validated locally but PDFs were not regenerated. Treat a clean Overleaf compile
of `main_technical.tex`, `main_overview.tex`, and `supplement_main.tex` as the
final render gate.
