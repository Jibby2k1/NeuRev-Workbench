# Overleaf Instructions

Upload `main.tex` and `references.bib` to one Overleaf project. Set `main.tex` as
the main document and compile with pdfLaTeX + BibTeX.

The manuscript is self-contained. Planned experimental figures are referenced
under `figures/`. Before results exist, it renders explicit placeholder panels;
it does not fabricate plots. Once the implementation produces the exact files,
create a `figures/` directory in Overleaf and upload them.

Expected compile sequence outside Overleaf:

```bash
pdflatex main.tex
bibtex main       # `bibtex8 main` is equivalent in environments lacking bibtex
pdflatex main.tex
pdflatex main.tex
```
