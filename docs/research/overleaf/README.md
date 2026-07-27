# Overleaf build

Upload this directory as one Overleaf project and set `main.tex` as the **Main
document**. `main.tex` loads the compatibility package needed for literal
software identifiers and then includes the canonical manuscript source:

```text
main.tex
  -> neurev_denoise_then_difference.tex
```

The bundle was compiled with `pdflatex` in two passes and visually checked as a
13-page PDF. The manuscript is a working research narrative and implementation
rationale; it does not claim that the proposed NeuRev experiment has already
been run.
