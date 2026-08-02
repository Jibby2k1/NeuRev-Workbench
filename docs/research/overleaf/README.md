# Overleaf build

This directory contains three independent manuscript entrypoints:

```text
main.tex
  -> neurev_denoise_then_difference.tex

hierarchical_parzen_noisy_ica_main.tex
  -> hierarchical_parzen_noisy_ica.tex

convolutional_infomax_ica_neural_imaging_main.tex
  -> convolutional_infomax_ica_neural_imaging.tex
```

For a dedicated Overleaf project, upload the directory and set the desired
entrypoint as the **Main document**. `main.tex` loads the compatibility package
needed by the latent-dynamics manuscript. The hierarchical entrypoint delegates
to its self-contained canonical source.

The latent-dynamics bundle was compiled with `pdflatex` in two passes and
visually checked as a 13-page PDF. The hierarchical manuscript arrived with its
corresponding package PDF. The convolutional Infomax manuscript is an
Overleaf-ready proposal; this workstation currently has no TeX compiler, so its
source was structurally validated but not locally typeset. All three are working
research narratives and implementation rationales; none claims that proposed
experiments have already been run.
