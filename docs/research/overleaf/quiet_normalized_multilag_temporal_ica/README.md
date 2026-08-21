# Quiet-normalized multi-lag temporal ICA paper

This is a living, Overleaf-ready manuscript built from the completed short-history
`spon_ca_burst_multilag_temporal_identification_v1` run and the guarded
long-history `spon_ca_burst_long_history_fir_v1` run, with validated scientific
audits. The included `next_research_program.tex` records the
measurement-first research roadmap and conservative PC execution schedule. Its
operational companion is
`../../SPON_CA_BURST_NEXT_RESEARCH_PROGRAM_2026_08_12.md`.

Upload the entire directory to Overleaf and set `main.tex` as the main file.
The manuscript deliberately distinguishes the frozen label-free `ica_k8`
audit lane from the post-hoc known-label `ica_k16` diagnostic winner.

Local compilation:

```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Do not strengthen the source-separation or specificity claims until the
exhaustive candidate review in the manuscript
checklist are complete.
