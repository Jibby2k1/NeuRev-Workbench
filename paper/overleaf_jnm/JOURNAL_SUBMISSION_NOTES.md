# Journal selection and template notes

## Selected venue

**Journal of Neuroscience Methods** — full Research Article.

The project is positioned as a neuroscience measurement and identifiability
study with a reproducible analysis framework. This is a better fit than framing
the submission as an exclusively software or machine-learning paper because the
central contribution is the empirical separation of shared burst structure,
local residual information, neuron/site observability, and quantities that are
not identifiable under sparse-positive annotation.

## Template

The project uses Elsevier's `elsarticle` class in preprint, 12-point,
author--year mode:

```tex
\documentclass[preprint,12pt,authoryear]{elsarticle}
\journal{Journal of Neuroscience Methods}
```

Overleaf provides the Elsevier article template and the class is also maintained
through CTAN. The journal may change production formatting after acceptance; do
not hard-code a two-column appearance into the scientific source files.

## Current submission constraints reflected in the package

- Research Article structure: Abstract, Introduction, Materials and methods,
  Results, Discussion, declarations, references, tables, and figure legends.
- Abstract limited to 250 words.
- One to seven keywords.
- Three to five highlights, each no more than 85 characters.
- Author--year references.
- CRediT authorship statement.
- Data-availability statement.
- Declaration of competing interests.
- Disclosure of generative-AI assistance used during manuscript preparation.

These requirements should be checked again immediately before submission. The
file `submission_checklist.md` is the release gate.

## Scope safeguard

The journal states that purely software/algorithm contributions without a
scientific or research component are outside its intended scope. The manuscript
therefore must retain the data-centered hypotheses, controlled measurement
analyses, and neuroscience interpretation. Repository engineering is essential
for reproducibility but is not written as the sole contribution.
