---
title: NeuRev Workbench
description: Evidence-first calcium-imaging measurement, detection, review, and research provenance.
---

<div class="neurev-hero" markdown>
  <p class="neurev-eyebrow">Calcium imaging · reproducible evidence · bounded claims</p>

# See the signal. Preserve the evidence chain.

NeuRev Workbench connects calcium-imaging measurement, candidate detection,
visual adjudication, and scientific provenance in one inspectable workflow.
Software status, evidence tier, review state, and scientific outcome remain
separate—so a finished run never masquerades as a finished conclusion.

[Explore current work](research/CURRENT_WORK.md){ .md-button .md-button--primary }
[Inspect the repository guide](REPOSITORY_GUIDE.md){ .md-button }
</div>

<div class="neurev-principles" role="list" aria-label="NeuRev operating principles">
  <div role="listitem"><strong>Audit first</strong><span>Every scientific workflow inherits a shared evidence contract.</span></div>
  <div role="listitem"><strong>Immutable runs</strong><span>New work receives a new output root; prior evidence remains inspectable.</span></div>
  <div role="listitem"><strong>Bounded claims</strong><span>Registered limitations travel with results into every presentation view.</span></div>
</div>

## Current work, with its limits in view

The [current-work synthesis](research/CURRENT_WORK.md) connects the latest
reports to their code and scientific gates. Read each result within its own
population, evaluation head, and audit boundary.

<div class="grid cards neurev-current-grid" markdown>

-   **Causal Gamma-LS candidate extraction**

    ---

    The September 9 campaign retains signed differencing and one fixed local
    context. Full-record proposals are audited; support sufficiency and strict
    sustained 1-kHz readiness remain unpassed.

    [Read the final campaign](research/SPON_CA_BURST_GAMMA_LS_PAPER_SUCCESS_RESULTS_2026_09_09.md)

-   **Representations and feature evidence**

    ---

    Follow ICA/whitening, PC-MITL, contextual envelopes, Feature Atlas, and
    uncertainty-aware fusion through their completed comparisons and
    unresolved confirmation gates.

    [Browse research by thread](research/README.md)

-   **Reproduce and inspect**

    ---

    Find frozen workflow contracts, small artifact indexes, and maintained
    modules. A recorded execution and a scientific advance are separate
    outcomes.

    [Choose a workflow](workflows/README.md)

</div>

## Choose your reading path

<div class="grid cards neurev-route-grid" markdown>

-   <span class="neurev-route-mark" aria-hidden="true">01</span> **Researcher**

    ---

    Start from the current methods and evidence, then descend into a frozen
    workflow only when you need its operational details.

    [Open the result index](research/README.md)

-   <span class="neurev-route-mark" aria-hidden="true">02</span> **Reviewer**

    ---

    Inspect the claim boundary, evidence contract, and independent-review
    package before reading narrative conclusions.

    [Review publication boundaries](PUBLICATION_BOUNDARY.md)

-   <span class="neurev-route-mark" aria-hidden="true">03</span> **Developer or coding agent**

    ---

    Use the maintained package map, authority order, and smallest focused test
    surface for the change you are making.

    [Navigate the codebase](CODEBASE_NAVIGATION.md)

</div>

## One evidence chain, many synchronized views

![NeuRev evidence flow from research question and provenance through frozen experiment, bounded execution, scientific audit, evidence capsule, review, canonical registry, and synchronized public views.](assets/diagrams/repository-evidence-flow.svg)

[Open the evidence-flow diagram at full size](assets/diagrams/repository-evidence-flow.svg){ .neurev-diagram-link }

The registry is the semantic center of the public research story. GitHub pages,
this documentation site, manuscript material, and machine-readable context are
downstream views—not competing sources of truth. The
[visual system](VISUAL_SYSTEM.md) defines how these relationships are drawn
without borrowing colors reserved for scientific annotations.

!!! info "Current research state"

    The canonical project story, claim ledger, and experiment timeline are
    generated from repository-root research records. Read the
    [current project story on GitHub](https://github.com/Jibby2k1/NeuRev-Workbench/blob/main/research/generated/PROJECT_STORY.md)
    for the latest claim states and unresolved boundaries.

## From experiment to conclusion

Computation is only one transition in the experiment lifecycle. Validation,
review, decision, and claim state are recorded independently; a failed gate is
preserved as evidence and can motivate a linked successor.

![NeuRev experiment lifecycle showing auditable state transitions and independent outcome, evidence-tier, decision, and claim-state records.](assets/diagrams/experiment-lifecycle.svg)

[Open the experiment-lifecycle diagram at full size](assets/diagrams/experiment-lifecycle.svg){ .neurev-diagram-link }

[Read the scientific audit standard](workflows/SCIENTIFIC_AUDIT_OUTPUT_STANDARD.md){ .md-button .md-button--primary }
[Browse all workflows](workflows/README.md){ .md-button }

## Work with NeuRev locally

```bash
git clone https://github.com/Jibby2k1/NeuRev-Workbench.git
cd NeuRev-Workbench
python -m venv .venv-neurobench
. .venv-neurobench/bin/activate
python -m pip install -e ".[dev,docs]"
```

Before running a scientific experiment, read the
[audit output standard](workflows/SCIENTIFIC_AUDIT_OUTPUT_STANDARD.md) and the
matching workflow contract. Generated views should be rebuilt from canonical
records rather than edited directly.

```bash
python -m neurobench.research.registry check
mkdocs serve
```

## What this repository does—and does not—claim

NeuRev supports method development and evidence inspection across a broad
pipeline. It does not collapse sparse positive annotations into exhaustive
negatives, treat passive intent as a causal action effect, or infer scientific
success from completed computation. The [repository guide](REPOSITORY_GUIDE.md)
explains the authority order; the [publication boundary](PUBLICATION_BOUNDARY.md)
explains what belongs in a public clone.
