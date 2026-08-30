# NeuRev Workbench Visual System

This foundation gives NeuRev Workbench a restrained visual identity without
changing the scientific meaning of its figures. The external bounded-review UI
provides the starting vocabulary: deep navy surfaces, cyan structure, and a
small amber attention accent. Scientific annotations remain a separate,
reserved color system.

## Design principles

1. **Evidence before decoration.** A visual should expose provenance, state,
   comparison, uncertainty, or navigation. It should not imply evidence that a
   record does not contain.
2. **One semantic authority.** Diagrams summarize canonical records; generated
   diagrams, GitHub pages, documentation, and manuscripts are presentation
   views, not scientific authorities.
3. **Color never works alone.** Pair color with labels, shape, line style, or
   marker type. This is especially important for review state and annotations.
4. **Grayscale scientific evidence stays grayscale.** Do not tint fluorescence,
   evidence maps, or comparison backgrounds with brand colors.
5. **Light and dark are first-class.** Standalone SVGs define both themes and
   keep the same hierarchy in either appearance.

## Brand palette

The reusable CSS tokens live in
[`assets/brand/palette.css`](assets/brand/palette.css).

| Role | Light | Dark | Intended use |
|---|---:|---:|---|
| Canvas | `#f7fafb` | `#091015` | Page and diagram background |
| Surface | `#ffffff` | `#111b22` | Cards and grouped content |
| Raised surface | `#eef4f6` | `#16232c` | Secondary generated views |
| Ink | `#102733` | `#eff7f8` | Primary text |
| Muted ink | `#526875` | `#a9bac1` | Supporting text |
| Border | `#b9cad1` | `#29404d` | Rules and card boundaries |
| Cyan accent | `#147d87` | `#61d4de` | Identity, focus, authoritative path |
| Cyan text | `#0f6670` | `#61d4de` | Small accent labels on tinted or neutral surfaces |
| Blue accent | `#3158a5` | `#89a7f5` | Execution and validation structure |
| Violet accent | `#6e53a4` | `#c4b5fd` | Registry, decision, generated story |
| Amber accent | — | `#f3bd59` | Sparse UI attention only; never annotation |

Use navy, cyan, blue, violet, and neutral surfaces for repository diagrams.
Amber should remain rare and local to review-interface attention states. It is
not a success color and should not appear in scientific overlays.

## Reserved scientific annotation colors

The scientific audit contract takes priority over branding:

| Meaning | Light | Dark | Required non-color cue |
|---|---:|---:|---|
| Expert ROI | `#2f9d67` | `#7bd7a5` | Solid expert outline and explicit `Expert` label |
| Model prediction | `#d97706` | `#ffad5c` | Dashed or distinct model outline and explicit `Model` label |
| One-to-one match | `#f4e7a1` | `#f4e7a1` | Thin link line plus match identifier |

Green is only for expert ROIs, orange only for model predictions, and pale
yellow only for one-to-one links. Do not use these colors for repository health,
experiment outcomes, lifecycle progress, buttons, decorative gradients, or
chart series. In particular, a green check mark must not stand in for “passed”
inside a scientific figure.

## Typography and composition

- Use the system sans-serif stack: `Inter`, `ui-sans-serif`, `system-ui`,
  `-apple-system`, `BlinkMacSystemFont`, `Segoe UI`, `sans-serif`.
- Use sentence case for titles and labels. Small uppercase text is reserved for
  short section eyebrows.
- Keep body and diagram labels at 12 screen pixels or larger; target 14–18
  pixels for card content in a full-width GitHub image.
- Prefer direct labels to legends. When a legend is unavoidable, keep it near
  the evidence it explains.
- Use rounded rectangles sparingly for bounded records or phases. Use arrows
  only for actual dependency, transition, or generated-from relationships.
- Keep authoritative evidence above generated presentation views. A generated
  page must never appear upstream of the evidence capsule or registry.

## Canonical diagrams

### Repository evidence flow

![NeuRev evidence flow: a research question and provenance lead through a frozen experiment, bounded execution, scientific audit, evidence capsule, independent review, and the canonical registry; the registry generates synchronized GitHub, documentation, Overleaf, and LLM views.](assets/diagrams/repository-evidence-flow.svg)

**Caption:** NeuRev’s public story is downstream of frozen experiments,
validated audit artifacts, explicit review, and canonical evidence records.

- Editable Mermaid:
  [`repository-evidence-flow.mmd`](assets/diagrams/src/repository-evidence-flow.mmd)
- Standalone SVG:
  [`repository-evidence-flow.svg`](assets/diagrams/repository-evidence-flow.svg)
- Recommended alt text: “A research question and provenance lead through a
  frozen experiment, bounded execution, scientific audit, evidence capsule,
  independent review, and the canonical registry. The registry generates
  synchronized GitHub, documentation, Overleaf, and LLM views.”

### Incremental experiment lifecycle

![NeuRev experiment lifecycle: draft, preregistered, running, computed, validated, reviewed, and closed are auditable states; outcome, evidence tier, decision, and claim state are recorded independently, with failed gates preserved as new linked versions or runs.](assets/diagrams/experiment-lifecycle.svg)

**Caption:** Computation is one lifecycle transition, not a scientific verdict;
outcome, evidence tier, decision, and claim state are recorded independently.

- Editable Mermaid:
  [`experiment-lifecycle.mmd`](assets/diagrams/src/experiment-lifecycle.mmd)
- Standalone SVG:
  [`experiment-lifecycle.svg`](assets/diagrams/experiment-lifecycle.svg)
- Recommended alt text: “Draft, preregistered, running, computed, validated,
  reviewed, and closed are auditable states. Outcome, evidence tier, decision,
  and claim state are independent. Failed gates preserve the result and open a
  linked version or run; passed gates update synchronized story views.”

## Editing and export contract

1. Edit the Mermaid source first when changing diagram meaning.
2. Update the matching standalone SVG in the same change. Keep its `<title>`,
   `<desc>`, visible caption, and light/dark styles synchronized.
3. Do not introduce green, orange, or pale yellow into repository diagrams.
4. Validate Mermaid syntax, SVG XML, embedded accessibility metadata, relative
   links, and light/dark contrast before publishing.
5. Use kebab-case filenames and deterministic paths under `docs/assets/` so
   GitHub, the documentation site, Overleaf exports, and LLM indexes can reuse
   the same artifact.

When a diagram becomes generated from the research registry, keep its semantic
source and generated output distinct. CI should reject stale renderings rather
than silently updating them.
