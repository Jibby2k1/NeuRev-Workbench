<div align="center">

# NeuRev Workbench

### From neural-imaging video to evidence you can audit

An evidence-first research workbench for calcium-imaging measurement,
source separation, candidate detection, human review, and bounded scientific
claims.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3158a5?style=flat-square)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-147d87?style=flat-square)](LICENSE)
[![Research registry](https://img.shields.io/badge/research-registry--backed-6e53a4?style=flat-square)](research/README.md)
[![Evidence boundary](https://img.shields.io/badge/evidence-bounded-102733?style=flat-square)](docs/PUBLICATION_BOUNDARY.md)

[Current work](docs/research/CURRENT_WORK.md) ·
[Research story](research/generated/PROJECT_STORY.md) ·
[Repository guide](docs/REPOSITORY_GUIDE.md) ·
[Documentation](docs/README.md) ·
[Manuscript](paper/overleaf_jnm/README.md)

</div>

![NeuRev evidence flow: a research question and provenance lead through a frozen experiment, bounded execution, scientific audit, evidence capsule, independent review, and the canonical registry; the registry generates synchronized GitHub, documentation, Overleaf, and LLM views.](docs/assets/diagrams/repository-evidence-flow.svg)

[Open the evidence-flow diagram at full size](docs/assets/diagrams/repository-evidence-flow.svg).

NeuRev is built for the awkward middle of scientific computing: the space
between a promising signal and a conclusion that another person can inspect.
It keeps measurements, algorithms, review decisions, execution provenance, and
scientific interpretation connected without treating them as the same thing.

<!-- BEGIN GENERATED RESEARCH SNAPSHOT -->

## Current research boundary

The flagship neuron-identifiability program currently studies **106 confirmed occurrences at 50 immutable sites in 1 recording**. Within that boundary, NeuRev has tested complete-trace features, identity-aware failure modes, truth-known movie stress tests, detector calibration, deblending alternatives, and a staged blinded-review package.

The current evidence still leaves three boundaries unresolved:

- Full-field precision is unresolved.
- One-to-one biological source identity is unresolved.
- Generalization to an independent recording is unresolved.

Those are registered open questions, not footnotes. See the generated
[claim ledger](research/generated/CLAIM_LEDGER.md),
[experiment timeline](research/generated/EXPERIMENT_TIMELINE.md), and
[current manuscript state](paper/overleaf_jnm/CURRENT_RESEARCH_STATE.md).

## What lives here

| Program | Purpose | Lifecycle |
| --- | --- | --- |
| [Neuron Identifiability](paper/overleaf_jnm/CURRENT_RESEARCH_STATE.md) | Evidence-first measurement, representation, detection, identity auditing, and blinded review in calcium imaging. | active |
| [Source Separation and Representation](docs/research/README.md) | Interpretable temporal, spatial, and multiscale representations for separating neural signal, structured artifact, and measurement noise. | active |
| [Fish Intent and Inverse Control](docs/programs/fish_inverse_control/README.md) | Stage-gated measurement, intent decoding, action-conditioned system identification, simulation, and safety-bounded control research. | draft |
| [Grid and Latent Dynamics](docs/GRID_LATENT_DYNAMICS.md) | Template-aligned neural-state grids, latent representations, forecasting baselines, and video-level classifiers. | active |
| [Gamma Local Standardization](docs/research/SPON_CA_BURST_GAMMA_LS_PAPER_SUCCESS_RESULTS_2026_09_09.md) | Single-scale causal signed-difference proposals with guarded radial Gamma-weighted local standardization and explicit calibration, support, and timing audits. | active |

<!-- END GENERATED RESEARCH SNAPSHOT -->

## Latest work

Start with the [current-work synthesis](docs/research/CURRENT_WORK.md) for the
latest reports, implementation routes, and unresolved gates. The compact
Gamma-LS architecture and the broader neuron-identifiability study use
different evaluation populations and readouts; their results stay separate.

| Research thread | Latest reading | What the evidence supports |
| --- | --- | --- |
| Causal candidate extraction | [Gamma-LS final campaign, September 9](docs/research/SPON_CA_BURST_GAMMA_LS_PAPER_SUCCESS_RESULTS_2026_09_09.md) | Signed differencing and one fixed Gamma-LS context; frame-level proposals, with support sufficiency and sustained 1-kHz gates unpassed |
| ICA and whitening | [Final real-data results](docs/research/ICA_WHITENING_REAL_DATA_V1_FINAL_RESULTS.md) · [synthetic screen](docs/research/ICA_WHITENING_HYPERPARAMETER_EVALUATION_RESULTS.md) | Concluded real-data evaluation and its completion gates; independent scoring differs from the Gamma-LS readout |
| Matrix-information ICA | [PC-MITL specialized confirmation](docs/research/PC_MITL_ICA_SPECIALIZED_CONFIRMATION_RESULTS.md) | The locked improvement threshold was missed; retain CS-Parzen |
| Context and feature engineering | [Envelope morphology](docs/research/CONTEXTUAL_ENVELOPE_MORPHOLOGY_V1_RESULTS.md) · [Feature Atlas](docs/research/SPON_CA_BURST_FEATURE_ATLAS_V1_RESULTS.md) | Distinct morphology roles and exploratory ranking signals; no detector promotion |
| Uncertainty-aware feature fusion | [v1.1 results](docs/research/UNCERTAINTY_AWARE_FEATURE_LEARNING_V1_1_RESULTS.md) | Completed engineering screen; no demonstrated nonlinear advantage over matched linear fusion |

[Browse the research index](docs/research/README.md) ·
[Find a workflow contract](docs/workflows/README.md) ·
[Inspect report artifacts](docs/reports/README.md)

## Choose your route

| I want to… | Start here |
| --- | --- |
| Catch up on the latest repository work | [Current work and evidence boundaries](docs/research/CURRENT_WORK.md) |
| Understand the research in five minutes | [Generated research story](research/generated/PROJECT_STORY.md) |
| Inspect every claim and its boundary | [Claim ledger](research/generated/CLAIM_LEDGER.md) |
| Follow the question → result → decision history | [Experiment timeline](research/generated/EXPERIMENT_TIMELINE.md) |
| Process a raw video into an auditable report | [Raw-video workflow](docs/workflows/raw_video_to_report.md) |
| Review or annotate candidates | [Workbench guide](docs/NEURON_WORKBENCH.md) |
| Run a reproducible experiment | [Scientific audit standard](docs/workflows/SCIENTIFIC_AUDIT_OUTPUT_STANDARD.md) |
| Add a new experiment record | [Research registry guide](research/README.md) |
| Extend the maintained Python package | [Codebase navigation](docs/CODEBASE_NAVIGATION.md) |
| Add a new pipeline stage | [Pipeline-stage guide](docs/developer/adding_pipeline_stage.md) |
| Inspect or build the paper | [Overleaf package](paper/overleaf_jnm/README.md) |
| Navigate as a coding agent | [`llms.txt`](llms.txt) and [`docs/navigation.json`](docs/navigation.json) |

## Quick start

NeuRev supports Python 3.10 or newer. CPU setup is sufficient for navigation,
tests, manifests, reports, and the local review workbench; CUDA is optional for
larger experiments.

```bash
git clone https://github.com/Jibby2k1/NeuRev-Workbench.git
cd NeuRev-Workbench
python -m venv .venv-neurobench
source .venv-neurobench/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Verify the portable research story and focused contracts:

```bash
python -m neurobench.research.registry check
make -C paper/overleaf_jnm story-check
python -m pytest -q tests/test_research_registry_schemas.py \
  tests/test_research_story.py tests/test_documentation_navigation.py
```

Explore the command surface:

```bash
neurobench --help
```

For an environment with pinned CPU or GPU dependencies, use
[`environment.cpu.yml`](environment.cpu.yml) or
[`environment.gpu.yml`](environment.gpu.yml).

## The evidence model

NeuRev does not use “complete” as a scientific verdict. It records six
independent questions:

| Dimension | What it answers |
| --- | --- |
| Lifecycle | Where is the work operationally? |
| Outcome | What did the registered comparison find? |
| Evidence tier | In what setting was the evidence produced? |
| Review state | What scrutiny has it received? |
| Claim state | What can currently be said? |
| Decision | What should happen next? |

![NeuRev experiment lifecycle: draft, preregistered, running, computed, validated, reviewed, and closed are auditable states; outcome, evidence tier, decision, and claim state are recorded independently, with failed gates preserved as new linked versions or runs.](docs/assets/diagrams/experiment-lifecycle.svg)

[Open the experiment-lifecycle diagram at full size](docs/assets/diagrams/experiment-lifecycle.svg).

Canonical YAML records live under [`research/registry/`](research/registry/).
Small, checksum-backed public evidence capsules live under
[`research/evidence/`](research/evidence/). GitHub summaries, machine context,
navigation, and manuscript story views are generated from those records.

```bash
# Regenerate synchronized views after changing a canonical record
python -m neurobench.research.registry build

# Fail if a record is invalid or a generated view is stale
python -m neurobench.research.registry check

# On the research workstation, also reconcile ignored local artifacts
python -m neurobench.research.registry verify-live --check
```

Historical v1 experiments intentionally have no invented run records: the old
story did not contain commits, input hashes, configurations, environments, or
seeds. Future native runs must satisfy the complete provenance schema.

## Repository map

```text
NeuRev-Workbench/
├── neurobench/                 maintained Python package and CLI
│   ├── algorithms/             signal, motion, detection, and representation methods
│   ├── experiments/            bounded scientific experiment packages
│   ├── workbench/              local review UI, server, and artifact contracts
│   ├── dynamics/               grid and latent-state forecasting
│   ├── programs/               stage-gated research-program audits
│   └── research/               registry compiler and shared hashing
├── research/
│   ├── registry/               canonical programs, claims, experiments, decisions, runs
│   ├── evidence/               sanitized evidence capsules
│   ├── schemas/                strict JSON Schemas
│   └── generated/              synchronized views; never edit by hand
├── paper/overleaf_jnm/         journal manuscript and generated story views
├── docs/                       human guides, workflows, decisions, and visual system
├── examples/                   small public manifests and configuration examples
├── tests/                      software and scientific-contract tests
├── Inputs/                     local raw/private data; ignored by Git
└── Outputs/                    local generated artifacts; ignored by Git
```

The maintained implementation is under [`neurobench/`](neurobench/). Older
top-level modules remain only for compatibility; consult
[`docs/CODEBASE_NAVIGATION.md`](docs/CODEBASE_NAVIGATION.md) before extending
them.

## Data and publication safety

Raw recordings, reviewer identities, private randomization keys, credentials,
and complete run directories do not belong in Git. `Inputs/` and `Outputs/`
are local workspaces. Public claims point instead to compact evidence capsules
with logical artifact roles, checksums, explicit limitations, and access state.

Before sharing a branch or release:

```bash
python tools/audit_publication_boundary.py
python -m neurobench.research.registry check
```

Read the full [publication boundary](docs/PUBLICATION_BOUNDARY.md) before
preparing a reviewer package or public release.

## Contributing and citation

Scientific changes should begin with a question, falsifier, comparison, data
scope, and gates—not with a result directory. See [CONTRIBUTING.md](CONTRIBUTING.md)
for the code and experiment workflow, [SECURITY.md](SECURITY.md) for private
reporting, and [CITATION.cff](CITATION.cff) for citation metadata.

NeuRev Workbench is released under the [MIT License](LICENSE).
