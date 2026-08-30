# Contributing to NeuRev Workbench

Thank you for helping make neural-imaging analysis more reproducible,
inspectable, and scientifically bounded. Contributions may improve maintained
code, tests, documentation, research records, or small sanitized public
artifacts. Raw recordings, local outputs, reviewer identities, credentials, and
private review material do not belong in Git.

## Start with the right context

1. Read `docs/REPOSITORY_GUIDE.md` and the nearest workflow documentation.
2. Read `AGENTS.md` when working with an automated coding assistant.
3. For any experiment work, read
   `docs/workflows/SCIENTIFIC_AUDIT_OUTPUT_STANDARD.md` before designing or
   running it.
4. For anything that may be shared or released, read
   `docs/PUBLICATION_BOUNDARY.md` and `docs/RELEASE_PROCESS.md`.

Completion, validation, review, publication, and scientific support are separate
states. A successful command or completed run does not by itself establish a
claim.

## Development setup

NeuRev supports Python 3.10 or newer. The maintained local environment name is
`.venv-neurobench`:

```bash
python3 -m venv .venv-neurobench
.venv-neurobench/bin/python -m pip install --upgrade pip
.venv-neurobench/bin/python -m pip install -e '.[dev]'
```

Run the smallest focused tests while iterating, then the relevant suite before a
pull request:

```bash
.venv-neurobench/bin/python -m pytest tests/test_changed_area.py -q
.venv-neurobench/bin/python -m pytest -q
```

Do not make a full-data, GPU, or long-running experiment a pull-request test.
Use synthetic or tiny fixtures, bounded resources, deterministic seeds, and a
new output root.

## Code and documentation changes

- Keep maintained implementation under `neurobench/`; keep one-off maintenance
  and audit entry points under `tools/`.
- Add focused regression tests for behavior changes. Tests must not require
  ignored `Inputs/` or `Outputs/` in a clean checkout.
- Preserve coordinate, frame, identity, and operating-point conventions from
  the owning workflow. Never silently reinterpret unlabeled samples as
  negatives.
- Use repository-relative paths and compact indexes. Do not commit absolute
  workstation paths or links that depend on a contributor's filesystem.
- Update the nearest durable workflow or decision record when semantics,
  evidence boundaries, or gates change.
- Keep public visuals small, attributed, accessible, and reproducible from
  canonical data or records.

## Proposing a new experiment

Open the research-experiment issue form before material new work. Define the
question, comparison, unit of analysis, data scope, falsifiers, scientific
gates, resource envelope, and public/private boundary. An issue or draft record
is a proposal, not run authorization.

Copy `research/templates/experiment-protocol.template.md` into the appropriate
`docs/workflows/` location and resolve every section. Then preview a linked
experiment, planning decision, and planned run:

```bash
.venv-neurobench/bin/python tools/new_research_experiment.py --help
```

The scaffold tool is dry-run by default and never overwrites record files or an
existing output root. Review its captured code, configuration, input, seed, and
environment provenance before repeating with `--write`. The emitted decision is
an explicit planning hold; it does not authorize execution or predict an
outcome.

After writing canonical records, rebuild all synchronized views:

```bash
.venv-neurobench/bin/python -m neurobench.research.registry build
.venv-neurobench/bin/python -m neurobench.research.registry check
make -C paper/overleaf_jnm story-check
```

Never hand-edit `research/generated/`, `docs/navigation.json`, `llms.txt`, or
generated manuscript story files. Correct the canonical registry record and
rebuild.

## Running and reporting experiments

- Obtain any explicit run authorization required by the owning workflow.
- Re-run read-only preflight immediately before execution: inputs and hashes,
  output collision, configuration, code state, disk/RAM/GPU headroom, active
  processes, and resource limits.
- Write into a new, non-colliding `Outputs/<program>/<experiment>/runs/<run>/`
  root. Use atomic metadata, bounded concurrency, progress heartbeats, and
  resumable outputs.
- Preserve negative, mixed, inconclusive, failed, and invalidated outcomes.
  Never recycle an ID to hide a result.
- Complete the scientific-audit output contract or retain the exact
  user-authorized opt-out reason.
- Commit only a small sanitized evidence capsule. Large artifacts belong in a
  checksummed release archive outside Git history.
- Update claims and decisions only to the scope justified by registered
  evidence; retain limitations and falsifiers.

## Publication boundary

Before staging a change, run:

```bash
.venv-neurobench/bin/python tools/audit_publication_boundary.py --root .
git status --short --branch
git diff --check
```

Inspect the exact staged diff. Passing an automated scan is necessary but not
sufficient: reviewers must still check data governance, licenses, identities,
links, generated views, and scientific wording. If a secret or restricted file
was committed, do not paste it into an issue; follow `SECURITY.md`.

## Pull requests

Keep each pull request reviewable and state what changed, which records and
claims are affected, which tests ran, and what remains unresolved. Complete the
pull-request template, include before/after visuals when presentation changes,
and call out any intentionally deferred validation. Do not combine large data
publication, methodological changes, and unrelated refactoring in one review.

By participating, contributors agree to follow `CODE_OF_CONDUCT.md`.
